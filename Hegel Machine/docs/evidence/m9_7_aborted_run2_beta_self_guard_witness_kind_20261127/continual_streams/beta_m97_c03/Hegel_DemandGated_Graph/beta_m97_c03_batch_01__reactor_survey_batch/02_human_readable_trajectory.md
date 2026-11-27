# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c03_batch_01__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c03_step_01` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `351,264` (`340,263` / `96,896` / `11,001`) |
| **Total Episode Wall Time (s)** | `385.3s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15275 ms` LLM)
- **Rationale**: Inspect the public schema and archived discretization to identify conventions that must remain unchanged; inspect actual sweep coverage to determine whether historical calibration is needed.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The new contract changes calibration, outflow boundary conditions, and source inversion.)

```bash
set -e
cd /holdout_task
cat public_asset_index.json
cat validate_public_schema.py
ls -lh /tmp/hegel_continual_archive
sed -n '1,320p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT' || true
python3 - <<'PY'
import pickle
with open('public_beta_m97_c03_batch_01.pkl','rb') as f: s=pickle.load(f)
for k,v in s.items():
 print(k, type(v).__name__, getattr(v,'shape',None), repr(v)[:480])
PY
```
- **Container Exit Code**: `1` (`523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18520 ms` LLM)
- **Rationale**: Unpack the public case list correctly and recover the archived discretization and certificate conventions that were truncated. Sweep counts and the history format determine which calibration paths the solver needs.
- **Alternatives Weighed**: `Use the prior gate's gain and offset payload as fixed coefficients` (rejected: The contract requires fitting current sweeps when available; the payload cannot establish coverage or identify EMI contamination.)

```bash
set -e
cd /holdout_task
cat /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py
cat validate_public_schema.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
python3 - <<'PY'
import json,pickle,numpy as np
with open('public_beta_m97_c03_batch_01.pkl','rb') as f: cases=pickle.load(f)
print('cases',len(cases))
for i,case in enumerate(cases):
 s=case['study']; print(i, list(s), 'sweeps:', repr(s['sensor_calibration_sweeps'])[:1700]); print('trace',repr(s['past_vortex_circulation_trace'])[:250], 'mode',s['recirculation_loop_mode']); print('weights',case['gt']['w_opt'])
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: h=json.load(f)
print('history format',repr(h)[:1600])
PY
```
- **Container Exit Code**: `0` (`402 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.231613,  1.167296, -1.484077,  0.88367 , -0.66923 ,  0.383197,
       -0.157314, -0.218079, -0.74952 ,  1.008611, -1.051865,  1.594549,
        1.701262, -1.33623 , -0.664275,  1.100048, -0.970694, -0.080674]), 'raw_readout': array([ 0.17961008,  1.410273  ,  0.15866002,  0.52544002,  2.29402905,
       -1.69914791,  1.64909779, -0.18230583, -0.86038284,  1.82225349,
        0.57553508, -1.00039659,  1.33774358,  1.49689181, -0.26713132,
        0.31482815,  1.52667372,  0.42307444, -0.92181834, -0.16811049,
       -1.29548281,  1.46702317,  1.33823418,  0.0738487 ,  2.06966888,
       -0.16836624,  0.5712838 ,  1.93640461, -0.69773548,  2.28540511,
        0.18330313,  1.28740916, -1.84121091,  0.95273052, -0.87969149,
        0.3621725 , -0.02214369, -0.07318637, -0.51959671,  0.95723336,
       -0.77356655,  1.44942106,  1.53905973, -1.01243321, -0.44799095,
        1.03404038, -0.705383  ,  0.04223377])}
trace array([0.311453, 0.204083, 0.162267, 0.324958, 0.799554, 0.932288,
       0.942991, 0.704902, 0.703121, 0.338325, 0.280892, 0.22806 ,
       0.181983, 0.334224, 0.28382 , 0.125608]) mode 0.0
weights [0.32473619 0.4924118  0.15113327 0.08570754 0.27999742 0.21527281
 0.15392076 0.20559869 0.16902636 0.0883555  0.21664032 0.21719935]
history format [{'episode_short_name': 'beta_m97_c03_s00', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [1.264348, -1.219958, 0.256522, 1.347465, 0.569354, 1.811065, -0.019818, 0.893868, -0.78079, 0.69839, -0.260974, -0.723373, -0.09476, -0.5241, -1.493419, -1.382773, 1.855897, 0.114249, 1.853062, -0.77891, 1.464848, -0.952095, 1.253344, -1.324626, 0.447921, -0.495095, 2.182426, 2.194405, -1.195034, 1.353202, 0.467707, -1.264416, 0.593127, 0.596117, 1.467343, -1.056263, 0.211325, -1.250988, 0.51585, 1.236978, -0.96195, 0.611761, 1.710794, -0.837501, 1.007049, -0.081748, 1.502034, 1.660641], 'raw_readout': [1.264348, -1.219958, 0.256522, 1.347465, 0.569354, 1.811065, -0.019818, 0.893868, -0.78079, 0.69839, -0.260974, -0.723373, -0.09476, -0.5241, -1.493419, -1.382773, 1.855897, 0.114249, 1.853062, -0.77891, 1.464848, -0.952095, 1.253344, -1.324626, 0.447921, -0.495095, 2.182426, 2.194405, -1.195034, 1.353202, 0.467707, -1.264416, 0.593127, 0.596117, 1.467343, -1.056263, 0.211325, -1.250988, 0.51585, 1.236978, -0.96195, 0.611761, 1.710794, -0.837501, 1.007049, -0.081748, 1.502034, 1.660641]}]
```

### Turn 3 — `execute_in_remote_docker` (`15750 ms` LLM)
- **Rationale**: Read only the archived forward and inversion code plus the validator. A changed flux or indexing convention would alter the public sensor matrix even with correct calibration, and the validator determines which independently computed certificate checks are needed.
- **Alternatives Weighed**: `Reconstruct the forward discretization from the task description alone` (rejected: The description changes the x-wall rows but leaves the interior flux convention, stacking, and field indexing to the archived discretization.)

```bash
set -e
sed -n '/^def _axis/,$p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py | head -160
printf '\n--- archived definitions ---\n'
grep -n '^def ' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py
printf '\n--- validator ---\n'
sed -n '1,175p' /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`333 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
g = S.T @ res + lw * w_feas
    # Unconditional convex supporting-hyperplane + box-Lagrange lower bound valid for ANY `(w_feas, nu)`:
    l_kkt = j_true - float(g @ w_feas) + nu * bt + u_b * float(np.sum(np.minimum(0.0, g - nu)))
    stat_res = float(np.max(np.abs(g - nu - ll + lu)))
    if abs(p_rep - j_true) > max(gap_tol, 1e-6 * max(1.0, abs(j_true))):
        return {"valid": False, "reason": f"reported_primal_mismatch:{abs(p_rep - j_true):.3e}", "gap": 1e9}
    if d_rep > l_kkt + gap_tol:
        return {"valid": False, "reason": f"unsound_dual_lower_bound:{d_rep:.8f}>{l_kkt:.8f}", "gap": 1e9}
    if stat_res > 1e-4:
        return {"valid": False, "reason": f"kkt_stationarity_violation:{stat_res:.3e}", "gap": 1e9}
    eff_lb = min(d_rep, l_kkt)
    gap = j_true - eff_lb
    if gap < -1e-9 or gap > gap_tol:
        return {"valid": False, "reason": f"duality_gap_too_large:{gap:.3e}", "gap": float(gap)}
    return {"valid": True, "reason": "ok", "true_primal": float(j_true), "verified_dual_lb": float(l_kkt), "gap": float(max(0.0, gap))}

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c03_batch_01_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c03_batch_01.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c03_batch_01(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out: errs.append("missing:kkt_certificate")
    elif not errs:
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
        if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
        elif float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("w_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 4 — `execute_in_remote_docker` (`35223 ms` LLM)
- **Rationale**: Build on the archived discretization and active-set solver. Compare the projected sensor matrix and measurements independently: a matrix mismatch points to transport or stacking, whereas a measurement-only mismatch points to calibration. Compare weights and terminal field separately because the schema validator does not check field accuracy.
- **Alternatives Weighed**: `Seal the y walls as well` (rejected: The specified sorption profile localizes to the x walls; preserve the archived y boundary condition unless the public forward comparison contradicts it.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py')
s=p.read_text()
import re
s=re.sub(r'\A""".*?"""', '"""Bank-calibrated reaction–transport tomography with certified source inversion."""', s, count=1, flags=re.S)
s=s[:s.rfind('\ndef solve_')]
a=s.index('    ax = _axis', s.index('def _forward'))
b=s.index('    ident = ', a)
s=s[:a]+'''    lx, ly = study['Lx'], study['Ly']
    trace = np.asarray(study['past_vortex_circulation_trace'])
    excess = np.maximum(0.0, trace - study['critical_vortex_circulation'])
    memory = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0.0
    potential = np.asarray(study['V_reaction_2d'])
    if sealed:
        x = np.linspace(0.0, lx, nx)[:, None]
        y = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(x/(0.25*lx))**2) + np.exp(-((lx-x)/(0.25*lx))**2)
        potential = potential + (0.95 + 0.55*np.tanh(memory))*wall*(0.6 + 0.4*np.sin(np.pi*y/ly))
    ax = _axis(study['kappa_x'], study['vel_x'], lx, 0.0 if sealed else study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], ly, study['robin_y'])
    operator = kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(potential.ravel(order='F'))
'''+s[b:]
s+='''

def solve_beta_m97_c03_batch_01(study: dict) -> dict:
    sensors, terminal = _forward(study)
    data = np.column_stack((_calibrate(study), sensors))
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge']*np.eye(z.shape[1])
    profiled = data - z @ np.linalg.solve(gram, z.T @ data)
    weights, certificate = _invert(profiled[:, 1:], profiled[:, 0], study['source_ridge'], study['u_box'], study['total_budget'])
    field = (terminal @ weights).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=weights, terminal_field=field, kkt_certificate=certificate)
'''
Path('beta_m97_c03_batch_01_solver.py').write_text(s)
PY
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','beta_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for c in pickle.load(open('public_beta_m97_c03_batch_01.pkl','rb')):
 s,g=c['study'],c['gt']; a,_=m._forward(s); z=s['Z_nuisance']; v=np.column_stack((m._calibrate(s),a)); p=v-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@v)
 out=m.solve_beta_m97_c03_batch_01(s)
 for k,x,y in [('survey',p[:,1:],g['S_tilde']),('calibration',p[:,0],g['r_y']),('field',out['terminal_field'],g['terminal_field_A']),('weights',out['weights'],g['w_opt'])]: print(k,'max abs',np.max(np.abs(x-y)),'relative L2',np.linalg.norm(x-y)/np.linalg.norm(y))
 print('robin x/y',s['robin_x'],s['robin_y'])
PY
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`605 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
survey max abs 0.032405724845724355 relative L2 0.25883982345803513
calibration max abs 1.779399241286228e-07 relative L2 1.4574188444100698e-06
field max abs 0.020661356788040076 relative L2 0.22238338050395967
weights max abs 0.09046632566009181 relative L2 0.21706258835585063
robin x/y 0.85 0.75
{"schema_valid": false, "errors": ["kkt_invalid:reported_primal_mismatch:5.701e-03"]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=04108231fc45285f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_04108231 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27064 ms` LLM)
- **Rationale**: The axis comparison will show whether the archived y Robin condition caused the mismatch. Evaluating the terminal field with reference weights isolates the forward discrepancy from inversion error.
- **Alternatives Weighed**: `Tune calibration or the active-set optimizer` (rejected: Neither can correct a mismatch in the projected sensor matrix.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived operator with only the x end rows sealed and the specified sorption reproduces the public forward map.` | Witness: `The projected survey matrix differs by 0.03241 max absolute and 25.88% relative L2; the projected measurement differs by only 1.78e-7 max absolute.` | Blanket Diagnosis: `Calibration agrees independently; the mismatch lies in the forward operator or its indexing, before optimization.` | Sublated Pivot: `Keep the calibrated measurements and test the discrete x/y boundary and sorption factors against both the survey matrix and terminal field.`

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
from pathlib import Path
from itertools import product
import pickle
import numpy as np
p = Path('beta_m97_c03_batch_01_solver.py')
source = p.read_text()
for case in pickle.load(open('public_beta_m97_c03_batch_01.pkl','rb')):
    s, gt = case['study'], case['gt']
    z = s['Z_nuisance']
    gram = z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1])
    for xseal,yseal,wall in product((False,True),repeat=3):
        code = source.replace("0.0 if sealed else study['robin_x']", "0.0" if xseal else "study['robin_x']").replace("ly, study['robin_y'])", "ly, 0.0)" if yseal else "ly, study['robin_y'])").replace('    if sealed:', '    if '+str(wall)+':')
        ns = {}; exec(compile(code,str(p),'exec'),ns)
        a,f = ns['_forward'](s)
        a -= z @ np.linalg.solve(gram,z.T@a)
        d = a - gt['S_tilde']
        f = (f@gt['w_opt']).reshape(gt['terminal_field_A'].shape,order='F')
        print(f'xseal={xseal} yseal={yseal} wall={wall} survey_rel={np.linalg.norm(d)/np.linalg.norm(gt["S_tilde"]):.8g} survey_max={np.max(np.abs(d)):.8g} field_max={np.max(np.abs(f-gt["terminal_field_A"])):.8g}')
    for key in ('reaction_x','reaction_y','vel_x','vel_y','mu_shifts'):
        print(key,repr(s[key]))
PY
```
- **Container Exit Code**: `0` (`718 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
xseal=False yseal=False wall=False survey_rel=0.26247603 survey_max=0.031142448 field_max=0.024794856
xseal=False yseal=False wall=True survey_rel=0.33262188 survey_max=0.035110159 field_max=0.025761436
xseal=False yseal=True wall=False survey_rel=0.12774313 survey_max=0.019616857 field_max=0.017091843
xseal=False yseal=True wall=True survey_rel=0.18380467 survey_max=0.026928147 field_max=0.019678352
xseal=True yseal=False wall=False survey_rel=0.23142939 survey_max=0.028754231 field_max=0.016673491
xseal=True yseal=False wall=True survey_rel=0.25883982 survey_max=0.032405725 field_max=0.017963171
xseal=True yseal=True wall=False survey_rel=0.36638249 survey_max=0.047073113 field_max=0.012626741
xseal=True yseal=True wall=True survey_rel=8.2984241e-16 survey_max=9.7144515e-17 field_max=5.5511151e-17
reaction_x array([0.3       , 0.30055363, 0.30221453, 0.3049827 , 0.30885813,
       0.31384083, 0.3199308 , 0.32712803, 0.33543253, 0.34484429,
       0.35536332, 0.36698962, 0.37972318, 0.39356401, 0.40851211,
       0.42456747, 0.4417301 , 0.46      ])
reaction_y array([0.4       , 0.39923307, 0.39694066, 0.39314791, 0.38789636,
       0.38124356, 0.37326238, 0.36404028, 0.35367828, 0.34228994,
       0.33      , 0.31694313, 0.30326238, 0.28910764, 0.27463398,
       0.26      ])
vel_x array([0.24      , 0.23647059, 0.23294118, 0.22941176, 0.22588235,
       0.22235294, 0.21882353, 0.21529412, 0.21176471, 0.20823529,
       0.20470588, 0.20117647, 0.19764706, 0.19411765, 0.19058824,
       0.18705882, 0.18352941, 0.18      ])
vel_y array([0.2       , 0.20266667, 0.20533333, 0.208     , 0.21066667,
       0.21333333, 0.216     , 0.21866667, 0.22133333, 0.224     ,
       0.22666667, 0.22933333, 0.232     , 0.23466667, 0.23733333,
       0.24      ])
mu_shifts array([0.  , 0.22, 0.52, 0.9 ])

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=04108231fc45285f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_04108231 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31532 ms` LLM)
- **Rationale**: Apply the confirmed y-wall correction, then verify the complete solver in an isolated process. The public field comparison catches output errors outside the schema check; the mode and threshold checks catch accidental sealing of the open branch. Inspect the inherited calibration fallback and active-set code before targeting their stress cases.
- **Alternatives Weighed**: `Change reaction or advection terms` (rejected: Sealing both axes with the stated sorption already matches the reference forward map to floating-point precision.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_batch_01_solver.py')
s = p.read_text()
old = "    ay = _axis(study['kappa_y'], study['vel_y'], ly, study['robin_y'])"
assert s.count(old) == 1
p.write_text(s.replace(old, "    ay = _axis(study['kappa_y'], study['vel_y'], ly, 0.0 if sealed else study['robin_y'])"))
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('beta_m97_c03_batch_01_solver.py')
a = load('/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py')
with open('public_beta_m97_c03_batch_01.pkl','rb') as f:
    cases = pickle.load(f)
for case in cases:
    s, gt = case['study'], case['gt']
    out = m.solve_beta_m97_c03_batch_01(s)
    for key, ref in [('weights','w_opt'), ('terminal_field','terminal_field_A')]:
        err = np.max(np.abs(out[key] - gt[ref]))
        print(key, 'max abs error:', err)
        assert err < 1e-5
    w = np.asarray(out['weights']); c = out['kkt_certificate']
    g = gt['S_tilde'].T @ (gt['S_tilde'] @ w - gt['r_y']) + s['source_ridge']*w
    d = g - c['nu_sum']
    gap = d @ w - s['u_box']*np.minimum(d,0).sum()
    print('public reference tangent gap:', gap)
    assert gap < 2e-4
    for mode, trace in [(1,s['past_vortex_circulation_trace']), (0,np.full_like(s['past_vortex_circulation_trace'],s['critical_vortex_circulation']))]:
        st = dict(s, recirculation_loop_mode=mode, past_vortex_circulation_trace=trace)
        for got, expected in zip(m._forward(st), a._forward(st)):
            np.testing.assert_allclose(got, expected, atol=1e-13, rtol=0)
    for got, expected in zip(m._forward(dict(s,robin_x=2*s['robin_x'],robin_y=2*s['robin_y'])),m._forward(s)):
        np.testing.assert_array_equal(got,expected)
print('Boundary checks passed: both Robin losses vanish only in the fouled, sealed branch.')
PY
sed -n '43,87p;122,195p' beta_m97_c03_batch_01_solver.py
```
- **Container Exit Code**: `0` (`701 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
nu = inv_g.sum()/inv_one.sum()
            direction = nu*inv_one - inv_g
            if np.max(np.abs(direction)) > 1e-12:
                ratio = np.full(direction.size, np.inf)
                up = direction > 1e-13
                down = direction < -1e-13
                wf = w[free]
                ratio[up] = (upper - wf[up])/direction[up]
                ratio[down] = -wf[down]/direction[down]
                block = np.argmin(ratio)
                step = min(1.0, max(0.0, ratio[block]))
                w[free] = np.clip(wf + step*direction, 0, upper)
                if step < 1.0:
                    idx = np.flatnonzero(free)[block]
                    active[idx] = 1 if up[block] else -1
                    w[idx] = upper if up[block] else 0.0
                continue
        else:
            lo = np.max(g[active == 1], initial=-np.inf)
            hi = np.min(g[active == -1], initial=np.inf)
            nu = np.clip(np.mean(g), min(lo, hi), max(lo, hi))
        violation = active*(g - nu)
        idx = np.argmax(violation)
        if violation[idx] <= 1e-11:
            break
        active[idx] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')
    res = s @ w - y
    grad = s.T @ res + ridge*w
    slack = grad - nu
    obj = 0.5*(res @ res + ridge*(w @ w))
    dual = obj - grad @ w + nu*budget + upper*np.minimum(slack, 0).sum()
    return w, dict(weights=w.copy(), nu_sum=float(nu), lam_lower=np.maximum(slack, 0),
                   lam_upper=np.maximum(-slack, 0), primal_objective=float(obj),
                   dual_lower_bound=float(min(obj, dual)))



def solve_beta_m97_c03_batch_01(study: dict) -> dict:
    sensors, terminal = _forward(study)
    data = np.column_stack((_calibrate(study), sensors))
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge']*np.eye(z.shape[1])
    profiled = data - z @ np.linalg.solve(gram, z.T @ data)
    weights, certificate = _invert(profiled[:, 1:], profiled[:, 0], study['source_ridge'], study['u_box'], study['total_budget'])
    field = (terminal @ weights).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=weights, terminal_field=field, kkt_certificate=certificate)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cec93a7a02549493 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_cec93a7a (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30617 ms` LLM)
- **Rationale**: Use the real archive and altered public sweeps to detect a fallback format mismatch or an EMI-biased fit. Stress the inherited optimizer at active bounds and budget endpoints; change it only if feasibility or certificate residuals fail.
- **Alternatives Weighed**: `Submit on public-case agreement alone` (rejected: The clean public sweeps never exercise the required historical fallback, and the public inversion does not cover the fully active budget endpoints.)

```bash
set -e
cd /holdout_task
sed -n '1,80p' beta_m97_c03_batch_01_solver.py
python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with m._ARCHIVE.open() as f: h = json.load(f)
def describe(v, level=0):
    if isinstance(v, dict):
        return {k: describe(x, level+1) for k,x in v.items()}
    if isinstance(v, list):
        return [len(v), describe(v[0], level+1) if v else None]
    return v
print('Calibration archive structure:', describe(h))
with open('public_beta_m97_c03_batch_01.pkl','rb') as f: s = pickle.load(f)[0]['study']
sw = {k: np.array(v) for k,v in s['sensor_calibration_sweeps'].items()}
idx = np.flatnonzero(sw['bank_id']==0)
base = m._calibrate(s)
for label, keep, spike in [('one current row', 1, False), ('three current rows, one spike', 3, True), ('twelve current rows, one spike', len(idx), True)]:
    mask = np.ones(sw['bank_id'].size, dtype=bool); mask[idx[keep:]] = False
    ss = {k: v[mask].copy() for k,v in sw.items()}
    if spike: ss['raw_readout'][np.flatnonzero(ss['bank_id']==0)[0]] += 5.0
    try:
        y = m._calibrate(dict(s, sensor_calibration_sweeps=ss))
        print(label, 'maximum change in calibrated measurements:', np.max(np.abs(y-base)))
    except (ValueError, KeyError, TypeError) as e:
        print(label, type(e).__name__, str(e))
rng = np.random.default_rng(707)
max_feas = max_gap = max_comp = 0.0
for i in range(120):
    A = rng.normal(size=(30,12))*10**rng.uniform(-2,1)
    y = rng.normal(size=30); u = 0.4; ridge = 0.005
    b = 0.0 if i==0 else 12*u if i==1 else (i%12)*u if i%4==0 else rng.uniform(0,12*u)
    w,c = m._invert(A,y,ridge,u,b)
    d = A.T @ (A@w-y) + ridge*w-c['nu_sum']
    max_feas = max(max_feas, abs(w.sum()-b), -w.min(), w.max()-u)
    max_gap = max(max_gap, d@w-u*np.minimum(d,0).sum())
    max_comp = max(max_comp, np.max(np.abs(w*c['lam_lower'])), np.max(np.abs((u-w)*c['lam_upper'])))
print('QP worst feasibility, tangent gap, complementarity:', max_feas, max_gap, max_comp)
assert max(max_feas,max_gap,max_comp) < 1e-8
PY
```
- **Container Exit Code**: `0` (`554 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
valid]]) / (x[jj[valid]] - x[ii[valid]])
        if slopes.size:
            a = np.median(slopes)
            d = np.median(y - a*x)
            return np.abs(y - (a*x + d)) <= 3.0
    return np.ones(x.size, dtype=bool)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gain_offset = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        if x.size >= 2:
            clean = _clean_sweep(x, y)
            if clean.all() or clean.sum() >= 4:
                gain_offset[b] = _affine(x[clean], y[clean])
                continue
        else:
            clean = np.ones(x.size, dtype=bool)
        if history is None and _ARCHIVE.is_file():
            history = json.loads(_ARCHIVE.read_text())
        if history is not None:
            entries = history.get('banks', history) if isinstance(history, dict) else history
            if isinstance(entries, dict):
                entry = entries.get(str(b))
            else:
                entry = next((e for e in reversed(entries) if e['bank_id'] == b), None)
            if entry is not None:
                gain_offset[b] = entry['gain'], entry['offset']
                continue
        if clean.sum() >= 2:
            gain_offset[b] = _affine(x[clean], y[clean])
        else:
            raise ValueError(f'Insufficient calibration data for bank {b}')
    gain, offset = gain_offset[np.asarray(study['row_bank_id'], dtype=int)].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain
Calibration archive structure: [1, {'episode_short_name': 'beta_m97_c03_s00', 'bank_id': [48, 0], 'ref_signal': [48, 1.264348], 'raw_readout': [48, 1.264348]}]
one current row ValueError Insufficient calibration data for bank 0
three current rows, one spike maximum change in calibrated measurements: 1.2559649942289752e-07
twelve current rows, one spike maximum change in calibrated measurements: 7.804789259324707e-09
QP worst feasibility, tangent gap, complementarity: 8.881784197001252e-16 2.2737367544323206e-13 1.2354188710574178e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cec93a7a02549493 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_cec93a7a (directory).
```

### Turn 8 — `execute_in_remote_docker` (`45857 ms` LLM)
- **Rationale**: Fix the observed archive parsing failure without changing the verified transport or optimizer. Exercise missing rows and EMI across all four banks, then verify each returned field, weight vector, and certificate against the public reference.
- **Alternatives Weighed**: `Use fixed gain and offset values as a fallback` (rejected: The actual archive contains per-episode sweep rows; fitting them preserves bank-specific calibration without hard-coding coefficients.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_batch_01_solver.py')
s = p.read_text()
start = s.index('            entries = history.get(')
end = s.index('        if clean.sum() >= 2:', start)
s = s[:start] + '''            for sweep in reversed(history):
                rows = np.asarray(sweep['bank_id']) == b
                hx = np.asarray(sweep['ref_signal'], dtype=float)[rows]
                hy = np.asarray(sweep['raw_readout'], dtype=float)[rows]
                usable = _clean_sweep(hx, hy)
                if usable.sum() >= 2 and np.ptp(hx[usable]) > 0:
                    gain_offset[b] = _affine(hx[usable], hy[usable])
                    break
            else:
                sweep = None
            if sweep is not None:
                continue
''' + s[end:]
s = s.replace('''                    break
            else:
                sweep = None
            if sweep is not None:
                continue
        if clean.sum() >= 2:''', '''                    break
            else:
                if clean.sum() < 2:
                    raise ValueError(f'Insufficient calibration data for bank {b}')
                gain_offset[b] = _affine(x[clean], y[clean])
            continue
        if clean.sum() >= 2:''')
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver','beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with m._ARCHIVE.open() as f: history = json.load(f)
with open('public_beta_m97_c03_batch_01.pkl','rb') as f: cases = pickle.load(f)
max_weight = max_field = max_fallback = max_gap = 0.0
for case in cases:
    s, gt = case['study'], case['gt']; baseline = m._calibrate(s)
    sw = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    hist = history[-1]
    slices = [s]
    for b in range(4):
        hr = np.asarray(hist['bank_id']) == b
        hx = np.asarray(hist['ref_signal'])[hr]; hy = np.asarray(hist['raw_readout'])[hr]
        gain, off = np.linalg.lstsq(np.column_stack((hx,np.ones(hx.size))), hy, rcond=None)[0]
        idx = np.flatnonzero(sw['bank_id'] == b)
        rows = np.asarray(s['row_bank_id']) == b
        for count, emi in [(0,False),(1,False),(4,True),(len(idx),True)]:
            keep = np.ones(sw['bank_id'].size, dtype=bool); keep[idx[count:]] = False
            sweep = {k:v[keep].copy() for k,v in sw.items()}
            if count <= 4:
                sweep['raw_readout'][sweep['bank_id']==b] = gain*sw['ref_signal'][idx[:count]] + off
                raw = np.asarray(s['y_obs_raw']).copy(); raw[rows] = gain*baseline[rows] + off
            else:
                raw = s['y_obs_raw']
            if emi: sweep['raw_readout'][np.flatnonzero(sweep['bank_id']==b)[0]] += 5.0
            st = dict(s, sensor_calibration_sweeps=sweep, y_obs_raw=raw)
            max_fallback = max(max_fallback, np.max(np.abs(m._calibrate(st)-baseline)))
            slices.append(st)
    for st in slices:
        out = m.solve_beta_m97_c03_batch_01(st); w = out['weights']; cert = out['kkt_certificate']
        max_weight = max(max_weight, np.max(np.abs(w-gt['w_opt'])))
        max_field = max(max_field, np.max(np.abs(out['terminal_field']-gt['terminal_field_A'])))
        g = gt['S_tilde'].T @ (gt['S_tilde'] @ w - gt['r_y']) + st['source_ridge']*w
        d = g - cert['nu_sum']; max_gap = max(max_gap, d@w - st['u_box']*np.minimum(d,0).sum())
        assert np.all(w >= 0) and np.all(w <= st['u_box']) and abs(w.sum()-st['total_budget']) < 1e-10
        assert cert['primal_objective'] - cert['dual_lower_bound'] < 1e-10
print('Worst bank fallback/EMI measurement error:', max_fallback)
print('Worst public-derived weight, field, and reference tangent gap:', max_weight,max_field,max_gap)
assert max_fallback < 1e-6 and max_weight < 1e-5 and max_field < 1e-5 and max_gap < 2e-4
PY
```
- **Container Exit Code**: `0` (`801 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Worst bank fallback/EMI measurement error: 4.481262294575572e-08
Worst public-derived weight, field, and reference tangent gap: 3.139651009453903e-07 2.429105618040328e-08 7.409739188324913e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9ba5a995ec38f2a2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9ba5a995 (directory).
```

### Turn 9 — `submit_completed_task` (`17269 ms` LLM)
- **Rationale**: The final artifact passed the isolated public validator and all-bank calibration stress tests. The archive fix eliminated the observed fallback failure while keeping weight, field, and certificate errors below the validation thresholds.
- **Alternatives Weighed**: `Further modify the transport operator or optimizer` (rejected: The corrected transport matches the public forward map to floating-point precision, and the optimizer passed the active-bound and budget-endpoint stress checks.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_01_solver.py': 'FILE:8031'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c03_batch_01_solver.py`. The public validator passed. Across all four banks’ missing-sweep and EMI stress cases, the maximum weight error was 3.14e-7, the maximum terminal-field error was 2.43e-8, and the worst reference KKT gap was 7.41e-8.`

### Turn 10 — `execute_in_remote_docker` (`39701 ms` LLM)
- **Rationale**: Check whether near-threshold EMI biases a bank despite four clean points; a failure would justify changing only sweep selection. Compare calibrated and uncalibrated residuals by bank on real measurements and verify that the vortex trace changes the forward map only in mode 0.
- **Alternatives Weighed**: `Add rational or drift terms and fit a batch classifier` (rejected: This is a specified reaction–transport inversion, not a stochastic control or classification task. The projected forward matrix already matches to floating-point precision; bank-wise projected residuals are the relevant nuisance diagnostic.)

```bash
set -e
cd /holdout_task
cat > /tmp/beta_m97_c03_calibration_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cases = pickle.load(open('/holdout_task/public_beta_m97_c03_batch_01.pkl', 'rb'))
for c in cases:
    s, g = c['study'], c['gt']
    sw = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    z = s['Z_nuisance']; gram = z.T@z + s['nuisance_ridge']*np.eye(z.shape[1])
    def profile(v):
        return v - z@np.linalg.solve(gram,z.T@v)
    a, _ = m._forward(s)
    print('projected design max error:', np.max(np.abs(profile(a)-g['S_tilde'])))
    y = m._calibrate(s); bank = np.asarray(s['row_bank_id']); raw = np.asarray(s['y_obs_raw'])
    for b in range(4):
        rows = bank == b
        print('bank',b,'profiled residual max (calibrated, uncalibrated):',*[np.max(np.abs(profile(v)[rows]-g['r_y'][rows])) for v in (y,raw)])
    # The documented direct-fit guarantee starts at four clean rows. Check that it
    # holds without history for both signs and every location of a near-threshold spike.
    worst = (0.0, None); failures = []
    for b in range(4):
        idx = np.flatnonzero(sw['bank_id']==b)
        for n in (5, len(idx)):
            for selection in (idx[:n], idx[-n:]):
                keep = (sw['bank_id']!=b); keep[selection] = True
                local = {k:v[keep].copy() for k,v in sw.items()}
                targets = np.flatnonzero(local['bank_id']==b)
                for j in targets:
                    for pulse in (-3.01, 3.01):
                        trial = dict(local,raw_readout=local['raw_readout'].copy())
                        trial['raw_readout'][j] += pulse
                        dev = np.max(np.abs(m._calibrate(dict(s,sensor_calibration_sweeps=trial))-y))
                        if dev > worst[0]: worst = (dev,(b,n,int(j),pulse))
                        if dev > 1e-6: failures.append((b,n,int(j),pulse,float(dev)))
    print('worst directly recoverable EMI case:',worst,'failures:',len(failures),'first:',failures[:3])
    v = np.asarray(s['past_vortex_circulation_trace']); critical = s['critical_vortex_circulation']
    for mode in (0,1):
        f = m._forward(dict(s,recirculation_loop_mode=mode,past_vortex_circulation_trace=v))[0]
        f0 = m._forward(dict(s,recirculation_loop_mode=mode,past_vortex_circulation_trace=np.full_like(v,critical)))[0]
        print('recirculation mode',mode,'above-vs-at-threshold forward response:',np.max(np.abs(f-f0)))
        assert (np.max(np.abs(f-f0)) > 0) == (mode == 0)
PY
python3 -I /tmp/beta_m97_c03_calibration_audit.py
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1215 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
projected design max error: 9.71445146547012e-17
bank 0 profiled residual max (calibrated, uncalibrated): 6.811176578302991e-08 0.5605337788291973
bank 1 profiled residual max (calibrated, uncalibrated): 1.7277626396250412e-07 0.744108245064532
bank 2 profiled residual max (calibrated, uncalibrated): 7.536757573589625e-08 0.5404381506145173
bank 3 profiled residual max (calibrated, uncalibrated): 1.779399241286228e-07 0.3818587035482599
worst directly recoverable EMI case: (3.0862972066891814e-07, (0, 5, 2, -3.01)) failures: 0 first: []
recirculation mode 0 above-vs-at-threshold forward response: 0.032100865866985864
recirculation mode 1 above-vs-at-threshold forward response: 0.0
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9ba5a995ec38f2a2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9ba5a995 (directory).
```

### Turn 11 — `submit_completed_task` (`19161 ms` LLM)
- **Rationale**: The verified artifact passed the public validator, independent field and certificate comparisons, active-bound inversion stress, and calibration fallback tests. The follow-up audit confirmed that vortex memory affects the forward map only in mode 0. No further edit is supported by the measured results.
- **Alternatives Weighed**: `Add more transport terms or change the sweep estimator` (rejected: The projected forward matrix matches the reference to floating-point precision, and the follow-up audit found no near-threshold EMI failures.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_01_solver.py': 'FILE:8031'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c03_batch_01_solver.py`. The isolated public validator passed. Across all four banks’ missing-sweep and EMI tests, the maximum weight error was 3.14e-7, the maximum terminal-field error was 2.43e-8, and the worst reference KKT gap was 7.41e-8. The follow-up audit passed for both recirculation modes and near-threshold EMI.`

### Turn 12 — `execute_in_remote_docker` (`32882 ms` LLM)
- **Rationale**: Use held-out real sweep readings to test the minimum clean-data guarantee bank by bank. A held-out calibration failure would isolate the affine fit; accurate calibration with a failed weight, field, or certificate margin would instead expose sensitivity downstream. The prior projected-design and mode checks already cover the specified transport algebra.
- **Alternatives Weighed**: `Repeat the full-sweep calibration and two-mode transport checks` (rejected: Those passed on this artifact; neither checks held-out prediction from exactly four clean current rows with archive access disabled.)

```bash
set -e
cd /holdout_task
cat > /tmp/beta_m97_c03_sweep_lobo.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class NoArchive:
    def open(self, *args, **kwargs):
        raise AssertionError('Clean current sweeps must not consult the archive')

m._ARCHIVE = NoArchive()
with open('/holdout_task/public_beta_m97_c03_batch_01.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    s, gt = case['study'], case['gt']
    sweeps = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    bank, x, raw = (sweeps[k] for k in ('bank_id', 'ref_signal', 'raw_readout'))
    worst = np.zeros((4, 4))
    for b in range(4):
        idx = np.flatnonzero(bank == b)
        for held in idx:
            pool = idx[idx != held]
            for selection in (pool, pool[:4], pool[-4:]):
                keep = bank != b
                keep[selection] = True
                sweep = {k: v[keep] for k, v in sweeps.items()}
                probe = dict(s, sensor_calibration_sweeps=sweep)
                recovered = m._calibrate(dict(probe, y_obs_raw=raw[held:held+1], row_bank_id=bank[held:held+1]))[0]
                out = m.solve_beta_m97_c03_batch_01(probe)
                w = np.asarray(out['weights'])
                d = gt['S_tilde'].T @ (gt['S_tilde'] @ w - gt['r_y']) + s['source_ridge']*w - out['kkt_certificate']['nu_sum']
                worst[b] = np.maximum(worst[b], [abs(recovered-x[held]), np.max(np.abs(w-gt['w_opt'])), np.max(np.abs(out['terminal_field']-gt['terminal_field_A'])), d@w-s['u_box']*np.minimum(d,0).sum()])
        print('bank', b, 'max held-out reference, weight, field, KKT gap:', *worst[b])
    assert np.all(worst < [1e-5, 1e-5, 1e-5, 2e-4]), worst
PY
python3 -I /tmp/beta_m97_c03_sweep_lobo.py
```
- **Container Exit Code**: `0` (`4974 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank 0 max held-out reference, weight, field, KKT gap: 2.0564958336244388e-07 4.5791376923620675e-07 3.6859698043789635e-08 2.7659056770817196e-07
bank 1 max held-out reference, weight, field, KKT gap: 3.243953211606865e-07 5.188412421780475e-07 3.249881752598327e-08 3.705993904390353e-07
bank 2 max held-out reference, weight, field, KKT gap: 2.3476054433757554e-07 3.2739744804111126e-07 2.3770799092992512e-08 8.35641212610903e-08
bank 3 max held-out reference, weight, field, KKT gap: 5.540685388272237e-07 4.1766298092271725e-07 3.292921304381968e-08 3.052478190004759e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9ba5a995ec38f2a2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9ba5a995 (directory).
```

### Turn 13 — `submit_completed_task` (`15551 ms` LLM)
- **Rationale**: The unchanged artifact passed the isolated validator, independent field and certificate comparisons, and targeted stress tests for calibration, boundary modes, and active bounds. The held-out audit confirmed the minimum clean-sweep guarantee without relying on the archive.
- **Alternatives Weighed**: `Change the estimator after the held-out sweep audit` (rejected: All four banks passed with archive access disabled and as few as four clean current rows; no tested slice exposed a failure.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_01_solver.py`. The isolated public validator passed. Across the held-out calibration tests, the maximum absolute weight and terminal-field errors were 5.19e-7 and 3.69e-8, and the worst reference KKT gap was 3.71e-7 against a 2e-4 tolerance. Missing-sweep fallback, EMI handling, and both recirculation modes also passed.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
