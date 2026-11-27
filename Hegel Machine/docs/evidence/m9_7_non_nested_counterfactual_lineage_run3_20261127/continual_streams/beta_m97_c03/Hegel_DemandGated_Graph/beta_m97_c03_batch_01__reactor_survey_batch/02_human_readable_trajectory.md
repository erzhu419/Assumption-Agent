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
| **Total Tokens (Prompt / Cached / Out)** | `346,609` (`334,582` / `92,032` / `12,027`) |
| **Total Episode Wall Time (s)** | `398.18s` |

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

### Turn 1 — `execute_in_remote_docker` (`14386 ms` LLM)
- **Rationale**: Inspect the public contract and prior solver to distinguish reusable discretization from boundary terms that must change, and look up the requested ridge profiling primitive.

```bash
cd /holdout_task && ls -l /holdout_task /tmp/hegel_continual_archive && cat validate_public_schema.py && cat public_asset_index.json && sed -n '1,280p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`301 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`14414 ms` LLM)
- **Rationale**: The prior output truncated the forward assembly. Read it directly to identify the boundary rows, axis order, and sensor stacking to preserve, and inspect the history format before implementing the permitted calibration fallback.
- **Alternatives Weighed**: `Copy the archived solver unchanged` (rejected: The current contract explicitly requires a sealed-boundary wall-sorption branch and in-batch calibration.)

```bash
cd /holdout_task && grep -n '^def ' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py validate_public_schema.py && sed -n '65,240p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py && cat /tmp/hegel_continual_archive/archive_calibration_history.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`348 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ecution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
        "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
        "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
      },
      "functions": [
        {
          "function": "fit_ridge_nuisance",
          "signature": "fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
        }
      ],
      "index_cards": [
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`16896 ms` LLM)
- **Rationale**: Bank row counts and sweep values determine whether to fit directly or invoke the permitted archive fallback. The forward assembly and validator excerpts resolve the sensor stacking, axis order, and KKT multiplier convention before implementation.

```bash
cd /holdout_task && sed -n '1,116p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py && sed -n '35,105p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_batch_01.pkl','rb') as f: s=pickle.load(f)
for k,v in s.items():
    print(k, v if np.isscalar(v) else list(v) if isinstance(v,dict) else np.shape(v))
c=s['sensor_calibration_sweeps']; b,x,y=[np.asarray(c[k]) for k in ('bank_id','ref_signal','raw_readout')]
for i in range(4):
    m=b==i
    print('calibration bank',i,'points',np.count_nonzero(m),'pairs',list(zip(x[m],y[m])))
print('v_mem',np.sum(np.maximum(0,np.asarray(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation'])**2))
PY
```
- **Container Exit Code**: `1` (`369 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lid": False, "reason": "negative_bound_multipliers", "gap": 1e9}

    w_feas = np.clip(w, 0.0, u_b)
    res = S @ w_feas - ry
    j_true = 0.5 * float(res @ res) + 0.5 * lw * float(w_feas @ w_feas)
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
    elif not isinstance(out["kkt_certificate"], dict) or not {"weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"} <= set(out["kkt_certificate"].keys()):
        errs.append("bad_kkt_certificate_keys")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 4 — `execute_in_remote_docker` (`21411 ms` LLM)
- **Rationale**: The pickle is a list of cases, not a study dict. Inspect its nested sweep counts and archive schema to implement the correct calibration fallback, and read the archived optimizer to see whether it can certify both interior and bound-active solutions.

```bash
cd /holdout_task && sed -n '117,220p' /tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py && python3 - <<'PY'
import json,pickle,numpy as np
items=pickle.load(open('public_beta_m97_c03_batch_01.pkl','rb'))
print('cases',len(items))
for item in items:
 s=item['study']; print('study', {k:(v if np.isscalar(v) else list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()}); print('gt keys',list(item['gt']))
 c=s['sensor_calibration_sweeps']; b,x,y=[np.asarray(c[k]) for k in ('bank_id','ref_signal','raw_readout')]
 for i in range(4):
  m=b==i; print('bank',i,'pairs',list(zip(x[m],y[m])))
 print('memory',np.maximum(0,np.asarray(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation']) .__pow__(2).sum())
a=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history structure',[(type(r).__name__,list(r) if isinstance(r,dict) else None) for r in a] if isinstance(a,list) else list(a))
print(open('public_asset_index.json').read()[:5600])
PY
```
- **Container Exit Code**: `0` (`362 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pe": [
                16
              ],
              "type": "ndarray"
            },
            "reaction_x": {
              "dtype": "float64",
              "shape": [
                18
              ],
              "type": "ndarray"
            },
            "reaction_y": {
              "dtype": "float64",
              "shape": [
                16
              ],
              "type": "ndarray"
            },
            "recirculation_loop_mode": 0.0,
            "robin_x": 0.85,
            "robin_y": 0.75,
            "row_bank_id": {
              "dtype": "int64",
              "shape": [
                144
              ],
              "type": "ndarray"
            },
            "sensor_calibration_sweeps": {
              "bank_id": {
                "dtype": "int64",
                "shape": [
                  48
                ],
                "type": "ndarray"
              },
              "raw_readout": {
                "dtype": "float64",
                "shape": [
                  48
                ],
                "type": "ndarray"
              },
              "ref_signal": {
                "dtype": "float64",
                "shape": [
                  48
                ],
                "type": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
                36,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                12,
                18,
                16
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 2.6,
            "u_box": 1.0,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                18
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                16
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                144
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 4702
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
```

### Turn 5 — `execute_in_remote_docker` (`60325 ms` LLM)
- **Rationale**: Implement the specified boundary and observation operators while preserving the archived grid and sensor ordering. The short archive summary will reveal any history-layout mismatch; the isolated schema run will catch input-key, shape, or import errors before comparison with the public reference.
- **Alternatives Weighed**: `Use archived gain and offset for every bank` (rejected: The contract requires current in-batch coefficients when enough clean measurements exist.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; a=json.load(open(p)); print('history:',type(a).__name__, [(list(v), {k:np.shape(x) for k,x in v.items()}) for v in a] if isinstance(a,list) else list(a))
it=pickle.load(open('public_beta_m97_c03_batch_01.pkl','rb'))[0]; s=it['study']; print('gt:',[(k,np.shape(v)) for k,v in it['gt'].items()]); c=s['sensor_calibration_sweeps']; b,x,y=[np.asarray(c[k]) for k in ('bank_id','ref_signal','raw_readout')]; print('banks:',[(i,int(sum(b==i)),np.polyfit(x[b==i],y[b==i],1).tolist(),float(np.max(np.abs(y[b==i]-np.polyval(np.polyfit(x[b==i],y[b==i],1),x[b==i]))))) for i in range(4)])
PY
cat > /tmp/build_batch01.py <<'PY'
from pathlib import Path
archive = Path('/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py').read_text()
axis = archive[archive.index('def _axis('):archive.index('def _forward(')]
qp = archive[archive.index('def _box_simplex('):archive.index('def solve_beta_m97_c03_s00(')]
Path('/holdout_task/beta_m97_c03_batch_01_solver.py').write_text('''"""Calibrated reaction-transport inversion with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_sweep(x, y):
    """Keep sweep points within the EMI threshold of the consensus line."""
    if x.size < 3:
        return np.ones(x.size, dtype=bool)
    i, j = np.triu_indices(x.size, 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    intercepts = y[i] - slopes * x[i]
    error = np.abs(y[None, :] - slopes[:, None] * x[None, :] - intercepts[:, None])
    inlier = error <= 3.0
    count = inlier.sum(axis=1)
    best = np.lexsort((np.where(inlier, error**2, 0).sum(axis=1), -count))[0]
    return inlier[best]


def _archived_fit(bank):
    for sweep in reversed(json.loads(_HISTORY.read_text())):
        b = np.asarray(sweep['bank_id'])
        x = np.asarray(sweep['ref_signal'], dtype=float)[b == bank]
        y = np.asarray(sweep['raw_readout'], dtype=float)[b == bank]
        if np.unique(x).size >= 2:
            keep = _clean_sweep(x, y)
            return _fit_line(x[keep], y[keep])
    raise ValueError(f'No archived calibration for bank {bank}')


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    b = np.asarray(sweep['bank_id'])
    x = np.asarray(sweep['ref_signal'], dtype=float)
    y = np.asarray(sweep['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    coeff = np.empty((4, 2))
    for bank in np.unique(row_banks):
        xb, yb = x[b == bank], y[b == bank]
        keep = _clean_sweep(xb, yb)
        if xb.size < 2 or (not keep.all() and keep.sum() < 4):
            coeff[bank] = _archived_fit(bank)
        else:
            coeff[bank] = _fit_line(xb[keep], yb[keep])
    gain, offset = coeff[row_banks].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


''' + axis + '''def _forward(study):
    nx, ny = study['nx'], study['ny']
    lx, ly = study['Lx'], study['Ly']
    v = np.asarray(study['past_vortex_circulation_trace'])
    v_mem = np.sum(np.maximum(0.0, v - study['critical_vortex_circulation'])**2)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                    0.0 if sealed else study['robin_' + a], study['L' + a])
              for a in ('x', 'y')]
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    op = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
          + diags(reaction.ravel(order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_rows = i + nx * j
    identity = eye(nx * ny, format='csc')
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * identity).solve(modes)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field


''' + qp + '''def solve_beta_m97_c03_batch_01(study: dict) -> dict:
    y = _calibrate(study)
    sensors, terminal = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    data = np.column_stack((y, sensors))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    residualized = data - z @ np.linalg.solve(gram, z.T @ data)
    target, design = residualized[:, 0], residualized[:, 1:]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    w, nu = _box_simplex(hessian, -design.T @ target, cap, budget)
    residual = design @ w - target
    gradient = design.T @ residual + ridge * w
    reduced = gradient - nu
    ll = np.where(w == 0.0, np.maximum(reduced, 0.0), 0.0)
    lu = np.where(w == cap, np.maximum(-reduced, 0.0), 0.0)
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    # Algebraically identical to the tangent lower bound, without cancellation
    # between the equality term and the full gradient inner product.
    gap = np.maximum(reduced, 0) @ w + np.maximum(-reduced, 0) @ (cap - w)
    certificate = {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': ll.tolist(),
                   'lam_upper': lu.tolist(), 'primal_objective': primal,
                   'dual_lower_bound': float(primal - gap)}
    field = (terminal @ w).reshape((study['nx'], study['ny']), order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': certificate}
''')
PY
python3 /tmp/build_batch01.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`527 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
history: list [(['episode_short_name', 'bank_id', 'ref_signal', 'raw_readout'], {'episode_short_name': (), 'bank_id': (48,), 'ref_signal': (48,), 'raw_readout': (48,)})]
gt: [('S_tilde', (144, 12)), ('r_y', (144,)), ('w_opt', (12,)), ('cert_opt', ()), ('terminal_field_A', (18, 16))]
banks: [(0, 12, [1.2699999880126034, 0.12999999408410204], 1.2839447949275495e-07), (1, 12, [0.7800000690169956, -0.15999999688734862], 2.04497461853137e-07), (2, 12, [1.1800000342817336, -0.0900000655104666], 2.0496489483434743e-07), (3, 12, [0.8399999466442427, 0.10999998897348956], 2.456033549869119e-07)]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=27786e673d1e35bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_27786e67 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29707 ms` LLM)
- **Rationale**: Compare the design, observations, weights, and terminal field separately with the public references to localize any discrepancy. Verify the certificate against the reference objective, then perturb one bank at a time to detect fallback failures or EMI masking that the clean public sweep cannot reveal.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: Shape and import checks do not verify the boundary operator, reference-field accuracy, or KKT certificate.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import importlib.util,json,pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('candidate',p/'beta_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d = ')[0],ns); verify=ns['verify_box_simplex_kkt_certificate']
for item in pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes()):
 s,g=item['study'],item['gt']; out=m.solve_beta_m97_c03_batch_01(s); c=out['kkt_certificate']; y=m._calibrate(s); S,_=m._forward(s); Z=s['Z_nuisance']; data=np.column_stack((y,S)); r=data-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@data)
 print('public', {k:float(np.max(np.abs(a-b))) for k,a,b in [('design_max_abs',r[:,1:],g['S_tilde']),('target_max_abs',r[:,0],g['r_y']),('weight_max_abs',out['weights'],g['w_opt']),('field_max_abs',out['terminal_field'],g['terminal_field_A'])]})
 for label,A,q in [('candidate',r[:,1:],r[:,0]),('reference',g['S_tilde'],g['r_y'])]:
  print(label,verify(A,q,s['source_ridge'],s['u_box'],s['total_budget'],c))
 cal=s['sensor_calibration_sweeps']; banks=cal['bank_id']; print('historical line fits',[m._archived_fit(i).tolist() for i in range(4)])
 max_error={}; failures=[]
 for bank in range(4):
  idx=np.flatnonzero(banks==bank); row=s['row_bank_id']==bank
  for mode in ('absent','one_point','emi_just_over_threshold','emi_large','emi_sparse'):
   errs=[]
   for i in idx:
    t={k:np.array(v,copy=True) for k,v in cal.items()}; keep=np.ones(banks.size,dtype=bool)
    if mode in ('absent','one_point'):
     keep[idx]=False
     if mode=='one_point': keep[i]=True
    else:
     t['raw_readout'][i]+=3.01 if mode=='emi_just_over_threshold' else 12.0
     if mode=='emi_sparse':
      keep[idx]=False; keep[np.r_[i,idx[idx!=i][:3]]]=True
    trial=dict(s,sensor_calibration_sweeps={k:v[keep] for k,v in t.items()})
    try: errs.append(float(np.max(np.abs(m._calibrate(trial)[row]-y[row]))))
    except (ValueError,IndexError) as e: failures.append((bank,mode,str(e)))
   if errs: max_error[mode]=max(max_error.get(mode,0),max(errs))
 print('bank stress max calibrated error:',max_error,'exceptions:',failures[:3])
PY
```
- **Container Exit Code**: `0` (`526 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public {'design_max_abs': 2.220446049250313e-16, 'target_max_abs': 1.779399241286228e-07, 'weight_max_abs': 2.923248668407208e-07, 'field_max_abs': 2.184216425088481e-08}
candidate {'valid': True, 'reason': 'ok', 'true_primal': 0.018398626559021376, 'verified_dual_lb': 0.018398626559021324, 'gap': 5.551115123125783e-17}
reference {'valid': True, 'reason': 'ok', 'true_primal': 0.018398630811939706, 'verified_dual_lb': 0.018398566393665057, 'gap': 6.441827464856442e-08}
historical line fits [[1.0000000000000004, -9.165413721101009e-17], [1.0, -6.477350574728079e-17], [1.0, 6.23307698252489e-17], [0.9999999999999996, 1.0143016187198231e-18]]
bank stress max calibrated error: {'absent': 0.7879345677684477, 'one_point': 0.7879345677684477, 'emi_just_over_threshold': 3.2250435423420973, 'emi_large': 1.4065350484315786e-07, 'emi_sparse': 1.9736921054295078} exceptions: []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=27786e673d1e35bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_27786e67 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`45267 ms` LLM)
- **Rationale**: Fix the observed threshold-masking failure without changing the forward operator, which matches the public design to machine precision. Sweep all four banks, both spike signs, and supported row counts to distinguish EMI classification failures from expected archive drift.
- **Alternatives Weighed**: `Treat archive fallback error against current gain and offset as a solver error` (rejected: The archive has different coefficients and is the required fallback when current calibration is unidentifiable. Validate that branch against the archived fit.)
- **ATMS NOGOOD DECLARED**: Refuted `Maximizing the number of points within the EMI threshold identifies the clean affine calibration line.` | Witness: `A single +3.01 spike on the public sweep produced calibrated row error up to 3.225, even though the clean affine fits had residuals below 2.46e-7.` | Blanket Diagnosis: `The spike can tilt a candidate line enough to bring itself under the threshold while leaving the other points nominally inliers.` | Sublated Pivot: `Estimate the affine line from a clean majority first, then apply the EMI threshold and use the archive only when the current sweep is insufficient.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_batch_01_solver.py')
s = p.read_text()
old = '''    inlier = error <= 3.0
    count = inlier.sum(axis=1)
    best = np.lexsort((np.where(inlier, error**2, 0).sum(axis=1), -count))[0]
    return inlier[best]
'''
new = '''    if x.size == 3:
        # Two points alone cannot identify the contaminated point. A third
        # point departing by more than the threshold triggers archive fallback.
        return error[np.argmax(error.max(axis=1))] <= 3.0
    # Estimate the line from a clean majority before applying the EMI threshold.
    # Counting threshold hits first can favor a tilted line that hides a spike.
    core_size = x.size // 2 + 1
    core = np.argpartition(error, core_size - 1, axis=1)[:, :core_size]
    core_error = np.take_along_axis(error, core, axis=1)
    best = np.argmin(np.sum(core_error**2, axis=1))
    gain, offset = _fit_line(x[core[best]], y[core[best]])
    return np.abs(y - (gain * x + offset)) <= 3.0
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task'); f=p/'beta_m97_c03_batch_01_solver.py'
spec=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d = ')[0],ns)
it=pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes())[0]; s,g=it['study'],it['gt']; out=m.solve_beta_m97_c03_batch_01(s)
verify=ns['verify_box_simplex_kkt_certificate']; cert=verify(g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']); print('public reference certificate',cert); assert cert['valid']
for k,r in [('weights','w_opt'),('terminal_field','terminal_field_A')]:
 err=np.max(np.abs(out[k]-g[r])); print(k,'max absolute error',err); assert err<1e-6
cal=s['sensor_calibration_sweeps']; banks=cal['bank_id']; raw=s['y_obs_raw']; baseline=m._calibrate(s); maxima={}
for bank in range(4):
 idx=np.flatnonzero(banks==bank); row=s['row_bank_id']==bank
 gain,offset=m._archived_fit(bank); expected_archive=(raw[row]-offset)/gain
 for n in [0,1,2,3,4,5,len(idx)]:
  for i in idx:
   retained=np.r_[i,idx[idx!=i]][:n]
   for spike in ([0] if n<3 else [0,-12,-3.01,3.01,12]):
    t={k:np.array(v,copy=True) for k,v in cal.items()}; keep=banks!=bank; keep[retained]=True
    t['raw_readout'][i]+=spike
    use_archive=n<2 or (spike!=0 and n<5)
    expected=expected_archive if use_archive else baseline[row]
    trial=dict(s,sensor_calibration_sweeps={k:v[keep] for k,v in t.items()})
    err=float(np.max(np.abs(m._calibrate(trial)[row]-expected)))
    label=('archive' if use_archive else 'in-batch',n,'EMI' if spike else 'clean'); maxima[label]=max(maxima.get(label,0),err)
print('bank stress (expected in-batch or archive)',maxima)
assert max(maxima.values())<1e-5
```
- **Container Exit Code**: `1` (`1231 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public reference certificate {'valid': True, 'reason': 'ok', 'true_primal': 0.018398630811939706, 'verified_dual_lb': 0.018398566393665057, 'gap': 6.441827464856442e-08}
weights max absolute error 2.923248668407208e-07
terminal_field max absolute error 2.184216425088481e-08
bank stress (expected in-batch or archive) {('archive', 0, 'clean'): 0, ('archive', 1, 'clean'): 0, ('in-batch', 2, 'clean'): 0.00018486779178417123, ('in-batch', 3, 'clean'): 3.6623033330407395e-06, ('archive', 3, 'EMI'): 0, ('in-batch', 4, 'clean'): 4.1027561792361666e-07, ('archive', 4, 'EMI'): 0, ('in-batch', 5, 'clean'): 3.212402743812959e-07, ('in-batch', 5, 'EMI'): 3.0862972066891814e-07, ('in-batch', 12, 'clean'): 0, ('in-batch', 12, 'EMI'): 1.4065350484315786e-07}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a7893dd56e5b3d52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a7893dd5 (directory).
```
- **Stderr (tail)**:
```text
bash: line 56: warning: here-document at line 29 delimited by end-of-file (wanted `PY')
Traceback (most recent call last):
  File "<stdin>", line 27, in <module>
AssertionError
```

### Turn 8 — `execute_in_remote_docker` (`37414 ms` LLM)
- **Rationale**: The prior check compared a sparse calibration fit with the full-sweep fit. Verify the two-point branch against an independent affine calculation instead; a mismatch would warrant a calibration fix. Check projected KKT residuals at vertex, near-endpoint, and mixed-bound budgets, where a failure would identify an active-set defect. No candidate change is needed unless these checks fail.
- **Alternatives Weighed**: `Use the archive for close or noisy two-point sweeps` (rejected: The contract allows archive use only below two points or when EMI is detected. A two-point fit should match the affine line through those measurements, not the 12-point fit.)

```bash
set -e
cd /holdout_task
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('candidate',p/'beta_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d = ')[0],ns)
it=pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes())[0]; s,g=it['study'],it['gt']; out=m.solve_beta_m97_c03_batch_01(s)
cert=ns['verify_box_simplex_kkt_certificate'](g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']); assert cert['valid'],cert
errs={k:float(np.max(np.abs(out[k]-g[r]))) for k,r in [('weights','w_opt'),('terminal_field','terminal_field_A')]}; assert max(errs.values()) < 1e-6,errs
print('public errors',errs,'reference KKT gap',cert['gap'])
cal=s['sensor_calibration_sweeps']; banks=cal['bank_id']; raw=s['y_obs_raw']; baseline=m._calibrate(s); worst={}
for bank in range(4):
 idx=np.flatnonzero(banks==bank); row=s['row_bank_id']==bank
 ah,ao=m._archived_fit(bank)
 for n in [0,1,2,3,4,5,len(idx)]:
  for i in idx:
   retained=np.r_[i,idx[idx!=i]][:n]
   for spike in ([0] if n<3 else [0,-12,-3.01,3.01,12]):
    t={k:np.array(v,copy=True) for k,v in cal.items()}; keep=banks!=bank; keep[retained]=True; t['raw_readout'][i]+=spike
    if n<2 or (spike and n<5):
     expected=(raw[row]-ao)/ah; label='archive'
    elif n==2:
     j,k=retained; x,y=t['ref_signal'],t['raw_readout']; h=(y[k]-y[j])/(x[k]-x[j]); o=y[j]-h*x[j]
     expected=(raw[row]-o)/h; label='two-point affine'
    else:
     expected=baseline[row]; label='EMI with sufficient clean points' if spike else 'clean in-batch'
    trial=dict(s,sensor_calibration_sweeps={k:v[keep] for k,v in t.items()}); err=float(np.max(np.abs(m._calibrate(trial)[row]-expected))); worst[label]=max(worst.get(label,0),err)
print('bank stress maximum absolute errors',worst)
assert max(worst['archive'],worst['two-point affine'])<1e-8,worst
assert max(worst['EMI with sufficient clean points'],worst['clean in-batch'])<1e-5,worst
rng=np.random.default_rng(731); worst_kkt=0.0
for cap in [0.2,1.0,1.7]:
 for budget in [0,cap*1e-5,cap,2.6*cap,6*cap,cap*(12-1e-5),12*cap]:
  for repeat in range(30):
   a=rng.normal(size=(8,12)); H=a.T@a+0.04*np.eye(12); c=rng.normal(size=12)*(30 if repeat%2 else 1)
   w,nu=m._box_simplex(H,c,cap,budget); d=H@w+c-nu
   residual=max(abs(w.sum()-budget),float(np.max(-w)),float(np.max(w-cap)),float(np.max(np.abs(w-np.clip(w-d,0,cap)))))
   worst_kkt=max(worst_kkt,residual)
print('active-set worst projected KKT residual across 630 cases',worst_kkt)
assert worst_kkt<1e-9,worst_kkt
PY
```
- **Container Exit Code**: `0` (`1514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public errors {'weights': 2.923248668407208e-07, 'terminal_field': 2.184216425088481e-08} reference KKT gap 6.441827464856442e-08
bank stress maximum absolute errors {'archive': 0, 'two-point affine': 2.795097486796294e-12, 'clean in-batch': 3.6623033330407395e-06, 'EMI with sufficient clean points': 3.0862972066891814e-07}
active-set worst projected KKT residual across 630 cases 1.4210854715202004e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a7893dd56e5b3d52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a7893dd5 (directory).
```

### Turn 9 — `submit_completed_task` (`13719 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema and reference checks, calibration stress tests across all four banks, and active-set tests at endpoint and mixed-bound budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_01_solver.py': 'FILE:8291'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_01_solver.py`. The public schema and KKT certificate passed. Maximum absolute errors were 2.92e-7 for weights and 2.18e-8 for the terminal field. Across 630 box-simplex stress cases, the worst projected KKT residual was 1.42e-14.`

### Turn 10 — `execute_in_remote_docker` (`43491 ms` LLM)
- **Rationale**: Reassemble the operator in the opposite flattening convention with independently assigned Neumann endpoint rows and the specified wall term. A design or field mismatch would identify a boundary, indexing, or branch defect; a KKT mismatch would identify an inversion defect. Only the measured fouled slice has ground truth; the other mode and threshold slices test the equations, not predictive accuracy.
- **Alternatives Weighed**: `Fit extra rational terms or compare pooled and within-batch classification scores` (rejected: This is a specified linear inverse problem with one public batch and no classification labels; the relevant nuisance check is agreement with the supplied ridge-profiled observations and design.)

```bash
set -e
cat > /tmp/audit_batch01_transport.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p = Path('/holdout_task')
m = load('candidate', p/'beta_m97_c03_batch_01_solver.py')
a = load('prior', '/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py')
item = pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes())[0]
s, gt = item['study'], item['gt']
nx, ny = s['nx'], s['ny']
source = np.asarray(s['source_modes']).reshape(-1, nx*ny).T
ij = np.asarray(s['sensor_ij']); rows = ij[:,0]*ny + ij[:,1]
worst = dict(design=0., terminal=0., neumann=0., kkt=0.)
for mode, trace_name in [(0,'measured'), (1,'measured'), (0,'threshold')]:
    q = dict(s, recirculation_loop_mode=mode)
    if trace_name == 'threshold':
        q['past_vortex_circulation_trace'] = np.minimum(s['past_vortex_circulation_trace'], s['critical_vortex_circulation'])
    excess = [float(v)-s['critical_vortex_circulation'] for v in q['past_vortex_circulation_trace'] if v>s['critical_vortex_circulation']]
    memory = sum(t*t for t in excess)
    sealed = mode == 0 and bool(excess)
    axes = []
    for tag,n in [('x',nx),('y',ny)]:
        k = np.asarray(q['kappa_'+tag]); length = q['L'+tag]
        op = a._axis(k, np.asarray(q['vel_'+tag]), q['robin_'+tag], length).toarray()
        if sealed:
            for end,neighbor in [(0,1),(n-1,n-2)]:
                d = k[end]*((n-1)/length)**2
                op[end] = 0.; op[end,end] = d; op[end,neighbor] = -d
            worst['neumann'] = max(worst['neumann'], float(np.max(np.abs(op[[0,-1]].sum(axis=1)))))
    
        axes.append(op)
    x, y = np.meshgrid(np.linspace(0, q['Lx'], nx), np.linspace(0, q['Ly'], ny), indexing='ij')
    v = np.array(q['V_reaction_2d'], copy=True)
    if sealed:
        v += (0.95+0.55*np.tanh(memory))*(np.exp(-16*(x/q['Lx'])**2)+np.exp(-16*(1-x/q['Lx'])**2))*(0.6+0.4*np.sin(np.pi*y/q['Ly']))
    operator = np.kron(axes[0],np.eye(ny)) + np.kron(np.eye(nx),axes[1]) + np.diag(v.ravel())
    blocks = []
    for mu in q['mu_shifts']:
        dense = solve(operator+mu*np.eye(nx*ny),source)
        blocks.append(dense[rows])
    stack = np.vstack(blocks)
    design, _ = m._forward(q)
    out = m.solve_beta_m97_c03_batch_01(q)
    w = out['weights']; cap = q['u_box']; c = out['kkt_certificate']
    d_err = float(np.max(np.abs(stack-design)))
    f_err = float(np.max(np.abs((dense@w).reshape(nx,ny)-out['terminal_field'])))
    z = q['Z_nuisance']; obs = np.column_stack((m._calibrate(q),stack))
    r = obs-z@solve(z.T@z+q['nuisance_ridge']*np.eye(z.shape[1]),z.T@obs,assume_a='pos')
    d = r[:,1:].T@(r[:,1:]@w-r[:,0])+q['source_ridge']*w-c['nu_sum']
    k_err = max(abs(float(w.sum()-q['total_budget'])),float(np.max(np.abs(w-np.clip(w-d,0,cap)))))
    print('mode',mode,trace_name,'sealed',sealed,'memory',memory,'design/terminal/KKT',d_err,f_err,k_err)
    for key,val in [('design',d_err),('terminal',f_err),('kkt',k_err)]:
        worst[key] = max(worst[key],val)
    assert max(d_err,f_err,k_err)<1e-9, worst
    if sealed:
        nuisance_error = float(np.max(np.abs(r[:,0]-gt['r_y'])))
        design_error = float(np.max(np.abs(r[:,1:]-gt['S_tilde'])))
        print('measured profiled reference target/design errors',nuisance_error,design_error)
        assert max(nuisance_error,design_error)<1e-6
print('worst across mode and trace slices',worst)
assert worst['neumann']<1e-12
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_batch01_transport.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`578 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mode 0 measured sealed True memory 0.5560895101860001 design/terminal/KKT 9.71445146547012e-17 2.7755575615628914e-17 4.440892098500626e-16
measured profiled reference target/design errors 1.779399243506674e-07 1.8041124150158794e-16
mode 1 measured sealed False memory 0.5560895101860001 design/terminal/KKT 9.020562075079397e-17 2.7755575615628914e-17 4.440892098500626e-16
mode 0 threshold sealed False memory 0 design/terminal/KKT 9.020562075079397e-17 2.7755575615628914e-17 4.440892098500626e-16
worst across mode and trace slices {'design': 9.71445146547012e-17, 'terminal': 2.7755575615628914e-17, 'neumann': 0.0, 'kkt': 4.440892098500626e-16}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a7893dd56e5b3d52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a7893dd5 (directory).
```

### Turn 11 — `submit_completed_task` (`14561 ms` LLM)
- **Rationale**: The saved candidate passed clean-process schema validation, public-reference checks, calibration stress tests, and an independent audit of both recirculation modes and the fouling threshold. The audit found no defect requiring another change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_01_solver.py': 'FILE:8291'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_01_solver.py`. The public schema and KKT certificate passed. Maximum absolute errors were 2.92e-7 for source weights and 2.18e-8 for the terminal field. An independent transport assembly agreed within 9.72e-17 across both recirculation modes and at the fouling threshold.`

### Turn 12 — `execute_in_remote_docker` (`31713 ms` LLM)
- **Rationale**: Stress the full inversion with simultaneous missing-bank, one-point, and near-threshold EMI cases, rotating roles across the measured banks. A mismatch against independently fitted calibration or the reference-design certificate would identify a bank-indexing or nuisance-projection defect; an output change after removing auxiliary telemetry would reveal an unintended dependency.
- **Alternatives Weighed**: `Repeat the transport-mode audit or add classification tests` (rejected: Both transport modes already matched an independent assembly, and this single public inverse problem has no classification labels.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('candidate',p/'beta_m97_c03_batch_01_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d = ')[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
item=pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes())[0]
s,g=item['study'],item['gt']; cal=s['sensor_calibration_sweeps']; b,x,y=(np.asarray(cal[k]) for k in ('bank_id','ref_signal','raw_readout'))
a=json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())[-1]; ab,ax,ay=(np.asarray(a[k]) for k in ('bank_id','ref_signal','raw_readout'))
z=s['Z_nuisance']; gram=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); obs=s['y_obs_raw']; rb=s['row_bank_id']
worst_y=worst_gap=0.0
for turn in range(4):
    keep=np.ones(b.size,dtype=bool); raw=y.copy(); expected=np.empty_like(obs)
    for bank in range(4):
        i=np.flatnonzero(b==bank); role=(bank-turn)%4
        if role==0:
            keep[i]=False; j=ab==bank; h,o=np.polyfit(ax[j],ay[j],1)
        elif role==1:
            keep[i[1:]]=False; j=ab==bank; h,o=np.polyfit(ax[j],ay[j],1)
        elif role==2:
            contaminated=i[-1]; raw[contaminated]+=(-1)**turn*3.01; j=i[:-1]; h,o=np.polyfit(x[j],y[j],1)
        else:
            h,o=np.polyfit(x[i],y[i],1)
        expected[rb==bank]=(obs[rb==bank]-o)/h
    q=dict(s,sensor_calibration_sweeps=dict(bank_id=b[keep],ref_signal=x[keep],raw_readout=raw[keep]))
    actual=m._calibrate(q); worst_y=max(worst_y,float(np.max(np.abs(actual-expected))))
    target=expected-z@np.linalg.solve(gram,z.T@expected)
    out=m.solve_beta_m97_c03_batch_01(q)
    cert=verify(g['S_tilde'],target,s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']); assert cert['valid'],cert
    worst_gap=max(worst_gap,cert['gap'])
    # The contract explicitly declares this telemetry uncoupled.
    q.pop('aux_manifold_telemetry',None)
    no_aux=m.solve_beta_m97_c03_batch_01(q)
    for k in ('weights','terminal_field'):
        np.testing.assert_array_equal(out[k],no_aux[k])
assert worst_y<1e-9 and worst_gap<1e-9,(worst_y,worst_gap)
print('rotating joint bank slices: worst calibration error',worst_y,'worst reference-design KKT gap',worst_gap,'telemetry invariant')
PY
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rotating joint bank slices: worst calibration error 1.3322676295501878e-15 worst reference-design KKT gap 8.881784197001252e-15 telemetry invariant

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a7893dd56e5b3d52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a7893dd5 (directory).
```

### Turn 13 — `submit_completed_task` (`17998 ms` LLM)
- **Rationale**: The latest end-to-end audit matched independently fitted calibration within 1.34e-15 and confirmed that auxiliary telemetry does not affect the outputs. The saved candidate is unchanged from the isolated schema and transport checks and has a bound verification receipt.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_01_solver.py`. The public schema and KKT certificate passed. Maximum absolute errors were 2.92e-7 for source weights and 2.18e-8 for the terminal field. Both recirculation modes, the fouling threshold, and simultaneous sparse-bank and EMI conditions passed the targeted checks.`

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
