# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c02_batch_01__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c02_step_01` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `274,579` (`265,547` / `65,408` / `9,032`) |
| **Total Episode Wall Time (s)** | `360.39s` |

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

### Turn 1 — `execute_in_remote_docker` (`14109 ms` LLM)
- **Rationale**: Inspect the public contract and prior forward operator to catch interface or stencil mismatches before coding; check whether a reusable ridge-projection primitive is available.
- **Alternatives Weighed**: `Implement from the prompt alone` (rejected: The prompt does not specify the full forward stencil, observation packing, or validator tolerances; the archive and public schema can resolve those before implementation.)

```bash
set -e
cd /holdout_task
ls -l /tmp/hegel_continual_archive
cat public_asset_index.json
sed -n '1,240p' validate_public_schema.py
sed -n '1,320p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
```
- **Container Exit Code**: `0` (`317 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`13616 ms` LLM)
- **Rationale**: The first inspection truncated the archived implementation. Read it and the small calibration history separately: the stencil determines where the sealed-wall change belongs, and the history format determines how to recover calibration for a bank with insufficient or contaminated sweeps.
- **Alternatives Weighed**: `Patch the archived solver from the truncated listing` (rejected: The missing lines determine where to change the wall stencil without affecting the other boundary branch.)

```bash
set -e
nl -ba /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py | sed -n '1,250p'
cat /tmp/hegel_continual_archive/archive_calibration_history.json
```
- **Container Exit Code**: `0` (`267 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
l = values - Z @ np.linalg.solve(gram, Z.T @ values)
   200	    w, cert = _box_simplex(residual[:, 1:], residual[:, 0], study['source_ridge'],
   201	                           study['u_box'], study['total_budget'])
   202	    field = (terminal @ w).reshape(study['nx'], study['ny'], order='F')
   203	    return {'weights': w, 'terminal_field': field, 'kkt_certificate': cert}
[
  {
    "episode_short_name": "beta_m97_c02_s00",
    "bank_id": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "ref_signal": [
      0.268167,
      1.973482,
      0.155493,
      -0.991373,
      0.005186,
      0.883223,
      0.855335,
      -0.111767,
      2.179932,
      0.515458,
      2.19924,
      1.8819,
      0.790972,
      -1.447989,
      -0.050426,
      1.319399,
      2.029099,
      -0.88645,
      0.286097,
      -0.734514,
      0.486405,
      -1.23754,
      -1.240911,
      1.652451,
      2.155713,
      0.453713,
      0.912999,
      1.858977,
      -1.367223,
      1.234135,
      -0.671947,
      -0.037994,
      -1.441484,
      1.234848,
      2.043594,
      2.149632,
      1.492805,
      -1.165341,
      -0.972774,
      0.868599,
      0.633375,
      1.932977,
      0.643473,
      0.809337,
      0.674338,
      1.857391,
      -0.593005,
      -0.394796
    ],
    "raw_readout": [
      0.268167,
      1.973482,
      0.155493,
      -0.991373,
      0.005186,
      0.883223,
      0.855335,
      -0.111767,
      2.179932,
      0.515458,
      2.19924,
      1.8819,
      0.790972,
      -1.447989,
      -0.050426,
      1.319399,
      2.029099,
      -0.88645,
      0.286097,
      -0.734514,
      0.486405,
      -1.23754,
      -1.240911,
      1.652451,
      2.155713,
      0.453713,
      0.912999,
      1.858977,
      -1.367223,
      1.234135,
      -0.671947,
      -0.037994,
      -1.441484,
      1.234848,
      2.043594,
      2.149632,
      1.492805,
      -1.165341,
      -0.972774,
      0.868599,
      0.633375,
      1.932977,
      0.643473,
      0.809337,
      0.674338,
      1.857391,
      -0.593005,
      -0.394796
    ]
  }
]
```

### Turn 3 — `execute_in_remote_docker` (`20252 ms` LLM)
- **Rationale**: Read only the stencil and calibration code omitted by truncation, and inspect public sweep rows for sparse banks or EMI. This determines where to change the boundary operator and whether each bank should use its current fit or archived sweeps.
- **Alternatives Weighed**: `Use the gate diagnostic as calibration` (rejected: Its coefficients do not establish which banks require archival fallback; that decision must come from the sweep rows.)

```bash
set -e
sed -n '49,135p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py
python3 - <<'PY'
import pickle
import numpy as np
p=pickle.load(open('/holdout_task/public_beta_m97_c02_batch_01.pkl','rb'))
print('public root:', type(p).__name__, list(p) if isinstance(p,dict) else len(p))
items=[p] if isinstance(p,dict) and 'study' in p else (p.values() if isinstance(p,dict) else p)
for i,item in enumerate(items):
    if not isinstance(item,dict) or 'study' not in item: continue
    s=item['study']; c=s['sensor_calibration_sweeps']; b=np.asarray(c['bank_id']); x=np.asarray(c['ref_signal']); y=np.asarray(c['raw_readout'])
    print('case',i,'mode',s['recirculation_loop_mode'],'v_mem',np.sum(np.maximum(0,np.asarray(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation'])**2)); print('study keys:',list(s))
    for j in range(4):
        ix=b==j; print('bank',j,'n',ix.sum(),'ref/raw',np.round(np.c_[x[ix],y[ix]],5).tolist())
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -25
```
- **Container Exit Code**: `0` (`545 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
bank 0 n 12 ref/raw [[-0.79114, -0.79083], [-0.21661, -0.32545], [-1.13301, -1.06774], [-1.30817, -1.20962], [1.71345, 1.23789], [0.8757, 0.55932], [0.94366, 0.61437], [-0.29371, -0.38791], [1.47519, 1.0449], [1.84362, 1.34334], [0.93259, 0.6054], [2.11283, 1.56139]]
bank 1 n 12 ref/raw [[0.8402, 1.25747], [-0.91794, -1.08085], [-1.38227, -1.69842], [-1.34658, -1.65096], [-0.36677, -0.34781], [-0.22818, -0.16348], [1.26379, 1.82085], [1.49297, 2.12565], [1.84316, 2.59141], [0.64334, 0.99564], [2.14704, 2.99557], [-0.01465, 0.12052]]
bank 2 n 12 ref/raw [[1.75838, 1.6222], [1.79159, 1.65077], [1.6395, 1.51997], [1.75139, 1.61619], [1.35727, 1.27725], [-0.88848, -0.65409], [0.61505, 0.63895], [0.71753, 0.72708], [-1.45787, -1.14376], [1.95727, 1.79325], [2.01516, 1.84304], [1.55983, 1.45145]]
bank 3 n 12 ref/raw [[0.90724, 0.96869], [-0.1322, -0.27863], [-0.45272, -0.66327], [-0.03936, -0.16723], [0.21602, 0.13922], [1.03367, 1.12041], [-0.93519, -1.24222], [-0.83887, -1.12665], [0.16561, 0.07873], [-0.35061, -0.54073], [-1.29108, -1.6693], [-1.18116, -1.53739]]
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
  "access_mode": "executable",
  "execution_available": true,
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
```

### Turn 4 — `execute_in_remote_docker` (`16023 ms` LLM)
- **Rationale**: Read the exact boundary-row assembly and certificate checks that were truncated. The stencil determines where to preserve reaction terms while removing Robin loss; the validator determines the multiplier sign and numerical tolerances.
- **Alternatives Weighed**: `Replace the archived stencil wholesale` (rejected: Only the wall rows and reaction potential are specified to change; replacing the interior would risk changing transport or sensor ordering.)

```bash
set -e
sed -n '89,129p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py
sed -n '131,191p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py
sed -n '1,180p' /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`273 ms`, timed_out=`False`)
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
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c02_batch_01_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c02_batch_01.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c02_batch_01(st)
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

### Turn 5 — `execute_in_remote_docker` (`40014 ms` LLM)
- **Rationale**: Build the solver using the known transport stencil and fit archived sweep rows only where needed. Compare projected observations, forward matrix, weights, and terminal field separately so a calibration or boundary mismatch is visible even if the certificate passes.
- **Alternatives Weighed**: `Seal both axes by default` (rejected: The specified sorption mask identifies the x walls; compare both interpretations against the public forward matrix before changing the y stencil.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py').read_text()
src = src.replace('Open-Robin reactor tomography', 'Vortex-gated reactor tomography', 1)
a = src.index('    try:\n', src.index('def _archived_affine'))
b = src.index('\n\ndef _calibration', a)
src = src[:a] + '''    for sweeps in reversed(json.loads(_ARCHIVE.read_text())):
        mask = np.asarray(sweeps['bank_id']) == bank
        x = np.asarray(sweeps['ref_signal'], dtype=float)[mask]
        y = np.asarray(sweeps['raw_readout'], dtype=float)[mask]
        if len(x) >= 2 and np.ptp(x) > 0:
            return _fit_affine(x, y)
    raise ValueError(f'No archived affine calibration for sensor bank {bank}')
''' + src[b:]
a = src.index('    ax = _axis', src.index('def _forward'))
b = src.index('    modes = ', a)
src = src[:a] + '''    lx, ly = study['Lx'], study['Ly']
    potential = np.asarray(study['V_reaction_2d'], dtype=float)
    excess = np.maximum(0., np.asarray(study['past_vortex_circulation_trace']) -
                        study['critical_vortex_circulation'])
    v_mem = excess @ excess
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0.
    robin_x = 0. if sealed else study['robin_x']
    if sealed:
        xs = np.linspace(0., lx, nx)[:, None]
        ys = np.linspace(0., ly, ny)[None, :]
        wall_mask = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        potential = potential + ((0.95 + 0.55 * np.tanh(v_mem)) * wall_mask *
                                 (0.6 + 0.4 * np.sin(np.pi * ys / ly)))
    ax = _axis(study['kappa_x'], study['vel_x'], robin_x, lx)
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], ly)
    L = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') +
         diags(potential.ravel(order='F'), format='csc'))
''' + src[b:]
src = src.replace('def solve_beta_m97_c02_s00(', 'def solve_beta_m97_c02_batch_01(')
Path('/holdout_task/beta_m97_c02_batch_01_solver.py').write_text(src)
PY
python3 -I /holdout_task/validate_public_schema.py || true
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root = '/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',root+'beta_m97_c02_batch_01_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open(root+'public_beta_m97_c02_batch_01.pkl', 'rb'))[0]
s,g=item['study'],item['gt']
for label, st in [('x_only',s), ('both_axes',dict(s,robin_y=0.)), ('open',dict(s,recirculation_loop_mode=1))]:
    S,_=m._forward(st); y=m._calibration(st); Z=s['Z_nuisance']; A=np.column_stack((y,S)); R=A-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@A)
    out=m.solve_beta_m97_c02_batch_01(st); cert=out['kkt_certificate']
    print(label, 'max_abs_error', {k:float(np.max(np.abs(d))) for k,d in [('ry',R[:,0]-g['r_y']), ('S',R[:,1:]-g['S_tilde']), ('w',out['weights']-g['w_opt']), ('field',out['terminal_field']-g['terminal_field_A'])]}, 'candidate_gap',cert['primal_objective']-cert['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`1704 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": false, "errors": ["kkt_invalid:reported_primal_mismatch:1.470e-03"]}
x_only max_abs_error {'ry': 1.1320478288112668e-07, 'S': 0.033194341855142785, 'w': 0.0830351303392883, 'field': 0.018506304764684265} candidate_gap 7.632783294297951e-17
both_axes max_abs_error {'ry': 1.1320478288112668e-07, 'S': 1.1796119636642288e-16, 'w': 6.268782038465481e-07, 'field': 3.51467013409712e-08} candidate_gap 6.245004513516506e-17
open max_abs_error {'ry': 1.1320478288112668e-07, 'S': 0.03391789998639244, 'w': 0.05783305596554275, 'field': 0.02455194310217086} candidate_gap 1.1796119636642288e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8286a94260fb81a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8286a942 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`55730 ms` LLM)
- **Rationale**: Apply the boundary condition supported by the public evidence, then verify the exact artifact in isolated Python. Check the unfouled gates against the archived forward operator, exercise sparse and EMI calibration in each bank, and test KKT conditions at ordinary and limiting budgets so a passing public schema does not mask a branch defect.
- **Alternatives Weighed**: `Keep y-axis Robin leakage because the sorption mask is x-localized` (rejected: The public forward matrix distinguishes the two effects and rules out residual y-wall leakage in the fouled state.)
- **ATMS NOGOOD DECLARED**: Refuted `The direction of the sorption mask also identifies the only axis whose boundary condition changes.` | Witness: `X-only sealing failed public KKT verification with a 1.47e-3 reported-primal mismatch; sealing both axes matched the public projected-forward matrix to 1.18e-16.` | Blanket Diagnosis: `The transient changes leakage on both axes, while the spatially varying sorption term remains concentrated at the x walls.` | Sublated Pivot: `Remove Robin leakage on both axes only when mode is zero and vortex memory is positive; preserve the specified x-wall sorption mask.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c02_batch_01_solver.py')
s = p.read_text()
s = s.replace("    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], ly)", "    robin_y = 0. if sealed else study['robin_y']\n    ay = _axis(study['kappa_y'], study['vel_y'], robin_y, ly)")
s = s.replace("                    archived = _archived_affine(b)\n                    if archived is None:\n                        raise ValueError(f'Ambiguous three-point EMI calibration for sensor bank {b}')\n                    gain[b], offset[b] = archived", "                    gain[b], offset[b] = _archived_affine(b)")
a = s.index('        archived = _archived_affine(b) if')
b = s.index('    banks = ', a)
s = s[:a] + '''        xx, yy = x[clean], y[clean]
        if len(xx) >= 4 or (len(xx) >= 2 and clean.all()):
            gain[b], offset[b] = _fit_affine(xx, yy)
        else:
            gain[b], offset[b] = _archived_affine(b)
''' + s[b:]
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np

root = '/holdout_task/'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('final_solver', root+'beta_m97_c02_batch_01_solver.py')
a = load('archived_solver', '/tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py')
with open(root+'public_beta_m97_c02_batch_01.pkl','rb') as f:
    e = pickle.load(f)[0]
s,gt = e['study'],e['gt']

def maxerr(x,y):
    return float(np.max(np.abs(x-y)))

def check_solution(st, label):
    sol = m.solve_beta_m97_c02_batch_01(st)
    w, c = sol['weights'], sol['kkt_certificate']
    z = st['Z_nuisance']
    S, term = m._forward(st)
    vals = np.column_stack((m._calibration(st), S))
    R = vals - z @ np.linalg.solve(z.T @ z + st['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ vals)
    S,y = R[:,1:],R[:,0]
    res = S @ w - y
    ridge, cap, budget = st['source_ridge'], st['u_box'], st['total_budget']
    p = .5 * (res @ res + ridge * (w @ w))
    g = S.T @ res + ridge*w
    nu, ll, lu = c['nu_sum'], c['lam_lower'], c['lam_upper']
    stat = np.max(np.abs(g-nu-ll+lu))
    gap = p-c['dual_lower_bound']
    assert abs(w.sum()-budget)<1e-10 and np.min(w)>=-1e-10 and np.max(w)<=cap+1e-10
    assert min(ll.min(),lu.min())>=0 and np.max(np.abs(ll*w)+np.abs(lu*(cap-w)))<1e-10
    assert stat<1e-9 and -1e-12<=gap<1e-9 and abs(p-c['primal_objective'])<1e-12
    assert maxerr(c['weights'],w)==0 and maxerr(sol['terminal_field'],(term@w).reshape(st['nx'],st['ny'],order='F'))<1e-12
    print(label,'stationarity',stat,'gap',gap)
    return sol, S

sol,S = check_solution(s,'public sealed, both axes')
for key, value, ref, tol in [('forward',S,gt['S_tilde'],1e-12),('weights',sol['weights'],gt['w_opt'],2e-6),('terminal field',sol['terminal_field'],gt['terminal_field_A'],2e-7)]:
    err = maxerr(value,ref)
    print(key,'max abs error',err)
    assert err<tol
for label, st in [('loop mode 1 with excursion',dict(s,recirculation_loop_mode=1)), ('loop mode 0 without excursion',dict(s,past_vortex_circulation_trace=np.full_like(s['past_vortex_circulation_trace'],s['critical_vortex_circulation'])))]:
    assert all(maxerr(x,y)<1e-12 for x,y in zip(m._forward(st),a._forward(st)))
    check_solution(st,label)
for b in [0.,s['u_box'],12*s['u_box']]:
    check_solution(dict(s,total_budget=b),f'budget {b}')
sweep = s['sensor_calibration_sweeps']
bank, ref, raw = (np.asarray(sweep[k]) for k in ('bank_id','ref_signal','raw_readout'))
obs = np.asarray(s['y_obs_raw'])
rows = np.asarray(s['row_bank_id'])
baseline = m._calibration(s)
for b in range(4):
    idx = np.flatnonzero(bank==b)
    spike = raw.copy(); spike[idx[0]] += 4.
    for label, keep, values, fallback in [('spike with clean majority',np.arange(bank.size),spike,False), ('two clean points',np.flatnonzero((bank!=b) | np.isin(np.arange(bank.size),idx[:2])),raw,False), ('one point',np.flatnonzero((bank!=b) | (np.arange(bank.size)==idx[0])),raw,True), ('spike with two clean points',np.flatnonzero((bank!=b) | np.isin(np.arange(bank.size),idx[:3])),spike,True)]:
        st = dict(s,sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=ref[keep],raw_readout=values[keep]))
        expected = baseline.copy()
        if fallback:
            expected[rows==b] = obs[rows==b]  # prefix sweep has exactly unit gain and zero offset
        else:
            clean = idx[1:] if values is spike else idx[:2]
            gain, offset = np.polyfit(ref[clean],raw[clean],1)
            expected[rows==b] = (obs[rows==b]-offset)/gain
        assert maxerr(m._calibration(st),expected)<1e-12, (b,label)
print('calibration: all 16 bank/condition slices passed')
PY
```
- **Container Exit Code**: `0` (`2000 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public sealed, both axes stationarity 2.7755575615628914e-17 gap 6.245004513516506e-17
forward max abs error 1.1796119636642288e-16
weights max abs error 6.268782038465481e-07
terminal field max abs error 3.51467013409712e-08
loop mode 1 with excursion stationarity 4.163336342344337e-17 gap 1.1796119636642288e-16
loop mode 0 without excursion stationarity 4.163336342344337e-17 gap 1.1796119636642288e-16
budget 0.0 stationarity 0.0 gap 0.0
budget 1.0 stationarity 4.163336342344337e-17 gap 6.591949208711867e-17
budget 12.0 stationarity 2.220446049250313e-16 gap 0.0
calibration: all 16 bank/condition slices passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=af492642f31bcec3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_af492642 (directory).
```

### Turn 7 — `submit_completed_task` (`13586 ms` LLM)
- **Rationale**: Isolated Python verification passed on the exact saved artifact. The corrected two-axis stencil matches the public projected forward matrix to 1.18e-16, resolving the boundary mismatch without changing either unfouled branch.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_01_solver.py': 'FILE:7815'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c02_batch_01_solver.py`. The public schema validator passed, along with both unfouled control cases, budget boundary checks, and all 16 calibration stress cases.`

### Turn 8 — `execute_in_remote_docker` (`45069 ms` LLM)
- **Rationale**: An independent dense stencil checks both recirculation modes and would localize a boundary or indexing defect. Leaving out each real calibration point checks whether any bank is driving the fit; compare bankwise errors with profiling alone to determine whether calibration is necessary and whether its uncertainty threatens the weight or field margins.
- **Alternatives Weighed**: `Run a pooled classification or drift-model audit` (rejected: The inputs define a continuous inverse problem, not a classifier or stochastic drift model. Bankwise residual errors and an independent transport stencil test the applicable confounding and algebraic risks.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/audit_beta_m97_c02.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate_audit', root/'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    items = pickle.load(f)

def err(a,b):
    return float(np.max(np.abs(a-b)))

def axis(k, v, robin, length):
    n = len(k)
    h = length/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
            a[i,i] = k[i]/h**2 + robin/h
        else:
            a[i,i-1] = -(k[i-1]+k[i])/(2*h*h) - max(v[i],0)/h
            a[i,i+1] = -(k[i+1]+k[i])/(2*h*h) - max(-v[i],0)/h
            a[i,i] = -a[i].sum()
    return a

for case,item in enumerate(items):
    s,g = item['study'],item['gt']
    nx,ny = s['nx'],s['ny']
    amp = sum(max(0.,v-s['critical_vortex_circulation'])**2 for v in s['past_vortex_circulation_trace'])
    x,y = np.meshgrid(np.linspace(0.,1.,nx),np.linspace(0.,1.,ny),indexing='ij')
    sink = (0.95+0.55*np.tanh(amp))*(np.exp(-16*x*x)+np.exp(-16*(1-x)**2))*(0.6+0.4*np.sin(np.pi*y))
    i,j = np.asarray(s['sensor_ij']).T
    rhs = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
    for mode in (0,1):
        sealed = mode == 0 and amp > 0
        ax = axis(s['kappa_x'],s['vel_x'],0. if sealed else s['robin_x'],s['Lx'])
        ay = axis(s['kappa_y'],s['vel_y'],0. if sealed else s['robin_y'],s['Ly'])
        if sealed:
            assert max(np.max(np.abs(ax.sum(1))),np.max(np.abs(ay.sum(1)))) < 1e-10
        L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag((s['V_reaction_2d']+(sink if sealed else 0.)).ravel(order='F'))
        T = [np.linalg.solve(L+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
        got,terminal = m._forward(dict(s,recirculation_loop_mode=mode))
        e = max(err(got,np.concatenate([t[i+nx*j] for t in T])),err(terminal,T[-1]))
        print('case',case,'loop mode',mode,'dense forward max error',e)
        assert e < 1e-11
    sweeps=s['sensor_calibration_sweeps']
    bank,ref,raw = [np.asarray(sweeps[k]) for k in ('bank_id','ref_signal','raw_readout')]
    rowbank=np.asarray(s['row_bank_id'])
    Z=s['Z_nuisance']
    def profile(v):
        return v-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@v)
    unaligned=profile(np.asarray(s['y_obs_raw']))
    for b in range(4):
        heldout=[]; weight=[]; field=[]; adjusted=[]
        ids=np.flatnonzero(bank==b)
        for k in ids:
            train=ids[ids!=k]
            slope,offset=np.polyfit(ref[train],raw[train],1)
            heldout.append(abs(slope*ref[k]+offset-raw[k]))
            keep=np.arange(bank.size)!=k
            st=dict(s,sensor_calibration_sweeps={key:np.asarray(val)[keep] for key,val in sweeps.items()})
            out=m.solve_beta_m97_c02_batch_01(st)
            weight.append(err(out['weights'],g['w_opt']))
            field.append(err(out['terminal_field'],g['terminal_field_A']))
            adjusted.append(err(profile(m._calibration(st))[rowbank==b],g['r_y'][rowbank==b]))
        unadjusted=err(unaligned[rowbank==b],g['r_y'][rowbank==b])
        print('case',case,'bank',b,'LOO sweep',max(heldout),'profiled adjusted/unadjusted',max(adjusted),unadjusted,'weight',max(weight),'field',max(field))
        assert max(heldout)<1e-5 and max(adjusted)<1e-6 and max(adjusted)<unadjusted
        assert max(weight)<2e-6 and max(field)<2e-7
PY
python3 -I /tmp/audit_beta_m97_c02.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1009 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 loop mode 0 dense forward max error 1.734723475976807e-16
case 0 loop mode 1 dense forward max error 7.632783294297951e-17
case 0 bank 0 LOO sweep 1.799955520187524e-07 profiled adjusted/unadjusted 1.39805238941193e-07 0.8826267624615367 weight 7.648153985129191e-07 field 4.320660514089969e-08
case 0 bank 1 LOO sweep 2.404731488797607e-07 profiled adjusted/unadjusted 1.270729845792573e-07 0.9796782399922508 weight 6.425146467370357e-07 field 3.628524191529037e-08
case 0 bank 2 LOO sweep 2.1864664212500884e-07 profiled adjusted/unadjusted 6.87345367289538e-08 0.5296550179714599 weight 6.350781563413399e-07 field 3.602830293825132e-08
case 0 bank 3 LOO sweep 3.2341516456591e-07 profiled adjusted/unadjusted 7.82769522533755e-08 0.41419995639561646 weight 6.393788133385314e-07 field 3.6903125659459857e-08
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=af492642f31bcec3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_af492642 (directory).
```

### Turn 9 — `submit_completed_task` (`15748 ms` LLM)
- **Rationale**: The verified artifact passed isolated execution, both recirculation branches, and the independent KKT probe. Calibration reduced the profiled residual error in every bank, and all leave-one-out weight and field errors stayed within the tested margins. No further change is supported by the audit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_01_solver.py': 'FILE:7815'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_01_solver.py`. The public schema passed, and an independent transport model agreed in both recirculation modes. Across all leave-one-out calibration folds, maximum absolute errors were 7.65e-7 for weights and 4.32e-8 for the terminal field.`

### Turn 10 — `execute_in_remote_docker` (`28406 ms` LLM)
- **Rationale**: The prior EMI checks covered a large clean majority and only two clean points, but not the specified boundary of exactly four clean sweeps. Test both spike signs just above threshold at real sweep locations. A discrepancy would identify a calibration branch that discards or contaminates a valid four-point fit; otherwise no edit is needed.
- **Alternatives Weighed**: `Repeat the dense-operator or classification audit` (rejected: Both transport modes already agree with the independent stencil; this continuous inverse problem has no classification labels or multi-batch accuracy metric.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat >/tmp/audit_beta_m97_c02_emi.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('batch_solver', root/'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    items = pickle.load(f)

worst = (0., None)
failures = 0
for case, item in enumerate(items):
    s = item['study']
    b, x, y = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id','ref_signal','raw_readout'))
    for bank in range(4):
        ids = np.flatnonzero(b == bank)
        ids = ids[np.argsort(x[ids])]
        selected = ids[np.linspace(0, len(ids)-1, 5, dtype=int)]
        for spike in selected:
            clean = selected[selected != spike]
            base = np.flatnonzero((b != bank) | np.isin(np.arange(len(b)), clean))
            contaminated = np.sort(np.append(base, spike))
            def study(rows, raw):
                return dict(s, sensor_calibration_sweeps=dict(bank_id=b[rows], ref_signal=x[rows], raw_readout=raw[rows]))
            expected = m._calibration(study(base, y))
            for emi in (-3.1, 3.1):
                raw = y.copy()
                raw[spike] += emi
                found = m._calibration(study(contaminated, raw))
                error = float(np.max(np.abs(found-expected)))
                label = (case, bank, float(x[spike]), emi)
                failures += error > 1e-10
                if error > worst[0]:
                    worst = (error, label)
print('Four clean sweeps plus one signed EMI spike:', 'failed slices', failures, 'worst calibrated-readout error and slice', worst)
if failures:
    import inspect
    print(inspect.getsource(m._calibration))
    raise SystemExit(1)
PY
python3 -I /tmp/audit_beta_m97_c02_emi.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`809 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Four clean sweeps plus one signed EMI spike: failed slices 0 worst calibrated-readout error and slice (0.0, None)
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=af492642f31bcec3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_af492642 (directory).
```

### Turn 11 — `submit_completed_task` (`13854 ms` LLM)
- **Rationale**: The latest audit covered the remaining four-clean-point calibration threshold and found no failures. The saved artifact passed the public validator and the independent transport and KKT checks; no further edit is warranted.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_01_solver.py`. The public validator passed in isolated Python. Both recirculation modes matched an independent transport calculation, and all 40 four-clean-point calibration tests recovered the uncontaminated fit after a signed EMI spike.`

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
