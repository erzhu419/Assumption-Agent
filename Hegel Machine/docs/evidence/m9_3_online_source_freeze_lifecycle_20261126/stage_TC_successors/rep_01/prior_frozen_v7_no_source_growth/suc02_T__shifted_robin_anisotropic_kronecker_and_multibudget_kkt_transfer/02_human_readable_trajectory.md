# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Anisotropic Reversed-Advection Kronecker Sylvester Transport & Multi-Budget KKT Pareto Certification` |
| **Wave / Wormhole Family** | `m93_stage_T_transfer` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `289,534` (`278,830` / `60,544` / `10,704`) |
| **Total Episode Wall Time (s)** | `400.69s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

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
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`22040 ms` LLM)
- **Rationale**: Inspect the validator and a compact study summary to catch mismatches in field names, boundary conventions, and output structure before writing the solver.
- **Alternatives Weighed**: `Implement from the task summary alone` (rejected: The supplied validator and study can resolve stencil, stacking, and certificate conventions before implementation.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_suc02_T_studies.pkl','rb') as f:
    data=pickle.load(f)
print(type(data), len(data))
if isinstance(data, dict):
    print('top-level keys:', list(data)[:30])
    if 'source_modes' not in data:
        data=next(iter(data.values()))
while isinstance(data, (list,tuple)):
    print('nested:', type(data), len(data))
    data=data[0]
if isinstance(data,dict):
    for k,v in data.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype, v.ravel()[:8])
        else:
            print(k,type(v),repr(v)[:450])
PY
ls -la /pretrain_task /tmp/hegel_wormhole_skills 2>/dev/null; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT' 2>&1
```
- **Container Exit Code**: `0` (`675 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`23350 ms` LLM)
- **Rationale**: Read the unseen validator checks and compact study fields to catch stencil or certificate convention mismatches; inspect the ridge primitive to determine whether it supplies the required profiled least-squares geometry.
- **Alternatives Weighed**: `Implement from the truncated inspection output` (rejected: The missing output contains the study fields and validator checks needed to choose the correct stencil and nuisance transform.)

```bash
cd /holdout_task && sed -n '45,180p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_suc02_T_studies.pkl','rb') as f: data=pickle.load(f)
def show(v):
 if isinstance(v,dict) and 'source_modes' in v:
  for k,a in v.items():
   print(k, (a.shape,a.dtype,a.flat[:5]) if isinstance(a,np.ndarray) else str(a)[:240])
 elif isinstance(v,dict):
  print('collection keys',list(v)[:12]); show(next(iter(v.values())))
 elif isinstance(v,(list,tuple)):
  print('collection len',len(v)); show(v[0])
 else: print(type(v),str(v)[:200])
show(data)
PY
find /pretrain_task -maxdepth 2 -type f 2>/dev/null | head -25
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`719 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`34081 ms` LLM)
- **Rationale**: Identify the exact profiling convention and available stencil comparison targets before constructing the solver; an incorrect convention changes the optimizer and invalidates its certificates.
- **Alternatives Weighed**: `Assume the objective uses the square root of the Schur complement` (rejected: Ordinary ridge residualization and square-root profiling produce different QPs; the public target residual can distinguish them directly.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
p=pickle.load(open('public_suc02_T_studies.pkl','rb'))[0]; s,g=p['study'],p['gt']
for k,v in g.items():
 a=np.asarray(v); print(k,a.shape,a.dtype,a.ravel()[:4])
Z=s['Z_nuisance']; Q,d,_=np.linalg.svd(Z,full_matrices=False); y=s['y_obs']; r=s['nuisance_ridge']
for e in [.5,1]:
 pred=y+Q@(((r/(d*d+r))**e-1)*(Q.T@y)); print('profile exponent',e,'max error',np.max(abs(pred-g['r_y'])))
for ax in 'xy':
 v=s['vel_'+ax]; print(ax,'velocity range and endpoints',v.min(),v.max(),v[[0,-1]])
print('scipy version',__import__('scipy').__version__)
PY
find /pretrain_task -maxdepth 3 -type f -print 2>/dev/null | head -20
```
- **Container Exit Code**: `0` (`511 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
711721685e-17, 2.0816681711721685e-17, 2.0816681711721685e-17, 1.3877787807814457e-17, 2.0816681711721685e-17, 1.734723475976807e-17, 0.0030432569481855867], 'lam_upper': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'primal_objective': 0.03906431692465978, 'dual_lower_bound': 0.03906431692465975}
 {'weights': [0.02092697527357751, 0.2947945512306602, 0.06558300993853924, 0.11951344540808621, 0.21312928697800154, 0.12624952136662335, 0.26196950164721655, 0.07924205411414664, 0.19760184272535883, 0.21399198151783247, 0.13247462534036084, 0.2638177366080791, 0.2193217894613717, 0.26950948941690067, 0.09225693537591272, 0.029617253597333082], 'nu_sum': 0.009527983302115323, 'lam_lower': [2.7755575615628914e-17, 3.8163916471489756e-17, 1.3877787807814457e-17, 1.3877787807814457e-17, 1.3877787807814457e-17, 6.938893903907228e-18, 0.0, 6.938893903907228e-18, 1.0408340855860843e-17, 0.0, 6.938893903907228e-18, 0.0, 6.938893903907228e-18, 0.0, 1.3877787807814457e-17, 1.3877787807814457e-17], 'lam_upper': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'primal_objective': 0.03932836821873202, 'dual_lower_bound': 0.03932836821873199}
 {'weights': [0.06310294113723114, 0.45669596638403975, 0.02110249353264459, 0.1798219216425899, 0.2568953247966629, 0.293270810065521, 0.35155318404467484, 0.11374231077347043, 0.33365351996975373, 0.30856333655220036, 0.18723167573933772, 0.3954445361886492, 0.2598276802867132, 0.3691174569083164, 0.10343219842972594, 0.10654464354846929], 'nu_sum': 0.02791499255736784, 'lam_lower': [4.85722573273506e-17, 1.734723475976807e-17, 1.3877787807814457e-17, 6.938893903907228e-18, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, 2.0816681711721685e-17, 3.469446951953614e-18, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, 0.0, 0.0, 1.3877787807814457e-17, 2.0816681711721685e-17], 'lam_upper': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.938893903907228e-18, 0.0, 0.0], 'primal_objective': 0.06179415373442193, 'dual_lower_bound': 0.06179415373442188}]
terminal_field_mid_budget (48, 36) float64 [0.00052723 0.00068274 0.00085471 0.00103221]
profile exponent 0.5 max error 0.1286710594787821
profile exponent 1 max error 4.440892098500626e-15
x velocity range and endpoints -0.6897045314069272 -0.41429546859307265 [-0.56102563 -0.54297437]
y velocity range and endpoints 0.36843760971944917 0.5975623902805506 [0.36843761 0.36843761]
scipy version 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`35757 ms` LLM)
- **Rationale**: Compare both full public operators against the inferred harmonic flux, upwind sign, and half-cell Robin resistance. Any remaining residual locates the stencil term to correct before implementation.
- **Alternatives Weighed**: `Assume the inferred stencil is correct from its first row` (rejected: A first-row match cannot establish the interior or opposite-boundary treatment under reversed advection.)
- **ATMS NOGOOD DECLARED**: Refuted `Square-root Schur-complement profiling reproduces the benchmark residual vector.` | Witness: `On the public study, the square-root transform differs from r_y by 0.128671 in max norm; the ordinary ridge residual differs by 4.44e-15.` | Blanket Diagnosis: `The benchmark squares an already ridge-residualized observation and design, so the square-root transform defines a different QP and certificate.` | Sublated Pivot: `Apply the same ordinary ridge residual map to the sensor matrix and observations.`

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
p=pickle.load(open('public_suc02_T_studies.pkl','rb'))[0]; s,g=p['study'],p['gt']
for name,key in [('x','Ax'),('y','By')]:
 kap,v,r=(s[k+'_'+name] for k in ('kappa','vel','robin')); h=s['L'+name]/len(kap)
 f=2*kap[:-1]*kap[1:]/(kap[:-1]+kap[1:])/h**2
 diag=np.r_[f,0]+np.r_[0,f]+abs(v)/h
 diag[[0,-1]]+=r/(h*(1+r*h/(2*kap[[0,-1]])))
 pred=np.diag(diag)+np.diag(-f-np.maximum(-v[:-1],0)/h,1)+np.diag(-f-np.maximum(v[1:],0)/h,-1)
 d=g[key]-pred
 print(name,'max error',np.abs(d).max(),'boundary diagonal residuals',d.diagonal()[[0,-1]],'interior max',np.abs(d[1:-1]).max())
PY
```
- **Container Exit Code**: `0` (`629 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x max error 0.0 boundary diagonal residuals [0. 0.] interior max 0.0
y max error 2.842170943040401e-14 boundary diagonal residuals [-2.84217094e-14 -2.84217094e-14] interior max 0.0
```

### Turn 5 — `execute_in_remote_docker` (`59338 ms` LLM)
- **Rationale**: Implement the measured stencil and residual transform, then run the public validator in an isolated Python process. A failed certificate will distinguish an objective or optimization mismatch from the already matched one-dimensional stencils.
- **Alternatives Weighed**: `Run 192 separate Sylvester factorizations` (rejected: A sparse Kronecker sum factors once per shift and reuses that factor for all 16 source modes.)

```bash
set -e
cd /holdout_task
cat > multibudget_kronecker_kkt_transfer_solver.py <<'PY'
"""Shifted transport responses and certified box-simplex source fits."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis_operator(length, kappa, velocity, robin):
    kappa = np.asarray(kappa, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    h = length / kappa.size
    face = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    diag = np.r_[face, 0.] + np.r_[0., face] + np.abs(velocity) / h
    ends = [0, kappa.size - 1]
    diag[ends] += robin / (h * (1 + robin * h / (2 * kappa[ends])))
    lower = -face - np.maximum(velocity[1:], 0.) / h
    upper = -face - np.maximum(-velocity[:-1], 0.) / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _transport(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    ax, by = (_axis_operator(study['L' + a], study['kappa_' + a],
                            study['vel_' + a], study['robin_' + a]) for a in 'xy')
    # With C-order flattening, U -> Ax U + U By.T has this Kronecker sum.
    base = (sparse.kron(ax, sparse.eye(ny), format='csc') +
            sparse.kron(sparse.eye(nx), by, format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    i, j = np.asarray(study['sensor_ij']).T
    sensors = i * ny + j
    rhs = np.asfortranarray(sources.reshape(m, nx * ny).T)
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(base + float(shift) * eye).solve(rhs)
        blocks.append(fields[sensors, :])
    return np.concatenate(blocks, axis=0), fields


def _profile(design, target, nuisance, ridge):
    # The objective uses ordinary ridge residuals, not the square-root map.
    values = np.column_stack((design, target))
    if ridge == 0:
        coef = np.linalg.lstsq(nuisance, values, rcond=None)[0]
    else:
        gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
        coef = np.linalg.solve(gram, nuisance.T @ values)
    residual = values - nuisance @ coef
    return residual[:, :-1], residual[:, -1]


def _fit_budget(hessian, linear, cap, budget):
    """Feasible active-set QP: minimize on a face, then test its multipliers."""
    m = linear.size
    if budget == 0 or budget == m * cap:
        return np.full(m, budget / m)
    w = np.full(m, budget / m)
    status = np.zeros(m, dtype=np.int8)  # -1 lower, +1 upper, 0 free
    grad_tol = 32 * np.finfo(float).eps * max(
        np.max(np.abs(linear)), cap * np.max(np.sum(np.abs(hessian), axis=1)))
    for _ in range(100 * m):
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            g = hessian @ w + linear
            lo = np.where(status < 0, g, np.inf).argmin()
            hi = np.where(status > 0, g, -np.inf).argmax()
            if g[hi] <= g[lo] + grad_tol:
                return w
            status[[lo, hi]] = 0
            continue

        fixed = np.flatnonzero(status)
        # Stationarity on the free set, with the remaining budget as an equality.
        n = free.size
        system = np.ones((n + 1, n + 1))
        system[:n, :n] = hessian[np.ix_(free, free)]
        system[n, n] = 0.
        rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed],
                    budget - cap * np.count_nonzero(status > 0)]
        sol = np.linalg.solve(system, rhs)
        trial, nu = sol[:-1], -sol[-1]
        if np.any((trial < 0.) | (trial > cap)):
            old = w[free]
            delta = trial - old
            steps = np.full(n, np.inf)
            low = trial < 0.
            high = trial > cap
            steps[low] = -old[low] / delta[low]
            steps[high] = (cap - old[high]) / delta[high]
            k = steps.argmin()
            w[free] = np.clip(old + steps[k] * delta, 0., cap)
            w[free[k]] = 0. if low[k] else cap
            status[free[k]] = -1 if low[k] else 1
            continue

        w[free] = trial
        reduced = hessian @ w + linear - nu
        violation = status * reduced
        k = violation.argmax()
        if violation[k] <= grad_tol:
            return w
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, target, ridge, w, cap, budget):
    residual = design @ w - target
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = design.T @ residual + ridge * w
    # The minimizer of the tangent plane fills its cheapest coordinates first.
    # A gradient at the budget cutoff is a maximizing equality multiplier.
    k = min(int(budget / cap), w.size - 1) if cap > 0 else 0
    nu = float(np.partition(g, k)[k])
    lower = np.maximum(g - nu, 0.)
    upper = np.maximum(nu - g, 0.)
    gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - budget)
    return {'weights': w, 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
            'primal_objective': primal,
            'dual_lower_bound': float(primal - max(0., gap))}


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    sensor_design, terminal = _transport(study)
    design, target = _profile(sensor_design, np.asarray(study['y_obs']).ravel(),
                              np.asarray(study['Z_nuisance']), float(study['nuisance_ridge']))
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    m = design.shape[1]
    hessian = design.T @ design + ridge * np.eye(m)
    linear = -(design.T @ target)
    certs = []
    for budget in study['budget_grid']:
        budget = float(budget)
        w = _fit_budget(hessian, linear, cap, budget)
        certs.append(_certificate(design, target, ridge, w, cap, budget))
    weights = np.stack([cert['weights'] for cert in certs])
    field = (terminal @ weights[1]).reshape(np.shape(study['source_modes'])[1:])
    return {'weights_by_budget': weights, 'terminal_field_mid_budget': field,
            'kkt_certificates': certs}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`483 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/multibudget_kronecker_kkt_transfer_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29b2623eda964fad (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_29b2623e (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37304 ms` LLM)
- **Rationale**: An independent Sylvester reference detects shift, stacking, or transpose errors; known-solution QPs isolate active-set and certificate failures. If either check fails, fix the corresponding component before submitting.
- **Alternatives Weighed**: `Submit after the public schema check` (rejected: The schema check does not compare field values or exercise upper-saturated and endpoint budgets.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve_sylvester

p = Path.cwd() / 'multibudget_kronecker_kkt_transfer_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with open('public_suc02_T_studies.pkl','rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
out = mod.solve_multibudget_kronecker_transfer(s)
we = np.max(np.abs(out['weights_by_budget'] - gt['weights_by_budget']))
fe = np.max(np.abs(out['terminal_field_mid_budget'] - gt['terminal_field_mid_budget']))
assert we < 1e-8 and fe < 1e-9, (we, fe)
gaps = []
for b, w, cert in zip(s['budget_grid'], out['weights_by_budget'], out['kkt_certificates']):
    np.testing.assert_array_equal(w, cert['weights'])
    ck = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], b, cert)
    assert ck['valid'], ck
    gaps.append(ck['gap'])
# Cross-check all shifted sensor blocks with independent 2D Sylvester solves.
stack, terminal = mod._transport(s)
i, j = s['sensor_ij'].T
blocks = []
for mu in s['mu_shifts']:
    fields = np.stack([solve_sylvester(gt['Ax'] + mu*np.eye(s['nx']), gt['By'].T, q) for q in s['source_modes']])
    blocks.append(fields[:, i, j].T)
se = np.max(np.abs(stack - np.concatenate(blocks)))
te = np.max(np.abs(terminal.T.reshape(fields.shape) - fields))
assert max(se, te) < 1e-10, (se, te)
print('public weight max error', we, 'terminal max error', fe, 'worst certificate gap', max(gaps), 'all-shift sensor max error', se, 'final-shift field max error', te)
# Construct independent QPs with known solutions, including mixed active bounds,
# fully active integer budgets and both degenerate endpoint budgets.
rng = np.random.default_rng(269)
worst_w = worst_gap = 0.
for k in range(240):
    m = 16
    cap = float(10**rng.uniform(-1, 1))
    A = rng.normal(size=(48, m))
    A *= np.geomspace(1., .01 if k % 2 else 1., m)
    ridge = 10**rng.uniform(-5, 0)
    mode = k % 6
    if mode < 2:
        w0 = np.full(m, cap * mode)
    else:
        w0 = rng.uniform(.01, .99, m)*cap
        perm = rng.permutation(m)
        w0[perm[:5]] = 0.
        w0[perm[5:16 if mode == 2 else 10]] = cap
    budget = float(w0.sum())
    if mode == 1:
        budget = m * cap
    grad = rng.normal() + (w0 == 0)*rng.uniform(.1, 4., m) - (w0 == cap)*rng.uniform(.1, 4., m)
    y = A @ w0 - np.linalg.lstsq(A.T, grad - ridge*w0, rcond=None)[0]
    w = mod._fit_budget(A.T @ A + ridge*np.eye(m), -(A.T @ y), cap, budget)
    cert = mod._certificate(A, y, ridge, w, cap, budget)
    ck = verify(A, y, ridge, cap, budget, cert)
    assert ck['valid'], (k, ck)
    e = np.max(np.abs(w-w0))
    assert e < 2e-8, (k, e)
    worst_w = max(worst_w, e)
    worst_gap = max(worst_gap, ck['gap'])
print('240 known-solution stress cases; worst weight max error', worst_w, 'worst verified gap', worst_gap)
PY
```
- **Container Exit Code**: `0` (`834 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public weight max error 1.2434497875801753e-14 terminal max error 2.0209528495129803e-16 worst certificate gap 2.456368441983159e-15 all-shift sensor max error 1.654926196081874e-15 final-shift field max error 3.226585665316861e-16
240 known-solution stress cases; worst weight max error 6.77125022718883e-13 worst verified gap 3.4393599079862724e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29b2623eda964fad (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_29b2623e (directory).
```

### Turn 7 — `submit_completed_task` (`14086 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against independent solutions for every shift, public reference outputs for all budgets, and known-solution QPs covering mixed bounds and endpoint budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/multibudget_kronecker_kkt_transfer_solver.py': 'FILE:5900'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed, and the terminal field matched the reference within 2.03e-16. Across 240 stress cases, the worst verified certificate gap was 3.44e-12, below the 1e-6 tolerance.`

### Turn 8 — `execute_in_remote_docker` (`36597 ms` LLM)
- **Rationale**: Reflect each axis of the measured study, reorder nonterminal shifts and endpoint budgets, and rotate the nuisance basis. A failure would identify a supported transfer case where the solver mishandles geometry, ordering, or profiling; a pass requires no artifact change.
- **Alternatives Weighed**: `Audit relay modes or batch-classification metrics` (rejected: The study is a transport regression with no relay states or batch labels. Orthogonal nuisance-basis rotation tests the relevant profiling invariance.)

```bash
set -e
cd /holdout_task
cat > /tmp/audit_kronecker_transfer.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'multibudget_kronecker_kkt_transfer_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root / 'public_suc02_T_studies.pkl').open('rb') as f:
    studies = pickle.load(f)
worst_gap = worst_weight = worst_field = 0.
rng = np.random.default_rng(917)
for item in studies:
    s, gt = item['study'], item['gt']
    nk = len(s['mu_shifts'])
    nr = len(s['sensor_ij'])
    perm = np.r_[rng.permutation(nk-1), nk-1]
    rows = (perm[:, None] * nr + np.arange(nr)).ravel()
    order = [2, 1, 0]
    z = s['Z_nuisance']
    q, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
    for axes in ('x', 'y', 'xy'):
        t = dict(s)
        t['sensor_ij'] = np.array(s['sensor_ij'], copy=True)
        field_slice = [slice(None), slice(None)]
        for a in axes:
            d = 'xy'.index(a)
            field_slice[d] = slice(None, None, -1)
            t['kappa_'+a] = s['kappa_'+a][::-1]
            t['vel_'+a] = -s['vel_'+a][::-1]
            t['robin_'+a] = np.asarray(s['robin_'+a])[::-1]
            t['sensor_ij'][:, d] = s['n'+a] - 1 - s['sensor_ij'][:, d]
        t['source_modes'] = s['source_modes'][(slice(None), *field_slice)]
        t['mu_shifts'] = np.asarray(s['mu_shifts'])[perm]
        t['y_obs'] = np.asarray(s['y_obs'])[rows]
        t['Z_nuisance'] = z[rows] @ q
        t['budget_grid'] = np.asarray(s['budget_grid'])[order]
        out = mod.solve_multibudget_kronecker_transfer(t)
        ew = np.max(np.abs(out['weights_by_budget'] - gt['weights_by_budget'][order]))
        ef = np.max(np.abs(out['terminal_field_mid_budget'] - gt['terminal_field_mid_budget'][tuple(field_slice)]))
        assert ew < 1e-8 and ef < 1e-9, (axes, ew, ef)
        worst_weight, worst_field = max(worst_weight, ew), max(worst_field, ef)
        for b, cert in zip(t['budget_grid'], out['kkt_certificates']):
            ck = verify(gt['S_tilde'][rows], gt['r_y'][rows], t['source_ridge'], t['u_box'], b, cert)
            assert ck['valid'], (axes, b, ck)
            worst_gap = max(worst_gap, ck['gap'])
print(f'{len(studies)} public studies, 3 reflection slices each: worst weight error {worst_weight:.3e}, field error {worst_field:.3e}, KKT gap {worst_gap:.3e}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_kronecker_transfer.py
```
- **Container Exit Code**: `1` (`529 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29b2623eda964fad (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_29b2623e (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/audit_kronecker_transfer.py", line 35, in <module>
    t['robin_'+a] = np.asarray(s['robin_'+a])[::-1]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

### Turn 9 — `execute_in_remote_docker` (`14966 ms` LLM)
- **Rationale**: Broadcast the scalar Robin coefficient to both endpoints before reflecting it in the audit. The rerun will detect geometry, ordering, or profiling errors in the solver; only a numerical failure would warrant a solver change.
- **Alternatives Weighed**: `Change the solver's Robin stencil` (rejected: The traceback is confined to the test fixture: the public study stores a scalar Robin coefficient, which the solver already handles correctly.)

```bash
set -e
sed -i "s/np.asarray(s\['robin_'+a\])\[::\-1\]/np.broadcast_to(s['robin_'+a], (2,))[::-1]/" /tmp/audit_kronecker_transfer.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_kronecker_transfer.py
```
- **Container Exit Code**: `0` (`535 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1 public studies, 3 reflection slices each: worst weight error 1.252e-14, field error 2.082e-16, KKT gap 2.498e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29b2623eda964fad (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_29b2623e (directory).
```

### Turn 10 — `submit_completed_task` (`14492 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against all 192 public Sylvester responses, all three budget certificates, and the independent active-bound and transfer tests. The audit found no solver defect requiring a change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/multibudget_kronecker_kkt_transfer_solver.py': 'FILE:5900'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed. The worst gap across 240 stress cases was 3.44e-12, below the 1e-6 tolerance. The reflection and reordering audit also passed, with a maximum terminal-field error of 2.08e-16.`

### Turn 11 — `execute_in_remote_docker` (`44656 ms` LLM)
- **Rationale**: Keep the public observations and source modes, vary the boundary and advection terms, and change which shift is last. An independent Sylvester reference and KKT verifier will distinguish stencil or terminal-field errors from profiling or budget-fit errors. No solver edit is warranted unless this check finds a failure.
- **Alternatives Weighed**: `Repeat the reflection audit` (rejected: Reflection preserved equal Robin endpoints and left the final shift unchanged, so it could not detect swapped endpoint coefficients or hard-coded terminal-shift selection.)

```bash
set -e
cd /holdout_task
cat > /tmp/audit_kronecker_boundaries.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve_sylvester

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'multibudget_kronecker_kkt_transfer_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
worst_sensor = worst_field = worst_gap = 0.
slices = 0
with (root / 'public_suc02_T_studies.pkl').open('rb') as f:
    studies = pickle.load(f)
for item in studies:
    s, gt = item['study'], item['gt']
    for axis in 'xy':
        t = dict(s)
        velocity = np.asarray(s['vel_' + axis])
        signs = np.ones(velocity.size)
        signs[::2] = -1.
        new_velocity = velocity * signs
        robin = np.broadcast_to(s['robin_' + axis], (2,))
        new_robin = robin * [0.45, 1.7]
        t['vel_' + axis], t['robin_' + axis] = new_velocity, new_robin
        h = s['L' + axis] / velocity.size
        key = {'x': 'Ax', 'y': 'By'}[axis]
        mat = gt[key].copy()
        # Changing only flow direction leaves its diagonal contribution intact.
        # Move the upwind contribution from the old neighbor to the new one.
        for row in range(0, velocity.size, 2):
            v = velocity[row]
            if v == 0:
                continue
            old = row - int(np.sign(v))
            new = row + int(np.sign(v))
            if 0 <= old < velocity.size:
                mat[row, old] += abs(v) / h
            if 0 <= new < velocity.size:
                mat[row, new] -= abs(v) / h
        for side, row in enumerate((0, velocity.size - 1)):
            half_cell = h / (2 * s['kappa_' + axis][row])
            for r, sign in ((robin[side], -1), (new_robin[side], 1)):
                if r:
                    mat[row, row] += sign / (h * (half_cell + 1 / r))
        ax, by = (mat, gt['By']) if axis == 'x' else (gt['Ax'], mat)
        t['mu_shifts'] = np.asarray(s['mu_shifts'])[::-1]
        nr = len(s['sensor_ij'])
        order = np.arange(len(s['y_obs'])).reshape(-1, nr)[::-1].ravel()
        z, y = s['Z_nuisance'][order], s['y_obs'][order]
        t['Z_nuisance'], t['y_obs'] = z, y
        i, j = s['sensor_ij'].T
        eye = np.eye(ax.shape[0])
        blocks = []
        for mu in t['mu_shifts']:
            fields = np.stack([solve_sylvester(ax + mu * eye, by.T, q) for q in s['source_modes']])
            blocks.append(fields[:, i, j].T)
        sensor = np.concatenate(blocks)
        sensor_error = np.max(np.abs(solver._transport(t)[0] - sensor))
        assert sensor_error < 1e-10, (axis, sensor_error)
        # Independent SVD form of the ordinary ridge residual map.
        q, d, _ = np.linalg.svd(z, full_matrices=False)
        values = np.column_stack((sensor, y))
        denom = d*d + s['nuisance_ridge']
        shrink = np.divide(d*d, denom, out=np.zeros_like(d), where=denom > 0)
        prof = values - q @ (shrink[:, None] * (q.T @ values))
        out = solver.solve_multibudget_kronecker_transfer(t)
        w = out['weights_by_budget']
        field_error = np.max(np.abs(out['terminal_field_mid_budget'] - np.einsum('m,mij->ij', w[1], fields)))
        assert field_error < 1e-10, (axis, field_error)
        for budget, weights, cert in zip(t['budget_grid'], w, out['kkt_certificates']):
            np.testing.assert_array_equal(weights, cert['weights'])
            check = verify(prof[:, :-1], prof[:, -1], t['source_ridge'], t['u_box'], budget, cert)
            assert check['valid'], (axis, budget, check)
            worst_gap = max(worst_gap, check['gap'])
        worst_sensor = max(worst_sensor, sensor_error)
        worst_field = max(worst_field, field_error)
        slices += 1
print(f'{slices} public-derived mixed-upwind/asymmetric-Robin slices: max sensor error {worst_sensor:.3e}, max terminal-field error {worst_field:.3e}, worst KKT gap {worst_gap:.3e}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_kronecker_boundaries.py
```
- **Container Exit Code**: `0` (`932 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2 public-derived mixed-upwind/asymmetric-Robin slices: max sensor error 2.356e-15, max terminal-field error 2.220e-15, worst KKT gap 6.102e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29b2623eda964fad (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_29b2623e (directory).
```

### Turn 12 — `submit_completed_task` (`14754 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against all 192 public Sylvester responses, all three budget certificates, and independent references for mixed advection, asymmetric boundaries, and a changed terminal shift. The final audit found no defect requiring an edit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed. The worst gap across 240 stress cases was 3.44e-12, below the 1e-6 tolerance. The mixed-advection and asymmetric Robin audit also passed, with a maximum terminal-field error of 2.22e-15.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/multibudget_kronecker_kkt_transfer_solver.py": "FILE:5900"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_weights_rel_l2": 0.0,
    "max_field_rel_l2": 0.0,
    "max_kkt_gap": 0.0,
    "student_elapsed_sec": 1.7677,
    "instance_diagnostics": [
      {
        "seed": 99611,
        "w_rel_l2": 0.0,
        "field_rel_l2": 0.0,
        "max_kkt_gap": 0.0
      },
      {
        "seed": 99612,
        "w_rel_l2": 0.0,
        "field_rel_l2": 0.0,
        "max_kkt_gap": 0.0
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.051247,
      "import_cpu_sec": 0.046589,
      "candidate_call_wall_sec": 0.090717,
      "candidate_call_cpu_sec": 0.082184,
      "ipc_and_codec_wall_sec": 0.208795,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 1.561928,
      "verifier_and_gt_cpu_sec": 1.494144,
      "total_supervisor_wall_sec": 1.912686
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 1.7677, \"instance_diagnostics\": [{\"seed\": 99611, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}, {\"seed\": 99612, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_4eab0d5d01b0d824477d642476717f64__:{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 1.7677, \"instance_diagnostics\": [{\"seed\": 99611, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}, {\"seed\": 99612, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.051247, \"import_cpu_sec\": 0.046589, \"candidate_call_wall_sec\": 0.090717, \"candidate_call_cpu_sec\": 0.082184, \"ipc_and_codec_wall_sec\": 0.208795, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 1.561928, \"verifier_and_gt_cpu_sec\": 1.494144, \"total_supervisor_wall_sec\": 1.912686}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 8240.45,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
