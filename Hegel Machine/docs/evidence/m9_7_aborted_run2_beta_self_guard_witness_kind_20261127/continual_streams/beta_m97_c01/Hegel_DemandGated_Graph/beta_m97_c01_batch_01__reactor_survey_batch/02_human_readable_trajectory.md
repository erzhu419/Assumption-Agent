# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c01_batch_01__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c01_step_01` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `291,283` (`282,847` / `107,520` / `8,436`) |
| **Total Episode Wall Time (s)** | `309.05s` |

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

### Turn 1 — `execute_in_remote_docker` (`15272 ms` LLM)
- **Rationale**: Inspect the public contract and archive before reusing the forward operator; this will identify missing inputs or interface mismatches without pulling the dataset out of the container.

```bash
ls -l /tmp/hegel_continual_archive /holdout_task; cat /holdout_task/public_asset_index.json; sed -n '1,240p' /holdout_task/validate_public_schema.py; wc -l /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`371 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`14336 ms` LLM)
- **Rationale**: Read the archived discretization, calibration history, and validator to identify which forward-model conventions carry over and how the certificate is checked before implementing the sealed-wall branch.

```bash
cat /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; sed -n '1,130p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -65
```
- **Container Exit Code**: `0` (`338 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`18358 ms` LLM)
- **Rationale**: The prior output omitted the forward assembly and certificate code. Reading that section separately will resolve axis ordering, sensor stacking, and multiplier conventions before implementation. The calibration rows will show which banks need archive fallback in the public case.

```bash
sed -n '44,161p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; python3 - <<'PY'
import pickle
import numpy as np
p=pickle.load(open('/holdout_task/public_beta_m97_c01_batch_01.pkl','rb'))
print('root:',type(p).__name__,len(p))
if isinstance(p,dict): print('root keys:',list(p))
a=p[0] if not isinstance(p,dict) else p.get('cases',p.get('studies',[p]))[0]
s=a['study']; print('study keys:',list(s)); print('truth keys:',list(a.get('gt',{})))
for key in ('recirculation_loop_mode','past_vortex_circulation_trace','critical_vortex_circulation','mu_shifts','source_ridge','nuisance_ridge','total_budget','u_box','sensor_calibration_sweeps'):
 print(key,s.get(key))
PY
```
- **Container Exit Code**: `0` (`391 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2173,
      -0.377959,
      -0.95691,
      -0.953174,
      1.419985,
      -0.938988,
      0.959744,
      -0.942318
    ]
  }
]root: list 1
study keys: ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'vel_x', 'robin_x', 'reaction_x', 'kappa_y', 'vel_y', 'robin_y', 'reaction_y', 'V_reaction_2d', 'recirculation_loop_mode', 'past_vortex_circulation_trace', 'critical_vortex_circulation', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
truth keys: ['S_tilde', 'r_y', 'w_opt', 'cert_opt', 'terminal_field_A']
recirculation_loop_mode 0.0
past_vortex_circulation_trace [0.136261 0.267565 0.206207 0.251921 0.767165 0.939135 0.91063  0.975683
 0.742526 0.336067 0.158823 0.253155 0.222958 0.185469 0.177629 0.092507]
critical_vortex_circulation 0.5
mu_shifts [0.   0.22 0.52 0.9 ]
source_ridge 0.04
nuisance_ridge 0.65
total_budget 2.6
u_box 1.0
sensor_calibration_sweeps {'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([-1.083514,  1.436139,  0.476672, -1.379075,  1.124264, -1.008754,
       -0.545696,  1.159503,  0.821322, -0.214967, -0.784196, -1.266671,
        0.900673, -1.346184,  1.985429,  0.582765,  1.624365,  0.687592,
        1.322749, -0.009779, -0.761243, -0.999244, -0.954807,  1.151292,
        0.387301, -0.283784, -1.382342,  0.966675,  1.48697 ,  0.484664,
       -1.125168, -0.379082,  2.114219,  0.406528, -0.92497 , -0.682303,
       -0.888164, -0.355165, -0.133275,  0.752048,  2.189532,  0.629129,
        2.079442, -0.055504,  0.834782, -1.335165,  0.520207,  1.01805 ]), 'raw_readout': array([-1.27940331,  2.02134213,  0.76444045, -1.66658821,  1.61278588,
       -1.18146772, -0.57486169,  1.65894877,  1.21593174, -0.14160686,
       -0.88729661, -1.51933912,  0.58153175, -1.19348541,  1.43848888,
        0.33038441,  1.15324843,  0.41319784,  0.91497183, -0.1377255 ,
       -0.73138198, -0.91940275, -0.88429754,  0.77952072,  0.3725073 ,
       -0.44621655, -1.78645703,  1.07934357,  1.71410356,  0.49129027,
       -1.47270499, -0.56248004,  2.47934717,  0.3959643 , -1.22846318,
       -0.93240976, -0.63493915, -0.18189029,  0.00671621,  0.75924087,
        1.98110217,  0.65475967,  1.88752566,  0.07282165,  0.8295648 ,
       -1.01489022,  0.56217583,  0.98534243])}
```

### Turn 4 — `execute_in_remote_docker` (`32011 ms` LLM)
- **Rationale**: Build the conditional operator and compare intermediate quantities against public references. A design mismatch would direct attention to boundary or sensor conventions; a target mismatch would direct attention to calibration; agreement in both isolates any remaining error to the constrained solve or terminal reconstruction.
- **Alternatives Weighed**: `Replace the archived calibration and active-set solver` (rejected: The supplied batch changes the forward mechanism; preserve the existing numerical components and measure their errors separately.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
import re
src = Path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py').read_text()
src = src.replace('Open-Robin reactor tomography', 'Reactor tomography with vortex-triggered membrane fouling')
start = src.index('    ax, ay =', src.index('def _forward('))
end = src.index('    op =', start)
src = src[:start] + '''    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                        - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    ax, ay = (_axis(study[f'kappa_{a}'], study[f'vel_{a}'],
                    0.0 if sealed else study[f'robin_{a}'], study[f'L{a}'])
              for a in ('x', 'y'))
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        lx, ly = study['Lx'], study['Ly']
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    reaction = reaction.ravel(order='F')
''' + src[end:]
src, count = re.subn(r'(?m)^def solve_beta_m97_c01_s00\([^\n]*', 'def solve_beta_m97_c01_batch_01(study: dict) -> dict:', src)
assert count == 1
Path('/holdout_task/beta_m97_c01_batch_01_solver.py').write_text(src)
PY
tail -78 /holdout_task/beta_m97_c01_batch_01_solver.py
python3 -I /holdout_task/validate_public_schema.py || true
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
p = '/holdout_task/'
m = runpy.run_path(p + 'beta_m97_c01_batch_01_solver.py')
a = pickle.load(open(p + 'public_beta_m97_c01_batch_01.pkl', 'rb'))[0]
s, gt = a['study'], a['gt']
out = m['solve_beta_m97_c01_batch_01'](s)
S, _ = m['_forward'](s)
Z = s['Z_nuisance']
A = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
def resid(x):
    return x - Z @ np.linalg.solve(A, Z.T @ x)
for name, pred, ref in [('design', resid(S), gt['S_tilde']), ('calibrated target', resid(m['_calibrate'](s)), gt['r_y']), ('weights', out['weights'], gt['w_opt']), ('terminal field', out['terminal_field'], gt['terminal_field_A'])]:
    d = np.asarray(pred) - ref
    print(name, 'max_abs', np.max(np.abs(d)), 'relative_l2', np.linalg.norm(d) / np.linalg.norm(ref))
PY
```
- **Container Exit Code**: `1` (`587 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
ssings[hit] < 1:
                w[free] += max(0.0, crossings[hit]) * step
                side[free[hit]] = -1 if low[hit] else 1
                w[free[hit]] = 0 if low[hit] else cap
                continue
            w[free] = np.clip(target, 0, cap)
        else:
            g = H @ w - c
            low, high = np.flatnonzero(side < 0), np.flatnonzero(side > 0)
            i, j = low[g[low].argmin()], high[g[high].argmax()]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            side[[i, j]] = 0
            continue
        g = H @ w - c
        violations = side * (g - nu)
        release = int(violations.argmax())
        if violations[release] <= 1e-12:
            return w, nu
        side[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_batch_01(study: dict) -> dict:
    if study['recirculation_loop_mode'] != 0 or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']:
        raise ValueError('Only the open-outflow Robin regime is specified')
    y = _calibrate(study)
    sensors, terminal_modes = _forward(study)
    z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((y, sensors))
    projected = data - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    r, s = projected[:, 0], projected[:, 1:]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), s.T @ r, cap, budget)
    error = s @ w - r
    g = s.T @ error + ridge * w
    primal = float(0.5 * (error @ error + ridge * (w @ w)))
    delta = g - nu
    dual = min(primal, primal - float(g @ w) + nu * budget + cap * float(np.minimum(delta, 0).sum()))
    return {
        'weights': w,
        'terminal_field': (terminal_modes @ w).reshape((study['nx'], study['ny']), order='F'),
        'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                            'lam_lower': np.maximum(delta, 0).tolist(),
                            'lam_upper': np.maximum(-delta, 0).tolist(),
                            'primal_objective': primal, 'dual_lower_bound': dual},
    }

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3bf2940c9b63573 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3bf2940 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 64, in <module>
    out = mod.solve_beta_m97_c01_batch_01(st)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/beta_m97_c01_batch_01_solver.py", line 153, in solve_beta_m97_c01_batch_01
    raise ValueError('Only the open-outflow Robin regime is specified')
ValueError: Only the open-outflow Robin regime is specified
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/holdout_task/beta_m97_c01_batch_01_solver.py", line 153, in solve_beta_m97_c01_batch_01
    raise ValueError('Only the open-outflow Robin regime is specified')
ValueError: Only the open-outflow Robin regime is specified
```

### Turn 5 — `execute_in_remote_docker` (`29689 ms` LLM)
- **Rationale**: Remove the inherited guard that rejects the required fouled regime. Then compare the projected design, calibrated target, weights, and terminal field separately so any discrepancy points to the component that needs correction.

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c01_batch_01_solver.py')
s = p.read_text()
start = s.index("    if study['recirculation_loop_mode']", s.index('def solve_beta_m97_c01_batch_01('))
end = s.index('    y = _calibrate(study)', start)
p.write_text(s[:start] + s[end:])
PY
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
p = '/holdout_task/'
m = runpy.run_path(p + 'beta_m97_c01_batch_01_solver.py')
for a in pickle.load(open(p + 'public_beta_m97_c01_batch_01.pkl', 'rb')):
    s, gt = a['study'], a['gt']
    out = m['solve_beta_m97_c01_batch_01'](s)
    S, _ = m['_forward'](s)
    Z = s['Z_nuisance']
    A = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def resid(x):
        return x - Z @ np.linalg.solve(A, Z.T @ x)
    for name, pred, ref in [('design', resid(S), gt['S_tilde']), ('target', resid(m['_calibrate'](s)), gt['r_y']), ('weights', out['weights'], gt['w_opt']), ('terminal', out['terminal_field'], gt['terminal_field_A'])]:
        d = np.asarray(pred) - ref
        print(name, 'max_abs', np.max(np.abs(d)), 'relative_l2', np.linalg.norm(d) / np.linalg.norm(ref))
    cert = out['kkt_certificate']
    print('primal-dual gap', cert['primal_objective'] - cert['dual_lower_bound'])
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
design max_abs 1.1796119636642288e-16 relative_l2 1.0852433518483205e-15
target max_abs 8.782890525971254e-08 relative_l2 1.0094203870170194e-06
weights max_abs 3.6528120894718796e-07 relative_l2 9.467749847995372e-07
terminal max_abs 2.922596074872441e-08 relative_l2 5.341770869250832e-07
primal-dual gap 1.3183898417423734e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b9a6e3040e0b6980 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b9a6e304 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34087 ms` LLM)
- **Rationale**: The public case verifies the fouled branch with clean calibration. Stress the untested branch conditions, bank fallback and spike handling, and tight budget faces. A failure would identify which component needs correction without changing the verified public solution.

```bash
python3 -I - <<'PY'
import copy, json, pickle, runpy
import numpy as np
p='/holdout_task/'
m=runpy.run_path(p+'beta_m97_c01_batch_01_solver.py')
old=runpy.run_path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py')
base=pickle.load(open(p+'public_beta_m97_c01_batch_01.pkl','rb'))[0]['study']
worst=0.
for mode, excursion in [(0,False),(1,True)]:
    s=copy.deepcopy(base)
    s['recirculation_loop_mode']=mode
    if not excursion:
        s['past_vortex_circulation_trace'][:]=s['critical_vortex_circulation']
    for got, want in zip(m['_forward'](s),old['_forward'](s)):
        worst=max(worst,float(np.max(np.abs(got-want))))
assert worst < 1e-12
print('open-branch maximum deviation:',worst)
s=copy.deepcopy(base)
prior=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
coeffs=np.array([[prior[str(b)]['gain'],prior[str(b)]['offset']] for b in range(4)])
coeffs[2:]=[[1.6,-.2],[.65,.3]]
banks=np.repeat(np.arange(4),[1,4,5,3]); x=np.concatenate([np.linspace(-1,1,n) for n in [1,4,5,3]])
y=coeffs[banks,0]*x+coeffs[banks,1]; y[2]-=9; y[7]+=9
s['sensor_calibration_sweeps']=dict(bank_id=banks,ref_signal=x,raw_readout=y)
b=s['row_bank_id']; target=np.linspace(-.2,.2,b.size)
s['y_obs_raw']=coeffs[b,0]*target+coeffs[b,1]
for bank in range(4):
    err=float(np.max(np.abs((m['_calibrate'](s)-target)[b==bank])))
    print('calibration bank',bank,'max error',err)
    assert err < 1e-12
solve=m['solve_beta_m97_c01_batch_01']; s=copy.deepcopy(base)
first=solve(s)
s['aux_manifold_telemetry']=None
assert np.array_equal(first['weights'],solve(s)['weights'])
S,_=m['_forward'](s); z=s['Z_nuisance']; y=m['_calibrate'](s)
a=np.column_stack((y,S)); a-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
r,T=a[:,0],a[:,1:]; cap=s['u_box']; ridge=s['source_ridge']; worst=np.zeros(3)
for budget in (0.,.001,1.,2.6,6.,11.999,12.):
    s['total_budget']=budget
    out=solve(s); w=out['weights']; c=out['kkt_certificate']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); e=T@w-r
    g=T.T@e+ridge*w; obj=.5*(e@e+ridge*(w@w))
    assert abs(c['primal_objective']-obj)<1e-12 and c['dual_lower_bound']<=obj+1e-12
    assert min(w.min(),cap-w.max(),lo.min(),hi.min())>=-1e-12
    worst=np.maximum(worst,[abs(w.sum()-budget),np.max(np.abs(g-c['nu_sum']-lo+hi)),max(np.max(lo*w),np.max(hi*(cap-w)),obj-c['dual_lower_bound'])])
assert np.max(worst)<1e-10
print('worst budget/stationarity/complementarity-or-gap residuals:',worst)
PY
```
- **Container Exit Code**: `1` (`478 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
open-branch maximum deviation: 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b9a6e3040e0b6980 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b9a6e304 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 19, in <module>
TypeError: list indices must be integers or slices, not str
```

### Turn 7 — `execute_in_remote_docker` (`14720 ms` LLM)
- **Rationale**: The fallback test failed before reaching the solver because it indexed the archive as a dictionary. Inspect the archive layout and the solver's reader together to determine whether the fallback itself needs a fix or only the test fixture does.

```bash
sed -n '1,58p' /holdout_task/beta_m97_c01_batch_01_solver.py; python3 -I - <<'PY'
import json
p=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print(type(p).__name__,len(p))
for row in p if isinstance(p,list) else p.values():
    print({k:(f'{type(v).__name__}[{len(v)}]' if isinstance(v,(list,dict)) else v) for k,v in row.items()})
PY
```
- **Container Exit Code**: `0` (`340 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
arse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_calibration(x, y):
    # Estimate the clean line before applying the EMI threshold. Counting
    # points within 3 units first can reward a tilted line that hides a spike.
    best = None
    for i in range(x.size):
        for j in range(i):
            if x[i] == x[j]:
                continue
            slope = (y[i] - y[j]) / (x[i] - x[j])
            residual = np.abs(y - (y[j] + slope * (x - x[j])))
            mask = residual <= 3.0
            score = (-float(np.median(residual)), int(mask.sum()))
            if best is None or score > best[0]:
                best = score, mask
    return best[1] if best is not None else np.zeros(x.size, dtype=bool)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, ref, raw = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    coeffs = np.empty((4, 2))
    for b in range(4):
        x, y = ref[banks == b], raw[banks == b]
        clean = _clean_calibration(x, y)
        has_emi = clean.sum() < x.size and np.unique(x).size >= 2
        prior = None
        if (x.size < 2 or has_emi) and _ARCHIVE.is_file():
            record = json.loads(_ARCHIVE.read_text()).get(str(b))
            if record is not None:
                prior = np.array([record['gain'], record['offset']])
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or prior is None):
            coeffs[b] = _affine(x[clean], y[clean])
        elif prior is not None:
            coeffs[b] = prior
        else:
            raise ValueError(f'Bank {b} needs two distinct clean references or an archived calibration')
    gain, offset = coeffs[np.asarray(study['row_bank_id'])].T
    return (np.asarray(study['y_obs_raw']).reshape(-1) - offset) / gain


def _axis(kappa, velocity, robin, length):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
list 1
{'episode_short_name': 'beta_m97_c01_s00', 'bank_id': 'list[48]', 'ref_signal': 'list[48]', 'raw_readout': 'list[48]'}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b9a6e3040e0b6980 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b9a6e304 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`43681 ms` LLM)
- **Rationale**: The fallback reader incorrectly treats the archive as a bank-keyed dictionary; the file actually contains raw sweeps. Fit the latest usable archived sweep only when fallback is needed, then verify sparse and spiked banks, public accuracy, and KKT residuals across tight budget faces.

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c01_batch_01_solver.py')
s = p.read_text().replace('Reactor tomography with vortex-triggered membrane fouling with a box-simplex KKT certificate.', 'Reactor tomography with vortex-triggered membrane fouling and a box-simplex KKT certificate.')
s = s.replace('    coeffs = np.empty((4, 2))\n', '    coeffs = np.empty((4, 2))\n    history = None\n', 1)
a = s.index('        prior = None', s.index('def _calibrate('))
b = s.index('    gain, offset', a)
s = s[:a] + '''        if clean.sum() >= 4 or (x.size >= 2 and not has_emi and np.unique(x[clean]).size >= 2):
            coeffs[b] = _affine(x[clean], y[clean])
            continue
        if x.size < 2 or has_emi:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            for sweep in reversed(history):
                select = np.asarray(sweep['bank_id']) == b
                x0 = np.asarray(sweep['ref_signal'])[select]
                y0 = np.asarray(sweep['raw_readout'])[select]
                good = _clean_calibration(x0, y0)
                if np.unique(x0[good]).size >= 2:
                    coeffs[b] = _affine(x0[good], y0[good])
                    break
            else:
                raise ValueError(f'No usable archived calibration for bank {b}')
            continue
        raise ValueError(f'Bank {b} needs two distinct clean references')
''' + s[b:]
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import copy, json, pickle, runpy
import numpy as np
p='/holdout_task/'
m=runpy.run_path(p+'beta_m97_c01_batch_01_solver.py')
case=pickle.load(open(p+'public_beta_m97_c01_batch_01.pkl','rb'))[0]; base=case['study']; gt=case['gt']
solve=m['solve_beta_m97_c01_batch_01']; first=solve(base)
for result, key, tol in [('weights','w_opt',1e-6),('terminal_field','terminal_field_A',1e-7)]:
    err=np.max(np.abs(first[result]-gt[key])); print(result,'public max error',err); assert err<tol
prior=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))[-1]
b0,x0,y0=(np.asarray(prior[k]) for k in ('bank_id','ref_signal','raw_readout'))
coeff=np.array([np.linalg.lstsq(np.column_stack((x0[b0==i],np.ones(np.count_nonzero(b0==i)))),y0[b0==i],rcond=None)[0] for i in range(4)])
coeff[2:]=[[1.6,-.2],[.65,.3]]
banks=np.repeat(np.arange(4),[1,4,5,3]); x=np.concatenate([np.linspace(-1,1,n) for n in [1,4,5,3]])
y=coeff[banks,0]*x+coeff[banks,1]; y[2]-=9; y[7]+=9
s=copy.deepcopy(base); s['sensor_calibration_sweeps']=dict(bank_id=banks,ref_signal=x,raw_readout=y)
b=np.asarray(s['row_bank_id']); target=np.linspace(-.2,.2,b.size); s['y_obs_raw']=coeff[b,0]*target+coeff[b,1]
cal_error=m['_calibrate'](s)-target
for i in range(4):
    err=np.max(np.abs(cal_error[b==i])); print('calibration bank',i,'max error',err); assert err<1e-12
s=copy.deepcopy(base); s['aux_manifold_telemetry']=None
assert np.array_equal(first['weights'],solve(s)['weights'])
S,_=m['_forward'](s); z=s['Z_nuisance']; y=m['_calibrate'](s)
a=np.column_stack((y,S)); a-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
r,T=a[:,0],a[:,1:]; cap=s['u_box']; ridge=s['source_ridge']; n=T.shape[1]; worst=np.zeros(3)
for budget in np.asarray([0.,.001,1.,2.6,6.,n-.001,n])*cap:
    s['total_budget']=budget
    out=solve(s); w=out['weights']; c=out['kkt_certificate']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); e=T@w-r
    g=T.T@e+ridge*w; obj=.5*(e@e+ridge*(w@w))
    assert abs(c['primal_objective']-obj)<1e-12 and c['dual_lower_bound']<=obj+1e-12
    assert min(w.min(),cap-w.max(),lo.min(),hi.min())>=-1e-12
    worst=np.maximum(worst,[abs(w.sum()-budget),np.max(np.abs(g-c['nu_sum']-lo+hi)),max(np.max(lo*w),np.max(hi*(cap-w)),obj-c['dual_lower_bound'])])
print('worst budget/stationarity/complementarity-or-gap residuals:',worst)
assert np.max(worst)<1e-10
PY
```
- **Container Exit Code**: `0` (`937 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
weights public max error 3.6528120894718796e-07
terminal_field public max error 2.922596074872441e-08
calibration bank 0 max error 0.0
calibration bank 1 max error 0.0
calibration bank 2 max error 8.326672684688674e-17
calibration bank 3 max error 2.220446049250313e-16
worst budget/stationarity/complementarity-or-gap residuals: [8.8817842e-16 0.0000000e+00 8.8817842e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a17c66bbddc14f1d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a17c66bb (directory).
```

### Turn 9 — `submit_completed_task` (`13272 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification. The archive reader fix preserved public accuracy and passed the sparse and spiked bank tests; budget stress tests verified feasibility, stationarity, and complementarity.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:7741'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c01_batch_01_solver.py`. Public schema validation passed. Maximum public errors were 3.65e-7 for weights and 2.92e-8 for the terminal field. Stress tests passed for archive fallback, EMI spikes, and tight budget constraints.`

### Turn 10 — `execute_in_remote_docker` (`32722 ms` LLM)
- **Rationale**: Hold out real calibration rows within each bank to detect whether any coefficient depends on one reference; a large error would trigger a bank-specific calibration review. Compare raw versus calibrated targets and Robin versus sealed designs to isolate the effects of calibration and boundary physics. A nonzero sealed boundary row sum would identify a discretization error rather than a need for a new model.
- **Alternatives Weighed**: `Run balanced-accuracy, relay, or stochastic-drift checks` (rejected: The task has continuous measurements, a recirculation mode already tested in both states, and no class labels or stochastic drift equation.)

```bash
set -e
cat > /tmp/beta_m97_public_audit.py <<'PY'
import json
import pickle
import runpy
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
m = runpy.run_path(str(root / 'beta_m97_c01_batch_01_solver.py'))
archived = runpy.run_path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py')
with (root / 'public_beta_m97_c01_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
prior = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())[-1]
for case in cases:
    s, gt = case['study'], case['gt']
    for label, sweep in [('public', s['sensor_calibration_sweeps']), ('prefix', prior)]:
        banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        for bank in range(4):
            xb, yb = x[banks == bank], y[banks == bank]
            errors = []
            for i in range(xb.size):
                train = np.arange(xb.size) != i
                keep = m['_clean_calibration'](xb[train], yb[train])
                g, o = m['_affine'](xb[train][keep], yb[train][keep])
                errors.append(abs(g * xb[i] + o - yb[i]))
            print(label, 'bank', bank, 'leave-one-reference-out max raw error', max(errors))
    out = m['solve_beta_m97_c01_batch_01'](s)
    z = s['Z_nuisance']
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    def project(a):
        return a - z @ np.linalg.solve(gram, z.T @ a)
    design, _ = m['_forward'](s)
    robin_design, _ = archived['_forward'](s)
    errors = dict(sealed_design=np.max(np.abs(project(design) - gt['S_tilde'])), robin_design=np.max(np.abs(project(robin_design) - gt['S_tilde'])), calibrated_target=np.max(np.abs(project(m['_calibrate'](s)) - gt['r_y'])), raw_target=np.max(np.abs(project(np.asarray(s['y_obs_raw']).reshape(-1)) - gt['r_y'])), weights=np.max(np.abs(out['weights'] - gt['w_opt'])), terminal=np.max(np.abs(out['terminal_field'] - gt['terminal_field_A'])))
    print('Public max errors:', errors)
    assert errors['sealed_design'] < 1e-12 and errors['weights'] < 1e-6 and errors['terminal'] < 1e-7
    for axis in ('x', 'y'):
        mat = m['_axis'](s[f'kappa_{axis}'], s[f'vel_{axis}'], 0.0, s[f'L{axis}'])
        boundary_sums = (mat @ np.ones(mat.shape[0]))[[0, -1]]
        print(axis, 'sealed boundary row sums:', boundary_sums)
        np.testing.assert_allclose(boundary_sums, 0.0, atol=1e-12)
PY
python3 -I /tmp/beta_m97_public_audit.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`825 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public bank 0 leave-one-reference-out max raw error 1.8526752598724272e-07
public bank 1 leave-one-reference-out max raw error 1.5761386262802546e-07
public bank 2 leave-one-reference-out max raw error 1.953269478383035e-07
public bank 3 leave-one-reference-out max raw error 2.3473052823685947e-07
prefix bank 0 leave-one-reference-out max raw error 6.661338147750939e-16
prefix bank 1 leave-one-reference-out max raw error 1.1102230246251565e-15
prefix bank 2 leave-one-reference-out max raw error 6.661338147750939e-16
prefix bank 3 leave-one-reference-out max raw error 7.771561172376096e-16
Public max errors: {'sealed_design': 1.1796119636642288e-16, 'robin_design': 0.03262427606397696, 'calibrated_target': 8.782890525971254e-08, 'raw_target': 0.4919194231094133, 'weights': 3.6528120894718796e-07, 'terminal': 2.922596074872441e-08}
x sealed boundary row sums: [0. 0.]
y sealed boundary row sums: [0. 0.]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a17c66bbddc14f1d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a17c66bb (directory).
```

### Turn 11 — `submit_completed_task` (`17143 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated Python validation and targeted stress tests. Held-out calibration rows supported the specified affine fits, and the sealed operator matched the public design to numerical precision.
- **ATMS NOGOOD DECLARED**: Refuted `The archived open-Robin operator also describes the fouled membrane regime.` | Witness: `On the public fouled case, the projected design maximum error was 3.26e-2 with the archived operator and 1.18e-16 with the sealed-wall operator.` | Blanket Diagnosis: `The forward operator must depend on recirculation mode and past threshold exceedance; sensor-bank calibration cannot correct the resulting design error.` | Sublated Pivot: `Use sealed Neumann boundaries and wall sorption when fouling is active; retain the archived operator otherwise.`
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:7741'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. Public schema validation passed. Maximum public errors were 3.65e-7 for source weights and 2.92e-8 for the terminal field. Stress tests passed for both recirculation modes, sparse and EMI-spiked calibration, and box-simplex budget limits.`

### Turn 12 — `submit_completed_task` (`15080 ms` LLM)
- **Rationale**: The verified candidate already resolves ng_t11: the sealed-wall branch reduced maximum projected design error from 3.26e-2 to 1.18e-16, and the unaffected open branches matched the archive exactly. The focused audit and isolated schema validation passed on this same artifact.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. Public schema validation passed. Maximum public errors were 3.65e-7 for source weights and 2.92e-8 for the terminal field. Stress tests passed for both recirculation modes, sparse and EMI-spiked calibration, and box-simplex budget limits.`

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
