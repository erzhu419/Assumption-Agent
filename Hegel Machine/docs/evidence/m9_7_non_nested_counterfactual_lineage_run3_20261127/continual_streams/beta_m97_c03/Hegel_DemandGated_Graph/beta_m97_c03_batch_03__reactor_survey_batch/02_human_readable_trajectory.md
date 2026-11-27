# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c03_batch_03__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c03_step_03` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `266,790` (`258,586` / `96,384` / `8,204`) |
| **Total Episode Wall Time (s)** | `299.1s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`root_A03_ill_posed_regularization_and_certificates`**: AXIOM A03 (HEAVY-TAILED REGULARIZATION & DUAL CERTIFICATES): Under outliers, stage drift, or ill-posed combinatorial landscapes, replace naive L2 least squares with heavy-tailed robust losses (Huber/Cauchy/MAD) or exact variable projection, and verify optimality/safety margins via dual bounds or worst-corner simulation.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14047 ms` LLM)
- **Rationale**: Read the validator, public input shapes, and archive inventory to identify contract mismatches before choosing which transport and inversion code to reuse.

```bash
cd /holdout_task && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && wc -l /tmp/hegel_continual_archive/*solver.py && python3 - <<'PY'
import pickle
with open('public_beta_m97_c03_batch_03.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items(): print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:180] if getattr(v,'ndim',0)<1 else '')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`15524 ms` LLM)
- **Rationale**: Inspect one study inside the public list and the latest archived operator to identify reusable code and pinpoint where periodic indexing, calibration, and nuisance projection need to change.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, collections
with open('public_beta_m97_c03_batch_03.pkl','rb') as f: d=pickle.load(f)
print(type(d).__name__,len(d)); print(collections.Counter(type(x).__name__ for x in d))
for x in d[:1]:
 for k,v in x.items():
  print(k,type(v).__name__,getattr(v,'shape',''),repr(v)[:180] if not getattr(v,'ndim',0) else '')
for i,x in enumerate(d):
 print(i,x.get('recirculation_loop_mode'),type(x.get('sensor_calibration_sweeps')).__name__)
PY
cat /tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -40
```
- **Container Exit Code**: `0` (`492 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`20784 ms` LLM)
- **Rationale**: The public pickle wraps each input under `study`. Inspect the nested sweep and the archived stencil to identify which banks need fallback and where periodic wrapping belongs. Read the validator tail to determine which outputs must be checked independently.

```bash
cd /holdout_task && sed -n '45,175p' /tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py && tail -48 validate_public_schema.py && python3 - <<'PY'
import pickle,json,numpy as np
with open('public_beta_m97_c03_batch_03.pkl','rb') as f: cases=pickle.load(f)
a=cases[0]; s=a['study']; print('STUDY',[(k,getattr(v,'shape',None),v if np.isscalar(v) else type(v).__name__) for k,v in s.items()]); print('GT',[(k,getattr(v,'shape',None)) for k,v in a['gt'].items()]); print('SWEEP',s['sensor_calibration_sweeps']); h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('HISTORY',len(h),[(list(t), {int(b):int(np.count_nonzero(np.asarray(t['bank_id'])==b)) for b in set(t['bank_id'])}) for t in h]); print('GRID',[(k,np.asarray(s[k]).ravel()[:3].tolist(),np.asarray(s[k]).ravel()[-2:].tolist()) for k in ('xs','ys','kappa_x','kappa_y') if k in s])
PY
```
- **Container Exit Code**: `0` (`520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array'), ('source_modes', (12, 18, 16), 'ndarray'), ('sensor_ij', (36, 2), 'ndarray'), ('row_bank_id', (144,), 'ndarray'), ('sensor_calibration_sweeps', None, 'dict'), ('Z_nuisance', (144, 5), 'ndarray'), ('y_obs_raw', (144,), 'ndarray'), ('aux_manifold_telemetry', (144, 3), 'ndarray'), ('nuisance_ridge', None, 0.65), ('source_ridge', None, 0.04), ('u_box', None, 1.0), ('total_budget', None, 2.6)]
GT [('S_tilde', (144, 12)), ('r_y', (144,)), ('w_opt', (12,)), ('cert_opt', None), ('terminal_field_A', (18, 16))]
SWEEP {'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([-0.309939,  0.509739, -0.301632, -0.012838, -0.227686,  0.417615,
        1.388939,  1.244269, -0.876755,  0.688231,  0.450253,  1.014505,
        0.246204, -1.103179, -1.256419,  1.842579, -1.490331,  1.224235,
        0.054897,  0.807965,  1.166814,  1.412796, -0.008659, -0.926917,
        1.725776, -1.403503,  0.883847,  1.197932,  0.834109,  2.047369,
        1.473478,  2.082333,  1.136001,  0.729196, -0.93955 , -1.041375,
       -0.822857, -1.237617,  0.822121, -0.80934 , -1.475581,  2.03562 ,
       -0.237924,  1.189686, -0.610196, -0.442137,  0.923241, -1.362116]), 'raw_readout': array([-0.26362257,  0.77736866, -0.25307273,  0.11369583, -0.15916133,
        0.66037106,  1.89395253,  1.71022171, -0.983479  ,  1.00405345,
        0.7018213 ,  1.4184213 ,  0.03203918, -1.0204797 , -1.14000685,
        1.27721184, -1.32245811,  0.79490319, -0.11718027,  0.47021277,
        0.75011483,  0.941981  , -0.16675413, -0.88299528,  1.94641572,
       -1.74613357,  0.95293944,  1.3235599 ,  0.89424861,  2.32589552,
        1.64870398,  2.36715296,  1.2504811 ,  0.77045117, -1.19866905,
       -1.31882236, -0.58119981, -0.92959809,  0.80058176, -0.5698455 ,
       -1.12948806,  1.81992087, -0.08985625,  1.10933639, -0.40256459,
       -0.26139504,  0.88552248, -1.03417761])}
HISTORY 3 [(['episode_short_name', 'bank_id', 'ref_signal', 'raw_readout'], {0: 12, 1: 12, 2: 12, 3: 12}), (['episode_short_name', 'bank_id', 'ref_signal', 'raw_readout'], {0: 12, 1: 12, 2: 12, 3: 12}), (['episode_short_name', 'bank_id', 'ref_signal', 'raw_readout'], {0: 12, 1: 12, 2: 12, 3: 12})]
GRID [('kappa_x', [0.12, 0.11957432749209755, 0.1183118057351089], [0.07042567250790245, 0.07]), ('kappa_y', [0.085, 0.0891582338163552, 0.09313473286151601], [0.0891582338163552, 0.085])]
```

### Turn 4 — `execute_in_remote_docker` (`35471 ms` LLM)
- **Rationale**: Extend the archived solver with wrapped diffusion and upwinding while preserving the nonperiodic regimes. Compare weights and terminal field independently against public references; a mismatch would trigger a forward-matrix comparison to isolate grid spacing, reaction coordinates, or sensor stacking.
- **Alternatives Weighed**: `Replace the archive's direct ridge solve with the SVD-based skill` (rejected: The requested positive-ridge projection uses a small Gram system, and the direct solve avoids an unnecessary dependency.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
import re
src = Path('/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py').read_text()
src = src.replace('from scipy.sparse import diags, eye, kron', 'from scipy.sparse import coo_matrix, diags, eye, kron')
periodic = '''def _periodic_axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    lower = -(kappa + kappa[im]) / (2 * h**2) - np.maximum(velocity, 0) / h
    upper = -(kappa + kappa[ip]) / (2 * h**2) - np.maximum(-velocity, 0) / h
    return coo_matrix((np.concatenate((lower, -lower - upper, upper)),
                       (np.tile(i, 3), np.concatenate((im, i, ip)))),
                      shape=(n, n)).tocsc()


'''
src = src.replace('def _forward(study):', periodic + 'def _forward(study):', 1)
src = src.replace("    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0", "    mode = study['recirculation_loop_mode']\n    sealed = mode == 0 and v_mem > 0")
start = src.index('    ax, ay = ')
end = src.index('        wall = ', start)
src = src[:start] + '''    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if mode > 0.50:
        ax, ay = [_periodic_axis(np.asarray(study['kappa_' + a]),
                                 np.asarray(study['vel_' + a]), study['L' + a])
                  for a in ('x', 'y')]
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        reaction = reaction + 0.75 * mode * np.cos(2 * np.pi * xs / lx) * np.cos(2 * np.pi * ys / ly)
    else:
        ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                        0.0 if sealed else study['robin_' + a], study['L' + a])
                  for a in ('x', 'y')]
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
''' + src[end:]
src, count = re.subn(r'def solve_beta_m97_c03_\w+\(study', 'def solve_beta_m97_c03_batch_03(study', src)
assert count == 1
Path('beta_m97_c03_batch_03_solver.py').write_text(src)
PY
sed -n '/    modes = /,$p' beta_m97_c03_batch_03_solver.py
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/beta_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for c in pickle.load(open('/holdout_task/public_beta_m97_c03_batch_03.pkl','rb')):
 s,g=c['study'],c['gt']; out=m.solve_beta_m97_c03_batch_03(s)
 for p,q in [('weights','w_opt'),('terminal_field','terminal_field_A')]:
  d=out[p]-g[q];print(p,'max_abs',np.max(np.abs(d)),'relative_l2',np.linalg.norm(d)/np.linalg.norm(g[q]))
 print('public primal comparison',out['kkt_certificate']['primal_objective'],g['cert_opt']['primal_objective'])
PY
```
- **Container Exit Code**: `0` (`602 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
teps[first] * direction
            w[f[first]] = bounds[first]
            status[f[first]] = -1 if bounds[first] == 0 else 1
            continue

        w[free] = target
        reduced = H @ w + c - nu
        violations = status * reduced
        release = int(np.argmax(violations))
        if violations[release] <= tol:
            return w, nu
        status[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c03_batch_03(study: dict) -> dict:
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
{"schema_valid": true, "errors": []}
weights max_abs 5.080723845252155e-07 relative_l2 8.881599540435168e-07
terminal_field max_abs 2.5943083147017543e-08 relative_l2 2.6254957747674686e-07
public primal comparison 0.02210909512143305 0.02210908714119178

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed37c0f76616937 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed37c0f (directory).
```

### Turn 5 — `execute_in_remote_docker` (`37726 ms` LLM)
- **Rationale**: The public outputs already agree closely. Check the forward matrix and certificate independently to detect errors hidden by the fitted weights, then exercise calibration fallback, EMI rejection, and budget endpoints to isolate any failing branch before submission.

```bash
cd /holdout_task && python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('candidate', '/holdout_task/beta_m97_c03_batch_03_solver.py')
with open('validate_public_schema.py') as f:
    src = f.read()
ns = {}
exec(src[:src.index('\ndef verify_box_simplex_kkt_certificate')] + '\n' + src[src.index('def verify_box_simplex_kkt_certificate'):src.index('\n    # Unconditional convex supporting-hyperplane')], ns)
with open('public_beta_m97_c03_batch_03.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, g = case['study'], case['gt']
z = s['Z_nuisance']; gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
def project(x): return x - z @ np.linalg.solve(gram, z.T @ x)
S = project(m._forward(s)[0]); r = project(m._calibrate(s))
print('public matrices: max S error', np.max(abs(S - g['S_tilde'])), 'max r error', np.max(abs(r - g['r_y'])))
assert np.max(abs(S - g['S_tilde'])) < 1e-10

def check(result, A, y, budget):
    cert = result['kkt_certificate']; w = np.asarray(cert['weights']); nu = cert['nu_sum']; u = s['u_box']; ridge = s['source_ridge']
    np.testing.assert_array_equal(w, result['weights'])
    res = A @ w - y; grad = A.T @ res + ridge*w
    p = float((res @ res + ridge*(w @ w))/2)
    d = float(p - grad @ w + nu*budget + u*np.minimum(0,grad-nu).sum())
    ll, lu = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
    stat = np.max(abs(grad-nu-ll+lu))
    assert max(abs(w.sum()-budget), -w.min(), w.max()-u, -ll.min(), -lu.min()) < 1e-9
    assert max(np.max(abs(ll*w)),np.max(abs(lu*(u-w)))) < 1e-9
    assert stat < 1e-6 and -1e-9 <= p-d < 1e-6
    assert max(abs(cert['primal_objective']-p),abs(cert['dual_lower_bound']-d)) < 1e-6
    return p-d, stat

out = m.solve_beta_m97_c03_batch_03(s)
print('public independent reference gap and stationarity',check(out,g['S_tilde'],g['r_y'],s['total_budget']))
base = m._calibrate(s); sweep = s['sensor_calibration_sweeps']; banks = sweep['bank_id']
worst = 0.0
for bank in range(4):
    idx = np.flatnonzero(banks == bank)
    for kind, dropped, spike in [('absent',idx,0),('one',idx[1:],0),('two_clean',idx[2:],0),('spike_majority_clean',[],8),('spike_three_points',idx[3:],8)]:
        mask = np.ones(len(banks), dtype=bool); mask[dropped] = False
        sw = {k: np.asarray(v)[mask].copy() for k,v in sweep.items()}
        sw['raw_readout'][np.flatnonzero(sw['bank_id']==bank)[0]] += spike if kind != 'absent' else 0 if np.any(sw['bank_id']==bank) else 0
        trial = dict(s,sensor_calibration_sweeps=sw)
        err = np.max(abs(m._calibrate(trial)-base)); worst=max(worst,err)
        assert err < 2e-5, (bank,kind,err)
        check(m.solve_beta_m97_c03_batch_03(trial), S, project(m._calibrate(trial)), s['total_budget'])
print('worst calibration perturbation',worst)
for budget in [0., 1., s['total_budget'], 11., 12.]:
    check(m.solve_beta_m97_c03_batch_03(dict(s,total_budget=budget)), S, r, budget)
print('budget endpoint and interior certificates passed')
PY
```
- **Container Exit Code**: `1` (`506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public matrices: max S error 1.5265566588595902e-16 max r error 1.2342505373297286e-07
public independent reference gap and stationarity (4.025286617359569e-07, 7.976561044314989e-08)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed37c0f76616937 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed37c0f (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 47, in <module>
IndexError: index 0 is out of bounds for axis 0 with size 0
```

### Turn 6 — `execute_in_remote_docker` (`37703 ms` LLM)
- **Rationale**: The prior stress harness indexed an intentionally absent bank before checking whether to inject a spike. Correct the harness, leave the solver unchanged, and complete the calibration fallback and budget endpoint tests; any failure now identifies the solver branch that needs attention.

```bash
cd /holdout_task && python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c03_batch_03.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, g = case['study'], case['gt']
z = s['Z_nuisance']; gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
def project(x):
    return x - z @ np.linalg.solve(gram, z.T @ x)
S_raw, terminal = m._forward(s)
S = project(S_raw)
base = m._calibrate(s)
u, ridge = s['u_box'], s['source_ridge']

def check(out, A, y, budget, tol):
    cert = out['kkt_certificate']; w = out['weights']; nu = cert['nu_sum']
    np.testing.assert_array_equal(cert['weights'], w)
    ll, lu = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
    residual = A @ w - y; grad = A.T @ residual + ridge * w
    primal = (residual @ residual + ridge * (w @ w)) / 2
    lower = primal - grad @ w + nu * budget + u * np.minimum(0, grad - nu).sum()
    stat = np.max(np.abs(grad - nu - ll + lu))
    assert max(abs(w.sum() - budget), -w.min(), w.max() - u, -ll.min(), -lu.min()) < 1e-9
    assert max(np.max(abs(ll * w)), np.max(abs(lu * (u - w)))) < 1e-9
    assert -1e-10 <= primal - lower < tol and stat < tol
    assert max(abs(cert['primal_objective'] - primal), abs(cert['dual_lower_bound'] - lower)) < tol
    np.testing.assert_allclose(out['terminal_field'], (terminal @ w).reshape(s['nx'], s['ny'], order='F'), atol=1e-12, rtol=0)
    return max(primal - lower, 0), stat

out = m.solve_beta_m97_c03_batch_03(s)
print('public certificate against reference, gap and stationarity:', check(out, g['S_tilde'], g['r_y'], s['total_budget'], 1e-6))
sweep = s['sensor_calibration_sweeps']; banks = sweep['bank_id']
worst_cal = worst_gap = worst_stat = 0.0
for bank in range(4):
    idx = np.flatnonzero(banks == bank)
    for kind, dropped, spike in [('absent', idx, 0), ('one', idx[1:], 0), ('two_clean', idx[2:], 0), ('spike_majority_clean', [], 8), ('spike_three_points', idx[3:], 8)]:
        mask = np.ones(len(banks), dtype=bool); mask[dropped] = False
        sw = {k: np.asarray(v)[mask].copy() for k, v in sweep.items()}
        if spike:
            sw['raw_readout'][np.flatnonzero(sw['bank_id'] == bank)[0]] += spike
        trial = dict(s, sensor_calibration_sweeps=sw)
        cal = m._calibrate(trial)
        err = np.max(abs(cal - base))
        assert err < 2e-5, (bank, kind, err)
        worst_cal = max(worst_cal, err)
        gap, stat = check(m.solve_beta_m97_c03_batch_03(trial), S, project(cal), s['total_budget'], 1e-9)
        worst_gap, worst_stat = max(worst_gap, gap), max(worst_stat, stat)
for budget in [0., 1., s['total_budget'], 11., 12.]:
    gap, stat = check(m.solve_beta_m97_c03_batch_03(dict(s, total_budget=budget)), S, project(base), budget, 1e-9)
    worst_gap, worst_stat = max(worst_gap, gap), max(worst_stat, stat)
print('20 bank perturbations and 5 budgets; worst calibration error, gap, stationarity:', worst_cal, worst_gap, worst_stat)
PY
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public certificate against reference, gap and stationarity: (4.025286617359569e-07, 7.976561044314989e-08)
20 bank perturbations and 5 budgets; worst calibration error, gap, stationarity: 1.4644949333231239e-06 9.020562075079397e-16 3.3306690738754696e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed37c0f76616937 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed37c0f (directory).
```

### Turn 7 — `submit_completed_task` (`15179 ms` LLM)
- **Rationale**: The verified artifact passes the public contract and independent certificate checks for missing calibration banks, EMI spikes, and budget endpoints. No further edits are warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_03_solver.py': 'FILE:9230'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema passed, the KKT gap against the public reference was 4.03e-7 (below 1e-6), and all 20 calibration stress cases and five budget cases passed.`

### Turn 8 — `execute_in_remote_docker` (`40652 ms` LLM)
- **Rationale**: Audit the supported failure modes on public measurements: a seam-sign or conservation failure would require a stencil correction; a large leave-one-sweep-out shift would identify a bank fit to investigate; telemetry dependence would violate the input contract.
- **Alternatives Weighed**: `Test binary classification, relay modes, or HJB drift` (rejected: The reactor contract defines a continuous inverse problem with a specified transport stencil; it supplies no classification labels, relay dynamics, or HJB operator.)

```bash
cd /holdout_task && python3 -I /holdout_task/validate_public_schema.py && python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_m97_c03_batch_03.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    s, g = case['study'], case['gt']
    if s['recirculation_loop_mode'] > 0.5:
        for a in ('x', 'y'):
            k, v, L = (np.asarray(s[key + a]) for key in ('kappa_', 'vel_', 'L'))
            A = m._periodic_axis(k, v, L).toarray()
            n = len(k); h = L / (n - 1)
            for i in (0, n - 1):
                im, ip = (i - 1) % n, (i + 1) % n
                row = np.zeros(n)
                row[im] = -(k[i] + k[im]) / (2*h*h) - max(v[i], 0)/h
                row[ip] = -(k[i] + k[ip]) / (2*h*h) - max(-v[i], 0)/h
                row[i] = -row[im] - row[ip]
                np.testing.assert_allclose(A[i], row, rtol=0, atol=1e-12)
            balance = float(np.max(np.abs(A @ np.ones(n))))
            print(a, 'constant-field transport residual', balance)
            assert balance < 1e-12
    Z = s['Z_nuisance']
    gram = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def project(x):
        return x - Z @ np.linalg.solve(gram, Z.T @ x)
    design = project(m._forward(s)[0])
    np.testing.assert_allclose(design, g['S_tilde'], atol=1e-12, rtol=0)
    base = m.solve_beta_m97_c03_batch_03(s)
    sweep = s['sensor_calibration_sweeps']
    b = np.asarray(sweep['bank_id'])
    r = project(m._calibrate(s)); ref = g['r_y']
    rows = np.asarray(s['row_bank_id'])
    for bank in range(4):
        ix = np.flatnonzero(b == bank)
        heldout = []
        if len(ix) >= 5:
            for j in ix:
                keep = np.arange(len(b)) != j
                sw = {k: np.asarray(v)[keep] for k,v in sweep.items()}
                trial = m.solve_beta_m97_c03_batch_03(dict(s, sensor_calibration_sweeps=sw))
                heldout.append(float(np.max(np.abs(trial['weights'] - base['weights']))))
        print('bank', bank, 'sweep points', len(ix), 'projected target max reference error', float(np.max(np.abs((r-ref)[rows == bank]), initial=0.0)), 'max leave-one-sweep-out weight shift', max(heldout, default=None))
        assert max(heldout, default=0) < 2e-5
    if 'aux_manifold_telemetry' in s:
        without_aux = dict(s)
        del without_aux['aux_manifold_telemetry']
        ret = m.solve_beta_m97_c03_batch_03(without_aux)
        for key in ('weights', 'terminal_field'):
            np.testing.assert_array_equal(base[key], ret[key])
        assert base['kkt_certificate'] == ret['kkt_certificate']
        print('uncoupled auxiliary telemetry: exact output invariance')
PY
```
- **Container Exit Code**: `0` (`928 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
x constant-field transport residual 3.552713678800501e-15
y constant-field transport residual 3.552713678800501e-15
bank 0 sweep points 12 projected target max reference error 1.2342505373297286e-07 max leave-one-sweep-out weight shift 5.840235549336992e-08
bank 1 sweep points 12 projected target max reference error 3.884386687302843e-08 max leave-one-sweep-out weight shift 1.2716562114434282e-07
bank 2 sweep points 12 projected target max reference error 1.0322043930699465e-07 max leave-one-sweep-out weight shift 1.1666926333364636e-07
bank 3 sweep points 12 projected target max reference error 8.55957846823685e-08 max leave-one-sweep-out weight shift 8.891026187507478e-08
uncoupled auxiliary telemetry: exact output invariance

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed37c0f76616937 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed37c0f (directory).
```

### Turn 9 — `submit_completed_task` (`15867 ms` LLM)
- **Rationale**: Clean-process verification confirmed the wrapped stencil, stable bank calibration, and exact output invariance to auxiliary telemetry. No failing check warrants a solver change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_03_solver.py': 'FILE:9230'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema, periodic boundary checks, 20 calibration stress cases, and five budget cases passed. The public KKT gap was 4.03e-7, below the 1e-6 tolerance.`

### Turn 10 — `execute_in_remote_docker` (`20232 ms` LLM)
- **Rationale**: The previous audit tested two clean points and EMI with at least three points, but not their intersection. Two points alone fit an affine line exactly even if one is corrupted. Inspect whether the sparse-sweep branch already detects that case and whether real prefix calibration is stable enough to distinguish EMI from a legitimate fit before considering a change.
- **Alternatives Weighed**: `Repeat boundary conservation, telemetry invariance, and the public schema check` (rejected: Those checks already passed on the unchanged artifact; repeating them would not address an untested failure mode.)

```bash
sed -n '1,85p' /holdout_task/beta_m97_c03_batch_03_solver.py && python3 -I - <<'PY'
import json
import numpy as np
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)
for sweep in history:
    b, x, y = (np.asarray(sweep[k]) for k in ('bank_id','ref_signal','raw_readout'))
    print(sweep['episode_short_name'], [(int(i), np.round(np.polyfit(x[b==i],y[b==i],1),7).tolist()) for i in np.unique(b)])
PY
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, core, axis=1)
    best = np.argmin(np.sum(core_error**2, axis=1))
    gain, offset = _fit_line(x[core[best]], y[core[best]])
    return np.abs(y - (gain * x + offset)) <= 3.0


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


def _axis(kappa, velocity, robin, length):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    main = np.empty(n)
    main[1:-1] = -lower[:-1] - upper[1:]
    main[0] = -upper[0] + robin / h
    main[-1] = -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _periodic_axis(kappa, velocity, length):
beta_m97_c03_s00 [(0, [1.0, 0.0]), (1, [1.0, 0.0]), (2, [1.0, -0.0]), (3, [1.0, -0.0])]
beta_m97_c03_batch_01 [(0, [1.27, 0.1299999]), (1, [0.7800001, -0.1599999]), (2, [1.18, -0.0899999]), (3, [0.8399999, 0.1099999])]
beta_m97_c03_batch_02 [(0, [1.27, 0.1300001]), (1, [0.7799999, -0.16]), (2, [1.18, -0.0900001]), (3, [0.84, 0.1099998])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed37c0f76616937 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed37c0f (directory).
```

### Turn 11 — `submit_completed_task` (`18322 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process validation, independent boundary and KKT checks, 20 calibration perturbations, and five budget cases. The final inspection identified a two-point identifiability limit, not a failure on the public measurements.
- **Alternatives Weighed**: `Use archived calibration to screen every two-point sweep for EMI` (rejected: Two points fit an affine line exactly, so an archive comparison would impose a stability assumption and consult history even for clean two-point sweeps, contrary to the fallback rule.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema and periodic boundary checks passed. The public KKT gap was 4.03e-7, below the 1e-6 tolerance. A spike in a two-point calibration sweep cannot be identified from those two readings alone.`

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
