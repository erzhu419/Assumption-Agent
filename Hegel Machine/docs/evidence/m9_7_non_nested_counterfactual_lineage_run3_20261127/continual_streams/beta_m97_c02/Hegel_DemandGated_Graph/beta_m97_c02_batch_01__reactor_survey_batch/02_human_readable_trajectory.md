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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `326,054` (`315,761` / `107,520` / `10,293`) |
| **Total Episode Wall Time (s)** | `352.62s` |

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

### Turn 1 — `execute_in_remote_docker` (`19006 ms` LLM)
- **Rationale**: Inspect the validator, prior implementation, and public data shapes to catch interface or grid-index mismatches before reusing code. Check for a verified profiling or KKT primitive before implementing either from scratch.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The current batch explicitly changes calibration routing and the fouled-wall operator.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; sed -n '1,320p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'; python3 - <<'PY'
import pickle
with open('/holdout_task/public_beta_m97_c02_batch_01.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None),str(v)[:450] if getattr(v,'size',0)<70 else '')
PY
```
- **Container Exit Code**: `1` (`465 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`19831 ms` LLM)
- **Rationale**: The pickle contains a list, so inspect each study separately. Compare current and archived sweep formats to determine when fallback applies, and check the profiling primitive against the required residual transform before choosing an implementation.
- **Alternatives Weighed**: `Implement from the truncated archive output` (rejected: The missing assembly code could conceal a grid-order or boundary-coefficient mismatch.)

```bash
cat /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; tail -45 /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance; python3 - <<'PY'
import pickle
import numpy as np
p='/holdout_task/public_beta_m97_c02_batch_01.pkl'
with open(p,'rb') as f: studies=pickle.load(f)
print('public study count:',len(studies))
for i,d in enumerate(studies):
 print('study',i)
 for k,v in d.items():
  print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:320] if getattr(v,'size',0)<=24 else '')
 c=d['sensor_calibration_sweeps']
 if isinstance(c,dict):
  b,x,y=(np.asarray(c[k]) for k in ('bank_id','ref_signal','raw_readout'))
 else:
  b,x,y=(np.asarray([r[k] for r in c]) for k in ('bank_id','ref_signal','raw_readout'))
 for j in range(4):
  m=b==j
  print('bank',j,'points',int(m.sum()),'reference',x[m].tolist(),'raw',y[m].tolist())
PY
```
- **Container Exit Code**: `1` (`418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_level_applicability": {
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
public study count: 1
study 0
study dict None {'Lx': 1.5, 'Ly': 1.3, 'nx': 18, 'ny': 16, 'kappa_x': array([0.12      , 0.11957433, 0.11831181, 0.11625543, 0.11347522,
       0.11006587, 0.10614346, 0.10184157, 0.09730671, 0.09269329,
       0.08815843, 0.08385654, 0.07993413, 0.07652478, 0.07374457,
       0.07168819, 0.07042567, 0.07      ]), 'vel_x': array([0.24
gt dict None {'S_tilde': array([[ 0.03780545,  0.03766456,  0.01397862, ...,  0.01028596,
         0.0020878 ,  0.00080043],
       [ 0.00425646,  0.01328167,  0.0217077 , ...,  0.01682021,
         0.02673095,  0.01175644],
       [ 0.00820823,  0.02566284,  0.03279221, ...,  0.00979239,
         0.01545349,  0.00577819],
       .
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
KeyError: 'sensor_calibration_sweeps'
```

### Turn 3 — `execute_in_remote_docker` (`19016 ms` LLM)
- **Rationale**: Read the study inside the public case wrapper and inspect just the missing assembly and calibration details. Sweep counts and outliers determine fallback routing; the archive code determines the boundary indices and field layout.
- **Alternatives Weighed**: `Modify the prior solver without reading its forward assembly` (rejected: The truncated output omitted the axis ordering and shift assembly, which determine where to apply the new boundary and sink terms.)

```bash
sed -n '52,190p' /tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; tail -35 /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_beta_m97_c02_batch_01.pkl','rb') as f:
    cases=pickle.load(f)
for case in cases:
    d=case['study']
    for k,v in d.items():
        a=np.asarray(v) if not isinstance(v,dict) else None
        print(k, list(v) if a is None else (a.shape, a.tolist() if 0<a.size<=16 else ''))
    print('public reference keys:', list(case['gt']))
    c=d['sensor_calibration_sweeps']; keys=('bank_id','ref_signal','raw_readout')
    b,x,y=(np.asarray(c[k] if isinstance(c,dict) else [r[k] for r in c]) for k in keys)
    for j in range(4):
        m=b==j
        print('bank',j,'ref/raw:',list(zip(x[m].tolist(),y[m].tolist())))
PY
```
- **Container Exit Code**: `0` (`430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7, 0.34228993532094626, 0.33, 0.31694313003061203, 0.30326237921249266, 0.28910763671448636, 0.2746339848574715, 0.26])
V_reaction_2d ((18, 16), '')
recirculation_loop_mode ((), 0.0)
past_vortex_circulation_trace ((16,), [0.204836, 0.269357, 0.145213, 0.152527, 0.933469, 0.74663, 0.730277, 0.926086, 1.047529, 0.185363, 0.150684, 0.33255, 0.334021, 0.121958, 0.17916, 0.110959])
critical_vortex_circulation ((), 0.5)
mu_shifts ((4,), [0.0, 0.22, 0.52, 0.9])
source_modes ((12, 18, 16), '')
sensor_ij ((36, 2), '')
row_bank_id ((144,), '')
sensor_calibration_sweeps ['bank_id', 'ref_signal', 'raw_readout']
Z_nuisance ((144, 5), '')
y_obs_raw ((144,), '')
aux_manifold_telemetry ((144, 3), '')
nuisance_ridge ((), 0.65)
source_ridge ((), 0.04)
u_box ((), 1.0)
total_budget ((), 2.6)
public reference keys: ['S_tilde', 'r_y', 'w_opt', 'cert_opt', 'terminal_field_A']
bank 0 ref/raw: [(-0.791143, -0.7908257940335697), (-0.216611, -0.3254549152630075), (-1.13301, -1.0677380668585081), (-1.30817, -1.20961752641936), (1.713448, 1.237892944400269), (0.875702, 0.5593188231729911), (0.943663, 0.6143670736469865), (-0.29371, -0.38790518472689206), (1.475187, 1.0449015765613343), (1.843625, 1.3433363422175935), (0.932592, 0.6053995411778677), (2.112826, 1.5613891081282518)]
bank 1 ref/raw: [(0.840203, 1.257469875152389), (-0.917936, -1.080854853810764), (-1.382269, -1.698417705018294), (-1.346584, -1.6509566778377247), (-0.366771, -0.34780542094514555), (-0.228184, -0.16348481598124392), (1.263794, 1.8208460827055826), (1.49297, 2.125649938606239), (1.843164, 2.5914080230106253), (0.64334, 0.995642255176037), (2.147042, 2.9955657600901424), (-0.014648, 0.1205183876174737)]
bank 2 ref/raw: [(1.758375, 1.6222025632880868), (1.79159, 1.650767588034738), (1.639504, 1.5199734062487433), (1.751388, 1.6161935134826684), (1.357273, 1.2772547888056007), (-0.888478, -0.6540911007883645), (0.615053, 0.6389457030069186), (0.717531, 0.7270766954773403), (-1.457866, -1.1437647812833778), (1.95727, 1.7932521895374542), (2.015157, 1.8430351157525988), (1.55983, 1.4514537767713291)]
bank 3 ref/raw: [(0.907239, 0.9686868152103886), (-0.132195, -0.27863398791233324), (-0.452725, -0.6632699895528311), (-0.039361, -0.16723307563141235), (0.216018, 0.13922170280841414), (1.033672, 1.1204064606192279), (-0.935186, -1.2422231705602746), (-0.838871, -1.1266451790428873), (0.165606, 0.07872692133455879), (-0.350606, -0.5407271942687467), (-1.291083, -1.6692996900180148), (-1.18116, -1.5373919662063993)]
```

### Turn 4 — `execute_in_remote_docker` (`40484 ms` LLM)
- **Rationale**: Reuse the archived stencil and active-set solve, and change the fouling and calibration rules. Compare the projected operator and calibrated observations separately so a boundary mismatch cannot be masked by the inverse fit.
- **Alternatives Weighed**: `Seal both axes` (rejected: The prescribed sorption profile identifies the x walls; compare the forward matrix directly with the public reference to test that interpretation.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py').read_text()
src = src.replace('Calibrated open-Robin reactor tomography', 'Calibrated history-dependent reactor tomography')
start = src.index('        clean = np.ones')
end = src.index('        gains[bank], offsets[bank]', start)
src = src[:start] + '''        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, 1)
            distinct = x[i] != x[j]
            if distinct.any():
                i, j = i[distinct], j[distinct]
                slope = np.median((y[i] - y[j]) / (x[i] - x[j]))
                delta = y - slope * x
                clean = np.abs(delta - np.median(delta)) <= 3.0
        if clean.sum() >= 4 or (x.size >= 2 and clean.all()):
            ref = x[clean]
            gain, offset = np.linalg.lstsq(np.column_stack((ref, np.ones_like(ref))), y[clean], rcond=None)[0]
        else:
            prior = _historical_calibration(bank)
            if prior is None:
                raise ValueError(f'No archived calibration for bank {bank}')
            gain, offset = prior
''' + src[end:]
a = '''    ax = _axis(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))'''
b = '''    lx, ly = study['Lx'], study['Ly']
    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace']) - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0.0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx / (nx - 1))'''
assert a in src
src = src.replace(a, b)
a = "    operator[diag] += np.asarray(study['V_reaction_2d']).ravel(order='F')"
b = '''    reaction = np.asarray(study['V_reaction_2d'])
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        walls = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * walls * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator[diag] += reaction.ravel(order='F')'''
assert a in src
src = src.replace(a, b)
import re
src, n = re.subn(r'def solve_\w+\(', 'def solve_beta_m97_c02_batch_01(', src)
assert n == 1
Path('/holdout_task/beta_m97_c02_batch_01_solver.py').write_text(src)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',root+'beta_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(root+'public_beta_m97_c02_batch_01.pkl','rb') as f: cases=pickle.load(f)
for c in cases:
 d,g=c['study'],c['gt']; out=m.solve_beta_m97_c02_batch_01(d)
 z=d['Z_nuisance']; s,_=m._forward(d); raw=np.column_stack((m._calibrate(d),s)); p=raw-z@np.linalg.solve(z.T@z+d['nuisance_ridge']*np.eye(z.shape[1]),z.T@raw)
 for k,pred,truth in [('S',p[:,1:],g['S_tilde']),('y',p[:,0],g['r_y']),('weights',out['weights'],g['w_opt']),('field',out['terminal_field'],g['terminal_field_A'])]:
  print(k,'maxabs',np.max(np.abs(pred-truth)),'relative L2',np.linalg.norm(pred-truth)/np.linalg.norm(truth))
 cert=out['kkt_certificate']; print('reported gap',cert['primal_objective']-cert['dual_lower_bound'])
PY
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
d=json.load(open(p)); print('history shape',type(d).__name__,list(d)[:6] if isinstance(d,dict) else d[:1])
PY
```
- **Container Exit Code**: `1` (`412 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bf2ba5d350c21e39 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bf2ba5d3 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 64, in <module>
    out = mod.solve_beta_m97_c02_batch_01(st)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/beta_m97_c02_batch_01_solver.py", line 154, in solve_beta_m97_c02_batch_01
    raise ValueError('Study is outside the open-Robin regime')
ValueError: Study is outside the open-Robin regime
```

### Turn 5 — `execute_in_remote_docker` (`29242 ms` LLM)
- **Rationale**: Remove the obsolete open-Robin guard, then compare the projected forward matrix and calibrated observations separately against the public reference. This distinguishes a boundary error from a calibration error and tests whether the certificate holds for both the reconstructed and reference objectives.
- **Alternatives Weighed**: `Change the wall stencil now` (rejected: The validator failed before reaching the forward assembly, so it gave no evidence about the stencil.)

```bash
set -e
python3 - <<'PY'
import ast
from pathlib import Path
p = Path('/holdout_task/beta_m97_c02_batch_01_solver.py')
s = p.read_text()
f = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == 'solve_beta_m97_c02_batch_01')
(guard,) = [n for n in f.body if isinstance(n, ast.If) and 'Study is outside the open-Robin regime' in ast.get_source_segment(s, n)]
assert not guard.orelse and len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
lines = s.splitlines(keepends=True)
del lines[guard.lineno-1:guard.end_lineno]
p.write_text(''.join(lines))
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import ast, importlib.util, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'beta_m97_c02_batch_01_solver.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
nodes = ast.parse((root/'validate_public_schema.py').read_text()).body
ver = {}
exec(compile(ast.Module(body=[n for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name=='verify_box_simplex_kkt_certificate'],type_ignores=[]), 'public_checker', 'exec'), ver)
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f: cases = pickle.load(f)
for c in cases:
 d,g=c['study'],c['gt']; out=m.solve_beta_m97_c02_batch_01(d)
 z=d['Z_nuisance']; s,_=m._forward(d); raw=np.column_stack((m._calibrate(d),s)); p=raw-z@np.linalg.solve(z.T@z+d['nuisance_ridge']*np.eye(z.shape[1]),z.T@raw)
 for k,pred,truth in [('S',p[:,1:],g['S_tilde']),('y',p[:,0],g['r_y']),('weights',out['weights'],g['w_opt']),('field',out['terminal_field'],g['terminal_field_A'])]:
  print(k,'maxabs',np.max(np.abs(pred-truth)),'relative L2',np.linalg.norm(pred-truth)/np.linalg.norm(truth))
 for label,design,y in [('reconstructed',p[:,1:],p[:,0]),('public_reference',g['S_tilde'],g['r_y'])]:
  print(label,ver['verify_box_simplex_kkt_certificate'](design,y,d['source_ridge'],d['u_box'],d['total_budget'],out['kkt_certificate']))
print((Path('/tmp/hegel_continual_archive')/'archive_calibration_history.json').read_text()[:2300])
PY
```
- **Container Exit Code**: `0` (`506 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
174941446927e-16}
public_reference {'valid': False, 'reason': 'reported_primal_mismatch:1.470e-03', 'gap': 1000000000.0}
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

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=691b37b6c2762f7e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_691b37b6 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26033 ms` LLM)
- **Rationale**: Compare the four boundary choices against the public design and the terminal field evaluated at fixed reference weights. If a choice matches both, update the stencil; if none does, isolate the sorption or interior assembly before changing the artifact.
- **Alternatives Weighed**: `Retune calibration or source ridge` (rejected: Neither changes the projected design, where the main discrepancy is already visible.)
- **ATMS NOGOOD DECLARED**: Refuted `The x-localized sorption mask determines which axis must lose Robin boundary flux.` | Witness: `The x-only model misses the projected design by 21.20% relative L2, while the projected observations agree to 1.23e-6 relative L2.` | Blanket Diagnosis: `Calibration and the optimizer do not explain the discrepancy: the design is independent of readings, and the certificate passes only for the reconstructed objective.` | Sublated Pivot: `Hold the sorption term fixed and test the discrete boundary choices independently on each axis.`

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I - <<'PY'
from pathlib import Path
import pickle
import numpy as np
root = Path('/holdout_task')
p = root/'beta_m97_c02_batch_01_solver.py'
src = p.read_text()
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
x_expr = "0.0 if sealed else study['robin_x']"
y_expr = "study['vel_y'], study['robin_y'],"
assert src.count(x_expr) == src.count(y_expr) == 1
for seal_x, seal_y in [(True, False), (True, True), (False, True), (False, False)]:
    variant = src if seal_x else src.replace(x_expr, "study['robin_x']")
    if seal_y:
        variant = variant.replace(y_expr, "study['vel_y'], 0.0 if sealed else study['robin_y'],")
    ns = {}
    exec(compile(variant, str(p), 'exec'), ns)
    for case in cases:
        d,g = case['study'],case['gt']
        S,F = ns['_forward'](d)
        Z = d['Z_nuisance']
        P = S-Z@np.linalg.solve(Z.T@Z+d['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@S)
        f = (F@g['w_opt']).reshape(d['nx'],d['ny'],order='F')
        print('sealed axes', seal_x, seal_y, 'design rel L2', np.linalg.norm(P-g['S_tilde'])/np.linalg.norm(g['S_tilde']), 'fixed-weight field rel L2', np.linalg.norm(f-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']))
PY
sed -n '1,54p' /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`387 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ll = np.asarray(cert["lam_lower"], dtype=float).ravel()
        lu = np.asarray(cert["lam_upper"], dtype=float).ravel()
        nu = float(cert["nu_sum"])
        p_rep = float(cert["primal_objective"])
        d_rep = float(cert["dual_lower_bound"])
    except Exception as exc:
        return {"valid": False, "reason": f"cast_error:{exc}", "gap": 1e9}
    if w.shape != (M,) or ll.shape != (M,) or lu.shape != (M,):
        return {"valid": False, "reason": f"shape_mismatch:{w.shape}", "gap": 1e9}
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(ll)) and np.all(np.isfinite(lu)) and math.isfinite(nu) and math.isfinite(p_rep) and math.isfinite(d_rep)):
        return {"valid": False, "reason": "non_finite_values", "gap": 1e9}
    u_b = float(u_box)
    bt = float(b_tot)
    lw = float(lam_w)
    if float(np.min(w)) < -feas_tol or float(np.max(w)) > u_b + feas_tol:
        return {"valid": False, "reason": "box_violation", "gap": 1e9}
    if abs(float(np.sum(w)) - bt) > feas_tol * max(1.0, bt):
        return {"valid": False, "reason": f"budget_sum_violation:{abs(float(np.sum(w)) - bt):.3e}", "gap": 1e9}
    if float(np.min(ll)) < -feas_tol or float(np.min(lu)) < -feas_tol:
        return {"valid": False, "reason": "negative_bound_multipliers", "gap": 1e9}

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

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=691b37b6c2762f7e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_691b37b6 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30749 ms` LLM)
- **Rationale**: Promote the stencil that matches the public reference. Check the certificate against the reference objective, and test the two ways the fouling gate stays inactive to catch accidental sealing. Inspect the fallback parser before exercising sparse and EMI-contaminated calibration sweeps.
- **Alternatives Weighed**: `Change the sorption mask or calibration` (rejected: Sealing both axes already reproduces the public design and fixed-weight terminal field to machine precision without either change.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c02_batch_01_solver.py')
s = p.read_text()
a = "study['vel_y'], study['robin_y'],"
assert s.count(a) == 1
p.write_text(s.replace(a, "study['vel_y'], 0.0 if sealed else study['robin_y'],"))
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import ast, importlib.util, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
m = load('candidate', root/'beta_m97_c02_batch_01_solver.py')
old = load('prior', '/tmp/hegel_continual_archive/beta_m97_c02_s00_solver.py')
nodes = ast.parse((root/'validate_public_schema.py').read_text()).body
ns = {}
exec(compile(ast.Module(body=[n for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name=='verify_box_simplex_kkt_certificate'],type_ignores=[]), 'public_checker', 'exec'), ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f: cases = pickle.load(f)
for case in cases:
    d,g = case['study'],case['gt']; out = m.solve_beta_m97_c02_batch_01(d)
    cert = verify(g['S_tilde'],g['r_y'],d['source_ridge'],d['u_box'],d['total_budget'],out['kkt_certificate'])
    print('reference certificate:',cert)
    assert cert['valid'], cert
    for key,truth in [('weights','w_opt'),('terminal_field','terminal_field_A')]:
        rel = np.linalg.norm(out[key]-g[truth])/np.linalg.norm(g[truth]); print(key,'relative L2',rel)
        assert rel < 1e-4
    for label, changes in [('loop-enabled', {'recirculation_loop_mode':1}), ('threshold-equality', {'past_vortex_circulation_trace': np.full_like(d['past_vortex_circulation_trace'],d['critical_vortex_circulation'])})]:
        branch = dict(d, **changes)
        err = max(np.max(np.abs(a-b)) for a,b in zip(m._forward(branch),old._forward(branch)))
        print(label,'maximum open-Robin regression error',err); assert err < 1e-12
PY
sed -n '1,57p' /holdout_task/beta_m97_c02_batch_01_solver.py
```
- **Container Exit Code**: `0` (`542 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
alibrated history-dependent reactor tomography with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(bank):
    if not _HISTORY.is_file():
        return None
    with _HISTORY.open() as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get('calibration_history', records.get('history', records))
    if isinstance(records, dict):
        records = records.get(str(bank), [])
        if isinstance(records, dict):
            records = [records]
    for row in reversed(records):
        if int(row.get('bank_id', bank)) == bank and 'gain' in row and 'offset' in row:
            return float(row['gain']), float(row['offset'])
    return None


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    banks, x_all, y_all = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    gains, offsets = np.empty(4), np.empty(4)
    for bank in range(4):
        x, y = x_all[banks == bank], y_all[banks == bank]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, 1)
            distinct = x[i] != x[j]
            if distinct.any():
                i, j = i[distinct], j[distinct]
                slope = np.median((y[i] - y[j]) / (x[i] - x[j]))
                delta = y - slope * x
                clean = np.abs(delta - np.median(delta)) <= 3.0
        if clean.sum() >= 4 or (x.size >= 2 and clean.all()):
            ref = x[clean]
            gain, offset = np.linalg.lstsq(np.column_stack((ref, np.ones_like(ref))), y[clean], rcond=None)[0]
        else:
            prior = _historical_calibration(bank)
            if prior is None:
                raise ValueError(f'No archived calibration for bank {bank}')
            gain, offset = prior
        gains[bank], offsets[bank] = gain, offset
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[rows]) / gains[rows]


def _axis(kappa, velocity, robin, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9b0466a3d066fa3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_f9b0466a (directory).
```

### Turn 8 — `execute_in_remote_docker` (`43208 ms` LLM)
- **Rationale**: Fit fallback coefficients from the actual archived sweeps. Test every bank with missing, sparse, and spiked calibration rows to detect incorrect fallback or accidental replacement of valid in-batch fits. Leave the verified transport unchanged.
- **Alternatives Weighed**: `Keep the inherited history parser` (rejected: The mounted history stores sweep columns, not scalar bank IDs or fitted gain and offset records.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c02_batch_01_solver.py')
s = p.read_text()
a = s.index('def _historical_calibration(')
b = s.index('\n\ndef _calibrate(', a)
s = s[:a] + '''def _historical_calibration(bank):
    with _HISTORY.open() as handle:
        records = json.load(handle)
    for sweep in reversed(records):
        banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        selected = banks == bank
        if np.count_nonzero(selected) >= 2 and np.ptp(x[selected]) > 0:
            return _fit_line(x[selected], y[selected])
    raise ValueError(f'No archived calibration for bank {bank}')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
''' + s[b:]
a = s.index('        clean = np.ones')
b = s.index('    rows = ', a)
s = s[:a] + '''        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            # Repeated-median slope keeps a spike from displacing the affine fit.
            slopes = []
            for i in range(x.size):
                different = x != x[i]
                if different.any():
                    slopes.append(np.median((y[different] - y[i]) / (x[different] - x[i])))
            if slopes:
                delta = y - np.median(slopes) * x
                clean = np.abs(delta - np.median(delta)) <= 3.0
        if clean.sum() >= 4 or (x.size >= 2 and clean.all()):
            gains[bank], offsets[bank] = _fit_line(x[clean], y[clean])
        else:
            gains[bank], offsets[bank] = _historical_calibration(bank)
''' + s[b:]
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'beta_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(root+'public_beta_m97_c02_batch_01.pkl','rb') as f: d=pickle.load(f)[0]['study']
bank,x,_=(np.asarray(d['sensor_calibration_sweeps'][k]) for k in ('bank_id','ref_signal','raw_readout'))
rb=np.asarray(d['row_bank_id']); truth=np.linspace(-.1,.2,len(rb)); gains=np.array([.81,1.33,.86,1.2]); offs=np.array([-.15,.14,.11,-.12]); y=gains[bank]*x+offs[bank]
worst=0.
for j in range(4):
    idx=np.flatnonzero(bank==j)
    for kind in ['missing','single','two-clean','three-clean','four-clean','four-with-spike','five-with-spike','full-with-spike']:
        count={'missing':0,'single':1,'two-clean':2,'three-clean':3,'four-clean':4,'four-with-spike':4,'five-with-spike':5}.get(kind,len(idx))
        keep=(bank!=j); keep[idx[:count]]=True
        for spike in idx[:count] if 'spike' in kind else [None]:
            z=y.copy()
            if spike is not None: z[spike]+=8
            raw=gains[rb]*truth+offs[rb]
            if kind in ('missing','single','four-with-spike'): raw[rb==j]=truth[rb==j]
            branch=dict(d,sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=x[keep],raw_readout=z[keep]),y_obs_raw=raw)
            err=float(np.max(np.abs(m._calibrate(branch)-truth))); worst=max(err,worst)
            assert err < 1e-12,(j,kind,spike,err)
print('calibration routing max absolute error',worst)
PY
sed -n '96,205p' /holdout_task/beta_m97_c02_batch_01_solver.py
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
= np.argmin(alpha)
                w[free] = np.clip(w[free] + alpha[hit] * direction, 0, upper)
                state[free[hit]] = -1 if below[hit] else 1
                w[free[hit]] = 0 if below[hit] else upper
                continue
            w[free] = candidate
        else:
            gradient = hess @ w - linear
            a = np.min(gradient[low], initial=np.inf)
            b = np.max(gradient[high], initial=-np.inf)
            if b > a + 1e-12:
                state[np.flatnonzero(low)[np.argmin(gradient[low])]] = 0
                state[np.flatnonzero(high)[np.argmax(gradient[high])]] = 0
                continue
            nu = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
        reduced = hess @ w - linear - nu
        violation = state * reduced
        if np.max(violation) <= 1e-12:
            return w, float(nu)
        state[np.argmax(violation)] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c02_batch_01(study: dict) -> dict:
    calibrated = _calibrate(study)
    forward, fields = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((calibrated, forward))
    values -= z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)
    target, design = values[:, 0], values[:, 1:]
    ridge, upper, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(design, target, ridge, upper, budget)
    residual = design @ w - target
    primal = 0.5 * float(residual @ residual + ridge * (w @ w))
    reduced = design.T @ residual + ridge * w - nu
    lower, upper_mult = np.maximum(reduced, 0), np.maximum(-reduced, 0)
    # Equivalent to the supporting-hyperplane bound, without cancellation in the gap.
    gap = float(lower @ w + upper_mult @ (upper - w) + nu * (w.sum() - budget))
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=lower, lam_upper=upper_mult,
                primal_objective=primal, dual_lower_bound=primal - max(0.0, gap))
    shape = int(study['nx']), int(study['ny'])
    return dict(weights=w, terminal_field=(fields @ w).reshape(shape, order='F'), kkt_certificate=cert)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29d4a7623d10e09c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_29d4a762 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`33121 ms` LLM)
- **Rationale**: Verify the exact final file in an isolated Python process. A public-reference failure would identify a regression from the calibration edit; an endpoint or mixed-bound failure would identify an active-set defect that the public budget does not exercise.

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import ast
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
path = root / 'beta_m97_c02_batch_01_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
nodes = ast.parse((root / 'validate_public_schema.py').read_text()).body
ns = {}
exec(compile(ast.Module(body=[n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate'], type_ignores=[]), 'public_checker', 'exec'), ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root / 'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
for case in cases:
    d, g = case['study'], case['gt']
    out = mod.solve_beta_m97_c02_batch_01(d)
    cert = verify(g['S_tilde'], g['r_y'], d['source_ridge'], d['u_box'], d['total_budget'], out['kkt_certificate'])
    assert cert['valid'], cert
    np.testing.assert_array_equal(out['weights'], out['kkt_certificate']['weights'])
    errors = {k: float(np.linalg.norm(out[k] - g[r]) / np.linalg.norm(g[r])) for k, r in [('weights', 'w_opt'), ('terminal_field', 'terminal_field_A')]}
    assert max(errors.values()) < 1e-4, errors
    print('public reference gap', cert['gap'], 'relative errors', errors)

rng = np.random.default_rng(873)
worst = 0.0
for trial in range(120):
    S = rng.normal(size=(24, 12)) * 0.1
    y = rng.normal(size=24) * (0.01 if trial % 2 else 1.0)
    ridge, upper = (1e-6 if trial % 3 else 0.02), 0.2
    budget = [0.0, 0.001, 0.8, 1.137, 2.399, 12*upper][trial % 6]
    w, nu = mod._box_simplex(S, y, ridge, upper, budget)
    res = S @ w - y
    gradient = S.T @ res + ridge*w
    feasible = max(abs(w.sum()-budget), -w.min(), w.max()-upper)
    gap = gradient@w - nu*budget - upper*np.minimum(0, gradient-nu).sum()
    assert feasible < 1e-10 and abs(gap) < 1e-9, (trial, feasible, gap)
    worst = max(worst, abs(gap))
print('active-set stress: 120 trials; max absolute tangential gap', worst)
PY
```
- **Container Exit Code**: `0` (`619 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public reference gap 4.6165279605006604e-07 relative errors {'weights': 9.95131593318338e-07, 'terminal_field': 4.821877280926584e-07}
active-set stress: 120 trials; max absolute tangential gap 8.881784197001252e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29d4a7623d10e09c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_29d4a762 (directory).
```

### Turn 10 — `submit_completed_task` (`14163 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated Python validation, the public reference certificate, calibration routing tests, and active-set stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_01_solver.py': 'FILE:7566'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_01_solver.py`. The public schema and independent KKT checks passed. Relative errors were 9.95e-7 for weights and 4.82e-7 for the terminal field. Calibration fallback and 120 box-simplex stress cases also passed.`

### Turn 11 — `execute_in_remote_docker` (`30889 ms` LLM)
- **Rationale**: A nonzero constant-field residual would expose an unintended transport sink and warrant inspecting the stencil. Deleting one measured sweep row at a time tests whether a single calibration point can consume the KKT tolerance margin; any failure would identify the bank and row to inspect before changing the fit.
- **Alternatives Weighed**: `Add rational, Fokker–Planck, or batch-classification terms` (rejected: The task specifies a discrete transport operator and a continuous constrained inverse problem, not an unknown drift law or a classification objective. The boundary-mode branches were already tested.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import ast, importlib.util, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
nodes = ast.parse((root/'validate_public_schema.py').read_text()).body
ns = {}
exec(compile(ast.Module(body=[n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate'], type_ignores=[]), 'checker', 'exec'), ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
for case in cases:
    d, g = case['study'], case['gt']
    for axis in ('x','y'):
        k, v = d['kappa_'+axis], d['vel_'+axis]
        h = d['L'+axis]/(len(k)-1)
        a = m._axis(k,v,0.0,h)
        err = float(np.max(np.abs(a@np.ones(len(k)))))
        assert err < 1e-12, (axis,err)
        print('sealed constant-field residual',axis,err)
    sweep = {k: np.asarray(d['sensor_calibration_sweeps'][k]) for k in ('bank_id','ref_signal','raw_readout')}
    def certificate(st):
        o = m.solve_beta_m97_c02_batch_01(st)
        return verify(g['S_tilde'],g['r_y'],d['source_ridge'],d['u_box'],d['total_budget'],o['kkt_certificate'])
    base = certificate(d); assert base['valid'], base
    print('unaltered public reference gap',base['gap'])
    for bank in range(4):
        ids = np.flatnonzero(sweep['bank_id']==bank)
        if len(ids)<5: continue
        holdout, gaps, failures = [], [], []
        for i in ids:
            keep = np.arange(len(sweep['bank_id'])) != i
            train = ids[ids != i]
            x, y = sweep['ref_signal'], sweep['raw_readout']
            gain, offset = m._fit_line(x[train],y[train]); holdout.append(abs(gain*x[i]+offset-y[i]))
            c = certificate(dict(d,sensor_calibration_sweeps={k:v[keep] for k,v in sweep.items()}))
            if c['valid']: gaps.append(c['gap'])
            else: failures.append((int(i),c['reason']))
        print('bank',bank,'max held-out raw residual',max(holdout),'max passing gap',max(gaps,default=None),'failed deletion slices',failures)
    print('available case fields',','.join(sorted(d)))
PY
```
- **Container Exit Code**: `0` (`791 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
sealed constant-field residual x 3.552713678800501e-15
sealed constant-field residual y 3.552713678800501e-15
unaltered public reference gap 4.6165279605006604e-07
bank 0 max held-out raw residual 1.79995552240797e-07 max passing gap 5.652420620311238e-07 failed deletion slices []
bank 1 max held-out raw residual 2.4047314889363847e-07 max passing gap 4.993588188194342e-07 failed deletion slices []
bank 2 max held-out raw residual 2.1864664279114265e-07 max passing gap 4.995792540277155e-07 failed deletion slices []
bank 3 max held-out raw residual 3.234151649544881e-07 max passing gap 5.821714383312526e-07 failed deletion slices []
available case fields Lx,Ly,V_reaction_2d,Z_nuisance,aux_manifold_telemetry,critical_vortex_circulation,kappa_x,kappa_y,mu_shifts,nuisance_ridge,nx,ny,past_vortex_circulation_trace,reaction_x,reaction_y,recirculation_loop_mode,robin_x,robin_y,row_bank_id,sensor_calibration_sweeps,sensor_ij,source_modes,source_ridge,total_budget,u_box,vel_x,vel_y,y_obs_raw

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=29d4a7623d10e09c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_29d4a762 (directory).
```

### Turn 12 — `submit_completed_task` (`14759 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python validation, the independent public certificate check, and all measured calibration-deletion slices. The inactive fouling gate reproduced the archived open-Robin operator exactly, and the sealed stencil preserved constant fields on both axes to numerical precision.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_01_solver.py`. The public schema passed, and the KKT gap stayed below the 1e-6 tolerance for the full study and every leave-one-out calibration slice. Public relative errors were 9.95e-7 for the weights and 4.82e-7 for the terminal field.`

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
