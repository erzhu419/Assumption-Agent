# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_c03_s00__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `409,264` (`394,228` / `134,656` / `15,036`) |
| **Total Episode Wall Time (s)** | `445.76s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15108 ms` LLM)
- **Rationale**: Inspect the public arrays and apparatus metadata before choosing a calibration model or boundary convention; the schema defines the output contract.

```bash
cd /holdout_task && cat validate_public_schema.py public_asset_index.json && python3 - <<'PY'
import pickle
s=pickle.load(open('public_batch.pkl','rb'))
def describe(x,p='',d=0):
 if isinstance(x,dict) and d<3:
  for k,v in x.items(): describe(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x),repr(x)[:200]);
  if x and isinstance(x[0],dict) and d<3: describe(x[0],p+'/0',d+1)
 elif hasattr(x,'shape'): print(p,x.shape,x.dtype,repr(x.ravel()[:8]))
 else: print(p,repr(x)[:350])
describe(s)
PY
if [ -d /tmp/hegel_continual_archive ]; then ls -lah /tmp/hegel_continual_archive; f=/tmp/hegel_continual_archive/archive_apparatus_log.json; if [ -f "$f" ]; then head -c 12000 "$f"; fi; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'sensor calibration ridge nuisance profiling box simplex quadratic programming' 2>&1 | head -60
```
- **Container Exit Code**: `0` (`326 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`16423 ms` LLM)
- **Rationale**: The first dump was truncated. Inspect the calibration sweep layout, sensor indexing, and available lineage metadata in a smaller read; these determine which inverse calibration and boundary conventions can be tested against the tracer pulses.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb'));print('items',len(b))
for item in b:
 s=item['study'];print('item keys',list(item),'metadata', {k:v for k,v in item.items() if k!='study' and not hasattr(v,'shape')});print('study keys',list(s));print({k:v for k,v in s.items() if np.isscalar(v)});print('sensors',s['sensor_ij']); c=s['sensor_calibration_sweeps'];
 for k,v in c.items():
  if isinstance(v,np.ndarray): print('cal',k,v.shape,v[:3])
  else: print('cal',k,type(v).__name__,str(v)[:1700])
 print('pulses',len(s['reference_tracer_pulses'] or []))
PY
ls -la /pretrain_task /tmp/hegel_continual_archive 2>/dev/null; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 5500 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance 2>&1 | head -22
```
- **Container Exit Code**: `0` (`379 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sensors [[ 4  2]
 [ 6  4]
 [ 8  4]
 [ 4  8]
 [12  6]
 [ 0  0]
 [ 8  6]
 [12  4]
 [ 4  4]
 [ 2 10]
 [ 2  8]
 [ 2  2]
 [ 8  2]
 [10  4]
 [12  2]
 [ 6  6]
 [ 8 10]
 [10 10]
 [ 8  8]
 [ 6 10]
 [ 4  6]
 [ 4 10]
 [ 6  0]
 [ 0  8]
 [12  0]
 [10  2]
 [ 8  0]
 [ 4  0]
 [10  6]
 [ 0  4]
 [ 6  2]
 [12 10]]
cal ref_rosette (80, 4) [[-1.2  0.   0.   0. ]
 [-0.4  0.   0.   0. ]
 [ 0.5  0.   0.   0. ]]
cal raw_rosette (80, 4) [[-1.20000058e+00 -2.12508168e-06 -4.52987804e-06  5.96587503e-06]
 [-3.99988780e-01  1.20387058e-05 -1.48824716e-05  5.01547279e-06]
 [ 5.00010299e-01 -1.66464292e-05  3.75001870e-06  7.44700775e-07]]
cal single_bank_sweeps dict {'bank_id': array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
       3, 3]), 'ref_signal': array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,  0.9,
        1.4, -1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,
        0.9,  1.4]), 'raw_readout': array([-1.09999207, -0.59999364, -0.0999917 ,  0.40001239,  0.89998748,
        1.40000783, -1.09999509, -0.60000436, -0.10001536,  0.40000101,
        0.89999925,  1.39998038, -1.10000176, -0.60000887, -0.0999957 ,
        0.3999991 ,  0.89998691,  1.40001779, -1.10000329, -0.59998056,
       -0.10002232,  0.39999812,  0.90001902,  1.40000264])}
pulses 2
/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 12:27 ..
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
```

### Turn 3 — `execute_in_remote_docker` (`19209 ms` LLM)
- **Rationale**: Check whether the supplied velocities are cell- or face-indexed, whether the bank IDs match rosette positions, and whether boundary sensors carry signal. These observations determine which exact discrete operators to test against the tracer measurements.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
for k in ('kappa_x','kappa_y','vel_x','vel_y','mu_shifts','row_bank_id'):
 a=np.asarray(s[k]);print(k,a.shape,a)
for k in ('source_modes','V_reaction_2d','Z_nuisance','aux_manifold_telemetry','y_obs_raw'):
 a=s[k]; print(k,a.shape,'range',a.min(),a.max(),'mean',a.mean(axis=tuple(range(1,a.ndim)))[:8] if a.ndim>1 else a.mean())
for p in s['reference_tracer_pulses']:
 print('pulse',p['w_pulse'],'min/max per shift',np.min(p['y_pulse_raw'].reshape(3,-1),axis=1),np.max(p['y_pulse_raw'].reshape(3,-1),axis=1));print('corners',p['y_pulse_raw'].reshape(3,-1)[:,[5,22,23,24,27,29]])
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; print('cal noise',np.sqrt(np.mean((y-x)**2,axis=0)), 'pattern',np.unique((x!=0).sum(axis=1),return_counts=True)); print('mixed examples',x[16:25]);
PY
ls -ld /tmp/hegel_continual_archive /holdout_task/* 2>/dev/null || true
```
- **Container Exit Code**: `0` (`315 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_y (12,) [0.5173  0.51244 0.49825 0.47588 0.44715 0.41438 0.38022 0.34745 0.31872
 0.29635 0.28216 0.2773 ]
vel_x (14,) [ 0.35     0.345    0.33016  0.30589  0.27289  0.23209  0.18467  0.13198
  0.07552  0.01691 -0.04219 -0.10008 -0.15512 -0.20572]
vel_y (12,) [-0.      -0.05561 -0.109   -0.15804 -0.2008  -0.23555 -0.26092 -0.2759
 -0.27989 -0.27272 -0.2547  -0.22652]
mu_shifts (3,) [0.15 0.55 1.1 ]
row_bank_id (96,) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1
 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
source_modes (8, 14, 12) range 2.4237710201634425e-17 0.9945483521672139 mean [0.08795 0.08608 0.08736 0.08436 0.08889 0.08423 0.0886  0.0887 ]
V_reaction_2d (14, 12) range 0.25 0.42738476356834465 mean [0.25    0.25515 0.26944 0.28958 0.31096 0.32868 0.33869 0.33869]
Z_nuisance (96, 3) range -3.3583366597691273 3.781466319842277 mean [ 0.31132  0.66424 -0.6101  -0.26771 -0.19183 -0.55526 -0.11733 -0.95129]
aux_manifold_telemetry (24, 4) range -2.4463474053826744 2.725212705095335 mean [-0.57982  0.10581  0.48768  0.57956  0.57613 -0.12092  0.24966 -0.01023]
y_obs_raw (96,) range -0.1703420606024825 0.39309667787587643 mean 0.07949776184632647
pulse [1.  0.  0.  0.5 0.  0.  0.  0. ] min/max per shift [0.01021 0.00637 0.00365] [0.2149  0.18624 0.15903]
corners [[0.05308 0.12223 0.02351 0.03026 0.1188  0.07646]
 [0.04433 0.104   0.01666 0.02262 0.10131 0.06247]
 [0.03605 0.08665 0.01119 0.01615 0.08454 0.04976]]
pulse [0.  1.  0.  0.  0.5 0.  0.  0. ] min/max per shift [0.01459 0.00975 0.00613] [0.20879 0.18042 0.1536 ]
corners [[0.01597 0.08885 0.0179  0.07584 0.05491 0.03106]
 [0.01145 0.0733  0.01229 0.06336 0.04313 0.02252]
 [0.00775 0.05899 0.00799 0.05161 0.03272 0.01557]]
cal noise [0.00001 0.00001 0.00001 0.00001] pattern (array([1, 4]), array([16, 64]))
mixed examples [[ 1.28868  0.81568 -1.23308  1.41015]
 [-1.05445  1.00838 -1.35885  1.45232]
 [ 1.42374 -1.10201  0.2692   1.41665]
 [ 1.34287 -0.34599 -0.10586  0.24647]
 [ 1.78133 -0.63069 -0.67244  0.23758]
 [-0.43045  0.50944 -1.2349  -1.03464]
 [-0.82837  0.84526  0.67909  1.30301]
 [ 1.15258  0.59972 -0.24039 -1.1431 ]
 [ 0.58681 -0.67269 -0.28329 -1.11576]]
-rw-r--r-- 1 root root  3667 Nov 27 12:27 /holdout_task/public_asset_index.json
-rw-r--r-- 1 root root 26378 Nov 27 12:27 /holdout_task/public_batch.pkl
-rw-r--r-- 1 root root  1064 Nov 27 12:27 /holdout_task/validate_public_schema.py
```

### Turn 4 — `execute_in_remote_docker` (`39286 ms` LLM)
- **Rationale**: Compare exact grid, interface, and Robin conventions against both measured tracer responses. Profiling the four wall coefficients separately will show whether any residual comes from the wall closure or the interior stencil before either is committed to the solver.
- **Alternatives Weighed**: `Fit a generic sensor-domain correction` (rejected: A sensor-space fit would not identify the transport operator needed to reconstruct the terminal field.)

```bash
cat > /tmp/transport_probe.py <<'PY'
import itertools,pickle
import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
nx,ny=s['nx'],s['ny']; coords=s['sensor_ij']; ids=coords[:,0]+nx*coords[:,1]; shifts=s['mu_shifts']; pulses=s['reference_tracer_pulses']; M=len(s['source_modes']); f=s['source_modes'].reshape(M,-1,order='F').T@np.stack([p['w_pulse'] for p in pulses],axis=1); y=np.stack([p['y_pulse_raw'] for p in pulses],axis=1); rx,ry=s['robin_x'],s['robin_y']

def axis(k,v,h,km,vm,form):
    n=len(k); j=np.arange(n-1); A=np.zeros((n,n)); d={'ar':(k[:-1]+k[1:])*.5,'ha':2*k[:-1]*k[1:]/(k[:-1]+k[1:]),'geo':np.sqrt(k[:-1]*k[1:])}[km]/h**2
    a={'face':(v[:-1]+v[1:])*.5,'left':v[:-1],'right':v[1:]}[vm]/h
    A[j,j]+=d; A[j+1,j+1]+=d; A[j,j+1]-=d; A[j+1,j]-=d
    if form=='flux':
        A[j,j]+=np.maximum(a,0); A[j+1,j]-=np.maximum(a,0); A[j,j+1]+=np.minimum(a,0); A[j+1,j+1]-=np.minimum(a,0)
    elif form=='velocity':
        p=np.maximum(v[1:]/h,0); q=np.maximum(-v[:-1]/h,0); A[j,j]+=q; A[j,j+1]-=q; A[j+1,j+1]+=p; A[j+1,j]-=p
    elif form=='facevelocity':
        p=np.maximum(a,0); q=np.maximum(-a,0); A[j,j]+=q; A[j,j+1]-=q; A[j+1,j+1]+=p; A[j+1,j]-=p
    elif form=='none': pass
    return A

def init(delt,km,vm,form):
    hx=s['Lx']/(nx-delt);hy=s['Ly']/(ny-delt)
    ax=axis(s['kappa_x'],s['vel_x'],hx,km,vm,form);ay=axis(s['kappa_y'],s['vel_y'],hy,km,vm,form)
    B=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    xx=np.tile(np.arange(nx),ny); yy=np.repeat(np.arange(ny),nx)
    return B,np.stack([xx==0,xx==nx-1,yy==0,yy==ny-1],axis=1).astype(float),hx,hy

def predict(B,d):
    return np.concatenate([solve(B+np.diag(d+mu),f,check_finite=False)[ids] for mu in shifts])

def boundary(hx,hy):
    h=np.repeat([hx,hy],2); r=np.repeat([rx,ry],2); k=np.array([s['kappa_x'][0],s['kappa_x'][-1],s['kappa_y'][0],s['kappa_y'][-1]]); v=np.array([-s['vel_x'][0],s['vel_x'][-1],-s['vel_y'][0],s['vel_y'][-1]])/h
    for name,c in {'r':r,'r/h':r/h,'kr/h':k*r/h,'r/k':r/k,'r/(h+r*h*h/2k)':r/(h+r*h*h/(2*k)),'kr/(h+rh*h/2)':k*r/(h+r*h*h/2),'2r/h':2*r/h,'r/h2':r/h**2,'kr/h2':k*r/h**2,'robin-extrap':k/(h*(1/r+h/2))}.items():
        for an,ad in [('zero',0),('out',np.maximum(v,0)),('signed',v),('in',np.maximum(-v,0))]:
            yield name,an,c+ad

if __name__=='__main__':
    ranks=[];free=[]
    for config in itertools.product([0,1,-1],['ar','ha','geo'],['face','left','right'],['flux','velocity','facevelocity','none']):
        B,E,hx,hy=init(*config)
        fit=least_squares(lambda c:(predict(B,E@c)-y).ravel(),np.ones(4)*3,bounds=(-.25,40),gtol=1e-9)
        free.append((np.sqrt(np.mean(fit.fun**2)),config,fit.x))
        for bn,an,c in boundary(hx,hy):
            e=predict(B,E@c)-y
            ranks.append((np.sqrt(np.mean(e**2)),config,bn,an,c))
    print('BEST exact'); print(*sorted(ranks,key=lambda z:z[0])[:12],sep='\n'); print('BEST profiled'); print(*sorted(free,key=lambda z:z[0])[:18],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_probe.py
```
- **Container Exit Code**: `0` (`7695 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
80436, 6.15702798, 6.15702798]))
(0.0003528696088492302, (1, 'geo', 'left', 'velocity'), 'r/h', 'out', array([5.04380436, 5.04380436, 6.15702798, 6.15702798]))
(0.0003528696088492302, (1, 'geo', 'right', 'velocity'), 'r/h', 'zero', array([5.04380436, 5.04380436, 6.15702798, 6.15702798]))
(0.0003528696088492302, (1, 'geo', 'right', 'velocity'), 'r/h', 'out', array([5.04380436, 5.04380436, 6.15702798, 6.15702798]))
BEST profiled
(2.250693195994971e-05, (1, 'ar', 'face', 'velocity'), array([5.23603755, 5.25970547, 6.1290073 , 6.1975699 ]))
(2.250693195994971e-05, (1, 'ar', 'left', 'velocity'), array([5.23603755, 5.25970547, 6.1290073 , 6.1975699 ]))
(2.250693195994971e-05, (1, 'ar', 'right', 'velocity'), array([5.23603755, 5.25970547, 6.1290073 , 6.1975699 ]))
(2.5367042687132495e-05, (1, 'geo', 'face', 'velocity'), array([5.23597877, 5.26182702, 6.13138313, 6.17802586]))
(2.5367042687132495e-05, (1, 'geo', 'left', 'velocity'), array([5.23597877, 5.26182702, 6.13138313, 6.17802586]))
(2.5367042687132495e-05, (1, 'geo', 'right', 'velocity'), array([5.23597877, 5.26182702, 6.13138313, 6.17802586]))
(3.179549958941112e-05, (1, 'ha', 'face', 'velocity'), array([5.23591934, 5.26394921, 6.13375946, 6.15854271]))
(3.179549958941112e-05, (1, 'ha', 'left', 'velocity'), array([5.23591934, 5.26394921, 6.13375946, 6.15854271]))
(3.179549958941112e-05, (1, 'ha', 'right', 'velocity'), array([5.23591934, 5.26394921, 6.13375946, 6.15854271]))
(0.0003545108633815396, (1, 'ha', 'face', 'facevelocity'), array([5.14245553, 4.90688975, 6.17400974, 6.02714496]))
(0.00036031553581156666, (1, 'geo', 'face', 'facevelocity'), array([5.1425281 , 4.9050516 , 6.17164009, 6.04628357]))
(0.00036633835485501404, (1, 'ar', 'face', 'facevelocity'), array([5.14259993, 4.90321379, 6.16927095, 6.06548154]))
(0.0005347255937055499, (1, 'ha', 'right', 'facevelocity'), array([5.09447104, 4.60196632, 6.24814295, 6.11579247]))
(0.0005405021131016906, (1, 'geo', 'right', 'facevelocity'), array([5.09457336, 4.60034282, 6.24576156, 6.1352318 ]))
(0.0005464236627542074, (1, 'ar', 'right', 'facevelocity'), array([5.09467508, 4.59871957, 6.24338061, 6.15473168]))
(0.0006583455648567372, (1, 'ha', 'left', 'facevelocity'), array([5.19044793, 5.23770076, 6.09743957, 5.93164979]))
(0.0006598708749308202, (1, 'geo', 'left', 'facevelocity'), array([5.19049119, 5.23561867, 6.09508318, 5.95046938]))
(0.0006615658981292267, (1, 'ar', 'left', 'facevelocity'), array([5.19053381, 5.23353708, 6.09272729, 5.96934689]))
```

### Turn 5 — `execute_in_remote_docker` (`28734 ms` LLM)
- **Rationale**: Compare nominal, axis-shared, and independent wall corrections on held-out pulses and shifts. Cross-validate the sweep calibration separately so a noise-level public sweep does not force an unnecessary nonlinear inverse.
- **Alternatives Weighed**: `Commit four wall coefficients based on pooled residual` (rejected: Holding out an entire tracer pulse tests whether the fitted corrections transfer to a different source configuration.)

```bash
cat > /tmp/transport_validate.py <<'PY'
from transport_probe import *
from scipy.optimize import least_squares
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; n=len(x)
lib=[np.ones((n,1)),x]
for degree in (2,3):
    lib.append(np.column_stack([np.prod(x[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),degree)]))
fold=np.tile(np.arange(4),n//4)
for a in (0,2,3,4):
    P=np.column_stack(lib[:a]) if a else None
    e=raw-x
    if a:
        for k in range(4):
            tr=fold!=k
            e[~tr]-=P[~tr]@np.linalg.lstsq(P[tr],e[tr],rcond=None)[0]
    print('cal degree',a-1,'rms',np.sqrt(np.mean(e**2)),'worst bank',np.sqrt(np.mean(e**2,axis=0)).max(),'single / mixed',*[np.sqrt(np.mean(v**2)) for v in (e[:16],e[16:])])
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
rope=[('nominal',np.zeros((4,0))),('axis scale',np.repeat(np.eye(2),2,axis=0)),('independent walls',np.eye(4))]
for name,W in rope:
    fitlist=[]; held=[]
    for train,test in [(0,1),(1,0)]:
        a=least_squares(lambda a:(predict(B,E@(d0+W@a))[:,train]-y[:,train]),np.zeros(W.shape[1]),gtol=1e-12).x if W.shape[1] else np.empty(0)
        dd=d0+W@a; fitlist.append(dd)
        e=(predict(B,E@dd)-y).reshape(3,-1,2)[:,:,test]
        held.append(list(np.sqrt(np.mean(e**2,axis=1))))
    print(name,'wall coefficients',np.array(fitlist),'held pulse-shift RMSE',np.array(held),'worst',np.max(held))
for k in ('robin_x','robin_y'):
 print(k,s[k])
print('source design',np.linalg.svd(np.concatenate([solve(B+np.diag(E@d0+mu),s['source_modes'].reshape(M,-1,order='F').T,check_finite=False)[ids] for mu in shifts]),compute_uv=False))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_validate.py
```
- **Container Exit Code**: `0` (`497 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cal degree -1 rms 1.0069909406158925e-05 worst bank 1.0485883781472711e-05 single / mixed 1.0932136750392963e-05 9.842557636508253e-06
cal degree 1 rms 1.0584633207264648e-05 worst bank 1.1127074084540948e-05 single / mixed 1.1118019628429866e-05 1.0447032356170528e-05
cal degree 2 rms 1.4068141619891952e-05 worst bank 1.5110399009610503e-05 single / mixed 1.1874438659720764e-05 1.4565033038953342e-05
cal degree 3 rms 2.9532277033855438e-05 worst bank 3.199859234212173e-05 single / mixed 1.813879658443664e-05 3.1748074715210515e-05
nominal wall coefficients [[5.04380436 5.04380436 6.15702798 6.15702798]
 [5.04380436 5.04380436 6.15702798 6.15702798]] held pulse-shift RMSE [[0.00047425 0.00032388 0.00021779]
 [0.00045823 0.00031762 0.00021978]] worst 0.0004742469238022118
axis scale wall coefficients [[5.24079038 5.24079038 6.12917831 6.12917831]
 [5.26176565 5.26176565 6.13011927 6.13011927]] held pulse-shift RMSE [[5.34812966e-05 4.79214978e-05 4.31557037e-05]
 [7.44170711e-05 4.42512303e-05 2.48635994e-05]] worst 7.441707106178088e-05
independent walls wall coefficients [[5.23343259 5.26641069 6.12788764 6.18567545]
 [5.2597309  5.25688235 6.12981812 6.19048974]] held pulse-shift RMSE [[3.71069876e-05 2.59771297e-05 2.55448446e-05]
 [7.30378275e-05 4.17716762e-05 2.30352689e-05]] worst 7.303782746550941e-05
robin_x 0.9311638816009302
robin_y 1.1194596332223248
source design [1.53395982 0.64659362 0.5136978  0.27362689 0.19085785 0.13525667
 0.10249094 0.05629228]
```

### Turn 6 — `execute_in_remote_docker` (`26830 ms` LLM)
- **Rationale**: The held-shift check will detect corrections that fit one range of spectral shifts but fail on another; if four-wall identification is unstable, use the axis-shared closure. The asset index may also document the apparatus metadata available on blind campaigns.
- **Alternatives Weighed**: `Use four free walls solely because they fit both pulses better` (rejected: The held-pulse advantage over two axis corrections is small; held-shift errors and Jacobian conditioning can reveal whether the extra wall terms are identified.)
- **ATMS NOGOOD DECLARED**: Refuted `The two nominal Robin values fully specify the active campaign's wall diagonals.` | Witness: `Correcting only the wall diagonals reduces worst held-out pulse-shift RMSE from 4.74e-4 to 7.30e-5; public rosette sweeps deviate from identity by about 1.01e-5 RMS.` | Blanket Diagnosis: `The nominal wall values leave a campaign-dependent mismatch even when the interior stencil and sensor calibration are held fixed.` | Sublated Pivot: `Keep the arithmetic-face diffusion and cell-centered upwind stencil; fit wall corrections from tracer measurements and use nominal values as the baseline when diagnostic evidence is absent.`

```bash
cat /holdout_task/public_asset_index.json; cat /holdout_task/validate_public_schema.py; if [ -d /tmp/hegel_continual_archive ]; then ls -lah /tmp/hegel_continual_archive; fi; cat > /tmp/transport_slices.py <<'PY'
from transport_probe import *
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
ns=len(ids); group=np.repeat(np.arange(len(shifts)),ns)
for name,W in [('axis',np.repeat(np.eye(2),2,axis=0)),('walls',np.eye(4))]:
    for k in range(len(shifts)):
        use=group!=k
        fit=least_squares(lambda a:(predict(B,E@(d0+W@a))-y)[use].ravel(),np.zeros(W.shape[1]),gtol=1e-12)
        errors=(predict(B,E@(d0+W@fit.x))-y)[~use]
        print(name,'held shift',k,'RMSE per pulse',np.sqrt(np.mean(errors**2,axis=0)))
    fit=least_squares(lambda a:(predict(B,E@(d0+W@a))-y).ravel(),np.zeros(W.shape[1]),gtol=1e-12)
    print(name,'relative coefficients',(d0+W@fit.x)/d0,'Jacobian singular values',np.linalg.svd(fit.jac,compute_uv=False))
    res=fit.fun.reshape(len(shifts),ns,2)
    print('largest residual sensors',[(coords[j].tolist(), round(float(e),8)) for j,e in sorted(enumerate(np.sqrt(np.mean(res**2,axis=(0,2)))),key=lambda v:-v[1])[:8]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_slices.py
```
- **Container Exit Code**: `0` (`544 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                96
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
      "bytes": 1064
    }
  },
  "schema": "hegel_public_asset_index_v1"
}import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
M, nx, ny = np.asarray(st["source_modes"]).shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out or not isinstance(out["kkt_certificate"], dict): errs.append("missing:kkt_certificate")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
axis held shift 0 RMSE per pulse [6.11679126e-05 3.36334201e-05]
axis held shift 1 RMSE per pulse [2.94694772e-05 3.55569385e-05]
axis held shift 2 RMSE per pulse [1.92175658e-05 3.51118966e-05]
axis relative coefficients [1.04094279 1.04094279 0.99556961 0.99556961] Jacobian singular values [0.04656917 0.01884632]
largest residual sensors [([0, 4], 7.721e-05), ([12, 2], 6.189e-05), ([6, 10], 5.662e-05), ([8, 10], 5.609e-05), ([0, 8], 5.269e-05), ([4, 10], 4.668e-05), ([0, 0], 4.504e-05), ([12, 4], 4.397e-05)]
walls held shift 0 RMSE per pulse [3.69362308e-05 4.16277757e-05]
walls held shift 1 RMSE per pulse [1.38129132e-05 2.55469613e-05]
walls held shift 2 RMSE per pulse [1.77684514e-05 2.91603807e-05]
walls relative coefficients [1.03811274 1.04280521 0.99544899 1.00658466] Jacobian singular values [0.04365669 0.01638204 0.01390012 0.0037321 ]
largest residual sensors [([0, 4], 6.059e-05), ([12, 6], 4.081e-05), ([12, 2], 4.07e-05), ([0, 8], 3.945e-05), ([12, 4], 3.262e-05), ([12, 0], 2.971e-05), ([0, 0], 2.796e-05), ([2, 2], 2.694e-05)]
```

### Turn 7 — `execute_in_remote_docker` (`65679 ms` LLM)
- **Rationale**: Implement the measured stencil, sweep-selected rosette inverse, and certified box-simplex solve. The public validator will detect output schema and import failures; numerical certificate and blind-branch checks should follow.
- **Alternatives Weighed**: `Use axis-shared tracer corrections` (rejected: Independent wall coefficients improve worst held-shift RMSE from 6.12e-5 to 4.16e-5 and also improve the worst held-pulse slice.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
import numpy as np
from itertools import combinations_with_replacement
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu


def _features(x, terms):
    return np.column_stack([np.prod(x[:, t], axis=1) for t in terms])


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], float)
    y = np.asarray(sweeps['raw_rosette'], float)
    terms = [()] + [t for degree in (1, 2, 3)
                    for t in combinations_with_replacement(range(4), degree)]
    features = _features(x, terms)
    delta = y - x
    folds = np.arange(len(x)) % 4
    scores = [np.mean(delta**2)]
    sizes = (0, 1, 5, 15, 35)
    for size in sizes[1:]:
        err = np.empty_like(delta)
        for fold in range(4):
            test = folds == fold
            p = features[:, :size]
            coef = np.linalg.lstsq(p[~test], delta[~test], rcond=None)[0]
            err[test] = delta[test] - p[test] @ coef
        scores.append(np.mean(err**2))
    size = sizes[int(np.argmin(scores))]
    terms = terms[:size]
    coef = np.linalg.lstsq(features[:, :size], delta, rcond=None)[0]
    deriv = [[(i, t.count(j), tuple(k for l, k in enumerate(t) if l != t.index(j)))
              for i, t in enumerate(terms) if j in t] for j in range(4)]

    def inverse(raw):
        raw = np.asarray(raw, float)
        if not size:
            return raw.copy()
        target = raw.reshape(-1, 4)
        z = target.copy()
        for _ in range(30):
            err = z + _features(z, terms) @ coef - target
            if np.max(np.abs(err)) < 1e-12:
                break
            J = np.broadcast_to(np.eye(4), (len(z), 4, 4)).copy()
            for j, row in enumerate(deriv):
                for i, count, t in row:
                    J[:, :, j] += (count * np.prod(z[:, t], axis=1))[:, None] * coef[i]
            step = np.linalg.solve(J, err[..., None])[..., 0]
            scale = 1.
            while scale > 1/1024:
                trial = z - scale * step
                trial_err = trial + _features(trial, terms) @ coef - target
                if np.linalg.norm(trial_err) < np.linalg.norm(err):
                    z = trial
                    break
                scale *= .5
            else:
                break
        return z.reshape(raw.shape)
    return inverse


def _axis(k, v, h):
    d = (k[:-1] + k[1:]) / (2*h*h)
    upper = d + np.maximum(-v[:-1], 0) / h
    lower = d + np.maximum(v[1:], 0) / h
    main = np.r_[upper, 0.] + np.r_[0., lower]
    return sparse.diags((-lower, main, -upper), (-1, 0, 1), format='csc')


def _transport(study, calibrate):
    nx, ny = study['nx'], study['ny']
    hx, hy = study['Lx']/(nx-1), study['Ly']/(ny-1)
    ax = _axis(np.asarray(study['kappa_x']), np.asarray(study['vel_x']), hx)
    ay = _axis(np.asarray(study['kappa_y']), np.asarray(study['vel_y']), hy)
    base = sparse.kronsum(ax, ay, format='csc')
    V = np.asarray(study['V_reaction_2d']).ravel(order='F')
    xx = np.tile(np.arange(nx), ny)
    yy = np.repeat(np.arange(ny), nx)
    walls = np.column_stack((xx == 0, xx == nx-1, yy == 0, yy == ny-1)).astype(float)
    d0 = np.repeat([study['robin_x']/hx, study['robin_y']/hy], 2)
    i, j = np.asarray(study['sensor_ij']).T
    ids = i + nx*j
    src = np.asarray(study['source_modes'], float).reshape(-1, nx*ny, order='F').T
    shifts = study['mu_shifts']

    def fields(d, rhs):
        diagonal = V + walls @ d
        for mu in shifts:
            yield splu(base + sparse.diags(diagonal + mu, format='csc')).solve(rhs)

    d = d0
    pulses = study['reference_tracer_pulses']
    if pulses:
        target = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
        forcing = src @ np.column_stack([p['w_pulse'] for p in pulses])

        def residual(d):
            return (np.concatenate([u[ids] for u in fields(d, forcing)]) - target).ravel()

        d = least_squares(residual, d0, bounds=(0, np.inf), gtol=1e-12).x
    observed = []
    for u in fields(d, src):
        observed.append(u[ids])
    return np.concatenate(observed), u.T.reshape(-1, nx, ny, order='F')


def _qp(S, y, ridge, upper, budget):
    n = S.shape[1]
    u = np.broadcast_to(np.asarray(upper, float), (n,))
    H = S.T @ S + ridge * np.eye(n)
    b = S.T @ y
    w = np.zeros(n)
    if budget >= u.sum():
        w = u.copy()
    elif budget > 0:
        w = u * (budget / u.sum())
        active = np.where(u == 0, -1, 0)
        for _ in range(20*n+100):
            free = active == 0
            g = H @ w - b
            if np.any(free):
                inv = solve(H[np.ix_(free, free)], np.column_stack((-g[free], np.ones(free.sum()))), assume_a='pos')
                nu = inv[:, 0].sum() / inv[:, 1].sum()
                p = inv[:, 0] - nu * inv[:, 1]
                if np.max(np.abs(p)) > 1e-12:
                    f = np.flatnonzero(free)
                    steps = np.full(len(f), np.inf)
                    neg, pos = p < -1e-14, p > 1e-14
                    steps[neg] = -w[f[neg]] / p[neg]
                    steps[pos] = (u[f[pos]] - w[f[pos]]) / p[pos]
                    j = int(np.argmin(steps))
                    if steps[j] < 1:
                        w[f] += max(0., steps[j]) * p
                        active[f[j]] = 1 if pos[j] else -1
                        w[f[j]] = u[f[j]] if pos[j] else 0.
                        continue
                    w[f] += p
                    g = H @ w - b
            else:
                lower, high = active < 0, active > 0
                nu = (-g[lower & (u > 0)].min() - g[high].max()) / 2
            violation = np.where(active > 0, g + nu, np.where((active < 0) & (u > 0), -g-nu, -np.inf))
            j = int(np.argmax(violation))
            if violation[j] <= 1e-11:
                break
            active[j] = 0
        else:
            raise RuntimeError('Box-simplex active set did not converge')
    g = H @ w - b
    lower, high = w <= 1e-11, w >= u - 1e-11
    free = ~(lower | high)
    if free.any():
        nu = -g[free].mean()
    else:
        lo = np.max(-g[lower & ~high], initial=-np.inf)
        hi = np.min(-g[high & ~lower], initial=np.inf)
        nu = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else lo if np.isfinite(lo) else hi if np.isfinite(hi) else 0.
    lam_lower = np.where(lower, np.maximum(g+nu, 0), 0.)
    lam_upper = np.where(high, np.maximum(-g-nu, 0), 0.)
    station = g + nu - lam_lower + lam_upper
    primal = .5 * (np.linalg.norm(S @ w - y)**2 + ridge * (w @ w))
    lagrangian = primal + nu*(w.sum()-budget) - lam_lower@w + lam_upper@(w-u)
    bound = lagrangian - .5 * station @ solve(H, station, assume_a='pos')
    return dict(weights=w, nu_sum=float(nu), lam_lower=lam_lower, lam_upper=lam_upper,
                primal_objective=float(primal), dual_lower_bound=float(bound))


def solve_batch(study: dict) -> dict:
    calibrate = _calibration(study['sensor_calibration_sweeps'])
    S, U_last = _transport(study, calibrate)
    Z = np.asarray(study['Z_nuisance'], float)
    y = calibrate(study['y_obs_raw'])
    T = np.column_stack((S, y))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    T -= Z @ solve(gram, Z.T @ T, assume_a='pos')
    cert = _qp(T[:, :-1], T[:, -1], study['source_ridge'], study['u_box'], study['total_budget'])
    w = cert['weights']
    return dict(weights=w, terminal_field=np.tensordot(w, U_last, axes=(0, 0)), kkt_certificate=cert)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 8 — `execute_in_remote_docker` (`32428 ms` LLM)
- **Rationale**: Check the reported objective and KKT conditions independently, including feasible budgets at box vertices where active-set errors can occur. Exercise the no-pulse branch and inspect the small public telemetry and any available apparatus log before deciding whether a blind wall correction is supported.
- **Alternatives Weighed**: `Treat the schema pass as numerical verification` (rejected: The schema does not check KKT residuals, field contraction, or execution without tracer pulses.)

```bash
cat > /tmp/verify_batch.py <<'PY'
import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
maxima=np.zeros(4)
def verify(S,y,r,u,b,c):
    global maxima
    w=c['weights']; l=c['lam_lower']; h=c['lam_upper']; nu=c['nu_sum']; g=S.T@(S@w-y)+r*w
    p=.5*(np.linalg.norm(S@w-y)**2+r*(w@w))
    assert np.all(l>=0) and np.all(h>=0) and abs(p-c['primal_objective'])<1e-9
    gap=p-c['dual_lower_bound']; assert gap>=-1e-9
    err=np.array([max(abs(w.sum()-b),float(np.max(-w)),float(np.max(w-u))), np.max(abs(g+nu-l+h)), max(np.max(abs(l*w)),np.max(abs(h*(w-u)))),abs(gap)])
    maxima=np.maximum(maxima,err)
    assert np.max(err)<1e-8, err
for pulses in (s['reference_tracer_pulses'], None):
    a=dict(s,reference_tracer_pulses=pulses); out=m.solve_batch(a)
    cal=m._calibration(a['sensor_calibration_sweeps']); S,U=m._transport(a,cal); y=cal(a['y_obs_raw']); Z=a['Z_nuisance']
    P=np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+a['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    verify(P@S,P@y,a['source_ridge'],a['u_box'],a['total_budget'],out['kkt_certificate'])
    np.testing.assert_allclose(out['weights'],out['kkt_certificate']['weights']); np.testing.assert_allclose(out['terminal_field'],out['weights']@U.reshape(len(U),-1).copy() if False else np.einsum('i,ixy->xy',out['weights'],U))
    print('mode', 'tracer' if pulses else 'blind', 'weights',np.round(out['weights'],7),'field range',out['terminal_field'].min(),out['terminal_field'].max())
rng=np.random.default_rng(17)
for _ in range(320):
    n=int(rng.integers(2,30)); u=rng.choice([0.,.2,.4,1.],n); S=rng.normal(size=(int(rng.integers(2,70)),n)); y=rng.normal(size=len(S)); r=10.**rng.uniform(-5,1)
    for b in (0.,u.sum(),u[:n//2].sum(),u.sum()*rng.uniform()):
        verify(S,y,r,u,b,m._qp(S,y,r,u,b))
print('worst feasibility / stationarity / complementarity / dual gap',maxima)
np.set_printoptions(precision=6,suppress=True)
print('aux telemetry (public only):\n',s['aux_manifold_telemetry']); print('raw rosette first rows:\n',s['y_obs_raw'].reshape(-1,4)[:4]); print('single-bank sweeps', {k:(np.asarray(v).shape,np.asarray(v).ravel()[:5]) for k,v in s['sensor_calibration_sweeps']['single_bank_sweeps'].items()})
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py
if [ -d /tmp/hegel_continual_archive ]; then ls -la /tmp/hegel_continual_archive; f=/tmp/hegel_continual_archive/archive_apparatus_log.json; if [ -f "$f" ]; then head -c 5500 "$f"; fi; else echo 'No apparatus archive mounted'; fi
```
- **Container Exit Code**: `0` (`929 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mode tracer weights [0.1346936 0.0952036 0.0773929 0.1039312 0.2042335 0.2085005 0.3589088
 0.217136 ] field range 0.004003296072036581 0.12260886255113727
mode blind weights [0.1349894 0.0975525 0.0752153 0.1055908 0.2029041 0.2095078 0.3590728
 0.2151672] field range 0.004095872810697297 0.12251192799068364
worst feasibility / stationarity / complementarity / dual gap [3.55271368e-15 5.68434189e-14 5.83902008e-17 1.13686838e-13]
aux telemetry (public only):
 [[-0.372782 -1.310024 -0.11026  -0.526211]
 [ 0.249485  0.319008 -1.004993  0.859745]
 [-0.318439  0.636866  0.543659  1.088639]
 [ 0.671071  1.234005  1.067794 -0.654629]
 [ 2.071094 -0.075506  0.379859 -0.070941]
 [ 0.131892  0.190632 -2.446347  1.640125]
 [ 0.092512 -0.649847  1.419192  0.136789]
 [ 0.122286 -1.219077  2.137495 -1.081636]
 [-2.188567  1.642671  0.618896 -0.825362]
 [-0.922469 -0.31315   0.638967  1.345774]
 [ 1.209491 -0.256597  1.46231  -0.48426 ]
 [ 0.677404 -0.183745 -0.691769  0.986769]
 [ 0.237818  0.752808  0.856403  0.302794]
 [-1.868696  0.364245  0.633042 -1.443353]
 [ 0.550339 -0.402233 -1.314976 -1.354106]
 [-0.599713 -0.829302 -1.091622 -0.564529]
 [ 0.906072  0.333799  0.219814 -0.929196]
 [-0.654857 -0.734987 -0.639025 -1.591541]
 [ 0.907659 -0.194235  0.101666  0.756835]
 [ 0.231591 -1.419205  1.253719  2.077682]
 [-0.16039   0.30125   1.150348  1.224609]
 [ 0.36048  -0.355227  2.725213  0.139827]
 [ 0.773867 -0.146844  0.705915  1.870586]
 [ 0.85838  -0.240611 -0.363336 -0.477195]]
raw rosette first rows:
 [[-0.021351  0.034918  0.277457 -0.002967]
 [ 0.035942  0.1001    0.235265  0.393097]
 [ 0.033381 -0.062614  0.085227 -0.020487]
 [ 0.131422  0.310197  0.140527  0.10644 ]]
single-bank sweeps {'bank_id': ((24,), array([0, 0, 0, 0, 0])), 'ref_signal': ((24,), array([-1.1, -0.6, -0.1,  0.4,  0.9])), 'raw_readout': ((24,), array([-1.099992, -0.599994, -0.099992,  0.400012,  0.899987]))}
No apparatus archive mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 9 — `execute_in_remote_docker` (`24921 ms` LLM)
- **Rationale**: Temporarily withhold tracer data and test whether routine observations identify wall corrections after profiling source weights and nuisance. Compare held-out rosettes and use the tracer fit only as an independent reference; if the inferred corrections are unstable, keep the nominal fallback.
- **Alternatives Weighed**: `Infer a blind correction from the public telemetry alone` (rejected: There is one campaign and no mounted apparatus log, so there is no independent evidence linking telemetry to the wall parameters.)

```bash
cat > /tmp/blind_probe.py <<'PY'
from transport_probe import *
from scipy.optimize import minimize
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
F=s['source_modes'].reshape(M,-1,order='F').T
Z=s['Z_nuisance']; raw=s['y_obs_raw']; A=np.column_stack((F, F@np.column_stack([p['w_pulse'] for p in pulses]))); ns=len(ids)

def design(d):
    return np.concatenate([solve(B+np.diag(E@d+mu),A,check_finite=False)[ids] for mu in shifts])

dref=least_squares(lambda d:(design(d)[:,M:]-y).ravel(),d0,gtol=1e-12).x
print('tracer fitted change:',dref-d0)
# Eliminate the budget before profiling sources and measurement nuisance.
from scipy.linalg import null_space
Q=null_space(np.ones((1,M))); w0=np.full(M,s['total_budget']/M)
def profile(d,train):
    S=design(d)[:,:M]; X=np.column_stack((S@Q,Z)); z=raw-S@w0
    beta=np.linalg.lstsq(X[train],z[train],rcond=None)[0]
    return X@beta-z, w0+Q@beta[:M-1]
for label,W in [('nominal',np.empty((4,0))),('axis',np.repeat(np.eye(2),2,axis=0)),('walls',np.eye(4)),('tracer',np.empty((4,0)))]:
    for fold in [-1,0,1,2,3]:
        train=np.ones(len(raw),bool) if fold<0 else np.arange(len(raw))//4%4!=fold
        d=(dref if label=='tracer' else d0).copy()
        if W.shape[1]:
            fit=least_squares(lambda a:profile(d0+W@a,train)[0][train],np.zeros(W.shape[1]),bounds=(-1.5,1.5),gtol=1e-12)
            d=d0+W@fit.x
        e,w=profile(d,train)
        print(label,fold,'d',np.round(d-d0,4),'RMSE train/test',np.sqrt(np.mean(e[train]**2)),np.sqrt(np.mean(e[~train]**2)) if fold>=0 else None,'w range',w.min(),w.max())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/blind_probe.py
```
- **Container Exit Code**: `0` (`765 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3870332 w range 0.06996986865767985 0.5639193384699136
axis -1 d [ 1.5     1.5    -1.3442 -1.3442] RMSE train/test 0.00774597570610177 None w range 0.022821863269907233 0.5101421703839448
axis 0 d [ 1.5     1.5    -1.2776 -1.2776] RMSE train/test 0.007921584977385387 0.00786291927848021 w range 0.0776504325958791 0.46374921470989033
axis 1 d [ 1.5    1.5   -1.336 -1.336] RMSE train/test 0.007542717016465054 0.008554996591436199 w range 0.021473316125228986 0.5287681095743169
axis 2 d [ 0.9729  0.9729 -0.713  -0.713 ] RMSE train/test 0.007415889061506389 0.009539765880809748 w range -0.06926592098244722 0.5552078042063717
axis 3 d [ 1.0727  1.0727 -1.2739 -1.2739] RMSE train/test 0.007610282303178156 0.00890506552753877 w range 0.045784031034199674 0.513810710617068
walls -1 d [ 1.5     1.1626 -1.5    -0.7939] RMSE train/test 0.007732796289705586 None w range 0.014322328986775407 0.5294018156171252
walls 0 d [ 1.5     0.6168 -1.5     0.3009] RMSE train/test 0.00787334761920623 0.008277174079534431 w range 0.07331427243111593 0.5211222191998957
walls 1 d [ 1.5     1.3093 -1.5    -0.7973] RMSE train/test 0.007522004923321852 0.008557933203540317 w range 0.0034115258409505855 0.5470607116151507
walls 2 d [ 0.4726  1.4356 -1.0215 -0.4009] RMSE train/test 0.007406906129825661 0.0096016027300857 w range -0.09347509400138443 0.5612536318947967
walls 3 d [ 0.7231  1.5    -1.4609 -1.2058] RMSE train/test 0.007605454587523668 0.008969523180520241 w range 0.026502900837807014 0.5107571731986438
tracer -1 d [ 0.1922  0.2159 -0.028   0.0405] RMSE train/test 0.007923546788173254 None w range 0.03963937864517392 0.5543225842411923
tracer 0 d [ 0.1922  0.2159 -0.028   0.0405] RMSE train/test 0.00807668211160212 0.008264566136907547 w range 0.08009199777350014 0.49904350533708786
tracer 1 d [ 0.1922  0.2159 -0.028   0.0405] RMSE train/test 0.007835710652876061 0.008344693844666285 w range 0.02914257403627568 0.5644667443699056
tracer 2 d [ 0.1922  0.2159 -0.028   0.0405] RMSE train/test 0.007448963025984187 0.009961324456311935 w range -0.06419934046097947 0.5827387787552117
tracer 3 d [ 0.1922  0.2159 -0.028   0.0405] RMSE train/test 0.007800416312172024 0.008984816595106944 w range 0.05511850163745011 0.5684562037364688

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 10 — `execute_in_remote_docker` (`30434 ms` LLM)
- **Rationale**: Test whether a specific transport term explains the residual left by wall fitting. Keep the existing stencil unless a physically plausible correction is stable and improves held-out pulses, shifts, and rosettes; retain nominal walls when neither tracers nor apparatus evidence are available.
- **Alternatives Weighed**: `Estimate blind walls directly from routine measurements` (rejected: The fitted wall changes are several times larger than the independent tracer estimates, hit the search bounds, and worsen some held-out rosettes.)

```bash
cat > /tmp/tracer_identify.py <<'PY'
from transport_probe import *
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
ns=len(ids); ng=len(shifts)
# Probe whether the remaining mismatch transfers as a shared transport term.
ax0=axis(s['kappa_x'],s['vel_x'],hx,'ar','face','none'); ay0=axis(s['kappa_y'],s['vel_y'],hy,'ar','face','none')
ax=axis(s['kappa_x'],s['vel_x'],hx,'ar','face','velocity'); ay=axis(s['kappa_y'],s['vel_y'],hy,'ar','face','velocity')
D=np.kron(np.eye(ny),ax0)+np.kron(ay0,np.eye(nx)); A=np.kron(np.eye(ny),ax-ax0)+np.kron(ay-ay0,np.eye(nx)); V=np.diag(s['V_reaction_2d'].ravel(order='F'))
Bface,*_=init(1,'ar','face','facevelocity'); Bflux,*_=init(1,'ar','face','flux'); Bharm,*_=init(1,'ha','face','velocity')
folds=[]
for p in range(y.shape[1]):
 test=np.zeros_like(y,dtype=bool); test[:,p]=True; folds.append(('pulse',test))
for k in range(ng):
 test=np.zeros_like(y,dtype=bool); test[k*ns:(k+1)*ns]=True; folds.append(('shift',test))
for k in range(4):
 test=np.broadcast_to(((np.arange(len(y))%ns)//4%4==k)[:,None],y.shape); folds.append(('rosette',test))
for name,mat in [('walls only',None),('diffusion scale',D),('velocity scale',A),('reaction scale',V),('velocity indexing',Bface-B),('conservative flux',Bflux-B),('diffusion averaging',Bharm-B)]:
 def pred(a): return predict(B if mat is None else B+a[-1]*mat,E@a[:4])
 ini=d0 if mat is None else np.r_[d0,0.]; lower=(0.,)*4 if mat is None else (0.,)*4+(-.4,); upper=(np.inf,)*4 if mat is None else (np.inf,)*4+(.4,)
 full=least_squares(lambda a:(pred(a)-y).ravel(),ini,bounds=(lower,upper),gtol=1e-12)
 results={}; corrections=[]
 for kind,test in folds:
  fit=least_squares(lambda a:(pred(a)-y)[~test],ini,bounds=(lower,upper),gtol=1e-12)
  err=(pred(fit.x)-y).reshape(ng,ns,-1); mask=test.reshape(err.shape)
  slices=[np.sqrt(np.mean(err[k,:,p][mask[k,:,p]]**2)) for k in range(ng) for p in range(y.shape[1]) if mask[k,:,p].any()]
  results[kind]=max(results.get(kind,0.),max(slices)); corrections.append(fit.x[-1])
 print(name,'full RMSE',np.sqrt(np.mean(full.fun**2)), 'worst by holdout',results,'coefficient range', (min(corrections),max(corrections)) if mat is not None else None)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/tracer_identify.py
```
- **Container Exit Code**: `0` (`2641 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
walls only full RMSE 2.250693195993642e-05 worst by holdout {'pulse': 7.303784405969491e-05, 'shift': 4.162776444639128e-05, 'rosette': 4.974111924546883e-05} coefficient range None
diffusion scale full RMSE 2.2430739559053802e-05 worst by holdout {'pulse': 7.272038972632479e-05, 'shift': 4.1408320543707856e-05, 'rosette': 5.1142387753284016e-05} coefficient range (-0.0001156901407712838, -1.6165818648973374e-05)
velocity scale full RMSE 2.2506406904337347e-05 worst by holdout {'pulse': 7.661564567292361e-05, 'shift': 4.163327895325835e-05, 'rosette': 4.978657376332939e-05} coefficient range (-0.0004682692783009773, 0.0002501394124811589)
reaction scale full RMSE 2.250016441667814e-05 worst by holdout {'pulse': 7.560257108326345e-05, 'shift': 4.1637816645096186e-05, 'rosette': 5.023435477092675e-05} coefficient range (-0.0005402236970936647, 0.0001272954884509778)
velocity indexing full RMSE 2.2317626691282703e-05 worst by holdout {'pulse': 7.492946135997907e-05, 'shift': 4.141953136418601e-05, 'rosette': 4.8901032437758634e-05} coefficient range (-0.0036506665531719096, 0.012588864247354307)
conservative flux full RMSE 2.250575844773039e-05 worst by holdout {'pulse': 7.498014031316654e-05, 'shift': 4.162499707347879e-05, 'rosette': 5.04699054376491e-05} coefficient range (-0.00013974690123962544, 0.0003572098881051052)
diffusion averaging full RMSE 2.248437027801472e-05 worst by holdout {'pulse': 7.3337595685848e-05, 'shift': 4.1793684996052527e-05, 'rosette': 5.026100777824715e-05} coefficient range (-0.11479937151866942, 0.05405455364222679)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 11 — `execute_in_remote_docker` (`33548 ms` LLM)
- **Rationale**: The public sweeps select identity, so they never exercise the nonlinear inverse. Test that branch with known cross-channel distortion, both with and without tracer pulses. An inverse failure would require fixing calibration; a field or KKT failure against the independently assembled operator would identify a transport or optimization error.
- **Alternatives Weighed**: `Add an interior transport correction` (rejected: None of the tested corrections improved all held-out tracer slices.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, importlib.util
import numpy as np
sys.path.insert(0, '/tmp')
import transport_probe as ref
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = ref.s
B, E, hx, hy = ref.init(1, 'ar', 'face', 'velocity')
nx, ny = s['nx'], s['ny']
M = len(s['source_modes'])
F = np.stack([f.ravel(order='F') for f in s['source_modes']], axis=1)
d0 = np.repeat([s['robin_x']/hx, s['robin_y']/hy], 2)
Z = s['Z_nuisance']
w = np.full(M, s['total_budget']/M)

def raw(v):
    a = np.asarray(v).reshape(-1, 4)
    x, y, z, t = a.T
    c = np.column_stack((1.14*x + .16*y + .07*x*z + .035*y**3 + .03, .91*y - .07*z + .05*t**2 - .04, 1.07*z + .12*t - .04*y*t + .02, .88*t - .06*x + .03*z**3 - .02))
    return c.reshape(np.shape(v))

sweeps = dict(s['sensor_calibration_sweeps'])
sweeps['raw_rosette'] = raw(sweeps['ref_rosette'])
cal = m._calibration(sweeps)
for label, d, tracer in [('nonlinear blind', d0, False), ('nonlinear tracer', d0 * [1.22, .81, 1.15, .90], True)]:
    U = [np.linalg.solve(B + np.diag(E @ d + mu), F) for mu in s['mu_shifts']]
    S = np.vstack([v[ref.ids] for v in U])
    y = S @ w + Z @ np.array([.11, -.08, .06])
    p = [{'w_pulse': p['w_pulse'], 'y_pulse_raw': raw(S @ p['w_pulse'])} for p in s['reference_tracer_pulses']] if tracer else None
    fitted = m.solve_batch(dict(s, sensor_calibration_sweeps=sweeps, reference_tracer_pulses=p, y_obs_raw=raw(y)))
    P = np.eye(len(y)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S0, y0 = P @ S, P @ y
    v = fitted['weights']; cert = fitted['kkt_certificate']
    field_error = np.max(np.abs(fitted['terminal_field'] - (U[-1] @ v).reshape(nx, ny, order='F')))
    station_error = np.max(np.abs(S0.T @ (S0 @ v - y0) + s['source_ridge']*v + cert['nu_sum'] - cert['lam_lower'] + cert['lam_upper']))
    inverse_error = np.max(np.abs(cal(raw(y))-y))
    print(label, 'inverse max error', inverse_error, 'independent field max error', field_error, 'independent KKT stationarity max error', station_error)
    assert max(field_error, station_error, inverse_error) < 1e-8
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`638 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nonlinear blind inverse max error 2.220446049250313e-16 independent field max error 9.71445146547012e-17 independent KKT stationarity max error 3.7296554733501353e-16
nonlinear tracer inverse max error 2.3592239273284576e-16 independent field max error 2.185751579730777e-16 independent KKT stationarity max error 4.701100619897147e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 12 — `submit_completed_task` (`16455 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, independent field and KKT checks for both tracer and blind cases, and stress tests at box-simplex vertices. The available evidence does not support adding an interior or blind wall correction.
- **Alternatives Weighed**: `Fit blind wall corrections from routine observations` (rejected: The corrections hit the fit bounds, disagreed with tracer estimates, and worsened some held-out rosettes.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:7490'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst QP dual gap in stress tests was 1.14e-13. Without tracer pulses, the solver uses nominal wall parameters; no apparatus archive was available to validate a lineage correction.`

### Turn 13 — `execute_in_remote_docker` (`33623 ms` LLM)
- **Rationale**: Test whether omitted rational compression or pairwise mixing improves held-out bank-by-sweep error; only promote it if the gain transfers. Independently test whether the saved solver's tracer branch reproduces the measured boundary correction on an unseen pulse, and verify the public interface in a clean process.
- **Alternatives Weighed**: `Run relay-mode, Fokker–Planck/HJB, and within-batch classification checks` (rejected: The public study is a single continuous-response steady transport campaign, with no relay state, stochastic control objective, or class labels.)

```bash
cat > /tmp/campaign_margin_audit.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
y = raw - x
single = np.count_nonzero(x, axis=1) == 1
fold = np.arange(len(x)) % 4
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
poly = m._features(x, [()] + [t for k in (1,2,3) for t in itertools.combinations_with_replacement(range(4),k)])
def design(a, mode):
    q = x if mode == 'channel' else np.column_stack([(x[:,i] + sign*x[:,j])/np.sqrt(2) for i,j in pairs for sign in (-1,1)])
    return np.column_stack((np.ones(len(x)), x, q/(1+a*q*q)-q))
def fit_predict(P, train, test):
    return P[test] @ np.linalg.lstsq(P[train], y[train], rcond=None)[0]
options = [('identity', None), ('affine', poly[:,:5]), ('quadratic', poly[:,:15]), ('cubic', poly)]
options += [(f'{mode}@{a}', design(a,mode)) for mode in ('channel','pair') for a in (.1,.5,2.)]
base = y.copy(); nested = np.empty_like(y); selections = []
for k in range(4):
    train, test = fold != k, fold == k
    scores = []
    for name,P in options:
        errors = []
        for j in range(4):
            if j == k: continue
            val = fold == j
            errors.append(y[val] if P is None else y[val] - fit_predict(P, train & ~val, val))
        scores.append(np.mean(np.concatenate(errors)**2))
    name,P = options[np.argmin(scores)]
    selections.append(name)
    nested[test] = y[test] if P is None else y[test]-fit_predict(P,train,test)
for name,e in [('identity',base),('nested candidate library',nested)]:
    rms = np.sqrt(np.mean(e**2))
    worst = max(np.sqrt(np.mean(e[mask]**2,axis=0)).max() for mask in (single,~single))
    print(name, 'pooled RMSE',rms,'worst bank-by-sweep RMSE',worst)
print('outer-fold choices:',selections)
bank = c['single_bank_sweeps']
print('dedicated single-bank identity RMSE:',[float(np.sqrt(np.mean((bank['raw_readout'][bank['bank_id']==k]-bank['ref_signal'][bank['bank_id']==k])**2))) for k in range(4)])
cal=m._calibration(c)
p=s['reference_tracer_pulses']; ns=len(s['sensor_ij']); ng=len(s['mu_shifts'])
for label in ('nominal','held-out tracer'):
    metrics=[]
    for k,pulse in enumerate(p):
        t=dict(s,reference_tracer_pulses=None if label=='nominal' else [q for j,q in enumerate(p) if j!=k]); S,_=m._transport(t,cal)
        err=(S@pulse['w_pulse']-cal(pulse['y_pulse_raw'])).reshape(ng,ns)
        metrics.extend(np.sqrt(np.mean(err**2,axis=1)))
    print(label,'worst held-pulse shift RMSE',max(metrics))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /tmp/campaign_margin_audit.py && python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`642 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
identity pooled RMSE 1.0069909406158925e-05 worst bank-by-sweep RMSE 1.2233911343079284e-05
nested candidate library pooled RMSE 1.0069909406158925e-05 worst bank-by-sweep RMSE 1.2233911343079284e-05
outer-fold choices: ['identity', 'identity', 'identity', 'identity']
dedicated single-bank identity RMSE: [9.519539708885588e-06, 1.0531631447314842e-05, 9.907212136119426e-06, 1.4486620143343573e-05]
nominal worst held-pulse shift RMSE 0.000474698453787017
held-out tracer worst held-pulse shift RMSE 7.3039273194914e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bca46b3c45bcf324 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bca46b3c (directory).
```

### Turn 14 — `submit_completed_task` (`15228 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, independent field and KKT checks in both branches, and box-simplex stress tests. The held-out tracer improvement supports the wall correction when pulses are available; it does not establish a lineage correction for blind campaigns.
- **Alternatives Weighed**: `Add rational or pairwise rosette terms` (rejected: All four outer folds selected identity on the public sweeps; the expanded model library did not improve held-out error.); `Infer blind wall corrections from routine observations` (rejected: The fitted corrections disagreed with tracer estimates and worsened some held-out rosettes.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, tracer fitting reduced worst held-out RMSE from 4.75e-4 to 7.30e-5, and the worst QP duality gap in stress tests was 1.14e-13. Without tracer pulses or a validated apparatus correction, the solver uses nominal wall parameters.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
