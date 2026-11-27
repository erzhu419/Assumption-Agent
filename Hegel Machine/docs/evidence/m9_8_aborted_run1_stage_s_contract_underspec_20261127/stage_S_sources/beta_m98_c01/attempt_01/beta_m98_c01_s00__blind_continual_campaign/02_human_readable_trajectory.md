# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_c01_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `369,029` (`354,295` / `77,952` / `14,734`) |
| **Total Episode Wall Time (s)** | `436.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`17100 ms` LLM)
- **Rationale**: Inspect the public schema, calibration sweeps, and apparatus log to determine which parameters are observable and which must come from apparatus state.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && { if [ -d /tmp/hegel_continual_archive ]; then ls -lah /tmp/hegel_continual_archive; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 14000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; fi; } && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))
for k,v in s.items():
 print(k, type(v).__name__,getattr(v,'shape',''),repr(v)[:240] if not isinstance(v,np.ndarray) else (str(v.dtype),float(v.min()),float(v.max())) if v.size and np.issubdtype(v.dtype,np.number) else '')
c=s.get('sensor_calibration_sweeps')
if isinstance(c,dict):
 for k,v in c.items(): print('CAL',k,type(v).__name__,getattr(v,'shape',''),repr(v)[:700])
p=s.get('reference_tracer_pulses')
if p: print('PULSE',len(p),[(k,getattr(v,'shape',None)) for k,v in p[0].items()])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic programming sensor calibration robust regression'
```
- **Container Exit Code**: `0` (`369 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
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

### Turn 2 — `execute_in_remote_docker` (`20625 ms` LLM)
- **Rationale**: Read the study inside its batch wrapper and inspect the bank and pulse measurements. Their shapes and the available archive determine how calibration and boundary mismatch can be estimated independently.

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('batch',type(b),len(b));
for rec in b:
 print('record keys',rec.keys()); s=rec.get('study',rec)
 for k,v in s.items():
  if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,np.round(v.ravel()[:8],5))
  elif k not in ('sensor_calibration_sweeps','reference_tracer_pulses'): print(k,repr(v)[:240])
 c=s['sensor_calibration_sweeps']; print('calibration',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in c.items()]); q=c['single_bank_sweeps']; print('banks',[(k,np.shape(v),np.asarray(v).ravel()[:10]) for k,v in q.items()]); print('pulses',len(s.get('reference_tracer_pulses') or []));
PY
if [ -d /tmp/hegel_continual_archive ]; then ls -lh /tmp/hegel_continual_archive; head -c 4500 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; find /pretrain_task -maxdepth 2 -type f -printf '%p %s\n' | head -30; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -30
```
- **Container Exit Code**: `0` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
6]
nuisance_ridge 0.08
source_ridge 0.04
u_box 0.42
total_budget 1.4
calibration [('ref_rosette', (80, 4)), ('raw_rosette', (80, 4)), ('single_bank_sweeps', ['bank_id', 'ref_signal', 'raw_readout'])]
banks [('bank_id', (24,), array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])), ('ref_signal', (24,), array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4])), ('raw_readout', (24,), array([-1.10000364, -0.60000137, -0.10001157,  0.39999986,  0.9000046 ,
        1.40000392, -1.10000593, -0.59999936, -0.10001481,  0.40000211]))]
pulses 2
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
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
        }
      ],
      "index_cards": [
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
```

### Turn 3 — `execute_in_remote_docker` (`24803 ms` LLM)
- **Rationale**: Held-out sweep errors will show whether affine calibration is adequate; pulse ranges and nuisance fits will determine how to compare candidate transport operators.

```bash
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as terms
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; n=len(X)
print('ref/raw ranges',X.min(0),X.max(0),Y.min(0),Y.max(0)); print('delta percentiles',np.percentile(Y-X,[0,25,50,75,100],axis=0)); print('sample',np.round(np.c_[X[:8],(Y-X)[:8]],5)); print('single bank errors',np.round((c['single_bank_sweeps']['raw_readout']-c['single_bank_sweeps']['ref_signal']).reshape(4,-1),6))
for j in range(4):
 a=np.ones((n,1)); names=[()]; target=Y[:,j]-X[:,j]
 for d in (1,2,3):
  idx=list(terms(range(4),d)); names+=idx; a=np.c_[a,np.column_stack([X[:,t].prod(1) for t in idx])]; e=np.zeros(n)
  for fold in range(5):
   m=np.arange(n)%5==fold; e[m]=a[m]@np.linalg.lstsq(a[~m],target[~m],rcond=None)[0]-target[m]
  co=np.linalg.lstsq(a,target,rcond=None)[0]; print(j,d,'cv',np.sqrt(np.mean(e**2)),max(abs(e)),'terms',[(names[k],round(co[k],6)) for k in np.argsort(abs(co))[-6:][::-1]])
for p in s['reference_tracer_pulses']:
 print('pulse',p['w_pulse'],np.percentile(p['y_pulse_raw'],[0,25,50,75,100]), p['y_pulse_raw'][:12]); Z=s['Z_nuisance']; print('Zfit',np.linalg.lstsq(Z,p['y_pulse_raw'],rcond=None)[0])
print('telemetry',np.round(s['aux_manifold_telemetry'][:6],4)); print('bank ids unique',np.unique(s['row_bank_id'].reshape(-1,4),axis=0)); print('velocity',s['vel_x'],s['vel_y'])
PY
ls -ld /tmp/hegel_continual_archive /pretrain_task; ls -a /holdout_task
```
- **Container Exit Code**: `0` (`442 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(), 0.0)]
1 2 cv 1.4732807388741495e-05 4.135438480576219e-05 terms [((1,), -3e-06), ((0, 2), -2e-06), ((1, 1), 2e-06), ((2, 3), -2e-06), ((0,), 1e-06), ((0, 1), 1e-06)]
1 3 cv 1.7754092244073646e-05 4.89979278591064e-05 terms [((1,), -1.1e-05), ((0,), 9e-06), ((2,), -8e-06), ((0, 0, 3), -7e-06), ((0, 1, 2), -6e-06), ((0, 1, 3), 6e-06)]
2 1 cv 1.096166775806491e-05 3.060365214643168e-05 terms [((2,), 1e-06), ((1,), -1e-06), ((), -1e-06), ((0,), -0.0), ((3,), -0.0)]
2 2 cv 1.0713274811617084e-05 3.020159052822508e-05 terms [((1, 3), -5e-06), ((0, 2), 4e-06), ((1, 2), 3e-06), ((2, 2), -3e-06), ((0, 1), -2e-06), ((2, 3), 2e-06)]
2 3 cv 1.3665396445976606e-05 5.5380149633043046e-05 terms [((2,), 1.4e-05), ((0, 2), 8e-06), ((0,), -6e-06), ((1, 3), -6e-06), ((1, 2), 5e-06), ((1, 2, 3), -5e-06)]
3 1 cv 9.10857855775337e-06 2.478217033974276e-05 terms [((1,), 2e-06), ((), -2e-06), ((3,), 1e-06), ((2,), 1e-06), ((0,), 0.0)]
3 2 cv 9.316905721852016e-06 2.5911714259845794e-05 terms [((1, 2), 3e-06), ((0, 1), -3e-06), ((), -2e-06), ((1,), 2e-06), ((0, 2), -1e-06), ((3,), 1e-06)]
3 3 cv 1.671255855111383e-05 7.329657680486842e-05 terms [((0,), 9e-06), ((1, 2), 7e-06), ((3,), -6e-06), ((0, 0, 0), -5e-06), ((1, 2, 2), -5e-06), ((0, 1), -5e-06)]
pulse [1.  0.  0.  0.5 0.  0.  0.  0. ] [0.00724405 0.03048236 0.06092573 0.10777952 0.24709452] [0.04319874 0.24709452 0.14093363 0.06667238 0.05782845 0.12313358
 0.09108712 0.08108133 0.03392663 0.16976159 0.07673145 0.04909334]
Zfit [0.01922082 0.01116233 0.01068125]
pulse [0.  1.  0.  0.  0.5 0.  0.  0. ] [0.00516586 0.02707042 0.05454539 0.11080504 0.23795152] [0.03416688 0.17535901 0.16364842 0.11369859 0.04603065 0.21294911
 0.06101251 0.04673735 0.03204479 0.10984053 0.16336997 0.08057604]
Zfit [0.01831615 0.01328904 0.00944968]
telemetry [[-0.6633  1.072   0.25    1.4838]
 [-2.7079  0.7721 -1.5507  0.7837]
 [ 0.5645  1.5413 -0.1817  0.5948]
 [-0.9031  0.0532  1.5623 -0.1868]
 [-0.6258  0.3797  0.8899  0.5274]
 [ 0.6583 -0.8545 -0.8581  0.9898]]
bank ids unique [[0 1 2 3]]
velocity [ 0.35        0.34500413  0.33015913  0.30588881  0.27288602  0.23209293
  0.18467409  0.13198319  0.07552447  0.01690968 -0.04218784 -0.10008099
 -0.15511704 -0.20572484] [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907 -0.27988581 -0.27272334 -0.25469696 -0.22652476]
drwxr-xr-x 2 root root 4096 Nov 23 09:31 /pretrain_task
.
..
public_asset_index.json
public_batch.pkl
validate_public_schema.py
```
- **Stderr (tail)**:
```text
ls: cannot access '/tmp/hegel_continual_archive': No such file or directory
```

### Turn 4 — `execute_in_remote_docker` (`33748 ms` LLM)
- **Rationale**: Fit four wall diagonals while comparing interior stencils so boundary errors do not determine the stencil choice. Compare the recovered diagonals with Robin formulas, then hold out sensor rosettes to see whether the fitted operator predicts untrained locations.
- **Alternatives Weighed**: `Fit unrestricted cubic rosette calibration` (rejected: On the public sweeps, held-out errors rise above the affine calibration noise floor.)

```bash
cat > /tmp/transport_probe.py <<'PY'
import pickle,numpy as np,itertools
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; idx=s['sensor_ij']@np.array([1,nx]); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; B=s['source_modes'].reshape(len(s['source_modes']),-1,order='F').T@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); I=eye(nx*ny,format='csc')
wall=np.zeros((nx,ny,4)); wall[0,:,0]=wall[-1,:,1]=wall[:,0,2]=wall[:,-1,3]=1; W=wall.reshape(-1,4,order='F')
def one(k,v,h,km,vm,form):
 d=(k[:-1]+k[1:])/2 if km==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); d=d/h**2
 if vm==0: f=(v[:-1]+v[1:])/2/h; p=np.maximum(f,0); m=np.maximum(-f,0)
 elif vm==1: p=np.maximum(v[:-1],0)/h; m=np.maximum(-v[1:],0)/h
 else: p=np.maximum(v[1:],0)/h; m=np.maximum(-v[:-1],0)/h
 if form==0: a=np.r_[d+p,0]+np.r_[0,d+m]
 else: a=np.r_[d+m,0]+np.r_[0,d+p]
 return diags([-d-p,a,-d-m],[-1,0,1],format='csc')
def op(km,vm,form,spacing):
 x=one(s['kappa_x'],s['vel_x'],s['Lx']/(nx-spacing),km,vm,form); y=one(s['kappa_y'],s['vel_y'],s['Ly']/(ny-spacing),km,vm,form)
 return (kron(eye(ny),x)+kron(y,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
def predict(A,a,b=B):
 D=A+diags(W@a); return np.concatenate([splu(D+m*I).solve(b)[idx] for m in mu])
def fit(A,train=None):
 if train is None: train=np.ones(Y.shape[0],bool)
 return least_squares(lambda a:(predict(A,a)-Y)[train].ravel(),np.full(4,5.),bounds=(0.,200.),ftol=1e-11,gtol=1e-11,xtol=1e-11)
if __name__=='__main__':
 results=[]
 for key in itertools.product(range(2),range(3),range(2),range(2)):
  A=op(*key); f=fit(A); results.append((np.sqrt(np.mean(f.fun**2)),key,f.x))
 for rm,key,a in sorted(results)[:12]: print('walls fit',key,'rmse',rm,'wall diag',a)
 rm,key,a=min(results); A=op(*key)
 for fold in range(4):
  mask=np.tile((np.arange(len(idx))//4)%4==fold,len(mu)); f=fit(A,~mask); print('rosette jackknife',fold,np.sqrt(np.mean((predict(A,f.x)[mask]-Y[mask])**2)),f.x)
 for j in 'xy':
  n=s['n'+j]; h=s['L'+j]/(n-key[-1]); k=s['kappa_'+j][[0,-1]]; v=s['vel_'+j][[0,-1]]; r=s['robin_'+j]; print(j,'h,k,v,r',h,k,v,r,'r/h',r/h,'r/h^2',r/h**2,'kr/h^2',k*r/h**2,'series',r/h/(1+r*h/(2*k)))
 print('sensor locations',s['sensor_ij'].tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_probe.py
```
- **Container Exit Code**: `0` (`1310 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
walls fit (0, 2, 1, 1) rmse 2.060683296583251e-05 wall diag [4.98908758 4.98945349 7.07548137 7.16769594]
walls fit (1, 2, 1, 1) rmse 3.353929956384133e-05 wall diag [4.99014881 4.99621767 7.07848584 7.1663283 ]
walls fit (1, 0, 1, 1) rmse 0.00020972119761090201 wall diag [4.90276287 4.81511354 7.21933484 7.04543189]
walls fit (0, 0, 1, 1) rmse 0.00022906348053320828 wall diag [4.90169802 4.8088621  7.21631203 7.04703047]
walls fit (1, 1, 1, 1) rmse 0.00044397942067466123 wall diag [4.81243991 4.65833643 7.35610934 6.91678423]
walls fit (0, 1, 1, 1) rmse 0.0004635342655936374 wall diag [4.81137955 4.65252884 7.3530722  6.91860584]
walls fit (0, 1, 0, 0) rmse 0.0037173136963456798 wall diag [ 5.70921199  9.47981668 12.49008676  9.48485999]
walls fit (1, 1, 0, 0) rmse 0.0037270250192059803 wall diag [ 5.7135349   9.49967539 12.49766035  9.48698324]
walls fit (0, 0, 0, 0) rmse 0.0038142935865394245 wall diag [ 6.55253111  9.2396491  12.19653602  9.94239303]
walls fit (1, 0, 0, 0) rmse 0.0038262938946876197 wall diag [ 6.55782073  9.25868378 12.20417697  9.9452044 ]
walls fit (0, 2, 0, 0) rmse 0.004049146685921188 wall diag [ 7.46188792  9.0313486  11.89995554 10.42427284]
walls fit (1, 2, 0, 0) rmse 0.004062956588657225 wall diag [ 7.46832176  9.04967848 11.90766329 10.42786012]
rosette jackknife 0 2.6785883910588063e-05 [4.986924   4.98736033 7.07626824 7.17081819]
rosette jackknife 1 1.3613466560726502e-05 [4.98911525 4.98961046 7.07511262 7.1678585 ]
rosette jackknife 2 2.4753937214543398e-05 [4.99003194 4.99132209 7.07538782 7.16309996]
rosette jackknife 3 1.712279623922218e-05 [4.99026446 4.98961708 7.07495151 7.16825172]
x h,k,v,r 0.1846153846153846 [0.46954204 0.46954204] [ 0.35       -0.20572484] 0.880939340203039 r/h 4.771754759433128 r/h^2 25.84700494692945 kr/h^2 [12.13625532 12.13625532] series [4.06735182 4.06735182]
y h,k,v,r 0.18181818181818182 [0.56785556 0.32785556] [-0.         -0.22652476] 1.2931872722124154 r/h 7.112529997168284 r/h^2 39.118914984425565 kr/h^2 [22.21389338 12.82535378] series [5.89259324 5.23526732]
sensor locations [[0, 6], [6, 4], [10, 4], [12, 6], [10, 0], [10, 6], [2, 4], [6, 0], [12, 0], [4, 4], [10, 8], [6, 10], [4, 6], [10, 2], [0, 8], [0, 10], [0, 0], [2, 2], [10, 10], [12, 10], [6, 6], [8, 0], [8, 4], [6, 2], [2, 6], [12, 4], [8, 8], [0, 2], [2, 0], [4, 2], [2, 10], [6, 8]]
```

### Turn 5 — `execute_in_remote_docker` (`32907 ms` LLM)
- **Rationale**: If split walls improve prediction beyond the calibration noise floor, retain them for diagnostic campaigns; otherwise prefer axis symmetry. The index check looks for public apparatus metadata that could guide blind campaigns.
- **ATMS NOGOOD DECLARED**: Refuted `Face-averaged advection can fit the tracer data once the walls are re-estimated.` | Witness: `Its best pulse RMSE was 2.10e-4 versus 2.06e-5 for the arithmetic-diffusion, row-local-upwind stencil with four fitted wall diagonals.` | Blanket Diagnosis: `Refitting wall diagonals does not absorb interior stencil error.` | Sublated Pivot: `Use the row-local stencil and compare axis-shared and split Robin walls on held-out rosettes, shifts, and pulses.`

```bash
cd /holdout_task; python3 - <<'PY'
import json
j=json.load(open('public_asset_index.json')); print('index keys',list(j)); print('other metadata',{k:v for k,v in j.items() if k!='files'}); print('files',list(j.get('files',{})))
PY
cat > /tmp/wall_probe.py <<'PY'
from transport_probe import *
A=op(0,2,1,1); h=np.array([s['Lx']/(nx-1),s['Ly']/(ny-1)]); r=np.array([s['robin_x'],s['robin_y']]); a0=np.repeat(r/h,2)
G=[np.array([[1,0],[1,0],[0,1],[0,1]]),np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1]]),np.eye(4)]
def solve(g,mask):
 a=np.linalg.lstsq(g,a0,rcond=None)[0]; return g@least_squares(lambda a:(predict(A,g@a)-Y)[mask],a,gtol=1e-13,ftol=1e-12).x
masks=[np.tile((np.arange(len(idx))//4)%4==f,len(mu)) for f in range(4)]
masks += [np.arange(len(Y))//len(idx)==f for f in range(len(mu))]
for name,g in [('nominal',None),('axis',G[0]),('x shared',G[1]),('split',G[2])]:
 a=a0 if g is None else solve(g,np.ones(Y.shape,bool)); E=predict(A,a)-Y
 cv=[np.sqrt(np.mean((predict(A,a0 if g is None else solve(g,~mask))-Y)[mask]**2)) for mask in masks];
 pcv=[]
 for i in range(Y.shape[1]):
  train=np.ones_like(Y,dtype=bool); train[:,i]=False
  pcv.append(np.sqrt(np.mean((predict(A,a0 if g is None else solve(g,train))-Y)[:,i]**2)))
 print(name,'walls',a,'rmse',np.sqrt(np.mean(E**2)),'max',abs(E).max(),'rosette/shift cv',np.round(cv,8),'pulse cv',np.round(pcv,8))
 if g is G[-1]:
  print('residual by shift mean/std',np.round(E.reshape(len(mu),len(idx),-1).mean(1),8),np.round(E.reshape(len(mu),len(idx),-1).std(1),8)); print('residual Z slopes',np.linalg.lstsq(s['Z_nuisance'],E,rcond=None)[0]); print('residual wall rmse',[(i,np.sqrt(np.mean(E[np.tile(W[idx,i].astype(bool),len(mu))]**2))) for i in (0,2)])
print('rhs/cal noise',[(p['w_pulse'].sum(),) for p in pulses],np.std(s['sensor_calibration_sweeps']['raw_rosette']-s['sensor_calibration_sweeps']['ref_rosette']))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `1` (`466 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
index keys ['files', 'schema']
other metadata {'schema': 'hegel_public_asset_index_v1'}
files ['/holdout_task/public_batch.pkl', '/holdout_task/validate_public_schema.py']
nominal walls [4.77175476 4.77175476 7.11253    7.11253   ] rmse 0.0003960567240137659 max 0.0013544333004868736 rosette/shift cv [0.00049604 0.00028208 0.00044588 0.00032096 0.00054344 0.00035447
 0.00022273] pulse cv [0.000364   0.00042571]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/wall_probe.py", line 10, in <module>
    cv=[np.sqrt(np.mean((predict(A,a0 if g is None else solve(g,~mask))-Y)[mask]**2)) for mask in masks];
                                                        ^^^^^^^^^^^^^^
  File "/tmp/wall_probe.py", line 5, in solve
    a=np.linalg.lstsq(g,a0,rcond=None)[0]; return g@least_squares(lambda a:(predict(A,g@a)-Y)[mask],a,gtol=1e-13,ftol=1e-12).x
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_lsq/least_squares.py", line 835, in least_squares
    raise ValueError("`fun` must return at most 1-d array_like. "
ValueError: `fun` must return at most 1-d array_like. f0.shape: (72, 2)
```

### Turn 6 — `execute_in_remote_docker` (`23921 ms` LLM)
- **Rationale**: Complete the held-out wall comparison after flattening the least-squares residual. On the same study, hide the pulses and compare observation residuals under nominal and tracer-fitted walls to see whether blind observations can distinguish the wall mismatch from source and nuisance variation. Read the validator before implementing the return contract.

```bash
sed -i 's/(predict(A,g@a)-Y)\[mask\],a/(predict(A,g@a)-Y)[mask].ravel(),a/' /tmp/wall_probe.py; OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py; cat /holdout_task/validate_public_schema.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys; sys.path.insert(0,'/tmp')
from transport_probe import *
A=op(0,2,1,1); F=s['source_modes'].reshape(len(s['source_modes']),-1,order='F').T; Z=s['Z_nuisance']; y=s['y_obs_raw']; u=s['u_box']; budget=s['total_budget']; a=fit(A).x
from scipy.optimize import minimize
for walls in [np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2),a]:
 S=predict(A,walls,F); X=np.c_[S,Z]; Xp=np.r_[X,np.r_[np.ones(S.shape[1]),np.zeros(Z.shape[1])][None,:]]; yp=np.r_[y,budget]
 f=minimize(lambda q:0.5*np.sum((Xp@q-yp)**2),np.r_[np.full(S.shape[1],budget/S.shape[1]),np.zeros(Z.shape[1])],jac=lambda q:Xp.T@(Xp@q-yp),bounds=[(0,u)]*S.shape[1]+[(None,None)]*Z.shape[1],method='SLSQP',constraints=[{'type':'eq','fun':lambda q:q[:S.shape[1]].sum()-budget}],options={'ftol':1e-14,'maxiter':500}); e=X@f.x-y; print('obs fit',walls,'rmse',np.sqrt(np.mean(e**2)), 'by shift',np.sqrt(np.mean(e.reshape(len(mu),-1)**2,axis=1)),'weights',f.x[:S.shape[1]])
T=s['aux_manifold_telemetry']; print('aux mean,std',T.mean(0),T.std(0)); print('aux covariance',np.round(np.cov(T.T),3))
PY
```
- **Container Exit Code**: `0` (`1111 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e-05 2.394e-05]
split walls [4.98908757 4.98945349 7.07548137 7.16769594] rmse 2.0606832965828788e-05 max 8.428908040275518e-05 rosette/shift cv [2.679e-05 1.361e-05 2.475e-05 1.712e-05 4.925e-05 1.862e-05 2.755e-05] pulse cv [2.318e-05 2.426e-05]
residual by shift mean/std [[-1.030e-05 -1.827e-05]
 [ 1.154e-05  8.420e-06]
 [ 1.942e-05  1.911e-05]] [[1.274e-05 1.384e-05]
 [1.025e-05 1.268e-05]
 [1.494e-05 1.785e-05]]
residual Z slopes [[-2.72430571e-06 -3.98441595e-07]
 [ 1.30125380e-06  2.18335187e-06]
 [-1.19058145e-06 -2.28946070e-07]]
residual wall rmse [(0, 2.7683219149742948e-05), (2, 1.4833654645430508e-05)]
rhs/cal noise [(1.5,), (1.5,)] 1.0042945790876763e-05
import importlib.util, json, os, pickle, sys
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
obs fit [4.77175476 4.77175476 7.11253    7.11253   ] rmse 0.008482387043441323 by shift [0.00829656 0.00884128 0.00829768] weights [7.77416195e-02 4.20000000e-01 2.62060308e-01 5.91688807e-02
 4.20000000e-01 1.03855107e-17 1.61029191e-01 1.51049797e-19]
obs fit [4.98908758 4.98945349 7.07548137 7.16769594] rmse 0.008432804058259247 by shift [0.00812934 0.0088297  0.00832388] weights [7.29373545e-02 4.20000000e-01 2.62268850e-01 6.18702196e-02
 4.20000000e-01 1.73662930e-17 1.62923576e-01 1.80846828e-18]
aux mean,std [-0.30329271  0.13139762  0.27366888  0.02608553] [1.0646252  1.02447233 1.09257829 1.1028606 ]
aux covariance [[ 1.183  0.016 -0.011  0.006]
 [ 0.016  1.095  0.089  0.498]
 [-0.011  0.089  1.246  0.066]
 [ 0.006  0.498  0.066  1.269]]
```

### Turn 7 — `execute_in_remote_docker` (`24720 ms` LLM)
- **Rationale**: Inspect residuals by rosette, bank, and auxiliary telemetry and measure source conditioning. A repeatable residual pattern would justify a targeted measurement-model test; otherwise the blind fallback should favor apparatus and nominal wall information over unconstrained observation fitting.
- **Alternatives Weighed**: `Fit independent walls directly to the routine observations` (rejected: The observation residual is about 400 times the tracer residual, so a wall fit could absorb an unrelated measurement discrepancy.)

```bash
cat > /tmp/obs_probe.py <<'PY'
from transport_probe import *
np.set_printoptions(precision=6,suppress=True,linewidth=150)
for k,v in s.items():
 if k not in ['sensor_calibration_sweeps','reference_tracer_pulses']: print(k,getattr(v,'shape',None),repr(v)[:185] if not isinstance(v,np.ndarray) else np.percentile(v,[0,25,50,75,100]))
A=op(0,2,1,1); a=fit(A).x; F=s['source_modes'].reshape(len(s['source_modes']),-1,order='F').T; S=predict(A,a,F); Z=s['Z_nuisance']; y=s['y_obs_raw']; u=np.broadcast_to(s['u_box'],S.shape[1]); b=s['total_budget']; X=np.c_[S,Z]; from scipy.optimize import minimize
r=minimize(lambda q:.5*np.sum((X@q-y)**2),np.r_[u*b/u.sum(),np.zeros(Z.shape[1])],jac=lambda q:X.T@(X@q-y),bounds=list(zip(np.zeros(len(u)),u))+[(None,None)]*Z.shape[1],method='SLSQP',constraints={'type':'eq','fun':lambda q:q[:len(u)].sum()-b},options={'ftol':1e-14,'maxiter':500}); e=(X@r.x-y).reshape(-1,4); print('fit',r.x); print('bank rmse',np.sqrt(np.mean(e**2,0))); print('obs, residual and telemetry each rosette'); print(np.c_[y.reshape(-1,4),e,s['aux_manifold_telemetry']]); print('singular projected source on equality',np.linalg.svd((S-Z@np.linalg.lstsq(Z,S,rcond=None)[0])@np.r_[np.eye(len(u)-1),-np.ones((1,len(u)-1))],compute_uv=False)); print('cal shapes',[(k,np.shape(v)) for k,v in s['sensor_calibration_sweeps'].items()]); print('bank sweep keys',s['sensor_calibration_sweeps']['single_bank_sweeps'].keys())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/obs_probe.py
ls -la /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.858094  0.989796]
 [ 0.110126  0.271497 -0.033956  0.045788 -0.001843  0.003274  0.009224 -0.003996 -1.833271 -0.659835  0.257321  0.431485]
 [ 0.181625  0.318516  0.124179  0.260097 -0.014386 -0.000164  0.008413 -0.006977 -0.224107 -1.55741  -0.492325 -0.259881]
 [ 0.144165  0.028434  0.152783  0.037118  0.009662  0.001114 -0.006674  0.003433 -0.334264 -0.650661 -1.772357  0.080161]
 [ 0.30166  -0.07604   0.046356  0.178886  0.003072 -0.010414  0.003188 -0.001627  1.526425  1.743434 -0.321334  1.156117]
 [-0.044373  0.057284  0.234498  0.250931 -0.004614  0.010804 -0.002187  0.00183   0.794921  0.503133 -0.18878   1.034697]
 [ 0.172963  0.008988 -0.14396  -0.136554 -0.004711  0.003765  0.005727 -0.01535  -0.48098   1.403099  1.502046  0.840121]
 [ 0.234872 -0.309786 -0.199573  0.192586  0.007133  0.015622 -0.003564  0.001168  0.293296 -1.874019 -0.042746 -1.559561]
 [ 0.188179  0.24401   0.413276  0.272868  0.001887  0.00386  -0.022523 -0.002756 -0.580948 -0.245131  1.385093 -1.675923]
 [ 0.073335  0.141559  0.287974  0.053822  0.009607  0.001091 -0.008121 -0.015879 -1.132372  1.311074  1.083966  0.370119]
 [ 0.094753  0.189437 -0.102215 -0.104655 -0.00907   0.018374  0.008758  0.003431  1.476135  0.395027  1.005606 -1.13919 ]
 [ 0.162093 -0.187824  0.34998   0.092052  0.018551 -0.010586 -0.009986  0.001368  0.393279  1.214653  1.032322  0.65732 ]
 [-0.23586   0.46504   0.113253  0.21246   0.00108  -0.002458  0.003247  0.000472 -0.765546  0.084032 -1.542867 -2.576488]
 [ 0.168094 -0.234278  0.31086   0.114496 -0.008369  0.008567  0.001788  0.006783 -2.189698  0.238324  1.827234  0.048241]
 [ 0.268933 -0.048738 -0.298856  0.20715   0.018484  0.008331 -0.007395  0.011284 -1.212282 -0.262893 -0.126865  1.080222]
 [ 0.040181 -0.010671 -0.086656  0.144881 -0.010039 -0.006659  0.016762 -0.001905  0.345035 -0.788612  2.542839  0.984789]
 [ 0.028531 -0.126808  0.105957  0.105504 -0.003913  0.006573 -0.003823  0.001678 -0.647564 -1.241454  0.786746 -2.275597]
 [-0.248163  0.024986  0.271002 -0.014219 -0.009789 -0.003893  0.003126 -0.00476   1.214176 -0.868006 -0.127091 -0.288177]
 [ 0.081982  0.024362 -0.292475  0.36859   0.009331  0.007912  0.002732 -0.007036 -0.244038  1.444956 -0.352421 -0.475062]]
singular projected source on equality [1.203157 0.476424 0.306849 0.150729 0.08664  0.051539 0.019684]
cal shapes [('ref_rosette', (80, 4)), ('raw_rosette', (80, 4)), ('single_bank_sweeps', ())]
bank sweep keys dict_keys(['bank_id', 'ref_signal', 'raw_readout'])
```

### Turn 8 — `execute_in_remote_docker` (`67935 ms` LLM)
- **Rationale**: Implement the empirically supported stencil and the stated projected QP. Check the schema now; next independently stress-test stationarity and field ordering, and test whether a shared shift or apparatus prior improves held-out predictions.
- **Alternatives Weighed**: `Estimate four blind wall terms from routine observations` (rejected: The routine residual is much larger than the tracer response to wall changes; no measured holdout supports this fit yet.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated transport inversion and box-simplex optimality certificate."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    terms = {d: list(combinations_with_replacement(range(4), d)) for d in (2, 3)}

    def design(z, degree):
        cols = [np.ones(len(z)), *z.T]
        for d in range(2, degree + 1):
            cols.extend(z[:, t].prod(axis=1) for t in terms[d])
        return np.column_stack(cols)

    # Compare forward models on shared held-out sweeps; invert only the winner.
    degree, best = 1, np.inf
    fold = np.arange(len(x)) % 5
    for d in (1, 2, 3):
        a = design(x, d)
        if a.shape[1] >= 0.8 * len(x):
            continue
        mse = np.mean([np.mean((a[fold == f] @ np.linalg.lstsq(a[fold != f], y[fold != f], rcond=None)[0] - y[fold == f])**2)
                       for f in np.unique(fold)])
        if mse < 0.92 * best:
            degree, best = d, mse
    c = np.linalg.lstsq(design(x, degree), y, rcond=None)[0]

    def correct(raw):
        shape = np.shape(raw)
        raw = np.asarray(raw, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(c[1:5].T, (raw - c[0]).T).T
        if degree > 1:
            for i, row in enumerate(raw):
                sol = least_squares(lambda u: (design(u[None, :], degree) @ c)[0] - row,
                                    z[i], gtol=1e-12, ftol=1e-12, xtol=1e-12)
                z[i] = sol.x
        return z.reshape(shape)

    return correct


def _axis(k, v, h):
    d = (k[:-1] + k[1:]) * (0.5 / h**2)
    lo = d + np.maximum(v[1:], 0) / h
    hi = d + np.maximum(-v[:-1], 0) / h
    return diags((-lo, np.r_[0., lo] + np.r_[hi, 0.], -hi), (-1, 0, 1), format='csc')


def _chamber(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx'] / (nx - 1), s['Ly'] / (ny - 1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    a = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    wall = np.zeros((nx, ny, 4))
    wall[0, :, 0] = wall[-1, :, 1] = wall[:, 0, 2] = wall[:, -1, 3] = 1
    wall = wall.reshape(-1, 4, order='F')
    ij = np.asarray(s['sensor_ij'], dtype=int)
    at = ij[:, 0] + nx * ij[:, 1]
    ident = eye(nx * ny, format='csc')
    source = np.asarray(s['source_modes'], dtype=float).reshape(-1, nx * ny, order='F').T

    def response(walls, rhs=source, shift_offset=0.):
        operator = a + diags(wall @ walls)
        rows = []
        for shift in s['mu_shifts']:
            field = splu(operator + (shift + shift_offset) * ident).solve(rhs)
            rows.append(field[at])
        return np.concatenate(rows, axis=0), field.T.reshape(-1, nx, ny, order='F')

    return response, source, np.repeat([s['robin_x'] / hx, s['robin_y'] / hy], 2)


def _fit_walls(s, correct, response, source, nominal):
    pulses = s.get('reference_tracer_pulses')
    if not pulses:
        return nominal
    rhs = source @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([correct(p['y_pulse_raw']) for p in pulses])

    def residual(walls):
        return (response(walls, rhs)[0] - observed).ravel()

    return least_squares(residual, np.maximum(nominal, 1e-8), bounds=(0., np.inf),
                         gtol=1e-13, ftol=1e-11, xtol=1e-11).x


def _box_simplex(h, c, upper, budget):
    """Primal active-set QP, followed by a Lagrangian lower bound."""
    n = len(c)
    w = np.zeros(n) if not upper.sum() else upper * (budget / upper.sum())
    lock = upper == 0.
    state = np.zeros(n, dtype=int)
    if budget == 0:
        state[:] = -1
    elif budget == upper.sum():
        state[:] = 1
    for _ in range(max(100, 10*n*n)):
        free = (state == 0) & ~lock
        g = h @ w + c
        if np.any(free):
            hf = h[np.ix_(free, free)]
            invg, inv1 = np.linalg.solve(hf, np.column_stack((g[free], np.ones(free.sum())))).T
            nu = -invg.sum() / inv1.sum()
            p = -invg - nu * inv1
            ix = np.flatnonzero(free)
            ratios = np.full(len(ix), np.inf)
            neg, pos = p < -1e-14, p > 1e-14
            ratios[neg] = -w[ix[neg]] / p[neg]
            ratios[pos] = (upper[ix[pos]] - w[ix[pos]]) / p[pos]
            k = ratios.argmin()
            step = np.clip(ratios[k], 0., 1.)
            w[ix] += step * p
            if step < 1. - 1e-12:
                j = ix[k]
                state[j] = 1 if p[k] > 0 else -1
                w[j] = upper[j] if p[k] > 0 else 0.
                continue
            g = h @ w + c
        else:
            lower = (state < 0) & ~lock
            capped = (state > 0) & ~lock
            lo = np.max(-g[lower], initial=-np.inf)
            hi = np.min(-g[capped], initial=np.inf)
            if lo > hi + 1e-11:
                state[np.flatnonzero(lower)[np.argmin(g[lower])]] = 0
                state[np.flatnonzero(capped)[np.argmax(g[capped])]] = 0
                continue
            nu = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.clip(0., lo, hi)
        shifted = g + nu
        bad = np.where(lock, 0., state * shifted)
        j = bad.argmax()
        if bad[j] <= 1e-11:
            break
        state[j] = 0
    else:
        raise RuntimeError('box-simplex active set did not converge')
    lam_lower = np.where((state < 0) | lock, np.maximum(shifted, 0.), 0.)
    lam_upper = np.where((state > 0) | lock, np.maximum(-shifted, 0.), 0.)
    return w, float(nu), lam_lower, lam_upper


def solve_batch(study: dict) -> dict:
    s = study
    correct = _calibration(s['sensor_calibration_sweeps'])
    y = correct(s['y_obs_raw'])
    response, source, nominal = _chamber(s)
    walls = _fit_walls(s, correct, response, source, nominal)
    design, fields = response(walls)
    z = np.asarray(s['Z_nuisance'], dtype=float)
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    yz = np.column_stack((design, y))
    projected = yz - z @ np.linalg.solve(gram, z.T @ yz)
    x, r = projected[:, :-1], projected[:, -1]
    ridge = s['source_ridge']
    h = x.T @ x + ridge * np.eye(x.shape[1])
    c = -x.T @ r
    upper = np.broadcast_to(s['u_box'], c.shape)
    budget = s['total_budget']
    w, nu, lower, cap = _box_simplex(h, c, upper, budget)
    err = x @ w - r
    objective = float((err @ err + ridge * (w @ w)) / 2)
    stationarity = h @ w + c + nu - lower + cap
    lagrangian = objective + nu * (w.sum() - budget) - lower @ w + cap @ (w - upper)
    dual = lagrangian - 0.5 * stationarity @ np.linalg.solve(h, stationarity)
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': cap,
                                'primal_objective': objective,
                                'dual_lower_bound': float(dual)}}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`435 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b56e70e13bfc7bea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b56e70e1 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`25524 ms` LLM)
- **Rationale**: The tracer residual changes sign across shifts after fitting the walls. Test a single shared offset on held-out shifts, rosettes, and pulses using the submitted stencil. Retain it only if it improves prediction beyond the calibration noise floor.
- **Alternatives Weighed**: `Fit a separate offset for every shift` (rejected: A separate offset cannot predict a held-out shift.)

```bash
cat > /tmp/offset_probe.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fix=m._calibration(s['sensor_calibration_sweeps']); R,F,a0=m._chamber(s); pulses=s['reference_tracer_pulses']; B=F@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([fix(p['y_pulse_raw']) for p in pulses]); ns=len(s['sensor_ij']); rows=np.arange(len(Y)); folds=[(rows%ns//4)%4==j for j in range(4)]+[rows//ns==j for j in range(len(s['mu_shifts']))]; masks=[np.broadcast_to(f[:,None],Y.shape) for f in folds]+[np.broadcast_to(np.arange(Y.shape[1])==j,Y.shape) for j in range(Y.shape[1])]
for label,G in [('axis',np.repeat(np.eye(2),2,axis=0)),('split',np.eye(4))]:
 q0=np.linalg.lstsq(G,a0,rcond=None)[0]
 for shift in [False,True]:
  x0=np.r_[q0,0.] if shift else q0
  def pred(q): return R(G@(q[:-1] if shift else q),B,q[-1] if shift else 0.)[0]
  def est(mask): return least_squares(lambda q:(pred(q)-Y)[mask],x0,gtol=1e-13,ftol=1e-10,xtol=1e-11).x
  q=est(np.ones(Y.shape,bool)); cv=[np.sqrt(np.mean((pred(est(~mask))-Y)[mask]**2)) for mask in masks]; print(label,shift,'params',q,'all',np.sqrt(np.mean((pred(q)-Y)**2)),'cv',np.round(cv,8))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/offset_probe.py
if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 4000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi
```
- **Container Exit Code**: `0` (`1519 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
axis False params [5.00254057 7.08687451] all 5.47543951610426e-05 cv [6.092e-05 8.524e-05 7.663e-05 4.392e-05 8.089e-05 5.042e-05 3.982e-05
 5.267e-05 6.990e-05]
axis True params [4.99200532e+00 7.07882496e+00 7.56462036e-04] all 5.188435327443295e-05 cv [6.683e-05 7.297e-05 7.178e-05 4.060e-05 7.704e-05 4.781e-05 3.827e-05
 4.764e-05 6.653e-05]
split False params [4.9891807  4.98944388 7.07556087 7.16706666] all 2.0623697078017906e-05 cv [2.676e-05 1.361e-05 2.473e-05 1.730e-05 4.908e-05 1.867e-05 2.749e-05
 2.320e-05 2.426e-05]
split True params [4.98801531e+00 4.98794098e+00 7.07456490e+00 7.16472072e+00
 1.10015820e-04] all 2.0481941739006323e-05 cv [2.722e-05 1.375e-05 2.442e-05 1.694e-05 4.883e-05 1.852e-05 2.736e-05
 2.336e-05 2.428e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b56e70e13bfc7bea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b56e70e1 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`36690 ms` LLM)
- **Rationale**: Stress-test feasibility, KKT conditions, and terminal field ordering. Fit blind wall candidates without tracer access and score them against the withheld tracers; adopt a blind fit only if it improves that independent measurement.
- **Alternatives Weighed**: `Add a shared shift offset` (rejected: Worst-slice improvement is smaller than calibration noise, and pulse holdouts become slightly worse.)

```bash
cat > /tmp/verify_solver.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fix=m._calibration(s['sensor_calibration_sweeps']); R,F,a0=m._chamber(s); w0=m.solve_batch(s); cert=w0['kkt_certificate']; a=m._fit_walls(s,fix,R,F,a0); S,U=R(a); T=U.transpose(1,2,0).reshape(s['nx']*s['ny'],-1,order='F'); assert np.allclose(T[s['sensor_ij'][:,0]+s['nx']*s['sensor_ij'][:,1]], S[-len(s['sensor_ij']):],atol=1e-14); print('public certificate gap',cert['primal_objective']-cert['dual_lower_bound'],'weights',w0['weights'])
rng=np.random.default_rng(4); worst=np.zeros(4)
for i in range(600):
 n=int(rng.integers(2,32)); D=rng.normal(size=(n,n)); H=D.T@D+10**rng.uniform(-4,0)*np.eye(n); c=rng.normal(size=n)*10**rng.uniform(-2,2); u=rng.uniform(.05,1,n); u[rng.random(n)<.15]=0
 if i%7==0: u[:]=float(rng.choice([.25,.5,1.]));
 b=[0.,u.sum(),u[:n//2].sum(),rng.uniform(0,u.sum())][i%4]; w,nu,l,h=m._box_simplex(H,c,u,b); e=H@w+c+nu-l+h; worst=np.maximum(worst,[max(abs(w.sum()-b),max(-w),max(w-u)),max(abs(e)),max(max(abs(w*l)),max(abs((u-w)*h))),abs(nu*(w.sum()-b)-l@w+h@(w-u)-.5*e@np.linalg.solve(H,e))]); assert min(l)>=0 and min(h)>=0
print('stress worst feasibility/stationarity/complementarity/gap',worst); assert max(worst)<1e-8
Z=s['Z_nuisance']; y=fix(s['y_obs_raw']); u=np.broadcast_to(s['u_box'],(F.shape[1],)); b=s['total_budget']; ridge=s['source_ridge']; ns=len(s['sensor_ij']); pulses=s['reference_tracer_pulses']; B=F@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([fix(p['y_pulse_raw']) for p in pulses]); I=np.eye(len(u)); G=np.repeat(np.eye(2),2,axis=0)
def profile(walls,mask):
 sm=R(walls)[0][mask]; z=Z[mask]; yz=np.column_stack((sm,y[mask])); zz=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); x,r=(yz-z@np.linalg.solve(zz,z.T@yz))[:,:-1],(yz-z@np.linalg.solve(zz,z.T@yz))[:,-1]; w=m._box_simplex(x.T@x+ridge*I,-x.T@r,u,b)[0]; g=np.linalg.solve(zz,z.T@(y[mask]-sm@w)); return w,g
rows=np.arange(len(y)); tests=[(rows%ns//4)%4==i for i in range(4)]+[rows//ns==i for i in range(len(s['mu_shifts']))]; allrows=np.ones(len(y),bool)
for name,g,prior in [('nominal',G,0),('axis',G,.1),('axis',G,.01),('split',np.eye(4),.01),('axis',G,0.)]:
 def wallfit(mask):
  if name=='nominal': return a0
  def f(delta):
   walls=a0+g@delta; w,gamma=profile(walls,mask); return np.r_[(R(walls)[0]@w+Z@gamma-y)[mask],prior*delta]
  return a0+g@least_squares(f,np.zeros(g.shape[1]),bounds=(-1.5,1.5),max_nfev=32).x
 aq=wallfit(allrows); out=[]
 for mask in tests:
  aa=wallfit(~mask); w,gamma=profile(aa,~mask); out.append(np.sqrt(np.mean((R(aa)[0]@w+Z@gamma-y)[mask]**2)))
 e=(R(aq,B)[0]-Y).reshape(len(s['mu_shifts']),ns,-1); print('blind',name,prior,'walls',np.round(aq,4),'worst routine CV',max(out),'worst independent tracer shift',max(np.sqrt(np.mean(e**2,axis=(1,2)))))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_solver.py
```
- **Container Exit Code**: `0` (`2334 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public certificate gap 0.0 weights [0.08854811 0.33161789 0.29302007 0.14214732 0.33659508 0.04563901
 0.16243251 0.        ]
stress worst feasibility/stationarity/complementarity/gap [7.10542736e-15 1.42108547e-14 0.00000000e+00 1.94975082e-13]
blind nominal 0 walls [4.7718 4.7718 7.1125 7.1125] worst routine CV 0.012097861835270948 worst independent tracer shift 0.0005432605491209258
blind axis 0.1 walls [4.7883 4.7883 7.1442 7.1442] worst routine CV 0.012093035343416644 worst independent tracer shift 0.0004557584448292425
blind axis 0.01 walls [4.7623 4.7623 8.0555 8.0555] worst routine CV 0.012009376352298403 worst independent tracer shift 0.0018944089963860973
blind split 0.01 walls [4.8009 4.8937 7.8974 7.6338] worst routine CV 0.012029451042997536 worst independent tracer shift 0.0015731862166702567
blind axis 0.0 walls [4.4433 4.4433 8.6125 8.6125] worst routine CV 0.01199033715561851 worst independent tracer shift 0.0026595651773833884

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b56e70e13bfc7bea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b56e70e1 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`36759 ms` LLM)
- **Rationale**: Compare small, shared corrections to the interior operator while refitting wall terms inside each fold. If gains do not persist across held-out shifts and pulses, retain the measured stencil. The public QP already passed 600 stress cases with maximum certificate gap 1.95e-13.
- **Alternatives Weighed**: `Select blind wall fits by routine-observation cross-validation alone` (rejected: The best routine score coincides with the worst independent tracer prediction.)

```bash
cat > /tmp/transport_residual_probe.py <<'PY'
import sys
sys.path.insert(0,'/tmp')
from transport_probe import *
from importlib.util import spec_from_file_location, module_from_spec
P=spec_from_file_location('solver','/holdout_task/batch_solver.py'); M=module_from_spec(P); P.loader.exec_module(M)
correct=M._calibration(s['sensor_calibration_sweeps']); Y=np.column_stack([correct(p['y_pulse_raw']) for p in pulses]); n=len(Y); nr=len(idx); rows=np.arange(n); masks=[np.broadcast_to(mask[:,None],Y.shape) for mask in [(rows%nr//4)%4==f for f in range(4)]+[rows//nr==f for f in range(len(mu))]]; masks.extend(np.broadcast_to(np.arange(Y.shape[1])==j,Y.shape) for j in range(Y.shape[1])); A=op(0,2,1,1)
def delta(j, which):
 k=s['kappa_'+j]; v=s['vel_'+j]; h=s['L'+j]/(s['n'+j]-1); D=one(k,np.zeros_like(v),h,0,2,1) if which=='k' else one(np.zeros_like(k),v,h,0,2,1)
 return (kron(eye(ny),D) if j=='x' else kron(D,eye(nx))).tocsc()
Ds={j+t:delta(j,t) for j in 'xy' for t in ['k','v']}; Ds['reaction']=diags(s['V_reaction_2d'].ravel(order='F')); a0=np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2)
for keys in [(),('xk','yk'),('xv','yv'),('reaction',),('xk','yk','xv','yv','reaction')]:
 def pred(q):
  D=A.copy()
  for key,b in zip(keys,q[4:]): D=D+b*Ds[key]
  return predict(D,q[:4])
 x0=np.r_[a0,np.zeros(len(keys))]
 def fit_mask(mask): return least_squares(lambda q:(pred(q)-Y)[mask],x0,gtol=1e-13,ftol=1e-10,xtol=1e-11).x
 q=fit_mask(np.ones(Y.shape,bool)); cv=np.array([np.sqrt(np.mean((pred(fit_mask(~mask))-Y)[mask]**2)) for mask in masks]); e=(pred(q)-Y).reshape(len(mu),nr,-1)
 print(keys,'fit',np.sqrt(np.mean(e**2)),'worst',cv.max(),'folds',np.round(cv*1e6,2),'params',np.round(q,7))
print('archive:',__import__('pathlib').Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').exists())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/transport_residual_probe.py
```
- **Container Exit Code**: `0` (`2789 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
() fit 2.0623697078030815e-05 worst 4.9081562795207376e-05 folds [26.76 13.61 24.73 17.3  49.08 18.67 27.49 23.2  24.26] params [4.9891807 4.9894439 7.0755609 7.1670667]
('xk', 'yk') fit 1.972444717313726e-05 worst 4.7700457834661405e-05 folds [26.62 13.95 23.84 16.26 47.7  17.95 26.58 20.95 21.91] params [ 4.9820346e+00  4.9835488e+00  7.0793110e+00  7.1746322e+00
 -3.5350000e-04  3.9000000e-04]
('xv', 'yv') fit 2.011060541407448e-05 worst 4.812314620355001e-05 folds [27.15 13.36 23.9  16.52 48.12 18.14 27.   22.15 23.99] params [4.9864936e+00 4.9882462e+00 7.0750394e+00 7.1596296e+00 5.5430000e-04
 7.4800000e-04]
('reaction',) fit 2.0496139180361624e-05 worst 4.88622705861743e-05 folds [27.19 13.76 24.45 16.97 48.86 18.53 27.38 23.34 24.28] params [4.9881537e+00 4.9881519e+00 7.0745226e+00 7.1650948e+00 3.2110000e-04]
('xk', 'yk', 'xv', 'yv', 'reaction') fit 1.92987523969877e-05 worst 4.702056536663851e-05 folds [26.52 14.9  23.13 15.57 47.02 17.57 26.25 26.04 24.11] params [ 4.9802400e+00  4.9827553e+00  7.0791290e+00  7.1666757e+00
 -4.4470000e-04  2.9160000e-04  2.9600000e-05  1.0227000e-03
  1.9780000e-04]
archive: False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b56e70e13bfc7bea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b56e70e1 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`28216 ms` LLM)
- **Rationale**: Test the pulse-free path end to end on public observations and on planted affine and nonlinear sweeps. A failed invariance check would point to calibration inversion; a failed field or gap check would point to the blind solver path.
- **Alternatives Weighed**: `Add five fitted interior terms` (rejected: The worst-fold gain is below calibration noise, and held-out pulse error increases.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('study keys',list(s))
for k,v in s.items():
 if not isinstance(v,(np.ndarray,list,dict)) and k not in ('reference_tracer_pulses',): print(k,v)
x=s['sensor_calibration_sweeps']['ref_rosette']; R,F,a=m._chamber(s); S,U=R(a); Z=s['Z_nuisance']; u=np.broadcast_to(s['u_box'],(F.shape[1],)); w=u*s['total_budget']/u.sum(); y=S@w+Z@np.array([.02,-.03,.04]); G=np.array([[1.06,.03,0,-.02],[.02,.94,.04,0],[0,-.03,1.03,.02],[.01,0,-.03,.98]])
def encode(v,nonlinear):
 shape=v.shape; z=v.reshape(-1,4); out=z@G+np.array([.015,-.025,.012,.004]);
 if nonlinear: out+=np.column_stack((.05*z[:,1]*z[:,2],-.04*z[:,0]*z[:,3],.03*z[:,2]**3,.06*z[:,0]*z[:,1]*z[:,2]))
 return out.reshape(shape)
ref=m.solve_batch(dict(s,reference_tracer_pulses=None,y_obs_raw=y,sensor_calibration_sweeps=dict(s['sensor_calibration_sweeps'],raw_rosette=x)))
for nonlinear in (False,True):
 q=m.solve_batch(dict(s, reference_tracer_pulses=None,y_obs_raw=encode(y,nonlinear),sensor_calibration_sweeps=dict(s['sensor_calibration_sweeps'],raw_rosette=encode(x,nonlinear))))
 dw=np.max(abs(ref['weights']-q['weights'])); df=np.max(abs(ref['terminal_field']-q['terminal_field'])); print('blind planted rosette nonlinear',nonlinear,'weight and field deviation',dw,df); assert max(dw,df)<1e-8
q=m.solve_batch(dict(s, reference_tracer_pulses=None)); c=q['kkt_certificate']; w=q['weights']; gap=c['primal_objective']-c['dual_lower_bound']; err=np.max(abs(np.tensordot(w,U,axes=(0,0))-q['terminal_field'])); print('public blind feasibility, gap, field error',abs(w.sum()-s['total_budget']),gap,err); assert abs(gap)<1e-10 and err<1e-12
PY
```
- **Container Exit Code**: `0` (`515 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study keys ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'reference_tracer_pulses', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
Lx 2.4
Ly 2.0
nx 14
ny 12
robin_x 0.880939340203039
robin_y 1.2931872722124154
nuisance_ridge 0.08
source_ridge 0.04
u_box 0.42
total_budget 1.4
blind planted rosette nonlinear False weight and field deviation 1.5265566588595902e-15 9.71445146547012e-17
blind planted rosette nonlinear True weight and field deviation 3.4139358007223564e-15 2.706168622523819e-16
public blind feasibility, gap, field error 0.0 0.0 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b56e70e13bfc7bea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b56e70e1 (directory).
```

### Turn 13 — `submit_completed_task` (`14568 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process schema, calibration, field-ordering, and KKT checks. Held-out tracer results support the row-local stencil, but the routine observations do not reliably identify wall corrections.
- **Alternatives Weighed**: `Infer blind wall corrections from routine observations` (rejected: Weakly constrained fits improved routine cross-validation slightly but worsened independent tracer predictions.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed, terminal-field ordering was verified, and 600 QP stress cases had a maximum primal–dual gap of 1.95e-13. No apparatus archive was available; without tracer pulses, the solver uses the nominal Robin parameters.`

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
