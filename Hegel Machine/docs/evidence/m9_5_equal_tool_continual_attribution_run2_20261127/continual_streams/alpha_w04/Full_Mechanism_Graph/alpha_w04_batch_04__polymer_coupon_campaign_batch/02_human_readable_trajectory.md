# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w04_batch_04__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w04_step_04` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `339,211` (`326,540` / `85,760` / `12,671`) |
| **Total Episode Wall Time (s)** | `368.06s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_10_split_rate_anchored_hereditary_cre`**: [BIFURCATED SUB-REGIME `missing_raw_feature_count >= 1`]: Lab-specific reference alignment removes shared affine measurement shifts, but pooled calibration and cell-mean imputation cannot recover individual coupon drift: the episode recorded endpoint errors up to 1.706 on incomplete coupons and correlations below −0.9996 between endpoint residuals and onset-rate errors in three labs. Accept a local stress calibration when at least three checks and at least half the panel agree within 0.003 of the stress scale and span at least 0.35 of the command scale; otherwise prefer a supported same-instrument archive, with a two-point fit allowed without an archive but unable to certify either check. Propagate both preloaded exponential memories and fit hold increments and onset rates jointly with a soft-L1 pilot and separately MAD-scaled Cauchy refinement rather than pooled L2; reject fits that fail the local normalized Jacobian rank screen. On incomplete rows, subtract the modeled viscoelastic onset rate from the measured onset rate to identify constant drift; on complete rows, retain feature-based drift with a lab correction only when at least eight complete training rows support an identifiable offset and slope. The supplied episode probe reported strain MAE 0.002891 and relative time-constant error 0.030179, but did not establish cross-epoch persistence or unseen-regime extrapolation. (Counter-regime `missing_raw_feature_count` warning: This is an observability split within one sub-yield constitutive law. Complete rows use feature drift plus any supported group correction; incomplete rows require a finite independent onset-rate measurement to identify specimen drift.)
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Aligned and imputed coupon features alone are sufficient to predict the per-coupon drift.` (witness: `The largest endpoint errors, up to 1.706 on the public set, occur on coupons with missing features; in labs 0–2 the endpoint residual and modeled instantaneous-rate error have correlations below −0.9996.`) -> Pivot `Keep the calibrated two-mode recurrence and feature alignment; compare rate fusion and drift-profiled kinetics inside leave-one-lab-out folds.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`13090 ms` LLM)
- **Rationale**: Inspect field shapes, available calibration history, and reusable primitives before choosing how to fit the constitutive model.

```bash
ls -lh /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference panel alignment ridge nuisance profiling' || true; python3 - <<'PY'
import pickle,numpy as np
p='/holdout_task/public_alpha_w04_batch_04.pkl'
d=pickle.load(open(p,'rb'))
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None), (v[:2] if isinstance(v,list) else list(v) if isinstance(v,dict) else v if np.isscalar(v) else np.asarray(v).ravel()[:5]))
PY
```
- **Container Exit Code**: `1` (`579 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18858 ms` LLM)
- **Rationale**: Read the documented dataset member and recover the seven archived basis terms. Inspect current calibration rows alongside the archive to distinguish sparse coverage from EMI outliers before choosing how to fit each lab.
- **Alternatives Weighed**: `Reuse the archived calibration coefficients unchanged` (rejected: The current sweeps must first establish which labs have enough clean leverage to estimate fresh coefficients and which need historical support.)

```bash
grep -n '^def\|^import\|^from ' /tmp/hegel_continual_archive/alpha_w04_batch_03_solver.py; cat /tmp/hegel_continual_archive/alpha_w04_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights; python3 - <<'PY'
import pickle,json,numpy as np
p='/holdout_task/public_alpha_w04_batch_04.pkl'
d=pickle.load(open(p,'rb'))[0]['dataset']
for k,v in d.items():
 print(k, type(v).__name__, getattr(v,'shape',None))
 if isinstance(v,dict):
  for a,b in v.items():
   print(' ',a,type(b).__name__,getattr(b,'shape',None))
 elif np.isscalar(v) or k=='eval_times': print(v)
c=d['calibration_load_cells']
for lab in range(4):
 m=c['lab_id']==lab; print('lab',lab,'checks',np.c_[c['cmd_stress'][m],c['true_stress'][m]].round(5).tolist())
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history preview',str(h)[:2400]); print('manifest',open('/tmp/hegel_continual_archive/archive_manifest.json').read())
PY
```
- **Container Exit Code**: `0` (`552 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5847, -1.720386, -1.480616, -1.678528, 0.237124, -0.254526, -0.266281, 0.02638, 0.483159, 1.042648, 1.803905], 'true_stress': [-1.998026, 0.001924, 1.587906, -0.097953, 0.383708, -1.367023, -0.468064, -0.423969, 0.49774, 1.659167, -0.715484, -0.580163, 0.686967, -0.502652, -1.4597, 0.841132, -0.5333, -1.588631, 0.262934, 0.637014, 1.737986, -1.669004, 1.147616, -1.123548, -1.481691, -1.455723, 0.375249, -0.711939, 0.607519, 0.385859, -1.720363, -1.48052, -1.678405, 0.23707, -0.254424, -0.266194, 0.026309, 0.48317, 1.042617, 1.803978]}, {'episode_short_name': 'alpha_w04_batch_02', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [2.303968, 1.773672, 1.87774, 1.106582, 1.179565, 1.608769, 1.057076, 1.748507, -2.006956, 1.404948, 1.821261, -1.615241, -0.479061, 
manifest [
  {
    "episode_short_name": "alpha_w04_s00",
    "task_id": "alpha_w04_s00__coupon_creep_baseline",
    "entrypoint": "solve_alpha_w04_s00",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "viscoplastic_weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/alpha_w04_s00_solver.py"
  },
  {
    "episode_short_name": "alpha_w04_batch_01",
    "task_id": "alpha_w04_batch_01__polymer_coupon_campaign_batch",
    "entrypoint": "solve_alpha_w04_batch_01",
    "output_schema_keys": [
      "estimated_taus",
      "hazard_crossing_times",
      "predicted_strain",
      "secant_compliance"
    ],
    "solver_path": "/tmp/hegel_continual_archive/alpha_w04_batch_01_solver.py"
  },
  {
    "episode_short_name": "alpha_w04_batch_02",
    "task_id": "alpha_w04_batch_02__polymer_coupon_campaign_batch",
    "entrypoint": "solve_alpha_w04_batch_02",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "viscoplastic_weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/alpha_w04_batch_02_solver.py"
  },
  {
    "episode_short_name": "alpha_w04_batch_03",
    "task_id": "alpha_w04_batch_03__polymer_coupon_campaign_batch",
    "entrypoint": "solve_alpha_w04_batch_03",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "viscoplastic_weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/alpha_w04_batch_03_solver.py"
  }
]
```

### Turn 3 — `execute_in_remote_docker` (`22647 ms` LLM)
- **Rationale**: Decode the archived solver to recover the exact hereditary basis and inspect compact per-lab residuals to determine when current data should override the archive.
- **Alternatives Weighed**: `Fit ordinary least squares to all current load-cell rows` (rejected: Calibration residuals must first establish whether any lab has EMI outliers or insufficient stress coverage.)

```bash
python3 - <<'PY'
import ast,base64,pathlib,pickle,json,numpy as np
root=pathlib.Path('/tmp/hegel_continual_archive')
for batch in ['02','03']:
 p=root/f'alpha_w04_batch_{batch}_solver.py'; s=p.read_text(); node=next((n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SRC_B64' for t in n.targets)),None)
 if node: s=base64.b64decode(ast.literal_eval(node.value)).decode()
 pathlib.Path(f'/tmp/w04_archive_{batch}.py').write_text(s)
s=pathlib.Path('/tmp/w04_archive_03.py').read_text(); print(s[s.index('def _archive_panels'):]);
d=pickle.load(open('/holdout_task/public_alpha_w04_batch_04.pkl','rb'))[0]['dataset']
print('TOP',[(k,getattr(v,'shape',list(v) if isinstance(v,dict) else v)) for k,v in d.items()]);
for k in ('train','test'):
 for a,b in d[k].items():
  v=np.asarray(b); print(k,a,v.shape, 'nan',np.isnan(v).sum() if np.issubdtype(v.dtype,np.number) else '-', 'range', (np.nanmin(v),np.nanmax(v)) if np.issubdtype(v.dtype,np.number) else '')
h=json.load(open(root/'archive_calibration_history.json')); print('history keys',list(h) if isinstance(h,dict) else len(h)); panels=next(v for v in h.values() if isinstance(v,list)) if isinstance(h,dict) else h
for c in [*panels,d['calibration_load_cells']]:
 print(c.get('episode_short_name','current'))
 for lab in range(4):
  m=np.asarray(c['lab_id'])==lab; x=np.asarray(c['cmd_stress'])[m]; y=np.asarray(c['true_stress'])[m]; w=np.polyfit(x,y,1) if len(x)>1 and np.ptp(x)>0 else np.array([np.nan,np.nan]); print(lab,len(x),w.round(6),np.sort(abs(y-np.polyval(w,x))).round(6), 'current rows',np.c_[x,y].round(5).tolist() if c is d['calibration_load_cells'] else '')
PY
```
- **Container Exit Code**: `0` (`554 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
04] current rows 
alpha_w04_batch_02
0 10 [1.223723 0.139491] [3.20e-05 3.50e-05 6.60e-05 7.40e-05 8.10e-05 1.30e-04 1.36e-04 1.53e-04
 2.67e-04 2.88e-04] current rows 
1 10 [ 0.707861 -0.088688] [3.00e-06 3.80e-05 5.90e-05 6.00e-05 7.80e-05 8.10e-05 1.40e-04 1.46e-04
 1.52e-04 2.02e-04] current rows 
2 10 [1.199251 0.084814] [1.7e-05 2.6e-05 3.3e-05 4.0e-05 4.4e-05 4.8e-05 5.2e-05 5.4e-05 7.0e-05
 8.1e-05] current rows 
3 10 [ 0.731626 -0.140979] [1.00e-06 4.00e-06 2.00e-05 2.80e-05 3.00e-05 3.60e-05 4.70e-05 8.70e-05
 1.06e-04 1.27e-04] current rows 
alpha_w04_batch_03
0 10 [9.99997e-01 2.50000e-05] [2.90e-05 4.80e-05 7.90e-05 8.00e-05 1.04e-04 1.17e-04 1.31e-04 1.46e-04
 1.69e-04 2.16e-04] current rows 
1 10 [1.000021e+00 2.600000e-05] [2.0e-06 2.1e-05 2.5e-05 2.7e-05 3.3e-05 4.3e-05 5.4e-05 5.8e-05 6.2e-05
 9.3e-05] current rows 
2 10 [1.000008e+00 1.000000e-06] [3.00e-06 6.00e-06 9.00e-06 1.20e-05 3.90e-05 5.20e-05 5.40e-05 8.60e-05
 1.29e-04 1.84e-04] current rows 
3 10 [1.000027e+00 3.000000e-06] [5.00e-06 2.30e-05 2.30e-05 5.00e-05 7.80e-05 9.40e-05 1.09e-04 1.16e-04
 1.34e-04 2.13e-04] current rows 
current
0 10 [1.223703 0.139498] [2.70e-05 5.90e-05 6.50e-05 9.40e-05 9.90e-05 1.19e-04 1.36e-04 1.37e-04
 1.49e-04 1.87e-04] current rows [[-1.96784, -2.26841], [-0.99697, -1.08047], [2.3611, 3.02867], [-2.0965, -2.42613], [1.0699, 1.44884], [-0.95587, -1.0303], [-2.11542, -2.44908], [-0.27571, -0.19783], [1.04126, 1.41383], [-0.69699, -0.7136]]
1 10 [ 0.70787  -0.088732] [4.00e-05 4.50e-05 6.30e-05 6.30e-05 6.30e-05 6.60e-05 6.80e-05 7.70e-05
 1.13e-04 1.17e-04] current rows [[-1.14557, -0.89958], [-1.21078, -0.94574], [1.72841, 1.13469], [-1.3218, -1.02432], [-0.93195, -0.74838], [-0.64965, -0.54871], [-1.45074, -1.11578], [-0.05144, -0.12521], [1.97837, 1.31174], [1.37573, 0.88517]]
2 10 [1.199312 0.08478 ] [2.40e-05 2.50e-05 2.60e-05 4.10e-05 4.50e-05 5.10e-05 7.80e-05 9.20e-05
 1.32e-04 2.33e-04] current rows [[-1.97988, -2.28968], [0.48096, 0.66158], [1.62233, 2.03041], [1.1055, 1.41085], [0.25237, 0.38736], [0.84096, 1.09322], [0.77874, 1.01879], [1.33514, 1.68596], [2.21898, 2.74605], [2.15133, 2.66492]]
3 10 [ 0.731603 -0.141   ] [5.00e-06 7.00e-06 3.40e-05 5.00e-05 6.60e-05 7.90e-05 1.10e-04 1.14e-04
 1.29e-04 1.64e-04] current rows [[-0.3841, -0.42185], [1.65124, 1.06712], [0.14828, -0.03257], [-1.71256, -1.39391], [0.88155, 0.50398], [-0.01818, -0.15419], [2.20383, 1.47125], [-0.71221, -0.66218], [0.12238, -0.05147], [-0.23774, -0.31505]]
```

### Turn 4 — `execute_in_remote_docker` (`15458 ms` LLM)
- **Rationale**: The previous output truncated the archived feature and response definitions. Read just that section and the current field shapes: discrepancies would change the basis, alignment, or strain/rate stacking before implementation.
- **Alternatives Weighed**: `Pool all historical sweeps with current checks` (rejected: The archive contains both near-identity and shifted calibrations. All four current sweeps are clean and well-spread, so pooling would introduce bias.)

```bash
grep -n '^def ' /tmp/w04_archive_03.py /tmp/w04_archive_02.py; sed -n '/^def _align/,/^def solve/p' /tmp/w04_archive_03.py; python3 - <<'PY'
import pickle,numpy as np
r=pickle.load(open('/holdout_task/public_alpha_w04_batch_04.pkl','rb'))[0]; d=r['dataset']; print('dataset fields',list(d),'public reference fields',list(r)); print('t',d['eval_times'],'yield',d.get('yield_stress'))
for k in ('train','test'):
 print(k, [(a,np.shape(b),int(np.isnan(b).sum()) if isinstance(b,np.ndarray) and np.issubdtype(b.dtype,np.inexact) else None) for a,b in d[k].items()])
PY
```
- **Container Exit Code**: `0` (`493 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    delta = hold[:, None] - np.einsum('ns,sm->nm', hist, memory)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    rate = np.column_stack((delta / taus, drift))
    curves = np.concatenate((delta[:, None, :] * (-np.expm1(-times[:, None] / taus)),
                             drift[:, None, :] * times[None, :, None]), axis=2)
    return curves, rate


def _fit(dt, times, hist, hold, aligned, curves, rate):
    n, k = curves.shape
    target = np.r_[curves.ravel(), rate]
    def build(taus):
        a, b = _design(taus, dt, times, hist, hold, aligned)
        return np.vstack((a.reshape(-1, 7), b))
    def unpack(p):
        return np.array([p[0], p[0] + p[1] * (8. - p[0])]), p[2:]
    lo = np.r_[0.20, 1e-5, np.full(7, -np.inf)]
    hi = np.r_[8. - 1e-5, 1., np.full(7, np.inf)]
    def optimize(p, scales, loss):
        def fun(q):
            t, w = unpack(q)
            return (np.einsum('ij,j->i', build(t), w) - target) / scales
        return least_squares(fun, p, bounds=(lo, hi), loss=loss, x_scale='jac', max_nfev=160)
    pilots = []
    for a, b in ((0.35, 1.3), (0.60, 3.0), (1.15, 5.4)):
        m = build(np.array([a, b]))
        w = np.linalg.lstsq(m, target, rcond=None)[0]
        pilots.append(optimize(np.r_[a, (b-a)/(8.-a), w], 0.05, 'soft_l1'))
    pilot = min(pilots, key=lambda o: o.cost)
    raw = pilot.fun * 0.05
    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))
    scales = np.r_[np.full(n*k, mad(raw[:n*k])), np.full(n, mad(raw[n*k:]))]
    fit = optimize(pilot.x, 2.385 * scales, 'cauchy')
    return unpack(fit.x)


def solve_alpha_w04_s00(dataset: dict) -> dict:
dataset fields ['past_dt', 'eval_times', 'yield_stress', 'calibration_load_cells', 'train', 'test'] public reference fields ['dataset', 'gt']
t [0.25 0.8  1.35 1.9  2.45 3.   3.55 4.1  4.65 5.2 ] yield 1.52
train [('lab_id', (200,), None), ('stratum', (200,), None), ('coupon_features_raw', (200, 4), 23), ('ref_coupons', (200, 32), 280), ('past_stress_cmd', (200, 18), 0), ('strain_0', (200,), 0), ('stress_hold_cmd', (200,), 0), ('instant_rate_obs', (200,), 0), ('strain_curves_obs', (200, 10), 0)]
test [('lab_id', (80,), None), ('stratum', (80,), None), ('coupon_features_raw', (80, 4), 8), ('ref_coupons', (80, 32), 104), ('past_stress_cmd', (80, 18), 0), ('strain_0', (80,), 0), ('stress_hold_cmd', (80,), 0), ('instant_rate_obs', (80,), 0), ('residual_plastic_state', (80,), 0)]
```

### Turn 5 — `execute_in_remote_docker` (`63521 ms` LLM)
- **Rationale**: Build the solver using the archived reference alignment and Prony basis, with the specified overstress terms and their exact initial derivatives. The isolated schema run will catch entrypoint, shape, and finiteness failures before testing numerical accuracy by lab and yield regime.
- **Alternatives Weighed**: `Fit a response surrogate directly to commanded stress` (rejected: It would confound lab calibration with the physical yield threshold and discard the exact hereditary basis.)

```bash
sed -n '186,240p' /tmp/w04_archive_03.py; python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('archive structure:',list(h) if isinstance(h,dict) else [r.get('episode_short_name') for r in h])
PY
cat > /holdout_task/alpha_w04_batch_04_solver.py <<'PY'
"""Fit the calibrated two-branch hereditary creep law with yield activation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _rows(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    use = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _consensus(x, y):
    i, j = np.triu_indices(len(x), 1)
    spread = x[j] - x[i]
    use = np.abs(spread) >= 0.12
    i, j, spread = i[use], j[use], spread[use]
    if not len(i):
        return None
    gain = (y[j] - y[i]) / spread
    bias = y[i] - gain * x[i]
    error = np.abs(y - gain[:, None] * x - bias[:, None])
    gate = 0.0015
    count = (error <= gate).sum(axis=1)
    loss = np.minimum((error / gate)**2, 1).sum(axis=1)
    keep = error[np.lexsort((loss, -count))[0]] <= gate
    if keep.sum() >= max(3, (len(x) + 1)//2) and np.ptp(x[keep]) >= 0.30:
        return _line(x[keep], y[keep])
    return None


def _historical_fits():
    if not _ARCHIVE.exists():
        return []
    archive = json.loads(_ARCHIVE.read_text())
    if isinstance(archive, dict):
        archive = next(v for v in archive.values() if isinstance(v, list))
    return [[_consensus(*_rows(p, lab)) for lab in range(4)] for p in reversed(archive)]


def _calibrate(panel):
    rows = [_rows(panel, lab) for lab in range(4)]
    fits = [_consensus(*r) for r in rows]
    if all(f is not None for f in fits):
        return np.array(fits)
    history = _historical_fits()
    for lab, fit in enumerate(fits.copy()):
        if fit is not None:
            continue
        x, y = rows[lab]
        peer = lab ^ 1
        xp, yp = rows[peer]
        options = []
        for old in history:
            if old[lab] is None:
                continue
            err = np.abs(y - old[lab][0]*x - old[lab][1])
            close = err < 0.004
            peer_err = (np.abs(yp - old[peer][0]*xp - old[peer][1])
                        if old[peer] is not None else np.full(len(xp), 0.004))
            peer_matches = int((peer_err < 0.004).sum())
            loss = float(np.minimum(err, 0.004).sum() + np.minimum(peer_err, 0.004).sum())
            options.append(((-int(close.sum()), -peer_matches, loss), old[lab], close))
        if options:
            score, prior, close = min(options, key=lambda item: item[0])
        else:
            score, prior, close = (0, 0, 0), np.array([1., 0.]), np.zeros(len(x), bool)
        if len(x) == 2 and np.ptp(x) >= 0.30 and not close.any():
            fits[lab] = _line(x, y)
        else:
            fits[lab] = prior
    return np.array(fits)
PY
sed -n '/^def _align(/,/^def _design(/{ /^def _design(/!p; }' /tmp/w04_archive_03.py >> /holdout_task/alpha_w04_batch_04_solver.py
cat >> /holdout_task/alpha_w04_batch_04_solver.py <<'PY'
def _inputs(split, affine, aligned, dt, yield_stress):
    hist, hold = _stress(split, affine)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    excess = np.maximum(np.abs(hold) - yield_stress, 0.)
    ov_hold = np.sign(hold) * excess**2
    work = np.maximum(np.abs(hist) - yield_stress, 0.)**2 @ dt
    return hist, hold, drift, ov_hold, work*hold


def _design(taus, dt, times, data, active):
    hist, hold, drift, ov, work_hold = data
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    delta = hold[:, None] - hist @ memory
    relaxation = -np.expm1(-times[:, None] / taus)
    a = [delta[:, None, :]*relaxation, drift[:, None, :]*times[None, :, None]]
    b = [delta / taus, drift]
    if active[0]:
        a.append((ov[:, None]*times)[:, :, None])
        b.append(ov[:, None])
    if active[1]:
        a.append((work_hold[:, None]*relaxation[:, 0])[:, :, None])
        b.append((work_hold / taus[0])[:, None])
    return np.concatenate(a, axis=2), np.concatenate(b, axis=1)


def _fit(dt, times, data, curves, rate, active):
    n, k = curves.shape
    cols = 7 + sum(active)
    target = np.r_[curves.ravel(), rate]

    def unpack(p):
        return np.array([p[0], p[0] + p[1]*(24.99 - p[0])])

    def build(t):
        a, b = _design(t, dt, times, data, active)
        return np.vstack((a.reshape(-1, cols), b))

    bounds = ([0.05001, 1e-6], [24.98, 1.])
    def profile(p, scale):
        m = build(unpack(p))/scale[:, None]
        y = target/scale
        w = np.linalg.lstsq(m, y, rcond=None)[0]
        return np.einsum('ij,j->i', m, w) - y, w

    scale = np.ones_like(target)
    pilots = [least_squares(lambda p: profile(p, scale)[0], [a, (b-a)/(24.99-a)],
                           bounds=bounds, max_nfev=100)
              for a, b in ((0.35, 1.5), (0.65, 3.0), (1.4, 6.0))]
    pilot = min(pilots, key=lambda fit: fit.cost)
    def mad(r):
        return max(1e-7, 1.4826*np.median(np.abs(r-np.median(r))))
    scale = 2.385*np.r_[np.full(n*k, mad(pilot.fun[:n*k])), np.full(n, mad(pilot.fun[n*k:]))]
    _, w = profile(pilot.x, scale)
    def fun(p):
        return (np.einsum('ij,j->i', build(unpack(p)), p[2:]) - target)/scale
    fit = least_squares(fun, np.r_[pilot.x, w],
                        bounds=(np.r_[bounds[0], np.full(cols, -np.inf)],
                                np.r_[bounds[1], np.full(cols, np.inf)]),
                        x_scale='jac', loss='cauchy', max_nfev=120)
    return unpack(fit.x), fit.x[2:]


def solve_alpha_w04_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    affine = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    a = _inputs(train, affine, x_train, dt, dataset['yield_stress'])
    b = _inputs(test, affine, x_test, dt, dataset['yield_stress'])
    active = tuple(bool(np.any(c)) for c in a[-2:])
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(dt, times, a, curves, train['instant_rate_obs'], active)
    vp = np.zeros(2)
    vp[list(active)] = weights[7:]
    curves, rate = _design(taus, dt, times, b, active)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': vp,
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`556 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w04_batch_04_solver.py']`
- **Stdout (tail)**:
```text
def solve_alpha_w04_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    affine = _calibrate(dataset['calibration_load_cells'])
    x, xt = _align(train, test)
    hist, hold = _stress(train, affine)
    taus, theta = _fit(dt, times, hist, hold, x,
                       train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    at, bt = _design(taus, dt, times, *_stress(test, affine), xt)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + at @ theta,
            'initial_rate_pred': bt @ theta}
archive structure: ['alpha_w04_s00', 'alpha_w04_batch_01', 'alpha_w04_batch_02', 'alpha_w04_batch_03']
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ceba74bbf1b17d1b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ceba74bb (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33798 ms` LLM)
- **Rationale**: Measure strain and initial-rate error separately in leave-one-lab-out folds, stratified by hold and history yield activation. If a slice fails, use the pattern to distinguish a constitutive or alignment error from a calibration fallback error.
- **Alternatives Weighed**: `Submit after the public schema check` (rejected: The schema check cannot detect response errors concentrated in a lab or yield regime, or the selection of the wrong historical calibration under EMI.)

```bash
cat > /tmp/w04_validate.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for record in pickle.load(open(root+'public_alpha_w04_batch_04.pkl','rb')):
    d = record['dataset']
    o = m.solve_alpha_w04_batch_04(d)
    print('fit', {k: np.round(v,7).tolist() for k,v in o.items() if v.size < 10})
    gt = record['gt']
    print('public reference fields', [(k,np.shape(v)) for k,v in gt.items()] if isinstance(gt,dict) else type(gt).__name__)
    if isinstance(gt,dict):
        for k,v in o.items():
            if k in gt and np.shape(gt[k]) == v.shape:
                e = np.asarray(v)-np.asarray(gt[k]); print('public reference',k,'RMSE',np.sqrt(np.mean(e**2)),'max',np.max(np.abs(e)))
    tr=d['train']; labs=tr['lab_id']; n=len(labs); obs=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; dt=d['past_dt']; t=d['eval_times']; y=d['yield_stress']
    c=d['calibration_load_cells']; affine=m._calibrate(c)
    x=m._align(tr,d['test'])[0]; a=m._inputs(tr,affine,x,dt,y)
    cv=np.empty_like(obs); rv=np.empty_like(rate)
    for lab in range(4):
        keep=labs!=lab; take=~keep
        fold=tuple(z[keep] for z in a); active=tuple(bool(np.any(z)) for z in fold[-2:]); tt,ww=m._fit(dt,t,fold,obs[keep]-tr['strain_0'][keep,None],rate[keep],active)
        ca,rb=m._design(tt,dt,t,tuple(z[take] for z in a),active)
        cv[take]=tr['strain_0'][take,None]+ca@ww; rv[take]=rb@ww
        print('fold',lab,'taus',tt.round(5),'plastic',ww[7:].round(6))
    categories=(np.abs(a[3])>0).astype(int)+2*(np.max(np.abs(a[0]),axis=1)>y)
    def slice_scores(mask):
        return tuple(float(np.sqrt(np.mean(e[mask]**2))) for e in (cv-obs,rv-rate))
    metrics=[(lab,rg,int(mask.sum()),*slice_scores(mask)) for lab in range(4) for rg in range(4) if (mask:=(labs==lab)&(categories==rg)).any()]
    print('lab x yield regime LOLO RMSE [lab, regime, n, curve, rate]',metrics)
    print('worst LOLO curve,rate',max(s[3] for s in metrics),max(s[4] for s in metrics))
    for lab in range(4):
        rows=np.flatnonzero(c['lab_id']==lab); xcal=c['cmd_stress'][rows]; keep=rows[[xcal.argmin(),xcal.argmax()]]
        mask=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),keep)
        sparse={k:v[mask].copy() for k,v in c.items()}; sparse['true_stress'][np.flatnonzero(sparse['lab_id']==lab)[0]]+=0.16
        recovered=m._calibrate(sparse)
        dirty={k:v.copy() for k,v in c.items()}; dirty['true_stress'][rows[:2]] += [0.16,-0.24]
        robust=m._calibrate(dirty)
        print('calibration stress test',lab,'sparse + EMI',np.max(np.abs(recovered-affine)), 'dense + EMI',np.max(np.abs(robust-affine)))
    sub=dict(d,yield_stress=1e6); so=m.solve_alpha_w04_batch_04(sub)
    assert np.array_equal(so['viscoplastic_weights'],np.zeros(2))
    print('sub-yield branch weights exactly zero')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/w04_validate.py
```
- **Container Exit Code**: `0` (`677 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit {'estimated_taus': [0.5683319, 3.8277706], 'hereditary_weights': [0.6872477, 0.5702721, 0.0796342, 0.1485239, -0.1283926, 0.1087959, -0.0889243], 'viscoplastic_weights': [0.2488719, 0.1701068]}
public reference fields [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_cross', (80,)), ('te_secant', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
fold 0 taus [0.56813 3.79442] plastic [0.249808 0.170075]
fold 1 taus [0.56798 3.86072] plastic [0.247395 0.169993]
fold 2 taus [0.56815 3.84017] plastic [0.249012 0.170052]
fold 3 taus [0.56884 3.83255] plastic [0.248446 0.170217]
lab x yield regime LOLO RMSE [lab, regime, n, curve, rate] [(0, 0, 16, 0.24787538111989588, 0.008206631316982745), (0, 3, 29, 0.33254393885947214, 0.052167009804758024), (1, 0, 31, 0.33306372273446483, 0.03702457822174857), (1, 3, 26, 0.3380621929020056, 0.039701377944306976), (2, 0, 28, 0.37632898224720324, 0.08009480207248121), (2, 3, 19, 0.3284873237289185, 0.04873260863356208), (3, 0, 25, 0.27635424202568376, 0.030217269033073254), (3, 3, 26, 0.2745014537163385, 0.01572435608306737)]
worst LOLO curve,rate 0.37632898224720324 0.08009480207248121
calibration stress test 0 sparse + EMI 2.027907815249641e-05 dense + EMI 1.8720929784804552e-05
calibration stress test 1 sparse + EMI 4.3906987656966434e-05 dense + EMI 1.743412054452942e-05
calibration stress test 2 sparse + EMI 6.116478983742013e-05 dense + EMI 4.553396101522611e-05
calibration stress test 3 sparse + EMI 2.3431994566558423e-05 dense + EMI 2.844739223983872e-05
sub-yield branch weights exactly zero

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ceba74bbf1b17d1b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ceba74bb (directory).
```

### Turn 7 — `execute_in_remote_docker` (`29300 ms` LLM)
- **Rationale**: The previous run discovered the public reference names but did not score them. Compare each output to its clean reference and test the two mixed-yield regimes absent from the public folds. A clean error concentrated in missing-feature rows would point to alignment, while a mixed-regime failure would require a basis correction.
- **Alternatives Weighed**: `Change the fitter based on the noisy leave-one-lab-out strain error` (rejected: The public clean references can distinguish model error from observation noise before any edit.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_w04_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for rec in pickle.load(open(p+'public_alpha_w04_batch_04.pkl','rb')):
 d,g=rec['dataset'],rec['gt']; out=m.solve_alpha_w04_batch_04(d); te=d['test']; dt=d['past_dt']; ts=d['eval_times']; ys=d['yield_stress']; cal=m._calibrate(d['calibration_load_cells']); aligned=m._align(d['train'],te)[1]; a=m._inputs(te,cal,aligned,dt,ys)
 for k,gk in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
  print(k,'reference',g[gk],'absolute error',np.abs(out[k]-g[gk]))
 for k,x in [('gain',cal[:,0]-g['stress_gains']),('bias',cal[:,1]-g['stress_biases'])]: print(k,'max abs error',np.max(np.abs(x)))
 errs=[out['predicted_strain']-g['te_clean'],out['initial_rate_pred']-g['te_true_rate0']]; lab=te['lab_id']; regime=(np.abs(a[1])>ys).astype(int)+2*(np.max(np.abs(a[0]),axis=1)>ys); missing=np.isnan(te['coupon_features_raw']).any(axis=1)
 for name,mask in [('all',np.ones(len(lab),bool)),('complete',~missing),('missing',missing)]+[(f'lab={i} regime={j}',(lab==i)&(regime==j)) for i in range(4) for j in range(4)]:
  if mask.any(): print(name,mask.sum(),'clean curve/rate RMSE',*[round(float(np.sqrt(np.mean(e[mask]**2))),7) for e in errs])
 split={k:v[:4].copy() for k,v in te.items()}; split['lab_id'][:]=0; split['past_stress_cmd'][:]=0; split['past_stress_cmd'][2:,-1]=(2*ys-cal[0,1])/cal[0,0]; split['stress_hold_cmd'][:]=(0.5*ys-cal[0,1])/cal[0,0]; split['stress_hold_cmd'][[1,3]]=(-2*ys-cal[0,1])/cal[0,0]; z=m._inputs(split,cal,aligned[:4],dt,ys); expected=np.array([[0,0],[-1,0],[0,1],[-1,-1]]); assert np.array_equal(np.sign(np.column_stack(z[-2:])),expected); eps=1e-7
 design,rate=m._design(out['estimated_taus'],dt,np.array([0.,eps]),z,(True,True)); assert np.all(design[:,0,:]==0); np.testing.assert_allclose(design[:,1,:]/eps,rate,rtol=2e-6,atol=1e-8); print('all four yield combinations and initial derivatives verified')
PY
```
- **Container Exit Code**: `0` (`479 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus reference [0.568 3.821] absolute error [0.00033192 0.00677058]
hereditary_weights reference [ 0.689  0.566  0.08   0.15  -0.13   0.11  -0.09 ] absolute error [0.00175226 0.00427207 0.00036577 0.00147606 0.00160735 0.00120413
 0.00107573]
viscoplastic_weights reference [0.25 0.17] absolute error [0.00112809 0.00010679]
gain max abs error 3.0460223156225652e-05
bias max abs error 3.2228761208286105e-05
all 80 clean curve/rate RMSE 0.1596245 0.0505804
complete 74 clean curve/rate RMSE 0.0047495 0.0033765
missing 6 clean curve/rate RMSE 0.5826274 0.1843123
lab=0 regime=0 12 clean curve/rate RMSE 0.2421019 0.0769372
lab=0 regime=3 7 clean curve/rate RMSE 0.3748454 0.1188261
lab=1 regime=0 5 clean curve/rate RMSE 0.0052081 0.0043063
lab=1 regime=3 9 clean curve/rate RMSE 0.0056487 0.0026543
lab=2 regime=0 6 clean curve/rate RMSE 0.2366952 0.0734937
lab=2 regime=3 12 clean curve/rate RMSE 0.0046998 0.0020527
lab=3 regime=0 17 clean curve/rate RMSE 0.0292038 0.0112243
lab=3 regime=3 12 clean curve/rate RMSE 0.0034177 0.001914
all four yield combinations and initial derivatives verified

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ceba74bbf1b17d1b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ceba74bb (directory).
```

### Turn 8 — `execute_in_remote_docker` (`26661 ms` LLM)
- **Rationale**: Use the supplied clean references only for diagnosis. If the strain discrepancy is the initial-rate discrepancy times elapsed time, the failure is confined to coupon drift. Compare the implied missing values with the prescribed cell means and quantify observed-rate noise before deciding whether any correction is justified by the contract.
- **Alternatives Weighed**: `Refit the Prony and plastic terms to reduce missing-row error` (rejected: The parameters and complete-row predictions already agree closely with the public references; the error is concentrated in rows with missing coupon measurements.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; s=importlib.util.spec_from_file_location('solver',p+'alpha_w04_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for rec in pickle.load(open(p+'public_alpha_w04_batch_04.pkl','rb')):
 d,g=rec['dataset'],rec['gt']; tr,te=d['train'],d['test']; dt,t,y=d['past_dt'],d['eval_times'],d['yield_stress']; xt=m._align(tr,te)[1]; a=m._inputs(te,m._calibrate(d['calibration_load_cells']),xt,dt,y); a=(g['te_hist_true'],g['te_hold_true'],*a[2:]); D,R=m._design(g['taus'],dt,t,a,(True,True)); w=np.r_[g['theta'],g['vp_theta']]; model=R@w; clean=g['te_true_rate0']; obs=te['instant_rate_obs']; e=clean-model; missing=np.isnan(te['coupon_features_raw']); rows=missing.any(axis=1); print('test fields',list(te)); print('row lab stratum missing-cols aligned-imputes true-minus-model rateobs-minus-clean effective-impute-if-single');
 for i in np.flatnonzero(rows):
  q=np.flatnonzero(missing[i]); v=xt[i,q]; inferred=v+e[i]/g['theta'][q+3] if len(q)==1 else []; print(i,te['lab_id'][i],te['stratum'][i],q,v.round(4),round(e[i],5),round(obs[i]-clean[i],5),np.round(inferred,4))
 pure=g['te_clean']-te['strain_0'][:,None]-D@w; print('largest discrepancy after subtracting rate discrepancy times t',np.max(abs(pure-e[:,None]*t),axis=1).max()); print('rate noise rmse missing/complete',*[np.sqrt(np.mean((obs[q]-clean[q])**2)) for q in (rows,~rows)]); print('observed-rate error quantiles',np.quantile(abs(obs-clean),[.5,.8,.9,.95,1])); print('feature values by missing cell: train/test finite means, counts')
 for i,j in zip(*np.where(missing)):
  vals=[]
  for split in (tr,te):
   x=split['coupon_features_raw'][(split['lab_id']==te['lab_id'][i])&(split['stratum']==te['stratum'][i]),j]; vals.append((round(np.nanmean(x),4),np.isfinite(x).sum()))
  print(i,j,vals)
PY
```
- **Container Exit Code**: `0` (`411 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
test fields ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'strain_0', 'stress_hold_cmd', 'instant_rate_obs', 'residual_plastic_state']
row lab stratum missing-cols aligned-imputes true-minus-model rateobs-minus-clean effective-impute-if-single
12 0 1 [1] [0.5076] 0.3141 0.01619 [-1.9086]
43 0 2 [1] [0.4871] -0.26512 -0.00992 [2.5265]
52 3 0 [0] [-0.0232] 0.00847 0.0117 [0.0333]
53 2 2 [3] [0.3577] 0.17947 -0.00891 [-1.6364]
60 3 1 [1 2 3] [-0.0258  0.1103  0.3672] 0.03728 0.00218 []
62 0 1 [2] [-0.3774] -0.00889 0.00901 [-0.4583]
largest discrepancy after subtracting rate discrepancy times t 0.008356238036643492
rate noise rmse missing/complete 0.010507974658476707 0.009330869592123814
observed-rate error quantiles [0.00755998 0.01130851 0.01399889 0.01616848 0.02704863]
feature values by missing cell: train/test finite means, counts
12 1 [(-0.2771, 14), (-0.2731, 6)]
43 1 [(-0.2991, 9), (-0.2827, 6)]
52 0 [(0.7318, 15), (0.7645, 9)]
53 3 [(0.1459, 23), (-0.0476, 7)]
60 1 [(0.9787, 15), (0.3434, 9)]
60 2 [(0.8067, 15), (1.1416, 9)]
60 3 [(0.6112, 15), (2.4329, 9)]
62 2 [(-1.0762, 14), (-0.8073, 6)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ceba74bbf1b17d1b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ceba74bb (directory).
```

### Turn 9 — `execute_in_remote_docker` (`44991 ms` LLM)
- **Rationale**: The correction uses only an input measurement and changes the linear drift consistently in strain and initial rate. Verify that it improves the missing slice without regressing complete rows, fitted weights, or the yield and calibration checks.
- **Alternatives Weighed**: `Use the observed rate for every coupon` (rejected: Complete-row model rates already have lower clean RMSE than the observed rates.)
- **ATMS NOGOOD DECLARED**: Refuted `Cell-balanced means alone adequately recover individual coupon drift when a feature is missing.` | Witness: `Clean initial-rate discrepancies on incomplete coupons reached 0.3141 and tracked the strain discrepancy linearly in time; their observed rates differed from clean rates by only 0.01051 RMSE.` | Blanket Diagnosis: `The residual is concentrated in the additive coupon drift, not the calibrated stress, Prony branches, or yield activation.` | Sublated Pivot: `Keep the required cell-balanced imputation and fitted mechanical response, and use the supplied coupon rate to correct only the unresolved scalar drift.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_w04_batch_04_solver.py')
s = p.read_text()
old = """    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': vp,
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rate @ weights}
"""
new = """    strain = test['strain_0'][:, None] + curves @ weights
    initial_rate = rate @ weights
    # Cell means keep the aligned feature design defined, but cannot recover
    # a missing coupon's individual drift. Its measured rate identifies that
    # additive term once the fitted instantaneous mechanical response is fixed.
    observed_rate = np.asarray(test['instant_rate_obs'])
    use_rate = (np.isnan(test['coupon_features_raw']).any(axis=1)
                & np.isfinite(observed_rate))
    drift_residual = observed_rate[use_rate] - initial_rate[use_rate]
    strain[use_rate] += drift_residual[:, None] * times
    initial_rate[use_rate] += drift_residual
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': vp, 'predicted_strain': strain,
            'initial_rate_pred': initial_rate}
"""
assert old in s
p.write_text(s.replace(old, new))
PY
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/w04_validate.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', p+'alpha_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
for rec in pickle.load(open(p+'public_alpha_w04_batch_04.pkl','rb')):
    d, g = rec['dataset'], rec['gt']; te = d['test']
    out = m.solve_alpha_w04_batch_04(d)
    lab = te['lab_id']; ys = d['yield_stress']
    missing = np.isnan(te['coupon_features_raw']).any(axis=1)
    regime = (abs(g['te_hold_true']) > ys).astype(int) + 2*(abs(g['te_hist_true']).max(axis=1) > ys)
    e = out['predicted_strain'] - g['te_clean']; r = out['initial_rate_pred'] - g['te_true_rate0']
    def rmse(mask): return np.sqrt(np.mean(e[mask]**2)), np.sqrt(np.mean(r[mask]**2))
    for name, mask in [('pooled', np.ones(len(lab), bool)), ('complete', ~missing), ('missing', missing)]:
        print(name, int(mask.sum()), rmse(mask))
    slices = [(i,j,*rmse(mask)) for i in range(4) for j in range(4) if (mask := (lab==i)&(regime==j)).any()]
    print('lab × regime clean RMSE', slices)
    print('worst clean lab × regime strain/rate', np.max(np.array(slices)[:,2:],axis=0))
    assert np.all(np.array(rmse(missing)) < [0.06, 0.02])
    assert np.all(np.array(rmse(~missing)) < [0.005, 0.004])
    for key, ref, limit in [('estimated_taus','taus',0.008), ('hereditary_weights','theta',0.005), ('viscoplastic_weights','vp_theta',0.002)]:
        np.testing.assert_allclose(out[key],g[ref],rtol=0,atol=limit)
    a = m._calibrate(d['calibration_load_cells']); dt=d['past_dt']
    hist=np.zeros((4,len(dt))); hist[2:,-1]=2*ys
    hold=ys*np.array([0.5,-2,0.5,-2]); drift=np.ones((4,5))
    ov=np.sign(hold)*np.maximum(abs(hold)-ys,0)**2; work=np.maximum(abs(hist)-ys,0)**2@dt
    eps=1e-7; des, rate=m._design(out['estimated_taus'],dt,np.array([0.,eps]),(hist,hold,drift,ov,work*hold),(True,True))
    np.testing.assert_array_equal(np.sign(rate[:,7:]), [[0,0],[-1,0],[0,1],[-1,-1]])
    np.testing.assert_allclose(des[:,1]/eps,rate,rtol=2e-6,atol=1e-8)
    print('all four yield regimes and t=0+ derivatives verified')
PY
```
- **Container Exit Code**: `0` (`1150 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w04_batch_04_solver.py']`
- **Stdout (tail)**:
```text
plastic_weights': [0.2488719, 0.1701068]}
public reference fields [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_cross', (80,)), ('te_secant', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
fold 0 taus [0.56813 3.79442] plastic [0.249808 0.170075]
fold 1 taus [0.56798 3.86072] plastic [0.247395 0.169993]
fold 2 taus [0.56815 3.84017] plastic [0.249012 0.170052]
fold 3 taus [0.56884 3.83255] plastic [0.248446 0.170217]
lab x yield regime LOLO RMSE [lab, regime, n, curve, rate] [(0, 0, 16, 0.24787538111989588, 0.008206631316982745), (0, 3, 29, 0.33254393885947214, 0.052167009804758024), (1, 0, 31, 0.33306372273446483, 0.03702457822174857), (1, 3, 26, 0.3380621929020056, 0.039701377944306976), (2, 0, 28, 0.37632898224720324, 0.08009480207248121), (2, 3, 19, 0.3284873237289185, 0.04873260863356208), (3, 0, 25, 0.27635424202568376, 0.030217269033073254), (3, 3, 26, 0.2745014537163385, 0.01572435608306737)]
worst LOLO curve,rate 0.37632898224720324 0.08009480207248121
calibration stress test 0 sparse + EMI 2.027907815249641e-05 dense + EMI 1.8720929784804552e-05
calibration stress test 1 sparse + EMI 4.3906987656966434e-05 dense + EMI 1.743412054452942e-05
calibration stress test 2 sparse + EMI 6.116478983742013e-05 dense + EMI 4.553396101522611e-05
calibration stress test 3 sparse + EMI 2.3431994566558423e-05 dense + EMI 2.844739223983872e-05
sub-yield branch weights exactly zero
pooled 80 (0.011627527151528627, 0.004338997592344837)
complete 74 (0.004749525785970788, 0.003376482297052404)
missing 6 (0.03904411899502136, 0.010507974658476698)
lab × regime clean RMSE [(0, 0, 0.015159316732299071, 0.004897344305532003), (0, 3, 0.019627575598887052, 0.006363197486990488), (1, 0, 0.005208062157277369, 0.004306260455363711), (1, 3, 0.005648735430132848, 0.002654277569123731), (2, 0, 0.017276193671669087, 0.00581835503393243), (2, 3, 0.004699794412498439, 0.0020527108379839233), (3, 0, 0.012701971380728318, 0.005319996277206033), (3, 3, 0.0034176935455337504, 0.00191404975609956)]
worst clean lab × regime strain/rate [0.01962758 0.0063632 ]
all four yield regimes and t=0+ derivatives verified

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4530818d9a44b92 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c4530818 (directory).
```

### Turn 10 — `submit_completed_task` (`12456 ms` LLM)
- **Rationale**: The verified solver reduced missing-row clean strain/rate RMSE from 0.58263/0.18431 to 0.03904/0.01051 without changing complete-row predictions or fitted parameters. Sparse and noisy calibration checks and the exact zero-plasticity check also passed.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_w04_batch_04_solver.py': 'FILE:8604'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w04_batch_04_solver.py`. The public schema passed, and all four yield regimes and initial-rate derivatives were checked. Worst clean lab-by-yield RMSE was 0.01963 for strain and 0.00636 for initial rate.`

### Turn 11 — `execute_in_remote_docker` (`40123 ms` LLM)
- **Rationale**: Audit three failure modes not covered by the prior run: a history-order or sign error would require a basis fix, joint sparse and noisy checks across a wing would expose a calibration fallback failure, and finer clean-reference slices would show whether pooled performance hides a lab or stratum offset.
- **Alternatives Weighed**: `Add rational, interaction, or relay terms to the constitutive model` (rejected: This task specifies a two-state Prony recurrence and signed quadratic yield terms, not a relay, diffusion operator, or classification problem. An independent recurrence and stratified residuals can test whether the specified basis or lab alignment is wrong.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util,pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_w04_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for rec in pickle.load(open(p+'public_alpha_w04_batch_04.pkl','rb')):
 d,g=rec['dataset'],rec['gt']; tr,te=d['train'],d['test']; dt=np.asarray(d['past_dt']); t=np.asarray(d['eval_times']); y=d['yield_stress']; o=m.solve_alpha_w04_batch_04(d); aa=m._calibrate(d['calibration_load_cells']); xs=m._align(tr,te)
 # Independent chronological recurrence and signed-yield oracle, including exact threshold and reversal.
 h=np.zeros((10,len(dt))); h[2:4,-1]=[2*y,-2*y]; h[6:8,:2]=[[2*y,-y],[-2*y,y]]; h[8:,-1]=[y,-y]
 q=y*np.array([.5,-.5,.5,-.5,2.,-2.,2.,-2.,1.,-1.]); x=np.ones((len(h),5)); x[1::2]=-x[::2]; tau=o['estimated_taus']; state=np.zeros((len(h),2)); work=np.zeros(len(h))
 for j,ds in enumerate(dt):
  state=h[:,j,None]+(state-h[:,j,None])*np.exp(-ds/tau)
  work+=np.maximum(0,np.abs(h[:,j])-y)**2*ds
 ov=(np.maximum(q-y,0)**2-np.maximum(-q-y,0)**2); delta=q[:,None]-state; f=1-np.exp(-t[:,None]/tau)
 A,B=m._design(tau,dt,t,(h,q,x,ov,work*q),(True,True)); oracle=np.concatenate((delta[:,None,:]*f, x[:,None,:]*t[None,:,None],(ov[:,None]*t)[:,:,None],((work*q)[:,None]*f[:,0])[:,:,None]),axis=-1)
 np.testing.assert_allclose(A,oracle,rtol=2e-12,atol=2e-12); np.testing.assert_allclose(B,np.column_stack((delta/tau,x,ov,work*q/tau[0])),rtol=2e-12,atol=2e-12)
 np.testing.assert_allclose(A[::2],-A[1::2],atol=2e-12); assert np.all(A[-2:,:,7:]==0)
 # Check wing-level fallback with both labs sparse and one bad spot check per lab.
 c=d['calibration_load_cells']; labels=np.asarray(c['lab_id']); cmd=np.asarray(c['cmd_stress']); rows=np.arange(len(labels))
 for wing in ((0,1),(2,3)):
  retain=~np.isin(labels,wing); bad=[]
  for lab in wing:
   j=rows[labels==lab]; retain[j[np.argmin(cmd[j])]]=True; k=j[np.argmax(cmd[j])]; retain[k]=True; bad.append(k)
  panel={k:np.asarray(v)[retain].copy() for k,v in c.items()}; panel['true_stress'][np.isin(rows[retain],bad)]+=[0.16,-0.24]; recovered=m._calibrate(panel)
  err=float(np.max(np.abs(recovered-aa))); print('joint sparse+EMI wing',wing,'max coefficient delta',err); assert err<0.001
 # Compare pooled errors with measured lab/stratum/yield slices and check if any lab has a systematic signed offset.
 reg=(abs(g['te_hold_true'])>y).astype(int)+2*(abs(g['te_hist_true']).max(axis=1)>y); miss=np.isnan(te['coupon_features_raw']).any(axis=1)
 err=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']
 groups=[]
 for lab in range(4):
  for stratum in np.unique(te['stratum']):
   for r in np.unique(reg):
    sel=(te['lab_id']==lab)&(te['stratum']==stratum)&(reg==r)
    if sel.any(): groups.append((int(sel.sum()),float(np.sqrt(np.mean(err[sel]**2))),float(np.sqrt(np.mean(er[sel]**2)))))
 print('worst clean lab x stratum x yield RMSE',np.max(np.array(groups)[:,1:],axis=0)); print('complete-row signed curve/rate mean by lab',[(i,round(float(err[k].mean()),6),round(float(er[k].mean()),6)) for i in range(4) if (k:=(te['lab_id']==i)&~miss).any()]); assert max(gp[1] for gp in groups)<0.06 and max(gp[2] for gp in groups)<0.02
 print('independent recurrence, sign symmetry, and yield cutoff passed')
PY
```
- **Container Exit Code**: `0` (`685 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
joint sparse+EMI wing (0, 1) max coefficient delta 4.3906987656966434e-05
joint sparse+EMI wing (2, 3) max coefficient delta 6.116478983742013e-05
worst clean lab x stratum x yield RMSE [0.02544618 0.00827185]
complete-row signed curve/rate mean by lab [(0, 0.004697, 0.000403), (1, -0.002505, -0.002234), (2, -0.003298, -0.00158), (3, 0.000542, -0.000964)]
independent recurrence, sign symmetry, and yield cutoff passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4530818d9a44b92 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c4530818 (directory).
```

### Turn 12 — `submit_completed_task` (`12253 ms` LLM)
- **Rationale**: The isolated audit confirmed the Prony recurrence, sign symmetry, exact yield cutoff, and calibration fallback when both labs in either wing had sparse and noisy checks. The drift correction improved incomplete-coupon predictions without changing complete-coupon predictions or fitted parameters.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w04_batch_04_solver.py`. The public schema, all four yield regimes, and sparse and noisy calibration checks passed. Across public lab × stratum × yield slices, the worst clean RMSE was 0.02545 for strain and 0.00827 for initial rate.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
