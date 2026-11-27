# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_alpha_m98_00_batch_04__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_hereditary_memory_and_factored_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `387,721` (`375,563` / `105,600` / `12,158`) |
| **Total Episode Wall Time (s)** | `421.19s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`18156 ms` LLM)
- **Rationale**: Check whether the public batch has chamber diagnostics and whether the latest apparatus entry matches an earlier regime; those findings determine which chamber law to implement. Inspect array shapes before fitting.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; python3 - <<'PY'
import pickle
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
def show(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+k)
 elif hasattr(x,'shape'): print(p,x.shape,x.dtype)
 else: print(p,type(x).__name__,str(x)[:90])
show(x)
PY
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.39954,
        -1.10636,
        -0.39918,
        -0.6763,
        -1.22919,
        -0.30638,
        -0.60015,
        -0.9703
      ],
      [
        -0.46072,
        -1.16263,
        -0.35029,
        -0.39254,
        -1.1539,
        -0.51141,
        -0.43087,
        -1.11441
      ],
      [
        -0.10304,
        -1.24036,
        -0.4348,
        -0.00583,
        -1.31512,
        -0.4152,
        -0.04874,
        -1.14728
      ],
      [
        0.56489,
        -1.10795,
        -0.30337,
        0.28112,
        -0.92364,
        -0.40711,
        0.23302,
        -1.0947
      ],
      [
        0.71322,
        -1.19965,
        -0.50163,
        0.47967,
        -1.11243,
        -0.57257,
        0.65344,
        -1.16418
      ],
      [
        0.92081,
        -0.89738,
        -0.4821,
        1.04546,
        -1.01061,
        -0.43449,
        0.75986,
        -1.00443
      ],
      [
        1.03018,
        -1.06114,
        -0.27844,
        0.8375,
        -1.26257,
        -0.44659,
        0.54816,
        -1.07536
      ],
      [
        0.799,
        -1.25644,
        -0.44577,
        0.70738,
        -1.0459,
        -0.3756,
        0.91263,
        -1.0694
      ]
    ]
  }
]/dataset/past_dt (6,) float64
/dataset/eval_times (12,) float64
/dataset/calibration_load_cells/bridge_raw (80, 4) float64
/dataset/calibration_load_cells/true_channel_loads (80, 4) float64
/dataset/calibration_load_cells/single_channel_sweeps/lab_id (24,) int64
/dataset/calibration_load_cells/single_channel_sweeps/cmd_stress (24,) float64
/dataset/calibration_load_cells/single_channel_sweeps/true_stress (24,) float64
/dataset/train/lab_id (72,) int64
/dataset/train/stratum (72,) int64
/dataset/train/coupon_features_raw (72, 4) float64
/dataset/train/ref_coupons (72, 8) float64
/dataset/train/past_channel_raw (72, 6, 4) float64
/dataset/train/hold_channel_raw (72, 4) float64
/dataset/train/strain_0 (72,) float64
/dataset/train/strain_curves_obs (72, 12) float64
/dataset/train/instant_rate_obs (72,) float64
/dataset/train/aux_chamber_telemetry (72, 4) float64
/dataset/test/lab_id (36,) int64
/dataset/test/stratum (36,) int64
/dataset/test/coupon_features_raw (36, 4) float64
/dataset/test/ref_coupons (36, 8) float64
/dataset/test/past_channel_raw (36, 6, 4) float64
/dataset/test/hold_channel_raw (36, 4) float64
/dataset/test/strain_0 (36,) float64
/dataset/test/aux_chamber_telemetry (36, 4) float64
/campaign_diagnostics NoneType None
```

### Turn 2 — `execute_in_remote_docker` (`19323 ms` LLM)
- **Rationale**: Compare complete traces and channel summaries to see whether the latest entry clearly matches a prior regime. Inspect archived chamber equations so the solver can reuse the matched law without reusing material or calibration constants.
- **Alternatives Weighed**: `Default to the benchtop chamber law` (rejected: The prefix log and prior verified solvers are available and may identify a different chamber mechanism.)

```bash
cat /tmp/hegel_continual_archive/archive_manifest.json; for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $f"; tail -n 57 "$f"; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference-panel alignment imputation' 2>&1 | head -30; python3 - <<'PY'
import json,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'; a=json.load(open(p));
for e in a:
 x=np.asarray(e['facility_telemetry_trace']); print(e.keys(),e['completed_batch_index'],x.shape); print('Mean',x.mean(0).round(4),'std',x.std(0).round(4))
x=np.stack([e['facility_telemetry_trace'] for e in a]); print('latest trace RMSE',np.sqrt(np.mean((x-x[-1])**2,axis=(1,2)))); print('latest channel RMSE',np.sqrt(np.mean((x-x[-1])**2,axis=1)).round(3))
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
dict_keys(['completed_batch_index', 'completed_short_name', 'facility_telemetry_trace']) 0 (16, 8)
Mean [-3.0700e-02  1.1281e+00  2.4594e+00 -2.3700e-02  1.1449e+00 -4.2040e-01
  1.3000e-03  1.1570e+00] std [0.6294 0.1285 0.0494 0.6401 0.179  0.0826 0.6407 0.1567]
dict_keys(['completed_batch_index', 'completed_short_name', 'facility_telemetry_trace']) 1 (16, 8)
Mean [ 0.0655 -1.1653 -0.3535  0.0686 -1.1927 -0.3863  0.0226 -1.1281] std [0.6538 0.1065 0.0668 0.6392 0.1166 0.0812 0.6685 0.1532]
dict_keys(['completed_batch_index', 'completed_short_name', 'facility_telemetry_trace']) 2 (16, 8)
Mean [ 0.0649  1.1447 -0.3282  0.0467  1.0926  2.5043  0.1039  1.1938] std [0.6625 0.1455 0.0945 0.6684 0.1384 0.0907 0.6433 0.1589]
dict_keys(['completed_batch_index', 'completed_short_name', 'facility_telemetry_trace']) 3 (16, 8)
Mean [ 0.0669 -1.1469 -0.3978  0.0293 -1.1459 -0.4242  0.0542 -1.1178] std [0.7177 0.135  0.0776 0.6551 0.1223 0.0801 0.5958 0.0867]
latest trace RMSE [1.85550942 0.51555362 1.7665609  0.        ]
latest channel RMSE [[1.194 2.282 2.859 1.109 2.299 0.109 1.006 2.28 ]
 [0.883 0.187 0.105 0.766 0.136 0.129 0.802 0.187]
 [0.5   2.299 0.167 0.413 2.25  2.931 0.44  2.32 ]
 [0.    0.    0.    0.    0.    0.    0.    0.   ]]
```

### Turn 3 — `execute_in_remote_docker` (`19631 ms` LLM)
- **Rationale**: The close telemetry match does not by itself identify which terms transfer. Comparing the archived implementations will distinguish shared chamber equations from changes to calibration or material fitting.

```bash
cd /tmp/hegel_continual_archive && cat dev_alpha_m98_00_batch_03_solver.py && diff -u dev_alpha_m98_00_batch_03_solver.py dev_alpha_m98_00_batch_01_solver.py || true
```
- **Container Exit Code**: `0` (`287 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
aset)
    taus, theta = _fit(dt, times, train_coupon,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    curve, rate = _chamber(taus, theta, dt, times, test_coupon)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
--- dev_alpha_m98_00_batch_03_solver.py	2026-11-27 13:51:58.090620726 +0000
+++ dev_alpha_m98_00_batch_01_solver.py	2026-11-27 13:51:58.090397084 +0000
@@ -91,17 +91,6 @@
     return np.exp(best[:2][order]), np.r_[best[2:4][order], best[4:]]
 
 
-def _chamber(taus, theta, dt, times, coupon):
-    curve, rate = _design(taus, dt, times, coupon)
-    drift = coupon[2] @ theta[2:]
-    # The chamber preserves the prepared branch memory. It redirects drift
-    # into both transient branches while preserving the initial derivative.
-    fractions = np.array([0.62686, 0.60591])
-    growth = -np.expm1(-times[:, None] / taus)
-    drift_correction = growth @ (fractions * taus) - times * fractions.sum()
-    return curve @ theta + drift[:, None] * drift_correction, rate @ theta
-
-
 def solve_batch(dataset: dict) -> dict:
     """Infer intrinsic parameters from training labels, not deployment telemetry."""
     dt, times = dataset['past_dt'], dataset['eval_times']
@@ -110,7 +99,15 @@
     taus, theta = _fit(dt, times, train_coupon,
                        train['strain_curves_obs'] - train['strain_0'][:, None],
                        train['instant_rate_obs'])
-    curve, rate = _chamber(taus, theta, dt, times, test_coupon)
+    # Chamber equilibration resets the initial branch memory. A fraction
+    # of the coupon drift enters each damped branch instead of persisting.
+    _, hold, x = test_coupon
+    drift = x @ theta[2:]
+    drift_fractions = np.array([0.52, 0.43])
+    amplitude = hold[:, None] * theta[:2] + drift[:, None] * (drift_fractions * taus)
+    steady_drift = (1.0 - drift_fractions.sum()) * drift
+    decay = -np.expm1(-times[:, None] / taus)
     return {'estimated_taus': taus, 'hereditary_weights': theta,
-            'predicted_strain': test['strain_0'][:, None] + curve,
-            'initial_rate_pred': rate}
+            'predicted_strain': (test['strain_0'][:, None] + amplitude @ decay.T
+                                 + steady_drift[:, None] * times),
+            'initial_rate_pred': amplitude @ (1.0 / taus) + steady_drift}
```

### Turn 4 — `execute_in_remote_docker` (`26958 ms` LLM)
- **Rationale**: The other two archived laws may distinguish which factors the log actually encodes. Separately measure affine calibration error to decide whether the inherited bridge fit needs revision.
- **Alternatives Weighed**: `Copy batch 01 because its full trace is closest to batch 03` (rejected: The verified batch 01 and 03 equations disagree on both memory and drift; aggregate distance alone cannot select the active terms.)
- **ATMS NOGOOD DECLARED**: Refuted `Matching the overall apparatus trace identifies a single transferable chamber law.` | Witness: `Batches 01 and 03 are the closest trace pair to the latest entry, but their verified solvers use different drift fractions and different initial-memory conditions.` | Blanket Diagnosis: `An aggregate telemetry distance can combine channels that may track different physical factors and obscure disagreement in those factors.` | Sublated Pivot: `Compare all archived equations and the persistent telemetry channels separately, retaining a campaign-specific benchtop material and calibration fit.`

```bash
cd /tmp/hegel_continual_archive; for f in *solver.py; do echo "=== $f"; sed -n '/^def _chamber/,$p' "$f" | tail -n 36; done; sed -n '1,105p' /holdout_task/public_asset_index.json; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
for k in ('train','test'):
 g=x[k]; a=g['aux_chamber_telemetry']; print(k,'telemetry mean/std',a.mean(0),a.std(0),'per lab',[(i,len(a[g['lab_id']==i]),a[g['lab_id']==i].mean(0).round(3)) for i in np.unique(g['lab_id'])]); print('missing',np.isnan(g['coupon_features_raw']).sum(0))
print('dt',x['past_dt'],'times',x['eval_times']); c=x['calibration_load_cells']; a=c['bridge_raw']; y=c['true_channel_loads']; X=np.c_[a,np.ones(len(a))]; print('affine inverse calibration RMSE',np.sqrt(np.mean((X@np.linalg.lstsq(X,y,rcond=None)[0]-y)**2,axis=0)))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation' 2>/dev/null | cut -c1-135 | head -12
```
- **Container Exit Code**: `0` (`381 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4
                ],
                "type": "ndarray"
              },
              "coupon_features_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              },
              "hold_channel_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              },
              "lab_id": {
                "dtype": "int64",
                "shape": [
                  36
                ],
                "type": "ndarray"
              },
              "past_channel_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  6,
                  4
                ],
                "type": "ndarray"
              },
              "ref_coupons": {
                "dtype": "float64",
                "shape": [
                  36,
                  8
                ],
                "type": "ndarray"
              },
              "strain_0": {
                "dtype": "float64",
                "shape": [
                  36
                ],
                "type": "ndarray"
              },
              "stratum": {
                "dtype": "int64",
                "shape": [
                  36
train telemetry mean/std [ 0.30433324  0.16824692 -0.00093925 -0.07019175] [1.00090813 0.97225306 0.89970879 1.06799247] per lab [(0, 20, array([ 0.424,  0.289, -0.217,  0.219])), (1, 16, array([ 0.486,  0.096,  0.286, -0.088])), (2, 18, array([ 0.139,  0.18 , -0.081, -0.342])), (3, 18, array([ 0.175,  0.087,  0.064, -0.104]))]
missing [4 8 5 6]
test telemetry mean/std [-0.14210112  0.06791686 -0.1638547   0.0609121 ] [1.09757969 0.92907382 1.01347223 0.93922327] per lab [(0, 9, array([ 0.147, -0.246, -0.286,  0.031])), (1, 6, array([ 0.03 , -0.634, -0.719,  0.545])), (2, 12, array([-0.153,  0.557,  0.239, -0.099])), (3, 9, array([-0.532,  0.198, -0.209, -0.019]))]
missing [1 3 2 1]
dt [0.35 0.35 0.4  0.45 0.5  0.55] times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
affine inverse calibration RMSE [0.36107573 0.43179923 0.32100513 0.51135968]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray,
```

### Turn 5 — `execute_in_remote_docker` (`24941 ms` LLM)
- **Rationale**: Read the two archived chamber laws missed by the previous extraction. Compare held-out forward and inverse calibration errors and check deployment load ranges; those results will determine whether to retain the affine fit or replace it with a nonlinear bridge model.

```bash
cd /tmp/hegel_continual_archive; for f in dev_alpha_m98_00_s00_solver.py dev_alpha_m98_00_batch_02_solver.py; do echo "=== $f"; tail -n 43 "$f"; done; python3 - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=x['calibration_load_cells']; a=c['bridge_raw']; y=c['true_channel_loads'];
print('calibration raw range',np.stack([a.min(0),a.max(0)]).round(3)); print('calibration true range',np.stack([y.min(0),y.max(0)]).round(3));
for split in ('train','test'):
 b=np.vstack([x[split]['past_channel_raw'].reshape(-1,4),x[split]['hold_channel_raw']]); print(split,'raw range',np.stack([b.min(0),b.max(0)]).round(3))
rng=np.random.default_rng(42); k=np.array_split(rng.permutation(len(a)),5)
def design(z,d):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,i],axis=1) for j in range(1,d+1) for i in combinations_with_replacement(range(4),j)])
for name,z,target in [('inverse',a,y),('forward',y,a)]:
 for d in (1,2,3):
  A=design(z,d); r=np.empty_like(target)
  for h in k:
   t=np.setdiff1d(np.arange(len(z)),h); r[h]=A[h]@np.linalg.lstsq(A[t],target[t],rcond=None)[0]-target[h]
  print(name,d,'channel RMSE',np.sqrt(np.mean(r*r,axis=0)).round(5),'sum RMSE',np.sqrt(np.mean(r.sum(1)**2)).round(5))
for i in range(10): print('sweep',a[i].round(3),y[i].round(3))
PY
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.median(np.abs(r - np.median(r))),
                      initial_scale * 0.002) for r in (e[:k], e[k:])]
        scale = 2.5 * np.repeat(scales, [k, len(rate_obs)])
        best = fit(best, 'cauchy', scale).x
    order = np.argsort(best[:2])
    return np.exp(best[:2][order]), np.r_[best[2:4][order], best[4:]]


def solve_batch(dataset: dict) -> dict:
    """Infer intrinsic parameters from training labels, not deployment telemetry."""
    dt, times = dataset['past_dt'], dataset['eval_times']
    train, test = dataset['train'], dataset['test']
    train_coupon, test_coupon = _prepare(dataset)
    taus, theta = _fit(dt, times, train_coupon,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # This blind campaign supplies no chamber diagnostics or lineage archive.
    # Apply the specified benchtop fallback to the deployment coupons.
    curve, rate = _design(taus, dt, times, test_coupon)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
calibration raw range [[ -2.805 -13.31   -2.925 -14.722]
 [ 14.765   1.983  13.701   1.663]]
calibration true range [[-1.6 -1.6 -1.6 -1.6]
 [ 1.7  1.7  1.7  1.7]]
train raw range [[-0.944 -0.803 -0.927 -0.852]
 [ 2.954  1.785  3.139  1.708]]
test raw range [[-1.076 -0.732 -1.081 -0.758]
 [ 2.982  1.767  3.268  1.657]]
inverse 1 channel RMSE [0.39727 0.47187 0.36072 0.55903] sum RMSE 0.38423
inverse 2 channel RMSE [0.84444 0.14045 0.6675  0.75307] sum RMSE 0.86701
inverse 3 channel RMSE [6.80246 6.99788 4.49785 6.77453] sum RMSE 24.96486
forward 1 channel RMSE [2.8737  2.59923 2.53285 2.74238] sum RMSE 0.40598
forward 2 channel RMSE [2.94479 2.69104 2.5683  2.81779] sum RMSE 0.3222
forward 3 channel RMSE [3.91511 3.56405 3.45604 3.80706] sum RMSE 0.00036
sweep [-2.805  0.104  0.16  -0.536] [-1.6  0.   0.   0. ]
sweep [-0.665 -0.03   0.16  -0.282] [-0.6  0.   0.   0. ]
sweep [ 1.1   -0.204  0.16   0.048] [0.7 0.  0.  0. ]
sweep [ 3.396 -0.337  0.16   0.302] [1.7 0.  0.  0. ]
sweep [-0.525 -2.003  0.508 -0.13 ] [ 0.  -1.6  0.   0. ]
sweep [-0.109 -0.628  0.291 -0.13 ] [ 0.  -0.6  0.   0. ]
sweep [ 0.431  0.507  0.008 -0.13 ] [0.  0.7 0.  0. ]
sweep [ 0.847  1.983 -0.21  -0.13 ] [0.  1.7 0.  0. ]
sweep [ 0.14  -0.538 -2.925  0.073] [ 0.   0.  -1.6  0. ]
sweep [ 0.14  -0.27  -0.683 -0.054] [ 0.   0.  -0.6  0. ]
```

### Turn 6 — `execute_in_remote_docker` (`30483 ms` LLM)
- **Rationale**: Inspect polynomial coefficients and the worst sweep rows to distinguish ordinary per-channel curvature from interactions that cancel across channels. That distinction determines which terms must be inverted and which can be eliminated algebraically.
- **Alternatives Weighed**: `Invert the unrestricted forward cubic` (rejected: Its large per-channel errors make the inverse unreliable even though it predicts the raw sum accurately.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine inverse bridge captures total-load calibration across the provided sweeps.` | Witness: `The affine inverse has held-out total-load RMSE 0.38423, while the forward cubic predicts the raw channel sum with RMSE 0.00036 despite large individual-channel errors.` | Blanket Diagnosis: `Strong nonlinear channel interactions may cancel in the raw channel sum; a generic inverse regression misses that structure.` | Sublated Pivot: `Identify conserved channel combinations and the smallest structured forward map before fitting and validating the physical inverse.`

```bash
python3 - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; Y=C['bridge_raw']; X=C['true_channel_loads']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
print('row nonzero counts',np.unique(np.count_nonzero(X,axis=1),return_counts=True)); print('paired extreme rows',np.c_[X,Y][np.argsort(np.linalg.norm(Y,axis=1))[-8:]])
for k,v in C['single_channel_sweeps'].items(): print(k,np.shape(v),np.asarray(v).ravel()[:16]);
for d in (1,2,3,4):
 idx=[()]+[i for j in range(1,d+1) for i in combinations_with_replacement(range(4),j)]; A=np.column_stack([np.prod(X[:,i],axis=1) for i in idx]); B=np.linalg.lstsq(A,Y,rcond=None)[0]; E=A@B-Y; print('degree',d,'rank',np.linalg.matrix_rank(A),'rms',np.sqrt(np.mean(E**2,axis=0)),'sum rms',np.sqrt(np.mean(E.sum(1)**2)))
 if d==3:
  for i,b in zip(idx,B):
   if np.max(np.abs(b))>0.02: print(i,b,b.sum())
  print('row residuals 0..30',np.c_[np.arange(30),np.linalg.norm(E[:30],axis=1),E[:30].sum(1)])
print('rows 16..32', np.c_[X,Y][16:32]); print('train median loads', np.median(D['train']['hold_channel_raw'],axis=0))
PY
wc -l /tmp/hegel_continual_archive/*; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`374 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
85581 -0.92348 -0.13024 -2.73624  0.51914]
 [-0.05103 -0.76346 -1.2716   0.67525 -0.37142 -0.95324 -1.79602  0.57546]
 [-0.64359 -0.0685  -1.30165  1.25976 -1.00966 -0.4116  -1.99467  0.98156]
 [ 0.71937 -0.48118 -0.64217  1.33997  0.55623 -0.72878 -0.31202  1.66286]
 [-0.8102  -0.68026  1.12616  0.51031 -1.2377  -0.43407  2.5233  -0.1376 ]
 [-1.08726 -1.24542  0.31576  1.0326  -1.87849 -1.24071  1.39996  0.31124]
 [ 1.01858  0.46454  0.68833  0.13483  1.95924  0.38574  1.13044  0.16899]
 [-0.86061 -1.0734   0.70662  0.53847 -1.39123 -0.99113  1.7554  -0.08084]
 [ 0.86618  0.48913 -1.56271 -0.52586  1.80883 -0.36642 -2.90643 -0.21385]
 [ 0.96162 -0.51301 -0.64414 -0.16966  1.23855 -0.78689 -0.68434  0.03721]
 [-0.21393  0.99709  1.38307  0.77771  0.06203  1.48628  3.04844  0.27571]
 [-0.19067  0.60221  0.95652  0.21325  0.07257  0.79358  1.65166 -0.14118]
 [-0.08014 -1.52942 -0.54627  1.23049 -0.81874 -1.85675  0.07415  1.10792]]
train median loads [1.7543  0.90518 1.94037 0.88545]
  665 /tmp/hegel_continual_archive/archive_apparatus_log.json
   49 /tmp/hegel_continual_archive/archive_manifest.json
  113 /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py
  107 /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_02_solver.py
  116 /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_03_solver.py
  107 /tmp/hegel_continual_archive/dev_alpha_m98_00_s00_solver.py
    4 /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
 1161 total
import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
ds = item["dataset"]
out = mod.solve_batch(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 7 — `execute_in_remote_docker` (`25157 ms` LLM)
- **Rationale**: Estimate the cubic law after excluding the extreme readings and test whether the remaining residuals collapse to one direction. A sparse fit plus a rank-one disturbance would support robust forward calibration; broader residual structure would require revising the calibration map. Compare the apparatus channels separately to identify which modes are shared.

```bash
cat > /tmp/m98_probe.py <<'PY'
import json,pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; Y=C['bridge_raw']; X=C['true_channel_loads']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
idx=[()]+[i for j in range(1,4) for i in combinations_with_replacement(range(4),j)]; A=np.column_stack([np.prod(X[:,i],axis=1) for i in idx]);
mask=np.max(abs(Y),axis=1)<8; B=np.linalg.lstsq(A[mask],Y[mask],rcond=None)[0]
for i in range(6):
 E=Y-A@B; mask=np.max(abs(E),axis=1)<max(0.002,0.1/3**i); B=np.linalg.lstsq(A[mask],Y[mask],rcond=None)[0]; print('retained',mask.sum(),'rms',np.sqrt(np.mean((Y[mask]-A[mask]@B)**2)))
for i,b in zip(idx,B):
 if np.max(abs(b))>0.001:print(i,b)
E=Y-A@B; u,s,v=np.linalg.svd(E,full_matrices=False); print('singular',s,'direction',v[0]);print('unretained',np.c_[np.where(~mask)[0],(E@v[0])[~mask],np.linalg.norm(E-(E@v[0])[:,None]*v[0],axis=1)[~mask]])
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 z=np.array(e['facility_telemetry_trace']); print(e['completed_short_name'],'mean',z.mean(0),'sd',z.std(0)); print('initial/final',z[[0,-1]])
PY
python3 /tmp/m98_probe.py; sed -n '61,94p' /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py
```
- **Container Exit Code**: `0` (`451 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0013  1.157 ] sd [0.6294 0.1285 0.0494 0.6401 0.179  0.0826 0.6407 0.1567]
initial/final [[ 0.0847  1.0133  2.4799  0.0012  1.2107 -0.4429  0.0241  1.2794]
 [-0.0224  1.1062  2.4211 -0.0886  1.188  -0.4813 -0.0634  1.2718]]
dev_alpha_m98_00_batch_01 mean [ 0.0655 -1.1653 -0.3535  0.0686 -1.1927 -0.3863  0.0226 -1.1281] sd [0.6538 0.1065 0.0668 0.6392 0.1166 0.0812 0.6685 0.1532]
initial/final [[ 0.6848 -1.2941 -0.4056  0.6593 -0.9974 -0.2132  0.2967 -0.9376]
 [ 0.5285 -1.1444 -0.3663  0.7792 -1.0664 -0.4305  0.5527 -1.251 ]]
dev_alpha_m98_00_batch_02 mean [ 0.0649  1.1447 -0.3282  0.0467  1.0926  2.5043  0.1039  1.1938] sd [0.6625 0.1455 0.0945 0.6684 0.1384 0.0907 0.6433 0.1589]
initial/final [[ 0.8567  1.0209 -0.403   0.7421  1.106   2.4081  0.8935  1.2402]
 [ 0.9978  1.1578 -0.3986  0.7765  1.2628  2.4426  0.6687  1.2503]]
dev_alpha_m98_00_batch_03 mean [ 0.0669 -1.1469 -0.3978  0.0293 -1.1459 -0.4242  0.0542 -1.1178] sd [0.7177 0.135  0.0776 0.6551 0.1223 0.0801 0.5958 0.0867]
initial/final [[ 0.7558 -1.2015 -0.4058  0.8618 -0.9731 -0.4796  0.9255 -1.1693]
 [ 0.799  -1.2564 -0.4458  0.7074 -1.0459 -0.3756  0.9126 -1.0694]]
    y = np.concatenate((curve_obs.ravel(), rate_obs))
    k = curve_obs.size
    low, high = np.log([0.0500001, 24.999999])
    bounds = ([low, low] + [-np.inf] * 7, [high, high] + [np.inf] * 7)

    def design(log_tau):
        curve, rate = _design(np.exp(log_tau), dt, times, coupon)
        return np.vstack((curve.reshape(-1, 7), rate))

    def residual(p):
        return design(p[:2]) @ p[2:] - y

    def fit(p, loss, scale):
        return least_squares(lambda a: residual(a) / scale, p, bounds=bounds,
                             x_scale='jac', loss=loss, max_nfev=250)

    initial_scale = max(0.02 * np.median(np.abs(y)), 1e-7)
    fits = []
    for start in ((0.25, 2.0), (0.7, 5.0), (1.3, 14.0)):
        lt = np.log(start)
        beta = np.linalg.lstsq(design(lt), y, rcond=None)[0]
        fits.append(fit(np.r_[lt, beta], 'soft_l1', initial_scale))
    best = min(fits, key=lambda result: result.cost).x
    if adaptive:
        e = residual(best)
        scales = [max(1.4826 * np.median(np.abs(r - np.median(r))),
                      initial_scale * 0.002) for r in (e[:k], e[k:])]
        scale = 2.5 * np.repeat(scales, [k, len(rate_obs)])
        best = fit(best, 'cauchy', scale).x
    order = np.argsort(best[:2])
    return np.exp(best[:2][order]), np.r_[best[2:4][order], best[4:]]


def solve_batch(dataset: dict) -> dict:
```

### Turn 8 — `execute_in_remote_docker` (`42801 ms` LLM)
- **Rationale**: The three rejected sweeps have residuals along the same sum-zero direction. Build the structured inverse and test it against the schema now; an unpacking or shape failure will reveal where its interface differs from the archived fit. The candidate uses the batch 01 law provisionally, pending a factor-wise comparison of apparatus traces.
- **Alternatives Weighed**: `Unrestricted cubic calibration` (rejected: The clean sweeps identify a much smaller cyclic model: own-channel cubic response, adjacent coupling, and one neighboring product per channel.)

```bash
cat > /tmp/m98_bridge.txt <<'PY'
def _bridge_basis(loads):
    return np.stack((np.ones_like(loads), loads, np.roll(loads, 1, axis=-1),
                     np.roll(loads, -1, axis=-1), loads * np.roll(loads, -1, axis=-1),
                     loads**3), axis=-1)


def _fit_bridge(raw, loads):
    basis = _bridge_basis(loads)

    def fit(rows):
        return np.stack([np.linalg.lstsq(basis[rows, i], raw[rows, i], rcond=None)[0]
                         for i in range(4)])

    # Start from the ordinary bridge readings, excluding common-mode spikes.
    radius = np.max(np.abs(raw - np.median(raw, axis=0)), axis=1)
    coef = fit(radius < 4 * np.median(radius))
    error = np.linalg.norm(np.sum(basis * coef, axis=-1) - raw, axis=1)
    return fit(error < max(8 * np.median(error), 1e-3))


def _channel_loads(raw, coef):
    offset, gain, prev, nxt, cross, cubic = coef.T
    v = (raw - offset) / gain
    eye = np.eye(4)
    for _ in range(14):
        forward = np.sum(_bridge_basis(v) * coef, axis=-1)
        jac = ((gain + cross * np.roll(v, -1, axis=-1) + 3 * cubic * v*v)[..., :, None] * eye
               + prev[:, None] * np.roll(eye, -1, axis=1)
               + (nxt + cross * v)[..., :, None] * np.roll(eye, 1, axis=1))
        delta = np.linalg.solve(jac, (forward - raw)[..., None])[..., 0]
        v -= delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return v


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    features = _aligned_features(train, test)
    cal = dataset['calibration_load_cells']
    coef = _fit_bridge(cal['bridge_raw'], cal['true_channel_loads'])
    coupons = []
    for group, x in zip((train, test), features):
        past = _channel_loads(group['past_channel_raw'], coef).sum(axis=-1)
        hold = _channel_loads(group['hold_channel_raw'], coef).sum(axis=-1)
        coupons.append((past, hold, np.column_stack((np.ones(len(x)), x))))
    return coupons


PY
sed -n '1,64p' /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py; tail -n 22 /tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py; python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py'); s=p.read_text(); print('Functions', [l for l in s.splitlines() if l.startswith('def ')]); a=s.index('def _prepare('); b=s.index('def _design(',a); Path('/tmp/m98_candidate.py').write_text(s[:a]+Path('/tmp/m98_bridge.txt').read_text()+s[b:])
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /tmp/m98_candidate.py
```
- **Container Exit Code**: `1` (`503 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t['calibration_load_cells'])
    return tuple((group['past_channel_raw'] @ cal[:-1] + cal[-1],
                  group['hold_channel_raw'] @ cal[:-1] + cal[-1], x)
                 for group, x in ((train, features[:n_train]),
                                  (test, features[n_train:])))


def _design(taus, dt, times, coupon):
    past, hold, x = coupon
    z = np.zeros((len(hold), 2))
    for alpha, stress in zip(-np.expm1(-dt[:, None] / taus), past.T):
        z += alpha * (stress[:, None] - z)
    gap = hold[:, None] - z
    growth = -np.expm1(-times[:, None] / taus)
    curve = np.concatenate((gap[:, None, :] * growth,
                            times[None, :, None] * x[:, None, :]), axis=2)
    rate = np.column_stack((gap / taus, x))
    return curve, rate


def _fit(dt, times, coupon, curve_obs, rate_obs, *, adaptive=True):
    y = np.concatenate((curve_obs.ravel(), rate_obs))
    k = curve_obs.size
    low, high = np.log([0.0500001, 24.999999])
    bounds = ([low, low] + [-np.inf] * 7, [high, high] + [np.inf] * 7)


def solve_batch(dataset: dict) -> dict:
    """Infer intrinsic parameters from training labels, not deployment telemetry."""
    dt, times = dataset['past_dt'], dataset['eval_times']
    train, test = dataset['train'], dataset['test']
    train_coupon, test_coupon = _prepare(dataset)
    taus, theta = _fit(dt, times, train_coupon,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Chamber equilibration resets the initial branch memory. A fraction
    # of the coupon drift enters each damped branch instead of persisting.
    _, hold, x = test_coupon
    drift = x @ theta[2:]
    drift_fractions = np.array([0.52, 0.43])
    amplitude = hold[:, None] * theta[:2] + drift[:, None] * (drift_fractions * taus)
    steady_drift = (1.0 - drift_fractions.sum()) * drift
    decay = -np.expm1(-times[:, None] / taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': (test['strain_0'][:, None] + amplitude @ decay.T
                                 + steady_drift[:, None] * times),
            'initial_rate_pred': amplitude @ (1.0 / taus) + steady_drift}
Functions ['def _aligned_features(train, test):', 'def _stress_calibration(calibration):', 'def _prepare(dataset):', 'def _design(taus, dt, times, coupon):', 'def _fit(dt, times, coupon, curve_obs, rate_obs, *, adaptive=True):', 'def solve_batch(dataset: dict) -> dict:']
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 9, in <module>
    out = mod.solve_batch(ds)
          ^^^^^^^^^^^^^^^^^^^
  File "/tmp/m98_candidate.py", line 137, in solve_batch
    taus, theta = _fit(dt, times, train_coupon,
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/m98_candidate.py", line 119, in _fit
    beta = np.linalg.lstsq(design(lt), y, rcond=None)[0]
                           ^^^^^^^^^^
  File "/tmp/m98_candidate.py", line 105, in design
    curve, rate = _design(np.exp(log_tau), dt, times, coupon)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/m98_candidate.py", line 92, in _design
    curve = np.concatenate((gap[:, None, :] * growth,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 72 and the array at index 1 has size 5
```

### Turn 9 — `execute_in_remote_docker` (`26111 ms` LLM)
- **Rationale**: The archived feature routine already returns one intercept-augmented matrix, so split it by coupon count. Held-out inverse errors and deployment round-trip residuals will reveal whether to keep the cubic inverse. Inspect the trace waveforms separately from their offsets to see whether either distinguishes archived chamber regimes.

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/m98_candidate.py'); s=p.read_text(); a=s.index('def _stress_calibration'); b=s.index('def _bridge_basis',a); s=s[:a]+s[b:]; s=s.replace('features = _aligned_features(train, test)','features = np.split(_aligned_features(train, test), [len(train[\'lab_id\'])])').replace('np.column_stack((np.ones(len(x)), x))','x'); p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py /tmp/m98_candidate.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys; sys.path.insert(0,'/tmp')
import m98_candidate as m, pickle,json,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; o=m.solve_batch(d); print('material',o['estimated_taus'],o['hereditary_weights']); print('keys',list(d)); c=d['calibration_load_cells']; u=c['bridge_raw']; l=c['true_channel_loads']; b=m._fit_bridge(u,l)
e=np.linalg.norm(np.sum(m._bridge_basis(l)*b,axis=-1)-u,axis=1); ok=e<.001; err=[]
for i in np.flatnonzero(ok):
 h=np.arange(len(u))!=i; ci=m._fit_bridge(u[h],l[h]); xi=m._channel_loads(u[i],ci); err.append(xi-l[i]);
err=np.asarray(err); print('LOO clean sweeps max channel error',abs(err).max(),'max and RMS total',abs(err.sum(1)).max(),np.sqrt(np.mean(err.sum(1)**2)))
for group in (d['train'],d['test']):
 u=np.vstack((group['past_channel_raw'].reshape(-1,4),group['hold_channel_raw'])); l=m._channel_loads(u,b); print('coupon min/max load',l.min(0),l.max(0),'max inversion residual',np.max(abs(np.sum(m._bridge_basis(l)*b,axis=-1)-u)))
idx=json.load(open('/holdout_task/public_asset_index.json')); print('asset index:',{k:(list(v) if isinstance(v,dict) else str(v)[:350]) for k,v in idx.items()})
q=np.array([r['facility_telemetry_trace'] for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))]);
for i in range(len(q)):
 z=q[i][:,[0,3,6]].mean(1); f=np.abs(np.fft.rfft(z-z.mean())); print(i,'shape',z.round(2),'spectrum',f.round(2),'codes',np.median(q[i][:,[1,2,4,5,7]],axis=0).round(3))
PY
```
- **Container Exit Code**: `0` (`976 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
material [0.66088422 3.9154807 ] [ 0.71251453  0.51929392  0.26509075  0.12968259 -0.12376963  0.13020807
 -0.08875421]
keys ['past_dt', 'eval_times', 'calibration_load_cells', 'train', 'test']
LOO clean sweeps max channel error 0.00038632196450619216 max and RMS total 0.0003870936230781219 0.0001587928491583908
coupon min/max load [-0.64958982 -0.64935799 -0.64547692 -0.64647527] [1.34370739 1.31386534 1.34965248 1.33831287] max inversion residual 1.3322676295501878e-15
coupon min/max load [-0.6457412  -0.64059713 -0.64175931 -0.64976534] [1.28832776 1.34208658 1.33928449 1.31977412] max inversion residual 8.881784197001252e-16
asset index: {'files': ['/holdout_task/public_batch.pkl', '/holdout_task/validate_public_schema.py'], 'schema': 'hegel_public_asset_index_v1'}
0 shape [ 0.04  0.32  0.72  0.89  0.84  0.8   0.54  0.04 -0.26 -0.34 -0.88 -0.88
 -0.88 -0.71 -0.45 -0.06] spectrum [0.   7.   0.4  0.48 0.21 0.69 0.2  0.26 0.39] codes [ 1.112  2.475  1.193 -0.415  1.201]
1 shape [ 0.55  0.76  1.1   0.83  0.64  0.23  0.05 -0.33 -0.58 -0.98 -0.88 -0.79
 -0.51  0.01  0.11  0.62] spectrum [0.   7.17 0.53 0.63 0.41 0.2  0.57 0.69 0.14] codes [-1.138 -0.371 -1.181 -0.383 -1.128]
2 shape [ 0.83  0.82  0.68  0.58  0.16 -0.23 -0.64 -0.87 -0.93 -0.75 -0.53 -0.3
  0.14  0.57  0.79  0.81] spectrum [0.   7.28 0.82 0.51 0.22 0.2  0.05 0.17 0.14] codes [ 1.143 -0.344  1.076  2.495  1.189]
3 shape [ 0.85  0.54  0.26 -0.16 -0.59 -0.8  -0.81 -0.95 -0.56 -0.43 -0.05  0.36
  0.62  0.91  0.81  0.81] spectrum [0.   7.28 0.45 0.41 0.19 0.19 0.5  0.31 0.24] codes [-1.167 -0.402 -1.149 -0.425 -1.131]
```

### Turn 10 — `execute_in_remote_docker` (`30048 ms` LLM)
- **Rationale**: Save the candidate and test the exact artifact. Leave-one-lab-out and held-out stress-quartile fits will show whether the inferred material is stable across reference panels and loading regimes; large parameter shifts would warrant revisiting the robust fit. Compare the archived deployment laws directly before finalizing the blind transfer.
- **Alternatives Weighed**: `Transfer batch 03 merely because it is the latest archived solver` (rejected: The task calls for comparing the latest apparatus trace with earlier entries. Batch 01 matches all five persistent channels, whereas the other earlier entries disagree on multiple channels.)

```bash
set -e
for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $(basename "$f")"; sed -n '/^def solve_batch/,$p' "$f"; done
cp /tmp/m98_candidate.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/m98_cv.py <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=d['train']; dt=d['past_dt']; times=d['eval_times']
a,_=m._prepare(d); y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ta,th=m._fit(dt,times,a,y,r)
def score(name, holdout):
    take=~holdout
    tau,th0=m._fit(dt,times,tuple(z[take] for z in a),y[take],r[take]); c,v=m._design(tau,dt,times,tuple(z[holdout] for z in a)); e=c@th0-y[holdout]; f=v@th0-r[holdout]
    print(f'{name}: n={holdout.sum()} taus={tau.round(6)} max_rel_tau_shift={np.max(abs(tau/ta-1)):.4g} max_weight_shift={np.max(abs(th0-th)):.4g} curve_medabs={np.median(abs(e)):.6g} curve_p90={np.quantile(abs(e),.9):.6g} curve_rmse={np.sqrt(np.mean(e**2)):.6g} rate_medabs={np.median(abs(f)):.6g} rate_rmse={np.sqrt(np.mean(f**2)):.6g}')
for lab in np.unique(t['lab_id']): score(f'lab={lab}',t['lab_id']==lab)
for i,ix in enumerate(np.array_split(np.argsort(a[1]),4)):
    h=np.zeros(len(r),bool); h[ix]=True; score(f'hold_stress_quartile={i}',h)
o=m.solve_batch(d); print('public material',ta,th); print('deployment range',[(k,float(o[k].min()),float(o[k].max())) for k in ('predicted_strain','initial_rate_pred')])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_cv.py
```
- **Container Exit Code**: `0` (`1266 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
ed benchtop fallback to the deployment coupons.
    curve, rate = _design(taus, dt, times, test_coupon)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
{"schema_valid": true, "errors": []}
lab=0: n=20 taus=[0.664062 4.137426] max_rel_tau_shift=0.05668 max_weight_shift=0.007249 curve_medabs=0.00713293 curve_p90=0.0204967 curve_rmse=0.0458241 rate_medabs=0.00496626 rate_rmse=0.00714086
lab=1: n=16 taus=[0.660468 3.814578] max_rel_tau_shift=0.02577 max_weight_shift=0.007241 curve_medabs=0.00972543 curve_p90=0.0241727 curve_rmse=0.0514132 rate_medabs=0.00400593 rate_rmse=0.00755347
lab=2: n=18 taus=[0.659662 3.852922] max_rel_tau_shift=0.01598 max_weight_shift=0.002788 curve_medabs=0.00835927 curve_p90=0.0213592 curve_rmse=0.0640442 rate_medabs=0.00858263 rate_rmse=0.00985806
lab=3: n=18 taus=[0.660132 3.89804 ] max_rel_tau_shift=0.004454 max_weight_shift=0.001401 curve_medabs=0.0100668 curve_p90=0.0242724 curve_rmse=0.0412669 rate_medabs=0.00625745 rate_rmse=0.00795665
hold_stress_quartile=0: n=18 taus=[0.660715 3.873231] max_rel_tau_shift=0.01079 max_weight_shift=0.004207 curve_medabs=0.00810171 curve_p90=0.023295 curve_rmse=0.0546999 rate_medabs=0.00575523 rate_rmse=0.00795985
hold_stress_quartile=1: n=18 taus=[0.662596 4.04024 ] max_rel_tau_shift=0.03186 max_weight_shift=0.004274 curve_medabs=0.00885309 curve_p90=0.0229744 curve_rmse=0.0449803 rate_medabs=0.00425444 rate_rmse=0.00726509
hold_stress_quartile=2: n=18 taus=[0.660622 3.964835] max_rel_tau_shift=0.01261 max_weight_shift=0.004673 curve_medabs=0.00833132 curve_p90=0.0232675 curve_rmse=0.0553894 rate_medabs=0.00611813 rate_rmse=0.00887286
hold_stress_quartile=3: n=18 taus=[0.659829 3.789803] max_rel_tau_shift=0.0321 max_weight_shift=0.004595 curve_medabs=0.0084686 curve_p90=0.0206333 curve_rmse=0.0487915 rate_medabs=0.00637101 rate_rmse=0.00821302
public material [0.66088422 3.9154807 ] [ 0.71251453  0.51929392  0.26509075  0.12968259 -0.12376963  0.13020807
 -0.08875421]
deployment range [('predicted_strain', 0.6375569502692362, 5.637111280587121), ('initial_rate_pred', 2.814349286659366, 5.820687410907057)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be736bef0aa2fa6c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be736bef (directory).
```

### Turn 11 — `execute_in_remote_docker` (`40173 ms` LLM)
- **Rationale**: Check that the lineage decision survives an oscillatory phase change, that a missing archive selects the baseline, and that the exact artifact preserves all four stress-tested public outputs. A failed check would identify a lineage-selection or equation regression. Public deployment labels are absent, so these checks cannot establish chamber prediction accuracy.
- **Alternatives Weighed**: `Keep the batch 01 chamber law unconditional` (rejected: It reproduces the public candidate but ignores the runtime apparatus log and the specified fallback when the archive is absent.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text().replace('import numpy as np', 'import json\nfrom pathlib import Path\n\nimport numpy as np', 1)
i = s.index('def solve_batch(')
s = s[:i] + '''def _chamber_lineage(log_path):
    if not log_path.exists():
        return None
    with log_path.open() as stream:
        entries = json.load(stream)
    if len(entries) < 2:
        return None
    traces = np.asarray([entry['facility_telemetry_trace'] for entry in entries])
    # The oscillatory channels change phase between batches. Match the
    # persistent levels instead, keeping the regulator channels separate.
    levels = np.median(traces[:, :, [1, 2, 4, 5, 7]], axis=1)
    nearest = np.argmin(np.linalg.norm(levels[:-1] - levels[-1], axis=1))
    return entries[nearest]['completed_short_name']


''' + s[i:]
a = s.index('    # Chamber equilibration')
s = s[:a] + '''    lineage = _chamber_lineage(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
    past, hold, x = test_coupon
    if lineage == 'dev_alpha_m98_00_batch_01':
        # Reset the prepared memory and divert part of the drift into the branches.
        drift = x @ theta[2:]
        fractions = np.array([0.52, 0.43])
        amplitude = hold[:, None] * theta[:2] + drift[:, None] * (fractions * taus)
        steady = (1.0 - fractions.sum()) * drift
        growth = -np.expm1(-times[:, None] / taus)
        curve = amplitude @ growth.T + steady[:, None] * times
        rate = amplitude @ (1.0 / taus) + steady
    else:
        curve_design, rate_design = _design(taus, dt, times, test_coupon)
        curve, rate = curve_design @ theta, rate_design @ theta
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
'''
p.write_text(s.replace('    past, hold, x = test_coupon', '    _, hold, x = test_coupon'))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m = load(root/'batch_solver.py'); prev = load('/tmp/m98_candidate.py')
archive = Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
assert m._chamber_lineage(archive) == 'dev_alpha_m98_00_batch_01'
traces = np.array([r['facility_telemetry_trace'] for r in json.loads(archive.read_text())]); levels = np.median(traces[:, :, [1,2,4,5,7]], axis=1)
print('Persistent-channel distances:', np.linalg.norm(levels[:-1]-levels[-1],axis=1))
assert m._chamber_lineage(Path('/nonexistent/archive_apparatus_log.json')) is None
with tempfile.TemporaryDirectory() as folder:
    f = Path(folder)/'shifted.json'; rows = json.loads(archive.read_text()); traces[-1, :, [0,3,6]] *= -1
    rows[-1]['facility_telemetry_trace'] = traces[-1].tolist(); f.write_text(json.dumps(rows)); assert m._chamber_lineage(f) == 'dev_alpha_m98_00_batch_01'
d = pickle.loads((root/'public_batch.pkl').read_bytes())[0]['dataset']; out = m.solve_batch(d); old = prev.solve_batch(d)
for k in out: np.testing.assert_allclose(out[k], old[k], rtol=0, atol=2e-14)
tau, theta = out['estimated_taus'], out['hereditary_weights']; assert 0.05 < tau[0] < tau[1] < 25
_, (_,hold,x) = m._prepare(d); exact_rate=(x @ theta[2:]) + hold * (theta[:2] @ (1/tau)); print('Max chamber initial-rate identity error:', np.max(abs(exact_rate-out['initial_rate_pred']))); np.testing.assert_allclose(out['initial_rate_pred'],exact_rate,atol=2e-14,rtol=0)
print('All 4 public outputs unchanged within 2e-14 from the stress-tested solver.')
PY
```
- **Container Exit Code**: `0` (`1006 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Persistent-channel distances: [4.93917052 0.06759422 4.91889443]
Max chamber initial-rate identity error: 8.881784197001252e-16
All 4 public outputs unchanged within 2e-14 from the stress-tested solver.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dbe77a01ec24bb54 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_dbe77a01 (directory).
```

### Turn 12 — `submit_completed_task` (`21910 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated Python schema verification and preserved all four outputs of the stress-tested candidate. Persistent-channel matching selected batch 01 at distance 0.068 versus 4.919 for the next closest entry; this supports the lineage selection but does not independently validate the chamber law.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:7004'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` and verified all four outputs with the public schema validator. Held-out calibration RMSE was 0.000159 for total axial load; worst held-out benchtop strain 90th-percentile absolute error was 0.0243, and worst initial-rate RMSE was 0.00986. Chamber predictions use the closest archived apparatus regime; public deployment labels were unavailable to verify accuracy.`

### Turn 13 — `execute_in_remote_docker` (`37534 ms` LLM)
- **Rationale**: Compare calibration alternatives on held-out single-channel, mixed-channel, and extreme sweeps. A meaningful improvement on the worst slice would justify changing the bridge. Separating the apparatus channels tests whether the closest lineage depends on pooling signals that disagree.
- **Alternatives Weighed**: `Test relay modes, stochastic drift operators, or within-batch balanced accuracy` (rejected: The measured outputs are continuous strain and strain rate; the supplied model and data have no relay, stochastic drift operator, or classification labels. Leave-one-lab-out regression is the relevant nuisance check.)

```bash
set -e
cat > /tmp/m98_margin.py <<'PY'
import importlib.util, json, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'batch_solver.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.loads((root/'public_batch.pkl').read_bytes())[0]['dataset']
c = d['calibration_load_cells']; raw, true = c['bridge_raw'], c['true_channel_loads']; basis = m._bridge_basis(true)
coef = m._fit_bridge(raw, true); resid = raw - (basis*coef).sum(-1)
clean = np.linalg.norm(resid, axis=1) < 0.001
print('Rejected sweep count:', (~clean).sum(), 'max absolute raw sum deviation on rejected sweeps:', np.max(np.abs(resid[~clean].sum(1))))
rng = np.random.default_rng(17); folds = np.array_split(rng.permutation(len(raw)), 8)
e_aff, e_cyc, e_rat = [np.zeros_like(raw) for _ in range(3)]
for h in folds:
    keep = np.ones(len(raw), bool); keep[h] = False
    b = m._fit_bridge(raw[keep], true[keep]); fitted = m._channel_loads(raw[h[clean[h]]], b)
    e_cyc[h[clean[h]]] = fitted - true[h[clean[h]]]
    affine = np.linalg.lstsq(np.c_[raw[keep], np.ones(sum(keep))], true[keep], rcond=None)[0]
    e_aff[h] = np.c_[raw[h], np.ones(len(h))] @ affine - true[h]
    good = keep & (np.linalg.norm(raw - (basis*b).sum(-1), axis=1) < 0.001)
    # Test saturating adjacent-channel difference, nested around the cyclic cubic model.
    def forward(v, b, sat):
        B = m._bridge_basis(v)
        out = (B*b).sum(-1)
        return out + b[:,4]*B[...,4]*(1/(1+sat*(B[...,1]-B[...,3])**2)-1)
    p = least_squares(lambda p: (forward(true[good],p[:24].reshape(4,6),p[24:])-raw[good]).ravel(), np.r_[b.ravel(), np.zeros(4)], bounds=(np.r_[np.full(24,-np.inf),np.zeros(4)], np.full(28,np.inf)), gtol=1e-10).x
    for j,v in zip(h[clean[h]], fitted):
        e_rat[j] = least_squares(lambda v: forward(v,p[:24].reshape(4,6),p[24:])-raw[j],v,gtol=1e-12).x - true[j]
for mask,name in [(clean & (np.count_nonzero(true,axis=1)==1),'single'),(clean & (np.count_nonzero(true,axis=1)==4),'mixed'),(clean & (np.max(abs(true),axis=1)>1.25),'extreme')]:
    print(name,int(mask.sum()),'axial RMSE affine/cyclic/rational',*[f'{np.sqrt(np.mean(e[mask].sum(1)**2)):.8g}' for e in (e_aff,e_cyc,e_rat)],'cyclic max channel',np.max(np.abs(e_cyc[mask])))
entries=json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text()); traces=np.array([e['facility_telemetry_trace'] for e in entries]); labels=[e['completed_short_name'] for e in entries[:-1]]
for cols in ([1,4,7],[2],[5],[1,2,4,5,7]):
    levels=np.median(traces[:,:,cols],axis=1); dist=np.linalg.norm(levels[:-1]-levels[-1],axis=1); print('lineage channels',cols,dict(zip(labels,dist.round(4))))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_margin.py
```
- **Container Exit Code**: `0` (`620 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Rejected sweep count: 3 max absolute raw sum deviation on rejected sweeps: 0.00028142526503138754
single 13 axial RMSE affine/cyclic/rational 0.29676819 0.00012966921 0.0001309569 cyclic max channel 0.0003636413728213258
mixed 64 axial RMSE affine/cyclic/rational 0.40470314 0.00016327399 0.00015851435 cyclic max channel 0.00041193551209245394
extreme 39 axial RMSE affine/cyclic/rational 0.45523599 0.00015537773 0.0001497139 cyclic max channel 0.0004048845967474235
lineage channels [1, 4, 7] {'dev_alpha_m98_00_s00': 4.0146, 'dev_alpha_m98_00_batch_01': 0.043, 'dev_alpha_m98_00_batch_02': 3.9582}
lineage channels [2] {'dev_alpha_m98_00_s00': 2.8772, 'dev_alpha_m98_00_batch_01': 0.0316, 'dev_alpha_m98_00_batch_02': 0.0583}
lineage channels [5] {'dev_alpha_m98_00_s00': 0.0097, 'dev_alpha_m98_00_batch_01': 0.0415, 'dev_alpha_m98_00_batch_02': 2.9198}
lineage channels [1, 2, 4, 5, 7] {'dev_alpha_m98_00_s00': 4.9392, 'dev_alpha_m98_00_batch_01': 0.0676, 'dev_alpha_m98_00_batch_02': 4.9189}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dbe77a01ec24bb54 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_dbe77a01 (directory).
```

### Turn 14 — `submit_completed_task` (`21179 ms` LLM)
- **Rationale**: The exact artifact passed isolated schema and equation checks. Held-out calibration directly supports replacing the affine bridge: mixed-sweep axial RMSE fell from 0.405 to 0.000163. Persistent-channel matching supports selecting batch 01, but without chamber labels it does not resolve the constitutive-transfer uncertainty.
- **Alternatives Weighed**: `Add a rational saturation term to the bridge` (rejected: It improved worst clean-slice axial RMSE only from 0.000163 to 0.000159 and slightly worsened the single-channel slice; it did not address the remaining uncertainty in chamber transfer.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` and verified all four required outputs in an isolated Python process. The worst held-out axial-load calibration RMSE was 0.000163 across clean sweep slices. On held-out benchtop coupons, the worst strain 90th-percentile absolute error was 0.0243 and the worst initial-rate RMSE was 0.00986. The apparatus log selected the batch 01 chamber law; public deployment labels were unavailable to verify its prediction accuracy.`

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
