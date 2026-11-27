# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_v2_c02_batch_04__blind_continual_campaign`

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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `394,246` (`382,794` / `134,400` / `11,452`) |
| **Total Episode Wall Time (s)** | `407.93s` |

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

### Turn 1 — `execute_in_remote_docker` (`16310 ms` LLM)
- **Rationale**: Inspect the log to detect a deployment regime mismatch and choose the corresponding prior solver; inspect schema and field shapes so the implementation uses only inputs available at inference.
- **Alternatives Weighed**: `Apply the benchtop law to deployment without checking the archive` (rejected: The task identifies chamber-specific regimes that may be distinguishable by the apparatus log.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; python3 - <<'PY'
import pickle
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
print('item:', x.keys(), 'diagnostics:',type(x.get('campaign_diagnostics')).__name__)
for k,v in x['dataset'].items():
 print(k, ([(a,getattr(b,'shape',type(b).__name__)) for a,b in v.items()] if isinstance(v,dict) else getattr(v,'shape',type(v).__name__)))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS reference-panel imputation viscoelastic relaxation' 2>&1 | head -20
```
- **Container Exit Code**: `0` (`501 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
uting_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`16423 ms` LLM)
- **Rationale**: Compare archive traces and isolate solver differences to detect a regime mismatch. Use that comparison to decide which chamber equations to transfer and which estimation steps to retain.
- **Alternatives Weighed**: `Reuse the most recent solver without comparing regimes` (rejected: Recency does not establish that its chamber law applies to the blind campaign.)

```bash
cd /tmp/hegel_continual_archive; cat archive_manifest.json; for f in *solver.py; do echo "=== $f"; if [[ $f == *s00* ]]; then cat "$f"; else diff -u alpha_m98_v2_c02_s00_solver.py "$f" || true; fi; done; python3 - <<'PY'
import json,numpy as np
r=json.load(open('archive_apparatus_log.json'))
for a in r:
 print({k:v for k,v in a.items() if k!='facility_telemetry_trace'})
 x=np.array(a['facility_telemetry_trace']);print('shape',x.shape,'mean',x.mean(0).round(4),'last',x[-1])
for a in r[:-1]:
 x=np.array(a['facility_telemetry_trace']);y=np.array(r[-1]['facility_telemetry_trace']);print(a['completed_short_name'],np.sqrt(np.mean((x-y)**2,axis=0)).round(4))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation' 2>&1 | head -8
```
- **Container Exit Code**: `0` (`405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Blind campaign: no chamber diagnostics or continual archive are available.
    strain, rate = _design(test, x_te, stress, dt, t)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c02_s00'}
shape (16, 8) mean [ 0.0115 -0.387  -0.6098  0.0452  1.0984  2.4549 -0.0531  1.13  ] last [ 0.18502 -0.32427 -0.34111  0.07911  1.09868  2.50478 -0.24166  1.37516]
{'completed_batch_index': 1, 'completed_short_name': 'alpha_m98_v2_c02_batch_01'}
shape (16, 8) mean [ 0.025  -0.391  -0.4471  0.0323 -1.1978 -0.3788  0.0347 -1.2345] last [ 0.80933 -0.30034 -0.21941  0.41144 -1.45165 -0.39638  0.74908 -1.22508]
{'completed_batch_index': 2, 'completed_short_name': 'alpha_m98_v2_c02_batch_02'}
shape (16, 8) mean [ 0.0703  2.5394 -0.2281  0.0067  1.1113 -0.3706  0.0145  1.1119] last [ 1.02881  2.47868 -0.16288  0.80619  1.10413 -0.33246  1.11407  1.04202]
{'completed_batch_index': 3, 'completed_short_name': 'alpha_m98_v2_c02_batch_03'}
shape (16, 8) mean [ 0.0324 -0.3923  0.0297  0.0876 -1.107  -0.3465  0.0265 -1.0919] last [ 0.5995  -0.57054 -0.04574  0.94841 -1.4782  -0.34073  0.74817 -1.12635]
alpha_m98_v2_c02_s00 [1.0797 0.12   0.6565 1.1201 2.217  2.8043 1.1785 2.2292]
alpha_m98_v2_c02_batch_01 [0.8023 0.1446 0.5007 0.8213 0.2614 0.1059 0.8271 0.2335]
alpha_m98_v2_c02_batch_02 [0.5088 2.9342 0.3175 0.4428 2.2284 0.066  0.4179 2.2125]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`21451 ms` LLM)
- **Rationale**: The earlier output truncated the solver differences. Read the relevant equations and compare trace ranges to determine which differences are steady setpoints and which are transient; that determines whether to transfer one chamber law or compose separate terms.
- **Alternatives Weighed**: `Transfer the entire nearest archived solver` (rejected: The steady channels favor batch 01, but the time-varying channels may encode a separate factor. Isolate the archived equations before transferring them.)

```bash
cd /tmp/hegel_continual_archive; cat alpha_m98_v2_c02_s00_solver.py; for f in *batch*solver.py; do echo "=== $f"; diff -u alpha_m98_v2_c02_s00_solver.py "$f" || true; done; echo '=== asset index'; python3 - <<'PY'
import json,pickle,numpy as np
p='/holdout_task/'
a=json.load(open(p+'public_asset_index.json'));print('index:', str(a)[:1800])
x=pickle.load(open(p+'public_batch.pkl','rb'))[0]; print('item fields',list(x));d=x['dataset']; c=d['calibration_load_cells']; print('sweeps',[(k,getattr(v,'shape',type(v).__name__)) for k,v in c['single_channel_sweeps'].items()]);
for k in ['train','test']:
 s=d[k]; print(k,'labs/strata',np.unique(s['lab_id'],return_counts=True),np.unique(s['stratum'],return_counts=True),'telemetry mean/std',np.mean(s['aux_chamber_telemetry'],0),np.std(s['aux_chamber_telemetry'],0))
r=json.load(open('archive_apparatus_log.json'))
for entry in r:
 x=np.array(entry['facility_telemetry_trace']);print(entry['completed_short_name'],'ranges',np.ptp(x,axis=0).round(3),'edges',x[[0,7,8,-1]].round(3).tolist())
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -26
```
- **Container Exit Code**: `0` (`481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mean/std [ 0.17290114 -0.04595283 -0.11608223 -0.31867301] [1.05334855 1.02348649 0.89457789 1.00711993]
test labs/strata (array([0, 1, 2, 3]), array([ 8,  9,  8, 11])) (array([0, 1, 2]), array([12, 11, 13])) telemetry mean/std [-0.1366334   0.13594158 -0.06585079 -0.04009536] [0.79567048 1.12157563 1.28634259 0.88900529]
alpha_m98_v2_c02_s00 ranges [2.057 0.274 0.483 2.151 0.596 0.23  2.101 0.554] edges [[0.109, -0.468, -0.586, 0.036, 1.182, 2.38, 0.098, 0.821], [0.095, -0.548, -0.756, 0.257, 1.004, 2.437, 0.189, 1.105], [-0.369, -0.379, -0.764, -0.391, 1.193, 2.516, -0.148, 1.145], [0.185, -0.324, -0.341, 0.079, 1.099, 2.505, -0.242, 1.375]]
alpha_m98_v2_c02_batch_01 ranges [1.9   0.441 0.41  1.923 0.469 0.264 2.018 0.467] edges [[0.563, -0.379, -0.44, 0.426, -1.24, -0.421, 0.792, -1.443], [-0.36, -0.32, -0.505, -0.378, -1.04, -0.347, -0.421, -1.323], [-0.767, -0.427, -0.428, -0.736, -0.983, -0.409, -0.757, -1.127], [0.809, -0.3, -0.219, 0.411, -1.452, -0.396, 0.749, -1.225]]
alpha_m98_v2_c02_batch_02 ranges [2.152 0.233 0.382 1.922 0.398 0.176 2.07  0.566] edges [[1.085, 2.579, -0.488, 0.857, 1.332, -0.321, 0.942, 1.2], [-0.917, 2.638, -0.14, -1.065, 1.082, -0.401, -0.872, 1.057], [-0.589, 2.536, -0.215, -0.828, 0.973, -0.455, -0.926, 1.066], [1.029, 2.479, -0.163, 0.806, 1.104, -0.332, 1.114, 1.042]]
alpha_m98_v2_c02_batch_03 ranges [2.053 0.351 0.489 1.985 0.694 0.273 2.173 0.386] edges [[0.623, -0.375, 0.148, 0.737, -0.928, -0.189, 0.891, -0.911], [-0.839, -0.407, -0.042, -0.864, -1.419, -0.367, -0.821, -1.07], [-0.618, -0.392, -0.019, -0.531, -1.228, -0.462, -0.627, -1.137], [0.6, -0.571, -0.046, 0.948, -1.478, -0.341, 0.748, -1.126]]
{
  "ok": true,
  "target": "autogenous_control_panel_deconfounding.align_control_panel",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_control_panel_deconfounding",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "aa502f6d83a30b6babf8dcdff2012cefc7cf8ce998e199d66c4221b48051b07f",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "observations_per_contributing_control",
          "op": ">=",
          "value": 2
        },
        {
          "field": "usable_control_features_per_center",
          "op": ">=",
          "value": 1
        },
        {
          "field": "estimated_shared_scanner_gain",
          "op": ">",
```

### Turn 4 — `execute_in_remote_docker` (`21180 ms` LLM)
- **Rationale**: Read just the deployment entrypoints and any upstream differences. This detects whether the archive candidates differ in chamber mechanics, fitting, or calibration, and determines what to transfer before testing the solver.
- **Alternatives Weighed**: `Copy batch 02 because the gate selected it` (rejected: The gate reported that row failed, and its steady telemetry differs sharply from the latest trace.)

```bash
cd /tmp/hegel_continual_archive; for f in *solver.py; do echo "=== $f"; sed -n '/^def _predict_chamber/,$p' "$f"; if ! grep -q '^def _predict_chamber' "$f"; then sed -n '/^def solve_batch/,$p' "$f"; fi; done; echo '=== Base fitter differences'; for f in *batch*solver.py; do echo "$f"; diff -u <(sed -n '3,/^def _fit/ p' alpha_m98_v2_c02_s00_solver.py) <(sed -n '3,/^def _fit/ p' "$f") | head -45; done
```
- **Container Exit Code**: `0` (`376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
03_solver.py
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _design(test, x_te, stress, dt, t, chamber=True)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
=== alpha_m98_v2_c02_s00_solver.py
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Blind campaign: no chamber diagnostics or continual archive are available.
    strain, rate = _design(test, x_te, stress, dt, t)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
=== Base fitter differences
alpha_m98_v2_c02_batch_01_solver.py
alpha_m98_v2_c02_batch_02_solver.py
alpha_m98_v2_c02_batch_03_solver.py
--- /dev/fd/63	2026-11-27 14:42:48.786619646 +0000
+++ /dev/fd/62	2026-11-27 14:42:48.786619646 +0000
@@ -29,11 +29,13 @@
     return lambda bridge: bridge @ coefficients[:4] + coefficients[4]
 
 
-def _design(site, x, stress, dt, t):
+def _design(site, x, stress, dt, t, *, chamber=False):
     past, hold = stress(site['past_channel_raw']), stress(site['hold_channel_raw'])
     tail = np.cumsum(dt[::-1])[::-1] - dt
     features = np.column_stack((np.ones(len(x)), x))
-    drift = features[:, None, :] * t[None, :, None]
+    # Saturating chamber creep has the same initial slope as benchtop creep.
+    drift_time = -np.expm1(-t) if chamber else t
+    drift = features[:, None, :] * drift_time[None, :, None]
 
     def matrices(taus):
         weights = -np.expm1(-dt[:, None] / taus) * np.exp(-tail[:, None] / taus)
```

### Turn 5 — `execute_in_remote_docker` (`37177 ms` LLM)
- **Rationale**: Build the standalone candidate. Compare affine and quadratic calibration out of fold to detect systematic bridge curvature that would change the calibrator, and hold out labs to detect whether benchtop estimation fails for specific groups.
- **Alternatives Weighed**: `Transfer batch 03's retained memory and saturating creep` (rejected: The latest trace's stable channels most closely match batch 01, which has a chamber law supported by verification coupons.)

```bash
set -e
cp /tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_01_solver.py /holdout_task/batch_solver.py
sed -i 's/# Diagnostics support a memory reset on entry to the chamber and a/# The latest apparatus trace matches the archived reset regime, with a/' /holdout_task/batch_solver.py
cat > /tmp/m98_public_check.py <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'batch_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test'];t=d['eval_times']; dt=d['past_dt']; c=d['calibration_load_cells']; raw=c['bridge_raw'];loads=c['true_channel_loads'];stress=m._stress_calibration(c)
print('calibration total residual quantiles',np.quantile(np.abs(stress(raw)-loads.sum(1)),[.5,.9,1]))
linear=np.c_[np.ones(len(raw)),raw]; quad=np.c_[linear,*[raw[:,i]*raw[:,j] for i in range(4) for j in range(i,4)]]
for a in [linear,quad]:
 target=loads.sum(1); e=[]
 for idx in np.array_split(np.arange(len(raw)),5):
  keep=np.setdiff1d(np.arange(len(raw)),idx);e.extend(a[idx]@np.linalg.lstsq(a[keep],target[keep],rcond=None)[0]-target[idx])
 print('calibration OOF columns, RMSE, max abs',a.shape[1],np.sqrt(np.mean(np.square(e))),np.max(np.abs(e)))
print('public estimated taus and weights',*(m.solve_batch(d)[k] for k in ['estimated_taus','hereditary_weights'])); print('eval times',t)
x,_=m._standardize(tr,te)
for group in np.unique(tr['lab_id']):
 mask=tr['lab_id']==group; fit=np.flatnonzero(~mask); hold=np.flatnonzero(mask)
 def sl(ix):return {k:v[ix] for k,v in tr.items()}
 taus,theta=m._fit(m._design(sl(fit),x[fit],stress,dt,t),tr['strain_curves_obs'][fit]-tr['strain_0'][fit,None],tr['instant_rate_obs'][fit]);a,b=m._design(sl(hold),x[hold],stress,dt,t)(taus)
 ec=np.abs(a@theta+tr['strain_0'][hold,None]-tr['strain_curves_obs'][hold]); er=np.abs(b@theta-tr['instant_rate_obs'][hold]); print('held lab',group,'taus',taus,'curve median/q90',np.quantile(ec,[.5,.9]),'rate median/q90',np.quantile(er,[.5,.9]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_public_check.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1425 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
calibration total residual quantiles [0.40081292 1.08931182 1.70984406]
calibration OOF columns, RMSE, max abs 5 2.155025250555156 10.852883298321625
calibration OOF columns, RMSE, max abs 15 9.621292655070523 54.7073654137313
public estimated taus and weights [0.050001   0.65914557] [ 0.00866785  1.15783728  0.60003803  0.10832166 -0.18239637  0.15808791
 -0.10361662]
eval times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
held lab 0 taus [0.050001   0.73250633] curve median/q90 [0.19743879 0.59338683] rate median/q90 [0.27772303 0.91430132]
held lab 1 taus [ 0.6269609  24.99997563] curve median/q90 [0.15856023 0.40070632] rate median/q90 [0.34983832 0.72535714]
held lab 2 taus [0.81177987 1.21481166] curve median/q90 [0.20228139 0.54120144] rate median/q90 [0.31136425 0.77811163]
held lab 3 taus [0.050001  0.6178973] curve median/q90 [0.20155605 0.47747751] rate median/q90 [0.37421898 0.94117382]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=deec38d96671cdff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_deec38d9 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32574 ms` LLM)
- **Rationale**: Inspect sweep layout and residuals by channel and load range to distinguish nonlinear bridge response from blocked extrapolation. That determines whether to change the calibrator or investigate spectrum identifiability with the existing affine fit.
- **Alternatives Weighed**: `Keep the schema-valid archived solver unchanged` (rejected: The relaxation estimates are unstable across held-out labs, which matters because both taus are required outputs.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived affine calibration and robust fit are sufficient for a stable two-branch estimate on this campaign.` | Witness: `Held-lab fits moved the fast tau to 0.050001 and the slow tau to 24.99998 in different folds; affine calibration had blocked out-of-fold maximum total-load error 10.85.` | Blanket Diagnosis: `Check the measured raw-to-load mapping before attributing boundary estimates to material physics; a generic quadratic inverse made the blocked calibration error worse.` | Sublated Pivot: `Retain the hereditary law and inspect the calibration sweep structure to identify a better-constrained bridge model.`

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,numpy as np
np.set_printoptions(precision=4,suppress=True,linewidth=150)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=d['calibration_load_cells']; r=c['bridge_raw']; y=c['true_channel_loads']; print('diagnostics',type(i['campaign_diagnostics']),'dataset keys',list(d),'cal keys',list(c)); print('shapes',r.shape,y.shape); print('first & last cal rows',np.c_[r,y][np.r_[:8,-8:0]]); print('sweeps',c['single_channel_sweeps']); print('nonzero per row',np.unique(np.count_nonzero(y,axis=1),return_counts=True)); print('range',np.quantile(np.c_[r,y],[0,.1,.5,.9,1],axis=0));
a=np.c_[np.ones(len(r)),r]; b=np.linalg.lstsq(a,y,rcond=None)[0]; e=a@b-y; print('inverse coef',b,'channel residual RMS',np.sqrt(np.mean(e**2,axis=0))); print('cal inverse residuals, consecutive groups');
for idx in np.array_split(np.arange(len(y)),8):print(idx[0],np.mean(e[idx],0),np.sqrt(np.mean(e[idx]**2,0)),np.mean(y[idx],0));
for name in ['train','test']:
 s=d[name];print(name,'raw min max and median',np.quantile(np.concatenate([s['hold_channel_raw'],s['past_channel_raw'].reshape(-1,4)]),[0,.5,1],axis=0),'hold head',s['hold_channel_raw'][:3]);
print('dt',d['past_dt']);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'transducer nonlinear calibration sensor bridge robust' 2>&1 | head -5
```
- **Container Exit Code**: `0` (`471 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4826]
 [ 1.0971  2.3315  1.4113  2.4166]] hold head [[0.0537 0.4565 0.8324 1.3329]
 [0.8677 0.5326 0.2557 1.66  ]
 [0.111  0.5694 0.663  0.7827]]
test raw min max and median [[-0.8599 -0.8667 -0.7701 -0.9733]
 [ 0.0749  0.6056  0.1172  0.5159]
 [ 1.3585  2.4787  1.3942  2.4554]] hold head [[0.2584 1.6122 0.7094 0.7696]
 [1.006  0.6346 0.5422 0.6102]
 [1.1312 0.4812 0.6269 0.1577]]
dt [0.35 0.35 0.4  0.45 0.5  0.55]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=deec38d96671cdff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_deec38d9 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30538 ms` LLM)
- **Rationale**: Known loads are clean predictors for the forward bridge. Compare robust channel-response bases out of fold to distinguish smooth nonlinearity from sparse raw-sensor outliers before choosing a mapping to invert.
- **Alternatives Weighed**: `Fit an unrestricted polynomial inverse` (rejected: The measured bridge contains extreme raw readings, and the quadratic inverse already worsened blocked prediction.)

```bash
cat > /tmp/m98_cal.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']
F={'linear':lambda x: [], 'square':lambda x:[x*x], 'signed_square':lambda x:[x*abs(x)], 'cubic':lambda x:[x**3], 'abs':lambda x:[abs(x)], 'mixed_square':lambda x:[x*x,x*abs(x)], 'odd_float':lambda x:[np.sign(x)*abs(x)**1.5], 'hinge':lambda x:[np.sign(x)*np.maximum(abs(x)-.5,0)]}
def design(x,j,k):return np.column_stack([np.ones(len(x)),x,*F[k](x[:,j])])
def fit(a,y):
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 for scale in (.5,.12,.025,.004,.001):
  b=least_squares(lambda b:a@b-y,b,jac=lambda b:a,loss='soft_l1',f_scale=scale).x
 return b
if __name__=='__main__':
 for key in F:
  oof=np.empty_like(Y); coeff=[]
  for j in range(4):
   a=design(X,j,key); coeff.append(fit(a,Y[:,j]));
   for fold in np.array_split(np.random.default_rng(23).permutation(len(X)),5):
    keep=np.setdiff1d(np.arange(len(X)),fold);oof[fold,j]=a[fold]@fit(a[keep],Y[keep,j])-Y[fold,j]
  print(key,'per channel q50/q90/q95',np.quantile(abs(oof),[.5,.9,.95],axis=0).round(5).tolist());
  if key in ('signed_square','cubic'):
   print('coef',np.asarray(coeff).round(5));e=np.column_stack([design(X,j,key)@coeff[j]-Y[:,j] for j in range(4)]); print('worst per channel',[list(zip(np.argsort(abs(v))[-6:],np.sort(abs(v))[-6:].round(4))) for v in e.T]); print('sweeps residual',e[:16].round(4).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_cal.py
```
- **Container Exit Code**: `0` (`1603 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
23, -0.2271, 0.9978], [-0.2513, -0.6079, -0.0619, 0.8577], [-0.2288, -0.4984, -0.0741, 0.6753], [-0.2115, -0.414, 0.0776, 0.535], [0.0097, 0.4196, -0.0802, -0.8695], [-14.647, 13.0089, -12.8774, 14.5198], [-14.8508, 12.2151, -12.8736, 15.3307], [-15.007, 11.6047, -12.871, 16.6505]]
cubic per channel q50/q90/q95 [[0.94521, 0.85072, 0.85758, 1.00592], [1.93067, 4.02677, 2.01699, 2.84533], [4.76952, 13.04992, 12.2671, 10.77888]]
coef [[-0.41854  0.84369 -0.1853   0.01694 -0.09814  0.09565]
 [-0.40753  0.19564  1.35738 -0.33698 -0.60966  0.22359]
 [-0.20329  0.01097  0.12244  0.77779 -0.29815  0.21106]
 [ 0.78885 -0.43478 -0.16132  0.11362  1.11835  0.91693]]
worst per channel [[(63, 0.7272), (22, 0.7924), (42, 0.8847), (13, 14.6638), (14, 14.956), (15, 15.1802)], [(56, 1.5118), (42, 1.5562), (44, 1.562), (15, 11.606), (14, 12.2156), (13, 13.0082)], [(50, 0.2604), (8, 0.2955), (37, 0.3951), (13, 12.8819), (14, 12.9282), (15, 12.9641)], [(72, 1.9946), (35, 2.1971), (56, 3.3205), (13, 14.7073), (14, 14.8811), (15, 17.8572)]]
sweeps residual [[-0.1937, -0.5377, -0.1209, 0.6989], [-0.3207, -0.5501, -0.1098, 0.674], [-0.277, -0.5662, -0.0958, 0.6412], [-0.4361, -0.5785, -0.0846, 0.6159], [-0.4066, -0.6066, -0.0892, 0.917], [-0.3391, -0.5898, -0.098, 0.7556], [-0.2512, -0.5208, -0.1093, 0.5458], [-0.1838, -0.5112, -0.1181, 0.3848], [-0.3255, -0.684, -0.2955, 0.8047], [-0.3087, -0.6049, -0.0916, 0.7137], [-0.2867, -0.5023, -0.1111, 0.5951], [-0.2698, -0.4232, 0.1366, 0.5039], [0.0608, 0.418, -0.0461, -1.8946], [-14.6638, 13.0082, -12.8819, 14.7073], [-14.956, 12.2156, -12.9282, 14.8811], [-15.1802, 11.606, -12.9641, 17.8572]]
abs per channel q50/q90/q95 [[0.95483, 0.97109, 0.85653, 1.07036], [1.93068, 3.5571, 2.01699, 2.77601], [14.47508, 13.04985, 18.43398, 13.08449]]
mixed_square per channel q50/q90/q95 [[0.9725, 1.00155, 0.85757, 0.9531], [1.96785, 4.47407, 2.86967, 2.90729], [14.31241, 13.04992, 12.53331, 13.08449]]
odd_float per channel q50/q90/q95 [[1.08269, 0.97108, 0.85758, 1.07125], [1.96788, 5.27932, 3.04274, 2.90729], [9.74769, 22.78038, 15.02223, 13.08449]]
hinge per channel q50/q90/q95 [[1.05119, 1.04207, 0.85758, 0.94046], [1.96788, 11.78819, 2.01699, 2.77601], [14.31241, 13.90787, 16.1498, 13.08449]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=deec38d96671cdff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_deec38d9 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`19176 ms` LLM)
- **Rationale**: Return a fresh Jacobian and rerun the basis comparison before deciding which physical mapping to invert. Compare telemetry channel by channel to detect a split regime match that would change the chamber law.
- **Alternatives Weighed**: `Choose a bridge basis from turn 7 residuals` (rejected: The robust optimizer can scale the returned Jacobian in place; returning the design array directly also changes the residual function, invalidating that comparison.)

```bash
sed -i 's/jac=lambda b:a,/jac=lambda b:a.copy(),/' /tmp/m98_cal.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_cal.py | grep -E 'q50/q90|^coef|sweeps residual'
python3 -I - <<'PY'
import json,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json';r=json.load(open(p)); y=np.asarray(r[-1]['facility_telemetry_trace']);
for e in r:
 x=np.asarray(e['facility_telemetry_trace']); print(e['completed_short_name'], 'means',x.mean(0).round(3),'stds',x.std(0).round(3),'RMSE',np.mean((x-y)**2,axis=0).__pow__(.5).round(3))
PY
```
- **Container Exit Code**: `0` (`1251 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-14.4516, 13.1994, -12.8102, 14.0597], [-14.418, 13.202, -12.8244, 13.9708]]
cubic per channel q50/q90/q95 [[0.06767, 0.09775, 0.06287, 0.10774], [0.37464, 0.4835, 0.2288, 0.43847], [0.45395, 0.65021, 0.28325, 0.54053]]
coef [[-9.71200e-02  8.15920e-01 -2.38510e-01  2.86200e-02  1.49860e-01
sweeps residual [[0.0372, -0.0103, 0.0006, -0.0797], [0.0103, -0.0046, 0.0004, -0.044], [0.0362, 0.0029, -0.0003, 0.0019], [-0.0001, 0.0087, -0.0004, 0.0371], [-0.0001, 0.0001, -0.0077, -0.0751], [0.0142, 0.0046, -0.0028, -0.0426], [0.0329, -0.0075, 0.0037, -0.0001], [0.0472, -0.0003, 0.0085, 0.0328], [-0.0227, -0.0019, -0.0496, -0.02], [0.0057, -0.0014, -0.0104, -0.0217], [0.0429, -0.0009, 0.013, -0.0241], [0.0715, -0.0004, 0.0564, -0.026], [-0.0146, 0.0001, 0.033, 0.0013], [-14.4912, 13.1992, -12.7875, 14.0787], [-14.461, 13.1981, -12.8142, 14.0744], [-14.4372, 13.1975, -12.835, 14.0485]]
abs per channel q50/q90/q95 [[0.10299, 0.16068, 0.0932, 0.0703], [0.40154, 0.57721, 0.29019, 0.57931], [0.66713, 0.75971, 0.36412, 0.84291]]
mixed_square per channel q50/q90/q95 [[0.07235, 0.12281, 0.07859, 0.07549], [0.36108, 0.46721, 0.24815, 0.52866], [0.5543, 0.57632, 0.31019, 0.80164]]
odd_float per channel q50/q90/q95 [[0.0683, 0.08716, 0.07421, 0.1186], [0.33173, 0.55621, 0.20738, 0.4597], [0.47789, 0.80139, 0.31067, 0.56818]]
hinge per channel q50/q90/q95 [[0.0871, 0.09632, 0.0761, 0.13639], [0.32166, 0.57676, 0.23967, 0.48409], [0.56594, 0.69348, 0.29638, 0.60161]]
alpha_m98_v2_c02_s00 means [ 0.012 -0.387 -0.61   0.045  1.098  2.455 -0.053  1.13 ] stds [0.648 0.079 0.136 0.661 0.155 0.071 0.701 0.129] RMSE [1.08  0.12  0.657 1.12  2.217 2.804 1.178 2.229]
alpha_m98_v2_c02_batch_01 means [ 0.025 -0.391 -0.447  0.032 -1.198 -0.379  0.035 -1.235] stds [0.636 0.104 0.115 0.645 0.146 0.061 0.678 0.136] RMSE [0.802 0.145 0.501 0.821 0.261 0.106 0.827 0.233]
alpha_m98_v2_c02_batch_02 means [ 0.07   2.539 -0.228  0.007  1.111 -0.371  0.014  1.112] stds [0.753 0.061 0.097 0.671 0.125 0.061 0.708 0.14 ] RMSE [0.509 2.934 0.318 0.443 2.228 0.066 0.418 2.213]
alpha_m98_v2_c02_batch_03 means [ 0.032 -0.392  0.03   0.088 -1.107 -0.346  0.027 -1.092] stds [0.639 0.102 0.129 0.696 0.18  0.079 0.678 0.093] RMSE [0. 0. 0. 0. 0. 0. 0. 0.]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=deec38d96671cdff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_deec38d9 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`30088 ms` LLM)
- **Rationale**: Compare explicit mixed-load terms on held-out sweeps to determine which forward bridge structure is worth inverting. Read compact log metadata to check whether entries describe the completed campaign or the following one; that changes the interpretation of the latest chamber trace.
- **Alternatives Weighed**: `Invert the channel-wise cubic fit now` (rejected: Its held-out residuals remain large relative to the sweep noise and may reflect missing cross-channel terms.)

```bash
cat > /tmp/m98_poly.py <<'PY'
import numpy as np, pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']
valid=np.max(abs(Y),axis=1)<6
folds=np.array_split(np.random.default_rng(23).permutation(np.flatnonzero(valid)),5)
def design(x,j,k):
 z=x[:,j]; Q=[x[:,a]*x[:,b] for a,b in combinations_with_replacement(range(4),2)]; C=[x[:,a]*x[:,b]*x[:,d] for a,b,d in combinations_with_replacement(range(4),3)]; base=[np.ones(len(x)),*x.T]; extras={'lin':[], 'diag':[z**3], 'pair':Q+[z**3], 'joint':Q+C, 'odd':[v**3 for v in x.T]+[x[:,a]*x[:,b] for a in range(4) for b in range(a+1,4)], 'mixpoly':Q+[v**3 for v in x.T], 'odd3':C, 'by_channel':Q+[z*z*v for v in x.T]+[z*v*v for i,v in enumerate(x.T) if i!=j]};return np.column_stack(base+extras[k])
for key in ['lin','diag','pair','joint','odd','mixpoly','odd3','by_channel']:
 e=np.zeros_like(Y); E=np.zeros_like(Y)
 for j in range(4):
  a=design(X,j,key); b=np.linalg.lstsq(a[valid],Y[valid,j],rcond=None)[0]; E[:,j]=a@b-Y[:,j]
  for ids in folds:
   keep=np.setdiff1d(np.flatnonzero(valid),ids); b=np.linalg.lstsq(a[keep],Y[keep,j],rcond=None)[0]; b=least_squares(lambda v:a[keep]@v-Y[keep,j],b,loss='soft_l1',f_scale=.005).x;e[ids,j]=a[ids]@b-Y[ids,j]
 print(key,'oof',np.quantile(abs(e[valid]),[.5,.9],axis=0).round(5),'in rms',np.sqrt(np.mean(E[valid]**2,axis=0)).round(5))
 if key=='joint':
  for j in range(4):
   a=design(X,j,key);b=np.linalg.lstsq(a[valid],Y[valid,j],rcond=None)[0];print('joint coef',j,b.round(3))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_poly.py
python3 -I - <<'PY'
import json,pickle,numpy as np
r=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('archive header', [{k:v for k,v in e.items() if k!='facility_telemetry_trace'} for e in r]); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];print('item metadata',{k:str(v)[:250] for k,v in i.items() if k!='dataset'}); print('test fields',list(i['dataset']['test']))
PY
```
- **Container Exit Code**: `0` (`1143 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8.0e-05 9.0e-05]
 [1.9e-04 1.7e-04 1.9e-04 1.9e-04]] in rms [8.e-05 8.e-05 9.e-05 8.e-05]
joint oof [[0.0001  0.0001  0.00012 0.00012]
 [0.00024 0.00026 0.00035 0.00032]] in rms [7.e-05 7.e-05 8.e-05 7.e-05]
joint coef 0 [-0.12   0.79  -0.253  0.     0.126  0.    -0.174  0.    -0.    -0.
 -0.     0.    -0.     0.    -0.     0.142  0.    -0.    -0.    -0.
 -0.     0.    -0.     0.     0.     0.     0.    -0.     0.    -0.
  0.     0.    -0.     0.    -0.   ]
joint coef 1 [ 0.15   0.208  1.3   -0.416  0.    -0.    -0.     0.     0.    -0.
 -0.286  0.    -0.     0.    -0.     0.     0.    -0.     0.     0.
  0.    -0.     0.    -0.    -0.     0.234  0.    -0.     0.    -0.
 -0.     0.    -0.     0.    -0.   ]
joint coef 2 [-0.1    0.     0.131  0.82  -0.262  0.     0.     0.     0.     0.
  0.    -0.     0.    -0.18   0.     0.    -0.     0.     0.     0.
 -0.    -0.    -0.    -0.    -0.    -0.    -0.     0.    -0.     0.
 -0.     0.148  0.    -0.     0.   ]
joint coef 3 [ 0.13  -0.41   0.     0.205  1.28   0.     0.     0.    -0.282 -0.
 -0.    -0.    -0.     0.     0.    -0.    -0.     0.     0.    -0.
 -0.    -0.    -0.     0.     0.    -0.    -0.     0.    -0.     0.
  0.    -0.     0.    -0.     0.23 ]
odd oof [[8.0e-05 6.0e-05 7.0e-05 9.0e-05]
 [1.9e-04 1.7e-04 1.7e-04 2.0e-04]] in rms [9.e-05 8.e-05 9.e-05 8.e-05]
mixpoly oof [[8.0e-05 7.0e-05 8.0e-05 9.0e-05]
 [2.1e-04 1.9e-04 1.9e-04 2.1e-04]] in rms [8.e-05 8.e-05 9.e-05 8.e-05]
odd3 oof [[0.07789 0.09072 0.05044 0.09443]
 [0.3544  0.51248 0.30432 0.5465 ]] in rms [0.11365 0.15674 0.07275 0.15528]
by_channel oof [[7.0e-05 7.0e-05 8.0e-05 1.0e-04]
 [1.8e-04 1.9e-04 2.2e-04 2.6e-04]] in rms [7.e-05 8.e-05 9.e-05 8.e-05]
archive header [{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c02_s00'}, {'completed_batch_index': 1, 'completed_short_name': 'alpha_m98_v2_c02_batch_01'}, {'completed_batch_index': 2, 'completed_short_name': 'alpha_m98_v2_c02_batch_02'}, {'completed_batch_index': 3, 'completed_short_name': 'alpha_m98_v2_c02_batch_03'}]
item metadata {'campaign_diagnostics': 'None'}
test fields ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'aux_chamber_telemetry']

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=deec38d96671cdff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_deec38d9 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`32128 ms` LLM)
- **Rationale**: Fit the sparse forward law and measure whether inversion stabilizes the held-lab spectrum. Compare current coupon telemetry with the archive because the final log entry describes a completed campaign, not a direct observation of deployment.
- **Alternatives Weighed**: `Keep the full third-order polynomial` (rejected: The fitted coefficients show a repeated cyclic structure, and unrestricted cubic terms increased held-out raw error.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/batch_solver.py'); s=p.read_text(); i=s.index('def _stress_calibration'); j=s.index('\ndef ',i+4)
s=s[:i]+'''def _stress_calibration(calibration):
    """Fit the cyclic forward bridge and invert it to recover axial load."""
    loads = calibration['true_channel_loads']
    raw = calibration['bridge_raw']
    prev = np.roll(loads, 1, axis=-1)
    nxt = np.roll(loads, -1, axis=-1)
    a = np.stack((np.ones_like(loads), loads, prev, nxt, loads * nxt, loads**3), axis=-1)
    coefs = []
    for j in range(4):
        design = a[:, j, :]
        y = raw[:, j]
        b = np.linalg.lstsq(design, y, rcond=None)[0]
        for scale in (0.3, 0.01, 0.0003):
            b = least_squares(lambda c: design @ c - y, b, loss='soft_l1', f_scale=scale).x
        e = design @ b - y
        mad = np.median(np.abs(e - np.median(e)))
        keep = np.abs(e - np.median(e)) < max(8 * mad, 0.001)
        coefs.append(np.linalg.lstsq(design[keep], y[keep], rcond=None)[0])
    offset, gain, left, right, cross, cubic = np.asarray(coefs).T
    ids = np.arange(4)
    before, after = (ids - 1) % 4, (ids + 1) % 4
    linear = np.diag(gain)
    linear[ids, before] = left
    linear[ids, after] = right

    def stress(bridge):
        y = bridge - offset
        x = y @ np.linalg.inv(linear).T
        for _ in range(20):
            nxt = x[..., after]
            residual = x @ linear.T + cross * x * nxt + cubic * x**3 - y
            jac = np.broadcast_to(linear, x.shape[:-1] + (4, 4)).copy()
            jac[..., ids, ids] += cross * nxt + 3 * cubic * x**2
            jac[..., ids, after] += cross * x
            step = np.linalg.solve(jac, residual[..., None])[..., 0]
            x -= np.clip(step, -0.8, 0.8)
            if np.max(np.abs(step)) < 1e-10:
                break
        return x.sum(axis=-1)

    return stress

''' + s[j+1:];p.write_text(s)
PY
cat /holdout_task/batch_solver.py | sed -n '/^def _fit/,$p'
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_public_check.py
python3 -I - <<'PY'
import json,pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; r=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for name in ('train','test'):
 t=x[name]['aux_chamber_telemetry']; print(name,'telemetry',t.shape,t.mean(0).round(3),t.std(0).round(3));print(t[:3].round(3))
for a in r:
 t=np.array(a['facility_telemetry_trace']);print(a['completed_short_name'],t[:2].round(3))
PY
```
- **Container Exit Code**: `0` (`1218 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
in['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _predict_chamber(test, x_te, stress, t, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
calibration total residual quantiles [1.04223843e-04 3.33022239e-04 1.28802596e+00]
calibration OOF columns, RMSE, max abs 5 2.155025250555156 10.852883298321625
calibration OOF columns, RMSE, max abs 15 9.621292655070523 54.7073654137313
public estimated taus and weights [0.58630756 3.77780577] [ 0.6945827   0.44652951  0.364204    0.11307699 -0.17556842  0.16962934
 -0.1040043 ]
eval times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
held lab 0 taus [0.58577956 3.72494251] curve median/q90 [0.00866227 0.0211302 ] rate median/q90 [0.00370946 0.01002786]
held lab 1 taus [0.58688338 3.77484384] curve median/q90 [0.00807543 0.02413894] rate median/q90 [0.00439245 0.01324697]
held lab 2 taus [0.58514952 3.75610962] curve median/q90 [0.00799728 0.02093904] rate median/q90 [0.00358657 0.00835298]
held lab 3 taus [0.58701733 3.82794262] curve median/q90 [0.00922326 0.02136516] rate median/q90 [0.00629398 0.01282378]
train telemetry (72, 4) [ 0.173 -0.046 -0.116 -0.319] [1.053 1.023 0.895 1.007]
[[ 0.684  0.143  0.307 -0.731]
 [-0.169 -0.693  0.688  0.122]
 [ 2.011  0.362  0.198 -0.897]]
test telemetry (36, 4) [-0.137  0.136 -0.066 -0.04 ] [0.796 1.122 1.286 0.889]
[[ 0.095 -0.165 -0.611 -0.189]
 [-0.659 -1.123  0.413 -0.197]
 [-1.597  1.181  2.432 -1.286]]
alpha_m98_v2_c02_s00 [[ 0.109 -0.468 -0.586  0.036  1.182  2.38   0.098  0.821]
 [ 0.376 -0.274 -0.518  0.645  1.209  2.538  0.274  1.109]]
alpha_m98_v2_c02_batch_01 [[ 0.563 -0.379 -0.44   0.426 -1.24  -0.421  0.792 -1.443]
 [ 0.648 -0.384 -0.465  1.035 -1.139 -0.269  0.823 -1.175]]
alpha_m98_v2_c02_batch_02 [[ 1.085  2.579 -0.488  0.857  1.332 -0.321  0.942  1.2  ]
 [ 0.776  2.44  -0.298  0.741  1.08  -0.455  0.851  1.444]]
alpha_m98_v2_c02_batch_03 [[ 0.623 -0.375  0.148  0.737 -0.928 -0.189  0.891 -0.911]
 [ 0.685 -0.22   0.275  0.609 -0.784 -0.406  0.578 -1.097]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=176e6cbf4dac69ce (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_176e6cbf (directory).
```

### Turn 11 — `execute_in_remote_docker` (`26950 ms` LLM)
- **Rationale**: The bridge correction stabilized both relaxation times across labs. Compare archived equation differences with steady and transient telemetry signatures to decide whether to retain batch 01's full law or transfer only the supported terms.
- **Alternatives Weighed**: `Finalize the batch 01 deployment law on the strength of the benchtop validation` (rejected: Benchtop observations verify calibration and parameter recovery but cannot distinguish chamber memory or creep laws.)

```bash
cat /holdout_task/public_asset_index.json; for n in 01 02 03; do echo "=== $n"; diff -u /tmp/hegel_continual_archive/alpha_m98_v2_c02_s00_solver.py /tmp/hegel_continual_archive/alpha_m98_v2_c02_batch_${n}_solver.py || true; done; python3 -I - <<'PY'
import json,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'; r=json.load(open(p));
for e in r:
 a=np.array(e['facility_telemetry_trace']); print(e['completed_short_name'],a.shape,'half means',np.stack([v.mean(0) for v in np.array_split(a,2)]).round(3).tolist(),'correlation',np.corrcoef(a.T).round(2).tolist())
PY
```
- **Container Exit Code**: `0` (`431 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.523, 1.159, 2.437, -0.665, 1.189]] correlation [[1.0, 0.1, 0.22, 0.97, -0.57, 0.38, 0.97, -0.2], [0.1, 1.0, 0.31, 0.07, 0.25, 0.19, 0.05, 0.05], [0.22, 0.31, 1.0, 0.21, -0.1, 0.21, 0.14, 0.17], [0.97, 0.07, 0.21, 1.0, -0.45, 0.38, 0.96, -0.19], [-0.57, 0.25, -0.1, -0.45, 1.0, -0.18, -0.48, -0.0], [0.38, 0.19, 0.21, 0.38, -0.18, 1.0, 0.4, 0.19], [0.97, 0.05, 0.14, 0.96, -0.48, 0.4, 1.0, -0.25], [-0.2, 0.05, 0.17, -0.19, -0.0, 0.19, -0.25, 1.0]]
alpha_m98_v2_c02_batch_01 (16, 8) half means [[0.425, -0.413, -0.487, 0.451, -1.191, -0.354, 0.464, -1.271], [-0.375, -0.369, -0.407, -0.386, -1.204, -0.404, -0.395, -1.198]] correlation [[1.0, -0.02, 0.06, 0.96, -0.48, 0.36, 0.99, -0.16], [-0.02, 1.0, -0.23, -0.04, 0.0, 0.38, -0.02, -0.29], [0.06, -0.23, 1.0, -0.02, -0.35, -0.31, 0.04, -0.15], [0.96, -0.04, -0.02, 1.0, -0.41, 0.37, 0.97, -0.17], [-0.48, 0.0, -0.35, -0.41, 1.0, 0.16, -0.45, -0.06], [0.36, 0.38, -0.31, 0.37, 0.16, 1.0, 0.31, 0.09], [0.99, -0.02, 0.04, 0.97, -0.45, 0.31, 1.0, -0.17], [-0.16, -0.29, -0.15, -0.17, -0.06, 0.09, -0.17, 1.0]]
alpha_m98_v2_c02_batch_02 (16, 8) half means [[0.124, 2.546, -0.245, 0.085, 1.099, -0.383, 0.071, 1.142], [0.017, 2.532, -0.211, -0.071, 1.123, -0.358, -0.042, 1.082]] correlation [[1.0, -0.49, -0.18, 0.97, 0.62, 0.45, 0.96, 0.3], [-0.49, 1.0, -0.05, -0.47, -0.04, 0.06, -0.57, -0.39], [-0.18, -0.05, 1.0, -0.18, 0.02, -0.22, -0.17, -0.18], [0.97, -0.47, -0.18, 1.0, 0.58, 0.43, 0.97, 0.27], [0.62, -0.04, 0.02, 0.58, 1.0, 0.3, 0.58, 0.11], [0.45, 0.06, -0.22, 0.43, 0.3, 1.0, 0.42, -0.13], [0.96, -0.57, -0.17, 0.97, 0.58, 0.42, 1.0, 0.29], [0.3, -0.39, -0.18, 0.27, 0.11, -0.13, 0.29, 1.0]]
alpha_m98_v2_c02_batch_03 (16, 8) half means [[-0.233, -0.404, 0.078, -0.198, -1.061, -0.356, -0.233, -1.074], [0.298, -0.38, -0.019, 0.374, -1.153, -0.337, 0.286, -1.109]] correlation [[1.0, 0.13, 0.4, 0.97, 0.24, 0.4, 0.98, -0.13], [0.13, 1.0, 0.15, -0.02, 0.34, -0.07, 0.07, 0.43], [0.4, 0.15, 1.0, 0.33, 0.42, -0.07, 0.38, -0.05], [0.97, -0.02, 0.33, 1.0, 0.13, 0.42, 0.98, -0.23], [0.24, 0.34, 0.42, 0.13, 1.0, 0.08, 0.19, -0.01], [0.4, -0.07, -0.07, 0.42, 0.08, 1.0, 0.54, 0.14], [0.98, 0.07, 0.38, 0.98, 0.19, 0.54, 1.0, -0.14], [-0.13, 0.43, -0.05, -0.23, -0.01, 0.14, -0.14, 1.0]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=176e6cbf4dac69ce (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_176e6cbf (directory).
```

### Turn 12 — `execute_in_remote_docker` (`27719 ms` LLM)
- **Rationale**: Read only the archived chamber equations to avoid another truncated comparison. Validate total load after inversion on held-out clean sweeps: if the inverse amplifies the small forward residuals, revise inversion before finalizing the chamber predictor.
- **Alternatives Weighed**: `Select batch 01 by excluding the latest archive entry from the regime candidates` (rejected: That selects the closest older trace, but does not establish that batch 01's reset and logarithmic creep apply when batch 03's chamber law is also available.)

```bash
set -e
cd /tmp/hegel_continual_archive
for f in *solver.py; do echo "=== $f"; grep -A20 '^def _predict_chamber' "$f" || grep -A19 '^def solve_batch' "$f"; done
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; r=c['bridge_raw']; y=c['true_channel_loads']; keep=np.max(abs(r),axis=1)<6
idx=np.flatnonzero(keep); folds=np.array_split(np.random.default_rng(23).permutation(idx),5); errors=[]
for fold in folds:
 tr=np.setdiff1d(idx,fold); stress=m._stress_calibration({'bridge_raw':r[tr],'true_channel_loads':y[tr]}); errors.extend(abs(stress(r[fold])-y[fold].sum(1)))
print('held-out clean total-load q50/q90/max', np.quantile(errors,[.5,.9,1])); print('corrupted rows',np.flatnonzero(~keep))
log=json.load(open('archive_apparatus_log.json')); a=np.array([e['facility_telemetry_trace'] for e in log]); print('DC estimates for telemetry channels 1,2,4,5,7',np.mean(a,axis=1)[:,[1,2,4,5,7]].round(3)); print('transient mean-absolute amplitude for channels 0,3,6',np.mean(abs(a),axis=1)[:,[0,3,6]].round(3))
PY
```
- **Container Exit Code**: `0` (`511 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rain, rate = _design(test, x_te, stress, dt, t)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
=== alpha_m98_v2_c02_batch_03_solver.py
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _design(test, x_te, stress, dt, t, chamber=True)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
=== alpha_m98_v2_c02_s00_solver.py
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_tr, x_te = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    tr_design = _design(train, x_tr, stress, dt, t)
    taus, theta = _fit(tr_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # Blind campaign: no chamber diagnostics or continual archive are available.
    strain, rate = _design(test, x_te, stress, dt, t)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain @ theta,
            'initial_rate_pred': rate @ theta}
held-out clean total-load q50/q90/max [0.00011131 0.00033074 0.00059174]
corrupted rows [13 14 15]
DC estimates for telemetry channels 1,2,4,5,7 [[-0.387 -0.61   1.098  2.455  1.13 ]
 [-0.391 -0.447 -1.198 -0.379 -1.235]
 [ 2.539 -0.228  1.111 -0.371  1.112]
 [-0.392  0.03  -1.107 -0.346 -1.092]]
transient mean-absolute amplitude for channels 0,3,6 [[0.564 0.578 0.612]
 [0.551 0.579 0.596]
 [0.693 0.619 0.646]
 [0.574 0.638 0.6  ]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=176e6cbf4dac69ce (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_176e6cbf (directory).
```

### Turn 13 — `execute_in_remote_docker` (`37505 ms` LLM)
- **Rationale**: Confirm the earlier regime match and verify the exact deliverable in isolated Python. An independent curve calculation and a zero-time derivative check can detect different mistakes in the two deployment outputs; neither is a substitute for unavailable chamber labels.
- **Alternatives Weighed**: `Use batch 03’s saturating creep` (rejected: Batch 03 supplies the reference trace in the instructed comparison; its match to itself is not evidence for transferring its chamber law.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_apparatus_log.json') as f: log=json.load(f)
a=np.array([e['facility_telemetry_trace'] for e in log]); delta=a[:-1]-a[-1]
print('prior-trace distances: steady channels 1,4,5,7 / all channels:')
for e, z in zip(log[:-1],delta): print(e['completed_short_name'],np.sqrt(np.mean(z[:,[1,4,5,7]]**2)),np.sqrt(np.mean(z**2)))
out=m.solve_batch(d); taus,theta=out['estimated_taus'],out['hereditary_weights']; assert 0.05<taus[0]<taus[1]<25 and all(np.isfinite(v).all() for v in out.values())
print('taus',taus,'weights',theta)
tr,te=d['train'],d['test']; _,x=m._standardize(tr,te); stress=m._stress_calibration(d['calibration_load_cells']); h=stress(te['hold_channel_raw']); drift=theta[2]+x@theta[3:]; t=d['eval_times']; expected=te['strain_0'][:,None]+(h[:,None,None]*theta[:2]*(1-np.exp(-t[None,:,None]/taus))).sum(2)+drift[:,None]*np.log1p(t); np.testing.assert_allclose(out['predicted_strain'],expected,atol=1e-12,rtol=0)
eps=1e-6; q,_=m._predict_chamber(te,x,stress,np.array([0.,eps]),taus,theta); np.testing.assert_allclose(q[:,0],te['strain_0'],atol=1e-12,rtol=0); err=abs((q[:,1]-q[:,0])/eps-out['initial_rate_pred']); print('max zero-time rate consistency error',err.max()); assert err.max()<1e-5
for site in (tr,te):
 for k in ('hold_channel_raw','past_channel_raw'):
  y=stress(site[k]); print('stress percentiles',len(y),k,np.quantile(y,[0,.1,.5,.9,1])); assert np.isfinite(y).all()
PY
```
- **Container Exit Code**: `0` (`1118 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
prior-trace distances: steady channels 1,4,5,7 / all channels:
alpha_m98_v2_c02_s00 2.10729615485846 1.658425139044758
alpha_m98_v2_c02_batch_01 0.19683540491121637 0.5486407914935099
alpha_m98_v2_c02_batch_02 2.149088895318854 1.5493862708943622
taus [0.58630756 3.77780577] weights [ 0.6945827   0.44652951  0.364204    0.11307699 -0.17556842  0.16962934
 -0.1040043 ]
max zero-time rate consistency error 5.0175836356913806e-06
stress percentiles 72 hold_channel_raw [2.18135427 2.54405637 3.41924833 4.26803194 4.48624461]
stress percentiles 72 past_channel_raw [-1.64084221 -0.45611919  0.99996598  2.16452484  4.00783562]
stress percentiles 36 hold_channel_raw [2.27984386 2.75693713 3.40162058 4.0707465  4.60725922]
stress percentiles 36 past_channel_raw [-1.71016167 -0.26370249  1.0392739   2.41513444  3.38411684]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=176e6cbf4dac69ce (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_176e6cbf (directory).
```

### Turn 14 — `submit_completed_task` (`18326 ms` LLM)
- **Rationale**: Held-out calibration sweeps and lab folds support replacing the affine bridge. The exact final artifact passed isolated schema, finiteness, and initial-rate consistency checks. These checks do not establish blind chamber accuracy.
- **Alternatives Weighed**: `Transfer batch 02’s benchtop deployment law` (rejected: Batch 02’s steady-channel trace distance was 2.149, compared with 0.197 for batch 01.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Worst held-out-lab 90th-percentile absolute errors were 0.0241 for strain and 0.0133 for initial rate. The blind deployment law was selected from the apparatus archive; no chamber labels were available to validate it directly.`

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
