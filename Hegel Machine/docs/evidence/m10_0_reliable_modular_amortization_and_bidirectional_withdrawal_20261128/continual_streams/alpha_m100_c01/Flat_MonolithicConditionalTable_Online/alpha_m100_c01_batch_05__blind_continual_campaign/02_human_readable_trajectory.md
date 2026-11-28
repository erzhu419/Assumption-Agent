# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m100_c01_batch_05__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_viscoelastic_chamber_hereditary_mechanics` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_alpha` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `364,146` (`352,843` / `121,472` / `11,303`) |
| **Total Episode Wall Time (s)** | `372.58s` |

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

### Turn 1 — `execute_in_remote_docker` (`12957 ms` LLM)
- **Rationale**: Inventory the public inputs and read bounded archive excerpts to determine which chamber laws and reusable fitting primitives can be validated.
- **Alternatives Weighed**: `Use the benchtop law for deployment by default` (rejected: The campaign can alter chamber physics, so inspect the public diagnostics and prior telemetry before choosing a law.)

```bash
ls -lah /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json; for f in /tmp/hegel_continual_archive/*.json; do echo "--- $f"; wc -c "$f"; head -c 9500 "$f"; echo; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment robust IRLS variable projection'
```
- **Container Exit Code**: `0` (`383 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
uting_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`17585 ms` LLM)
- **Rationale**: Read the prior solver, current public arrays, and telemetry structure to distinguish reusable calibration and fitting code from campaign-specific assumptions.
- **Alternatives Weighed**: `Copy the previous solver unchanged` (rejected: The current campaign is blind and may differ from the previous chamber regime; inspect what the previous solver assumed and what the current telemetry supports.)

```bash
cd /holdout_task; sed -n '1,165p' validate_public_schema.py; cat /tmp/hegel_continual_archive/alpha_m100_c01_batch_04_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'; a=json.load(open(p)); print(type(a)); print([(k,type(v).__name__,len(v) if hasattr(v,'__len__') else None) for k,v in a.items()] if isinstance(a,dict) else len(a));
for v in (a.values() if isinstance(a,dict) else [a]):
 if isinstance(v,list) and v: print('first',str(v[0])[:2200], '\nlast',str(v[-1])[:6200]);
 elif not isinstance(v,list): print(str(v)[:1500]);
i=pickle.load(open('public_batch.pkl','rb'))[0]; print(i.keys(),i.get('campaign_diagnostics')); d=i['dataset']; print(d.keys());
for name in ('train','test'):
 for k,v in d[name].items(): print(name,k,type(v).__name__,getattr(v,'shape',None),str(v[:2] if isinstance(v,np.ndarray) else v)[:280]);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -50
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
          "value": 0
        },
        {
          "field": "observations_per_training_center_class_cell",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_control_panel",
          "signature": "align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
        "autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_control_panel_deconfounding",
      "reference_source_txt": null
```

### Turn 3 — `execute_in_remote_docker` (`17087 ms` LLM)
- **Rationale**: The previous output truncated the telemetry. Read it alongside the archived chamber equations to determine whether the current apparatus state supports reusing or updating a prior transfer law.
- **Alternatives Weighed**: `Reuse the latest archived chamber coefficients unchanged` (rejected: The current batch has no verification coupons, and the preceding campaign may have operated in a different apparatus regime.)

```bash
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'; a=json.load(open(p)); print('root', list(a) if isinstance(a,dict) else type(a));
for k,v in (a.items() if isinstance(a,dict) else [('records',a)]):
 if isinstance(v,list):
  print(k,len(v));
  for r in v: print(json.dumps(r,separators=(',',':'))[:1850]);
 else: print(k,str(v)[:700]);
PY
for f in /tmp/hegel_continual_archive/*solver.py; do echo ---$(basename "$f"); grep -nE '^(def |[A-Z_]+ *=)|telemetry|chamber|equilibr|_LAW' "$f" | tail -27; done
```
- **Container Exit Code**: `0` (`324 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9,-0.68868,-0.32707,0.2797,0.65048,-0.8098,-0.68171,-0.40091]]}
---alpha_m100_c01_batch_01_solver.py
6:def _features(train, test):
28:def _bridge(raw, loads):
33:def _inputs(split, x, bridge):
39:def _design(taus, past, hold, features, dt, times):
50:def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
91:# Campaign chamber calibration from the public verification coupons. The
93:_CHAMBER_PROJECTION = np.array([0.188729, 2.002465])
96:def _predict_chamber(taus, theta, hold, features, times, projection):
103:def solve_batch(dataset: dict) -> dict:
114:    curve, rate = _predict_chamber(taus, theta, hold, features, times,
---alpha_m100_c01_batch_02_solver.py
6:def _features(train, test):
28:def _bridge(raw, loads):
33:def _inputs(split, x, bridge):
39:def _design(taus, past, hold, features, dt, times):
50:def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
103:_CHAMBER_DRIFT_TIME = 1.0116511228821314
106:def solve_batch(dataset: dict) -> dict:
118:    # The chamber replaces the linear drift ramp with exponential saturation.
---alpha_m100_c01_batch_03_solver.py
6:def _features(train, test):
28:def _design(taus, past, hold, features, dt, times):
39:def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
91:def _bridge_forward(loads, bridge):
96:def _fit_bridge(raw, loads):
117:def _load_from_raw(raw, bridge):
132:def _calibrated_inputs(split, features, bridge):
138:def solve_batch(dataset: dict) -> dict:
149:    # The latest apparatus record has both known chamber indicators at idle.
---alpha_m100_c01_batch_04_solver.py
6:def _features(train, test):
28:def _design(taus, past, hold, features, dt, times):
39:def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
91:def _bridge_forward(loads, bridge):
96:def _fit_bridge(raw, loads):
117:def _load_from_raw(raw, bridge):
132:def _calibrated_inputs(split, features, bridge):
139:_CHAMBER_PROJECTION = np.array([0.188729, 2.002465])
142:def _predict_chamber(taus, theta, hold, features, times, projection):
149:def solve_batch(dataset: dict) -> dict:
160:    # In this regime the chamber resets the hereditary state at entry.
163:    curve, rate = _predict_chamber(taus, theta, hold, features, times,
---alpha_m100_c01_s00_solver.py
6:def _features(train, test):
28:def _bridge(raw, loads):
33:def _inputs(split, x, bridge):
39:def _design(taus, past, hold, features, dt, times):
50:def _fit(past, hold, features, dt, times, curve_obs, rate_obs):
91:def solve_batch(dataset: dict) -> dict:
```

### Turn 4 — `execute_in_remote_docker` (`19953 ms` LLM)
- **Rationale**: Summarize each record compactly to determine whether the latest entry describes the current campaign or only its predecessor, and compare sustained settings with the archived equations before choosing a transfer law.
- **Alternatives Weighed**: `Match the full raw telemetry traces` (rejected: Oscillatory channels can differ in phase even when the sustained apparatus settings agree; compare channel medians and spread across known regimes first.)

```bash
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/'; logs=json.load(open(p+'archive_apparatus_log.json'))
for r in logs:
 a=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_index'],r['completed_campaign_short_name'],a.shape,'median',np.round(np.median(a,axis=0),3),'sd',np.round(a.std(0),3)); print({k:v for k,v in r.items() if k not in ('facility_telemetry_trace','completed_campaign_index','completed_campaign_short_name')})
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; print('keys',d.keys()); print('dt',d['past_dt'],'times',d['eval_times']); print('train',[(k,np.shape(v)) for k,v in d['train'].items()]); print('test',[(k,np.shape(v)) for k,v in d['test'].items()]); print('bridge sweeps',[(k,np.shape(v)) for k,v in d['calibration_load_cells']['single_channel_sweeps'].items()])
PY
for v in s00 batch_01 batch_02 batch_03 batch_04; do echo ---$v; sed -n '/^def solve_batch/,$p' /tmp/hegel_continual_archive/alpha_m100_c01_${v}_solver.py; done; sed -n '1,138p' /tmp/hegel_continual_archive/alpha_m100_c01_batch_04_solver.py
```
- **Container Exit Code**: `0` (`392 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
idual(q):
        e = residual(q).reshape(n, k + 1)
        e[:, -1] *= 4.0
        return e.ravel()

    result = least_squares(rate_weighted_residual, result.x, loss='soft_l1',
                           f_scale=.008, x_scale='jac', max_nfev=200,
                           bounds=(np.r_[bounds[0], np.full(7, -np.inf)],
                                   np.r_[bounds[1], np.full(7, np.inf)]))
    return unpack(result.x[:2]), result.x[2:]


def _bridge_forward(loads, bridge):
    offset, linear, mixed, cubic = bridge
    return offset + loads @ linear + mixed * loads * np.roll(loads, -1, axis=-1) + cubic * loads**3


def _fit_bridge(raw, loads):
    # Each output has linear cross-talk, a cyclic adjacent-channel product,
    # and a cubic response to its own load. Fit every coefficient on this batch.
    a = [np.column_stack((np.ones(len(loads)), loads[:, [(j-1)%4, j, (j+1)%4]],
                         loads[:, j]*loads[:, (j+1)%4], loads[:, j]**3)) for j in range(4)]
    keep = np.ones(len(raw), dtype=bool)
    while True:
        b = np.array([np.linalg.lstsq(a[j][keep], raw[keep, j], rcond=None)[0] for j in range(4)])
        residual = raw - np.column_stack([a[j] @ b[j] for j in range(4)])
        distance = np.linalg.norm(residual, axis=1)
        worst = np.argmax(np.where(keep, distance, -1))
        if distance[worst] < 0.005 or keep.sum() <= len(raw)*0.85:
            break
        # The calibration sweep has an isolated, large shared offset on a few rows.
        keep[worst] = False
    linear = np.zeros((4, 4))
    for j in range(4):
        linear[[(j-1)%4, j, (j+1)%4], j] = b[j, 1:4]
    return (b[:, 0], linear, b[:, 4], b[:, 5]), keep


def _load_from_raw(raw, bridge):
    offset, linear, mixed, cubic = bridge
    j = np.arange(4)
    loads = (raw-offset) @ np.linalg.inv(linear)
    for _ in range(15):
        error = _bridge_forward(loads, bridge) - raw
        if np.max(np.abs(error)) < 1e-12:
            break
        jac = np.broadcast_to(linear.T, loads.shape[:-1] + (4, 4)).copy()
        jac[..., j, j] += mixed*np.roll(loads, -1, axis=-1) + 3*cubic*loads**2
        jac[..., j, (j+1)%4] += mixed*loads
        loads -= np.linalg.solve(jac, error[..., None])[..., 0]
    return loads


def _calibrated_inputs(split, features, bridge):
    return (_load_from_raw(split['past_channel_raw'], bridge).sum(-1),
            _load_from_raw(split['hold_channel_raw'], bridge).sum(-1),
            np.column_stack((np.ones(len(features)), features)))
```

### Turn 5 — `execute_in_remote_docker` (`23597 ms` LLM)
- **Rationale**: Test whether the archived bridge basis fits the current calibration and inverts across the actual coupon range. A poor fit would require changing the bridge model before validating material parameters.
- **Alternatives Weighed**: `Carry forward the last solver's reset law` (rejected: The last solver applies a reset learned from a different apparatus state; the latest trace is near the earlier idle trace in both indicator channels.)

```bash
cd /holdout_task; A=/tmp/hegel_continual_archive; sed -n '1,90p' "$A/alpha_m100_c01_batch_03_solver.py"; for v in batch_02 batch_03 batch_04; do echo "--- $v"; sed -n '/^def solve_batch/,$p' "$A/alpha_m100_c01_${v}_solver.py" | tail -25; done; python3 - <<'PY'
import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m100_c01_batch_03_solver.py'; s=importlib.util.spec_from_file_location('arch',p); a=importlib.util.module_from_spec(s); s.loader.exec_module(a)
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; raw,load=c['bridge_raw'],c['true_channel_loads']; b,keep=a._fit_bridge(raw,load); print('bridge',b); print('row residual all',np.quantile(np.linalg.norm(raw-a._bridge_forward(load,b),axis=1),[0,.5,.85,.9,.95,1]),'removed',np.flatnonzero(~keep)); print('load calib range',load.min(0),load.max(0));
for name in ('train','test'):
 z=d[name]; l=a._load_from_raw(np.vstack((z['hold_channel_raw'],z['past_channel_raw'].reshape(-1,4))),b); print(name,'load range',l.min(0),l.max(0),'inversion max',abs(a._bridge_forward(l,b)-np.vstack((z['hold_channel_raw'],z['past_channel_raw'].reshape(-1,4)))).max()); print('ref',np.isnan(z['ref_coupons']).sum(), 'missing features',np.isnan(z['coupon_features_raw']).sum(0)); print('aux median/sd',np.median(z['aux_chamber_telemetry'],0),np.std(z['aux_chamber_telemetry'],0));
o=a.solve_batch(d); print('archive baseline taus',o['estimated_taus'],'theta',o['hereditary_weights']);
PY
```
- **Container Exit Code**: `0` (`699 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
theta}
--- batch_04
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    xtr, xte = _features(train, test)
    cal = dataset['calibration_load_cells']
    bridge, _ = _fit_bridge(cal['bridge_raw'], cal['true_channel_loads'])
    taus, theta = _fit(*_calibrated_inputs(train, xtr, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])

    # The pre-batch facility trace activates the projection regime observed in batch 01.
    # In this regime the chamber resets the hereditary state at entry.
    hold = _load_from_raw(test['hold_channel_raw'], bridge).sum(axis=-1)
    features = np.column_stack((np.ones(len(xte)), xte))
    curve, rate = _predict_chamber(taus, theta, hold, features, times,
                                   _CHAMBER_PROJECTION)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
bridge (array([ 0.11824793, -0.16885151,  0.07871429, -0.16817686]), array([[ 1.75481718,  0.00306479,  0.        , -0.01543011],
       [ 0.0423546 ,  1.06021084, -0.04116389,  0.        ],
       [ 0.        , -0.00819444,  1.80277171,  0.04319022],
       [-0.01184163,  0.        , -0.02394252,  1.0459678 ]]), array([ 0.06981364, -0.01868379,  0.00475006,  0.03392632]), array([-0.22100705, -0.16983952, -0.22823871, -0.16891917]))
row residual all [ 0.09181083  0.37929481  0.54776986  0.69456638  0.80296886 27.40875092] removed [ 0  8 13 14 15 19 23 26 33 41 56 68]
load calib range [-1.6 -1.6 -1.6 -1.6] [1.7 1.7 1.7 1.7]
train load range [ -3.29516167 -20.11551565  -3.7219901   -3.36860481] [6.80297906 3.5540194  2.07132943 3.18126715] inversion max 1360.1329803079152
ref 0 missing features [5 2 4 3]
aux median/sd [ 0.23866647  0.26198016 -0.08625506 -0.44220047] [1.10046845 0.92958859 1.07231339 0.91162623]
test load range [-4.15979907 -8.88713077 -3.2710747  -2.91985575] [ 6.79889646 23.09150532  6.48070571 14.58508485] inversion max 2068.5661806602675
ref 0 missing features [4 3 0 2]
aux median/sd [ 0.42702758 -0.1108441   0.43804939 -0.19120878] [1.01381159 0.98346796 1.1151294  1.09488551]
archive baseline taus [ 0.5110895 24.999    ] theta [ 0.56975822 -5.79762732  1.67627606  0.22129092 -0.13899493  0.10548066
 -0.10988178]
```

### Turn 6 — `execute_in_remote_docker` (`22522 ms` LLM)
- **Rationale**: Isolated sweeps can distinguish saturation, piecewise response, and calibration contamination without confounding them with cross-talk. Mixed sweeps then test which interaction terms are necessary.
- **Alternatives Weighed**: `Retune Newton inversion for the archived cubic` (rejected: A better numerical inverse would not fix the large residuals in the forward calibration.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived cyclic-product plus cubic forward basis can calibrate this campaign's transducer.` | Witness: `The median calibration residual norm was 0.379 after fitting, and archived Newton inversion missed an observed deployment bridge vector by up to 2068.57.` | Blanket Diagnosis: `Bridge model error corrupts the calibrated stress histories before the hereditary fit, so the fitted relaxation time at the upper bound cannot diagnose chamber physics.` | Sublated Pivot: `Use zero-load and isolated-channel rows to identify the one-channel response, then test interaction terms on mixed sweeps before refitting material parameters.`

```bash
cd /holdout_task; cat public_asset_index.json; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json | head -c 2600; python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_batch.pkl','rb'))[0]; print('diagnostics:',type(item.get('campaign_diagnostics'))); d=item['dataset']; c=d['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']; np.set_printoptions(precision=4,suppress=True,linewidth=150); print('calibration columns',list(c)); print('shape',R.shape); g=(L!=0).sum(1); print('nonzero loads per row',np.unique(g,return_counts=True)); print('single and zero rows: i, load, raw'); print(np.c_[np.arange(len(L)),L,R][g<=1]); print('first 12 mixed rows'); print(np.c_[np.arange(len(L)),L,R][g>1][:12]); print('raw ranges',np.quantile(R,[0,.05,.5,.95,1],axis=0)); print('train/test raw ranges');
for n in ('train','test'):
 a=np.vstack((d[n]['hold_channel_raw'],d[n]['past_channel_raw'].reshape(-1,4))); print(n,np.quantile(a,[0,.05,.95,1],axis=0))
PY
```
- **Container Exit Code**: `0` (`414 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.       2.0296  -0.1099   0.4431  -0.11  ]
 [  4.       0.      -1.6      0.       0.       0.1401  -1.2245   0.1299  -0.4747]
 [  5.       0.      -0.6      0.       0.       0.14    -0.781    0.13    -0.2413]
 [  6.       0.       0.7      0.       0.       0.14     0.6373   0.1301  -0.0322]
 [  7.       0.       1.7      0.       0.       0.1399   1.0564   0.13     0.0813]
 [  8.       0.       0.      -1.6      0.      -0.4435  -0.11    -1.7181  -0.11  ]
 [  9.       0.       0.      -0.6      0.      -0.0702  -0.11    -0.9826  -0.1101]
 [ 10.       0.       0.       0.7      0.       0.2645  -0.1099   1.3693  -0.1098]
 [ 11.       0.       0.       1.7      0.       0.4458  -0.1101   2.0639  -0.11  ]
 [ 12.       0.       0.       0.      -1.6      0.14    -0.4701   0.1299  -1.2388]
 [ 13.       0.       0.       0.      -0.6     14.6399 -13.4397  12.93   -14.8896]
 [ 14.       0.       0.       0.       0.7     14.64   -13.2332  12.93   -13.4531]
 [ 15.       0.       0.       0.       1.7     14.6401 -13.1211  12.9299 -13.029 ]]
first 12 mixed rows
[[16.      0.0826 -0.1776 -0.0403 -0.8368  0.3003 -0.526   0.0659 -0.9858]
 [17.      0.4506 -1.0949 -1.1546 -0.7925  0.5668 -1.2321 -1.371  -1.187 ]
 [18.      0.8132  1.278   0.6426 -1.1837  1.5849  0.6114  1.4399 -0.9665]
 [19.     -0.3238 -0.5389 -1.382  -1.4706 -1.0257 -1.0661 -1.6843 -1.3002]
 [20.      0.1713 -0.8534  0.4327 -0.4656  0.5816 -1.0476  1.0288 -0.8641]
 [21.     -1.061   0.0023 -0.7867  1.1535 -1.6568  0.0052 -1.6144  0.8642]
 [22.     -0.2016  1.4256  1.0662 -0.3545 -0.1038  0.8651  1.615  -0.4031]
 [23.      1.3984 -0.9319 -1.2414  0.3262  1.3477 -0.9453 -1.2767  0.0782]
 [24.      0.9642  0.6156 -0.6607  1.0554  1.3524  0.677  -0.9028  0.9023]
 [25.     -0.1609  1.1637  0.6531  1.4788 -0.0723  1.0061  1.2641  1.0821]
 [26.     -1.5009 -1.3987 -0.5673 -0.6207 -1.7947 -1.2813 -1.5062 -1.1346]
 [27.     -0.8328 -1.1915 -0.7916  0.1569 -1.4956 -1.0595 -1.5228 -0.1909]]
raw ranges [[ -2.0799 -13.4397  -2.0627 -14.8896]
 [ -1.7982  -1.52    -1.7305  -1.5325]
 [  0.2023  -0.1918   0.2342  -0.444 ]
 [  2.0306   0.7981   2.1216   0.8299]
 [ 14.6401   1.0564  12.93     1.0821]]
train/test raw ranges
train [[-1.2156 -0.9201 -1.2299 -0.9546]
 [-0.854  -0.6869 -0.9025 -0.7241]
 [ 1.8122  0.8959  1.8367  0.9297]
 [ 1.9494  1.0248  2.0084  1.0397]]
test [[-1.1331 -0.9125 -1.1312 -0.9504]
 [-0.8473 -0.726  -0.8762 -0.7404]
 [ 1.804   0.9284  1.849   0.9203]
 [ 1.9909  1.0259  2.0231  1.0412]]
```

### Turn 7 — `execute_in_remote_docker` (`27698 ms` LLM)
- **Rationale**: The isolated sweeps show saturation in each channel, opposite-channel cross-talk, and three grossly shifted rows. Compare candidate response laws on held-out mixed loads before inverting the bridge.
- **Alternatives Weighed**: `Interpolate the calibration with a generic multivariate model` (rejected: The isolated sweeps identify a sparse opposite-channel structure; test that structure before using a model that may extrapolate poorly.)

```bash
cat > /tmp/bridge_probe.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; clean=np.max(abs(Y),axis=1)<5; X,Y=X[clean],Y[clean]; ids=np.flatnonzero(clean)
F={'tanh':np.tanh,'atan':np.arctan,'sqrt':lambda z:z/np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z),'softsign':lambda z:z/(1+abs(z)),'exp':lambda z:np.sign(z)*-np.expm1(-abs(z))}
# Separate held-out sets for mixed and isolated sweeps.
fit=(ids%4!=0); fit|=(ids<16)
for j in range(4):
 x=X[:,j]; z=X[:,(j+2)%4]; u=X[:,(j+1)%4]; v=X[:,(j-1)%4]; y=Y[:,j]
 for name,f in F.items():
  for variant in (0,1,2,3):
   terms=[np.ones(len(x)),np.maximum(z,0),np.minimum(z,0),x,z*z,x*z]
   if variant==1: terms=terms[:4]
   if variant==2: terms += [u,v,x*u,x*v,u*v]
   if variant==3: terms += [abs(x),u,v,x*u,x*v,u*v,x*x,x*z*z,x*x*z]
   M=np.array(terms).T
   def pred(p): return M@p[3:]+p[0]*f(x*np.exp(np.where(x>=0,p[1],p[2])))
   r=least_squares(lambda p:(pred(p)-y)[fit],np.r_[1,0,0,np.zeros(M.shape[1])],loss='soft_l1',f_scale=.005,max_nfev=250,bounds=(np.r_[0,-3,-3,np.full(M.shape[1],-np.inf)],np.r_[10,4,4,np.full(M.shape[1],np.inf)])); e=pred(r.x)-y
   print(j,name,variant,np.round([np.sqrt(np.mean(e[fit]**2)),np.sqrt(np.mean(e[~fit]**2)),max(abs(e[~fit]))],6),np.round(r.x,4) if name=='tanh' or max(abs(e[~fit]))<.002 else '')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_probe.py
```
- **Container Exit Code**: `0` (`2062 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e+00  3.8490e-01  3.4060e-01  1.3690e-01  1.6630e-01  3.8050e-01
  3.4790e-01 -2.7000e-03 -2.4000e-03 -4.2000e-02 -3.8000e-03  3.2000e-03
  4.8000e-03  3.5000e-03  8.0000e-04  2.7100e-02 -1.0500e-02  1.0000e-03]
2 atan 0 [0.018302 0.020101 0.038676] 
2 atan 1 [0.018295 0.017572 0.032238] 
2 atan 2 [0.019109 0.020216 0.035241] 
2 atan 3 [0.015524 0.018328 0.031845] 
2 sqrt 0 [0.017294 0.018719 0.035277] 
2 sqrt 1 [0.017352 0.016363 0.029439] 
2 sqrt 2 [0.017921 0.018854 0.031353] 
2 sqrt 3 [0.014784 0.017322 0.03018 ] 
2 rational 0 [0.015123 0.015846 0.026546] 
2 rational 1 [0.015295 0.014118 0.023268] 
2 rational 2 [0.015418 0.016283 0.026064] 
2 rational 3 [0.013154 0.015582 0.027334] 
2 softsign 0 [0.024382 0.030102 0.048779] 
2 softsign 1 [0.024139 0.028088 0.045919] 
2 softsign 2 [0.025683 0.031045 0.055666] 
2 softsign 3 [0.020338 0.030513 0.056257] 
2 exp 0 [0.023375 0.029294 0.048798] 
2 exp 1 [0.023422 0.027068 0.045311] 
2 exp 2 [0.024502 0.029869 0.055134] 
2 exp 3 [0.019657 0.030563 0.055455] 
3 tanh 0 [0.007175 0.00595  0.011305] [ 8.963e-01  2.833e-01  2.788e-01 -1.086e-01  1.088e-01  2.295e-01
  1.423e-01 -4.300e-03 -3.000e-04]
3 tanh 1 [0.007093 0.006286 0.010971] [ 0.8906  0.289   0.2811 -0.1081  0.103   0.2357  0.1461]
3 tanh 2 [0.006674 0.006157 0.013336] [ 8.901e-01  2.875e-01  2.861e-01 -1.083e-01  1.083e-01  2.300e-01
  1.457e-01 -4.200e-03 -2.000e-04 -5.000e-04  3.300e-03 -5.000e-04
  2.300e-03 -6.000e-04]
3 tanh 3 [0.005974 0.008994 0.02032 ] [ 9.481e-01  2.198e-01  2.949e-01 -1.083e-01  1.111e-01  2.338e-01
  1.037e-01 -8.000e-04  2.700e-03  4.690e-02 -8.000e-04  3.600e-03
 -1.000e-03  1.200e-03 -9.000e-04 -3.330e-02  4.900e-03 -1.500e-03]
3 atan 0 [0.007959 0.006974 0.012544] 
3 atan 1 [0.007923 0.007298 0.012334] 
3 atan 2 [0.007431 0.00721  0.014278] 
3 atan 3 [0.006782 0.009662 0.021952] 
3 sqrt 0 [0.007646 0.006595 0.012322] 
3 sqrt 1 [0.007593 0.006883 0.011825] 
3 sqrt 2 [0.007141 0.006774 0.013959] 
3 sqrt 3 [0.006459 0.009385 0.0213  ] 
3 rational 0 [0.007006 0.005533 0.010054] 
3 rational 1 [0.006891 0.006079 0.01048 ] 
3 rational 2 [0.006454 0.005945 0.012708] 
3 rational 3 [0.005804 0.00891  0.020077] 
3 softsign 0 [0.011012 0.010689 0.02026 ] 
3 softsign 1 [0.011065 0.011434 0.023145] 
3 softsign 2 [0.010241 0.011774 0.024466] 
3 softsign 3 [0.010046 0.014371 0.031896] 
3 exp 0 [0.010898 0.010549 0.018731] 
3 exp 1 [0.010948 0.011263 0.022094] 
3 exp 2 [0.010213 0.011642 0.022759] 
3 exp 3 [0.009778 0.013955 0.030679]
```

### Turn 8 — `execute_in_remote_docker` (`45553 ms` LLM)
- **Rationale**: Isolated sweeps on channels 0 and 2 have nearly proportional centered responses. Test whether a shared response shape and a sparse opposite-channel term reduce held-out error before fitting the material.
- **Alternatives Weighed**: `Use the first-pass tanh fit directly` (rejected: Its held-out errors are far above the roughly 0.0001 scatter visible on isolated sweeps; those errors could bias recovered stresses near saturation.)

```bash
cat > /tmp/bridge_structure.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
X,Y=D['true_channel_loads'],D['bridge_raw']; keep=np.max(np.abs(Y),axis=1)<5; ids=np.flatnonzero(keep); X,Y=X[keep],Y[keep]; Z=np.roll(X,2,axis=1)
fit=(ids<16)|(ids%4!=0)
F={'tanh':np.tanh,'rat':lambda t:t/(1+t*t),'atan':np.arctan,'sqrt':lambda t:t/np.sqrt(1+t*t),'clip':lambda t:np.clip(t,-1,1),'sinh':lambda t:t/np.cosh(t),'poly':lambda t:t**3,'erf':None}
from scipy.special import erf
F['erf']=erf
# Common response shape with per-channel gain and offset; independent opposite-channel coefficients.
for name,f in F.items():
 for form in ('sep','sep3','sep5','shift','mix','split','clip2','quot'):
  if form=='clip2' and name!='clip': continue
  if form=='quot' and name!='rat': continue
  def pred(p):
   k,b,c,m,a,e,h,j,v=p[:9]; offsets=p[9:13]; gains=np.r_[1,np.exp(p[13:16])]; z=Z
   cross=c*z+m*np.abs(z)+a*z**3
   if form=='sep': cross=c*z+m*np.abs(z)
   if form=='sep5': cross=c*z+m*np.abs(z)+a*z**3+e*z*np.abs(z)+h*np.abs(z)**3
   if form=='split': cross=c*z+m*np.abs(z)+a*z**3+e*z*np.abs(z)+h*np.abs(z)**3+j*X**5+v*X**3
   if form=='shift': t=(X+h*z)*np.exp(k)
   elif form=='mix': t=X*np.exp(k)*(1+h*z+v*z*z)
   elif form=='split': t=X*np.exp(k+j*np.sign(X))
   else: t=X*np.exp(k)
   if form=='clip2': cross+=h*np.clip(X*np.exp(v),-1,1)
   if form=='quot': cross+=h*X*(X*X)/(1+np.exp(v)*X*X)
   return offsets+gains*(f(t)*b+X*e+cross) if form not in ('sep5','split') else offsets+gains*(f(t)*b+X*0.3+cross)
  sols=[]
  for k in (-.6,.35,.9):
   p=np.r_[k,1,.27,-.09,.005,.3,.0,0.,0.,.14,-.11,.13,-.11,-.47,.02,-.46]
   res=least_squares(lambda p:(pred(p)-Y)[fit].ravel(),p,loss='soft_l1',f_scale=.003,max_nfev=240,bounds=(np.r_[[-2],np.full(7,-np.inf),[-4],np.full(4,-np.inf),np.full(3,-2)],np.r_[[2],np.full(7,np.inf),[4],np.full(4,np.inf),np.full(3,2)])); sols.append(res)
  r=min(sols,key=lambda r:r.cost); err=pred(r.x)-Y
  print(name,form,'fit/hold/max',np.round([np.sqrt(np.mean(err[fit]**2)),np.sqrt(np.mean(err[~fit]**2)),np.max(abs(err[~fit]))],6), 'pars',np.round(r.x,5) if np.sqrt(np.mean(err[~fit]**2))<.009 else '')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_structure.py
```
- **Container Exit Code**: `0` (`6274 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2] pars 
atan sep3 fit/hold/max [0.015037 0.01236  0.030621] pars 
atan sep5 fit/hold/max [0.018179 0.022911 0.0543  ] pars 
atan shift fit/hold/max [0.014997 0.012459 0.030706] pars 
atan mix fit/hold/max [0.014478 0.013144 0.038928] pars 
atan split fit/hold/max [0.011988 0.010706 0.027185] pars 
sqrt sep fit/hold/max [0.014184 0.011153 0.03041 ] pars 
sqrt sep3 fit/hold/max [0.014254 0.011767 0.028721] pars 
sqrt sep5 fit/hold/max [0.014918 0.01708  0.044614] pars 
sqrt shift fit/hold/max [0.014201 0.011901 0.029056] pars 
sqrt mix fit/hold/max [0.013803 0.012557 0.038683] pars 
sqrt split fit/hold/max [0.010982 0.009511 0.023024] pars 
clip sep fit/hold/max [0.020883 0.024824 0.076962] pars 
clip sep3 fit/hold/max [0.020573 0.024904 0.080087] pars 
clip sep5 fit/hold/max [0.048197 0.053927 0.121686] pars 
clip shift fit/hold/max [0.020629 0.026029 0.083066] pars 
clip mix fit/hold/max [0.020878 0.026019 0.08637 ] pars 
clip split fit/hold/max [0.026138 0.031835 0.11282 ] pars 
clip clip2 fit/hold/max [0.010086 0.010373 0.029416] pars 
sinh sep fit/hold/max [0.011641 0.009615 0.023584] pars 
sinh sep3 fit/hold/max [0.011715 0.010154 0.025606] pars 
sinh sep5 fit/hold/max [0.029469 0.022551 0.09894 ] pars 
sinh shift fit/hold/max [0.0116   0.01048  0.026301] pars 
sinh mix fit/hold/max [0.011653 0.010279 0.024224] pars 
sinh split fit/hold/max [0.008826 0.008387 0.020184] pars [ 9.97600e-02  1.65828e+00  2.63800e-01 -1.09690e-01 -5.94000e-03
  1.06800e-02 -4.10000e-04  6.10000e-03  6.97800e-02  1.47490e-01
 -1.06580e-01  1.37560e-01 -1.06370e-01 -4.85950e-01  2.14800e-02
 -4.73350e-01]
poly sep fit/hold/max [0.054285 0.048286 0.153748] pars 
poly sep3 fit/hold/max [0.053801 0.047716 0.148801] pars 
poly sep5 fit/hold/max [0.455855 0.483862 1.120782] pars 
poly shift fit/hold/max [0.056141 0.044989 0.124692] pars 
poly mix fit/hold/max [0.161419 0.153765 0.408924] pars 
poly split fit/hold/max [0.30523  0.314915 0.729819] pars 
erf sep fit/hold/max [0.011172 0.009044 0.021544] pars 
erf sep3 fit/hold/max [0.011249 0.009546 0.024067] pars 
erf sep5 fit/hold/max [0.015417 0.010792 0.043507] pars 
erf shift fit/hold/max [0.011155 0.009844 0.024335] pars 
erf mix fit/hold/max [0.011165 0.010239 0.032855] pars 
erf split fit/hold/max [0.008865 0.007629 0.018146] pars [ 2.2628e-01  1.2868e+00  2.6585e-01 -1.1303e-01 -3.5900e-03  6.1400e-03
  5.7000e-04  1.8870e-02 -3.7650e-02  1.4315e-01 -1.0815e-01  1.3409e-01
 -1.0857e-01 -4.8571e-01  2.1530e-02 -4.7289e-01]
```

### Turn 9 — `execute_in_remote_docker` (`24134 ms` LLM)
- **Rationale**: If the additive spline decomposition predicts held-out sweeps well, its component curves can expose the missing shape. If it does not, test a joint dependence on the paired channels before committing to an inverse.
- **Alternatives Weighed**: `Continue testing single saturation functions` (rejected: Several families leave similar residuals; estimating both component curves directly can reveal which shape or coupling is missing.)

```bash
cat > /tmp/decompose.py <<'PY'
import pickle,numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
C=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=C['true_channel_loads'],C['bridge_raw']; ok=np.max(abs(Y),1)<5; ids=np.nonzero(ok)[0]; X,Y=X[ok],Y[ok]; Z=np.roll(X,2,axis=1)
for n in (3,5,7,9,13):
 def basis(x): return BSpline.design_matrix(x,np.r_[[-1.8]*4,np.linspace(-1.8,1.8,n+2)[1:-1],[1.8]*4],3).toarray()
 errors=[]
 for j in range(4):
  a,b=basis(X[:,j]),basis(Z[:,j]); M=np.c_[a,b[:,1:]]; y=Y[:,j]; errs=[]
  for fold in range(4):
   keep=ids%4!=fold; p=np.linalg.lstsq(M[keep],y[keep],rcond=None)[0]; errs.extend((M[~keep]@p-y[~keep]).tolist())
  p=np.linalg.lstsq(M,y,rcond=None)[0]; errors.append([np.std(M@p-y),np.std(errs),max(abs(np.array(errs)))]);
  if n==9:
   grid=np.arange(-1.6,1.61,.1); base=basis(grid)-basis(np.array([0.])); own=base@p[:a.shape[1]]; cross=base[:,1:]@p[a.shape[1]:]; print('ch',j,'own',np.round(own,4),'cross',np.round(cross,4))
 print(n,np.round(errors,6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/decompose.py; sed -n '1,105p' /tmp/hegel_continual_archive/alpha_m100_c01_batch_03_solver.py
```
- **Container Exit Code**: `0` (`530 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.4, 2.5, 4.):
        for b in (.5, 1., 2., 4., 8., 16.):
            if b <= a:
                continue
            u = np.array([np.log(a), np.log(b/a) / np.log(24.999/a)])
            M = matrix(u)
            beta = np.linalg.lstsq(M, y, rcond=None)[0]
            for _ in range(4):
                w = (1 + ((M @ beta - y) / .01)**2)**(-.25)
                beta = np.linalg.lstsq(w[:, None] * M, w * y, rcond=None)[0]
            resid = M @ beta - y
            initials.append((np.sum(np.hypot(.01, resid) - .01), u, beta))

    def residual(q):
        return matrix(q[:2]) @ q[2:] - y

    results = []
    for _, u, beta in sorted(initials, key=lambda entry: entry[0])[:4]:
        results.append(least_squares(residual, np.r_[u, beta], loss='soft_l1',
                                     f_scale=.008, x_scale='jac', max_nfev=200,
                                     bounds=(np.r_[bounds[0], np.full(7, -np.inf)],
                                             np.r_[bounds[1], np.full(7, np.inf)])))
    result = min(results, key=lambda r: r.cost)
    # Give the direct initial-rate measurements more leverage than any
    # single strain reading while retaining robust loss for both outputs.
    def rate_weighted_residual(q):
        e = residual(q).reshape(n, k + 1)
        e[:, -1] *= 4.0
        return e.ravel()

    result = least_squares(rate_weighted_residual, result.x, loss='soft_l1',
                           f_scale=.008, x_scale='jac', max_nfev=200,
                           bounds=(np.r_[bounds[0], np.full(7, -np.inf)],
                                   np.r_[bounds[1], np.full(7, np.inf)]))
    return unpack(result.x[:2]), result.x[2:]


def _bridge_forward(loads, bridge):
    offset, linear, mixed, cubic = bridge
    return offset + loads @ linear + mixed * loads * np.roll(loads, -1, axis=-1) + cubic * loads**3


def _fit_bridge(raw, loads):
    # Each output has linear cross-talk, a cyclic adjacent-channel product,
    # and a cubic response to its own load. Fit every coefficient on this batch.
    a = [np.column_stack((np.ones(len(loads)), loads[:, [(j-1)%4, j, (j+1)%4]],
                         loads[:, j]*loads[:, (j+1)%4], loads[:, j]**3)) for j in range(4)]
    keep = np.ones(len(raw), dtype=bool)
    while True:
        b = np.array([np.linalg.lstsq(a[j][keep], raw[keep, j], rcond=None)[0] for j in range(4)])
        residual = raw - np.column_stack([a[j] @ b[j] for j in range(4)])
        distance = np.linalg.norm(residual, axis=1)
```

### Turn 10 — `execute_in_remote_docker` (`34637 ms` LLM)
- **Rationale**: The recovered cross-channel response has a nearly linear odd component and a cosine-shaped even component. Fit that structure jointly with an odd own-channel response; use the holdout error to decide whether the additional terms are needed.
- **Alternatives Weighed**: `Deploy the unconstrained spline inverse` (rejected: Sparse edge coverage destabilized withheld-sweep predictions; the component curves reveal stronger symmetry that can be used before inversion.)

```bash
cat > /tmp/fourier_bridge.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
C=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=C['true_channel_loads'],C['bridge_raw']; keep=np.max(abs(Y),1)<5; ids=np.flatnonzero(keep); X,Y=X[keep],Y[keep]; Z=np.roll(X,2,axis=1)
a=ids%4!=1
for form in ('sin','sin_tanh','sin_cubic','two_sin','sin_rat','sin2','sin_cos2'):
 def pred(p):
  k,l,m,A,B,C,D,E,F=p[:9]; gains=np.r_[1,np.exp(p[13:16])]; out=A*X+B*np.sin(k*X); cross=C*Z+D*(np.cos(l*Z)-1)
  if form=='sin_tanh': out+=E*np.tanh(m*X)
  if form=='sin_cubic': out+=E*X**3
  if form=='two_sin': out+=E*np.sin(m*X)
  if form=='sin_rat': out+=E*X/(1+(m*X)**2)
  if form=='sin2': out+=E*np.sin(m*X); cross+=F*(np.cos(m*Z)-1)
  if form=='sin_cos2': cross+=E*(np.cos(m*Z)-1)
  return p[9:13]+gains*(out+cross)
 res=[]
 for m in (.8,1.8,3.2,5.):
  p=np.r_[2.5,2,m,1.27,.32,.27,.077,.001,.001,.14,-.11,.13,-.11,-.48,.023,-.47]; r=least_squares(lambda p:(pred(p)-Y)[a].ravel(),p,max_nfev=240,loss='soft_l1',f_scale=.0005); res.append(r)
 r=min(res,key=lambda r:r.cost); e=pred(r.x)-Y; print(form,np.round(np.r_[np.sqrt(np.mean(e[a]**2)),np.sqrt(np.mean(e[~a]**2)),max(abs(e[~a]).ravel())],7),np.round(r.x,6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/fourier_bridge.py
sed -n '/^def solve_batch/,$p' /tmp/hegel_continual_archive/alpha_m100_c01_batch_03_solver.py
python3 - <<'PY'
import json,numpy as np
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print(r['completed_campaign_short_name'],np.round(np.median(r['facility_telemetry_trace'],axis=0)[[2,7]],3))
PY
```
- **Container Exit Code**: `0` (`2430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
179000e-02 -4.7000800e-01]
sin_cubic [9.130e-05 9.810e-05 2.375e-04] [ 2.500803e+00  2.000171e+00  7.969540e-01  1.280649e+00  3.196740e-01
  2.687960e-01  7.679600e-02 -2.440000e-04  1.611000e-02  1.399980e-01
 -1.099690e-01  1.300120e-01 -1.099970e-01 -4.825930e-01  2.317900e-02
 -4.700090e-01]
two_sin [9.130e-05 9.850e-05 2.368e-04] [ 2.500030e+00  2.000180e+00  5.356293e+00  1.280015e+00  3.200120e-01
  2.687960e-01  7.679600e-02  1.100000e-05  3.851188e+00  1.399980e-01
 -1.099690e-01  1.300120e-01 -1.099980e-01 -4.825930e-01  2.317900e-02
 -4.700090e-01]
sin_rat [9.110e-05 9.720e-05 2.345e-04] [ 2.5000200e+00  2.0001710e+00  1.1594807e+01  1.2800090e+00
  3.1999700e-01  2.6879600e-01  7.6795000e-02  9.0100000e-04
 -5.7830820e+00  1.3999600e-01 -1.0996800e-01  1.3001200e-01
 -1.0999800e-01 -4.8259200e-01  2.3180000e-02 -4.7000800e-01]
sin2 [8.940e-05 1.039e-04 2.441e-04] [ 2.500119e+00  2.000019e+00  6.805220e+00  1.280037e+00  3.199970e-01
  2.687980e-01  7.680900e-02  8.000000e-06 -3.400000e-05  1.399730e-01
 -1.099810e-01  1.299910e-01 -1.100100e-01 -4.825890e-01  2.318100e-02
 -4.700060e-01]
sin_cos2 [8.940e-05 1.035e-04 2.371e-04] [ 2.500086e+00  2.000030e+00  6.884380e+00  1.280028e+00  3.200000e-01
  2.687980e-01  7.680900e-02 -3.400000e-05 -1.066905e+01  1.399730e-01
 -1.099820e-01  1.299910e-01 -1.100100e-01 -4.825890e-01  2.318200e-02
 -4.700060e-01]
def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    xtr, xte = _features(train, test)
    cal = dataset['calibration_load_cells']
    bridge, _ = _fit_bridge(cal['bridge_raw'], cal['true_channel_loads'])
    taus, theta = _fit(*_calibrated_inputs(train, xtr, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])

    # The latest apparatus record has both known chamber indicators at idle.
    curve, rate = _design(taus, *_calibrated_inputs(test, xte, bridge), dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
facility_init [-0.337 -0.421]
alpha_m100_c01_s00 [ 2.434 -0.416]
alpha_m100_c01_batch_01 [-0.345  2.565]
alpha_m100_c01_batch_02 [-0.38  -0.407]
alpha_m100_c01_batch_03 [ 2.474 -0.4  ]
alpha_m100_c01_batch_04 [-0.34  -0.411]
```

### Turn 11 — `execute_in_remote_docker` (`44354 ms` LLM)
- **Rationale**: Implement the noise-floor calibration and verify its inverse on the actual coupon inputs. If closure holds, blocked validation can test material recovery separately from the blind chamber assumption.
- **Alternatives Weighed**: `Add harmonics to the bridge response` (rejected: Additional terms did not reduce the worst held-out residual, which is already near the calibration noise floor.)

```bash
set -e
src=/tmp/hegel_continual_archive/alpha_m100_c01_batch_03_solver.py
target=/holdout_task/batch_solver.py
sed '/^def _bridge_forward/,$d' "$src" > "$target"
cat >> "$target" <<'PY'
def _bridge_terms(loads, p):
    k, ell, a, b, c, d = p[:6]
    opposite = np.roll(loads, 2, axis=-1)
    return a * loads + b * np.sin(k * loads) + c * opposite + d * (np.cos(ell * opposite) - 1)


def _bridge_forward(loads, p):
    return p[6:10] + np.exp(np.r_[0., p[10:]]) * _bridge_terms(loads, p)


def _fit_bridge(raw, loads):
    # Identify a shared response in channel-0 units; the other gains are relative.
    initial = np.r_[2.4, 2.1, 1.2, .3, .25, .08, np.zeros(4), np.zeros(3)]
    bounds = (np.r_[.5, .5, .1, 0., -.8, -.5, np.full(4, -np.inf), np.full(3, -2.)],
              np.r_[5., 5., 3., 1., .8, .5, np.full(4, np.inf), np.full(3, 2.)])
    result = least_squares(lambda p: (_bridge_forward(loads, p) - raw).ravel(),
                           initial, bounds=bounds, loss='soft_l1', f_scale=.002)
    err = np.linalg.norm(_bridge_forward(loads, result.x) - raw, axis=1)
    # A contaminated sweep is removed as a row, not as independent channels.
    keep = err < 12 * np.median(err)
    p = least_squares(lambda p: (_bridge_forward(loads[keep], p) - raw[keep]).ravel(),
                      result.x, bounds=bounds).x
    return p, keep


def _load_from_raw(raw, p):
    k, ell, a, b, c, d = p[:6]
    y = (raw - p[6:10]) / np.exp(np.r_[0., p[10:]])
    loads = y / (a + b*k)
    for _ in range(30):
        opposite = np.roll(loads, 2, axis=-1)
        r = _bridge_terms(loads, p) - y
        if np.max(np.abs(r)) < 1e-12:
            break
        own = a + b*k*np.cos(k*loads)
        cross = c - d*ell*np.sin(ell*opposite)
        det = own*np.roll(own, 2, axis=-1) - cross*np.roll(cross, 2, axis=-1)
        delta = (np.roll(own, 2, axis=-1)*r - cross*np.roll(r, 2, axis=-1)) / det
        loads -= np.clip(delta, -.4, .4)
    return loads


def _calibrated_inputs(split, features, bridge):
    past = _load_from_raw(split['past_channel_raw'], bridge).sum(axis=-1)
    hold = _load_from_raw(split['hold_channel_raw'], bridge).sum(axis=-1)
    return past, hold, np.column_stack((np.ones(len(features)), features))


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    xtr, xte = _features(train, test)
    cal = dataset['calibration_load_cells']
    bridge, _ = _fit_bridge(cal['bridge_raw'], cal['true_channel_loads'])
    taus, theta = _fit(*_calibrated_inputs(train, xtr, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The latest apparatus trace shows both chamber indicators at idle.
    curve, rate = _design(taus, *_calibrated_inputs(test, xte, bridge), dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
PY
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import batch_solver as b,pickle,numpy as np
D=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; L=C['true_channel_loads']; R=C['bridge_raw']; p,keep=b._fit_bridge(R,L); print('removed',np.flatnonzero(~keep),'fit',np.round(p,6)); print('inverse cal max',abs(b._load_from_raw(R[keep],p)-L[keep]).max());
for s in ('train','test'):
 for k in ('past_channel_raw','hold_channel_raw'):
  r=D[s][k]; print(s,k,'inverse closure',np.max(abs(b._bridge_forward(b._load_from_raw(r,p),p)-r)))
o=b.solve_batch(D); print('intrinsic',o['estimated_taus'],o['hereditary_weights']); print('pred spans',[(k,np.min(v),np.max(v)) for k,v in o.items()]); print('train labs/strata',[(k,np.unique(D['train'][k],return_counts=True)) for k in ('lab_id','stratum')]); print('eval times',D['eval_times'])
PY
```
- **Container Exit Code**: `0` (`999 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 0.0, "ref_swap_rate_max_diff": 0.0, "tol": 1e-05}}
removed [13 14 15] fit [ 2.500051  2.000032  1.280019  0.319987  0.268792  0.076798  0.139995
 -0.109974  0.130006 -0.109991 -0.482593  0.023179 -0.470006]
inverse cal max 0.0009226472969108279
train past_channel_raw inverse closure 3.552713678800501e-14
train hold_channel_raw inverse closure 6.661338147750939e-16
test past_channel_raw inverse closure 7.993605777301127e-15
test hold_channel_raw inverse closure 4.440892098500626e-16
intrinsic [0.4926665  3.17869135] [ 0.69976378  0.67276894  0.35759372  0.1496219  -0.17120315  0.14392167
 -0.08965325]
pred spans [('estimated_taus', 0.4926664983560879, 3.1786913478973537), ('hereditary_weights', -0.17120314915987764, 0.699763775003538), ('predicted_strain', 0.4080835120233465, 7.486323898937288), ('initial_rate_pred', 1.2793768811072064, 8.166149494695496)]
train labs/strata [('lab_id', (array([0, 1, 2, 3]), array([21, 20, 16, 15]))), ('stratum', (array([0, 1, 2]), array([18, 25, 29])))]
eval times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eb25dd0d945753b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_eb25dd0d (directory).
```

### Turn 12 — `execute_in_remote_docker` (`27644 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated process, then leave out entire labs and strata to detect unstable relaxation estimates or poor late-time or initial-rate predictions that pooled scores could hide. If a slice fails, inspect it before changing the fit.
- **Alternatives Weighed**: `Retune the robust loss using pooled residuals` (rejected: The verified candidate has low errors on both outputs; only a material failure on held-out labs or strata would justify retuning.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
d = pickle.load(open('public_batch.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; dt, t = d['past_dt'], d['eval_times']
o = b.solve_batch(d)
shapes = {'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (len(te['strain_0']),len(t)), 'initial_rate_pred': (len(te['strain_0']),)}
assert o.keys() == shapes.keys()
for k,shape in shapes.items():
    assert o[k].shape == shape and np.all(np.isfinite(o[k])), k
assert .05 < o['estimated_taus'][0] < o['estimated_taus'][1] < 25
cal=d['calibration_load_cells']; p,keep=b._fit_bridge(cal['bridge_raw'],cal['true_channel_loads']); x,_=b._features(tr,te)
P,H,F=b._calibrated_inputs(tr,x,p); Y=tr['strain_curves_obs']-tr['strain_0'][:,None]; R=tr['instant_rate_obs']; full=o['estimated_taus']; worst=[0.,0.,0.,0.]
for axis in ('lab_id','stratum'):
    for value in np.unique(tr[axis]):
        val=tr[axis]==value; fit=~val
        tau,theta=b._fit(P[fit],H[fit],F[fit],dt,t,Y[fit],R[fit]); a,r=b._design(tau,P[val],H[val],F[val],dt,t)
        e=np.abs(a@theta-Y[val]); er=np.abs(r@theta-R[val]); m=[np.median(e),np.median(er),np.median(e[:,-3:]),max(abs(tau/full-1))]; worst=np.maximum(worst,m)
        print(axis,value,'n',sum(val),'median strain/rate/late-strain; max relative tau shift',np.round(m,6))
print('worst-fold:', worst)
print('isolated process: all four outputs finite with required shapes and ordered taus')
assert worst[0]<.04 and worst[1]<.04 and worst[2]<.04 and worst[3]<.20, worst
PY
```
- **Container Exit Code**: `0` (`1112 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab_id 0 n 21 median strain/rate/late-strain; max relative tau shift [0.008269 0.006783 0.007846 0.01347 ]
lab_id 1 n 20 median strain/rate/late-strain; max relative tau shift [0.008218 0.010028 0.009397 0.005587]
lab_id 2 n 16 median strain/rate/late-strain; max relative tau shift [0.008023 0.006224 0.007749 0.008529]
lab_id 3 n 15 median strain/rate/late-strain; max relative tau shift [0.008984 0.006337 0.008877 0.021341]
stratum 0 n 18 median strain/rate/late-strain; max relative tau shift [0.008425 0.007301 0.008283 0.013088]
stratum 1 n 25 median strain/rate/late-strain; max relative tau shift [0.008903 0.00834  0.008691 0.041379]
stratum 2 n 29 median strain/rate/late-strain; max relative tau shift [0.007791 0.00666  0.007862 0.018813]
worst-fold: [0.00898426 0.01002841 0.00939711 0.04137911]
isolated process: all four outputs finite with required shapes and ordered taus

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eb25dd0d945753b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_eb25dd0d (directory).
```

### Turn 13 — `submit_completed_task` (`14207 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process verification, the public schema and invariance checks, and held-out lab and stratum validation. The host probe measured strain MAE 0.001011 and initial-rate MAE 0.000621; no validated failure warrants another edit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and pooled-reference invariance checks passed. Across held-out labs and strata, the worst median absolute errors were 0.0090 for strain and 0.0100 for initial rate.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
