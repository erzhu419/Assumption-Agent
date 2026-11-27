# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_02__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_02` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `16 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `482,512` (`465,047` / `123,520` / `17,465`) |
| **Total Episode Wall Time (s)** | `572.63s` |

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

### Turn 1 — `execute_in_remote_docker` (`18215 ms` LLM)
- **Rationale**: Inspect the public schema and archive to identify training observables, panel partition conventions, and calibration fallback inputs before fitting; print only bounded data samples.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The current batch explicitly specifies a linear-flow regime and in-batch calibration; inspect the archived conventions before reusing components.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))
def show(d,depth=0):
 for k,v in d.items():
  print('  '*depth,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''),len(v) if hasattr(v,'__len__') else repr(v))
  if isinstance(v,dict) and depth<2: show(v,depth+1)
  elif isinstance(v,(list,tuple)) and len(v): print('  '*(depth+1),str(v[0])[:450])
  elif isinstance(v,np.ndarray): print('  '*(depth+1),str(v[:2] if v.ndim else v)[:450])
show(p)
PY
wc -l /tmp/hegel_continual_archive/*; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`427 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"task_id": "alpha_m97_c01_batch_01__rheology_campaign_batch",
    "entrypoint": "solve_alpha_m97_c01_batch_01",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "regime_mode_weights"
    ],
    "input_coordinate_contract": "raw_measured",
    "solver_path": "/tmp/hegel_continual_archive/alpha_m97_c01_batch_01_solver.py"
  }
]  195 /tmp/hegel_continual_archive/alpha_m97_c01_batch_01_solver.py
  149 /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py
  259 /tmp/hegel_continual_archive/archive_calibration_history.json
   29 /tmp/hegel_continual_archive/archive_manifest.json
   17 /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
  649 total
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`22868 ms` LLM)
- **Rationale**: Inspect the dataset inside the list wrapper and read the archived implementation to identify the measured training targets, panel layout, and calibration fallback schema. A mismatch changes which components can be reused.
- **Alternatives Weighed**: `Port the prior batch solver unchanged` (rejected: The current batch specifies linear terminal flow and its own calibration rules; only compatible components should carry over.)

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c01_batch_01_solver.py; cat /holdout_task/public_asset_index.json; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))
print('items',len(p),list(p[0]))
def show(d,depth=0):
 for k,v in d.items():
  print('  '*depth,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''))
  if isinstance(v,dict): show(v,depth+1)
  elif isinstance(v,(np.ndarray,list,tuple)): print('  '*(depth+1),str(v[:2])[:320])
  else: print('  '*(depth+1),repr(v)[:320])
show(p[0]['dataset'])
PY
head -65 /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[
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
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`17663 ms` LLM)
- **Rationale**: The previous log truncated the dataset layout and archived preprocessing. Read only those parts to detect schema differences that would change calibration, alignment, or the measured targets used in fitting.

```bash
sed -n '86,195p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_01_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))
print('items/keys',len(p),list(p[0]))
for k,v in p[0]['dataset'].items():
 print(k,[(a,np.shape(b)) for a,b in v.items()] if isinstance(v,dict) else (np.shape(v),str(v)[:180]))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive',[(i['episode_short_name'],list(i),len(i['lab_id'])) for i in h])
PY
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x[take] - center) / scale
        for level in np.unique(stratum[take]):
            cell = take & (stratum == level)
            values = x[cell]
            x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    plateau = dataset['shear_banding_index'] <= 0.50
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)**2 @ dt
    act = np.tanh(damage) if plateau else np.zeros_like(damage)
    # Low-banding campaigns have logarithmic baseline flow. Damage gates the
    # sample-specific crystallization terms, not the baseline time dependence.
    return dt, t, past, hold, x, act, np.full(len(lab), plateau, dtype=bool)


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    dt, t, *samples = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    p = fit(dt, t, *(a[:n] for a in samples), yc, train['instant_rate_obs'])
    c, r, *_ = basis(p[:2], dt, t, *(a[n:] for a in samples))
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:9],
                regime_mode_weights=p[9:], predicted_strain=test['strain_0'][:, None] + c @ p[2:],
                initial_rate_pred=r @ p[2:])
items/keys 1 ['dataset', 'gt']
past_dt ((18,), '[0.631  0.4019 0.4324 0.613  0.7743 0.3251 0.4696 0.7581 0.2512 0.3061\n 0.3394 0.5896 0.5022 0.5857 0.449  0.4712 0.7702 0.449 ]')
eval_times ((10,), '[0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]')
crit_stress ((), '2.65')
yield_stress ((), '2.65')
shear_banding_index ((), '0.12')
calibration_load_cells [('lab_id', (40,)), ('cmd_stress', (40,)), ('true_stress', (40,))]
train [('lab_id', (180,)), ('stratum', (180,)), ('coupon_features_raw', (180, 4)), ('ref_coupons', (180, 32)), ('past_stress_cmd', (180, 18)), ('stress_hold_cmd', (180,)), ('strain_0', (180,)), ('aux_chamber_telemetry', (180, 3)), ('strain_curves_obs', (180, 10)), ('instant_rate_obs', (180,))]
test [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 32)), ('past_stress_cmd', (72, 18)), ('stress_hold_cmd', (72,)), ('strain_0', (72,)), ('aux_chamber_telemetry', (72, 3))]
archive [('alpha_m97_c01_s00', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'], 40), ('alpha_m97_c01_batch_01', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'], 40)]
```

### Turn 4 — `execute_in_remote_docker` (`81803 ms` LLM)
- **Rationale**: Build the specified stress, alignment, memory, and flow operators, then check the public schema and residual scales. Calibration residuals will expose a bad stress conversion; curve and rate residuals will show whether either target needs a different noise treatment before held-out validation.
- **Alternatives Weighed**: `Use the archived batch-01 flow basis` (rejected: Its logarithmic and crystallization terms do not belong to the specified linear-flow regime.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_02_solver.py <<'PY'
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = xc @ (y - y.mean()) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _spikes(x, y):
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not len(i):
        return np.zeros(len(x), dtype=bool)
    gains = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[:, None] - x[:, None] * gains
    errors = np.abs(offsets - np.median(offsets, axis=0))
    best = np.argmin(np.median(errors, axis=0))
    return errors[:, best] > 3.0


def _prior_line(channel):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        row = np.asarray(sweep['lab_id']) == channel
        x = np.asarray(sweep['cmd_stress'])[row]
        y = np.asarray(sweep['true_stress'])[row]
        keep = ~_spikes(x, y)
        if np.unique(x[keep]).size >= 2:
            return _line(x[keep], y[keep])
    raise ValueError(f'No usable calibration sweep for channel {channel}')


def _calibrate(x, y, channel):
    if len(x) >= 4:
        keep = ~_spikes(x, y)
        if keep.sum() >= 4 or keep.all():
            return _line(x[keep], y[keep])
    else:
        keep = np.ones(len(x), dtype=bool)
        if len(x) == 3:
            # Two surviving points cannot identify which of three rows spiked.
            # Look for a >3 discrepancy before consulting the prior sweep.
            errors = [abs(y[k] - np.polyval(_line(np.delete(x, k), np.delete(y, k)), x[k]))
                      for k in range(3) if np.unique(np.delete(x, k)).size == 2]
            slope, bias = _line(x, y)
            if not errors or max(errors) <= 3 or np.max(np.abs(y - slope*x - bias)) < .01:
                return slope, bias
        elif len(x) == 2:
            return _line(x, y)
    gain, bias = _prior_line(channel)
    if len(x) >= 3 and keep.all():
        keep = np.abs(y - (gain*x + bias)) <= 3.0
    if np.unique(x[keep]).size >= 2:
        return _line(x[keep], y[keep])
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd'])).astype(float)
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']].astype(float)
    for c in np.unique(lab):
        rows = np.asarray(cal['lab_id']) == c
        gain, bias = _calibrate(np.asarray(cal['cmd_stress'])[rows], np.asarray(cal['true_stress'])[rows], c)
        take = lab == c
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        x[take] = (x[take] - np.nanmedian(np.nanmean(panel, axis=0))) / np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
    cells = []
    for c in np.unique(lab):
        for stratum_id in np.unique(stratum[lab == c]):
            take = (lab == c) & (stratum == stratum_id)
            valid = np.isfinite(x[take]).sum(axis=0)
            mean = np.divide(np.nansum(x[take], axis=0), valid, out=np.full(x.shape[1], np.nan), where=valid != 0)
            cells.append((c, take, mean))
    for c, take, mean in cells:
        if np.isnan(mean).any():
            with np.errstate(invalid='ignore', divide='ignore'):
                local = np.array([v for lab_id, _, v in cells if lab_id == c])
                count = np.isfinite(local).sum(axis=0)
                local = np.divide(np.nansum(local, axis=0), count, out=np.zeros(x.shape[1]), where=count != 0)
            mean = np.where(np.isnan(mean), local, mean)
        x[take] = np.where(np.isnan(x[take]), mean, x[take])
    return np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times']), past, hold, x


def _basis(logtau, dt, t, past, hold, x):
    tau = np.exp(logtau)
    h = dt[:, None] / tau
    a = np.exp(-h)
    b = -np.expm1(-h)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for s in range(len(dt)):
        u = past[:, s, None]
        dz = a[s] * (dz + h[s] * (z - u))
        z = a[s] * z + b[s] * u
    gap = hold[:, None] - z
    k = t[:, None] / tau
    phi = -np.expm1(-k)
    dphi = -k * np.exp(-k)
    drift = np.column_stack((np.ones(len(hold)), x))
    curves = np.concatenate((gap[:, None, :] * phi, drift[:, None, :] * t[None, :, None]), axis=2)
    rates = np.column_stack((gap / tau, drift))
    dc = -dz[:, None, :] * phi + gap[:, None, :] * dphi
    dr = (-dz - gap) / tau
    return curves, rates, dc, dr


def _fit(dt, t, past, hold, x, yc, yr):
    y = np.r_[yc.ravel(), yr]
    n = yc.size
    low = np.log(min(dt.min(), t[t > 0].min()) / 20)
    high = np.log((dt.sum() + t.max()) * 20)
    bounds = (np.r_[[low]*2, [-np.inf]*7], np.r_[[high]*2, [np.inf]*7])
    scale = np.ones_like(y)
    last = None

    def design(u):
        nonlocal last
        if last is None or not np.array_equal(u, last[0]):
            c, r, dc, dr = _basis(u, dt, t, past, hold, x)
            last = (u.copy(), np.vstack((c.reshape(-1, 7), r)), np.vstack((dc.reshape(-1, 2), dr)))
        return last[1:]

    def fun(p):
        a, _ = design(p[:2])
        return (np.einsum('ij,j->i', a, p[2:]) - y) / scale

    def jac(p):
        a, d = design(p[:2])
        return np.column_stack((d * p[2:4], a)) / scale[:, None]

    starts = []
    for a, b in ((.25, 1.2), (.5, 3), (1., 5.), (2., 10.)):
        u = np.clip(np.log([a, b]), low + 1e-6, high - 1e-6)
        matrix, _ = design(u)
        w = np.linalg.lstsq(matrix, y, rcond=None)[0]
        starts.append(least_squares(fun, np.r_[u, w], jac=jac, x_scale='jac', bounds=bounds, max_nfev=80))
    best = min(starts, key=lambda r: r.cost)
    residual = fun(best.x)
    for sl in (slice(n), slice(n, None)):
        v = residual[sl]
        scale[sl] = max(1.4826 * np.median(np.abs(v - np.median(v))), 1e-6)
    p = least_squares(fun, best.x, jac=jac, loss='soft_l1', f_scale=2.5,
                      x_scale='jac', bounds=bounds, max_nfev=80).x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def solve_alpha_m97_c01_batch_02(dataset: dict) -> dict:
    dt, t, past, hold, x = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    p = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])
    curves, rates, _, _ = _basis(p[:2], dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:], regime_mode_weights=np.zeros(2),
                predicted_strain=test['strain_0'][:, None] + curves @ p[2:], initial_rate_pred=rates @ p[2:])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=item['dataset']; tr=d['train']
a=m.solve_alpha_m97_c01_batch_02(d); print({k:v for k,v in a.items() if v.size<10})
print('public gt schema',[(k,np.shape(v)) for k,v in item['gt'].items()] if isinstance(item['gt'],dict) else type(item['gt']))
cal=d['calibration_load_cells']
for lab in range(4):
 q=cal['lab_id']==lab; x,y=cal['cmd_stress'][q],cal['true_stress'][q]; g,b=m._calibrate(x,y,lab); print('calibration',lab,len(x),g,b,np.max(abs(y-g*x-b)))
dt,t,past,hold,x=m._prepare(d); n=len(tr['lab_id']); c,r,*_=m._basis(np.log(a['estimated_taus']),dt,t,past[:n],hold[:n],x[:n]); w=a['hereditary_weights']; dy=c@w+tr['strain_0'][:,None]-tr['strain_curves_obs']; dr=r@w-tr['instant_rate_obs']; print('fit residuals: rms, median abs, max abs',[(np.sqrt(np.mean(v*v)),np.median(abs(v)),np.max(abs(v))) for v in (dy,dr)])
PY
```
- **Container Exit Code**: `0` (`2405 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{'estimated_taus': array([0.54056113, 3.67168015]), 'hereditary_weights': array([ 0.67713786,  0.52818146,  0.13653608,  0.16286681, -0.12897331,
        0.12336928, -0.09362185]), 'regime_mode_weights': array([0., 0.])}
public gt schema [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
calibration 0 10 1.3400299423834945 0.14997668313572482 0.0002582160902684927
calibration 1 10 0.7700064844710877 -0.16997721760221676 0.00033584875306449935
calibration 2 10 1.2099979428688714 -0.12001293005243174 0.00024783127968208696
calibration 3 10 0.8299389094426014 0.11008922263461629 0.0002486197515758348
fit residuals: rms, median abs, max abs [(3.9855023202595503, 0.019662609505568973, 166.6831762054872), (0.043356116298390424, 0.01442620547854212, 0.21665601287822983)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31f1ac8155bdfc5a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31f1ac81 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29241 ms` LLM)
- **Rationale**: Compare clean public outputs by lab and refit with each training lab held out. Inspect the worst observation to distinguish isolated corruption from a preprocessing or model failure; those findings call for different fixes.
- **Alternatives Weighed**: `Change the flow basis immediately` (rejected: First determine whether the extreme residual is an isolated observation or a systematic error that persists by lab or time.)

```bash
cat > /tmp/rheology_validate.py <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
a=m.solve_alpha_m97_c01_batch_02(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'pred',a[k],'reference',g[j])
for k,j in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=a[k]-g[j]; print(k,'rmse by lab',[(int(c),round(np.sqrt(np.mean(e[te['lab_id']==c]**2)),6)) for c in np.unique(te['lab_id'])]); print('abs q50/q95/max',np.quantile(abs(e),[.5,.95,1]))
print('test stress max error',np.max(abs(past[n:]-g['te_hist_true'])),np.max(abs(hold[n:]-g['te_hold_true'])))
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; cp,rp,*_=m._basis(np.log(a['estimated_taus']),dt,t,past[:n],hold[:n],x[:n]); residual=yc-cp@a['hereditary_weights']; idx=np.unravel_index(np.argmax(abs(residual)),residual.shape); i,k=idx
print('largest residual',idx,'lab',tr['lab_id'][i],'raw feature',tr['coupon_features_raw'][i],'aligned',x[i],'observed curve delta',yc[i],'fitted curve delta',cp[i]@a['hereditary_weights'],'rate obs/fitted',yr[i],rp[i]@a['hereditary_weights'])
print('curve abs residual quantiles by time',np.quantile(abs(residual),[.5,.95,1],axis=0).round(4)); print('max feature abs by column',np.max(abs(x),axis=0))
for lab in np.unique(tr['lab_id']):
 q=tr['lab_id']==lab; p=m._fit(dt,t,past[:n][~q],hold[:n][~q],x[:n][~q],yc[~q],yr[~q]); c,r,*_=m._basis(p[:2],dt,t,past[:n][q],hold[:n][q],x[:n][q]); e1=c@p[2:]-yc[q]; e2=r@p[2:]-yr[q]; print('LOLO',lab,'taus',np.exp(p[:2]),'curve median/p95',np.quantile(abs(e1),[.5,.95]),'rate median/p95',np.quantile(abs(e2),[.5,.95]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_validate.py
```
- **Container Exit Code**: `0` (`843 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus pred [0.54056113 3.67168015] reference [0.54 3.9 ]
hereditary_weights pred [ 0.67713786  0.52818146  0.13653608  0.16286681 -0.12897331  0.12336928
 -0.09362185] reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
regime_mode_weights pred [0. 0.] reference [0. 0.]
predicted_strain rmse by lab [(0, 0.075148), (1, 0.048955), (2, 0.30426), (3, 0.044485)]
abs q50/q95/max [0.01646309 0.24310572 1.36181089]
initial_rate_pred rmse by lab [(0, 0.024408), (1, 0.01596), (2, 0.095161), (3, 0.01344)]
abs q50/q95/max [0.0069431  0.06898328 0.255661  ]
test stress max error 0.00024945654239560966 0.00021756432731012687
largest residual (118, 7) lab 1 raw feature [-0.07432666 -2.33158026  1.3982826   1.41897117] aligned [-0.4378669  -2.23086977  0.73186991  0.74830344] observed curve delta [-2.01738154e-01 -2.16362473e-01 -1.76950580e-01 -3.67423013e-02
  1.38999077e-01  2.96586052e-01  4.75763981e-01  1.67326044e+02
  8.40202889e-01  2.73498861e-01] fitted curve delta [-0.17359188 -0.24178207 -0.16623817 -0.03428349  0.12169651  0.28933433
  0.46379149  0.64286748  0.82554824  1.01111429] rate obs/fitted -0.810812911462429 -0.8361449486783439
curve abs residual quantiles by time [[1.200000e-02 1.390000e-02 1.380000e-02 1.640000e-02 2.100000e-02
  2.340000e-02 2.370000e-02 2.850000e-02 2.780000e-02 3.140000e-02]
 [9.250000e-02 1.623000e-01 2.319000e-01 4.081000e-01 5.304000e-01
  3.723000e-01 6.212000e-01 6.657000e-01 6.150000e-01 6.897000e-01]
 [3.122400e+00 1.410220e+01 1.201660e+01 4.994400e+00 1.371340e+01
  8.174000e-01 3.261700e+00 1.666832e+02 1.091580e+01 1.103800e+00]]
max feature abs by column [3.5617925  3.2605272  3.11534951 2.49959555]
LOLO 0 taus [0.53836588 3.15981825] curve median/p95 [0.02408884 0.50060614] rate median/p95 [0.01228994 0.09438474]
LOLO 1 taus [0.54254687 3.97778466] curve median/p95 [0.0179731  0.66274422] rate median/p95 [0.01323313 0.12054747]
LOLO 2 taus [0.53563428 3.42550206] curve median/p95 [0.02186193 0.42036514] rate median/p95 [0.01392096 0.05224258]
LOLO 3 taus [0.53696784 3.57499537] curve median/p95 [0.02406472 0.36550263] rate median/p95 [0.01405551 0.06300599]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31f1ac8155bdfc5a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31f1ac81 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32399 ms` LLM)
- **Rationale**: Use the provided clean references only to diagnose error, not to fit the solver. Holding physical parameters and stress fixed separates feature preprocessing error from fitting error; slicing by lab and missingness shows whether to inspect alignment or imputation first.
- **Alternatives Weighed**: `Tune the robust fit immediately` (rejected: Lab 2 has much larger clean-target error than the other labs despite accurate stress calibration; first isolate whether feature alignment accounts for that gap.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
c,r,*_=m._basis(np.log(g['taus']),dt,t,g['te_hist_true'],g['te_hold_true'],x[n:]); w=g['theta']; ec=g['te_clean']-te['strain_0'][:,None]-c@w; er=g['te_true_rate0']-r@w
missing=np.isnan(te['coupon_features_raw']); print('missing test pattern counts',np.unique(missing.sum(axis=1),return_counts=True)); print('oracle drift mismatch',np.max(abs(ec-er[:,None]*t)))
for lab in np.unique(te['lab_id']):
 for miss in [False,True]:
  q=(te['lab_id']==lab)&(missing.any(axis=1)==miss); print('lab/has_missing/n and drift q50,q95,max',lab,miss,q.sum(),np.quantile(abs(er[q]),[.5,.95,1]) if q.any() else [])
for k in np.argsort(abs(er))[-14:][::-1]: print('sample/lab/cell',k,te['lab_id'][k],te['stratum'][k], 'drift error',round(er[k],6),'raw',np.round(te['coupon_features_raw'][k],4),'aligned',np.round(x[n+k],4))
lab=np.r_[tr['lab_id'],te['lab_id']]; st=np.r_[tr['stratum'],te['stratum']]; raw=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); ref=np.vstack((tr['ref_coupons'],te['ref_coupons']));
for c0 in np.unique(lab):
 q=lab==c0; print('lab',c0,'ref mean/std medians',np.median(ref[q].mean(0)),np.median(ref[q].std(0)))
 for st0 in np.unique(st[q]):
  cell=q&(st==st0); print(' cell',st0,'n train/test',cell[:n].sum(),cell[n:].sum(),'missing',np.isnan(raw[cell]).sum(0))
PY
```
- **Container Exit Code**: `0` (`1332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[    nan -0.1838  0.1837  0.8012] aligned [0.4086 0.0604 0.4517 1.1092]
sample/lab/cell 21 2 1 drift error 0.142192 raw [-1.0196 -0.0963     nan -1.6326] aligned [-0.8297  0.1535  0.0362 -1.4823]
sample/lab/cell 24 0 2 drift error -0.065294 raw [ 1.3949 -0.5994  0.7893     nan] aligned [ 2.4743  0.0516  1.7386 -0.4079]
sample/lab/cell 62 2 0 drift error -0.064943 raw [-0.657  -0.3375 -0.5926     nan] aligned [-0.4435 -0.1033 -0.375  -0.2643]
sample/lab/cell 33 0 1 drift error 0.062964 raw [-0.7343 -0.0688  0.0947     nan] aligned [-0.1122  0.6962  0.8948  0.0422]
sample/lab/cell 55 1 1 drift error 0.060017 raw [ 1.2217     nan  1.5344 -0.4428] aligned [ 0.5916 -0.0118  0.84   -0.7306]
sample/lab/cell 17 3 1 drift error 0.047371 raw [-0.2994     nan  2.6936  3.4483] aligned [-0.7685 -0.1366  1.3356  1.8662]
sample/lab/cell 63 1 0 drift error -0.015882 raw [1.4587 0.7168    nan 0.1334] aligned [ 0.7799  0.1905 -0.1484 -0.2728]
sample/lab/cell 48 1 2 drift error -0.006627 raw [ 1.8101  1.4724  2.3172 -1.6441] aligned [ 1.059   0.7907  1.4618 -1.6848]
sample/lab/cell 5 1 0 drift error -0.005506 raw [ 1.0382 -0.4853  1.1021 -0.856 ] aligned [ 0.4458 -0.7643  0.4966 -1.0588]
sample/lab/cell 31 1 2 drift error 0.00477 raw [-2.3833 -0.3042 -0.0605 -0.0211] aligned [-2.272  -0.6205 -0.4269 -0.3956]
sample/lab/cell 58 1 1 drift error -0.004686 raw [ 3.3110e-01 -5.8080e-01  2.0616e+00  6.0000e-04] aligned [-0.1158 -0.8402  1.2587 -0.3783]
sample/lab/cell 46 1 0 drift error 0.004263 raw [ 0.7794  1.2753 -0.7959  1.909 ] aligned [ 0.2403  0.6342 -1.0111  1.1375]
lab 0 ref mean/std medians nan nan
 cell 0 n train/test 17 9 missing [1 0 1 1]
 cell 1 n train/test 17 5 missing [1 0 0 1]
 cell 2 n train/test 12 4 missing [1 2 1 1]
lab 1 ref mean/std medians nan nan
 cell 0 n train/test 8 7 missing [0 0 1 2]
 cell 1 n train/test 23 10 missing [1 2 2 1]
 cell 2 n train/test 11 3 missing [0 1 0 0]
lab 2 ref mean/std medians nan nan
 cell 0 n train/test 14 2 missing [1 0 0 2]
 cell 1 n train/test 7 5 missing [0 1 1 1]
 cell 2 n train/test 23 8 missing [2 1 1 0]
lab 3 ref mean/std medians nan nan
 cell 0 n train/test 16 7 missing [0 1 0 0]
 cell 1 n train/test 17 7 missing [0 1 1 0]
 cell 2 n train/test 15 5 missing [1 0 0 0]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31f1ac8155bdfc5a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31f1ac81 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`39432 ms` LLM)
- **Rationale**: The supplied observations contain large strain spikes. Compare robust fitting on all versus complete training rows, estimating residual scales inside each leave-one-lab-out fold. Change the fit only if the comparison improves identifiable complete-row behavior without materially worsening the missing-row slices.
- **Alternatives Weighed**: `Replace cell means with conditional feature predictions` (rejected: The task specifies lab-by-stratum column means, and clean-reference diagnostics locate the error in missing features rather than in the flow equation.)

```bash
cat > /tmp/rheology_fit_compare.py <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)

def predict(p,q):
 c,r,*_=m._basis(p[:2],dt,t,past[q],hold[q],x[q]); return c@p[2:],r@p[2:]

def refine(p,q,loss,include_missing):
 if not include_missing: q=q&complete
 y=np.column_stack((yc[q],yr[q])); c,r=predict(p,np.flatnonzero(q)); e=y-np.column_stack((c,r)); scale=np.maximum(1.4826*np.median(abs(e-np.median(e,axis=0)),axis=0),1e-6)
 def res(v):
  c,r=predict(v,np.flatnonzero(q)); return ((np.column_stack((c,r))-y)/scale).ravel()
 z=least_squares(res,p,loss=loss,f_scale=2.5,x_scale='jac',max_nfev=60).x
 return z

configs=[('baseline',None,True),('cauchy/all','cauchy',True),('cauchy/complete','cauchy',False),('softl1/complete','soft_l1',False)]
cv={k:[] for k,_,_ in configs}
for held in np.unique(lab):
 q=lab!=held; p=m._fit(dt,t,past[:n][q],hold[:n][q],x[:n][q],yc[q],yr[q]); h=~q
 for name,loss,inc in configs:
  z=p if loss is None else refine(p,q,loss,inc); c,r=predict(z,np.flatnonzero(h)); e1=abs(c-yc[h]); e2=abs(r-yr[h]); cc=complete[h]; cv[name].append([np.median(e1[cc]),np.quantile(e1[cc],.90),np.median(e2[cc]),np.quantile(e2[cc],.90),np.median(e1[~cc]),np.median(e2[~cc])])
p=m._fit(dt,t,past[:n],hold[:n],x[:n],yc,yr); tc=np.isfinite(te['coupon_features_raw']).all(axis=1)
for name,loss,inc in configs:
 z=p if loss is None else refine(p,np.ones(n,dtype=bool),loss,inc); c,r=predict(z,np.arange(n,len(x))); e1=c+te['strain_0'][:,None]-g['te_clean']; e2=r-g['te_true_rate0']; rm=lambda v:np.sqrt(np.mean(v*v))
 print(name,'LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med]',np.max(cv[name],axis=0).round(5),'taus',np.exp(z[:2]).round(5),'hw max error',round(np.max(abs(z[2:]-g['theta'])),5),'public complete worst-lab rmse',[round(max(rm(e[(te['lab_id']==j)&tc]) for j in np.unique(lab)),5) for e in (e1,e2)],'public all worst-lab rmse',[round(max(rm(e[te['lab_id']==j]) for j in np.unique(lab)),5) for e in (e1,e2)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_fit_compare.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS nonlinear least squares' | head -12
```
- **Container Exit Code**: `0` (`1190 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med] [0.02281 0.07264 0.01393 0.03808 0.28161 0.11225] taus [0.54056 3.67168] hw max error 0.02182 public complete worst-lab rmse [0.03101, 0.01095] public all worst-lab rmse [0.30426, 0.09516]
cauchy/all LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med] [0.01473 0.04588 0.0159  0.0392  0.2868  0.11372] taus [0.54298 3.92348] hw max error 0.00587 public complete worst-lab rmse [0.01097, 0.00472] public all worst-lab rmse [0.3045, 0.09617]
cauchy/complete LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med] [0.01409 0.04481 0.01583 0.03875 0.28749 0.11397] taus [0.54147 3.90167] hw max error 0.00426 public complete worst-lab rmse [0.00918, 0.00342] public all worst-lab rmse [0.30436, 0.09602]
softl1/complete LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med] [0.01404 0.04532 0.01559 0.03832 0.28731 0.11507] taus [0.54134 3.9058 ] hw max error 0.00366 public complete worst-lab rmse [0.00966, 0.00338] public all worst-lab rmse [0.30444, 0.0961]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31f1ac8155bdfc5a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31f1ac81 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`46697 ms` LLM)
- **Rationale**: Promote the challenger because it improves parameter recovery and held-out complete-row strain error without materially changing the missingness-limited worst-lab error. Check the exact edited artifact in an isolated process. The schema, held-out slices, and sparse or spiked calibration checks distinguish integration errors, fit regressions, and fallback failures.
- **Alternatives Weighed**: `Keep the baseline soft-L1 fit` (rejected: It estimates the slow time constant at 3.672 rather than 3.900 and has more than three times the complete-row worst-lab strain RMSE of the challenger.); `Change test-time imputation to improve the missing-row slice` (rejected: The task specifies cell means; changing them would violate the contract.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c01_batch_02_solver.py')
s = p.read_text()
s = s.replace('def _fit(dt, t, past, hold, x, yc, yr):', 'def _fit(dt, t, past, hold, x, yc, yr, complete=None):')
needle = "    if p[0] > p[1]:\n"
assert s.count(needle) == 1
s = s.replace(needle, '''    # Imputation is required for predictions, but imputed training rows
    # need not determine the physical parameters when there are enough
    # complete rows to identify all seven linear coefficients.
    if complete is None:
        complete = np.ones(len(hold), dtype=bool)
    rows = np.r_[np.repeat(complete, len(t)), complete]
    a, _ = design(p[:2])
    if np.linalg.matrix_rank(a[rows]) < 7:
        rows[:] = True
    residual = np.einsum('ij,j->i', a, p[2:]) - y
    by_target = np.column_stack((residual[:n].reshape(yc.shape), residual[n:]))
    v = by_target[rows[n:]]
    target_scale = np.maximum(1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0), 1e-6)
    scale[:] = np.r_[np.tile(target_scale[:-1], len(hold)), np.full(len(hold), target_scale[-1])]
    p = least_squares(lambda v: fun(v)[rows], p, jac=lambda v: jac(v)[rows],
                      loss='cauchy', f_scale=2.5, x_scale='jac', bounds=bounds, max_nfev=80).x
''' + needle)
needle = "    p = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])\n"
assert s.count(needle) == 1
s = s.replace(needle, "    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)\n    p = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'], complete)\n")
p.write_text(s)
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr=d['train']; te=d['test']; a=m.solve_alpha_m97_c01_batch_02(d); n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
assert all(np.isfinite(v).all() for v in a.values())
assert np.all(a['regime_mode_weights']==0)
print('taus',a['estimated_taus'],'max weight error',np.max(abs(a['hereditary_weights']-g['theta'])))
assert np.max(abs(a['estimated_taus']-g['taus'])) < .035
assert np.max(abs(a['hereditary_weights']-g['theta'])) < .01
ok=np.isfinite(te['coupon_features_raw']).all(axis=1)
for k,j,threshold,full_limit in [('predicted_strain','te_clean',.013,.31),('initial_rate_pred','te_true_rate0',.005,.10)]:
 e=a[k]-g[j]; by=lambda mask: max(np.sqrt(np.mean(e[q]**2)) for c in range(4) if (q:=(te['lab_id']==c)&mask).any()); full,complete,missing=by(np.ones(len(ok),bool)),by(ok),by(~ok); print(k,'worst-lab RMSE all/complete/missing',full,complete,missing); assert complete < threshold and full < full_limit
complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; cv=[]
for c in range(4):
 q=tr['lab_id']==c; ix=np.flatnonzero(~q); h=np.flatnonzero(q&complete); p=m._fit(dt,t,past[ix],hold[ix],x[ix],y[ix],r[ix],complete[ix]); b,v,*_=m._basis(p[:2],dt,t,past[h],hold[h],x[h]); cv.append([np.quantile(abs(b@p[2:]-y[h]),.9),np.quantile(abs(v@p[2:]-r[h]),.9)])
worst=np.max(cv,axis=0); print('LOLO complete-row worst P90 strain/rate',worst); assert np.all(worst < [.05,.045])
cal=d['calibration_load_cells']
for c in range(4):
 q=cal['lab_id']==c; z=cal['cmd_stress'][q]; w=cal['true_stress'][q]; expected=np.array(m._line(z,w)); spike=w.copy(); spike[np.argsort(z)[len(z)//2]]+=15
 for zx,wy in [(z[:0],w[:0]),(z[:1],w[:1]),(z[:2],w[:2]),(z[:3],w[:3]),(z,spike)]:
  assert np.allclose(m._calibrate(zx,wy,c),expected,atol=.005,rtol=0)
print('Sparse calibration and EMI checks passed for four channels')
PY
```
- **Container Exit Code**: `0` (`1213 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
taus [0.54146975 3.90167168] max weight error 0.004263354431329414
predicted_strain worst-lab RMSE all/complete/missing 0.3043645382574913 0.009176620341185639 0.5893917186636382
initial_rate_pred worst-lab RMSE all/complete/missing 0.09602103225612052 0.0034183077991709557 0.1859091227817877
LOLO complete-row worst P90 strain/rate [0.04480607 0.03874637]
Sparse calibration and EMI checks passed for four channels

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7828a9c93d768573 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7828a9c9 (directory).
```

### Turn 9 — `submit_completed_task` (`16511 ms` LLM)
- **Rationale**: The final artifact passed isolated-process schema, calibration, and held-out slice checks. The robust refinement improved parameter recovery and complete-coupon predictions without materially changing the missing-feature error under the prescribed imputation.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_02_solver.py': 'FILE:8154'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_02_solver.py`. The public schema and sparse/EMI calibration checks passed. Worst-lab strain and initial-rate RMSE were 0.00918 and 0.00342 on complete coupons, and 0.30436 and 0.09602 across all coupons. Missing coupon features remain the main source of prediction error.`

### Turn 10 — `execute_in_remote_docker` (`46550 ms` LLM)
- **Rationale**: An independent convolution can reveal a stress-coordinate, history-order, sign, or derivative error that would require a solver correction. Fit nonlinear residual terms only inside training-lab folds to check whether a missing basis term improves held-out margins rather than fitting nuisance variation. Leave the artifact unchanged during this audit.
- **Alternatives Weighed**: `Run relay-mode, HJB, and classification audits` (rejected: The specified model has continuous strain and rate outcomes, no relay or control state, and no class labels. Lab-held-out residuals are the applicable nuisance check.); `Add pairwise sums and differences as standalone terms` (rejected: These are already spanned by the four linear coupon features; nonlinear interactions and saturation supply a meaningful completeness test.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr=d['train']; te=d['test']; a=m.solve_alpha_m97_c01_batch_02(d); dt,t,past,hold,x=m._prepare(d); n=len(tr['lab_id']); tau=a['estimated_taus']; w=a['hereditary_weights']
# Closed-form convolution checks history order, stress coordinates, signs, and the initial derivative independently.
ends=np.cumsum(dt); lag=ends[-1]-ends
mem=past[n:] @ ((1-np.exp(-dt[:,None]/tau))*np.exp(-lag[:,None]/tau))
gap=hold[n:,None]-mem; drift=w[2]+x[n:]@w[3:]; curve=te['strain_0'][:,None]+(gap*w[:2])@(1-np.exp(-t[None,:]/tau[:,None]))+drift[:,None]*t
rate=(gap*w[:2])@(1/tau)+drift
np.testing.assert_allclose(curve,a['predicted_strain'],rtol=1e-12,atol=1e-12); np.testing.assert_allclose(rate,a['initial_rate_pred'],rtol=1e-12,atol=1e-12); np.testing.assert_array_equal(a['regime_mode_weights'],[0,0]); print('Independent convolution and derivative passed')
print('regime inputs', {k:d[k] for k in d if ('band' in k.lower() or 'cryst' in k.lower()) and np.size(d[k])==1}); print('top-level fields',sorted(d))
# Test pairwise/three-way drift, a rational saturation correction, and a nonlinear time correction against held-out labs.
masks=np.isfinite(tr['coupon_features_raw']).all(1); lab=tr['lab_id']; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; X=x[:n]; ii,jj=np.triu_indices(X.shape[1],1); terms=np.column_stack((X[:,ii]*X[:,jj],np.prod(X[:,[0,1,2]],axis=1),np.prod(X[:,[0,1,3]],axis=1),np.prod(X[:,[0,2,3]],axis=1),np.prod(X[:,[1,2,3]],axis=1),X/(1+X**2)-X)); y=np.column_stack((yc,yr)); score=[]
for c in np.unique(lab):
 fit=lab!=c; take=fit&masks; test=~fit&masks
 p=m._fit(dt,t,past[:n][fit],hold[:n][fit],X[fit],yc[fit],yr[fit],masks[fit]); b,r,*_=m._basis(p[:2],dt,t,past[:n],hold[:n],X); resid=y-np.column_stack((b@p[2:],r@p[2:])); scale=np.maximum(1.4826*np.median(abs(resid[take]-np.median(resid[take],axis=0)),axis=0),1e-6)
 f=np.r_[t,1.]; ar=terms[:,None,:]*f[None,:,None]; arc=np.broadcast_to(np.r_[np.log1p(t)-t,0][None,:,None],(n,len(f),1)); design=np.concatenate((ar,arc),axis=2); model=least_squares(lambda v:((design[take]@v-resid[take])/scale).ravel(),np.zeros(design.shape[-1]),loss='cauchy',f_scale=2.5,max_nfev=60); ext=resid-design@model.x
 pair=[]
 for z in (resid[test],ext[test]): pair.append(np.array([np.quantile(abs(z[:,:-1]),.9),np.quantile(abs(z[:,-1]),.9)]))
 score.append(pair); print('held-out lab',int(c),'base/augmented P90',np.round(pair,6),'base signed median late/rate',np.round(np.median(resid[test][:,[-2,-1]],axis=0),6))
print('worst P90 by model',np.max(score,axis=0))
for k,j in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=a[k]-g[j]; print(k,'pooled RMSE',np.sqrt(np.mean(e**2)))
print('absolute tau/weight margins',np.max(abs(tau-g['taus'])),np.max(abs(w-g['theta'])))
PY
```
- **Container Exit Code**: `0` (`1556 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Independent convolution and derivative passed
regime inputs {'shear_banding_index': 0.12}
top-level fields ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress']
held-out lab 0 base/augmented P90 [[0.033063 0.028617]
 [0.03434  0.029297]] base signed median late/rate [ 0.002486 -0.004635]
held-out lab 1 base/augmented P90 [[0.044806 0.023657]
 [0.044925 0.023685]] base signed median late/rate [-0.007321 -0.002778]
held-out lab 2 base/augmented P90 [[0.034735 0.026473]
 [0.037392 0.026839]] base signed median late/rate [0.00646 0.00054]
held-out lab 3 base/augmented P90 [[0.035127 0.038746]
 [0.034848 0.037721]] base signed median late/rate [-0.007169  0.000322]
worst P90 by model [[0.04480607 0.03874637]
 [0.0449255  0.03772058]]
predicted_strain pooled RMSE 0.1458092386550015
initial_rate_pred pooled RMSE 0.04603589026735011
absolute tau/weight margins 0.0016716794736946206 0.004263354431329414

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7828a9c93d768573 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7828a9c9 (directory).
```

### Turn 11 — `submit_completed_task` (`15680 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated-process validation and matched an independent closed-form convolution and initial derivative. Held-out lab results did not justify departing from the specified linear-flow model.
- **Alternatives Weighed**: `Add nonlinear interaction, saturation, and flow terms` (rejected: In held-out labs, the larger basis slightly worsened worst-lab strain P90 and only marginally improved rate P90. It did not address the dominant missing-feature error.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_02_solver.py': 'FILE:8154'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_02_solver.py`. The public schema, sparse and EMI calibration checks, and independent strain and initial-rate calculations passed. Worst-lab strain and initial-rate RMSE were 0.00918 and 0.00342 on complete coupons, and 0.30436 and 0.09602 across all coupons. Missing coupon features were the main source of error.`

### Turn 12 — `execute_in_remote_docker` (`25798 ms` LLM)
- **Rationale**: The previous calibration check tested sparse sweeps and EMI separately. Combine them using actual calibration subsets and measured test stresses to detect whether a spike can steer a short-sweep fit. If it can, correct the spike-selection rule without changing the verified flow model.
- **Alternatives Weighed**: `Repeat the nonlinear basis and lab-confounding audit` (rejected: The prior held-out audit tested those alternatives and found no meaningful improvement. Relay states and classification metrics are not part of this regression task.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,itertools,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; cal=d['calibration_load_cells']; te=d['test']; g=i['gt']
for k in (3,4,5):
 count=bad=0; worst=(0,None)
 for c in range(4):
  q=cal['lab_id']==c; X,Y=cal['cmd_stress'][q],cal['true_stress'][q]; take=te['lab_id']==c; probe=te['stress_hold_cmd'][take]; expected=g['te_hold_true'][take]
  for ix in itertools.combinations(range(len(X)),k):
   z,w=X[list(ix)],Y[list(ix)]
   for j in range(k):
    for delta in (-4.,4.):
     v=w.copy(); v[j]+=delta; gain,bias=m._calibrate(z,v,c); e=np.max(abs(gain*probe+bias-expected)); count+=1; bad+=e>.01
     if e>worst[0]: worst=(e,(c,ix,j,delta,gain,bias))
 print('calibration rows',k,'stress errors > .01',bad,'/',count,'worst',worst)
PY
```
- **Container Exit Code**: `0` (`4276 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration rows 3 stress errors > .01 48 / 2880 worst (0.014240505057506736, (1, (0, 2, 9), 0, -4.0, 0.7755729911076914, -0.1709079848961183))
calibration rows 4 stress errors > .01 14 / 6720 worst (0.011108816545580158, (1, (0, 1, 4, 7), 0, -4.0, 0.7674764570577147, -0.16490402594567422))
calibration rows 5 stress errors > .01 0 / 10080 worst (0.002076748525410288, (2, (0, 1, 2, 4, 9), 2, -4.0, 1.2106216605079176, -0.12115851331238758))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7828a9c93d768573 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7828a9c9 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`42339 ms` LLM)
- **Rationale**: The audit found a calibration failure, not a missing flow term: after spike removal, too narrow a command span can amplify measurement noise into prediction error. Borrow only a compatible archived slope in that case, retain the in-batch level, and rerun the measured-subset stress audit and public checks on the edited artifact.
- **Alternatives Weighed**: `Use the archived affine calibration for every EMI-affected channel` (rejected: That would discard sufficient clean in-batch points and could carry an older bias into the current batch.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c01_batch_02_solver.py')
s=p.read_text()
old='''    if np.unique(x[keep]).size >= 2:
        return _line(x[keep], y[keep])
    return gain, bias
'''
new='''    if np.unique(x[keep]).size >= 2:
        clean_x, clean_y = x[keep], y[keep]
        fitted_gain, fitted_bias = _line(clean_x, clean_y)
        # A short command span cannot reliably estimate a slope after spike
        # removal. Use the archived slope only if compatible with the clean
        # in-batch contrasts, and retain the in-batch level.
        x_centered = clean_x - clean_x.mean()
        span = np.ptp(clean_x)
        if keep.sum() < 4 and span < 0.2:
            contrasts = clean_y - clean_y.mean() - gain * x_centered
            if np.ptp(contrasts) <= 0.001:
                return gain, clean_y.mean() - gain * clean_x.mean()
        return fitted_gain, fitted_bias
    return gain, bias
'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,itertools,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; cal=d['calibration_load_cells']; te=d['test']; g=i['gt']
a=m.solve_alpha_m97_c01_batch_02(d)
assert all(np.isfinite(v).all() for v in a.values())
np.testing.assert_array_equal(a['regime_mode_weights'],[0,0])
assert np.max(abs(a['estimated_taus']-g['taus'])) < .002
assert np.max(abs(a['hereditary_weights']-g['theta'])) < .005
complete=np.isfinite(te['coupon_features_raw']).all(1)
for key,ref,limit,complete_limit in [('predicted_strain','te_clean',.305,.01),('initial_rate_pred','te_true_rate0',.097,.004)]:
 e=a[key]-g[ref]; rms=lambda mask:max(np.sqrt(np.mean(e[q]**2)) for c in range(4) if (q:=(te['lab_id']==c)&mask).any()); all_rmse=rms(np.ones(len(complete),bool)); complete_rmse=rms(complete); print(key,'worst-lab all/complete RMSE',all_rmse,complete_rmse); assert all_rmse<limit and complete_rmse<complete_limit
for k in (3,4,5):
 count=bad=0; worst=0.
 for c in range(4):
  q=cal['lab_id']==c; X,Y=cal['cmd_stress'][q],cal['true_stress'][q]; take=te['lab_id']==c; probe=te['stress_hold_cmd'][take]; expected=g['te_hold_true'][take]
  if k==3:
   for num in (0,1,2):
    gain,bias=m._calibrate(X[:num],Y[:num],c); assert np.max(abs(gain*probe+bias-expected)) < .01
  for ix in itertools.combinations(range(len(X)),k):
   z,w=X[list(ix)],Y[list(ix)]
   for j in range(k):
    for delta in (-4.,4.):
     v=w.copy(); v[j]+=delta; gain,bias=m._calibrate(z,v,c); e=np.max(abs(gain*probe+bias-expected)); count+=1; bad+=e>.01; worst=max(worst,e)
 print('calibration rows',k,'stress errors > .01',bad,'/',count,'worst',worst)
 assert bad==0
PY
```
- **Container Exit Code**: `1` (`1924 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
predicted_strain worst-lab all/complete RMSE 0.3043645382574913 0.009176620341185639
initial_rate_pred worst-lab all/complete RMSE 0.09602103225612052 0.0034183077991709557
calibration rows 3 stress errors > .01 16 / 2880 worst 0.011079948591770306

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=65aa38f1ab1e4556 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_65aa38f1 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 26, in <module>
AssertionError
```

### Turn 14 — `execute_in_remote_docker` (`31281 ms` LLM)
- **Rationale**: Compare the current cutoff with a contrast-only rule on measured calibration subsets and on both hold and history stresses. Print the remaining worst case to distinguish a faulty spike decision from slope extrapolation. Leave the deliverable unchanged until the comparison shows whether the new rule closes the margin.
- **Alternatives Weighed**: `Increase the command-span cutoff` (rejected: A wider fixed cutoff would still be an indirect test. Check whether clean in-batch stress differences directly support the archived slope, while preserving the in-batch intercept.)

```bash
cat > /tmp/rheology_emi_compare.py <<'PY'
import functools
import importlib.util
import itertools
import pickle
import numpy as np

path = '/holdout_task/alpha_m97_c01_batch_02_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
cal, te = d['calibration_load_cells'], d['test']
m._prior_line = functools.lru_cache(None)(m._prior_line)
source = open(path).read()
needle = 'if keep.sum() < 4 and span < 0.2:'
variants = {'current': m._calibrate}
if needle in source:
    scope = {}
    exec(compile(source.replace(needle, 'if keep.sum() < 4:' ).replace('        span = np.ptp(clean_x)\n', ''), path, 'exec'), scope)
    scope['_prior_line'] = m._prior_line
    variants['compatible-prior'] = scope['_calibrate']
for c in range(4):
    q = cal['lab_id'] == c
    print('channel', c, 'archived minus full in-batch gain/bias', np.array(m._prior_line(c)) - m._line(cal['cmd_stress'][q], cal['true_stress'][q]))
for k in (3, 4, 5):
    for name, fit in variants.items():
        count = bad = 0
        worst = (0., None)
        for c in range(4):
            q = cal['lab_id'] == c
            x, y = cal['cmd_stress'][q], cal['true_stress'][q]
            sel = te['lab_id'] == c
            probe = np.r_[te['stress_hold_cmd'][sel], te['past_stress_cmd'][sel].ravel()]
            ref = np.r_[gt['te_hold_true'][sel], gt['te_hist_true'][sel].ravel()]
            for ix in itertools.combinations(range(len(x)), k):
                z, w = x[list(ix)], y[list(ix)]
                for j in range(k):
                    for spike in (-4., 4.):
                        v = w.copy(); v[j] += spike
                        a, b = fit(z, v, c)
                        error = np.max(abs(a*probe + b - ref))
                        count += 1; bad += error > .01
                        if error > worst[0]:
                            keep = np.arange(k) != j
                            worst = (error, (c, ix, j, spike, np.ptp(z[keep]), np.ptp(w[keep] - m._prior_line(c)[0]*z[keep])))
        print(k, name, 'above .01', bad, '/', count, 'worst (error; channel, rows, spike row, size, clean span, prior contrast range)', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_emi_compare.py
```
- **Container Exit Code**: `0` (`4222 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
channel 0 archived minus full in-batch gain/bias [-3.05503109e-05  1.71684536e-05]
channel 1 archived minus full in-batch gain/bias [-2.55519969e-06 -1.67347402e-05]
channel 2 archived minus full in-batch gain/bias [2.90552561e-05 9.51243310e-06]
channel 3 archived minus full in-batch gain/bias [ 7.09903686e-05 -1.16458922e-04]
3 current above .01 16 / 2880 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.0112834619216029, (1, (0, 1, 7), 0, -4.0, 0.26317899999999983, 0.0006627833181507281))
3 compatible-prior above .01 0 / 2880 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00030449920711506095, (3, (0, 2, 4), 0, -4.0, 0.924455, 0.00014910480802501902))
4 current above .01 14 / 6720 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.011313066212137457, (1, (0, 1, 4, 7), 0, -4.0, 0.26317899999999983, 0.0006627833181507281))
4 compatible-prior above .01 0 / 6720 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00026950563452565746, (3, (0, 2, 4, 5), 0, -4.0, 1.0335509999999999, 0.00017953312178067837))
5 current above .01 0 / 10080 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00215882158421854, (2, (0, 1, 2, 4, 9), 2, -4.0, 0.715141, 0.00044551868020148433))
5 compatible-prior above .01 0 / 10080 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00215882158421854, (2, (0, 1, 2, 4, 9), 2, -4.0, 0.715141, 0.00044551868020148433))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=65aa38f1ab1e4556 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_65aa38f1 (directory).
```

### Turn 15 — `execute_in_remote_docker` (`34675 ms` LLM)
- **Rationale**: Apply the tested contrast rule while preserving direct in-batch fitting whenever four clean readings remain. Verify the saved file in isolated processes to catch both sparse-calibration regressions and any unintended change to public predictions.
- **Alternatives Weighed**: `Retain the fixed command-span cutoff` (rejected: It left 30 three- and four-row cases above 0.01 stress error. The contrast-based rule eliminated all 30 without changing the five-row results.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c01_batch_02_solver.py')
s = p.read_text()
old = '''        # A short command span cannot reliably estimate a slope after spike
        # removal. Use the archived slope only if compatible with the clean
        # in-batch contrasts, and retain the in-batch level.
        x_centered = clean_x - clean_x.mean()
        span = np.ptp(clean_x)
        if keep.sum() < 4 and span < 0.2:
'''
new = '''        # With fewer than four clean readings after EMI rejection, borrow
        # the archived slope only if supported by the in-batch contrasts.
        # Keep the stress level anchored to the current measurements.
        x_centered = clean_x - clean_x.mean()
        if keep.sum() < 4:
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/rheology_emi_compare.py > /tmp/rheology_emi_final.log
cat /tmp/rheology_emi_final.log
python3 -I - <<'PY'
import importlib.util, pickle, re
from pathlib import Path
import numpy as np
lines = Path('/tmp/rheology_emi_final.log').read_text()
rows = re.findall(r'^(\d) current above \.01 (\d+) / (\d+) worst [^\n]+\((\d+\.\d+(?:e[+-]\d+)?), \(', lines, re.M)
assert [(int(k), int(n)) for k, _, n, _ in rows] == [(3,2880),(4,6720),(5,10080)], rows
assert all(int(bad) == 0 and float(worst) < cap for (_,bad,_,worst),cap in zip(rows,[.0004,.0004,.0023])), rows
path = '/holdout_task/alpha_m97_c01_batch_02_solver.py'
spec = importlib.util.spec_from_file_location('solver',path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']; te = d['test']; cal = d['calibration_load_cells']
a = m.solve_alpha_m97_c01_batch_02(d)
assert all(np.isfinite(v).all() for v in a.values())
np.testing.assert_array_equal(a['regime_mode_weights'], [0,0])
assert np.max(abs(a['estimated_taus']-g['taus'])) < .002
assert np.max(abs(a['hereditary_weights']-g['theta'])) < .005
complete = np.isfinite(te['coupon_features_raw']).all(1)
for k,j,limit,clean_limit in [('predicted_strain','te_clean',.305,.01),('initial_rate_pred','te_true_rate0',.097,.004)]:
    e = a[k]-g[j]
    def worst(mask):
        return max(np.sqrt(np.mean(e[q]**2)) for c in range(4) if (q := ((te['lab_id']==c)&mask)).any())
    full, clean = worst(np.ones_like(complete)), worst(complete)
    print(k, 'worst-lab RMSE all/complete', full, clean)
    assert full < limit and clean < clean_limit
for c in range(4):
    q = cal['lab_id']==c; x=cal['cmd_stress'][q]; y=cal['true_stress'][q]; t=te['lab_id']==c
    for n in [0,1,2]:
        gain,bias=m._calibrate(x[:n],y[:n],c)
        assert np.max(abs(gain*te['stress_hold_cmd'][t]+bias-g['te_hold_true'][t])) < .01
print('Exact artifact passed clean-process public and sparse/EMI checks.')
PY
```
- **Container Exit Code**: `0` (`3278 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
channel 0 archived minus full in-batch gain/bias [-3.05503109e-05  1.71684536e-05]
channel 1 archived minus full in-batch gain/bias [-2.55519969e-06 -1.67347402e-05]
channel 2 archived minus full in-batch gain/bias [2.90552561e-05 9.51243310e-06]
channel 3 archived minus full in-batch gain/bias [ 7.09903686e-05 -1.16458922e-04]
3 current above .01 0 / 2880 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00030449920711506095, (3, (0, 2, 4), 0, -4.0, 0.924455, 0.00014910480802501902))
4 current above .01 0 / 6720 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00026950563452565746, (3, (0, 2, 4, 5), 0, -4.0, 1.0335509999999999, 0.00017953312178067837))
5 current above .01 0 / 10080 worst (error; channel, rows, spike row, size, clean span, prior contrast range) (0.00215882158421854, (2, (0, 1, 2, 4, 9), 2, -4.0, 0.715141, 0.00044551868020148433))
predicted_strain worst-lab RMSE all/complete 0.3043645382574913 0.009176620341185639
initial_rate_pred worst-lab RMSE all/complete 0.09602103225612052 0.0034183077991709557
Exact artifact passed clean-process public and sparse/EMI checks.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=322772295adad557 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_32277229 (directory).
```

### Turn 16 — `submit_completed_task` (`15757 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process schema, prediction, and sparse calibration checks. The calibration change closed the EMI margin on measured subsets while preserving direct in-batch fits when at least four clean readings remained.
- **Alternatives Weighed**: `Retain the fixed command-span cutoff` (rejected: It left 30 sparse EMI cases above 0.01 stress error. The contrast-based rule eliminated those failures without changing public predictions.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_02_solver.py`. The public schema and sparse calibration checks passed. Across 19,680 injected EMI cases, the maximum stress error was 0.00216. Worst-lab strain and initial-rate RMSE were 0.00918 and 0.00342 on complete coupons, versus 0.30436 and 0.09602 across all coupons. Missing coupon features remain the main source of prediction error.`

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
