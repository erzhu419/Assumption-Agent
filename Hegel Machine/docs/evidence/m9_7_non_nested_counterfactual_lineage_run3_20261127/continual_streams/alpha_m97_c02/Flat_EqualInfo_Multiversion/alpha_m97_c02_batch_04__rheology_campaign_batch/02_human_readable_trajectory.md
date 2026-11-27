# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `391,341` (`376,285` / `91,392` / `15,056`) |
| **Total Episode Wall Time (s)** | `468.13s` |

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

### Turn 1 — `execute_in_remote_docker` (`14367 ms` LLM)
- **Rationale**: Inspect the actual schema and archive manifest before choosing a fit procedure; a mismatch in split, feature, or calibration layout would change the parser or alignment method.

```bash
cd /holdout_task && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && cat /tmp/hegel_continual_archive/archive_manifest.json && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)
def show(d, depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,getattr(v,'shape',None),repr(v)[:220] if not isinstance(v,dict) else '')
  if isinstance(v,dict): show(v,depth+2)
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`436 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "regime_mode_weights"
    ],
    "input_coordinate_contract": "raw_measured",
    "solver_path": "/tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py"
  },
  {
    "episode_short_name": "alpha_m97_c02_batch_03",
    "task_id": "alpha_m97_c02_batch_03__rheology_campaign_batch",
    "entrypoint": "solve_alpha_m97_c02_batch_03",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "regime_mode_weights"
    ],
    "input_coordinate_contract": "raw_measured",
    "solver_path": "/tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py"
  }
]autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "<stdin>", line 5, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19309 ms` LLM)
- **Rationale**: The validator shows that the pickle wraps the dataset in a list. Inspect the corrected schema and prior alignment code before fitting: calibration coverage determines whether archive fallback is permitted, and the observed label layout determines the regression objective.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: items=pickle.load(f)
print('items',len(items),'first-item keys',list(items[0]))
def show(d,depth=0):
 for k,v in d.items():
  p=' '*depth
  if isinstance(v,dict):
   print(p,k,'dict'); show(v,depth+2)
  elif isinstance(v,np.ndarray):
   print(p,k,v.shape,v.dtype,repr(v.reshape(-1)[:8]))
  else:
   print(p,k,type(v).__name__,repr(v)[:360])
show(items[0]['dataset'])
PY
sed -n '1,235p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -55
```
- **Container Exit Code**: `0` (`421 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`18390 ms` LLM)
- **Rationale**: Read the archive's calibration fields and prior panel implementation to determine how to select clean historical sweeps for channels 0, 2, and 3. Count complete training rows per lab before choosing whether to exclude imputed rows from parameter fitting.
- **Alternatives Weighed**: `Use the rejected gate's calibration payload as fixed coefficients` (rejected: Three channels need archive fallback, but channel 1 has two distinct in-batch readings that identify its affine map directly.)

```bash
sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('Calibration history:',json.dumps(h)[:7600]); d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; print('Dataset keys:',list(d)); print('Test keys:',list(d['test'])); print('Calibration:',d['calibration_load_cells']);
for k in ('train','test'):
 b=d[k]; print(k,[(int(c),int(np.sum(b['lab_id']==c)),int(np.sum(np.isfinite(b['coupon_features_raw'][b['lab_id']==c]).all(axis=1)))) for c in range(4)])
PY
```
- **Container Exit Code**: `0` (`462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "cmd_stress": [2.117655, -1.165749, -1.673814, -1.249019, -1.48004, 2.119091, 1.208208, 0.301724, 0.137891, -1.560518, -0.944846, -1.229132, 0.019873, -1.425842, 1.95715, -1.991932, 1.147843, -1.750378, 2.055936, 1.428677, -1.812528, 0.193572, -0.712096, 0.209373, -1.182443, -1.167675, -2.02483, 0.645453, -0.272135, -0.243067, -0.560621, -0.24358, -1.962731, 0.308464, -1.386231, -0.387364, -1.555944, 0.04562, 0.946391, 0.974083], "true_stress": [1.532903, -1.060971, -1.462342, -1.126581, -1.309376, 1.53423, 0.814577, 0.09824, -0.031172, -1.372753, -1.07762, -1.45007, 0.186019, -1.707929, 2.723961, -2.449405, 1.663513, -2.133159, 2.853096, 2.031436, -1.44056, 0.264403, -0.505239, 0.277706, -0.905162, -0.892615, -1.621148, 0.648664, -0.131155, -0.106387, -0.825134, -0.432034, -2.564003, 0.252585, -1.848978, -0.610312, -2.059283, -0.07338, 1.043509, 1.077867]}, {"episode_short_name": "alpha_m97_c02_batch_03", "lab_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "cmd_stress": [0.946041, 0.026539, -1.868094, 0.186519, 1.478584, -1.876496, 0.722244, -0.491277, 1.000356, -1.752239, -0.257067, -0.430637, 0.442373, 1.855268, 0.321991, -1.993656, -2.094652, -1.028239, -1.643449, 0.129373, 2.036062, 2.193845, -1.436971, -1.129126, 1.519563, 0.48526, -1.347118, -1.835224, -0.864378, -0.591004, -1.143398, 0.664015, -0.898756, 1.989517, 1.564437, -1.108175, -0.501381, 0.364932, -1.903257, 0.817037], "true_stress": [0.607655, -0.119198, -1.615612, 0.007318, 1.028166, -1.622395, 0.430456, -0.528034, 0.650233, -1.524129, -0.176842, -0.404107, 0.739445, 2.59022, 0.581818, -2.451662, -2.584005, -1.187026, -1.992969, 0.329584, 1.830528, 1.964853, -1.121335, -0.859911, 1.391591, 0.512527, -1.045019, -1.459926, -0.634564, -0.402297, -1.547764, 0.69341, -1.244464, 2.336847, 1.809704, -1.503979, -0.751799, 0.322649, -2.49024, 0.883051]}]
Dataset keys: ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
Test keys: ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry']
Calibration: {'lab_id': array([1, 1]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-1.54314271,  1.99396734])}
train [(0, 47, 41), (1, 37, 34), (2, 40, 37), (3, 56, 48)]
test [(0, 22, 19), (1, 14, 12), (2, 16, 13), (3, 20, 19)]
```

### Turn 4 — `execute_in_remote_docker` (`68162 ms` LLM)
- **Rationale**: Build the solver and test calibration transfer by holding out each lab's strain and rate labels. Reporting complete and imputed rows separately will distinguish a shared-law or calibration failure from imputation error.
- **Alternatives Weighed**: `Fit strain and rate with separate regressors` (rejected: A joint fit enforces the specified initial derivative of the strain law and keeps the returned weights consistent with both predictions.)

```bash
set -e
cd /holdout_task
sed -n '55,108p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py
cat > alpha_m97_c02_batch_04_solver.py <<'PY'
"""Calibrated two-branch Prony rheology with a damage-gated logarithmic plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _clean(x, y):
    good = np.ones(len(x), dtype=bool)
    if len(x) < 3:
        return good
    # Select the most coherent majority before measuring EMI residuals.
    best = (np.inf, np.inf)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] == x[i]:
                continue
            g = (y[j] - y[i]) / (x[j] - x[i])
            r = np.abs((y - y[i]) - g*(x - x[i]))
            score = (np.median(r), np.mean(np.sort(r)[:len(x)-1]))
            if score < best:
                best, good = score, r <= 3.0
    return good


def _calibrate(sweep):
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _channel(sweep, lab)
        good = _clean(x, y)
        if good.sum() >= 4 or (good.all() and len(x) >= 2):
            result[lab] = _affine(x[good], y[good])
            continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        for archived_sweep in reversed(history):
            ax, ay = _channel(archived_sweep, lab)
            clean = _clean(ax, ay)
            if clean.sum() >= 2 and np.ptp(ax[clean]) > 0:
                result[lab] = _affine(ax[clean], ay[clean])
                break
        else:
            raise ValueError(f'No usable calibration for lab {lab}')
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    refs, raw = join('ref_coupons'), join('coupon_features_raw')
    aligned = np.empty_like(raw, dtype=float)
    for c in range(4):
        use = lab == c
        ref = refs[use]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[use] = (raw[use] - center) / scale
    # First compute each cell mean; an unavailable cell mean borrows equally
    # from observed cells rather than from cells in proportion to their count.
    cells = [(lab == c) & (stratum == s) for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    means = np.array([np.divide(np.nansum(aligned[use], axis=0), count,
                                out=np.full(raw.shape[1], np.nan), where=count > 0)
                      for use in cells for count in [np.isfinite(aligned[use]).sum(axis=0)]])
    fallback = np.nan_to_num(np.nanmean(means, axis=0))
    for use, mean in zip(cells, means):
        aligned[use] = np.where(np.isfinite(aligned[use]), aligned[use],
                                np.where(np.isfinite(mean), mean, fallback))
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    dt = np.asarray(dataset['past_dt'])
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)**2 @ dt
    act = np.tanh(damage) if dataset['shear_banding_index'] <= 0.50 else np.zeros_like(damage)
    return aligned, past, hold, act


def _design(aligned, past, hold, act, dt, times, taus):
    # Unroll the exact recurrence: each step's input is attenuated by the
    # remaining history duration, not by the step's own duration twice.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (-np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus))
    gap = hold[:, None] - memory
    relax = -np.expm1(-times[:, None]/taus)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    terminal = np.where(act[:, None] > 0, np.log1p(times), times)
    curve = np.empty((len(hold), len(times), 9))
    curve[:, :, :2] = gap[:, None, :]*relax
    curve[:, :, 2:7] = drift[:, None, :]*terminal[:, :, None]
    curve[:, :, 7] = (act*hold)[:, None]*relax[:, 0]
    curve[:, :, 8] = (act*memory[:, 0])[:, None]*terminal
    rate = np.column_stack((gap/taus, drift, act*hold/taus[0], act*memory[:, 0]))
    return curve, rate


def _fit(aligned, past, hold, act, dt, times, y, r):
    # Variable projection: only the two relaxation times need a nonlinear fit.
    rate_scale = 1.0
    y_all = np.r_[y.ravel(), rate_scale*r]

    def profile(logtau, output=False):
        taus = np.exp(logtau)
        curve, rate = _design(aligned, past, hold, act, dt, times, taus)
        a = np.vstack((curve.reshape(-1, 9), rate_scale*rate))
        w = np.linalg.lstsq(a, y_all, rcond=None)[0]
        return (taus, w) if output else np.einsum('ij,j->i', a, w) - y_all

    starts = ((0.36, 1.8), (0.65, 4.0), (1.25, 7.0), (2.7, 0.6))
    fits = [least_squares(profile, np.log(start), bounds=(np.log(0.035), np.log(60.)),
                         max_nfev=100, ftol=1e-10, xtol=1e-9) for start in starts]
    return profile(min(fits, key=lambda fit: fit.cost).x, output=True)


def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                         y[complete], train['instant_rate_obs'][complete])
    curve, rate = _design(*(a[n:] for a in data), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c02_batch_04_solver as m
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
t=d['train']; n=len(t['lab_id']); a=tuple(x[:n] for x in m._prepare(d)); dt,ts=d['past_dt'],d['eval_times']; y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ok=np.isfinite(t['coupon_features_raw']).all(1)
print('calibration',m._calibrate(d['calibration_load_cells']).round(7).tolist(),'damage active',np.sum(a[-1]>0),'of',n)
for c in [None,0,1,2,3]:
 fit=ok if c is None else ok & (t['lab_id']!=c); val=ok if c is None else (t['lab_id']==c)
 tau,w=m._fit(*(x[fit] for x in a),dt,ts,y[fit],r[fit]); x,z=m._design(*(x[val] for x in a),dt,ts,tau)
 for u in (True,False):
  mask=ok[val]==u
  if mask.any(): print('lab',c,'complete',u,'n',mask.sum(),'curve/rate rmse',np.sqrt(np.mean((x[mask]@w-y[val][mask])**2)),np.sqrt(np.mean((z[mask]@w-r[val][mask])**2)))
 print('params',tau.round(5),w.round(5))
PY
```
- **Container Exit Code**: `0` (`1079 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
s.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts != 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gains, biases = _calibrate(dataset['calibration_load_cells'])[labs].T
    past = gains[:, None] * both('past_stress_cmd') + biases[:, None]
    hold = gains * both('stress_hold_cmd') + biases
    return past, hold, x


def _design(past, hold, x, dt, times, taus):
    # Unroll the relaxation recurrence: each increment decays for the
    # duration of all the steps that follow it.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus)
    gap = hold[:, None] - past @ kernel
    relaxation = -np.expm1(-times[:, None]/taus)
{"schema_valid": true, "errors": []}
calibration [[0.7899748, -0.1399601], [1.3100408, 0.1599103], [0.8499869, 0.1000162], [1.2399722, -0.1300353]] damage active 120 of 180
lab None complete True n 160 curve/rate rmse 0.918959946744072 0.13705539221031024
params [0.59928 3.4861 ] [ 0.71737  0.45187  0.07382  0.08358 -0.0745   0.07647 -0.04072  0.29752
  0.19795]
lab 0 complete True n 41 curve/rate rmse 0.2968925903301257 0.14829415567466633
lab 0 complete False n 6 curve/rate rmse 4.432105533015313 0.20193199571937048
params [0.6028  3.08812] [ 0.72323  0.44714  0.0801   0.0762  -0.08103  0.07877 -0.03607  0.2898
  0.2157 ]
lab 1 complete True n 34 curve/rate rmse 0.5329510690302487 0.1501045576334473
lab 1 complete False n 3 curve/rate rmse 0.15067040662778333 0.1765639875833833
params [0.5993  4.26467] [ 0.72359  0.49201  0.07316  0.09501 -0.07148  0.08086 -0.03753  0.29868
  0.18766]
lab 2 complete True n 37 curve/rate rmse 0.8792519833949906 0.1293256256228635
lab 2 complete False n 3 curve/rate rmse 0.23359820687888197 0.1382120615859923
params [0.62295 4.80905] [ 0.76021  0.46959  0.07622  0.08059 -0.07576  0.06932 -0.03955  0.31525
  0.21167]
lab 3 complete True n 48 curve/rate rmse 1.4036145495180092 0.1316661103948016
lab 3 complete False n 8 curve/rate rmse 0.6463485240903093 0.17362288538062648
params [0.57024 2.95219] [ 0.6645   0.48044  0.06604  0.08191 -0.07083  0.07668 -0.04939  0.27686
  0.17423]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f146534dbd0f4c4d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f146534d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`28194 ms` LLM)
- **Rationale**: Out-of-fold residuals at each time distinguish isolated spikes, which warrant robust fitting, from coherent regime-dependent errors, which would warrant revisiting the design. Reference-panel counts check whether low support could explain the lab differences.
- **Alternatives Weighed**: `Change the terminal-flow law immediately` (rejected: The aggregate errors do not show whether errors follow a coherent time law or come from isolated corrupted measurements.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c02_batch_04_solver as m
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
t=d['train']; n=len(t['lab_id']); a=tuple(x[:n] for x in m._prepare(d)); dt,ts=d['past_dt'],d['eval_times']; y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ok=np.isfinite(t['coupon_features_raw']).all(1)
pred=np.empty_like(y); rate=np.empty_like(r)
for c in range(4):
 fit=ok & (t['lab_id']!=c); val=t['lab_id']==c
 tau,w=m._fit(*(x[fit] for x in a),dt,ts,y[fit],r[fit]); x,z=m._design(*(x[val] for x in a),dt,ts,tau); pred[val]=x@w; rate[val]=z@w
for active in (True,False):
 mask=((a[-1]>0)==active)&ok
 e=y[mask]-pred[mask]; er=r[mask]-rate[mask]; print('active',active,'n',len(e),'abs curve quantiles by time',np.quantile(abs(e),[.5,.9,.98,1],axis=0).round(4),'signed median',np.median(e,axis=0).round(4),'rate quant',np.quantile(abs(er),[.5,.9,.98,1]).round(4))
print('largest residual rows: row lab act complete rate_error curve_errors')
for i in np.argsort(np.max(np.abs(y-pred),axis=1))[-22:][::-1]: print(i,t['lab_id'][i],round(a[-1][i],4),ok[i],round(r[i]-rate[i],4),(y[i]-pred[i]).round(3))
print('panel finite counts by lab',[(int(c),int(np.isfinite(np.r_[t['ref_coupons'],d['test']['ref_coupons']][np.r_[t['lab_id'],d['test']['lab_id']]==c]).sum(axis=0).min())) for c in range(4)])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection outliers' | head -14
```
- **Container Exit Code**: `0` (`731 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.2   -0.063 -0.063 -0.061 -0.051 -0.051 -0.054 -0.022 -0.038 -0.042]
167 0 0.0 True -0.0559 [ 0.006  0.017  0.06  -2.191  0.136  0.19   0.29   0.341  0.369  0.433]
166 1 0.0 True 0.0973 [0.023 0.032 0.071 0.147 0.145 0.191 1.93  0.536 0.284 0.296]
111 2 0.9952 True -0.1396 [-0.031 -0.079 -0.096 -0.106 -0.145 -0.157 -0.202 -1.427 -0.212 -0.249]
38 0 0.9918 True 0.3435 [-0.159  0.262  0.414  0.395  0.426  0.457  0.494  0.537  0.521 -1.304]
28 2 0.9758 True -0.0108 [-0.03  -0.094 -0.115 -0.125 -0.113 -1.265 -0.154 -0.131 -0.145 -0.143]
159 0 0.0 True -0.1126 [-0.029 -0.053 -0.089 -0.098 -0.588 -0.121 -0.12  -1.148 -0.358 -0.112]
154 0 0.0 True 0.3243 [ 1.139 -0.245  0.149  0.065  0.007 -0.044 -0.056 -0.232 -0.465 -0.453]
74 0 0.9623 True 0.0046 [ 0.015  0.05  -0.053  0.083  0.093  0.082  0.092  0.09  -1.127  0.056]
175 1 0.0 True 0.3393 [ 0.06   0.044  0.008 -0.104 -0.234 -0.364 -0.576 -0.727 -0.878 -1.102]
112 0 0.9714 True 0.0082 [-0.006  0.004  0.003  1.066 -0.043 -0.061 -0.094 -0.09  -0.094 -0.149]
83 2 0.989 True -0.0529 [-0.011 -0.061 -0.054 -0.091 -0.059 -0.045 -0.023 -1.052 -0.014  0.012]
panel finite counts by lab [(0, 62), (1, 46), (2, 51), (3, 69)]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f146534dbd0f4c4d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f146534d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`45107 ms` LLM)
- **Rationale**: Test whether isolated spikes, including a 34.25-unit excursion, explain the failure of ordinary least squares. Compare the robust fit on the same held-out labs and check whether damage-inactive rows retain a systematic time trend after spike downweighting; that would change the terminal basis, not the calibration.
- **Alternatives Weighed**: `Drop entire curves with large residuals` (rejected: Several of the largest excursions affect one measurement while adjacent times remain coherent; pointwise weighting preserves those usable observations.)

```bash
set -e
cd /holdout_task
cat >/tmp/rheo_robust_probe.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('m','/holdout_task/alpha_m97_c02_batch_04_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def design(data,dt,times,tau,mode):
    x,z=m._design(*data,dt,times,tau)
    if mode=='log':
        inactive=data[-1]<=0
        x[inactive,:,2:7] *= (np.log1p(times)/times)[None,:,None]
    return x,z

def fit(data,dt,times,y,r,mode,curve_scale=.055,rate_scale=.13):
    response=np.r_[y.ravel(),r]; n=y.size
    scales=np.r_[np.full(n,curve_scale),np.full(len(r),rate_scale)]
    def matrix(p):
        c,z=design(data,dt,times,np.exp(p),mode)
        return np.vstack((c.reshape(-1,9),z))
    p=np.log([.6,3.5]); sw=np.ones(len(response)); bounds=(np.log(.035),np.log(60.))
    for step in range(25):
        b=sw*response
        def profile(p,ret=False):
            a=matrix(p)*sw[:,None]
            w=np.linalg.lstsq(a,b,rcond=None)[0]
            return w if ret else np.einsum('ij,j->i',a,w)-b
        p=least_squares(profile,p,bounds=bounds,max_nfev=45,ftol=1.e-8,xtol=1.e-8).x
        w=profile(p,True)
        e=matrix(p)@w-response
        s=scales*(3 if step<3 else 1)
        new=1/np.sqrt(s*s+e*e)
        if step>4 and np.max(abs(new/sw-1)) < 1.e-4: break
        sw=new
    return np.exp(p),w

if __name__=='__main__':
    with open('/holdout_task/public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
    t=d['train']; n=len(t['lab_id']); a=tuple(x[:n] for x in m._prepare(d)); dt,ts=d['past_dt'],d['eval_times']; y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ok=np.isfinite(t['coupon_features_raw']).all(1)
    errs={}
    for mode in ('baseline','linear','log'):
        e=np.empty_like(y); er=np.empty_like(r)
        for c in (0,1,2,3,None):
            tr=ok if c is None else ok&(t['lab_id']!=c); va=np.ones(n,dtype=bool) if c is None else (t['lab_id']==c)
            data=tuple(x[tr] for x in a)
            if mode=='baseline': tau,w=m._fit(*data,dt,ts,y[tr],r[tr])
            else: tau,w=fit(data,dt,ts,y[tr],r[tr],mode)
            if c is None:
                print(mode,'full parameters',tau.round(5),w.round(5)); continue
            x,z=design(tuple(x[va] for x in a),dt,ts,tau,mode); e[va]=y[va]-x@w; er[va]=r[va]-z@w
        errs[mode]=(e,er)
        print(mode,'complete damage/time residual signed medians',[np.median(e[ok&((a[-1]>0)==p)],axis=0).round(3).tolist() for p in (True,False)])
    # Compare all models on identical observations, retaining an uncensored rate metric.
    common=abs(errs['linear'][0])<.6
    for mode,(e,er) in errs.items():
        scores=[]
        for c in range(4):
            for complete in (True,False):
                for active in (True,False):
                    row=(t['lab_id']==c)&(ok==complete)&((a[-1]>0)==active)
                    if not row.any(): continue
                    ec=e[row]; kept=common[row]
                    scores.append((round(np.sqrt(np.mean(ec[kept]**2)),4),round(np.sqrt(np.mean(er[row]**2)),4),c,complete,active,int(row.sum())))
        print(mode,'worst clean curve',max(scores),'worst rate',max(scores,key=lambda x:x[1]),'pooled median abs curve/rate',np.median(abs(e)),np.median(abs(er)))
        for active in (True,False):
            row=ok&((a[-1]>0)==active)
            print(mode,'complete active',active,'curve/rate clean rms',np.sqrt(np.mean(e[row][common[row]]**2)),np.sqrt(np.mean(er[row]**2)))
    print('common mask omitted points',int((~common).sum()),'of',common.size)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_robust_probe.py
```
- **Container Exit Code**: `0` (`1436 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline full parameters [0.59928 3.4861 ] [ 0.71737  0.45187  0.07382  0.08358 -0.0745   0.07647 -0.04072  0.29752
  0.19795]
baseline complete damage/time residual signed medians [[0.015, 0.037, 0.045, 0.065, 0.066, 0.084, 0.092, 0.091, 0.107, 0.125], [0.018, 0.026, 0.014, 0.003, -0.022, -0.057, -0.064, -0.127, -0.164, -0.195]]
linear full parameters [0.56791 4.38705] [ 0.68879  0.56037  0.13787  0.1541  -0.12209  0.11428 -0.08449  0.25295
  0.2212 ]
linear complete damage/time residual signed medians [[-0.003, 0.001, 0.004, 0.004, 0.002, 0.005, 0.003, -0.001, 0.002, 0.006], [-0.005, -0.035, -0.087, -0.173, -0.239, -0.291, -0.367, -0.476, -0.533, -0.619]]
log full parameters [0.56521 4.16719] [ 0.67788  0.54363  0.13991  0.15613 -0.12683  0.11709 -0.08774  0.26141
  0.21937]
log complete damage/time residual signed medians [[-0.003, -0.0, 0.002, 0.001, -0.001, 0.0, -0.001, -0.002, 0.0, 0.002], [0.002, -0.004, -0.002, 0.003, -0.001, 0.001, 0.002, -0.0, 0.004, -0.004]]
baseline worst clean curve (0.2696, 0.2495, 0, False, True, 2) worst rate (0.2696, 0.2495, 0, False, True, 2) pooled median abs curve/rate 0.09388607268204674 0.09268980025612356
baseline complete active True curve/rate clean rms 0.18376306724574074 0.132852308583809
baseline complete active False curve/rate clean rms 0.11098534695630043 0.15269933070840325
linear worst clean curve (0.3772, 0.1916, 3, False, False, 1) worst rate (0.2987, 0.2485, 0, False, True, 2) pooled median abs curve/rate 0.02288689961052659 0.015063681228138559
linear complete active True curve/rate clean rms 0.06818849401336889 0.02035413583202776
linear complete active False curve/rate clean rms 0.2415266554786535 0.020933542633280566
log worst clean curve (0.2995, 0.2471, 0, False, True, 2) worst rate (0.2995, 0.2471, 0, False, True, 2) pooled median abs curve/rate 0.01384122064688683 0.014723850499109709
log complete active True curve/rate clean rms 0.06781473689946542 0.01910555123813419
log complete active False curve/rate clean rms 0.06902623788949269 0.015560080446529267
common mask omitted points 178 of 1800

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f146534dbd0f4c4d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f146534d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38824 ms` LLM)
- **Rationale**: Imputed rows now dominate the worst rate slice. Compare cell and cell-balanced fills on those rows, and check whether tighter robust scales destabilize the relaxation times. Report tail strain errors without a model-dependent exclusion mask.
- **Alternatives Weighed**: `Keep linear drift to minimize the change` (rejected: Even after spike downweighting, it leaves a coherent late-time bias on complete rows across held-out labs.)
- **ATMS NOGOOD DECLARED**: Refuted `A zero damage activation switches the terminal drift basis from logarithmic to linear in this low-shear campaign.` | Witness: `After pointwise robust fitting, complete zero-damage rows still had a signed median final-time residual of −0.619 with linear drift, versus −0.004 with shared logarithmic drift in leave-one-lab-out predictions.` | Blanket Diagnosis: `The terminal basis and damage activation were coupled unnecessarily: activation should gate the two crystallization corrections, not disable logarithmic drift in this campaign.` | Sublated Pivot: `Use the logarithmic terminal basis in the low-shear campaign while preserving exact Prony memory and both activation-weighted terms.`

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys
sys.path.insert(0, '/tmp')
from rheo_robust_probe import m, fit, design
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
test,train=d['test'],d['train']; n=len(train['lab_id'])
lab=np.r_[train['lab_id'],test['lab_id']]; stratum=np.r_[train['stratum'],test['stratum']]
raw=np.r_[train['coupon_features_raw'],test['coupon_features_raw']]; observed=np.isfinite(raw)
x,*mech=m._prepare(d); empty=np.where(observed,x,np.nan)
cells=np.unique(np.column_stack((lab,stratum)),axis=0)
means=np.array([np.nanmean(empty[(lab==c)&(stratum==s)],axis=0) for c,s in cells]); balanced=np.nanmean(means,axis=0)
xalt={'cell':x,'balanced':np.where(observed,x,balanced),'lab_balanced':x.copy()}
for c in range(4):
    xalt['lab_balanced'][lab==c]=np.where(observed[lab==c],x[lab==c],np.nanmean(means[cells[:,0]==c],axis=0))
y=train['strain_curves_obs']-train['strain_0'][:,None]; r=train['instant_rate_obs']; ok=observed[:n].all(1)
dt,ts=d['past_dt'],d['eval_times']
act=mech[-1][:n]>0
for cs,rs in ((.055,.13),(.035,.04),(.025,.025)):
    residuals={k:(np.empty_like(y),np.empty_like(r)) for k in xalt}
    params=[]
    for c in range(4):
        tr=ok&(lab[:n]!=c); va=lab[:n]==c
        tau,w=fit(tuple(a[:n][tr] for a in (x,*mech)),dt,ts,y[tr],r[tr],'log',cs,rs)
        params.append(np.r_[tau,w]);
        for k,X in xalt.items():
            b,z=design(tuple(a[:n][va] for a in (X,*mech)),dt,ts,tau,'log')
            e,er=residuals[k]; e[va]=b@w-y[va]; er[va]=z@w-r[va]
    print('scales',cs,rs,'held-lab tau range',np.min(params,axis=0)[:2].round(4),np.max(params,axis=0)[:2].round(4))
    for k,(e,er) in residuals.items():
        for complete in ([True,False] if k=='cell' else [False]):
            groups=[(lab[:n]==c)&(ok==complete)&(act==active) for c in range(4) for active in (True,False)]
            groups=[g for g in groups if g.any()]
            q=lambda arr: round(float(np.quantile(abs(arr),.9)),4)
            worst_curve=max(q(e[g]) for g in groups)
            worst_rate=max(float(np.sqrt(np.mean(er[g]**2))) for g in groups)
            g=ok==complete
            print(k,'complete',complete,'worst lab x damage q90 strain/rate RMSE',worst_curve,round(worst_rate,4),'pooled med/q90 strain, rate RMSE',round(float(np.median(abs(e[g]))),4),q(e[g]),round(float(np.sqrt(np.mean(er[g]**2))),4))
PY
```
- **Container Exit Code**: `0` (`930 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scales 0.055 0.13 held-lab tau range [0.5648 4.1376] [0.566  4.2155]
cell complete True worst lab x damage q90 strain/rate RMSE 0.0535 0.0201 pooled med/q90 strain, rate RMSE 0.0122 0.0354 0.018
cell complete False worst lab x damage q90 strain/rate RMSE 0.6523 0.2471 pooled med/q90 strain, rate RMSE 0.116 0.4533 0.1563
balanced complete False worst lab x damage q90 strain/rate RMSE 0.6506 0.264 pooled med/q90 strain, rate RMSE 0.135 0.4252 0.1567
lab_balanced complete False worst lab x damage q90 strain/rate RMSE 0.6118 0.2429 pooled med/q90 strain, rate RMSE 0.1409 0.4054 0.1538
scales 0.035 0.04 held-lab tau range [0.5651 4.1369] [0.5657 4.2218]
cell complete True worst lab x damage q90 strain/rate RMSE 0.0533 0.0203 pooled med/q90 strain, rate RMSE 0.0123 0.0355 0.0179
cell complete False worst lab x damage q90 strain/rate RMSE 0.6525 0.2469 pooled med/q90 strain, rate RMSE 0.1164 0.4537 0.157
balanced complete False worst lab x damage q90 strain/rate RMSE 0.6508 0.2639 pooled med/q90 strain, rate RMSE 0.1348 0.4258 0.1576
lab_balanced complete False worst lab x damage q90 strain/rate RMSE 0.612 0.2428 pooled med/q90 strain, rate RMSE 0.141 0.4054 0.1546
scales 0.025 0.025 held-lab tau range [0.565  4.1252] [0.5657 4.2186]
cell complete True worst lab x damage q90 strain/rate RMSE 0.0533 0.0205 pooled med/q90 strain, rate RMSE 0.0124 0.0356 0.0179
cell complete False worst lab x damage q90 strain/rate RMSE 0.6526 0.2469 pooled med/q90 strain, rate RMSE 0.1164 0.4536 0.1571
balanced complete False worst lab x damage q90 strain/rate RMSE 0.6509 0.2639 pooled med/q90 strain, rate RMSE 0.1347 0.4262 0.1577
lab_balanced complete False worst lab x damage q90 strain/rate RMSE 0.6121 0.2428 pooled med/q90 strain, rate RMSE 0.1409 0.4053 0.1547

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f146534dbd0f4c4d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f146534d (directory).
```

### Turn 8 — `execute_in_remote_docker` (`70502 ms` LLM)
- **Rationale**: Promote the robust logarithmic model: the held-out zero-damage late-time bias disappeared without changing the damage-gated corrections. Verify the exact artifact against an independent memory recurrence and finite-difference rate, then test whether EMI routing or uncoupled telemetry can change predictions incorrectly.
- **Alternatives Weighed**: `Use a lab-specific fill` (rejected: Its small improvement on imputed tail errors does not justify replacing the specified equal weighting of lab–stratum cells with a lab-conditional estimator.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_04_solver.py')
s = p.read_text()
s = s.replace("        if good.sum() >= 4 or (good.all() and len(x) >= 2):\n            result[lab] = _affine(x[good], y[good])\n            continue", "        if good.sum() >= 4 or (good.all() and len(x) >= 2):\n            if np.ptp(x[good]) == 0:\n                raise ValueError(f'No identifiable load-cell calibration for lab {lab}')\n            result[lab] = _affine(x[good], y[good])\n            continue")
a = s.index('    # First compute each cell mean;')
b = s.index('    gain, bias', a)
s = s[:a] + '''    # Give each (lab, stratum) cell equal weight, regardless of its size.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = []
    for c, stratum_id in cells:
        values = aligned[(lab == c) & (stratum == stratum_id)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(raw.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))
''' + s[b:]
s = s.replace('def _design(aligned, past, hold, act, dt, times, taus):', 'def _design(aligned, past, hold, act, dt, times, taus, low_shear):')
s = s.replace('terminal = np.where(act[:, None] > 0, np.log1p(times), times)', 'terminal = np.log1p(times) if low_shear else times').replace('terminal[:, :, None]', 'terminal[None, :, None]')
a = s.index('def _fit(')
b = s.index('\ndef solve_', a)
s = s[:a] + '''def _fit(aligned, past, hold, act, dt, times, y, r, low_shear):
    # Profile the nine linear weights at each pair of relaxation times.
    # Cauchy weights suppress individual spikes, not the rest of their curve.
    response = np.r_[y.ravel(), r]
    scales = np.r_[np.full(y.size, .055), np.full(len(r), .13)]
    logtau = np.log([.6, 3.5])
    sw = np.ones_like(response)

    def matrix(p):
        curve, rate = _design(aligned, past, hold, act, dt, times, np.exp(p), low_shear)
        return np.vstack((curve.reshape(-1, 9), rate))

    for step in range(25):
        target = sw*response

        def profile(p, coefficients=False):
            design = sw[:, None]*matrix(p)
            weights = np.linalg.lstsq(design, target, rcond=None)[0]
            if coefficients:
                return weights
            return np.einsum('ij,j->i', design, weights) - target

        logtau = least_squares(profile, logtau, bounds=(np.log(.035), np.log(60.)),
                               max_nfev=45, ftol=1e-8, xtol=1e-8).x
        weights = profile(logtau, coefficients=True)
        residual = np.einsum('ij,j->i', matrix(logtau), weights) - response
        updated = 1/np.hypot(scales*(3 if step < 3 else 1), residual)
        if step > 4 and np.max(np.abs(updated/sw - 1)) < 1e-4:
            break
        sw = updated
    return np.exp(logtau), weights

''' + s[b:]
s = s.replace("    y = train['strain_curves_obs']", "    low_shear = dataset['shear_banding_index'] <= .50\n    y = train['strain_curves_obs']").replace("y[complete], train['instant_rate_obs'][complete])", "y[complete], train['instant_rate_obs'][complete], low_shear)").replace('dt, times, taus)\n    return', 'dt, times, taus, low_shear)\n    return')
p.write_text(s)
PY
python3 -I validate_public_schema.py
cat >/tmp/rheo_batch04_verify.py <<'PY'
import copy
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c02_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
solve = m.solve_alpha_m97_c02_batch_04
out = solve(d)
for v in out.values():
    assert np.isfinite(v).all()
assert np.all(np.diff(out['estimated_taus']) > 0)
print('taus, hereditary and regime weights:', *(out[k].round(6).tolist() for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights')))
training, test = d['train'], d['test']
n = len(training['lab_id'])
dt, ts = d['past_dt'], d['eval_times']
prepared = m._prepare(d)
aligned, past, hold, act = (v[n:] for v in prepared)
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
z = np.zeros((len(hold),2))
for step, duration in enumerate(dt):
    decay = np.exp(-duration/tau)
    z = decay*z + (1-decay)*past[:, step, None]
act_independent = np.tanh(np.sum(dt*np.maximum(np.abs(past)-d['crit_stress'], 0)**2, axis=1))
np.testing.assert_allclose(act, act_independent, rtol=1e-13, atol=1e-13)
assert (act == 0).any() and (act > 0).any()
gap = hold[:, None]-z
drift = hw[2]+aligned@hw[3:]+rw[1]*act*z[:, 0]
def strain(t):
    e = -np.expm1(-t[:, None]/tau)
    return test['strain_0'][:,None] + (gap*hw[:2])@e.T + (rw[0]*act*hold)[:, None]*e[:, 0] + drift[:, None]*np.log1p(t)
np.testing.assert_allclose(out['predicted_strain'], strain(ts), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], (strain(np.array([1e-6]))[:,0]-test['strain_0'])/1e-6, rtol=1e-5, atol=1e-5)
changed = copy.deepcopy(d)
for split in ('train','test'):
    changed[split]['aux_chamber_telemetry'] = np.full_like(changed[split]['aux_chamber_telemetry'], 1000.)
for k,v in solve(changed).items():
    np.testing.assert_array_equal(out[k],v)
base_cal = m._calibrate(d['calibration_load_cells'])
x = np.tile([-2., -1., 0., 1., 2.], 4); lab = np.repeat(np.arange(4), 5)
g = np.array([.9, 1.1, .8, 1.3]); b = np.array([.2, -.1, .1, -.2])
ycal = g[lab]*x+b[lab]
def calibrate(y):
    return m._calibrate(dict(lab_id=lab, cmd_stress=x, true_stress=y))
np.testing.assert_allclose(calibrate(ycal), np.column_stack((g,b)), atol=1e-12)
ycal[[0,5,6]] += [7,8,-9]
np.testing.assert_allclose(calibrate(ycal)[0], [g[0],b[0]], atol=1e-12)
np.testing.assert_allclose(calibrate(ycal)[1], m._calibrate(dict(lab_id=np.array([],dtype=int),cmd_stress=[],true_stress=[]))[1], atol=1e-12)
print('Passed recurrence, initial derivative, telemetry independence, and clean/EMI calibration checks.')
lab = training['lab_id']; complete = np.isfinite(training['coupon_features_raw']).all(1)
a = tuple(v[:n] for v in prepared)
y = training['strain_curves_obs']-training['strain_0'][:,None]; r = training['instant_rate_obs']
e, er = np.empty_like(y), np.empty_like(r)
for c in range(4):
    tr, va = complete & (lab != c), lab == c
    t, w = m._fit(*(v[tr] for v in a), dt, ts, y[tr], r[tr], True)
    curve, rate = m._design(*(v[va] for v in a), dt, ts, t, True)
    e[va], er[va] = y[va]-curve@w, r[va]-rate@w
for flag in (True,False):
    groups = [(lab==c)&(complete==flag)&((a[-1]>0)==active) for c in range(4) for active in (True,False)]
    scores = [(np.quantile(abs(e[q]),.9), np.sqrt(np.mean(er[q]**2))) for q in groups if q.any()]
    print('complete',flag,'worst lab x damage: strain absolute error q90 and rate RMSE',np.max(scores,axis=0).round(5).tolist())
print('Pooled median absolute strain and rate errors:', np.median(abs(e)), np.median(abs(er)))
print('Complete zero-damage: median final-time signed residual', np.median(e[complete & (a[-1]==0),-1]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_batch04_verify.py
```
- **Container Exit Code**: `0` (`1013 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
taus, hereditary and regime weights: [0.565209, 4.16719] [0.677879, 0.543631, 0.139908, 0.156135, -0.126825, 0.117089, -0.087744] [0.261409, 0.219366]
Passed recurrence, initial derivative, telemetry independence, and clean/EMI calibration checks.
complete True worst lab x damage: strain absolute error q90 and rate RMSE [0.05345, 0.02013]
complete False worst lab x damage: strain absolute error q90 and rate RMSE [0.65061, 0.26396]
Pooled median absolute strain and rate errors: 0.013768823832004673 0.01472385032701104
Complete zero-damage: median final-time signed residual -0.004091095359145452

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e49d047886fe97d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_e49d0478 (directory).
```

### Turn 9 — `submit_completed_task` (`17391 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification and the host scientific probe. Shared logarithmic drift removed the systematic late-time bias on zero-damage rows, and the remaining worst slice is tied to missing features under the prescribed imputation.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:6786'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema, memory recurrence, initial-rate derivative, EMI calibration, and telemetry independence checks passed. Missing coupon features remain the main source of held-out error.`

### Turn 10 — `execute_in_remote_docker` (`42236 ms` LLM)
- **Rationale**: Test whether the sparse channels inherit a noisy or stale calibration by comparing chronological sweeps and held-out signed errors within each lab, keeping the directly identified in-batch channel fixed. Damage-stratified horizon residuals test for a missed term without letting pooled scores conceal a local bias.
- **Alternatives Weighed**: `Add rational, relay, or stochastic drift terms` (rejected: The specified model has no relay state or stochastic drift operator. A repeatable signed error by damage and horizon would justify revisiting the physical basis before adding terms.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
history = json.loads(m._ARCHIVE.read_text())
original = m._calibrate
base = original(d['calibration_load_cells'])
pooled = base.copy()
for c in range(4):
    x, y = m._channel(d['calibration_load_cells'], c)
    print('lab', c, 'in-batch support', len(x), 'used map', base[c].round(7).tolist())
    if len(x) >= 2:
        continue
    sweeps = []
    for h in history:
        x, y = m._channel(h, c)
        ok = m._clean(x, y)
        if ok.sum() >= 2 and np.ptp(x[ok]) > 0:
            sweeps.append((x[ok], y[ok]))
    maps = np.array([m._affine(x, y) for x, y in sweeps])
    print('  archival maps, chronological', maps.round(7).tolist())
    pooled[c] = m._affine(*(np.concatenate(v) for v in zip(*sweeps)))
    if len(sweeps) > 1:
        prev = m._affine(*(np.concatenate(v) for v in zip(*sweeps[:-1])))
        x, y = sweeps[-1]
        print('  preceding-sweeps prediction RMSE on latest sweep', round(float(np.sqrt(np.mean((x*prev[0]+prev[1]-y)**2))), 7))
t = d['train']; n = len(t['lab_id']); lab = t['lab_id']; dt = d['past_dt']; ts = d['eval_times']
complete = np.isfinite(t['coupon_features_raw']).all(1)
y = t['strain_curves_obs'] - t['strain_0'][:,None]; r = t['instant_rate_obs']
base_act = m._prepare(d)[-1][:n]
active = base_act > 0
for label, cal in [('latest', base), ('pooled', pooled)]:
    m._calibrate = lambda _, cal=cal: cal
    data = tuple(x[:n] for x in m._prepare(d))
    print(label, 'largest map change', np.max(np.abs(cal-base)), 'activation switches', np.count_nonzero((data[-1]>0)!=active))
    e = np.empty_like(y); er = np.empty_like(r)
    for c in range(4):
        tr = complete & (lab != c); va = lab == c
        tau, w = m._fit(*(x[tr] for x in data), dt, ts, y[tr], r[tr], True)
        curve, rate = m._design(*(x[va] for x in data), dt, ts, tau, True)
        e[va], er[va] = y[va]-curve@w, r[va]-rate@w
        q = va & complete
        print('  held lab', c, 'complete n', q.sum(), 'strain p90 / median final / rate RMSE / median rate', np.round([np.quantile(abs(e[q]),.9),np.median(e[q,-1]),np.sqrt(np.mean(er[q]**2)),np.median(er[q])],6).tolist())
    for flag in (True, False):
        groups = [(lab==c)&(complete==flag)&(active==a) for c in range(4) for a in (False,True)]
        print('  complete', flag, 'worst strain p90 / rate RMSE', np.max([[np.quantile(abs(e[g]),.9),np.sqrt(np.mean(er[g]**2))] for g in groups if g.any()],axis=0).round(6).tolist())
    if label == 'latest':
        for g in (complete & ~active, complete & (base_act>0) & (base_act<.1), complete & (base_act>=.1)):
            if g.any():
                print('  damage slice n', g.sum(), 'signed strain median by horizon', np.median(e[g],axis=0).round(5).tolist(), 'strain p90', round(float(np.quantile(abs(e[g]),.9)),5))
PY
```
- **Container Exit Code**: `0` (`908 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.1029718
lab 1 in-batch support 2 used map [1.3100408, 0.1599103]
lab 2 in-batch support 0 used map [0.8499869, 0.1000162]
  archival maps, chronological [[1.0000075, 2.74e-05], [0.8499614, 0.1000368], [0.8499857, 0.0999836], [0.8499869, 0.1000162]]
  preceding-sweeps prediction RMSE on latest sweep 0.0747259
lab 3 in-batch support 0 used map [1.2399722, -0.1300353]
  archival maps, chronological [[1.0000095, -3.07e-05], [1.2400016, -0.1299619], [1.2400385, -0.1299844], [1.2399722, -0.1300353]]
  preceding-sweeps prediction RMSE on latest sweep 0.1156661
latest largest map change 0.0 activation switches 0
  held lab 0 complete n 41 strain p90 / median final / rate RMSE / median rate [0.038918, -0.001655, 0.018115, -0.005923]
  held lab 1 complete n 34 strain p90 / median final / rate RMSE / median rate [0.028205, 0.003036, 0.018454, 0.008122]
  held lab 2 complete n 37 strain p90 / median final / rate RMSE / median rate [0.037714, -0.002899, 0.018891, -0.002453]
  held lab 3 complete n 48 strain p90 / median final / rate RMSE / median rate [0.034397, -0.000109, 0.016941, -0.001376]
  complete True worst strain p90 / rate RMSE [0.05345, 0.020126]
  complete False worst strain p90 / rate RMSE [0.650613, 0.263965]
  damage slice n 52 signed strain median by horizon [0.00181, -0.00357, -0.00187, 0.0032, -0.00079, 0.00133, 0.00248, -0.0004, 0.00437, -0.00409] strain p90 0.03944
  damage slice n 108 signed strain median by horizon [-0.00328, -5e-05, 0.00215, 0.00124, -0.00132, 0.00034, -0.00072, -0.00239, 0.00048, 0.00215] strain p90 0.03406
pooled largest map change 0.06759443086687678 activation switches 0
  held lab 0 complete n 41 strain p90 / median final / rate RMSE / median rate [0.19992, -0.054978, 0.151767, -0.033226]
  held lab 1 complete n 34 strain p90 / median final / rate RMSE / median rate [0.103519, 0.04513, 0.091181, 0.049298]
  held lab 2 complete n 37 strain p90 / median final / rate RMSE / median rate [0.104376, 0.009372, 0.091956, 0.016867]
  held lab 3 complete n 48 strain p90 / median final / rate RMSE / median rate [0.22919, 0.002095, 0.197878, -0.001961]
  complete True worst strain p90 / rate RMSE [0.250832, 0.226214]
  complete False worst strain p90 / rate RMSE [0.671428, 0.435076]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e49d047886fe97d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_e49d0478 (directory).
```

### Turn 11 — `submit_completed_task` (`17528 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated Python verification. The follow-up audit supported retaining the latest clean calibration and found no systematic horizon bias across the measured damage slices that would warrant another model term.
- **Alternatives Weighed**: `Pool all historical calibration sweeps` (rejected: The earliest sweep used a different calibration. Including it raised worst complete-row strain p90 from 0.05345 to 0.25083 and initial-rate RMSE from 0.02013 to 0.22621.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:6786'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema and scientific checks passed. Rows with missing coupon features remain the largest source of held-out error.`

### Turn 12 — `execute_in_remote_docker` (`37229 ms` LLM)
- **Rationale**: Use real held-out lab measurements to test two plausible missing terms without editing the verified solver. Fit each correction using only the training labs and require it to improve both strain and its initial derivative before reconsidering the specified law.
- **Alternatives Weighed**: `Retest archive pooling, panel gauge invariance, or relay modes` (rejected: Calibration and gauge transfer were already tested. The campaign has no relay state, stochastic drift equation, or classification labels.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
t = d['train']; lab = t['lab_id']; n = len(lab)
a = tuple(v[:n] for v in m._prepare(d)); x, past, hold, act = a
ok = np.isfinite(t['coupon_features_raw']).all(1)
dt, ts = d['past_dt'], d['eval_times']
y = t['strain_curves_obs'] - t['strain_0'][:,None]; r = t['instant_rate_obs']
# Test one structured departure at a time, with the same added term in
# the curve and its analytic initial derivative.
errors = {key:(np.empty_like(y), np.empty_like(r)) for key in ('exact','rational_stress','pairwise_drift')}
for c in np.unique(lab):
    tr, va = ok & (lab != c), ok & (lab == c)
    tau, w = m._fit(*(v[tr] for v in a), dt, ts, y[tr], r[tr], True)
    for key, shape, impulse in (
        ('exact', None, None),
        ('rational_stress', -np.expm1(-ts/tau[0]), (hold/(1+(hold/d['crit_stress'])**2)-hold)*act),
        ('pairwise_drift', np.log1p(ts), (x[:,0]-x[:,1])*(x[:,2]-x[:,3])),
    ):
        def predict(mask):
            cx, rx = m._design(*(v[mask] for v in a), dt, ts, tau, True)
            return cx@w, rx@w
        py, pr = predict(va)
        if shape is not None:
            fy, fr = predict(tr)
            derivative = 1/tau[0] if key == 'rational_stress' else 1.
            design = np.r_[(impulse[tr,None]*shape).ravel(), impulse[tr]*derivative]
            residual = np.r_[(y[tr]-fy).ravel(), r[tr]-fr]
            scales = np.r_[np.full(fy.size,.055), np.full(fr.size,.13)]
            beta = 0.
            for _ in range(20):
                weights = 1/(scales**2+(residual-beta*design)**2)
                beta = np.dot(weights*design,residual)/np.dot(weights*design,design)
            py += beta*impulse[va,None]*shape
            pr += beta*impulse[va]*derivative
        ey, er = errors[key]; ey[va] = y[va]-py; er[va] = r[va]-pr
for key,(ey,er) in errors.items():
    groups = [ok & (lab==c) & ((act>0)==g) for c in np.unique(lab) for g in (False,True)]
    scores = np.array([(np.quantile(np.abs(ey[q]),.9), np.sqrt(np.mean(er[q]**2))) for q in groups if q.any()])
    print(key, 'worst complete lab x damage strain p90 / rate RMSE', scores.max(0).round(6).tolist(), 'pooled medians', np.round([np.median(abs(ey[ok])),np.median(abs(er[ok]))],6).tolist())
PY
```
- **Container Exit Code**: `0` (`614 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
exact worst complete lab x damage strain p90 / rate RMSE [0.05345, 0.020126] pooled medians [0.012208, 0.012688]
rational_stress worst complete lab x damage strain p90 / rate RMSE [0.05345, 0.020126] pooled medians [0.012324, 0.012369]
pairwise_drift worst complete lab x damage strain p90 / rate RMSE [0.053369, 0.019977] pooled medians [0.012221, 0.012773]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e49d047886fe97d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_e49d0478 (directory).
```

### Turn 13 — `submit_completed_task` (`18424 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification and the host scientific probe. The final leave-one-lab-out audit found no material gain from extra terms; retaining the latest clean calibration for sparse channels avoids the measured degradation from pooling.
- **Alternatives Weighed**: `Add rational stress saturation or a pairwise feature interaction` (rejected: Neither materially improved held-out strain and rate errors, and neither is part of the specified law.); `Pool all historical calibration sweeps` (rejected: The earliest sweep used a different calibration; pooling increased worst complete-row strain p90 to 0.25083 and rate RMSE to 0.22621.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema and independent checks of the memory recurrence, initial rate, and calibration passed. Missing coupon features remain the main source of held-out error.`

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
