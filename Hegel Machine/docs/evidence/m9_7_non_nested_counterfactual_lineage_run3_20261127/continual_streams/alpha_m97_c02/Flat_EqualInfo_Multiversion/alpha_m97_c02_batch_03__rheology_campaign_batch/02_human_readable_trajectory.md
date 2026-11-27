# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `423,590` (`410,994` / `108,032` / `12,596`) |
| **Total Episode Wall Time (s)** | `430.45s` |

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

### Turn 1 — `execute_in_remote_docker` (`19652 ms` LLM)
- **Rationale**: Inspect the schema and array shapes before fitting; discrepancies in calibration layout, reference grouping, or target channels would change the implementation and validation splits.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The current batch specifies a distinct jammed-slip basis and requires in-batch load-cell fitting where sufficient clean calibration exists.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' ; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb') as f: d=pickle.load(f)
def show(d, p='', depth=0):
 for k,v in d.items():
  key=p+str(k)
  if isinstance(v,dict) and depth<2:
   print(key, 'dict'); show(v,key+'.',depth+1)
  else:
   a=np.asarray(v); print(key, type(v).__name__, a.shape, a.dtype, repr(a.reshape(-1)[:4])[:280])
show(d)
PY
```
- **Container Exit Code**: `1` (`459 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
float64",
              "shape": [
                2
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
      "bytes": 1111
    }
  },
  "schema": "hegel_public_asset_index_v1"
}total 40
-rw-r--r-- 1 root root 7679 Nov 27 08:59 alpha_m97_c02_batch_01_solver.py
-rw-r--r-- 1 root root 6161 Nov 27 08:59 alpha_m97_c02_batch_02_solver.py
-rw-r--r-- 1 root root 7113 Nov 27 08:59 alpha_m97_c02_s00_solver.py
-r--r--r-- 1 root root 5402 Nov 27 08:59 archive_calibration_history.json
-rw-r--r-- 1 root root 1416 Nov 27 08:59 archive_manifest.json
-rw-r--r-- 1 root root  339 Nov 27 08:59 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18182 ms` LLM)
- **Rationale**: Inspect only the dataset inputs and prior solver conventions. The calibration residuals determine whether to use in-batch fits or investigate EMI, while the reference and target layouts determine how to build and validate the regression.
- **Alternatives Weighed**: `Fit directly from the asset index` (rejected: The index does not establish the observation fields, reference aggregation, or whether any calibration channel needs a fallback.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c02_batch_03.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
def show(d,p=''):
 for k,v in d.items():
  key=p+str(k)
  if isinstance(v,dict): show(v,key+'.'); continue
  a=np.asarray(v); print(key,a.shape,a.dtype,repr(a.reshape(-1)[:4])[:180])
show(d)
c=d['calibration_load_cells']
for l in range(4):
 idx=c['lab_id']==l;x=c['cmd_stress'][idx];y=c['true_stress'][idx]; b=np.polyfit(x,y,1); print('cal',l,len(x),b,'max residual',abs(y-np.polyval(b,x)).max())
PY
sed -n '1,200p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -50
```
- **Container Exit Code**: `0` (`448 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`22865 ms` LLM)
- **Rationale**: All four public channels have ten clean points, so their calibration needs no archive fallback. Check the archived parser and actual cell coverage to determine the fallback and imputation behavior, and identify which public fields can independently verify predictions without entering the fit.
- **Alternatives Weighed**: `Use the control-panel skill directly` (rejected: Its API does not encode the required stratum-balanced imputation, and the archived solver has a batch-specific reference implementation to inspect.)

```bash
sed -n '1,/^def solve_/p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p))
print('history layout',repr(h)[:1500])
a=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))[0];d=a['dataset']
print('public verification fields',[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in a.items() if k!='dataset'])
for split in ('train','test'):
 t=d[split];print(split,'missing per column',np.sum(~np.isfinite(t['coupon_features_raw']),axis=0),'complete per lab',[(l,int(np.sum((t['lab_id']==l)&np.isfinite(t['coupon_features_raw']).all(1)))) for l in range(4)])
 for l in range(4):
  print('cells',l,[(int(s),np.sum(np.isfinite(t['coupon_features_raw'][(t['lab_id']==l)&(t['stratum']==s)]),axis=0).tolist()) for s in np.unique(t['stratum'])])
PY
```
- **Container Exit Code**: `0` (`408 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
p(p[:2]), p[2:]


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
history layout [{'episode_short_name': 'alpha_m97_c02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646936, 0.95752, 2.133164, -0.418978, -0.546283, -1.402876, 1.416566, -0.474167, 0.562697, -1.431478, -1.102292, 0.1, 0.333007, 0.979026, 1.410368, 2.014512, 2.037978, -1.520941, -1.132263, -0.400526, 1.428582, 2.184857, 2.114623, 2.024408, 0.373485, 1.83932, -1.383844, 0.25249, -1.650978, 2.152261, 0.672955, 0.634083, 2.133016, 0.157033, -0.816183, -1.031889, -0.02981, 1.574906]}, {'episode_short_name': 'alpha_m97_c02_batch_01', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [-1.006249, -1.809938, -1.639722, 0.686336, 1.257617, -1.795823, -1.10985, 0.855402, 0.387867, -1.661626, 0.375391, 1.92561, 2.080645, 1.659176, 0.999466, 1.338557, -0.977277, 2.183171, -0.406287, 0.845669, 2.186524, 1.679156, 1.00863, -1.6169, 0.078971, 0.018612, 1.244379
public verification fields [('gt', ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true'])]
train missing per column [6 8 4 1] complete per lab [(0, 33), (1, 54), (2, 39), (3, 36)]
cells 0 [(0, [15, 15, 14, 15]), (1, [13, 13, 12, 13]), (2, [9, 9, 9, 9])]
cells 1 [(0, [12, 11, 12, 12]), (1, [26, 24, 26, 26]), (2, [19, 19, 19, 19])]
cells 2 [(0, [12, 12, 12, 12]), (1, [8, 7, 8, 8]), (2, [21, 22, 23, 23])]
cells 3 [(0, [8, 8, 7, 8]), (1, [13, 12, 13, 13]), (2, [18, 20, 21, 21])]
test missing per column [1 2 0 1] complete per lab [(0, 15), (1, 17), (2, 16), (3, 21)]
cells 0 [(0, [3, 3, 3, 3]), (1, [7, 7, 7, 7]), (2, [6, 5, 6, 5])]
cells 1 [(0, [4, 4, 4, 4]), (1, [10, 10, 10, 10]), (2, [3, 3, 3, 3])]
cells 2 [(0, [5, 5, 5, 5]), (1, [2, 2, 2, 2]), (2, [10, 9, 10, 10])]
cells 3 [(0, [6, 7, 7, 7]), (1, [8, 8, 8, 8]), (2, [7, 7, 7, 7])]
```

### Turn 4 — `execute_in_remote_docker` (`47662 ms` LLM)
- **Rationale**: Build the specified nine-column model using the archived alignment convention. The isolated schema check detects implementation and output-shape failures before held-out prediction and parameter validation.
- **Alternatives Weighed**: `Fit on imputed training rows as if fully observed` (rejected: 162 of 180 training rows are complete; fitting on those avoids folding imputation error into the constitutive coefficients while retaining all labs.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
old = Path('/tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py').read_text()
prep = old[old.index('def _affine'):old.index('def _design')]
print(prep[prep.index('def _prepare'):])
p = Path('alpha_m97_c02_batch_03_solver.py')
p.write_text('''"""Calibrated two-branch Prony model with jammed-slip terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


''' + prep)
PY
cat >> alpha_m97_c02_batch_03_solver.py <<'PY'
def _design(past, hold, x, dt, times, taus):
    # Unroll the relaxation recurrence: each increment decays for the
    # duration of all the steps that follow it.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus)
    gap = hold[:, None] - past @ kernel
    relaxation = -np.expm1(-times[:, None]/taus)
    phi = times / (1.0 + 0.85*times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    delta = gap[:, 0] - gap[:, 1]
    flow = np.column_stack((np.ones_like(hold), x, slip))
    curve = np.concatenate((gap[:, None, :]*relaxation[None, :, :],
                            flow[:, None, :]*phi[None, :, None],
                            delta[:, None, None]*relaxation[None, :, :1]), axis=2)
    rate = np.column_stack((gap/taus, flow, delta/taus[0]))
    return curve, rate


def _mad(x):
    return max(1e-7, 1.4826*np.median(np.abs(x-np.median(x))))


def _fit(past, hold, x, dt, times, curves, rates):
    cm, rm = np.isfinite(curves), np.isfinite(rates)
    target = np.r_[curves[cm], rates[rm]]
    nc = cm.sum()
    # Positive separation keeps branch 0 fast in both the profile and the
    # robust fit; the asymmetric slip term is not invariant to a branch swap.
    lower, upper = np.log([.015, .015]), np.log([20., 60.])

    def unpack(q):
        fast, separation = np.exp(q[:2])
        return np.array([fast, fast + separation])

    def design(q):
        c, r = _design(past, hold, x, dt, times, unpack(q))
        return np.concatenate((c[cm], r[rm]))

    scale = np.r_[np.ones(nc), np.full(rm.sum(), 2.)]

    def profile(q, ret_weights=False):
        a = design(q)/scale[:, None]
        b = target/scale
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        return w if ret_weights else np.einsum('ij,j->i', a, w) - b

    starts = ((.15, .7), (.4, 2.), (.9, 3.5), (1.8, 7.))
    best = min((least_squares(profile, np.log([f, s-f]), bounds=(lower, upper),
                              max_nfev=75) for f, s in starts), key=lambda r: r.cost)
    p = np.r_[best.x, profile(best.x, True)]

    def residual(q):
        return np.einsum('ij,j->i', design(q), q[2:]) - target

    bounds = (np.r_[lower, np.full(9, -np.inf)], np.r_[upper, np.full(9, np.inf)])
    for _ in range(2):
        e = residual(p)
        noise = np.r_[np.full(nc, _mad(e[:nc])), np.full(rm.sum(), _mad(e[nc:]))]
        p = least_squares(lambda q: residual(q)/noise, p, bounds=bounds,
                          loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100).x
    return unpack(p), p[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                         curves[complete], train['instant_rate_obs'][complete])
    curve, rate = _design(*(a[n:] for a in data), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS nonlinear least squares' | head -8
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`621 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
in[key], test[key]))

    labs, strata = both('lab_id'), both('stratum')
    raw, refs = both('coupon_features_raw'), both('ref_coupons')
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        use = labs == lab
        panel = refs[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (raw[use] - center) / scale
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for index in np.unique(cell):
        values = x[cell == index]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts != 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gains, biases = _calibrate(dataset['calibration_load_cells'])[labs].T
    past = gains[:, None] * both('past_stress_cmd') + biases[:, None]
    hold = gains * both('stress_hold_cmd') + biases
    return past, hold, np.column_stack((np.ones(len(x)), x))



autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7be3469635b339af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7be34696 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 9, in <module>
    out = mod.solve_alpha_m97_c02_batch_03(ds)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c02_batch_03_solver.py", line 155, in solve_alpha_m97_c02_batch_03
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c02_batch_03_solver.py", line 142, in _fit
    p = least_squares(lambda q: residual(q)/noise, p, bounds=bounds,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_lsq/least_squares.py", line 813, in least_squares
    raise ValueError("Inconsistent shapes between bounds and `x0`.")
ValueError: Inconsistent shapes between bounds and `x0`.
```

### Turn 5 — `execute_in_remote_docker` (`23060 ms` LLM)
- **Rationale**: The archived preparation added an intercept that the new design also adds. Return only the four aligned features, then verify the output contract in isolation and compare both endpoints against public references by lab, stratum, missingness, stress sign, and horizon. These slices distinguish imputation errors from constitutive-fit errors.
- **Alternatives Weighed**: `Expand the optimizer bounds to match the current design` (rejected: That would retain a duplicate intercept and make the seven hereditary coefficients non-identifiable.)

```bash
set -e
cd /holdout_task
sed -i 's/return past, hold, np.column_stack((np.ones(len(x)), x))/return past, hold, x/' alpha_m97_c02_batch_03_solver.py
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
a=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]
d,g=a['dataset'],a['gt']; o=s.solve_alpha_m97_c02_batch_03(d)
for k,ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(k,'estimated',o[k],'reference',g[ref],'delta',o[k]-g[ref])
c=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; t=d['test']; complete=np.isfinite(t['coupon_features_raw']).all(axis=1)
for key, groups in [('lab',t['lab_id']),('stratum',t['stratum']),('complete',complete),('stress_sign',np.sign(g['te_hold_true']))]:
 for v in np.unique(groups):
  m=groups==v
  print(key,v,m.sum(),'curve/rate RMSE',np.sqrt(np.mean(c[m]**2)),np.sqrt(np.mean(r[m]**2)))
print('curve RMSE by horizon',np.sqrt(np.mean(c**2,axis=0)))
PY
```
- **Container Exit Code**: `0` (`773 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus estimated [0.57577169 4.21677425] reference [0.576 4.235] delta [-0.00022831 -0.01822575]
hereditary_weights estimated [ 0.67906535  0.55127838  0.14041606  0.15677964 -0.12799244  0.11826274
 -0.0892787 ] reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] delta [-0.00093465  0.00127838  0.00041606 -0.00322036  0.00200756 -0.00173726
  0.0007213 ]
regime_mode_weights estimated [ 0.31021314 -0.21887076] reference [ 0.31 -0.22] delta [0.00021314 0.00112924]
lab 0 16 curve/rate RMSE 0.00909549197965546 0.011799876225567376
lab 1 17 curve/rate RMSE 0.0025067579032315265 0.0035959922006833186
lab 2 17 curve/rate RMSE 0.0176778201285979 0.022837539281763283
lab 3 22 curve/rate RMSE 0.025365727078866158 0.03262167619044711
stratum 0 19 curve/rate RMSE 0.02561877682270156 0.03281197297436434
stratum 1 27 curve/rate RMSE 0.007602030718599663 0.010092179364137909
stratum 2 26 curve/rate RMSE 0.016252620403331437 0.021049354035349438
complete False 3 curve/rate RMSE 0.07596880072629487 0.09725920731230207
complete True 69 curve/rate RMSE 0.007207614409818219 0.00959150426642948
stress_sign -1.0 39 curve/rate RMSE 0.01428323171241235 0.018613530189540123
stress_sign 1.0 33 curve/rate RMSE 0.019803495338609794 0.025354321438428803
curve RMSE by horizon [0.00524934 0.01079333 0.01398483 0.01606022 0.01751937 0.01860124
 0.01943555 0.02009807 0.02063682 0.02108315]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27887 ms` LLM)
- **Rationale**: Use public reference parameters only for diagnosis: if the residual follows the flow basis even with exact stress and coefficients, inspect feature alignment rather than retuning the Prony fit. Leave-one-lab-out fitting checks whether the constitutive coefficients transfer across labs.
- **Alternatives Weighed**: `Change the imputation based on the three public held-out coupons` (rejected: That would tune on public verification labels without establishing whether the error comes from alignment, missing features, or the fitted mechanics.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s = importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
a = pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g = a['dataset'],a['gt']
tr,te=d['train'],d['test']; n=len(tr['lab_id']); past,hold,x=m._prepare(d); dt,ts=d['past_dt'],d['eval_times']
w=np.r_[g['theta'],g['vp_theta']]; phi=ts/(1+.85*ts)
cp, rp=m._design(g['te_hist_true'],g['te_hold_true'],x[n:],dt,ts,g['taus'])
ec=te['strain_0'][:,None]+cp@w-g['te_clean']; er=rp@w-g['te_true_rate0']; complete=np.isfinite(te['coupon_features_raw']).all(axis=1)
print('oracle stress and coefficients, unchanged alignment:')
for k in (True,False):
 b=complete==k
 print('complete',k,'curve/rate RMSE',np.sqrt(np.mean(ec[b]**2)),np.sqrt(np.mean(er[b]**2)),'curve residual minus rate residual times phi, max',np.max(abs(ec[b]-er[b,None]*phi)))
for lab in range(4):
 fit=(tr['lab_id']!=lab)&np.isfinite(tr['coupon_features_raw']).all(axis=1)
 ev=te['lab_id']==lab
 t,v=m._fit(past[:n][fit],hold[:n][fit],x[:n][fit],dt,ts,(tr['strain_curves_obs']-tr['strain_0'][:,None])[fit],tr['instant_rate_obs'][fit])
 c,r=m._design(past[n:][ev],hold[n:][ev],x[n:][ev],dt,ts,t)
 print('leave-lab-out',lab,'curve/rate RMSE',np.sqrt(np.mean((te['strain_0'][ev,None]+c@v-g['te_clean'][ev])**2)),np.sqrt(np.mean((r@v-g['te_true_rate0'][ev])**2)),'taus',t)
PY
```
- **Container Exit Code**: `0` (`720 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
oracle stress and coefficients, unchanged alignment:
complete True curve/rate RMSE 0.007445699121403575 0.009600625477897798 curve residual minus rate residual times phi, max 1.502270530195915e-15
complete False curve/rate RMSE 0.07697558210630072 0.0992537735806941 curve residual minus rate residual times phi, max 7.008282842946301e-16
leave-lab-out 0 curve/rate RMSE 0.01091284932192718 0.014113142867562782 taus [0.57570017 4.22990543]
leave-lab-out 1 curve/rate RMSE 0.0048907289029753885 0.006510556994992629 taus [0.57487039 4.20788199]
leave-lab-out 2 curve/rate RMSE 0.018133798186379592 0.02343344079402495 taus [0.57553549 4.24739971]
leave-lab-out 3 curve/rate RMSE 0.025644776239860834 0.033222500868876056 taus [0.5766839  4.16374062]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`25618 ms` LLM)
- **Rationale**: Separate a systematic reference-panel offset on complete coupons from error caused by filling missing values. If the panel statistics reveal a mismatch with the specified alignment, correct it and validate on training folds; do not choose an imputer based on three public test coupons.
- **Alternatives Weighed**: `Retune the relaxation model` (rejected: With exact parameters the residual follows the terminal-flow basis to floating-point precision, so relaxation cannot explain it.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
a=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
for split in (tr,te):
 print({k:np.shape(v) for k,v in split.items()})
labs=np.r_[tr['lab_id'],te['lab_id']]; strata=np.r_[tr['stratum'],te['stratum']]; raw=np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; refs=np.r_[tr['ref_coupons'],te['ref_coupons']]; past,hold,x=m._prepare(d)
c,r=m._design(g['te_hist_true'],g['te_hold_true'],x[n:],d['past_dt'],d['eval_times'],g['taus']); w=np.r_[g['theta'],g['vp_theta']]; e=g['te_true_rate0']-r@w
for l in range(4):
 mask=labs==l; panel=refs[mask]; ii=(labs[n:]==l)&np.isfinite(raw[n:]).all(axis=1)
 print('lab',l,'ref centers',np.nanmean(panel,axis=0),'ref scales',np.nanstd(panel,axis=0),'oracle residual mean/std on complete',e[ii].mean(),e[ii].std())
for i,j in zip(*np.where(~np.isfinite(raw[n:]))):
 cell=(labs==labs[n+i])&(strata==strata[n+i]); lab=labs==labs[n+i]
 print('missing coupon',i,'column',j,'lab/stratum',labs[n+i],strata[n+i],'used',x[n+i,j],'same cell mean',np.nanmean(x[cell & np.isfinite(raw[:,j]),j]),'same lab mean',np.nanmean(x[lab & np.isfinite(raw[:,j]),j]),'rate residual',e[i])
PY
```
- **Container Exit Code**: `0` (`461 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
33 -0.09432483 -0.41438162
 -0.12394929 -0.41969517 -0.21040911 -0.23411288 -0.19701279 -0.41613474
 -0.32015715 -0.43296835 -0.25757082 -0.4109186  -0.16494532 -0.35546389
 -0.18484847 -0.34238826 -0.2540934   0.05581934 -0.28304305 -0.29146946
 -0.38943965 -0.11477489] ref scales [0.90564868 1.09161931 0.91379948 1.02761075 0.85879645 0.90087836
 1.01816697 0.85880052 0.85830178 0.81951341 0.91817687 0.80089424
 0.97682072 0.95657376 0.92743129 0.97958637 0.77738028 0.79338603
 1.06098072 0.88029477 0.9504062  1.0024251  0.93512569 0.98082288
 0.90723918 0.94852632 0.8333666  1.02681541 1.03173116 0.84654091
 0.95426498 0.93773273] oracle residual mean/std on complete -0.0022714361514607727 0.0025440383624941725
lab 3 ref centers [1.05227276 0.7677475  0.96509969 0.90345557 0.89106502 0.97950498
 0.64209199 0.38315549 0.65360456 0.95912114 0.26523406 0.77159127
 0.69149898 0.96800453 0.32712326 0.76121523 0.83695466 0.41586552
 0.69946983 0.67734875 0.918704   1.11124879 0.85666719 1.12327532
 0.67041382 0.65548576 0.74934263 0.99739107 1.14914512 0.65176055
 0.87561348 1.04994433] ref scales [1.56021861 1.59308339 1.48618028 1.31687612 1.62555842 1.52410834
 1.46010693 1.27432889 1.53236233 1.5116845  1.24839603 1.50973088
 1.6203494  1.30478989 1.32859488 1.50082235 1.46576045 1.52762919
 1.57086793 1.22395153 1.47059516 1.47787157 1.5404081  1.43272523
 1.23574463 1.4669324  1.24410312 1.37502475 1.43998149 1.13818986
 1.19324291 1.52224303] oracle residual mean/std on complete 0.0003493296119152989 0.00943618543642906
missing coupon 10 column 1 lab/stratum 2 2 used -0.05035189827540867 same cell mean 0.4431832524744951 same lab mean 0.11638225017573194 rate residual -0.09076803936461854
missing coupon 46 column 1 lab/stratum 0 2 used -0.05035189827540867 same cell mean 0.4785335480724373 same lab mean -0.10636231900683425 rate residual -0.028754080709281205
missing coupon 46 column 3 lab/stratum 0 2 used 0.026112901621787202 same cell mean 0.3762793538718691 same lab mean 0.12378054411290346 rate residual -0.028754080709281205
missing coupon 65 column 0 lab/stratum 3 0 used -0.04177296832901997 same cell mean -0.39599888217584217 same lab mean -0.06914597004388294 rate residual -0.14313734866375816

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`19665 ms` LLM)
- **Rationale**: The first public case passes, but it exercises only clean calibration. Inspect the inherited fallback before testing sparse and EMI-contaminated channels: a mismatch with the archive format or trigger would require a targeted fix. Also check whether the validator skipped any additional public cases.
- **Alternatives Weighed**: `Replace cell-balanced imputation based on the three public test coupons` (rejected: The residuals isolate the feature contribution but do not establish a better rule than the specified cell-balanced mean.)

```bash
set -e
cd /holdout_task
sed -n '1,/^def _prepare/p' alpha_m97_c02_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json,pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))
print('public cases:',len(items))
for a in items:
 d=a['dataset']; print('banding and sizes:',d['shear_banding_index'],len(d['train']['lab_id']),len(d['test']['lab_id']))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive structure:', list(h) if isinstance(h,dict) else (len(h),list(h[0])))
PY
```
- **Container Exit Code**: `0` (`397 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"""Calibrated two-branch Prony model with jammed-slip terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _historical(lab):
    if _ARCHIVE.is_file():
        for sweep in reversed(json.loads(_ARCHIVE.read_text())):
            x, y = _channel(sweep, lab)
            if len(x) >= 2 and np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(sweep):
    calibration = np.empty((4, 2))
    for lab in range(4):
        x, y = _channel(sweep, lab)
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            gain = np.median(slopes)
            offset = np.median(y - gain*x)
            clean = np.abs(y - gain*x - offset) <= 3.0
        if len(x) < 2 or (not clean.all() and clean.sum() < 4):
            archived = _historical(lab)
            if archived is not None:
                calibration[lab] = archived
                continue
        if clean.sum() < 2 or np.ptp(x[clean]) == 0:
            raise ValueError(f'No identifiable load-cell calibration for channel {lab}')
        calibration[lab] = _affine(x[clean], y[clean])
    return calibration


def _prepare(dataset):
public cases: 1
banding and sizes: 0.82 180 72
archive structure: (3, ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'])

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`26559 ms` LLM)
- **Rationale**: Thin or contaminate one real public calibration channel at a time and measure both prediction endpoints. If the archive is consulted for a clean channel, or a spike corrupts an otherwise identifiable fit, correct that calibration branch without changing the verified constitutive model.
- **Alternatives Weighed**: `Submit after clean-calibration validation` (rejected: The public sweep does not exercise the specified sparse or EMI-contaminated calibration paths.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
a=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; c=d['calibration_load_cells']; baseline=s._calibrate(c)
lookup=s._historical; calls=[]
def recorded(lab):
    calls.append(lab)
    return lookup(lab)
s._historical=recorded
for lab in range(4):
    ix=np.flatnonzero(c['lab_id']==lab)
    for mode in ('sparse','emi_full','emi_four'):
        calls.clear()
        keep=np.ones(len(c['lab_id']),dtype=bool)
        if mode=='sparse': keep[ix[1:]]=False
        if mode=='emi_four': keep[ix[4:]]=False
        sweep={k:v[keep].copy() for k,v in c.items()}
        if mode.startswith('emi'):
            ch=np.flatnonzero(sweep['lab_id']==lab)
            j=ch[np.argmax(np.abs(sweep['cmd_stress'][ch]))]
            sweep['true_stress'][j]+=8.
        cal=s._calibrate(sweep)
        np.testing.assert_allclose(cal[np.arange(4)!=lab],baseline[np.arange(4)!=lab],atol=1e-12)
        out=s.solve_alpha_m97_c02_batch_03(dict(d,calibration_load_cells=sweep))
        te=d['test']['lab_id']==lab
        curve=np.sqrt(np.mean((out['predicted_strain'][te]-g['te_clean'][te])**2)); rate=np.sqrt(np.mean((out['initial_rate_pred'][te]-g['te_true_rate0'][te])**2))
        print(lab,mode,'archive',calls,'calibration delta',np.round(cal[lab]-baseline[lab],5),'curve/rate RMSE',round(curve,5),round(rate,5))
PY
```
- **Container Exit Code**: `0` (`1267 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 sparse archive [0, 0] calibration delta [ 4.e-05 -4.e-05] curve/rate RMSE 0.0091 0.0118
0 emi_full archive [] calibration delta [-1.e-05  1.e-05] curve/rate RMSE 0.0091 0.0118
0 emi_four archive [0, 0] calibration delta [ 4.e-05 -4.e-05] curve/rate RMSE 0.0091 0.0118
1 sparse archive [1, 1] calibration delta [ 1.e-05 -0.e+00] curve/rate RMSE 0.00251 0.00359
1 emi_full archive [] calibration delta [-0.  0.] curve/rate RMSE 0.00251 0.0036
1 emi_four archive [1, 1] calibration delta [ 1.e-05 -0.e+00] curve/rate RMSE 0.00251 0.00359
2 sparse archive [2, 2] calibration delta [-0.e+00 -3.e-05] curve/rate RMSE 0.01768 0.02284
2 emi_full archive [] calibration delta [-2.e-05 -2.e-05] curve/rate RMSE 0.01768 0.02284
2 emi_four archive [2, 2] calibration delta [-0.e+00 -3.e-05] curve/rate RMSE 0.01768 0.02284
3 sparse archive [3, 3] calibration delta [7.e-05 5.e-05] curve/rate RMSE 0.02537 0.03262
3 emi_full archive [] calibration delta [1.e-05 1.e-05] curve/rate RMSE 0.02537 0.03262
3 emi_four archive [3, 3] calibration delta [7.e-05 5.e-05] curve/rate RMSE 0.02537 0.03262

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 10 — `submit_completed_task` (`19469 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, public reference comparisons, leave-one-lab-out checks, and sparse and EMI calibration stress tests. Those stress tests preserved the unaffected channels and used the archive only when the in-batch sweep was insufficient.
- **Alternatives Weighed**: `Retune the constitutive model or change the imputation rule` (rejected: Exact constitutive parameters did not reduce the missing-feature error, and tuning the prescribed imputation on three held-out coupons would not establish a general improvement.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:6530'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The public schema passed, and sparse and EMI-contaminated calibration checks passed on all four channels. The three coupons with missing features were the hardest public slice: strain RMSE 0.0760 and initial-rate RMSE 0.0973.`

### Turn 11 — `execute_in_remote_docker` (`41645 ms` LLM)
- **Rationale**: Check the rational flow, gap-difference term, and initial derivative against an independent recurrence; any disagreement would require correcting the design. Within-lab residuals can reveal a pooled fit that hides group bias. Moving one EMI spike across the actual calibration pairs tests whether the existing filter fails at a placement the earlier stress test missed.
- **Alternatives Weighed**: `Add unrelated PDE operators, relay modes, or classification metrics` (rejected: The batch specifies continuous strain and rate, a two-branch recurrence, and a rational flow basis. It has no PDE, relay state, or class labels; the relevant nuisance check is signed error within held-out lab and stratum.)

```bash
set -e
cd /holdout_task
sed -n '/^def _prepare/,/^def _mad/p' alpha_m97_c02_batch_03_solver.py
cat >/tmp/audit_rheology_b03.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c02_batch_03.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
dt, times = d['past_dt'], d['eval_times']
past, hold, x = m._prepare(d)
out = m.solve_alpha_m97_c02_batch_03(d)
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
z = np.zeros((len(hold), 2))
for step, duration in enumerate(dt):
    decay = np.exp(-duration/tau)
    z = z*decay + (1-decay)*past[:,step,None]
g0, g1 = (hold[:,None]-z).T
terminal = hw[2] + x@hw[3:] + rw[0]*np.sign(hold)*np.sqrt(abs(hold))
amp0 = hw[0]*g0 + rw[1]*(g0-g1)
amp1 = hw[1]*g1
def response(t):
    return (amp0[:,None]*(-np.expm1(-t/tau[0])) + amp1[:,None]*(-np.expm1(-t/tau[1])) + terminal[:,None]*t/(1+.85*t))
np.testing.assert_allclose(out['predicted_strain'], te['strain_0'][:,None]+response(times)[n:], rtol=0, atol=2e-12)
np.testing.assert_allclose(out['initial_rate_pred'], (amp0/tau[0]+amp1/tau[1]+terminal)[n:], rtol=0, atol=2e-12)
fd = ((4*response(np.array([5e-6]))-response(np.array([1e-5])))/1e-5)[n:,0]
print('scalar recurrence and closed-form derivative agree; max finite-difference rate error', np.max(abs(fd-out['initial_rate_pred'])))
assert np.max(abs(fd-out['initial_rate_pred'])) < 1e-8

complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    fit = (tr['lab_id']!=lab)&complete
    val = (tr['lab_id']==lab)&complete
    t, w = m._fit(past[:n][fit], hold[:n][fit], x[:n][fit], dt, times, y[fit], r[fit])
    c, a = m._design(past[:n][val], hold[:n][val], x[:n][val], dt, times, t)
    e, er = y[val]-c@w, r[val]-a@w
    for stratum in np.unique(tr['stratum'][val]):
        q = tr['stratum'][val]==stratum
        print('held lab/stratum',lab,stratum,'median signed curve/rate residual',round(np.median(e[q]),6),round(np.median(er[q]),6),'median abs',round(np.median(abs(e[q])),6),round(np.median(abs(er[q])),6))

sweep = d['calibration_load_cells']; base = m._calibrate(sweep)
issue = []
for lab in range(4):
    ii = np.flatnonzero(sweep['lab_id']==lab)
    for count in (4, 5, len(ii)):
        keep = np.ones(len(sweep['lab_id']),dtype=bool); keep[ii[count:]]=False
        for point in range(count):
            for spike in (-8.,8.):
                s = {k:v[keep].copy() for k,v in sweep.items()}
                j = np.flatnonzero(s['lab_id']==lab)[point]
                s['true_stress'][j] += spike
                error = np.max(abs(m._calibrate(s)[lab]-base[lab]))
                if error > 0.002: issue.append((lab,count,point,spike,round(error,6)))
print('single-spike placement deviations > 0.002:', len(issue), 'worst:', sorted(issue, key=lambda v:v[-1], reverse=True)[:6])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_rheology_b03.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1237 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
elaxation recurrence: each increment decays for the
    # duration of all the steps that follow it.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus)
    gap = hold[:, None] - past @ kernel
    relaxation = -np.expm1(-times[:, None]/taus)
    phi = times / (1.0 + 0.85*times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    delta = gap[:, 0] - gap[:, 1]
    flow = np.column_stack((np.ones_like(hold), x, slip))
    curve = np.concatenate((gap[:, None, :]*relaxation[None, :, :],
                            flow[:, None, :]*phi[None, :, None],
                            delta[:, None, None]*relaxation[None, :, :1]), axis=2)
    rate = np.column_stack((gap/taus, flow, delta/taus[0]))
    return curve, rate


def _mad(x):
scalar recurrence and closed-form derivative agree; max finite-difference rate error 9.43005673548214e-11
held lab/stratum 0 0 median signed curve/rate residual -0.003043 -0.005751 median abs 0.011503 0.00807
held lab/stratum 0 1 median signed curve/rate residual 0.000732 0.002653 median abs 0.01106 0.018966
held lab/stratum 0 2 median signed curve/rate residual -0.008258 -0.015744 median abs 0.011933 0.024551
held lab/stratum 1 0 median signed curve/rate residual 0.009089 0.010859 median abs 0.012989 0.015463
held lab/stratum 1 1 median signed curve/rate residual 0.005659 0.001618 median abs 0.012584 0.011799
held lab/stratum 1 2 median signed curve/rate residual 0.003068 0.001568 median abs 0.013605 0.009206
held lab/stratum 2 0 median signed curve/rate residual -0.003563 -0.013223 median abs 0.012391 0.015926
held lab/stratum 2 1 median signed curve/rate residual 0.001325 -0.020398 median abs 0.012398 0.020398
held lab/stratum 2 2 median signed curve/rate residual -0.003301 -0.009847 median abs 0.013194 0.012361
held lab/stratum 3 0 median signed curve/rate residual 0.002256 -0.002001 median abs 0.01128 0.011879
held lab/stratum 3 1 median signed curve/rate residual -0.000901 0.005985 median abs 0.016668 0.013287
held lab/stratum 3 2 median signed curve/rate residual -0.000702 0.016785 median abs 0.014883 0.020106
single-spike placement deviations > 0.002: 0 worst: []
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`28235 ms` LLM)
- **Rationale**: Use the measured calibration pairs to test the specified EMI threshold directly. If a spike measurably shifts the affine fit, correct the spike detector without changing the verified rheology model; otherwise retain the current artifact.
- **Alternatives Weighed**: `Submit after the ±8 EMI audit` (rejected: That audit does not test whether a high-leverage spike just above the specified 3.0 threshold can bias its own residual enough to escape detection.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; sweep=d['calibration_load_cells']; base=m._calibrate(sweep)
for size in (4,5,10):
 for amplitude in (3.01,3.1):
  errors=[]
  for lab in range(4):
   idx=np.flatnonzero(sweep['lab_id']==lab); chosen=idx[np.argsort(sweep['cmd_stress'][idx])]; take=chosen[np.rint(np.linspace(0,len(idx)-1,size)).astype(int)]
   keep=(sweep['lab_id']!=lab); keep[take]=True
   for point in range(size):
    for sign in (-1,1):
     c={k:v[keep].copy() for k,v in sweep.items()}; ii=np.flatnonzero(c['lab_id']==lab); j=ii[np.argsort(c['cmd_stress'][ii])[point]]; c['true_stress'][j]+=sign*amplitude
     err=np.max(abs(m._calibrate(c)[lab]-base[lab])); errors.append((err,lab,point,sign))
  print('rows',size,'spike',amplitude,'cases above 0.002',sum(e[0]>.002 for e in errors),'worst',max(errors))
PY
```
- **Container Exit Code**: `0` (`614 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rows 4 spike 3.01 cases above 0.002 16 worst (0.9661268975396716, 1, 3, -1)
rows 4 spike 3.1 cases above 0.002 16 worst (0.9950143496274716, 1, 3, -1)
rows 5 spike 3.01 cases above 0.002 0 worst (0.00010378128874724557, 0, 2, 1)
rows 5 spike 3.1 cases above 0.002 0 worst (0.00010378128874724557, 0, 2, 1)
rows 10 spike 3.01 cases above 0.002 0 worst (4.20978614275036e-05, 3, 0, 1)
rows 10 spike 3.1 cases above 0.002 0 worst (4.20978614275036e-05, 3, 0, 1)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`22212 ms` LLM)
- **Rationale**: The supported four-row slice fails in 16 placements at each tested spike amplitude, while five- and ten-row cases remain accurate. Inspect the existing detector and archive routing so the fix targets the sparse failure without disrupting clean-channel fits or working EMI handling.
- **Alternatives Weighed**: `Lower the 3.0 EMI threshold` (rejected: The threshold is specified in physical stress units. Evaluate the suspected row against a fit that excludes it rather than lowering the threshold on a residual that the row can bias.)

```bash
cd /holdout_task && sed -n '/^def _historical/,/^def _prepare/p' alpha_m97_c02_batch_03_solver.py
```
- **Container Exit Code**: `0` (`287 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
def _historical(lab):
    if _ARCHIVE.is_file():
        for sweep in reversed(json.loads(_ARCHIVE.read_text())):
            x, y = _channel(sweep, lab)
            if len(x) >= 2 and np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(sweep):
    calibration = np.empty((4, 2))
    for lab in range(4):
        x, y = _channel(sweep, lab)
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            gain = np.median(slopes)
            offset = np.median(y - gain*x)
            clean = np.abs(y - gain*x - offset) <= 3.0
        if len(x) < 2 or (not clean.all() and clean.sum() < 4):
            archived = _historical(lab)
            if archived is not None:
                calibration[lab] = archived
                continue
        if clean.sum() < 2 or np.ptp(x[clean]) == 0:
            raise ValueError(f'No identifiable load-cell calibration for channel {lab}')
        calibration[lab] = _affine(x[clean], y[clean])
    return calibration


def _prepare(dataset):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc7aeed9fdb7c718 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc7aeed9 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`38404 ms` LLM)
- **Rationale**: Test each candidate spike against the affine fit that excludes it, so it cannot suppress its own residual. Verify both sides of the 3.0 threshold, unchanged neighboring channels, and clean-process prediction. If the new detector misroutes a clean or below-threshold sweep, narrow it before submitting.
- **Alternatives Weighed**: `Always use the archive for four-row sweeps` (rejected: Four clean in-batch points must be fitted directly; the archive is appropriate only when contamination is detected and too few clean points remain.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_03_solver.py')
s = p.read_text()
old = '            clean = np.abs(y - gain*x - offset) <= 3.0\n'
assert s.count(old) == 1
s = s.replace(old, old + '''            # With one transient in a small sweep, a fit that includes
            # the spike can mask it. Select the most self-consistent
            # leave-one-out sweep before testing its omitted row.
            if len(x) >= 4:
                best_error = np.inf
                omitted = None
                for k in range(len(x)):
                    keep = np.arange(len(x)) != k
                    if np.ptp(x[keep]) == 0:
                        continue
                    g, b = _affine(x[keep], y[keep])
                    error = np.max(np.abs(y[keep] - (g*x[keep] + b)))
                    if error < best_error:
                        best_error = error
                        omitted = k
                        deleted_error = abs(y[k] - (g*x[k] + b))
                if omitted is not None and best_error <= 3.0 < deleted_error:
                    clean = np.arange(len(x)) != omitted
''')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/audit_rheology_b03.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
sp = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
a = pickle.load(open('public_alpha_m97_c02_batch_03.pkl', 'rb'))[0]
d, gt = a['dataset'], a['gt']; sweep = d['calibration_load_cells']; base = m._calibrate(sweep)
lookup = m._historical; calls = []
def record(lab):
    calls.append(lab)
    return lookup(lab)
m._historical = record
worst = (0., None)
for size in (4, 5, 10):
    for lab in range(4):
        ix = np.flatnonzero(sweep['lab_id'] == lab)
        ix = ix[np.argsort(sweep['cmd_stress'][ix])]
        chosen = ix[np.rint(np.linspace(0, len(ix)-1, size)).astype(int)]
        keep = sweep['lab_id'] != lab; keep[chosen] = True
        for amplitude in (0., 2.99, 3.01, 3.10, 8.):
            for point in range(size):
                for sign in (-1, 1):
                    calls.clear()
                    c = {k: v[keep].copy() for k, v in sweep.items()}
                    j = np.flatnonzero(np.flatnonzero(keep) == chosen[point])[0]
                    c['true_stress'][j] += sign*amplitude
                    cal = m._calibrate(c)
                    assert calls == ([lab] if amplitude > 3 and size == 4 else []), (size, lab, point, sign, amplitude, calls)
                    np.testing.assert_array_equal(cal[np.arange(4) != lab], base[np.arange(4) != lab])
                    if amplitude == 0 or amplitude > 3:
                        error = np.max(abs(cal[lab]-base[lab]))
                        if error > worst[0]:
                            worst = (error, (size, lab, point, sign, amplitude))
                        assert error < .002, worst
print('Worst clean or EMI-corrected calibration deviation:', worst)
out = m.solve_alpha_m97_c02_batch_03(d)
missing = ~np.isfinite(d['test']['coupon_features_raw']).all(axis=1)
for k, ref in [('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    print(k, 'missing-feature RMSE:', np.sqrt(np.mean((out[k][missing] - gt[ref][missing])**2)))
PY
```
- **Container Exit Code**: `0` (`2807 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
scalar recurrence and closed-form derivative agree; max finite-difference rate error 9.43005673548214e-11
held lab/stratum 0 0 median signed curve/rate residual -0.003043 -0.005751 median abs 0.011503 0.00807
held lab/stratum 0 1 median signed curve/rate residual 0.000732 0.002653 median abs 0.01106 0.018966
held lab/stratum 0 2 median signed curve/rate residual -0.008258 -0.015744 median abs 0.011933 0.024551
held lab/stratum 1 0 median signed curve/rate residual 0.009089 0.010859 median abs 0.012989 0.015463
held lab/stratum 1 1 median signed curve/rate residual 0.005659 0.001618 median abs 0.012584 0.011799
held lab/stratum 1 2 median signed curve/rate residual 0.003068 0.001568 median abs 0.013605 0.009206
held lab/stratum 2 0 median signed curve/rate residual -0.003563 -0.013223 median abs 0.012391 0.015926
held lab/stratum 2 1 median signed curve/rate residual 0.001325 -0.020398 median abs 0.012398 0.020398
held lab/stratum 2 2 median signed curve/rate residual -0.003301 -0.009847 median abs 0.013194 0.012361
held lab/stratum 3 0 median signed curve/rate residual 0.002256 -0.002001 median abs 0.01128 0.011879
held lab/stratum 3 1 median signed curve/rate residual -0.000901 0.005985 median abs 0.016668 0.013287
held lab/stratum 3 2 median signed curve/rate residual -0.000702 0.016785 median abs 0.014883 0.020106
single-spike placement deviations > 0.002: 0 worst: []
Worst clean or EMI-corrected calibration deviation: (0.00010378128874724557, (5, 0, 2, -1, 3.01))
predicted_strain missing-feature RMSE: 0.07596880072629487
initial_rate_pred missing-feature RMSE: 0.09725920731230207

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e4687781140a0e7f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_e4687781 (directory).
```

### Turn 15 — `submit_completed_task` (`13893 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, independent recurrence and derivative checks, and calibration tests on both sides of the specified EMI threshold. The calibration fix left clean public predictions unchanged.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The public schema passed. The EMI fix reduced the worst gain or bias deviation from 0.9950 to 0.000104 across the calibration stress tests. Coupons with missing features remained the hardest public slice: strain RMSE 0.0760 and initial-rate RMSE 0.0973.`

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
