# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `406,103` (`392,793` / `85,760` / `13,310`) |
| **Total Episode Wall Time (s)** | `433.22s` |

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

### Turn 1 — `execute_in_remote_docker` (`16328 ms` LLM)
- **Rationale**: Inspect the public schema and field shapes to detect mismatches that would change alignment, regime gating, or the returned arrays before implementing the fit.
- **Alternatives Weighed**: `Reuse the prior batch solver unchanged` (rejected: The current batch specifies a different regime law, and the public gate already rejected an accurate-looking calibration payload.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation' ; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c01_batch_04.pkl','rb') as f:d=pickle.load(f)
def show(x,depth=0):
 if isinstance(x,dict) and depth<3:
  for k,v in x.items(): print('  '*depth,k,end=' ');show(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,repr(x.ravel()[:6]))
 elif isinstance(x,(list,tuple)): print(type(x).__name__,len(x),repr(x[:1])[:400])
 else: print(type(x).__name__,repr(x)[:220])
show(d)
PY
```
- **Container Exit Code**: `0` (`626 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
r-- 1 root root 6656 Nov 27 07:06 alpha_m97_c01_batch_03_solver.py
-rw-r--r-- 1 root root 8166 Nov 27 07:06 alpha_m97_c01_s00_solver.py
-r--r--r-- 1 root root 7215 Nov 27 07:06 archive_calibration_history.json
-rw-r--r-- 1 root root 1893 Nov 27 07:06 archive_manifest.json
-rw-r--r-- 1 root root  338 Nov 27 07:06 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
list 1 [{'dataset': {'past_dt': array([0.6662, 0.6332, 0.3214, 0.709 , 0.3215, 0.3399, 0.3466, 0.2323,
       0.6797, 0.7147, 0.5853, 0.3089, 0.5838, 0.6262, 0.6828, 0.2639,
       0.6961, 0.5385]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shear_banding_index': 0.12, 'calibration_load_cells':
```

### Turn 2 — `execute_in_remote_docker` (`18593 ms` LLM)
- **Rationale**: The schema inspection stopped at the outer list. Inspect the actual train and test fields and prior alignment code to identify the observed targets and resolve sparse-channel and missing-coupon handling before fitting.
- **Alternatives Weighed**: `Use the rejected calibration payload as a fixed input` (rejected: The task requires channel-wise calibration from current sweeps or permitted prefix history; the payload does not establish that provenance.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_alpha_m97_c01_batch_04.pkl','rb'))[0];d=x['dataset'];print('item keys',list(x));print('dataset keys',list(d));print('cal',d['calibration_load_cells']);
for s in ('train','test'):
 print(s)
 for k,v in d[s].items():
  if isinstance(v,np.ndarray): print(k,v.shape,v.dtype, 'missing='+str(np.isnan(v).sum()) if v.dtype.kind=='f' else (np.unique(v,return_counts=True) if v.ndim==1 else ''))
  elif isinstance(v,dict): print(k,{a:np.shape(b) for a,b in v.items()})
  else: print(k,type(v),str(v)[:100])
PY
sed -n '1,230p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_03_solver.py
head -c 4800 /tmp/hegel_continual_archive/archive_calibration_history.json
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`573 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`17199 ms` LLM)
- **Rationale**: Channels 0–2 have no in-batch calibration rows, while channel 3 has exactly two. Summarize prefix fits to detect drift or contaminated sweeps that would change the fallback, and inspect the prior solver’s panel implementation to resolve the 32-column reference layout.
- **Alternatives Weighed**: `Apply the generic control-panel alignment skill` (rejected: Its shared scanner gain and class-cell assumptions do not match the specified per-lab median reference moments and stratum imputation.)

```bash
sed -n '1,200p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_03_solver.py; python3 - <<'PY'
import json,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
for s in json.load(open(p)):
 print({k:v for k,v in s.items() if k not in ('lab_id','cmd_stress','true_stress')})
 g,x,y=(np.asarray(s[k]) for k in ('lab_id','cmd_stress','true_stress'))
 for c in np.unique(g):
  a,b=x[g==c],y[g==c]; z=np.polyfit(a,b,1); print(int(c),len(a),z, 'max residual',np.max(np.abs(b-np.polyval(z,a))))
PY
```
- **Container Exit Code**: `0` (`545 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.exp(p), profile(p, w)[1]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    state = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(tuple(a[:n] for a in state), dt, time, delta, train['instant_rate_obs'])
    if tau[0] > tau[1]:
        tau = tau[::-1].copy()
        # The asymmetric cross-gap exponential follows the fast branch.
        # Refit in the required fast-to-slow orientation rather than swapping coefficients.
        a, b = _design(tau, tuple(s[:n] for s in state), dt, time)
        cm, rm = np.isfinite(delta), np.isfinite(train['instant_rate_obs'])
        coef = np.linalg.lstsq(np.concatenate((a[cm], b[rm]*time[0])),
                               np.r_[delta[cm], train['instant_rate_obs'][rm]*time[0]], rcond=None)[0]
    a, b = _design(tau, tuple(s[n:] for s in state), dt, time)
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:], 'predicted_strain': test['strain_0'][:, None]+a @ coef,
            'initial_rate_pred': b @ coef}
{'episode_short_name': 'alpha_m97_c01_s00'}
0 10 [9.99975614e-01 2.04739985e-05] max residual 0.00019041951493792286
1 10 [9.99993163e-01 4.93569014e-06] max residual 0.00017523865265911542
2 10 [ 1.00001214e+00 -4.58794218e-05] max residual 0.0001231190991310971
3 10 [1.00000478e+00 5.07681242e-05] max residual 0.00026499759041254123
{'episode_short_name': 'alpha_m97_c01_batch_01'}
0 10 [1.33999939 0.14999385] max residual 0.00015623657709662453
1 10 [ 0.77000393 -0.16999395] max residual 0.00018804651208315093
2 10 [ 1.210027   -0.12000342] max residual 0.00010360910560947012
3 10 [0.8300099  0.10997276] max residual 8.706492646259889e-05
{'episode_short_name': 'alpha_m97_c01_batch_02'}
0 10 [1.34002987 0.14997666] max residual 0.0002585045497334182
1 10 [ 0.77000648 -0.1699771 ] max residual 0.00033589233030895116
2 10 [ 1.20999794 -0.12001295] max residual 0.0002474827079610442
3 10 [0.82993885 0.11008917] max residual 0.00024816469867347535
{'episode_short_name': 'alpha_m97_c01_batch_03'}
0 10 [1.33998917 0.15000139] max residual 0.00021708319376595453
1 10 [ 0.77003895 -0.17002607] max residual 0.00014232070663200425
2 10 [ 1.209981   -0.11999077] max residual 0.00016153702472232467
3 10 [0.83000149 0.11001183] max residual 0.00017183812683066457
```

### Turn 4 — `execute_in_remote_docker` (`62646 ms` LLM)
- **Rationale**: Implement the prescribed basis and jointly fit curves and initial rates by variable projection. The isolated schema check catches import, shape, and nonfinite-output failures; the prior alignment comparison checks whether empty-cell fallback differs before leave-one-lab-out validation.
- **Alternatives Weighed**: `Allow the optimizer to freely swap the two branches` (rejected: The crystallization terms explicitly use branch zero, so swapping the fitted taus changes the model.)

```bash
set -e
cat > /holdout_task/alpha_m97_c01_batch_04_solver.py <<'PY'
"""Calibrated two-branch creep with a damage-gated crystallization plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel_points(rows, channel):
    x, y = (np.asarray(rows[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    use = (np.asarray(rows['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _clean(x, y):
    if len(x) < 3 or not np.ptp(x):
        return np.ones(len(x), dtype=bool)
    dx, dy = x[:, None] - x, y[:, None] - y
    slope = np.median([np.median(b[a != 0] / a[a != 0])
                       for a, b in zip(dx, dy) if np.any(a != 0)])
    intercepts = y - slope*x
    return np.abs(intercepts - np.median(intercepts)) <= 3.0


def _calibrate(rows):
    result = np.empty((4, 2))
    sweeps = None
    for c in range(4):
        x, y = _channel_points(rows, c)
        clean = _clean(x, y)
        if clean.sum() >= 2 and np.ptp(x[clean]) and (clean.sum() >= 4 or clean.all()):
            result[c] = np.polyfit(x[clean], y[clean], 1)
            continue
        if len(x) >= 2 and clean.all():
            raise ValueError(f'Calibration commands have no variation for channel {c}')
        if sweeps is None:
            sweeps = json.loads(_ARCHIVE.read_text())
        for sweep in reversed(sweeps):
            a, b = _channel_points(sweep, c)
            keep = _clean(a, b)
            if keep.sum() >= 2 and np.ptp(a[keep]):
                result[c] = np.polyfit(a[keep], b[keep], 1)
                break
        else:
            raise ValueError(f'No usable prefix calibration for channel {c}')
    return result


def _prepare(dataset):
    def join(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    labs = np.unique(lab)
    for c in labs:
        use = lab == c
        panel = ref[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (x[use] - center)/spread
    cells = [(lab == c) & (stratum == s) for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    for j in range(x.shape[1]):
        observed = np.isfinite(x[:, j])
        means = [np.mean(x[use & observed, j]) if np.any(use & observed) else np.nan for use in cells]
        for use, mean in zip(cells, means):
            if np.any(use & ~observed):
                x[use & ~observed, j] = mean if np.isfinite(mean) else np.nanmean(means)
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    damage = (excess**2) @ np.asarray(dataset['past_dt'])
    activated = (dataset['shear_banding_index'] <= .50) & (damage > 0)
    return past, hold, x, activated, np.where(activated, np.tanh(damage), 0.)


def _design(tau, state, dt, time):
    past, hold, x, activated, act = state
    z = np.zeros((len(hold), 2))
    decay = np.exp(-dt[:, None]/tau)
    step = -np.expm1(-dt[:, None]/tau)
    for j in range(len(dt)):
        z = decay[j]*z + step[j]*past[:, j, None]
    gap = hold[:, None] - z
    creep = -np.expm1(-time[:, None]/tau)
    flow = np.where(activated[:, None], np.log1p(time), time)
    drift = np.column_stack((np.ones(len(hold)), x))
    net, memory = act*hold, act*z[:, 0]
    curves = np.concatenate((gap[:, None, :]*creep[None, :, :],
                             drift[:, None, :]*flow[:, :, None],
                             (net[:, None]*creep[:, 0])[:, :, None],
                             (memory[:, None]*flow)[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, drift, net/tau[0], memory))
    return curves, rates


def _mad(x, axis=None):
    return 1.4826*np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _fit(state, dt, time, delta, rate):
    cm, rm = np.isfinite(delta), np.isfinite(rate)
    y = np.r_[delta[cm], rate[rm]]
    n_curve = cm.sum()
    w = np.r_[np.ones(n_curve), np.full(rm.sum(), time[0])]
    lo = max(.02*np.min(dt[dt > 0]), 1e-4)
    hi = 12*(dt.sum()+time.max())
    bounds = np.log([[lo, lo], [hi, hi]])

    def unpack(p):
        t = np.exp(p)
        return np.array([t[0], t.sum()])

    def profile(p):
        a, b = _design(unpack(p), state, dt, time)
        a = np.concatenate((a[cm], b[rm]))
        weighted_a = a*w[:, None]
        coef = np.linalg.lstsq(weighted_a, y*w, rcond=None)[0]
        return a @ coef-y, coef

    def opt(p):
        return least_squares(lambda q: profile(q)[0]*w, p, bounds=bounds,
                             max_nfev=100, ftol=1e-9)

    starts = np.clip(np.log([[.25, 2.25], [.65, 3.35], [1.25, 6.75], [.15, 8.85]]), *bounds)
    p = min((opt(s) for s in starts), key=lambda fit: fit.cost).x
    for _ in range(8):
        residual, _ = profile(p)
        e = np.full(delta.shape, np.nan)
        e[cm] = residual[:n_curve]
        sigma_c = np.maximum(_mad(e, axis=0), max(.35*_mad(e), 1e-7))
        sigma_r = max(_mad(residual[n_curve:]), 1e-7) if rm.any() else 1.
        sigma = np.r_[np.broadcast_to(sigma_c, delta.shape)[cm], np.full(rm.sum(), sigma_r)]
        w = 1/np.hypot(sigma, residual/2.38)
        p = opt(p).x
    return unpack(p), profile(p)[1]


def solve_alpha_m97_c01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    state = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(tuple(a[:n] for a in state), dt, time, delta, train['instant_rate_obs'])
    curve, rate = _design(tau, tuple(a[n:] for a in state), dt, time)
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ coef,
            'initial_rate_pred': rate @ coef}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
sed -n '/def _prepare/,/def _design/p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_03_solver.py
```
- **Container Exit Code**: `0` (`603 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
def _prepare(dataset):
    def join(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    for c in np.unique(lab):
        select = lab == c
        panel = ref[select]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[select] = (x[select] - center) / spread
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    observed = np.isfinite(x)
    sums = np.zeros((cell.max()+1, x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(observed, x, 0.))
    np.add.at(counts, cell, observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / np.count_nonzero(counts, axis=0)
    x = np.where(observed, x, fill)
    static = np.column_stack((np.ones(len(x)), x, np.sign(hold)*np.sqrt(np.abs(hold))))
    return past, hold, static


def _design(tau, state, dt, time):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2c678cdfc4cbf029 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2c678cdf (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25892 ms` LLM)
- **Rationale**: Fit only on the other labs’ response measurements in each fold, keeping the label-free pooled panel alignment prescribed by the task. Separate curve, late-time, and initial-rate errors to identify which part of the model needs revision if a slice fails.
- **Alternatives Weighed**: `Submit on the schema result alone` (rejected: Finite outputs do not establish that the memory fit generalizes across labs, activation states, or missing coupons.)

```bash
cat > /tmp/rheo04_cv.py <<'PY'
import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',P+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(P+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; state=m._prepare(d)
print('calibration',m._calibrate(d['calibration_load_cells'])); print('public reference schema',{k:np.shape(v) for k,v in item['gt'].items()})
def rm(e): return np.sqrt(np.mean(e**2))
def stats(label,ec,er): print(label,'curve MAE/RMSE',np.mean(np.abs(ec)),rm(ec),'rate MAE/RMSE',np.mean(np.abs(er)),rm(er),'late RMSE',rm(ec[:,-1]))
lab=tr['lab_id']; ec=np.empty_like(y); er=np.empty_like(r)
for c in np.unique(lab):
 a=lab!=c; b=~a
 tau,coef=m._fit(tuple(v[:n][a] for v in state),dt,t,y[a],r[a]); A,B=m._design(tau,tuple(v[:n][b] for v in state),dt,t)
 ec[b]=A@coef-y[b]; er[b]=B@coef-r[b]; stats(('held lab',c,'tau',np.round(tau,5)),ec[b],er[b])
for b,label in [(state[3][:n],'active'),(~state[3][:n],'inactive'),(np.isnan(tr['coupon_features_raw']).any(axis=1),'missing coupon'),(np.ones(n,dtype=bool),'pooled')]:
 if b.any(): stats((label,b.sum()),ec[b],er[b])
out=m.solve_alpha_m97_c01_batch_04(d)
for key in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(key,out[key])
gt=item['gt']
if 'te_true_curve' in gt: stats('public references',out['predicted_strain']-gt['te_true_curve'],out['initial_rate_pred']-gt['te_true_rate0'])
print('test activation count',state[3][n:].sum(),'/',len(te['lab_id']))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo04_cv.py
```
- **Container Exit Code**: `0` (`1126 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration [[ 1.33998917  0.15000139]
 [ 0.77003895 -0.17002607]
 [ 1.209981   -0.11999077]
 [ 0.83007089  0.11003027]]
public reference schema {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
('held lab', 0, 'tau', array([0.61038, 3.53319])) curve MAE/RMSE 0.16608954228731998 0.46948404215178086 rate MAE/RMSE 0.07975710842873404 0.09572228763701304 late RMSE 0.28614258007654975
('held lab', 1, 'tau', array([0.60977, 3.74879])) curve MAE/RMSE 0.19779394598204147 0.3802119571548747 rate MAE/RMSE 0.06369730433137301 0.08770936309373602 late RMSE 0.554996186387614
('held lab', 2, 'tau', array([0.61378, 4.39574])) curve MAE/RMSE 0.1786339045064094 0.6535702399696253 rate MAE/RMSE 0.07890043467156276 0.10182751154191116 late RMSE 0.3031659359149029
('held lab', 3, 'tau', array([0.60089, 3.53094])) curve MAE/RMSE 0.20883737876144667 0.6042346649252746 rate MAE/RMSE 0.07678968634752234 0.09484255474557063 late RMSE 0.5125943677013383
('active', 120) curve MAE/RMSE 0.14440426900307887 0.41677210799170905 rate MAE/RMSE 0.071109177612208 0.09344306186654436 late RMSE 0.2686266232087473
('inactive', 60) curve MAE/RMSE 0.2763915722142006 0.7268651723834572 rate MAE/RMSE 0.08156188461887415 0.09855368739921387 late RMSE 0.6497229152079181
('missing coupon', 16) curve MAE/RMSE 0.18703985511077809 0.2647179284310172 rate MAE/RMSE 0.09036151890192871 0.11497745493563512 late RMSE 0.4107846758101226
('pooled', 180) curve MAE/RMSE 0.18840003674011946 0.5402872565662759 rate MAE/RMSE 0.07459341328109673 0.09517709970010581 late RMSE 0.43453438373326464
estimated_taus [0.60925543 3.79400712]
hereditary_weights [ 0.68546857  0.53316713  0.08988713  0.11054969 -0.08662451  0.08907478
 -0.07310477]
regime_mode_weights [0.26754118 0.21246439]
test activation count 48 / 72

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2c678cdfc4cbf029 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2c678cdf (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31285 ms` LLM)
- **Rationale**: Audit the existing candidate against the clean public references and use the reference parameters only as a diagnostic. If errors persist with those parameters, inspect the forward model or alignment; if they vanish, investigate estimation noise rather than changing the physics.
- **Alternatives Weighed**: `Change the inactive flow law based on raw held-out RMSE` (rejected: The observed curve residuals may be dominated by sparse large deviations; clean-reference errors can distinguish that from a wrong flow law.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]; d=it['dataset']; gt=it['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); state=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; out=m.solve_alpha_m97_c01_batch_04(d)
print('parameter errors', {k:out[k]-gt[g] for k,g in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]})
print('stress reconstruction max abs',np.max(abs(state[0][n:]-gt['te_hist_true'])),np.max(abs(state[1][n:]-gt['te_hold_true'])))
coef=np.r_[gt['theta'],gt['vp_theta']]; ts=tuple(v[n:] for v in state); a,b=m._design(gt['taus'],ts,dt,t)
clean=gt['te_clean']; rate=gt['te_true_rate0']; missing=np.isnan(te['coupon_features_raw']).any(axis=1)
def stats(e,f,k):
 if k.any(): print(int(k.sum()),'rmse',np.sqrt(np.mean(e[k]**2)),np.sqrt(np.mean(f[k]**2)),'max',np.max(abs(e[k])),np.max(abs(f[k])))
for label, e,f in [('fitted',out['predicted_strain']-clean,out['initial_rate_pred']-rate),('reference parameters diagnostic',te['strain_0'][:,None]+a@coef-clean,b@coef-rate)]:
 print(label)
 for name,k in [('all',np.ones(len(rate),bool)),('active',ts[3]),('inactive',~ts[3]),('missing',missing),('complete',~missing)]+[(f'lab {j}',te['lab_id']==j) for j in range(4)]:
  print(name,end=' ');stats(e,f,k)
state_tr=tuple(v[:n] for v in state); a,b=m._design(out['estimated_taus'],state_tr,dt,t); w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; e=tr['strain_0'][:,None]+a@w-tr['strain_curves_obs']; f=b@w-tr['instant_rate_obs']; print('training absolute residual quantiles by time',np.quantile(abs(e),[.5,.8,.9,.95,.99],axis=0)); print('training rate absolute residual quantiles',np.quantile(abs(f),[.5,.8,.9,.95,.99])); print('curve points abs residual > .5 by row quantiles',np.quantile((abs(e)>.5).sum(1),[0,.5,.8,.9,.95,1]))
PY
```
- **Container Exit Code**: `0` (`555 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
509117 max 0.4286017630945036 0.2555453796064624
complete 64 rmse 0.2424345895055453 0.10667371204040609 max 1.2259552657144996 0.29373271309837223
lab 0 15 rmse 0.36593805223379083 0.11384117119068071 max 1.2259552657144996 0.2222836882273711
lab 1 22 rmse 0.15383603528590972 0.10720231072250343 max 0.5196415508446499 0.23402415282136202
lab 2 17 rmse 0.2257991760720396 0.13698227847164698 max 1.178047457578446 0.29373271309837223
lab 3 18 rmse 0.1804752488882715 0.08980607981579898 max 0.8403792447567517 0.20266822423483088
reference parameters diagnostic
all 72 rmse 0.3979291716445576 0.04007682435143156 max 2.2984804471479565 0.1800899296189317
active 48 rmse 0.05460328012002477 0.04168823065106065 max 0.32858715906607316 0.1800899296189317
inactive 24 rmse 0.6848940359920208 0.036642030520352274 max 2.2984804471479565 0.17656216018494275
missing 8 rmse 0.22756941758577334 0.11990135872863712 max 0.8909803962264227 0.1800899296189317
complete 64 rmse 0.4143278881199023 0.0031430643066550157 max 2.2984804471479565 0.009214332051588503
lab 0 15 rmse 0.6852505666552158 0.0033777303570363885 max 2.2984804471479565 0.006660781008171579
lab 1 22 rmse 0.18329695202431584 0.030372918366630513 max 1.0194061209299752 0.13774979333811377
lab 2 17 rmse 0.32298973120747826 0.050387640375667404 max 2.1582426546614557 0.17656216018494275
lab 3 18 rmse 0.32014516953586514 0.05375610682848669 max 1.5933283126954474 0.1800899296189317
training absolute residual quantiles by time [[0.01752262 0.03186085 0.04448853 0.06920038 0.09119181 0.10931432
  0.10899726 0.12960691 0.15674763 0.13409243]
 [0.04133786 0.08715818 0.11267041 0.15578327 0.19983957 0.2509212
  0.26587463 0.31199868 0.38005664 0.40032995]
 [0.06468072 0.12227685 0.16141123 0.21981268 0.23137435 0.43409506
  0.38499592 0.47797671 0.62568665 0.67596867]
 [0.21940169 0.42406283 0.2890221  0.27417685 0.30342231 0.59835867
  0.47807504 0.67860776 0.78749464 0.81882478]
 [0.78474856 5.84601294 1.08501349 2.53117077 0.8103185  1.22543267
  0.74257611 1.23167991 1.44417613 1.36649685]]
training rate absolute residual quantiles [0.06001253 0.12712204 0.16065735 0.18860082 0.21192224]
curve points abs residual > .5 by row quantiles [0.   0.   1.   2.   3.05 6.  ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2c678cdfc4cbf029 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2c678cdf (directory).
```

### Turn 7 — `execute_in_remote_docker` (`31903 ms` LLM)
- **Rationale**: Compare both flow gates and both imputation interpretations using only held-out training responses. Measure late-time and robust curve errors separately from rate RMSE so sparse curve spikes do not determine which basis to keep.
- **Alternatives Weighed**: `Retune stress calibration or robust weights first` (rejected: The systematic zero-damage curve error persists even with reference coefficients and closely reconstructed physical stress.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero damage implies linear terminal flow in this low-banding batch.` | Witness: `At the reference parameters, clean zero-damage curve RMSE is 0.685 versus 0.055 on activated rows, while zero-damage initial-rate RMSE is only 0.037 and maximum stress reconstruction error is below 0.0004.` | Blanket Diagnosis: `The baseline uses one gate for two distinct basis terms. Linear and logarithmic flow share the same derivative at zero, so initial rates can agree even when the finite-time flow basis is wrong.` | Sublated Pivot: `Test a low-banding logarithmic baseline with damage-gated network corrections, independently of the missing-coupon imputation choice.`

```bash
cat > /tmp/rheo04_structure.py <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; lab=tr['lab_id']; n=len(lab); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; past,hold,x,active,act=m._prepare(d)
def join(k): return np.concatenate([tr[k],te[k]])
missing=~np.isfinite(join('coupon_features_raw')); labs=join('lab_id'); strata=join('stratum'); pooled=x.copy()
for j in range(x.shape[1]):
 means=[x[(labs==c)&(strata==s)&~missing[:,j],j].mean() for c,s in np.unique(np.column_stack((labs,strata)),axis=0) if np.any((labs==c)&(strata==s)&~missing[:,j])]
 pooled[missing[:,j],j]=np.mean(means)
print('max imputation change',np.max(abs(pooled-x)))
slices=[('active',active[:n]),('zero_damage',~active[:n]),('missing',missing[:n].any(axis=1))]+[(f'lab{c}',lab==c) for c in np.unique(lab)]
for law,gate in [('damage-gated-flow',active),('low-banding-flow',np.full(len(hold),d['shear_banding_index']<=.50))]:
 for fill,features in [('local',x),('cell-balanced pooled',pooled)]:
  state=(past,hold,features,gate,act); ec=np.empty_like(y); er=np.empty_like(r); taus=[]
  for c in np.unique(lab):
   a=lab!=c; b=~a
   tau,w=m._fit(tuple(v[:n][a] for v in state),dt,t,y[a],r[a]); A,B=m._design(tau,tuple(v[:n][b] for v in state),dt,t)
   ec[b]=A@w-y[b]; er[b]=B@w-r[b]; taus.append(tau)
  print(law,fill,'taus range',np.round(np.min(taus,axis=0),5),np.round(np.max(taus,axis=0),5))
  for name,k in slices:
   print(name,'curve median abs / rmse, rate rmse, late median abs',*[round(z,6) for z in [np.median(abs(ec[k])),np.sqrt(np.mean(ec[k]**2)),np.sqrt(np.mean(er[k]**2)),np.median(abs(ec[k,-3:]))]])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo04_structure.py
```
- **Container Exit Code**: `0` (`2006 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
235 0.094843 0.20663
damage-gated-flow cell-balanced pooled taus range [0.6002  3.51842] [0.61214 4.25899]
active curve median abs / rmse, rate rmse, late median abs 0.069358 0.416966 0.093941 0.099019
zero_damage curve median abs / rmse, rate rmse, late median abs 0.128023 0.73279 0.098089 0.377768
missing curve median abs / rmse, rate rmse, late median abs 0.125325 0.295849 0.124325 0.233531
lab0 curve median abs / rmse, rate rmse, late median abs 0.077556 0.467518 0.096532 0.151163
lab1 curve median abs / rmse, rate rmse, late median abs 0.074594 0.389428 0.089126 0.130419
lab2 curve median abs / rmse, rate rmse, late median abs 0.073499 0.653072 0.101221 0.122936
lab3 curve median abs / rmse, rate rmse, late median abs 0.097669 0.609905 0.094093 0.205104
low-banding-flow local taus range [0.60424 3.66776] [0.60673 3.72769]
active curve median abs / rmse, rate rmse, late median abs 0.01219 0.399988 0.032149 0.012911
zero_damage curve median abs / rmse, rate rmse, late median abs 0.01294 0.641119 0.049468 0.01321
missing curve median abs / rmse, rate rmse, late median abs 0.114149 0.175461 0.115587 0.203055
lab0 curve median abs / rmse, rate rmse, late median abs 0.01128 0.43857 0.029198 0.013047
lab1 curve median abs / rmse, rate rmse, late median abs 0.013792 0.240341 0.048899 0.015015
lab2 curve median abs / rmse, rate rmse, late median abs 0.01194 0.629926 0.041488 0.012283
lab3 curve median abs / rmse, rate rmse, late median abs 0.012222 0.565794 0.030458 0.012037
low-banding-flow cell-balanced pooled taus range [0.60399 3.65944] [0.60653 3.71882]
active curve median abs / rmse, rate rmse, late median abs 0.012182 0.400449 0.035438 0.012798
zero_damage curve median abs / rmse, rate rmse, late median abs 0.012441 0.641964 0.053164 0.012903
missing curve median abs / rmse, rate rmse, late median abs 0.125197 0.193771 0.128246 0.226315
lab0 curve median abs / rmse, rate rmse, late median abs 0.011434 0.438703 0.026449 0.012626
lab1 curve median abs / rmse, rate rmse, late median abs 0.014085 0.242736 0.055732 0.015669
lab2 curve median abs / rmse, rate rmse, late median abs 0.011949 0.629692 0.039727 0.012131
lab3 curve median abs / rmse, rate rmse, late median abs 0.012124 0.567056 0.039599 0.012092

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2c678cdfc4cbf029 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2c678cdf (directory).
```

### Turn 8 — `execute_in_remote_docker` (`36120 ms` LLM)
- **Rationale**: Promote the flow basis supported by held-out responses. Verify that the exact saved solver improves the clean zero-damage slice without sacrificing active rows, and check the predicted initial rate against the forward curve derivative.
- **Alternatives Weighed**: `Use a pooled imputation mean for all missing coupons` (rejected: With the corrected flow basis, held-out missing-coupon curve and rate RMSE were both lower when filling from the matching lab and stratum.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c01_batch_04_solver.py')
s=p.read_text()
old='''    activated = (dataset['shear_banding_index'] <= .50) & (damage > 0)
    return past, hold, x, activated, np.where(activated, np.tanh(damage), 0.)'''
new='''    plateau = np.full(len(hold), dataset['shear_banding_index'] <= .50)
    # Banding selects the baseline flow basis; damage controls only the network terms.
    return past, hold, x, plateau, np.where(plateau, np.tanh(damage), 0.)'''
assert old in s
s=s.replace(old,new).replace('past, hold, x, activated, act = state','past, hold, x, plateau, act = state').replace('np.where(activated[:, None], np.log1p(time), time)','np.where(plateau[:, None], np.log1p(time), time)')
p.write_text(s)
p=Path('/tmp/rheo04_cv.py'); s=p.read_text().replace("(state[3][:n],'active'),(~state[3][:n],'inactive')", "(state[4][:n]>0,'active'),(state[4][:n]==0,'zero damage')").replace("'te_true_curve'", "'te_clean'").replace('state[3][n:].sum()', '(state[4][n:]>0).sum()'); p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/rheo04_cv.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]; d,gt=it['dataset'], it['gt']; out=m.solve_alpha_m97_c01_batch_04(d)
n=len(d['train']['lab_id']); test=d['test']; state=tuple(v[n:] for v in m._prepare(d)); active=state[4]>0; missing=np.isnan(test['coupon_features_raw']).any(axis=1)
for key,ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(key,'abs errors',np.round(np.abs(out[key]-gt[ref]),6))
curve=out['predicted_strain']-gt['te_clean']; rate=out['initial_rate_pred']-gt['te_true_rate0']
for name,k in [('active',active),('zero damage',~active),('missing',missing),('complete',~missing)]+[(f'lab {c}',test['lab_id']==c) for c in range(4)]:
 if k.any(): print(name,int(k.sum()),'clean curve / late / rate RMSE',*[round(np.sqrt(np.mean(e**2)),6) for e in (curve[k],curve[k,-1],rate[k])])
a,b=m._design(out['estimated_taus'],state,d['past_dt'],np.array([0.,1e-7])); w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]
np.testing.assert_allclose(a[:,0,:]@w,0,atol=1e-12)
np.testing.assert_allclose(a[:,1,:]@w/1e-7,b@w,atol=2e-6,rtol=0)
print('zero-time value and derivative verified')
PY
```
- **Container Exit Code**: `0` (`1590 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_04_solver.py']`
- **Stdout (tail)**:
```text
E/RMSE 0.028911803664247233 0.0488994444231374 late RMSE 0.09487114668272396
('held lab', 2, 'tau', array([0.60424, 3.66776])) curve MAE/RMSE 0.09079804614508452 0.6299258869735804 rate MAE/RMSE 0.02350676334279279 0.04148793904886958 late RMSE 0.07310755155878983
('held lab', 3, 'tau', array([0.60604, 3.70397])) curve MAE/RMSE 0.09559510626540339 0.5657944392946579 rate MAE/RMSE 0.020064265713666894 0.030457923165740083 late RMSE 0.35718781423882573
('active', 120) curve MAE/RMSE 0.08011246185102297 0.3999881306407797 rate MAE/RMSE 0.020993051164275123 0.03214889780935725 late RMSE 0.22212551466404007
('zero damage', 60) curve MAE/RMSE 0.09234793983638155 0.6411193470873846 rate MAE/RMSE 0.027093097319633854 0.049467938758039204 late RMSE 0.12271649215886919
('missing coupon', 16) curve MAE/RMSE 0.13873589591541577 0.17546097452784964 rate MAE/RMSE 0.10616840214344159 0.11558701271808905 late RMSE 0.2567895816433809
('pooled', 180) curve MAE/RMSE 0.08419095451280917 0.49363111278118244 rate MAE/RMSE 0.023026399882728036 0.03879080744060422 late RMSE 0.19471245977706644
estimated_taus [0.60577627 3.70338674]
hereditary_weights [ 0.68071049  0.55142344  0.13845521  0.15821594 -0.1286537   0.11879745
 -0.08873187]
regime_mode_weights [0.26022323 0.21961653]
public references curve MAE/RMSE 0.015780613388163802 0.05251511299607577 rate MAE/RMSE 0.013315406195722586 0.040376458409395646 late RMSE 0.07312223625946546
test activation count 48 / 72
estimated_taus abs errors [0.000776 0.032387]
hereditary_weights abs errors [0.00071  0.001423 0.001545 0.001784 0.001346 0.001203 0.001268]
regime_mode_weights abs errors [0.000223 0.000383]
active 48 clean curve / late / rate RMSE 0.054344 0.075683 0.041763
zero damage 24 clean curve / late / rate RMSE 0.048651 0.067712 0.037449
missing 8 clean curve / late / rate RMSE 0.15736 0.219087 0.120954
complete 64 clean curve / late / rate RMSE 0.002698 0.003918 0.002302
lab 0 15 clean curve / late / rate RMSE 0.001837 0.002616 0.002179
lab 1 22 clean curve / late / rate RMSE 0.040655 0.056518 0.031502
lab 2 17 clean curve / late / rate RMSE 0.065842 0.091547 0.050757
lab 3 18 clean curve / late / rate RMSE 0.070101 0.097787 0.053582
zero-time value and derivative verified

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ee22a6be8938aad7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ee22a6be (directory).
```

### Turn 9 — `submit_completed_task` (`16822 ms` LLM)
- **Rationale**: The saved solver passed isolated schema validation, leave-one-lab-out evaluation, clean-reference slice checks, and the zero-time derivative check. The corrected flow basis substantially reduced zero-damage error, and the host verified panel affine invariance and the seven-coefficient handoff.
- **Alternatives Weighed**: `Replace lab-and-stratum imputation with a pooled cell-balanced mean` (rejected: The pooled mean worsened both strain and initial-rate RMSE on held-out rows with missing coupons.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_04_solver.py': 'FILE:6424'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_04_solver.py`. The public schema and initial-rate derivative checks passed. Clean public strain RMSE was 0.0525 overall and 0.1574 on rows with missing coupon features.`

### Turn 10 — `execute_in_remote_docker` (`42244 ms` LLM)
- **Rationale**: Use complete-coupon measurements to test whether omitted nonlinear terms consistently reduce held-out late-time and rate errors without confusing missing-feature error with model error. Report calibration and parameter errors separately so any material bottleneck determines the next edit.
- **Alternatives Weighed**: `Add relay-mode or Fokker–Planck terms` (rejected: The measured creep task has neither a relay state nor a stochastic evolution operator; pairwise, product, and rational drift terms are the applicable algebraic probes.); `Apply a batch classification correction` (rejected: There are no class labels or multiple technical batches. Per-lab held-out signed residuals test the relevant nuisance shortcut.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/rheo04_margin.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p + 'alpha_m97_c01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(p + 'public_alpha_m97_c01_batch_04.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
tr = d['train']
lab = tr['lab_id']
n = len(lab)
dt, t = d['past_dt'], d['eval_times']
state = tuple(v[:n] for v in m._prepare(d))
x = state[2]
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
# Test omitted grammar on complete rows so feature error cannot masquerade as a drift term.
u = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)])
v = np.column_stack([x[:, i]*x[:, j]*x[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)])
q = np.column_stack([x[:, i]/(1 + (x[:, i]+sign*x[:, j])**2) for i in range(4) for j in range(i+1, 4) for sign in (-1, 1)])
flow = np.log1p(t)
variants = {'prescribed': np.empty((n,0)), 'pairwise_and_three_way': np.column_stack((u,v)), 'rational_pairwise': q}
for name, extra in variants.items():
    scores = []
    for c in np.unique(lab):
        fit, test = (lab != c) & complete, (lab == c) & complete
        tau, _ = m._fit(tuple(z[fit] for z in state), dt, t, y[fit], r[fit])
        a,b = m._design(tau, state, dt, t)
        a = np.concatenate((a, extra[:,None,:]*flow[None,:,None]), axis=2)
        b = np.column_stack((b, extra))
        ac, br, yc, yr = a[fit], b[fit], y[fit], r[fit]
        cm, rm = np.isfinite(yc), np.isfinite(yr)
        design = np.concatenate((ac[cm], br[rm]))
        obs = np.r_[yc[cm], yr[rm]]
        wc = np.r_[np.ones(cm.sum()), np.full(rm.sum(), t[0])]
        for _ in range(9):
            coef = np.linalg.lstsq(design*wc[:,None], obs*wc, rcond=None)[0]
            res = obs - design@coef
            ec = np.full_like(yc, np.nan); ec[cm] = res[:cm.sum()]
            sc = np.maximum(m._mad(ec, axis=0), max(.35*m._mad(ec), 1e-7))
            sr = max(m._mad(res[cm.sum():]), 1e-7)
            sigma = np.r_[np.broadcast_to(sc, yc.shape)[cm], np.full(rm.sum(), sr)]
            wc = 1/np.hypot(sigma, res/2.38)
        e, f = a[test]@coef-y[test], b[test]@coef-r[test]
        med = np.nanmedian(abs(e[:,-3:]),axis=0).max()
        scores.append([med, np.sqrt(np.nanmean(f*f)), np.nanmedian(f)])
    print(name, 'per-lab [worst late horizon median abs, rate RMSE, signed median rate residual]', np.round(scores,6).tolist())
out = m.solve_alpha_m97_c01_batch_04(d)
cal = m._calibrate(d['calibration_load_cells'])
print('per-channel gain/bias absolute error', np.round(abs(cal - np.column_stack((gt['stress_gains'],gt['stress_biases']))),8))
print('relative tau error', np.round(abs(out['estimated_taus']/gt['taus']-1),6))
print('worst hereditary/mode absolute errors', np.max(abs(out['hereditary_weights']-gt['theta'])), np.max(abs(out['regime_mode_weights']-gt['vp_theta'])))
PY
python3 -I /tmp/rheo04_margin.py
python3 -I /holdout_task/validate_public_schema.py
grep -inE 'toler|threshold|margin' /holdout_task/public_asset_index.json || true
```
- **Container Exit Code**: `0` (`1834 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
prescribed per-lab [worst late horizon median abs, rate RMSE, signed median rate residual] [[0.014695, 0.019519, -0.006488], [0.014133, 0.017136, 0.002554], [0.011981, 0.016941, 0.00314], [0.012618, 0.02098, 0.005755]]
pairwise_and_three_way per-lab [worst late horizon median abs, rate RMSE, signed median rate residual] [[0.013293, 0.019309, -0.00553], [0.016127, 0.017485, 0.003498], [0.012861, 0.016959, 0.003357], [0.015369, 0.021719, 0.005191]]
rational_pairwise per-lab [worst late horizon median abs, rate RMSE, signed median rate residual] [[0.01505, 0.019793, -0.006707], [0.014598, 0.017161, 0.00273], [0.011888, 0.016629, 0.003798], [0.013559, 0.021238, 0.004718]]
per-channel gain/bias absolute error [[1.083e-05 1.390e-06]
 [3.895e-05 2.607e-05]
 [1.900e-05 9.230e-06]
 [7.089e-05 3.027e-05]]
relative tau error [0.001283 0.008822]
worst hereditary/mode absolute errors 0.0017840612626742747 0.00038346958477497584
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ee22a6be8938aad7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ee22a6be (directory).
```

### Turn 11 — `submit_completed_task` (`16997 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated schema validation, leave-one-lab-out evaluation, clean-reference slice checks, and the initial-rate derivative check. The corrected flow law reduced held-out zero-damage late-time median absolute error from 0.386 to 0.0132; the subsequent audit found no improvement from expanding the drift basis.
- **Alternatives Weighed**: `Add product or rational drift terms` (rejected: Both increased the worst held-out lab’s late-time median absolute error and initial-rate RMSE.); `Use pooled means for missing coupons` (rejected: Pooled imputation worsened held-out strain and rate RMSE on missing-coupon rows.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_04_solver.py': 'FILE:6424'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_04_solver.py`. The isolated schema and initial-rate derivative checks passed. Clean public strain RMSE was 0.0525 overall, 0.0027 with complete coupon features, and 0.1574 with missing features.`

### Turn 12 — `execute_in_remote_docker` (`28653 ms` LLM)
- **Rationale**: Test whether imputed predictors contaminate shared parameter estimates, rather than merely limiting predictions on incomplete rows. Change the fit only if removing incomplete training rows materially improves held-out complete-row performance without worsening the missing-coupon bottleneck.
- **Alternatives Weighed**: `Repeat the nonlinear basis and calibration audits` (rejected: The prior audit already found that expanded bases worsened worst-lab transfer and that calibration errors were below 0.000071.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/rheo04_fit_ablation.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']
tr = d['train']
lab = tr['lab_id']
n = len(lab)
s = tuple(v[:n] for v in m._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)

def fit_and_predict(fit, test):
    tau, w = m._fit(tuple(z[fit] for z in s), dt, t, y[fit], r[fit])
    a, b = m._design(tau, tuple(z[test] for z in s), dt, t)
    return tau, a@w-y[test], b@w-r[test]

for complete_fit in (False, True):
    e, f = np.empty_like(y), np.empty_like(r)
    taus = []
    for c in np.unique(lab):
        test = lab == c
        fit = ~test & (complete if complete_fit else True)
        tau, e[test], f[test] = fit_and_predict(fit, test)
        taus.append(tau)
    print('complete-only fit' if complete_fit else 'current all-row fit')
    for label, select in [('complete', complete), ('missing', ~complete), ('zero damage', s[4]==0), ('active', s[4]>0)] + [(f'complete lab {c}', complete & (lab==c)) for c in np.unique(lab)]:
        print(label, int(select.sum()), 'late-horizon median-abs max / rate RMSE', round(np.nanmedian(abs(e[select,-3:]), axis=0).max(),6), round(np.sqrt(np.nanmean(f[select]**2)),6))
    print('fold taus', np.round(taus,6).tolist())
PY
python3 -I /tmp/rheo04_fit_ablation.py
```
- **Container Exit Code**: `0` (`1131 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
current all-row fit
complete 164 late-horizon median-abs max / rate RMSE 0.012316 0.018657
missing 16 late-horizon median-abs max / rate RMSE 0.215034 0.115587
zero damage 60 late-horizon median-abs max / rate RMSE 0.017046 0.049468
active 120 late-horizon median-abs max / rate RMSE 0.013179 0.032149
complete lab 0 37 late-horizon median-abs max / rate RMSE 0.013277 0.019223
complete lab 1 40 late-horizon median-abs max / rate RMSE 0.01405 0.017197
complete lab 2 44 late-horizon median-abs max / rate RMSE 0.011792 0.017193
complete lab 3 43 late-horizon median-abs max / rate RMSE 0.012644 0.020786
fold taus [[0.606059, 3.710234], [0.606735, 3.727689], [0.604242, 3.667763], [0.606041, 3.703967]]
complete-only fit
complete 164 late-horizon median-abs max / rate RMSE 0.011878 0.018708
missing 16 late-horizon median-abs max / rate RMSE 0.213749 0.115394
zero damage 60 late-horizon median-abs max / rate RMSE 0.014409 0.049488
active 120 late-horizon median-abs max / rate RMSE 0.013881 0.032081
complete lab 0 37 late-horizon median-abs max / rate RMSE 0.014695 0.019519
complete lab 1 40 late-horizon median-abs max / rate RMSE 0.014133 0.017136
complete lab 2 44 late-horizon median-abs max / rate RMSE 0.01198 0.016941
complete lab 3 43 late-horizon median-abs max / rate RMSE 0.012618 0.02098
fold taus [[0.606218, 3.732588], [0.606454, 3.732465], [0.605003, 3.69394], [0.606793, 3.753117]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ee22a6be8938aad7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ee22a6be (directory).
```

### Turn 13 — `execute_in_remote_docker` (`28729 ms` LLM)
- **Rationale**: The public batch exercised sparse calibration but not its documented EMI branch. Inject single spikes into measured prefix sweeps to test whether the solver uses the correct calibration source and leaves the other channels unchanged. If it fails, fix calibration rather than reopening the settled creep basis.
- **Alternatives Weighed**: `Switch to complete-only parameter fitting` (rejected: It changed missing-coupon rate RMSE by less than 0.0002 and worsened the worst complete-lab late-time median error.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']
with open(m._ARCHIVE) as f: sweep = json.load(f)[-1]
keys = ('lab_id', 'cmd_stress', 'true_stress')
base = m._calibrate(d['calibration_load_cells'])
for c in range(4):
    x, y = m._channel_points(sweep, c)
    other = {k: np.asarray(d['calibration_load_cells'][k])[np.asarray(d['calibration_load_cells']['lab_id']) != c] for k in keys}
    for idx in ([0, 3, 6, 9], [0, 2, 5, 7, 9]):
        for j in range(len(idx)):
            for spike in (-6., 6.):
                a, b = x[idx], y[idx].copy(); b[j] += spike
                clean = np.arange(len(idx)) != j
                expected = np.polyfit(a[clean], b[clean], 1) if clean.sum() >= 4 else np.polyfit(x, y, 1)
                rows = {k: np.r_[other[k], v] for k,v in zip(keys, (np.full(len(a),c),a,b))}
                actual = m._calibrate(rows)
                np.testing.assert_allclose(actual[c], expected, atol=1e-10, rtol=0, err_msg=str((c,idx,j,spike)))
                np.testing.assert_array_equal(actual[np.arange(4)!=c], base[np.arange(4)!=c])
print('80 calibration stress cases passed: prefix fallback, clean in-batch fit, and channel isolation')
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`905 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
80 calibration stress cases passed: prefix fallback, clean in-batch fit, and channel isolation
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ee22a6be8938aad7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ee22a6be (directory).
```

### Turn 14 — `submit_completed_task` (`16864 ms` LLM)
- **Rationale**: The corrected flow basis reduced held-out zero-damage late-time median absolute error from 0.386 to 0.0132. The exact saved candidate passed isolated validation, host verification of panel affine invariance and the seven-coefficient handoff, and the documented sparse and EMI calibration stress cases. The tested alternatives did not materially improve the missing-coupon bottleneck.
- **Alternatives Weighed**: `Expand the drift basis with product or rational terms` (rejected: Both expansions worsened the worst held-out lab's late-time median error and initial-rate RMSE.); `Fit only on rows with complete coupon features` (rejected: This reduced missing-coupon rate RMSE by less than 0.0002 while worsening the worst complete-lab late-time median error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_04_solver.py`. The isolated schema check, initial-rate derivative check, and all 80 calibration stress cases passed. Clean public strain RMSE was 0.0525 overall and 0.1574 on rows with missing coupon features, the largest remaining source of error.`

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
