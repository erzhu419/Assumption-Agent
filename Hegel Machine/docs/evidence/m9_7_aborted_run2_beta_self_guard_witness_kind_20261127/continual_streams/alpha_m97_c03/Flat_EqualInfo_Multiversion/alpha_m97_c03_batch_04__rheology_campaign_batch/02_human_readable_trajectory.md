# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `402,161` (`388,813` / `101,248` / `13,348`) |
| **Total Episode Wall Time (s)** | `441.39s` |

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

### Turn 1 — `execute_in_remote_docker` (`14925 ms` LLM)
- **Rationale**: Inspect the public schema and archive inventory to identify the available labels, panel layout, and any calibration fallback needed before implementing the fit.
- **Alternatives Weighed**: `Start from the rejected calibration payload` (rejected: The contract calls for fitting clean in-batch rows before consulting archive calibration.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation' ; python3 - <<'PY'
import pickle
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_04.pkl','rb'))
for k,v in p.items():
 print(k,type(v).__name__,getattr(v,'shape',None),list(v)[:25] if isinstance(v,dict) else (len(v) if hasattr(v,'__len__') else v))
PY
```
- **Container Exit Code**: `1` (`434 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
}total 56
-rw-r--r-- 1 root root 7294 Nov 27 08:10 alpha_m97_c03_batch_01_solver.py
-rw-r--r-- 1 root root 8716 Nov 27 08:10 alpha_m97_c03_batch_02_solver.py
-rw-r--r-- 1 root root 9430 Nov 27 08:10 alpha_m97_c03_batch_03_solver.py
-rw-r--r-- 1 root root 7801 Nov 27 08:10 alpha_m97_c03_s00_solver.py
-r--r--r-- 1 root root 7215 Nov 27 08:10 archive_calibration_history.json
-rw-r--r-- 1 root root 1893 Nov 27 08:10 archive_manifest.json
-rw-r--r-- 1 root root  337 Nov 27 08:10 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`21696 ms` LLM)
- **Rationale**: Read only the dataset inputs, not the public reference targets, to establish the panel layout and identify which channels require archive fallback. Inspect the prior implementation and skill preconditions before choosing reusable preprocessing.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: Its calibration and panel preprocessing may transfer, but the current plateau basis must be fitted for this batch.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: ds=pickle.load(f)[0]['dataset']
for k,v in ds.items():
 print(k, {a:(np.shape(b) if not isinstance(b,dict) else {x:np.shape(y) for x,y in b.items()}) for a,b in v.items()} if isinstance(v,dict) else (v if np.size(v)<22 else np.shape(v)))
for k,v in ds['calibration_load_cells'].items(): print(k,v)
PY
cat /tmp/hegel_continual_archive/archive_calibration_history.json | head -85
sed -n '1,185p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -45
```
- **Container Exit Code**: `0` (`448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
q = r[ix]
                s[ix] = max(1e-5, 1.4826 * np.median(np.abs(q - np.median(q))))
        return s

    low = min(np.min(dt), np.min(t[t > 0])) / 12
    high = max(dt.sum(), t.max()) * 12
    bounds = np.log([low, high])
    short = np.median(dt)
    long = np.median(t)
    starts = [(0.4 * short, long), (short, long), (2 * short, 2 * long), (short, 4 * long)]
    w = np.ones_like(y)
    for b in blocks:
        n = len(y[b]); w[b] = 1 / np.sqrt(max(n, 1))
    fits = [least_squares(profile, np.log(pair), args=(w,), bounds=bounds, max_nfev=75)
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
```

### Turn 3 — `execute_in_remote_docker` (`18664 ms` LLM)
- **Rationale**: Compare the archived affine fits to determine whether pooling would mix incompatible calibrations for channels 1–3, and inspect the prior panel code to resolve how the 32 reference values are grouped into four columns.
- **Alternatives Weighed**: `Use the generic control-panel alignment skill` (rejected: The task specifies per-lab reference means and standard deviations, rather than a shared scanner gain.)

```bash
sed -n '1,110p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: history=json.load(f)
for a in history:
 l,x,y=(np.asarray(a[k]) for k in ('lab_id','cmd_stress','true_stress'))
 print(a['episode_short_name'],[(len(x[l==c]),np.polyfit(x[l==c],y[l==c],1).round(5).tolist()) for c in range(4) if np.sum(l==c)>=2])
with open('/holdout_task/public_alpha_m97_c03_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
for name in ('train','test'):
 s=d[name]; print(name,'lab counts',np.bincount(s['lab_id']),'strata',np.unique(s['stratum']).tolist()); print('NaNs',{k:int(np.isnan(v).sum()) for k,v in s.items() if np.issubdtype(v.dtype,np.floating)})
PY
```
- **Container Exit Code**: `0` (`353 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
False
        possible_emi = False
        for j in range(len(x)):
            keep = np.arange(len(x)) != j
            if np.ptp(x[keep]) == 0:
                continue
            inverse = np.linalg.pinv(design[keep])
            line = inverse @ y[keep]
            predict = design[j] @ inverse
            deleted = abs(y[j] - predict @ y[keep])
            uncertainty = noise * (1 + np.abs(predict).sum())
            # Check line consistency separately from possible spike amplitude:
            # the highest-leverage reading can have the least precise prediction.
            # Three-point sweeps need a tighter consistency screen to
            # distinguish a borderline spike from a clean near-repeat.
            consistency_limit = uncertainty * (0.625 if len(x) == 3 else 1.0)
            inconsistent |= deleted > consistency_limit
            possible_emi |= deleted + uncertainty > 3.0
            error = np.max(np.abs(y[keep] - design[keep] @ line))
            if deleted + uncertainty > 3.0:
                candidates.append((error, keep, line))
        if not (inconsistent and possible_emi):
            continue
        prior = archived_line(lab)
        archive_error = np.abs(y - design @ prior)
        keep = archive_error <= 2 * noise
        if keep.sum() == len(x) - 1 and archive_error[~keep][0] > 3.0 - 2 * noise:
            # The archive identifies the spike independently of extrapolation.
            coefficients[lab] = (np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
                                 if np.ptp(x[keep]) >= 0.1 else prior)
alpha_m97_c03_s00 [(10, [1.0, 1e-05]), (10, [0.99999, -2e-05]), (10, [1.00004, 2e-05]), (10, [1.00004, -8e-05])]
alpha_m97_c03_batch_01 [(10, [1.28999, 0.12]), (10, [0.82004, -0.15003]), (10, [1.16998, 0.14001]), (10, [0.76, -0.10999])]
alpha_m97_c03_batch_02 [(10, [1.28996, 0.11998]), (10, [0.82003, -0.15001]), (10, [1.16999, 0.13998]), (10, [0.75997, -0.11003])]
alpha_m97_c03_batch_03 [(10, [1.28996, 0.11998]), (10, [0.81998, -0.14996]), (10, [1.17, 0.14001]), (10, [0.76, -0.11004])]
train lab counts [48 43 42 47] strata [0, 1, 2]
NaNs {'coupon_features_raw': 25, 'ref_coupons': 223, 'past_stress_cmd': 0, 'stress_hold_cmd': 0, 'strain_0': 0, 'aux_chamber_telemetry': 0, 'strain_curves_obs': 0, 'instant_rate_obs': 0}
test lab counts [18 17 16 21] strata [0, 1, 2]
NaNs {'coupon_features_raw': 6, 'ref_coupons': 85, 'past_stress_cmd': 0, 'stress_hold_cmd': 0, 'strain_0': 0, 'aux_chamber_telemetry': 0}
```

### Turn 4 — `execute_in_remote_docker` (`60812 ms` LLM)
- **Rationale**: Fit only the two relaxation times nonlinearly and project the nine weights jointly from curves and initial rates. The schema check detects output-contract failures; held-out labs and damage counts reveal whether alignment generalizes or a reachable regime branch is mishandled.
- **Alternatives Weighed**: `Pool historical calibration across all episodes` (rejected: The initial episode has identity calibration, unlike the three subsequent sweeps; pooling would bias the missing-channel fits.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
import ast
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py').read_text()
helpers = {node.name: ast.get_source_segment(p, node) for node in ast.parse(p).body if isinstance(node, ast.FunctionDef)}
Path('alpha_m97_c03_batch_04_solver.py').write_text('''"""Channel-calibrated two-branch strain-crystallization model."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


''' + '\n\n\n'.join(helpers[k] for k in ('_align', '_calibrate')) + '\n')
PY
cat >> alpha_m97_c03_batch_04_solver.py <<'PY'


def _design(sample, aligned, calibration, dt, times, threshold, banding):
    gain, bias = calibration[np.asarray(sample['lab_id'])].T
    history = gain[:, None] * np.asarray(sample['past_stress_cmd']) + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(history) - threshold, 0.0)**2 @ dt
    plateau = (banding <= 0.50) & (damage > 0)
    activation = np.where(plateau, np.tanh(damage), 0.0)
    flow = np.column_stack((np.ones(len(hold)), aligned))
    tail = np.where(plateau[:, None], np.log1p(times), times)

    def evaluate(log_tau):
        tau = np.exp(log_tau)
        alpha = np.exp(-dt[:, None] / tau)
        beta = -np.expm1(-dt[:, None] / tau)
        z = np.zeros((len(hold), 2))
        for j in range(len(dt)):
            z = alpha[j] * z + beta[j] * history[:, j, None]
        gap = hold[:, None] - z
        relax = -np.expm1(-times[:, None] / tau)
        extra = activation * hold
        arrest = activation * z[:, 0]
        curves = np.concatenate((gap[:, None, :] * relax,
                                 flow[:, None, :] * tail[:, :, None],
                                 (extra[:, None] * relax[:, 0])[:, :, None],
                                 (arrest[:, None] * np.log1p(times))[:, :, None]), axis=2)
        rates = np.column_stack((gap / tau, flow, extra / tau[0], arrest))
        return curves, rates

    return evaluate


def _fit(evaluate, curves, rates, dt, times):
    curves_ok, rates_ok = np.isfinite(curves), np.isfinite(rates)
    y = np.r_[curves[curves_ok], rates[rates_ok]]
    group = np.r_[np.nonzero(curves_ok)[1], np.full(rates_ok.sum(), len(times))]
    counts = np.bincount(group, minlength=len(times) + 1)
    base_weight = np.where(group == len(times), 1.0, 1.0 / np.sqrt(len(times)))

    def matrix(p):
        a, b = evaluate(p)
        return np.concatenate((a[curves_ok], b[rates_ok]))

    def project(a, w):
        return np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]

    def residual(p, w):
        a = matrix(p)
        return (np.einsum('ij,j->i', a, project(a, w)) - y) * w

    short, long = np.median(dt), np.median(times)
    low, high = min(dt.min(), times[times > 0].min()) / 12, max(dt.sum(), times.max()) * 12
    bounds = np.log([low, high])
    starts = [(0.4*short, long), (short, long), (2*short, 2*long), (short, 4*long)]
    fits = [least_squares(residual, np.log(pair), args=(base_weight,), bounds=bounds,
                          max_nfev=70) for pair in starts]
    p = np.sort(min(fits, key=lambda fit: fit.cost).x)
    w = base_weight
    for _ in range(4):
        r = residual(p, np.ones_like(y))
        scale = np.empty_like(y)
        for j in np.flatnonzero(counts):
            ix = group == j
            q = r[ix]
            scale[ix] = max(1e-5, 1.4826 * np.median(np.abs(q - np.median(q))))
        w = base_weight / (scale * np.sqrt(1 + (r / (2.5 * scale))**2))**0.5
        # The two branches have distinct parameter roles; the fast one carries activation.
        p = least_squares(residual, p, args=(w,), bounds=bounds, max_nfev=45).x
    return np.exp(p), project(matrix(p), w)


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    a_train, a_test = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    args = calibration, dt, times, dataset['crit_stress'], dataset['shear_banding_index']
    training = _design(train, a_train, *args)
    taus, weights = _fit(training, train['strain_curves_obs'] - train['strain_0'][:, None],
                        train['instant_rate_obs'], dt, times)
    curves, rates = _design(test, a_test, *args)(np.log(taus))
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']
a=m._calibrate(d['calibration_load_cells']); print('calibration', a)
for key in ('train','test'):
 b=d[key]; g,c=a[b['lab_id']].T; hist=g[:,None]*b['past_stress_cmd']+c[:,None]; print(key,'positive damage rows by lab',np.bincount(b['lab_id'][np.any(abs(hist)>d['crit_stress'],axis=1)],minlength=4))
for lab in range(4):
 v=d['train']; keep=v['lab_id']!=lab
 x={**d,'train':{k:b[keep] for k,b in v.items()},'test':{k:b[~keep] for k,b in v.items()}}
 o=m.solve_alpha_m97_c03_batch_04(x)
 print('held out lab',lab,'taus',o['estimated_taus'].round(5),'curve rmse',np.mean((o['predicted_strain']-x['test']['strain_curves_obs'])**2)**.5,'rate rmse',np.mean((o['initial_rate_pred']-x['test']['instant_rate_obs'])**2)**.5)
print({k:v for k,v in m.solve_alpha_m97_c03_batch_04(d).items() if v.ndim==1 and len(v)<10})
PY
```
- **Container Exit Code**: `1` (`475 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=667fa59b014d39ac (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_667fa59b (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 9, in <module>
    out = mod.solve_alpha_m97_c03_batch_04(ds)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c03_batch_04_solver.py", line 200, in solve_alpha_m97_c03_batch_04
    training = _design(train, a_train, *args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c03_batch_04_solver.py", line 125, in _design
    gain, bias = calibration[np.asarray(sample['lab_id'])].T
                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: only integer scalar arrays can be converted to a scalar index
```

### Turn 5 — `execute_in_remote_docker` (`26768 ms` LLM)
- **Rationale**: Convert the reused calibration helper’s gain and bias vectors into the lab-indexed matrix the design expects. Rerun the isolated schema check, then measure held-out curve and rate errors separately by lab and damage state to identify whether either regime needs correction.

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_04_solver.py')
s = p.read_text(); old = "    calibration = _calibrate(dataset['calibration_load_cells'])"
assert s.count(old) == 1
p.write_text(s.replace(old, "    gains, biases = _calibrate(dataset['calibration_load_cells'])\n    calibration = np.column_stack((gains, biases))"))
PY
sed -n '105,127p' alpha_m97_c03_batch_04_solver.py
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']
a=np.column_stack(m._calibrate(d['calibration_load_cells'])); print('calibration', a)
def damage(b):
 g,c=a[b['lab_id']].T; hist=g[:,None]*b['past_stress_cmd']+c[:,None]
 return np.any(abs(hist)>d['crit_stress'],axis=1)
for key in ('train','test'):
 b=d[key]; print(key,'positive damage rows by lab',np.bincount(b['lab_id'][damage(b)],minlength=4))
for lab in range(4):
 v=d['train']; keep=v['lab_id']!=lab
 x={**d,'train':{k:b[keep] for k,b in v.items()},'test':{k:b[~keep] for k,b in v.items()}}
 o=m.solve_alpha_m97_c03_batch_04(x); b=x['test']; damaged=damage(b)
 for name,sel in [('all',np.ones(len(damaged),bool)),('plateau',damaged),('undamaged',~damaged)]:
  if sel.any():
   print('lab',lab,name,'n',sel.sum(),'taus',o['estimated_taus'].round(5),'curve rmse',np.mean((o['predicted_strain'][sel]-b['strain_curves_obs'][sel])**2)**.5,'rate rmse',np.mean((o['initial_rate_pred'][sel]-b['instant_rate_obs'][sel])**2)**.5)
o=m.solve_alpha_m97_c03_batch_04(d); print({k:o[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights')})
PY
```
- **Container Exit Code**: `0` (`1088 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
en their clean readings already agree with the archived line.
        if np.ptp(x[keep]) < 0.1 and np.max(np.abs(y[keep] - design[keep] @ prior)) <= 2 * noise:
            line = prior
        coefficients[lab] = line
    return coefficients[:, 0], coefficients[:, 1]


def _design(sample, aligned, calibration, dt, times, threshold, banding):
    gain, bias = calibration[np.asarray(sample['lab_id'])].T
    history = gain[:, None] * np.asarray(sample['past_stress_cmd']) + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
{"schema_valid": true, "errors": []}
calibration [[ 1.2899706   0.11999929]
 [ 0.81997725 -0.14996309]
 [ 1.16999803  0.14000889]
 [ 0.76000326 -0.11003605]]
train positive damage rows by lab [35 29 24 32]
test positive damage rows by lab [10 12 12 14]
lab 0 all n 48 taus [0.54607 3.4142 ] curve rmse 1.8821707691110243 rate rmse 0.11085688388215813
lab 0 plateau n 35 taus [0.54607 3.4142 ] curve rmse 2.1965200470641113 rate rmse 0.12030941173599838
lab 0 undamaged n 13 taus [0.54607 3.4142 ] curve rmse 0.3011127435413346 rate rmse 0.08003918678027495
lab 1 all n 43 taus [0.5415  3.42627] curve rmse 0.42959829387241505 rate rmse 0.09644607924895324
lab 1 plateau n 29 taus [0.5415  3.42627] curve rmse 0.48134507214504635 rate rmse 0.1043597284063048
lab 1 undamaged n 14 taus [0.5415  3.42627] curve rmse 0.2948065431463474 rate rmse 0.07752499699599552
lab 2 all n 42 taus [0.53112 3.17674] curve rmse 1.141300005673958 rate rmse 0.08419209261309506
lab 2 plateau n 24 taus [0.53112 3.17674] curve rmse 1.4670113906012467 rate rmse 0.0820257068948812
lab 2 undamaged n 18 taus [0.53112 3.17674] curve rmse 0.4120963641530223 rate rmse 0.08699672952393858
lab 3 all n 47 taus [0.53334 3.26497] curve rmse 0.7566000967305978 rate rmse 0.09320354456833957
lab 3 plateau n 32 taus [0.53334 3.26497] curve rmse 0.5845800291610811 rate rmse 0.09368373823341852
lab 3 undamaged n 15 taus [0.53334 3.26497] curve rmse 1.0318065795394082 rate rmse 0.0921707704732217
{'estimated_taus': array([0.5384536 , 3.32905142]), 'hereditary_weights': array([ 0.67907466,  0.5219226 ,  0.09047916,  0.11874342, -0.09351553,
        0.08235472, -0.06740764]), 'regime_mode_weights': array([0.264909  , 0.20557892])}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3a7aceab0b6ab74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3a7aceab (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29308 ms` LLM)
- **Rationale**: Inspect out-of-fold residuals across horizons and damage states. If large errors are isolated readings, compare stronger robust fitting; if they persist along individual curves or horizons, revisit the design or time indexing.
- **Alternatives Weighed**: `Change the plateau equation based on raw RMSE` (rejected: Isolated measurement spikes and persistent curve mismatch imply different fixes; raw RMSE alone does not distinguish them.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
v=d['train']; t=d['eval_times']; a=np.column_stack(m._calibrate(d['calibration_load_cells'])); g,b=a[v['lab_id']].T; h=g[:,None]*v['past_stress_cmd']+b[:,None]; dd=np.maximum(abs(h)-d['crit_stress'],0)**2@d['past_dt']; print('times',t,'first observed column matches baseline fraction', np.mean(v['strain_curves_obs'][:,0]==v['strain_0'])); print('damage positive quantiles',np.quantile(dd[dd>0],[0,.1,.5,.9,1]));
E=np.empty_like(v['strain_curves_obs']); R=np.empty_like(v['instant_rate_obs']);
for lab in range(4):
 k=v['lab_id']!=lab; data={**d,'train':{n:x[k] for n,x in v.items()},'test':{n:x[~k] for n,x in v.items()}}; o=m.solve_alpha_m97_c03_batch_04(data); E[~k]=o['predicted_strain']-v['strain_curves_obs'][~k]; R[~k]=o['initial_rate_pred']-v['instant_rate_obs'][~k]
for lab in range(4):
 for plateau in (0,1):
  k=(v['lab_id']==lab)&((dd>0)==plateau); print('lab,plateau', lab,plateau,'curve abs q50/q90/q99/max',np.round(np.quantile(abs(E[k]),[.5,.9,.99,1]),4),'rate abs q50/q90/q99/max',np.round(np.quantile(abs(R[k]),[.5,.9,.99,1]),4),'how many curve/rate > .3', (abs(E[k])>.3).sum(),(abs(R[k])>.3).sum())
print('horizon med abs',np.round(np.median(abs(E),axis=0),5),'horizon q90',np.round(np.quantile(abs(E),.90,axis=0),5)); print('horizon median signed',np.round(np.median(E,axis=0),5)); print('largest curve residuals (row, horizon, residual, neighboring residuals)');
for i,j in zip(*np.unravel_index(np.argsort(abs(E).ravel())[-10:][::-1],E.shape)):
 print(i,j,round(E[i,j],4),np.round(E[i],3))
PY
```
- **Container Exit Code**: `0` (`713 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.3372 0.5985 2.7114] rate abs q50/q90/q99/max [0.0682 0.1074 0.1318 0.1346] how many curve/rate > .3 23 0
lab,plateau 1 1 curve abs q50/q90/q99/max [0.1034 0.2511 1.669  6.5169] rate abs q50/q90/q99/max [0.0898 0.1409 0.2153 0.224 ] how many curve/rate > .3 22 0
lab,plateau 2 0 curve abs q50/q90/q99/max [0.0879 0.5426 1.6929 2.6805] rate abs q50/q90/q99/max [0.0449 0.1263 0.1908 0.2016] how many curve/rate > .3 40 0
lab,plateau 2 1 curve abs q50/q90/q99/max [ 0.0678  0.1852  1.8698 22.1967] rate abs q50/q90/q99/max [0.0623 0.1232 0.1754 0.179 ] how many curve/rate > .3 15 0
lab,plateau 3 0 curve abs q50/q90/q99/max [ 0.1682  0.6993  0.9515 11.7997] rate abs q50/q90/q99/max [0.0923 0.13   0.1541 0.1572] how many curve/rate > .3 53 0
lab,plateau 3 1 curve abs q50/q90/q99/max [0.0566 0.2704 3.3033 6.4648] rate abs q50/q90/q99/max [0.0596 0.1766 0.2284 0.2369] how many curve/rate > .3 27 0
horizon med abs [0.01985 0.03044 0.0502  0.07922 0.10005 0.12648 0.14887 0.1619  0.17236
 0.19782] horizon q90 [0.05221 0.13333 0.1502  0.22333 0.25259 0.3291  0.42192 0.47824 0.61421
 0.70854]
horizon median signed [-0.012   -0.01462 -0.01676 -0.01391 -0.02216 -0.02293 -0.02323 -0.02275
 -0.03243 -0.04545]
largest curve residuals (row, horizon, residual, neighboring residuals)
54 5 39.1754 [ 1.1000e-02 -2.1000e-02 -1.6000e-02 -1.3000e-02 -3.0000e-02  3.9175e+01
  4.8520e+00  5.0000e-03  8.0000e-03 -1.8000e-02]
118 9 -22.1967 [ -0.037  -0.122  -0.169  -3.522  -0.223  -0.227  -0.256  -0.284  -0.324
 -22.197]
168 8 11.7997 [-0.    -0.036  0.055  0.14   0.167  0.303  0.367  0.46  11.8    0.648]
39 5 -7.2576 [ 0.037  1.97   0.148  0.151  0.187 -7.258  0.234  0.248  0.269  0.273]
37 6 -6.5169 [-0.023 -0.091 -0.108 -0.123 -0.1   -0.142 -6.517 -0.172 -0.198 -0.174]
75 0 -6.4648 [-6.465  0.021  0.05   0.052  0.033  0.06   0.049  1.401  0.101  0.116]
16 0 6.4199 [ 6.42  -0.117 -0.166 -0.219 -0.662 -0.329 -0.315 -0.318 -0.38  -0.4  ]
54 6 4.8518 [ 1.1000e-02 -2.1000e-02 -1.6000e-02 -1.3000e-02 -3.0000e-02  3.9175e+01
  4.8520e+00  5.0000e-03  8.0000e-03 -1.8000e-02]
96 9 -4.514 [-0.019 -0.032 -0.044 -0.073 -0.078 -0.103 -0.183 -0.143 -0.167 -4.514]
89 9 -4.2145 [ 0.045  0.126  0.167  0.472  0.273  0.299 -4.102  0.324  0.045 -4.215]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3a7aceab0b6ab74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3a7aceab (directory).
```

### Turn 7 — `execute_in_remote_docker` (`35332 ms` LLM)
- **Rationale**: Separate isolated strain spikes from systematic late-time error. Compare the two flow limits only as diagnostic controls, with fitting inside each held-out-lab fold. If late error is concentrated in rows with missing coupon features, prioritize checking imputation rather than changing the plateau law.
- **Alternatives Weighed**: `Use raw RMSE alone to select a model` (rejected: A single strain observation misses by 39.18 while most other observations on that curve miss by less than 0.04.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('m','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('public_alpha_m97_c03_batch_04.pkl','rb'))[0]; d=item['dataset']; v=d['train']; t=d['eval_times']; dt=d['past_dt']; n=len(v['lab_id']); a=np.column_stack(m._calibrate(d['calibration_load_cells'])); g,b=a[v['lab_id']].T; h=g[:,None]*v['past_stress_cmd']+b[:,None]; damage=np.maximum(abs(h)-d['crit_stress'],0)**2@dt; pos=damage>0; missing=~np.isfinite(v['coupon_features_raw']).all(axis=1); y=v['strain_curves_obs']; r=v['instant_rate_obs']; print('public entries',list(item),'calibration rows',d['calibration_load_cells']); print('missing coupon counts zero/positive damage',[(int((k&missing).sum()),int(k.sum())) for k in (~pos,pos)])
for mode in ('specified','all_log','all_linear'):
    E=np.empty_like(y); R=np.empty_like(r)
    for lab in range(4):
        k=v['lab_id']!=lab; f=[{j:w[q] for j,w in v.items()} for q in (k,~k)]; aligned=m._align(*f)
        def wrap(sample,x,act):
            ev=m._design(sample,x,a,dt,t,d['crit_stress'],d['shear_banding_index'])
            def evaluate(p):
                c,rr=ev(p)
                if mode!='specified':
                    tail=np.log1p(t) if mode=='all_log' else t
                    c[:,:,2:7]=rr[:,None,2:7]*tail[None,:,None]
                return c,rr
            return evaluate
        ta,weights=m._fit(wrap(f[0],aligned[0],pos[k]),y[k]-v['strain_0'][k,None],r[k],dt,t)
        c,rr=wrap(f[1],aligned[1],pos[~k])(np.log(ta)); E[~k]=v['strain_0'][~k,None]+c@weights-y[~k]; R[~k]=rr@weights-r[~k]
    groups=[(v['lab_id']==lab)&(pos==p)&(missing==q) for lab in range(4) for p in (0,1) for q in (0,1)]
    print(mode,'worst lab/damage/availability (min n=5) curve median, clipped(.5) RMSE, terminal abs med, rate clipped(.3) RMSE:',np.round(np.max([[np.median(abs(E[k])), np.mean(np.minimum(E[k]**2,.25))**.5, np.median(abs(E[k,-1])), np.mean(np.minimum(R[k]**2,.09))**.5] for k in groups if k.sum()>=5],axis=0),5))
    for k in (~pos&~missing,pos&~missing,~pos&missing,pos&missing):
        print(' n',k.sum(),'curve median',round(np.median(abs(E[k])),5),'signed late',np.round(np.median(E[k,-3:],axis=0),5),'rate median',round(np.median(abs(R[k])),5))
PY
```
- **Container Exit Code**: `0` (`1335 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public entries ['dataset', 'gt'] calibration rows {'lab_id': array([0, 0]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-1.55696249,  1.92595813])}
missing coupon counts zero/positive damage [(7, 60), (17, 120)]
specified worst lab/damage/availability (min n=5) curve median, clipped(.5) RMSE, terminal abs med, rate clipped(.3) RMSE: [0.21831 0.30433 0.67558 0.18226]
 n 53 curve median 0.11237 signed late [0.21786 0.26392 0.31263] rate median 0.06242
 n 103 curve median 0.07431 signed late [-0.08059 -0.07794 -0.09704] rate median 0.06427
 n 7 curve median 0.09633 signed late [0.16745 0.22427 0.30328] rate median 0.11308
 n 17 curve median 0.16772 signed late [ 0.01522 -0.14988 -0.15681] rate median 0.14114
all_log worst lab/damage/availability (min n=5) curve median, clipped(.5) RMSE, terminal abs med, rate clipped(.3) RMSE: [0.18329 0.24786 0.29765 0.16567]
 n 53 curve median 0.01419 signed late [0.00476 0.00046 0.00036] rate median 0.01408
 n 103 curve median 0.01506 signed late [-0.00157 -0.00064 -0.00644] rate median 0.01364
 n 7 curve median 0.05861 signed late [-0.04826 -0.05589 -0.12375] rate median 0.03898
 n 17 curve median 0.14734 signed late [ 0.05349 -0.02666  0.04248] rate median 0.11955
all_linear worst lab/damage/availability (min n=5) curve median, clipped(.5) RMSE, terminal abs med, rate clipped(.3) RMSE: [0.15263 0.24847 0.23378 0.21188]
 n 53 curve median 0.04561 signed late [0.06939 0.09784 0.13448] rate median 0.09989
 n 103 curve median 0.04436 signed late [0.03269 0.05183 0.06522] rate median 0.11764
 n 7 curve median 0.08269 signed late [0.00725 0.02561 0.02762] rate median 0.15933
 n 17 curve median 0.13594 signed late [0.0906  0.05318 0.0669 ] rate median 0.17692

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3a7aceab0b6ab74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3a7aceab (directory).
```

### Turn 8 — `execute_in_remote_docker` (`36633 ms` LLM)
- **Rationale**: Promote the tail change supported by held-out data without changing the damaged-plateau formula. Verify the exact artifact in an isolated process, check the analytic initial rate and both gate boundaries, and inspect public reference fields for a direct accuracy check without using them in fitting.
- **Alternatives Weighed**: `Use linear flow for all rows` (rejected: It violates the specified damaged response and increases the worst clipped initial-rate RMSE to 0.21188.)
- **ATMS NOGOOD DECLARED**: Refuted `Undamaged specimens in the low-banding batch require a linear base tail.` | Witness: `On held-out labs, complete-coupon undamaged rows had a +0.31263 median terminal residual under the switched tail versus +0.00036 under the shared log tail; median absolute curve error fell from 0.11237 to 0.01419.` | Blanket Diagnosis: `The damage indicator was applied to both the base time basis and the two activation terms, creating a late-time bias even in complete-coupon rows.` | Sublated Pivot: `Keep the logarithmic base tail throughout the low-banding phase, and gate the two existing activation terms by positive damage without changing the specified damaged response.`

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_04_solver.py')
s = p.read_text()
old = '    tail = np.where(plateau[:, None], np.log1p(times), times)'
new = '    # In the low-banding phase, damage gates the two extra terms, not the base tail.\n    tail = np.log1p(times) if banding <= 0.50 else times'
assert s.count(old) == 1
s = s.replace(old, new).replace('flow[:, None, :] * tail[:, :, None]', 'flow[:, None, :] * tail[None, :, None]')
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: items=pickle.load(f)
print('public episodes',len(items))
for item in items:
 d=item['dataset']; gt=item.get('gt'); print('public reference fields', {k:np.shape(v) for k,v in gt.items()} if isinstance(gt,dict) else type(gt).__name__)
 out=m.solve_alpha_m97_c03_batch_04(d)
 for key in ('estimated_taus','hereditary_weights','regime_mode_weights'):
  print(key,np.round(out[key],6))
  if isinstance(gt,dict) and key in gt and np.shape(gt[key])==out[key].shape:
   print('public max abs error',float(np.max(abs(out[key]-gt[key]))))
 for key in ('predicted_strain','initial_rate_pred'):
  if isinstance(gt,dict) and key in gt and np.shape(gt[key])==out[key].shape:
   print('public',key,'RMSE',float(np.sqrt(np.mean((out[key]-gt[key])**2))))
 a=np.column_stack(m._calibrate(d['calibration_load_cells'])); _,x=m._align(d['train'],d['test']); b=d['test']; g,c=a[b['lab_id']].T; hist=g[:,None]*b['past_stress_cmd']+c[:,None]; dt=d['past_dt']; crit=d['crit_stress']; damaged=np.any(abs(hist)>crit,axis=1); assert damaged.any() and (~damaged).any()
 for banding in (0.50,0.500001):
  curve, rate=m._design(b,x,a,dt,np.array([0.,1e-7]),crit,banding)(np.log(out['estimated_taus']))
  np.testing.assert_array_equal(curve[:,0],0.)
  np.testing.assert_allclose(curve[:,1]/1e-7,rate,rtol=2e-6,atol=2e-7)
  assert np.all(curve[~damaged,:,7:]==0) and np.all(rate[~damaged,7:]==0)
  assert bool(np.any(rate[damaged,7:]!=0)) == (banding<=.50)
 print('zero-time derivative and both sides of damage/banding gate passed')
PY
```
- **Container Exit Code**: `0` (`1031 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public episodes 1
public reference fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
estimated_taus [0.538037 3.803978]
hereditary_weights [ 0.684112  0.560017  0.137367  0.160786 -0.127656  0.122557 -0.087865]
regime_mode_weights [0.264738 0.212075]
zero-time derivative and both sides of damage/banding gate passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4f959ab579f55a4b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4f959ab5 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`23991 ms` LLM)
- **Rationale**: The schema and analytic derivative checks passed, but the previous script did not compare accuracy because the public reference fields use different names. Compare the unchanged candidate against those fields after fitting: stress errors would point to calibration, while errors isolated to missing coupons would point to imputation rather than the plateau law.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: item=pickle.load(f)[0]
d,gt=item['dataset'],item['gt']; out=m.solve_alpha_m97_c03_batch_04(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 e=out[k]-gt[j]; print(k,'max abs error',np.max(abs(e)))
b=d['test']; g,c=np.column_stack(m._calibrate(d['calibration_load_cells']))[b['lab_id']].T; hist=g[:,None]*b['past_stress_cmd']+c[:,None]; hold=g*b['stress_hold_cmd']+c
print('calibrated stress max abs errors: history, hold',np.max(abs(hist-gt['te_hist_true'])),np.max(abs(hold-gt['te_hold_true'])))
pos=np.any(abs(hist)>d['crit_stress'],axis=1); missing=~np.isfinite(b['coupon_features_raw']).all(axis=1)
e=out['predicted_strain']-gt['te_clean']; r=out['initial_rate_pred']-gt['te_true_rate0']
for name,k in [('all',np.ones(len(pos),bool))]+[(f'lab {i} damage {p}',(b['lab_id']==i)&(pos==p)) for i in range(4) for p in (False,True)]+[('complete coupons',~missing),('missing coupons',missing)]:
 if k.any(): print(name,'n',k.sum(),'strain RMSE',np.mean(e[k]**2)**.5,'terminal RMSE',np.mean(e[k,-1]**2)**.5,'initial-rate RMSE',np.mean(r[k]**2)**.5)
PY
```
- **Container Exit Code**: `0` (`574 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus max abs error 0.21297821275001194
hereditary_weights max abs error 0.01001683622554983
regime_mode_weights max abs error 0.007924783611099856
calibrated stress max abs errors: history, hold 0.00014127481208925374 7.707487746499986e-05
all n 72 strain RMSE 0.048380119199149474 terminal RMSE 0.06751470758329552 initial-rate RMSE 0.037001883979209345
lab 0 damage False n 8 strain RMSE 0.024546816059245805 terminal RMSE 0.034353554211655125 initial-rate RMSE 0.019470235145674652
lab 0 damage True n 10 strain RMSE 0.09287697977371163 terminal RMSE 0.1279097487164892 initial-rate RMSE 0.07145851587601512
lab 1 damage False n 5 strain RMSE 0.006622384692225431 terminal RMSE 0.009469198074363517 initial-rate RMSE 0.006320737717917295
lab 1 damage True n 12 strain RMSE 0.010392071405295132 terminal RMSE 0.015481888370631821 initial-rate RMSE 0.007735224590359725
lab 2 damage False n 4 strain RMSE 0.008090332759695609 terminal RMSE 0.01082682264758397 initial-rate RMSE 0.006383075272640899
lab 2 damage True n 12 strain RMSE 0.014472029013803785 terminal RMSE 0.022039580797083676 initial-rate RMSE 0.009978745009008878
lab 3 damage False n 7 strain RMSE 0.010523688425047816 terminal RMSE 0.01390462668566493 initial-rate RMSE 0.009750687973555537
lab 3 damage True n 14 strain RMSE 0.07190198529917263 terminal RMSE 0.1014772452791218 initial-rate RMSE 0.054472109165263446
complete coupons n 66 strain RMSE 0.010706645600023447 terminal RMSE 0.015868720767190054 initial-rate RMSE 0.00830329731854534
missing coupons n 6 strain RMSE 0.16378851102759748 terminal RMSE 0.2278790240325715 initial-rate RMSE 0.1251849863448834

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4f959ab579f55a4b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4f959ab5 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`29983 ms` LLM)
- **Rationale**: Inspect the imputation implementation and test whether incomplete rows degrade the fitted spectrum, without changing the artifact or fitting on public reference labels. Change training weights only if the held-out results show a clear benefit without worsening the missing-coupon slice.
- **Alternatives Weighed**: `Replace the specified cell-mean imputation` (rejected: Missing-coupon errors alone do not show the required imputation was implemented incorrectly or justify a different prediction rule.)

```bash
set -e
cd /holdout_task
sed -n '1,48p' alpha_m97_c03_batch_04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
v=d['train']; dt=d['past_dt']; t=d['eval_times']; a=np.column_stack(m._calibrate(d['calibration_load_cells'])); g,b=a[v['lab_id']].T; pos=np.any(abs(g[:,None]*v['past_stress_cmd']+b[:,None])>d['crit_stress'],axis=1); cc=np.isfinite(v['coupon_features_raw']).all(axis=1); y=v['strain_curves_obs']; r=v['instant_rate_obs']; args=(a,dt,t,d['crit_stress'],d['shear_banding_index'])
def sub(q): return {k:z[q] for k,z in v.items()}
for fraction in (1.,.25,0.):
 E=np.empty_like(y); R=np.empty_like(r); taus=[]
 for lab in range(4):
  k=v['lab_id']!=lab; tr,te=sub(k),sub(~k); x,xx=m._align(tr,te); yy=y[k]-tr['strain_0'][:,None]; rr=r[k].copy(); missing=~cc[k]
  if fraction==0:
   yy[missing]=np.nan; rr[missing]=np.nan
  ev=m._design(tr,x,*args)
  if fraction==.25:
   # Drop three quarters of imputed-label rows deterministically for this diagnostic only.
   drop=np.flatnonzero(missing); drop=drop[np.arange(len(drop))%4!=0]; yy[drop]=np.nan; rr[drop]=np.nan
  tau,w=m._fit(ev,yy,rr,dt,t); taus.append(tau)
  c,h=m._design(te,xx,*args)(np.log(tau)); E[~k]=te['strain_0'][:,None]+c@w-y[~k]; R[~k]=h@w-r[~k]
 def metrics(q): return np.array([np.sqrt(np.mean(np.minimum(E[q]**2,.25))),np.sqrt(np.mean(np.minimum(R[q]**2,.09))),np.median(abs(E[q,-1]))])
 print('fraction',fraction,'LOLO tau fits',np.round(taus,4).tolist())
 for mask in (cc,~cc):
  slices=[mask&(v['lab_id']==l)&(pos==j) for l in range(4) for j in (0,1)]; print('overall and worst lab/damage (n>=5)',metrics(mask).round(5),np.max([metrics(q) for q in slices if q.sum()>=5],axis=0).round(5))
PY
```
- **Container Exit Code**: `0` (`1275 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    coefficients = np.empty((4, 2))
    history = None
    # Measured clean sweeps have residuals below 2e-4 stress units.
    noise = 2e-4

    def archived_line(lab):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
fraction 1.0 LOLO tau fits [[0.5401, 3.6631], [0.5375, 3.7877], [0.5361, 3.7414], [0.536, 3.903]]
overall and worst lab/damage (n>=5) [0.09982 0.0219  0.01898] [0.12129 0.02905 0.02876]
overall and worst lab/damage (n>=5) [0.2131  0.13884 0.17795] [0.24786 0.16567 0.29765]
fraction 0.25 LOLO tau fits [[0.5379, 3.6197], [0.5399, 3.7323], [0.5403, 3.7269], [0.5393, 3.8879]]
overall and worst lab/damage (n>=5) [0.09957 0.02126 0.01618] [0.12105 0.02774 0.02041]
overall and worst lab/damage (n>=5) [0.212   0.13711 0.16597] [0.24496 0.16106 0.2957 ]
fraction 0.0 LOLO tau fits [[0.5381, 3.6251], [0.5401, 3.7432], [0.539, 3.6393], [0.5389, 3.7301]]
overall and worst lab/damage (n>=5) [0.09944 0.02076 0.01396] [0.12066 0.02611 0.01849]
overall and worst lab/damage (n>=5) [0.21129 0.13679 0.16247] [0.2437  0.15971 0.29557]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4f959ab579f55a4b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4f959ab5 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`34795 ms` LLM)
- **Rationale**: Promote the training-only change supported by the held-out folds without altering the required imputation or plateau law. Check the exact edited artifact in an isolated process for schema, parameter recovery, and damage-slice errors; retain the verified prior version if the edit causes a material regression.
- **Alternatives Weighed**: `Keep all incomplete rows in the fit` (rejected: Complete-case fitting reduced held-out errors in both availability groups and narrowed the spread of the slow relaxation time across folds.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_04_solver.py')
s = p.read_text()
old = "    training = _design(train, a_train, *args)\n    taus, weights = _fit(training, train['strain_curves_obs'] - train['strain_0'][:, None],\n                        train['instant_rate_obs'], dt, times)"
new = "    training = _design(train, a_train, *args)\n    # Imputed coupons support prediction, but their unknown covariates can bias the fit.\n    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)\n    def observed_design(log_tau):\n        curves, rates = training(log_tau)\n        return curves[complete], rates[complete]\n\n    taus, weights = _fit(observed_design,\n                         train['strain_curves_obs'][complete] - train['strain_0'][complete, None],\n                         train['instant_rate_obs'][complete], dt, times)"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl','rb') as f: item=pickle.load(f)[0]
d,gt=item['dataset'],item['gt']; out=m.solve_alpha_m97_c03_batch_04(d); b=d['test']
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(k,'max abs error',float(np.max(np.abs(out[k]-gt[j]))))
a=np.column_stack(m._calibrate(d['calibration_load_cells'])); g,c=a[b['lab_id']].T; damage=np.any(np.abs(g[:,None]*b['past_stress_cmd']+c[:,None])>d['crit_stress'],axis=1); missing=~np.isfinite(b['coupon_features_raw']).all(axis=1)
e=out['predicted_strain']-gt['te_clean']; r=out['initial_rate_pred']-gt['te_true_rate0']
for name,k in [('all',np.ones(len(damage),bool)),('complete',~missing),('missing',missing)]+[(f'lab {i} damage {j}',(b['lab_id']==i)&(damage==j)) for i in range(4) for j in (False,True)]:
 print(name,int(k.sum()),'strain/terminal/rate RMSE',np.round([np.mean(e[k]**2)**.5,np.mean(e[k,-1]**2)**.5,np.mean(r[k]**2)**.5],6))
_,x=m._align(d['train'],b)
for banding in (0.50,0.500001):
 cv,rt=m._design(b,x,a,d['past_dt'],np.array([0.,1e-7]),d['crit_stress'],banding)(np.log(out['estimated_taus']))
 np.testing.assert_array_equal(cv[:,0],0.)
 np.testing.assert_allclose(cv[:,1]/1e-7,rt,rtol=2e-6,atol=2e-7)
 assert np.all(rt[~damage,7:]==0) and bool(np.any(rt[damage,7:]!=0))==(banding<=.50)
print('Zero-time derivative and activation gates passed.')
PY
```
- **Container Exit Code**: `0` (`918 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus max abs error 0.0808859570798317
hereditary_weights max abs error 0.006057979028392224
regime_mode_weights max abs error 0.0013466014115580127
all 72 strain/terminal/rate RMSE [0.048146 0.067053 0.036839]
complete 66 strain/terminal/rate RMSE [0.005647 0.007351 0.004397]
missing 6 strain/terminal/rate RMSE [0.165728 0.230995 0.126777]
lab 0 damage False 8 strain/terminal/rate RMSE [0.023949 0.033294 0.019008]
lab 0 damage True 10 strain/terminal/rate RMSE [0.092286 0.128322 0.071011]
lab 1 damage False 5 strain/terminal/rate RMSE [0.005126 0.006972 0.004229]
lab 1 damage True 12 strain/terminal/rate RMSE [0.008598 0.011515 0.007217]
lab 2 damage False 4 strain/terminal/rate RMSE [0.003314 0.003609 0.003086]
lab 2 damage True 12 strain/terminal/rate RMSE [0.004446 0.005228 0.003074]
lab 3 damage False 7 strain/terminal/rate RMSE [0.005022 0.006633 0.004111]
lab 3 damage True 14 strain/terminal/rate RMSE [0.073518 0.102701 0.055684]
Zero-time derivative and activation gates passed.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9816ded675991bb0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_9816ded6 (directory).
```

### Turn 12 — `submit_completed_task` (`15464 ms` LLM)
- **Rationale**: Complete-case fitting improved held-out performance and public parameter recovery while preserving the specified prediction-time imputation. The final artifact passed isolated schema and analytic boundary checks, and its verification receipt is bound.
- **Alternatives Weighed**: `Retain the all-row fit` (rejected: It had higher worst-slice held-out errors and a larger public relaxation-time error (0.21298 versus 0.08089), despite slightly lower RMSE on the six public specimens with missing coupon features.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_04_solver.py': 'FILE:10267'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema, zero-time derivative, and activation-boundary checks passed. Public strain and initial-rate RMSE were 0.04815 and 0.03684 overall, and 0.16573 and 0.12678 on the six specimens with missing coupon features.`

### Turn 13 — `execute_in_remote_docker` (`39420 ms` LLM)
- **Rationale**: Use the real public measurements to compare the solver with an independently computed memory and curve/rate handoff. Compare signed errors within labs and use public reference parameters only as a diagnostic: if they leave the missing-coupon error essentially unchanged and that error follows the logarithmic drift, improving fitted parameters or adding response terms will not address the bottleneck.
- **Alternatives Weighed**: `Add rational drift, PDE operators, or relay modes` (rejected: The task specifies a continuous two-branch constitutive response, with no PDE state, relay input, or classification target. An independent closed-form check and within-lab residuals can detect failures in the actual response.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
cat > /tmp/audit_batch04.py <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with (root/'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
a, b = d['train'], d['test']
out = m.solve_alpha_m97_c03_batch_04(d)
labs = b['lab_id']
complete = np.isfinite(b['coupon_features_raw']).all(axis=1)
cal = np.column_stack(m._calibrate(d['calibration_load_cells']))
g, c = cal[labs].T
hist = g[:, None]*b['past_stress_cmd'] + c[:, None]
hold = g*b['stress_hold_cmd'] + c
dt, t = d['past_dt'], d['eval_times']
damage = (np.maximum(0., np.abs(hist)-d['crit_stress'])**2) @ dt
act = np.tanh(damage)
_, x = m._align(a,b)

def direct(tau, w, rw):
    z = hist @ ((-np.expm1(-dt[:,None]/tau))*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau))
    gap = hold[:,None]-z
    coeff = w[2]+x@w[3:]+rw[1]*act*z[:,0]
    amp = gap*w[:2]
    amp[:,0] += rw[0]*act*hold
    return b['strain_0'][:,None] + amp @ (-np.expm1(-t[None,:]/tau[:,None])) + coeff[:,None]*np.log1p(t), amp @ (1/tau) + coeff

cv, rt = direct(out['estimated_taus'],out['hereditary_weights'],out['regime_mode_weights'])
np.testing.assert_allclose(cv,out['predicted_strain'],atol=1e-11,rtol=1e-11)
np.testing.assert_allclose(rt,out['initial_rate_pred'],atol=1e-11,rtol=1e-11)
print('Independent closed-form memory and curve/rate handoff agree.')
print('Stress reconstruction max abs, history/hold:',np.max(abs(hist-gt['te_hist_true'])),np.max(abs(hold-gt['te_hold_true'])))
print('Max relative tau error:', np.max(abs(out['estimated_taus']/gt['taus']-1)))
comparison = [('fitted',cv,rt),('public reference parameters with measured inputs',*direct(gt['taus'],gt['theta'],gt['vp_theta']))]
for label, yy, rr in comparison:
    e, r = yy-gt['te_clean'], rr-gt['te_true_rate0']
    print(label)
    for name, q in [('complete',complete),('missing',~complete)]+[(f'complete lab {i} damage {p}',complete&(labs==i)&((damage>0)==p)) for i in range(4) for p in (False,True)]:
        if q.any():
            print(name,int(q.sum()),'strain/rate RMSE',np.round([np.sqrt(np.mean(e[q]**2)),np.sqrt(np.mean(r[q]**2))],6),'signed terminal/rate',np.round([np.mean(e[q,-1]),np.mean(r[q])],6))
    print('missing residual unexplained by tail alone, RMS:', np.sqrt(np.mean((e[~complete]-r[~complete,None]*np.log1p(t))**2)))
PY
python3 -I /tmp/audit_batch04.py
python3 - <<'PY'
import json
with open('public_asset_index.json') as f:
    def scan(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if any(s in k.lower() for s in ('toler','threshold','metric')): print(k, str(v)[:300])
                elif isinstance(v,(dict,list)): scan(v)
        elif isinstance(x,list):
            for v in x: scan(v)
    scan(json.load(f))
PY
```
- **Container Exit Code**: `0` (`799 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
signed terminal/rate [0.000146 0.000422]
missing 6 strain/rate RMSE [0.165728 0.126777] signed terminal/rate [-0.000538  0.000773]
complete lab 0 damage False 7 strain/rate RMSE [0.003984 0.003391] signed terminal/rate [-0.004196 -0.002746]
complete lab 0 damage True 8 strain/rate RMSE [0.004211 0.003008] signed terminal/rate [-0.002639 -0.000951]
complete lab 1 damage False 4 strain/rate RMSE [0.003716 0.003555] signed terminal/rate [0.002991 0.000152]
complete lab 1 damage True 11 strain/rate RMSE [0.008742 0.0072  ] signed terminal/rate [0.007575 0.004928]
complete lab 2 damage False 4 strain/rate RMSE [0.003314 0.003086] signed terminal/rate [0.003469 0.002824]
complete lab 2 damage True 12 strain/rate RMSE [0.004446 0.003074] signed terminal/rate [5.60e-05 6.53e-04]
complete lab 3 damage False 7 strain/rate RMSE [0.005022 0.004111] signed terminal/rate [-0.001924 -0.001494]
complete lab 3 damage True 13 strain/rate RMSE [0.006191 0.004228] signed terminal/rate [-0.002786 -0.000676]
missing residual unexplained by tail alone, RMS: 0.0022171629373874064
public reference parameters with measured inputs
complete 66 strain/rate RMSE [0.004027 0.003079] signed terminal/rate [0.000453 0.00025 ]
missing 6 strain/rate RMSE [0.166806 0.127386] signed terminal/rate [0.002509 0.001398]
complete lab 0 damage False 7 strain/rate RMSE [0.001825 0.001381] signed terminal/rate [-0.002495 -0.001353]
complete lab 0 damage True 8 strain/rate RMSE [0.001732 0.001324] signed terminal/rate [-0.002382 -0.001307]
complete lab 1 damage False 4 strain/rate RMSE [0.003208 0.002451] signed terminal/rate [0.002097 0.001139]
complete lab 1 damage True 11 strain/rate RMSE [0.00796  0.006089] signed terminal/rate [0.00681  0.003748]
complete lab 2 damage False 4 strain/rate RMSE [0.00212  0.001617] signed terminal/rate [0.002527 0.001383]
complete lab 2 damage True 12 strain/rate RMSE [0.001155 0.000881] signed terminal/rate [0.00029  0.000162]
complete lab 3 damage False 7 strain/rate RMSE [0.003165 0.002418] signed terminal/rate [-0.001459 -0.000796]
complete lab 3 damage True 13 strain/rate RMSE [0.003728 0.002853] signed terminal/rate [-0.001558 -0.000865]
missing residual unexplained by tail alone, RMS: 3.9158465713834935e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9816ded675991bb0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_9816ded6 (directory).
```

### Turn 14 — `submit_completed_task` (`17400 ms` LLM)
- **Rationale**: Complete-case fitting improved held-out errors and public parameter recovery. The isolated audit confirmed the independent constitutive calculation and found a maximum history-stress reconstruction error of 0.0001413. The final artifact has a bound verification receipt, and the audit found no evidence for a further change.
- **Alternatives Weighed**: `Refit the spectrum or add response terms to reduce missing-coupon errors` (rejected: Public reference parameters did not improve that slice, and its residuals follow the logarithmic drift term. The prescribed cell-mean imputation remains in place.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema, independent curve and initial-rate calculations, and activation boundary checks passed. Public strain and initial-rate RMSE were 0.04815 and 0.03684 overall.

Limitation: On the six specimens with missing coupon features, RMSE was 0.16573 for strain and 0.12678 for initial rate. Using the public reference parameters did not reduce these errors.`

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
