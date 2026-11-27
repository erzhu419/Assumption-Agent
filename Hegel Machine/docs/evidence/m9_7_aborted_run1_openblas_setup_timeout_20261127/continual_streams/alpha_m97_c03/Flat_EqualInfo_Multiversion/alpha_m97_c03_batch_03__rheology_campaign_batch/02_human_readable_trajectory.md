# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `347,468` (`334,799` / `94,464` / `12,669`) |
| **Total Episode Wall Time (s)** | `471.58s` |

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

### Turn 1 — `execute_in_remote_docker` (`19479 ms` LLM)
- **Rationale**: Inspect the public schema and compact data summaries to identify missing or contaminated calibration channels and determine which observations can support validation. Those findings decide when to use prefix calibration and how to fit and validate the model.
- **Alternatives Weighed**: `Reuse the prior stress gains directly` (rejected: The public gate rejected that proposal, and the contract gives eligible in-batch rows priority.)

```bash
cd /holdout_task && ls -lah && cat validate_public_schema.py && cat public_asset_index.json; echo '=== archive ==='; ls -lah /tmp/hegel_continual_archive; echo '=== primitive search ==='; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel lab stratum imputation viscoelastic Prony relaxation'; echo '=== dataset structure ==='; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb') as f: d=pickle.load(f)
def show(x, prefix=''):
 for k,v in x.items():
  if isinstance(v,dict):
   print(prefix,k,'dict',list(v)[:30]);
   if len(prefix)<4: show(v,prefix+'  ')
  elif hasattr(v,'shape'):
   a=np.asarray(v); print(prefix,k,a.shape,a.dtype,repr(a.ravel()[:8]))
  elif isinstance(v,(list,tuple)):
   print(prefix,k,type(v).__name__,len(v),repr(v[:1])[:350])
  else: print(prefix,k,repr(v)[:350])
show(d)
PY
```
- **Container Exit Code**: `1` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1111
    }
  },
  "schema": "hegel_public_asset_index_v1"
}=== archive ===
total 52K
drwxr-xr-x 2 root root 4.0K Nov 27 07:00 .
drwxrwxrwt 1 root root 4.0K Nov 27 07:00 ..
-rw-r--r-- 1 root root 8.4K Nov 27 07:00 alpha_m97_c03_batch_01_solver.py
-rw-r--r-- 1 root root 7.5K Nov 27 07:00 alpha_m97_c03_batch_02_solver.py
-rw-r--r-- 1 root root 6.7K Nov 27 07:00 alpha_m97_c03_s00_solver.py
-r--r--r-- 1 root root 5.3K Nov 27 07:00 archive_calibration_history.json
-rw-r--r-- 1 root root 1.4K Nov 27 07:00 archive_manifest.json
-rw-r--r-- 1 root root  337 Nov 27 07:00 pre_llm_gate_diagnostic.json
=== primitive search ===
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
=== dataset structure ===
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19719 ms` LLM)
- **Rationale**: Unwrap the public item correctly and check calibration coverage and contamination. Those checks determine which channels use a direct fit and which, if any, need prefix calibration.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The current batch specifies a different terminal-flow basis. Inspect the archived solver for preprocessing conventions, not as a substitute for the new model.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb')); item=items[0]; d=item['dataset']; print('items:',len(items),'item keys:',list(item));
def show(x,p=''):
 for k,v in x.items():
  if isinstance(v,dict):
   print(p,k,'dict'); show(v,p+'  ')
  elif hasattr(v,'shape'):
   a=np.asarray(v); print(p,k,a.shape,a.dtype,a.ravel()[:8])
  else: print(p,k,str(v)[:200])
show(d)
c=d['calibration_load_cells'];
for lab in range(4):
 m=c['lab_id']==lab; print('calibration lab',lab,np.round(np.c_[c['cmd_stress'][m],c['true_stress'][m]],5).tolist())
PY
cat /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; a=json.load(open(p)); print(str(a)[:3600])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`630 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`22564 ms` LLM)
- **Rationale**: The prior output was truncated at the preprocessing details. A compact residual summary determines whether any channel needs prefix calibration, and the previous solver clarifies the imputation convention before implementation.
- **Alternatives Weighed**: `Reuse the control-panel primitive` (rejected: Its shared scanner-gain and classifier assumptions do not match the specified per-lab reference-column statistics and stratum imputation.)

```bash
cd /holdout_task && python3 - <<'PY'
import json,pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d=item['dataset']; a=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history shape',type(a), list(a) if isinstance(a,dict) else len(a)); h=next(iter(a.values())) if isinstance(a,dict) else a[0]; print('first value:',str(h)[:450]); print('public comparison keys',[(k,np.shape(v)) for k,v in item['gt'].items()]); c=d['calibration_load_cells']
for lab in range(4):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('channel',lab,'n',len(x),'OLS',p,'residuals',np.round(y-np.polyval(p,x),4))
for split in ['train','test']:
 s=d[split]; print(split,'missing raw',np.isnan(s['coupon_features_raw']).sum(axis=0),'counts',np.unique(np.c_[s['lab_id'],s['stratum']],axis=0,return_counts=True)[1]); print('measurement shapes',[(k,np.shape(v)) for k,v in s.items() if 'obs' in k])
PY
p=/tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; grep -n '^def ' "$p"; sed -n '1,115p' "$p"
```
- **Container Exit Code**: `0` (`622 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
if line is None:
                raise ValueError(f'No usable calibration for channel {c}')
            lines[c] = line
    return lines


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    return np.where(np.isfinite(aligned), aligned, fills[ids])


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.0]
    u = remaining[:, None] / tau
    decay = np.exp(-u)
    gap = hold[:, None] - stress @ np.diff(decay, axis=0)
    dgap = -stress @ np.diff(u * decay, axis=0)
    v = times[:, None] / tau
    prony = -np.expm1(-v)
    a = np.empty((len(hold), len(times) + 1, 7))
    a[:, :-1, :2] = gap[:, None, :] * prony
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:] = xd[:, None, :] * times[None, :, None]
    a[:, -1, 2:] = xd
    deriv = np.empty(a.shape[:-1] + (2,))
    deriv[:, :-1] = dgap[:, None, :] * prony - gap[:, None, :] * (v * np.exp(-v))
    deriv[:, -1] = (dgap - gap) / tau
    return a, deriv
```

### Turn 4 — `execute_in_remote_docker` (`47890 ms` LLM)
- **Rationale**: All four public channels have ten clean calibration points, so use direct fits. Build the exact design and audit each output against the public reference separately from training; the results will show whether calibration, parameter recovery, or prediction needs further investigation.
- **Alternatives Weighed**: `Fit strain curves and initial rates independently` (rejected: The initial rate is the derivative of the same constitutive model; independent fitting would break that constraint and obscure parameter errors.)

```bash
set -e
cd /holdout_task
p=/tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
sed -n '12,42p;118,162p' "$p"
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py')
s = p.read_text().split('def _design', 1)[0]
s = s.replace('amorphous linear flow', 'shear-banding jammed slip')
Path('alpha_m97_c03_batch_03_solver.py').write_text(s)
PY
cat >> alpha_m97_c03_batch_03_solver.py <<'PY'
def _design(logtau, inputs, dt, times):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.0]
    # Each stress step contributes its own relaxation increment, decayed
    # through all subsequent steps (zero memory before the first step).
    kernel = np.exp(-remaining[1:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    gap = hold[:, None] - stress @ kernel
    diff = gap[:, 0] - gap[:, 1]
    relaxation = -np.expm1(-times[:, None] / tau)
    flow = times / (1.0 + 0.85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    a = np.empty((len(hold), len(times) + 1, 9))
    a[:, :-1, :2] = gap[:, None, :] * relaxation
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:7] = xd[:, None, :] * flow[None, :, None]
    a[:, -1, 2:7] = xd
    a[:, :-1, 7] = slip[:, None] * flow
    a[:, -1, 7] = slip
    a[:, :-1, 8] = diff[:, None] * relaxation[:, 0]
    a[:, -1, 8] = diff / tau[0]
    return a


def _fit(inputs, dt, times, y, rate_weight=1.0):
    lo = np.log(max(min(np.min(dt), np.min(times[times > 0])) / 20, 1e-5))
    hi = np.log(max(np.sum(dt), np.max(times)) * 20)
    weights = np.ones(y.shape)
    weights[:, -1] = rate_weight
    valid = np.isfinite(y)
    weights = weights[valid]
    target = y[valid] * weights

    def profile(logtau):
        mat = _design(logtau, inputs, dt, times)[valid] * weights[:, None]
        beta = np.linalg.lstsq(mat, target, rcond=None)[0]
        return np.einsum('ij,j->i', mat, beta) - target, beta

    fastest = min(np.median(dt), np.min(times[times > 0]))
    horizon = np.max(times)
    starts = [(fastest, horizon / 2), (2 * fastest, horizon),
              (fastest / 2, horizon), (fastest, 2 * horizon),
              (horizon / 4, horizon)]
    best = None
    for start in starts:
        trial = least_squares(lambda x: profile(x)[0], np.log(start),
                              bounds=(lo, hi), max_nfev=100,
                              ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if trial.x[0] < trial.x[1] and (best is None or trial.cost < best.cost):
            best = trial
    if best is None:
        raise ValueError('Could not fit ordered relaxation times')
    return best.x, profile(best.x)[1]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    lines = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    inputs = _inputs(train, aligned[:n], lines)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    logtau, beta = _fit(inputs, dt, times, y)
    design = _design(logtau, _inputs(test, aligned[n:], lines), dt, times)
    pred = np.einsum('nkj,j->nk', design, beta)
    return {'estimated_taus': np.exp(logtau),
            'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:],
            'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
import alpha_m97_c03_batch_03_solver as s
item=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d=item['dataset']; gt=item['gt']; o=s.solve_alpha_m97_c03_batch_03(d)
for k,g in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(k,'estimate',np.round(o[k],6),'public reference',gt[g],'max abs error',np.max(np.abs(o[k]-gt[g])))
for k,g in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=o[k]-gt[g]; lab=d['test']['lab_id']; strata=d['test']['stratum']; print(k,'RMSE',np.sqrt(np.mean(e**2)),'worst cell RMSE',max(np.sqrt(np.mean(e[(lab==i)&(strata==j)]**2)) for i,j in set(zip(lab,strata))))
PY
```
- **Container Exit Code**: `0` (`1227 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
]
    d2 -= np.nanmedian(d2, axis=0)
    curve_scale = max(1.4826 * np.nanmedian(np.abs(d2) / np.sqrt(1 + a0**2 + (1-a0)**2)), 1e-8)
    scale = np.full(y.shape[1], curve_scale)
    bounds = (np.r_[np.log([0.01, 0.01]), np.zeros(2), np.full(5, -np.inf)],
              np.r_[np.log([100., 100.]), np.full(7, np.inf)])

    def run(p, width):
        def fun(p):
            a, _ = _design(p[:2], inputs, dt, times)
            return ((a @ p[2:] - y) / scale)[observed]

        def jac(p):
            a, da = _design(p[:2], inputs, dt, times)
            return (np.concatenate((da * p[2:4], a), axis=-1) / scale[None, :, None])[observed]

        return least_squares(fun, p, jac=jac, bounds=bounds, loss='cauchy', f_scale=width,
                             x_scale='jac', max_nfev=120)

    trials = []
    for taus in ((0.25, 2.), (0.55, 4.), (0.9, 8.), (0.15, 12.)):
        lt = np.log(taus)
        a, _ = _design(lt, inputs, dt, times)
        # Rate measurements are far less prone to impulses than strain.
        rate = observed[:, -1]
        w = np.linalg.lstsq(a[rate, -1], y[rate, -1], rcond=None)[0]
        w[:2] = np.maximum(w[:2], 1e-5)
        p = run(np.r_[lt, w], 10.).x
        p = run(p, 4.).x
        trials.append(run(p, 2.5))
    p = min(trials, key=lambda result: result.cost).x
    residual = _design(p[:2], inputs, dt, times)[0] @ p[2:] - y
    scale[:] = np.maximum(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual, axis=0)), axis=0),
                          curve_scale * 0.35)
    p = run(p, 2.5).x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


{"schema_valid": true, "errors": []}
estimated_taus estimate [0.738836 7.770045] public reference [0.533 4.019] max abs error 3.751044500733949
hereditary_weights estimate [ 1.031575  0.064012  0.180683  0.162812 -0.109143  0.107073 -0.115564] public reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] max abs error 0.4859880946032045
regime_mode_weights estimate [ 0.327028 -0.341769] public reference [ 0.31 -0.22] max abs error 0.12176918814599982
predicted_strain RMSE 0.10262097084307854 worst cell RMSE 0.1312547545907454
initial_rate_pred RMSE 0.10834013190769609 worst cell RMSE 0.14680921018042137

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3556719331ca0393 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35567193 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`33769 ms` LLM)
- **Rationale**: Residual tails distinguish impulsive contamination from a broadly wrong fit. Compare robust fits without reading public reference values during estimation, and inspect prefix line variation to inform the calibration fallback.
- **Alternatives Weighed**: `Change the constitutive basis or add telemetry` (rejected: The basis is specified, and telemetry is explicitly uncoupled. First test whether the observation noise explains the mismatch.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,json
from scipy.optimize import least_squares
import alpha_m97_c03_batch_03_solver as s
item=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d=item['dataset']; gt=item['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); times=d['eval_times']; dt=d['past_dt']; al=s._align(tr,te); lines=s._calibrate(d['calibration_load_cells']); inp=s._inputs(tr,al[:n],lines); test=s._inputs(te,al[n:],lines)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; ok=np.isfinite(y); p0,b0=s._fit(inp,dt,times,y); r=s._design(p0,inp,dt,times)@b0-y
mad=lambda a: 1.4826*np.nanmedian(np.abs(a-np.nanmedian(a,axis=0)),axis=0)
print('OLS residual MAD',mad(r)); print('percentiles absolute residuals curve/rate', [np.nanpercentile(np.abs(a),[50,90,95,99,100]) for a in (r[:,:-1],r[:,-1])]); print('missing',(~ok).sum(axis=0))
ref=s._design(np.log(gt['taus']),inp,dt,times)@np.r_[gt['theta'],gt['vp_theta']]-y
print('diagnostic only, public reference residual MAD',mad(ref)); print('reference residual absolute percentiles', [np.nanpercentile(np.abs(a),[50,90,95,99,100]) for a in (ref[:,:-1],ref[:,-1])]);
scale=np.maximum(mad(r),1e-8); scale[:-1]=np.median(scale[:-1]);
def fun(p): return ((s._design(p[:2],inp,dt,times)@p[2:]-y)/scale)[ok]
def score(p):
 pred=s._design(p[:2],test,dt,times)@p[2:]; e=np.c_[pred[:,:-1]+te['strain_0'][:,None]-gt['te_clean'],pred[:,-1]-gt['te_true_rate0']]; masks=[(te['lab_id']==i)&(te['stratum']==j) for i,j in set(zip(te['lab_id'],te['stratum']))]; return [(round(np.mean(z*z)**.5,6),round(max(np.mean(z[m]**2)**.5 for m in masks),6)) for z in (e[:,:-1], e[:,-1])]
for loss in ('soft_l1','cauchy'):
 for width in (1.5,2.5,4.):
  fits=[]
  for taus in (np.exp(p0), (.3,2.),(.6,4.),(.9,8.)):
   lt=np.log(taus); design=s._design(lt,inp,dt,times); beta=np.linalg.lstsq(design[ok],y[ok],rcond=None)[0]; p=np.r_[lt,beta]; p=least_squares(fun,p,loss='soft_l1',f_scale=4,x_scale='jac',max_nfev=100).x; res=least_squares(fun,p,loss=loss,f_scale=width,x_scale='jac',max_nfev=150); fits.append(res)
  p=min((r for r in fits if r.x[0]<r.x[1]),key=lambda r:r.cost).x; print(loss,width,'tau',np.exp(p[:2]),'weights',np.round(p[2:],4),'RMSE (overall,worst)',score(p))
for rec in json.load(open(str(s._HISTORY))):
 print(rec['episode_short_name'],[np.round(s._historical_line(rec,i),5).tolist() for i in range(4)])
PY
```
- **Container Exit Code**: `0` (`1280 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
OLS residual MAD [0.0354488  0.0986544  0.14932026 0.15707875 0.14385244 0.11797224
 0.08681241 0.06956433 0.07098511 0.07947802 0.10778051]
percentiles absolute residuals curve/rate [array([ 0.06712997,  0.20852318,  0.31343347,  1.85529252, 64.07236546]), array([0.08403773, 0.19348486, 0.23480401, 0.3244751 , 0.58097288])]
missing [0 0 0 0 0 0 0 0 0 0 0]
diagnostic only, public reference residual MAD [0.01878654 0.01920433 0.01763612 0.01792011 0.0216614  0.01505758
 0.01949026 0.01896501 0.01960556 0.01898156 0.01990115]
reference residual absolute percentiles [array([1.26412020e-02, 8.47575844e-02, 2.36885428e-01, 1.74799813e+00,
       6.43612077e+01]), array([0.01333273, 0.04158071, 0.12969237, 0.27890812, 0.43037649])]
soft_l1 1.5 tau [0.53423827 4.29687207] weights [ 0.6928  0.579   0.1361  0.1563 -0.1244  0.12   -0.0863  0.285  -0.2289] RMSE (overall,worst) [(0.030674, 0.069616), (0.039305, 0.08966)]
soft_l1 2.5 tau [0.53522531 4.37836469] weights [ 0.6974  0.5857  0.1356  0.1558 -0.123   0.1206 -0.0861  0.2789 -0.232 ] RMSE (overall,worst) [(0.031339, 0.070482), (0.040223, 0.091044)]
soft_l1 4.0 tau [0.53656484 4.45000348] weights [ 0.7018  0.5903  0.1354  0.155  -0.1217  0.1212 -0.0862  0.2747 -0.2351] RMSE (overall,worst) [(0.031756, 0.07102), (0.041024, 0.092304)]
cauchy 1.5 tau [0.53256773 4.17079716] weights [ 0.6856  0.5691  0.1369  0.1571 -0.1265  0.1192 -0.0867  0.2943 -0.2242] RMSE (overall,worst) [(0.029874, 0.068446), (0.038129, 0.08767)]
cauchy 2.5 tau [0.5329371  4.26197651] weights [ 0.69    0.5786  0.1358  0.157  -0.1251  0.1198 -0.0861  0.2857 -0.2269] RMSE (overall,worst) [(0.030839, 0.069714), (0.039116, 0.089076)]
cauchy 4.0 tau [0.53374127 4.34752987] weights [ 0.6945  0.5859  0.1349  0.1567 -0.1236  0.1204 -0.0858  0.2787 -0.229 ] RMSE (overall,worst) [(0.031665, 0.070671), (0.040046, 0.090288)]
alpha_m97_c03_s00 [[1.0, 1e-05], [0.99999, -2e-05], [1.00004, 2e-05], [1.00004, -8e-05]]
alpha_m97_c03_batch_01 [[1.28999, 0.12], [0.82004, -0.15003], [1.16998, 0.14001], [0.76, -0.10999]]
alpha_m97_c03_batch_02 [[1.28996, 0.11998], [0.82003, -0.15001], [1.16999, 0.13998], [0.75997, -0.11003]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3556719331ca0393 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35567193 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32714 ms` LLM)
- **Rationale**: The robust trial substantially improved both outputs, but a public-reference forward check can distinguish remaining model-fit error from panel alignment or imputation error. If the remaining gap is confined to missing rows, investigate that convention rather than tuning the constitutive fit to absorb it.
- **Alternatives Weighed**: `Tune the loss directly against public test targets` (rejected: Use the targets only to diagnose residual structure; estimate noise and choose fitting settings from training measurements.)
- **ATMS NOGOOD DECLARED**: Refuted `A thin-tailed, comparably scaled measurement error model is adequate for recovering constitutive parameters by least squares.` | Witness: `Curve residuals reach 64 stress-response units versus a typical clean-model residual scale near 0.02; robust fitting reduces slow-tau error from 3.75 to 0.15.` | Blanket Diagnosis: `Measurement impulses dominate the squared loss and displace the inferred relaxation parameters; the specified constitutive basis need not change.` | Sublated Pivot: `Fit the same basis with a bounded-influence observation loss and measure the remaining error separately on complete and imputed coupon rows.`

```bash
cd /holdout_task
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS residual MAD' | head -c 2600
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_03_solver as s
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); lines=s._calibrate(d['calibration_load_cells']); a=s._align(tr,te); dt,t=d['past_dt'],d['eval_times']; b=np.r_[g['theta'],g['vp_theta']]; lt=np.log(g['taus']); ph=t/(1+.85*t)
for name,part,x in [('train',tr,a[:n]),('test',te,a[n:])]:
 y=s._design(lt,s._inputs(part,x,lines),dt,t)@b
 missing=np.isnan(part['coupon_features_raw']).any(axis=1)
 if name=='test':
  err=np.c_[y[:,:-1]+part['strain_0'][:,None]-g['te_clean'],y[:,-1]-g['te_true_rate0']]; print('hypothesis check: exact public parameters, test RMSE present/missing',[(m.tolist() if hasattr(m,'tolist') else m,np.sqrt(np.mean(err[missing==m]**2,axis=0)).round(5)) for m in (False,True)])
  for i in np.argsort(abs(err[:,-1]))[-10:]: print('rate residual',i,part['lab_id'][i],part['stratum'][i],np.isnan(part['coupon_features_raw'][i]).astype(int),np.round(err[i,-1],5),'phi mismatch',np.max(abs(err[i,:-1]-err[i,-1]*ph)).round(6))
 else:
  yy=np.c_[part['strain_curves_obs']-part['strain_0'][:,None],part['instant_rate_obs']]; e=y-yy
  for lab in range(4):
   q=e[part['lab_id']==lab]; print('train lab',lab,'residual median',np.median(q,axis=0).round(4),'MAD',np.median(abs(q-np.median(q,axis=0)),axis=0).round(4))
PY
```
- **Container Exit Code**: `0` (`614 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
train lab 0 residual median [0.0007 0.001  0.0004 0.0028 0.0025 0.0047 0.0022 0.0004 0.0021 0.0079
 0.002 ] MAD [0.0105 0.0135 0.0103 0.0146 0.0206 0.0111 0.0151 0.0138 0.0168 0.0145
 0.0132]
train lab 1 residual median [ 0.0003  0.0068  0.0066  0.0024 -0.0007  0.0003  0.0051  0.0056  0.0057
  0.0025  0.0022] MAD [0.0123 0.0154 0.0112 0.0156 0.0096 0.0101 0.0107 0.0157 0.0122 0.0084
 0.0138]
train lab 2 residual median [ 0.0033  0.0029 -0.003  -0.0015  0.002   0.0027  0.001  -0.0033  0.0015
  0.0022  0.0005] MAD [0.0141 0.0148 0.0106 0.0115 0.0148 0.0092 0.0154 0.0112 0.0129 0.0108
 0.0153]
train lab 3 residual median [ 0.0021  0.0013  0.0048 -0.003   0.003  -0.0044  0.0024  0.0028 -0.0024
  0.0036  0.0002] MAD [0.0138 0.012  0.0128 0.0108 0.0136 0.0132 0.0099 0.0091 0.011  0.0126
 0.0121]
hypothesis check: exact public parameters, test RMSE present/missing [(False, array([0.001  , 0.00205, 0.00266, 0.00305, 0.00333, 0.00353, 0.00369,
       0.00381, 0.00392, 0.004  , 0.00417])), (True, array([0.02636, 0.05421, 0.07024, 0.08065, 0.08797, 0.09338, 0.09756,
       0.10087, 0.10356, 0.1058 , 0.11028]))]
rate residual 24 0 0 [0 0 0 0] -0.00854 phi mismatch 1.8e-05
rate residual 46 0 1 [0 0 0 0] -0.0104 phi mismatch 1.3e-05
rate residual 7 3 0 [0 0 0 0] 0.01104 phi mismatch 1e-06
rate residual 62 2 0 [1 0 0 0] -0.02302 phi mismatch 1e-06
rate residual 31 1 1 [0 1 0 0] 0.04306 phi mismatch 1.7e-05
rate residual 23 0 2 [0 0 1 0] -0.06985 phi mismatch 7e-06
rate residual 58 0 0 [1 0 0 0] 0.09183 phi mismatch 1.8e-05
rate residual 67 0 1 [0 1 0 0] -0.12559 phi mismatch 1.1e-05
rate residual 48 3 1 [0 1 0 0] 0.13551 phi mismatch 1e-06
rate residual 41 0 1 [0 0 1 0] -0.21782 phi mismatch 1e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3556719331ca0393 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35567193 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`47424 ms` LLM)
- **Rationale**: The public-reference diagnostic isolates imputation error from constitutive error. Verify that the saved robust candidate improves material recovery and complete-row predictions before testing transfer across training labs and exercising the calibration fallback.
- **Alternatives Weighed**: `Change the test imputation to optimize public scores` (rejected: With public reference weights, the missing-row error already follows the terminal-flow basis almost exactly. Preserve the specified cell means and keep those descriptor errors out of material estimation.)

```bash
set -e
cd /holdout_task
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -c 1700; echo
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c03_batch_03_solver.py')
s=p.read_text(); i=s.index('def _fit('); j=s.index('def solve_',i)
s=s[:i]+'''def _fit(inputs, dt, times, y):
    lo = np.log(max(min(np.min(dt), np.min(times[times > 0])) / 20, 1e-5))
    hi = np.log(max(np.sum(dt), np.max(times)) * 20)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)],
              np.r_[hi, hi, np.full(9, np.inf)])
    valid = np.isfinite(y)
    yobs = y[valid]

    # Interpolation in the terminal-flow coordinate removes coupon drift.
    # The resulting local differences are a robust initial noise estimate.
    flow = times / (1 + 0.85 * times)
    alpha = (flow[1:-1] - flow[:-2]) / (flow[2:] - flow[:-2])
    curve = y[:, :-1]
    local = (curve[:, 1:-1] - (1-alpha) * curve[:, :-2] - alpha * curve[:, 2:])
    local /= np.sqrt(1 + (1-alpha)**2 + alpha**2)
    local -= np.nanmedian(local, axis=0)
    width = max(1.4826 * np.nanmedian(np.abs(local)), 1e-7)
    widths = np.full(y.shape[1], width)

    def run(p, loss, multiplier):
        scale = np.broadcast_to(widths, y.shape)[valid]

        def residual(p):
            a = _design(p[:2], inputs, dt, times)[valid]
            return (np.einsum('ij,j->i', a, p[2:]) - yobs) / scale

        return least_squares(residual, p, bounds=bounds, loss=loss,
                             f_scale=multiplier, x_scale='jac', max_nfev=150)

    fastest = min(np.median(dt), np.min(times[times > 0]))
    horizon = np.max(times)
    trials = []
    for start in ((fastest, horizon / 2), (2*fastest, horizon),
                  (fastest / 2, 2*horizon), (horizon / 4, horizon)):
        lt = np.log(start)
        a = _design(lt, inputs, dt, times)
        rate = valid[:, -1]
        beta = np.linalg.lstsq(a[rate, -1], y[rate, -1], rcond=None)[0]
        warm = run(np.r_[lt, beta], 'soft_l1', 6.)
        trial = run(warm.x, 'cauchy', 2.5)
        if trial.x[0] < trial.x[1]:
            trials.append(trial)
    if not trials:
        raise ValueError('Could not fit ordered relaxation times')
    p = min(trials, key=lambda result: result.cost).x
    res = _design(p[:2], inputs, dt, times) @ p[2:] - y
    mad = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res, axis=0)), axis=0)
    widths[:-1] = max(np.nanmedian(mad[:-1]), width * 0.35)
    widths[-1] = max(mad[-1], width * 0.35) if np.isfinite(mad[-1]) else width
    refined = run(p, 'cauchy', 2.5).x
    if refined[0] < refined[1]:
        p = refined
    return p[:2], p[2:]


'''+s[j:]
s=s.replace('    logtau, beta = _fit(inputs, dt, times, y)', '''    # Missing measured descriptors introduce a coupon-specific flow error.
    # Estimate the shared material law on the directly observed coupons.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    logtau, beta = _fit(tuple(a[complete] for a in inputs), dt, times, y[complete])''')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_03_solver as s
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; o=s.solve_alpha_m97_c03_batch_03(d)
for key, gt in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(key,np.round(o[key],6),'max absolute error',np.max(abs(o[key]-g[gt])))
t=d['test']; groups=np.c_[t['lab_id'],t['stratum']]; missing=np.isnan(t['coupon_features_raw']).any(axis=1)
for key, gt in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=o[key]-g[gt]; rms=lambda v: np.mean(v*v)**.5; print(key,'overall/complete/missing/worst-cell',rms(e),rms(e[~missing]),rms(e[missing]),max(rms(e[(groups==v).all(axis=1)]) for v in np.unique(groups,axis=0)))
print('train keys',list(d['train']),'test keys',list(d['test']))
PY
```
- **Container Exit Code**: `0` (`1401 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
23d6",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "dt",
          "op": ">",
          "value": 0
        },
        {
          "field": "observation_sample_count",
          "op": ">=",
          "value": 3
        },
        {
          "field": "forecast_sample_count",
          "op": ">=",
          "value": 1
        },
        {
          "field": "harmonic_count",
          "op": ">=",
          "value": 1
        },
        {
          "field": "twice_harmonic_count_over_horizon",
          "op": "<",
          "value": 1
        },
        {
          "field": "gain_floor",
          "op": ">",
          "value": 0
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "assimilate_diffusive_network",
          "signature": "assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
          "doc_summa
{"schema_valid": true, "errors": []}
estimated_taus [0.53224  3.964048] max absolute error 0.05495227468818875
hereditary_weights [ 0.677201  0.547575  0.139188  0.156738 -0.127845  0.118099 -0.088351] max absolute error 0.0032623650106998747
regime_mode_weights [ 0.312073 -0.217057] max absolute error 0.002943060956786564
predicted_strain overall/complete/missing/worst-cell 0.02819266988933553 0.002396935292501461 0.08430585584068229 0.06560842787983485
initial_rate_pred overall/complete/missing/worst-cell 0.03626948973942861 0.0035349693249053207 0.10834811908197799 0.08432398986654548
train keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs'] test keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry']

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c6ea47167527687c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c6ea4716 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`73160 ms` LLM)
- **Rationale**: Leave-one-lab-out fits test whether the shared material law transfers across calibrated labs; a systematic failure would warrant revisiting alignment or fit weighting. Sparse and spiked sweeps test the documented fallback paths, which the clean public sweep does not exercise.
- **Alternatives Weighed**: `Adjust the imputation rule to lower public worst-cell RMSE` (rejected: The reference-parameter diagnostic traces most of that error to missing descriptors, and the contract specifies cell-balanced imputation.)

```bash
set -e
cd /holdout_task
sed -n '1,103p' alpha_m97_c03_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, copy
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c03_batch_03_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']
tr,te=d['train'],d['test']; n=len(tr['lab_id']); lines=s._calibrate(d['calibration_load_cells']); x=s._align(tr,te)
dt,t=d['past_dt'],d['eval_times']; a=s._inputs(tr,x[:n],lines)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; clean=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for lab in range(4):
    fit=clean&(tr['lab_id']!=lab); hold=clean&(tr['lab_id']==lab)
    lt,b=s._fit(tuple(v[fit] for v in a),dt,t,y[fit]); e=np.abs(s._design(lt,tuple(v[hold] for v in a),dt,t)@b-y[hold]); print('leave-lab-out',lab,'n',sum(hold),'taus',np.exp(lt),'curve/rate median AE',np.median(e[:,:-1]),np.median(e[:,-1]),'curve worst horizon median AE',np.max(np.median(e[:,:-1],axis=0)))
    assert np.median(e[:,:-1]) < .04 and np.median(e[:,-1]) < .04
c=d['calibration_load_cells']; solve=s.solve_alpha_m97_c03_batch_03; base=solve(d)
for scenario in ('one point', 'EMI'):
    for lab in range(4):
        q=copy.deepcopy(d); idx=np.flatnonzero(c['lab_id']==lab)
        if scenario=='one point':
            keep=np.ones(len(c['lab_id']),dtype=bool); keep[idx[1:]]=False
            q['calibration_load_cells']={k:v[keep] for k,v in c.items()}
        else:
            q['calibration_load_cells']['true_stress'][idx[0]]+=8.
        err=np.max(np.abs(s._calibrate(q['calibration_load_cells'])-lines)); assert err<.002,(scenario,lab,err)
        pred=solve(q); delta={k:float(np.max(np.abs(v-base[k]))) for k,v in pred.items()}; print(scenario,lab,'calibration max shift',err,'output max shifts',delta); assert max(delta.values())<.01
PY
```
- **Container Exit Code**: `0` (`4680 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7877e-06, 'hereditary_weights': 3.599878547477786e-07, 'regime_mode_weights': 3.760498956673075e-08, 'predicted_strain': 3.0059710312002608e-06, 'initial_rate_pred': 2.9847911210412192e-06}
one point 1 calibration max shift 5.620998376643094e-05 output max shifts {'estimated_taus': 6.278807687554178e-06, 'hereditary_weights': 7.451664926105828e-06, 'regime_mode_weights': 1.4711262342037568e-05, 'predicted_strain': 0.0001474216231978076, 'initial_rate_pred': 0.00018604887993056707}
one point 2 calibration max shift 3.151677004933906e-05 output max shifts {'estimated_taus': 3.474391238267316e-06, 'hereditary_weights': 3.2533983587157422e-06, 'regime_mode_weights': 1.8308830978530644e-06, 'predicted_strain': 2.231225675508597e-05, 'initial_rate_pred': 2.7379540417982184e-05}
one point 3 calibration max shift 3.68638644228092e-05 output max shifts {'estimated_taus': 0.00014894772728535344, 'hereditary_weights': 8.287590182010085e-06, 'regime_mode_weights': 1.4227552638190755e-05, 'predicted_strain': 0.00010592888698468528, 'initial_rate_pred': 0.00014204432512254783}
EMI 0 calibration max shift 1.1406673993574756e-05 output max shifts {'estimated_taus': 3.1648443879372934e-05, 'hereditary_weights': 3.2542704864191307e-06, 'regime_mode_weights': 1.0316031595625219e-07, 'predicted_strain': 1.78082126383039e-05, 'initial_rate_pred': 2.1617688688024828e-05}
EMI 1 calibration max shift 9.655035643763021e-07 output max shifts {'estimated_taus': 1.6921582619566777e-06, 'hereditary_weights': 2.816207910205648e-07, 'regime_mode_weights': 1.32704101429848e-07, 'predicted_strain': 2.3493115670625286e-06, 'initial_rate_pred': 2.9847820055550756e-06}
EMI 2 calibration max shift 1.68280640432672e-05 output max shifts {'estimated_taus': 2.2014701752670618e-05, 'hereditary_weights': 1.5508024915344976e-06, 'regime_mode_weights': 1.2434219637813548e-06, 'predicted_strain': 7.4279984543235145e-06, 'initial_rate_pred': 9.501524862454147e-06}
EMI 3 calibration max shift 5.9302624412227445e-06 output max shifts {'estimated_taus': 1.2376318605511472e-05, 'hereditary_weights': 5.882928065525928e-07, 'regime_mode_weights': 7.445789285687177e-07, 'predicted_strain': 8.009198325886757e-06, 'initial_rate_pred': 1.0417238493154457e-05}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c6ea47167527687c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c6ea4716 (directory).
```

### Turn 9 — `submit_completed_task` (`19045 ms` LLM)
- **Rationale**: The saved solver passed isolated schema verification and leave-one-lab-out validation. One-point and EMI calibration tests across all four channels shifted no output by more than 0.00019.
- **Alternatives Weighed**: `Change the missing-descriptor imputation rule` (rejected: The contract specifies cell-balanced means, and the public-reference diagnostic showed that most missing-row error persists even with the reference material parameters.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:8191'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema passed. Worst lab-by-stratum RMSE was 0.0656 for strain and 0.0843 for initial rate, with most remaining error on coupons with missing descriptors.`

### Turn 10 — `execute_in_remote_docker` (`39111 ms` LLM)
- **Rationale**: An independent recurrence would expose a sign, derivative, or basis mismatch. Signed cell residuals can reveal bias hidden by pooled errors. The previous EMI test used ten-point sweeps; combining a spike with four measured rows tests whether the screen can instead absorb the spike into the fitted gain.
- **Alternatives Weighed**: `Test relay modes, PDE drift, and classification balance` (rejected: This is a continuous regression task with no relay state, PDE, or class labels. The relevant checks are the stress-memory recurrence, rational flow term, signed gap difference, and residual bias within lab-by-stratum cells.)

```bash
set -e
cd /holdout_task
sed -n '56,105p' alpha_m97_c03_batch_03_solver.py
python3 - <<'PY'
import json
x=json.load(open('public_asset_index.json'))
print('Asset-index keys:', list(x) if isinstance(x,dict) else type(x).__name__)
for line in json.dumps(x,indent=2).splitlines():
 if any(k in line.lower() for k in ('tolerance', 'rmse', 'tau_rel', 'threshold')): print(line)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,itertools
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c03_batch_03_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); c=d['calibration_load_cells']; lines=s._calibrate(c); a=s._align(tr,te); o=s.solve_alpha_m97_c03_batch_03(d); tau=o['estimated_taus']; hw=o['hereditary_weights']; rw=o['regime_mode_weights']
for split,part,x in [('train',tr,a[:n]),('test',te,a[n:])]:
 gain,bias=lines[part['lab_id']].T; past=part['past_stress_cmd']*gain[:,None]+bias[:,None]; hold=part['stress_hold_cmd']*gain+bias; mem=np.zeros((len(hold),2))
 for stress,dt in zip(past.T,d['past_dt']):
  decay=np.exp(-dt/tau); mem=decay*mem+(1-decay)*stress[:,None]
 g=hold[:,None]-mem; t=d['eval_times']; rel=1-np.exp(-t[:,None]/tau); drift=hw[2]+x@hw[3:]+rw[0]*np.copysign(np.sqrt(np.abs(hold)),hold); link=rw[1]*(g[:,0]-g[:,1]); eps=part['strain_0'][:,None]+(g*hw[:2])@rel.T+drift[:,None]*(t/(1+0.85*t))+link[:,None]*rel[:,0]; rate=(g*hw[:2])@(1/tau)+drift+link/tau[0]
 if split=='test':
  de=np.max(abs(eps-o['predicted_strain'])); dr=np.max(abs(rate-o['initial_rate_pred'])); print('Independent recurrence / rational flow / signed gap check:',de,dr); assert max(de,dr)<1e-10
 else:
  err=np.c_[eps-part['strain_curves_obs'],rate-part['instant_rate_obs']]; cells=np.c_[part['lab_id'],part['stratum']]; complete=np.isfinite(part['coupon_features_raw']).all(axis=1)
  for label,mask in [('complete',complete),('imputed',~complete)]:
   meds=np.array([np.median(err[mask & (cells==cell).all(axis=1)],axis=0) for cell in np.unique(cells[mask],axis=0)]); print(label,'max absolute lab×stratum signed median curve/rate',np.max(abs(meds[:,:-1])),np.max(abs(meds[:,-1])))
for lab in range(4):
 idx=np.flatnonzero(c['lab_id']==lab); x=c['cmd_stress'][idx]; y=c['true_stress'][idx]; bad=[]
 for sub in itertools.combinations(range(len(idx)),4):
  sub=np.array(sub); xx=x[sub]
  for j in range(4):
   for impulse in (4.,-4.,8.,-8.):
    yy=y[sub].copy(); yy[j]+=impulse; keep=s._clean_sweep(xx,yy)
    if keep.all(): bad.append((xx.copy(),j,impulse,float(np.max(abs(s._line(xx,yy)-lines[lab])))))
 print('lab',lab,'missed EMI in four-row sweeps',len(bad),'worst',max(bad,key=lambda b:b[-1]) if bad else None)
PY
```
- **Container Exit Code**: `0` (`2014 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    return np.where(np.isfinite(aligned), aligned, fills[ids])


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.0]
    # Each stress step contributes its own relaxation increment, decayed
    # through all subsequent steps (zero memory before the first step).
    kernel = np.exp(-remaining[1:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    gap = hold[:, None] - stress @ kernel
    diff = gap[:, 0] - gap[:, 1]
Asset-index keys: ['files', 'schema']
{"schema_valid": true, "errors": []}
complete max absolute lab×stratum signed median curve/rate 0.01569826917891376 0.0072406630700132535
imputed max absolute lab×stratum signed median curve/rate 0.275226307902966 0.2027782318332214
Independent recurrence / rational flow / signed gap check: 8.881784197001252e-16 8.881784197001252e-16
lab 0 missed EMI in four-row sweeps 840 worst (array([1.945973, 2.07103 , 1.652455, 2.006752]), 2, -4.0, 20.948663809614214)
lab 1 missed EMI in four-row sweeps 840 worst (array([1.291089, 1.104415, 1.197564, 1.256207]), 1, 4.0, 27.175348870139953)
lab 2 missed EMI in four-row sweeps 840 worst (array([1.391379, 1.800799, 1.290929, 1.767257]), 2, -4.0, 9.416553723591537)
lab 3 missed EMI in four-row sweeps 840 worst (array([1.060949, 2.159046, 1.94408 , 1.834298]), 0, -4.0, 8.016561398943116)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c6ea47167527687c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c6ea4716 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`50755 ms` LLM)
- **Rationale**: The independent recurrence matched both outputs to machine precision, so the constitutive basis stays unchanged. Verify the targeted calibration fix on measured four- and five-row subsets, check that clean shifted sweeps still take priority, and rerun the exact artifact in an isolated Python process.
- **Alternatives Weighed**: `Use archive calibration for every short sweep` (rejected: That would override clean in-batch readings and hide genuine gain or bias changes.)
- **ATMS NOGOOD DECLARED**: Refuted `A residual measured against a line fitted to the same short sweep reliably identifies a single EMI spike greater than 3.0 stress units.` | Witness: `Injecting individual spikes into four-row subsets of actual sweeps produced 840 misses per lab and a gain error as large as 27.18.` | Blanket Diagnosis: `The spike can pull the short-sweep fit toward itself, leaving a small fitted residual despite a large measurement error.` | Sublated Pivot: `Identify a spike against a line supported by the other readings, then retain direct fits with at least four clean points and use prefix history when the spike leaves fewer.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_03_solver.py')
s = p.read_text()
a = s.index('    i, j = np.triu_indices', s.index('def _clean_sweep'))
s = s[:a] + '''    # A spike can tilt a short sweep enough to hide in its fitted residuals.
    # Test it against the line supported by the other readings instead.
    if x.size >= 3:
        candidates = []
        for omit in range(x.size):
            keep = np.arange(x.size) != omit
            if np.unique(x[keep]).size < 2:
                continue
            gain, bias = _line(x[keep], y[keep])
            residual = y - (gain * x + bias)
            # Clean public and prefix sweeps agree with an affine line well
            # within 0.01 stress units; require that agreement before removal.
            if np.max(np.abs(residual[keep])) < 0.01 and abs(residual[omit]) > 3.0:
                candidates.append((np.sum(residual[keep]**2), keep))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
''' + s[a:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, itertools, pickle
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c03_batch_03_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; c=d['calibration_load_cells']; lines=s._calibrate(c)
count=0
for lab in range(4):
    idx=np.flatnonzero(c['lab_id']==lab); x=c['cmd_stress'][idx]; y=c['true_stress'][idx]
    for size in (4,5):
        for sub in itertools.combinations(range(len(x)),size):
            sel=list(sub); xx=x[sel]; yy=y[sel]
            assert s._clean_sweep(xx,yy).all()
            for j in range(size):
                for impulse in (3.1,-3.1,4.,-4.,8.,-8.):
                    z=yy.copy(); z[j]+=impulse
                    assert np.array_equal(np.flatnonzero(~s._clean_sweep(xx,z)),[j]),(lab,xx,j,impulse)
                    count+=1
print('Localized spike cases passed:',count)
for lab in range(4):
    idx=np.flatnonzero(c['lab_id']==lab)
    def sweep(size):
        take=(c['lab_id']!=lab); take[idx[:size]]=True
        return {k:v[take].copy() for k,v in c.items()}
    hist=s._HISTORY
    for size in (1,4,5):
        q=sweep(size); mask=q['lab_id']==lab
        if size==5:
            q['true_stress'][mask] += 0.08*q['cmd_stress'][mask]+0.06
            s._HISTORY=Path('/tmp/no-calibration-history-needed.json')
        if size>1:
            q['true_stress'][np.flatnonzero(mask)[0]]+=3.1
        got=s._calibrate(q)[lab]; s._HISTORY=hist
        expected=lines[lab]+([0.08,0.06] if size==5 else [0,0]); assert max(abs(got-expected))<.002,(lab,size,got,expected)
    for size in (2,3,4):
        q=sweep(size); mask=q['lab_id']==lab; q['true_stress'][mask]+=0.08*q['cmd_stress'][mask]+0.06
        s._HISTORY=Path('/tmp/no-calibration-history-needed.json'); got=s._calibrate(q)[lab]; s._HISTORY=hist
        assert max(abs(got-lines[lab]-[.08,.06]))<.002,(lab,size,got)
print('Archive and direct-fit routing passed for every channel')
o=s.solve_alpha_m97_c03_batch_03(d); groups=np.c_[d['test']['lab_id'],d['test']['stratum']]
for k, ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta'),('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=o[k]-g[ref]; print(k,'max parameter error' if not k.endswith(('strain','pred')) else 'worst-cell RMSE',np.max(abs(e)) if not k.endswith(('strain','pred')) else max(np.mean(e[(groups==v).all(axis=1)]**2)**.5 for v in np.unique(groups,axis=0)))
PY
```
- **Container Exit Code**: `0` (`8544 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Localized spike cases passed: 50400
Archive and direct-fit routing passed for every channel
estimated_taus max parameter error 0.05495227468818875
hereditary_weights max parameter error 0.0032623650106998747
regime_mode_weights max parameter error 0.002943060956786564
predicted_strain worst-cell RMSE 0.06560842787983485
initial_rate_pred worst-cell RMSE 0.08432398986654548

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a1c813317186c6cb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a1c81331 (directory).
```

### Turn 12 — `submit_completed_task` (`13985 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification, detected every injected spike in the measured four- and five-row sweeps, and chose direct or archive calibration as specified across all four channels. Public prediction errors were unchanged, so the calibration fix can be submitted without further tuning.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema passed, and calibration passed all 50,400 single-spike stress cases. Worst lab-by-stratum RMSE was 0.0656 for strain and 0.0843 for initial rate. Most remaining error was on coupons with missing descriptors.`

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
