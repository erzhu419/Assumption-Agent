# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `279,014` (`268,526` / `76,544` / `10,488`) |
| **Total Episode Wall Time (s)** | `348.24s` |

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

### Turn 1 — `execute_in_remote_docker` (`16148 ms` LLM)
- **Rationale**: Inspect the actual fields and validator before coding to detect schema mismatches or sparse calibration channels that would change the alignment path.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The current batch specifies a distinct terminal-flow basis; prior code is useful only where the input contract and mechanism match.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection'; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c01_batch_03.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None),getattr(v,'dtype',None),str(v if isinstance(v,(float,int,str,dict)) else (v[:2] if hasattr(v,'__getitem__') else v))[:450])
PY
```
- **Container Exit Code**: `1` (`730 ms`, timed_out=`False`)
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
}total 44
-rw-r--r-- 1 root root 8422 Nov 27 07:01 alpha_m97_c01_batch_01_solver.py
-rw-r--r-- 1 root root 6674 Nov 27 07:01 alpha_m97_c01_batch_02_solver.py
-rw-r--r-- 1 root root 8166 Nov 27 07:01 alpha_m97_c01_s00_solver.py
-r--r--r-- 1 root root 5407 Nov 27 07:01 archive_calibration_history.json
-rw-r--r-- 1 root root 1416 Nov 27 07:01 archive_manifest.json
-rw-r--r-- 1 root root  338 Nov 27 07:01 pre_llm_gate_diagnostic.json
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

### Turn 2 — `execute_in_remote_docker` (`18892 ms` LLM)
- **Rationale**: Unwrap the documented list envelope and inspect the previous implementation to identify which alignment conventions transfer and which must change for the new regime.
- **Alternatives Weighed**: `Code from the asset index alone` (rejected: The index does not resolve calibration history, feature imputation, or training-response conventions.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
item=pickle.load(open('public_alpha_m97_c01_batch_03.pkl','rb'))[0]
def schema(x, depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*depth,k, type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''), v if np.isscalar(v) else (len(v) if isinstance(v,list) else ''))
   if isinstance(v,dict): schema(v,depth+2)
schema(item)
d=item['dataset']; print('past dt',d['past_dt']); print('eval times',d['eval_times'])
for split in ('train','test'):
 s=d[split]; print(split,'lab counts',np.unique(s['lab_id'],return_counts=True)); print('strata',[(k,np.unique(v,return_counts=True)) for k,v in s.items() if 'strat' in k])
PY
sed -n '1,245p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
head -c 1100 /tmp/hegel_continual_archive/archive_calibration_history.json; echo
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -65
```
- **Container Exit Code**: `0` (`643 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`22899 ms` LLM)
- **Rationale**: Check for calibration spikes, empty imputation cells, or missing training responses; each would change the implementation. Read only the archived solver section needed to establish the panel and fallback conventions.
- **Alternatives Weighed**: `Use the control-panel skill` (rejected: It estimates shared scanner gain rather than the required per-lab median reference means and standard deviations.)

```bash
sed -n '1,105p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
with open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
a=d['calibration_load_cells']
for c in range(4):
 m=a['lab_id']==c; x=a['cmd_stress'][m]; y=a['true_stress'][m]; p=np.polyfit(x,y,1); print('calibration',c,len(x),p,'max residual',np.max(abs(y-np.polyval(p,x))))
for k in ('train','test'):
 s=d[k]; x=s['coupon_features_raw']; print(k,'missing coupon columns',np.isnan(x).sum(0),'missing refs',np.isnan(s['ref_coupons']).sum()); print('minimum observed counts per lab/stratum',np.min([np.isfinite(x[(s['lab_id']==c)&(s['stratum']==j)]).sum(0) for c in range(4) for j in range(3)],axis=0))
 if k=='train': print('finite training responses',[(h,np.isfinite(s[h]).sum(),s[h].size) for h in ('strain_curves_obs','instant_rate_obs')])
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: a=json.load(f)
print('archive sweep fields', {k:list(v) for k,v in a.items()})
PY
```
- **Container Exit Code**: `1` (`627 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(x[sel]-np.median(np.nanmean(panel, axis=0)))/np.median(np.nanstd(panel, axis=0, ddof=0))
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    observed = np.isfinite(x)
    sums = np.zeros((cell.max()+1, x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(observed, x, 0.))
    np.add.at(counts, cell, observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    n_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), n_cells, out=np.zeros(x.shape[1]), where=n_cells > 0)
    x = np.where(observed, x, fill)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(logtau, state, dt, times):
    past, hold, x = state
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        u = -width/tau
        z = np.exp(u)*z - np.expm1(u)*stress[:, None]
    gap = hold[:, None]-z
    curves = np.concatenate((gap[:, None, :]*(-np.expm1(-times[:, None]/tau)),
                             x[:, None, :]*times[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x))
    return curves, rates


def _scale(e):
    return max(1.4826*np.median(np.abs(e-np.median(e))), 1e-7)


def _fit(state, dt, times, delta, rate_obs, loss='cauchy'):
    cm, rm = np.isfinite(delta), np.isfinite(rate_obs)
    y = delta[cm]
    r = rate_obs[rm]
    horizon = np.broadcast_to(times, delta.shape)[cm]
    # Exclude gross values from initialization only; the final loss handles them directly.
    center = np.median(y/horizon)
    initial = np.abs(y/horizon-center) < 10*_scale(y/horizon)
    bounds = np.log([max(0.02*np.min(dt), 1e-4), 20*(dt.sum()+times.max())])
    starts = np.clip(np.log([[.25, 2.5], [.65, 4.], [1.25, 8.]]), *bounds)

    def matrices(p):
        a, b = _design(p, state, dt, times)
calibration 0 10 [1.33998908 0.15000143] max residual 0.0002167704491329303
calibration 1 10 [ 0.77003896 -0.17002605] max residual 0.00014201764236299041
calibration 2 10 [ 1.20998093 -0.11999076] max residual 0.00016161625739985475
calibration 3 10 [0.83000147 0.11001181] max residual 0.00017209599156450817
train missing coupon columns [ 5  2  5 11] missing refs 238
minimum observed counts per lab/stratum [7 8 8 6]
finite training responses [('strain_curves_obs', 1800, 1800), ('instant_rate_obs', 180, 180)]
test missing coupon columns [3 1 1 3] missing refs 95
minimum observed counts per lab/stratum [2 2 2 2]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 4 — `execute_in_remote_docker` (`68187 ms` LLM)
- **Rationale**: Build the prescribed basis and check all outputs against the public fixture. Per-lab and horizon errors will show whether alignment or the fit needs correction; the public reference values are not read by the solver.
- **Alternatives Weighed**: `Use archived gains or the diagnostic payload` (rejected: All four channels have ten clean in-batch points; the direct fit is required and has maximum residual below 0.00022.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_03_solver.py <<'PY'
"""Calibrated two-branch rheology in the jammed-slip regime."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _calibrate(rows):
    cmd, truth = (np.asarray(rows[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    lab = np.asarray(rows['lab_id'])
    result = np.empty((4, 2))
    sweeps = None
    for c in range(4):
        mask = (lab == c) & np.isfinite(cmd) & np.isfinite(truth)
        x, y = cmd[mask], truth[mask]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            dx, dy = x[:, None] - x, y[:, None] - y
            slope = np.median([np.median(b[a != 0] / a[a != 0])
                               for a, b in zip(dx, dy) if np.any(a != 0)])
            clean = np.abs(y - slope*x - np.median(y - slope*x)) <= 3.0
            spikes = not clean.all()
            x, y = x[clean], y[clean]
        usable = len(x) >= 2 and np.ptp(x) > 0
        if usable and (len(x) >= 4 or not spikes):
            result[c] = np.polyfit(x, y, 1)
            continue
        if len(x) < 2 or spikes:
            if sweeps is None:
                sweeps = json.loads(_ARCHIVE.read_text())
            for sweep in reversed(sweeps):
                sel = np.asarray(sweep['lab_id']) == c
                a, b = (np.asarray(sweep[k])[sel] for k in ('cmd_stress', 'true_stress'))
                if len(a) >= 2 and np.ptp(a) > 0:
                    result[c] = np.polyfit(a, b, 1)
                    break
            else:
                raise ValueError(f'No usable load-cell calibration for channel {c}')
        else:
            raise ValueError(f'Load-cell commands have no variation for channel {c}')
    return result


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
    past, hold, static = state
    # Unroll the memory recurrence: each commanded step decays over the later steps.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = -np.expm1(-dt[:, None]/tau) * np.exp(-tail[:, None]/tau)
    gap = hold[:, None] - past @ kernel
    expo = -np.expm1(-time[:, None]/tau)
    phi = time/(1.0 + 0.85*time)
    diff = gap[:, 0]-gap[:, 1]
    curve = np.concatenate((gap[:, None, :]*expo,
                            static[:, None, :]*phi[None, :, None],
                            diff[:, None, None]*expo[None, :, :1]), axis=2)
    rate = np.column_stack((gap/tau, static, diff/tau[0]))
    return curve, rate


def _mad(x, axis=None):
    return 1.4826*np.nanmedian(np.abs(x-np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _fit(state, dt, time, delta, rate):
    cm, rm = np.isfinite(delta), np.isfinite(rate)
    y = np.r_[delta[cm], rate[rm]]
    n_curve = cm.sum()
    def design(p):
        a, b = _design(np.exp(p), state, dt, time)
        return np.concatenate((a[cm], b[rm]))

    # Before estimating instrument noise, keep the initial-rate scale comparable
    # to the observed strain changes near the start of the hold.
    w = np.r_[np.ones(n_curve), np.full(rm.sum(), min(time[time > 0]))]
    bounds = np.log([max(0.02*np.min(dt), 1e-4), 20*(dt.sum()+time.max())])
    starts = np.clip(np.log([[.25, 2.5], [.65, 4.], [1.25, 8.], [.15, 9.]]), *bounds)

    def profile(p, weights):
        a = design(p)
        coef = np.linalg.lstsq(a*weights[:, None], y*weights, rcond=None)[0]
        return a @ coef-y, coef

    def opt(p):
        return least_squares(lambda q: profile(q, w)[0]*w, p, bounds=bounds,
                             diff_step=1e-4, max_nfev=100, ftol=1e-9)

    p = min((opt(s) for s in starts), key=lambda fit: fit.cost).x
    for _ in range(10):
        residual, _ = profile(p, w)
        e = np.full(delta.shape, np.nan)
        e[cm] = residual[:n_curve]
        sigma_c = np.maximum(_mad(e, axis=0), max(0.35*_mad(e), 1e-7))
        sigma_r = max(_mad(residual[n_curve:]), 1e-7) if rm.any() else 1.
        sigma = np.r_[np.broadcast_to(sigma_c, delta.shape)[cm], np.full(rm.sum(), sigma_r)]
        w = 1. / np.hypot(sigma, residual/2.38)
        p = opt(p).x
    return np.exp(p), profile(p, w)[1]


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
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; out=m.solve_alpha_m97_c01_batch_03(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,out[k], 'public reference',g[j])
for k,j in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=out[k]-g[j]; print(k,'RMSE',np.sqrt(np.mean(e**2)), 'by lab',[np.sqrt(np.mean(e[d['test']['lab_id']==c]**2)) for c in range(4)],'by horizon',np.sqrt(np.mean(e**2,axis=0)))
PY
```
- **Container Exit Code**: `0` (`1013 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus [0.59377599 3.89441967] public reference [0.593 3.877]
hereditary_weights [ 0.67645326  0.54853989  0.13787026  0.15521454 -0.12840318  0.11537022
 -0.08801557] public reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
regime_mode_weights [ 0.31705355 -0.21543172] public reference [ 0.31 -0.22]
predicted_strain RMSE 0.021283407163790114 by lab [0.02142030371776289, 0.02862509733469196, 0.019230251695559794, 0.014792101755742413] by horizon [0.00651902 0.0134194  0.01741522 0.02002716 0.02186872 0.02323608
 0.02429208 0.02513256 0.02581837 0.0263891 ]
initial_rate_pred RMSE 0.02727068286730199 by lab [0.02668690982017493, 0.03655702502142504, 0.025295002769107847, 0.019176378702030313] by horizon 0.02727068286730199

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dccac2a1bac5ee23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dccac2a1 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`40171 ms` LLM)
- **Rationale**: Compare the closed-form memory against the explicit recurrence to catch indexing or derivative errors. Separate missing-feature errors from complete-case errors and hold out each lab's responses to distinguish imputation limits from a fit that fails to transfer.
- **Alternatives Weighed**: `Submit after the public schema and aggregate errors pass` (rejected: The aggregate results do not test sparse or spiked calibration, response holdouts by lab, or whether prediction errors concentrate in missing-feature rows.)

```bash
cat > /tmp/check_batch03.py <<'PY'
import importlib.util, json, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'
spec=importlib.util.spec_from_file_location('solver',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; o=m.solve_alpha_m97_c01_batch_03(d)
rms=lambda x: np.sqrt(np.mean(x*x))
n=len(d['train']['lab_id']); t=d['eval_times']; dt=d['past_dt']; state=m._prepare(d); p,h,x=state
a,b=m._design(o['estimated_taus'],state,dt,t)
z=np.zeros((len(h),2)); tau=o['estimated_taus']
for j,width in enumerate(dt):
    q=np.exp(-width/tau); z=q*z+(1-q)*p[:,j,None]
wa=o['hereditary_weights']; wr=o['regime_mode_weights']; gap=h[:,None]-z; drift=x[:,0:5]@wa[2:]+wr[0]*np.sign(h)*np.sqrt(abs(h)); cross=wr[1]*(gap[:,0]-gap[:,1]); f=1-np.exp(-t[:,None]/tau)
assert np.allclose(o['predicted_strain'],d['test']['strain_0'][:,None]+((gap*wa[:2])@f.T+drift[:,None]*(t/(1+.85*t))+cross[:,None]*f[:,0])[n:])
assert np.allclose(o['initial_rate_pred'],((gap*wa[:2])@(1/tau)+drift+cross/tau[0])[n:])
for key,ref in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=o[key]-g[ref]; s=d['test']; miss=~np.isfinite(s['coupon_features_raw']).all(1)
    print(key, 'complete/missing',[(int(k.sum()),rms(e[k])) for k in (~miss,miss)], 'by stratum',[rms(e[s['stratum']==j]) for j in np.unique(s['stratum'])], 'by hold sign',[rms(e[k]) for k in (h[n:]<0,h[n:]>=0)])
    print('worst lab/stratum',max((rms(e[k]),c,j,int(k.sum())) for c in range(4) for j in range(3) if (k:=(s['lab_id']==c)&(s['stratum']==j)).any()))
s=d['train']; delta=s['strain_curves_obs']-s['strain_0'][:,None]; y=s['instant_rate_obs']; print('leave-one-lab-out, observed response RMSE:')
for c in range(4):
    keep=s['lab_id']!=c; tr=tuple(v[:n][keep] for v in state); va=tuple(v[:n][~keep] for v in state)
    tau,w=m._fit(tr,dt,t,delta[keep],y[keep]); a,b=m._design(tau,va,dt,t)
    print(c,tau,rms(a@w-delta[~keep]),rms(b@w-y[~keep]))
cal=d['calibration_load_cells']; c0=np.flatnonzero(cal['lab_id']==0)
for name,keep,spikes in [('sparse', np.arange(len(cal['lab_id']))!=c0[-1], []),('EMI', np.ones(len(cal['lab_id']),bool), c0[:1])]:
    if name=='sparse': keep[c0[1:]]=False
    rows={k:v.copy() for k,v in cal.items()}; rows['true_stress'][spikes]+=5
    print(name,m._calibrate({k:v[keep] for k,v in rows.items()})[0])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch03.py
```
- **Container Exit Code**: `0` (`997 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
predicted_strain complete/missing [(65, 0.0067360155571309045), (7, 0.06509937723302327)] by stratum [0.005992063761105894, 0.026257936848650736, 0.022474159036915218] by hold sign [0.014792451557197972, 0.027307360130268944]
worst lab/stratum (0.03579361942685585, 1, 1, 10)
initial_rate_pred complete/missing [(65, 0.009304499932590542), (7, 0.08273744931737163)] by stratum [0.00838806917015394, 0.033754282093769156, 0.028500534954433176] by hold sign [0.018888632402375132, 0.035033231103768984]
worst lab/stratum (0.04589137234824945, 1, 1, 10)
leave-one-lab-out, observed response RMSE:
0 [0.59291126 3.89314856] 0.8749262525489254 0.05130905826154803
1 [0.59623139 3.94856372] 0.41774292684803493 0.034034256395817634
2 [0.59170661 3.87482318] 0.5471741930076216 0.06316235679555152
3 [0.59365315 3.86344454] 0.1711747187596817 0.04419813972078792
sparse [1.34002987 0.14997666]
EMI [1.34000119 0.15001017]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dccac2a1bac5ee23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dccac2a1 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33132 ms` LLM)
- **Rationale**: Missing-feature rows dominate clean-target error. Compare fitting with and without those rows, while retaining supplied imputation for prediction. Change the estimator only if robust held-out lab errors and public worst-slice metrics support it.
- **Alternatives Weighed**: `Replace the specified imputation rule` (rejected: The missing rows expose an error concentration, not evidence that a different imputation rule is authorized or transfers.)

```bash
cat > /tmp/compare_batch03.py <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i,=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb')); d=i['dataset']; g=i['gt']; a=d['train']; b=d['test']; n=len(a['lab_id']); dt=d['past_dt']; t=d['eval_times']; state=m._prepare(d)
y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; complete=np.isfinite(a['coupon_features_raw']).all(1)
train=lambda idx: tuple(v[:n][idx] for v in state)
rmse=lambda x: np.sqrt(np.mean(x*x))
for c in (*np.unique(a['lab_id']),None):
    keep=np.ones(n,bool) if c is None else a['lab_id']!=c
    for name,sel in [('all',keep),('complete',keep&complete)]:
        tau,w=m._fit(train(sel),dt,t,y[sel],r[sel]);
        if c is None:
            A,B=m._design(tau,tuple(v[n:] for v in state),dt,t); e=A@w+b['strain_0'][:,None]-g['te_clean']; f=B@w-g['te_true_rate0'];
            mx=lambda z: max(rmse(z[k]) for lab in np.unique(b['lab_id']) for st in np.unique(b['stratum']) if (k:=(b['lab_id']==lab)&(b['stratum']==st)).any()); print('public',name,'tau',tau,'weight deviations',max(abs(w[:7]-g['theta'])),max(abs(w[7:]-g['vp_theta'])),'rmse',rmse(e),rmse(f),'worst lab/stratum',mx(e),mx(f))
        else:
            A,B=m._design(tau,train(~keep),dt,t); e=A@w-y[~keep]; f=B@w-r[~keep]; print('LOLO',c,name,'curve median absolute',np.median(abs(e)),'rate median absolute',np.median(abs(f)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/compare_batch03.py
```
- **Container Exit Code**: `0` (`1325 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
LOLO 0 all curve median absolute 0.014062818658160003 rate median absolute 0.010318438065294422
LOLO 0 complete curve median absolute 0.014519043637115148 rate median absolute 0.009234703405947897
LOLO 1 all curve median absolute 0.012698993506840658 rate median absolute 0.016665607454522613
LOLO 1 complete curve median absolute 0.012471119127170915 rate median absolute 0.015404840095633698
LOLO 2 all curve median absolute 0.012705315904267905 rate median absolute 0.014545889901996123
LOLO 2 complete curve median absolute 0.012558407512489489 rate median absolute 0.01449663263069012
LOLO 3 all curve median absolute 0.016717924281575897 rate median absolute 0.020962757286118627
LOLO 3 complete curve median absolute 0.017154417925145216 rate median absolute 0.022840725122040162
public all tau [0.59377599 3.89441967] weight deviations 0.0047854631606242815 0.007053549642700274 rmse 0.021283407163790114 0.02727068286730199 worst lab/stratum 0.03579361942685585 0.04589137234824945
public complete tau [0.59455392 3.92236434] weight deviations 0.004009429784409699 0.00211332674272921 rmse 0.021655741584431117 0.027760461298364535 worst lab/stratum 0.037069164056827385 0.04765053228268741

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dccac2a1bac5ee23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dccac2a1 (directory).
```

### Turn 7 — `submit_completed_task` (`17989 ms` LLM)
- **Rationale**: The verified candidate passed the schema check in an isolated Python process, matched the explicit memory recurrence and initial-rate formula, and passed the sparse-calibration and EMI checks. The complete-case alternative did not improve the worst-slice bottleneck.
- **Alternatives Weighed**: `Fit only training rows with complete coupon features` (rejected: It worsened both worst public lab-by-stratum RMSEs and did not consistently improve leave-one-lab-out errors.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_03_solver.py': 'FILE:6656'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed. Worst public lab-by-stratum RMSE was 0.03579 for strain and 0.04589 for initial rate. Missing coupon features were the largest source of error.`

### Turn 8 — `execute_in_remote_docker` (`29268 ms` LLM)
- **Rationale**: Test whether the specified slip and cross-gap terms remain identifiable in each held-out lab, whether residual bias indicates a lab-dependent alignment error, and whether one calibration point controls the physical stress estimates. A failed check would direct a change to the basis fit or calibration; otherwise the verified solver stays unchanged.
- **Alternatives Weighed**: `Apply relay-mode and balanced-accuracy checks` (rejected: The task predicts continuous strain and rate and specifies no relay state or class label. Held-out lab residuals test the relevant nuisance effect.)

```bash
set -e
sed -n '1,78p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
a=d['train']; lab=a['lab_id']; n=len(lab); dt=d['past_dt']; t=d['eval_times']; y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; state=tuple(v[:n] for v in m._prepare(d)); take=lambda k: tuple(v[k] for v in state)
for c in np.unique(lab):
 k=lab!=c; tau,w=m._fit(take(k),dt,t,y[k],r[k]); A,B=m._design(tau,take(~k),dt,t); ec=A@w-y[~k]; er=B@w-r[~k]; AB=np.vstack((A.reshape(-1,9),B)); AB/=np.linalg.norm(AB,axis=0); sv=np.linalg.svd(AB,compute_uv=False)
 print('held lab',int(c),'rank',np.linalg.matrix_rank(AB),'scaled condition',round(sv[0]/sv[-1],2),'curve/rate median bias',np.round([np.median(ec),np.median(er)],6),'curve/rate median abs',np.round([np.median(abs(ec)),np.median(abs(er))],6),'curve residuals > 0.2',int((abs(ec)>.2).sum()))
cal=d['calibration_load_cells']; base=m._calibrate(cal)
for c in range(4):
 idx=np.flatnonzero(cal['lab_id']==c); exc=[]
 for j in idx:
  keep=np.arange(len(cal['lab_id']))!=j; coeff=m._calibrate({name:v[keep] for name,v in cal.items()}); exc.append(np.max(np.abs((coeff[c,0]-base[c,0])*np.r_[a['past_stress_cmd'][lab==c].ravel(),a['stress_hold_cmd'][lab==c]]+coeff[c,1]-base[c,1])))
 print('channel',c,'max stress shift deleting one calibration point',max(exc))
print('regime',d['shear_banding_index'],'state keys',sorted(a))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1160 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        x[sel] = (x[sel]-np.median(np.nanmean(panel, axis=0)))/np.median(np.nanstd(panel, axis=0, ddof=0))
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    observed = np.isfinite(x)
    sums = np.zeros((cell.max()+1, x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(observed, x, 0.))
    np.add.at(counts, cell, observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    n_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), n_cells, out=np.zeros(x.shape[1]), where=n_cells > 0)
    x = np.where(observed, x, fill)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(logtau, state, dt, times):
    past, hold, x = state
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
held lab 0 rank 9 scaled condition 7.09 curve/rate median bias [0.000736 0.000643] curve/rate median abs [0.014063 0.010318] curve residuals > 0.2 19
held lab 1 rank 9 scaled condition 6.3 curve/rate median bias [-0.00056  -0.002294] curve/rate median abs [0.012699 0.016666] curve residuals > 0.2 19
held lab 2 rank 9 scaled condition 6.54 curve/rate median bias [-9.200e-05 -1.755e-03] curve/rate median abs [0.012705 0.014546] curve residuals > 0.2 38
held lab 3 rank 9 scaled condition 7.72 curve/rate median bias [ 0.000476 -0.000806] curve/rate median abs [0.016718 0.020963] curve residuals > 0.2 18
channel 0 max stress shift deleting one calibration point 9.126242094270598e-05
channel 1 max stress shift deleting one calibration point 6.395452443239136e-05
channel 2 max stress shift deleting one calibration point 5.175324520809155e-05
channel 3 max stress shift deleting one calibration point 6.81512179227517e-05
regime 0.82 state keys ['aux_chamber_telemetry', 'coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd']
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dccac2a1bac5ee23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dccac2a1 (directory).
```

### Turn 9 — `submit_completed_task` (`15504 ms` LLM)
- **Rationale**: The implemented basis matched the explicit memory recurrence and initial-rate equation and retained full rank in every held-out lab. Removing any one calibration point shifted physical stress by less than 0.0001, and the sparse-calibration and EMI checks passed. The final isolated schema check passed on the unchanged artifact.
- **Alternatives Weighed**: `Fit only rows with complete coupon features` (rejected: It worsened both worst-slice public errors and did not consistently improve leave-one-lab-out performance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_03_solver.py': 'FILE:6656'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed in isolated Python. Worst public lab-by-stratum RMSE was 0.03579 for strain and 0.04589 for initial rate. Errors were largest on rows with missing coupon features.`

### Turn 10 — `execute_in_remote_docker` (`34920 ms` LLM)
- **Rationale**: Refit each ablation inside leave-one-lab-out folds to test whether a simpler basis is hiding a transfer problem. Split residuals by horizon and hold sign to expose cancellation in pooled summaries. If large residuals cluster by specimen rather than time point, evaluate specimen-level robust weighting; otherwise retain the current weighting.
- **Alternatives Weighed**: `Repeat the calibration-point jackknife` (rejected: The previous audit already bounded the maximum stress shift below 0.0001; it did not test the contribution of individual rheology terms.)

```bash
set -e
cat > /tmp/audit_batch03_basis.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c01_batch_03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
s = d['train']
lab = s['lab_id']
n = len(lab)
dt, t = d['past_dt'], d['eval_times']
y = s['strain_curves_obs'] - s['strain_0'][:, None]
r = s['instant_rate_obs']
state = tuple(v[:n] for v in m._prepare(d))
original = m._design

def basis(kind):
    def design(tau, inputs, dt, t):
        a, b = original(tau, inputs, dt, t)
        if kind == 'no cross-gap':
            return a[..., :-1], b[..., :-1]
        if kind == 'no signed slip':
            cols = [0, 1, 2, 3, 4, 5, 6, 8]
            return a[..., cols], b[..., cols]
        if kind == 'linear terminal flow':
            a[..., 2:8] *= (1 + .85*t)[None, :, None]
        return a, b
    return design

for kind in ('full', 'no cross-gap', 'no signed slip', 'linear terminal flow'):
    m._design = basis(kind)
    rows = []
    bias = []
    for c in np.unique(lab):
        train = lab != c
        test = ~train
        tau, w = m._fit(tuple(v[train] for v in state), dt, t, y[train], r[train])
        a, b = m._design(tau, tuple(v[test] for v in state), dt, t)
        e, f = a @ w - y[test], b @ w - r[test]
        edge = np.array_split(np.arange(len(t)), 2)
        rows.append([np.median(np.abs(e[:, j])) for j in edge] + [np.median(np.abs(f))])
        hold = state[1][test]
        for sign in (hold < 0, hold >= 0):
            if np.any(sign):
                bias.append(np.abs([np.median(e[sign]), np.median(f[sign])]))
        if kind == 'full':
            hits = (np.abs(e) > .2).sum(axis=1)
            print('lab', int(c), 'gross curve residuals by specimen', np.bincount(hits, minlength=len(t)+1).tolist())
    print(kind, 'held-lab median absolute errors (early strain, late strain, initial rate)', np.round(rows, 6).tolist(), 'worst', np.round(np.max(rows, axis=0), 6).tolist(), 'max lab-by-hold-sign absolute median biases', np.round(np.max(bias, axis=0), 6).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_batch03_basis.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2537 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 gross curve residuals by specimen [21, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0]
lab 1 gross curve residuals by specimen [32, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0]
lab 2 gross curve residuals by specimen [35, 18, 1, 1, 0, 0, 0, 1, 1, 0, 0]
lab 3 gross curve residuals by specimen [25, 11, 2, 1, 0, 0, 0, 0, 0, 0, 0]
full held-lab median absolute errors (early strain, late strain, initial rate) [[0.01513, 0.013977, 0.010318], [0.010273, 0.014205, 0.016666], [0.012112, 0.013193, 0.014546], [0.013623, 0.021695, 0.020963]] worst [0.01513, 0.021695, 0.020963] max lab-by-hold-sign absolute median biases [0.003923, 0.009706]
no cross-gap held-lab median absolute errors (early strain, late strain, initial rate) [[0.052836, 0.034625, 0.146985], [0.039751, 0.040836, 0.11551], [0.036881, 0.03908, 0.11088], [0.056422, 0.0332, 0.195926]] worst [0.056422, 0.040836, 0.195926] max lab-by-hold-sign absolute median biases [0.018757, 0.22]
no signed slip held-lab median absolute errors (early strain, late strain, initial rate) [[0.049204, 0.076721, 0.081519], [0.043658, 0.076381, 0.074354], [0.043251, 0.067868, 0.073565], [0.035499, 0.058889, 0.057117]] worst [0.049204, 0.076721, 0.081519] max lab-by-hold-sign absolute median biases [0.045333, 0.09464]
linear terminal flow held-lab median absolute errors (early strain, late strain, initial rate) [[0.065476, 0.041129, 0.19201], [0.057274, 0.032884, 0.148971], [0.075706, 0.035767, 0.19037], [0.059677, 0.041817, 0.165031]] worst [0.075706, 0.041817, 0.19201] max lab-by-hold-sign absolute median biases [0.032444, 0.267142]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dccac2a1bac5ee23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dccac2a1 (directory).
```

### Turn 11 — `submit_completed_task` (`20535 ms` LLM)
- **Rationale**: The final audit supported the full prescribed basis: worst held-out-lab median absolute errors were 0.01513 for early strain, 0.02170 for late strain, and 0.02096 for initial rate. The unchanged artifact passed isolated schema verification after the audit.
- **Alternatives Weighed**: `Omit signed slip or cross-gap memory, or use linear terminal flow` (rejected: Each ablation worsened worst held-out-lab median absolute errors for early strain, late strain, and initial rate.); `Downweight whole specimens instead of individual observations` (rejected: Most large curve residuals were isolated within specimens, so whole-specimen weighting would also discard useful observations.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed in isolated Python. Worst public lab-by-stratum RMSE was 0.03579 for strain and 0.04589 for initial rate. Missing coupon features remain the largest source of prediction error.`

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
