# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_07__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_07` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `344,126` (`331,941` / `83,328` / `12,185`) |
| **Total Episode Wall Time (s)** | `394.17s` |

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

### Turn 1 — `execute_in_remote_docker` (`19355 ms` LLM)
- **Rationale**: Inspect the schema, calibration coverage, and archived conventions before fitting. The available labels determine which residuals can be fitted and which held-out slices can independently validate the model.
- **Alternatives Weighed**: `Reuse the preceding batch's model unchanged` (rejected: Batch 07 specifies a new plateau law and requires direct in-batch load-cell calibration when enough clean points are available.)

```bash
cd /holdout_task && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && sed -n '1,200p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_06_solver.py && python3 - <<'PY'
import pickle
from collections import Counter
D=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb'))
def show(d, depth=0):
 for k,v in d.items():
  p='  '*depth; print(p,k,type(v).__name__,getattr(v,'shape',''),end=' ')
  if isinstance(v,dict):
   print();
   if depth<2: show(v,depth+1)
  elif isinstance(v,(list,tuple)):
   print('n=',len(v),'first=',repr(v[0])[:400] if len(v) else None)
   if k=='calibration_load_cells': print(p,Counter(row['lab_id'] for row in v))
  else: print(repr(v)[:180])
show(D)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' || true
```
- **Container Exit Code**: `0` (`412 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
train = dataset['train']
    n = len(train['lab_id'])
    y = np.column_stack((np.asarray(train['strain_curves_obs']) - strain0[:n, None],
                         train['instant_rate_obs']))
    rows = np.flatnonzero(np.isfinite(train['coupon_features_raw']).all(axis=1))
    if len(rows) < 9 or np.linalg.matrix_rank(model.design(np.log([0.7, 5.0]), rows).reshape(-1, 9)) < 9:
        rows = np.arange(n)
    logtau, coeff = _fit(model, y, rows)
    prediction = model.design(logtau, slice(n, None)) @ coeff
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': coeff[:7],
            'regime_mode_weights': coeff[7:], 'predicted_strain': prediction[:, :-1] + strain0[n:, None],
            'initial_rate_pred': prediction[:, -1]}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "<stdin>", line 5, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`24217 ms` LLM)
- **Rationale**: Read the dataset through the list wrapper shown by the validator. Inspect calibration coverage, reference-panel layout, and observed response columns to decide whether archive fallback is needed and which slices can validate the fit.
- **Alternatives Weighed**: `Copy the prior batch’s preprocessing without checking the new data` (rejected: Different calibration coverage or coupon missingness could change which fallback and fitting rows are appropriate.)

```bash
cd /holdout_task && cat /tmp/hegel_continual_archive/alpha_m97_c01_batch_06_solver.py && python3 - <<'PY'
import json,pickle
from collections import Counter
import numpy as np
items=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb'))
print('items:', len(items), 'item keys:', list(items[0]))
d=items[0]['dataset']
def show(d,p=''):
 for k,v in d.items():
  name=p+k
  if isinstance(v,dict):
   print(name, 'dict');show(v,name+'.')
  elif isinstance(v,np.ndarray):
   print(name,v.shape,v.dtype,repr(v.ravel()[:8]), 'nonfinite='+str(np.count_nonzero(~np.isfinite(v))) if np.issubdtype(v.dtype,np.number) else '')
  elif isinstance(v,(list,tuple)):
   print(name,'n=',len(v),'first=',repr(v[0])[:400] if len(v) else None)
  else: print(name,repr(v)[:250])
show(d)
r=d['calibration_load_cells'];print('calibration coverage:',Counter(x['lab_id'] for x in r));print('calibration by lab ranges:',[(l,min(x['true_stress'] for x in r if x['lab_id']==l),max(x['true_stress'] for x in r if x['lab_id']==l)) for l in sorted({x['lab_id'] for x in r})])
a=json.load(open('public_asset_index.json')); print('asset index overview:',list(a) if isinstance(a,dict) else type(a).__name__)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -35
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy IRLS viscoelastic Prony variable projection' | head -12
```
- **Container Exit Code**: `0` (`449 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9d66c4221b48051b07f",
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
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 18, in <module>
  File "/usr/lib/python3.12/collections/__init__.py", line 607, in __init__
    self.update(iterable, **kwds)
  File "/usr/lib/python3.12/collections/__init__.py", line 699, in update
    _count_elements(self, iterable)
  File "<stdin>", line 18, in <genexpr>
TypeError: string indices must be integers, not 'str'
```

### Turn 3 — `execute_in_remote_docker` (`23375 ms` LLM)
- **Rationale**: Read the calibration as a columnar dictionary and isolate the archived preprocessing code. Per-lab calibration and complete-coupon counts determine whether archive fallback is needed and whether held-out lab fits can use complete coupons.
- **Alternatives Weighed**: `Use a control-panel alignment primitive` (rejected: The inspected primitive requires a shared scanner gain and control columns, whereas this task specifies per-lab reference coupon statistics.)

```bash
cd /holdout_task && sed -n '45,145p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_06_solver.py && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb')); print('items',len(items));
for item in items:
 d=item['dataset']; print('item keys',list(item),'dataset keys',list(d))
 for k in d:
  v=d[k]
  if isinstance(v,dict): print(k,{a:np.shape(b) for a,b in v.items()})
  else: print(k,np.shape(v),repr(v)[:150])
 r=d['calibration_load_cells']; l=np.asarray(r['lab_id']); print('calibration counts',np.unique(l,return_counts=True)); print('calibration cmd/true ranges by lab',[(i, *[(np.min(np.asarray(r[k])[l==i]),np.max(np.asarray(r[k])[l==i])) for k in ('cmd_stress','true_stress')]) for i in np.unique(l)])
 for part in ('train','test'):
  s=d[part]; x=np.asarray(s['coupon_features_raw']); print(part,'counts by lab and fully observed',[(int(i),int(np.sum(s['lab_id']==i)),int(np.sum((s['lab_id']==i)&np.isfinite(x).all(axis=1)))) for i in np.unique(s['lab_id'])]); print('shear', np.quantile(s['shear_banding_index'],[0,.5,1]) if 'shear_banding_index' in s else 'absent')
PY
```
- **Container Exit Code**: `0` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ernel = np.exp(-self.age[:, None] / tau) * -np.expm1(-self.dt[:, None] / tau)
        gap = self.hold[indices, None] - self.past[indices] @ kernel
        response = np.vstack((-np.expm1(-self.t[:, None] / tau), 1.0 / tau))
        memory = gap[:, None, :] * response
        exchange = (gap[:, :1] - gap[:, 1:]) * response[:, 0]
        return np.concatenate((memory, self.drift[indices], exchange[:, :, None]), axis=2)


def _fit(model, y, rows):
    y = y[rows]
    ncols = 9
    lower, upper = np.log([0.025, 100.0])

    def optimize(logtau, weights):
        root = np.sqrt(weights.ravel())
        target = y.ravel() * root

        def project(t):
            design = model.design(t, rows).reshape(-1, ncols)
            a = design * root[:, None]
            coeff = np.linalg.lstsq(a, target, rcond=1e-10)[0]
            return coeff, design @ coeff - y.ravel()

        result = least_squares(lambda t: project(t)[1] * root, logtau,
                               bounds=(lower, upper), diff_step=1e-4,
                               max_nfev=65, ftol=1e-9)
items 1
item keys ['dataset', 'gt'] dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
past_dt (18,) array([0.7867, 0.4058, 0.3132, 0.2019, 0.5463, 0.226 , 0.6171, 0.2499,
       0.3886, 0.4643, 0.5796, 0.6558, 0.4157, 0.6134, 0.5506, 0.6883,
       0
eval_times (10,) array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ])
crit_stress () 2.65
yield_stress () 2.65
shear_banding_index () 0.12
calibration_load_cells {'lab_id': (2,), 'cmd_stress': (2,), 'true_stress': (2,)}
train {'lab_id': (180,), 'stratum': (180,), 'coupon_features_raw': (180, 4), 'ref_coupons': (180, 32), 'past_stress_cmd': (180, 18), 'stress_hold_cmd': (180,), 'strain_0': (180,), 'aux_chamber_telemetry': (180, 3), 'strain_curves_obs': (180, 10), 'instant_rate_obs': (180,)}
test {'lab_id': (72,), 'stratum': (72,), 'coupon_features_raw': (72, 4), 'ref_coupons': (72, 32), 'past_stress_cmd': (72, 18), 'stress_hold_cmd': (72,), 'strain_0': (72,), 'aux_chamber_telemetry': (72, 3)}
calibration counts (array([3]), array([2]))
calibration cmd/true ranges by lab [(3, (-1.3, 1.4), (-1.9969954749803027, 1.7559876049385603))]
train counts by lab and fully observed [(0, 52, 46), (1, 49, 47), (2, 37, 35), (3, 42, 36)]
shear absent
test counts by lab and fully observed [(0, 14, 12), (1, 21, 19), (2, 19, 18), (3, 18, 16)]
shear absent
```

### Turn 4 — `execute_in_remote_docker` (`52880 ms` LLM)
- **Rationale**: Build the solver and check the output contract. Train residuals split by crystallization activation will show whether the linear fallback for zero-damage coupons needs revision before held-out validation.
- **Alternatives Weighed**: `Use the rejected gate calibration payload` (rejected: Labs 0–2 require the permitted archive fallback; lab 3 has two direct calibration points, which determine its affine line.)

```bash
set -eu
cd /holdout_task
p=alpha_m97_c01_batch_07_solver.py
sed '/^class _Model:/,$d' /tmp/hegel_continual_archive/alpha_m97_c01_batch_06_solver.py > "$p"
sed -i '1c\"\"\"Two-branch hereditary creep in the strain-crystallized plateau regime.\"\"\"' "$p"
sed -i '/from scipy.optimize/i from scipy.linalg import lstsq' "$p"
cat >> "$p" <<'PY'
class _Model:
    def __init__(self, dataset, past, hold, aligned):
        self.dt = np.asarray(dataset['past_dt'], dtype=float)
        self.t = np.asarray(dataset['eval_times'], dtype=float)
        self.age = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.past, self.hold = past, hold
        excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
        damage = np.sum(excess**2 * self.dt, axis=1)
        self.active = (dataset['shear_banding_index'] <= 0.50) & (damage > 0)
        self.activation = np.where(self.active, np.tanh(damage), 0.0)
        terminal = np.where(self.active[:, None], np.log1p(self.t), self.t)
        self.flow = np.column_stack((terminal, np.ones(len(hold))))
        features = np.column_stack((np.ones(len(hold)), aligned))
        self.drift = self.flow[:, :, None] * features[:, None, :]

    def design(self, logtau, rows=slice(None)):
        tau = np.exp(logtau)
        kernel = np.exp(-self.age[:, None] / tau) * -np.expm1(-self.dt[:, None] / tau)
        state = self.past[rows] @ kernel
        holding = self.hold[rows]
        response = np.vstack((-np.expm1(-self.t[:, None] / tau), 1.0 / tau))
        memory = (holding[:, None] - state)[:, None, :] * response
        activation = self.activation[rows]
        growth = (activation * holding)[:, None] * response[:, 0]
        plateau = (activation * state[:, 0])[:, None] * self.flow[rows]
        return np.concatenate((memory, self.drift[rows], growth[:, :, None], plateau[:, :, None]), axis=2)


def _fit(model, y, rows):
    y = y[rows]
    median = np.median(y, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(y - median), axis=0), 1e-6)
    initial = 1.0 / (1.0 + ((y - median) / (8.0 * spread))**2)

    def fit(seed, weights):
        root = np.sqrt(weights.ravel())
        target = y.ravel() * root

        def project(logtau):
            design = model.design(logtau, rows).reshape(-1, 9)
            coeff = lstsq(design * root[:, None], target, cond=1e-10, lapack_driver='gelsy')[0]
            error = np.einsum('ij,j->i', design, coeff) - y.ravel()
            return coeff, error

        opt = least_squares(lambda t: project(t)[1] * root, seed,
                            bounds=np.log([0.025, 100.0]), diff_step=1e-4,
                            max_nfev=60, ftol=1e-9)
        coeff, error = project(opt.x)
        return opt.x, coeff, error.reshape(y.shape)

    trials = []
    for seed in ([0.35, 3.0], [0.85, 6.0], [2.0, 0.45], [6.0, 0.85]):
        tau, coeff, error = fit(np.log(seed), initial)
        for _ in range(6):
            scale = np.maximum(1.4826 * np.median(np.abs(error - np.median(error, axis=0)), axis=0), 1e-7)
            weights = 1 / (1 + (error / (3 * scale))**2)
            tau, coeff, error = fit(tau, weights)
        central = np.sort(np.abs(error), axis=0)[:max(1, int(0.85 * len(y)))]
        trials.append((np.mean(central**2), tau, coeff))
    ordered = [trial for trial in trials if trial[1][0] < trial[1][1]]
    _, tau, coeff = min(ordered or trials, key=lambda trial: trial[0])
    return tau, coeff


def solve_alpha_m97_c01_batch_07(dataset: dict) -> dict:
    past, hold, aligned, strain0 = _inputs(dataset)
    model = _Model(dataset, past, hold, aligned)
    train = dataset['train']
    n = len(train['lab_id'])
    y = np.column_stack((np.asarray(train['strain_curves_obs']) - strain0[:n, None],
                         train['instant_rate_obs']))
    rows = np.flatnonzero(np.isfinite(train['coupon_features_raw']).all(axis=1))
    if len(rows) < 9 or np.linalg.matrix_rank(model.design(np.log([0.7, 5.0]), rows).reshape(-1, 9)) < 9:
        rows = np.arange(n)
    logtau, coeff = _fit(model, y, rows)
    prediction = model.design(logtau, slice(n, None)) @ coeff
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': coeff[:7],
            'regime_mode_weights': coeff[7:], 'predicted_strain': prediction[:, :-1] + strain0[n:, None],
            'initial_rate_pred': prediction[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_batch_07_solver as s
item=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb'))[0]; d=item['dataset']; gt=item['gt']
p,h,x,z=s._inputs(d); m=s._Model(d,p,h,x); result=s.solve_alpha_m97_c01_batch_07(d)
print('calibration',s._calibrate(d)); print('active train/test',m.active[:180].sum(),m.active[180:].sum())
print('gt metadata', {k:np.shape(v) for k,v in gt.items()} if isinstance(gt,dict) else type(gt)); print('fit', {k:v for k,v in result.items() if v.size<=9})
y=np.column_stack((d['train']['strain_curves_obs']-z[:180,None],d['train']['instant_rate_obs']))
w=np.r_[result['hereditary_weights'],result['regime_mode_weights']]; err=m.design(np.log(result['estimated_taus']),slice(180))@w-y
for name,mask in [('active',m.active[:180]),('inactive',~m.active[:180])]:
 if mask.any(): print(name, 'median and p90 abs error',np.quantile(np.abs(err[mask]),[.5,.9],axis=0).round(5))
PY
```
- **Container Exit Code**: `0` (`1044 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[ 3.53047619  2.12142857]
 [ 0.72004539  0.18001838]
 [ 1.13857143 -4.76285714]
 [ 1.38999373 -0.19000362]]
active train/test 148 58
gt metadata {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
fit {'estimated_taus': array([0.64470602, 2.84248851]), 'hereditary_weights': array([ 0.61712291,  0.24349547,  0.14314057,  0.10563649, -0.07154949,
        0.08280042, -0.03986873]), 'regime_mode_weights': array([-0.18356959,  0.15587196])}
active median and p90 abs error [[0.19973 0.40795 0.48713 0.53634 0.57029 0.5624  0.59695 0.59932 0.66721
  0.57357 0.77786]
 [0.54545 0.94112 1.09694 1.18756 1.28716 1.32207 1.36668 1.35917 1.62387
  1.50768 2.20647]]
inactive median and p90 abs error [[0.14463 0.25442 0.27451 0.28391 0.26459 0.36978 0.33302 0.45257 0.42899
  0.47011 0.74113]
 [0.20685 0.35714 0.41144 0.52938 0.60406 0.70581 0.789   1.02207 1.01723
  1.02593 0.95488]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e61e5c2b229ab878 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e61e5c2b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25778 ms` LLM)
- **Rationale**: Compare the sparse fallback lines with independently supported archive sweeps and localize residuals by lab. If sparse lines disagree with repeated clean sweeps, replace the latest-line fallback with a robust archive estimate before testing the constitutive model.
- **Alternatives Weighed**: `Change the plateau law first` (rejected: Stress calibration changes both hereditary memory and the damage gate, so the constitutive law cannot be judged reliably until calibration is resolved.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json,pickle,numpy as np
import alpha_m97_c01_batch_07_solver as s
hist=json.loads(s._HISTORY.read_text())
for j,r in enumerate(hist):
 lab,x,y=(np.asarray(r[k]) for k in ('lab_id','cmd_stress','true_stress'))
 print('sweep',j,'metadata',{k:v for k,v in r.items() if k not in ('lab_id','cmd_stress','true_stress')})
 for c in range(4):
  a,b=x[lab==c],y[lab==c]
  if len(a):
   clean,emi=s._clean_line(a,b)
   fit=s._affine(a[clean],b[clean]).round(6) if clean.sum()>=2 and np.ptp(a[clean])>0 else None
   print(' lab',c,'rows',len(a),'accepted',clean.sum(),'fit',fit,'cmd/true',np.column_stack((a,b)).round(5).tolist() if len(a)<=3 else '')
d=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb'))[0]['dataset']; p,h,x,z=s._inputs(d); m=s._Model(d,p,h,x); o=s.solve_alpha_m97_c01_batch_07(d); n=len(d['train']['lab_id']); w=np.r_[o['hereditary_weights'],o['regime_mode_weights']]; e=m.design(np.log(o['estimated_taus']),slice(n))@w-np.column_stack((d['train']['strain_curves_obs']-z[:n,None],d['train']['instant_rate_obs']))
for c in range(4):
 k=d['train']['lab_id']==c
 print('lab',c,'active fraction',m.active[:n][k].mean().round(3),'curve abs p90',round(float(np.quantile(abs(e[k,:-1]),.9)),4),'rate abs p90',round(float(np.quantile(abs(e[k,-1]),.9)),4))
PY
```
- **Container Exit Code**: `0` (`741 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
00e-06] cmd/true 
 lab 2 rows 10 accepted 10 fit [ 1.000012e+00 -4.600000e-05] cmd/true 
 lab 3 rows 10 accepted 10 fit [1.000005e+00 5.100000e-05] cmd/true 
sweep 1 metadata {'episode_short_name': 'alpha_m97_c01_batch_01'}
 lab 0 rows 10 accepted 10 fit [1.339999 0.149994] cmd/true 
 lab 1 rows 10 accepted 10 fit [ 0.770004 -0.169994] cmd/true 
 lab 2 rows 10 accepted 10 fit [ 1.210027 -0.120003] cmd/true 
 lab 3 rows 10 accepted 10 fit [0.83001  0.109973] cmd/true 
sweep 2 metadata {'episode_short_name': 'alpha_m97_c01_batch_02'}
 lab 0 rows 10 accepted 10 fit [1.34003  0.149977] cmd/true 
 lab 1 rows 10 accepted 10 fit [ 0.770006 -0.169977] cmd/true 
 lab 2 rows 10 accepted 10 fit [ 1.209998 -0.120013] cmd/true 
 lab 3 rows 10 accepted 10 fit [0.829939 0.110089] cmd/true 
sweep 3 metadata {'episode_short_name': 'alpha_m97_c01_batch_03'}
 lab 0 rows 10 accepted 10 fit [1.339989 0.150001] cmd/true 
 lab 1 rows 10 accepted 10 fit [ 0.770039 -0.170026] cmd/true 
 lab 2 rows 10 accepted 10 fit [ 1.209981 -0.119991] cmd/true 
 lab 3 rows 10 accepted 10 fit [0.830001 0.110012] cmd/true 
sweep 4 metadata {'episode_short_name': 'alpha_m97_c01_batch_04'}
 lab 3 rows 2 accepted 2 fit [0.830071 0.11003 ] cmd/true [[-1.3, -0.96906], [1.4, 1.27213]]
sweep 5 metadata {'episode_short_name': 'alpha_m97_c01_batch_05'}
 lab 0 rows 10 accepted 10 fit [1.34003  0.149977] cmd/true 
 lab 1 rows 10 accepted 10 fit [ 0.770006 -0.169977] cmd/true 
 lab 2 rows 10 accepted 10 fit [ 1.209998 -0.120013] cmd/true 
 lab 3 rows 10 accepted 10 fit [0.829939 0.110089] cmd/true 
sweep 6 metadata {'episode_short_name': 'alpha_m97_c01_batch_06'}
 lab 0 rows 3 accepted 2 fit [3.530476 2.121429] cmd/true [[-1.6, 2.906], [1.2, 6.358], [-0.9, -1.056]]
 lab 1 rows 12 accepted 12 fit [0.720045 0.180018] cmd/true 
 lab 2 rows 3 accepted 2 fit [ 1.138571 -4.762857] cmd/true [[1.5, -3.055], [-1.3, -6.243], [1.1, 1.211]]
 lab 3 rows 12 accepted 12 fit [ 1.390048 -0.190014] cmd/true 
lab 0 active fraction 1.0 curve abs p90 1.0879 rate abs p90 1.138
lab 1 active fraction 0.673 curve abs p90 1.2271 rate abs p90 2.2663
lab 2 active fraction 1.0 curve abs p90 1.4804 rate abs p90 2.7642
lab 3 active fraction 0.619 curve abs p90 1.1204 rate abs p90 2.1208

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e61e5c2b229ab878 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e61e5c2b (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29242 ms` LLM)
- **Rationale**: Run leave-one-lab-out validation after correcting the unsupported fallback. If large residuals persist on complete coupons, inspect the regime basis or response noise; if they concentrate on incomplete coupons, inspect imputation instead.
- **Alternatives Weighed**: `Pool all archive sweeps` (rejected: Pooling would obscure the well-supported recent gain and bias change for lab 1.)
- **ATMS NOGOOD DECLARED**: Refuted `The latest archive sweep that yields an affine fit provides a reliable transferable calibration.` | Witness: `The latest lab 0 and lab 2 sweeps each lose one of three points to the EMI filter, leaving lines with biases +2.12143 and -4.76286 instead of the repeatedly supported +0.15 and -0.12; both labs then classify every training coupon as damaged.` | Blanket Diagnosis: `With only two retained points, the fitted line has no independent residual check. Its error propagates through both branch memory and the crystallization gate.` | Sublated Pivot: `Keep direct in-batch calibration and the exact constitutive basis. For archive fallback, prefer the latest sweep with four or more accepted points, retaining a sparse non-EMI fallback only when no such sweep exists.`

```bash
set -eu
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_07_solver.py')
s = p.read_text()
a = s.index('        for sweep in reversed(history):')
b = s.index("        raise ValueError(f'No usable calibration sweep", a)
s = s[:a] + '''        for minimum in (4, 2):
            for sweep in reversed(history):
                take = np.asarray(sweep['lab_id']) == channel
                x, y = (np.asarray(sweep[k])[take] for k in ('cmd_stress', 'true_stress'))
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                if len(x) < minimum:
                    continue
                clean, emi = _clean_line(x, y)
                if clean.sum() >= minimum and (minimum == 4 or not emi) and np.ptp(x[clean]) > 0:
                    return _affine(x[clean], y[clean])
''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_batch_07_solver as s
d=pickle.load(open('public_alpha_m97_c01_batch_07.pkl','rb'))[0]['dataset']; tr=d['train']; n=len(tr['lab_id'])
p,h,x,z=s._inputs(d); m=s._Model(d,p,h,x)
y=np.column_stack((tr['strain_curves_obs']-z[:n,None],tr['instant_rate_obs']))
lab=tr['lab_id']; full=np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('calibration',s._calibrate(d)); print('active train/test',m.active[:n].sum(),m.active[n:].sum())
for c in [-1,0,1,2,3]:
 rows=np.flatnonzero(full & (lab!=c)); tau,w=s._fit(m,y,rows)
 test=np.arange(n) if c==-1 else np.flatnonzero(lab==c)
 err=m.design(tau,test)@w-y[test]
 print('fold',c,'tau',np.exp(tau).round(6),'weights',w.round(6))
 for gate in [False,True]:
  for complete in [False,True]:
   mask=(m.active[test]==gate)&(full[test]==complete)
   if mask.any():
    print(' active',gate,'complete',complete,'n',mask.sum(),'median/p90 curve',np.quantile(abs(err[mask,:-1]),[.5,.9]).round(5),'rate',np.quantile(abs(err[mask,-1]),[.5,.9]).round(5))
PY
```
- **Container Exit Code**: `0` (`1800 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_07_solver.py']`
- **Stdout (tail)**:
```text
]
 active False complete True n 52 median/p90 curve [0.04816 0.37316] rate [0.08924 0.1786 ]
 active True complete False n 8 median/p90 curve [0.15466 0.41443] rate [0.12512 0.25229]
 active True complete True n 112 median/p90 curve [0.08522 0.25847] rate [0.0732  0.15405]
fold 0 tau [0.503174 3.67205 ] weights [ 0.671894  0.558865  0.082794  0.09227  -0.076532  0.0697   -0.04
  0.258476  0.206025]
 active False complete False n 3 median/p90 curve [0.12124 0.32458] rate [0.06327 0.10245]
 active False complete True n 13 median/p90 curve [0.0465  0.29791] rate [0.09841 0.16264]
 active True complete False n 3 median/p90 curve [0.12199 0.37187] rate [0.11237 0.24884]
 active True complete True n 33 median/p90 curve [0.09412 0.29049] rate [0.0979  0.21928]
fold 1 tau [0.506235 3.48568 ] weights [ 0.676752  0.535186  0.086537  0.104674 -0.077094  0.07567  -0.041214
  0.250624  0.247448]
 active False complete False n 1 median/p90 curve [0.14796 0.16799] rate [0.18404 0.18404]
 active False complete True n 15 median/p90 curve [0.0674  0.43973] rate [0.0738  0.16421]
 active True complete False n 1 median/p90 curve [0.11521 0.15265] rate [0.10335 0.10335]
 active True complete True n 32 median/p90 curve [0.0899  0.24606] rate [0.07631 0.13373]
fold 2 tau [0.507992 3.767307] weights [ 0.684771  0.537233  0.089073  0.101555 -0.081434  0.072161 -0.043878
  0.252231  0.229684]
 active False complete False n 1 median/p90 curve [0.3     0.62366] rate [0.01499 0.01499]
 active False complete True n 11 median/p90 curve [0.06274 0.414  ] rate [0.11317 0.1787 ]
 active True complete False n 1 median/p90 curve [0.22808 0.29949] rate [0.1265 0.1265]
 active True complete True n 24 median/p90 curve [0.07422 0.22523] rate [0.06181 0.12939]
fold 3 tau [0.525144 5.024722] weights [ 0.726843  0.59886   0.082463  0.101584 -0.067571  0.083986 -0.041037
  0.254705  0.260781]
 active False complete False n 3 median/p90 curve [0.2362  0.49285] rate [0.2248  0.24883]
 active False complete True n 13 median/p90 curve [0.0475  0.42291] rate [0.03897 0.17479]
 active True complete False n 3 median/p90 curve [0.21748 0.47733] rate [0.21853 0.28921]
 active True complete True n 23 median/p90 curve [0.11073 0.37162] rate [0.08253 0.1536 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8a66dbc0c7305784 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8a66dbc0 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28131 ms` LLM)
- **Rationale**: The well-supported archive fallback corrected the implausible activation pattern. Compare linear and logarithmic flow on zero-damage coupons, and use active-only fitting to check whether inactive observations destabilize the slower relaxation time. Revise the flow law only if the held-out residuals show a consistent improvement.
- **Alternatives Weighed**: `Tune weights against the largest individual residuals` (rejected: Several worst intersections contain only one coupon; signed errors by horizon on complete coupons better distinguish a flow-law mismatch from observation noise or imputation error.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
import alpha_m97_c01_batch_07_solver as s
with open('public_alpha_m97_c01_batch_07.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; lab = tr['lab_id']; n = len(lab)
p,h,x,z = s._inputs(d)
y = np.column_stack((tr['strain_curves_obs']-z[:n,None],tr['instant_rate_obs']))
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
features = np.column_stack((np.ones(len(h)),x))
for variant in ('gated_linear','global_log','active_fit_only'):
    m = s._Model(d,p,h,x)
    if variant == 'global_log':
        m.flow[:] = np.r_[np.log1p(m.t),1.0]
        m.drift = m.flow[:,:,None]*features[:,None,:]
    active = m.active[:n]
    oof = np.empty_like(y)
    taus = []
    for c in np.unique(lab):
        rows = np.flatnonzero(complete & (lab != c) & (active if variant == 'active_fit_only' else True))
        lt,w = s._fit(m,y,rows)
        taus.append(np.exp(lt))
        test = np.flatnonzero(lab == c)
        oof[test] = m.design(lt,test)@w-y[test]
    print(variant,'fold taus',np.round(taus,4).tolist())
    for gate in (False,True):
        take = complete & (active == gate)
        e = oof[take]
        print(' active',gate,'n',len(e),'signed horizon medians',np.median(e,axis=0).round(4),'abs curve/rate median',np.median(abs(e[:,:-1])).round(4),np.median(abs(e[:,-1])).round(4),'worst lab p90 curve/rate',np.max([np.quantile(abs(oof[take & (lab == c),:-1]),.9) for c in np.unique(lab)]).round(4),np.max([np.quantile(abs(oof[take & (lab == c),-1]),.9) for c in np.unique(lab)]).round(4))
PY
```
- **Container Exit Code**: `0` (`2482 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
gated_linear fold taus [[0.5032, 3.672], [0.5062, 3.4857], [0.508, 3.7673], [0.5251, 5.0247]]
 active False n 52 signed horizon medians [-0.0083 -0.0098 -0.0025 -0.001   0.0242  0.0591  0.0835  0.0892  0.141
  0.0953 -0.0323] abs curve/rate median 0.0511 0.0779 worst lab p90 curve/rate 0.4397 0.1787
 active True n 112 signed horizon medians [-0.0183 -0.0315 -0.044  -0.0611 -0.0654 -0.0805 -0.0722 -0.0955 -0.1041
 -0.0974 -0.0578] abs curve/rate median 0.0917 0.083 worst lab p90 curve/rate 0.3716 0.2193
global_log fold taus [[0.5141, 3.6717], [0.5135, 3.633], [0.5141, 3.6607], [0.5149, 3.6546]]
 active False n 52 signed horizon medians [-0.0001  0.0006  0.0016 -0.0075  0.0058  0.0023  0.0035  0.0025  0.0028
 -0.0029  0.0013] abs curve/rate median 0.013 0.0135 worst lab p90 curve/rate 0.0437 0.0313
 active True n 112 signed horizon medians [-0.0013  0.0003 -0.0001 -0.0043 -0.0011 -0.0004 -0.0003 -0.0009 -0.0033
 -0.001   0.002 ] abs curve/rate median 0.0128 0.0141 worst lab p90 curve/rate 0.0443 0.0403
active_fit_only fold taus [[0.5132, 3.6518], [0.5132, 3.6547], [0.5135, 3.6475], [0.5145, 3.6314]]
 active False n 52 signed horizon medians [0.0064 0.0067 0.0479 0.0373 0.1038 0.2077 0.1566 0.1673 0.24   0.2318
 0.0038] abs curve/rate median 0.181 0.0132 worst lab p90 curve/rate 1.0652 0.0321
 active True n 112 signed horizon medians [-0.0012 -0.0001  0.0006 -0.0046  0.      0.0006  0.0022  0.0003 -0.003
 -0.0003  0.0026] abs curve/rate median 0.0126 0.0145 worst lab p90 curve/rate 0.0434 0.0419

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8a66dbc0c7305784 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8a66dbc0 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`36093 ms` LLM)
- **Rationale**: Promote the baseline that removes the held-out horizon bias without adding parameters. Verify the exact saved solver in an isolated process, then check public clean strain and initial-rate targets separately by lab, activation, and missingness, along with parameter and stress reconstruction errors. Those checks distinguish preprocessing errors from fit or imputation errors.
- **Alternatives Weighed**: `Fit active coupons only and retain linear extrapolation` (rejected: It fits active coupons but produces much larger held-out strain errors on inactive coupons, even with complete features.)
- **ATMS NOGOOD DECLARED**: Refuted `Outside the active damage gate, terminal drift must revert to linear time.` | Witness: `On complete inactive leave-one-lab-out coupons, a linear baseline produced worst-lab strain p90 of 0.4397, versus 0.0437 for the logarithmic baseline. Active-only fitting with linear inactive extrapolation worsened that metric to 1.0652.` | Blanket Diagnosis: `The damage gate modulates the two activated corrections, not the logarithmic baseline. Imposing a linear baseline on inactive rows also biases the shared fit on active rows.` | Sublated Pivot: `Retain the two-branch memory, calibrated stresses, and exact damage gate, while applying logarithmic baseline drift throughout batch 07.`

```bash
set -eu
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_07_solver.py')
s = p.read_text()
old = '        terminal = np.where(self.active[:, None], np.log1p(self.t), self.t)\n        self.flow = np.column_stack((terminal, np.ones(len(hold))))'
new = '        # The plateau baseline applies throughout this campaign; damage gates only the two corrections.\n        self.flow = np.broadcast_to(np.r_[np.log1p(self.t), 1.0], (len(hold), len(self.t) + 1))'
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_m97_c01_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, gt = item['dataset'], item['gt']
    o = s.solve_alpha_m97_c01_batch_07(d)
    p,h,x,z = s._inputs(d)
    m = s._Model(d,p,h,x)
    for split in ('train','test'):
        q = d[split]
        n = len(d['train']['lab_id'])
        lab = np.asarray(q['lab_id'])
        complete = np.isfinite(q['coupon_features_raw']).all(axis=1)
        if split == 'train':
            y = np.column_stack((q['strain_curves_obs']-z[:n,None],q['instant_rate_obs']))
            e = np.empty_like(y)
            for c in np.unique(lab):
                t,w = s._fit(m,y,np.flatnonzero(complete & (lab != c)))
                ids = np.flatnonzero(lab == c)
                e[ids] = m.design(t,ids) @ w - y[ids]
            active = m.active[:n]
        else:
            e = np.column_stack((o['predicted_strain']-gt['te_clean'],o['initial_rate_pred']-gt['te_true_rate0']))
            active = m.active[n:]
        assert np.isfinite(e).all()
        for full in (True, False):
            slices = [(int(c), a, (lab == c) & (active == a) & (complete == full)) for c in np.unique(lab) for a in (False,True)]
            summaries = [(c,a,int(k.sum()),float(np.quantile(abs(e[k,:-1]),.9)),float(np.quantile(abs(e[k,-1]),.9))) for c,a,k in slices if k.any()]
            if summaries:
                print(split,'complete',full,'worst strain/rate slices (lab, active, count, strain p90, rate p90)',max(summaries,key=lambda v:v[3]),max(summaries,key=lambda v:v[4]))
    for got,key in (('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')):
        print(got,'estimate',o[got].round(6),'public reference',np.asarray(gt[key]).round(6),'max abs error',float(np.max(abs(o[got]-gt[key]))))
    print('stress history/hold max abs error',float(np.max(abs(p[n:]-gt['te_hist_true']))),float(np.max(abs(h[n:]-gt['te_hold_true']))))
PY
sed -n '1,118p' alpha_m97_c01_batch_07_solver.py
```
- **Container Exit Code**: `0` (`1842 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_07_solver.py']`
- **Stdout (tail)**:
```text
hannel] = _affine(x[clean], y[clean])
            continue
        if len(x) < 2 or emi:
            gain, bias = prior(channel)
            # The archived sweep supplies an absolute reference: an EMI burst
            # can move most of a sparse sweep in the same direction.
            clean = np.abs(y - (gain * x + bias)) <= 3.0
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            coefficients[channel] = _affine(x[clean], y[clean])
        elif len(x) < 2 or emi:
            if clean.any():
                bias = np.median(y[clean] - gain * x[clean])
            coefficients[channel] = gain, bias
        else:
            raise ValueError(f'Calibration commands do not vary for lab {channel}')
    return coefficients


def _inputs(dataset):
    def stack(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        take = lab == channel
        center = np.nanmedian(np.nanmean(ref[take], axis=0))
        scale = np.nanmedian(np.nanstd(ref[take], axis=0, ddof=0))
        aligned[take] = (raw[take] - center) / scale

    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = []
    for channel, group in cells:
        values = aligned[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(raw.shape[1], np.nan), where=counts > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dataset, past, hold, aligned):
        self.dt = np.asarray(dataset['past_dt'], dtype=float)
        self.t = np.asarray(dataset['eval_times'], dtype=float)
        self.age = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.past, self.hold = past, hold

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1aedf91e000ea598 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_1aedf91e (directory).
```

### Turn 9 — `execute_in_remote_docker` (`33119 ms` LLM)
- **Rationale**: Check the saved solver against a step-by-step memory recurrence and the analytical initial slope. This can detect a state-update or damage-gating error that fit residuals alone would not expose. Report active and inactive test errors separately so missing-feature error is not mistaken for a failure of the plateau law.
- **Alternatives Weighed**: `Tune imputation against the public clean test targets` (rejected: The task specifies cell-balanced imputation; fitting against test targets would bias validation and would not recover the missing coupon measurements.)

```bash
set -eu
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_m97_c01_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, gt = item['dataset'], item['gt']
    out = s.solve_alpha_m97_c01_batch_07(d)
    tau, hw, rw = (out[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights'))
    assert 0 < tau[0] < tau[1]
    p, h, x, z0 = s._inputs(d)
    n = len(d['train']['lab_id'])
    p, h, x, z0 = p[n:], h[n:], x[n:], z0[n:]
    t = np.asarray(d['eval_times'])
    z = np.zeros((len(h), 2))
    damage = np.zeros(len(h))
    for stress, dt in zip(p.T, d['past_dt']):
        a = np.exp(-dt / tau)
        z = z * a + stress[:, None] * (1 - a)
        damage += np.maximum(0, np.abs(stress) - d['crit_stress'])**2 * dt
    active = (damage > 0) & (d['shear_banding_index'] <= 0.50)
    act = np.where(active, np.tanh(damage), 0)
    amps = (h[:, None] - z) * hw[:2]
    amps[:, 0] += rw[0] * act * h
    flow = hw[2] + x @ hw[3:] + rw[1] * act * z[:, 0]
    curve = z0[:, None] + amps @ (1 - np.exp(-t[None, :] / tau[:, None])) + flow[:, None] * np.log1p(t)
    rate = amps @ (1 / tau) + flow
    np.testing.assert_allclose(out['predicted_strain'], curve, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out['initial_rate_pred'], rate, rtol=0, atol=1e-12)
    complete = np.isfinite(d['test']['coupon_features_raw']).all(axis=1)
    ec = np.abs(curve - gt['te_clean'])
    er = np.abs(rate - gt['te_true_rate0'])
    for gate in (False, True):
        for full in (False, True):
            mask = (active == gate) & (complete == full)
            if mask.any():
                print('active', gate, 'complete', full, 'n', mask.sum(), 'strain/rate MAE', ec[mask].mean(), er[mask].mean())
    print('all-test strain/rate MAE', ec.mean(), er.mean())
    print('independent recurrence max strain/rate difference', np.max(np.abs(out['predicted_strain']-curve)), np.max(np.abs(out['initial_rate_pred']-rate)))
PY
```
- **Container Exit Code**: `0` (`988 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
active False complete False n 4 strain/rate MAE 0.14980137877223892 0.12437065691979848
active False complete True n 20 strain/rate MAE 0.005034034078504419 0.0039247733393764726
active True complete False n 3 strain/rate MAE 0.20676329052864867 0.17115873757562214
active True complete True n 45 strain/rate MAE 0.004737468172359581 0.004327472540961217
all-test strain/rate MAE 0.0212966963333496 0.017835980160233952
independent recurrence max strain/rate difference 1.3322676295501878e-15 1.7763568394002505e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1aedf91e000ea598 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_1aedf91e (directory).
```

### Turn 10 — `submit_completed_task` (`17517 ms` LLM)
- **Rationale**: The saved solver passed isolated schema and scientific verification. Held-out lab residuals support the logarithmic baseline, and the independent recurrence confirms the strain and rate calculations. No further edits are warranted.
- **Alternatives Weighed**: `Tune imputation against the public test targets` (rejected: The task specifies cell-balanced imputation, and using test targets would compromise independent validation.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_07_solver.py': 'FILE:8447'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_07_solver.py`. The public schema passed, and an independent recurrence matched both predictions to within 2 × 10⁻¹⁵. Public test MAE was 0.0213 for strain and 0.0178 for initial rate. Errors were larger on coupons with missing features.`

### Turn 11 — `execute_in_remote_docker` (`46220 ms` LLM)
- **Rationale**: Audit whether the contracted basis is identifiable, whether fixed pairwise, three-way, or rational corrections explain held-out residuals, and whether lab-centered telemetry adds transferable signal. Separately check whether observed calibration error is large enough to change the damage gate. Only a material failure would warrant changing the saved solver.
- **Alternatives Weighed**: `Apply classification, relay-mode, or Fokker–Planck/HJB tests` (rejected: Batch 07 has continuous strain and rate targets and a closed-form constitutive model, not class labels, a relay state, or a stochastic control operator. Within-lab signed residuals and held-out-lab errors test the relevant confounding risk.)

```bash
set -eu
cd /holdout_task
cat > /tmp/audit_alpha_m97_c01_b07.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_m97_c01_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d = item['dataset']
    tr = d['train']
    lab = np.asarray(tr['lab_id'])
    n = len(lab)
    p, h, x, start = s._inputs(d)
    m = s._Model(d, p, h, x)
    y = np.column_stack((tr['strain_curves_obs'] - start[:n, None], tr['instant_rate_obs']))
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    gate = m.active[:n]
    errors = {name: np.empty_like(y) for name in ('exact', 'algebraic_extensions', 'centered_telemetry')}
    tele = np.asarray(tr['aux_chamber_telemetry'], float)
    tele = tele - np.vstack([np.median(tele[lab == c], axis=0) for c in lab])
    for c in np.unique(lab):
        fit = np.flatnonzero(complete & (lab != c))
        val = np.flatnonzero(lab == c)
        lt, w = s._fit(m, y, fit)
        a = m.design(lt, slice(n))
        err = a @ w - y
        errors['exact'][val] = err[val]
        gap = a[:, -1, 0] * np.exp(lt[0])
        flow = m.flow[:n]
        pairs = np.column_stack([(x[:n, i] - x[:n, j])**2 for i in range(4) for j in range(i + 1, 4)])
        triple = (x[:n, 0] * x[:n, 1] * x[:n, 2])[:, None]
        frac = (gap / (1 + h[:n]**2))[:, None]
        proposals = {'algebraic_extensions': np.column_stack((pairs, triple, frac)), 'centered_telemetry': tele}
        core = a[fit].reshape(-1, 9)
        print('held-out lab', c, 'core rank', np.linalg.matrix_rank(core), 'taus', np.exp(lt).round(5))
        for name, features in proposals.items():
            add = flow[:, :, None] * features[:, None, :]
            mtx = np.concatenate((a, add), axis=2)
            ef = err[fit]
            sd = np.maximum(1.4826 * np.median(abs(ef - np.median(ef, axis=0)), axis=0), 1e-7)
            root = (1 + (ef / (3 * sd))**2)**-0.5
            z = mtx[fit].reshape(-1, mtx.shape[-1]) * root.reshape(-1, 1)
            coeff = np.linalg.lstsq(z, (y[fit] * root).ravel(), rcond=1e-10)[0]
            errors[name][val] = mtx[val] @ coeff - y[val]
    for name, e in errors.items():
        for full in (True, False):
            rows = complete == full
            slices = [rows & (lab == c) & (gate == g) for c in np.unique(lab) for g in (False, True)]
            slice_scores = [(np.quantile(abs(e[k,:-1]), .9), np.quantile(abs(e[k,-1]), .9)) for k in slices if k.any()]
            if slice_scores:
                print(name, 'complete', full, 'worst curve/rate p90', np.max(slice_scores, axis=0).round(6), 'pooled MAE', np.array([abs(e[rows,:-1]).mean(), abs(e[rows,-1]).mean()]).round(6))
    base = errors['exact']
    for c in np.unique(lab):
        for g in (False, True):
            k = complete & (lab == c) & (gate == g)
            if k.any():
                print('exact signed lab/gate residual', c, g, k.sum(), 'last-horizon/rate', np.median(base[k][:, [-2, -1]], axis=0).round(6))
    gt = item['gt']
    stress_error = np.max(abs(p[n:] - gt['te_hist_true']))
    max_stress = np.max(abs(p[n:]), axis=1)
    threshold = float(d['crit_stress'])
    print('public stress reconstruction max error', stress_error, 'test coupons within that error of damage threshold', np.sum(abs(max_stress - threshold) <= stress_error))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_alpha_m97_c01_b07.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out lab 0 core rank 9 taus [0.51407 3.67169]
held-out lab 1 core rank 9 taus [0.51352 3.63304]
held-out lab 2 core rank 9 taus [0.5141  3.66069]
held-out lab 3 core rank 9 taus [0.51488 3.65462]
exact complete True worst curve/rate p90 [0.044334 0.04028 ] pooled MAE [0.111604 0.016278]
exact complete False worst curve/rate p90 [0.501608 0.227575] pooled MAE [0.314321 0.126161]
algebraic_extensions complete True worst curve/rate p90 [0.043778 0.040636] pooled MAE [0.111736 0.016218]
algebraic_extensions complete False worst curve/rate p90 [0.501607 0.227129] pooled MAE [0.313906 0.125849]
centered_telemetry complete True worst curve/rate p90 [0.044872 0.039729] pooled MAE [0.111587 0.016323]
centered_telemetry complete False worst curve/rate p90 [0.502008 0.225808] pooled MAE [0.31408  0.126023]
exact signed lab/gate residual 0 False 13 last-horizon/rate [-0.01639  -0.011373]
exact signed lab/gate residual 0 True 33 last-horizon/rate [-0.00236   0.000132]
exact signed lab/gate residual 1 False 15 last-horizon/rate [0.00289  0.008525]
exact signed lab/gate residual 1 True 32 last-horizon/rate [0.007959 0.003471]
exact signed lab/gate residual 2 False 11 last-horizon/rate [-0.001218  0.010079]
exact signed lab/gate residual 2 True 24 last-horizon/rate [-0.00524   0.000738]
exact signed lab/gate residual 3 False 13 last-horizon/rate [ 0.000579 -0.002547]
exact signed lab/gate residual 3 True 23 last-horizon/rate [-0.005794  0.010162]
public stress reconstruction max error 0.00025845199065654967 test coupons within that error of damage threshold 0
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1aedf91e000ea598 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_1aedf91e (directory).
```

### Turn 12 — `submit_completed_task` (`15835 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema and recurrence checks. The fitted basis had full rank in every held-out lab fold, and no public test coupon was close enough to the damage threshold for the observed calibration error to change its activation. The audit found no reason to change the solver.
- **Alternatives Weighed**: `Add algebraic corrections or lab-centered telemetry` (rejected: Neither challenger materially improved held-out strain and rate together or addressed the larger errors on incomplete coupons.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_07_solver.py`. The public schema passed. Public test MAE was 0.0213 for strain and 0.0178 for initial rate. Errors were larger on coupons with missing features.`

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
