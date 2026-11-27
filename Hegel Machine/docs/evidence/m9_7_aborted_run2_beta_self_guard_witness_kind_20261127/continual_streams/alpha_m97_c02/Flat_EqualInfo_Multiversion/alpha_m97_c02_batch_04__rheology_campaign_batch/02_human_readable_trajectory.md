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
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `436,014` (`423,030` / `105,600` / `12,984`) |
| **Total Episode Wall Time (s)** | `483.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`17801 ms` LLM)
- **Rationale**: Inspect field names, shapes, and archive metadata before coding so calibration fallback, panel alignment, and phase selection use the published inputs. Print only bounded summaries to keep data transfer small.
- **Alternatives Weighed**: `Reuse the previous batch solver unchanged` (rejected: The current batch specifies a different phase law; inspect the schema and archive first to identify reusable calibration and memory code.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && cat /tmp/hegel_continual_archive/archive_manifest.json && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)
def show(obj,depth=0):
 if isinstance(obj,dict) and depth<3:
  for k,v in obj.items():
   print(' '*depth,k,end=': ');show(v,depth+1)
 elif isinstance(obj,np.ndarray):
  print(obj.shape,obj.dtype,obj.ravel()[:5])
 elif isinstance(obj,(list,tuple)):
  print(type(obj).__name__,len(obj),str(obj[:1])[:350])
 else: print(str(obj)[:250])
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation Prony variable projection'
```
- **Container Exit Code**: `0` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2_batch_03__rheology_campaign_batch",
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
]list 1 [{'dataset': {'past_dt': array([0.7303, 0.4761, 0.699 , 0.3755, 0.2141, 0.3417, 0.556 , 0.323 ,
       0.4575, 0.2982, 0.5806, 0.2753, 0.3471, 0.5171, 0.6661, 0.4308,
       0.4055, 0.3143]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shea
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`17369 ms` LLM)
- **Rationale**: The first listing omitted nested field names and calibration coverage. Those determine the training design and which channels may use archived sweeps; inspect only the necessary schema and prior implementation.

```bash
python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_04.pkl','rb'))[0]
print('item keys',list(p)); d=p['dataset']
for k,v in d.items():
 if isinstance(v,dict):
  print(k,[(a,np.shape(b)) for a,b in v.items()]);
  if k=='calibration_load_cells': print(v)
 else: print(k,np.shape(v),v if np.size(v)<=20 else '')
PY
ls -lh /tmp/hegel_continual_archive; head -c 2200 /tmp/hegel_continual_archive/archive_calibration_history.json; printf '\n'; sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py
```
- **Container Exit Code**: `0` (`410 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
a[:, :, 8] = delta[:, None]*growth[:, 0]
    b = np.column_stack((gap/tau, flow, delta/tau[0]))
    return a, b


def _fit(dt, t, inputs, curves, rates):
    n, k = curves.shape
    y = np.r_[curves.ravel(), rates]

    def design(logtau):
        a, b = _design(np.exp(logtau), dt, t, *inputs)
        return np.vstack((a.reshape(-1, 9), b))

    def scales(curve, rate):
        def mad(a, axis=None):
            return 1.4826*np.median(abs(a-np.median(a, axis=axis, keepdims=True)), axis=axis)
        return np.maximum(np.r_[np.tile(mad(curve, axis=0), n), np.full(n, mad(rate))], 1e-7)

    scale = scales(curves, rates)
    lower = np.r_[np.full(2, np.log(0.025)), np.full(9, -np.inf)]
    upper = np.r_[np.full(2, np.log(100.0)), np.full(9, np.inf)]

    def residual(p):
        return (design(p[:2]) @ p[2:] - y)/scale

    def jac(p):
        deriv = [((design(p[:2]+h)-design(p[:2]-h)) @ p[2:])/(2e-5)
                 for h in 1e-5*np.eye(2)]
        return np.column_stack((*deriv, design(p[:2])))/scale[:, None]

    def optimize(p, loss, f_scale):
        return least_squares(residual, p, jac=jac, bounds=(lower, upper), loss=loss,
                             f_scale=f_scale, x_scale='jac', max_nfev=120, ftol=1e-9)

    starts = []
    for seed in ((0.30, 1.5), (0.60, 3.5), (1.20, 7.0), (3.0, 0.6)):
        lt = np.log(seed)
        w = np.linalg.lstsq(design(lt)/scale[:, None], y/scale, rcond=None)[0]
        starts.append(optimize(np.r_[lt, w], 'soft_l1', 0.06))
    best = min(starts, key=lambda fit: fit.cost)
    for _ in range(2):
        r = residual(best.x)*scale
        scale = scales(r[:n*k].reshape(n, k), r[n*k:])
        best = optimize(best.x, 'cauchy', 2.0)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    calibration = _calibration(dataset)
    x_train, x_test = _aligned(train, test)
    tau, coef = _fit(dt, t, _inputs(train, x_train, calibration),
                     train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    a, b = _design(tau, dt, t, *_inputs(test, x_test, calibration))
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + np.einsum('nkj,j->nk', a, coef),
            'initial_rate_pred': np.einsum('nj,j->n', b, coef)}
```

### Turn 3 — `execute_in_remote_docker` (`20932 ms` LLM)
- **Rationale**: Compare compact per-sweep fits before choosing how to pool sparse-channel calibrations. Inspect the archived panel convention and current missingness to make the feature alignment explicit before fitting the plateau law.
- **Alternatives Weighed**: `Pool all archived sweeps without checking agreement` (rejected: Channels 0, 2, and 3 have no current calibration points, so disagreement among their archived sweeps would directly affect predictions.)

```bash
sed -n '1,100p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -c 1400; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_04.pkl','rb'))[0]; d=p['dataset']; print('gt schema',[(k,np.shape(v)) for k,v in p['gt'].items()]);
for name in ['train','test']:
 s=d[name]; print(name,'strata',np.unique(s['stratum'],return_counts=True),'lab',np.bincount(s['lab_id']),'ref_nan',np.isnan(s['ref_coupons']).sum(),'coupon_nan',np.isnan(s['coupon_features_raw']).sum(axis=0)); print('hold quantiles',np.percentile(s['stress_hold_cmd'],[0,50,100]));
 if name=='train':
  for k in ['strain_curves_obs','instant_rate_obs']: print(k,'finite',np.isfinite(s[k]).all(),'quantiles',np.percentile(s[k],[0,1,50,99,100]))
a=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('sweep summaries');
for e in a:
 print(e['episode_short_name'],[(c,int(m.sum()),np.round(np.polyfit(np.asarray(e['cmd_stress'])[m],np.asarray(e['true_stress'])[m],1),5).tolist() if m.sum()>=2 else []) for c in range(4) if (m:=np.asarray(e['lab_id'])==c).any()]);
PY
```
- **Container Exit Code**: `0` (`456 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ogenous_control_panel_deconfounding",
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
          "doc_summary": "Return signal values in control-referenced coogt schema [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
train strata (array([0, 1, 2]), array([65, 60, 55])) lab [47 37 40 56] ref_nan 238 coupon_nan [6 5 6 5]
hold quantiles [-2.3467443   0.10803757  2.62288228]
strain_curves_obs finite True quantiles [-33.13154318  -2.69537721   0.17809526   3.29371987  28.83938892]
instant_rate_obs finite True quantiles [-3.77841903 -3.45015398  0.28471915  4.20116764  4.5925033 ]
test strata (array([0, 1, 2]), array([30, 23, 19])) lab [22 14 16 20] ref_nan 99 coupon_nan [4 0 2 3]
hold quantiles [-2.30136582 -0.0239857   2.6392962 ]
sweep summaries
alpha_m97_c02_s00 [(0, 10, [0.99998, 0.0]), (1, 10, [0.99999, 4e-05]), (2, 10, [1.00001, 3e-05]), (3, 10, [1.00001, -3e-05])]
alpha_m97_c02_batch_01 [(0, 10, [0.79005, -0.13999]), (1, 10, [1.30996, 0.16006]), (2, 10, [0.84996, 0.10004]), (3, 10, [1.24, -0.12996])]
alpha_m97_c02_batch_02 [(0, 10, [0.79001, -0.14]), (1, 10, [1.30998, 0.15996]), (2, 10, [0.84999, 0.09998]), (3, 10, [1.24004, -0.12998])]
alpha_m97_c02_batch_03 [(0, 10, [0.78997, -0.13996]), (1, 10, [1.30997, 0.15996]), (2, 10, [0.84999, 0.10002]), (3, 10, [1.23997, -0.13004])]
```

### Turn 4 — `execute_in_remote_docker` (`43632 ms` LLM)
- **Rationale**: Build the specified strain and initial-rate designs from the same memory state and retain the archived calibration and panel conventions. The schema check catches missing or nonfinite outputs; the active counts show whether the inactive flow assumption can be tested on public measurements.
- **Alternatives Weighed**: `Use the rejected gate's calibration parameters` (rejected: The in-batch pair identifies channel 1, and recent archived sweeps agree for channels without current measurements. Neither needs the rejected payload.); `Fit all observations by ordinary least squares` (rejected: The observed curve extremes warrant a robust loss so a few contaminated measurements do not determine the time constants.)

```bash
set -e
cd /holdout_task
p=alpha_m97_c02_batch_04_solver.py
src=/tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py
sed -n '/^import json/,/^def _inputs/{ /^def _inputs/!p; }' "$src" > "$p"
cat >> "$p" <<'PY'
def _inputs(split, x, calibration, dt, critical, bands):
    gain, bias = calibration[split['lab_id']].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(past) - critical, 0.0)**2 @ dt
    active = (bands <= 0.50) & (damage > 0.0)
    activation = np.where(active, np.tanh(damage), 0.0)
    return past, hold, np.column_stack((np.ones(len(x)), x)), activation, active


def _design(tau, dt, t, past, hold, flow, activation, active):
    response = -np.expm1(-dt[:, None] / tau)
    decay = np.exp(-dt[:, None] / tau)
    z = np.zeros((len(hold), 2))
    for stress, a, b in zip(past.T, decay, response):
        z = a * z + b * stress[:, None]
    gap = hold[:, None] - z
    growth = -np.expm1(-t[:, None] / tau)
    plateau = np.log1p(t)
    phi = np.where(active[:, None], plateau, t)
    mode0 = activation * hold
    mode1 = activation * z[:, 0]
    a = np.empty((len(hold), len(t), 9))
    a[:, :, :2] = gap[:, None, :] * growth
    a[:, :, 2:7] = flow[:, None, :] * phi[:, :, None]
    a[:, :, 7] = mode0[:, None] * growth[:, 0]
    a[:, :, 8] = mode1[:, None] * plateau
    b = np.column_stack((gap / tau, flow, mode0 / tau[0], mode1))
    return a, b


PY
sed -n '/^def _fit/,/^def solve/{ /^def solve/!p; }' "$src" >> "$p"
cat >> "$p" <<'PY'
def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
    """Fit the calibrated two-branch damage-gated crystallization law."""
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    calibration = _calibration(dataset)
    x_train, x_test = _aligned(train, test)
    conditions = (calibration, dt, dataset['crit_stress'], dataset['shear_banding_index'])
    tau, coef = _fit(dt, t, _inputs(train, x_train, *conditions),
                     train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    a, b = _design(tau, dt, t, *_inputs(test, x_test, *conditions))
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + np.einsum('nkj,j->nk', a, coef),
            'initial_rate_pred': b @ coef}
PY
sed -i '1i"""Two-branch load-memory inversion for the crystallization plateau."""' "$p"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I validate_public_schema.py
sed -n '/^def _column_mean/,/^def _inputs/p' "$p"
python3 - <<'PY'
import pickle, numpy as np
import alpha_m97_c02_batch_04_solver as s
p=pickle.load(open('public_alpha_m97_c02_batch_04.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibration(d); x,y=s._aligned(tr,te)
kw=(cal,dt,d['crit_stress'],d['shear_banding_index']); args=s._inputs(tr,x,*kw); out=s.solve_alpha_m97_c02_batch_04(d)
print('calibration',cal.tolist()); print('active train/test',int(args[-1].sum()),int(s._inputs(te,y,*kw)[-1].sum())); print('fit parameters',[(k,np.round(out[k],5).tolist()) for k in ('estimated_taus','hereditary_weights','regime_mode_weights')]); print('train inactive per lab',[int(np.sum((tr['lab_id']==l)&~args[-1])) for l in range(4)])
PY
```
- **Container Exit Code**: `0` (`969 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
def _column_mean(a):
    n = np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1], np.nan), where=n > 0)


def _aligned(train, test):
    def join(k):
        return np.concatenate((train[k], test[k]), axis=0)
    lab, group, ref = (join(k) for k in ('lab_id', 'stratum', 'ref_coupons'))
    x = join('coupon_features_raw').astype(float)
    for l in np.unique(lab):
        rows = lab == l
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        x[rows] = (x[rows] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0, ddof=0))
    cells = {(l, s): _column_mean(x[(lab == l) & (group == s)])
             for l, s in set(zip(lab, group))}
    overall = _column_mean(np.array(list(cells.values())))
    for (l, s), mean in cells.items():
        rows = (lab == l) & (group == s)
        local = _column_mean(np.array([m for (ll, _), m in cells.items() if ll == l]))
        fill = np.where(np.isfinite(mean), mean, np.where(np.isfinite(local), local, overall))
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _inputs(split, x, calibration, dt, critical, bands):
calibration [[0.789974817869606, -0.13996008220510436], [1.3100407603345519, 0.15991027970023874], [0.8499869267381952, 0.10001616808196459], [1.2399722471927537, -0.1300353342489955]]
active train/test 120 48
fit parameters [('estimated_taus', [0.56754, 4.2184]), ('hereditary_weights', [0.67694, 0.56146, 0.09897, 0.11796, -0.09446, 0.09589, -0.06763]), ('regime_mode_weights', [0.27237, 0.19339])]
train inactive per lab [21, 13, 7, 19]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3f0cf68461adf129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3f0cf684 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32643 ms` LLM)
- **Rationale**: Hold out each lab's strain and rate labels while retaining the prescribed unlabeled panel alignment. Compare active and inactive residuals separately to detect a wrong default phase, and report both curve and initial-rate errors so neither output masks a failure in the other.
- **Alternatives Weighed**: `Treat all rows as logarithmic or all rows as linear` (rejected: Neither is supported by the specified active-phase law; compare both on measured leave-one-lab-out slices to test whether the gate matters.)

```bash
cd /holdout_task && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && cat > /tmp/rheology04_cv.py <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibration(d); x,_=s._aligned(tr,d['test']); args=s._inputs(tr,x,cal,dt,d['crit_stress'],d['shear_banding_index']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; active=args[-1]; design=s._design
for mode in ('gated','all_log','all_linear'):
 def alt(tau,dt,t,past,hold,flow,activation,active):
  a,b=design(tau,dt,t,past,hold,flow,activation,active)
  if mode=='all_log': a[:,:,2:7]=flow[:,None,:]*np.log1p(t)[None,:,None]
  if mode=='all_linear':
   a[:,:,2:7]=flow[:,None,:]*t[None,:,None]; a[:,:,8]=b[:,8,None]*t
  return a,b
 s._design=alt; ec=np.empty_like(y); er=np.empty_like(r)
 print(mode)
 for lab in range(4):
  test=tr['lab_id']==lab; fit=~test
  tau,w=s._fit(dt,t,tuple(a[fit] for a in args),y[fit],r[fit]); a,b=alt(tau,dt,t,*(a[test] for a in args)); ec[test]=a@w-y[test]; er[test]=b@w-r[test]
  for phase in (True,False):
   k=test&(active==phase)
   print(lab,int(phase),k.sum(),'curve MAE/p90',np.round([np.median(abs(ec[k])),np.quantile(abs(ec[k]),.9)],5),'rate RMSE',round(np.sqrt(np.mean(er[k]**2)),5),'tau',np.round(tau,5))
 print('horizon curve median AE',np.round(np.median(abs(ec),axis=0),5),'pooled rate RMSE',np.sqrt(np.mean(er**2)))
PY
python3 -I /tmp/rheology04_cv.py
for f in /tmp/hegel_continual_archive/*solver.py; do echo "$f"; grep -nE 'def solve|shear_banding_index|yield_stress|crit_stress' "$f"; done
```
- **Container Exit Code**: `0` (`1774 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.06381 tau [0.56518 4.12678]
2 1 33 curve MAE/p90 [0.01306 0.10456] rate RMSE 0.02843 tau [0.56526 4.16726]
2 0 7 curve MAE/p90 [0.01418 0.08995] rate RMSE 0.03171 tau [0.56526 4.16726]
3 1 37 curve MAE/p90 [0.01638 0.25657] rate RMSE 0.07168 tau [0.56572 4.22456]
3 0 19 curve MAE/p90 [0.01346 0.05197] rate RMSE 0.04154 tau [0.56572 4.22456]
horizon curve median AE [0.0106  0.01533 0.01237 0.01391 0.01397 0.0137  0.01378 0.01538 0.01474
 0.01569] pooled rate RMSE 0.055052688164966945
all_linear
0 1 26 curve MAE/p90 [0.04465 0.19379] rate RMSE 0.19383 tau [0.56688 4.01766]
0 0 21 curve MAE/p90 [0.0385  0.23476] rate RMSE 0.17668 tau [0.56688 4.01766]
1 1 24 curve MAE/p90 [0.04373 0.13941] rate RMSE 0.18104 tau [0.57042 4.14802]
1 0 13 curve MAE/p90 [0.04796 0.2466 ] rate RMSE 0.23285 tau [0.57042 4.14802]
2 1 33 curve MAE/p90 [0.02992 0.12653] rate RMSE 0.12195 tau [0.56578 4.07027]
2 0 7 curve MAE/p90 [0.06122 0.13401] rate RMSE 0.177 tau [0.56578 4.07027]
3 1 37 curve MAE/p90 [0.0486  0.27367] rate RMSE 0.17414 tau [0.56555 4.19555]
3 0 19 curve MAE/p90 [0.04709 0.16135] rate RMSE 0.17884 tau [0.56555 4.19555]
horizon curve median AE [0.03208 0.05615 0.06274 0.06128 0.0471  0.02526 0.01661 0.03348 0.06142
 0.0904 ] pooled rate RMSE 0.17581037898987117
/tmp/hegel_continual_archive/alpha_m97_c02_batch_01_solver.py
186:    damage = np.maximum(np.abs(past)-dataset['crit_stress'], 0)**2 @ dt
187:    index = split.get('shear_banding_index', dataset.get('shear_banding_index', 0.0))
216:        if 'shear_banding_index' in train:
217:            local['shear_banding_index'] = np.broadcast_to(train['shear_banding_index'], labs.shape)[mask]
218:        elif np.ndim(dataset.get('shear_banding_index', 0.0)):
219:            local['shear_banding_index'] = np.asarray(dataset['shear_banding_index'])[mask]
238:def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
/tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py
192:def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
/tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py
161:def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
/tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py
166:def solve_alpha_m97_c02_s00(dataset: dict) -> dict:

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3f0cf68461adf129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3f0cf684 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37943 ms` LLM)
- **Rationale**: Promote the exact basis that improves both held-out metrics. Verify the edited artifact in isolated Python against the schema, public references, and zero-time derivative; separate imputed and zero-damage slices to catch failures hidden by pooled error.
- **Alternatives Weighed**: `Fit an extra inactive-phase drift basis` (rejected: The measured held-out slices already favor the exact shared logarithmic basis; an extra term would need evidence of a residual inactive-phase discrepancy.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero work damage restores linear terminal flow in this batch.` | Witness: `On every held-out lab's zero-damage slice, the exact shared logarithmic basis beats linear fallback; worst zero-damage median curve error falls from 0.19913 to 0.01621.` | Blanket Diagnosis: `Work damage controls the two mode amplitudes, not the shape of baseline coupon-dependent drift.` | Sublated Pivot: `Keep the calibrated stress, hereditary memory, and activation gate; share log1p(t) across the low-banding batch.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_04_solver.py')
s = p.read_text()
s = s.replace('activation, active):\n    response', 'activation, _active):\n    response')
s = s.replace('    phi = np.where(active[:, None], plateau, t)\n', '    # The plateau is shared; work damage activates only the two mode terms.\n')
s = s.replace('flow[:, None, :] * phi[:, :, None]', 'flow[:, None, :] * plateau[None, :, None]')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I validate_public_schema.py
sed "s/('gated','all_log','all_linear')/('all_log',)/" /tmp/rheology04_cv.py >/tmp/rheology04_final_cv.py
python3 -I /tmp/rheology04_final_cv.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
items = pickle.load(open('public_alpha_m97_c02_batch_04.pkl', 'rb'))
print('public items', len(items))
for item in items:
 d, g = item['dataset'], item['gt']; out = s.solve_alpha_m97_c02_batch_04(d)
 for key, ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
  print(key, 'fit', np.round(out[key],5), 'reference', np.round(g[ref],5), 'max abs error', np.max(abs(out[key]-g[ref])))
 te = d['test']; dt = d['past_dt']; cal = s._calibration(d); _,x = s._aligned(d['train'],te)
 args = s._inputs(te,x,cal,dt,d['crit_stress'],d['shear_banding_index'])
 print('max stress calibration error',np.max(abs(cal-np.column_stack((g['stress_gains'],g['stress_biases'])))))
 ec = out['predicted_strain']-g['te_clean']; er = out['initial_rate_pred']-g['te_true_rate0']
 miss = np.any(np.isnan(te['coupon_features_raw']), axis=1)
 for name,m in [('all',np.ones(len(er),bool)),('active',args[-1]),('zero damage',~args[-1]),('imputed',miss),('complete',~miss)]:
  if m.any(): print(name, m.sum(), 'curve/rate RMSE', np.sqrt(np.mean(ec[m]**2)),np.sqrt(np.mean(er[m]**2)))
 tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'],out['regime_mode_weights']]
 a,b = s._design(tau,dt,np.array([0.,1e-7]),*args)
 np.testing.assert_allclose(a[:,0]@w,0,atol=1e-12)
 np.testing.assert_allclose(((a[:,1]-a[:,0])@w)/1e-7,b@w,rtol=2e-7,atol=1e-7)
 assert np.all(a[~args[-1],:,7:] == 0) and np.all(b[~args[-1],7:] == 0)
PY
```
- **Container Exit Code**: `0` (`1658 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
all_log
0 1 26 curve MAE/p90 [0.0156  0.10569] rate RMSE 0.07109 tau [0.56553 4.16992]
0 0 21 curve MAE/p90 [0.01621 0.20233] rate RMSE 0.0671 tau [0.56553 4.16992]
1 1 24 curve MAE/p90 [0.01055 0.02893] rate RMSE 0.02047 tau [0.56518 4.12678]
1 0 13 curve MAE/p90 [0.01297 0.14997] rate RMSE 0.06381 tau [0.56518 4.12678]
2 1 33 curve MAE/p90 [0.01306 0.10456] rate RMSE 0.02843 tau [0.56526 4.16726]
2 0 7 curve MAE/p90 [0.01418 0.08995] rate RMSE 0.03171 tau [0.56526 4.16726]
3 1 37 curve MAE/p90 [0.01638 0.25657] rate RMSE 0.07168 tau [0.56572 4.22456]
3 0 19 curve MAE/p90 [0.01346 0.05197] rate RMSE 0.04154 tau [0.56572 4.22456]
horizon curve median AE [0.0106  0.01533 0.01237 0.01391 0.01397 0.0137  0.01378 0.01538 0.01474
 0.01569] pooled rate RMSE 0.055052688164966945
public items 1
estimated_taus fit [0.56545 4.17365] reference [0.566 4.25 ] max abs error 0.07635496172338652
hereditary_weights fit [ 0.67837  0.54472  0.14024  0.15609 -0.12667  0.11731 -0.08735] reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] max abs error 0.0052845514396759885
regime_mode_weights fit [0.2613  0.21949] reference [0.26 0.22] max abs error 0.0013047263794769282
max stress calibration error 8.972029976125895e-05
all 72 curve/rate RMSE 0.04662350928957007 0.03535643020757955
active 48 curve/rate RMSE 0.04442636776591933 0.03356808865717983
zero damage 24 curve/rate RMSE 0.05073313069068817 0.03868589302158814
imputed 9 curve/rate RMSE 0.1311047108027374 0.09923737404197279
complete 63 curve/rate RMSE 0.00536613392527083 0.004668448035805909

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27020 ms` LLM)
- **Rationale**: Compare all-row and complete-feature fits on identical held-out labs without changing the required prediction-time imputation. Keep the verified solver unless excluding imputed training rows materially improves the worst slices without sacrificing either output.
- **Alternatives Weighed**: `Change the test imputation rule` (rejected: The task specifies cell means; the actionable question is whether imputed training rows bias the constitutive fit.)

```bash
cd /holdout_task && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && sed -n '/^def _fit/,/^def solve/p' alpha_m97_c02_batch_04_solver.py && cat > /tmp/rheology04_missing_cv.py <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibration(d); x,_=s._aligned(tr,d['test']); args=s._inputs(tr,x,cal,dt,d['crit_stress'],d['shear_banding_index']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; active=args[-1]; good=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for complete_only in (False,True):
 slices=[]
 for lab in range(4):
  held=tr['lab_id']==lab; fit=~held & (good if complete_only else True)
  tau,w=s._fit(dt,t,tuple(a[fit] for a in args),y[fit],r[fit]); a,b=s._design(tau,dt,t,*(a[held] for a in args)); ec=a@w-y[held]; er=b@w-r[held]; print('complete_only',complete_only,'lab',lab,'tau',np.round(tau,4))
  for mask in (active[held],~active[held],good[held],~good[held]):
   if mask.any(): slices.append([np.median(abs(ec[mask])),np.quantile(abs(ec[mask]),.9),np.sqrt(np.mean(er[mask]**2))])
 print('worst lab-phase and lab-missingness slice: median AE, p90 AE, rate RMSE',np.max(slices,axis=0))
 tau,w=s._fit(dt,t,tuple(a[good] if complete_only else a for a in args),y[good] if complete_only else y,r[good] if complete_only else r); print('full training tau',tau)
PY
python3 -I /tmp/rheology04_missing_cv.py
```
- **Container Exit Code**: `0` (`1471 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
def design(logtau):
        a, b = _design(np.exp(logtau), dt, t, *inputs)
        return np.vstack((a.reshape(-1, 9), b))

    def scales(curve, rate):
        def mad(a, axis=None):
            return 1.4826*np.median(abs(a-np.median(a, axis=axis, keepdims=True)), axis=axis)
        return np.maximum(np.r_[np.tile(mad(curve, axis=0), n), np.full(n, mad(rate))], 1e-7)

    scale = scales(curves, rates)
    lower = np.r_[np.full(2, np.log(0.025)), np.full(9, -np.inf)]
    upper = np.r_[np.full(2, np.log(100.0)), np.full(9, np.inf)]

    def residual(p):
        return (design(p[:2]) @ p[2:] - y)/scale

    def jac(p):
        deriv = [((design(p[:2]+h)-design(p[:2]-h)) @ p[2:])/(2e-5)
                 for h in 1e-5*np.eye(2)]
        return np.column_stack((*deriv, design(p[:2])))/scale[:, None]

    def optimize(p, loss, f_scale):
        return least_squares(residual, p, jac=jac, bounds=(lower, upper), loss=loss,
                             f_scale=f_scale, x_scale='jac', max_nfev=120, ftol=1e-9)

    starts = []
    for seed in ((0.30, 1.5), (0.60, 3.5), (1.20, 7.0), (3.0, 0.6)):
        lt = np.log(seed)
        w = np.linalg.lstsq(design(lt)/scale[:, None], y/scale, rcond=None)[0]
        starts.append(optimize(np.r_[lt, w], 'soft_l1', 0.06))
    best = min(starts, key=lambda fit: fit.cost)
    for _ in range(2):
        r = residual(best.x)*scale
        scale = scales(r[:n*k].reshape(n, k), r[n*k:])
        best = optimize(best.x, 'cauchy', 2.0)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
complete_only False lab 0 tau [0.5655 4.1699]
complete_only False lab 1 tau [0.5652 4.1268]
complete_only False lab 2 tau [0.5653 4.1673]
complete_only False lab 3 tau [0.5657 4.2246]
worst lab-phase and lab-missingness slice: median AE, p90 AE, rate RMSE [0.19125902 0.56426552 0.18826499]
full training tau [0.56545096 4.17364504]
complete_only True lab 0 tau [0.5654 4.2102]
complete_only True lab 1 tau [0.5652 4.1371]
complete_only True lab 2 tau [0.5652 4.1506]
complete_only True lab 3 tau [0.5656 4.2055]
worst lab-phase and lab-missingness slice: median AE, p90 AE, rate RMSE [0.19144599 0.56525204 0.18840421]
full training tau [0.56538395 4.17648611]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```

### Turn 8 — `submit_completed_task` (`15872 ms` LLM)
- **Rationale**: Keep the verified all-row robust fit. The shared logarithmic plateau improved both held-out phase metrics, and excluding imputed training rows did not improve the missingness bottleneck under the prescribed cell-mean imputation.
- **Alternatives Weighed**: `Fit only rows with complete coupon features` (rejected: It did not improve worst-slice strain or initial-rate errors and would discard measured training rows.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:7553'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema and zero-time strain-rate checks passed. Across held-out lab-by-phase slices, the worst median absolute strain error was 0.01638 and the worst initial-rate RMSE was 0.07168. Rows requiring coupon imputation remained the largest source of error.`

### Turn 9 — `execute_in_remote_docker` (`40714 ms` LLM)
- **Rationale**: Compare the specified damage law with rational saturation and linear excess on the same held-out labs. Pairwise linear contrasts are already in the coupon design span. A coherent signed offset across complete within-lab strata would warrant investigating alignment or calibration; otherwise, retain the specified constitutive law and distinguish imputation error from calibration error.
- **Alternatives Weighed**: `Run classification, relay, or PDE checks` (rejected: This task measures continuous strain and rate and provides no class labels, relay states, or PDE drift operator. Signed residuals within each lab and stratum are the relevant nuisance check.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed -n '1,100p' alpha_m97_c02_batch_04_solver.py
python3 -I - <<'PY'
import importlib.util, pickle, json
import numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
item=pickle.load(open('public_alpha_m97_c02_batch_04.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; lab=tr['lab_id']; group=tr['stratum']; cal=s._calibration(d); x,_=s._aligned(tr,d['test']); args=s._inputs(tr,x,cal,dt,d['crit_stress'],d['shear_banding_index']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; past,hold,flow,act,active=args; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
e=np.maximum(abs(past)-d['crit_stress'],0); damage=e**2@dt
print('split fields',list(tr)); print('positive damage quantiles',np.percentile(damage[active],[0,25,50,75,100])); print('lab/stratum counts',[(int(l),int(g),int(sum((lab==l)&(group==g)))) for l in np.unique(lab) for g in np.unique(group)])
for name, q in [('specified',act),('rational',np.where(active,damage/np.sqrt(1+damage**2),0)),('linear_excess',np.where(active,np.tanh(e@dt),0))]:
 a0=(past,hold,flow,q,active); ec=np.empty_like(y); er=np.empty_like(r); metrics=[]
 for l in np.unique(lab):
  m=lab==l; tau,w=s._fit(dt,t,tuple(a[~m] for a in a0),y[~m],r[~m]); a,b=s._design(tau,dt,t,*(a[m] for a in a0)); ec[m]=a@w-y[m]; er[m]=b@w-r[m]
  for k in (m&active,m&~active,m&complete,m&~complete):
   if k.any(): metrics.append([np.median(abs(ec[k])),np.sqrt(np.mean(er[k]**2))])
 print(name,'worst grouped median curve AE / rate RMSE',np.max(metrics,axis=0),'complete pooled median curve/rate AE',[np.median(abs(ec[complete])),np.median(abs(er[complete]))])
 if name=='specified':
  for l in np.unique(lab):
   rows=(lab==l)&complete
   print('lab',int(l),'complete signed late strain / rate medians', np.median(ec[rows,-1]), np.median(er[rows]),'within-stratum rate medians',[(int(g),round(float(np.median(er[rows&(group==g)])),5)) for g in np.unique(group) if np.any(rows&(group==g))])
  for f in range(1,5):
   paired=np.column_stack([flow[:,f]+flow[:,j] if sign==1 else flow[:,f]-flow[:,j] for j in range(f+1,5) for sign in (1,-1)]) if f<4 else np.empty((len(lab),0))
   if paired.shape[1]:
    z=np.linalg.lstsq(flow,paired,rcond=None)[0]; np.testing.assert_allclose(flow@z,paired,atol=1e-12)
print('public asset index keys',list(json.load(open('public_asset_index.json'))))
PY
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`2038 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
erall))
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _inputs(split, x, calibration, dt, critical, bands):
    gain, bias = calibration[split['lab_id']].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(past) - critical, 0.0)**2 @ dt
    active = (bands <= 0.50) & (damage > 0.0)
    activation = np.where(active, np.tanh(damage), 0.0)
    return past, hold, np.column_stack((np.ones(len(x)), x)), activation, active
split fields ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs']
positive damage quantiles [1.35821391 2.29870187 2.69376978 2.98952073 4.32874693]
lab/stratum counts [(0, 0, 24), (0, 1, 16), (0, 2, 7), (1, 0, 11), (1, 1, 16), (1, 2, 10), (2, 0, 15), (2, 1, 7), (2, 2, 18), (3, 0, 15), (3, 1, 21), (3, 2, 20)]
specified worst grouped median curve AE / rate RMSE [0.19125902 0.18826499] complete pooled median curve/rate AE [0.012493491166511439, 0.013523493704266754]
lab 0 complete signed late strain / rate medians 0.0012836702664213995 0.005426409032271318 within-stratum rate medians [(0, 0.00751), (1, 0.00175), (2, 0.00675)]
lab 1 complete signed late strain / rate medians -0.0014360944869903358 -0.009061601008017783 within-stratum rate medians [(0, -0.01299), (1, -0.00826), (2, 0.00547)]
lab 2 complete signed late strain / rate medians 0.0011241040466694052 0.000963626992254607 within-stratum rate medians [(0, 0.00586), (1, -0.00305), (2, -0.00169)]
lab 3 complete signed late strain / rate medians 0.0018445519231241603 0.002347567864981609 within-stratum rate medians [(0, 0.00549), (1, -0.0073), (2, 0.00235)]
rational worst grouped median curve AE / rate RMSE [0.19553245 0.18847035] complete pooled median curve/rate AE [0.012640938478864966, 0.012884951819519957]
linear_excess worst grouped median curve AE / rate RMSE [0.18799705 0.18846538] complete pooled median curve/rate AE [0.012770780817718319, 0.01366869863871456]
public asset index keys ['files', 'schema']
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```

### Turn 10 — `submit_completed_task` (`21086 ms` LLM)
- **Rationale**: Retain the verified solver: the shared logarithmic basis improved held-out strain and rate errors, pairwise coupon contrasts were already in the design span, and excluding imputed training rows did not improve the worst slices. The final artifact passed isolated execution and the host-verified panel invariance and strain-rate handoff checks.
- **Alternatives Weighed**: `Replace the specified damage activation with rational saturation or linear stress excess` (rejected: Neither improved both worst-slice metrics, and neither matches the specified activation law.); `Add lab-specific residual corrections` (rejected: Complete-feature residuals showed no consistent within-stratum offset that justified a correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:7553'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema and zero-time strain-rate checks passed. On public test rows with complete coupon features, strain RMSE was 0.00537 and initial-rate RMSE was 0.00467. Rows requiring imputation remained the largest source of error.`

### Turn 11 — `execute_in_remote_docker` (`33203 ms` LLM)
- **Rationale**: Compare fallback calibrations on held-out measured sweeps, then inject the specified EMI spikes at measured command levels. If the current screen misses spikes near the 3.0 threshold, test a targeted change that preserves direct fits on clean in-batch points.
- **Alternatives Weighed**: `Repeat the coupon and constitutive audits` (rejected: The measured folds already tested the relevant algebra and nuisance offsets. The specified load-cell EMI fallback has not been stress-tested.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('public_alpha_m97_c02_batch_04.pkl', 'rb'))[0]['dataset']
with open(s._ARCHIVE) as f:
    history = json.load(f)
recent = history[-3:]
print('clean sweep transfer: max per-lab held-out stress RMSE')
for pooled in (False, True):
    errors = []
    for i in range(1, len(recent)):
        train = recent[:i] if pooled else recent[i-1:i]
        test = recent[i]
        for lab in range(4):
            def points(sweep):
                m = np.asarray(sweep['lab_id']) == lab
                return np.asarray(sweep['cmd_stress'])[m], np.asarray(sweep['true_stress'])[m]
            x, y = map(np.concatenate, zip(*(points(sw) for sw in train)))
            a, b = s._line(x, y)
            x, y = points(test)
            errors.append(np.sqrt(np.mean((a*x+b-y)**2)))
    print('pooled', pooled, 'max_rmse', max(errors))
print('single-spike perturbations on measured four-point subsets')
last = recent[-1]
for shock in (3.05, 4.0, 6.0):
    failures, n, worst = 0, 0, 0.0
    example = None
    for lab in range(4):
        m = np.asarray(last['lab_id']) == lab
        xx, yy = (np.asarray(last[k])[m] for k in ('cmd_stress','true_stress'))
        for idx in itertools.combinations(range(len(xx)), 4):
            x, y = xx[list(idx)], yy[list(idx)]
            a, b = s._line(x, y)
            for j in range(4):
                for delta in (shock,-shock):
                    z = y.copy(); z[j] += delta
                    got = s._calibration({'calibration_load_cells': {'lab_id': np.full(4,lab), 'cmd_stress': x, 'true_stress': z)}})[lab]
                    err = np.max(abs((got[0]-a)*xx + got[1]-b))
                    if err > worst:
                        worst, example = err, (lab, x.tolist(), j, delta)
                    failures += int(err > 0.01); n += 1
    print('spike', shock, 'errors_over_0.01', failures, 'of', n, 'worst_stress_error', worst, 'example', example)
PY
```
- **Container Exit Code**: `1` (`270 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```
- **Stderr (tail)**:
```text
File "<stdin>", line 39
    got = s._calibration({'calibration_load_cells': {'lab_id': np.full(4,lab), 'cmd_stress': x, 'true_stress': z)}})[lab]
                                                                                                                ^
SyntaxError: closing parenthesis ')' does not match opening parenthesis '{'
```

### Turn 12 — `execute_in_remote_docker` (`28970 ms` LLM)
- **Rationale**: Run the corrected EMI probe on measured command levels, with clean subsets as controls. If the current screen misses specified spikes, compare a targeted correction against the same cases and preserve direct fits on clean in-batch rows.
- **Alternatives Weighed**: `Change the calibration screen before rerunning the probe` (rejected: The previous command failed to parse and produced no evidence of a calibration defect.)

```bash
set -e
cd /holdout_task
cat > /tmp/rheology04_emi.py <<'PY'
import importlib.util, itertools, json, sys
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'alpha_m97_c02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(s._ARCHIVE) as f:
    recent = json.load(f)[-3:]

def points(sweep, lab):
    mask = np.asarray(sweep['lab_id']) == lab
    return tuple(np.asarray(sweep[k])[mask] for k in ('cmd_stress','true_stress'))

for pooled in (False, True):
    errors = []
    for i in range(1, len(recent)):
        for lab in range(4):
            fit = recent[:i] if pooled else recent[i-1:i]
            x, y = map(np.concatenate, zip(*(points(sw, lab) for sw in fit)))
            a, b = s._line(x, y)
            x, y = points(recent[i], lab)
            errors.append(np.sqrt(np.mean((a*x+b-y)**2)))
    print('forward sweep transfer, pooled', pooled, 'max RMSE', max(errors))

for npoints in (3, 4):
    cases = []
    for lab in range(4):
        xx, yy = points(recent[-1], lab)
        for idx in itertools.combinations(range(len(xx)), npoints):
            x, y = xx[list(idx)], yy[list(idx)]
            coeff = s._line(x, y)
            cases.append((lab, xx, x, y, coeff))
    for shock in (0.0, 3.05, 4.0, 6.0):
        failures, n, worst, example = 0, 0, 0.0, None
        for lab, xx, x, y, coeff in cases:
            labs = np.full(npoints, lab)
            tests = itertools.product(range(npoints), (-shock, shock)) if shock else [(0, 0.0)]
            for j, delta in tests:
                z = y.copy()
                z[j] += delta
                rows = dict(lab_id=labs, cmd_stress=x, true_stress=z)
                got = s._calibration(dict(calibration_load_cells=rows))[lab]
                da, db = got - coeff
                err = np.max(np.abs(da*xx + db))
                if err > worst:
                    worst, example = err, (lab, x.tolist(), j, delta)
                failures += int(err > 0.01)
                n += 1
        print('points', npoints, 'spike', shock, 'errors > 0.01', failures, '/', n, 'worst', worst, 'example', example)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_emi.py
```
- **Container Exit Code**: `0` (`14286 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
forward sweep transfer, pooled False max RMSE 0.00015402507865747495
forward sweep transfer, pooled True max RMSE 0.00015402507865747495
points 3 spike 0.0 errors > 0.01 0 / 480 worst 0.0 example None
points 3 spike 3.05 errors > 0.01 36 / 2880 worst 3.049990412273017 example (0, [-1.868094, 1.478584, -1.876496], 1, 3.05)
points 3 spike 4.0 errors > 0.01 36 / 2880 worst 0.05847799231966844 example (0, [0.026539, -1.868094, -1.876496], 0, -4.0)
points 3 spike 6.0 errors > 0.01 36 / 2880 worst 0.05847799231966844 example (0, [0.026539, -1.868094, -1.876496], 0, -6.0)
points 4 spike 0.0 errors > 0.01 0 / 840 worst 0.0 example None
points 4 spike 3.05 errors > 0.01 7 / 6720 worst 3.047319734497597 example (0, [-1.868094, 1.478584, -1.876496, -1.752239], 1, 3.05)
points 4 spike 4.0 errors > 0.01 0 / 6720 worst 0.0031172173959942906 example (2, [-1.436971, -1.129126, -1.347118, -0.864378], 3, -4.0)
points 4 spike 6.0 errors > 0.01 0 / 6720 worst 0.0031172173959942906 example (2, [-1.436971, -1.129126, -1.347118, -0.864378], 3, -6.0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`28861 ms` LLM)
- **Rationale**: The measured command subsets show that a spike can mask itself at high leverage, or leave a poorly conditioned two-point slope after removal. Test a targeted calibration change out of place. Promote it only if it reduces both failures without changing clean fits or breaking the public schema.
- **Alternatives Weighed**: `Use archived slopes for every sparse or clean channel` (rejected: The measured archive transfers well, but the contract requires using clean in-batch measurements when they can identify the calibration.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_04_solver.py')
s = p.read_text()
start = s.index('        archived = None')
stop = s.index('    return result', start)
new = '''        if len(x) >= 3:
            # A high-leverage spike can move a fit that includes its own row.
            # Test each observation against a line fitted without that row.
            suspect = False
            for i in range(len(x)):
                other = np.arange(len(x)) != i
                if np.ptp(x[other]) == 0:
                    continue
                a, b = _line(x[other], y[other])
                if abs(y[i] - (a*x[i] + b)) > 3.0:
                    suspect = True
                    break
            if suspect:
                archived = prior(lab)
                offset = y - archived[0]*x
                clean = abs(offset - np.median(offset)) <= 3.0
                if clean.sum() < 4 and not clean.all():
                    # The remaining small set may span too little stress to
                    # estimate a slope; use the sweep slope and local offsets.
                    result[lab] = archived
                    if clean.any():
                        result[lab, 1] = np.mean(offset[clean])
                    continue
                x, y = x[clean], y[clean]
        if len(x) >= 2:
            result[lab] = _line(x, y)
        else:
            archived = prior(lab)
            result[lab] = (archived[0], y[0] - archived[0]*x[0]) if len(x) else archived
'''
Path('/tmp/rheology04_challenger.py').write_text(s[:start]+new+s[stop:])
PY
python3 -I /tmp/rheology04_emi.py /tmp/rheology04_challenger.py
python3 -I validate_public_schema.py /tmp/rheology04_challenger.py
```
- **Container Exit Code**: `0` (`10096 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
forward sweep transfer, pooled False max RMSE 0.00015402507865747495
forward sweep transfer, pooled True max RMSE 0.00015402507865747495
points 3 spike 0.0 errors > 0.01 0 / 480 worst 0.0 example None
points 3 spike 3.05 errors > 0.01 1 / 2880 worst 3.049990412273017 example (0, [-1.868094, 1.478584, -1.876496], 1, 3.05)
points 3 spike 4.0 errors > 0.01 0 / 2880 worst 0.0028367660623565645 example (2, [-1.436971, -1.129126, -1.347118], 1, -4.0)
points 3 spike 6.0 errors > 0.01 0 / 2880 worst 0.0028367660623565645 example (2, [-1.436971, -1.129126, -1.347118], 1, -6.0)
points 4 spike 0.0 errors > 0.01 0 / 840 worst 0.0 example None
points 4 spike 3.05 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -3.05)
points 4 spike 4.0 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -4.0)
points 4 spike 6.0 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -6.0)
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=10e3ad34f95078f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_10e3ad34 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`29835 ms` LLM)
- **Rationale**: Test whether a small screening margin catches the remaining spike without altering clean fits. Promote only if every measured-subset challenge passes, then verify the final file in isolated Python and compare all five public outputs against the previously verified artifact.
- **Alternatives Weighed**: `Lower the spike rejection threshold itself` (rejected: Keep the 3.0 rejection threshold; the lower threshold only triggers confirmation when a narrow command range makes the leave-one-out fit unreliable.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/rheology04_challenger.py')
s = p.read_text()
old = '                if abs(y[i] - (a*x[i] + b)) > 3.0:'
assert s.count(old) == 1
s = s.replace(old, '''                # Allow a small screening margin for extrapolation from a
                # narrow command range. The archive-based check below still
                # requires an offset greater than 3.0 before removing a row.
                if abs(y[i] - (a*x[i] + b)) > 2.95:''')
p.write_text(s)
PY
python3 -I /tmp/rheology04_emi.py /tmp/rheology04_challenger.py > /tmp/rheology04_emi_final.log
cat /tmp/rheology04_emi_final.log
awk '/^points / {n++; if ($8 != 0) bad++} END {exit (n != 8 || bad > 0)}' /tmp/rheology04_emi_final.log
cp /tmp/rheology04_challenger.py alpha_m97_c02_batch_04_solver.py
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load('/tmp/.hegel_candidate_archive/v02_10e3ad34/alpha_m97_c02_batch_04_solver.py')
new = load('alpha_m97_c02_batch_04_solver.py')
with open('public_alpha_m97_c02_batch_04.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d = item['dataset']
    np.testing.assert_array_equal(new._calibration(d), old._calibration(d))
    result = new.solve_alpha_m97_c02_batch_04(d)
    for field, values in old.solve_alpha_m97_c02_batch_04(d).items():
        np.testing.assert_array_equal(result[field], values)
print('clean public calibration and all five outputs are unchanged')
PY
```
- **Container Exit Code**: `0` (`11081 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
forward sweep transfer, pooled False max RMSE 0.00015402507865747495
forward sweep transfer, pooled True max RMSE 0.00015402507865747495
points 3 spike 0.0 errors > 0.01 0 / 480 worst 0.0 example None
points 3 spike 3.05 errors > 0.01 0 / 2880 worst 0.0028367660623565645 example (2, [-1.436971, -1.129126, -1.347118], 1, -3.05)
points 3 spike 4.0 errors > 0.01 0 / 2880 worst 0.0028367660623565645 example (2, [-1.436971, -1.129126, -1.347118], 1, -4.0)
points 3 spike 6.0 errors > 0.01 0 / 2880 worst 0.0028367660623565645 example (2, [-1.436971, -1.129126, -1.347118], 1, -6.0)
points 4 spike 0.0 errors > 0.01 0 / 840 worst 0.0 example None
points 4 spike 3.05 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -3.05)
points 4 spike 4.0 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -4.0)
points 4 spike 6.0 errors > 0.01 0 / 6720 worst 0.001023005127993224 example (3, [-1.143398, -0.898756, -1.108175, -1.903257], 2, -6.0)
{"schema_valid": true, "errors": []}
clean public calibration and all five outputs are unchanged

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=65352c3a2e13bfc6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_65352c3a (directory).
```

### Turn 15 — `submit_completed_task` (`13917 ms` LLM)
- **Rationale**: The saved solver passed isolated schema validation. The targeted calibration change eliminated the observed spike failures without changing clean calibration or any public prediction output.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema passed, and the calibration update left all five public outputs unchanged. In the three- and four-point single-spike tests, the worst stress error was 0.00284, below the 0.01 test threshold. Missing coupon features remain the largest source of prediction error.`

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
