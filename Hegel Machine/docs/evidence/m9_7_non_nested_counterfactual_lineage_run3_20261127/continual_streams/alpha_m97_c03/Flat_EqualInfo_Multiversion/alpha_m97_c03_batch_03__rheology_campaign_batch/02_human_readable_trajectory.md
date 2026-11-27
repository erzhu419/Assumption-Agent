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
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `300,548` (`290,837` / `103,168` / `9,711`) |
| **Total Episode Wall Time (s)** | `344.25s` |

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

### Turn 1 — `execute_in_remote_docker` (`12648 ms` LLM)
- **Rationale**: Inspect the public schema and prior alignment implementation before fitting; a mismatch in field names or reference-panel semantics would change preprocessing, and the skill lookup may identify reusable primitives.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json; ls -lh /tmp/hegel_continual_archive; sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x[:2])
    return np.exp(best.x[:2][order]), np.r_[best.x[2:4][order], best.x[4:]]


def solve_alpha_m97_c03_batch_02(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    taus, weights = _fit((past[:n], hold[:n], features[:n], dt, t), target)
    a, _ = _basis(np.log(taus), past[n:], hold[n:], features[n:], dt, t)
    pred = a @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`23670 ms` LLM)
- **Rationale**: Inspect the untruncated preprocessing and compact measurement summaries. A reference schema mismatch changes alignment, while excess channel residuals or sparse sweeps determine whether archive calibration is needed.
- **Alternatives Weighed**: `Reuse prior solver unchanged` (rejected: Its displayed prediction stage omits both jammed-slip terms; only applicable preprocessing should be reused.)

```bash
sed -n '1,100p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -45; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))
print('items',len(p),p[0].keys()); d=p[0]['dataset']; print('dataset',d.keys())
for name in ('train','test','calibration_load_cells'):
 a=d[name]; print(name,[(k,np.shape(v)) for k,v in a.items() if not isinstance(v,dict)]); print('labs',np.unique(a['lab_id'],return_counts=True));
 if 'ref_coupons' in a: print('reference schema',[(k,np.shape(v)) for k,v in a['ref_coupons'].items()] if isinstance(a['ref_coupons'],dict) else np.shape(a['ref_coupons']))
 if 'coupon_features_raw' in a: print('missing features',np.isnan(a['coupon_features_raw']).sum(axis=0))
c=d['calibration_load_cells'];
for i in range(4):
 ix=c['lab_id']==i; x=c['cmd_stress'][ix]; y=c['true_stress'][ix]; print('cal',i,'max reading',np.max(abs(y)),'fit / max residual',np.polyfit(x,y,1),max(abs(y-np.polyval(np.polyfit(x,y,1),x))))
print('times',d['eval_times'],'past dt',d['past_dt']); print('scipy',__import__('scipy').__version__)
PY
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"function": "align_control_panel",
          "signature": "align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
items 1 dict_keys(['dataset', 'gt'])
dataset dict_keys(['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test'])
train [('lab_id', (180,)), ('stratum', (180,)), ('coupon_features_raw', (180, 4)), ('ref_coupons', (180, 32)), ('past_stress_cmd', (180, 18)), ('stress_hold_cmd', (180,)), ('strain_0', (180,)), ('aux_chamber_telemetry', (180, 3)), ('strain_curves_obs', (180, 10)), ('instant_rate_obs', (180,))]
labs (array([0, 1, 2, 3]), array([49, 47, 45, 39]))
reference schema (180, 32)
missing features [6 4 8 4]
test [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 32)), ('past_stress_cmd', (72, 18)), ('stress_hold_cmd', (72,)), ('strain_0', (72,)), ('aux_chamber_telemetry', (72, 3))]
labs (array([0, 1, 2, 3]), array([26, 18, 14, 14]))
reference schema (72, 32)
missing features [3 3 2 0]
calibration_load_cells [('lab_id', (40,)), ('cmd_stress', (40,)), ('true_stress', (40,))]
labs (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
cal 0 max reading 2.791644186794195 fit / max residual [1.28996286 0.11997817] 0.00017199813083901283
cal 1 max reading 1.354561142136367 fit / max residual [ 0.8199772  -0.14996294] 0.0002451248283934504
cal 2 max reading 2.246884794299574 fit / max residual [1.16999799 0.14000869] 0.00017239041523975374
cal 3 max reading 1.530996264069709 fit / max residual [ 0.76000329 -0.11003612] 0.00015032266651648918
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] past dt [0.7303 0.4761 0.699  0.3755 0.2141 0.3417 0.556  0.323  0.4575 0.2982
 0.5806 0.2753 0.3471 0.5171 0.6661 0.4308 0.4055 0.3143]
scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`27402 ms` LLM)
- **Rationale**: Read the prior panel implementation without truncation to resolve how the 32 reference values map onto four features. Check training-label coverage before choosing a joint strain/rate objective; missing labels would require masking.
- **Alternatives Weighed**: `Use the retrieved control-panel skill` (rejected: Its shared scanner-gain model does not implement per-lab location and scale estimated from the reference coupons.)

```bash
sed -n '49,120p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']; a=D['train']; b=D['test']; labs=np.r_[a['lab_id'],b['lab_id']]; strata=np.r_[a['stratum'],b['stratum']]; ref=np.r_[a['ref_coupons'],b['ref_coupons']]; x=np.r_[a['coupon_features_raw'],b['coupon_features_raw']]; print('reference NaN count',np.isnan(ref).sum()); print('strata',np.unique(strata));
for l in range(4):
 m=labs==l; print('lab',l,'cell sizes',np.unique(strata[m],return_counts=True)[1], 'reference first row',np.round(ref[m][0,:12],3),'feature range',np.round(np.nanmin(x[m],axis=0),2),np.round(np.nanmax(x[m],axis=0),2))
y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']]; print('response quantiles',np.nanpercentile(y,[0,10,50,90,100],axis=0)); print('nonfinite labels',(~np.isfinite(y)).sum())
PY
```
- **Container Exit Code**: `0` (`421 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
len(hold), len(t) + 1, 7))
    a[:, :, :2] = gap[:, None, :] * phi
    a[:, :, 2:] = np.column_stack((np.ones(len(hold)), features))[:, None, :] * np.r_[t, 1.0][None, :, None]
    da = gap[:, None, :] * dphi - dz[:, None, :] * phi
    return a, da


def _scale(e):
    return max(1.4826 * np.median(np.abs(e - np.median(e))), 1e-7)


def _fit(args, target, mode='cauchy'):
    shape = target.shape
    y = target.ravel()
    multiplier = np.ones(shape).ravel()
    cached_log = None

    def terms(logs):
        nonlocal cached_log, basis, deriv
        if cached_log is None or not np.array_equal(logs, cached_log):
            a, da = _basis(logs, *args)
            basis, deriv = a.reshape(-1, 7), da.reshape(-1, 2)
reference NaN count 340
strata [0 1 2]
lab 0 cell sizes [32 27 16] reference first row [-2.377 -1.187 -0.587 -0.938 -0.038 -3.131    nan -1.025 -0.133 -0.69
  1.201 -0.441] feature range [-2.96 -3.33 -2.84 -2.41] [1.92 1.23 0.83 1.34]
lab 1 cell sizes [12 34 19] reference first row [-0.177  0.217  1.108    nan  1.195  2.041  1.363 -0.318  0.353  0.93
  0.573  2.466] feature range [-2.05 -3.43 -3.   -3.6 ] [3.97 2.67 3.7  3.42]
lab 2 cell sizes [24 12 23] reference first row [-0.663 -0.58   1.464  0.13  -0.249 -0.413 -0.464  0.067 -0.49  -0.114
 -0.184 -0.003] feature range [-1.59 -2.57 -1.8  -2.45] [1.63 1.39 1.98 2.42]
lab 3 cell sizes [13 22 18] reference first row [-0.327  2.058  0.457 -0.981 -0.369    nan -0.376  1.034  2.466  1.452
  1.899  0.135] feature range [-2.23 -2.44 -2.33 -2.69] [4.58 3.57 5.59 4.25]
response quantiles [[-1.15957744e+00 -4.32351410e+00 -6.65085035e+00 -2.66236565e+00
  -4.61687797e+00 -2.92572312e+00 -3.03599628e+00 -3.14827512e+00
  -1.13449499e+01 -3.99935716e+00 -4.34179815e+00]
 [-5.97847056e-01 -1.18739315e+00 -1.43614712e+00 -1.59142627e+00
  -1.77836499e+00 -1.78129107e+00 -1.86467629e+00 -1.90787505e+00
  -1.99628071e+00 -2.02519863e+00 -2.49670150e+00]
 [ 2.35770763e-02  2.51044113e-03  1.25673560e-03  1.02433912e-02
   8.17052259e-03  3.83432426e-02  6.44980004e-02  6.89113858e-02
   2.37167132e-02  4.26297791e-02  7.63107184e-02]
 [ 6.29639709e-01  1.22502716e+00  1.67428491e+00  1.66432200e+00
   1.85453301e+00  1.87513728e+00  2.14407062e+00  2.00950112e+00
   2.06119239e+00  2.10365382e+00  2.63753495e+00]
 [ 1.64337882e+00  2.03118398e+00  4.16421492e+00  7.13454131e+00
   6.65437371e+01  5.28319693e+00  5.94817253e+00  3.45793837e+00
   4.51890395e+00  1.19773640e+01  4.46191846e+00]]
nonfinite labels 0
```

### Turn 4 — `execute_in_remote_docker` (`47410 ms` LLM)
- **Rationale**: Reuse the archived panel alignment and implement the specified saturated flow, slip, and initial-rate equations. The public schema check detects a broken output contract before leave-one-lab-out validation.
- **Alternatives Weighed**: `Ordinary least squares on all responses` (rejected: The observed strain spike above 66 is far larger than the bulk response and would dominate the fit.); `Sort fitted branches before returning` (rejected: The cross-gap slip term specifically uses branch 0, so swapping branches changes the physical prediction.)

```bash
set -e
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS' | head -12
python3 - <<'PY'
from pathlib import Path
src=Path('/tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py').read_text()
header=src[:src.index('def _basis')].replace('Calibrated two-branch creep model for the amorphous melt campaign.', 'Calibrated two-branch creep model for the jammed-slip campaign.')
Path('/holdout_task/alpha_m97_c03_batch_03_solver.py').write_text(header)
PY
cat >> /holdout_task/alpha_m97_c03_batch_03_solver.py <<'PY'
def _basis(logtau, past, hold, features, dt, t):
    tau = np.exp(logtau)
    u, v = dt[:, None] / tau, t[:, None] / tau
    decay = np.exp(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for step, (a, b, da) in enumerate(zip(decay, -np.expm1(-u), u * decay)):
        stress = past[:, step, None]
        dz = a * dz + da * (z - stress)
        z = a * z + b * stress
    gap = hold[:, None] - z
    phi = np.vstack((-np.expm1(-v), 1 / tau))
    dphi = np.vstack((-v * np.exp(-v), -1 / tau))
    terminal = np.r_[t / (1 + 0.85 * t), 1.0]
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = gap[:, None, :] * phi
    a[:, :, 2:8] = np.column_stack((np.ones(len(hold)), features, slip))[:, None, :] * terminal[None, :, None]
    delta = gap[:, :1] - gap[:, 1:]
    a[:, :, 8] = delta * phi[:, 0]
    da = gap[:, None, :] * dphi - dz[:, None, :] * phi
    ds = np.stack((delta * dphi[:, 0] - dz[:, :1] * phi[:, 0], dz[:, 1:] * phi[:, 0]), axis=-1)
    return a, da, ds


def _scale(e):
    return max(1.4826 * np.median(np.abs(e - np.median(e))), 1e-7)


def _fit(args, target, mode='cauchy'):
    y = target.ravel()
    mult = np.ones_like(target)
    cached = None

    def terms(logs):
        nonlocal cached, a, da, ds
        if cached is None or not np.array_equal(logs, cached):
            basis, derivative, cross = _basis(logs, *args)
            a, da, ds = basis.reshape(-1, 9), derivative.reshape(-1, 2), cross.reshape(-1, 2)
            cached = logs.copy()
        return a, da, ds

    a = da = ds = None

    def residual(p):
        basis, _, _ = terms(p[:2])
        return np.einsum('ij,j->i', basis, p[2:]) - y

    def fun(p):
        return residual(p) * mult.ravel()

    def jac(p):
        basis, derivative, cross = terms(p[:2])
        return np.column_stack((derivative * p[2:4] + cross * p[-1], basis)) * mult.reshape(-1, 1)

    bounds = (np.r_[np.log([0.015, 0.015]), [-np.inf] * 9],
              np.r_[np.log([120., 120.]), [np.inf] * 9])

    def fit(p, loss, scale):
        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=scale, x_scale='jac', max_nfev=140)

    starts = []
    for pair in ((0.3, 2.0), (0.65, 4.5), (1.5, 8.0), (2.0, 0.3), (4.5, 0.65), (8.0, 1.5)):
        logs = np.log(pair)
        basis, _, _ = terms(logs)
        starts.append(np.r_[logs, np.linalg.lstsq(basis, y, rcond=None)[0]])
    scale = min(_scale(residual(p)) for p in starts)
    best = min((fit(p, 'soft_l1', scale) for p in starts), key=lambda result: result.cost)
    if mode != 'soft_l1':
        for _ in range(3):
            err = residual(best.x).reshape(target.shape)
            curve_scale, rate_scale = _scale(err[:, :-1]), _scale(err[:, -1])
            mult[:, -1] = curve_scale / rate_scale
            best = fit(best.x, mode, 2.25 * curve_scale)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    taus, weights = _fit((past[:n], hold[:n], features[:n], dt, t), target)
    basis, _, _ = _basis(np.log(taus), past[n:], hold[n:], features[n:], dt, t)
    pred = basis @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`723 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`28841 ms` LLM)
- **Rationale**: Compare robust objectives on four held-out labs, reporting strain and rate separately. A loss that worsens the weakest lab or destabilizes the two relaxation times needs adjustment; a derivative mismatch requires fixing the optimizer Jacobian before interpreting those results.
- **Alternatives Weighed**: `Verify only the public output schema` (rejected: Correct shapes and finite values do not catch a failed lab transfer, unstable branch identification, or a wrong derivative.)

```bash
set -e
cat > /tmp/rheology_oof.py <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_m97_c03_batch_03_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]
d=item['dataset']; tr=d['train']; lab=tr['lab_id']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d)
y=np.column_stack((tr['strain_curves_obs']-o[:n,None],tr['instant_rate_obs']))
def args(mask): return p[:n][mask],h[:n][mask],x[:n][mask],dt,t
for mode in ('cauchy','soft_l1'):
    folds=[]
    for l in np.unique(lab):
        tau,w=m._fit(args(lab!=l),y[lab!=l],mode)
        a,_,_=m._basis(np.log(tau),*args(lab==l)); e=abs(a@w-y[lab==l]); folds.append(e)
        print(mode,'lab',l,'taus',np.round(tau,4),'rw',np.round(w[7:],4),'abs median/q90 strain',np.round(np.quantile(e[:,:-1],[.5,.9]),4),'rate',np.round(np.quantile(e[:,-1],[.5,.9]),4),'worst horizon median',round(np.median(e[:,:-1],axis=0).max(),4))
    print(mode,'worst lab medians',max(np.median(e[:,:-1]) for e in folds),max(np.median(e[:,-1]) for e in folds))
tau,w=m._fit(args(np.ones(n,bool)),y); print('full fit taus/weights',tau,w)
a,da,ds=m._basis(np.log(tau),*args(np.ones(n,bool))); logs=np.log(tau); eps=1e-6
for j in range(2):
    delta=np.eye(2)[j]*eps
    fd=((m._basis(logs+delta,*args(np.ones(n,bool)))[0]-m._basis(logs-delta,*args(np.ones(n,bool)))[0])@w)/(2*eps)
    err=np.max(abs(fd-da[:,:,j]*w[j]-ds[:,:,j]*w[-1])); print('logtau derivative max error',j,err); assert err<1e-7
print('public truth field names',list(item['gt']) if isinstance(item['gt'],dict) else type(item['gt']).__name__)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_oof.py
```
- **Container Exit Code**: `0` (`1131 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cauchy lab 0 taus [0.5316 4.0171] rw [ 0.308  -0.2178] abs median/q90 strain [0.0135 0.1284] rate [0.016  0.0433] worst horizon median 0.0188
cauchy lab 1 taus [0.5316 4.0154] rw [ 0.3105 -0.2173] abs median/q90 strain [0.0122 0.0951] rate [0.0094 0.0355] worst horizon median 0.0159
cauchy lab 2 taus [0.5318 3.9824] rw [ 0.3105 -0.2175] abs median/q90 strain [0.0123 0.0717] rate [0.0156 0.0442] worst horizon median 0.0161
cauchy lab 3 taus [0.5322 4.039 ] rw [ 0.3047 -0.2217] abs median/q90 strain [0.0117 0.035 ] rate [0.0116 0.0284] worst horizon median 0.0135
cauchy worst lab medians 0.013539666184140486 0.016006157989956415
soft_l1 lab 0 taus [0.53   4.1749] rw [ 0.292  -0.2221] abs median/q90 strain [0.0144 0.1289] rate [0.0133 0.0507] worst horizon median 0.0203
soft_l1 lab 1 taus [0.5341 4.205 ] rw [ 0.2955 -0.2235] abs median/q90 strain [0.0124 0.0894] rate [0.0127 0.0399] worst horizon median 0.0136
soft_l1 lab 2 taus [0.5333 4.1723] rw [ 0.2971 -0.2212] abs median/q90 strain [0.0133 0.0645] rate [0.0178 0.0473] worst horizon median 0.0162
soft_l1 lab 3 taus [0.5318 4.2272] rw [ 0.2839 -0.2287] abs median/q90 strain [0.0135 0.0447] rate [0.0154 0.0338] worst horizon median 0.0164
soft_l1 worst lab medians 0.01441699481802372 0.017791848279418793
full fit taus/weights [0.53178338 4.01119402] [ 0.67805223  0.55284698  0.13897391  0.1572188  -0.12771668  0.11840395
 -0.08773985  0.30865636 -0.21854023]
logtau derivative max error 0 2.689640177244712e-10
logtau derivative max error 1 2.2281201189633393e-10
public truth field names ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true']

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27165 ms` LLM)
- **Rationale**: Keep the Cauchy fit, which improved both worst-lab median errors over soft L1. Audit all five outputs against the public clean references and test the specified sparse and EMI calibration paths; failures would identify whether to revise parameter recovery, prediction, or calibration.
- **Alternatives Weighed**: `Submit based on leave-one-lab-out results alone` (rejected: Noisy response residuals do not directly verify parameter recovery or clean test predictions, and the public sweep does not exercise the required calibration fallbacks.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',root+'alpha_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(root+'public_alpha_m97_c03_batch_03.pkl','rb')); d,g=item['dataset'],item['gt']
start=time.monotonic(); out=m.solve_alpha_m97_c03_batch_03(d); print('solve_seconds',round(time.monotonic()-start,3))
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta'),('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    ref=np.asarray(g[b]); assert ref.shape==out[a].shape,(a,ref.shape,out[a].shape)
    err=out[a]-ref
    print(a,'rmse',np.sqrt(np.mean(err**2)),'max_abs',np.max(abs(err)))
    if err.ndim==2:
        print('worst horizon RMSE',np.max(np.sqrt(np.mean(err**2,axis=0))))
    if err.ndim==2 or a=='initial_rate_pred':
        for lab in range(4):
            e=err[d['test']['lab_id']==lab]; print('lab',lab,'RMSE',np.sqrt(np.mean(e**2)))
cal=d['calibration_load_cells']; clean=m._calibration(cal); emi={k:v.copy() for k,v in cal.items()}; i=np.flatnonzero(emi['lab_id']==1)[0]; emi['true_stress'][i]+=5
keep=np.arange(len(cal['lab_id']))!=i; expected=m._calibration({k:v[keep] for k,v in cal.items()}); np.testing.assert_allclose(m._calibration(emi),expected,atol=1e-10)
keep=(cal['lab_id']!=2); keep[np.flatnonzero(~keep)[0]]=True; fallback=m._calibration({k:v[keep] for k,v in cal.items()}); np.testing.assert_allclose(fallback[2],m._prior_calibration(2)); np.testing.assert_allclose(fallback[[0,1,3]],clean[[0,1,3]])
print('EMI and sparse-channel calibration checks passed')
PY
```
- **Container Exit Code**: `1` (`715 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
TypeError: list indices must be integers or slices, not str
```

### Turn 7 — `execute_in_remote_docker` (`25155 ms` LLM)
- **Rationale**: The schema passed, but the follow-up audit stopped before testing the solver because it indexed the pickle's outer list as a dictionary. Select its first record and finish the clean-reference and calibration fallback checks; their results will determine whether the solver needs changes.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',root+'alpha_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_03.pkl','rb') as f:
    item=pickle.load(f)[0]
d,g=item['dataset'],item['gt']
start=time.monotonic(); out=m.solve_alpha_m97_c03_batch_03(d); print('solve_seconds', round(time.monotonic()-start,3))
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta'),('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    ref=np.asarray(g[b]); assert ref.shape==out[a].shape,(a,ref.shape,out[a].shape)
    err=out[a]-ref
    print(a,'rmse',np.sqrt(np.mean(err**2)),'max_abs',np.max(abs(err)))
    if err.ndim==2:
        print('worst horizon RMSE',np.max(np.sqrt(np.mean(err**2,axis=0))))
    if err.ndim==2 or a=='initial_rate_pred':
        print('lab RMSEs',[float(np.sqrt(np.mean(err[d['test']['lab_id']==lab]**2))) for lab in range(4)])
cal=d['calibration_load_cells']; clean=m._calibration(cal); emi={k:v.copy() for k,v in cal.items()}; i=np.flatnonzero(emi['lab_id']==1)[0]; emi['true_stress'][i]+=5
keep=np.arange(len(cal['lab_id']))!=i; expected=m._calibration({k:v[keep] for k,v in cal.items()}); np.testing.assert_allclose(m._calibration(emi),expected,atol=1e-10)
keep=cal['lab_id']!=2; keep[np.flatnonzero(~keep)[0]]=True; fallback=m._calibration({k:v[keep] for k,v in cal.items()}); np.testing.assert_allclose(fallback[2],m._prior_calibration(2)); np.testing.assert_allclose(fallback[[0,1,3]],clean[[0,1,3]])
print('EMI and sparse-channel calibration checks passed')
PY
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
solve_seconds 0.084
estimated_taus rmse 0.005586301729941727 max_abs 0.007805982547387735
hereditary_weights rmse 0.0021894651089126656 max_abs 0.0028469824179774283
regime_mode_weights rmse 0.0014029086806799632 max_abs 0.0014597699214941307
predicted_strain rmse 0.028580740442306136 max_abs 0.18421465062708364
worst horizon RMSE 0.03526323564484198
lab RMSEs [0.04384251044198794, 0.012813116471624452, 0.002382538893697766, 0.020359064894582536]
initial_rate_pred rmse 0.03660613525835706 max_abs 0.19073434896736563
lab RMSEs [0.05611744601436323, 0.016664665765574325, 0.0032562639767319684, 0.0259877051241276]
EMI and sparse-channel calibration checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`23845 ms` LLM)
- **Rationale**: The schema and calibration fallback checks passed. Split clean-test errors by lab and coupon missingness, and compare reconstructed stresses to their public references. If complete coupons also show large errors, revisit alignment or stress conversion; if errors concentrate on missing coupons, check the imputation rule rather than retuning the optimizer.
- **Alternatives Weighed**: `Retune the robust optimizer` (rejected: The relaxation times and weights are already close to the clean public references; first determine whether the worst-lab errors come from missing coupon features or stress conversion.)

```bash
set -e
sed -n '19,100p' /holdout_task/alpha_m97_c03_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_03.pkl','rb') as f: item=pickle.load(f)[0]
d,g=item['dataset'],item['gt']; o=m.solve_alpha_m97_c03_batch_03(d); lab=d['test']['lab_id']; miss=np.isnan(d['test']['coupon_features_raw']).sum(axis=1)
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; rmse=lambda a: float(np.sqrt(np.mean(a*a)))
for l in range(4):
 for missing in (False,True):
  ix=(lab==l)&((miss>0)==missing)
  if ix.any(): print('lab',l,'missing',missing,'n',ix.sum(),'strain/rate RMSE',rmse(e[ix]),rmse(r[ix]))
p,h,x,offset,dt,t=m._prepare(d); n=len(d['train']['lab_id']); print('max stress errors, history/hold', np.max(abs(p[n:]-g['te_hist_true'])),np.max(abs(h[n:]-g['te_hold_true'])))
PY
```
- **Container Exit Code**: `0` (`570 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[rows] = (features[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            cell = features[rows & (stratum == s)]
            count = np.isfinite(cell).sum(axis=0)
            cells.append(np.divide(np.nansum(cell, axis=0), count,
                                   out=np.full(features.shape[1], np.nan), where=count > 0))
    features = np.where(np.isnan(features), np.nanmean(cells, axis=0), features)
    gain, bias = _calibration(dataset['calibration_load_cells'])[lab].T
    past = join('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    return past, hold, features, join('strain_0'), np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])


def _basis(logtau, past, hold, features, dt, t):
    tau = np.exp(logtau)
    u, v = dt[:, None] / tau, t[:, None] / tau
    decay = np.exp(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for step, (a, b, da) in enumerate(zip(decay, -np.expm1(-u), u * decay)):
        stress = past[:, step, None]
        dz = a * dz + da * (z - stress)
        z = a * z + b * stress
    gap = hold[:, None] - z
    phi = np.vstack((-np.expm1(-v), 1 / tau))
    dphi = np.vstack((-v * np.exp(-v), -1 / tau))
    terminal = np.r_[t / (1 + 0.85 * t), 1.0]
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
lab 0 missing False n 21 strain/rate RMSE 0.0016707683103876607 0.0021977077111152727
lab 0 missing True n 5 strain/rate RMSE 0.09991765406229718 0.12788818016040093
lab 1 missing False n 17 strain/rate RMSE 0.003792915502008247 0.005128304056359865
lab 1 missing True n 1 strain/rate RMSE 0.05206343852249949 0.06746634708507182
lab 2 missing False n 13 strain/rate RMSE 0.001758686337628162 0.0025277369553937585
lab 2 missing True n 1 strain/rate RMSE 0.0062659534690002764 0.008085954965410647
lab 3 missing False n 13 strain/rate RMSE 0.0021106640391838015 0.002937909571009583
lab 3 missing True n 1 strain/rate RMSE 0.07579556446404746 0.09665839012855493
max stress errors, history/hold 9.035782730437703e-05 8.635183194005691e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`23597 ms` LLM)
- **Rationale**: Check whether the asset index specifies a separate imputation convention or acceptance threshold, whether the public records exercise another regime, and whether any occupied lab–stratum cell lacks observations for a feature. These results decide whether the existing verified artifact can be submitted unchanged or needs a targeted correction.
- **Alternatives Weighed**: `Replace the prescribed mean imputation with a predictive model` (rejected: The error is concentrated on missing inputs, but that does not show the specified cell-balanced mean was implemented incorrectly or authorize replacing it.)

```bash
set -e
python3 -I - <<'PY'
import json, pickle
import numpy as np
root='/holdout_task/'
with open(root+'public_asset_index.json') as f:
    idx=json.load(f)
for line in json.dumps(idx,indent=2).splitlines():
    if any(word in line.lower() for word in ('imput','cell-balanced','toleran','threshold','rmse','library','runtime')):
        print(line.strip())
with open(root+'public_alpha_m97_c03_batch_03.pkl','rb') as f:
    items=pickle.load(f)
print('Public records:',len(items))
for i,item in enumerate(items):
    d=item['dataset']
    lab=np.r_[d['train']['lab_id'],d['test']['lab_id']]
    stratum=np.r_[d['train']['stratum'],d['test']['stratum']]
    x=np.r_[d['train']['coupon_features_raw'],d['test']['coupon_features_raw']]
    counts=np.asarray([np.isfinite(x[(lab==l)&(stratum==s)]).sum(axis=0) for l,s in sorted(set(zip(lab,stratum)))])
    print(i,'banding',d['shear_banding_index'],'occupied cells',len(counts),'minimum observed counts per feature',counts.min(axis=0).tolist(),'missing train/test',[int(np.isnan(d[k]['coupon_features_raw']).sum()) for k in ('train','test')])
PY
```
- **Container Exit Code**: `0` (`365 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public records: 1
0 banding 0.82 occupied cells 12 minimum observed counts per feature [12, 12, 11, 12] missing train/test [22, 8]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 10 — `submit_completed_task` (`17674 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process schema validation, clean-reference checks for all five outputs, and both calibration stress checks. Leave-one-lab-out validation favored the Cauchy fit, and the remaining error was concentrated on incomplete coupons rather than stress conversion.
- **Alternatives Weighed**: `Replace cell-balanced mean imputation` (rejected: The task prescribes cell-balanced means, and validation did not identify an error in that implementation.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:7194'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema and sparse-channel and EMI calibration checks passed. Against clean public references, RMSE was 0.02858 for strain and 0.03661 for initial rate. Missing coupon features accounted for most of the remaining prediction error.`

### Turn 11 — `execute_in_remote_docker` (`36472 ms` LLM)
- **Rationale**: An independent convolution checks the exact rational and cross-gap terms and the initial derivative; a discrepancy requires correcting the physical basis. A matched raw-versus-aligned leave-one-lab-out comparison tests whether panel correction improves transfer or leaves a lab-specific bias. The isolated schema check verifies the unchanged artifact.
- **Alternatives Weighed**: `Audit relay modes, PDE drift, or classification recall` (rejected: This batch specifies a continuous rheology response with no relay states, PDE operator, or classification target; the relevant checks are the slip denominator, cross-gap term, signed stress slices, and lab transfer.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_03.pkl','rb') as f: item=pickle.load(f)[0]
d,g=item['dataset'],item['gt']; out=m.solve_alpha_m97_c03_batch_03(d)
tr,te=d['train'],d['test']; n=len(tr['lab_id']); p,h,x,o,dt,t=m._prepare(d)
tau=out['estimated_taus']; hw=out['hereditary_weights']; rw=out['regime_mode_weights']
# Independent discrete convolution, without the solver's recurrence or basis.
after=np.cumsum(dt[::-1])[::-1]-dt
gaps=np.column_stack([h[n:]-p[n:]@((1-np.exp(-dt/u))*np.exp(-after/u)) for u in tau]); a,b=gaps.T
flow=hw[2]+x[n:]@hw[3:]+rw[0]*np.sign(h[n:])*np.sqrt(abs(h[n:]))
def direct(times):
    f,j=(1-np.exp(-times[:,None]/tau)).T
    return o[n:,None]+np.outer(hw[0]*a+rw[1]*(a-b),f)+np.outer(hw[1]*b,j)+np.outer(flow,times/(1+0.85*times))
rate=(hw[0]*a+rw[1]*(a-b))/tau[0]+hw[1]*b/tau[1]+flow
np.testing.assert_allclose(out['predicted_strain'],direct(t),rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'],rate,rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(direct(np.array([0.]))[:,0],o[n:]); eps=1e-5
np.testing.assert_allclose((direct(np.array([eps]))[:,0]-direct(np.array([-eps]))[:,0])/(2*eps),rate,atol=1e-8)
print('Exact jammed-slip grammar and zero-time slope passed')
# Compare raw vs panel-aligned regressors using held-out lab responses, with identical complete cases.
y=np.column_stack((tr['strain_curves_obs']-o[:n,None],tr['instant_rate_obs']))
raw=tr['coupon_features_raw']; complete=np.isfinite(raw).all(axis=1); labs=tr['lab_id']; worst={}
for name,features in [('aligned',x[:n]),('raw',raw)]:
    scores=[]
    for lab in np.unique(labs):
        fit=complete&(labs!=lab); val=complete&(labs==lab)
        u,w=m._fit((p[:n][fit],h[:n][fit],features[fit],dt,t),y[fit]); B,_,_=m._basis(np.log(u),p[:n][val],h[:n][val],features[val],dt,t)
        err=B@w-y[val]; scores.append((np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])))); print(name,'lab',lab,'median absolute strain/rate',np.round(scores[-1],6),'median signed strain/rate',np.round([np.median(err[:,:-1]),np.median(err[:,-1])],6))
    worst[name]=np.max(scores,axis=0)
print('Worst complete-case held-out lab errors',worst)
for name,pred,truth in [('strain',out['predicted_strain'],g['te_clean']),('rate',rate,g['te_true_rate0'])]:
    err=pred-truth
    print('Clean',name,'RMSE by hold stress sign',[(int(mask.sum()),float(np.sqrt(np.mean(err[mask]**2)))) for mask in [h[n:]<0,h[n:]>=0] if mask.any()])
PY
```
- **Container Exit Code**: `0` (`1467 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Exact jammed-slip grammar and zero-time slope passed
aligned lab 0 median absolute strain/rate [0.01066  0.011318] median signed strain/rate [-2.7e-05 -8.1e-05]
aligned lab 1 median absolute strain/rate [0.010822 0.008015] median signed strain/rate [ 0.001526 -0.002176]
aligned lab 2 median absolute strain/rate [0.011595 0.014008] median signed strain/rate [-0.002751 -0.001738]
aligned lab 3 median absolute strain/rate [0.011361 0.010514] median signed strain/rate [0.000437 0.000728]
raw lab 0 median absolute strain/rate [0.047078 0.075818] median signed strain/rate [-0.017568 -0.02095 ]
raw lab 1 median absolute strain/rate [0.025395 0.040608] median signed strain/rate [0.017111 0.03023 ]
raw lab 2 median absolute strain/rate [0.048347 0.073307] median signed strain/rate [-0.030174 -0.035313]
raw lab 3 median absolute strain/rate [0.053527 0.069577] median signed strain/rate [0.027731 0.040338]
Worst complete-case held-out lab errors {'aligned': array([0.01159539, 0.01400764]), 'raw': array([0.05352738, 0.07581832])}
Clean strain RMSE by hold stress sign [(33, 0.021876885951040065), (39, 0.033212636443040316)]
Clean rate RMSE by hold stress sign [(33, 0.027983711155625195), (39, 0.04255878317991856)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=abe938e5165fbbb9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_abe938e5 (directory).
```

### Turn 12 — `submit_completed_task` (`17255 ms` LLM)
- **Rationale**: The final artifact passed isolated-process verification. Independent convolution confirmed the specified response and zero-time slope, and panel alignment improved held-out transfer in every lab for both strain and rate. No further edit is warranted.
- **Alternatives Weighed**: `Replace cell-balanced mean imputation` (rejected: Missing coupons drive the largest errors, but the task prescribes cell-balanced mean imputation and the audit did not identify an implementation error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema, sparse and EMI calibration checks, and independent strain and initial-rate calculations passed. Against clean public references, RMSE was 0.02858 for strain and 0.03661 for initial rate. Missing coupon features accounted for most of the remaining error.`

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
