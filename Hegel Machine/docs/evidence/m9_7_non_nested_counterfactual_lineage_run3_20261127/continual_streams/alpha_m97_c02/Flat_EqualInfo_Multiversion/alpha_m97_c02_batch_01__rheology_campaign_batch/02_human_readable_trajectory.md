# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `407,997` (`394,786` / `125,440` / `13,211`) |
| **Total Episode Wall Time (s)** | `434.2s` |

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

### Turn 1 — `execute_in_remote_docker` (`14704 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver before fitting: this will distinguish reusable calibration and alignment code from prior-phase constitutive terms that would mispredict the plateau. Look up the relevant primitives before implementing them.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; sed -n '1,250p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation' || true
```
- **Container Exit Code**: `0` (`315 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    train_past, train_hold = _physical(train, calibration)
    taus, weights = _fit(train_past, train_hold, aligned[:n], dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    a, b = _basis(*_physical(test, calibration), dt, times, taus)
    drift = weights[2] + aligned[n:] @ weights[3:]
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + a @ weights[:2] + drift[:, None]*times,
            'initial_rate_pred': b @ weights[:2] + drift}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`24899 ms` LLM)
- **Rationale**: The first log truncated the alignment code and data summary. Inspect those narrowly, then count lab and regime coverage and measure calibration residuals: these determine which slices can be validated and whether archive calibration is warranted.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: Its linear-flow basis omits the specified logarithmic plateau and damage-coupled terms.)

```bash
sed -n '1,160p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -75; python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb')); print('items',len(items)); d=items[0]['dataset']; print('dataset keys',list(d)); print('times/dt/crit',d['eval_times'],d['past_dt'],d['crit_stress']);
for part in ('train','test'):
 s=d[part]; print(part)
 for k,v in s.items():
  if isinstance(v,dict):
   print(k,[(f,np.shape(a),np.asarray(a).dtype) for f,a in v.items()]); continue
  a=np.asarray(v); print(k,a.shape,a.dtype, (np.nanmin(a),np.nanmax(a),np.sum(~np.isfinite(a))) if a.dtype.kind in 'fiu' else '')
 c=d['calibration_load_cells']; l=s['lab_id']; x=s['past_stress_cmd']; gain=[]; bias=[]
 for j in range(4):
  m=c['lab_id']==j; p=np.polyfit(c['cmd_stress'][m],c['true_stress'][m],1); gain.append(p[0]);bias.append(p[1]); print('calibration',j,m.sum(),'max residual',np.max(abs(np.polyval(p,c['cmd_stress'][m])-c['true_stress'][m])))
 z=x*np.asarray(gain)[l,None]+np.asarray(bias)[l,None]; active=np.sum(np.maximum(0,abs(z)-d['crit_stress'])**2*d['past_dt'],axis=1)>0
 if 'shear_banding_index' in s:
  low=s['shear_banding_index']<=.5; print('counts by lab/low/damage',[(j,int(b),int(a),int(np.sum((l==j)&(low==b)&(active==a)))) for j in range(4) for b in (0,1) for a in (0,1)])
PY
```
- **Container Exit Code**: `0` (`434 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_control_panel_deconfounding",
      "reference_source_txt": null
    }
  ]
}
items 1
dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
times/dt/crit [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] [0.6862 0.3536 0.4115 0.5101 0.4903 0.7741 0.4359 0.5228 0.5538 0.2493
 0.5622 0.3497 0.6793 0.3183 0.7778 0.7773 0.754  0.2048] 2.65
train
lab_id (180,) int64 (0, 3, 0)
stratum (180,) int64 (0, 2, 0)
coupon_features_raw (180, 4) float64 (-3.872691122990179, 5.177864993438177, 20)
ref_coupons (180, 32) float64 (-4.76307065482229, 5.395998560530416, 225)
past_stress_cmd (180, 18) float64 (-4.873684035853066, 5.22664370440438, 0)
stress_hold_cmd (180,) float64 (-2.3022823529411767, 2.6337329113924053, 0)
strain_0 (180,) float64 (-0.823836, 0.820409, 0)
aux_chamber_telemetry (180, 3) float64 (-3.266351, 3.284674, 0)
strain_curves_obs (180, 10) float64 (-17.781020196386567, 25.323727123410176, 0)
instant_rate_obs (180,) float64 (-5.111013802613548, 4.849736995223994, 0)
calibration 0 10 max residual 0.00019010847293876587
calibration 1 10 max residual 0.00019176057234515653
calibration 2 10 max residual 0.00012464239328244253
calibration 3 10 max residual 0.00014402498836063593
test
lab_id (72,) int64 (0, 3, 0)
stratum (72,) int64 (0, 2, 0)
coupon_features_raw (72, 4) float64 (-3.0836794474795886, 4.363033579332407, 4)
ref_coupons (72, 32) float64 (-3.850053664920206, 4.970931605611035, 99)
past_stress_cmd (72, 18) float64 (-4.884058104327808, 5.13679653485851, 0)
stress_hold_cmd (72,) float64 (-2.3543253164556956, 2.5809911392405063, 0)
strain_0 (72,) float64 (-1.061798, 0.518879, 0)
aux_chamber_telemetry (72, 3) float64 (-2.668785, 3.016834, 0)
calibration 0 10 max residual 0.00019010847293876587
calibration 1 10 max residual 0.00019176057234515653
calibration 2 10 max residual 0.00012464239328244253
calibration 3 10 max residual 0.00014402498836063593
```

### Turn 3 — `execute_in_remote_docker` (`22126 ms` LLM)
- **Rationale**: The previous output omitted the alignment helper and archive format. Read those sections directly, then measure zero-damage and lab–stratum coverage to determine whether the terminal-flow fallback can be validated on public observations.
- **Alternatives Weighed**: `Reuse the prior solver's entire constitutive model` (rejected: Its terminal-flow basis omits the specified logarithmic plateau and both damage-coupled terms.)

```bash
sed -n '53,119p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d=item['dataset']; print('item keys',list(item)); print('shear, yield',d['shear_banding_index'],d['yield_stress']); c=d['calibration_load_cells']; fits=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==i],c['true_stress'][c['lab_id']==i],1) for i in range(4)])
for key in ('train','test'):
 s=d[key]; l=s['lab_id']; past=s['past_stress_cmd']*fits[l,0,None]+fits[l,1,None]; D=(np.maximum(abs(past)-d['crit_stress'],0)**2*d['past_dt']).sum(1); print(key,'zero damage per lab',np.bincount(l[D==0],minlength=4),'positive damage quantiles',np.quantile(D[D>0],[0,.1,.5,.9,1])); print('lab/stratum counts',np.bincount(3*l+s['stratum'],minlength=12)); print('max missing in cell',max(np.isnan(s['coupon_features_raw'][(l==j)&(s['stratum']==k)]).sum(axis=0).max() for j in range(4) for k in range(3)))
PY
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
)*30)), np.full(7, np.inf)])

    def design(logtau):
        a, b = _basis(past, hold, dt, times, np.exp(logtau))
        return np.vstack((np.column_stack((a.reshape(n*k, 2), tdrift)),
                          np.column_stack((b, drift))))

    def residual(p):
[
  {
    "episode_short_name": "alpha_m97_c02_s00",
    "lab_id": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "cmd_stress": [
      1.619455,
      -1.307963,
      -0.646756,
      0.95759,
      2.133391,
      -0.419148,
      -0.546245,
      -1.403033,
      1.416515,
      -0.474218,
      0.56263,
      -1.431472,
      -1.102381,
      0.100096,
      0.332884,
      0.978926,
      1.410301,
      2.014558,
      2.037985,
      -1.520981,
      -1.132282,
      -0.400487,
      1.428566,
      2.184907,
      2.114515,
      2.024265,
      0.373395,
      1.839347,
      -1.383881,
      0.252459,
      -1.650907,
      2.152332,
      0.673079,
      0.633995,
      2.133025,
      0.157003,
      -0.816267,
      -1.031715,
      -0.029779,
      1.574899
    ],
    "true_stress": [
      1.619593,
      -1.308033,
      -0.646936,
      0.95752,
      2.133164,
      -0.418978,
      -0.546283,
      -1.402876,
      1.416566,
      -0.474167,
      0.562697,
      -1.431478,
      -1.102292,
      0.1,
      0.333007,
      0.979026,
      1.410368,
      2.014512,
      2.037978,
      -1.520941,
      -1.132263,
      -0.400526,
      1.428582,
      2.184857,
      2.114623,
      2.024408,
      0.373485,
      1.83932,
      -1.383844,
      0.25249,
      -1.650978,
      2.152261,
      0.672955,
      0.634083,
      2.133016,
      0.157033,
      -0.816183,
      -1.031889,
      -0.02981,
      1.574906
    ]
  }
]item keys ['dataset', 'gt']
shear, yield 0.12 2.65
train zero damage per lab [11 13 21 15] positive damage quantiles [1.99669092 2.48437064 3.54616811 4.41294492 5.0070849 ]
lab/stratum counts [19 16  6  7 20 10 17 15 20 10 15 25]
max missing in cell 2
test zero damage per lab [ 7  3  2 12] positive damage quantiles [2.08632099 2.43794021 3.29681505 4.38624978 4.67608706]
lab/stratum counts [6 5 6 4 8 4 4 5 7 8 6 9]
max missing in cell 1
```

### Turn 4 — `execute_in_remote_docker` (`51692 ms` LLM)
- **Rationale**: Implement the specified basis and fit the two times jointly across curves and initial rates. Run the isolated schema gate now to catch output shape or finiteness errors before measuring held-out lab, damage, and missing-feature slices.
- **Alternatives Weighed**: `Apply logarithmic flow to all rows` (rejected: The specified gate excludes zero-damage rows, which occur in all four labs.)

```bash
set -e
p=/holdout_task/alpha_m97_c02_batch_01_solver.py
sed '/^def _basis/,$d' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py > "$p"
sed -i '1c\"""Calibrated two-branch Prony regression with a damage-gated plateau."""' "$p"
cat >> "$p" <<'PY'
def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    aligned = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    past, hold = (np.concatenate([a, b]) for a, b in
                  zip(_physical(train, calibration), _physical(test, calibration)))
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    damage = np.square(np.maximum(np.abs(past) - dataset['crit_stress'], 0.)) @ dt
    plateau = (np.asarray(dataset['shear_banding_index']) <= .50) & (damage > 0.)
    active = np.where(plateau, np.tanh(damage), 0.)
    return past, hold, aligned, plateau, active, dt, times


def _design(past, hold, aligned, plateau, active, dt, times, taus):
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (-np.expm1(-dt[:, None] / taus) * np.exp(-later[:, None] / taus))
    gap = hold[:, None] - memory
    response = -np.expm1(-times[:, None] / taus)
    terminal = np.where(plateau[:, None], np.log1p(times), times)
    x = np.column_stack((np.ones(len(hold)), aligned))
    extra = np.column_stack((active * hold, active * memory[:, 0]))
    curve = np.concatenate((gap[:, None, :] * response,
                            x[:, None, :] * terminal[:, :, None],
                            extra[:, None, :1] * response[None, :, :1],
                            extra[:, None, 1:] * terminal[:, :, None]), axis=2)
    rate = np.column_stack((gap / taus, x, extra[:, 0] / taus[0], extra[:, 1]))
    return curve, rate


def _fit(data, curves, rates):
    *_, dt, times = data
    target = np.r_[curves.ravel(), rates]
    bounds = (np.full(2, np.log(dt.min() / 30)),
              np.full(2, np.log(times.max() * 30)))

    def design(log_tau):
        c, r = _design(*data, np.exp(log_tau))
        return np.vstack((c.reshape(-1, 9), r))

    def profile(log_tau, inv_scale):
        a = design(log_tau) * inv_scale[:, None]
        y = target * inv_scale
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, w) - y, w

    unit = np.ones(len(target))
    starts = np.log([[.15, 1.0], [.4, 2.5], [.9, 5.]])
    best = None
    for start in starts:
        start = np.clip(start, *bounds)
        fit = least_squares(lambda q: profile(q, unit)[0], start, bounds=bounds,
                            max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    err, weights = profile(best.x, unit)
    taus = np.exp(best.x)
    if taus[0] > taus[1]:
        taus = taus[::-1]
        weights = profile(np.log(taus), unit)[1]
    # Downweight large residuals, including rows whose missing features were imputed.
    err = profile(np.log(taus), unit)[0]
    nc = curves.size
    mad = lambda v: max(1e-6, 1.4826 * np.median(np.abs(v - np.median(v))))
    scales = np.r_[np.full(nc, mad(err[:nc])), np.full(len(rates), mad(err[nc:]))]
    lo, hi = bounds

    def residual(p):
        return (np.einsum('ij,j->i', design(p[:2]), p[2:]) - target) / scales

    robust = least_squares(residual, np.r_[np.log(taus), weights],
                           bounds=(np.r_[lo, np.full(9, -np.inf)],
                                   np.r_[hi, np.full(9, np.inf)]),
                           loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    taus = np.exp(robust.x[:2])
    if taus[0] <= taus[1]:
        return taus, robust.x[2:]
    # The damage terms reference the fast branch; re-estimate their coefficients.
    taus = taus[::-1]
    return taus, profile(np.log(taus), 1. / scales)[1]


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    taus, w = _fit(tuple(a[:n] for a in data[:5]) + data[5:],
                   train['strain_curves_obs'] - train['strain_0'][:, None],
                   train['instant_rate_obs'])
    curve, rate = _design(*(tuple(a[n:] for a in data[:5]) + data[5:]), taus)
    return {'estimated_taus': taus, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
python3 - <<'PY'
import pickle, numpy as np
item=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]
print('Public reference fields:',[(k,np.shape(v)) for k,v in item['gt'].items()])
PY
```
- **Container Exit Code**: `0` (`664 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public reference fields: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d65475800b790a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d654758 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29221 ms` LLM)
- **Rationale**: The schema gate cannot detect a wrong constitutive basis or biased parameter fit. Compare the frozen solver with the supplied public reference, then refit without each lab’s observations to detect lab-specific errors. Large clean errors on complete rows would warrant checking the basis and alignment; errors concentrated on missing rows would warrant checking imputation.

```bash
cat > /tmp/check_batch01.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(root+'public_alpha_m97_c02_batch_01.pkl','rb'))[0]
d, gt = item['dataset'], item['gt']
train, test = d['train'], d['test']
n = len(train['lab_id'])
data = m._prepare(d)
out = m.solve_alpha_m97_c02_batch_01(d)
for k, g in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta')]:
    print(k, 'fit', np.round(out[k],6), 'ref', gt[g], 'max abs error', round(float(np.max(np.abs(out[k]-gt[g]))),7))
past, hold = data[:2]
print('calibration max history, hold error', np.max(np.abs(past[n:]-gt['te_hist_true'])), np.max(np.abs(hold[n:]-gt['te_hold_true'])))

def errors(label, ce, re, split, plateau):
    lab, st = split['lab_id'], split['stratum']
    missing = np.isnan(split['coupon_features_raw']).any(axis=1)
    subsets = [('all', np.ones(len(lab),bool))]
    subsets += [(f'lab{i}/plateau{int(p)}', (lab==i)&(plateau==p)) for i in range(4) for p in (False,True)]
    subsets += [(f'stratum{i}',st==i) for i in range(3)]
    subsets += [('missing',missing),('complete',~missing)]
    print(label, '(slice, n, curve RMSE, rate RMSE)')
    for s, idx in subsets:
        if idx.any(): print(s, idx.sum(), *[round(float(np.sqrt(np.mean(e[idx]**2))),6) for e in (ce,re)])
    print('curve horizon RMSE', np.round(np.sqrt(np.mean(ce**2,axis=0)),6))

errors('public clean holdout', out['predicted_strain']-gt['te_clean'], out['initial_rate_pred']-gt['te_true_rate0'], test, data[3][n:])
y = train['strain_curves_obs']-train['strain_0'][:,None]
r = train['instant_rate_obs']
cerr, rerr = np.empty_like(y), np.empty_like(r)
for lab in range(4):
    keep = train['lab_id'] != lab
    def sub(mask): return tuple(a[:n][mask] for a in data[:5])+data[5:]
    taus, w = m._fit(sub(keep), y[keep], r[keep])
    ca, ra = m._design(*sub(~keep), taus)
    cerr[~keep], rerr[~keep] = ca@w-y[~keep], ra@w-r[~keep]
    print('lab fold',lab,'taus',np.round(taus,6))
errors('leave-one-lab-out measured', cerr, rerr, train, data[3][:n])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch01.py
```
- **Container Exit Code**: `0` (`852 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus fit [0.496837 3.366375] ref [0.501 3.825] max abs error 0.4586249
hereditary_weights fit [ 0.659688  0.538991  0.098105  0.108761 -0.089821  0.089509 -0.052596] ref [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] max abs error 0.0512387
regime_mode_weights fit [0.28899  0.229752] ref [0.28 0.24] max abs error 0.0102485
calibration max history, hold error 0.0002544414505800674 0.00013367516081164155
public clean holdout (slice, n, curve RMSE, rate RMSE)
all 72 0.16608 0.089341
lab0/plateau0 7 0.231208 0.069673
lab0/plateau1 10 0.107919 0.088918
lab1/plateau0 3 0.228212 0.046676
lab1/plateau1 13 0.137388 0.103557
lab2/plateau0 2 0.144736 0.057035
lab2/plateau1 14 0.126073 0.096008
lab3/plateau0 12 0.228027 0.074545
lab3/plateau1 11 0.136613 0.10156
stratum0 22 0.146942 0.073878
stratum1 24 0.191679 0.098601
stratum2 26 0.155558 0.092254
missing 3 0.230552 0.172641
complete 69 0.162699 0.083863
curve horizon RMSE [0.022236 0.049914 0.07277  0.096195 0.121898 0.150177 0.180871 0.213666
 0.248267 0.284387]
lab fold 0 taus [0.505387 3.584426]
lab fold 1 taus [0.491966 3.559723]
lab fold 2 taus [0.499541 3.476358]
lab fold 3 taus [0.483908 2.724775]
leave-one-lab-out measured (slice, n, curve RMSE, rate RMSE)
all 180 0.962734 0.104797
lab0/plateau0 11 0.839072 0.118009
lab0/plateau1 30 1.079169 0.131996
lab1/plateau0 13 2.286864 0.079229
lab1/plateau1 24 0.386515 0.074882
lab2/plateau0 21 0.415582 0.074028
lab2/plateau1 31 1.15681 0.064451
lab3/plateau0 15 0.224584 0.118902
lab3/plateau1 35 0.437902 0.134043
stratum0 53 0.317035 0.09291
stratum1 66 1.490602 0.109467
stratum2 61 0.493602 0.109279
missing 19 0.543977 0.146667
complete 161 1.000658 0.098691
curve horizon RMSE [0.144934 0.602005 0.217153 0.377313 0.501659 1.907732 0.406858 0.389466
 1.475709 1.519736]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d65475800b790a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d654758 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`30636 ms` LLM)
- **Rationale**: Separate residuals that follow the prescribed terminal basis from irregular residuals across horizons. Using public reference parameters only for diagnosis, check whether discrepancies persist on complete rows and agree with the independently observed initial rate. This determines whether to correct alignment or the design, or instead improve resistance to contaminated observations.
- **Alternatives Weighed**: `Retune the robust loss immediately` (rejected: The late-horizon error and feature-weight bias could come from alignment or a missing basis term; a loss change would not distinguish those from observation spikes.)

```bash
cat /holdout_task/public_asset_index.json; cat > /tmp/diagnose_batch01.py <<'PY'
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
y = train['strain_curves_obs'] - train['strain_0'][:, None]
r = train['instant_rate_obs']
w = np.r_[gt['theta'], gt['vp_theta']]
c, a = m._design(*data, gt['taus'])
for name, split, sl, obs, rate in [('train',train,slice(0,n),y,r),('reference',test,slice(n,None),gt['te_clean']-test['strain_0'][:,None],gt['te_true_rate0'])]:
    p = data[3][sl]; miss = np.isnan(split['coupon_features_raw']).any(1)
    ec = obs - c[sl] @ w; er = rate - a[sl] @ w
    phi = np.where(p[:,None],np.log1p(data[-1]),data[-1]); u=ec/phi
    drift = np.median(u,axis=1)
    irregular = np.max(np.abs(u-drift[:,None]),axis=1)
    q=lambda v: np.round(np.quantile(v,[0,.1,.5,.9,.95,.99,1]),6)
    for gate in [False,True]:
        ix = (p == gate) & ~miss
        print(name,gate,'drift',q(drift[ix]),'rate',q(er[ix]),'rate minus drift',q((er-drift)[ix]),'irregular',q(irregular[ix]))
        for lab in range(4):
            aidx = ix & (split['lab_id']==lab)
            print('lab',lab,'drift quantiles',q(drift[aidx]),'rmse curve/rate',np.sqrt(np.mean(ec[aidx]**2)), np.sqrt(np.mean(er[aidx]**2)))
    z=np.c_[np.ones(len(p)),data[2][sl],np.eye(4)[split['lab_id']][:,1:]]
    ix=(~miss)&(irregular < np.median(irregular)*3)
    if ix.sum()>8:
        for v in (drift,er):
            b=np.linalg.lstsq(z[ix],v[ix],rcond=None)[0]; print(name,'drift regression',np.round(b,6),'residual',q((v-z@b)[ix]))
for maskname, mask in [('complete',~np.isnan(train['coupon_features_raw']).any(1)), ('all',np.ones(n,bool))]:
    a,wfit=m._fit(tuple(a[:n][mask] for a in data[:5])+data[5:],y[mask],r[mask]); print(maskname,'taus/weights',a,wfit)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/diagnose_batch01.py
```
- **Container Exit Code**: `0` (`591 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ntiles [-0.228397 -0.184515 -0.018417  0.09027   0.101515  0.110511  0.11276 ] rmse curve/rate 0.4464793633988278 0.0020276540013014228
lab 1 drift quantiles [-0.200147 -0.157088  0.015148  0.048064  0.052179  0.05547   0.056293] rmse curve/rate 0.43619592814690084 0.005051611334991382
lab 2 drift quantiles [0.080923 0.082918 0.090902 0.098885 0.099883 0.100681 0.100881] rmse curve/rate 0.33035917485374533 0.00444977313115816
lab 3 drift quantiles [-0.25157  -0.160969 -0.025546  0.120886  0.14913   0.175721  0.182369] rmse curve/rate 0.45473064772080873 0.008215621258386825
reference True drift [-0.024669 -0.005987  0.00139   0.010728  0.0132    0.016569  0.017167] rate [-0.024684 -0.006071  0.001333  0.010638  0.013207  0.016556  0.017153] rate minus drift [-1.47e-04 -1.17e-04 -1.40e-05  7.40e-05  1.42e-04  1.81e-04  1.89e-04] irregular [3.00e-06 9.00e-06 3.50e-05 1.00e-04 1.16e-04 1.40e-04 1.46e-04]
lab 0 drift quantiles [-0.003805 -0.000835  0.002659  0.003262  0.003448  0.003597  0.003634] rmse curve/rate 0.003563075531762484 0.0027304054572164106
lab 1 drift quantiles [-0.001448  0.001366  0.006727  0.011186  0.011247  0.011297  0.011309] rmse curve/rate 0.009540029930389037 0.007245594015231691
lab 2 drift quantiles [-0.008103 -0.005824 -0.000364  0.006881  0.009619  0.012862  0.013673] rmse curve/rate 0.007418221614795576 0.005680923963283243
lab 3 drift quantiles [-2.4669e-02 -1.9033e-02  5.5000e-05  1.5809e-02  1.6488e-02  1.7031e-02
  1.7167e-02] rmse curve/rate 0.016149004230971344 0.012333168647150514
reference drift regression [-0.001338 -0.003589  0.002114 -0.002777  0.002722  0.002445 -0.000541
 -0.001063] residual [-8.6170e-03 -2.7230e-03 -9.5000e-05  2.2270e-03  4.6180e-03  9.0440e-03
  1.0254e-02]
reference drift regression [-0.001342 -0.003609  0.002097 -0.002756  0.002734  0.00242  -0.00055
 -0.001062] residual [-0.008595 -0.002676 -0.000119  0.002176  0.004607  0.009029  0.010233]
complete taus/weights [0.49779785 3.38811357] [ 0.66226491  0.53744819  0.10722525  0.12006338 -0.09761444  0.09906609
 -0.05866734  0.28871338  0.23169087]
all taus/weights [0.49683689 3.36637513] [ 0.65968773  0.53899054  0.09810519  0.10876131 -0.08982084  0.08950926
 -0.05259585  0.28899004  0.22975154]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d65475800b790a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d654758 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28461 ms` LLM)
- **Rationale**: Use the initial-rate residual to remove drift-amplitude error, then test whether a common zero-damage time basis explains the remaining curve discrepancy. Fit active rows separately to check whether the current fallback is also biasing the shared relaxation times.
- **Alternatives Weighed**: `Refit the linear fallback with a different robust loss` (rejected: The zero-damage curve discrepancy persists even when the public reference coefficients are supplied.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero-damage rows inherit linear terminal flow from the prior episode.` | Witness: `At the public reference parameters, zero-damage complete rows show curve RMSE of 0.330–0.455 across labs despite initial-rate RMSE of only 0.002–0.0083; active-row residuals instead follow the specified logarithmic drift basis.` | Blanket Diagnosis: `The discrepancy is tied to time evolution and the damage gate, not primarily to stress calibration or a common drift-amplitude error.` | Sublated Pivot: `Separate the damage-amplitude gate from the terminal time-basis choice and test candidate time bases against both measurements.`

```bash
cat > /tmp/zero_damage_batch01.py <<'PY'
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
t = data[-1]
print('times',t)
wref = np.r_[gt['theta'],gt['vp_theta']]
c,a = m._design(*data,gt['taus'])
mask = ~data[3][n:] & np.isfinite(test['coupon_features_raw']).all(1)
o = gt['te_clean'] - test['strain_0'][:,None]
rate_error = gt['te_true_rate0'] - a[n:] @ wref
base = c[n:] @ wref
b = wref[2] + data[2][n:] @ wref[3:7]
for name, phi in [('linear',t),('log',np.log1p(t)),('sqrt',2*(np.sqrt(1+t)-1)),('exp',-np.expm1(-t)),('sat',t/(1+t)),('asinh',np.arcsinh(t))]:
    e=o-base-b[:,None]*(phi-t)-rate_error[:,None]*phi
    print(name,'zero damage diagnostic RMSE',np.sqrt(np.mean(e[mask]**2)))
known_prony = c[n:,:,:2] @ wref[:2]
true_drift = b + rate_error
ratio = (o-known_prony)[mask] / true_drift[mask,None]
print('implied terminal function median',np.median(ratio,axis=0),'spread',np.ptp(ratio,axis=0))
complete=np.isfinite(train['coupon_features_raw']).all(1)
for label, mask in [('active-complete',data[3][:n]&complete),('active-all',data[3][:n])]:
    taus,w=m._fit(tuple(x[:n][mask] for x in data[:5])+data[5:],y[mask],r[mask]); print(label,taus,w)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/zero_damage_batch01.py
```
- **Container Exit Code**: `0` (`548 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
linear zero damage diagnostic RMSE 0.4249806736615531
log zero damage diagnostic RMSE 5.9093795639989146e-05
sqrt zero damage diagnostic RMSE 0.15202000789183678
exp zero damage diagnostic RMSE 0.10878752185319757
sat zero damage diagnostic RMSE 0.14272469948991404
asinh zero damage diagnostic RMSE 0.08856133813883786
implied terminal function median [0.26236448 0.61215524 0.8708354  1.07613156 1.24640437 1.39183433
 1.51879691 1.63142138 1.73265354 1.82455718] spread [0.00039436 0.00219557 0.00430863 0.00627707 0.00801287 0.00953282
 0.01087596 0.01207747 0.01316604 0.01416259]
active-complete [0.50034272 3.774053  ] [ 0.67740533  0.54831458  0.13879136  0.15729553 -0.12765457  0.11787006
 -0.08777321  0.28114004  0.24122384]
active-all [0.49922323 3.74605574] [ 0.67368715  0.54996439  0.13952342  0.15745358 -0.12942988  0.11555669
 -0.08667957  0.280923    0.23984634]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d65475800b790a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d654758 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`24017 ms` LLM)
- **Rationale**: Refit with the corrected low-shear basis and rerun the same lab-by-damage and horizon slices to test whether the improvement holds with estimated parameters. Inspect alignment separately: if missing-feature errors still dominate, the panel and imputation convention is the next issue to check.
- **Alternatives Weighed**: `Fit an unconstrained time basis for zero-damage rows` (rejected: The existing logarithmic basis matches both the curve and the initial slope without adding parameters.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
old = '''    plateau = (np.asarray(dataset['shear_banding_index']) <= .50) & (damage > 0.)
    active = np.where(plateau, np.tanh(damage), 0.)
    return past, hold, aligned, plateau, active, dt, times'''
new = '''    # The low-shear phase uses logarithmic flow even when damage is zero.
    # Damage gates the corrections, not the terminal time basis.
    logarithmic = np.broadcast_to(np.asarray(dataset['shear_banding_index']) <= .50, damage.shape)
    active = np.where(logarithmic, np.tanh(damage), 0.)
    return past, hold, aligned, logarithmic, active, dt, times'''
assert old in s
s = s.replace(old, new).replace('aligned, plateau, active, dt, times, taus', 'aligned, logarithmic, active, dt, times, taus').replace('np.where(plateau[:, None],', 'np.where(logarithmic[:, None],')
p.write_text(s)
PY
sed -i 's/test, data\[3\]\[n:\])/test, data[4][n:] > 0)/; s/train, data\[3\]\[:n\])/train, data[4][:n] > 0)/' /tmp/check_batch01.py
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/check_batch01.py
sed -n '/^def _align(/,/^def _physical(/p' /holdout_task/alpha_m97_c02_batch_01_solver.py
```
- **Container Exit Code**: `0` (`1046 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
ab1/plateau0 3 0.00442 0.005813
lab1/plateau1 13 0.089207 0.067213
lab2/plateau0 2 0.002241 0.003164
lab2/plateau1 14 0.002818 0.003481
lab3/plateau0 12 0.004957 0.004675
lab3/plateau1 11 0.006591 0.005609
stratum0 22 0.028173 0.022491
stratum1 24 0.064535 0.048513
stratum2 26 0.005651 0.00554
missing 3 0.19711 0.149155
complete 69 0.004949 0.004918
curve horizon RMSE [0.008108 0.018946 0.026938 0.033282 0.038556 0.043066 0.047003 0.050492
 0.053627 0.056474]
lab fold 0 taus [0.498868 3.663688]
lab fold 1 taus [0.499368 3.708691]
lab fold 2 taus [0.498536 3.697431]
lab fold 3 taus [0.498184 3.697024]
leave-one-lab-out measured (slice, n, curve RMSE, rate RMSE)
all 180 0.945807 0.046
lab0/plateau0 11 0.870761 0.028279
lab0/plateau1 30 1.077425 0.063969
lab1/plateau0 13 2.253391 0.073456
lab1/plateau1 24 0.37059 0.026481
lab2/plateau0 21 0.2743 0.029161
lab2/plateau1 31 1.152037 0.024004
lab3/plateau0 15 0.151114 0.088584
lab3/plateau1 35 0.395721 0.015972
stratum0 53 0.278095 0.023449
stratum1 66 1.481579 0.07007
stratum2 61 0.44437 0.021308
missing 19 0.468665 0.130748
complete 161 0.987015 0.018664
curve horizon RMSE [0.143708 0.598539 0.207817 0.362876 0.488858 1.893737 0.362359 0.294945
 1.446827 1.501638]
def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        mean = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0))
        aligned[take] = (raw[take] - mean) / scale
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    cell_means = []
    for cell in np.unique(cells):
        values = aligned[cells.ravel() == cell]
        counts = np.isfinite(values).sum(axis=0)
        cell_means.append(np.divide(np.nansum(values, axis=0), counts,
                                    out=np.full(raw.shape[1], np.nan), where=counts > 0))
    fill = np.nanmean(cell_means, axis=0)
    return np.where(np.isfinite(aligned), aligned, fill)


def _physical(split, calibration):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d2f3aef51c6eb408 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d2f3aef5 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`28258 ms` LLM)
- **Rationale**: Check whether the current global fill has collapsed the lab–stratum distinction. Compare candidate fills with the weighted contribution implied by the public clean rates, using reference parameters only for this diagnosis. If no candidate matches, test whether masking observed training features reproduces the failure before changing imputation.
- **Alternatives Weighed**: `Tune relaxation times first` (rejected: The dominant public prediction error is confined to three rows with missing features; the complete-row errors are much smaller for both outputs.)

```bash
cat > /tmp/missing_batch01.py <<'PY'
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
labs, strata, raw, refs = (np.concatenate((train[k],test[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
a = np.empty_like(raw)
for lab in range(4):
    ix = labs == lab
    r = refs[ix]
    a[ix] = (raw[ix]-np.nanmedian(np.nanmean(r,axis=0)))/np.nanmedian(np.nanstd(r,axis=0))
w = np.r_[gt['theta'],gt['vp_theta']]
c, rate = m._design(*data,gt['taus'])
e = gt['te_true_rate0']-rate[n:]@w
for j in np.flatnonzero(np.isnan(a[n:]).any(1)):
    i=n+j; missing=np.isnan(a[i]); cell=(labs==labs[i])&(strata==strata[i]); feat=w[3:7][missing]; old=data[2][i,missing]
    candidates={'global balanced':old, 'own cell':np.nanmean(a[cell],axis=0)[missing], 'own lab':np.nanmean(a[labs==labs[i]],axis=0)[missing], 'own stratum':np.nanmean(a[strata==strata[i]],axis=0)[missing], 'zero':np.zeros(missing.sum())}
    print('row',j,'lab/stratum',labs[i],strata[i],'missing',np.flatnonzero(missing),'inferred weighted fill',round(old@feat+e[j],6),'candidate weighted fills', {k:round(v@feat,6) for k,v in candidates.items()})
print('complete feature correlation',np.round(np.corrcoef(a[np.isfinite(a).all(1)].T),3))
print('own-cell feature means',[(int(l),int(s),np.round(np.nanmean(a[(labs==l)&(strata==s)],axis=0),3).tolist()) for l,s in np.unique(np.column_stack((labs,strata)),axis=0)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/missing_batch01.py
```
- **Container Exit Code**: `0` (`490 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
row 32 lab/stratum 0 0 missing [2] inferred weighted fill -0.102039 candidate weighted fills {'global balanced': -0.018906, 'own cell': -0.011555, 'own lab': -0.003582, 'own stratum': -0.055786, 'zero': 0.0}
row 40 lab/stratum 1 0 missing [0] inferred weighted fill 0.057187 candidate weighted fills {'global balanced': 0.005448, 'own cell': -0.013155, 'own lab': 0.010847, 'own stratum': -0.035908, 'zero': 0.0}
row 43 lab/stratum 1 1 missing [2 3] inferred weighted fill 0.214666 candidate weighted fills {'global balanced': -0.026118, 'own cell': -0.047537, 'own lab': -0.022329, 'own stratum': -0.033486, 'zero': 0.0}
complete feature correlation [[ 1.    -0.002 -0.011 -0.093]
 [-0.002  1.     0.057  0.031]
 [-0.011  0.057  1.     0.044]
 [-0.093  0.031  0.044  1.   ]]
own-cell feature means [(0, 0, [-0.303, 0.01, -0.096, -0.085]), (0, 1, [-0.085, 0.098, -0.172, -0.236]), (0, 2, [0.44, 0.536, 0.339, 0.305]), (1, 0, [-0.082, 0.318, -0.402, -0.057]), (1, 1, [0.105, 0.233, -0.298, 0.131]), (1, 2, [0.104, -0.192, 0.268, -0.096]), (2, 0, [-0.421, -0.113, -0.568, -0.109]), (2, 1, [-0.002, -0.099, -0.199, 0.421]), (2, 2, [0.379, 0.241, -0.102, 0.28]), (3, 0, [0.035, -0.166, -0.875, 0.134]), (3, 1, [-0.051, 0.221, -0.137, 0.057]), (3, 2, [0.291, 0.104, 0.35, 0.217])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d2f3aef51c6eb408 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d2f3aef5 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`31483 ms` LLM)
- **Rationale**: Keep the prescribed imputation and test whether excluding imputed training rows improves parameter estimation. Compare median and 90th-percentile errors on measured, complete rows held out by lab, so isolated spikes do not obscure ordinary prediction error. Keep the existing fit unless a variant improves the worst slices; inspect calibration separately for the specified sparse and EMI fallback cases.
- **Alternatives Weighed**: `Replace balanced imputation with the matching lab–stratum mean` (rejected: The matching cell does not explain the missing contribution and is farther from the implied contribution on all three public incomplete rows.)

```bash
cat > /tmp/fit_batch01_probe.py <<'PY'
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
from scipy.optimize import least_squares
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
complete = np.isfinite(train['coupon_features_raw']).all(1)
lab = train['lab_id']
def sub(mask): return tuple(a[:n][mask] for a in data[:5])+data[5:]
def fit(mask, variant):
    p, w = m._fit(sub(mask), y[mask], r[mask])
    if variant in ('all', 'complete'): return p, w
    yy, rr = y[mask], r[mask]
    c, a = m._design(*sub(mask), p)
    e, f = c @ w - yy, a @ w - rr
    def mad(v, axis):
        center = np.median(v, axis=axis, keepdims=True)
        return np.maximum(1e-6, 1.4826*np.median(abs(v-center),axis=axis))
    scale = mad(e, 0) if variant == 'per_horizon' else mad(e, None)
    sr = mad(f, None)
    def fun(q):
        c,a = m._design(*sub(mask),np.exp(q[:2]))
        return np.r_[((c@q[2:]-yy)/scale).ravel(), (a@q[2:]-rr)/sr]
    q = least_squares(fun,np.r_[np.log(p),w],loss='cauchy',f_scale=1.5 if variant=='tight' else 2.5,x_scale='jac',max_nfev=100).x
    return np.exp(q[:2]),q[2:]
for variant in ('all','complete','per_horizon','tight'):
    eligible = np.ones(n,bool) if variant == 'all' else complete
    pooled = []
    for j in range(4):
        tr = eligible & (lab != j)
        te = complete & (lab == j)
        p,w=fit(tr,variant)
        c,a=m._design(*sub(te),p)
        e,f=abs(c@w-y[te]),abs(a@w-r[te])
        pooled.append([np.median(e),np.median(f),np.quantile(e,.9),np.quantile(f,.9),p[1]])
    p,w=fit(eligible,variant)
    print(variant,'worst lab (median curve/rate, p90 curve/rate)',np.max(np.array(pooled)[:,:4],axis=0).round(6),'fold slow taus',np.array(pooled)[:,4].round(5),'full taus',p.round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/fit_batch01_probe.py
sed -n '1,72p' /holdout_task/alpha_m97_c02_batch_01_solver.py
```
- **Container Exit Code**: `0` (`1520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
CHIVE.read_text())
    if isinstance(data, dict):
        data = data.get('calibration_load_cells', data.get('channels', data))
    if isinstance(data, dict) and 'lab_id' in data:
        ids = np.asarray(data['lab_id'])
        x = np.asarray(data['cmd_stress'])[ids == lab]
        y = np.asarray(data['true_stress'])[ids == lab]
        return _affine(x, y) if len(x) >= 2 and np.ptp(x) > 0 else None
    if isinstance(data, dict):
        row = data.get(str(lab))
        data = [row] if isinstance(row, dict) else row
    if isinstance(data, list):
        rows = [r for r in data if r.get('lab_id', lab) == lab]
        fits = [(r['gain'], r['bias']) for r in rows if 'gain' in r and 'bias' in r]
        if fits:
            return np.median(fits, axis=0)
        points = [(r['cmd_stress'], r['true_stress']) for r in rows
                  if 'cmd_stress' in r and 'true_stress' in r]
        if len(points) >= 2:
            x, y = np.asarray(points).T
            if np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(cal):
    out = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        take = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        spike = False
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            g = np.median(slopes)
            b = np.median(y - g*x)
            clean = np.abs(y - (g*x + b)) <= 3.
            spike = not clean.all()
        identifiable = clean.sum() >= 2 and np.ptp(x[clean]) > 0
        if identifiable:
            out[lab] = _affine(x[clean], y[clean])
        if len(x) < 2 or (spike and clean.sum() < 4):
            fallback = _historical_calibration(lab)
            if fallback is not None:
                out[lab] = fallback
            elif len(x) == 1:
                out[lab, 1] = y[0] - x[0]
    return out


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d2f3aef51c6eb408 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d2f3aef5 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`41299 ms` LLM)
- **Rationale**: Promote the complete-row fit based on leave-one-lab-out results, without changing the prescribed prediction-time fill. Recheck the public slices to detect a prediction regression, exercise the sparse and EMI calibration branches to catch unintended archive use or spike leakage, and compare the returned rate with the curve slope at zero to catch a mismatch between the two outputs.
- **Alternatives Weighed**: `Tighten the robust loss at the same time` (rejected: The tighter loss improves median errors but raises worst-lab p90 rate error and does not materially change the slow relaxation time.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
old = '''    taus, w = _fit(tuple(a[:n] for a in data[:5]) + data[5:],
                   train['strain_curves_obs'] - train['strain_0'][:, None],
                   train['instant_rate_obs'])'''
new = '''    # Estimate material parameters from observed features; imputation is for prediction.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    taus, w = _fit(tuple(a[:n][complete] for a in data[:5]) + data[5:],
                   (train['strain_curves_obs'] - train['strain_0'][:, None])[complete],
                   train['instant_rate_obs'][complete])'''
assert old in s
p.write_text(s.replace(old, new))
p = Path('/tmp/check_batch01.py')
s = p.read_text().replace("    keep = train['lab_id'] != lab", "    held_out = train['lab_id'] == lab\n    keep = ~held_out & np.isfinite(train['coupon_features_raw']).all(axis=1)").replace('sub(~keep)', 'sub(held_out)').replace('[~keep]', '[held_out]')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/check_batch01.py
python3 -I - <<'PY'
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
cal = d['calibration_load_cells']
baseline = m._calibrate(cal)
for lab in range(4):
    ids = np.flatnonzero(cal['lab_id'] == lab)
    for k in (0,1):
        keep = cal['lab_id'] != lab
        keep[ids[:k]] = True
        trial = m._calibrate({name: arr[keep] for name, arr in cal.items()})
        np.testing.assert_allclose(trial[lab], m._historical_calibration(lab))
        np.testing.assert_allclose(trial[np.arange(4) != lab], baseline[np.arange(4) != lab])
    for spike in (-8., 8.):
        for i in ids:
            trial = {name: arr.copy() for name, arr in cal.items()}
            trial['true_stress'][i] += spike
            np.testing.assert_allclose(m._calibrate(trial), baseline, atol=3e-4, rtol=0)
print('Calibration: 8 sparse-channel cases and 80 injected EMI cases passed')
# Compare the analytic rate with the forward curve slope in both modes.
out = m.solve_alpha_m97_c02_batch_01(d)
w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
past, hold, x, log_mode, act, dt, _ = data
for use_log in (False, True):
    curve, rate = m._design(past, hold, x, np.full_like(log_mode, use_log), act * use_log,
                            dt, np.array([0., 1e-7]), out['estimated_taus'])
    np.testing.assert_allclose(curve[:, 0] @ w, 0., atol=1e-12)
    np.testing.assert_allclose(curve[:, 1] @ w / 1e-7, rate @ w, atol=2e-6, rtol=0)
print('Zero-time boundary and rate handoff passed for both time bases')
PY
```
- **Container Exit Code**: `1` (`1265 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus fit [0.500613 3.772188] ref [0.501 3.825] max abs error 0.0528118
hereditary_weights fit [ 0.677886  0.548377  0.139054  0.15696  -0.12738   0.117932 -0.088175] ref [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] max abs error 0.00304
regime_mode_weights fit [0.280683 0.241316] ref [0.28 0.24] max abs error 0.0013156
calibration max history, hold error 0.0002544414505800674 0.00013367516081164155
public clean holdout (slice, n, curve RMSE, rate RMSE)
all 72 0.040566 0.031007
lab0/plateau0 7 0.003399 0.003079
lab0/plateau1 10 0.035196 0.027872
lab1/plateau0 3 0.00353 0.004862
lab1/plateau1 13 0.089822 0.068229
lab2/plateau0 2 0.000856 0.00204
lab2/plateau1 14 0.001622 0.00224
lab3/plateau0 12 0.005035 0.004366
lab3/plateau1 11 0.00827 0.006686
stratum0 22 0.027515 0.021806
stratum1 24 0.064892 0.049228
stratum2 26 0.005433 0.004805
missing 3 0.197396 0.150503
complete 69 0.004799 0.004289
curve horizon RMSE [0.008131 0.018964 0.026977 0.03334  0.038617 0.043121 0.047051 0.050535
 0.053665 0.056505]
lab fold 0 taus [0.500285 3.756741]
lab fold 1 taus [0.500277 3.758662]
lab fold 2 taus [0.500935 3.77157 ]
lab fold 3 taus [0.501387 3.83629 ]
leave-one-lab-out measured (slice, n, curve RMSE, rate RMSE)
all 180 0.945709 0.044945
lab0/plateau0 11 0.870267 0.027707
lab0/plateau1 30 1.077515 0.062183
lab1/plateau0 13 2.253655 0.07187
lab1/plateau1 24 0.370805 0.025968
lab2/plateau0 21 0.274426 0.029044
lab2/plateau1 31 1.151695 0.022223
lab3/plateau0 15 0.14957 0.086754
lab3/plateau1 35 0.395027 0.016834
stratum0 53 0.278162 0.022477
stratum1 66 1.481601 0.068582
stratum2 61 0.443634 0.020806
missing 19 0.468414 0.127991
complete 161 0.986924 0.018035
curve horizon RMSE [0.143657 0.598005 0.207367 0.362785 0.489259 1.894015 0.362831 0.294188
 1.446262 1.501418]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=36538212af7e9266 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_36538212 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 1504, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/usr/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 742, in assert_array_compare
    val = comparison(x, y)
          ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 1499, in compare
    return np.core.numeric.isclose(x, y, rtol=rtol, atol=atol,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/core/numeric.py", line 2349, in isclose
    yfin = isfinite(y)
           ^^^^^^^^^^^
TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```

### Turn 12 — `execute_in_remote_docker` (`17216 ms` LLM)
- **Rationale**: Read the archive field names and nesting without dumping the sweeps. This will distinguish a parser mismatch from genuinely absent channel data and allow the fallback test to use an independent expected fit.
- **Alternatives Weighed**: `Skip the sparse-channel check because the public batch has ten calibration points per lab` (rejected: The sparse fallback is part of the requested interface, and the current lookup returned no calibration when exercised.)

```bash
python3 -I - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
a = json.loads(p.read_text())
def summary(x):
    if isinstance(x, dict): return {k: summary(v) for k,v in x.items()}
    if isinstance(x, list): return {'length':len(x), 'first':summary(x[0]) if x else None}
    return x
print(json.dumps(summary(a), indent=2))
print(Path('/tmp/hegel_continual_archive/archive_manifest.json').read_text())
PY
```
- **Container Exit Code**: `0` (`375 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{
  "length": 1,
  "first": {
    "episode_short_name": "alpha_m97_c02_s00",
    "lab_id": {
      "length": 40,
      "first": 0
    },
    "cmd_stress": {
      "length": 40,
      "first": 1.619455
    },
    "true_stress": {
      "length": 40,
      "first": 1.619593
    }
  }
}
[
  {
    "episode_short_name": "alpha_m97_c02_s00",
    "task_id": "alpha_m97_c02_s00__rheology_campaign_baseline",
    "entrypoint": "solve_alpha_m97_c02_s00",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain",
      "regime_mode_weights"
    ],
    "input_coordinate_contract": "raw_measured",
    "solver_path": "/tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py"
  }
]
```

### Turn 13 — `execute_in_remote_docker` (`35915 ms` LLM)
- **Rationale**: The archive contains a list of sweeps with parallel arrays, not scalar calibration rows. Fit the requested channel from the latest usable sweep and check the result against an independent affine fit. Run the previously blocked EMI and zero-time checks and confirm that the archive fix leaves ordinary predictions unchanged.

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
a = s.index('def _historical_calibration(')
b = s.index('def _calibrate(', a)
s = s[:a] + '''def _historical_calibration(lab):
    if not _ARCHIVE.is_file():
        return None
    # Prefer the most recent prefix sweep with an identifiable channel fit.
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        take = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[take]
        y = np.asarray(sweep['true_stress'])[take]
        if len(x) >= 2 and np.ptp(x) > 0:
            return _affine(x, y)
    return None


''' + s[b:]
s = s.replace('# Downweight large residuals, including rows whose missing features were imputed.', '# Downweight transient observation spikes without discarding the rest of a curve.')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import json
exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
cal = d['calibration_load_cells']
baseline = m._calibrate(cal)
sweep = json.loads(m._ARCHIVE.read_text())[-1]
for lab in range(4):
    ids = np.flatnonzero(cal['lab_id'] == lab)
    hist = np.asarray(sweep['lab_id']) == lab
    expected = np.polyfit(np.asarray(sweep['cmd_stress'])[hist], np.asarray(sweep['true_stress'])[hist], 1)
    for k in (0, 1):
        keep = cal['lab_id'] != lab
        keep[ids[:k]] = True
        trial = m._calibrate({name: arr[keep] for name, arr in cal.items()})
        np.testing.assert_allclose(trial[lab], expected, atol=1e-12)
        np.testing.assert_allclose(trial[np.arange(4) != lab], baseline[np.arange(4) != lab])
    for spike in (-8., 8.):
        for i in ids:
            trial = {name: arr.copy() for name, arr in cal.items()}
            trial['true_stress'][i] += spike
            np.testing.assert_allclose(m._calibrate(trial), baseline, atol=3e-4, rtol=0)
print('Passed: 8 sparse-channel and 80 injected EMI cases')
out = m.solve_alpha_m97_c02_batch_01(d)
w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
past, hold, x, log_mode, act, dt, _ = data
for use_log in (False, True):
    curve, rate = m._design(past, hold, x, np.full_like(log_mode, use_log), act * use_log,
                            dt, np.array([0., 1e-7]), out['estimated_taus'])
    np.testing.assert_allclose(curve[:, 0] @ w, 0., atol=1e-12)
    np.testing.assert_allclose(curve[:, 1] @ w / 1e-7, rate @ w, atol=2e-6, rtol=0)
print('Passed: zero-time strain and rate consistency in both flow modes')
for name, key in [('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    err = out[name] - gt[key]
    complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
    print(name, 'RMSE complete / missing', *(float(np.sqrt(np.mean(err[mask] ** 2))) for mask in (complete, ~complete)))
PY
```
- **Container Exit Code**: `0` (`986 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Passed: 8 sparse-channel and 80 injected EMI cases
Passed: zero-time strain and rate consistency in both flow modes
predicted_strain RMSE complete / missing 0.0047986920286867815 0.19739622804211923
initial_rate_pred RMSE complete / missing 0.00428886221450122 0.15050348327684143

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c1b4c2800b3267bf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_c1b4c280 (directory).
```

### Turn 14 — `submit_completed_task` (`14849 ms` LLM)
- **Rationale**: The final artifact passed isolated verification, including eight sparse-channel cases and 80 injected EMI cases. The tested alternative fills did not justify changing the prescribed cell-balanced imputation.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema, sparse and EMI-contaminated calibration checks, and zero-time strain–rate consistency checks passed. Public holdout RMSE was 0.00480 for strain and 0.00429 for initial rate on complete rows; on the three rows with missing features, it was 0.1974 and 0.1505, respectively.`

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
