# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_02__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_02` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `18 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `517,759` (`500,155` / `135,168` / `17,604`) |
| **Total Episode Wall Time (s)** | `845.44s` |

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

### Turn 1 — `execute_in_remote_docker` (`12757 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver interfaces to identify required fields and reusable alignment routines before implementing the fit.

```bash
ls -l /tmp/hegel_continual_archive /holdout_task; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; for f in /tmp/hegel_continual_archive/*solver.py; do echo "$f"; grep -nE '^def |^from |^import ' "$f"; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_index_v1"
}/tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py
2:import json
4:import numpy as np
5:from scipy.optimize import least_squares
10:def _align(train, test):
39:def _calibrate(cal):
71:def _design(split, x, gain, bias, dt, t, crit, banding):
104:def _mad(r):
108:def _fit(evaluate, curve_y, rate_y, dt, t):
147:def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
/tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py
2:import json
4:import numpy as np
5:from scipy.optimize import least_squares
11:def _archive_pairs():
31:def _calibrate(cal):
61:def _align(train, test):
90:def _design(split, x, gain, bias, dt, t):
122:def _fit(evaluate, curve_y, rate_y, dt, t):
178:def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`19312 ms` LLM)
- **Rationale**: Inspect the measured fields and prior preprocessing code to catch panel-layout or calibration-fallback mismatches before fitting. Print compact summaries without transferring datasets.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The current batch specifies linear terminal flow and zero regime weights; reuse should be limited to compatible calibration and alignment routines.)

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_02.pkl','rb'))
d=items[0]['dataset']
for k,v in d.items():
 print(k)
 for a,b in (v.items() if isinstance(v,dict) else [(k,v)]):
  x=np.asarray(b); print(' ',a,x.shape,x.dtype, ('range',np.nanmin(x),np.nanmax(x),'nonfinite',np.sum(~np.isfinite(x))) if x.dtype.kind in 'iuf' and x.size else '')
 if k in ('train','test','calibration_load_cells','ref_coupons') and isinstance(v,dict) and 'lab_id' in v: print(' lab counts:',np.unique(v['lab_id'],return_counts=True))
print('time grid',d.get('eval_times'))
PY
```
- **Container Exit Code**: `0` (`433 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.820516,
      -1.502377,
      0.013775,
      0.72799
    ]
  }
]past_dt
  past_dt (18,) float64 ('range', 0.283, 0.7877, 'nonfinite', 0)
eval_times
  eval_times (10,) float64 ('range', 0.3, 5.2, 'nonfinite', 0)
crit_stress
  crit_stress () float64 ('range', 2.65, 2.65, 'nonfinite', 0)
yield_stress
  yield_stress () float64 ('range', 2.65, 2.65, 'nonfinite', 0)
shear_banding_index
  shear_banding_index () float64 ('range', 0.12, 0.12, 'nonfinite', 0)
calibration_load_cells
  lab_id (40,) int64 ('range', 0, 3, 'nonfinite', 0)
  cmd_stress (40,) float64 ('range', -1.896441, 2.168228, 'nonfinite', 0)
  true_stress (40,) float64 ('range', -2.135178884187165, 2.2477476065188697, 'nonfinite', 0)
 lab counts: (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
train
  lab_id (180,) int64 ('range', 0, 3, 'nonfinite', 0)
  stratum (180,) int64 ('range', 0, 2, 'nonfinite', 0)
  coupon_features_raw (180, 4) float64 ('range', -2.9801484684657047, 4.000723734955689, 'nonfinite', 12)
  ref_coupons (180, 32) float64 ('range', -3.82549249486226, 6.038532829734818, 'nonfinite', 235)
  past_stress_cmd (180, 18) float64 ('range', -2.575, 2.8644736842105263, 'nonfinite', 0)
  stress_hold_cmd (180,) float64 ('range', -2.3888763157894735, 2.7213302631578946, 'nonfinite', 0)
  strain_0 (180,) float64 ('range', -0.856078, 1.107412, 'nonfinite', 0)
  aux_chamber_telemetry (180, 3) float64 ('range', -3.410697, 3.133747, 'nonfinite', 0)
  strain_curves_obs (180, 10) float64 ('range', -321.29532909766385, 118.55168403566387, 'nonfinite', 0)
  instant_rate_obs (180,) float64 ('range', -4.417439220338145, 4.022174511833956, 'nonfinite', 0)
 lab counts: (array([0, 1, 2, 3]), array([49, 38, 50, 43]))
test
  lab_id (72,) int64 ('range', 0, 3, 'nonfinite', 0)
  stratum (72,) int64 ('range', 0, 2, 'nonfinite', 0)
  coupon_features_raw (72, 4) float64 ('range', -3.2451786679964822, 4.362423893699004, 'nonfinite', 12)
  ref_coupons (72, 32) float64 ('range', -3.935204289907075, 4.614170280154203, 'nonfinite', 103)
  past_stress_cmd (72, 18) float64 ('range', -2.575, 2.8644736842105263, 'nonfinite', 0)
  stress_hold_cmd (72,) float64 ('range', -2.4729539473684206, 2.6272105263157894, 'nonfinite', 0)
  strain_0 (72,) float64 ('range', -0.601437, 0.767692, 'nonfinite', 0)
  aux_chamber_telemetry (72, 3) float64 ('range', -2.934733, 2.350602, 'nonfinite', 0)
 lab counts: (array([0, 1, 2, 3]), array([17, 15, 22, 18]))
time grid [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
```

### Turn 3 — `execute_in_remote_docker` (`21283 ms` LLM)
- **Rationale**: Check whether extreme readings are isolated or trajectory-wide, which determines how to weight the fit, and confirm that the public load-cell sweeps do not require archive fallback.
- **Alternatives Weighed**: `Fit ordinary least squares immediately` (rejected: Public measured strains reach magnitudes above 300, so the contamination pattern should determine whether robustness is needed at the reading or trajectory level.)

```bash
sed -n '39,71p;104,167p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS Cauchy viscoelastic' | head -15; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_02.pkl','rb'))[0];d=p['dataset'];tr=d['train'];y=tr['strain_curves_obs']-tr['strain_0'][:,None];t=d['eval_times'];print('public item keys',list(p));print('abs curve change quantiles per horizon',np.quantile(abs(y),[.5,.9,.95,.98,.99,1],axis=0).round(3));print('large residual candidates',[(int(i),int(j),round(y[i,j],2)) for i,j in np.argwhere(abs(y)>15)]);print('rate quantiles',np.quantile(tr['instant_rate_obs'],[0,.05,.5,.95,1])); c=d['calibration_load_cells'];
for lab in range(4):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; z=c['true_stress'][m]; a,b=np.polyfit(x,z,1); print('cal',lab,a,b,np.max(abs(z-a*x-b)))
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
public item keys ['dataset', 'gt']
abs curve change quantiles per horizon [[  0.329   0.663   0.787   0.885   0.92    1.025   1.083   1.178   1.17
    1.327]
 [  0.763   1.356   1.69    1.957   2.174   2.315   2.452   2.595   2.75
    3.176]
 [  0.966   1.607   1.997   2.636   2.773   2.707   3.139   3.297   3.704
    4.238]
 [  1.628   1.903   2.439   2.97    3.326   3.451   3.91    4.028   4.45
    5.082]
 [  9.282   2.464   2.658   3.321   3.9     3.697   4.257   4.352   5.048
    6.419]
 [ 13.817  25.953 118.431  11.467   7.97    4.374   5.62    5.405   9.379
  321.037]]
large residual candidates [(35, 9, -321.04), (117, 2, 118.43), (148, 1, 25.95)]
rate quantiles [-4.41743922 -2.48857115  0.07623382  3.1047268   4.02217451]
cal 0 1.289963286421332 0.119983233928751 0.00017066189373383944
cal 1 0.8200333935955562 -0.15000569371240166 0.00013417502581250518
cal 2 1.1699864656264194 0.1399770396145451 0.00017242595550148732
cal 3 0.7599664980599373 -0.11003498033048446 0.00016507313927653644
```

### Turn 4 — `execute_in_remote_docker` (`51403 ms` LLM)
- **Rationale**: Implement the specified memory and flow operators. The schema check catches interface failures, and public holdout errors will show whether the calibrated fit loses accuracy in any lab or among coupons with missing features.
- **Alternatives Weighed**: `Remove entire trajectories with extreme strain readings` (rejected: The extreme readings are isolated by horizon; observation-level weighting retains the remaining measurements.)

```bash
set -e
p=/holdout_task/alpha_m97_c03_batch_02_solver.py
{ printf '%s\n' '"""Calibrated two-branch hereditary response with linear terminal flow."""'; sed -n '2,70p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py; } > "$p"
cat >> "$p" <<'PY'

def _design(split, aligned, gain, bias, dt, t):
    lab = split['lab_id']
    past = gain[lab, None] * split['past_stress_cmd'] + bias[lab, None]
    hold = gain[lab] * split['stress_hold_cmd'] + bias[lab]
    features = np.column_stack((np.ones(len(lab)), aligned))
    remaining = np.cumsum(dt[::-1])[::-1] - dt

    def evaluate(log_tau):
        taus = np.exp(log_tau)
        kernel = np.exp(-remaining[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
        gaps = hold[:, None] - past @ kernel
        curve = np.concatenate((
            gaps[:, None, :] * -np.expm1(-t[:, None] / taus),
            t[None, :, None] * features[:, None, :],
        ), axis=2)
        return curve, np.column_stack((gaps / taus, features))

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    groups = np.r_[np.broadcast_to(np.arange(len(t)), cm.shape)[cm], np.full(rm.sum(), len(t))]
    masks = [groups == j for j in np.unique(groups)]
    lo = np.log(min(dt.min(), t[t > 0].min()) / 20)
    hi = np.log(20 * max(dt.sum(), t.max()))

    def matrix(log_tau):
        curve, rate = evaluate(log_tau)
        return np.vstack((curve[cm], rate[rm]))

    def project(m, weights):
        return np.linalg.lstsq(m * weights[:, None], y * weights, rcond=None)[0]

    def scales(residual):
        result = np.empty_like(residual)
        for mask in masks:
            r = residual[mask]
            result[mask] = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        return result

    def profile(log_tau, weights):
        m = matrix(log_tau)
        return (m @ project(m, weights) - y) * weights

    rough = np.clip(np.log([0.45, 3.5]), lo, hi)
    base = matrix(rough)
    weights = np.ones_like(y)
    for _ in range(7):
        residual = base @ project(base, weights) - y
        scale = scales(residual)
        weights = 1 / (scale * (1 + (residual / (3 * scale))**2)**0.25)
    starts = [rough, np.clip(np.log([0.18, 1.6]), lo, hi), np.clip(np.log([1.2, 8.0]), lo, hi)]
    fits = [least_squares(profile, s, args=(weights,), bounds=(lo, hi), max_nfev=60) for s in starts]
    tau = min(fits, key=lambda fit: fit.cost).x
    for _ in range(9):
        m = matrix(tau)
        residual = m @ project(m, weights) - y
        scale = scales(residual)
        weights = 1 / np.hypot(scale, residual / 2.5)
        tau = least_squares(profile, tau, args=(weights,), bounds=(lo, hi), max_nfev=40).x
    beta = project(matrix(tau), weights)
    order = np.argsort(tau)
    return tau[order], np.r_[beta[:2][order], beta[2:]]


def solve_alpha_m97_c03_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    train_design = _design(train, xtr, gain, bias, dt, t)
    # Fit on measured features; use aligned cell means when predicting missing ones.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    curve_y = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate_y = train['instant_rate_obs']
    log_tau, beta = _fit(train_design, np.where(complete[:, None], curve_y, np.nan),
                         np.where(complete, rate_y, np.nan), dt, t)
    curve, rate = _design(test, xte, gain, bias, dt, t)(log_tau)
    return dict(estimated_taus=np.exp(log_tau), hereditary_weights=beta,
                regime_mode_weights=np.zeros(2),
                predicted_strain=test['strain_0'][:, None] + curve @ beta,
                initial_rate_pred=rate @ beta)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',base+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open(base+'public_alpha_m97_c03_batch_02.pkl','rb'))[0];d=p['dataset'];g=p['gt'];o=m.solve_alpha_m97_c03_batch_02(d); print('truth schema',[(k,np.shape(v)) for k,v in g.items()]);print('tau',o['estimated_taus'],'weights',o['hereditary_weights']);
for k in ['taus','theta','vp_theta']:
 if k in g:print(k,g[k])
for key,gkey in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 if gkey not in g: continue
 e=o[key]-g[gkey];rows=[d['test']['lab_id']==j for j in range(4)];rows+=[~np.isfinite(d['test']['coupon_features_raw']).all(axis=1)];print(key,'rmse',np.sqrt(np.mean(e**2)),'lab and missing RMSE',[round(np.sqrt(np.mean(e[r]**2)),7) for r in rows])
PY
```
- **Container Exit Code**: `0` (`2055 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
truth schema [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
tau [0.54021981 3.90220058] weights [ 0.67921618  0.54935723  0.13936578  0.15987654 -0.12933048  0.11950825
 -0.09002158]
taus [0.54 3.9 ]
theta [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
vp_theta [0. 0.]
predicted_strain rmse 0.17297392712376225 lab and missing RMSE [0.2363548, 0.0811551, 0.2062971, 0.0970332, 0.4874157]
initial_rate_pred rmse 0.0547944107072266 lab and missing RMSE [0.0755474, 0.0257223, 0.0646499, 0.0309775, 0.1542115]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32474 ms` LLM)
- **Rationale**: Compare the solver against the same physical equations evaluated with public truth parameters, and verify the memory kernel against direct recurrence. Leaving each lab's labels out tests whether the shared coefficients transfer across labs.
- **Alternatives Weighed**: `Change the imputer immediately` (rejected: First determine whether the same error remains with public truth parameters and the required alignment; a large missing-feature error alone does not identify a defect.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',base+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open(base+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; xtr,xte=m._align(tr,te); gain,bias=m._calibrate(d['calibration_load_cells']); o=m.solve_alpha_m97_c03_batch_02(d)
def rmse(e): return np.sqrt(np.mean(e**2))
def report(label, curve, rate):
    ec=curve-g['te_clean']; er=rate-g['te_true_rate0']; missing=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    print(label, 'curve/rate RMSE complete', rmse(ec[~missing]),rmse(er[~missing]), 'missing',rmse(ec[missing]),rmse(er[missing]), 'worst horizon',max(map(rmse,ec.T)))
tau=g['taus']; beta=g['theta']; z=np.zeros((len(te['lab_id']),2))
for s,ds in enumerate(dt):
    a=np.exp(-ds/tau); z=a*z+(1-a)*g['te_hist_true'][:,s,None]
gaps=g['te_hold_true'][:,None]-z; drift=beta[2]+xte @ beta[3:]; c=te['strain_0'][:,None]+(gaps*beta[:2]) @ (1-np.exp(-t[None,:]/tau[:,None]))+drift[:,None]*t; r=(gaps/tau) @ beta[:2]+drift
report('solver',o['predicted_strain'],o['initial_rate_pred']); report('truth parameters + required alignment',c,r)
cv,rv=m._design(te,xte,g['stress_gains'],g['stress_biases'],dt,t)(np.log(tau)); np.testing.assert_allclose(te['strain_0'][:,None]+cv @ beta,c,atol=5e-5,rtol=0); np.testing.assert_allclose(rv @ beta,r,atol=5e-5,rtol=0)
# Train without one lab's labeled measurements at a time; keep the specified reference panel.
train_eval=m._design(tr,xtr,gain,bias,dt,t); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for lab in range(4):
    fit=complete & (tr['lab_id']!=lab); test=te['lab_id']==lab; lt,b=m._fit(train_eval,np.where(fit[:,None],y,np.nan),np.where(fit,tr['instant_rate_obs'],np.nan),dt,t)
    cv,rv=m._design(te,xte,gain,bias,dt,t)(lt); print('excluded lab',lab,'curve/rate RMSE',rmse((te['strain_0'][:,None]+cv @ b-g['te_clean'])[test]),rmse((rv @ b-g['te_true_rate0'])[test]),'max rel tau error',max(abs(np.exp(lt)/tau-1)))
assert np.array_equal(o['regime_mode_weights'],np.zeros(2))
PY
```
- **Container Exit Code**: `0` (`793 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
solver curve/rate RMSE complete 0.01597204593255129 0.005833783176565937 missing 0.4874157280577148 0.15421150532615763 worst horizon 0.2843032885897982
truth parameters + required alignment curve/rate RMSE complete 0.016583875792727645 0.005242185732632464 missing 0.488334469708531 0.15436319120173847 worst horizon 0.2849365040030683
excluded lab 0 curve/rate RMSE 0.23659444589494977 0.07588796824637813 max rel tau error 0.002125259063140983
excluded lab 1 curve/rate RMSE 0.08190841979928133 0.025947546664608308 max rel tau error 0.014959151180067076
excluded lab 2 curve/rate RMSE 0.20613713221082655 0.06463942174295126 max rel tau error 0.014755702505995782
excluded lab 3 curve/rate RMSE 0.10047056229023141 0.03179284941500424 max rel tau error 0.007058028731501187

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27097 ms` LLM)
- **Rationale**: The public schema, memory recurrence, and held-out lab checks passed. The public calibration rows did not exercise sparse-channel fallback or EMI rejection; test both on measured sweeps and revise calibration only if a spike moves the fit or a sparse channel fails to recover its archived line.
- **Alternatives Weighed**: `Change the imputation rule` (rejected: Public truth parameters yield essentially the same missing-feature errors with the required cell means; the error does not establish an implementation defect.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',base+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open(base+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d=p['dataset']; cal=d['calibration_load_cells']; lab=cal['lab_id']; baseline=np.array(m._calibrate(cal))
with open(m._ARCHIVE) as f: history=json.load(f)
worst_emi=worst_sparse=0.
for c in range(4):
    rows=np.flatnonzero(lab==c)
    for row in rows:
        for spike in [-8.,8.]:
            altered={k:v.copy() for k,v in cal.items()}; altered['true_stress'][row]+=spike
            delta=np.max(np.abs(np.array(m._calibrate(altered))-baseline)); worst_emi=max(worst_emi,delta)
        keep=(lab!=c); keep[row]=True
        sparse={k:v[keep] for k,v in cal.items()}; actual=np.array(m._calibrate(sparse)); sweep=next(s for s in reversed(history) if c in s['lab_id']); use=np.asarray(sweep['lab_id'])==c
        expected=np.polyfit(np.asarray(sweep['cmd_stress'])[use],np.asarray(sweep['true_stress'])[use],1)
        worst_sparse=max(worst_sparse,np.max(np.abs(actual[:,c]-expected)))
        np.testing.assert_allclose(actual[:,lab[:0]], baseline[:,lab[:0]]) if False else None
        other=np.arange(4)!=c; np.testing.assert_allclose(actual[:,other],baseline[:,other],rtol=0,atol=1e-12)
assert worst_emi < 1e-3 and worst_sparse < 1e-12, (worst_emi,worst_sparse)
print('worst gain/bias change under one signed EMI spike:', worst_emi, 'max sparse fallback discrepancy:',worst_sparse)
PY
```
- **Container Exit Code**: `0` (`632 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
worst gain/bias change under one signed EMI spike: 2.237812175696907e-05 max sparse fallback discrepancy: 4.85722573273506e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 7 — `submit_completed_task` (`13687 ms` LLM)
- **Rationale**: The verified artifact passes the schema and stress checks. The reference-parameter comparison shows that refitting the hereditary coefficients would not materially improve the worst slice under the specified imputation rule.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_02_solver.py': 'FILE:6726'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_02_solver.py`. The public schema, direct memory recurrence, leave-one-lab-out, and sparse and EMI calibration checks passed. The worst public slice was coupons with missing features: strain RMSE 0.4874 and initial-rate RMSE 0.1542, essentially unchanged when using the reference parameters with the required imputation.`

### Turn 8 — `execute_in_remote_docker` (`28181 ms` LLM)
- **Rationale**: Isolate complete-feature lab slices to detect systematic signed bias that missing-feature error could mask. Large lab bias would warrant revisiting alignment; derivative mismatch would warrant correcting the rate operator.
- **Alternatives Weighed**: `Add nonlinear terms, relay states, or classification diagnostics` (rejected: The specified regime is continuous linear terminal flow with zero regime weights. Held-out signed residuals test lab confounding directly without conflating it with imputation error.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',root+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(root+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; xtr,xte=m._align(tr,te); gain,bias=m._calibrate(d['calibration_load_cells']); print('split fields:',sorted(tr),sorted(te)); print('max gain/bias error:',max(abs(gain-g['stress_gains'])),max(abs(bias-g['stress_biases'])))
a=m._design(tr,xtr,gain,bias,dt,t); b=m._design(te,xte,gain,bias,dt,t); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); observed=np.isfinite(te['coupon_features_raw']).all(axis=1)
def rmse(x): return np.sqrt(np.mean(x*x))
for lab in np.unique(te['lab_id']):
    train=complete & (tr['lab_id']!=lab); keep=observed & (te['lab_id']==lab)
    lt,w=m._fit(a,np.where(train[:,None],y,np.nan),np.where(train,tr['instant_rate_obs'],np.nan),dt,t); c,r=b(lt)
    ec=(te['strain_0'][:,None]+c@w-g['te_clean'])[keep]; er=(r@w-g['te_true_rate0'])[keep]
    print('held-out lab',lab,'complete n',keep.sum(),'strain/rate RMSE',rmse(ec),rmse(er),'signed strain/rate bias',ec.mean(),er.mean(),'max horizon |bias|',max(abs(ec.mean(0))))
o=m.solve_alpha_m97_c03_batch_02(d); eps=1e-6; c,r=m._design(te,xte,gain,bias,dt,np.array([eps]))(np.log(o['estimated_taus'])); finite_diff=c[:,0,:]@o['hereditary_weights']/eps; delta=max(abs(finite_diff-o['initial_rate_pred'])); print('max derivative closure error',delta); assert delta < 1e-5; assert np.array_equal(o['regime_mode_weights'],np.zeros(2))
PY
```
- **Container Exit Code**: `0` (`1082 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
split fields: ['aux_chamber_telemetry', 'coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd'] ['aux_chamber_telemetry', 'coupon_features_raw', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'stratum', 'stress_hold_cmd']
max gain/bias error: 3.6713578666969227e-05 3.498033048483784e-05
held-out lab 0 complete n 15 strain/rate RMSE 0.015373346070570088 0.0073769449183002395 signed strain/rate bias -0.006668936866407994 -0.004602882059065627 max horizon |bias| 0.011756747251519034
held-out lab 1 complete n 14 strain/rate RMSE 0.01899546880285763 0.0065424853671061335 signed strain/rate bias 0.0028928722908898685 0.0014929125612227559 max horizon |bias| 0.0049325509150844405
held-out lab 2 complete n 19 strain/rate RMSE 0.020274839763501976 0.007659899824708607 signed strain/rate bias 0.00791838635034798 0.0021127891798754417 max horizon |bias| 0.015261161688507493
held-out lab 3 complete n 15 strain/rate RMSE 0.03416316462587579 0.009724970035150618 signed strain/rate bias 0.008379795484169615 0.004101565200925353 max horizon |bias| 0.015965463758220538
max derivative closure error 3.6723641185787415e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 9 — `submit_completed_task` (`15351 ms` LLM)
- **Rationale**: The final artifact passed isolated execution and the targeted stress tests. The reference-parameter comparison identifies the required imputation, not the fitted physical coefficients, as the main limitation on the public worst slice.
- **Alternatives Weighed**: `Add lab-specific drift or nonlinear flow terms` (rejected: Held-out complete-feature lab errors were small, and reference parameters did not improve the dominant missing-feature slice under the specified imputation.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_02_solver.py': 'FILE:6726'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_02_solver.py`. The public schema, memory recurrence, initial-rate derivative, and sparse and EMI calibration checks passed. In leave-one-lab-out tests, the worst complete-feature lab had strain RMSE 0.0342 and initial-rate RMSE 0.00972. Coupons with missing features had RMSE 0.4874 and 0.1542, respectively, essentially unchanged when using the reference parameters with the required imputation.`

### Turn 10 — `execute_in_remote_docker` (`72464 ms` LLM)
- **Rationale**: Leave early, middle, and late horizons out of the fit in turn, then score signed errors separately by lab on complete-feature public coupons. A systematic time-by-lab error would warrant auditing the alignment or memory terms; without it, adding nonlinear terms would be unsupported.
- **Alternatives Weighed**: `Fit rational drift, relay modes, or a classifier` (rejected: The specified outputs are continuous, with linear terminal flow and zero regime weights. A held-out time point tests the relevant algebraic failure: systematic curvature missed by the prescribed model.)

```bash
set -e
sed -n '1,72p' /holdout_task/alpha_m97_c03_batch_02_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',root+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(root+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; xtr,xte=m._align(tr,te); gain,bias=m._calibrate(d['calibration_load_cells']); a=m._design(tr,xtr,gain,bias,dt,t); b=m._design(te,xte,gain,bias,dt,t)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); complete_te=np.isfinite(te['coupon_features_raw']).all(axis=1); errs=[]
for horizon in (0,len(t)//2,len(t)-1):
    fit_y=np.where(complete[:,None],y,np.nan); fit_y[:,horizon]=np.nan
    lt,w=m._fit(a,fit_y,np.where(complete,tr['instant_rate_obs'],np.nan),dt,t); cv,_=b(lt)
    e=te['strain_0']+cv[:,horizon]@w-g['te_clean'][:,horizon]
    lab_stats=[(float(np.sqrt(np.mean(e[q]**2))),float(e[q].mean())) for lab in range(4) if (q:=complete_te & (te['lab_id']==lab)).any()]; errs.extend(lab_stats)
    print('withheld horizon',horizon,'complete-feature lab RMSE/signed error',np.round(lab_stats,6).tolist())
print('worst held-out horizon/lab RMSE',max(r for r,_ in errs),'worst absolute signed error',max(abs(b) for _,b in errs))
PY
```
- **Container Exit Code**: `0` (`826 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    gain, bias = np.ones(4), np.zeros(4)
    history = None
    for lab in range(4):
        x, y = (np.asarray(cal[key])[cal['lab_id'] == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                r = y - slope*x
                clean = np.abs(r - np.median(r)) <= 3.0
        if len(x) < 2 or not clean.all():
            if history is None:
                with open(_ARCHIVE) as f:
                    history = json.load(f)
            if clean.sum() < 2:
                sweep = next(row for row in reversed(history) if lab in row['lab_id'])
                ref_lab, ref_x, ref_y = (np.asarray(sweep[key])
                                         for key in ('lab_id', 'cmd_stress', 'true_stress'))
                x, y = ref_x[ref_lab == lab], ref_y[ref_lab == lab]
                clean = np.ones(len(x), dtype=bool)
        gain[lab], bias[lab] = np.linalg.lstsq(
            np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
    return gain, bias



def _design(split, aligned, gain, bias, dt, t):
withheld horizon 0 complete-feature lab RMSE/signed error [[0.001324, -0.000756], [0.001628, 0.000378], [0.001366, 0.00045], [0.002097, 0.000989]]
withheld horizon 5 complete-feature lab RMSE/signed error [[0.010061, -0.004333], [0.014765, 0.002875], [0.011775, 0.005312], [0.022536, 0.008238]]
withheld horizon 9 complete-feature lab RMSE/signed error [[0.016733, -0.00743], [0.025293, 0.004247], [0.020082, 0.008775], [0.038536, 0.013131]]
worst held-out horizon/lab RMSE 0.03853640630398093 worst absolute signed error 0.013130983779949904

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`28412 ms` LLM)
- **Rationale**: Prior tests exercised sparse calibration and EMI separately. A spike can dominate the median pairwise slope when only three or four readings remain. Test that interaction on subsets of the measured sweeps; if it materially changes gain or bias, revise spike detection without changing clean in-batch fits.
- **Alternatives Weighed**: `Repeat the flow-model audit` (rejected: Direct recurrence, derivative closure, and held-out horizons already test the specified flow model.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle, numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',root+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(root+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]['dataset']; cal=d['calibration_load_cells']; labs=cal['lab_id']; baseline=np.array(m._calibrate(cal));
for n in (3,4):
    worst=(0,None); clean_worst=0; failures=0; total=0
    for lab in range(4):
        for rows in itertools.combinations(np.flatnonzero(labs==lab),n):
            if np.ptp(cal['cmd_stress'][list(rows)])<0.5: continue
            keep=(labs!=lab); keep[list(rows)]=True
            small={k: v[keep].copy() for k,v in cal.items()}; clean=np.array(m._calibrate(small)); clean_worst=max(clean_worst,float(max(abs(clean[:,lab]-baseline[:,lab]))))
            for i in np.flatnonzero(small['lab_id']==lab):
                for spike in (-4.,4.,-8.,8.):
                    case={**small,'true_stress':small['true_stress'].copy()}; case['true_stress'][i]+=spike
                    actual=np.array(m._calibrate(case)); err=float(max(abs(actual[:,lab]-baseline[:,lab]))); total+=1; failures+=err>0.05
                    if err>worst[0]: worst=(err,(lab,small['cmd_stress'][small['lab_id']==lab].tolist(),float(small['cmd_stress'][i]),spike,actual[:,lab].tolist()))
    print(n,'rows; clean worst gain/bias delta',clean_worst,'spiked cases above 0.05',failures,'of',total,'worst',worst)
PY
```
- **Container Exit Code**: `0` (`9891 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3 rows; clean worst gain/bias delta 0.0004774187243475059 spiked cases above 0.05 3768 of 5652 worst (33.592913482124416, (3, [1.70414, 1.601029, 2.101495], 1.601029, -8.0, [16.74525247936645, -33.7029484624549]))
4 rows; clean worst gain/bias delta 0.00039026397994386397 spiked cases above 0.05 3384 of 13440 worst (8.87478085304387, (0, [1.649489, 1.648912, 1.320542, 1.005033], 1.005033, 4.0, [-4.310899476416018, 8.994764086972621]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`49181 ms` LLM)
- **Rationale**: Test whether deleted-reading residuals detect spikes without flagging clean subsets. Compare archived and measured lines to determine whether the archive can identify the contaminated reading directly or should only break ties between plausible in-batch fits.
- **Alternatives Weighed**: `Use archived calibration for every short sweep` (rejected: Clean short sweeps fit accurately in-batch, and the archive is only permitted for fewer than two readings or suspected EMI.)
- **ATMS NOGOOD DECLARED**: Refuted `The median pairwise slope reliably separates transient EMI from the load-cell line on three- and four-reading sweeps.` | Witness: `On measured three- and four-row subsets with one injected EMI spike, 3,768 of 5,652 and 3,384 of 13,440 cases exceeded 0.05 gain/bias error; the worst error was 33.59.` | Blanket Diagnosis: `A single high-leverage reading can tilt the screening line toward itself, hiding the spike and causing the fitted calibration to distort all stresses in that channel.` | Sublated Pivot: `Keep direct fits for clean sweeps and the prescribed affine calibration law; use deleted-reading residuals to flag possible spikes before consulting the archive.`

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, itertools, numpy as np
root='/holdout_task/'
d=pickle.load(open(root+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; hist=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for lab in range(4):
    pick=c['lab_id']==lab; x,y=c['cmd_stress'][pick],c['true_stress'][pick]; a=np.column_stack((x,np.ones(len(x)))); base=np.linalg.lstsq(a,y,rcond=None)[0]
    sweep=next(r for r in reversed(hist) if lab in r['lab_id']); pm=np.asarray(sweep['lab_id'])==lab; ref=np.polyfit(np.asarray(sweep['cmd_stress'])[pm],np.asarray(sweep['true_stress'])[pm],1)
    print('lab',lab,'in-batch/archive lines',base.round(6),ref.round(6),'max archive discrepancy on clean sweep',max(abs(y-a@ref)))
    for n in (3,4):
        clean_alarm=miss=total=0; min_bad=np.inf
        for rows in itertools.combinations(range(len(x)),n):
            idx=list(rows); ax=a[idx]; obs=y[idx]
            if np.ptp(ax[:,0])<0.5: continue
            q,_=np.linalg.qr(ax); h=np.sum(q*q,axis=1); pr=np.eye(n)-q@q.T
            def decision(z):
                r=pr@z; loo=np.divide(abs(r),1-h,out=np.zeros(n),where=(1-h)>1e-12)
                return max(abs(r))>0.002 and max(loo)>3.0
            clean_alarm+=decision(obs)
            for j in range(n):
                for spike in (-8.,-4.,4.,8.):
                    z=obs.copy(); z[j]+=spike; total+=1; miss+=not decision(z); min_bad=min(min_bad,max(abs(pr@z)))
        print(' rows',n,'clean alarms',clean_alarm,'misses',miss,'of',total,'min contaminated full-fit max residual',min_bad)
PY
```
- **Container Exit Code**: `0` (`621 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 in-batch/archive lines [1.289963 0.119983] [1.289989 0.120002] max archive discrepancy on clean sweep 0.0002314036618931148
 rows 3 clean alarms 0 misses 21 of 1392 min contaminated full-fit max residual 0.0002622977184144064
 rows 4 clean alarms 0 misses 0 of 3360 min contaminated full-fit max residual 0.16650977108973125
lab 1 in-batch/archive lines [ 0.820033 -0.150006] [ 0.820039 -0.150026] max archive discrepancy on clean sweep 0.0001510362678877608
 rows 3 clean alarms 0 misses 28 of 1416 min contaminated full-fit max residual 0.00020555949396155367
 rows 4 clean alarms 0 misses 0 of 3360 min contaminated full-fit max residual 0.08599605852235614
lab 2 in-batch/archive lines [1.169986 0.139977] [1.169981 0.140009] max archive discrepancy on clean sweep 0.00021245500360089764
 rows 3 clean alarms 0 misses 0 of 1416 min contaminated full-fit max residual 0.02787007346706245
 rows 4 clean alarms 0 misses 0 of 3360 min contaminated full-fit max residual 0.28283487220312087
lab 3 in-batch/archive lines [ 0.759966 -0.110035] [ 0.760002 -0.109988] max archive discrepancy on clean sweep 0.00017442504795739389
 rows 3 clean alarms 0 misses 0 of 1428 min contaminated full-fit max residual 0.04981253218019421
 rows 4 clean alarms 0 misses 0 of 3360 min contaminated full-fit max residual 0.28356594665207824

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`45422 ms` LLM)
- **Rationale**: Isolate whether the ordinary-residual gate or the deleted-reading threshold causes the misses. If deleted residuals alone detect spikes without flagging clean sweeps, use them to trigger archive consultation; if they flag clean sweeps, inspect those command subsets before changing the solver.
- **Alternatives Weighed**: `Require an ordinary-fit residual above 0.002 before testing for EMI` (rejected: The previous audit found 49 misses on measured three-reading subsets; near-duplicate commands can make an ordinary fit absorb nearly the entire spike.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, pickle, numpy as np
p='/holdout_task/public_alpha_m97_c03_batch_02.pkl'
c=pickle.load(open(p,'rb'))[0]['dataset']['calibration_load_cells']
for n in (3,4,5):
    max_clean=[0.,0.]; misses=np.zeros((4,2),int); total=0; worst=None
    for lab in range(4):
        use=c['lab_id']==lab; x=c['cmd_stress'][use]; y=c['true_stress'][use]
        for rows in itertools.combinations(range(len(x)),n):
            idx=list(rows); a=np.column_stack((x[idx],np.ones(n))); z=y[idx]
            if np.ptp(a[:,0])<0.5: continue
            # Each row is a deleted-reading prediction residual, computed directly
            # rather than dividing two small quantities from the hat matrix.
            L=np.eye(n)
            for j in range(n):
                keep=np.arange(n)!=j
                L[j,keep]=-a[j]@np.linalg.pinv(a[keep])
            base=L@z; max_clean=np.maximum(max_clean,[max(abs(base)),max(abs(a@np.linalg.lstsq(a,z,rcond=None)[0]-z))])
            for j in range(n):
                for s in (-1,1):
                    total+=1
                    for i,amp in enumerate((3.01,3.1,4.,8.)):
                        score=max(abs(base+s*amp*L[:,j])); misses[i,0]+=score<=3.; misses[i,1]+=score<=2.99
                        if i>=2 and score<=3 and (worst is None or score<worst[0]): worst=(score,lab,a[:,0].tolist(),j,s*amp)
    print(n,'rows: clean max deleted and ordinary residual',max_clean.tolist(),'misses by amplitude at triggers 3.0 and 2.99:',misses.tolist(),'of',total,'worst >=4 miss',worst)
PY
```
- **Container Exit Code**: `0` (`1009 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3 rows: clean max deleted and ordinary residual [1.0041758785919668, 0.00017109391817249353] misses by amplitude at triggers 3.0 and 2.99: [[14, 14], [14, 14], [1, 0], [0, 0]] of 2826 worst >=4 miss (2.9958241214080332, 1, [-1.41018, 2.168228, -1.409689], 1, -4.0)
4 rows: clean max deleted and ordinary residual [0.0036016952471911168, 0.00017974534856168844] misses by amplitude at triggers 3.0 and 2.99: [[0, 0], [0, 0], [0, 0], [0, 0]] of 6720 worst >=4 miss None
5 rows: clean max deleted and ordinary residual [0.0009428346421885798, 0.00019252907497158134] misses by amplitude at triggers 3.0 and 2.99: [[0, 0], [0, 0], [0, 0], [0, 0]] of 10080 worst >=4 miss None

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a721c45317f6cf9f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a721c453 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`55857 ms` LLM)
- **Rationale**: Use the deleted reading’s prediction uncertainty to screen for EMI, then consult the archive to disambiguate plausible exclusions. Test that clean subsets still fit directly, that the sparse-EMI error falls, and that public strain and rate predictions do not regress.
- **Alternatives Weighed**: `Use an unadjusted deleted-residual threshold of 3.0` (rejected: It missed documented stress-test cases when the remaining commands were nearly identical and amplified ordinary measurement noise.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_02_solver.py')
s = p.read_text()
a = s.index('def _calibrate(')
b = s.index('def _design(', a)
p.write_text(s[:a] + '''def _calibrate(cal):
    coefficients = np.empty((4, 2))
    history = None
    # Measured clean sweeps have residuals below 2e-4 stress units.
    noise = 3e-4

    def archived_line(lab):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        sweep = next(row for row in reversed(history) if lab in row['lab_id'])
        keep = np.asarray(sweep['lab_id']) == lab
        return np.polyfit(np.asarray(sweep['cmd_stress'])[keep],
                          np.asarray(sweep['true_stress'])[keep], 1)

    labs = np.asarray(cal['lab_id'])
    for lab in range(4):
        x, y = (np.asarray(cal[key])[labs == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 2:
            coefficients[lab] = archived_line(lab)
            continue
        design = np.column_stack((x, np.ones(len(x))))
        coefficients[lab] = np.linalg.lstsq(design, y, rcond=None)[0]
        candidates = []
        for j in range(len(x)) if len(x) >= 3 else ():
            keep = np.arange(len(x)) != j
            if np.ptp(x[keep]) == 0:
                continue
            inverse = np.linalg.pinv(design[keep])
            line = inverse @ y[keep]
            predict = design[j] @ inverse
            deleted = abs(y[j] - predict @ y[keep])
            uncertainty = noise * (1 + np.abs(predict).sum())
            # A nearly repeated pair can amplify ordinary reading noise in
            # the deleted prediction. Account for that before calling EMI.
            if deleted > uncertainty and deleted + uncertainty > 3.0:
                error = np.max(np.abs(y[keep] - design[keep] @ line))
                candidates.append((error, j, keep, line))
        if not candidates:
            continue
        prior = archived_line(lab)
        best_error = min(item[0] for item in candidates)
        plausible = [item for item in candidates if item[0] <= best_error + noise]
        _, _, keep, line = min(plausible, key=lambda item: np.max(
            np.abs(y[item[2]] - design[item[2]] @ prior)))
        # Do not estimate a new slope from two effectively repeated commands
        # when their clean readings already agree with the archived line.
        if np.ptp(x[keep]) < 0.1 and np.max(np.abs(y[keep] - design[keep] @ prior)) <= 2 * noise:
            line = prior
        coefficients[lab] = line
    return coefficients[:, 0], coefficients[:, 1]


''' + s[b:])
PY
python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/audit_rheology_cal.py <<'PY'
import importlib.util, itertools, pickle, numpy as np
r='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',r+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(r+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; c=d['calibration_load_cells']; labs=c['lab_id']; base=np.array(m._calibrate(c)); original=m._ARCHIVE; o=m.solve_alpha_m97_c03_batch_02(d)
for out, truth in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=(o[out]-g[truth])**2; missing=~np.isfinite(d['test']['coupon_features_raw']).all(axis=1); print(out,'RMSE complete/missing',np.sqrt(e[~missing].mean()),np.sqrt(e[missing].mean()))
for n in (3,4):
    worst={amp:(0.,None) for amp in (3.01,3.1,4.,8.)}; alarms=0
    for lab in range(4):
        for rows in itertools.combinations(np.flatnonzero(labs==lab),n):
            idx=list(rows); sub={k:v[idx].copy() for k,v in c.items()}; x=sub['cmd_stress']; y=sub['true_stress']; line=np.polyfit(x,y,1)
            m._ARCHIVE='/tmp/nonexistent_clean_calibration_history.json'
            try: actual=np.array(m._calibrate({k:np.concatenate((v[labs!=lab],sub[k])) for k,v in c.items()}))[:,lab]
            except FileNotFoundError: alarms+=1
            else: np.testing.assert_allclose(actual,line,atol=1e-9,rtol=0)
            finally: m._ARCHIVE=original
            if np.ptp(x)<0.5: continue
            sweep={k:np.concatenate((sub[k],v[labs!=lab])) for k,v in c.items()}
            for j in range(n):
                for amp in worst:
                    for spike in (-amp,amp):
                        sweep['true_stress'][j]=y[j]+spike
                        est=np.array(m._calibrate(sweep))[:,lab]; err=float(max(abs(est-base[:,lab])))
                        if err>worst[amp][0]: worst[amp]=(err,(lab,x.tolist(),j,spike,est.tolist()))
                sweep['true_stress'][j]=y[j]
    print(n,'rows: clean archive alarms',alarms,'worst errors by spike amplitude',worst)
PY
python3 -I /tmp/audit_rheology_cal.py
```
- **Container Exit Code**: `0` (`68235 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
predicted_strain RMSE complete/missing 0.01597204593255129 0.4874157280577148
initial_rate_pred RMSE complete/missing 0.005833783176565937 0.15421150532615763
3 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (2.089641599015108, (1, [-1.41018, 0.620892, -1.409689], 1, -3.01, [-0.6620668020040323, -2.2396472927275095])), 3.1: (1.852170657408162, (0, [1.649489, 1.648912, -1.111114], 2, -3.1, [2.4129785664341004, -1.7321874234794108])), 4.0: (2.0915008229610064, (0, [1.649489, -1.504968, 1.648912], 1, -4.0, [2.558097467951382, -1.9715175890322552])), 8.0: (0.0027097261215492385, (2, [0.830827, -1.400488, -1.531561], 0, -8.0, [1.168174739880791, 0.13726731349299567]))}
4 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (0.0014352602819732163, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.01, [0.8210354040181903, -0.14857043343042833])), 3.1: (0.0014352602819732163, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.1, [0.8210354040181903, -0.14857043343042833])), 4.0: (0.0014352602819732163, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -4.0, [0.8210354040181903, -0.14857043343042833])), 8.0: (0.0014352602819732163, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -8.0, [0.8210354040181903, -0.14857043343042833]))}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eb9c202961e3e222 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_eb9c2029 (directory).
```

### Turn 15 — `execute_in_remote_docker` (`34764 ms` LLM)
- **Rationale**: A near-duplicate command pair can make the spiked reading hard to predict, while a different deleted reading can still show that the sweep is inconsistent. Separate those tests and let the archive identify the spike when every other reading agrees with it. Re-run the measured-subset audit to check whether this closes the three-row gap without consulting the archive on clean sweeps or regressing public predictions.
- **Alternatives Weighed**: `Accept the four-row improvement and leave three-row sweeps unchanged` (rejected: Three-row sweeps with near-duplicate measured commands still show gain or bias errors above 2.0 under supported EMI amplitudes.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_02_solver.py')
s = p.read_text()
s = s.replace('    noise = 3e-4', '    noise = 2e-4')
s = s.replace('        candidates = []\n', '        candidates = []\n        inconsistent = False\n        possible_emi = False\n')
a = s.index('            # A nearly repeated pair')
b = s.index('        best_error =', a)
s = s[:a] + '''            # Check line consistency separately from possible spike amplitude:
            # the highest-leverage reading can have the least precise prediction.
            inconsistent |= deleted > uncertainty
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
            continue
''' + s[b:]
s = s.replace('        _, _, keep, line = min(plausible, key=lambda item: np.max(\n            np.abs(y[item[2]] - design[item[2]] @ prior)))', '        _, keep, line = min(plausible, key=lambda item: np.max(\n            np.abs(y[item[1]] - design[item[1]] @ prior)))')
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/audit_rheology_cal.py
```
- **Container Exit Code**: `0` (`69410 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
predicted_strain RMSE complete/missing 0.01597204593255129 0.4874157280577148
initial_rate_pred RMSE complete/missing 0.005833783176565937 0.15421150532615763
3 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (1.5738656743803134, (0, [1.649489, -1.504968, 1.648912], 1, -3.01, [2.2442271166519028, -1.4538824404515622])), 3.1: (1.5049015686463787, (0, [1.649489, 1.648912, -1.748203], 2, -3.1, [2.202410428254438, -1.3849183347176275])), 4.0: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -4.0, [1.1681747398807898, 0.13726731349299778])), 8.0: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -8.0, [1.1681747398807898, 0.13726731349299778]))}
4 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.01, [0.8210354040181891, -0.14857043343042992])), 3.1: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.1, [0.8210354040181891, -0.14857043343042992])), 4.0: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -4.0, [0.8210354040181891, -0.14857043343042992])), 8.0: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -8.0, [0.8210354040181891, -0.14857043343042992]))}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6c6f9c2eee8eef84 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6c6f9c2e (directory).
```

### Turn 16 — `execute_in_remote_docker` (`41458 ms` LLM)
- **Rationale**: Check whether a lower noise threshold catches borderline spikes without flagging clean sweeps. If the residuals overlap, threshold tuning cannot close the gap without sacrificing clean in-batch calibration; retain the verified improvement and report the remaining limitation.
- **Alternatives Weighed**: `Consult archived calibration on every three-reading sweep` (rejected: The contract limits archive use, and clean three-reading sweeps should still fit directly.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools,pickle,numpy as np
c=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_02.pkl','rb'))[0]['dataset']['calibration_load_cells']; worst_clean=[]; ambiguous=[]; levels=(1e-4,1.25e-4,1.5e-4,1.75e-4,2e-4); alarms=np.zeros(len(levels),int); misses=alarms.copy()
for lab in range(4):
 use=c['lab_id']==lab; x=c['cmd_stress'][use]; y=c['true_stress'][use]
 for idx in itertools.combinations(range(x.size),3):
  xs=x[list(idx)]; ys=y[list(idx)]; a=np.column_stack((xs,np.ones(3))); L=np.eye(3)
  for j in range(3):
   k=np.arange(3)!=j; L[j,k]=-a[j]@np.linalg.pinv(a[k])
  w=np.abs(L).sum(axis=1); r=L@ys
  def flag(z,e): return np.max(abs(z)/w)>e and np.max(abs(z)+e*w)>3
  for i,e in enumerate(levels): alarms[i]+=flag(r,e)
  if np.ptp(xs)<0.5: continue
  for j,amp in itertools.product(range(3),(-3.01,3.01,-3.1,3.1)):
   z=r+amp*L[:,j]
   for i,e in enumerate(levels): misses[i]+=not flag(z,e)
   if not flag(z,2e-4):
    alt=ys.copy(); alt[j]+=amp; beta=np.linalg.lstsq(a,alt,rcond=None)[0]; ambiguous.append((lab,xs.tolist(),amp,float(np.max(abs(z)/w)),float(np.max(abs(z)+2e-4*w)),beta.tolist()))
print('noise, clean alarms, borderline misses:',list(zip(levels,alarms.tolist(),misses.tolist())))
print('current borderline misses:',ambiguous[:12], 'total',len(ambiguous))
PY
```
- **Container Exit Code**: `0` (`643 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
noise, clean alarms, borderline misses: [(0.0001, 0, 0), (0.000125, 0, 0), (0.00015, 0, 2), (0.000175, 0, 5), (0.0002, 0, 9)]
current borderline misses: [(0, [1.649489, -1.504968, 1.648912], -3.01, 0.0001978954552233811, 4.350586149037408, [2.2442271166519028, -1.4538824404515622]), (0, [1.649489, 1.648912, -1.748203], -3.01, 0.00017819170760464083, 4.453999789927559, [2.1759196101773326, -1.3412296636504626]), (0, [1.649489, 1.648912, -1.748203], -3.1, 0.0001858336598475195, 4.54399978992756, [2.202410428254438, -1.3849183347176275]), (1, [-1.41018, 1.74467, -1.409689], -3.01, 0.0001653398593905797, 4.694877618728595, [-0.13414582763479854, -1.495313274902485]), (1, [-1.41018, 1.74467, -1.409689], -3.1, 0.00017234336193429068, 4.784877618728596, [-0.1626755524750369, -1.5355383187753178]), (1, [-1.41018, 1.684877, -1.409689], -3.01, 0.00016986742731599667, 4.66297665982197, [-0.15259053426437338, -1.521319103729723]), (1, [-1.41018, 1.684877, -1.409689], -3.1, 0.00017700622960621913, 4.75297665982197, [-0.181671464815711, -1.5623213115724983]), (1, [-1.41018, 2.168228, -1.409689], -3.01, 0.0001376114243557404, 4.921024121408235, [-0.021185096190119993, -1.336046039067448]), (1, [-1.41018, 2.168228, -1.409689], -3.1, 0.00014378595783543773, 5.011024121408235, [-0.046337670655085525, -1.37150952199309])] total 9

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6c6f9c2eee8eef84 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6c6f9c2e (directory).
```

### Turn 17 — `execute_in_remote_docker` (`27126 ms` LLM)
- **Rationale**: Apply the narrower threshold where the audit found separation. Keep the edit only if isolated verification confirms zero clean archive alarms, gain/bias error below 0.01 across the tested spikes, and no material change in public prediction errors.
- **Alternatives Weighed**: `Lower the noise allowance throughout calibration` (rejected: Only the three-reading consistency check needs to change; the spike-amplitude and archive agreement tests already work on the measured sweeps.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
p=/holdout_task/alpha_m97_c03_batch_02_solver.py
backup=$(mktemp /tmp/rheology_v03.XXXXXX)
cp "$p" "$backup"
trap 'cp "$backup" "$p"' EXIT
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_02_solver.py')
s = p.read_text()
old = '            inconsistent |= deleted > uncertainty\n'
assert s.count(old) == 1
new = '''            # Three-point sweeps need a tighter consistency screen to
            # distinguish a borderline spike from a clean near-repeat.
            consistency_limit = uncertainty * (0.625 if len(x) == 3 else 1.0)
            inconsistent |= deleted > consistency_limit
'''
p.write_text(s.replace(old, new))
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/audit_rheology_cal.py > /tmp/rheology_final_audit.log
cat /tmp/rheology_final_audit.log
python3 -I - <<'PY'
from ast import literal_eval
from pathlib import Path
lines = Path('/tmp/rheology_final_audit.log').read_text().splitlines()
for n in (3, 4):
    row = next(s for s in lines if s.startswith(f'{n} rows:'))
    assert 'clean archive alarms 0' in row, row
    cases = literal_eval(row.split('worst errors by spike amplitude ', 1)[1])
    assert max(err for err, _ in cases.values()) < 0.01, row
for name, bounds in [('predicted_strain', (0.016, 0.488)), ('initial_rate_pred', (0.00584, 0.155))]:
    row = next(s for s in lines if s.startswith(name + ' RMSE complete/missing '))
    values = tuple(map(float, row.split()[-2:]))
    assert all(v <= b for v, b in zip(values, bounds)), row
PY
trap - EXIT
rm -f "$backup"
```
- **Container Exit Code**: `0` (`67089 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
predicted_strain RMSE complete/missing 0.01597204593255129 0.4874157280577148
initial_rate_pred RMSE complete/missing 0.005833783176565937 0.15421150532615763
3 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -3.01, [1.1681747398807898, 0.13726731349299778])), 3.1: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -3.1, [1.1681747398807898, 0.13726731349299778])), 4.0: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -4.0, [1.1681747398807898, 0.13726731349299778])), 8.0: (0.002709726121547129, (2, [0.830827, -1.400488, -1.531561], 0, -8.0, [1.1681747398807898, 0.13726731349299778]))}
4 rows: clean archive alarms 0 worst errors by spike amplitude {3.01: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.01, [0.8210354040181891, -0.14857043343042992])), 3.1: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -3.1, [0.8210354040181891, -0.14857043343042992])), 4.0: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -4.0, [0.8210354040181891, -0.14857043343042992])), 8.0: (0.0014352602819716342, (1, [-1.41018, 0.620892, -1.528098, -1.409689], 1, -8.0, [0.8210354040181891, -0.14857043343042992]))}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8cc7a6438ae8287 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_c8cc7a64 (directory).
```

### Turn 18 — `submit_completed_task` (`14211 ms` LLM)
- **Rationale**: The final saved artifact passed isolated schema validation and corrected the measured sparse-calibration failure without triggering archive lookup on clean subsets or changing public prediction errors.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_02_solver.py`. The public schema passed in an isolated Python process. On measured three- and four-reading calibration subsets with injected EMI spikes, the worst gain or bias error fell from 33.59 to 0.00271. No clean subset triggered archive lookup, and public predictions were unchanged.`

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
