# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_07__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_07` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `377,754` (`363,830` / `81,408` / `13,924`) |
| **Total Episode Wall Time (s)** | `442.04s` |

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

### Turn 1 — `execute_in_remote_docker` (`12924 ms` LLM)
- **Rationale**: Inspect the public contract and available calibration and alignment primitives before fitting; a schema or convention mismatch would change the loader and validation splits.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -lh /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`297 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lic_asset_index_v1"
}total 76K
-rw-r--r-- 1 root root 7.2K Nov 27 07:16 alpha_m97_c02_batch_01_solver.py
-rw-r--r-- 1 root root 7.0K Nov 27 07:16 alpha_m97_c02_batch_02_solver.py
-rw-r--r-- 1 root root 7.9K Nov 27 07:16 alpha_m97_c02_batch_03_solver.py
-rw-r--r-- 1 root root 8.0K Nov 27 07:16 alpha_m97_c02_batch_04_solver.py
-rw-r--r-- 1 root root 8.0K Nov 27 07:16 alpha_m97_c02_batch_05_solver.py
-rw-r--r-- 1 root root 7.8K Nov 27 07:16 alpha_m97_c02_batch_06_solver.py
-rw-r--r-- 1 root root 7.3K Nov 27 07:16 alpha_m97_c02_s00_solver.py
-r--r--r-- 1 root root  11K Nov 27 07:16 archive_calibration_history.json
-rw-r--r-- 1 root root 3.3K Nov 27 07:16 archive_manifest.json
-rw-r--r-- 1 root root  340 Nov 27 07:16 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`20172 ms` LLM)
- **Rationale**: The public batch contains only two calibration rows across four channels. Identify which channels require prefix calibration and check the reference-panel and strain observation formats before writing the fit. Inspect only solver inputs, not the answer arrays.
- **Alternatives Weighed**: `Use the pre-LLM parameter payload directly` (rejected: The public gate rejected that candidate; inspect channel coverage and prefix evidence before choosing the calibration.)

```bash
cd /holdout_task
sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_06_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
a=json.load(open(p)); print('calibration history structure', list(a) if isinstance(a,dict) else len(a)); print('calibration head',json.dumps(a)[:2300])
d=pickle.load(open('public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']; print('dataset keys',list(d))
for split in ['train','test']:
 s=d[split];print(split)
 for k,v in s.items():
  if isinstance(v,dict): print(k,[(kk,np.shape(vv)) for kk,vv in v.items()]); continue
  x=np.asarray(v);print(k,x.shape,x.dtype,'nonfinite',np.sum(~np.isfinite(x)) if x.dtype.kind in 'fiu' else '-', 'values' ,np.unique(x,return_counts=True) if x.ndim==1 and len(np.unique(x))<=12 else '')
print('in-batch calibration',d['calibration_load_cells']);print('durations',d['past_dt'],'evaluation grid',d['eval_times'])
PY
```
- **Container Exit Code**: `0` (`495 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
999466, 1.338557, -0.977277, 2.183171, -0.406287, 0.845669, 2.186524, 1.679156, 1.00863, -1.6169, 0.078971, 0.018612, 1.244379, -0.508587, -1.248699, 1.995971, 0.932757, 1.092876, -1.329735, -0.12659, 1.905708, 0.686713, -0.873095, 0.074674, -1.141014, -0.732653], "true_stress": [-0.935006, -1.569947, -1.435516, 0.40229, 0.853447, -1.558858, -1.016829, 0.535962, 0.166432, -1.452561, 0.651821, 2.682504, 2.885814, 2.333603, 1.469255, 1.91348, -1.120155, 3.019816, -0.372068, 1.267711, 1.958514, 1.527341, 0.957286, -1.27431, 0.167035, 0.115871, 1.157698, -0.332211, -0.96121, 1.796516, 1.026744, 1.225138, -1.778862, -0.287078, 2.233145, 0.721528, -1.212573, -0.03726, -1.544707, -1.038538]}, {"episode_short_name": "alpha_m97_c02_batch_02", "lab_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "cmd_stress": [2.117655, -1.165749, -1.673814, -1.
dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
train
lab_id (180,) int64 nonfinite 0 values (array([0, 1, 2, 3]), array([43, 53, 38, 46]))
stratum (180,) int64 nonfinite 0 values (array([0, 1, 2]), array([55, 74, 51]))
coupon_features_raw (180, 4) float64 nonfinite 16 values 
ref_coupons (180, 32) float64 nonfinite 229 values 
past_stress_cmd (180, 18) float64 nonfinite 0 values 
stress_hold_cmd (180,) float64 nonfinite 0 values 
strain_0 (180,) float64 nonfinite 0 values 
aux_chamber_telemetry (180, 3) float64 nonfinite 0 values 
strain_curves_obs (180, 10) float64 nonfinite 0 values 
instant_rate_obs (180,) float64 nonfinite 0 values 
test
lab_id (72,) int64 nonfinite 0 values (array([0, 1, 2, 3]), array([16, 18, 15, 23]))
stratum (72,) int64 nonfinite 0 values (array([0, 1, 2]), array([19, 30, 23]))
coupon_features_raw (72, 4) float64 nonfinite 12 values 
ref_coupons (72, 32) float64 nonfinite 89 values 
past_stress_cmd (72, 18) float64 nonfinite 0 values 
stress_hold_cmd (72,) float64 nonfinite 0 values 
strain_0 (72,) float64 nonfinite 0 values 
aux_chamber_telemetry (72, 3) float64 nonfinite 0 values 
in-batch calibration {'lab_id': array([1, 1]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-0.76184545,  1.23592146])}
durations [0.7514 0.49   0.6846 0.4219 0.4716 0.4433 0.219  0.4691 0.6796 0.7763
 0.6334 0.3123 0.7556 0.708  0.4274 0.7273 0.7632 0.3231] evaluation grid [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
```

### Turn 3 — `execute_in_remote_docker` (`20419 ms` LLM)
- **Rationale**: Compare robust prefix fits to detect calibration changes or EMI that would invalidate a latest-sweep fallback. Read only the relevant prior solver section to recover the panel convention and variable-projection setup before fitting the plateau model.
- **Alternatives Weighed**: `Use the same archived episode for all undersampled channels without checking the calibration history` (rejected: Three channels have no in-batch points; the prefix may contain distinct calibration states or contaminated sweeps.)

```bash
cd /holdout_task
p=/tmp/hegel_continual_archive/alpha_m97_c02_batch_06_solver.py
sed -n '59,205p' "$p"
python3 - <<'PY'
import json,numpy as np
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for r in h:
 a,x,y=map(np.asarray,(r['lab_id'],r['cmd_stress'],r['true_stress'])); out=[]
 for c in range(4):
  u,v=x[a==c],y[a==c]
  if np.unique(u).size<2: out.append((len(u),)); continue
  i,j=np.triu_indices(len(u),1); ok=u[i]!=u[j]; i,j=i[ok],j[ok]; g=(v[j]-v[i])/(u[j]-u[i]); b=v[i]-g*u[i]; E=abs(v-g[:,None]*u-b[:,None]); ix=np.argmin(np.median(E,axis=1)); q=E[ix]<.035
  f=np.polyfit(u[q],v[q],1); out.append((len(u),int(q.sum()),*f.round(5),round(float(np.max(abs(v-np.polyval(f,u)))),4)))
 print(r['episode_short_name'],out)
PY
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nc:]))]
    bounds = (np.r_[np.full(2, limits[0]), np.full(9, -np.inf)],
              np.r_[np.full(2, limits[1]), np.full(9, np.inf)])
    fit = min((least_squares(lambda p: raw(p) / scale, start, bounds=bounds,
                             loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=110)
               for _, start in projected[:3]), key=lambda result: result.cost)
    r = raw(fit.x)
    scale = np.r_[np.full(nc, mad(r[:nc])), np.full(nr, mad(r[nc:]))]
    fit = least_squares(lambda p: raw(p) / scale, fit.x, bounds=bounds,
                        loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=75)
    return fit.x[:2], fit.x[2:]


def solve_alpha_m97_c02_batch_06(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], dt, times, curves, rates)
    curve, rate = _design(log_tau, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
alpha_m97_c02_s00 [(10, 10, 0.99998, 0.0, 0.0002), (10, 10, 0.99999, 4e-05, 0.0001), (10, 10, 1.00001, 3e-05, 0.0001), (10, 10, 1.00001, -3e-05, 0.0001)]
alpha_m97_c02_batch_01 [(10, 10, 0.79005, -0.13999, 0.0002), (10, 10, 1.30996, 0.16006, 0.0002), (10, 10, 0.84996, 0.10004, 0.0001), (10, 10, 1.24, -0.12996, 0.0001)]
alpha_m97_c02_batch_02 [(10, 10, 0.79001, -0.14, 0.0002), (10, 10, 1.30998, 0.15996, 0.0002), (10, 10, 0.84999, 0.09998, 0.0002), (10, 10, 1.24004, -0.12998, 0.0002)]
alpha_m97_c02_batch_03 [(10, 10, 0.78997, -0.13996, 0.0003), (10, 10, 1.30997, 0.15996, 0.0001), (10, 10, 0.84999, 0.10002, 0.0002), (10, 10, 1.23997, -0.13004, 0.0002)]
alpha_m97_c02_batch_04 [(0,), (2, 2, 1.31004, 0.15991, 0.0), (0,), (0,)]
alpha_m97_c02_batch_05 [(10, 10, 0.79001, -0.14, 0.0002), (10, 10, 1.30998, 0.15996, 0.0002), (10, 10, 0.84999, 0.09998, 0.0002), (10, 10, 1.24004, -0.12998, 0.0002)]
alpha_m97_c02_batch_06 [(3, 2, 0.71857, -4.78286, 4.7214), (12, 12, 0.74, 0.20002, 0.0002), (12, 12, 1.37002, -0.17003, 0.0002), (3, 2, 1.13286, 4.59857, 4.825)]
```

### Turn 4 — `execute_in_remote_docker` (`25415 ms` LLM)
- **Rationale**: Check which calibration states change damage eligibility and recover the archived zero-damage convention. Those results determine which design columns and held-out slices can distinguish the calibration hypotheses.
- **Alternatives Weighed**: `Assume every coupon is in the damage-active plateau regime` (rejected: The damage threshold is imposed in calibrated coordinates, so active and inactive counts must be checked before choosing the design.)

```bash
cd /holdout_task
for p in /tmp/hegel_continual_archive/*solver.py; do echo "--- $(basename "$p")"; grep -nE 'terminal =|damage =|activation =|np.log1p|def _design|yield_stress|crit_stress' "$p"; done
python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']; s=D['train']; t=D['eval_times']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; dt=D['past_dt']; lab=s['lab_id']; opts=[np.array([[.79,-.14],[.739914,.200044],[.85,.10],[1.24,-.13]]), np.array([[.79,-.14],[.739914,.200044],[1.37,-.17],[1.24,-.13]])]; print('yield',D['yield_stress']);
for c in range(4):
 m=lab==c; p=s['past_stress_cmd'][m]; h=s['stress_hold_cmd'][m]; print('lab',c,'past',np.quantile(p,[0,.05,.5,.95,1]),'hold',np.quantile(h,[0,.05,.5,.95,1]),'strain change',np.quantile(y[m,-1],[0,.5,1]),'rate',np.quantile(r[m],[0,.5,1]));
 for o in opts:
  g,b=o[c]; dmg=np.maximum(abs(g*p+b)-D['crit_stress'],0)**2@dt; print('active',sum(dmg>0),'damage quantiles',np.quantile(dmg,[0,.25,.5,.75,1]))
print('finite per reference column:', [np.isfinite(s['ref_coupons'][lab==c]).sum(0).min() for c in range(4)])
PY
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
excess = np.maximum(0., np.abs(past) - dataset['crit_stress'])
102:    phi = np.broadcast_to(np.log1p(times) if plateau else times, (len(past), len(times)))
--- alpha_m97_c02_batch_02_solver.py
--- alpha_m97_c02_batch_03_solver.py
--- alpha_m97_c02_batch_04_solver.py
97:def _design(log_tau, past, hold, features, dt, times, act):
103:    phi = np.log1p(times)
115:    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
--- alpha_m97_c02_batch_05_solver.py
97:def _design(log_tau, past, hold, features, dt, times, act):
103:    phi = np.log1p(times)
115:    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
--- alpha_m97_c02_batch_06_solver.py
101:def _design(log_tau, past, hold, features, dt, times):
107:    terminal = times / (1.0 + 0.85 * times)
--- alpha_m97_c02_s00_solver.py
yield 2.65
lab 0 past [-4.87111843 -4.55605821  0.08741617  4.77556234  5.23784017] hold [-2.17534557 -1.89213722  0.57726835  2.65134481  2.71706456] strain change [-2.12080043  0.7711303   3.38920112] rate [-3.29119082  0.91239323  3.85020497]
active 28 damage quantiles [0.         0.         3.16309942 3.76527982 4.85689263]
active 28 damage quantiles [0.         0.         3.16309942 3.76527982 4.85689263]
lab 1 past [-5.66833283 -5.38518436 -0.25233335  4.71044481  5.12927322] hold [-2.95265541 -2.73311892 -0.35740405  2.01778351  2.2865027 ] strain change [-2.97142002  0.0852002   3.61268925] rate [-3.82595335 -0.25261377  4.70889713]
active 41 damage quantiles [0.         2.44757685 3.26421636 3.93876958 5.12917986]
active 41 damage quantiles [0.         2.44757685 3.26421636 3.93876958 5.12917986]
lab 2 past [-2.79530639 -2.42681025  0.32212507  2.87181372  3.03047245] hold [-1.34128029 -1.15027464 -0.28062336  1.53916894  1.58597153] strain change [-3.44396207 -0.46260029  3.97638032] rate [-4.32629264 -0.85833529  4.19275747]
active 6 damage quantiles [0.         0.         0.         0.         0.00034188]
active 24 damage quantiles [0.         0.         3.24937142 3.68737423 5.20817877]
lab 3 past [-3.12000965e+00 -2.85173609e+00  2.70282258e-03  3.05352605e+00
  3.32866191e+00] hold [-1.4871     -1.37224214  0.18801573  1.21434214  1.70722339] strain change [-2.53341061  0.21576679  2.39720519] rate [-3.30830838  0.19024884  3.35539737]
active 27 damage quantiles [0.         0.         2.87326168 3.94206261 4.9144064 ]
active 27 damage quantiles [0.         0.         2.87326168 3.94206261 4.9144064 ]
finite per reference column: [37, 48, 31, 41]
```

### Turn 5 — `execute_in_remote_docker` (`49148 ms` LLM)
- **Rationale**: Use measured training curves and rates to screen the unresolved calibration, alignment, and zero-damage conventions. Lab-by-damage residuals will show which hypotheses warrant held-out testing.
- **Alternatives Weighed**: `Trust the pre-LLM gate rejection as evidence against its calibration` (rejected: A parameter payload does not isolate calibration from constitutive-model error. Compare both archived states using the same measured strain and initial-rate model.)

```bash
sed -n '60,155p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_05_solver.py
cd /holdout_task
cat > /tmp/rheo07_probe.py <<'PY'
import json, pickle
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares
from itertools import product

F='/tmp/hegel_continual_archive/archive_calibration_history.json'
D=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']
N=len(D['train']['lab_id']); dt=D['past_dt']; t=D['eval_times']; K=len(t)
lab,grp,x,ref,pc,hc,s0=(np.concatenate((D['train'][k],D['test'][k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons','past_stress_cmd','stress_hold_cmd','strain_0'))
for c in range(4):
    a=lab==c; p=ref[a]; x[a]=(x[a]-np.nanmedian(np.nanmean(p,axis=0)))/np.nanmedian(np.nanstd(p,axis=0))
cell=np.array([[np.nanmean(x[(lab==c)&(grp==g)],axis=0) for g in range(3)] for c in range(4)])
xf={}
for mode in ('balanced','local'):
    f=np.nanmean(cell,axis=0)[grp] if mode=='balanced' else cell[lab,grp]
    xf[mode]=np.c_[np.ones(len(lab)),np.where(np.isfinite(x),x,f)]
yc=D['train']['strain_curves_obs']-s0[:N,None]; yr=D['train']['instant_rate_obs']; age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
arc=json.load(open(F)); h=arc[-1]; l,u,v=[np.asarray(h[k]) for k in ('lab_id','cmd_stress','true_stress')]
for c in [0,3]:
    print('recent suspected EMI',c,list(zip(u[l==c],v[l==c])))
cal=[(.79,-.14), tuple(np.polyfit(D['calibration_load_cells']['cmd_stress'],D['calibration_load_cells']['true_stress'],1)), (1.37,-.17), (1.24,-.13)]

def design(p,hold,features,act,lt,gate):
    tau=np.exp(lt); z=p@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); gap=hold[:,None]-z
    e=-np.expm1(-t[:,None]/tau); phi=np.log1p(t)[None,:] if not gate else np.where(act[:,None]>0,np.log1p(t),t)
    ah=act*hold; az=act*z[:,0]
    return np.concatenate((gap[:,None,:]*e, features[:,None,:]*phi[:,:,None], ah[:,None,None]*e[None,:,0,None], az[:,None,None]*phi[:,:,None]),axis=2), np.c_[gap/tau,features,ah/tau[0],az]

def run(cc,gate,mode,train,test,rob=True):
    g,b=np.asarray(cc)[lab].T; p=pc*g[:,None]+b[:,None]; hold=hc*g+b; damage=np.maximum(np.abs(p)-D['crit_stress'],0.)**2@dt; act=np.tanh(damage)
    y=np.r_[yc[train].ravel(),yr[train]]; m=len(train)*K
    def mat(lt,ids):
        ac,ar=design(p[ids],hold[ids],xf[mode][ids],act[ids],lt,gate)
        return np.r_[ac.reshape(-1,9),ar]
    w=np.r_[np.ones(m),np.full(len(train),2.)]
    def project(lt,get=False):
        a=mat(lt,train)*w[:,None]; b=y*w
        beta=lstsq(a,b,lapack_driver='gelsy',check_finite=False)[0]; e=a@beta-b
        return (e@e,np.r_[lt,beta]) if get else e
    lim=np.log([.025,80.]); fits=[]
    for a in [(0.24,2.2),(0.8,7.),(2.2,0.24),(7.,0.8)]:
        fit=least_squares(project,np.log(a),bounds=lim,max_nfev=45)
        fits.append(project(fit.x,True))
    _,params=min(fits,key=lambda q:q[0]); raw=lambda p: mat(p[:2],train)@p[2:]-y
    if rob:
        for _ in range(2):
            r=raw(params); med=lambda a:max(1e-6,1.4826*np.median(np.abs(a-np.median(a)))); scale=np.r_[np.full(m,med(r[:m])),np.full(len(train),med(r[m:]))]
            params=least_squares(lambda q:raw(q)/scale,params,bounds=(np.r_[[lim[0]]*2,[-np.inf]*9],np.r_[[lim[1]]*2,[np.inf]*9]),loss='soft_l1',f_scale=1.5,x_scale='jac',max_nfev=55).x
    ae=mat(params[:2],test)@params[2:]-np.r_[yc[test].ravel(),yr[test]]
    return ae[:len(test)*K].reshape(-1,K),ae[len(test)*K:],params,damage[test]

def report(e,r,p,damage,ids):
    def rm(v):return np.sqrt(np.mean(v*v)).round(5)
    print('all',rm(e),rm(r),'tau',np.exp(p[:2]).round(5),'hw',p[2:].round(5),'lab/active',[(str(c)+str(a),int(m.sum()),rm(e[m]),rm(r[m])) for c,a in product(range(4),[0,1]) if (m:=(lab[ids]==c)&((damage>0)==a)).any()])

if __name__=='__main__':
    ids=np.arange(N)
    for stress,gate,mode in product([(.85,.1),(1.37,-.17)],[False,True],['balanced','local']):
        cc=cal.copy(); cc[2]=stress
        print('candidate',stress,gate,mode); report(*run(cc,gate,mode,ids,ids),ids)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo07_probe.py
```
- **Container Exit Code**: `0` (`1293 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8836)]
candidate (0.85, 0.1) True balanced
all 1.10919 0.44207 tau [0.5604  3.44606] hw [ 0.74992  0.58811  0.05778  0.07869 -0.09562  0.07813 -0.05957  0.13486
  0.33538] lab/active [('00', 15, 0.23489, 0.24488), ('01', 28, 1.24104, 0.21071), ('10', 12, 0.2695, 0.15372), ('11', 41, 0.49172, 0.13669), ('20', 32, 1.06271, 0.82432), ('21', 6, 0.68621, 1.18321), ('30', 19, 0.35425, 0.25188), ('31', 27, 2.15483, 0.15336)]
candidate (0.85, 0.1) True local
all 1.10933 0.44192 tau [0.56039 3.42401] hw [ 0.75011  0.5882   0.05755  0.07884 -0.09545  0.07734 -0.05898  0.13357
  0.3369 ] lab/active [('00', 15, 0.2379, 0.24693), ('01', 28, 1.24135, 0.21129), ('10', 12, 0.26922, 0.15512), ('11', 41, 0.49173, 0.13737), ('20', 32, 1.06182, 0.82339), ('21', 6, 0.68376, 1.18039), ('30', 19, 0.35772, 0.25195), ('31', 27, 2.15524, 0.15679)]
candidate (1.37, -0.17) False balanced
all 1.08811 0.0443 tau [0.565   4.01449] hw [ 0.67821  0.54529  0.1394   0.15711 -0.129    0.11711 -0.08885  0.26608
  0.25244] lab/active [('00', 15, 0.19893, 0.0852), ('01', 28, 1.25452, 0.02073), ('10', 12, 0.17285, 0.01477), ('11', 41, 0.47251, 0.01941), ('20', 14, 1.12851, 0.02126), ('21', 24, 0.80442, 0.05057), ('30', 19, 0.26023, 0.03746), ('31', 27, 2.1456, 0.06665)]
candidate (1.37, -0.17) False local
all 1.0884 0.04684 tau [0.56536 4.02693] hw [ 0.67883  0.54493  0.13943  0.15721 -0.12905  0.11697 -0.08879  0.26636
  0.25253] lab/active [('00', 15, 0.20292, 0.08889), ('01', 28, 1.25487, 0.0236), ('10', 12, 0.17287, 0.01479), ('11', 41, 0.4725, 0.01937), ('20', 14, 1.12833, 0.01472), ('21', 24, 0.80469, 0.05217), ('30', 19, 0.26129, 0.04379), ('31', 27, 2.14603, 0.07147)]
candidate (1.37, -0.17) True balanced
all 1.09797 0.09551 tau [0.56149 4.16137] hw [ 0.67571  0.54174  0.09315  0.11128 -0.09215  0.09221 -0.06882  0.26335
  0.25367] lab/active [('00', 15, 0.24582, 0.12737), ('01', 28, 1.24154, 0.11151), ('10', 12, 0.33033, 0.08026), ('11', 41, 0.47932, 0.07767), ('20', 14, 1.14781, 0.05655), ('21', 24, 0.80141, 0.10053), ('30', 19, 0.41805, 0.08943), ('31', 27, 2.15184, 0.10299)]
candidate (1.37, -0.17) True local
all 1.09848 0.09691 tau [0.56227 4.18033] hw [ 0.67709  0.54044  0.09272  0.1115  -0.09138  0.09216 -0.06824  0.26367
  0.25427] lab/active [('00', 15, 0.24756, 0.13075), ('01', 28, 1.24178, 0.11217), ('10', 12, 0.32931, 0.08073), ('11', 41, 0.47935, 0.07816), ('20', 14, 1.14727, 0.05421), ('21', 24, 0.80168, 0.10199), ('30', 19, 0.42501, 0.08914), ('31', 27, 2.15247, 0.10736)]
```

### Turn 6 — `execute_in_remote_docker` (`22001 ms` LLM)
- **Rationale**: The recent channel-2 calibration lowers rate RMSE from 0.449 to 0.044, but curve RMSE remains 1.088. Residuals by time, coupon, and lab–damage slice will show whether to fit around sparse corrupt measurements or revisit a systematic curve mismatch before held-out validation.
- **Alternatives Weighed**: `Change the constitutive law to reduce pooled strain RMSE` (rejected: The current RMSE does not distinguish isolated measurement spikes from a systematic error in the time law.)

```bash
cd /holdout_task
python3 - <<'PY'
import json
j=json.load(open('public_asset_index.json')); print(json.dumps(j,indent=1)[:7000])
PY
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
from rheo07_probe import *
ids=np.arange(N); e,r,p,dmg=run(cal,False,'balanced',ids,ids)
q=[0,.1,.25,.5,.75,.9,.95,.98,1.]
print('absolute curve residual quantiles / time'); print(np.quantile(abs(e),q,axis=0).round(4));print('absolute rate residual quantiles',np.quantile(abs(r),q).round(4));print('large curve entries per coupon',np.unique((abs(e)>.2).sum(1),return_counts=True));print('large curve entries per time',np.sum(abs(e)>.2,axis=0));
for c,a in product(range(4),[0,1]):
 m=(lab[:N]==c)&((dmg>0)==a); print(c,a,'curve absolute medians',np.median(abs(e[m]),axis=0).round(4),'rate median',np.median(abs(r[m])).round(4))
for i in np.argsort(np.max(abs(e),axis=1))[-14:]:
 print('coupon',i,'lab',lab[i],'damage',round(dmg[i],3),'residual',e[i].round(3),'rate error',round(r[i],3))
PY
```
- **Container Exit Code**: `0` (`542 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ian 0.022
3 0 curve absolute medians [0.0081 0.0125 0.0113 0.0163 0.0188 0.0172 0.0149 0.0208 0.0159 0.0145] rate median 0.0146
3 1 curve absolute medians [0.0104 0.0109 0.0145 0.0103 0.0133 0.0163 0.0111 0.0202 0.0137 0.0078] rate median 0.0125
coupon 101 lab 3 damage 2.974 residual [ 2.91e+00  3.00e-03  2.40e-02  1.00e-02  6.00e-03 -6.00e-03  2.30e-02
  2.30e-02 -5.86e-01 -2.00e-03] rate error 0.02
coupon 169 lab 2 damage 0.0 residual [ 0.011 -0.012  0.025 -0.015 -0.005  0.007  0.007  2.917  0.023  0.003] rate error 0.01
coupon 108 lab 1 damage 3.516 residual [ 0.000e+00 -3.000e-02  1.200e-02 -2.600e-02  1.200e-02 -1.000e-03
  1.000e-03 -1.000e-03  7.000e-03 -3.599e+00] rate error -0.009
coupon 176 lab 2 damage 0.0 residual [-4.000e-03 -4.000e-03 -4.000e-03  3.200e-02 -5.000e-03  8.000e-03
  1.000e-02  1.200e-02  7.000e-03  4.023e+00] rate error -0.004
coupon 100 lab 2 damage 3.614 residual [-0.02  -0.029 -0.011 -0.009 -0.041  0.027 -0.005  0.016  4.117 -0.005] rate error -0.02
coupon 117 lab 2 damage 3.388 residual [ 2.000e-03 -1.000e-02  4.000e-03  1.500e-02 -3.000e-02  6.561e+00
 -4.300e-02 -6.000e-03  4.000e-03  7.600e-02] rate error -0.006
coupon 58 lab 2 damage 4.16 residual [ 2.000e-03  7.183e+00  2.300e-02  6.120e+00  5.000e-03  4.000e-03
  1.100e-02 -2.800e-02 -1.500e-02  2.000e-03] rate error -0.045
coupon 92 lab 3 damage 3.137 residual [0.01  0.016 0.055 0.067 0.074 0.097 7.281 0.132 0.128 0.114] rate error 0.081
coupon 25 lab 1 damage 3.768 residual [ 1.000e-03  9.000e-03 -2.100e-02 -3.000e-03 -5.000e-03 -2.000e-03
  3.100e-02  7.971e+00  7.000e-03  1.000e-02] rate error 0.001
coupon 107 lab 0 damage 4.234 residual [ 1.8000e-02 -2.5000e-02 -1.0000e-03 -4.0000e-03  0.0000e+00  2.0000e-03
  1.0386e+01 -3.6000e-02  2.0000e-02 -2.4000e-02] rate error -0.024
coupon 50 lab 0 damage 4.107 residual [-1.5000e-02  2.5000e-02  1.0949e+01  4.0000e-03  1.0000e-02  8.0000e-03
 -3.0000e-03  8.0000e-03 -2.3000e-02  1.1000e-02] rate error -0.0
coupon 148 lab 2 damage 0.0 residual [-2.0000e-03 -2.0000e-02  4.0000e-03  9.0000e-03  2.0000e-03  5.0000e-03
  1.4000e-02 -4.2170e+00  1.1376e+01 -5.0000e-03] rate error 0.018
coupon 111 lab 0 damage 4.857 residual [ 9.000e-03  7.000e-03  1.600e-02  3.050e-01 -3.800e-02 -8.000e-03
 -7.820e-01  1.378e+01 -2.100e-02 -7.000e-03] rate error -0.008
coupon 81 lab 3 damage 2.964 residual [ 4.0000e-03 -2.0000e-03  1.4000e-02 -8.0000e-03 -2.0000e-02  3.4000e-02
 -7.0000e-03 -3.4204e+01  4.0000e-03  3.0000e-03] rate error 0.008
```

### Turn 7 — `execute_in_remote_docker` (`78772 ms` LLM)
- **Rationale**: Implement the calibrated recurrence, panel alignment, and shared curve/rate fit. Run the isolated public validator to catch contract failures, then inspect fitted calibration and weights before held-out lab and horizon tests.
- **Alternatives Weighed**: `Add time-law terms to reduce raw strain RMSE` (rejected: The largest deviations are isolated within otherwise well-fit curves, so extra constitutive terms would fit measurement spikes.); `Use the older channel-2 sweep` (rejected: It raised measured rate RMSE from 0.044 to 0.449 under the same plateau model.)

```bash
cat > /holdout_task/alpha_m97_c02_batch_07_solver.py <<'PY'
"""Two-branch hereditary strain and logarithmic plateau for campaign batch 07."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(table, channel):
    x, y = (np.asarray(table[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    keep = (np.asarray(table['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _sweep_fit(x, y):
    i, j = np.triu_indices(x.size, 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not i.size:
        return None, 0, False
    g = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - g * x[i]
    error = np.abs(y - g[:, None] * x - b[:, None])
    consensus = error < 0.025
    best = np.lexsort((np.minimum(error, 0.025).sum(axis=1), -consensus.sum(axis=1)))[0]
    clean = consensus[best]
    emi = bool(np.max(error[best]) > 3.0)
    use = clean if emi else np.ones(x.size, dtype=bool)
    return np.polyfit(x[use], y[use], 1), int(use.sum()), emi


def _calibrate(table):
    archive = None
    calibration = []
    for channel in range(4):
        x, y = _sweep(table, channel)
        fit, count, emi = _sweep_fit(x, y)
        if fit is None or (emi and count < 4):
            if archive is None:
                archive = json.loads(_ARCHIVE.read_text())
            # Use the most recent identifiable sweep; a pair among EMI
            # spikes is not a reliable line, even when it fits exactly.
            for sweep in reversed(archive):
                u, v = _sweep(sweep, channel)
                candidate, clean_count, contaminated = _sweep_fit(u, v)
                if candidate is not None and (not contaminated or clean_count >= 4):
                    fit = candidate
                    break
        if fit is None:
            raise ValueError(f'No usable load-cell sweep for channel {channel}')
        calibration.append(fit)
    return np.array(calibration)


def _align(lab, stratum, raw, ref):
    x = np.array(raw, dtype=float, copy=True)
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).any(axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale

    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        count = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), count, out=means[i], where=count > 0)
    overall = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        rows = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=overall.copy(), where=count > 0)
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.column_stack((np.ones(len(x)), x))


def _prepare(dataset, calibration=None):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, raw, ref, past_cmd, hold_cmd, strain0 = (
        np.concatenate((train[k], test[k])) for k in keys)
    if calibration is None:
        calibration = _calibrate(dataset['calibration_load_cells'])
    gain, bias = calibration[lab].T
    past = gain[:, None] * past_cmd + bias[:, None]
    hold = gain * hold_cmd + bias
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    damage = (excess**2) @ dt
    plateau = dataset['shear_banding_index'] <= 0.50
    act = np.tanh(damage) if plateau else np.zeros_like(damage)
    return (past, hold, _align(lab, stratum, raw, ref), act), strain0, dt, times, plateau


def _design(log_tau, inputs, dt, times, plateau):
    past, hold, features, act = inputs
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    z = past @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = hold[:, None] - z
    step = -np.expm1(-times[:, None] / tau)
    phi = np.log1p(times) if plateau else times
    ah, az = act * hold, act * z[:, 0]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            ah[:, None, None] * step[None, :, :1],
                            az[:, None, None] * phi[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, features, ah / tau[0], az))
    return curve, rate


def _fit(inputs, dt, times, plateau, curves, rates):
    yc, yr = np.asarray(curves), np.asarray(rates)
    mc, mr = np.isfinite(yc), np.isfinite(yr)
    y = np.r_[yc[mc], yr[mr]]
    nc = mc.sum()
    row_weight = np.r_[np.ones(nc), np.full(mr.sum(), 2.)]
    yw = y * row_weight

    def matrix(t):
        curve, rate = _design(t, inputs, dt, times, plateau)
        return np.concatenate((curve[mc], rate[mr]), axis=0)

    def project(t, return_params=False):
        a = matrix(t) * row_weight[:, None]
        w = lstsq(a, yw, lapack_driver='gelsy', check_finite=False)[0]
        r = np.einsum('ij,j->i', a, w) - yw
        return (r @ r, np.r_[t, w]) if return_params else r

    lo, hi = np.log([0.025, 80.])
    starts = ((0.25, 2.5), (0.8, 7.), (2.5, 0.25), (7., 0.8))
    trials = [project(least_squares(project, np.log(t), bounds=(lo, hi),
                                   max_nfev=55).x, True) for t in starts]
    trials.sort(key=lambda fit: fit[0])
    limits = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])

    def residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - y

    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else 1.

    def scales(p):
        r = residual(p)
        return np.r_[np.full(nc, mad(r[:nc])), np.full(mr.sum(), mad(r[nc:]))]

    scale = scales(trials[0][1])
    soft = [least_squares(lambda p: residual(p) / scale, start, bounds=limits,
                          loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=85)
            for _, start in trials[:3]]
    p = min(soft, key=lambda f: f.cost).x
    # Suppress isolated high-amplitude strain readout errors after finding
    # the constitutive fit with the broader basin of the soft-L1 objective.
    scale = scales(p)
    fit = least_squares(lambda q: residual(q) / scale, p, bounds=limits,
                        loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=70)
    return fit.x[:2], fit.x[2:]


def solve_alpha_m97_c02_batch_07(dataset: dict) -> dict:
    inputs, strain0, dt, times, plateau = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, w = _fit(tuple(a[:n] for a in inputs), dt, times, plateau, curves, rates)
    curve, rate = _design(log_tau, tuple(a[n:] for a in inputs), dt, times, plateau)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': strain0[n:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_07_solver as s
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']
print('calibration',s._calibrate(d['calibration_load_cells']))
a=s.solve_alpha_m97_c02_batch_07(d)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,a[k])
PY
```
- **Container Exit Code**: `0` (`1130 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[ 0.7900147  -0.14000156]
 [ 0.73991367  0.20004232]
 [ 1.37001768 -0.17003273]
 [ 1.24003845 -0.12998439]]
estimated_taus [0.56571378 4.09186268]
hereditary_weights [ 0.67937634  0.54877258  0.13887757  0.15731563 -0.12890766  0.11724049
 -0.08842963]
regime_mode_weights [0.26756865 0.25112131]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e29b7ae8cbcdbb34 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e29b7ae8 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`34824 ms` LLM)
- **Rationale**: Refit without each lab and, separately, without the last three time points. Report strain and rate errors by damage state, including raw RMSE so readout spikes remain visible. If a slice fails, revisit calibration or the time law; if an identity check fails, fix the recurrence or calibration implementation.
- **Alternatives Weighed**: `Submit after the public schema pass` (rejected: The schema cannot detect poor transfer to a held-out lab or late time points, nor a mismatch between the curve and its initial derivative.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_07_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']; x,z0,dt,t,plateau=s._prepare(d); tr=d['train']; lab=tr['lab_id']; n=len(lab); y=tr['strain_curves_obs']-z0[:n,None]; r=tr['instant_rate_obs']; x=tuple(a[:n] for a in x); active=x[-1]>0
abs_curve=np.empty_like(y); abs_rate=np.empty_like(r); params=[]
for channel in np.unique(lab):
    train=lab!=channel; valid=~train
    for name,keep in [('all',np.ones(len(t),bool)),('early',np.arange(len(t))<len(t)-3)]:
        target=np.where(keep,y[train],np.nan)
        tau,w=s._fit(tuple(a[train] for a in x),dt,t,plateau,target,r[train]); params.append(np.r_[np.exp(tau),w]) if name=='all' else None
        a,b=s._design(tau,tuple(a[valid] for a in x),dt,t,plateau)
        ce=np.abs(a@w-y[valid]); re=np.abs(b@w-r[valid]); horizon=~keep if name=='early' else keep
        if name=='all': abs_curve[valid]=ce; abs_rate[valid]=re
        for state in (False,True):
            m=active[valid]==state
            if m.any(): print(name,'lab',channel,'active',int(state),'n',m.sum(),'curve med/p90',np.quantile(ce[m][:,horizon],[.5,.9]).round(4),'rate med/p90',np.quantile(re[m],[.5,.9]).round(4))
print('all-fold curve median per time',np.median(abs_curve,axis=0).round(4)); print('all-fold rate median',np.median(abs_rate).round(4)); print('all-fold raw RMSE curve/rate',np.sqrt(np.mean(abs_curve**2)).round(4),np.sqrt(np.mean(abs_rate**2)).round(4)); print('LOLO parameter ranges',np.min(params,axis=0).round(5),np.max(params,axis=0).round(5))
fit=s.solve_alpha_m97_c02_batch_07(d); w=np.r_[fit['hereditary_weights'],fit['regime_mode_weights']]; a,b=s._design(np.log(fit['estimated_taus']),x,dt,np.array([0.,1e-7]),plateau); assert np.allclose(a[:,0]@w,0.); error=np.max(np.abs(a[:,1]@w/1e-7-b@w)); assert error < 3e-6; print('initial derivative max error',error)
assert all(np.isfinite(v).all() for v in fit.values())
# All four measured sweeps must override the archive even with an isolated EMI spike.
cmd=np.linspace(-2,2,7); gain=np.array([.86,1.16,.92,1.29]); bias=np.array([-.1,.08,.2,-.17]); stress=gain[:,None]*cmd+bias[:,None]; stress[2,3]+=4.5
cal=s._calibrate(dict(lab_id=np.repeat(np.arange(4),len(cmd)),cmd_stress=np.tile(cmd,4),true_stress=stress.ravel())); assert np.allclose(cal,np.c_[gain,bias]); print('clean/EMI calibration and finite outputs pass')
PY
```
- **Container Exit Code**: `0` (`1730 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
all lab 0 active 0 n 15 curve med/p90 [0.0149 0.1263] rate med/p90 [0.0199 0.0411]
all lab 0 active 1 n 28 curve med/p90 [0.0144 0.1095] rate med/p90 [0.0139 0.0307]
early lab 0 active 0 n 15 curve med/p90 [0.0171 0.2748] rate med/p90 [0.0215 0.0415]
early lab 0 active 1 n 28 curve med/p90 [0.0195 0.0579] rate med/p90 [0.0149 0.0326]
all lab 1 active 0 n 12 curve med/p90 [0.0131 0.0309] rate med/p90 [0.0104 0.0193]
all lab 1 active 1 n 41 curve med/p90 [0.0121 0.0341] rate med/p90 [0.0114 0.0347]
early lab 1 active 0 n 12 curve med/p90 [0.0138 0.0283] rate med/p90 [0.0103 0.0195]
early lab 1 active 1 n 41 curve med/p90 [0.0135 0.0525] rate med/p90 [0.0119 0.0348]
all lab 2 active 0 n 14 curve med/p90 [0.0118 0.0788] rate med/p90 [0.0097 0.0322]
all lab 2 active 1 n 24 curve med/p90 [0.0141 0.1679] rate med/p90 [0.0237 0.046 ]
early lab 2 active 0 n 14 curve med/p90 [0.0168 0.8993] rate med/p90 [0.0089 0.0321]
early lab 2 active 1 n 24 curve med/p90 [0.0137 0.3164] rate med/p90 [0.0236 0.0462]
all lab 3 active 0 n 19 curve med/p90 [0.0162 0.1454] rate med/p90 [0.0125 0.0469]
all lab 3 active 1 n 27 curve med/p90 [0.0128 0.1293] rate med/p90 [0.0129 0.0388]
early lab 3 active 0 n 19 curve med/p90 [0.0194 0.1867] rate med/p90 [0.0136 0.046 ]
early lab 3 active 1 n 27 curve med/p90 [0.0129 0.3587] rate med/p90 [0.0128 0.0402]
all-fold curve median per time [0.0104 0.0137 0.0131 0.0123 0.0136 0.0124 0.0128 0.0156 0.0143 0.0152]
all-fold rate median 0.0134
all-fold raw RMSE curve/rate 1.0881 0.0448
LOLO parameter ranges [ 0.56534  4.04976  0.67766  0.54671  0.13872  0.15614 -0.12996  0.11651
 -0.08951  0.26711  0.25066] [ 0.56636  4.16147  0.68135  0.55136  0.13908  0.15808 -0.12773  0.11796
 -0.08807  0.26774  0.2517 ]
initial derivative max error 4.0453627647707435e-07
clean/EMI calibration and finite outputs pass

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e29b7ae8cbcdbb34 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e29b7ae8 (directory).
```

### Turn 9 — `submit_completed_task` (`18796 ms` LLM)
- **Rationale**: The exact saved candidate passed isolated-process schema and numerical checks. Leave-one-lab-out fits remained stable across damage states and late horizons, and the maximum initial-derivative discrepancy was 4.05e-7.
- **Alternatives Weighed**: `Add constitutive terms to lower raw strain RMSE` (rejected: The largest errors are isolated readout spikes, while held-out lab and late-horizon median errors remain low.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_07_solver.py': 'FILE:7615'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema, calibration under EMI, and initial-rate consistency checks passed. Across held-out lab and damage slices, the worst median absolute errors were 0.0195 for strain and 0.0237 for initial rate. Isolated readout spikes still dominate raw strain RMSE (1.0881).`

### Turn 10 — `execute_in_remote_docker` (`43395 ms` LLM)
- **Rationale**: Compare the two plausible structural alternatives on held-out labs and inspect signed errors by lab and imputation status for bias hidden by pooling. An independent step-by-step recurrence checks both predictions; a mismatch would require an implementation fix before submission.
- **Alternatives Weighed**: `Test relay modes, classification accuracy, or Fokker–Planck/HJB drift` (rejected: This dataset contains continuous creep and rate measurements, with no relay, class labels, or stochastic control equation. The relevant checks are within-lab residuals and constitutive identities.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d = pickle.load(open('public_alpha_m97_c02_batch_07.pkl', 'rb'))[0]['dataset']
tr = d['train']; lab = tr['lab_id']; n = len(lab)
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
base = s._design

def gated(log_tau, inputs, dt, times, plateau):
    curve, rate = base(log_tau, inputs, dt, times, plateau)
    inactive = inputs[-1] == 0
    ratio = np.divide(times, np.log1p(times), out=np.ones_like(times), where=times != 0)
    curve[inactive, :, 2:7] *= ratio[None, :, None]
    return curve, rate

cal = s._calibrate(d['calibration_load_cells']); older = cal.copy(); older[2] = (.85, .10)
for name, design, line in [('candidate', base, cal), ('linear at zero damage', gated, cal), ('old channel 2', base, older)]:
    s._design = design
    inputs, z0, dt, times, plateau = s._prepare(d, line)
    y = tr['strain_curves_obs'] - z0[:n, None]; r = tr['instant_rate_obs']
    ec, er = np.empty_like(y), np.empty_like(r)
    for c in range(4):
        keep = lab != c; ids = np.flatnonzero(keep); val = np.flatnonzero(~keep)
        p, w = s._fit(tuple(a[ids] for a in inputs), dt, times, plateau, y[ids], r[ids])
        a, b = design(p, tuple(a[val] for a in inputs), dt, times, plateau)
        ec[val], er[val] = a @ w - y[val], b @ w - r[val]
    def summary(mask):
        return np.round([np.median(abs(ec[mask])), np.median(abs(er[mask])), np.median(ec[mask]), np.median(er[mask])], 4).tolist()
    print(name, 'absolute then signed medians: strain, rate')
    for c in range(4):
        for state in (False, True):
            mask = (lab == c) & ((inputs[-1][:n] > 0) == state)
            if mask.any(): print(c, int(state), int(mask.sum()), summary(mask))
    if name == 'candidate':
        print('complete versus imputed, by lab:', [(c, int(sum((lab == c) & missing)), summary((lab == c) & ~missing), summary((lab == c) & missing)) for c in range(4) if ((lab == c) & missing).any()])

s._design = base
fit = s.solve_alpha_m97_c02_batch_07(d)
inputs, z0, dt, t, _ = s._prepare(d)
past, hold, features, act = (a[n:] for a in inputs)
tau, hw, rw = (fit[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights'))
z = np.zeros((len(hold), 2))
for j, step in enumerate(dt):
    decay = np.exp(-step / tau)
    z = decay * z + (1 - decay) * past[:, j, None]
gap = hold[:, None] - z
prony = (gap * hw[:2]) @ (1 - np.exp(-t[:, None] / tau)).T
flow = features @ hw[2:] + rw[1] * act * z[:, 0]
quick = rw[0] * act * hold
pred = z0[n:, None] + prony + quick[:, None] * (1 - np.exp(-t / tau[0])) + flow[:, None] * np.log1p(t)
rate = (gap * hw[:2]) @ (1 / tau) + quick / tau[0] + flow
for name, value in [('predicted_strain', pred), ('initial_rate_pred', rate)]:
    err = np.max(np.abs(value - fit[name])); print(name, 'explicit recurrence max difference:', err); assert err < 1e-11
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2969 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
candidate absolute then signed medians: strain, rate
0 0 15 [0.0149, 0.0199, 0.0027, 0.0006]
0 1 28 [0.0144, 0.0139, 0.0013, -0.0029]
1 0 12 [0.0131, 0.0104, 0.0012, 0.0053]
1 1 41 [0.0121, 0.0114, 0.0, -0.002]
2 0 14 [0.0118, 0.0097, -0.0006, 0.0022]
2 1 24 [0.0141, 0.0237, -0.0023, -0.0012]
3 0 19 [0.0162, 0.0125, -0.001, 0.0101]
3 1 27 [0.0128, 0.0129, -0.0022, 0.0007]
complete versus imputed, by lab: [(0, 4, [0.0127, 0.0136, 0.0031, -0.0015], [0.1094, 0.0503, -0.1, -0.0483]), (1, 1, [0.0123, 0.011, 0.0008, 0.0002], [0.0228, 0.0389, -0.0228, -0.0389]), (2, 5, [0.0115, 0.0125, -0.0004, 0.0041], [0.0853, 0.0677, -0.062, -0.0371]), (3, 5, [0.0125, 0.0121, -0.0017, 0.0071], [0.0989, 0.0902, -0.0074, 0.011])]
linear at zero damage absolute then signed medians: strain, rate
0 0 15 [0.0415, 0.0324, -0.0072, -0.0226]
0 1 28 [0.0979, 0.0784, -0.0609, -0.0603]
1 0 12 [0.1599, 0.0522, 0.1126, -0.0522]
1 1 41 [0.0496, 0.0489, -0.0171, -0.01]
2 0 14 [0.1384, 0.0311, 0.0682, -0.0244]
2 1 24 [0.0475, 0.0353, -0.0404, -0.0252]
3 0 19 [0.1842, 0.0265, 0.0579, -0.0185]
3 1 27 [0.0361, 0.0283, -0.0322, -0.021]
old channel 2 absolute then signed medians: strain, rate
0 0 15 [0.0933, 0.1505, 0.0559, 0.0944]
0 1 28 [0.0393, 0.0728, -0.0179, -0.0413]
1 0 12 [0.0436, 0.0625, -0.0136, 0.0196]
1 1 41 [0.0363, 0.0374, -0.018, -0.0043]
2 0 32 [0.4475, 0.7492, 0.269, 0.4132]
2 1 6 [0.6293, 1.2568, 0.2115, 0.507]
3 0 19 [0.1127, 0.1679, -0.0321, -0.0454]
3 1 27 [0.0483, 0.0628, -0.0411, 0.006]
predicted_strain explicit recurrence max difference: 1.3322676295501878e-15
initial_rate_pred explicit recurrence max difference: 8.881784197001252e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e29b7ae8cbcdbb34 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e29b7ae8 (directory).
```

### Turn 11 — `submit_completed_task` (`22889 ms` LLM)
- **Rationale**: The saved candidate passed isolated-process validation. Held-out measurements favored the implemented time law and calibration, and the independent recurrence confirmed both prediction outputs.
- **Alternatives Weighed**: `Use linear flow at zero damage` (rejected: In leave-one-lab-out validation, it raised the worst lab–damage median strain error from 0.0162 to 0.1842 on the same evaluation horizons.); `Use the older channel-2 calibration` (rejected: It raised median absolute initial-rate error above 0.74 in both held-out channel-2 damage slices.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_07_solver.py': 'FILE:7615'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema passed, and both predictions matched an independent step-by-step recurrence to within 1.4e-15. Across held-out lab and damage slices, the worst median absolute errors were 0.0195 for strain and 0.0237 for initial rate. Errors were higher for coupons with imputed features, and isolated readout spikes dominated raw strain RMSE.`

### Turn 12 — `execute_in_remote_docker` (`43127 ms` LLM)
- **Rationale**: The previous audit found the largest median errors among coupons with missing features. Test whether excluding those coupons from fitting improves transfer without changing the required imputation at prediction time. Separately, test fixed interaction and saturation terms fitted inside each lab fold; retain the specified law unless the challenger reveals a material, reproducible residual pattern.
- **Alternatives Weighed**: `Repeat the public schema and recurrence checks on the unchanged candidate` (rejected: Both already passed in an isolated process; another run would not test whether an omitted drift term or imputed training row biases the fit.)

```bash
cd /holdout_task && cat > /tmp/rheo07_margin.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c02_batch_07.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; lab = tr['lab_id']; n = len(lab)
a, z0, dt, times, plateau = s._prepare(d)
a = tuple(v[:n] for v in a)
y = tr['strain_curves_obs'] - z0[:n, None]; r = tr['instant_rate_obs']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
# Fixed alternatives for an omitted terminal-drift interaction, all with
# the same derivative at zero as their curve amplitude.
x = a[2][:, 1:]; hold, act = a[1], a[3]
q = np.c_[x[:,0]*x[:,1], x[:,2]*x[:,3], x[:,0]*x[:,1]*x[:,2],
          (x[:,0]-x[:,1])/(1+(x[:,2]+x[:,3])**2),
          act*hold/(1+hold**2)]
phi = np.log1p(times) if plateau else times

def augment(c, v, ids):
    return np.concatenate((c, q[ids,None,:]*phi[None,:,None]),axis=2), np.c_[v,q[ids]]

for mode in ('baseline', 'complete-fit', 'drift-challenger'):
    es, er, params = np.empty_like(y), np.empty_like(r), []
    for ch in np.unique(lab):
        u = np.flatnonzero((lab != ch) & (~missing if mode == 'complete-fit' else True))
        v = np.flatnonzero(lab == ch)
        p,w = s._fit(tuple(z[u] for z in a),dt,times,plateau,y[u],r[u])
        params.append(np.r_[np.exp(p),w])
        cv,rv = s._design(p,tuple(z[v] for z in a),dt,times,plateau)
        if mode == 'drift-challenger':
            # Profile only the extra drift coefficients, so the audit asks
            # whether they improve an otherwise identical constitutive fit.
            cu,ru = s._design(p,tuple(z[u] for z in a),dt,times,plateau)
            from scipy.optimize import least_squares
            ec,ee = cu@w-y[u],ru@w-r[u]
            sc,sr = [max(1e-6, 1.4826*np.median(abs(e-np.median(e)))) for e in (ec,ee)]
            bu,du = augment(cu,ru,u); cv,rv = augment(cv,rv,v)
            def residual(k):
                return np.r_[((bu[:,:,9:]@k+ec)/sc).ravel(), (du[:,9:]@k+ee)/sr]
            w = np.r_[w,least_squares(residual,np.zeros(q.shape[1]),loss='cauchy',f_scale=2.5).x]
        es[v] = abs(cv@w-y[v]); er[v] = abs(rv@w-r[v])
    def worst(subsets):
        return np.round(np.max([np.r_[np.median(es[m]),np.median(er[m]),np.median(es[m,-3:])] for m in subsets if m.any()], axis=0),5)
    print(mode, 'worst curve/rate/late medians:',
          'lab x damage',worst([(lab==i)&((act>0)==j) for i in np.unique(lab) for j in (0,1)]),
          'imputed lab',worst([(lab==i)&missing for i in np.unique(lab)]),
          'complete lab',worst([(lab==i)&~missing for i in np.unique(lab)]),
          'tau span',np.round(np.ptp(params,axis=0)[:2],5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo07_margin.py
```
- **Container Exit Code**: `0` (`2326 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline worst curve/rate/late medians: lab x damage [0.01621 0.02367 0.01899] imputed lab [0.10941 0.09021 0.13381] complete lab [0.01267 0.01361 0.01626] tau span [0.00102 0.11172]
complete-fit worst curve/rate/late medians: lab x damage [0.01537 0.02407 0.01838] imputed lab [0.11228 0.08945 0.13703] complete lab [0.0127  0.01441 0.01528] tau span [0.0005  0.06975]
drift-challenger worst curve/rate/late medians: lab x damage [0.01575 0.02364 0.01874] imputed lab [0.10994 0.09025 0.13443] complete lab [0.01281 0.013   0.01676] tau span [0.00102 0.11172]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e29b7ae8cbcdbb34 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e29b7ae8 (directory).
```

### Turn 13 — `submit_completed_task` (`18084 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process schema and recurrence checks. The final held-out comparison found no consistent improvement on the imputed-feature bottleneck, so no edit or repeat verification is warranted.
- **Alternatives Weighed**: `Fit only on complete-feature coupons` (rejected: It narrowed the fold-to-fold range of the slow relaxation time, but raised worst imputed-slice late-horizon median strain error from 0.13381 to 0.13703.); `Add interaction and saturation terms to terminal drift` (rejected: The small gains on lab–damage slices did not transfer to the worst imputed slices, and the added terms fall outside the specified constitutive law.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema passed, and both predictions matched an independent recurrence to within 1.4e-15. The worst held-out lab–damage median absolute errors were 0.0195 for strain, including late-horizon extrapolation, and 0.0237 for initial rate. Errors were higher on coupons with imputed features; neither tested alternative improved that slice consistently.`

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
