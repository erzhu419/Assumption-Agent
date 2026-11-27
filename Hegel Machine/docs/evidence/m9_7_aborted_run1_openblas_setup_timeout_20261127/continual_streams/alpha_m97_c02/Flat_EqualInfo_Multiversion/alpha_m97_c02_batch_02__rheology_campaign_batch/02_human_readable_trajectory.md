# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_02__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_02` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `314,910` (`304,637` / `96,128` / `10,273`) |
| **Total Episode Wall Time (s)** | `363.58s` |

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

### Turn 1 — `execute_in_remote_docker` (`16747 ms` LLM)
- **Rationale**: Inspect the actual fields, split layout, and prior solver to catch schema and alignment mismatches before fitting; look up reusable primitives for alignment and relaxation.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_01_solver.py; cat /holdout_task/public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' || true; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_alpha_m97_c02_batch_02.pkl','rb') as f: d=pickle.load(f)
def show(x, depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*depth,k,type(v).__name__,end=' ')
   if isinstance(v,dict): print(); show(v,depth+2)
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype, v.reshape(-1)[:6])
   elif isinstance(v,(list,tuple)): print(len(v),str(v[:1])[:260])
   else: print(str(v)[:220])
show(d)
PY
```
- **Container Exit Code**: `0` (`611 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"type": "ndarray"
            },
            "te_true_rate0": {
              "dtype": "float64",
              "shape": [
                72
              ],
              "type": "ndarray"
            },
            "theta": {
              "dtype": "float64",
              "shape": [
                7
              ],
              "type": "ndarray"
            },
            "vp_theta": {
              "dtype": "float64",
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
}autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`15637 ms` LLM)
- **Rationale**: The first output was truncated before the dataset field listing. Inspect the actual field names, calibration layout, and prior implementation to catch schema mismatches and identify what can be reused without carrying over plateau-specific terms.

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c02_batch_01_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -22; python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_02.pkl','rb'))[0]
d=item['dataset']
print('wrapper keys:',list(item))
for k,v in d.items():
 if isinstance(v,dict):
  print(k,[(a,np.shape(b),str(np.asarray(b).dtype)) for a,b in v.items()]);
  if k in ('calibration_load_cells','ref_coupons'):
   for a,b in v.items(): print(a,np.asarray(b)[:12])
 else: print(k,np.shape(v),str(v)[:200])
PY
```
- **Container Exit Code**: `0` (`607 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
features[n:], dt, times, act[n:], plateau)
    return {'estimated_taus': np.exp(taus), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
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
wrapper keys: ['dataset', 'gt']
past_dt (18,) [0.4591 0.5943 0.5823 0.434  0.3268 0.5912 0.4711 0.2337 0.5456 0.3562
 0.7073 0.6748 0.5176 0.4162 0.2682 0.7841 0.6458 0.5548]
eval_times (10,) [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
crit_stress () 2.65
yield_stress () 2.65
shear_banding_index () 0.12
calibration_load_cells [('lab_id', (40,), 'int64'), ('cmd_stress', (40,), 'float64'), ('true_stress', (40,), 'float64')]
lab_id [0 0 0 0 0 0 0 0 0 0 1 1]
cmd_stress [ 2.117655 -1.165749 -1.673814 -1.249019 -1.48004   2.119091  1.208208
  0.301724  0.137891 -1.560518 -0.944846 -1.229132]
true_stress [ 1.53290266 -1.06097068 -1.46234156 -1.12658138 -1.3093763   1.53422972
  0.81457744  0.09823977 -0.0311721  -1.3727534  -1.07762012 -1.45007044]
train [('lab_id', (180,), 'int64'), ('stratum', (180,), 'int64'), ('coupon_features_raw', (180, 4), 'float64'), ('ref_coupons', (180, 32), 'float64'), ('past_stress_cmd', (180, 18), 'float64'), ('stress_hold_cmd', (180,), 'float64'), ('strain_0', (180,), 'float64'), ('aux_chamber_telemetry', (180, 3), 'float64'), ('strain_curves_obs', (180, 10), 'float64'), ('instant_rate_obs', (180,), 'float64')]
test [('lab_id', (72,), 'int64'), ('stratum', (72,), 'int64'), ('coupon_features_raw', (72, 4), 'float64'), ('ref_coupons', (72, 32), 'float64'), ('past_stress_cmd', (72, 18), 'float64'), ('stress_hold_cmd', (72,), 'float64'), ('strain_0', (72,), 'float64'), ('aux_chamber_telemetry', (72, 3), 'float64')]
```

### Turn 3 — `execute_in_remote_docker` (`25184 ms` LLM)
- **Rationale**: Recover the alignment and fitting code omitted from the previous output. Check calibration residuals, lab-stratum coverage, and missing observations to determine whether the fit needs archive fallback, cell-balanced imputation, or masked residuals.
- **Alternatives Weighed**: `Reuse the plateau solver unchanged` (rejected: The current regime specifies linear terminal flow and zero regime-mode weights; only the shared calibration, alignment, and relaxation are reusable.)

```bash
sed -n '46,140p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_01_solver.py; python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_02.pkl','rb'))[0]['dataset']
a=D['calibration_load_cells']; k=a['lab_id']; x=a['cmd_stress']; y=a['true_stress']
for j in range(4):
 m=k==j; p=np.polyfit(x[m],y[m],1); print('calibration',j,p,'max residual',np.max(np.abs(y[m]-np.polyval(p,x[m]))))
for s in ['train','test']:
 d=D[s]; print(s,'cells',np.unique(np.c_[d['lab_id'],d['stratum']],axis=0,return_counts=True),'feature missing',np.isnan(d['coupon_features_raw']).sum(0),'panel missing',np.isnan(d['ref_coupons']).sum())
for k in ['strain_curves_obs','instant_rate_obs']:
 a=D['train'][k]; print(k,'missing',np.isnan(a).sum(),'quantiles',np.nanquantile(a,[0,.01,.5,.99,1]))
PY
```
- **Container Exit Code**: `0` (`700 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ra = np.column_stack((-drive/taus[0], act*dz[:, 0]))
    return curve, rate, np.concatenate((dc, dc_extra), axis=-1), np.column_stack((dr, dr_extra))


def _fit(past, hold, features, dt, times, act, curves, rates, plateau, scales=(0.025, 0.025)):
    target = np.r_[curves.ravel()/scales[0], rates/scales[1]]

    def design(log_taus):
        c, r, dc, dr = _basis(log_taus, past, hold, features, dt, times, act, plateau)
        return (np.vstack((c.reshape(-1, 9)/scales[0], r/scales[1])),
                np.vstack((dc.reshape(-1, 4)/scales[0], dr/scales[1])))

    def residual(params):
        matrix, _ = design(params[:2])
        return np.einsum('ij,j->i', matrix, params[2:]) - target

    def jacobian(params):
        matrix, d = design(params[:2])
        fast = d[:, 0]*params[2] + d[:, 2]*params[9] + d[:, 3]*params[10]
        return np.column_stack((fast, d[:, 1]*params[3], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(9, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(9, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        matrix, _ = design(log_taus)
        weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', x_scale='jac', max_nfev=130)
calibration 0 [ 0.79001467 -0.14000165] max residual 0.00016360402143522634
calibration 1 [1.30997914 0.15995982] max residual 0.0001755180569942283
calibration 2 [0.84998578 0.09998369] max residual 0.00024219014072818235
calibration 3 [ 1.24003837 -0.12998453] max residual 0.00015674836549228743
train cells (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([13, 15, 12,  5, 24, 22,  7, 11, 21, 12, 23, 15])) feature missing [2 3 5 4] panel missing 237
test cells (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([ 9,  7,  8,  7, 14,  1,  2,  6,  6,  8,  4])) feature missing [3 1 0 1] panel missing 94
strain_curves_obs missing 0 quantiles [-5.16184183 -3.08189397  0.29534937  4.0854329  32.91495445]
instant_rate_obs missing 0 quantiles [-4.21436229 -3.81940612  0.19707913  4.38538827  4.65861021]
```

### Turn 4 — `execute_in_remote_docker` (`45083 ms` LLM)
- **Rationale**: Reuse the verified alignment convention and implement the specified terminal flow without plateau or damage regressors. The isolated schema run checks importability, output shapes, and finiteness; prediction accuracy still needs lab-level validation.
- **Alternatives Weighed**: `Ordinary least squares on all strain and rate observations` (rejected: The public strain range includes a large positive extreme. Fit with bounded influence so potentially corrupted observations cannot dominate the shared relaxation parameters.)

```bash
set -e
cd /holdout_task
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust nonlinear least squares viscoelastic Prony relaxation' | head -8
python3 - <<'PY'
from pathlib import Path
prior = Path('/tmp/hegel_continual_archive/alpha_m97_c02_batch_01_solver.py').read_text()
shared = prior[prior.index('import json'):prior.index('def _activity')]
Path('alpha_m97_c02_batch_02_solver.py').write_text('"""Calibrated two-branch creep model for an amorphous melt."""\n' + shared)
PY
cat >> alpha_m97_c02_batch_02_solver.py <<'PY'
def _basis(log_taus, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_taus)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # A history step contributes its uptake times its decay over subsequent steps.
    memory = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap/tau, features))
    if not derivatives:
        return curve, rate
    dz = past @ (a*memory - q*ea*eq)
    dc = -dz[:, None, :]*step - gap[:, None, :]*r*er
    dr = (-dz - gap)/tau
    return curve, rate, dc, dr


def _fit(past, hold, features, age, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]
    observed = np.isfinite(target)
    target = target[observed]
    args = past, hold, features, age, dt, times

    def design(log_taus, derivatives=False):
        c, r, *jac = _basis(log_taus, *args, derivatives=derivatives)
        matrix = np.vstack((c.reshape(-1, 7), r))[observed]
        if derivatives:
            dc, dr = jac
            return matrix, np.vstack((dc.reshape(-1, 2), dr))[observed]
        return matrix

    def residual(params):
        return np.einsum('ij,j->i', design(params[:2]), params[2:]) - target

    def jacobian(params):
        matrix, diff = design(params[:2], derivatives=True)
        return np.column_stack((diff * params[2:4], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0),
                 (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        weights = np.linalg.lstsq(design(log_taus), target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', f_scale=0.025,
                            x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x.copy()
    if params[0] > params[1]:
        params[[0, 1, 2, 3]] = params[[1, 0, 3, 2]]
    return params[:2], params[2:]


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_taus, weights = _fit(past[:n], hold[:n], features[:n], age, dt, times, curves, rates)
    curve, rate = _basis(log_taus, past[n:], hold[n:], features[n:], age, dt, times)
    return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`921 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_02_solver.py']`
- **Stdout (tail)**:
```text
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8c6b38cafc2af0d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8c6b38ca (directory).
```

### Turn 5 — `execute_in_remote_docker` (`35527 ms` LLM)
- **Rationale**: Hold out each lab's response measurements and score strain and initial rate independently by lab, stratum, and horizon. Systematic residuals would motivate revisiting alignment or fitting; isolated spikes would support retaining the robust loss. A separate step-by-step recurrence check would catch history indexing or rate errors.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: Shape and finiteness checks cannot detect lab-specific prediction bias or a mistake in the compressed history formula.)

```bash
set -e
cat > /tmp/check_m97_rheology.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_02_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
past, hold, features, base = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
tr = d['train']
y, r = tr['strain_curves_obs'] - base[:len(tr['lab_id']), None], tr['instant_rate_obs']
lab, strata = tr['lab_id'], tr['stratum']
n = len(lab)
oof_y, oof_r = np.empty_like(y), np.empty_like(r)

def fit(mask):
    return m._fit(past[:n][mask], hold[:n][mask], features[:n][mask], age, dt, t, y[mask], r[mask])

def predict(mask, log_tau, w):
    c, b = m._basis(log_tau, past[:n][mask], hold[:n][mask], features[:n][mask], age, dt, t)
    return c @ w, b @ w

for j in range(4):
    mask = lab == j
    tau, w = fit(~mask)
    oof_y[mask], oof_r[mask] = predict(mask, tau, w)
    print('held lab', j, 'tau', np.round(np.exp(tau), 5), 'weights', np.round(w, 5))

e = oof_y - y
v = oof_r - r
for name, keys in [('lab', lab), ('stratum', strata), ('lab/stratum', lab*10 + strata)]:
    for value in np.unique(keys):
        k = keys == value
        z = e[k]
        print(name, value, 'n', k.sum(), 'curve median abs / p90 / RMSE', np.round([np.median(abs(z)), np.quantile(abs(z), .9), np.sqrt(np.mean(z*z))], 5), 'rate RMSE', round(np.sqrt(np.mean(v[k]**2)), 5))
print('by horizon, median abs', np.round(np.median(abs(e), axis=0), 5))
print('by horizon, count above 0.15', (abs(e) > .15).sum(0))
print('curve abs > .15:', np.round(e[abs(e) > .15], 3), 'rate p90/max abs', np.round(np.quantile(abs(v), [.9, 1]), 5))

out = m.solve_alpha_m97_c02_batch_02(d)
assert np.array_equal(out['regime_mode_weights'], np.zeros(2))
assert np.all(np.diff(out['estimated_taus']) > 0)
# Compute the history step by step independently of the compressed basis.
tau, w = out['estimated_taus'], out['hereditary_weights']
f = past[n:]
z = np.zeros((len(f), 2))
for j, h in enumerate(dt):
    a = np.exp(-h/tau)
    z = a*z + (1-a)*f[:, j, None]
gap = hold[n:, None] - z
drift = features[n:] @ w[2:]
curve = base[n:, None] + (gap*w[:2]) @ (1-np.exp(-t[:, None]/tau)).T + drift[:, None]*t
rate = (gap/tau) @ w[:2] + drift
np.testing.assert_allclose(out['predicted_strain'], curve, atol=1e-12, rtol=0)
np.testing.assert_allclose(out['initial_rate_pred'], rate, atol=1e-12, rtol=0)
print('Full fit', tau, w, '; recurrence and rate agree to 1e-12')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_m97_rheology.py
```
- **Container Exit Code**: `0` (`3152 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.01191 0.05553 1.13926] rate RMSE 0.02673
lab/stratum 20 n 7 curve median abs / p90 / RMSE [0.01596 0.05337 0.04695] rate RMSE 0.01837
lab/stratum 21 n 11 curve median abs / p90 / RMSE [0.01814 0.06339 0.11469] rate RMSE 0.01446
lab/stratum 22 n 21 curve median abs / p90 / RMSE [0.01549 0.04536 0.37101] rate RMSE 0.01984
lab/stratum 30 n 12 curve median abs / p90 / RMSE [0.01728 0.29762 0.33168] rate RMSE 0.04728
lab/stratum 31 n 23 curve median abs / p90 / RMSE [0.01343 0.0701  0.49438] rate RMSE 0.02634
lab/stratum 32 n 15 curve median abs / p90 / RMSE [0.01528 0.08089 0.2513 ] rate RMSE 0.0231
by horizon, median abs [0.01127 0.01154 0.0113  0.0127  0.01636 0.01557 0.01514 0.01717 0.01704
 0.01961]
by horizon, count above 0.15 [ 4  9 12 15 13 21 17 23 21 21]
curve abs > .15: [  0.637  -0.166  -0.22   -0.263  -0.266  -0.293  -0.221 -16.259   0.581
  -0.352  -0.2     0.883  -0.43   -8.822  -2.268  -0.438  -0.779  -0.802
  -0.194  -0.226   1.622   0.592  -1.867   1.566   1.755   0.227   0.939
  -0.226   4.583   0.94    0.611  -0.201 -32.931   0.444   2.219   0.567
  -0.282  -0.722   2.67    0.411   2.722   0.381  -0.453  -0.967  -1.156
   0.281  -0.152  -0.182  -0.254   0.196   0.272   0.33    0.388   0.446
   0.494   0.974   0.16    0.187   0.214   0.234   0.279  -0.302   0.37
  -0.939   0.447   0.19    0.202  -1.318   0.301   0.309   0.448  -0.351
  -0.176  -0.188  -0.248  -0.269   1.607   0.4     1.772  -3.604  -0.2
  -0.23   -0.293  -0.354  -0.372  -0.398  -0.499   2.534  -1.173  -1.509
  -0.315  -0.397  -0.477  -0.559  -0.626  -0.738  -0.809  -0.628  -0.19
   1.052  -3.155   0.152   0.181   0.227   0.246   0.278   0.166   0.225
   0.275   0.337   0.429   0.52    0.558   0.623   0.167   0.239   0.311
   0.369   0.436   0.485   0.546   0.593  -4.372   0.406   0.984  -0.433
  -0.935  -0.366   0.228   0.173  -0.33    0.531   0.547   0.484   0.791
  -0.183  -0.204  -0.327  -0.379  -0.42   -0.488  -0.586  -0.663   1.973
  -0.433  -1.195  -0.417  -0.389  -1.644   2.983  -0.475   0.751   7.129
  -0.297   1.689   0.384] rate p90/max abs [0.03552 0.15785]
Full fit [0.5400331  3.93447469] [ 0.67887463  0.55597004  0.14045892  0.15632026 -0.12750569  0.11672176
 -0.08819245] ; recurrence and rate agree to 1e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8c6b38cafc2af0d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8c6b38ca (directory).
```

### Turn 6 — `execute_in_remote_docker` (`24621 ms` LLM)
- **Rationale**: Several held-out residuals grow roughly linearly with time. Check whether they concentrate on imputed coupons or persist on complete rows: the latter would warrant revisiting alignment or terminal drift. Compare the unchanged artifact with any available public reference outputs to distinguish prediction error from measurement contamination.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, pickle, runpy
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    v = runpy.run_path('/tmp/check_m97_rheology.py')
d, m, out = (v[k] for k in ('d', 'm', 'out'))
e, r, lab = (v[k] for k in ('e', 'v', 'lab'))
missing = np.isnan(d['train']['coupon_features_raw']).any(axis=1)
t = d['eval_times']
slopes = np.median(e/t, axis=1)
for name,k in [('complete', ~missing), ('imputed', missing)]:
    print(name, 'rows', k.sum(), 'strain p90', np.quantile(abs(e[k]), .9), 'rate RMSE', np.sqrt(np.mean(r[k]**2)), 'absolute median-slope quantiles', np.quantile(abs(slopes[k]), [.5,.9,1]))
    print('largest drift rows:', [(int(i), int(lab[i]), round(slopes[i],4), round(r[i],4)) for i in np.flatnonzero(k & (abs(slopes) > .025))])
past=v['past']; print('max |calibrated history stress|', np.max(abs(past)), 'threshold', d['crit_stress'])
with open('/holdout_task/public_alpha_m97_c02_batch_02.pkl','rb') as f:
    items=pickle.load(f)
print('public episodes', len(items))
for i,item in enumerate(items):
    gt=item['gt']
    print('reference fields', i, [(k,np.shape(x)) for k,x in gt.items()])
    pred=out if i==0 else m.solve_alpha_m97_c02_batch_02(item['dataset'])
    for k in pred.keys() & gt.keys():
        a,b=np.asarray(pred[k]),np.asarray(gt[k]);
        if a.shape == b.shape:
            z=a-b
            print(k, 'RMSE',float(np.sqrt(np.mean(z*z))), 'max abs',float(np.max(abs(z))))
PY
```
- **Container Exit Code**: `0` (`3632 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete rows 166 strain p90 0.04189852749016745 rate RMSE 0.0185978304804099 absolute median-slope quantiles [0.00259607 0.00780738 0.01747893]
largest drift rows: []
imputed rows 14 strain p90 0.522937310179979 rate RMSE 0.08249512583081052 absolute median-slope quantiles [0.06009409 0.12466157 0.15822559]
largest drift rows: [(3, 3, -0.0556, -0.0786), (80, 3, -0.0325, -0.0324), (82, 1, 0.0937, 0.0815), (83, 0, 0.0603, 0.0624), (87, 1, 0.0598, 0.0199), (92, 3, -0.0482, -0.0523), (103, 3, -0.0943, -0.08), (107, 1, -0.1582, -0.1578), (113, 3, 0.0532, 0.0682), (116, 3, 0.1198, 0.1128), (118, 0, 0.1222, 0.0915), (151, 1, -0.1257, -0.1293)]
max |calibrated history stress| 2.067083443066902 threshold 2.65
public episodes 1
reference fields 0 [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8c6b38cafc2af0d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8c6b38ca (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30309 ms` LLM)
- **Rationale**: Compare the unchanged solver with the correctly mapped public reference fields to separate measurement spikes from clean prediction error. Test a complete-row fit on held-out labs to see whether imputed training rows bias the shared parameters; change the solver only if the held-out comparison shows a meaningful gain.
- **Alternatives Weighed**: `Replace cell-balanced imputation with a learned predictor` (rejected: The contract prescribes cell-balanced imputation; first test whether missing training features bias the shared fit without changing inference.)

```bash
set -e
sed -n '1,51p' /holdout_task/alpha_m97_c02_batch_02_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, pickle, runpy
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    v = runpy.run_path('/tmp/check_m97_rheology.py')
with open('/holdout_task/public_alpha_m97_c02_batch_02.pkl', 'rb') as f:
    gt = pickle.load(f)[0]['gt']
d, m, out = (v[k] for k in ('d', 'm', 'out'))
for p, g in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta')]:
    print(p, 'reference', gt[g], 'error', out[p]-gt[g])
for p,g in [('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    z=out[p]-gt[g]
    test=d['test']; mis=np.isnan(test['coupon_features_raw']).any(axis=1)
    for name,k in [('all',np.ones(len(mis),bool)), ('complete',~mis), ('imputed',mis)]+[(f'lab {i}',test['lab_id']==i) for i in range(4)]:
        print(p,name,'n',k.sum(),'RMSE',np.sqrt(np.mean(z[k]**2)), 'max abs',np.max(np.abs(z[k])))
for p,g in [('past','te_hist_true'),('hold','te_hold_true')]:
    print(p,'calibration max abs error',np.max(np.abs(v[p][v['n']:]-gt[g])))
# Change the fitting labels only: preserve the prescribed imputation at inference.
n, lab = v['n'], v['lab']; miss=np.isnan(d['train']['coupon_features_raw']).any(axis=1)
y,r=v['y'],v['r']; es,rs=np.empty_like(y),np.empty_like(r)
for j in np.unique(lab):
    k=lab==j
    tau,w=v['fit'](~k & ~miss)
    es[k],rs[k]=v['predict'](k,tau,w)
es-=y;rs-=r
for name,k in [('all',np.ones(n,bool)),('complete',~miss),('imputed',miss)]:
    for label,e,s in [('baseline',v['e'],v['v']),('complete-only fit',es,rs)]:
        print(name,label,'strain median/p90',np.quantile(abs(e[k]),[.5,.9]),'rate RMSE',np.sqrt(np.mean(s[k]**2)))
for label,e,s in [('baseline',v['e'],v['v']),('complete-only fit',es,rs)]:
    print(label,'worst lab strain median and rate RMSE',max(np.median(abs(e[lab==i])) for i in range(4)),max(np.sqrt(np.mean(s[lab==i]**2)) for i in range(4)))
PY
```
- **Container Exit Code**: `0` (`1273 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ror [3.30971071e-05 3.44746877e-02]
hereditary_weights reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] error [-0.00112537  0.00597004  0.00045892 -0.00367974  0.00249431 -0.00327824
  0.00180755]
regime_mode_weights reference [0. 0.] error [0. 0.]
predicted_strain all n 72 RMSE 0.14740413644086256 max abs 1.4425678486447038
predicted_strain complete n 67 RMSE 0.01061620263270784 max abs 0.06498856303095413
predicted_strain imputed n 5 RMSE 0.5580077719642279 max abs 1.4425678486447038
predicted_strain lab 0 n 24 RMSE 0.08505500753699759 max abs 0.6568858102762212
predicted_strain lab 1 n 22 RMSE 0.003075369528017217 max abs 0.014415943775687223
predicted_strain lab 2 n 8 RMSE 0.16005983438705268 max abs 0.7383991819804758
predicted_strain lab 3 n 18 RMSE 0.25664813113569657 max abs 1.4425678486447038
initial_rate_pred all n 72 RMSE 0.046156066618613004 max abs 0.27871301901834045
initial_rate_pred complete n 67 RMSE 0.004182763763607679 max abs 0.013221023382864905
initial_rate_pred imputed n 5 RMSE 0.17447941977178202 max abs 0.27871301901834045
initial_rate_pred lab 0 n 24 RMSE 0.026237462568401773 max abs 0.12168418810798665
initial_rate_pred lab 1 n 22 RMSE 0.002544213675757558 max abs 0.006066253647551534
initial_rate_pred lab 2 n 8 RMSE 0.04913063335754513 max abs 0.1375742751274034
initial_rate_pred lab 3 n 18 RMSE 0.08076470377926084 max abs 0.27871301901834045
past calibration max abs error 8.34430669018893e-05
hold calibration max abs error 6.818820113618784e-05
all baseline strain median/p90 [0.0144737  0.09418938] rate RMSE 0.02912541888737741
all complete-only fit strain median/p90 [0.0146186  0.09484124] rate RMSE 0.02914443985734168
complete baseline strain median/p90 [0.01331068 0.04189853] rate RMSE 0.0185978304804099
complete complete-only fit strain median/p90 [0.01332929 0.04195603] rate RMSE 0.018633225720826478
imputed baseline strain median/p90 [0.17176051 0.52293731] rate RMSE 0.08249512583081052
imputed complete-only fit strain median/p90 [0.17176625 0.52284194] rate RMSE 0.08248679058793701
baseline worst lab strain median and rate RMSE 0.01637004140201248 0.035898637677876905
complete-only fit worst lab strain median and rate RMSE 0.016150486661845265 0.036011271030534094

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8c6b38cafc2af0d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8c6b38ca (directory).
```

### Turn 8 — `execute_in_remote_docker` (`24708 ms` LLM)
- **Rationale**: Confirm that the inherited imputation follows the stated cell-balanced rule, then exercise the explicitly supported sparse and EMI calibration paths. If an endpoint spike defeats the current detector, fix calibration without changing the clean-channel fit.
- **Alternatives Weighed**: `Fit only complete training rows` (rejected: Leave-one-lab-out results showed no meaningful improvement in either strain or initial rate.)

```bash
set -e
sed -n '52,97p' /holdout_task/alpha_m97_c02_batch_02_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c02_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c02_batch_02.pkl','rb'))[0]['dataset']; a=d['calibration_load_cells']; k=np.asarray(a['lab_id']); x=np.asarray(a['cmd_stress']); y=np.asarray(a['true_stress']); fit=m._calibrate(a)
for j in range(4):
 ix=np.flatnonzero(k==j); archive=m._archived_calibration(j); print('lab',j,'rows',len(ix),'archive minus current',archive-fit[j]);
 for size in (1,4,5,len(ix)):
  if size>len(ix): continue
  keep=np.r_[np.flatnonzero(k!=j),ix[:size]]
  for spike in ([None] if size==1 else range(size)):
   noisy=y.copy()
   if spike is not None: noisy[ix[spike]]+=5
   est=m._calibrate(dict(lab_id=k[keep],cmd_stress=x[keep],true_stress=noisy[keep]))[j]
   err=min(np.max(abs(est-fit[j])),np.max(abs(est-archive)))
   if err>0.01: print('missed fallback/clean fit',j,size,spike,'coefficient error',round(float(err),5))
PY
```
- **Container Exit Code**: `0` (`800 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_taus)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # A history step contributes its uptake times its decay over subsequent steps.
    memory = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap/tau, features))
    if not derivatives:
        return curve, rate
    dz = past @ (a*memory - q*ea*eq)
lab 0 rows 10 archive minus current [3.25033519e-05 1.34751299e-05]
missed fallback/clean fit 0 4 0 coefficient error 1.94645
lab 1 rows 10 archive minus current [-1.68585155e-05  9.62425598e-05]
missed fallback/clean fit 1 4 2 coefficient error 4.5705
lab 2 rows 10 archive minus current [-2.43444825e-05  5.31280759e-05]
missed fallback/clean fit 2 4 0 coefficient error 2.33269
lab 3 rows 10 archive minus current [-3.67727450e-05  2.26719608e-05]
missed fallback/clean fit 3 4 2 coefficient error 2.39858

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8c6b38cafc2af0d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8c6b38ca (directory).
```

### Turn 9 — `execute_in_remote_docker` (`35779 ms` LLM)
- **Rationale**: Fix the confirmed EMI failure without changing the clean fit. Check both spike signs, sparse archive fallback, and preference for four or more clean in-batch points; rerun the isolated schema, recurrence, and held-lab checks on the edited artifact.
- **Alternatives Weighed**: `Use archived coefficients to screen every channel` (rejected: Clean in-batch calibration must be fit directly without consulting the archive.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_02_solver.py')
s = p.read_text()
a = s.index('        if i.size:')
b = s.index('        archive =', a)
s = s[:a] + '''        if i.size:
            # Choose the line with the most near-collinear points before
            # testing for EMI. A spike can skew the median slope of four rows.
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            offsets = y[i] - slopes*x[i]
            residuals = np.abs(y - slopes[:, None]*x - offsets[:, None])
            near = residuals <= 0.1
            support = near.sum(axis=1)
            candidates = np.flatnonzero(support == support.max())
            best = candidates[np.argmin(np.sum(np.where(near[candidates], residuals[candidates], 0.), axis=1))]
            gain, bias = np.polyfit(x[near[best]], y[near[best]], 1)
            clean = np.abs(y - (gain*x + bias)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, runpy
import numpy as np
with contextlib.redirect_stdout(io.StringIO()) as log:
    v = runpy.run_path('/tmp/check_m97_rheology.py')
m, d, out = (v[k] for k in ('m', 'd', 'out'))
a = d['calibration_load_cells']
k, x, y = (np.asarray(a[name]) for name in ('lab_id', 'cmd_stress', 'true_stress'))
count = 0
for ch in range(4):
    ix = np.flatnonzero(k==ch)
    archive = m._archived_calibration(ch)
    for size in (0, 1, 2, 3, 4, 5, len(ix)):
        rows = ix[:size]
        keep = np.r_[np.flatnonzero(k!=ch), rows]
        for spike in [None] + (list(rows) if size >= 4 else []):
            for delta in ([0] if spike is None else [-5, 5]):
                obs = y.copy()
                if spike is not None:
                    obs[spike] += delta
                clean = rows[rows!=spike] if spike is not None else rows
                expected = archive if size < 2 or (spike is not None and len(clean)<4) else np.polyfit(x[clean], y[clean], 1)
                result = m._calibrate(dict(lab_id=k[keep], cmd_stress=x[keep], true_stress=obs[keep]))[ch]
                np.testing.assert_allclose(result, expected, atol=1e-11, rtol=0, err_msg=str((ch,size,spike,delta)))
                count += 1
print('Calibration cases passed:',count)
print(log.getvalue().splitlines()[-1])
print('Worst held-out lab strain median absolute error:',max(np.median(abs(v['e'][v['lab']==j])) for j in range(4)))
print('Worst held-out lab initial-rate RMSE:',max(np.sqrt(np.mean(v['v'][v['lab']==j]**2)) for j in range(4)))
PY
```
- **Container Exit Code**: `0` (`1552 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Calibration cases passed: 180
 -0.08819245] ; recurrence and rate agree to 1e-12
Worst held-out lab strain median absolute error: 0.01637004140201248
Worst held-out lab initial-rate RMSE: 0.035898637677876905

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2b30b2fe735e6a09 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2b30b2fe (directory).
```

### Turn 10 — `submit_completed_task` (`16960 ms` LLM)
- **Rationale**: The final calibration change fixed the demonstrated four-point EMI failure without degrading held-lab metrics. The exact artifact passed the isolated schema check, and its strain and rate predictions matched an independent history recurrence to 1e-12.
- **Alternatives Weighed**: `Fit only complete training coupons` (rejected: Leave-one-lab-out validation showed no meaningful improvement, and the inference-time imputation is prescribed.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_02_solver.py': 'FILE:7067'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_02_solver.py`. The public schema and all 180 calibration stress cases passed. Worst held-out lab median absolute strain error was 0.0164, and worst initial-rate RMSE was 0.0359. Missing coupon features remained the main source of error.`

### Turn 11 — `execute_in_remote_docker` (`31613 ms` LLM)
- **Rationale**: Fit residual corrections only on the non-held labs to test whether pair and triple interactions or rational drift terms expose repeatable lack of fit. Use complete coupons to isolate structural bias from prescribed imputation. A consistent held-lab gain would warrant investigating alignment or model assumptions; no gain would support retaining the verified artifact.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck/HJB operators, or balanced accuracy` (rejected: This task has continuous rheology outputs and no relay state, stochastic control operator, or class labels. Held-lab signed drift and rate residuals are the relevant nuisance diagnostics.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, runpy
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
    v = runpy.run_path('/tmp/check_m97_rheology.py')
d, lab, n, t = (v[k] for k in ('d', 'lab', 'n', 't'))
x = v['features'][:n, 1:]
complete = ~np.isnan(d['train']['coupon_features_raw']).any(axis=1)
y, r = v['y'], v['r']
# Pairwise linear sums and differences are already in the feature span.
pairs = list(combinations(range(4), 2))
triples = list(combinations(range(4), 3))
q = np.column_stack([x[:, i]*x[:, j] for i,j in pairs] + [np.prod(x[:, ijk], axis=1) for ijk in triples] + [x[:, i]/(1+.25*(x[:, i]-x[:, j])**2)-x[:, i] for i,j in pairs])
shapes = (t[:, None]/(1+(t[:, None]/np.array([.5, 4.]))**2))-t[:, None]
rows = []
for j in np.unique(lab):
    tr, te = (lab != j) & complete, (lab == j) & complete
    tau,w = v['fit'](tr)
    a,b = v['predict'](complete,tau,w)
    e = np.zeros_like(y); f = np.zeros_like(r)
    e[complete], f[complete] = a-y[complete], b-r[complete]
    c = np.concatenate((q[:,None,:]*t[None,:,None], np.broadcast_to(shapes,(n,*shapes.shape))),axis=2)
    h = np.column_stack((q,np.zeros((n,shapes.shape[1]))))
    matrix = np.vstack((c[tr].reshape(-1,h.shape[1]),h[tr])); err = np.r_[e[tr].ravel(),f[tr]]
    p = least_squares(lambda p: matrix@p+err, np.zeros(h.shape[1]), jac=lambda p: matrix.copy(), loss='cauchy', f_scale=.025).x
    ec,fc = e[te]+c[te]@p, f[te]+h[te]@p
    row = [np.median(abs(e[te])),np.median(abs(ec)),np.sqrt(np.mean(f[te]**2)),np.sqrt(np.mean(fc**2))]
    rows.append(row)
    slope = np.median(e[te][:,t>0]/t[t>0],axis=1)
    print('held lab',int(j),'complete n',int(te.sum()),'strain median baseline/augmented; rate RMSE baseline/augmented',np.round(row,6),'median signed slope',round(float(np.median(slope)),6),'median signed rate error',round(float(np.median(f[te])),6))
print('Worst complete held-lab metrics',np.round(np.max(rows,axis=0),6))
PY
```
- **Container Exit Code**: `0` (`1162 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 complete n 38 strain median baseline/augmented; rate RMSE baseline/augmented [0.0146   0.015029 0.017006 0.01698 ] median signed slope 0.000433 median signed rate error -0.001587
held lab 1 complete n 45 strain median baseline/augmented; rate RMSE baseline/augmented [0.010328 0.010572 0.019207 0.019134] median signed slope -0.000958 median signed rate error 0.000155
held lab 2 complete n 39 strain median baseline/augmented; rate RMSE baseline/augmented [0.01615  0.016353 0.018253 0.018255] median signed slope 0.002365 median signed rate error 0.005186
held lab 3 complete n 44 strain median baseline/augmented; rate RMSE baseline/augmented [0.012938 0.013285 0.019685 0.019408] median signed slope -0.002339 median signed rate error 0.000567
Worst complete held-lab metrics [0.01615  0.016353 0.019685 0.019408]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2b30b2fe735e6a09 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2b30b2fe (directory).
```

### Turn 12 — `submit_completed_task` (`15277 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and calibration checks, and its strain and initial-rate predictions matched an independent history recurrence to 1e-12. The follow-up audit found no material improvement over the specified linear-flow model and no consistent signed drift across labs.
- **Alternatives Weighed**: `Add interaction and rational drift terms` (rejected: On complete coupons, the extra terms worsened median strain error in every held-out lab and reduced worst-lab initial-rate RMSE by only 0.00028.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_02_solver.py`. The public schema and all 180 calibration stress cases passed. Across held-out labs, the worst median absolute strain error was 0.0164 and the worst initial-rate RMSE was 0.0359. Missing coupon features remained the main limitation.`

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
