# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m100_c03_s00__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_viscoelastic_chamber_hereditary_mechanics` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_alpha` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 16` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `242,723` (`233,638` / `83,200` / `9,085`) |
| **Total Episode Wall Time (s)** | `286.43s` |

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

### Turn 1 — `execute_in_remote_docker` (`17522 ms` LLM)
- **Rationale**: Inspect the schema and compact telemetry first: a different chamber law changes prediction, while different calibration or reference-panel layouts change preprocessing. No broad data transfer is needed.
- **Alternatives Weighed**: `Assume the benchtop law also governs deployment` (rejected: The specification makes that a fallback only when campaign evidence is unavailable.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json; cat /holdout_task/validate_public_schema.py; for f in /tmp/hegel_continual_archive/{archive_manifest,archive_apparatus_log,pre_llm_gate_diagnostic}.json; do if [ -f "$f" ]; then echo "=== $f"; wc -c "$f"; head -c 12000 "$f"; echo; fi; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference-panel alignment imputation' 2>&1; python3 - <<'PY'
import pickle
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
def walk(x,p='',d=0):
 if isinstance(x,dict) and d<3:
  for k,v in x.items(): walk(v,p+'/'+str(k),d+1)
 else:
  print(p,type(x).__name__,getattr(x,'shape',None),getattr(x,'dtype',''),repr(x)[:160])
walk(x)
PY
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array (72,) float64 array([-0.00837735, -0.05632574,  0.06192332,  0.09138664, -0.13442071,
        0.15401749, -0.13237955,  0.09635441,  0.18524911,  0.18712373,
        0.067963
/dataset/train/strain_curves_obs ndarray (72, 12) float64 array([[0.61111047, 1.29083669, 1.77329201, 2.16409387, 2.48952661,
        2.74403508, 2.98658385, 3.22959366, 3.43924676, 3.63959271,
        3.83929492, 4.01
/dataset/train/instant_rate_obs ndarray (72,) float64 array([2.92769077, 2.99666742, 4.23994621, 2.62829153, 2.01162246,
       3.85637708, 2.57917501, 4.48996112, 3.55651061, 2.55581202,
       2.26812494, 2.69408
/dataset/train/aux_chamber_telemetry ndarray (72, 4) float64 array([[ 0.44774936,  0.55580422, -1.41684465,  0.60919955],
       [-0.29130554,  0.67896929,  0.14142758,  0.50635156],
       [-1.03425339, -0.27094256,  0.0
/dataset/test/lab_id ndarray (36,) int64 array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 2, 2, 2, 2,
       3, 0, 2, 3, 2, 2, 0, 2, 0, 2, 0, 2, 3, 2])
/dataset/test/stratum ndarray (36,) int64 array([0, 2, 0, 0, 1, 0, 2, 2, 0, 1, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 2,
       0, 2, 2, 1, 2, 2, 1, 1, 2, 0, 2, 1, 2, 2])
/dataset/test/coupon_features_raw ndarray (36, 4) float64 array([[-2.22425617, -0.92244105, -1.62060168, -1.42301021],
       [ 0.75427657, -0.17559954,  0.27580474, -0.06580103],
       [ 2.14888592, -1.38202235, -0.6
/dataset/test/ref_coupons ndarray (36, 8) float64 array([[-2.08531119, -1.95390101, -0.05255352,  0.02446336, -1.04813652,
         1.07756948,  0.16088953,  0.30062589],
       [ 0.88002425,  0.83516547,  0.18
/dataset/test/past_channel_raw ndarray (36, 6, 4) float64 array([[[-1.82689499e-01,  3.75764347e-01,  8.18424873e-01,
          7.49021651e-02],
        [ 3.02377703e-01,  4.20241495e-02,  7.16155909e-01,
          5.1
/dataset/test/hold_channel_raw ndarray (36, 4) float64 array([[0.36096641, 1.26058125, 0.62656863, 0.42893802],
       [0.36380612, 0.90306243, 1.11023018, 0.35991518],
       [1.11583072, 1.01160646, 0.45307795, 0.
/dataset/test/strain_0 ndarray (36,) float64 array([ 0.09738064,  0.08877973,  0.22295686, -0.11961088, -0.12518303,
       -0.0014805 ,  0.06023398, -0.04213011, -0.00721783,  0.12194816,
        0.135534
/dataset/test/aux_chamber_telemetry ndarray (36, 4) float64 array([[ 0.15130083,  0.21403194,  0.3151847 ,  1.27454971],
       [ 0.44816288,  0.29027383,  0.61974511,  0.64175607],
       [-0.83217486,  0.93747065, -0.5
/campaign_diagnostics NoneType None  None
```

### Turn 2 — `execute_in_remote_docker` (`18655 ms` LLM)
- **Rationale**: The prior output was truncated. Read the compact apparatus log directly to recover the chamber law, and measure affine calibration residuals to decide whether the inverse needs nonlinear terms.
- **Alternatives Weighed**: `Fit an affine bridge without checking its residuals` (rejected: If the sweep residuals reveal a nonlinear bridge, the affine inverse would bias both the fitted relaxation and the deployment prediction.)

```bash
cat /tmp/hegel_continual_archive/archive_apparatus_log.json; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle, numpy as np, importlib.util
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; print('keys',list(D),'scipy',bool(importlib.util.find_spec('scipy')))
print('times',D['eval_times'],'history',D['past_dt'])
for k in ('train','test'):
 d=D[k]; r=d['ref_coupons']; print(k,'keys',list(d),'missing',np.isnan(d['coupon_features_raw']).sum(0)); print('reference labs',[(int(l),r[d['lab_id']==l].mean(0).round(3).tolist(),r[d['lab_id']==l].std(0).round(3).tolist()) for l in np.unique(d['lab_id'])])
c=D['calibration_load_cells']; x=c['bridge_raw']; y=c['true_channel_loads']; A=np.c_[np.ones(len(x)),x]; print('affine inverse residual std',np.std(y-A@np.linalg.lstsq(A,y,rcond=None)[0],axis=0),'truth std',y.std(0)); print('sweeps',[(k,v.shape) for k,v in c['single_channel_sweeps'].items()])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference-panel' 2>&1
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
uting_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`18116 ms` LLM)
- **Rationale**: The previous output truncated the apparatus metadata. Read it without the full trace: explicit controls can determine the chamber correction, whereas unlabeled telemetry summaries alone cannot identify a constitutive law.
- **Alternatives Weighed**: `Add nonlinear bridge terms` (rejected: The affine residual is approximately 0.01% of channel load variation; there is no evidence that higher-order terms would improve the calibration.)

```bash
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for row in json.load(open(p)):
 print({k:v for k,v in row.items() if k!='facility_telemetry_trace'})
 a=np.asarray(row['facility_telemetry_trace']); print('trace',a.shape,'mean',a.mean(0).round(4),'std',a.std(0).round(4),'last',a[-3:].round(4))
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=x['dataset']; print('dataset keys',list(d)); print('times',d['eval_times'],'past_dt',d['past_dt']);
for s in ('train','test'):
 a=d[s]['aux_chamber_telemetry']; print(s,'aux mean/std',a.mean(0).round(3),a.std(0).round(3)); print('aux correlation',np.corrcoef(a.T).round(2))
PY
sed -n '1,160p' /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`382 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
out1 = solve_fn(ds_swap)
    def difference(key):
        before = np.asarray(out0[key], dtype=float)
        after = np.asarray(out1[key], dtype=float)
        if before.size and before.shape == after.shape and np.all(np.isfinite(before)) and np.all(np.isfinite(after)):
            return float(np.max(np.abs(after - before)))
        return None

    strain_diff = difference("predicted_strain")
    rate_diff = (difference("initial_rate_pred") if "initial_rate_pred" in out0 and "initial_rate_pred" in out1
                 else (None if "initial_rate_pred" in out0 or "initial_rate_pred" in out1 else 0.0))
    return {
        "valid": bool(swapped_rows and strain_diff is not None and rate_diff is not None
                      and max(strain_diff, rate_diff) <= float(tol)),
        "swapped_ref_rows_per_split": int(swapped_rows),
        "ref_swap_strain_max_diff": round(strain_diff, 10) if strain_diff is not None else None,
        "ref_swap_rate_max_diff": round(rate_diff, 10) if rate_diff is not None else None,
        "tol": float(tol),
    }

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    ds = pickle.load(fh)[0]["dataset"]
out = mod.solve_batch(copy.deepcopy(ds))
n_te = len(ds["test"]["lab_id"])
k_t = len(ds["eval_times"])
errs = []
coupon = None
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        try:
            coupon = verify_alpha_pooled_ref_coupon_invariance(mod.solve_batch, ds, base_out=out)
            if not coupon["valid"]:
                errs.append("pooled_reference_coupon_invariance_failed")
        except (KeyError, TypeError, ValueError, IndexError):
            errs.append("invalid_output_on_reference_exchange")
print(json.dumps({"schema_valid": not errs, "errors": errs, "pooled_reference_check": coupon}, allow_nan=False))
sys.exit(1 if errs else 0)
```

### Turn 4 — `execute_in_remote_docker` (`58506 ms` LLM)
- **Rationale**: Build the specified two-branch recurrence and its analytic initial rate from the same calibrated loads. The schema test will detect split-dependent reference alignment, and the residual summary will reveal systematic preprocessing errors before lab-held-out validation.
- **Alternatives Weighed**: `Infer chamber coefficients from the facility initialization trace` (rejected: The trace has no documented channel-to-coefficient mapping, prior verified solver, or paired strain and rate observations.); `Use ordinary least squares for all observations` (rejected: The specified heavy-tailed strain outliers can bias relaxation times and branch weights.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel 2>&1 | head -25
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrate the bridge and fit a two-branch hereditary material law."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x = np.vstack([s['coupon_features_raw'] for s in (train, test)]).astype(float)
    ref = np.vstack([s['ref_coupons'] for s in (train, test)])
    labs = np.concatenate([s['lab_id'] for s in (train, test)])
    strata = np.concatenate([s['stratum'] for s in (train, test)])
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale
    _, cell, count = np.unique(np.column_stack((labs, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    w = 1.0 / count[cell, None]
    missing = np.isnan(x)
    denominator = np.sum(w * ~missing, axis=0)
    fill = np.divide(np.nansum(w * x, axis=0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)
    return np.where(missing, fill, x)


def _bridge(cal):
    raw = cal['bridge_raw']
    # The sum functional of the full affine four-channel calibration.
    return np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))),
                           cal['true_channel_loads'], rcond=None)[0].sum(axis=1)


def _loads(split, bridge):
    return tuple(split[k] @ bridge[:-1] + bridge[-1]
                 for k in ('past_channel_raw', 'hold_channel_raw'))


def _design(taus, x, past, hold, dt, times):
    decay = np.exp(-dt[:, None] / taus)
    z = np.zeros((len(hold), 2))
    for s, a in enumerate(decay):
        z = a * z + (-np.expm1(-dt[s] / taus)) * past[:, s, None]
    gap = hold[:, None] - z
    drift = np.column_stack((np.ones(len(x)), x))
    relax = -np.expm1(-times[:, None] / taus)
    curve = np.concatenate((gap[:, None, :] * relax,
                            drift[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _fit(x, past, hold, dt, times, strain, initial_rate):
    n_curve = strain.size
    y = np.r_[strain.ravel(), initial_rate]

    def matrix(log_tau):
        a, b = _design(np.exp(log_tau), x, past, hold, dt, times)
        return np.vstack((a.reshape(-1, 7), b))

    def residual(p):
        return (matrix(p[:2]) @ p[2:] - y) / scale

    def mad(r):
        return max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-7)

    grid = np.array([0.14, 0.35, 0.8, 1.7, 3.5, 7.0, 15.0])
    starts = []
    for i, t0 in enumerate(grid):
        for t1 in grid[i + 1:]:
            log_tau = np.log([t0, t1])
            a = matrix(log_tau)
            theta = np.linalg.lstsq(a, y, rcond=None)[0]
            r = a @ theta - y
            starts.append((np.mean(np.abs(r)), np.r_[log_tau, theta], r))
    starts.sort(key=lambda row: row[0])
    r = starts[0][2]
    scale = np.r_[np.full(n_curve, mad(r[:n_curve])),
                  np.full(len(initial_rate), mad(r[n_curve:]))]
    bounds = (np.r_[np.full(2, np.log(0.050001)), np.full(7, -np.inf)],
              np.r_[np.full(2, np.log(24.999999)), np.full(7, np.inf)])
    best = min((least_squares(residual, p, bounds=bounds, loss='soft_l1',
                              x_scale='jac', max_nfev=150)
                for _, p, _ in starts[:4]), key=lambda obj: obj.cost)
    r = residual(best.x) * scale
    scale = 2.5 * np.r_[np.full(n_curve, mad(r[:n_curve])),
                        np.full(len(initial_rate), mad(r[n_curve:]))]
    p = least_squares(residual, best.x, bounds=bounds, loss='cauchy',
                      x_scale='jac', max_nfev=150).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    x = _features(train, test)
    n = len(train['lab_id'])
    bridge = _bridge(dataset['calibration_load_cells'])
    past, hold = _loads(train, bridge)
    taus, theta = _fit(x[:n], past, hold, dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The blind batch has no labeled chamber data or documented mapping from
    # the initialization telemetry to a deployment-specific correction.
    past, hold = _loads(test, bridge)
    curve, rate = _design(taus, x[n:], past, hold, dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as s
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; o=s.solve_batch(d); print('tau',o['estimated_taus'],'theta',o['hereditary_weights']); tr=d['train']; x=s._features(tr,d['test'])[:len(tr['lab_id'])]; p,h=s._loads(tr,s._bridge(d['calibration_load_cells'])); C,R=s._design(o['estimated_taus'],x,p,h,d['past_dt'],d['eval_times']); e=C@o['hereditary_weights']+tr['strain_0'][:,None]-tr['strain_curves_obs']; er=R@o['hereditary_weights']-tr['instant_rate_obs']; print('train strain absolute error 50/90/99/max',np.quantile(abs(e),[.5,.9,.99,1])); print('train rate absolute error 50/90/99/max',np.quantile(abs(er),[.5,.9,.99,1])); print('correlation of strain and rate residuals with train aux', np.corrcoef(np.column_stack((np.median(e,axis=1),er,tr['aux_chamber_telemetry'])).T)[:2,2:].round(3))
PY
```
- **Container Exit Code**: `0` (`906 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
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
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 23, "ref_swap_strain_max_diff": 2.7e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}
tau [0.65153034 3.7144084 ] theta [ 0.75550768  0.66173058  0.31027026  0.19105444 -0.16080958  0.11005292
 -0.15789183]
train strain absolute error 50/90/99/max [0.00854508 0.02116885 0.28050805 0.29861818]
train rate absolute error 50/90/99/max [0.00619152 0.01303606 0.01930521 0.02003174]
correlation of strain and rate residuals with train aux [[-0.053  0.07  -0.069  0.116]
 [ 0.062 -0.187 -0.109 -0.044]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1b8a9722399259a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1b8a9722 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32131 ms` LLM)
- **Rationale**: Hold out each lab's response labels while preserving the specified joint, unlabeled reference alignment. A large late-horizon error would prompt a check of the slow branch; an error concentrated in imputed rows would prompt a check of the cell weights. The isolated schema and derivative checks verify the exact candidate's output contract.
- **Alternatives Weighed**: `Treat the deployment derivative check as chamber accuracy validation` (rejected: It verifies agreement between the two output columns, not agreement with unobserved deployment measurements.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x = s._features(tr, te)[:len(tr['lab_id'])]
p, h = s._loads(tr, s._bridge(d['calibration_load_cells']))
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
e, er = np.empty_like(y), np.empty_like(r)
o = s.solve_batch(d)
for lab in np.unique(tr['lab_id']):
    holdout = tr['lab_id'] == lab
    keep = ~holdout
    tau, theta = s._fit(x[keep], p[keep], h[keep], dt, t, y[keep], r[keep])
    assert .05 < tau[0] < tau[1] < 25
    a, b = s._design(tau, x[holdout], p[holdout], h[holdout], dt, t)
    e[holdout], er[holdout] = abs(a @ theta - y[holdout]), abs(b @ theta - r[holdout])
    print('held-out lab', lab, 'n', int(holdout.sum()), 'taus', tau.round(4), 'strain median/90th', np.quantile(e[holdout], [.5,.9]).round(5), 'rate median/90th', np.quantile(er[holdout], [.5,.9]).round(5))
for name, ids in [('early', slice(None,4)), ('middle', slice(4,8)), ('late', slice(8,None))]:
    print(name, 'worst lab strain 90th', round(max(np.quantile(e[tr['lab_id']==lab, ids], .9) for lab in np.unique(tr['lab_id'])), 5))
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for name, m in [('imputed', missing), ('complete', ~missing)]:
    if m.any():
        print(name, 'n', int(m.sum()), 'strain/rate 90th', round(np.quantile(e[m], .9), 5), round(np.quantile(er[m], .9), 5))
# Exact t=0 and forward derivative on the actual deployment histories.
pt, ht = s._loads(te, s._bridge(d['calibration_load_cells']))
a, b = s._design(o['estimated_taus'], s._features(tr, te)[len(x):], pt, ht, dt, np.array([0.,1e-7]))
y0, y1 = (a @ o['hereditary_weights']).T
rate_err = np.max(abs((y1-y0)/1e-7 - o['initial_rate_pred']))
print('deployment baseline max error', float(np.max(abs(y0))), 'initial-rate derivative max error', float(rate_err))
assert np.max(abs(y0)) == 0 and rate_err < 1e-5
PY
```
- **Container Exit Code**: `0` (`1187 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 23, "ref_swap_strain_max_diff": 2.7e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}
held-out lab 0 n 16 taus [0.6517 3.7265] strain median/90th [0.00866 0.02405] rate median/90th [0.00582 0.01246]
held-out lab 1 n 20 taus [0.654  3.7771] strain median/90th [0.00902 0.01963] rate median/90th [0.00563 0.01159]
held-out lab 2 n 16 taus [0.6511 3.7121] strain median/90th [0.00839 0.02109] rate median/90th [0.00422 0.01459]
held-out lab 3 n 20 taus [0.6493 3.6428] strain median/90th [0.00828 0.02128] rate median/90th [0.00908 0.01476]
early worst lab strain 90th 0.02637
middle worst lab strain 90th 0.02371
late worst lab strain 90th 0.02444
imputed n 19 strain/rate 90th 0.0228 0.01365
complete n 53 strain/rate 90th 0.02016 0.01443
deployment baseline max error 0.0 initial-rate derivative max error 3.6990140728221377e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1b8a9722399259a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1b8a9722 (directory).
```

### Turn 6 — `submit_completed_task` (`14617 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated Python verification. Leave-one-lab-out fits produced stable relaxation times, and the predicted deployment initial rate agreed with the strain derivative to within 3.70e-7.
- **Alternatives Weighed**: `Add a chamber-specific correction from the initialization trace` (rejected: The trace has no documented mapping to constitutive parameters, and no paired chamber responses are available to validate a correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4689'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and pooled-reference invariance checks passed. Deployment predictions use the benchtop law because the available telemetry does not identify a chamber-specific correction.`

### Turn 7 — `execute_in_remote_docker` (`39722 ms` LLM)
- **Rationale**: Cross-validated bridge sweeps will show whether quadratic terms improve calibration. Held-out lab and horizon slices will show whether pressure differences, rational terms, products, or lab offsets explain residual drift better than the specified law. Those tests can justify a benchtop change, but cannot validate a chamber correction without paired chamber responses.
- **Alternatives Weighed**: `Run classification, relay-mode, or stochastic-control checks` (rejected: The data contain continuous strain and rate outcomes from a deterministic hereditary recurrence. Held-out lab residuals and strain derivative consistency address the applicable nuisance and physical checks.)

```bash
cat > /tmp/audit_hereditary.py <<'PY'
import importlib.util, json, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; raw,load=c['bridge_raw'],c['true_channel_loads']; truth=load.sum(axis=1)
def cal(a,degree):
    z=[np.ones(len(a)),*a.T]
    if degree==2: z.extend(a[:,i]*a[:,j] for i in range(4) for j in range(i,4))
    return np.column_stack(z)
idx=np.arange(len(raw)); order=np.argsort(truth); fold=np.empty(len(raw),int); fold[order]=idx%5
for degree in (1,2):
    A=cal(raw,degree); e=np.zeros(len(raw))
    for f in range(5):
        m=fold==f; e[m]=A[m]@np.linalg.lstsq(A[~m], truth[~m],rcond=None)[0]-truth[m]
    print('bridge degree',degree,'RMSE',np.sqrt(np.mean(e**2)),'max load-quartile p90',max(np.quantile(abs(e[ids]),.9) for ids in np.array_split(order,4)))
bridge=s._bridge(c); p,h=s._loads(tr,bridge); x=s._features(tr,te)[:len(h)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; labs=tr['lab_id']; aux=tr['aux_chamber_telemetry']; e={k:[] for k in ('specified','lab_intercepts','pressure_grammar','auxiliary_grammar')}
for l in np.unique(labs):
    m=labs!=l; tau,theta=s._fit(x[m],p[m],h[m],dt,t,y[m],r[m]); C,R=s._design(tau,x,p,h,dt,t); ey=y-C@theta; er=r-R@theta
    delta=h-p[:,-1]; asym=aux[:,0]-aux[:,1]; cross=aux[:,2]*aux[:,3]
    # Additive drift challengers; every correction contributes t to strain and one to rate.
    Z={'specified':np.empty((len(h),0)), 'lab_intercepts':np.eye(len(np.unique(labs)))[np.searchsorted(np.unique(labs),labs)], 'pressure_grammar':np.c_[delta,delta/(1+h*h),delta*h*p[:,-1]], 'auxiliary_grammar':np.c_[aux,asym/(1+aux[:,2]**2),cross*asym]}
    for key,z in Z.items():
        a=np.zeros(len(h))
        if z.shape[1]:
            z=z/(1e-8+np.sqrt(np.mean(z[m]**2,axis=0)))
            def res(b):
                v=z[m]@b
                return np.r_[((v[:,None]*t-ey[m])/.012).ravel(),(v-er[m])/.008]
            a=z@least_squares(res,np.zeros(z.shape[1]),loss='cauchy',f_scale=2.5).x
        sy=abs(ey[~m]-a[~m,None]*t); sr=abs(er[~m]-a[~m]); e[key].append([max(np.quantile(sy[:,j],.9) for j in np.array_split(np.arange(len(t)),3)),np.quantile(sr,.9),np.median(sy),np.median(sr)])
    print('lab',l,'signed median residual strain/rate',np.median(ey[~m]),np.median(er[~m]))
for key,v in e.items(): print(key,'worst lab-horizon p90 strain / worst lab rate p90',np.max(v,axis=0)[:2].round(6),'mean lab median',np.mean(v,axis=0)[2:].round(6))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    a=np.asarray(row['facility_telemetry_trace']); print('facility fields',list(row),'samples',len(a),'offsets',a.mean(0).round(4),'centered RMS',a.std(0).round(4))
print('verification coupons available',pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['campaign_diagnostics'] is not None)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_hereditary.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1148 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bridge degree 1 RMSE 0.00019609962448106246 max load-quartile p90 0.00029471726073984086
bridge degree 2 RMSE 0.0002138014053113014 max load-quartile p90 0.0003165195178226777
lab 0 signed median residual strain/rate -0.00027256599941971604 0.002589986579407677
lab 1 signed median residual strain/rate -0.0007514256995176383 -0.0020152608154473395
lab 2 signed median residual strain/rate 0.0009466805931234723 0.0021117799695538064
lab 3 signed median residual strain/rate -0.00020625159784581903 -0.00425740309249667
specified worst lab-horizon p90 strain / worst lab rate p90 [0.026369 0.014761] mean lab median [0.008589 0.006186]
lab_intercepts worst lab-horizon p90 strain / worst lab rate p90 [0.026369 0.014761] mean lab median [0.008589 0.006186]
pressure_grammar worst lab-horizon p90 strain / worst lab rate p90 [0.026455 0.014832] mean lab median [0.008696 0.006217]
auxiliary_grammar worst lab-horizon p90 strain / worst lab rate p90 [0.02615  0.014935] mean lab median [0.008708 0.006281]
facility fields ['completed_campaign_index', 'completed_campaign_short_name', 'facility_telemetry_trace'] samples 24 offsets [-0.0035 -0.0009 -0.0107 -0.3938 -0.0071 -0.0125 -0.3481  0.0077] centered RMS [0.6053 0.6115 0.598  0.0652 0.5956 0.6066 0.0766 0.6019]
verification coupons available False
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 23, "ref_swap_strain_max_diff": 2.7e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1b8a9722399259a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1b8a9722 (directory).
```

### Turn 8 — `submit_completed_task` (`14529 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema verification and leave-one-lab-out stress tests. The predicted deployment initial rate agreed with the derivative of predicted strain to within 3.70e-7.
- **Alternatives Weighed**: `Quadratic bridge calibration` (rejected: Held-out calibration RMSE rose from 0.000196 to 0.000214.); `Additional pressure or telemetry drift terms` (rejected: Neither improved both held-out strain and rate, and the unlabeled facility trace does not identify a deployment correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4689'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and pooled-reference invariance checks passed. Deployment predictions use the benchtop law because the available telemetry does not identify a chamber-specific correction.`

### Turn 9 — `execute_in_remote_docker` (`33715 ms` LLM)
- **Rationale**: Interleaved calibration folds can miss bias at the ends of the load range; contiguous folds would expose it. Comparing robust cutoffs on held-out labs tests whether outliers materially bias either response or destabilize the relaxation times. Neither check requires deployment labels or changes to the submitted file.
- **Alternatives Weighed**: `Repeat the pressure and telemetry drift comparison or add relay-mode checks` (rejected: The drift comparison is already settled on measured responses. The dataset has no relay state, class labels, or stochastic-control equation.)

```bash
cat > /tmp/audit_robust_hereditary.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = d['calibration_load_cells']
x = s._features(tr, te)[:len(tr['lab_id'])]
p, h = s._loads(tr, s._bridge(cal))
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
dt, t = d['past_dt'], d['eval_times']
labs = tr['lab_id']
real_ls = s.least_squares
metrics = {}
for cutoff in (1.5, 2.5, 4.0, None):
    def ls(fun, x0, **kw):
        if kw['loss'] == 'cauchy':
            if cutoff is None:
                return real_ls(fun, x0, loss='linear', bounds=kw['bounds'], x_scale=kw['x_scale'], max_nfev=kw['max_nfev'])
            kw['f_scale'] = cutoff / 2.5
        return real_ls(fun, x0, **kw)
    s.least_squares = ls
    result = []
    for lab in np.unique(labs):
        m = labs != lab
        tau, theta = s._fit(x[m], p[m], h[m], dt, t, y[m], r[m])
        a, b = s._design(tau, x[~m], p[~m], h[~m], dt, t)
        ec, er = abs(a @ theta - y[~m]), abs(b @ theta - r[~m])
        result.append([max(np.quantile(ec[:, cols], .9) for cols in np.array_split(np.arange(len(t)), 3)), np.quantile(er, .9), np.median(ec), np.median(er), *tau])
    v = np.asarray(result)
    metrics[cutoff] = v
    print('Cauchy cutoff in robust SD', cutoff, 'worst p90 strain/rate', v[:, :2].max(axis=0), 'mean lab median strain/rate', v[:, 2:4].mean(axis=0), 'tau ranges', np.ptp(v[:, 4:], axis=0))
s.least_squares = real_ls
raw, true = cal['bridge_raw'], cal['true_channel_loads']
a = np.c_[raw, np.ones(len(raw))]
y_total = true.sum(axis=1)
# Contiguous stress folds, so the extreme held-out sweep groups test extrapolation.
order = np.argsort(y_total)
for i, test in enumerate(np.array_split(order, 5)):
    keep = np.ones(len(raw), dtype=bool)
    keep[test] = False
    affine = np.linalg.lstsq(a[keep], y_total[keep], rcond=None)[0]
    err = abs(a[test] @ affine - y_total[test])
    print('contiguous load fold', i, 'p90/max total stress error', np.quantile(err, .9), err.max())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_robust_hereditary.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2127 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Cauchy cutoff in robust SD 1.5 worst p90 strain/rate [0.02643269 0.01493289] mean lab median strain/rate [0.00860909 0.00623663] tau ranges [0.00444709 0.1246683 ]
Cauchy cutoff in robust SD 2.5 worst p90 strain/rate [0.02636919 0.01476112] mean lab median strain/rate [0.00858898 0.00618618] tau ranges [0.00462748 0.13429479]
Cauchy cutoff in robust SD 4.0 worst p90 strain/rate [0.02631089 0.01464102] mean lab median strain/rate [0.00857383 0.00616983] tau ranges [0.00460457 0.13451095]
Cauchy cutoff in robust SD None worst p90 strain/rate [0.02850541 0.01443973] mean lab median strain/rate [0.00911733 0.00622284] tau ranges [0.00792578 0.4441841 ]
contiguous load fold 0 p90/max total stress error 0.0003134386911479714 0.0004909271775170865
contiguous load fold 1 p90/max total stress error 0.0002561410681430898 0.0003046388860981075
contiguous load fold 2 p90/max total stress error 0.00042529345818376196 0.0006613956378222485
contiguous load fold 3 p90/max total stress error 0.00025840040615887627 0.0003024665561557338
contiguous load fold 4 p90/max total stress error 0.0001989987371356916 0.0002617812961704935
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 23, "ref_swap_strain_max_diff": 2.7e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1b8a9722399259a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1b8a9722 (directory).
```

### Turn 10 — `submit_completed_task` (`16093 ms` LLM)
- **Rationale**: The submitted artifact passed isolated schema and reference-invariance checks. Contiguous load-range holdouts had a worst 90th-percentile total-stress error of 0.000425, and the predicted initial rate agreed with the strain derivative to within 3.70e-7.
- **Alternatives Weighed**: `Widen the Cauchy cutoff from 2.5 to 4 robust standard deviations` (rejected: The worst held-out errors improved by only 0.00006 for strain and 0.00012 for rate, with no improvement in the stability of the slow relaxation time.); `Use quadratic bridge calibration` (rejected: Held-out calibration RMSE increased from 0.000196 to 0.000214.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and pooled-reference invariance checks passed. Deployment uses the benchtop law because the available telemetry does not identify a chamber-specific correction.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
