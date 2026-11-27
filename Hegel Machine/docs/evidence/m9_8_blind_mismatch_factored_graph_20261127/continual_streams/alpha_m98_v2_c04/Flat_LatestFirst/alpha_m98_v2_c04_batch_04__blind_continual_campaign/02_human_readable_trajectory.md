# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_v2_c04_batch_04__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_hereditary_memory_and_factored_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `375,359` (`364,386` / `139,136` / `10,973`) |
| **Total Episode Wall Time (s)** | `380.5s` |

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

### Turn 1 — `execute_in_remote_docker` (`24937 ms` LLM)
- **Rationale**: Inspect the public schema and compact apparatus records in place to detect incompatible inputs or an incorrect chamber-law choice before writing the solver.
- **Alternatives Weighed**: `Reuse the latest solver unchanged` (rejected: The latest campaign may not match the blind chamber's active regime; inspect the apparatus record and input contract first.)

```bash
ls -lh /tmp/hegel_continual_archive /holdout_task; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; cat /tmp/hegel_continual_archive/archive_manifest.json; python3 - <<'PY'
import pickle
from pathlib import Path
p=pickle.loads(Path('/holdout_task/public_batch.pkl').read_bytes()); print('items',len(p)); i=p[0]; print('item keys',list(i),'diagnostics',type(i.get('campaign_diagnostics')))
for k,v in i['dataset'].items():
 print(k, {a:(type(b).__name__,getattr(b,'shape',None)) for a,b in v.items()} if isinstance(v,dict) else (type(v).__name__,getattr(v,'shape',None)))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference panel alignment imputation'
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
uting_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`20863 ms` LLM)
- **Rationale**: Compare archived implementations and summarize telemetry in place. If memory and drift channels match different campaigns, compose their laws instead of copying either archived solver unchanged.
- **Alternatives Weighed**: `Choose a single archived solver by overall trace distance` (rejected: An aggregate distance can hide a mismatch in one chamber component; inspect channel-wise evidence and the solver differences first.)

```bash
cd /tmp/hegel_continual_archive; cat archive_manifest.json; diff -u *s00_solver.py *batch_02_solver.py; cat *batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
r=json.load(open('archive_apparatus_log.json')); a=pickle.load(open('/holdout_task/public_batch.pkl','rb'));print('public items and diagnostics:',len(a),[type(i.get('campaign_diagnostics')).__name__ for i in a]);print('dataset keys',a[0]['dataset'].keys());
z=[]
for d in r:
 t=np.array(d['facility_telemetry_trace']);z.append(t);print({k:v for k,v in d.items() if k!='facility_telemetry_trace'},'trace',t.shape,'mean',np.round(t.mean(0),4),'sd',np.round(t.std(0),4))
for j,t in enumerate(z[:-1]):
 if t.shape==z[-1].shape:print('latest vs',j,'channel RMSE',np.round(np.mean((t-z[-1])**2,axis=0)**.5,4))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`450 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The blind campaign provides no chamber labels or prior lineage archive.
    curve, rate = _design(x[n:], *_loads(test, bridge), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + (curve @ theta).reshape(-1, len(times)),
            'initial_rate_pred': rate @ theta}
public items and diagnostics: 1 ['NoneType']
dataset keys dict_keys(['past_dt', 'eval_times', 'calibration_load_cells', 'train', 'test'])
{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c04_s00'} trace (16, 8) mean [-0.0378  1.1815 -0.3903 -0.026   1.1071 -0.6551 -0.0374  2.4689] sd [0.5809 0.1387 0.0906 0.6017 0.1216 0.1673 0.6758 0.0879]
{'completed_batch_index': 1, 'completed_short_name': 'alpha_m98_v2_c04_batch_01'} trace (16, 8) mean [ 0.0775 -1.164  -0.3893  0.0475 -1.1953 -0.4108  0.0355 -0.3529] sd [0.6323 0.1291 0.0845 0.651  0.1478 0.1683 0.6245 0.076 ]
{'completed_batch_index': 2, 'completed_short_name': 'alpha_m98_v2_c04_batch_02'} trace (16, 8) mean [ 0.0678  1.1389  2.56    0.0634  1.1293 -0.2206  0.0521 -0.3623] sd [0.6989 0.1288 0.0651 0.6037 0.1494 0.1331 0.7079 0.0453]
{'completed_batch_index': 3, 'completed_short_name': 'alpha_m98_v2_c04_batch_03'} trace (16, 8) mean [ 0.084  -1.1095 -0.421   0.0369 -1.153  -0.0039  0.0344 -0.3147] sd [0.5643 0.1716 0.0856 0.6347 0.1676 0.1253 0.6762 0.0999]
latest vs 0 channel RMSE [0.9928 2.3015 0.1382 1.0742 2.2694 0.6796 1.1401 2.7874]
latest vs 1 channel RMSE [0.7287 0.254  0.1259 0.8226 0.2111 0.4557 0.717  0.1178]
latest vs 2 channel RMSE [0.4555 2.2564 2.9832 0.4571 2.2928 0.2668 0.442  0.1233]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`24172 ms` LLM)
- **Rationale**: Inspect asset metadata and time-ordered telemetry for a link to the chamber law, and inspect calibration sweep structure to decide whether the archived affine bridge needs a nonlinear term.
- **Alternatives Weighed**: `Treat campaign 01 as a verified match based on trace distance alone` (rejected: Campaign 01 has no mounted solver, and the two mounted solvers are identical. Trace proximity does not yet identify a correction.)

```bash
cat /holdout_task/public_asset_index.json; sed -n '33,115p' /tmp/hegel_continual_archive/alpha_m98_v2_c04_s00_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
np.set_printoptions(precision=4,suppress=True,linewidth=140)
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for r in json.load(open(p)):
 print('TELEMETRY',r['completed_batch_index']);print(np.asarray(r['facility_telemetry_trace']))
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'];print('times',d['eval_times'],'dt',d['past_dt'])
for k in ['train','test']:
 b=d[k]; x=b['aux_chamber_telemetry'];print(k,'aux shape, mean, sd',x.shape,x.mean(0),x.std(0),'first rows',x[:10]);print('aux corr',np.corrcoef(x.T));print('labs',[(l,int((b['lab_id']==l).sum()),x[b['lab_id']==l].mean(0)) for l in np.unique(b['lab_id'])]);print('load averages',b['past_channel_raw'].mean((0,1)),b['hold_channel_raw'].mean(0))
c=d['calibration_load_cells'];print('cal fields',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in c.items()]);print('sweeps fields',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in c.get('single_channel_sweeps',{}).items()] if isinstance(c.get('single_channel_sweeps'),dict) else type(c.get('single_channel_sweeps')))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`407 ms`, timed_out=`False`)
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

### Turn 4 — `execute_in_remote_docker` (`29221 ms` LLM)
- **Rationale**: Compare held-out calibration errors, including at high loads, to distinguish actual bridge curvature from extra terms that only improve in-sample fit. Summarize the telemetry without dumping raw arrays; if it cannot identify a transferable correction, keep the archived baseline law.
- **Alternatives Weighed**: `Add a chamber correction based on the nearest trace alone` (rejected: Campaign 01 has no mounted solver, so proximity does not supply its constitutive equation.)

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json,pickle,numpy as np
from itertools import combinations_with_replacement
np.set_printoptions(precision=4,suppress=True,linewidth=120)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; print('times',d['eval_times'],'dt',d['past_dt'])
raw,y=c['bridge_raw'],c['true_channel_loads']; print('bridge ranges',np.array([raw.min(0),raw.max(0),y.min(0),y.max(0)])); print('sweep',c['single_channel_sweeps']);
for name,a,b in [('inverse',raw,y),('forward',y,raw)]:
 models={'affine':np.c_[np.ones(len(a)),a], 'diagonal-quadratic':np.c_[np.ones(len(a)),a,a*a], 'quadratic':np.c_[np.ones(len(a)),a,np.column_stack([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(4),2)])], 'odd-cubic':np.c_[np.ones(len(a)),a,a**3]}
 folds=np.argsort(np.linalg.norm(a,axis=1))%5
 for m,X in models.items():
  e=np.empty_like(b)
  for f in range(5):
   t=folds!=f;e[~t]=X[~t]@np.linalg.lstsq(X[t],b[t],rcond=None)[0]-b[~t]
  end=np.linalg.norm(a,axis=1)>=np.quantile(np.linalg.norm(a,axis=1),.8)
  print(name,m,'OOF per-channel, summed, high-load-summed RMSE',np.round(np.r_[np.sqrt((e**2).mean(0)),np.sqrt(np.mean(e.sum(1)**2)),np.sqrt(np.mean(e[end].sum(1)**2))],6))
for k in ['train','test']:
 b=d[k]; a=b['aux_chamber_telemetry']; p=b['past_channel_raw']; h=b['hold_channel_raw']; print(k,'aux mean/std',a.mean(0),a.std(0),'aux correlations to hold total and past totals',np.corrcoef(np.column_stack([a,h.sum(1),p.sum(2)]).T)[:4,4:]); print('groups',np.unique(np.c_[b['lab_id'],b['stratum']],axis=0,return_counts=True))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(row['facility_telemetry_trace']); print('trace',row['completed_batch_index'],'first/last',a[[0,-1]],'range',np.ptp(a,axis=0),'adjacent r',np.array([np.corrcoef(v[:-1],v[1:])[0,1] for v in a.T]))
PY
ls -A /pretrain_task
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.2562]
forward odd-cubic OOF per-channel, summed, high-load-summed RMSE [2.9419 2.7027 2.5862 2.8803 0.3518 0.5394]
train aux mean/std [ 0.1223  0.3043 -0.2727  0.0506] [0.8019 0.9796 1.0443 0.9351] aux correlations to hold total and past totals [[ 0.1384 -0.0547 -0.0674  0.0378  0.0185 -0.0588  0.0337]
 [-0.0432 -0.0411 -0.1645  0.1651  0.0879 -0.0978  0.069 ]
 [ 0.1581 -0.098   0.0423  0.0429  0.0995  0.1416 -0.1333]
 [-0.0655 -0.0981 -0.0196 -0.0707 -0.2083 -0.2119  0.1062]]
groups (array([[0, 0],
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
       [3, 2]]), array([4, 9, 8, 6, 7, 6, 7, 6, 4, 7, 3, 5]))
test aux mean/std [-0.0453 -0.0161  0.3538 -0.1017] [1.192  1.152  1.0943 1.1244] aux correlations to hold total and past totals [[ 0.0603 -0.2018 -0.0497  0.4463 -0.1844 -0.1111 -0.2764]
 [ 0.1259 -0.2002  0.0002 -0.0692 -0.0734 -0.102   0.0703]
 [ 0.1046  0.1004 -0.0816 -0.1929 -0.0654 -0.2503 -0.3842]
 [-0.1355 -0.1881  0.0076 -0.177  -0.1332 -0.103   0.1743]]
groups (array([[0, 0],
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
       [3, 2]]), array([4, 4, 4, 1, 4, 3, 3, 1, 2, 3, 1, 6]))
trace 0 first/last [[-0.0854  1.0972 -0.4263 -0.1003  1.1456 -0.4984 -0.0265  2.4581]
 [-0.071   0.8997 -0.3738 -0.1964  1.0877 -0.482  -0.2074  2.3955]] range [1.7305 0.5527 0.4152 1.8795 0.5369 0.5968 2.0712 0.3252] adjacent r [ 0.8883  0.3994 -0.1605  0.86    0.2376  0.2521  0.9026 -0.1789]
trace 1 first/last [[ 0.6873 -1.352  -0.3801  0.648  -0.9652 -0.2907  0.7004 -0.2262]
 [ 0.839  -1.2348 -0.3302  0.592  -1.0562 -0.0743  0.6203 -0.4148]] range [1.8499 0.4229 0.33   1.9994 0.471  0.6964 1.7764 0.2485] adjacent r [ 0.8432 -0.489   0.0525  0.8705  0.0767  0.1635  0.8905  0.2009]
trace 2 first/last [[ 0.9762  1.0882  2.5452  0.9801  1.0587 -0.2308  1.0406 -0.3522]
 [ 1.0475  0.9963  2.5715  0.8935  1.0996 -0.1898  1.0075 -0.3573]] range [2.1915 0.5563 0.3022 1.7012 0.5931 0.4878 2.0886 0.1834] adjacent r [ 0.8549 -0.0909 -0.3629  0.8997 -0.1727  0.1724  0.8569  0.1585]
trace 3 first/last [[ 0.4294 -1.2481 -0.403   0.878  -1.0759 -0.1377  0.6413 -0.2796]
 [ 0.7401 -1.0255 -0.4386  0.9176 -0.8074  0.1256  0.7091 -0.2818]] range [1.7432 0.5037 0.2853 1.8019 0.6357 0.4511 1.985  0.4224] adjacent r [ 0.8838  0.1749  0.2358  0.8611 -0.0038 -0.1004  0.8847  0.0267]
```

### Turn 5 — `execute_in_remote_docker` (`28562 ms` LLM)
- **Rationale**: Paired observations and residual singular vectors distinguish a shared bridge distortion from independent channel nonlinearities. Coupon raw ranges show whether the large sweep deviations also occur in the inference inputs.
- **Alternatives Weighed**: `Deploy the full quadratic inverse` (rejected: It improves channel estimates but worsens the measured total-stress error, especially at high loads.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine inverse recovers the four physical loads to calibration precision.` | Witness: `Held-out channel RMSE is 0.305–0.608 for the affine inverse versus 0.110–0.201 for the quadratic inverse, but total-stress RMSE worsens from 0.311 to 0.348.` | Blanket Diagnosis: `Channel residuals may share a distortion that largely cancels in the axial sum. Separate channel fits can improve individual channels while spoiling that cancellation.` | Sublated Pivot: `Inspect paired sweeps and the residual subspace to identify a compact bridge law, then validate the axial sum directly.`

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
np.set_printoptions(precision=4,suppress=True,linewidth=150)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; a=np.c_[np.ones(len(x)),x]; b=np.linalg.lstsq(a,y,rcond=None)[0]; e=y-a@b
print('forward affine matrix',b); print('residual svd',np.linalg.svd(e,full_matrices=False)[1:]); print('truth unique values and counts by channel',[(len(np.unique(v)),len(v)) for v in x.T]); print('first 20 paired rows [truth | raw | totals]'); print(np.c_[x,y,x.sum(1),y.sum(1)][:20]); i=np.argsort(np.linalg.norm(e,axis=1)); print('largest paired rows [index | truth | raw | totals]'); print(np.c_[i,x[i],y[i],x[i].sum(1),y[i].sum(1)][-14:]);
for q in (.35,.55,.75,1):
 j=i[:int(q*len(x))]; B=np.linalg.lstsq(a[j],y[j],rcond=None)[0]; E=y-a@B; print('fraction',q,'matrix',B,'median vector error',np.median(np.linalg.norm(E,axis=1)),'median sum error',np.median(abs(E.sum(1))))
for k in ('train','test'):
 s=d[k]; z=np.vstack((s['hold_channel_raw'],s['past_channel_raw'].reshape(-1,4))); print(k,'raw quantiles',np.quantile(z,[0,.05,.5,.95,1],axis=0),'sum quantiles',np.quantile(z.sum(1),[0,.05,.5,.95,1]));print(k,'raw covariance singular values',np.linalg.svd(z-z.mean(0),compute_uv=False));
PY
find /pretrain_task -maxdepth 2 -type f -printf '%f %s\n' | head -25
```
- **Container Exit Code**: `0` (`399 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.303    2.1314  -1.2894   1.7494   0.1313   1.2884]
 [ 46.       0.7039   1.3196  -0.8373   1.1353   0.1629   3.0695  -0.9954   0.58     2.3215   2.8169]
 [ 29.      -1.2684  -0.5821  -1.5434   1.4103  -1.2537  -0.5536  -2.8049   1.8218  -1.9836  -2.7904]
 [ 62.      -0.7784  -0.9656   1.0693   1.1803  -0.5652  -1.556    0.8521   1.5171   0.5056   0.2481]
 [ 57.      -1.1871  -0.9433   1.3101   1.48    -1.106   -1.6257   1.1048   2.252    0.6596   0.6251]
 [ 15.       0.       0.       0.       1.7     14.5966 -13.0699  12.2917 -12.1782   1.7      1.6402]
 [ 14.       0.       0.       0.       0.7     14.4751 -13.0699  12.6907 -13.6182   0.7      0.4778]
 [ 13.       0.       0.       0.      -0.6     14.317  -13.07    13.2095 -14.7249  -0.6     -0.2684]]
fraction 0.35 matrix [[-0.0718  0.0433  0.3255  0.1735]
 [ 1.0575  0.1276 -0.0394 -0.0723]
 [-0.2645  1.6124  0.1914  0.0181]
 [-0.0677 -0.1905  1.9017  0.1386]
 [ 0.1579 -0.1136 -0.2755  1.2722]] median vector error 0.4332979593579893 median sum error 0.257665863497841
fraction 0.55 matrix [[-0.0291  0.0177  0.2164 -0.049 ]
 [ 1.0756  0.1868  0.0122 -0.1052]
 [-0.2009  1.5966  0.1707  0.0219]
 [-0.0494 -0.266   1.8474  0.1393]
 [ 0.1989 -0.1185 -0.3638  1.0607]] median vector error 0.3371336663994657 median sum error 0.21846462213964957
fraction 0.75 matrix [[-0.0557  0.1084  0.2006 -0.0875]
 [ 1.0891  0.1832 -0.0111 -0.1305]
 [-0.2169  1.6088  0.1895  0.0352]
 [-0.0319 -0.2807  1.8129  0.1227]
 [ 0.1767 -0.0266 -0.3731  1.0184]] median vector error 0.32533304676111996 median sum error 0.21302042527698606
fraction 1 matrix [[ 0.4818 -0.3824  0.7163 -0.6818]
 [ 1.1008  0.1621  0.0346 -0.2424]
 [-0.2032  1.5806  0.2839 -0.0427]
 [-0.0135 -0.3152  1.7065  0.1215]
 [ 0.6489 -0.446   0.1058  0.4861]] median vector error 1.0350388228735756 median sum error 0.19353351625942883
train raw quantiles [[-0.806  -0.9834 -0.9751 -0.8123]
 [-0.5872 -0.649  -0.6768 -0.6068]
 [ 0.1028  0.5197  0.5473  0.0822]
 [ 1.0047  1.8654  1.9248  0.8929]
 [ 1.2738  2.3008  2.4809  1.2663]] sum quantiles [-1.4604 -0.5874  1.3408  3.2453  4.4745]
train raw covariance singular values [18.9679 16.38   11.2063  9.7032]
test raw quantiles [[-0.7741 -0.9511 -0.9387 -0.8126]
 [-0.5876 -0.6272 -0.6372 -0.6055]
 [ 0.1929  0.4863  0.4307  0.0074]
 [ 0.8917  1.7781  1.8477  0.8019]
 [ 1.2762  2.1966  2.5632  1.3392]] sum quantiles [-1.1886 -0.454   1.1979  3.2617  4.7843]
test raw covariance singular values [12.7023 11.3024  7.4112  7.0851]
```

### Turn 6 — `execute_in_remote_docker` (`23818 ms` LLM)
- **Rationale**: Compare leave-one-out errors for compact forward laws on the ordinary sweeps. If cross-products and diagonal curvature explain the data down to sensor precision, invert that law and validate axial stress on held-out sweeps in the coupon load range.
- **Alternatives Weighed**: `Use an unconstrained inverse polynomial` (rejected: Its high-load axial error worsened, and it does not distinguish ordinary bridge curvature from the three gross sweep deviations.)

```bash
cat > /tmp/bridge_probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
np.set_printoptions(precision=6,suppress=True)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'];c=d['calibration_load_cells'];x=c['true_channel_loads'];y=c['bridge_raw']; good=np.max(abs(y),axis=1)<8
xs=x[good];ys=y[good]; n=len(xs); pairs=list(combinations(range(4),2));
base={'linear':lambda a:a,'sq':lambda a:a*a,'signed_sq':lambda a:a*abs(a),'cube':lambda a:a**3,'pair':lambda a:np.column_stack([a[:,j]*a[:,k] for j,k in pairs]),'triple':lambda a:np.column_stack([a[:,p].prod(1) for p in combinations(range(4),3)]),'mixedcube':lambda a:np.column_stack([a[:,j]**2*a[:,k] for j in range(4) for k in range(4) if j!=k])}
def design(a,terms):return np.column_stack([np.ones(len(a))]+[base[k](a) for k in terms.split()]);
for terms in ['linear','linear sq','linear signed_sq','linear cube','linear cube pair','linear signed_sq pair','linear sq pair','linear sq cube pair','linear cube pair triple','linear cube pair mixedcube']:
 A=design(xs,terms); beta=np.linalg.lstsq(A,ys,rcond=None)[0]; E=ys-A@beta; L=np.sum(np.linalg.qr(A)[0]**2,axis=1); ee=E/(1-L[:,None]);print(terms,'LOO RMSE',np.sqrt(np.mean(ee**2,axis=0)),'fit RMSE',np.sqrt(np.mean(E**2,axis=0)))
 if terms in ('linear cube pair','linear signed_sq pair','linear sq cube pair'):print('coef',beta)
 for i in range(4):
  h=(np.count_nonzero(xs,axis=1)==1)&(xs[:,i]!=0)
  if terms in ('linear signed_sq pair','linear cube pair'):print('axis',i,'diagonal residual',E[h,i]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_probe.py
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.00007   0.000002 -0.000062  0.000104]
axis 1 diagonal residual [-0.000034  0.000012  0.000118  0.00003 ]
axis 2 diagonal residual [ 0.000095 -0.000046 -0.000159 -0.000053]
axis 3 diagonal residual [-0.000038]
linear signed_sq pair LOO RMSE [0.018237 0.031122 0.031936 0.014489] fit RMSE [0.014544 0.024161 0.02463  0.011406]
coef [[-0.11139   0.134224  0.169834 -0.119744]
 [ 0.648068  0.186935  0.005683 -0.241748]
 [-0.241243  1.011005  0.195841 -0.005562]
 [ 0.003223 -0.395664  1.058537  0.114185]
 [ 0.117624  0.001939 -0.394029  0.65539 ]
 [ 0.323695  0.003454  0.001634  0.003682]
 [ 0.000795  0.503693  0.002475  0.004602]
 [-0.002304  0.014137  0.535996  0.006027]
 [ 0.002123 -0.004272 -0.002746  0.291901]
 [-0.17849  -0.001261  0.000488  0.000074]
 [ 0.001841  0.002348  0.002683  0.001847]
 [-0.002881 -0.003868  0.000654 -0.175167]
 [-0.004009 -0.277086 -0.000784  0.000686]
 [ 0.001861 -0.002539  0.004217 -0.000827]
 [ 0.001521  0.001035 -0.286684 -0.00073 ]]
axis 0 diagonal residual [-0.026281 -0.010721  0.006094  0.057605]
axis 1 diagonal residual [-0.04219  -0.021271  0.00116   0.077705]
axis 2 diagonal residual [-0.042599 -0.021487  0.009501  0.088872]
axis 3 diagonal residual [-0.050888]
linear sq pair LOO RMSE [0.110213 0.176029 0.176259 0.088662] fit RMSE [0.088693 0.137768 0.140174 0.065924]
linear sq cube pair LOO RMSE [0.000111 0.000103 0.000126 0.000103] fit RMSE [0.000085 0.000078 0.000097 0.000079]
coef [[-0.10999   0.129985  0.170027 -0.120046]
 [ 0.81001   0.189007  0.000006 -0.23697 ]
 [-0.242967  1.259995  0.199539 -0.00006 ]
 [ 0.000012 -0.378004  1.329961  0.118433]
 [ 0.121538 -0.00003  -0.399048  0.790022]
 [-0.000012 -0.000006 -0.000005  0.000011]
 [ 0.000006  0.000009 -0.000004  0.000035]
 [ 0.000009  0.000004 -0.000021  0.000008]
 [ 0.000001  0.000006  0.000006 -0.00002 ]
 [ 0.145792 -0.000005  0.000001 -0.000014]
 [-0.000008  0.226805 -0.000001  0.000011]
 [ 0.000001  0.000001  0.23944   0.000026]
 [-0.000015  0.000009  0.000034  0.142191]
 [-0.178217  0.000018  0.000009  0.000006]
 [ 0.000025  0.000002  0.000001  0.000017]
 [ 0.000005 -0.000019  0.00001  -0.173809]
 [-0.000015 -0.277194 -0.000025 -0.00001 ]
 [-0.000004 -0.000007  0.000014 -0.000027]
 [ 0.000012  0.000015 -0.292594 -0.000005]]
linear cube pair triple LOO RMSE [0.000113 0.000103 0.000128 0.000113] fit RMSE [0.000084 0.000077 0.000096 0.000084]
linear cube pair mixedcube LOO RMSE [0.000133 0.000096 0.000132 0.000123] fit RMSE [0.000083 0.000064 0.000085 0.000076]
```

### Turn 7 — `execute_in_remote_docker` (`35709 ms` LLM)
- **Rationale**: Implement the measured bridge law and test axial error after inversion, including at high loads. Check every coupon input for inversion failure before using decoded loads to estimate the relaxation spectrum. Read the archived preprocessing and log fields to identify what can be reused for the blind chamber.
- **Alternatives Weighed**: `Add quadratic and three-way terms` (rejected: The extra terms do not improve the worst held-out channel; the smaller forward basis already reaches measurement precision.)

```bash
cat /tmp/hegel_continual_archive/alpha_m98_v2_c04_s00_solver.py; python3 - <<'PY'
import json,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for r in json.load(open(p)):
 a=np.array(r['facility_telemetry_trace']);print({k:v for k,v in r.items() if k!='facility_telemetry_trace'}); print('means',np.round(a.mean(0),3),'stds',np.round(a.std(0),3))
PY
cat > /tmp/bridge_candidate.py <<'PY'
import numpy as np

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _bridge_basis(x):
    return np.column_stack((np.ones(len(x)), x, x**3,
                            *(x[:, i]*x[:, j] for i, j in _PAIRS)))


def _fit_bridge(cal):
    x = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    a = _bridge_basis(x)
    # Find the ordinary forward response despite large discordant sweep readings.
    mag = np.max(np.abs(raw - np.median(raw, axis=0)), axis=1)
    keep = mag < 5*np.median(mag)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]
        err = np.linalg.norm(a@coef-raw, axis=1)
        keep = err < 8*max(np.median(err), 1e-8)
    return np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]


def _decode_bridge(raw, coef):
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    lin = coef[1:5]
    x = np.linalg.solve(lin.T, (y-coef[0]).T).T
    for _ in range(18):
        err = _bridge_basis(x)@coef-y
        if np.max(np.abs(err), initial=0.) < 1e-10:
            break
        jac = np.broadcast_to(lin, (len(x), 4, 4)).copy()
        jac += 3*x[:, :, None]**2*coef[5:9]
        for v, (i, j) in zip(coef[9:], _PAIRS):
            jac[:, i, :] += x[:, j, None]*v
            jac[:, j, :] += x[:, i, None]*v
        step = np.linalg.solve(jac.transpose(0, 2, 1), err[:, :, None])[:, :, 0]
        x -= step/np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return x.reshape(np.shape(raw))
PY
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/tmp'); from bridge_candidate import _fit_bridge,_decode_bridge,_bridge_basis
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; b=_fit_bridge(c); clean=np.linalg.norm(_bridge_basis(x)@b-y,axis=1)<.01; errors=[]
for j in np.flatnonzero(clean):
 i=np.arange(len(x))!=j; m=_fit_bridge({'true_channel_loads':x[i],'bridge_raw':y[i]}); errors.append((_decode_bridge(y[j:j+1],m)[0]-x[j]).sum())
e=np.array(errors); top=np.linalg.norm(x[clean],axis=1)>=np.quantile(np.linalg.norm(x[clean],axis=1),.8); print('clean, axial OOF RMSE, high-load axial OOF RMSE and worst absolute',clean.sum(),np.sqrt(np.mean(e*e)),np.sqrt(np.mean(e[top]**2)),np.max(np.abs(e)))
for name in ('train','test'):
 for field in ('past_channel_raw','hold_channel_raw'):
  y=d[name][field].reshape(-1,4); z=_decode_bridge(y,b); print(name,field,'max forward roundtrip',np.max(np.abs(_bridge_basis(z)@b-y)),'load range',np.min(z),np.max(z))
PY
```
- **Container Exit Code**: `0` (`588 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
= 2.5 * np.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(x[:n], *_loads(train, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The blind campaign provides no chamber labels or prior lineage archive.
    curve, rate = _design(x[n:], *_loads(test, bridge), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + (curve @ theta).reshape(-1, len(times)),
            'initial_rate_pred': rate @ theta}
{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c04_s00'}
means [-0.038  1.181 -0.39  -0.026  1.107 -0.655 -0.037  2.469] stds [0.581 0.139 0.091 0.602 0.122 0.167 0.676 0.088]
{'completed_batch_index': 1, 'completed_short_name': 'alpha_m98_v2_c04_batch_01'}
means [ 0.078 -1.164 -0.389  0.047 -1.195 -0.411  0.036 -0.353] stds [0.632 0.129 0.085 0.651 0.148 0.168 0.624 0.076]
{'completed_batch_index': 2, 'completed_short_name': 'alpha_m98_v2_c04_batch_02'}
means [ 0.068  1.139  2.56   0.063  1.129 -0.221  0.052 -0.362] stds [0.699 0.129 0.065 0.604 0.149 0.133 0.708 0.045]
{'completed_batch_index': 3, 'completed_short_name': 'alpha_m98_v2_c04_batch_03'}
means [ 0.084 -1.109 -0.421  0.037 -1.153 -0.004  0.034 -0.315] stds [0.564 0.172 0.086 0.635 0.168 0.125 0.676 0.1  ]
clean, axial OOF RMSE, high-load axial OOF RMSE and worst absolute 77 0.0001954502733370019 0.0001454545828504706 0.0005002550256570371
train past_channel_raw max forward roundtrip 3.68594044175552e-14 load range -0.6498758307217656 1.1489159184728388
train hold_channel_raw max forward roundtrip 1.3322676295501878e-15 load range 0.35260298121722544 1.3473320415964924
test past_channel_raw max forward roundtrip 3.064215547965432e-14 load range -0.6465436600154716 1.1490093808341328
test hold_channel_raw max forward roundtrip 6.661338147750939e-16 load range 0.35286259304676315 1.3496071887407357
```

### Turn 8 — `execute_in_remote_docker` (`35802 ms` LLM)
- **Rationale**: Integrate the validated bridge, then test whether the material fit holds across labs and whether both deployment outputs agree with a direct chronological recurrence. The archive distinguishes regimes but does not establish an equation for the active chamber.
- **Alternatives Weighed**: `Copy the nearest campaign’s chamber correction` (rejected: The trace matches campaign 01, but that campaign has no mounted solver; the two mounted solvers both use the benchtop law.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
base = Path('/tmp/hegel_continual_archive/alpha_m98_v2_c04_s00_solver.py').read_text()
bridge = Path('/tmp/bridge_candidate.py').read_text().split('\n', 1)[1].strip()
lo = base.index('def _bridge(')
hi = base.index('def _design(', lo)
loads = '''def _loads(block, bridge):
    return tuple(_decode_bridge(block[key], bridge).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


'''
base = base[:lo] + bridge + '\n\n\n' + loads + base[hi:]
base = base.replace("bridge = _bridge(dataset['calibration_load_cells'])", "bridge = _fit_bridge(dataset['calibration_load_cells'])")
base = base.replace('    # The blind campaign provides no chamber labels or prior lineage archive.', '    # The archive identifies a telemetry regime but supplies no chamber law\n    # for that regime; use the measured benchtop constitutive law.')
Path('/holdout_task/batch_solver.py').write_text(base)
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
cat > /tmp/validate_rheology.py <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'
s=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
for item in items:
 d=item['dataset']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=m._features(tr,te)[:n]; c=d['calibration_load_cells']; B=m._fit_bridge(c)
 past,hold=m._loads(tr,B); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; labs=tr['lab_id']; pred=m.solve_batch(d); tau=pred['estimated_taus']; theta=pred['hereditary_weights']; print('all-coupon fit',tau,theta)
 assert .05<tau[0]<tau[1]<25 and all(np.isfinite(v).all() for v in pred.values())
 ec=np.empty_like(yc); er=np.empty_like(yr)
 for lab in np.unique(labs):
  u=labs!=lab; v=~u
  q,b=m._fit(x[u],past[u],hold[u],dt,t,yc[u],yr[u]); D,R=m._design(x[v],past[v],hold[v],dt,t,q)
  ec[v]=(D@b).reshape(-1,len(t))-yc[v]; er[v]=R@b-yr[v]
  print('held lab',lab,'taus',q,'max weights change',np.max(np.abs(theta-b)),'curve/rate MAE',np.mean(abs(ec[v])),np.mean(abs(er[v])),'medAE',np.median(abs(ec[v])),np.median(abs(er[v])))
 for idx in [0,len(t)-1]: print('horizon',t[idx],'lab MAE',[np.mean(abs(ec[labs==l,idx])) for l in np.unique(labs)])
 pp,hh=m._loads(te,B); z=np.zeros((len(te['lab_id']),2));
 for j,step in enumerate(dt):
  a=np.exp(-step/tau); z=a*z+(1-a)*pp[:,j,None]
 gap=hh[:,None]-z; drift=theta[2]+m._features(tr,te)[n:]@theta[3:]
 curve=te['strain_0'][:,None]+(gap*theta[:2])@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*t
 rate=(gap*theta[:2]/tau).sum(1)+drift
 np.testing.assert_allclose(pred['predicted_strain'],curve,atol=1e-10)
 np.testing.assert_allclose(pred['initial_rate_pred'],rate,atol=1e-10)
 print('test curve and initial rate agree with independent chronological recurrence')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/validate_rheology.py
```
- **Container Exit Code**: `0` (`1581 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
all-coupon fit [0.47589406 4.37017306] [ 0.6323159   0.55940605  0.33122391  0.16797299 -0.09644901  0.15789495
 -0.11044029]
held lab 0 taus [0.47720838 4.47499027] max weights change 0.002946558720170356 curve/rate MAE 0.01805807320083739 0.008191745926580521 medAE 0.008786375486025122 0.008509328937020832
held lab 1 taus [0.47599245 4.41439969] max weights change 0.0030227452845141833 curve/rate MAE 0.01874262377981778 0.005031601996811273 medAE 0.007201009045136431 0.004243766611912392
held lab 2 taus [0.47558656 4.33450579] max weights change 0.0012540774697777923 curve/rate MAE 0.015824953520718676 0.005943157472442817 medAE 0.0076159670423991965 0.0056688173483863125
held lab 3 taus [0.47507186 4.27693179] max weights change 0.004467091096649223 curve/rate MAE 0.02295169668603879 0.005779106707056429 medAE 0.009588581875185298 0.0038724685333608733
horizon 0.25 lab MAE [0.011673529767444433, 0.007858975518185823, 0.026995160390296674, 0.012094085340875612]
horizon 4.0 lab MAE [0.006300672454159629, 0.009192350630784333, 0.011563221498920568, 0.009489116602991062]
test curve and initial rate agree with independent chronological recurrence

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75bdd66063dc4ecf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75bdd660 (directory).
```

### Turn 9 — `submit_completed_task` (`15881 ms` LLM)
- **Rationale**: The saved candidate passed the public schema check, isolated execution, held-out calibration and lab tests, and independent verification of both deployment outputs. No remaining measured failure warrants another edit.
- **Alternatives Weighed**: `Add a chamber-specific correction from telemetry` (rejected: The archive provides no verified constitutive equation for the closest telemetry regime, and the current solver passed the isolated accuracy probe.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5741'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified /holdout_task/batch_solver.py. The public schema passed. The isolated host probe measured strain MAE of 0.000754, initial-rate MAE of 0.000477, and relative relaxation-time error of 0.46%.`

### Turn 10 — `execute_in_remote_docker` (`43208 ms` LLM)
- **Rationale**: Test whether signed pairwise or rational terms materially reduce held-out axial error. Compare pooled and within-lab residual associations to detect whether auxiliary telemetry is a lab shortcut or a candidate for further validation. Leave the verified artifact unchanged unless the audit reveals a measured failure.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck drift, or balanced classification accuracy` (rejected: The solver predicts continuous strain and strain rates from deterministic viscoelastic histories; it has no relay states, diffusion operator, or class labels.)

```bash
set -e
cat > /tmp/rheology_margin_audit.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
x, raw = c['true_channel_loads'], c['bridge_raw']
b = m._fit_bridge(c)
ordinary = np.linalg.norm(m._bridge_basis(x) @ b - raw, axis=1) < .01
# Evaluate only ordinary sweeps, while giving every training fold its gross errors.
# Both additions are specified before any held-out observations are scored.
def extras(a, family):
    qs = [a[:,i] + sign*a[:,j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1)]
    if family == 'signed_pair':
        return np.column_stack([q*abs(q) for q in qs])
    return np.column_stack([q**3/(1+q*q) for q in [*a.T,*qs]])

norm = np.linalg.norm(x[ordinary], axis=1)
high = norm >= np.quantile(norm, .8)
errors = {key: [] for key in ('cubic_pair','signed_pair','rational')}
for j in np.flatnonzero(ordinary):
    v = np.arange(len(x)) != j
    bh = m._fit_bridge({'true_channel_loads':x[v], 'bridge_raw':raw[v]})
    keep = v & (np.linalg.norm(m._bridge_basis(x) @ bh - raw, axis=1) < .01)
    est = m._decode_bridge(raw[j:j+1], bh)[0]
    errors['cubic_pair'].append((est-x[j]).sum())
    # A correction this small can be evaluated via the local inverse Jacobian.
    h = 1e-5
    A = m._bridge_basis(x[keep])
    for family in ('signed_pair','rational'):
        def basis(a):
            return np.column_stack((m._bridge_basis(a),extras(a,family)))
        C = np.linalg.lstsq(basis(x[keep]),raw[keep],rcond=None)[0]
        J = (basis(est+np.eye(4)*h)-basis(est-np.eye(4)*h))@C/(2*h)
        dx = np.linalg.solve(J.T, raw[j]-basis(est[None,:])[0]@C)
        errors[family].append((est+dx-x[j]).sum())
for name, error in errors.items():
    e = np.array(error)
    print(name, 'axial RMSE overall/high load, max absolute', *(f'{v:.7g}' for v in (np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[high]**2)),max(abs(e)))))

tr,te = d['train'],d['test']
lab = tr['lab_id']; strata = tr['stratum']; n = len(lab)
features = m._features(tr,te)[:n]
past,hold = m._loads(tr,b)
yc = tr['strain_curves_obs'] - tr['strain_0'][:,None]
yr = tr['instant_rate_obs']
t = d['eval_times']; dt = d['past_dt']
aux = np.asarray(tr['aux_chamber_telemetry'])
errc, errr = np.empty_like(yc), np.empty_like(yr)
for l in np.unique(lab):
    keep = lab != l
    tau,theta = m._fit(features[keep],past[keep],hold[keep],dt,t,yc[keep],yr[keep])
    D,R = m._design(features[~keep],past[~keep],hold[~keep],dt,t,tau)
    errc[~keep] = (D@theta).reshape(-1,len(t)) - yc[~keep]
    errr[~keep] = R@theta - yr[~keep]
# Sample-level robust residual summaries avoid letting a single curve outlier determine group bias.
summary = np.column_stack((np.median(errc,axis=1),errr))
print('worst lab/horizon MAE',max(np.mean(abs(errc[lab==l]),axis=0).max() for l in np.unique(lab)))
for l,s in np.unique(np.c_[lab,strata],axis=0):
    keep=(lab==l)&(strata==s)
    print('lab/stratum/n',l,s,keep.sum(),'median curve/rate bias',np.round(np.median(summary[keep],axis=0),5))
a = np.c_[aux,np.clip(summary,-.03,.03)]
z = a.copy()
for l in np.unique(lab):
    v=lab==l
    z[v]-=z[v].mean(0)
print('max absolute auxiliary correlation per output, pooled',np.max(np.abs(np.corrcoef(a.T)[:-2,-2:]),axis=0),'within-lab centered',np.max(np.abs(np.corrcoef(z.T)[:-2,-2:]),axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1472 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cubic_pair axial RMSE overall/high load, max absolute 0.0001954503 0.0001454546 0.000500255
signed_pair axial RMSE overall/high load, max absolute 0.0002019747 0.00017016 0.0006039228
rational axial RMSE overall/high load, max absolute 0.0002012662 0.0001868166 0.0005153497
worst lab/horizon MAE 0.044091403431015794
lab/stratum/n 0 0 4 median curve/rate bias [ 0.00152 -0.00482]
lab/stratum/n 0 1 9 median curve/rate bias [ 0.00301 -0.00092]
lab/stratum/n 0 2 8 median curve/rate bias [0.00177 0.00239]
lab/stratum/n 1 0 6 median curve/rate bias [-0.00329  0.00128]
lab/stratum/n 1 1 7 median curve/rate bias [-0.0045   0.00046]
lab/stratum/n 1 2 6 median curve/rate bias [-0.00204 -0.00535]
lab/stratum/n 2 0 7 median curve/rate bias [-0.00168  0.00285]
lab/stratum/n 2 1 6 median curve/rate bias [ 0.00032 -0.00497]
lab/stratum/n 2 2 4 median curve/rate bias [-2.06e-03 -2.00e-05]
lab/stratum/n 3 0 7 median curve/rate bias [-0.00474  0.00301]
lab/stratum/n 3 1 3 median curve/rate bias [-2.00e-05 -6.54e-03]
lab/stratum/n 3 2 5 median curve/rate bias [-0.00041  0.00174]
max absolute auxiliary correlation per output, pooled [0.10473612 0.35102758] within-lab centered [0.13773658 0.33040005]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75bdd66063dc4ecf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75bdd660 (directory).
```

### Turn 11 — `submit_completed_task` (`17362 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema and scientific verification. The stress audit found no measured improvement from the broader bridge models and no evidence that justifies a chamber correction.
- **Alternatives Weighed**: `Add rational or signed pairwise bridge terms` (rejected: Both increased held-out axial RMSE overall and at high load.); `Add a chamber correction from auxiliary telemetry` (rejected: The archive does not identify a constitutive equation for the closest regime, and the benchtop residual associations do not establish a transferable correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5741'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. The isolated host probe measured strain MAE of 0.000754, initial-rate MAE of 0.000477, and relative relaxation-time error of 0.46%.`

### Turn 12 — `execute_in_remote_docker` (`30693 ms` LLM)
- **Rationale**: Test whether the bridge fit depends on seeing neighboring loads from the same sweep. Compare it with a sparse response fitted independently inside each fold; change the solver only if the alternative fixes a material error in a measured slice.
- **Alternatives Weighed**: `Repeat the signed-pair and rational-term comparisons` (rejected: Those terms already worsened held-out axial error. Withholding an entire sweep or the upper mixed-load band tests extrapolation that leave-one-out did not.)

```bash
set -e
cat > /tmp/rheology_blocked_bridge.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
A = m._bridge_basis(x)
axis = np.count_nonzero(x, axis=1) == 1
folds = [(f'axis {j}', axis & (x[:,j] != 0)) for j in range(4)]
radius = np.linalg.norm(x, axis=1)
folds.extend([('upper mixed-load band', (~axis) & (radius >= np.quantile(radius[~axis], .8)))])
for name, test in folds:
    tr = ~test
    b = m._fit_bridge({'true_channel_loads':x[tr], 'bridge_raw':y[tr]})
    e = np.linalg.norm(A@b-y, axis=1)
    ordinary = e < .01
    # Infer the sparse response independently in each training fold. The
    # 0.002 cutoff separates the observed order-0.1 terms from sensor noise.
    bs = b.copy()
    for j in range(4):
        use = abs(b[:,j]) > .002
        bs[:,j] = 0
        bs[use,j] = np.linalg.lstsq(A[tr & ordinary][:,use], y[tr & ordinary,j], rcond=None)[0]
    v = test & ordinary
    print(name, 'ordinary/total', v.sum(), test.sum(), 'min rejected forward residual', min(e[test & ~ordinary], default=float('nan')))
    for label, coef in [('current', b), ('sparse', bs)]:
        delta = m._decode_bridge(y[v], coef)-x[v]
        z = delta.sum(1)
        print(label, 'axial RMSE/max', np.sqrt(np.mean(z*z)), max(abs(z)), 'max channel error', np.max(abs(delta)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_blocked_bridge.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`929 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
axis 0 ordinary/total 4 4 min rejected forward residual nan
current axial RMSE/max 0.0002839850683025386 0.0004012833437983362 max channel error 0.0002564052038946515
sparse axial RMSE/max 0.0002193380421676125 0.0003408205934275031 max channel error 0.00024050989185364256
axis 1 ordinary/total 4 4 min rejected forward residual nan
current axial RMSE/max 0.00023948857568226078 0.00041166162725749986 max channel error 0.00026082510777240966
sparse axial RMSE/max 0.00011782989046739643 0.00018503268817762346 max channel error 0.00017938808537066057
axis 2 ordinary/total 4 4 min rejected forward residual nan
current axial RMSE/max 0.00028287211604384693 0.00034184803327327363 max channel error 0.00018783365388387962
sparse axial RMSE/max 0.00020855612009957128 0.00030262174370703263 max channel error 0.00019746264328442746
axis 3 ordinary/total 1 4 min rejected forward residual 27.333732288700727
current axial RMSE/max 0.0001305344416484759 0.0001305344416484759 max channel error 7.25832322484496e-05
sparse axial RMSE/max 9.592199782958744e-05 9.592199782958744e-05 max channel error 4.9364517756703e-05
upper mixed-load band ordinary/total 13 13 min rejected forward residual nan
current axial RMSE/max 0.00019438026914401273 0.00031312959871043766 max channel error 0.0001893721056149822
sparse axial RMSE/max 0.00018245210465821075 0.0004086772691049845 max channel error 0.0002392454092823293
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75bdd66063dc4ecf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75bdd660 (directory).
```

### Turn 13 — `submit_completed_task` (`19368 ms` LLM)
- **Rationale**: Whole-sweep and upper-load holdouts confirmed the calibration result without relying on neighboring points from the same sweep. The unchanged artifact passed isolated schema validation and the host scientific probe; the audit found no prediction failure that warrants another edit.
- **Alternatives Weighed**: `Use the sparse bridge fit` (rejected: It lowered RMSE in the blocked calibration folds but increased the maximum axial error in the upper mixed-load band, with no demonstrated improvement in strain or rate prediction.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed in isolated Python. The host probe measured strain MAE of 0.000754, initial-rate MAE of 0.000477, and relative relaxation-time error of 0.46%. Prediction accuracy in unseen chamber regimes was not verified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
