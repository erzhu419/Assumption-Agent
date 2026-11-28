# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_alpha_m100_01_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `9 / 16` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `211,449` (`203,086` / `21,760` / `8,363`) |
| **Total Episode Wall Time (s)** | `258.0s` |

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

### Turn 1 — `execute_in_remote_docker` (`16554 ms` LLM)
- **Rationale**: Inspect the public schema and chronological archive first: calibration and reference-panel field layouts determine preprocessing, and available chamber evidence determines whether the benchtop law needs a deployment correction.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference panel alignment imputation'; python3 - <<'PY'
import pickle, numpy as np, pathlib
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
def scan(x,p='',d=0):
 if isinstance(x,dict) and d<3:
  for k,v in x.items(): scan(v,p+'/'+str(k),d+1)
 elif isinstance(x,np.ndarray): print(p,x.shape,x.dtype,repr(x.ravel()[:7]))
 else: print(p,type(x).__name__,repr(x)[:340])
scan(x)
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
 print('\n',p,p.stat().st_size); print(p.read_text()[:9500])
PY
```
- **Container Exit Code**: `0` (`431 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9332,
        -0.32439,
        -0.76509,
        -0.71243,
        -0.41442,
        0.01206,
        -0.53966,
        0.51224
      ],
      [
        -0.73733,
        -0.32105,
        -0.84512,
        -0.78131,
        -0.70337,
        -0.56806,
        -0.42546,
        0.12069
      ],
      [
        -0.8324,
        -0.39429,
        -0.46351,
        -0.82546,
        -0.82772,
        -0.85506,
        -0.49949,
        -0.41772
      ],
      [
        -0.9239,
        -0.35428,
        0.25232,
        -0.85603,
        -0.79605,
        -0.65665,
        -0.39622,
        -0.72437
      ],
      [
        -0.78462,
        -0.38442,
        0.70322,
        -0.8108,
        -0.48389,
        -0.04807,
        -0.42083,
        -0.81883
      ],
      [
        -0.68748,
        -0.29172,
        0.90021,
        -0.70659,
        -0.06455,
        0.53055,
        -0.40594,
        -0.75763
      ],
      [
        -0.53495,
        -0.4987,
        0.41055,
        -0.51845,
        0.36621,
        0.85091,
        -0.26818,
        -0.54652
      ],
      [
        -0.43038,
        -0.3939,
        -0.30253,
        -0.28176,
        0.70084,
        0.61857,
        -0.30253,
        -0.07274
      ],
      [
        -0.19397,
        -0.42052,
        -0.72851,
        -0.10335,
        0.83354,
        0.03881,
        -0.44449,
        0.36222
      ],
      [
        -0.00326,
        -0.26357,
        -0.82123,
        0.07646,
        0.74671,
        -0.53949,
        -0.47263,
        0.68959
      ],
      [
        0.23744,
        -0.35799,
        -0.37133,
        0.30502,
        0.44755,
        -0.89235,
        -0.4852,
        0.81225
      ],
      [
        0.47234,
        -0.19447,
        0.24952,
        0.51053,
        0.03419,
        -0.66792,
        -0.45439,
        0.75886
      ],
      [
        0.61609,
        -0.34988,
        0.74465,
        0.72801,
        -0.44199,
        -0.03046,
        -0.42081,
        0.45391
      ],
      [
        0.75259,
        -0.44144,
        0.81392,
        0.83841,
        -0.78719,
        0.53318,
        -0.31846,
        0.09116
      ],
      [
        0.83913,
        -0.39415,
        0.39395,
        0.84511,
        -0.86061,
        0.84499,
        -0.42297,
        -0.29851
      ],
      [
        0.87752,
        -0.44908,
        -0.24768,
        0.87413,
        -0.7199,
        0.58863,
        -0.34096,
        -0.6691
      ]
    ]
  }
]
```

### Turn 2 — `execute_in_remote_docker` (`15602 ms` LLM)
- **Rationale**: The initial output was truncated. Extract only field shapes and compact telemetry summaries to distinguish a baseline fallback from an identifiable chamber correction and determine the calibration and reference-panel axes.

```bash
python3 - <<'PY'
import pickle,json,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
for k,v in x.items():
 print(k,type(v).__name__)
 if isinstance(v,dict):
  for a,b in v.items():
   print(' ',a,[(p,np.shape(q)) for p,q in b.items()] if isinstance(b,dict) else (np.shape(b),str(b)[:150] if np.ndim(b)==0 else np.asarray(b).ravel()[:3].tolist()))
p=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in p if isinstance(p,list) else [p]:
 print('telemetry',[(k,(np.shape(v),str(v)[:350]) if isinstance(v,list) else v) for k,v in r.items()])
a=json.load(open('/holdout_task/public_asset_index.json')); print('index',str(a)[:1300])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation' 2>&1 | head -20
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
d_campaign_short_name', 'facility_init'), ('facility_telemetry_trace', ((24, 8), '[[0.77552, -0.40175, -0.72976, 0.82025, -0.46189, 0.07187, -0.43126, -0.8963], [0.71649, -0.41895, -0.81354, 0.65635, -0.03225, -0.58464, -0.41244, -0.77664], [0.56917, -0.269, -0.42458, 0.43763, 0.37479, -0.82883, -0.35109, -0.53951], [0.35946, -0.48357, 0.27278, 0.29897, 0.70102, -0.65232, -0.27549, -0.09031], [0.23177, -0.34922, 0.72921, 0.05601'))]
index {'files': {'/holdout_task/public_batch.pkl': {'bytes': 55906, 'summary': {'first': {'campaign_diagnostics': None, 'dataset': {'calibration_load_cells': {'bridge_raw': {'dtype': 'float64', 'shape': [80, 4], 'type': 'ndarray'}, 'single_channel_sweeps': {'cmd_stress': 'ndarray', 'lab_id': 'ndarray', 'true_stress': 'ndarray'}, 'true_channel_loads': {'dtype': 'float64', 'shape': [80, 4], 'type': 'ndarray'}}, 'eval_times': {'dtype': 'float64', 'shape': [12], 'type': 'ndarray'}, 'past_dt': {'dtype': 'float64', 'shape': [6], 'type': 'ndarray'}, 'test': {'aux_chamber_telemetry': {'dtype': 'float64', 'shape': [36, 4], 'type': 'ndarray'}, 'coupon_features_raw': {'dtype': 'float64', 'shape': [36, 4], 'type': 'ndarray'}, 'hold_channel_raw': {'dtype': 'float64', 'shape': [36, 4], 'type': 'ndarray'}, 'lab_id': {'dtype': 'int64', 'shape': [36], 'type': 'ndarray'}, 'past_channel_raw': {'dtype': 'float64', 'shape': [36, 6, 4], 'type': 'ndarray'}, 'ref_coupons': {'dtype': 'float64', 'shape': [36, 8], 'type': 'ndarray'}, 'strain_0': {'dtype': 'float64', 'shape': [36], 'type': 'ndarray'}, 'stratum': {'dtype': 'int64', 'shape': [36], 'type': 'ndarray'}}, 'train': {'aux_chamber_telemetry': {'dtype': 'float64', 'shape': [72, 4], 'type': 'ndarray'}, 'coupon_features_raw': {'dtype': 'float64', 'shape': [7
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`29707 ms` LLM)
- **Rationale**: Held-out calibration error will show whether an affine bridge distorts total stress. Reference-panel column summaries will establish the axes for pooled per-lab standardization before fitting the hereditary law.
- **Alternatives Weighed**: `Infer a chamber correction from the facility trace alone` (rejected: There is only one initialization trace and no paired chamber response or prior campaign fit to identify a correction.)

```bash
python3 - <<'PY'
import pickle,json,numpy as np
from scipy.linalg import lstsq
np.set_printoptions(precision=4,suppress=True,linewidth=120)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; r=c['bridge_raw']; y=c['true_channel_loads']; print('cal samples',r[:5],y[:5]); print('sweep',[(k,np.shape(v),np.asarray(v).ravel()[:12]) for k,v in c['single_channel_sweeps'].items()]); rng=np.random.default_rng(863); f=np.array_split(rng.permutation(len(r)),5); i,j=np.triu_indices(4,1)
for tag,a in [('affine',np.c_[np.ones(len(r)),r]),('diagonal quadratic',np.c_[np.ones(len(r)),r,r*r]),('full quadratic',np.c_[np.ones(len(r)),r,r*r,r[:,i]*r[:,j]]),('diagonal cubic',np.c_[np.ones(len(r)),r,r*r,r**3]),('forward affine',np.c_[np.ones(len(r)),y]),('forward diagonal quadratic',np.c_[np.ones(len(r)),y,y*y]),('forward full quadratic',np.c_[np.ones(len(r)),y,y*y,y[:,i]*y[:,j]])]:
 z=r if tag.startswith('forward') else y; e=[]
 for h in f:
  g=np.setdiff1d(np.arange(len(r)),h); e.extend(a[h]@lstsq(a[g],z[g])[0]-z[h]);
 e=np.asarray(e); print(tag,'channel and sum RMSE',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e.sum(1)**2)), 'max total', max(abs(e.sum(1))))
for tag in ('train','test'):
 p=d[tag]; print(tag,'features',p['coupon_features_raw'][:6]); print('reference',p['ref_coupons'][:6]); print('range per lab')
 for l in np.unique(p['lab_id']):
  m=p['lab_id']==l; r0=p['ref_coupons'][m]; print(l, m.sum(), np.mean(r0,0), np.std(r0,0), 'strata',np.unique(p['stratum'][m],return_counts=True))
 print('aux',p['aux_chamber_telemetry'][:7], 'mean/std',np.mean(p['aux_chamber_telemetry'],0),np.std(p['aux_chamber_telemetry'],0))
 print('hold minmax',p['hold_channel_raw'].min(0),p['hold_channel_raw'].max(0),'past minmax',p['past_channel_raw'].min((0,1)),p['past_channel_raw'].max((0,1)))
print('dt,t',d['past_dt'],d['eval_times']); i=json.load(open('/holdout_task/public_asset_index.json')); print('index entries',list(i['files']),{k:v for k,v in i.items() if k!='files'}); print('batch count',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'cauchy irls' | head -8
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel imputation' | head -8
```
- **Container Exit Code**: `0` (`451 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 4 — `execute_in_remote_docker` (`52626 ms` LLM)
- **Rationale**: Implement the specified pooled preprocessing and stress recurrence, then validate output symmetry and hold out each lab to detect material-fit instability or preprocessing errors that could affect deployment.
- **Alternatives Weighed**: `Polynomial bridge calibration` (rejected: All tested quadratic and cubic extensions increased held-out total-stress RMSE.); `Chamber correction inferred from initialization telemetry` (rejected: The archive contains one initialization trace and no paired chamber responses or prior fits.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -24
cat > /holdout_task/batch_solver.py <<'PY'
"""Estimate a two-branch hereditary response from the benchtop coupons."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    parts = (train, test)
    def pooled(key):
        return np.concatenate([np.asarray(p[key]) for p in parts], axis=0)

    labs = pooled('lab_id')
    strata = pooled('stratum')
    refs = pooled('ref_coupons')
    x = pooled('coupon_features_raw').astype(float, copy=True)
    for lab in np.unique(labs):
        m = labs == lab
        panel = refs[m]
        x[m] = (x[m] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = (1.0 / counts[cell])[:, None]
    valid = np.isfinite(x)
    means = np.sum(np.where(valid, x, 0.0) * w, axis=0) / np.sum(valid * w, axis=0)
    return np.where(valid, x, means)


def _bridge(cal):
    raw = np.asarray(cal['bridge_raw'])
    loads = np.asarray(cal['true_channel_loads'])
    return np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))), loads, rcond=None)[0].sum(axis=1)


def _loads(part, bridge):
    return tuple(np.asarray(part[key]) @ bridge[:-1] + bridge[-1]
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(taus, past, hold, x, dt, t):
    a = np.exp(-dt[:, None] / taus)
    b = -np.expm1(-dt[:, None] / taus)
    z = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        z = a[s] * z + b[s] * past[:, s, None]
    gap = hold[:, None] - z
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
                            t[None, :, None] * drift[:, None, :]), axis=2)
    return curve, np.column_stack((gap / taus, drift))


def _fit(past, hold, x, dt, t, curves, rates):
    yc, yr = curves.ravel(), rates.ravel()
    mc, mr = np.isfinite(yc), np.isfinite(yr)
    y = np.concatenate((yc[mc], yr[mr]))
    nc = mc.sum()

    def design(logtaus):
        c, r = _design(np.exp(logtaus), past, hold, x, dt, t)
        return np.vstack((c.reshape(-1, 7)[mc], r[mr]))

    def profile(logtaus):
        a = design(logtaus)
        return a @ np.linalg.lstsq(a, y, rcond=None)[0] - y

    lo, hi = np.log([0.050001, 24.9999])
    fits = [least_squares(profile, np.log(pair), bounds=(lo, hi), max_nfev=75)
            for pair in ((0.20, 1.4), (0.48, 3.8), (1.0, 10.0), (2.0, 18.0))]
    q = min(fits, key=lambda f: f.cost).x
    theta = np.linalg.lstsq(design(q), y, rcond=None)[0]
    p = np.r_[q, theta]
    for _ in range(3):
        err = design(p[:2]) @ p[2:] - y
        def scale(a):
            return max(1e-8, 1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else 1.0
        scales = np.r_[np.full(nc, scale(err[:nc])), np.full(len(y)-nc, scale(err[nc:]))]
        p = least_squares(lambda v: (design(v[:2]) @ v[2:] - y) / scales,
                          p, bounds=(np.r_[lo, lo, np.full(7, -np.inf)],
                                     np.r_[hi, hi, np.full(7, np.inf)]),
                          loss='soft_l1', f_scale=1.5, max_nfev=120).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_batch(dataset):
    train, test = dataset['train'], dataset['test']
    x = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    n = len(train['lab_id'])
    dt, t = (np.asarray(dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train, bridge), x[:n], dt, t,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    # The initialization log has no paired chamber response for this blind batch.
    c, r = _design(taus, *_loads(test, bridge), x[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + c @ theta,
            'initial_rate_pred': r @ theta}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; p,h=m._loads(tr,m._bridge(d['calibration_load_cells'])); x=m._features(tr,te)[:len(h)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,weights=m._fit(p,h,x,dt,t,y,r); print('fit',tau,weights); print('missing per column',np.isnan(tr['coupon_features_raw']).sum(0));
for l in np.unique(tr['lab_id']):
 q=tr['lab_id']!=l; tau,w=m._fit(p[q],h[q],x[q],dt,t,y[q],r[q]); c,z=m._design(tau,p[~q],h[~q],x[~q],dt,t); e=c @ w-y[~q]; er=z @ w-r[~q]; print('held-out lab',l,'taus',tau,'curve abs [p50,p90,max]',np.quantile(abs(e),[.5,.9,1]),'rate abs',np.quantile(abs(er),[.5,.9,1]),'endpoint median',np.median(abs(e[:,-1])))
PY
```
- **Container Exit Code**: `0` (`913 ms`, timed_out=`False`)
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
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 4.4e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}
fit [0.58565269 4.44937826] [ 0.81643811  0.57835808  0.391566    0.19314586 -0.12814082  0.09440882
 -0.15609246]
missing per column [5 6 6 0]
held-out lab 0 taus [0.58557855 4.48117228] curve abs [p50,p90,max] [0.0088718  0.02209067 0.28133053] rate abs [0.00404595 0.00936227 0.01253438] endpoint median 0.009603210129846218
held-out lab 1 taus [0.58588826 4.44655268] curve abs [p50,p90,max] [0.00883776 0.02219404 0.27998705] rate abs [0.00360383 0.0098378  0.01411198] endpoint median 0.009187009655078127
held-out lab 2 taus [0.5861626  4.51707751] curve abs [p50,p90,max] [0.00844437 0.02219392 0.29297983] rate abs [0.00696798 0.01934747 0.02197994] endpoint median 0.007947137935184756
held-out lab 3 taus [0.5846506 4.3271713] curve abs [p50,p90,max] [0.00888324 0.02193118 0.28887861] rate abs [0.0050891  0.01421327 0.01985912] endpoint median 0.008446386328374411

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=618f5207edf6597e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_618f5207 (directory).
```

### Turn 5 — `submit_completed_task` (`15971 ms` LLM)
- **Rationale**: The saved artifact passed clean-process schema and pooled-reference checks. Leave-one-lab-out validation measured both response outputs, and no labeled chamber slice was available to justify changing the specified fallback.
- **Alternatives Weighed**: `Polynomial bridge calibration` (rejected: Every tested polynomial extension increased held-out total-stress RMSE.); `Chamber correction from initialization telemetry` (rejected: The archive contains no paired chamber response or prior solver to identify a correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4094'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It fits the bridge calibration from the supplied sweeps and estimates the two-branch material response from the benchtop data. The public schema and pooled-reference invariance checks passed. Deployment uses the benchtop law because no paired chamber measurements were available to validate a correction.`

### Turn 6 — `execute_in_remote_docker` (`31398 ms` LLM)
- **Rationale**: Check whether missing features, load extremes, or load reversals expose systematic errors masked by pooled metrics. Compare separate noise models on held-out labs, and check the zero-time derivative for a mismatch between strain and initial-rate formulas.
- **Alternatives Weighed**: `Classification, relay, and HJB audits` (rejected: The public task is continuous hereditary regression and contains no class labels, relay state, or PDE operator.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; p,h=m._loads(tr,m._bridge(d['calibration_load_cells'])); x=m._features(tr,te)[:len(h)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; labs=tr['lab_id']; missing=np.isnan(tr['coupon_features_raw']).any(1)
for lab in np.unique(labs):
 q=labs!=lab; tau,w=m._fit(p[q],h[q],x[q],dt,t,y[q],r[q]); c,z=m._design(tau,p[~q],h[~q],x[~q],dt,t); e=c@w-y[~q]; f=z@w-r[~q]; print('lab',lab,'signed medians curve/rate',np.median(e),np.median(f));
 for key,v in [('any missing',missing[~q]),('complete',~missing[~q]),('high hold',h[~q]>=np.quantile(h[q],.75)),('large history reversal',np.ptp(p[~q],axis=1)>=np.quantile(np.ptp(p[q],axis=1),.75))]:
  if v.any(): print(' ',key,int(v.sum()),'median abs early/late, p90 rate',*(round(a,6) for a in [np.median(abs(e[v,:3])),np.median(abs(e[v,-3:])),np.quantile(abs(f[v]),.9)]))
 c0,r0=m._design(tau,p[q],h[q],x[q],dt,t); ec=c0@w-y[q]; er=r0@w-r[q]; sc,sr=[1.4826*np.median(abs(a-np.median(a))) for a in (ec,er)]
 def residual(v):
  cc,rr=m._design(np.exp(v[:2]),p[q],h[q],x[q],dt,t); a=((cc@v[2:]-y[q])/sc).ravel(); b=(rr@v[2:]-r[q])/sr; return np.r_[np.sign(a)*np.sqrt(np.log1p(a*a)),b]
 v=least_squares(residual,np.r_[np.log(tau),w],bounds=(np.r_[np.log([.050001]*2),[-np.inf]*7],np.r_[np.log([24.9999]*2),[np.inf]*7])).x; cc,rr=m._design(np.exp(v[:2]),p[~q],h[~q],x[~q],dt,t)
 print(' baseline vs Cauchy curve/Gaussian rate p90',np.quantile(abs(e),.9),np.quantile(abs(cc@v[2:]-y[~q]),.9),np.quantile(abs(f),.9),np.quantile(abs(rr@v[2:]-r[~q]),.9))
c,z=m._design(np.array([.6,4.4]),p,h,x,dt,np.array([0.,1e-6])); np.testing.assert_allclose(c[:,0],0,atol=1e-12); np.testing.assert_allclose(c[:,1]/1e-6,z,rtol=1e-6,atol=1e-5)
print('zero-time curve and rate derivative agree')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1121 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0003992878139553735 0.0027068972505195354
  any missing 6 median abs early/late, p90 rate 0.010908 0.007298 0.004981
  complete 12 median abs early/late, p90 rate 0.010313 0.010893 0.011395
  high hold 1 median abs early/late, p90 rate 0.005817 0.01015 0.002275
  large history reversal 6 median abs early/late, p90 rate 0.007545 0.009603 0.008339
 baseline vs Cauchy curve/Gaussian rate p90 0.02209066941861515 0.022133064412866332 0.009362269263317671 0.009135188721816848
lab 1 signed medians curve/rate 0.0014637473873804208 -0.00040942314229441834
  any missing 4 median abs early/late, p90 rate 0.011178 0.008259 0.0066
  complete 18 median abs early/late, p90 rate 0.009667 0.00939 0.010737
  high hold 11 median abs early/late, p90 rate 0.0102 0.008198 0.009617
  large history reversal 2 median abs early/late, p90 rate 0.009495 0.012201 0.009138
 baseline vs Cauchy curve/Gaussian rate p90 0.022194037298697334 0.022393236630121324 0.009837798796052157 0.01008150040919409
lab 2 signed medians curve/rate -0.002312796700116637 0.0009363536306512543
  any missing 3 median abs early/late, p90 rate 0.007881 0.009054 0.009168
  complete 10 median abs early/late, p90 rate 0.006409 0.007101 0.021165
  high hold 2 median abs early/late, p90 rate 0.007158 0.008501 0.009059
  large history reversal 3 median abs early/late, p90 rate 0.002339 0.006464 0.018602
 baseline vs Cauchy curve/Gaussian rate p90 0.022193915698025313 0.02205067953078177 0.019347466021072898 0.01948790570424493
lab 3 signed medians curve/rate 0.0011522278016466991 -0.0005662985709333768
  any missing 4 median abs early/late, p90 rate 0.011213 0.006007 0.017276
  complete 15 median abs early/late, p90 rate 0.006691 0.011147 0.011561
  high hold 5 median abs early/late, p90 rate 0.006181 0.013989 0.012458
  large history reversal 8 median abs early/late, p90 rate 0.006635 0.011389 0.010136
 baseline vs Cauchy curve/Gaussian rate p90 0.021931182408374654 0.02167842293666622 0.01421327324034509 0.014497211981606025
zero-time curve and rate derivative agree
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 4.4e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=618f5207edf6597e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_618f5207 (directory).
```

### Turn 7 — `submit_completed_task` (`15650 ms` LLM)
- **Rationale**: The final artifact passed clean-process verification. Audits of missing features, high loads, and load reversals did not establish a reason to change the model, and the strain derivative agrees with the initial-rate formula.
- **Alternatives Weighed**: `Cauchy loss for strain and Gaussian loss for rate` (rejected: Held-out lab performance was mixed, with no improvement in either worst-lab p90 metric.); `Chamber correction from initialization telemetry` (rejected: The archive contains no paired chamber response or prior fit that identifies a correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4094'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema, pooled-reference invariance, and zero-time strain–rate consistency checks passed. Worst held-out-lab 90th-percentile absolute errors were 0.0222 for strain and 0.0193 for initial rate. Deployment uses the documented benchtop fallback because no paired chamber measurements were available to validate a correction.`

### Turn 8 — `execute_in_remote_docker` (`45596 ms` LLM)
- **Rationale**: The audit checks whether missing nonlinear terms or between-lab drift materially worsen either deliverable. Challenger selection stays inside the outer lab folds; change the solver only if the held-out results support it.
- **Alternatives Weighed**: `Add linear pairwise differences as new regressors` (rejected: They are already in the span of the unrestricted linear columns; the audit instead tests nonlinear terms that can change predictions.); `Run relay, HJB, or balanced-accuracy tests` (rejected: There are no relay states, PDEs, or class labels. Held-out labs and within-lab drift estimation test the relevant nuisance mechanism.)

```bash
cat >/tmp/audit_heredity.py <<'PY'
import importlib.util, json, pickle
import numpy as np
from scipy.optimize import least_squares

s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']; tr = d['train']; te = d['test']
lab = tr['lab_id']; labs = np.unique(lab)
dt, t = d['past_dt'], d['eval_times']
p, h = m._loads(tr, m._bridge(d['calibration_load_cells']))
x = m._features(tr, te)[:len(h)]
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]; r = tr['instant_rate_obs']

def fit(mask):
    tau, w = m._fit(p[mask], h[mask], x[mask], dt, t, y[mask], r[mask])
    c, z = m._design(tau, p, h, x, dt, t)
    return tau, c, z, c @ w - y, z @ w - r

def bases(tau, c, z, mask, name):
    if name == 'saturation':
        gap = z[:, :2] * tau
        unit = np.maximum(np.sqrt(np.mean(gap[mask]**2, axis=0)), 1e-12)
        f = 1 / (1 + (gap / unit)**2) - 1
        return c[:, :, :2] * f[:, None, :], z[:, :2] * f
    if name == 'interactions':
        i, j = np.triu_indices(x.shape[1], 1)
        f = np.column_stack((x[:, i] * x[:, j], (h-p[:, -1]) * x[:, 0] * x[:, 1]))
        return t[None, :, None] * f[:, None, :], f
    if name == 'lab_centered':
        g = np.column_stack((np.ones(len(h)), x))
        for l in labs:
            q = mask & (lab == l)
            if q.any():
                g[q] -= g[q].mean(axis=0)
        return t[None, :, None] * g[:, None, :], g
    raise ValueError(name)

def predict(fit_result, train, test, name):
    tau, c, z, ec, er = fit_result
    if name == 'baseline':
        return ec[test], er[test]
    u, v = bases(tau, c, z, train, name)
    def scale(a):
        return 1.4826 * np.median(abs(a-np.median(a)))
    sc, sr = scale(ec[train]), scale(er[train]); k = v.shape[1]
    if name == 'lab_centered':
        # Fit slopes within each training lab, then set a shared intercept.
        # Never estimate a nuisance offset from a held-out lab's responses.
        ec, er = ec.copy(), er.copy()
        for l in labs:
            q = train & (lab == l)
            if q.any():
                ec[q] -= np.median(ec[q], axis=0)
                er[q] -= np.median(er[q])
    a = np.vstack((u[train].reshape(-1, k)/sc, v[train]/sr))
    b = np.r_[ec[train].ravel()/sc, er[train]/sr]
    w = least_squares(lambda w: np.r_[a @ w+b, 2*w], np.zeros(k), loss='soft_l1', f_scale=1.5).x
    return ec[test]+u[test]@w, er[test]+v[test]@w

def metrics(errors):
    ec, er = errors
    return np.array([np.quantile(abs(ec), .9), np.quantile(abs(er), .9)])

names = ('baseline', 'saturation', 'interactions', 'lab_centered')
results = {n: [] for n in (*names, 'nested')}
cache = {}
for l in labs:
    trn = lab != l; tst = ~trn
    inner = {n: [] for n in names}
    for hld in labs[labs != l]:
        a = trn & (lab != hld); b = lab == hld
        key = tuple(sorted((l,hld)))
        if key not in cache: cache[key] = fit(a)
        ref = metrics(predict(cache[key], a, b, 'baseline'))
        for name in names:
            inner[name].append(metrics(predict(cache[key], a, b, name))/ref)
    selected = min(names, key=lambda n: np.max(inner[n]))
    basefit = fit(trn)
    for name in names:
        results[name].append(metrics(predict(basefit, trn, tst, name)))
    results['nested'].append(metrics(predict(basefit, trn, tst, selected)))
    print('outer lab', l, 'selected', selected, 'baseline signed median strain/rate', np.median(basefit[3][tst]), np.median(basefit[4][tst]))
for name, rows in results.items():
    print(name, 'per lab p90', np.round(rows, 6).tolist(), 'worst', np.max(rows, axis=0))
print('archive prior fit count:', len(json.load(open('/tmp/hegel_continual_archive/archive_manifest.json'))), 'chamber observations:', item.get('campaign_diagnostics') is not None)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_heredity.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1151 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
outer lab 0 selected baseline baseline signed median strain/rate -0.0003992878139553735 0.0027068972505195354
outer lab 1 selected baseline baseline signed median strain/rate 0.0014637473873804208 -0.00040942314229441834
outer lab 2 selected baseline baseline signed median strain/rate -0.002312796700116637 0.0009363536306512543
outer lab 3 selected baseline baseline signed median strain/rate 0.0011522278016466991 -0.0005662985709333768
baseline per lab p90 [[0.022091, 0.009362], [0.022194, 0.009838], [0.022194, 0.019347], [0.021931, 0.014213]] worst [0.02219404 0.01934747]
saturation per lab p90 [[0.022061, 0.009307], [0.022186, 0.009746], [0.022177, 0.019335], [0.021955, 0.014012]] worst [0.02218604 0.01933464]
interactions per lab p90 [[0.022303, 0.009633], [0.022318, 0.009887], [0.02129, 0.01953], [0.022813, 0.01423]] worst [0.02281258 0.01953028]
lab_centered per lab p90 [[0.022057, 0.009264], [0.022086, 0.009789], [0.022183, 0.019351], [0.022133, 0.014166]] worst [0.02218254 0.01935141]
nested per lab p90 [[0.022091, 0.009362], [0.022194, 0.009838], [0.022194, 0.019347], [0.021931, 0.014213]] worst [0.02219404 0.01934747]
archive prior fit count: 0 chamber observations: False
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 4.4e-09, "ref_swap_rate_max_diff": 1.4e-09, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=618f5207edf6597e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_618f5207 (directory).
```

### Turn 9 — `submit_completed_task` (`15539 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated verification, and nested held-out-lab tests found no material bottleneck that justified replacing the specified model.
- **Alternatives Weighed**: `Rational saturation, feature interactions, or within-lab drift correction` (rejected: None was selected in any inner validation fold; the small outer-slice gains did not establish a material improvement.); `Chamber correction from telemetry` (rejected: The initialization trace lacks paired chamber responses, and there is no prior fit to identify a correction.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema, pooled-reference invariance, and zero-time strain–rate consistency checks passed. Worst held-out-lab 90th-percentile absolute errors were 0.0222 for strain and 0.0193 for initial rate. Deployment uses the documented benchtop fallback because no paired chamber measurements were available to identify a correction.`

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
