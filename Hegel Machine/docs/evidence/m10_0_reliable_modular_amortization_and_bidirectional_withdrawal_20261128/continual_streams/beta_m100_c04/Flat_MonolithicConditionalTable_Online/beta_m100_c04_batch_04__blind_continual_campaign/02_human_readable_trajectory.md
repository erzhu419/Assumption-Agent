# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c04_batch_04__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `290,663` (`279,350` / `102,784` / `11,313`) |
| **Total Episode Wall Time (s)** | `356.29s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16156 ms` LLM)
- **Rationale**: Inspect the public schema and campaign evidence to determine whether the open-wall stencil applies and which calibration and solver paths need testing.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py /holdout_task/public_asset_index.json; python3 - <<'PY'
import pickle,json,pathlib
with open('/holdout_task/public_batch.pkl','rb') as f: b=pickle.load(f)
print('batch',type(b),len(b)); i=b[0]; print('item keys',list(i));
for k,v in i['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else (v if isinstance(v,(str,int,float,bool,type(None))) else ''))
d=i.get('campaign_diagnostics'); print('diagnostics',type(d).__name__,len(d) if isinstance(d,list) else d)
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
 x=json.loads(p.read_text()); print('\n',p.name, str(x)[:14000])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1
```
- **Container Exit Code**: `0` (`406 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
88, 0.88893, 0.80928], [-0.31101, 0.67728, -0.34017, 0.62438, 2.46304, 0.21575, 0.79996, 0.63892]]}, {'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c04_batch_03', 'facility_telemetry_trace': [[-0.22962, 0.10719, 2.64951, -0.66575, -0.29263, 0.40309, -0.19864, -0.70584], [-0.02714, -0.38235, 2.52797, -0.83542, -0.22378, 0.78042, 0.10354, -0.93853], [0.22646, -0.68273, 2.50594, -0.87243, -0.28964, 0.79855, 0.29171, -0.79466], [0.36455, -0.88172, 2.52261, -0.82955, -0.43045, 0.19829, 0.48034, -0.42248], [0.57707, -0.78548, 2.60233, -0.79056, -0.30209, -0.41606, 0.64489, 0.01113], [0.76938, -0.47565, 2.66704, -0.62245, -0.35926, -0.81412, 0.82031, 0.37089], [0.86695, -0.0708, 2.59059, -0.47257, -0.38055, -0.70554, 0.88586, 0.68737], [0.81472, 0.38788, 2.57922, -0.26049, -0.29953, -0.25393, 0.86316, 0.86368], [0.80308, 0.71976, 2.40784, -0.00756, -0.3648, 0.41371, 0.79628, 0.77768], [0.74868, 0.83598, 2.43002, 0.25987, -0.41101, 0.82442, 0.73812, 0.52223], [0.5921, 0.79589, 2.56827, 0.39033, -0.22266, 0.7655, 0.57381, 0.05583], [0.39411, 0.54668, 2.56686, 0.50397, -0.30978, 0.23831, 0.41312, -0.47027], [0.22963, 0.0832, 2.55486, 0.70594, -0.42936, -0.42088, 0.19668, -0.69147], [0.00833, -0.34502, 2.53056, 0.81721, -0.25927, -0.82373, -0.06761, -0.87024], [-0.1886, -0.59688, 2.54233, 0.84162, -0.37371, -0.72904, -0.30898, -0.72965], [-0.39126, -0.87756, 2.54609, 0.77485, -0.32311, -0.22312, -0.46586, -0.44395], [-0.58318, -0.76802, 2.422, 0.79196, -0.4078, 0.44582, -0.63657, -0.0102], [-0.70057, -0.53081, 2.67716, 0.616, -0.20167, 0.80214, -0.80791, 0.33006], [-0.82044, -0.04824, 2.59708, 0.45429, -0.46884, 0.77102, -0.84406, 0.74657], [-0.8878, 0.39077, 2.63159, 0.20684, -0.33795, 0.16575, -0.83, 0.82312], [-0.78931, 0.68047, 2.67807, 0.04801, -0.45772, -0.38758, -0.80075, 0.71674], [-0.70692, 0.80085, 2.40011, -0.15864, -0.42787, -0.80076, -0.7099, 0.46031], [-0.60813, 0.77741, 2.55478, -0.35213, -0.26968, -0.7772, -0.57405, 0.02286], [-0.41648, 0.47626, 2.50471, -0.58124, -0.29767, -0.20017, -0.4138, -0.38212]]}]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`21245 ms` LLM)
- **Rationale**: Identify the archived boundary and calibration variants before fitting them. Confirm tracer availability, since that determines whether boundary selection can rely on direct measurements or must use telemetry.

```bash
cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /tmp/hegel_continual_archive/beta_m100_c04_batch_03_solver.py; grep -nE 'wall|robin|bank|calib|telemetry' /tmp/hegel_continual_archive/*_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
b=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('Batch length',len(b))
for i in b:
 s=i['study']; d=i.get('campaign_diagnostics'); print('diag', type(d).__name__,len(d) if isinstance(d,list) else d); print('study',[(k,getattr(v,'shape',None),v if np.isscalar(v) else list(v) if isinstance(v,dict) else '') for k,v in s.items()]); print('calibration',[(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else '') for k,v in s['sensor_calibration_sweeps'].items()])
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log keys',list(a) if isinstance(a,dict) else type(a));
if isinstance(a,dict):
 for k,v in a.items():
  if isinstance(v,list) and v and isinstance(v[0],dict):
   for row in v:
    print(k,{j:(np.round(np.mean(x,axis=0),3).tolist() if j.endswith('_trace') else x) for j,x in row.items()})
  else: print(k,str(v)[:1000])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`504 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_ridge', None, 0.04), ('u_box', None, 0.42), ('total_budget', None, 1.4)]
calibration [('ref_rosette', 'ndarray', (80, 4), ''), ('raw_rosette', 'ndarray', (80, 4), ''), ('single_bank_sweeps', 'dict', None, ['bank_id', 'ref_signal', 'raw_readout'])]
log keys <class 'list'>
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
          "op": ">=",
          "value": 0
        },
        {
          "field": "eps",
          "op": ">",
          "value": 0
        },
        {
          "field": "max_leaves",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "solve_and_certify_l0",
          "signature": "solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict",
          "doc_summary": "Certify boxed sparse ridge regression with at most 16 real predictors."
        }
      ],
      "index_cards": [
        "autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_sparse_ridge_certification",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`20488 ms` LLM)
- **Rationale**: Recover the exact archived boundary equations and check whether observation amplitudes fall inside the calibration sweeps. This determines which wall variants to compare on held-out pulses and whether sensor extrapolation needs separate validation.
- **Alternatives Weighed**: `Select the boundary model from telemetry alone` (rejected: Three current-campaign tracer pulses are available to test the wall operator directly.)

```bash
for f in /tmp/hegel_continual_archive/*_solver.py; do echo ==== "$f"; sed -n '/^def _axis/,/^def _project/p' "$f" | head -85; done; sed -n '1,58p' /tmp/hegel_continual_archive/beta_m100_c04_batch_03_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; c=s['sensor_calibration_sweeps']; print('calibration ranges',np.quantile(c['ref_rosette'],[0,.01,.5,.99,1]),np.quantile(c['raw_rosette'],[0,.01,.5,.99,1])); print('bank ranges',[(k,np.shape(v),np.min(v),np.max(v)) for k,v in c['single_bank_sweeps'].items()]); print('tracers',[(d['w_pulse'].tolist(),np.min(d['y_pulse_raw']),np.max(d['y_pulse_raw'])) for d in i['campaign_diagnostics']]); print('obs',np.min(s['y_obs_raw']),np.max(s['y_obs_raw'])); print('bank order',s['row_bank_id'][:16]);
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:(np.round(np.mean(v,axis=0),3).tolist() if k.endswith('_trace') else str(v)[:200]) for k,v in row.items()})
PY
```
- **Container Exit Code**: `0` (`362 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
c = _sensor_fit(study['sensor_calibration_sweeps'])
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    def forward(x):
        return np.stack([_sensor_features(x, k) @ c[k] for k in range(4)], axis=-1)
    linear = np.stack([forward(e) - c[:, 0] - c[:, 5] * e for e in np.eye(4)], axis=1)
    init = np.linalg.solve(linear, (observed - c[:, 0]).T).T
    calibrated = []
    for y, x0 in zip(observed, init):
        sol = root(lambda x: forward(x) - y, x0, tol=1e-11)
        if np.linalg.norm(sol.fun, np.inf) > 1e-9:
            raise RuntimeError('rosette inversion did not converge')
        calibrated.append(sol.x)
    return np.asarray(calibrated).ravel()


def _axis(kappa, vel, h, leakage):
    n = len(kappa)
    face = (kappa[1:] + kappa[:-1]) * (0.5 / h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(vel[1:-1], 0) / h
    upper[1:] -= np.maximum(-vel[1:-1], 0) / h
    upper[0] = -kappa[0] / h**2
calibration ranges [-1.38506264 -1.34790083  0.          1.75856909  1.77744152] [-13.58510137 -12.71999702   0.02018055  12.5533496   13.59741422]
bank ranges [('bank_id', (48,), 0, 3), ('ref_signal', (48,), -1.3, 1.5), ('raw_readout', (48,), -2.1340965535841327, 14.427893956164436)]
tracers [([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], -0.027872945241891968, 0.4935111149817031), ([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], -0.05256800695727481, 0.5932793864618259), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0], -0.0696853088403248, 0.513089841265716)]
obs -0.27896712009709634 0.8724201183107407
bank order [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
{'completed_campaign_index': '0', 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': [-0.008, -0.002, -0.423, -0.004, -0.363, 0.0, -0.002, -0.01]}
{'completed_campaign_index': '1', 'completed_campaign_short_name': 'beta_m100_c04_s00', 'facility_telemetry_trace': [0.006, -0.002, -0.397, 0.008, 2.459, 0.01, 0.005, -0.003]}
{'completed_campaign_index': '2', 'completed_campaign_short_name': 'beta_m100_c04_batch_01', 'facility_telemetry_trace': [0.007, 0.001, -0.418, -0.002, -0.349, -0.001, 0.003, -0.002]}
{'completed_campaign_index': '3', 'completed_campaign_short_name': 'beta_m100_c04_batch_02', 'facility_telemetry_trace': [0.002, -0.011, -0.392, 0.003, 2.453, -0.004, -0.01, 0.002]}
{'completed_campaign_index': '4', 'completed_campaign_short_name': 'beta_m100_c04_batch_03', 'facility_telemetry_trace': [0.002, 0.007, 2.552, -0.002, -0.339, 0.002, 0.006, -0.003]}
```

### Turn 4 — `execute_in_remote_docker` (`43630 ms` LLM)
- **Rationale**: Compare exact open and closed wall combinations first. Fit continuous wall terms on two pulses and predict the third to distinguish missing boundary physics from calibration error or overfitting.
- **Alternatives Weighed**: `Copy the previous campaign solver unchanged` (rejected: The current sweeps and tracer pulses can distinguish a transferable operator from a campaign-specific setting.)

```bash
for f in /tmp/hegel_continual_archive/*_solver.py; do echo ====; grep -A9 '^def _forward' "$f"; grep -A6 '^def _sensor_features' "$f"; done
cat >/tmp/wallstudy.py <<'PY'
import pickle,sys,itertools
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c04_batch_03_solver as arc
from scipy import sparse
from scipy.sparse.linalg import splu
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; pulses=a['campaign_diagnostics']; cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; good=np.max(np.abs(raw),axis=1)<8
bank=cal['single_bank_sweeps']; ids=bank['bank_id']; bx=np.eye(4)[ids]*bank['ref_signal'][:,None]; by=bank['raw_readout']
def feat(x,k,kind):
 v=x[...,k]; p=x[...,(k-1)%4]; n=x[...,(k+1)%4]; o=x[...,(k+2)%4]; f=[np.ones_like(v),v,p,n,v*n,v**3]
 if kind>=1:f += [o,v*p,v*o,v*v,p*n,p*o,n*o,p*p,n*n,o*o]
 if kind>=2:f += [x[...,i]*x[...,j]*x[...,l] for i,j,l in itertools.combinations_with_replacement(range(4),3) if (i,j,l)!=(k,k,k)]
 return np.stack(f,axis=-1)
def fit(kind,mask):
 return [np.linalg.lstsq(np.vstack((feat(ref[mask],k,kind),feat(bx[(ids==k)&(np.abs(by)<8)],k,kind))),np.r_[raw[mask,k],by[(ids==k)&(np.abs(by)<8)]],rcond=None)[0] for k in range(4)]
def pred(x,kind,c):return np.stack([feat(x,k,kind)@c[k] for k in range(4)],axis=-1)
for kind in range(3):
 errs=[]
 for t in np.array_split(np.random.default_rng(2).permutation(np.flatnonzero(good)),5):
  mask=good.copy(); mask[t]=False; errs.append(np.sqrt(np.mean((pred(ref[t],kind,fit(kind,mask))-raw[t])**2)))
 print('sensor',kind,'CV',np.round(errs,7),'coeff base',np.array(fit(kind,good))[:,:6].round(5).tolist())
nx,ny=s['nx'],s['ny']; dx=s['Lx']/(nx-1); dy=s['Ly']/(ny-1); rob=np.repeat([s['robin_x'],s['robin_y']],2); ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; eye=sparse.eye(nx*ny,format='csc'); vdiag=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); w=np.array([p['w_pulse'] for p in pulses]); yy=np.array([p['y_pulse_raw'] for p in pulses]).reshape(3,-1,4)
def forward(z,q=(1,1,1,1)):
 ax=arc._axis(s['kappa_x'],s['vel_x'],dx,z[:2]*rob[:2]); ay=arc._axis(s['kappa_y'],s['vel_y'],dy,z[2:]*rob[2:]);
 def rescale(M,f):
  M=M.tolil(); n=M.shape[0]
  for a,b,t in [(0,1,f[0]),(n-1,n-2,f[1])]:
   d=M[a,b]*(t-1); M[a,b]+=d; M[a,a]-=d
  return M.tocsc()
 L=sparse.kron(sparse.eye(ny),rescale(ax,q[:2]),format='csc')+sparse.kron(rescale(ay,q[2:]),sparse.eye(nx),format='csc')+vdiag
 return np.vstack([splu(L+u*eye).solve(rhs)[rows] for u in s['mu_shifts']])
def residual(z,kind,c):return pred((w@forward(z[:4],z[4:] if len(z)>4 else (1,)*4).T).reshape(3,-1,4),kind,c)-yy
if __name__=='__main__':
 for kind in (0,1,2):
  c=fit(kind,good); ranks=[]
  for z in itertools.product((0.,1.),repeat=4):
   e=residual(np.array(z),kind,c); ranks.append((np.sqrt(np.mean(e**2)),z,np.sqrt(np.mean(e**2,axis=(1,2))).round(7)))
  print('walls',kind,sorted(ranks)[:4]);
  for n in (4,8):
   for test in (-1,0,1,2):
    take=np.arange(3)!=test; x0=np.r_[sorted(ranks)[0][1],np.ones(n-4)]; sol=least_squares(lambda z:residual(z,kind,c)[take].ravel(),x0,bounds=(np.r_[[-.2]*4,[.2]*(n-4)],np.r_[[4.]*4,[4.]*(n-4)]),diff_step=1e-4)
    e=residual(sol.x,kind,c); print('fitted',kind,n,test,sol.x.round(6),np.sqrt(np.mean(e**2,axis=(1,2))).round(7))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wallstudy.py
```
- **Container Exit Code**: `0` (`5143 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4.        4.        0.279982
  0.335901] [0.0351177 0.0306586 0.0553952]
walls 1 [(0.04269550159708488, (0.0, 0.0, 0.0, 0.0), array([0.0287721, 0.0619227, 0.0283984])), (0.05156891998231714, (0.0, 0.0, 0.0, 1.0), array([0.0561255, 0.0605677, 0.0340521])), (0.056296413152506095, (0.0, 1.0, 0.0, 0.0), array([0.0534125, 0.0704703, 0.0410962])), (0.06256127495615187, (1.0, 0.0, 0.0, 0.0), array([0.0712553, 0.0656091, 0.0485786]))]
fitted 1 4 -1 [ 0.23736   0.037534 -0.054973 -0.092156] [0.0422942 0.033237  0.0149742]
fitted 1 4 0 [ 0.271814 -0.181052 -0.020144 -0.035587] [0.0716674 0.0206328 0.0141737]
fitted 1 4 1 [ 0.021698  0.051899  0.030554 -0.057426] [0.0286573 0.0613793 0.0245681]
fitted 1 4 2 [ 0.267851  0.091336 -0.098105 -0.105485] [0.0406189 0.03452   0.0197576]
fitted 1 8 -1 [ 0.26815   0.197216 -0.052431 -0.080085  1.266081  4.        0.597808
  0.311115] [0.0414227 0.0290016 0.0143943]
fitted 1 8 0 [ 0.111669 -0.2      -0.043935 -0.05408   0.487007  1.957736  0.609955
  0.508626] [0.0595805 0.019657  0.011917 ]
fitted 1 8 1 [-0.039691  0.369604  0.258426 -0.043533  0.502598  4.        4.
  0.710927] [0.026914  0.081101  0.0224286]
fitted 1 8 2 [ 0.920682  0.41781  -0.100894 -0.084911  4.        4.        0.279987
  0.335907] [0.0351176 0.0306584 0.0553946]
walls 2 [(0.042696023362392956, (0.0, 0.0, 0.0, 0.0), array([0.0287717, 0.0619235, 0.0283992])), (0.05156919836030532, (0.0, 0.0, 0.0, 1.0), array([0.0561254, 0.0605681, 0.0340527])), (0.05629684290784131, (0.0, 1.0, 0.0, 0.0), array([0.0534125, 0.070471 , 0.0410968])), (0.06256127779629667, (1.0, 0.0, 0.0, 0.0), array([0.0712551, 0.0656092, 0.0485786]))]
fitted 2 4 -1 [ 0.237365  0.037532 -0.054972 -0.092157] [0.0422942 0.033237  0.0149744]
fitted 2 4 0 [ 0.271818 -0.181054 -0.020142 -0.035587] [0.071668  0.0206327 0.0141738]
fitted 2 4 1 [ 0.0217    0.051898  0.030556 -0.057426] [0.0286573 0.0613793 0.0245683]
fitted 2 4 2 [ 0.267855  0.091334 -0.098103 -0.105486] [0.0406189 0.03452   0.0197577]
fitted 2 8 -1 [ 0.268174  0.197213 -0.05243  -0.080085  1.266158  4.        0.597815
  0.311127] [0.0414226 0.0290016 0.0143946]
fitted 2 8 0 [ 0.111678 -0.2      -0.043934 -0.05408   0.487028  1.957671  0.60997
  0.508647] [0.059581  0.019657  0.0119171]
fitted 2 8 1 [-0.03969   0.369601  0.258436 -0.043533  0.502602  4.        4.
  0.710977] [0.026914  0.0811011 0.0224288]
fitted 2 8 2 [ 0.920697  0.417804 -0.100894 -0.084911  4.        4.        0.279991
  0.335916] [0.0351177 0.0306584 0.0553939]
```

### Turn 5 — `execute_in_remote_docker` (`22795 ms` LLM)
- **Rationale**: Test whether the residual falls in the measured nuisance subspace before changing the boundary physics. If profiling does not remove it, use its spatial and shift structure to identify the missing stencil terms.
- **Alternatives Weighed**: `Use the continuously fitted wall coefficients` (rejected: The coefficients vary substantially across held-out pulses, and the worst prediction is worse than the closed-wall baseline.)

```bash
cat >/tmp/pulsecheck.py <<'PY'
from wallstudy import *
from scipy.optimize import root
c=fit(0,good); print('cal rmse',np.sqrt(np.mean((pred(ref[good],0,c)-raw[good])**2)))
y=np.array([[root(lambda x: pred(x,0,c)-r,r).x for r in p] for p in yy]).reshape(3,-1)
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('Z characteristics',Z[:6].round(4),np.linalg.svd(Z,compute_uv=False)); print('aux mean',s['aux_manifold_telemetry'].mean(0)); print('sensor coords',ij.tolist()); print('pulses keys',[list(p) for p in pulses]);
def resid(z):return w@forward(z[:4],z[4:] if len(z)>4 else (1,)*4).T-y
def project(r):return r-(r@Q)@Q.T
def rm(r):return np.sqrt(np.mean(r*r,axis=-1)).round(8)
for z in [(0,)*4,(1,)*4]:
 r=resid(z); print('wall',z,'plain',rm(r),'profiled',rm(project(r)),'Z coeff',np.linalg.lstsq(Z,r.T,rcond=None)[0].T.round(5)); print('plain shift',rm(r.reshape(3,3,-1))); print('projected extremes',np.max(np.abs(project(r)),1));
for n in (4,8):
 for test in (-1,0,1,2):
  take=np.arange(3)!=test
  sol=least_squares(lambda z:project(resid(z))[take].ravel(),np.r_[[.1]*4,[1.]*(n-4)],bounds=(np.r_[[-.5]*4,[.1]*(n-4)],np.r_[[5.]*4,[8.]*(n-4)]),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('profile fit',n,test,sol.x.round(7),rm(project(resid(sol.x))))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/pulsecheck.py
```
- **Container Exit Code**: `0` (`2925 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
38, -0.2816, 0.23039], [0.07, 1.3, 0.182, -0.36399, -0.28599, 0.234], [-0.06, 0.79, 0.1106, -0.2212, -0.1738, 0.1422]]
cal rmse 9.561190934068857e-06
Z characteristics [[ 2.033  -0.0723 -0.7287]
 [ 0.7271 -1.4237 -1.6985]
 [ 0.1575  0.6075  0.9562]
 [-0.1224  2.2199  1.2293]
 [-0.1842 -1.4978  2.1334]
 [ 0.5003 -0.0772  0.9679]] [11.3574175   9.88043021  8.95595245]
aux mean [ 0.16068444 -0.08183605  0.25918203 -0.07514852]
sensor coords [[6, 4], [2, 10], [4, 6], [10, 10], [0, 6], [2, 6], [8, 6], [4, 8], [10, 2], [6, 10], [8, 4], [2, 8], [12, 8], [12, 4], [4, 10], [10, 8], [8, 8], [0, 0], [6, 6], [0, 8], [12, 0], [10, 4], [6, 0], [12, 6], [2, 4], [4, 0], [2, 0], [10, 6], [0, 4], [2, 2], [8, 2], [8, 0]]
pulses keys [['w_pulse', 'y_pulse_raw'], ['w_pulse', 'y_pulse_raw'], ['w_pulse', 'y_pulse_raw']]
wall (0, 0, 0, 0) plain [0.0279858  0.05908801 0.02485091] profiled [0.02753868 0.05813379 0.02446252] Z coeff [[-0.00077 -0.00172  0.00425]
 [ 0.00126  0.00343 -0.0092 ]
 [ 0.00013 -0.00018 -0.00447]]
plain shift [[0.03801774 0.02494075 0.01679958]
 [0.08141018 0.05191465 0.03393273]
 [0.03363393 0.02237561 0.01485915]]
projected extremes [0.09474533 0.20120785 0.09212222]
wall (1, 1, 1, 1) plain [0.12005755 0.11316958 0.10495074] profiled [0.11755679 0.11086869 0.10310321] Z coeff [[-0.02268 -0.00903 -0.00408]
 [-0.02152 -0.00582 -0.0064 ]
 [-0.01935 -0.00144 -0.00324]]
plain shift [[0.18857371 0.07885486 0.03825331]
 [0.17912461 0.07226671 0.03337601]
 [0.16095498 0.07449763 0.03984428]]
projected extremes [0.26057373 0.23256819 0.2729542 ]
profile fit 4 -1 [ 0.2407588  0.0463321 -0.0679377 -0.1010512] [0.04001815 0.03044024 0.01360069]
profile fit 4 0 [ 0.2790806 -0.1669172 -0.0271099 -0.045009 ] [0.06493934 0.0198671  0.01362812]
profile fit 4 1 [ 0.0169925  0.0399101  0.0246377 -0.0530623] [0.02667891 0.05944562 0.02277247]
profile fit 4 2 [ 0.2722034  0.1021164 -0.1104171 -0.1173279] [0.03898142 0.03110558 0.01781943]
profile fit 8 -1 [ 0.3912806  0.4500154 -0.0566676 -0.0858151  1.7130967  8.
  0.6682885  0.3907479] [0.03905655 0.02756519 0.01348712]
profile fit 8 0 [ 0.1089979 -0.5       -0.0383339 -0.0485092  0.470145   5.2654248
  0.7457975  0.6650775] [0.05706502 0.01777293 0.01217888]
profile fit 8 1 [-0.0285654  0.7508985  0.579507  -0.0500962  0.6701004  8.
  8.         0.6773136] [0.02531336 0.07446462 0.02179933]
profile fit 8 2 [ 2.0537736  0.9445142 -0.1070629 -0.095756   8.         8.
  0.2985723  0.4538054] [0.0340731  0.0284361  0.05093534]
```

### Turn 6 — `execute_in_remote_docker` (`29573 ms` LLM)
- **Rationale**: If a discrete boundary pairing predicts the third pulse from a choice made on two, use that stencil in the deliverable. If none does, inspect the signed residuals for the missing coupling.
- **Alternatives Weighed**: `Use freely fitted leakage and wall diffusion` (rejected: It worsens worst-pulse prediction relative to the simpler closed-wall stencil.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived local wall stencil plus the measured nuisance directions accounts for the current tracer response.` | Witness: `Calibration cross-validation RMSE is at most 1.15e-5, but nuisance projection only reduces the worst closed-wall tracer RMSE from 0.05909 to 0.05813. Independently fitting local wall leakage and face diffusion gives a worst held-out RMSE of 0.07446.` | Blanket Diagnosis: `Neither sensor calibration nor the supplied nuisance directions explains the shift-dependent tracer discrepancy.` | Sublated Pivot: `Test exact wraparound, reflection, and wall sink stencils on held-out pulses.`

```bash
cat >/tmp/wrapcheck.py <<'PY'
from pulsecheck import y,rm
from wallstudy import *

def axis(k,u,h,r,typ):
 n=len(k); A=arc._axis(k,u,h,(0,0)).tolil(); name,ff,ad=typ
 for i,j,p in [(0,1,n-1),(n-1,n-2,0)]:
  out = u[i] if i==n-1 else -u[i]
  if name=='open': A[i,i]+=r/h
  if name=='sink': A[i,i]+=ff*k[i]/h**2
  if name=='reflect':
   A[i,j]*=ff; A[i,i]=-A[i,j]
  if name in ('wrap','wrap_full','wrap_rep'):
   a=k[i] if ff==0 else (k[i]+k[p])/2
   if name=='wrap_full': A[i,j]=-(k[i]+k[j])/(2*h*h)
   if name=='wrap_rep': A[i,j]=0
   A[i,p]=-a/(h*h)
   if ad:
    A[i,j]-=max(out,0)/h; A[i,p]-=max(-out,0)/h
   A[i,i]=-A[i,j]-A[i,p]
 return A.tocsc()

def fw(tx,ty):
 A=axis(s['kappa_x'],s['vel_x'],dx,rob[0],tx); B=axis(s['kappa_y'],s['vel_y'],dy,rob[2],ty)
 L=sparse.kron(sparse.eye(ny),A,format='csc')+sparse.kron(B,sparse.eye(nx),format='csc')+vdiag
 return np.vstack([splu(L+u*eye).solve(rhs)[rows] for u in s['mu_shifts']])

types=[('closed',0,0),('open',0,0),('sink',1,0),('sink',2,0),('reflect',2,0),('reflect',.5,0)]+list(itertools.product(['wrap','wrap_full','wrap_rep'],[0,1],[0,1]))
a=[]
for tx,ty in itertools.product(types,repeat=2):
 e=rm(w@fw(tx,ty).T-y); a.append((np.linalg.norm(e),tx,ty,e.tolist()))
print('best pairs',*sorted(a)[:12],sep='\n');
for t in range(3):
 best=min(a,key=lambda x:sum(e**2 for j,e in enumerate(x[3]) if j!=t)); print('holdout',t,best)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wrapcheck.py | tail -16
```
- **Container Exit Code**: `0` (`3759 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
best pairs
(2.8638086528258136e-06, ('wrap_full', 0, 1), ('closed', 0, 0), [1.63e-06, 1.67e-06, 1.66e-06])
(2.8638086528258136e-06, ('wrap_full', 1, 1), ('closed', 0, 0), [1.63e-06, 1.67e-06, 1.66e-06])
(0.0013696527806710721, ('wrap', 0, 1), ('closed', 0, 0), [0.00066476, 0.00082548, 0.00086754])
(0.0013696527806710721, ('wrap', 1, 1), ('closed', 0, 0), [0.00066476, 0.00082548, 0.00086754])
(0.003030453246694296, ('wrap', 0, 0), ('closed', 0, 0), [0.0020242, 0.00221852, 0.0004055])
(0.003030453246694296, ('wrap', 1, 0), ('closed', 0, 0), [0.0020242, 0.00221852, 0.0004055])
(0.003580705510287044, ('wrap_full', 0, 0), ('closed', 0, 0), [0.00139881, 0.00305707, 0.00123252])
(0.003580705510287044, ('wrap_full', 1, 0), ('closed', 0, 0), [0.00139881, 0.00305707, 0.00123252])
(0.02313667105257366, ('wrap', 0, 1), ('reflect', 2, 0), [0.01512511, 0.01527323, 0.0085595])
(0.02313667105257366, ('wrap', 1, 1), ('reflect', 2, 0), [0.01512511, 0.01527323, 0.0085595])
(0.02436131158648688, ('wrap_full', 0, 1), ('reflect', 2, 0), [0.01578573, 0.01603733, 0.00933211])
(0.02436131158648688, ('wrap_full', 1, 1), ('reflect', 2, 0), [0.01578573, 0.01603733, 0.00933211])
holdout 0 (2.8638086528258136e-06, ('wrap_full', 0, 1), ('closed', 0, 0), [1.63e-06, 1.67e-06, 1.66e-06])
holdout 1 (2.8638086528258136e-06, ('wrap_full', 0, 1), ('closed', 0, 0), [1.63e-06, 1.67e-06, 1.66e-06])
holdout 2 (2.8638086528258136e-06, ('wrap_full', 0, 1), ('closed', 0, 0), [1.63e-06, 1.67e-06, 1.66e-06])
```

### Turn 7 — `execute_in_remote_docker` (`69464 ms` LLM)
- **Rationale**: All three leave-one-pulse-out splits selected the same boundary stencil. Implement it without runtime tracer or archive dependencies, fit calibration from each study, and run the public validator to catch interface failures before independently checking the field and KKT certificate.
- **Alternatives Weighed**: `Wrap x using the archived local endpoint face coefficient` (rejected: Its worst pulse RMSE is 8.68e-4, versus 1.67e-6 with the face-averaged upwind row.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated inversion for the periodic-x, closed-y transport chamber."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import root


def _sensor_features(x, k):
    v = x[..., k]
    prev = x[..., (k - 1) % 4]
    nxt = x[..., (k + 1) % 4]
    return np.stack((np.ones_like(v), v, prev, nxt, v * nxt, v**3), axis=-1)


def _sensor_coefficients(sweeps):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    bank_ref = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
    bank_raw = np.asarray(bank['raw_readout'])
    # Discard gross electronic faults, separated from the ordinary sweep range.
    good = np.max(np.abs(raw), axis=1) < 8
    coefficients = []
    for k in range(4):
        use = (ids == k) & (np.abs(bank_raw) < 8)
        x = np.concatenate((ref[good], bank_ref[use]))
        y = np.concatenate((raw[good, k], bank_raw[use]))
        coefficients.append(np.linalg.lstsq(_sensor_features(x, k), y, rcond=None)[0])
    return np.asarray(coefficients)


def _calibrate(study):
    c = _sensor_coefficients(study['sensor_calibration_sweeps'])
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    linear = np.diag(c[:, 1])
    k = np.arange(4)
    linear[k, (k - 1) % 4] = c[:, 2]
    linear[k, (k + 1) % 4] = c[:, 3]
    initial = np.linalg.solve(linear, (observed - c[:, 0]).T).T

    def response(x):
        return np.array([_sensor_features(x, j) @ c[j] for j in range(4)])

    calibrated = np.empty_like(initial)
    for j, (y, x0) in enumerate(zip(observed, initial)):
        sol = root(lambda x: response(x) - y, x0, tol=1e-11)
        if np.linalg.norm(sol.fun, np.inf) > 1e-9:
            raise RuntimeError('rosette inversion did not converge')
        calibrated[j] = sol.x
    return calibrated.ravel()


def _axis(kappa, vel, h, periodic):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    n = len(kappa)
    face = (kappa[1:] + kappa[:-1]) / (2 * h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(vel[1:-1], 0) / h
    upper[1:] -= np.maximum(-vel[1:-1], 0) / h
    diagonal = np.empty(n)
    diagonal[1:-1] = -lower[:-1] - upper[1:]
    if periodic:
        seam = (kappa[0] + kappa[-1]) / (2 * h**2)
        upper[0] -= max(-vel[0], 0) / h
        lower[-1] -= max(vel[-1], 0) / h
        wrap_lo = -seam - max(vel[0], 0) / h
        wrap_hi = -seam - max(-vel[-1], 0) / h
        diagonal[0] = -upper[0] - wrap_lo
        diagonal[-1] = -lower[-1] - wrap_hi
        return sparse.diags(([wrap_hi], lower, diagonal, upper, [wrap_lo]),
                            (1 - n, -1, 0, 1, n - 1), format='csc')
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    diagonal[0], diagonal[-1] = -upper[0], -lower[-1]
    return sparse.diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), True)
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), False)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    rhs = np.column_stack([np.asarray(mode).ravel(order='F') for mode in study['source_modes']])
    ij = np.asarray(study['sensor_ij'])
    rows = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    sampled = []
    for shift in study['mu_shifts']:
        sol = splu(operator + shift * eye).solve(rhs)
        sampled.append(sol[rows])
    return np.concatenate(sampled), sol.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, b, cap, budget):
    """Primal active-set QP; the gradient is H @ w - b."""
    n = len(b)
    if budget == 0:
        w = np.zeros(n)
        return w, float(np.min(-b))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w - b))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1., np.max(np.abs(b)), cap * np.linalg.norm(H, np.inf))
    for _ in range(20 * n * n):
        free = active == 0
        lower, upper = active < 0, active > 0
        if not np.any(free):
            g = H @ w - b
            lo = np.argmin(np.where(lower, g, np.inf))
            hi = np.argmax(np.where(upper, g, -np.inf))
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue
        count = np.count_nonzero(free)
        system = np.zeros((count + 1, count + 1))
        system[:-1, :-1] = H[np.ix_(free, free)]
        system[:-1, -1] = -1
        system[-1, :-1] = 1
        rhs = np.r_[b[free] - H[np.ix_(free, upper)] @ w[upper], budget - np.sum(w[upper])]
        result = np.linalg.solve(system, rhs)
        target = result[:-1]
        nu = float(result[-1])
        idx = np.flatnonzero(free)
        cur = w[idx]
        delta = target - cur
        steps = np.full(count, np.inf)
        below = (target < 0) & (delta < 0)
        above = (target > cap) & (delta > 0)
        steps[below] = -cur[below] / delta[below]
        steps[above] = (cap - cur[above]) / delta[above]
        hit = np.argmin(steps)
        if steps[hit] < 1:
            w[idx] = np.clip(cur + steps[hit] * delta, 0, cap)
            active[idx[hit]] = -1 if below[hit] else 1
            w[idx[hit]] = 0 if below[hit] else cap
            continue
        w[idx] = np.clip(target, 0, cap)
        g = H @ w - b
        violation = np.where(lower, nu - g, np.where(upper, g - nu, -np.inf))
        release = np.argmax(violation)
        if violation[release] <= tol:
            return w, nu
        active[release] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    y = _calibrate(study)
    S, terminal_modes = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    profile = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    design = S - Z @ (profile @ S)
    data = y - Z @ (profile @ y)
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    budget = float(study['total_budget'])
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, design.T @ data, cap, budget)
    residual = design @ w - data
    g = design.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    dual = float(objective - g @ w + nu * budget - cap * np.sum(upper))
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective, 'dual_lower_bound': dual,
        },
    }
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py
```
- **Container Exit Code**: `0` (`499 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2a0232f51603881b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2a0232f5 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`37462 ms` LLM)
- **Rationale**: The public schema and host probe passed. Run the exact artifact in isolated Python and independently assemble the dense stencil to catch axis or field-ordering errors. Check the certificate on public inputs and stress the active-set solver at tight and saturated budgets; a failure would identify which component to fix.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('campaign_solver',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: batch=pickle.load(f)
worst=np.zeros(5)
for item in batch:
 s=item['study']; out=m.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; cap=s['u_box']; budget=s['total_budget']; nx,ny=s['nx'],s['ny']
 A=np.diag(s['V_reaction_2d'].ravel(order='F'))
 for j in range(ny):
  for i in range(nx):
   row=i+nx*j
   for k,n,L,kap,vel,stride,wrap in [(i,nx,s['Lx'],s['kappa_x'],s['vel_x'],1,True),(j,ny,s['Ly'],s['kappa_y'],s['vel_y'],nx,False)]:
    h=L/(n-1)
    for d in (-1,1):
     nb=k+d
     if wrap: nb%=n
     elif not 0<=nb<n: continue
     if not wrap and k in (0,n-1): flux=kap[k]/h**2
     else: flux=(kap[k]+kap[nb])/(2*h**2)+max(0,-d*vel[k])/h
     A[row,row]+=flux; A[row,row+(nb-k)*stride]-=flux
 rhs=np.stack([a.ravel(order='F') for a in s['source_modes']],axis=1)
 ix=s['sensor_ij']; indices=ix[:,0]+nx*ix[:,1]
 modes=[np.linalg.solve(A+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
 S=np.concatenate([u[indices] for u in modes]); field=modes[-1]@w
 y=m._calibrate(s); Z=s['Z_nuisance']; B=np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
 X=B@S; r=B@y; e=X@w-r; ridge=s['source_ridge']; g=X.T@e+ridge*w; J=(e@e+ridge*(w@w))/2
 lo=np.asarray(c['lam_lower']); hi=np.asarray(c['lam_upper']); nu=c['nu_sum']
 assert np.allclose(c['weights'],w,atol=1e-14,rtol=0) and np.min(lo)>=0 and np.min(hi)>=0
 assert abs(J-c['primal_objective'])<1e-11 and abs(J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()-c['dual_lower_bound'])<1e-11
 err=[max(abs(w.sum()-budget),max(0,-w.min(),w.max()-cap)), np.max(np.abs(g-nu-lo+hi)), max(np.max(lo*w),np.max(hi*(cap-w))), abs(J-c['dual_lower_bound']),np.linalg.norm(field-out['terminal_field'].ravel(order='F'))/np.linalg.norm(field)]
 worst=np.maximum(worst,err)
assert max(worst)<1e-10,worst
print('Public studies:',len(batch),'max [feasibility, stationarity, complementarity, gap, relative field assembly error]:',worst)
rng=np.random.default_rng(25); maxgap=0.
for n in (3,8,24):
 for cap in (.25,1.75):
  for fraction in (0.,.001,1/n,.5,(n-1)/n,.999,1.):
   for scale in (.03,80.):
    V=rng.normal(size=(max(2,n//2),n)); H=V.T@V+1e-4*np.eye(n); b=scale*rng.normal(size=n); budget=cap*n*fraction
    w,nu=m._box_simplex(H,b,cap,budget); g=H@w-b
    gap=g@w-nu*budget-cap*np.minimum(g-nu,0).sum()
    assert abs(w.sum()-budget)<1e-11 and np.all((w>=0)&(w<=cap)) and abs(gap)<1e-9
    maxgap=max(maxgap,abs(gap))
print('84 QP stress cases; maximum absolute gap:',maxgap)
PY
```
- **Container Exit Code**: `0` (`581 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public studies: 1 max [feasibility, stationarity, complementarity, gap, relative field assembly error]: [2.22044605e-16 2.38004061e-15 9.20621976e-17 3.24393290e-16
 5.07550383e-16]
84 QP stress cases; maximum absolute gap: 2.2453150450019166e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2a0232f51603881b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2a0232f5 (directory).
```

### Turn 9 — `submit_completed_task` (`12277 ms` LLM)
- **Rationale**: All three held-out tracer splits selected the same wall stencil. The unchanged artifact passed the public validator, isolated field and certificate checks, and budget stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:7183'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and 84 box-simplex stress cases had a maximum absolute duality gap of 2.25e-12.`

### Turn 10 — `execute_in_remote_docker` (`32795 ms` LLM)
- **Rationale**: The isolated field and KKT checks already passed. This focused audit tests a remaining plausible failure: amplitude-dependent rosette bias hidden by aggregate sweep error. A material improvement on both the held-out amplitude tail and independent tracer slices would justify revisiting calibration.
- **Alternatives Weighed**: `Run relay, drift, or within-batch classification checks` (rejected: This is a steady transport inverse problem with no relay state, evolution law, or classification labels.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
s=item['study']; sweeps=s['sensor_calibration_sweeps']; x=sweeps['ref_rosette']; y=sweeps['raw_rosette']; banks=sweeps['single_bank_sweeps']; ids=banks['bank_id']; br=banks['raw_readout']; bx=np.eye(4)[ids]*banks['ref_signal'][:,None]; good=np.flatnonzero(np.max(np.abs(y),axis=1)<8)
# Stratify folds by sweep amplitude; also hold out the high-amplitude tail as its own slice.
order=good[np.argsort(np.max(np.abs(x[good]),axis=1))]; folds=[order[j::5] for j in range(5)]+[order[-max(1,len(order)//5):]]
def design(v,k,a=0):
    F=m._sensor_features(v,k); F[..., -1]/=1+a*v[..., k]**2
    return F
def fit(train,rational):
    coeff=[]
    for k in range(4):
        keep=(ids==k)&(np.abs(br)<8); X=np.concatenate((x[train],bx[keep])); Y=np.r_[y[train,k],br[keep]]
        def linear(a):
            F=design(X,k,a); return F,np.linalg.lstsq(F,Y,rcond=None)[0]
        def residual(p):
            F,b=linear(p[0]); return F@b-Y
        a=least_squares(residual,[0.],bounds=(0.,2.),gtol=1e-14).x[0] if rational else 0.
        coeff.append((a,linear(a)[1]))
    return coeff
def predict(v,coeff):
    return np.stack([design(v,k,a)@c for k,(a,c) in enumerate(coeff)],axis=-1)
S,_=m._forward(s); pulses=item['campaign_diagnostics']; shifts=len(s['mu_shifts']); w=np.stack([p['w_pulse'] for p in pulses]); tracer=(w@S.T).reshape(len(pulses),-1,4); measured=np.stack([p['y_pulse_raw'] for p in pulses]).reshape(tracer.shape)
for rational in (False,True):
    errs=[]
    for test in folds:
        c=fit(np.setdiff1d(good,test),rational); e=predict(x[test],c)-y[test]; errs.append(np.sqrt(np.mean(e**2,axis=0)))
    c=fit(good,rational); e=(predict(tracer,c)-measured).reshape(len(pulses),shifts,-1); t=np.sqrt(np.mean(e**2,axis=2))
    print('rational',rational,'worst stratified sweep channel RMSE',np.max(errs[:-1]),'amplitude-tail worst channel RMSE',np.max(errs[-1]),'worst tracer pulse/shift raw RMSE',t.max(),'pulse/shift RMSEs',t.tolist())
PY
```
- **Container Exit Code**: `0` (`631 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rational False worst stratified sweep channel RMSE 1.410050241299369e-05 amplitude-tail worst channel RMSE 1.3099018689820575e-05 worst tracer pulse/shift raw RMSE 2.3708607882283826e-06 pulse/shift RMSEs [[2.3435442964574455e-06, 1.790662655520062e-06, 1.5598118151451254e-06], [2.3708607882283826e-06, 1.836252028707154e-06, 1.6084603035735373e-06], [2.268280243096907e-06, 1.8202960400529978e-06, 1.6088809964905575e-06]]
rational True worst stratified sweep channel RMSE 1.4100499290110035e-05 amplitude-tail worst channel RMSE 1.309900568659687e-05 worst tracer pulse/shift raw RMSE 2.586414689707427e-06 pulse/shift RMSEs [[2.555821054024867e-06, 1.90868414150238e-06, 1.617003460538863e-06], [2.586414689707427e-06, 1.969599285353758e-06, 1.6856355258800283e-06], [2.448750427371544e-06, 1.918708463918875e-06, 1.6572575032812508e-06]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2a0232f51603881b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2a0232f5 (directory).
```

### Turn 11 — `submit_completed_task` (`14197 ms` LLM)
- **Rationale**: All three leave-one-pulse-out splits selected the same wall stencil. The follow-up audit found no calibration improvement, and the unchanged artifact already passed isolated verification and the independent host probe.
- **Alternatives Weighed**: `Add rational saturation to the rosette calibration` (rejected: It left held-out sweep errors essentially unchanged and increased worst raw pulse-by-shift RMSE from 2.37e-6 to 2.59e-6.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and independent field and KKT checks passed. Across 84 box-simplex stress cases, the maximum absolute duality gap was 2.25e-12.`

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
