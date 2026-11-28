# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c03_batch_08__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `12 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `289,983` (`280,451` / `112,128` / `9,532`) |
| **Total Episode Wall Time (s)** | `314.45s` |

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

### Turn 1 — `execute_in_remote_docker` (`15738 ms` LLM)
- **Rationale**: Inspect the calibrated interface and campaign telemetry first: a wall-regime mismatch would change the forward operator, while study dimensions and the schema determine the optimizer and output contract.

```bash
cd /holdout_task; cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py; ls -lh /tmp/hegel_continual_archive; python3 - <<'PY'
import pickle, numpy as np, json, pathlib
items=pickle.load(open('public_batch.pkl','rb')); print('items',len(items))
for i,item in enumerate(items):
 print(i,item.keys()); print({k:(type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else len(v) if isinstance(v,list) else v if np.isscalar(v) else None) for k,v in item['study'].items()}); print('diagnostics', None if item.get('campaign_diagnostics') is None else [(k,np.shape(v)) for k,v in item['campaign_diagnostics'][0].items()])
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
 s=p.read_text(); print(p.name,len(s),s[-9500:])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box capped simplex active set'
```
- **Container Exit Code**: `0` (`473 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.56688,
        0.64307
      ],
      [
        -0.40533,
        -0.12376,
        -0.28714,
        -0.03565,
        -0.73655,
        2.65986,
        0.7148,
        0.29447
      ],
      [
        -0.39113,
        0.26705,
        -0.7769,
        -0.2616,
        -0.86656,
        2.59069,
        0.85159,
        -0.11944
      ],
      [
        -0.45852,
        0.64144,
        -0.80305,
        -0.46301,
        -0.73504,
        2.64222,
        0.88373,
        -0.50447
      ],
      [
        -0.37686,
        0.84195,
        -0.37853,
        -0.58233,
        -0.41559,
        2.57231,
        0.82591,
        -0.78974
      ],
      [
        -0.35316,
        0.81678,
        0.325,
        -0.78025,
        -0.01507,
        2.63128,
        0.66515,
        -0.84159
      ],
      [
        -0.32431,
        0.60404,
        0.74386,
        -0.82756,
        0.44732,
        2.66788,
        0.57155,
        -0.5948
      ],
      [
        -0.30042,
        0.21173,
        0.7881,
        -0.84944,
        0.72582,
        2.47129,
        0.4217,
        -0.30968
      ],
      [
        -0.47997,
        -0.27165,
        0.36365,
        -0.81459,
        0.83142,
        2.47261,
        0.14357,
        0.16559
      ],
      [
        -0.44931,
        -0.59588,
        -0.31516,
        -0.75687,
        0.75702,
        2.48327,
        0.02624,
        0.55277
      ],
      [
        -0.3698,
        -0.87464,
        -0.80882,
        -0.57871,
        0.41804,
        2.59698,
        -0.25093,
        0.75152
      ],
      [
        -0.38081,
        -0.82133,
        -0.8067,
        -0.40917,
        0.01458,
        2.5539,
        -0.38155,
        0.80382
      ],
      [
        -0.41069,
        -0.57647,
        -0.37878,
        -0.18214,
        -0.39895,
        2.52846,
        -0.58101,
        0.72412
      ],
      [
        -0.17331,
        -0.11723,
        0.33716,
        0.01013,
        -0.73447,
        2.51906,
        -0.72041,
        0.28283
      ]
    ]
  }
]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18453 ms` LLM)
- **Rationale**: Summarize archive structure without dumping telemetry arrays. Pulse availability and archived wall regimes determine whether to fit the operator on current diagnostics or transfer it from earlier campaigns.

```bash
cd /holdout_task; tail -27 _verified_upstream.py; cat validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np,pathlib,json
items=pickle.load(open('public_batch.pkl','rb')); print('items',len(items))
for item in items:
 print('study',[(k,np.shape(v) if not isinstance(v,dict) else list(v),v if np.isscalar(v) else '') for k,v in item['study'].items()]); d=item.get('campaign_diagnostics'); print('pulses',None if d is None else (len(d),[(k,np.shape(v)) for k,v in d[0].items()]))
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
 x=json.loads(p.read_text()); print('\n',p.name, 'keys',list(x) if isinstance(x,dict) else (len(x),list(x[-1]) if x else []));
 if 'apparatus' not in p.name: print(str(x)[-4200:]); continue
 entries=x if isinstance(x,list) else next((v for v in x.values() if isinstance(v,list)),[])
 for e in entries: print({k:v for k,v in e.items() if not isinstance(v,(dict,list))},[(k,np.shape(v) if isinstance(v,list) else list(v)) for k,v in e.items() if isinstance(v,(dict,list))])
PY
find /tmp/hegel_continual_archive -maxdepth 3 -name '*.py' | tail -30; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`409 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
llm_gate_accepted': False, 'repair_target': 'downstream_module_solve', 'failure_kind': 'scientific_or_numerical'}

 archive_apparatus_log.json keys (9, ['completed_campaign_index', 'completed_campaign_short_name', 'facility_telemetry_trace'])
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m100_c03_s00'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c03_batch_01'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c03_batch_02'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c03_batch_03'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 5, 'completed_campaign_short_name': 'beta_m100_c03_batch_04'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 6, 'completed_campaign_short_name': 'beta_m100_c03_batch_05'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 7, 'completed_campaign_short_name': 'beta_m100_c03_batch_06'} [('facility_telemetry_trace', (24, 8))]
{'completed_campaign_index': 8, 'completed_campaign_short_name': 'beta_m100_c03_batch_07'} [('facility_telemetry_trace', (24, 8))]
/tmp/hegel_continual_archive/beta_m100_c03_batch_01_solver.py
/tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py
/tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py
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
```

### Turn 3 — `execute_in_remote_docker` (`17676 ms` LLM)
- **Rationale**: Recover the pulse availability and archived wall-law references lost in truncated output. If no current pulses are available, telemetry continuity and archived boundary formulas will determine which operator to test rather than defaulting to open Robin walls.

```bash
cd /holdout_task; grep -n '^def\|telemetry\|wall\|regime' /tmp/hegel_continual_archive/*solver.py; python3 - <<'PY'
import pickle,json,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))[0]; s=x['study']; print('STUDY',[(k,list(v) if isinstance(v,dict) else np.shape(v),v if np.isscalar(v) else '') for k,v in s.items()]); print('DIAG',[(k,None if v is None else len(v)) for k,v in x.items() if k!='study']);
for p in (x.get('campaign_diagnostics'),s.get('reference_tracer_pulses')):
 if p: print('PULSE',list(p[0]),[(k,np.shape(v)) for k,v in p[0].items()]);
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']); print(e['completed_campaign_index'],np.round(a[[0,1,-2,-1]],4).tolist(), 'mean',np.round(a.mean(0),4).tolist())
PY
cat public_asset_index.json | head -80
```
- **Container Exit Code**: `0` (`409 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
34, 0.6674, 0.0322, 0.4787, 2.4463, -0.1391, -0.7333], [-0.4791, -0.8329, -0.1026, 0.6866, -0.6581, 2.6264, 0.5179, -0.3535], [-0.4056, -0.5921, 0.5105, 0.5143, -0.3422, 2.6304, 0.3285, -0.6826]] mean [-0.3548, 0.0036, -0.0061, -0.0096, -0.0117, 2.5579, -0.0029, 0.0016]
8 [[-0.4372, 0.2552, 0.7801, 0.2428, -0.8885, 2.4821, -0.8155, -0.1362], [-0.3306, 0.6243, 0.7875, 0.4579, -0.8109, 2.6206, -0.8389, -0.5196], [-0.4107, -0.5765, -0.3788, -0.1821, -0.399, 2.5285, -0.581, 0.7241], [-0.1733, -0.1172, 0.3372, 0.0101, -0.7345, 2.5191, -0.7204, 0.2828]] mean [-0.3564, 0.0107, -0.0023, 0.0028, 0.0024, 2.5486, -0.0059, 0.0002]
{
  "files": {
    "/holdout_task/_upstream_param_payload.json": {
      "bytes": 121,
      "summary": {
        "gains": {
          "first": 1.25999485,
          "len": 4,
          "type": "list"
        },
        "offsets": {
          "first": 0.06000529,
          "len": 4,
          "type": "list"
        }
      }
    },
    "/holdout_task/_verified_upstream.py": {
      "bytes": 3745
    },
    "/holdout_task/batch_solver.py": {
      "bytes": 408
    },
    "/holdout_task/public_batch.pkl": {
      "bytes": 27864,
      "summary": {
        "first": {
          "campaign_diagnostics": {
            "first": {
              "w_pulse": {
                "dtype": "float64",
                "shape": [
                  8
                ],
                "type": "ndarray"
              },
              "y_pulse_raw": {
                "dtype": "float64",
                "shape": [
                  96
                ],
                "type": "ndarray"
              }
            },
            "len": 3,
            "type": "list"
          },
          "study": {
            "Lx": 2.4,
            "Ly": 2.0,
            "V_reaction_2d": {
              "dtype": "float64",
              "shape": [
                14,
                12
              ],
              "type": "ndarray"
            },
            "Z_nuisance": {
              "dtype": "float64",
              "shape": [
                96,
                3
              ],
              "type": "ndarray"
            },
            "aux_manifold_telemetry": {
              "dtype": "float64",
              "shape": [
                24,
                4
              ],
              "type": "ndarray"
            },
            "kappa_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
```

### Turn 4 — `execute_in_remote_docker` (`16606 ms` LLM)
- **Rationale**: Read the archived discretization to identify the wall terms worth comparing, and check pulse excitation and telemetry summaries before fitting. Weak pulse excitation would favor simpler or tied wall terms rather than separate coefficients for every face.
- **Alternatives Weighed**: `Transfer the last verified wall operator unchanged` (rejected: Current tracer pulses can test whether that operator still applies; archive recency alone cannot establish it.)

```bash
cd /holdout_task; cat /tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py; python3 - <<'PY'
import pickle, numpy as np, json
x=pickle.load(open('public_batch.pkl','rb'))[0]; s=x['study']; print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('rosette ranges',*[np.quantile(s['sensor_calibration_sweeps'][k],[0,.05,.95,1],axis=0).round(3).tolist() for k in ('ref_rosette','raw_rosette')]); print('archive means',[(e['completed_campaign_index'],np.asarray(e['facility_telemetry_trace']).mean(0).round(3).tolist()) for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))]); print('pulse weight vectors',[p['w_pulse'].round(3).tolist() for p in x['campaign_diagnostics']]); print('shift values',s['mu_shifts']); print('scipy',__import__('scipy').__version__)
PY
```
- **Container Exit Code**: `0` (`458 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
= np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s, last = _responses(cal_item)
    z = np.asarray(cal_item['Z_nuisance'])
    combined = np.column_stack((s, cal_item['y_obs_raw']))
    combined -= z @ np.linalg.solve(z.T @ z + cal_item['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ combined)
    s, y = combined[:, :-1], combined[:, -1]
    ridge, cap, budget = (float(cal_item[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_qp(s.T @ s + ridge * np.eye(s.shape[1]), -(s.T @ y), cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, reduced).tolist(),
                                'lam_upper': np.maximum(0, -reduced).tolist(),
                                'primal_objective': obj,
                                'dual_lower_bound': float(obj - g @ w + nu * budget + cap * np.minimum(0, reduced).sum())}}
aux mean/std [ 0.1768566  -0.07998856  0.13256436 -0.31904299] [0.77416666 0.71137999 1.12421637 1.09518345]
rosette ranges [[-1.2, -1.324, -1.364, -1.371], [-1.144, -1.14, -1.202, -1.291], [1.654, 1.434, 1.489, 1.523], [1.799, 1.785, 1.664, 1.733]] [[-2.146, -12.85, -1.649, -13.655], [-1.586, -1.411, -1.296, -2.24], [4.29, 1.705, 2.137, 3.131], [13.636, 3.119, 12.638, 3.712]]
archive means [(0, [-0.356, 0.005, 0.003, -0.002, 0.0, -0.389, -0.005, 0.014]), (1, [2.459, -0.008, -0.004, 0.006, 0.007, -0.413, -0.007, 0.004]), (2, [-0.356, 0.011, -0.002, 0.003, 0.002, 2.549, -0.006, 0.0]), (3, [-0.341, -0.0, -0.013, 0.001, -0.006, -0.422, -0.002, 0.003]), (4, [2.456, 0.003, -0.002, 0.0, 0.0, -0.381, 0.009, -0.008]), (5, [-0.377, -0.006, -0.0, 0.005, -0.003, -0.414, -0.005, 0.001]), (6, [2.447, -0.003, 0.004, 0.002, 0.015, -0.39, -0.002, 0.003]), (7, [-0.355, 0.004, -0.006, -0.01, -0.012, 2.558, -0.003, 0.002]), (8, [-0.356, 0.011, -0.002, 0.003, 0.002, 2.549, -0.006, 0.0])]
pulse weight vectors [[1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]]
shift values [0.15 0.55 1.1 ]
scipy 1.13.0
```

### Turn 5 — `execute_in_remote_docker` (`26988 ms` LLM)
- **Rationale**: Compare archived wall laws independently on both axes against every calibrated pulse and shift. If none matches, the raw and nuisance-profiled errors will distinguish a missing boundary term from a nuisance offset before fitting a richer law.

```bash
cat > /tmp/test_wall.py <<'PY'
import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import coo_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
nx,ny=s['nx'],s['ny']; modes=s['source_modes']; rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T@W
idx=s['sensor_ij']@np.array([1,nx]); I=eye(nx*ny,format='csc')
def axis(d,law):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); i=np.arange(n); j=(i-1)%n; l=(i+1)%n
 lo=(k+k[j])/(2*h*h)+np.maximum(v,0)/h; hi=(k+k[l])/(2*h*h)+np.maximum(-v,0)/h; leak=np.zeros(n)
 if law != 'periodic':
  lo[0]=hi[-1]=0; hi[0]=k[0]/h**2; lo[-1]=k[-1]/h**2
  if law=='robin': leak[[0,-1]]=s['robin_'+d]/h
 return coo_matrix((np.r_[lo+hi+leak,-lo,-hi],(np.tile(i,3),np.r_[i,j,l])),shape=(n,n)).tocsc()
def predict(ax,ay):
 A=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.vstack([splu(A+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
print('calibrated pulse RMS',np.sqrt(np.mean(Y**2,axis=0)))
for x,y in itertools.product(['robin','sealed','periodic'],repeat=2):
 E=predict(axis('x',x),axis('y',y))-Y; R=E-Q@(Q.T@E)
 print(x,y,'pulse RMSE',np.sqrt(np.mean(E**2,axis=0)).round(8),'profiled',np.sqrt(np.mean(R**2,axis=0)).round(8),'shift RMSE',np.sqrt(np.mean(E.reshape(3,-1)**2,axis=1)).round(8))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/test_wall.py
```
- **Container Exit Code**: `0` (`530 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibrated pulse RMS [0.17314701 0.17717792 0.21815038]
robin robin pulse RMSE [0.10415763 0.10457008 0.13942178] profiled [0.1036238  0.10419831 0.13908123] shift RMSE [0.18362125 0.07782129 0.03806878]
robin sealed pulse RMSE [0.06790536 0.07745011 0.10877059] profiled [0.06777428 0.07730748 0.10857323] shift RMSE [0.13899925 0.05122877 0.02226028]
robin periodic pulse RMSE [0.07493902 0.07833617 0.11455561] profiled [0.07441363 0.07804313 0.11432099] shift RMSE [0.14316757 0.05877356 0.03039929]
sealed robin pulse RMSE [0.08904206 0.08286089 0.11071248] profiled [0.08848068 0.08245257 0.11032753] shift RMSE [0.15022638 0.06021294 0.02929342]
sealed sealed pulse RMSE [0.00344003 0.00098092 0.01047998] profiled [0.0034367  0.00097786 0.0102731 ] shift RMSE [0.00892345 0.00555396 0.00348591]
sealed periodic pulse RMSE [0.05082022 0.03216018 0.03215206] profiled [0.05029424 0.03185876 0.03193838] shift RMSE [0.05510487 0.03363926 0.0219677 ]
periodic robin pulse RMSE [0.08853005 0.08268691 0.11065101] profiled [0.0879655  0.08227267 0.11019981] shift RMSE [0.14998421 0.05986108 0.02897807]
periodic sealed pulse RMSE [8.2e-07 8.0e-07 8.2e-07] profiled [8.0e-07 7.8e-07 8.0e-07] shift RMSE [8.3e-07 8.0e-07 8.0e-07]
periodic periodic pulse RMSE [0.05100143 0.03241334 0.02980402] profiled [0.0504628  0.03210185 0.02953324] shift RMSE [0.05490591 0.03273095 0.02130906]
```

### Turn 6 — `execute_in_remote_docker` (`36536 ms` LLM)
- **Rationale**: Implement the mixed boundary law and retain the archived capped-simplex solver. Verify pulse predictions, output schema, and the KKT certificate separately so a correct forward fit cannot mask an optimization or interface error.
- **Alternatives Weighed**: `Fit continuous wall coefficients` (rejected: The discrete mixed wall law already matches every calibrated pulse at sub-micro-unit RMSE.)
- **ATMS NOGOOD DECLARED**: Refuted `The same archived wall law applies on both chamber axes.` | Witness: `The best shared law, sealed on both axes, has worst pulse RMSE 0.01048; periodic x and sealed y reduce it to 8.2e-7 across all three pulses.` | Blanket Diagnosis: `The x-axis requires wraparound flux while the y-axis has sealed endpoints. Nuisance projection does not remove the shared-law error.` | Sublated Pivot: `Retain the interior discretization and select the exact boundary rows separately by axis.`

```bash
set -e
cd /holdout_task
cp /tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py downstream_solver.py
python3 - <<'PY'
from pathlib import Path
p = Path('downstream_solver.py')
s = p.read_text().replace('Periodic transport–reaction inversion in calibrated sensor coordinates.', 'Transport–reaction inversion with periodic x and sealed y walls.')
s = s.replace('def _axis(kappa, velocity, h):', 'def _axis(kappa, velocity, h, periodic):')
s = s.replace('    return coo_matrix', '    if not periodic:\n        lo[0] = hi[-1] = 0.0\n        hi[0], lo[-1] = kappa[0] / h**2, kappa[-1] / h**2\n    return coo_matrix', 1)
s = s.replace("study['L' + d] / (n - 1))", "study['L' + d] / (n - 1), periodic=(d == 'x'))")
p.write_text(s)
PY
cat > batch_solver.py <<'PY'
"""Calibrate the raw study, then solve the chamber inverse problem."""
import inspect
import json
from pathlib import Path

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream

_NEEDS_PAYLOAD = len(inspect.signature(calibrate_upstream).parameters) >= 2
if _NEEDS_PAYLOAD:
    _PAYLOAD = json.loads(Path(__file__).with_name('_upstream_param_payload.json').read_text())


def solve_batch(study: dict) -> dict:
    calibrated = calibrate_upstream(study, _PAYLOAD) if _NEEDS_PAYLOAD else calibrate_upstream(study)
    return solve_downstream(calibrated)
PY
OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _responses
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
study = item['study']
s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
S, U = _responses(s)
errors = np.column_stack([S @ p['w_pulse'] - p['y_pulse_raw'] for p in s['reference_tracer_pulses']])
rmse = lambda a, axis: np.sqrt(np.mean(a*a, axis=axis))
print('pulse RMSE', rmse(errors, 0), 'shift RMSE', rmse(errors.reshape(len(s['mu_shifts']), -1), 1))
assert np.max(rmse(errors, 0)) < 2e-6
out = solve_batch(study)
w = out['weights']; c = out['kkt_certificate']; lo = np.asarray(c['lam_lower']); hi = np.asarray(c['lam_upper'])
z = s['Z_nuisance']; r = S @ w - s['y_obs_raw']; M = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]); r -= z @ np.linalg.solve(M, z.T @ r)
g = S.T @ (r - z @ np.linalg.solve(M, z.T @ r)) + s['source_ridge'] * w
obj = 0.5*(r@r + s['source_ridge']*(w@w)); cap = s['u_box']; budget = s['total_budget']
checks = {'budget': abs(w.sum()-budget), 'stationarity': np.max(abs(g-c['nu_sum']-lo+hi)), 'complementarity': max(np.max(abs(lo*w)), np.max(abs(hi*(cap-w)))), 'duality_gap': abs(obj - c['dual_lower_bound']), 'objective': abs(obj-c['primal_objective']), 'field': np.max(abs(np.tensordot(w,U,axes=1)-out['terminal_field']))}
print('certificate', checks, 'weights', w)
assert w.min() >= -1e-12 and w.max() <= cap+1e-12 and lo.min() >= 0 and hi.min() >= 0
assert np.array_equal(w, c['weights']) and max(checks.values()) < 1e-11
PY
```
- **Container Exit Code**: `0` (`636 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
pulse RMSE [8.18270465e-07 7.99327361e-07 8.16253364e-07] shift RMSE [8.33973451e-07 8.00379196e-07 7.99150651e-07]
certificate {'budget': 0.0, 'stationarity': 8.153200337090993e-17, 'complementarity': 2.6835708708797604e-16, 'duality_gap': 6.314393452555578e-16, 'objective': 3.469446951953614e-18, 'field': 0.0} weights [0.08653505 0.1205153  0.15846775 0.19762235 0.16059492 0.23546016
 0.29936595 0.14143852]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f62add5ac0504754 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f62add5a (directory).
```

### Turn 7 — `execute_in_remote_docker` (`25342 ms` LLM)
- **Rationale**: Hold out each pulse during wall-law selection to detect dependence on a single tracer. The public weights are all interior, so stress active bounds and exact vertices separately; a failure there would require fixing the optimizer, not the wall model.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, runpy, contextlib, io, itertools
import numpy as np
sys.path.insert(0, '/holdout_task')
from downstream_solver import _box_qp
with contextlib.redirect_stdout(io.StringIO()):
    wall = runpy.run_path('/tmp/test_wall.py')
laws = list(itertools.product(('robin', 'sealed', 'periodic'), repeat=2))
y = wall['Y']
errors = np.array([wall['predict'](wall['axis']('x', a), wall['axis']('y', b)) - y for a, b in laws])
held = []
for k in range(y.shape[1]):
    choice = np.argmin(np.mean(errors[:, :, np.arange(y.shape[1]) != k]**2, axis=(1,2)))
    assert laws[choice] == ('periodic', 'sealed')
    held.append(float(np.sqrt(np.mean(errors[choice, :, k]**2))))
print('Leave-one-pulse-out RMSE:', held)
assert max(held) < 2e-6
rng = np.random.default_rng(781)
n, cap = 8, 0.42
worst = 0.0
counts = {'lower': 0, 'upper': 0, 'vertex': 0}
for j in range(300):
    a = rng.normal(size=(n,n)); h = a.T @ a + 0.04*np.eye(n)
    linear = rng.normal(size=n) * (1, 10, 100)[j%3]
    for budget in (0., cap, 3*cap, 1.4, 7*cap, n*cap):
        w, nu = _box_qp(h, linear, cap, budget)
        r = h @ w + linear - nu
        comp = max(np.max(np.maximum(r,0)*w), np.max(np.maximum(-r,0)*(cap-w)))
        worst = max(worst, abs(w.sum()-budget), max(0., -w.min(), w.max()-cap), comp)
        if 0 < budget < n*cap:
            low, high = w < 1e-12, w > cap-1e-12
            counts['lower'] += bool(low.any()); counts['upper'] += bool(high.any()); counts['vertex'] += bool((low|high).all())
print('Box-simplex stress: 1800 cases; worst feasibility/complementarity residual:', worst, 'active slices:', counts)
assert worst < 1e-10 and all(counts.values())
PY
```
- **Container Exit Code**: `0` (`1107 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Leave-one-pulse-out RMSE: [8.182704654883446e-07, 7.993273606362532e-07, 8.162533638742634e-07]
Box-simplex stress: 1800 cases; worst feasibility/complementarity residual: 1.1937117960769683e-14 active slices: {'lower': 1075, 'upper': 1020, 'vertex': 463}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f62add5ac0504754 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f62add5a (directory).
```

### Turn 8 — `execute_in_remote_docker` (`28770 ms` LLM)
- **Rationale**: Test whether each held-out pulse still supports the mixed wall law when the probe imports the shipped code directly. A disagreement would reopen boundary selection; agreement supports submission without a solver edit.
- **Alternatives Weighed**: `Submit on the existing verification receipt` (rejected: The host rejected the earlier holdout probe; a direct probe using the original fixture and installed solver avoids the temporary helper.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, pickle, sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _responses
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
study = item['study']
s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
pulses = s['reference_tracer_pulses']
y = np.column_stack([p['y_pulse_raw'] for p in pulses])
w = np.column_stack([p['w_pulse'] for p in pulses])
S, U = _responses(s)
nx, ny = s['nx'], s['ny']
rhs = s['source_modes'].transpose(0,2,1).reshape(len(w),-1).T @ w
ij = s['sensor_ij']
idx = ij[:,0] + nx*ij[:,1]
I = eye(nx*ny, format='csc')
R = diags(s['V_reaction_2d'].ravel(order='F'))
laws = list(itertools.product((False, True), repeat=2))
errors = []
for law in laws:
    ax, ay = [_axis(s['kappa_'+d], s['vel_'+d], s['L'+d]/(n-1), flag) for d,n,flag in zip('xy',(nx,ny),law)]
    L = (kron(eye(ny),ax) + kron(ay,eye(nx)) + R).tocsc()
    pred = np.vstack([splu(L + mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
    if law == (True, False):
        np.testing.assert_allclose(pred, S @ w, atol=1e-12, rtol=0)
    errors.append(pred-y)
errors = np.array(errors)
held = []
for k in range(len(pulses)):
    selected = np.argmin(np.mean(errors[:,:,np.arange(len(pulses))!=k]**2, axis=(1,2)))
    assert laws[selected] == (True, False)
    held.append(float(np.sqrt(np.mean(errors[selected,:,k]**2))))
assert max(held) < 2e-6
out = solve_batch(study)
c = out['kkt_certificate']
gap = abs(c['primal_objective']-c['dual_lower_bound'])
assert gap < 1e-11
print('Leave-one-pulse-out RMSE:', held, 'public KKT duality gap:', gap)
PY
```
- **Container Exit Code**: `0` (`672 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Leave-one-pulse-out RMSE: [8.182704654883446e-07, 7.993273606362532e-07, 8.162533638742634e-07] public KKT duality gap: 6.279698983036042e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f62add5ac0504754 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f62add5a (directory).
```

### Turn 9 — `execute_in_remote_docker` (`25420 ms` LLM)
- **Rationale**: The host did not accept the fixture-based evidence for closing the boundary finding. Check the shipped endpoint rows against hand-computed values in isolation: a failure would identify a wraparound or endpoint-flux error rather than a calibration or optimizer error.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from downstream_solver import _axis
k = np.array([0.7, 1.1, 0.8, 1.4, 0.9])
v = np.array([0.6, -0.9, 0.5, -0.3, -0.4])
h = 0.25
x = _axis(k, v, h, True).toarray()
y = _axis(k, v, h, False).toarray()
x_ends = np.array([[17.6, -14.4, 0., 0., -15.2], [-14.4, 0., 0., -18.4, 0.]])
x_ends[0,0] = 29.6
x_ends[1,-1] = 32.8
# At the right periodic endpoint, negative drift acts on the wraparound face.
y_ends = np.array([[11.2, -11.2, 0., 0., 0.], [0., 0., 0., -14.4, 14.4]])
np.testing.assert_allclose(x[[0,-1]], x_ends, atol=1e-13, rtol=0)
np.testing.assert_allclose(y[[0,-1]], y_ends, atol=1e-13, rtol=0)
np.testing.assert_allclose(x[1:-1], y[1:-1], atol=1e-13, rtol=0)
residual = max(np.max(np.abs(x[[0,-1]]-x_ends)), np.max(np.abs(y[[0,-1]]-y_ends)), np.max(np.abs(x.sum(axis=1))), np.max(np.abs(y.sum(axis=1))))
assert residual < 1e-13
print(f'Boundary structure PASS: periodic x wraparound, sealed y endpoints, common interior, and constant-state conservation; worst residual={residual:.3e}')
PY
```
- **Container Exit Code**: `0` (`474 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Boundary structure PASS: periodic x wraparound, sealed y endpoints, common interior, and constant-state conservation; worst residual=3.553e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f62add5ac0504754 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f62add5a (directory).
```

### Turn 10 — `submit_completed_task` (`14638 ms` LLM)
- **Rationale**: The verified bundle passed schema validation, held-out pulse selection, direct endpoint checks, and active-bound KKT stress tests. The host did not accept formal closure of ng_t06, so this submission does not claim it.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:595', '/holdout_task/downstream_solver.py': 'FILE:4843'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `batch_solver.py` and `downstream_solver.py`. The solver uses periodic x and sealed y boundaries. Public schema validation passed, and the worst KKT residual across 1,800 stress cases was 1.19e-14.`

### Turn 11 — `execute_in_remote_docker` (`45029 ms` LLM)
- **Rationale**: The earlier comparison tied opposite faces together. Test all 81 face-wise combinations on real pulses, holding out complete rosettes, shifts, and pulses. If an asymmetric law wins or nuisance profiling changes the winner, reopen the boundary fit; otherwise retain the shipped law and report the measured margin.
- **Alternatives Weighed**: `Audit classification confounding, relay modes, or HJB terms` (rejected: This task has continuous tracer measurements and a steady linear chamber, not class labels, relay dynamics, or an HJB operator. The relevant nuisance test is whether profiling changes the wall-law selection.)

```bash
set -e
cat >/tmp/audit_mixed_walls.py <<'PY'
import itertools
import pickle
import sys

import numpy as np
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _responses

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
study = item['study']
s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
W = np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']])
Y = np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']])

# Assemble endpoint rows independently, allowing each face its own law.
def axis(d, ends):
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k)
    h = s['L'+d]/(n-1)
    A = lil_matrix((n,n))
    for i in range(n):
        law = ends[0] if i == 0 else ends[1] if i == n-1 else 'periodic'
        if law == 'periodic':
            for step in (-1, 1):
                j = (i+step) % n
                a = (k[i]+k[j])/(2*h*h) + max(0., -step*v[i])/h
                A[i,j] -= a
                A[i,i] += a
        else:
            j = 1 if i == 0 else n-2
            a = k[i]/(h*h)
            A[i,j] = -a
            A[i,i] = a + (s['robin_'+d]/h if law == 'robin' else 0.)
    return A.tocsc()

pairs = list(itertools.product(('sealed','periodic','robin'), repeat=2))
ax, ay = ([axis(d, p) for p in pairs] for d in 'xy')
nx, ny = s['nx'], s['ny']
I = eye(nx*ny, format='csc')
R = diags(s['V_reaction_2d'].ravel(order='F'))
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ W
i, j = s['sensor_ij'].T
ns = len(i)
laws, predictions = [], []
for a, b in itertools.product(range(len(pairs)), repeat=2):
    L = (kron(eye(ny),ax[a]) + kron(ay[b],eye(nx)) + R).tocsc()
    predictions.append(np.vstack([splu(L+mu*I).solve(rhs)[i+nx*j] for mu in s['mu_shifts']]))
    laws.append(pairs[a]+pairs[b])
P = np.array(predictions)
E = P - Y
want = laws.index(('periodic','periodic','sealed','sealed'))
S, U = _responses(s)
np.testing.assert_allclose(S @ W, P[want], atol=1e-12, rtol=0)

# Hold out complete rosettes across shifts, entire shifts, and individual pulses.
row = np.arange(len(Y))
folds = [(row % ns // 4 == a, slice(None)) for a in range(ns//4)]
folds += [(row//ns == a, slice(None)) for a in range(len(s['mu_shifts']))]
folds += [(np.zeros(len(Y), dtype=bool), a) for a in range(Y.shape[1])]
worst, margin = 0., np.inf
Z = s['Z_nuisance']
for test, col in folds:
    if isinstance(col, int):
        train = E[:,:,np.arange(Y.shape[1])!=col]
        design = Z
        held = E[:,:,col]
    else:
        train = E[:,~test,:]
        design = Z[~test]
        held = E[:,test,:]
    # Check selection both before and after fitting nuisance on training rows only.
    q = np.linalg.qr(design, mode='reduced')[0]
    for residual in (train, train - q @ (q.T @ train)):
        rmse = np.sqrt(np.mean(residual**2, axis=(1,2)))
        order = np.argsort(rmse)
        assert order[0] == want, (laws[order[0]], rmse[order[:2]])
        margin = min(margin, rmse[order[1]]-rmse[order[0]])
    worst = max(worst, float(np.sqrt(np.mean(held[want]**2))))
assert worst < 2e-6
out = solve_batch(study)
c = out['kkt_certificate']
gap = abs(c['primal_objective']-c['dual_lower_bound'])
assert gap < 1e-11
print(f'{len(laws)} face-wise laws, {len(folds)} held-out folds: worst RMSE={worst:.3e}, smallest runner-up training RMSE margin={margin:.3e}, public KKT gap={gap:.3e}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_mixed_walls.py
```
- **Container Exit Code**: `0` (`598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
81 face-wise laws, 14 held-out folds: worst RMSE=8.784e-07, smallest runner-up training RMSE margin=2.095e-03, public KKT gap=6.280e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f62add5ac0504754 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f62add5a (directory).
```

### Turn 12 — `submit_completed_task` (`14460 ms` LLM)
- **Rationale**: The exact installed bundle passed clean-process schema and KKT verification. Independent assembly and held-out measurements support the mixed boundary law, so no further edit is warranted. Formal closure of the earlier finding is not claimed.
- **Alternatives Weighed**: `Change the wall law or nuisance treatment` (rejected: The installed wall law beat all 80 alternatives in every fold, both before and after nuisance profiling.)
- **Submitted Final Answer / Summary**: `Implemented and verified `batch_solver.py` and `downstream_solver.py`. Periodic x and sealed y boundaries won all 14 held-out folds, with a worst RMSE of 8.78e-7. Public schema validation passed, and the worst KKT residual across 1,800 stress cases was 1.19e-14.`

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
