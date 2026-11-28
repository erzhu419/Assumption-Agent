# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c03_batch_02__blind_continual_campaign__downstream_module`

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
| **Total Tokens (Prompt / Cached / Out)** | `277,325` (`267,044` / `102,784` / `10,281`) |
| **Total Episode Wall Time (s)** | `331.42s` |

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

### Turn 1 — `execute_in_remote_docker` (`14372 ms` LLM)
- **Rationale**: Inspect the composition contract and telemetry first. A different upstream schema would change the solver inputs; evidence of a non-baseline wall would change the boundary rows before validation.
- **Alternatives Weighed**: `Implement the open-wall baseline immediately.` (rejected: The staged upstream interface and apparatus telemetry may identify a different calibration handoff or boundary law.)

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py public_asset_index.json && ls -l /tmp/hegel_continual_archive && for f in /tmp/hegel_continual_archive/{archive_manifest,archive_apparatus_log,pre_llm_gate_diagnostic}.json; do if [ -f "$f" ]; then echo "=== $f"; head -c 18000 "$f"; echo; fi; done
```
- **Container Exit Code**: `0` (`344 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.77189,
        0.86704,
        0.40733,
        -0.00449,
        2.52413,
        0.44825,
        0.84341
      ],
      [
        -0.34749,
        -0.54814,
        0.38413,
        0.17571,
        -0.40219,
        2.50632,
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
=== /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "pre_llm_gate_accepted": false,
  "repair_target": "downstream_module_solve",
  "failure_kind": "scientific_or_numerical"
}
```

### Turn 2 — `execute_in_remote_docker` (`19366 ms` LLM)
- **Rationale**: Summarize the fixture and archive without dumping arrays. Diagnostic availability determines whether to validate wall coefficients against calibrated pulses or infer them from apparatus telemetry; the schema determines operator and output shapes.
- **Alternatives Weighed**: `Fit the open-wall model directly to the study observations.` (rejected: Unknown source weights can absorb wall-model error; independent pulse or apparatus data can distinguish the two.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -lh /tmp/hegel_continual_archive && python3 - <<'PY'
import json,pickle,numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive')
def summary(v):
 if isinstance(v,dict): return {k: summary(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):
  if len(v) and isinstance(v[0],dict): return (len(v),summary(v[0]))
  a=np.asarray(v);return (a.shape,str(a.dtype),a.ravel()[:5].tolist())
 return v
items=pickle.load(open('public_batch.pkl','rb')); print('public count',len(items));print(summary(items[0]))
for name in ['archive_manifest.json','archive_apparatus_log.json']:
 f=p/name
 if not f.exists(): continue
 d=json.loads(f.read_text()); print(name)
 if name=='archive_manifest.json':print(str(d)[:9000])
 else:
  if isinstance(d,list):
   for r in d: print({k:(summary(v) if not isinstance(v,list) else str(summary(v))[:250]) for k,v in r.items()})
  else: print(str(summary(d))[:6000])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling constrained quadratic box simplex'
```
- **Container Exit Code**: `0` (`501 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7]), 'raw_readout': ((48,), 'float64', [-1.5779870582511275, -1.25727278884349, -0.9365243851712544, -0.6158095410793637, -0.29508765842385337])}}, 'Z_nuisance': ((96, 3), 'float64', [-1.7369832039184012, -0.952669637987892, -1.334110554154472, 0.7481795981200382, 0.16185449117365086]), 'y_obs_raw': ((96,), 'float64', [0.3134083282791028, 0.04659719016784088, 0.2382898359627464, 0.3715759390315677, 0.5633989194076223]), 'aux_manifold_telemetry': ((24, 4), 'float64', [0.018282732241760094, -0.32586955407098506, 1.1838466371832097, 1.123164590053547, -0.3674272439809623]), 'nuisance_ridge': 0.08, 'source_ridge': 0.04, 'u_box': 0.42, 'total_budget': 1.4}, 'campaign_diagnostics': (3, {'w_pulse': ((8,), 'float64', [1.0, 0.0, 0.0, 0.5, 0.0]), 'y_pulse_raw': ((96,), 'float64', [0.33215434487336026, 0.1403484652059261, 0.16624511641558043, 0.35925343544532723, 0.44618783948579555])})}
archive_manifest.json
[{'episode_short_name': 'beta_m100_c03_s00', 'task_id': 'beta_m100_c03_s00__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py'}, {'episode_short_name': 'beta_m100_c03_batch_01', 'task_id': 'beta_m100_c03_batch_01__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/beta_m100_c03_batch_01_solver.py'}]
archive_apparatus_log.json
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': "((24, 8), 'float64', [-0.41903, -0.17227, 0.70276, -0.5966, -0.40769])"}
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m100_c03_s00', 'facility_telemetry_trace': "((24, 8), 'float64', [2.50078, -0.80696, 0.83836, 0.83277, 0.80397])"}
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c03_batch_01', 'facility_telemetry_trace': "((24, 8), 'float64', [-0.43717, 0.25524, 0.78009, 0.24276, -0.88846])"}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`17924 ms` LLM)
- **Rationale**: Extract prior boundary mechanisms and compact telemetry summaries. This identifies which operator variants to test against independently calibrated pulses and whether the suggested sparse-ridge primitive matches the equality-constrained objective.
- **Alternatives Weighed**: `Start a generic wall coefficient search without reading the prior solvers.` (rejected: Archived mechanisms may already define the relevant boundary terms and scaling; testing them first narrows the pulse comparison.)

```bash
cd /holdout_task; for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $f"; wc -l "$f"; grep -nE 'def |robin|wall|telemetry|axis|boundary|bank|kron|ridge' "$f" | head -100; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -45; python3 - <<'PY'
import json,pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('items:',len(b));
for i,t in enumerate(b):
 s=t['study'];print(i,s['nx'],s['ny'],list(s), 'pulses', None if t.get('campaign_diagnostics') is None else len(t['campaign_diagnostics']));print('telemetry',np.mean(s['aux_manifold_telemetry'],axis=0),np.std(s['aux_manifold_telemetry'],axis=0))
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in a:
 x=np.array(r['facility_telemetry_trace']);print(r['completed_campaign_short_name'],x.mean(0),x.std(0))
PY
```
- **Container Exit Code**: `0` (`448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
102:    w, nu = _box_qp(s.T @ s + ridge * np.eye(s.shape[1]), -(s.T @ y), cap, budget)
104:    g = s.T @ residual + ridge * w
105:    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
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
items: 1
0 14 12 ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget'] pulses 3
telemetry [ 0.20031774  0.54389262 -0.05028571 -0.10401326] [0.87027627 1.08038945 0.85998344 1.06851992]
facility_init [-3.55525417e-01  4.84166667e-03  2.58583333e-03 -1.55958333e-03
  2.65833333e-04 -3.89467500e-01 -4.99208333e-03  1.36566667e-02] [0.06632241 0.60569699 0.59779727 0.59488489 0.59788013 0.07122488
 0.59146729 0.58740939]
beta_m100_c03_s00 [ 2.4588     -0.0078325  -0.0036      0.00595917  0.00677625 -0.4131725
 -0.00664917  0.00368917] [0.04790632 0.60081267 0.60364007 0.59452763 0.60322289 0.06641669
 0.60785931 0.59254421]
beta_m100_c03_batch_01 [-3.56382500e-01  1.06925000e-02 -2.35000000e-03  2.75541667e-03
  2.44500000e-03  2.54864083e+00 -5.85041667e-03  2.25000000e-04] [0.06931078 0.59356095 0.61295766 0.61299281 0.6112247  0.06734741
 0.59674719 0.59574837]
```

### Turn 4 — `execute_in_remote_docker` (`37155 ms` LLM)
- **Rationale**: Compare exact boundary variants in raw and nuisance-profiled residuals. Fit broader wall terms on pulse subsets to determine whether any improvement holds on an independent pulse before committing the operator.
- **Alternatives Weighed**: `Infer the current wall law directly from prior campaign telemetry.` (rejected: The current campaign has calibrated tracer pulses that can test wall laws without fitting unknown source weights.)

```bash
cd /holdout_task && cat /tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py && cat > /tmp/wall_probe.py <<'PY'
import sys,json,pickle,itertools
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
Y=np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in item['campaign_diagnostics']],axis=1)
W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; z=s['Z_nuisance']
N=nx*ny; M=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); idx=ij[:,0]+nx*ij[:,1]
def axis(d,p):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); face=(k[:-1]+k[1:])/(2*h*h)
 l=-face-np.maximum(v[1:],0)/h; r=-face-np.maximum(-v[:-1],0)/h
 l[-1]=-k[-1]/h**2*p[3]; r[0]=-k[0]/h**2*p[0]
 A=diags([l,np.r_[0,-l[:-1]-r[1:],0],r],[-1,0,1],format='lil')
 for i,j,b in [(0,1,p[:3]),(n-1,n-2,p[3:])]:
  A[i,n-1-i]=-b[2]*k[i]/h**2
  A[i,i]=-A[i,j]-A[i,n-1-i]+b[1]*s['robin_'+d]/h
 return A.tocsc()
def calc(p):
 a,b=axis('x',p[:6]),axis('y',p[6:]); I=eye(N,format='csc'); L=(kron(eye(ny),a)+kron(b,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(M@W)[idx] for mu in s['mu_shifts']],axis=0)
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
def prof(x): return x-z@np.linalg.lstsq(z,x,rcond=None)[0]
opts={'open':[1,1,0],'sealed':[1,0,0],'periodic':[1,0,1],'double':[2,0,0],'double_robin':[2,2,0],'ghost_wrap':[0,0,1]}
if __name__=='__main__':
 print('pulse weights',W.T,'sample Y',Y[:4], 'baseline noise along nuisance',rms(Y-prof(Y)))
 scores=[]
 for x,y in itertools.product(opts,repeat=2):
  p=np.array(opts[x]*2+opts[y]*2); e=calc(p)-Y; scores.append((max(rms(prof(e))),x,y,rms(e),rms(prof(e))))
 for row in sorted(scores)[:12]: print(row)
 def sym(p): return np.repeat(p.reshape(2,3),2,axis=0).ravel()
 for grouped in (True,False):
  for train in ([0,1,2],[1,2],[0,2],[0,1]):
   conv=sym if grouped else np.asarray
   def fun(p):return (1e4*prof(calc(conv(p))-Y)[:,train]).ravel()
   fit=least_squares(fun,[1,1,.03]*(2 if grouped else 4), bounds=([.001,0,0]*(2 if grouped else 4),[5,12,5]*(2 if grouped else 4)),max_nfev=200)
   e=calc(conv(fit.x))-Y
   print('symmetric',grouped,'train',train,'params',np.round(fit.x,6),'raw RMS',rms(e),'profile RMS',rms(prof(e)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `0` (`6522 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7094480496, 'double', 'double', array([0.04066638, 0.03735134, 0.02410737]), array([0.03995737, 0.03701845, 0.02373186]))
(0.08785528577414375, 'double_robin', 'double', array([0.07994269, 0.08836056, 0.07873432]), array([0.07973925, 0.08785529, 0.07842653]))
(0.08831152269518927, 'open', 'double', array([0.08070079, 0.08882411, 0.07948338]), array([0.08048996, 0.08831152, 0.07917091]))
(0.09124100275800401, 'periodic', 'double_robin', array([0.09124648, 0.09179025, 0.08847593]), array([0.09075896, 0.091241  , 0.08804071]))
symmetric True train [0, 1, 2] params [1.347917 0.022108 0.688146 1.24487  0.       1.384059] raw RMS [0.00432075 0.00883065 0.00535712] profile RMS [0.00427388 0.00879743 0.00529122]
symmetric True train [1, 2] params [1.504381 0.029271 0.827659 1.12305  0.       1.180091] raw RMS [0.0060146  0.0070622  0.00708623] profile RMS [0.00594244 0.00704747 0.00699525]
symmetric True train [0, 2] params [1.053355 0.       1.299768 1.584704 0.003394 1.725919] raw RMS [0.00048238 0.01525024 0.00043456] profile RMS [0.0004811  0.01511805 0.00043373]
symmetric True train [0, 1] params [1.338839 0.018455 0.675947 1.094144 0.       1.125685] raw RMS [0.00652546 0.00644465 0.00778875] profile RMS [0.00646679 0.00642694 0.00770325]
symmetric False train [0, 1, 2] params [1.036682e+00 2.300000e-05 1.140292e+00 1.036475e+00 0.000000e+00
 1.082377e+00 9.952310e-01 0.000000e+00 7.781670e-01 1.008353e+00
 0.000000e+00 1.535112e+00] raw RMS [3.93648584e-06 3.97895755e-06 3.96600979e-06] profile RMS [3.81069895e-06 3.83178568e-06 3.83691345e-06]
symmetric False train [1, 2] params [1.036591e+00 2.200000e-05 1.140252e+00 1.036525e+00 0.000000e+00
 1.082463e+00 9.952540e-01 0.000000e+00 7.781520e-01 1.008337e+00
 1.000000e-06 1.535051e+00] raw RMS [3.97627708e-06 3.96949351e-06 3.95914681e-06] profile RMS [3.86309194e-06 3.82270266e-06 3.84080231e-06]
symmetric False train [0, 2] params [1.036738e+00 2.600000e-05 1.140338e+00 1.036482e+00 0.000000e+00
 1.082343e+00 9.950610e-01 0.000000e+00 7.779190e-01 1.009403e+00
 0.000000e+00 1.535426e+00] raw RMS [3.93948381e-06 1.56889090e-05 3.96912352e-06] profile RMS [3.80817726e-06 1.55831867e-05 3.83061034e-06]
symmetric False train [0, 1] params [1.036661e+00 2.100000e-05 1.140271e+00 1.036483e+00 0.000000e+00
 1.082388e+00 9.952290e-01 0.000000e+00 7.781690e-01 1.008347e+00
 0.000000e+00 1.535109e+00] raw RMS [3.93405963e-06 3.97446245e-06 3.96865173e-06] profile RMS [3.81056101e-06 3.83024296e-06 3.84104302e-06]
```

### Turn 5 — `execute_in_remote_docker` (`27030 ms` LLM)
- **Rationale**: Compare minimal nested stencils on held-out calibrated pulses. If fixing the neighbor coefficient raises error above the measurement floor, retain that term. Compare effective coefficients with endpoint and face conductivities before deciding whether to transfer an exact stencil or campaign-specific fitted multipliers.
- **Alternatives Weighed**: `Deploy all twelve fitted coefficients immediately.` (rejected: The leakage coefficients are near zero. Removing them may preserve pulse accuracy and avoid fitting calibration noise.)
- **ATMS NOGOOD DECLARED**: Refuted `Opposite walls can share the same neighbor, leakage, and wrap coefficients.` | Witness: `The shared-coefficient fit missed a held-out pulse by 0.01525 RMS; separate coefficients brought the worst held-out pulse to 0.00001569 RMS.` | Blanket Diagnosis: `The archived open and sealed operators omit opposite-edge transport, and shared coefficients erase directional differences visible in the tracer measurements.` | Sublated Pivot: `Test zero leakage and distinguish wrap-only coupling from a stencil that also modifies boundary-to-neighbor flux.`

```bash
cat > /tmp/wrap_fit.py <<'PY'
from wall_probe import *
np.set_printoptions(precision=9,suppress=True)
for d in 'xy':
 k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 print(d,'h',h,'kappa ends',k[[0,-1]],'vel ends',v[[0,-1]],'first/last face',((k[:-1]+k[1:])/2)[[0,-1]],'wrap mean',np.mean(k[[0,-1]]))
def convert(p,kind):
 p=np.asarray(p).reshape(4,-1)
 if kind=='wrap-only': return np.column_stack([np.ones(4),np.zeros(4),p[:,0]]).ravel()
 if kind=='no-leak': return np.column_stack([p[:,0],np.zeros(4),p[:,1]]).ravel()
 if kind=='full': return p.ravel()
for kind,p0 in [('wrap-only',[1]*4),('no-leak',[1,1]*4),('full',[1,0,1]*4)]:
 for omit in [-1,0,1,2]:
  keep=[i for i in range(3) if i!=omit]
  fit=least_squares(lambda p:1e5*(calc(convert(p,kind))-Y)[:,keep].ravel(),p0,gtol=1e-10)
  p=convert(fit.x,kind); print(kind,omit,'coef',p,'RMS',rms(calc(p)-Y))
  if kind=='no-leak' and omit==-1:
   for d,b in zip('xy',p.reshape(2,2,3)):
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    eff=b[:,[0,2]]*k[[0,-1],None]; print(d,'effective k',eff,'neighbor minus endpoint',eff[:,0]-k[[0,-1]],'wrap over arithmetic mean',eff[:,1]/np.mean(k[[0,-1]]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wrap_fit.py
```
- **Container Exit Code**: `0` (`2020 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.          1.583960187] RMS [0.001042443 0.000140113 0.000239505]
wrap-only 1 coef [1.          0.          1.080134534 1.          0.          1.032303261
 1.          0.          0.780831212 1.          0.          1.597150089] RMS [0.000248865 0.00139576  0.000296077]
wrap-only 2 coef [1.          0.          1.102816892 1.          0.          0.991705036
 1.          0.          0.774398201 1.          0.          1.657952559] RMS [0.000597984 0.000382808 0.000819794]
no-leak -1 coef [1.036448288 0.          1.140021886 1.036397776 0.          1.082276464
 0.99524093  0.          0.778126594 1.008310123 0.          1.534988698] RMS [0.000004024 0.000004033 0.000004062]
x effective k [[0.478767912 0.526611799]
 [0.478744579 0.499937381]] neighbor minus endpoint [0.016836605 0.016813272] wrap over arithmetic mean [1.140021886 1.082276464]
y effective k [[0.538881684 0.421323276]
 [0.30396368  0.462735426]] neighbor minus endpoint [-0.002576839  0.002505157] wrap over arithmetic mean [0.999679098 1.097938232]
no-leak 0 coef [1.036291546 0.          1.139953846 1.036546703 0.          1.082525596
 0.995279399 0.          0.778147551 1.008277629 0.          1.534915106] RMS [0.000004061 0.000004021 0.000004068]
no-leak 1 coef [1.036489267 0.          1.140118333 1.036340679 0.          1.082222513
 0.994505321 0.          0.776929827 1.008696054 0.          1.535081609] RMS [0.000003983 0.000010964 0.000004043]
no-leak 2 coef [1.036440791 0.          1.140015803 1.036417998 0.          1.08230159
 0.995249196 0.          0.778165274 1.008291099 0.          1.535013588] RMS [0.000004024 0.000004032 0.000004066]
full -1 coef [ 1.036802781  0.000041976  1.140414167  1.036275001 -0.000019743
  1.082165999  0.995123507 -0.000025211  0.778093932  1.00846046
  0.000020686  1.535295639] RMS [0.000003913 0.000003946 0.000003953]
full 0 coef [ 1.0367003    0.000039016  1.140375726  1.036325124 -0.000022131
  1.08229871   0.995181969 -0.000020375  0.778099869  1.008468566
  0.0000217    1.535226768] RMS [0.000003943 0.000003935 0.000003955]
full 1 coef [ 1.036990553  0.000055965  1.140597345  1.036133521 -0.000031784
  1.081978966  0.995002225 -0.000026565  0.777932795  1.009034257
  0.000020892  1.535479798] RMS [0.000003907 0.000009477 0.00000395 ]
full 2 coef [ 1.036821823  0.000042784  1.140434168  1.036231897 -0.000022123
  1.082135414  0.995034492 -0.000033336  0.77803193   1.008571746
  0.000028226  1.535422228] RMS [0.000003904 0.000003949 0.00000399 ]
```

### Turn 6 — `execute_in_remote_docker` (`21510 ms` LLM)
- **Rationale**: Test the periodic stencil without fitting any coefficients, separately for every tracer and shift. If it matches the fitted stencil at the calibration floor, implement the exact law; if not, use the slice residuals to isolate the missing term. Read the upstream handoff before deciding whether any downstream calibration is needed.
- **Alternatives Weighed**: `Use the fitted no-leak multipliers in the solver.` (rejected: The fitted neighbor terms match arithmetic face averages, and the directional seam terms match wrap-face diffusion plus row-local upwind advection. The exact stencil can recompute those terms when grid spacing or transport coefficients change.)

```bash
cd /holdout_task && cat _verified_upstream.py batch_solver.py && cat > /tmp/exact_periodic_probe.py <<'PY'
from wall_probe import *

def formula(d):
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    lo,hi=k[0],k[-1]
    seam=(lo+hi)/2
    return [(lo+k[1])/(2*lo),0,(seam+h*max(v[0],0))/lo,(hi+k[-2])/(2*hi),0,(seam+h*max(-v[-1],0))/hi]

p=np.array(formula('x')+formula('y'))
for name,q in [('periodic_fvm',p),('open_wall',np.array([1,1,0]*4)),('sealed_wall',np.array([1,0,0]*4))]:
    e=calc(q)-Y
    print(name,'coefficients',q,'pulse RMS',rms(e))
    print('shift-pulse RMS',np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),len(ij),-1)**2,axis=1)))
    print('pulse max abs',np.max(np.abs(e),axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/exact_periodic_probe.py
```
- **Container Exit Code**: `0` (`585 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    observed = np.asarray(sweeps['raw_rosette'])
    inlier_rows = np.all(np.abs(observed) <= 8.5, axis=1)

    payload = {} if param_payload is None else param_payload
    gains = np.asarray(payload.get('gains'))
    offsets = np.asarray(payload.get('offsets'))
    if gains.shape != (4,) or offsets.shape != (4,):
        x, y = ref[inlier_rows], observed[inlier_rows]
        gains, offsets = np.empty(4), np.empty(4)
        for c in range(4):
            design = np.column_stack((x[:, c], np.ones(len(x))))
            gains[c], offsets[c] = np.linalg.lstsq(design, y[:, c], rcond=None)[0]

    raw = np.asarray(calibrated['y_obs_raw']).reshape(-1, 4)
    calibrated['y_obs_raw'] = ((raw - offsets) / gains).ravel()
    sweeps['raw_rosette'] = np.where(inlier_rows[:, None], ref, observed)

    if 'single_bank_sweeps' in calibrated:
        bank = calibrated['single_bank_sweeps']
        readout = np.asarray(bank['raw_readout'])
        bank['raw_readout'] = np.where(np.abs(readout) <= 8.5, bank['ref_signal'], readout)
    return calibrated
from __future__ import annotations
import inspect, json
import _verified_upstream as _up
import downstream_solver as _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
periodic_fvm coefficients [1.03626534 0.         1.13988094 1.03626534 0.         1.08221995
 0.99551134 0.         0.77837638 1.00806221 0.         1.53468822] pulse RMS [4.80740820e-06 4.82300653e-06 4.79541673e-06]
shift-pulse RMS [[4.96826358e-06 4.99733745e-06 4.95215493e-06]
 [4.76342448e-06 4.77535743e-06 4.75261294e-06]
 [4.68611406e-06 4.69113587e-06 4.67727448e-06]]
pulse max abs [8.10259742e-06 8.17218765e-06 7.91139849e-06]
open_wall coefficients [1 1 0 1 1 0 1 1 0 1 1 0] pulse RMS [0.1107987  0.11995001 0.10793514]
shift-pulse RMS [[0.17639753 0.18974938 0.17195152]
 [0.06926498 0.07687779 0.06728601]
 [0.03025438 0.03534104 0.02924468]]
pulse max abs [0.1981208  0.23579381 0.19282407]
sealed_wall coefficients [1 0 0 1 0 0 1 0 0 1 0 0] pulse RMS [0.02581743 0.02942806 0.02220042]
shift-pulse RMS [[0.03449028 0.0401592  0.03180135]
 [0.02374801 0.02636334 0.01842565]
 [0.01568669 0.01703656 0.01130243]]
pulse max abs [0.06492286 0.08612241 0.05754562]
```

### Turn 7 — `execute_in_remote_docker` (`56816 ms` LLM)
- **Rationale**: Implement the exact stencil and solve the profiled box-simplex problem. Validate the composed interface, pulse ordering, and public primal-dual gap; compare staged and dynamically fitted calibration to detect a handoff discrepancy before final stress testing.
- **Alternatives Weighed**: `Carry forward fitted seam multipliers.` (rejected: The coefficient-free periodic stencil already matches all pulse and shift slices at the calibration noise floor and transfers across grid and coefficient changes.)

```bash
cd /holdout_task && cat > downstream_solver.py <<'PY'
"""Periodic transport–reaction inversion in calibrated sensor coordinates."""

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    face = (kappa + kappa[right]) / (2 * h**2)
    lo = face[left] + np.maximum(velocity, 0) / h
    hi = face + np.maximum(-velocity, 0) / h
    return coo_matrix((np.r_[lo + hi, -lo, -hi],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n)).tocsc()


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                    study['L' + d] / (n - 1)) for d, n in zip('xy', (nx, ny))]
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx * sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(indices) == len(study['y_obs_raw']) and len(shifts) > 1:
        indices = indices.reshape(len(shifts), -1)
    identity = eye(nx * ny, format='csc')
    samples = []
    for k, mu in enumerate(shifts):
        field = splu(operator + mu * identity).solve(rhs)
        samples.append(field[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(samples), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _box_qp(h, linear, cap, budget):
    """Feasible active-set Newton method for one equality and box bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = h @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    state = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1, np.max(np.abs(linear)), cap * np.linalg.norm(h, np.inf))
    for _ in range(20 * n + 100):
        f = np.flatnonzero(state == 0)
        g = h @ w + linear
        if not len(f):
            lower, upper = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[lo] >= g[hi] - tol:
                return w, float((g[lo] + g[hi]) / 2)
            state[[lo, hi]] = 0
            continue
        fixed = np.flatnonzero(state)
        hf = h[np.ix_(f, f)]
        gf = linear[f] + h[np.ix_(f, fixed)] @ w[fixed]
        v, q = np.linalg.solve(hf, np.column_stack((-gf, np.ones(len(f))))).T
        nu = (budget - w[fixed].sum() - v.sum()) / q.sum()
        target = v + nu * q
        step = target - w[f]
        ratios = np.full(len(f), np.inf)
        below, above = (target < 0) & (step < 0), (target > cap) & (step > 0)
        ratios[below] = -w[f][below] / step[below]
        ratios[above] = (cap - w[f][above]) / step[above]
        j = np.argmin(ratios)
        if ratios[j] < 1:
            w[f] += ratios[j] * step
            state[f[j]] = -1 if below[j] else 1
            w[f[j]] = 0 if below[j] else cap
            continue
        w[f] = np.clip(target, 0, cap)
        g = h @ w + linear
        violation = state * (g - nu)
        j = np.argmax(violation)
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
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys
sys.path.insert(0,'/holdout_task')
sys.path.insert(0,'/tmp')
from wall_probe import s, W, Y, rms, item, pay, calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _responses
import numpy as np
cal=calibrate_upstream(s,pay)
a, _ = _responses(cal)
e=(a@W-Y).reshape(len(s['mu_shifts']),-1,W.shape[1]); print('Worst shift/pulse RMS', np.sqrt(np.mean(e**2,axis=1)).max())
out=solve_batch(s); w=out['weights']; k=out['kkt_certificate']; print('weight range',w.min(),w.max(),'cap',s['u_box'],'budget error',abs(w.sum()-s['total_budget']),'objective',k['primal_objective'],'gap',k['primal_objective']-k['dual_lower_bound'])
print('dynamic vs staged calibration max delta', np.max(abs(calibrate_upstream(s)['y_obs_raw']-cal['y_obs_raw'])))
assert np.sqrt(np.mean(e**2,axis=1)).max() < 5.1e-6
assert np.all(np.isfinite(out['terminal_field'])) and abs(w.sum()-s['total_budget'])<1e-10
assert abs(k['primal_objective']-k['dual_lower_bound'])<1e-10
PY
```
- **Container Exit Code**: `0` (`685 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Worst shift/pulse RMS 4.997337453052916e-06
weight range 0.060044710543427485 0.2898940157221609 cap 0.42 budget error 0.0 objective 0.009305813305855953 gap 5.30825383648903e-16
dynamic vs staged calibration max delta 5.561459494762833e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=58bc8257ebad4632 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_58bc8257 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`38191 ms` LLM)
- **Rationale**: Compare the terminal field and certificate against an independent dense construction to catch indexing or projection errors. Stress active bounds separately; a KKT failure would require correcting the box-simplex solver before submission.
- **Alternatives Weighed**: `Submit after the public fixture passes.` (rejected: The public optimum is entirely interior, so it does not test bound activation or the multiplier interval at a vertex.)

```bash
cd /holdout_task && python3 validate_public_schema.py /holdout_task/batch_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_PAY
from _verified_upstream import calibrate_upstream
from downstream_solver import _box_qp
item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']
# Independent dense reference: row-by-row periodic stencil and per-shift solves.
nx,ny=s['nx'],s['ny']; Ax=[]
for d,n in zip('xy',(nx,ny)):
    h=s['L'+d]/(n-1); k=s['kappa_'+d]; v=s['vel_'+d]; a=np.zeros((n,n))
    for i in range(n):
        for j,adv in [((i-1)%n,max(v[i],0)),((i+1)%n,max(-v[i],0))]:
            flux=(k[i]+k[j])/(2*h*h)+adv/h
            a[i,i]+=flux; a[i,j]-=flux
    Ax.append(a)
L=np.kron(np.eye(ny),Ax[0])+np.kron(Ax[1],np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
M=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); i,j=s['sensor_ij'].T
fields=[np.linalg.solve(L+mu*np.eye(nx*ny),M) for mu in s['mu_shifts']]
S=np.vstack([f[i+nx*j] for f in fields]); out=solve_batch(s); w=out['weights']; cert=out['kkt_certificate']; cap=s['u_box']; budget=s['total_budget']
field_err=np.max(np.abs(out['terminal_field']-(fields[-1]@w).reshape((nx,ny),order='F')))
Z=s['Z_nuisance']; y=calibrate_upstream(s,_PAY)['y_obs_raw']; P=np.eye(len(y))-Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
T=P@S; residual=T@w-P@y; g=T.T@residual+s['source_ridge']*w; obj=.5*(residual@residual+s['source_ridge']*(w@w))
lo,hi=np.asarray(cert['lam_lower']),np.asarray(cert['lam_upper']); station=np.max(np.abs(g-cert['nu_sum']-lo+hi)); comp=max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w))))
assert field_err<1e-11 and station<1e-10 and comp<1e-10
assert np.all(w>=0) and np.all(w<=cap) and abs(w.sum()-budget)<1e-11
assert np.array_equal(w,np.asarray(cert['weights'])) and abs(cert['primal_objective']-obj)<1e-11 and abs(obj-cert['dual_lower_bound'])<1e-10
print('Dense reference: terminal max error',field_err,'stationarity',station,'complementarity',comp)
# KKT conditions independently certify optimality, including full vertices and mixed active sets.
rng=np.random.default_rng(517); mx=np.zeros(4); cases=0; vertex=0; mixed=0
for n in (1,3,9,24):
    for trial in range(75):
        a=rng.normal(size=(max(1,n//2),n)); H=a.T@a+10**rng.uniform(-7,-.3)*np.eye(n)
        c=rng.normal(size=n)*10**rng.uniform(-2,2); u=.42
        for b in (0.,n*u, min(2,n)*u, .53*n*u, .001*n*u, .999*n*u):
            x,nu=_box_qp(H,c,u,b); d=H@x+c-nu
            mx=np.maximum(mx,[abs(x.sum()-b),max(0.,-x.min(),x.max()-u),max(np.max(abs(np.maximum(d,0)*x)),np.max(abs(np.minimum(d,0)*(u-x)))),abs(x@d-u*np.minimum(d,0).sum())]); cases+=1
            if 0<b<n*u:
                vertex+=int(np.all((x==0)|(x==u))); mixed+=int(np.any(x==0) and np.any(x==u) and np.any((x>0)&(x<u)))
print('QP stress:',cases,'cases;',vertex,'interior-budget vertices;',mixed,'mixed active sets; worst budget, bound, complementarity and gap:',mx)
assert vertex>0 and mixed>0 and np.all(mx<1e-9)
PY
```
- **Container Exit Code**: `0` (`1144 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Dense reference: terminal max error 1.1102230246251565e-16 stationarity 2.5743296383495817e-15 complementarity 1.3862188178476018e-16
QP stress: 1800 cases; 67 interior-budget vertices; 270 mixed active sets; worst budget, bound, complementarity and gap: [1.76369253e-10 0.00000000e+00 3.65591505e-10 4.47656412e-10]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=58bc8257ebad4632 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_58bc8257 (directory).
```

### Turn 9 — `submit_completed_task` (`13915 ms` LLM)
- **Rationale**: The verified bundle matches calibrated tracer measurements without fitted wall coefficients. Independent dense verification confirmed field ordering and the certificate, and stress tests covered vertices and mixed active bounds absent from the public fixture.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4711'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented the periodic transport–reaction solver in `/holdout_task/downstream_solver.py` and verified its integration through `/holdout_task/batch_solver.py`. The public schema passed. Worst tracer RMS across shifts was 5.00e-6, the terminal field agreed with an independent dense solve to 1.11e-16, and all 1,800 box-simplex KKT stress cases passed.`

### Turn 10 — `execute_in_remote_docker` (`31720 ms` LLM)
- **Rationale**: Audit the exact directional seam coefficients and stratify real tracer residuals so a pooled score cannot hide a bad channel or shift. Compare staged and sweep-only calibration through both outputs; a material discrepancy would identify whether calibration, rather than the wall stencil, needs correction.
- **Alternatives Weighed**: `Apply cohort classification or relay-mode tests.` (rejected: This fixture is a continuous inverse problem with no class labels or relay states. Channel and shift residuals before and after nuisance projection are the relevant stratified checks.)

```bash
cat > /tmp/audit_periodic_campaign.py <<'PY'
import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from batch_solver import _PAY, solve_batch
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _responses, solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    fixture = pickle.load(f)
for item in fixture:
    s = item['study']
    calibrated = [calibrate_upstream(s, _PAY), calibrate_upstream(s)]
    S, last = _responses(calibrated[0])
    pulses = item['campaign_diagnostics']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    prediction = S @ W
    ns = len(s['mu_shifts'])
    for name, pay in [('staged', _PAY), ('sweep-only', None)]:
        Y = np.column_stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), pay)['y_obs_raw'] for p in pulses])
        e = (prediction-Y).reshape(ns, -1, 4, len(pulses))
        rms = lambda a, axis=None: np.sqrt(np.mean(a*a, axis=axis))
        print(name, 'worst shift/channel/pulse RMS', rms(e,1).max(), 'per-channel RMS', rms(e,(0,1,3)))
        print(name, 'worst whole pulse/shift RMS', rms(e,(1,2)).max(), 'peak abs', abs(e).max())
        Z = np.asarray(s['Z_nuisance'])
        beta = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ (prediction-Y))
        ep = (prediction-Y-Z@beta).reshape(e.shape)
        print(name, 'worst nuisance-profiled shift/channel/pulse RMS', rms(ep,1).max())
        assert rms(e,(1,2)).max() < 2e-5
    outs = [solve_batch(s), solve_downstream(calibrated[1])]
    print('calibration sensitivity: weights and terminal field max absolute change', *[np.max(abs(outs[0][key]-outs[1][key])) for key in ('weights', 'terminal_field')])
    for d in 'xy':
        k, v = (np.asarray(s[key+'_'+d]) for key in ('kappa','vel'))
        h = s['L'+d]/(len(k)-1)
        a = _axis(k,v,h).toarray()
        seam = (k[0]+k[-1])/(2*h**2)
        err = max(abs(a[0,-1]+seam+max(v[0],0)/h), abs(a[-1,0]+seam+max(-v[-1],0)/h), np.max(abs(a.sum(axis=1))))
        print(d, 'directional seam and constant-mode maximum error', err)
        assert err < 1e-12
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_periodic_campaign.py && python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
staged worst shift/channel/pulse RMS 7.81771450220318e-06 per-channel RMS [3.56658172e-06 2.86272656e-06 7.50189230e-06 3.91118307e-06]
staged worst whole pulse/shift RMS 4.997337453052916e-06 peak abs 8.172187651478424e-06
staged worst nuisance-profiled shift/channel/pulse RMS 7.402372621930658e-06
sweep-only worst shift/channel/pulse RMS 2.274502123158796e-06 per-channel RMS [8.25765579e-08 6.47691779e-07 1.97667026e-06 4.51836868e-07]
sweep-only worst whole pulse/shift RMS 1.2588915375300487e-06 peak abs 2.614698462910514e-06
sweep-only worst nuisance-profiled shift/channel/pulse RMS 2.077453591260254e-06
calibration sensitivity: weights and terminal field max absolute change 1.9839877968275133e-05 1.363985693947356e-06
x directional seam and constant-mode maximum error 3.552713678800501e-15
y directional seam and constant-mode maximum error 3.552713678800501e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=58bc8257ebad4632 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_58bc8257 (directory).
```

### Turn 11 — `submit_completed_task` (`19051 ms` LLM)
- **Rationale**: The exact stencil accounts for both directional seam terms without fitted wall coefficients. The isolated audit found no failing channel or shift, and the unchanged bundle passed independent field, certificate, and active-bound verification.
- **Alternatives Weighed**: `Replace the staged upstream calibration with a sweep-only fit.` (rejected: The sweep-only fit reduced tracer residuals, but replacing the verified upstream handoff is outside this downstream repair. The measured maximum changes were 1.98e-5 in weights and 1.36e-6 in the terminal field.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/downstream_solver.py` and verified it through the staged `/holdout_task/batch_solver.py`. The public schema passed. With the staged calibration, worst tracer RMS across shifts was 5.00e-6, the terminal field agreed with an independent dense solve to 1.11e-16, and all 1,800 box-simplex KKT stress cases passed.`

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
