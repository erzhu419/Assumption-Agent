# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m100_01_batch_02__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `14 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `408,143` (`392,212` / `115,456` / `15,931`) |
| **Total Episode Wall Time (s)** | `496.33s` |

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

### Turn 1 — `execute_in_remote_docker` (`19120 ms` LLM)
- **Rationale**: Inspect the staged composition and metadata to avoid applying calibration twice, and identify whether pulse measurements or archived telemetry can distinguish boundary operators before choosing one.
- **Alternatives Weighed**: `Implement the baseline immediately` (rejected: The staged interface and campaign evidence determine which coordinate transforms and boundary rows the solver needs.)

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py && python3 - <<'PY'
import json,os,pickle,numpy as np
with open('public_batch.pkl','rb') as f: b=pickle.load(f)
print('BATCH',len(b))
for i,it in enumerate(b[:3]):
 print('ITEM',i,list(it)); s=it['study']; d=it.get('campaign_diagnostics'); print('DIAGNOSTICS',None if d is None else (len(d),list(d[0]) if len(d) else []))
 for k,v in s.items():
  print(k,type(v).__name__,v.shape if hasattr(v,'shape') else list(v) if isinstance(v,dict) else (len(v) if isinstance(v,(tuple,list)) else v))
p='/tmp/hegel_continual_archive'; print('ARCHIVE',os.listdir(p) if os.path.isdir(p) else None)
for n in ['archive_manifest.json','archive_apparatus_log.json','pre_llm_gate_diagnostic.json']:
 fn=os.path.join(p,n)
 if os.path.isfile(fn):
  a=json.load(open(fn)); print(n,repr(a)[-8500:])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT active set' 2>&1 | head -40
```
- **Container Exit Code**: `0` (`444 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
hort_name': 'dev_beta_m100_01_batch_01', 'facility_telemetry_trace': [[-0.84085, -0.84992, -0.75803, -0.23384, -0.69788, 0.05121, 0.70335, 2.44987], [-0.80679, -0.69476, -0.18238, -0.27114, -0.82901, -0.58941, 0.89118, 2.6746], [-0.71013, -0.28212, 0.44132, -0.36751, -0.80597, -0.8556, 0.81997, 2.49876], [-0.5621, 0.12155, 0.85819, -0.35596, -0.56822, -0.55206, 0.79065, 2.53508], [-0.33724, 0.56372, 0.74052, -0.29233, -0.11042, -0.04449, 0.75805, 2.54205], [-0.17814, 0.78154, 0.17119, -0.34482, 0.30316, 0.57291, 0.57438, 2.47673], [0.07147, 0.88381, -0.47238, -0.42572, 0.65726, 0.8559, 0.39254, 2.5599], [0.30424, 0.63053, -0.86801, -0.49322, 0.83476, 0.57012, 0.18135, 2.59806], [0.43853, 0.29209, -0.72116, -0.30572, 0.88043, -0.06311, -0.09098, 2.57085], [0.64243, -0.1622, -0.23106, -0.4564, 0.57858, -0.62987, -0.30236, 2.59386], [0.77317, -0.52864, 0.49639, -0.40231, 0.19624, -0.82677, -0.47523, 2.49275], [0.83462, -0.80662, 0.81848, -0.39144, -0.23836, -0.63291, -0.69553, 2.60291], [0.85694, -0.85099, 0.71146, -0.2216, -0.62999, -0.00174, -0.69093, 2.4587], [0.79739, -0.66599, 0.18548, -0.3426, -0.83172, 0.5994, -0.79581, 2.46649], [0.66483, -0.28019, -0.41838, -0.25877, -0.84423, 0.8464, -0.80781, 2.65844], [0.55501, 0.18346, -0.78394, -0.42122, -0.49805, 0.63502, -0.79155, 2.48224], [0.39152, 0.59442, -0.68846, -0.41462, -0.15414, -0.00059, -0.6849, 2.53617], [0.15981, 0.79192, -0.21936, -0.34322, 0.2834, -0.58685, -0.59791, 2.58152], [-0.09902, 0.83881, 0.43486, -0.33661, 0.68063, -0.80792, -0.34111, 2.46773], [-0.30649, 0.65407, 0.82429, -0.41993, 0.86384, -0.59527, -0.1717, 2.55298], [-0.52358, 0.2863, 0.71578, -0.3963, 0.78551, -0.02606, 0.02462, 2.6031], [-0.64698, -0.08956, 0.23433, -0.42752, 0.5496, 0.57679, 0.24479, 2.54726], [-0.75148, -0.55865, -0.38863, -0.36872, 0.1268, 0.90517, 0.45156, 2.56119], [-0.90514, -0.81764, -0.83327, -0.37205, -0.26075, 0.63913, 0.55443, 2.72128]]}]
pre_llm_gate_diagnostic.json {'pre_llm_gate_accepted': False, 'repair_target': 'downstream_module_solve', 'failure_kind': 'scientific_or_numerical'}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`22572 ms` LLM)
- **Rationale**: Read compact input shapes, validator expectations, and archive metadata to locate the current campaign evidence and avoid transferring the wrong boundary operator.
- **Alternatives Weighed**: `Infer the current wall mode from the telemetry tail alone` (rejected: The previous output was truncated; the visible tail identifies a completed campaign, not necessarily the current one.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import json,pickle,pathlib,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('items',len(b))
for i,it in enumerate(b):
 s=it['study']; d=it.get('campaign_diagnostics'); print(i,'diagnostics',None if d is None else len(d));
 if i == 0:
  for k,v in s.items():
   if isinstance(v,dict): print(k,{a:np.shape(z) for a,z in v.items() if not isinstance(z,dict)})
   elif np.isscalar(v): print(k,v)
   else: print(k,np.shape(v))
print('asset index',pathlib.Path('public_asset_index.json').read_text()[:3000]); p=pathlib.Path('/tmp/hegel_continual_archive'); print('archive files',[(x.name,x.stat().st_size) for x in p.iterdir()]); m=json.loads((p/'archive_manifest.json').read_text()); print('manifest',str(m)[:4500]); a=json.loads((p/'archive_apparatus_log.json').read_text()); print('apparatus metadata',[(k,str(v)[:400]) for k,v in a.items() if not isinstance(v,list)] if isinstance(a,dict) else 'list'); rows=[a] if isinstance(a,dict) else a
if isinstance(a,dict): rows += [z for v in a.values() if isinstance(v,list) for z in v if isinstance(z,dict)]
for z in rows:
 for k,v in z.items():
  if isinstance(v,list):
   try: x=np.asarray(v,float)
   except (ValueError,TypeError): continue
   if x.ndim==2: print({q:w for q,w in z.items() if not isinstance(w,(list,dict))},k,x.shape,'mean',np.round(x.mean(0),3),'std',np.round(x.std(0),3))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -30
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
],
                "type": "ndarray"
      
archive files [('archive_manifest.json', 693), ('pre_llm_gate_diagnostic.json', 127), ('dev_beta_m100_01_s00_solver.py', 4819), ('archive_apparatus_log.json', 11629), ('dev_beta_m100_01_batch_01_solver.py', 4946)]
manifest [{'episode_short_name': 'dev_beta_m100_01_s00', 'task_id': 'dev_beta_m100_01_s00__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/dev_beta_m100_01_s00_solver.py'}, {'episode_short_name': 'dev_beta_m100_01_batch_01', 'task_id': 'dev_beta_m100_01_batch_01__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/dev_beta_m100_01_batch_01_solver.py'}]
apparatus metadata list
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'} facility_telemetry_trace (24, 8) mean [ 0.004 -0.007 -0.004 -0.354  0.009 -0.007  0.002 -0.372] std [0.606 0.589 0.598 0.067 0.611 0.6   0.605 0.069]
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'dev_beta_m100_01_s00'} facility_telemetry_trace (24, 8) mean [-1.000e-03 -9.000e-03  3.000e-03  2.455e+00  5.000e-03  2.000e-03
  3.000e-03 -4.170e-01] std [0.604 0.603 0.608 0.068 0.607 0.599 0.611 0.056]
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'dev_beta_m100_01_batch_01'} facility_telemetry_trace (24, 8) mean [-7.000e-03  1.000e-03  3.000e-03 -3.610e-01  1.100e-02  2.000e-03
 -2.000e-03  2.551e+00] std [0.605 0.609 0.603 0.069 0.611 0.6   0.592 0.07 ]
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
```

### Turn 3 — `execute_in_remote_docker` (`15992 ms` LLM)
- **Rationale**: Read the archived row formulas and the current sensor layout to identify which wall mechanisms to compare and whether the measurements cover those walls.
- **Alternatives Weighed**: `Reuse the latest archived wall operator unchanged` (rejected: The archive shows different apparatus states across campaigns, and the current pulses can test whether the previous operator transfers.)

```bash
cd /holdout_task && diff -u /tmp/hegel_continual_archive/dev_beta_m100_01_s00_solver.py /tmp/hegel_continual_archive/dev_beta_m100_01_batch_01_solver.py; cat /tmp/hegel_continual_archive/dev_beta_m100_01_batch_01_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
it=pickle.load(open('public_batch.pkl','rb'))[0]; s=it['study']; print('telemetry mean and std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('shift',s['mu_shifts']); print('sensors',s['sensor_ij']);
for p in it['campaign_diagnostics']: print('pulse', p['w_pulse'], 'raw range', (min(p['y_pulse_raw']),max(p['y_pulse_raw'])))
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fied face. Accept the
    # refinement only if it improves the certificate and remains feasible.
    free = ~(lower | upper)
    def multiplier(v):
        g = grad(v)
        mask = (v > 1e-9 * cap) & (v < (1-1e-9) * cap)
        if np.any(mask):
            return float(np.mean(g[mask]))
        return float(np.clip(0., np.max(g[v > cap/2]), np.min(g[v <= cap/2])))
    def gap(v, nu):
        d = grad(v) - nu
        return d.clip(0) @ v + (-d).clip(0) @ (cap-v)
    nu = multiplier(w)
    if np.any(free):
        v = np.where(upper, cap, 0.)
        hf = H[np.ix_(free, free)]
        lin = c[free] - H[free] @ v
        x, p = np.linalg.solve(hf, np.column_stack((lin, np.ones(len(lin))))).T
        mu = (budget - v.sum() - x.sum()) / p.sum()
        v[free] = x + mu * p
        if np.min(v) >= 0. and np.max(v) <= cap and gap(v, mu) <= gap(w, nu):
            w, nu = v, float(mu)
    return w, nu


def solve_batch(study):
    S, U_last = _forward(study)
    A, b = _project(study, S, _calibrate(study))
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(A, b, ridge, cap, budget)
    r = A @ w - b
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    g = A.T @ r + ridge * w
    d = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': J,
                                'dual_lower_bound': float(J - g @ w + nu * budget +
                                                          cap * np.minimum(0, d).sum())}}
telemetry mean and std [-0.19312945 -0.23598127 -0.00518928  0.12786467] [0.81246823 1.03681144 1.28370291 0.79028067]
shift [0.15 0.55 1.1 ]
sensors [[ 0  2]
 [12 10]
 [ 2  6]
 [ 2  8]
 [ 2  4]
 [ 8  6]
 [10  4]
 [12  8]
 [12  2]
 [ 6  8]
 [12  0]
 [ 8  2]
 [ 6 10]
 [12  4]
 [ 8  0]
 [ 0  6]
 [ 6  0]
 [ 8 10]
 [ 2  2]
 [ 4  0]
 [ 4  6]
 [10  2]
 [12  6]
 [ 2 10]
 [ 0  8]
 [10  8]
 [ 0  0]
 [ 2  0]
 [10  0]
 [ 8  4]
 [ 6  2]
 [ 4  2]]
pulse [1.  0.  0.  0.5 0.  0.  0.  0. ] raw range (-0.021451323595624357, 0.4930712099239312)
pulse [0.  1.  0.  0.  0.5 0.  0.  0. ] raw range (-0.013859433660747142, 0.5417745711418475)
pulse [0.  0.  1.  0.  0.  0.5 0.  0. ] raw range (-0.019255917140160998, 0.5123785327450897)
```

### Turn 4 — `execute_in_remote_docker` (`38031 ms` LLM)
- **Rationale**: Compare archived operators and nested wall stencils on calibrated tracer pulses. Hold out each pulse and check whether nuisance projection changes the inferred wall law before transferring it to the downstream solver.
- **Alternatives Weighed**: `Choose Robin or reflecting from the apparatus log alone` (rejected: The log describes prior campaigns; current tracer measurements can directly test both boundary laws.)

```bash
sed -n '1,68p' /tmp/hegel_continual_archive/dev_beta_m100_01_batch_01_solver.py
cat > /tmp/wall_probe.py <<'PY'
import sys,pickle,json
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pul=it['campaign_diagnostics']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); y=np.array([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pul]); w=np.array([p['w_pulse'] for p in pul]); nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T; sensor=i+nx*j; nsen=len(i); z=s['Z_nuisance']; I=sparse.eye(nx*ny,format='csc'); src=(w@s['source_modes'].reshape(len(w[0]),-1,order='F')).T

def axis(k,v,h,r,b):
 f=(k[1:]+k[:-1])/(2*h*h); l=-f-np.maximum(v[1:],0)/h; u=-f-np.maximum(-v[:-1],0)/h
 l[-1]=-k[-1]/h**2*b[1,0]; u[0]=-k[0]/h**2*b[0,0]; d=-np.r_[0,l]-np.r_[u,0]; d[[0,-1]]+=r/h*b[:,1]; return sparse.diags([l,d,u],[-1,0,1],format='csc')

def predict(p):
 p=np.array(p).reshape(4,2); ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:2]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[2:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

def resid(p, pi=slice(None),si=slice(None),nuis=False):
 r=(predict(p)-y)[pi][:,si]; q=z[si]; return r-(q@np.linalg.lstsq(q,r.T,rcond=None)[0]).T if nuis else r

def fit(decode,p0,pi=slice(None),si=slice(None),nuis=False):
 return least_squares(lambda p:resid(decode(p),pi,si,nuis).ravel(),p0,bounds=(-.08,12),xtol=1e-13,gtol=1e-13,ftol=1e-13)

def rms(a,axis=None): return np.sqrt(np.mean(a*a,axis=axis))
if __name__=='__main__':
 for x in [0,1,2]:
  for a in [0,1,2]: print('uniform end leak',x,a,'rms',rms(resid([1,x,1,x,1,a,1,a]),1))
 configs=[('leak-all',lambda p: np.c_[np.ones(4),p].ravel(),np.ones(4)),('free-wall',lambda p:p,np.ones(8)),('axis-symmetric',lambda p:np.repeat(np.reshape(p,(2,2)),2,axis=0).ravel(),np.ones(4))]
 for name,dec,start in configs:
  for nuis in (False,True):
   f=fit(dec,start,nuis=nuis); p=dec(f.x); print(name,nuis,'p',p,'res',rms(resid(p,nuis=nuis),1),'sing',np.linalg.svd(f.jac,compute_uv=False))
   for k in range(len(y)):
    f=fit(dec,start,np.arange(len(y))!=k,nuis=nuis); print('loo',k,dec(f.x),rms(resid(dec(f.x),np.array([k]),nuis=nuis)))
   print('mean/res by shift',resid(p,nuis=nuis).reshape(3,3,-1).mean(2),rms(resid(p,nuis=nuis).reshape(3,3,-1),2))
 print('nuisance basis range',z.min(0),z.max(0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `0` (`8672 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
602 0.01206592] sing [7.91821167 3.37184066 1.76002492 0.57023945 0.20994991 0.13226297
 0.05818091 0.01199602]
loo 0 [ 1.31444797  0.09516648  3.57746921  0.14559543  0.78238916 -0.05719181
  0.41647037 -0.05446388] 0.024998647333394994
loo 1 [ 0.57911343 -0.05503336  0.65154062 -0.04949901  0.90845188  0.05309003
 11.21061252  0.75342813] 0.03660243815897214
loo 2 [ 1.16502671  0.09348509  8.52281376  0.14875265  0.46329425 -0.04265355
  0.80810389 -0.06550916] 0.01932583711628621
mean/res by shift [[-0.01305986 -0.00508052 -0.00214507]
 [ 0.00615179 -0.00097456  0.00075319]
 [ 0.00199471 -0.00274454 -0.00283756]] [[0.02767565 0.01721757 0.01162257]
 [0.02118769 0.01556944 0.01236847]
 [0.01803615 0.00883657 0.00577679]]
axis-symmetric False p [ 1.76470355  0.11374519  1.76470355  0.11374519  0.53724845 -0.06270926
  0.53724845 -0.06270926] res [0.02168788 0.02027829 0.00851239] sing [12.13840445  0.95589585  0.35905308  0.03592187]
loo 0 [ 1.6531878   0.13235932  1.6531878   0.13235932  0.6186555  -0.06696274
  0.6186555  -0.06696274] 0.022884803548156193
loo 1 [ 0.65705303 -0.05216192  0.65705303 -0.05216192  1.17114705  0.0556721
  1.17114705  0.0556721 ] 0.03279831898533006
loo 2 [ 2.05838008  0.17556154  2.05838008  0.17556154  0.4450423  -0.07746701
  0.4450423  -0.07746701] 0.010584222583801532
mean/res by shift [[-0.01238745 -0.00542012 -0.00180428]
 [ 0.00636379 -0.00171908 -0.00109324]
 [ 0.00096296 -0.00476125 -0.00321267]] [[0.02989395 0.01944494 0.01180418]
 [0.02580614 0.01901831 0.01435181]
 [0.01216007 0.00688153 0.00470742]]
axis-symmetric True p [ 1.7437511   0.10946578  1.7437511   0.10946578  0.53828678 -0.06179591
  0.53828678 -0.06179591] res [0.02147137 0.01992598 0.00841127] sing [12.09442764  0.9623989   0.35289384  0.03651515]
loo 0 [ 1.6228711   0.12585976  1.6228711   0.12585976  0.61597022 -0.06576947
  0.61597022 -0.06576947] 0.022598458967050122
loo 1 [ 0.65385672 -0.05235849  0.65385672 -0.05235849  1.18753588  0.0575805
  1.18753588  0.0575805 ] 0.03228807539391901
loo 2 [ 2.04079408  0.16968167  2.04079408  0.16968167  0.44390204 -0.07640352
  0.44390204 -0.07640352] 0.010459104668775835
mean/res by shift [[-0.01208272 -0.00548977 -0.00194041]
 [ 0.00620722 -0.0021136   0.00065056]
 [ 0.00090492 -0.00462237 -0.00337695]] [[0.02959227 0.01884016 0.01234523]
 [0.02606118 0.01785371 0.01389942]
 [0.01202342 0.00669074 0.00478743]]
nuisance basis range [-2.46797804 -2.32701866 -3.50708586] [2.22684619 3.7984934  2.03128809]
```

### Turn 5 — `execute_in_remote_docker` (`29283 ms` LLM)
- **Rationale**: Test a cross-wall stencil that the local coefficient fit cannot represent. Select the discrete wall law inside each leave-one-pulse-out fold to see whether it predicts unseen tracer measurements.
- **Alternatives Weighed**: `Deploy the eight-parameter local wall fit` (rejected: Wall coefficients move substantially when a pulse is withheld, and the fit still leaves systematic error at the smallest shift.)

```bash
cat > /tmp/wall_discrete.py <<'PY'
from wall_probe import *
from itertools import product
from functools import lru_cache

def ax(k,v,h,r,kind):
 a=axis(k,v,h,r,np.array([[1.,1.],[1.,1.]])).tolil(); n=len(k)
 for t,b,j in [(0,n-1,1),(n-1,0,n-2)]:
  if kind in ('open','reflect'):
   if kind=='reflect': a[t,t]-=r/h
  elif kind=='pin':
   a[t,j]=0; a[t,t]=1/h**2
  else:
   f=k[t] if kind=='periodic-local' else (k[t]+k[j])/2
   g=k[t] if kind=='periodic-local' else (k[t]+k[b])/2
   if kind=='periodic-upwind':
    f+=h*max(v[t] if t==0 else -v[t],0); g+=h*max(-v[t] if t==0 else v[t],0)
   a[t,j]=-f/h**2; a[t,b]=-g/h**2; a[t,t]=(f+g)/h**2
 return a.tocsc()

@lru_cache(None)
def pred(x,q):
 A=ax(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],x); B=ax(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],q)
 L=sparse.kron(sparse.eye(ny),A,format='csc')+sparse.kron(B,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

if __name__=='__main__':
 kinds=['open','reflect','pin','periodic-local','periodic-mid','periodic-upwind']; candidates=list(product(kinds,repeat=2)); errors=np.array([rms(pred(*p)-y,1) for p in candidates]); order=np.argsort(np.max(errors,1));
 for j in order[:10]:
  p=candidates[j]; print(p,errors[j], 'shift RMSE',rms((pred(*p)-y).reshape(3,3,-1),2).round(6).tolist())
 for p in [('open','open'),('reflect','reflect')]: print('archive baseline',p,rms(pred(*p)-y,1))
 for k in range(3):
  j=np.argmin(np.mean(errors[:,np.arange(3)!=k]**2,1)); print('LOPO',k,candidates[j],errors[j,k])
 print('sensor range', y.min(),y.max()); print('sweeps',np.quantile(np.abs(s['sensor_calibration_sweeps']['raw_rosette']),[0.5,.9,.95,.98,1])); print('pulse data by row bank',np.unique(s['row_bank_id'],return_counts=True))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_discrete.py
```
- **Container Exit Code**: `0` (`547 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
('periodic-mid', 'periodic-mid') [0.00157306 0.00102098 0.00026227] shift RMSE [[0.002354, 0.001199, 0.000665], [0.001492, 0.000781, 0.00054], [0.000418, 0.000153, 9.1e-05]]
('periodic-local', 'periodic-mid') [0.0024175  0.00036624 0.00069669] shift RMSE [[0.003654, 0.001803, 0.000964], [0.000428, 0.00036, 0.000299], [0.001, 0.000583, 0.000342]]
('periodic-mid', 'periodic-upwind') [0.00216587 0.00249517 0.00237546] shift RMSE [[0.003237, 0.001661, 0.000914], [0.003826, 0.001744, 0.000997], [0.003679, 0.001624, 0.000869]]
('periodic-local', 'periodic-upwind') [0.00300427 0.0016714  0.00158128] shift RMSE [[0.00453, 0.002258, 0.001207], [0.002591, 0.001117, 0.000646], [0.002417, 0.001106, 0.000659]]
('periodic-upwind', 'periodic-mid') [0.00116522 0.00415211 0.00253516] shift RMSE [[0.00163, 0.000978, 0.000678], [0.006126, 0.003199, 0.001991], [0.003872, 0.001823, 0.000983]]
('periodic-upwind', 'periodic-upwind') [0.00089397 0.00567523 0.00463672] shift RMSE [[0.001093, 0.000864, 0.000675], [0.008537, 0.004204, 0.002466], [0.007144, 0.003254, 0.001697]]
('periodic-upwind', 'periodic-local') [0.0097546  0.00385205 0.00488782] shift RMSE [[0.014431, 0.007597, 0.004415], [0.005684, 0.002859, 0.00201], [0.0077, 0.003135, 0.001598]]
('periodic-mid', 'periodic-local') [0.01186773 0.00149352 0.00263037] shift RMSE [[0.017735, 0.009061, 0.005089], [0.001691, 0.001506, 0.001252], [0.004219, 0.001525, 0.000791]]
('periodic-local', 'periodic-local') [0.01264677 0.00157933 0.00184822] shift RMSE [[0.018946, 0.009606, 0.005347], [0.001702, 0.001694, 0.001309], [0.002961, 0.001049, 0.000618]]
('periodic-upwind', 'reflect') [0.02013664 0.00542693 0.00600399] shift RMSE [[0.026954, 0.018296, 0.012458], [0.006955, 0.004971, 0.003908], [0.009329, 0.003995, 0.002272]]
archive baseline ('open', 'open') [0.11210111 0.12475805 0.10939667]
archive baseline ('reflect', 'reflect') [0.0215246  0.02433235 0.00532423]
LOPO 0 ('periodic-local', 'periodic-mid') 0.002417503019827961
LOPO 1 ('periodic-mid', 'periodic-mid') 0.001020983334273112
LOPO 2 ('periodic-mid', 'periodic-mid') 0.00026226631077090753
sensor range 0.043173583882108894 0.38674811812956844
sweeps [0.70370707 1.54282624 1.85691482 2.08780173 2.30760347]
pulse data by row bank (array([0, 1, 2, 3]), array([24, 24, 24, 24]))
```

### Turn 6 — `execute_in_remote_docker` (`37871 ms` LLM)
- **Rationale**: Compare endpoint face conventions and upwind directions, using continuous fits only to diagnose missing terms. Check whether the upstream already calibrates the rosettes before adding any downstream correction.
- **Alternatives Weighed**: `Deploy the midpoint periodic stencil immediately` (rejected: It identifies the seam, but the fold disagreement and structured residual leave the exact boundary coefficients unresolved.)
- **ATMS NOGOOD DECLARED**: Refuted `The stencil needs no coupling between opposite ends of an axis.` | Witness: `The Robin and reflecting baselines have worst pulse RMSE 0.12476 and 0.02433; periodic coupling is selected in all three held-out-pulse folds and reduces worst held-out RMSE to 0.00242.` | Blanket Diagnosis: `Changing adjacent-cell diffusion and leakage cannot reproduce transport through the opposite wall.` | Sublated Pivot: `Retain the prescribed interior stencil and compare exact cyclic boundary-row formulas.`

```bash
cat /holdout_task/_verified_upstream.py; cat > /tmp/seam_probe.py <<'PY'
from wall_probe import *
from itertools import product

def sax(k,v,h,r,p):
 a=axis(k,v,h,r,np.array([[1.,0.],[1.,0.]])).tolil(); n=len(k)
 for t,b,j,side in [(0,n-1,1,0),(n-1,0,n-2,1)]:
  f,g,adv=p[side]; orient=1-2*side
  inward=(f+h*adv*max(-orient*v[t],0))/h**2
  outward=(g+h*adv*max(orient*v[t],0))/h**2
  a[t,j]=-inward; a[t,b]=-outward; a[t,t]=inward+outward
 return a.tocsc()

def response(p):
 ax=sax(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:2]); ay=sax(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[2:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

ks=[s['kappa_x'],s['kappa_y']]
base=np.array([[(k[t]+k[j])/2,(k[t]+k[b])/2,0.] for k in ks for t,b,j in [(0,-1,1),(-1,0,-2)]])
sw=s['sensor_calibration_sweeps']; cal=calibrate_upstream(s,pay); print('keys after upstream',list(cal)); print('sweep keys',list(sw))
for q in (s,cal):
 u=q['sensor_calibration_sweeps']; ref=u['ref_rosette']; raw=u['raw_rosette']; a=np.c_[ref,np.ones(len(ref))]; m=np.linalg.lstsq(a,raw,rcond=None)[0]; print('forward fit',m.round(7),'RMSE',rms(a@m-raw))
print('base face pairs',base)
for held in [None,0,1,2]:
 train=np.arange(3)!=held
 for n in [4,8,12]:
  def decode(x):
   p=base.copy(); p[:,:2]*=np.array(x[:8]).reshape(4,2) if n>=8 else np.tile(x[:4],2).reshape(4,2)
   if n==12:p[:,2]=x[8:]
   return p
  f=least_squares(lambda x:(response(decode(x))-y)[train].ravel(),np.r_[np.ones(min(n,8)),np.zeros(max(0,n-8))],ftol=1e-12,gtol=1e-12,xtol=1e-12)
  print('fit',held,n,'params',np.round(f.x,5),'faces',decode(f.x).round(5).tolist(),'errs',rms(response(decode(f.x))-y,1))

def grammar(k,inner,outer,adv):
 return [[{'local':k[t],'mid':(k[t]+k[j])/2,'neighbor':k[j]}[inner],{'local':k[t],'mid':(k[t]+k[b])/2,'neighbor':k[b]}[outer],adv] for t,b,j in [(0,-1,1),(-1,0,-2)]]
options=list(product(['local','mid','neighbor'],['local','mid','neighbor'],[0,1])); specs=list(product(options,repeat=2)); errors=np.array([rms(response(grammar(ks[0],*a)+grammar(ks[1],*b))-y,1) for a,b in specs]);
for j in np.argsort(np.max(errors,1))[:6]: print('discrete',specs[j],errors[j])
for held in range(3):
 j=np.argmin(np.mean(errors[:,np.arange(3)!=held]**2,1)); print('discrete LOO',held,specs[j],errors[j])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/seam_probe.py
```
- **Container Exit Code**: `0` (`4381 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
45952], [0.55129, 0.43434, -0.0], [0.31701, 0.16238, 7.62467]] errs [6.44839683e-06 4.17600593e-06 4.37802127e-06]
fit 1 4 params [1.00592 1.02508 1.00296 1.09624] faces [[0.46346, 0.45512, 0.0], [0.46209, 0.48671, 0.0], [0.55505, 0.44511, 0.0], [0.31758, 0.476, 0.0]] errs [1.09042523e-04 2.38849827e-03 8.39991435e-05]
fit 1 8 params [0.99946 1.1451  1.00032 1.08546 0.9993  0.99919 1.00091 1.09579] faces [[0.46048, 0.5084, 0.0], [0.46088, 0.48192, 0.0], [0.5514, 0.43387, 0.0], [0.31694, 0.47581, 0.0]] errs [4.36595537e-06 8.94699419e-06 4.41068799e-06]
fit 1 12 params [ 0.99946  1.32147  1.00032  0.5202   0.9993   0.99919  1.00091  0.50388
 -1.21181  6.60783  0.       6.24038] faces [[0.46048, 0.5867, -1.21181], [0.46088, 0.23096, 6.60783], [0.5514, 0.43387, 0.0], [0.31694, 0.21879, 6.24038]] errs [4.36595585e-06 8.94687420e-06 4.41068751e-06]
fit 2 4 params [1.45345 1.94243 0.70615 1.05932] faces [[0.66965, 0.8624, 0.0], [0.32535, 0.47032, 0.0], [0.80199, 0.84344, 0.0], [0.2236, 0.45997, 0.0]] errs [0.00049156 0.00053845 0.00139934]
fit 2 8 params [0.99951 1.14502 1.00023 1.08582 0.9995  0.99983 1.00063 1.09597] faces [[0.4605, 0.50836, 0.0], [0.46084, 0.48208, 0.0], [0.55151, 0.43414, 0.0], [0.31685, 0.47589, 0.0]] errs [4.42976307e-06 4.31819764e-06 5.00644221e-06]
fit 2 12 params [ 0.99951 -2.76478  1.00023  0.20983  0.9995   0.99983  1.00063 -1.28587
 26.86466 10.24017 -0.      25.11114] faces [[0.4605, -1.22751, 26.86466], [0.46084, 0.09316, 10.24017], [0.55151, 0.43414, -0.0], [0.31685, -0.55835, 25.11114]] errs [4.42976222e-06 4.31819851e-06 5.00644137e-06]
discrete (('mid', 'mid', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
discrete (('mid', 'local', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
discrete (('mid', 'neighbor', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
discrete (('mid', 'mid', 1), ('neighbor', 'mid', 1)) [1.80938432e-04 2.83859183e-05 7.77472549e-05]
discrete (('mid', 'neighbor', 1), ('neighbor', 'mid', 1)) [1.80938432e-04 2.83859183e-05 7.77472549e-05]
discrete (('mid', 'local', 1), ('neighbor', 'mid', 1)) [1.80938432e-04 2.83859183e-05 7.77472549e-05]
discrete LOO 0 (('mid', 'local', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
discrete LOO 1 (('mid', 'local', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
discrete LOO 2 (('mid', 'local', 1), ('mid', 'mid', 1)) [4.80685102e-06 4.82201415e-06 4.80475095e-06]
```

### Turn 7 — `execute_in_remote_docker` (`50368 ms` LLM)
- **Rationale**: Implement the cyclic stencil that transfers across pulses, then apply the specified nuisance projection and certify the box-simplex optimum. Run the public validator now to catch interface or output-shape errors before numerical stress testing.
- **Alternatives Weighed**: `Use independently fitted seam coefficients` (rejected: The exact cyclic stencil predicts every held-out pulse at the measurement noise level without fitting wall coefficients.); `Recalibrate inside the downstream solver` (rejected: The staged upstream already transforms observations to canonical coordinates, and its transformed sweeps show an identity map.)

```bash
cd /holdout_task && cat batch_solver.py && cat > downstream_solver.py <<'PY'
"""Periodic transport and constrained source inversion in calibrated coordinates."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    i = np.arange(kappa.size)
    prev, nxt = (i - 1) % i.size, (i + 1) % i.size
    face = (kappa + kappa[nxt]) / (2 * h * h)
    left = face[prev] + np.maximum(velocity, 0) / h
    right = face + np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[left + right, -left, -right],
                              (np.tile(i, 3), np.r_[i, prev, nxt])),
                             shape=(i.size, i.size))


def _forward(st):
    nx, ny = st['nx'], st['ny']
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx'] / (nx - 1))
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly'] / (ny - 1))
    L = (sparse.kron(sparse.eye(ny), ax, format='csc')
         + sparse.kron(ay, sparse.eye(nx), format='csc')
         + sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = st['sensor_ij']
    at = ij[:, 0] + nx * ij[:, 1]
    sampled = []
    for mu in st['mu_shifts']:
        U = splu(L + mu * eye).solve(rhs)
        sampled.append(U[at])
    return np.vstack(sampled), U.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(st, S):
    Z = np.asarray(st['Z_nuisance'])
    both = np.column_stack((S, st['y_obs_raw']))
    gram = Z.T @ Z + st['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = both - Z @ np.linalg.solve(gram, Z.T @ both)
    return projected[:, :-1], projected[:, -1]


def _qp(A, b, ridge, cap, budget):
    n = A.shape[1]
    H = A.T @ A + ridge * np.eye(n)
    c = A.T @ b

    def grad(w):
        return H @ w - c

    if budget == 0:
        w = np.zeros(n)
        return w, float(grad(w).min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(grad(w).max())

    # Scale the quadratic, then solve its exact equality system on the active face.
    scale = np.linalg.norm(H, ord=np.inf)
    res = minimize(lambda w: (0.5 * w @ H @ w - c @ w) / scale,
                   np.full(n, budget / n), jac=lambda w: grad(w) / scale,
                   method='SLSQP', bounds=[(0., cap)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                'jac': lambda w: np.ones(n)},
                   options={'ftol': 1e-14, 'maxiter': 500})
    w = res.x
    lower, upper = w <= 1e-8 * cap, w >= (1 - 1e-8) * cap
    free = ~(lower | upper)

    def gap(v, nu):
        d = grad(v) - nu
        return np.maximum(d, 0) @ v + np.maximum(-d, 0) @ (cap - v)

    g = grad(w)
    if np.any(free):
        nu = float(g[free].mean())
        v = np.where(upper, cap, 0.)
        hf = H[np.ix_(free, free)]
        rhs = c[free] - H[free] @ v
        x, direction = np.linalg.solve(hf, np.column_stack((rhs, np.ones(rhs.size)))).T
        mu = (budget - v.sum() - x.sum()) / direction.sum()
        v[free] = x + mu * direction
        if v.min() >= 0 and v.max() <= cap and gap(v, mu) <= gap(w, nu):
            return v, float(mu)
    else:
        nu = float((g[upper].max() + g[lower].min()) / 2)
        v = np.where(upper, cap, 0.)
        if v.sum() == budget and gap(v, nu) <= gap(w, nu):
            return v, nu
    return w, nu


def solve_downstream(cal_item: dict) -> dict:
    S, U_last = _forward(cal_item)
    A, b = _project(cal_item, S)
    ridge, cap, budget = (cal_item[key] for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(A, b, ridge, cap, budget)
    residual = A @ w - b
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = A.T @ residual + ridge * w
    d = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': objective,
                                'dual_lower_bound': float(objective - g @ w + nu * budget
                                                          + cap * np.minimum(0, d).sum())}}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
from __future__ import annotations
import inspect, json
import _verified_upstream as _up
import downstream_solver as _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b999a532bf44b45f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b999a532 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`45995 ms` LLM)
- **Rationale**: Run the saved artifacts in isolation and compare them with an independent dense forward solve. Check calibrated tracer residuals by shift and wall location, then test the KKT certificate at the observed budget and at tight and saturated budgets.
- **Alternatives Weighed**: `Submit after schema validation` (rejected: The schema does not check periodic transport accuracy, terminal-field orientation, or certificate validity on active bounds.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')

items = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
worst = dict(pulse_rmse=0., pulse_shift_wall_rmse=0., mass=0., box=0., stationarity=0., dual_gap=0., complementarity=0., forward=0., field=0.)
def record(key, value):
    worst[key] = max(worst[key], float(value))

def reference_axis(k, vel, h):
    n = len(k)
    a = np.zeros((n, n))
    for i in range(n):
        for j, upstream in (((i - 1) % n, vel[i]), ((i + 1) % n, -vel[i])):
            coeff = (k[i] + k[j]) / (2*h**2) + max(0., upstream)/h
            a[i,j] -= coeff
            a[i,i] += coeff
    return a

for item in items:
    st = item['study']
    out = batch.solve_batch(st)
    cal = calibrate_upstream(st, batch._PAY)
    nx, ny = st['nx'], st['ny']
    ax = reference_axis(st['kappa_x'], st['vel_x'], st['Lx']/(nx-1))
    ay = reference_axis(st['kappa_y'], st['vel_y'], st['Ly']/(ny-1))
    lin = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx)) + np.diag(st['V_reaction_2d'].flatten(order='F'))
    modes = np.stack([m.flatten(order='F') for m in st['source_modes']], axis=1)
    ix, iy = st['sensor_ij'].T
    boundary = (ix == 0) | (ix == nx-1) | (iy == 0) | (iy == ny-1)
    blocks = []
    for shift in st['mu_shifts']:
        uref = np.linalg.solve(lin + shift*np.eye(nx*ny), modes)
        blocks.append(uref[ix + nx*iy])
    S = np.vstack(blocks)
    S_impl, _ = dn._forward(cal)
    record('forward', np.max(np.abs(S-S_impl)))
    shift_count = len(blocks)
    for pulse in item.get('campaign_diagnostics') or []:
        y = calibrate_upstream(dict(st, y_obs_raw=pulse['y_pulse_raw']), batch._PAY)['y_obs_raw']
        e = (S_impl @ pulse['w_pulse'] - y).reshape(shift_count, -1)
        record('pulse_rmse', np.sqrt(np.mean(e**2)))
        for mask in (boundary, ~boundary):
            if mask.any():
                record('pulse_shift_wall_rmse', np.sqrt(np.mean(e[:,mask]**2, axis=1)).max())
    z = st['Z_nuisance']
    P = np.eye(len(S)) - z @ np.linalg.solve(z.T @ z + st['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    A, b = P @ S, P @ cal['y_obs_raw']
    ridge, cap, budget = (st[k] for k in ('source_ridge','u_box','total_budget'))
    w = out['weights']
    cert = out['kkt_certificate']
    lower, upper = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
    g = A.T @ (A @ w - b) + ridge*w
    J = (.5*np.linalg.norm(A @ w-b)**2 + .5*ridge*(w@w))
    assert np.allclose(cert['weights'],w) and np.min(lower) >= 0 and np.min(upper) >= 0
    assert abs(J - cert['primal_objective']) < 1e-11
    assert cert['dual_lower_bound'] <= J + 1e-11
    record('stationarity', np.max(np.abs(g-cert['nu_sum']-lower+upper)))
    record('field', np.max(np.abs(out['terminal_field']-(uref @ w).reshape((nx,ny), order='F'))))
    record('dual_gap', abs(J-cert['dual_lower_bound']))
    for test_budget in (0., cap*1e-7, cap, 2.5*cap, budget, (len(w)-1)*cap, (len(w)-1e-7)*cap, len(w)*cap):
        # Alter the observation target to exercise both faces and tight budgets.
        for y_test in (b, b + 3*A[:,0], b - 3*A[:,0]):
            v, nu = dn._qp(A,y_test,ridge,cap,test_budget)
            d = A.T @ (A @ v-y_test) + ridge*v - nu
            record('mass', abs(v.sum()-test_budget))
            record('box', max(0., -v.min(), v.max()-cap))
            record('complementarity', np.maximum(d,0)@v+np.maximum(-d,0)@(cap-v))
            record('dual_gap', abs(d@v - nu*(test_budget-v.sum()) - cap*np.minimum(d,0).sum()))
print('public_items', len(items), 'worst_metrics', worst)
assert max(worst[k] for k in ('mass','box','stationarity','forward','field')) < 1e-10
assert max(worst[k] for k in ('dual_gap','complementarity')) < 1e-8
assert worst['pulse_rmse'] < 1e-5 and worst['pulse_shift_wall_rmse'] < 2e-5
PY
```
- **Container Exit Code**: `0` (`529 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b999a532bf44b45f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b999a532 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`42491 ms` LLM)
- **Rationale**: Run the numerical checks separately from the schema validator. A dense reference will detect stencil or field-ordering errors, tracer slices will test wall prediction, and tight or saturated budgets will expose active-bound certificate failures. Leave the verified bundle unchanged unless a numerical check fails.
- **Alternatives Weighed**: `Submit on the schema receipt alone` (rejected: The previous invocation printed only the schema result; it did not report the numerical stress metrics.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

def axis(k, v, h):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        for j, vel in (((i-1)%n,v[i]),((i+1)%n,-v[i])):
            value = (k[i]+k[j])/(2*h*h) + max(0,vel)/h
            a[i,i] += value
            a[i,j] -= value
    return a

with open('/holdout_task/public_batch.pkl','rb') as f:
    items = pickle.load(f)
worst = {k:0. for k in ('forward','field','pulse','pulse_shift_wall','budget','box','stationarity','dual_gap','complementarity','reported_objective','reported_dual')}
def record(k,v):
    assert np.isfinite(v), (k,v)
    worst[k] = max(worst[k],float(v))
for item in items:
    s = item['study']
    cal = calibrate_upstream(s,batch._PAY)
    nx,ny = s['nx'],s['ny']
    ax = axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1))
    ay = axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1))
    L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
    rhs = np.column_stack([m.flatten(order='F') for m in s['source_modes']])
    i,j = s['sensor_ij'].T
    wall = (i==0)|(i==nx-1)|(j==0)|(j==ny-1)
    fields = [np.linalg.solve(L+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
    S = np.vstack([f[i+nx*j] for f in fields])
    impl,_ = dn._forward(cal)
    record('forward',np.max(np.abs(S-impl)))
    for p in item.get('campaign_diagnostics') or []:
        y = calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),batch._PAY)['y_obs_raw']
        e = (impl@p['w_pulse']-y).reshape(len(fields),-1)
        record('pulse',np.sqrt(np.mean(e**2)))
        for mask in (wall,~wall):
            if mask.any(): record('pulse_shift_wall',np.sqrt(np.mean(e[:,mask]**2,axis=1)).max())
    z = s['Z_nuisance']
    P = np.eye(len(S))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    A,b = P@S,P@cal['y_obs_raw']
    ridge,cap,B = (s[k] for k in ('source_ridge','u_box','total_budget'))
    out = batch.solve_batch(s)
    w = out['weights']; cert = out['kkt_certificate']; nu = cert['nu_sum']
    assert np.allclose(cert['weights'],w)
    lo,hi = map(np.asarray,(cert['lam_lower'],cert['lam_upper']))
    assert lo.min()>=0 and hi.min()>=0
    residual = A@w-b
    J = .5*(residual@residual+ridge*(w@w))
    g = A.T@residual+ridge*w
    dual = J-g@w+nu*B+cap*np.minimum(g-nu,0).sum()
    record('reported_objective',abs(J-cert['primal_objective']))
    record('reported_dual',abs(dual-cert['dual_lower_bound']))
    record('stationarity',np.max(np.abs(g-nu-lo+hi)))
    record('field',np.max(np.abs(out['terminal_field']-(fields[-1]@w).reshape((nx,ny),order='F'))))
    def feasible(v,m,target,budget):
        d = A.T@(A@v-target)+ridge*v-m
        record('budget',abs(v.sum()-budget))
        record('box',max(0.,-v.min(),v.max()-cap))
        record('complementarity',np.maximum(d,0)@v+np.maximum(-d,0)@(cap-v))
        record('dual_gap',abs(d@v+m*(v.sum()-budget)-cap*np.minimum(d,0).sum()))
    feasible(w,nu,b,B)
    n=len(w)
    for budget in (0,cap*1e-7,cap,2.5*cap,B,(n-1)*cap,(n-1e-7)*cap,n*cap):
        for target in (b,b+3*A[:,0],b-3*A[:,0]):
            v,m = dn._qp(A,target,ridge,cap,budget)
            feasible(v,m,target,budget)
print('Public items:',len(items),'worst metrics:',worst,flush=True)
assert max(worst[k] for k in ('forward','field','budget','box','stationarity','reported_objective','reported_dual')) < 1e-10
assert max(worst[k] for k in ('dual_gap','complementarity')) < 1e-8
assert worst['pulse'] < 1e-5 and worst['pulse_shift_wall'] < 2e-5
PY
```
- **Container Exit Code**: `0` (`500 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public items: 1 worst metrics: {'forward': 3.608224830031759e-16, 'field': 1.3877787807814457e-16, 'pulse': 4.8220141463450855e-06, 'pulse_shift_wall': 4.991881088193726e-06, 'budget': 4.440892098500626e-16, 'box': 0.0, 'stationarity': 1.1518563880486e-15, 'dual_gap': 3.1086244689504383e-15, 'complementarity': 2.517568377625298e-15, 'reported_objective': 5.204170427930421e-18, 'reported_dual': 1.6254358969902682e-15}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b999a532bf44b45f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b999a532 (directory).
```

### Turn 10 — `submit_completed_task` (`13796 ms` LLM)
- **Rationale**: The unchanged, receipt-bound bundle passed the public schema and isolated numerical checks. An independent dense solve matched the sampled responses and terminal field to floating-point precision, and certificate checks passed at tight and saturated budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4412'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema passed. Worst held-out tracer RMSE was 4.83e-6, and the worst KKT duality gap across tested budgets was 3.11e-15.`

### Turn 11 — `execute_in_remote_docker` (`39194 ms` LLM)
- **Rationale**: Withhold each pulse and sensor bank jointly to test whether the selected stencil depends on one excitation or bank. Compare against distinct seam formulas and measure whether a fresh sweep fit materially improves tracer prediction. Reopen the stencil or calibration only if the corresponding comparison warrants it.
- **Alternatives Weighed**: `Audit relay states or within-batch classification accuracy` (rejected: This is a steady continuous inverse problem with neither relay states nor class labels. Sensor-bank holdout tests the relevant risk that one bank drives seam selection.)

```bash
cat > /tmp/wall_margin_audit.py <<'PY'
import itertools
import pickle
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

def rmse(x):
    return np.sqrt(np.mean(x*x))

def alternate_axis(k, v, h, robin, seam, adv, leak):
    a = dn._axis(k, v, h).tolil()
    for i, j, p in [(0, len(k)-1, 1), (len(k)-1, 0, len(k)-2)]:
        side = 1 if i == 0 else -1
        inner = (k[i]+k[p])/(2*h*h) + adv*max(-side*v[i],0)/h
        coeff = {'mid': (k[i]+k[j])/2, 'harmonic': 2*k[i]*k[j]/(k[i]+k[j]), 'local': k[i], 'reflect': 0}[seam]
        outer = coeff/(h*h) + (adv and seam != 'reflect')*max(side*v[i],0)/h
        a[i,p], a[i,j], a[i,i] = -inner, -outer, inner+outer+leak*robin/h
    return a.tocsc()

for item in items:
    s = item['study']
    pulses = item['campaign_diagnostics']
    W = np.stack([p['w_pulse'] for p in pulses])
    def measure(payload):
        return np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),payload)['y_obs_raw'] for p in pulses])
    y = measure(batch._PAY)
    y_fit = measure(None)
    cal = calibrate_upstream(s,batch._PAY)
    production = W @ dn._forward(cal)[0].T
    nx,ny = s['nx'],s['ny']
    i,j = s['sensor_ij'].T
    bank = np.asarray(s['row_bank_id'])
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ W.T
    eye = sparse.eye(nx*ny,format='csc')
    reaction = sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    opts = [('mid',1,0),('mid',0,0),('harmonic',1,0),('local',1,0),('mid',1,1),('reflect',0,0),('reflect',0,1)]
    ops = [[alternate_axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(n-1),s['robin_'+a],*o) for o in opts] for a,n in [('x',nx),('y',ny)]]
    errors = []
    for ix,iy in itertools.product(range(len(opts)),repeat=2):
        L = sparse.kron(sparse.eye(ny),ops[0][ix],format='csc') + sparse.kron(ops[1][iy],sparse.eye(nx),format='csc') + reaction
        pred = np.vstack([splu(L+mu*eye).solve(rhs)[i+nx*j] for mu in s['mu_shifts']]).T
        if (ix,iy)==(0,0):
            np.testing.assert_allclose(pred,production,atol=1e-12,rtol=0)
        errors.append(pred-y)
    E = np.asarray(errors)
    max_held = 0.
    for p in range(len(pulses)):
        for b in np.unique(bank):
            train = E[:,np.arange(len(pulses))!=p][:,:,bank!=b]
            mse = (train**2).mean(axis=(1,2))
            rank = np.argsort(mse)
            held = rmse(E[rank[0],p,bank==b])
            print('held pulse/bank',p,int(b),'winner',int(rank[0]),'held RMSE',held,'train runner-up/winner',np.sqrt(mse[rank[1]]/mse[rank[0]]))
            assert rank[0]==0
            max_held = max(max_held,held)
    print('worst held pulse/bank RMSE',max_held,'worst pulse RMSE staged',max(map(rmse,production-y)),'worst pulse RMSE sweep refit',max(map(rmse,production-y_fit)))
    assert max_held < 1e-5
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/wall_margin_audit.py
```
- **Container Exit Code**: `0` (`641 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held pulse/bank 0 0 winner 0 held RMSE 5.2343553529179e-06 train runner-up/winner 1.0
held pulse/bank 0 1 winner 0 held RMSE 4.679530688380107e-06 train runner-up/winner 1.0
held pulse/bank 0 2 winner 0 held RMSE 6.154481241170159e-06 train runner-up/winner 1.0
held pulse/bank 0 3 winner 0 held RMSE 2.291101048979506e-06 train runner-up/winner 1.0
held pulse/bank 1 0 winner 0 held RMSE 5.316154497719083e-06 train runner-up/winner 1.0
held pulse/bank 1 1 winner 0 held RMSE 4.681118412864243e-06 train runner-up/winner 1.0
held pulse/bank 1 2 winner 0 held RMSE 6.141861066637046e-06 train runner-up/winner 1.0
held pulse/bank 1 3 winner 0 held RMSE 2.2606317943383498e-06 train runner-up/winner 1.0
held pulse/bank 2 0 winner 0 held RMSE 5.230579134192644e-06 train runner-up/winner 1.0
held pulse/bank 2 1 winner 0 held RMSE 4.640190254656983e-06 train runner-up/winner 1.0
held pulse/bank 2 2 winner 0 held RMSE 6.1731652214201965e-06 train runner-up/winner 1.0
held pulse/bank 2 3 winner 0 held RMSE 2.31176000987003e-06 train runner-up/winner 1.0
worst held pulse/bank RMSE 6.1731652214201965e-06 worst pulse RMSE staged 4.8220141463450855e-06 worst pulse RMSE sweep refit 6.961260185311333e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b999a532bf44b45f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b999a532 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`36435 ms` LLM)
- **Rationale**: Use the existing upstream fitter without overriding it with frozen gains. Recheck the changed composition in an isolated process: a bank-specific residual regression or an invalid certificate on the recalibrated observations would block submission.
- **Alternatives Weighed**: `Keep the frozen payload in the batch wrapper` (rejected: The contract calls for fitting the supplied sweeps dynamically, and the verified upstream fitter reduces the worst tracer residual nearly sevenfold without changing the boundary solver.)

```bash
set -e
cd /holdout_task
cat > batch_solver.py <<'PY'
"""Compose sweep-based sensor calibration with the downstream chamber solver."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, json, pickle, sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _forward
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    items=pickle.load(f)
with open('/holdout_task/_upstream_param_payload.json') as f:
    frozen=json.load(f)
for item in items:
    s=item['study']; cal=calibrate_upstream(s); result=solve_batch(s)
    S,U=_forward(cal)
    p=item['campaign_diagnostics']; W=np.stack([x['w_pulse'] for x in p]); pred=W@S.T
    def observed(payload):
        return np.stack([calibrate_upstream(dict(s,y_obs_raw=x['y_pulse_raw']),payload)['y_obs_raw'] for x in p])
    raw_fit=observed(None); err=pred-raw_fit; previous=pred-observed(frozen)
    rms=lambda x: np.sqrt(np.mean(x*x))
    bank=np.asarray(s['row_bank_id'])
    worst=max(rms(err[i,bank==j]) for i,j in itertools.product(range(len(p)),np.unique(bank)))
    print('worst tracer RMSE frozen vs sweep fit:',max(map(rms,previous)),max(map(rms,err)), 'worst pulse/bank:',worst)
    assert worst < 2e-6
    z=s['Z_nuisance']; P=np.eye(len(S))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    A,b=P@S,P@cal['y_obs_raw']; w=result['weights']; cert=result['kkt_certificate']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']
    r=A@w-b; g=A.T@r+ridge*w; d=g-cert['nu_sum']; J=.5*(r@r+ridge*(w@w)); dual=J-g@w+cert['nu_sum']*budget+cap*np.minimum(d,0).sum()
    np.testing.assert_allclose(w,solve_downstream(cal)['weights'],rtol=0,atol=1e-13)
    np.testing.assert_allclose(cert['weights'],w,rtol=0,atol=1e-13)
    np.testing.assert_allclose(result['terminal_field'],np.einsum('m,mij->ij',w,U),rtol=0,atol=1e-13)
    np.testing.assert_allclose(cert['lam_lower'],np.maximum(d,0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(cert['lam_upper'],np.maximum(-d,0),rtol=0,atol=1e-12)
    assert max(abs(w.sum()-budget),-w.min(),w.max()-cap,abs(J-cert['primal_objective']),abs(dual-cert['dual_lower_bound']),abs(J-dual)) < 1e-10
    print('recalibrated study: budget error',abs(w.sum()-budget),'KKT dual gap',abs(J-dual))
PY
```
- **Container Exit Code**: `0` (`757 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
worst tracer RMSE frozen vs sweep fit: 4.8220141463450855e-06 6.961260185311333e-07 worst pulse/bank: 1.0251873016627035e-06
recalibrated study: budget error 2.220446049250313e-16 KKT dual gap 5.134781488891349e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1076bd0e33d92da1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1076bd0e (directory).
```

### Turn 13 — `execute_in_remote_docker` (`38202 ms` LLM)
- **Rationale**: Run a read-only boundary probe against the saved solver and public measurements. An independent dense reference checks the implemented stencil, and joint pulse-and-bank holdout tests whether a genuinely distinct alternative beats it. If any fold favors a distinct alternative, reopen the wall model; otherwise retain the verified bundle.
- **Alternatives Weighed**: `Reuse the previous mixed edit-and-validation command as resolution evidence` (rejected: The gate did not accept that probe, and schema validation alone cannot distinguish competing wall stencils.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
pulses = item['campaign_diagnostics']
cal = calibrate_upstream(s)
y = np.stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']))['y_obs_raw'] for p in pulses])
w = np.stack([p['w_pulse'] for p in pulses])
production = w @ _forward(cal)[0].T

def axis(k, v, h, robin, kind):
    n = len(k)
    a = np.zeros((n, n))
    for i in range(n):
        if i in (0, n-1) and kind in ('reflect', 'robin'):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
            a[i,i] = -a[i,j] + (robin/h if kind == 'robin' else 0)
            continue
        for j, vel in (((i-1)%n, v[i]), ((i+1)%n, -v[i])):
            rate = (k[i]+k[j])/(2*h**2)
            if kind != 'diffusion_only_at_seam' or 0 < i < n-1:
                rate += max(vel, 0)/h
            a[i,j] -= rate
            a[i,i] += rate
    return a

nx, ny = s['nx'], s['ny']
ix, iy = s['sensor_ij'].T
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ w.T
kinds = ('cyclic', 'diffusion_only_at_seam', 'reflect', 'robin')
ops = [[axis(s['kappa_'+a], s['vel_'+a], s['L'+a]/(n-1), s['robin_'+a], k) for k in kinds] for a,n in [('x',nx), ('y',ny)]]
candidates = list(itertools.product(range(len(kinds)), repeat=2))
pred = []
for a,b in candidates:
    L = np.kron(np.eye(ny), ops[0][a]) + np.kron(ops[1][b], np.eye(nx)) + np.diag(s['V_reaction_2d'].ravel(order='F'))
    pred.append(np.vstack([np.linalg.solve(L+mu*np.eye(nx*ny), rhs)[ix+nx*iy] for mu in s['mu_shifts']]).T)
pred = np.stack(pred)
np.testing.assert_allclose(pred[0], production, atol=1e-12, rtol=0)
distinct = np.max(np.abs(pred-pred[0]),axis=(1,2)) > 1e-12
assert distinct.any()
errors = pred-y
bank = np.asarray(s['row_bank_id'])
worst, min_margin = 0., np.inf
for p,b in itertools.product(range(len(pulses)),np.unique(bank)):
    train = errors[:,np.arange(len(pulses))!=p][:,:,bank!=b]
    score = np.sqrt(np.mean(train**2,axis=(1,2)))
    margin = score[distinct].min()/score[0]
    held = np.sqrt(np.mean(errors[0,p,bank==b]**2))
    assert margin > 1 and held < 2e-6, (p,b,margin,held)
    worst, min_margin = max(worst,held), min(min_margin,margin)
print('Distinct-stencil pulse/bank audit: worst held-out RMSE',worst,'minimum distinct alternative / cyclic training RMSE',min_margin)
PY
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Distinct-stencil pulse/bank audit: worst held-out RMSE 1.0251873016491402e-06 minimum distinct alternative / cyclic training RMSE 172.83006310119654

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1076bd0e33d92da1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1076bd0e (directory).
```

### Turn 14 — `submit_completed_task` (`17161 ms` LLM)
- **Rationale**: The saved bundle passed the public schema and isolated numerical checks. In every holdout fold, the best distinct alternative stencil had at least 172 times the cyclic stencil’s training RMSE. The external gate did not record formal closure of ng_t06; no closure is claimed.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema passed. With per-study sensor calibration, the worst held-out tracer RMSE across pulse and sensor-bank splits was 1.03e-6. The public study’s KKT duality gap was 5.13e-16.`

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
