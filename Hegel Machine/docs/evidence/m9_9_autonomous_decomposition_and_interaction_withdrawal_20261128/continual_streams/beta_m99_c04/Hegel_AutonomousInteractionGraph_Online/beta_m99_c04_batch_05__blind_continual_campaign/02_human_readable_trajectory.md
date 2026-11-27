# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m99_c04_batch_05__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `353,120` (`341,974` / `164,352` / `11,146`) |
| **Total Episode Wall Time (s)** | `328.78s` |

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

### Turn 1 — `execute_in_remote_docker` (`13109 ms` LLM)
- **Rationale**: Inspect the wrapper, data contract, and apparatus log before selecting a boundary model. A changed wall setting would require changing the operator; the wrapper determines where calibration must occur.

```bash
cd /holdout_task && cat batch_solver.py _verified_upstream.py validate_public_schema.py && ls -l /tmp/hegel_continual_archive && cat /tmp/hegel_continual_archive/archive_manifest.json && cat /tmp/hegel_continual_archive/archive_apparatus_log.json && python3 - <<'PY'
import pickle
p=pickle.load(open('public_batch.pkl','rb'))
print('batch size',len(p))
for i in p[:2]:
 print(i.keys(), 'diagnostics', type(i.get('campaign_diagnostics')))
 for k,v in i['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:150])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
470884, 0.31597536, 0.29360822,
       0.2794194
vel_x ndarray (14,) array([ 0.35      ,  0.34500413,  0.33015913,  0.30588881,  0.27288602,
        0.23209293,  0.18467409,  0.13198319,  0.07552447,  0.01690968,
      
vel_y ndarray (12,) array([-0.        , -0.05560533, -0.10899563, -0.1580441 , -0.20079691,
       -0.23555099, -0.26092193, -0.27589907, -0.27988581, -0.27272334,
      
robin_x float None 1.180195883769873
robin_y float None 1.1407550528855488
V_reaction_2d ndarray (14, 12) array([[0.25      , 0.25      , 0.25      , 0.25      , 0.25      ,
        0.25      , 0.25      , 0.25      , 0.25      , 0.25      ,
        0.25  
mu_shifts ndarray (3,) array([0.15, 0.55, 1.1 ])
source_modes ndarray (8, 14, 12) array([[[2.11191803e-02, 5.68859895e-02, 9.69360556e-02, ...,
         1.09419575e-05, 4.78397821e-07, 1.32322936e-08],
        [6.47530627e-02, 1.744
sensor_ij ndarray (32, 2) array([[10,  0],
       [ 4,  2],
       [ 8,  2],
       [ 6,  4],
       [ 6,  8],
       [ 2,  4],
       [ 0,  4],
       [ 8,  4],
       [ 4, 10
row_bank_id ndarray (96,) array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
       2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
    
sensor_calibration_sweeps dict None {'ref_rosette': array([[-1.2       ,  0.        ,  0.        ,  0.        ],
       [-0.4       ,  0.        ,  0.        ,  0.        ],
       [ 0.5
Z_nuisance ndarray (96, 3) array([[-0.09300333, -0.57527804,  1.11476722],
       [ 0.72181442, -0.80470371,  0.57732623],
       [ 0.74098358, -0.37629679, -1.87335619],
      
y_obs_raw ndarray (96,) array([-0.06078901,  0.02786394,  0.69539879,  0.06744906, -0.03403519,
        0.60822055,  0.02658893, -0.05182381,  0.03445458,  0.56910911,
      
aux_manifold_telemetry ndarray (24, 4) array([[-1.62395251, -0.45317532,  1.44587737,  2.09058282],
       [ 1.76290986, -0.37555605,  1.2246719 ,  0.03391205],
       [ 0.6565525 , -1.8550
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`14745 ms` LLM)
- **Rationale**: Read the upstream diagnostic interface and extract prior boundary formulas without dumping telemetry arrays. This identifies which wall laws to compare on calibrated pulses and whether the suggested L0 skill matches the equality-constrained problem.
- **Alternatives Weighed**: `Select the latest archived wall setting without testing it` (rejected: The public campaign has tracer pulses that can distinguish an inherited setting from a new one.)

```bash
cd /holdout_task && cat validate_public_schema.py && tail -45 _verified_upstream.py && ls -l /tmp/hegel_continual_archive && grep -nEi 'robin|boundar|wall|def |archive' /tmp/hegel_continual_archive/*solver.py | head -110; python3 - <<'PY'
import pickle,json,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
for d in p['campaign_diagnostics']:
 print({k:(np.shape(v), np.asarray(v).ravel()[:5].tolist()) for k,v in d.items()})
f=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for d in f:
 print({k:(v if len(str(v))<250 else str(v)[:150]) for k,v in d.items()})
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -25
```
- **Container Exit Code**: `0` (`446 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 1.0, 0.0, 0.0]), 'y_pulse_raw': ((96,), [0.09866087734574837, 0.38808807467395634, 0.3187830515081727, 0.19513207789118436, 0.14550300070086677])}
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': '[[0.27396, 0.16584, -0.45269, -0.81828, -0.3228, 0.90096, -0.15661, -0.05952], [0.46578, -0.24839, -0.38341, -0.79001, -0.27955, 0.49218, -0.30185, -0'}
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m99_c04_s00', 'facility_telemetry_trace': '[[-0.61643, -0.46394, -0.37374, -0.89622, 2.51849, 0.75235, -0.8251, -0.76873], [-0.44979, -0.0562, -0.58125, -0.73775, 2.471, 0.78399, -0.72279, -0.8'}
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m99_c04_batch_01', 'facility_telemetry_trace': '[[-0.00163, 0.6468, 2.54639, 0.27117, -0.39804, 0.08397, 0.77822, 0.76126], [0.19143, 0.28324, 2.6155, 0.02551, -0.17902, 0.632, 0.68131, 0.83375], [0'}
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m99_c04_batch_02', 'facility_telemetry_trace': '[[-0.31525, -0.73499, -0.50689, -0.61922, -0.25033, -0.1684, -0.792, -0.26874], [-0.12624, -0.81827, -0.44524, -0.52014, -0.41205, -0.62742, -0.84656,'}
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m99_c04_batch_03', 'facility_telemetry_trace': '[[-0.34055, 0.08192, -0.45046, 0.77817, 2.40498, 0.81787, -0.79753, -0.77973], [-0.07856, -0.30889, -0.4701, 0.55518, 2.37159, 0.73953, -0.76341, -0.8'}
{'completed_campaign_index': 5, 'completed_campaign_short_name': 'beta_m99_c04_batch_04', 'facility_telemetry_trace': '[[-0.00163, 0.6468, 2.54639, 0.27117, -0.39804, 0.08397, 0.77822, 0.76126], [0.19143, 0.28324, 2.6155, 0.02551, -0.17902, 0.632, 0.68131, 0.83375], [0'}
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
```

### Turn 3 — `execute_in_remote_docker` (`10476 ms` LLM)
- **Rationale**: Recover the upstream return keys and archived function names that were truncated in the previous output. They determine how to calibrate the pulses and which boundary operators to test.

```bash
cd /holdout_task && tail -20 _verified_upstream.py && for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $f"; grep -nE '^def |^class |^_[A-Z]+ *=' "$f"; done && cat public_asset_index.json
```
- **Container Exit Code**: `0` (`322 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"kappa_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "mu_shifts": {
              "dtype": "float64",
              "shape": [
                3
              ],
              "type": "ndarray"
            },
            "nuisance_ridge": 0.08,
            "nx": 14,
            "ny": 12,
            "robin_x": 1.180195883769873,
            "robin_y": 1.1407550528855488,
            "row_bank_id": {
              "dtype": "int64",
              "shape": [
                96
              ],
              "type": "ndarray"
            },
            "sensor_calibration_sweeps": {
              "raw_rosette": {
                "dtype": "float64",
                "shape": [
                  80,
                  4
                ],
                "type": "ndarray"
              },
              "ref_rosette": {
                "dtype": "float64",
                "shape": [
                  80,
                  4
                ],
                "type": "ndarray"
              },
              "single_bank_sweeps": {
                "bank_id": "ndarray",
                "raw_readout": "ndarray",
                "ref_signal": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
                32,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                8,
                14,
                12
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 1.4,
            "u_box": 0.42,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                96
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
      "bytes": 1070
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
```

### Turn 4 — `execute_in_remote_docker` (`21235 ms` LLM)
- **Rationale**: Calibrate the tracer readings once, then compare archived stencils by shift and pulse on both wall and interior sensors. If none matches, use their exact boundary formulas as the starting point for identifying the missing stencil terms.
- **Alternatives Weighed**: `Fit new wall coefficients immediately` (rejected: An archived stencil may already explain all three pulses without fitting; compare exact candidates first.)

```bash
cd /holdout_task && for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $f"; sed -n '1,35p' "$f"; done; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import glob, importlib.util, json, pickle
import numpy as np
import _verified_upstream as up
p=pickle.load(open('public_batch.pkl','rb'))[0]
s=up.calibrate_upstream(dict(p['study'],reference_tracer_pulses=p['campaign_diagnostics']))
W=np.array([d['w_pulse'] for d in s['reference_tracer_pulses']]).T
Y=np.array([d['y_pulse_raw'] for d in s['reference_tracer_pulses']]).T
sensor=s['sensor_ij']; wall=np.any((sensor==0)|(sensor==np.array([s['nx']-1,s['ny']-1])),axis=1)
print('sensors wall/interior',wall.sum(),(~wall).sum())
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 spec=importlib.util.spec_from_file_location('old',f); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 S,U=m._design(s); R=(S@W-Y).reshape(len(s['mu_shifts']),len(sensor),-1)
 print(f.split('/')[-1], 'RMSE by shift x pulse',np.sqrt(np.mean(R**2,axis=1)).round(7),'wall/interior maxima',[np.max(np.abs(R[:,q])) for q in (wall,~wall)])
for entry in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(entry['facility_telemetry_trace']); print(entry['completed_campaign_short_name'],a.mean(axis=0).round(4))
print('study telemetry mean',s['aux_manifold_telemetry'].mean(axis=0).round(4))
PY
```
- **Container Exit Code**: `0` (`624 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/ h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    main = -np.r_[0., lower] - np.r_[upper, 0.]
    main[[0, -1]] += robin / h
    return diags((lower, main, upper), (-1, 0, 1), shape=(n, n), format='csc')


def _design(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), study['robin_y'])
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    rhs = np.stack([mode.ravel(order='F') for mode in study['source_modes']], axis=1)
    i, j = study['sensor_ij'].T
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        blocks.append(fields[i + nx * j])
    terminal = np.stack([column.reshape((nx, ny), order='F') for column in fields.T])
    return np.vstack(blocks), terminal


def _calibrate(study):
sensors wall/interior 9 23
beta_m99_c04_batch_01_solver.py RMSE by shift x pulse [[0.0542468 0.0116543 0.0216314]
 [0.0358081 0.0072668 0.0142256]
 [0.0236143 0.0045991 0.0096862]] wall/interior maxima [0.13566499395594345, 0.07990678639971682]
beta_m99_c04_batch_02_solver.py RMSE by shift x pulse [[0.0572575 0.0017077 0.0100197]
 [0.034599  0.0007598 0.0062062]
 [0.0224487 0.0006444 0.0048254]] wall/interior maxima [0.10371890938288464, 0.12084824863238258]
beta_m99_c04_s00_solver.py RMSE by shift x pulse [[0.1691801 0.169692  0.1927325]
 [0.076817  0.0645201 0.0772865]
 [0.0397942 0.0272829 0.0352521]] wall/interior maxima [0.251413170637991, 0.22285096211371616]
facility_init [-0.0041 -0.0056 -0.4122 -0.0006 -0.3311  0.0009  0.0087  0.0005]
beta_m99_c04_s00 [ 1.8000e-03  6.1000e-03 -3.9480e-01 -5.1000e-03  2.4492e+00  9.1000e-03
  6.0000e-03  6.2000e-03]
beta_m99_c04_batch_01 [ 8.0000e-04 -8.7000e-03  2.5577e+00  3.4000e-03 -3.6280e-01 -2.4000e-03
  1.6000e-03 -6.1000e-03]
beta_m99_c04_batch_02 [-3.000e-03 -2.900e-03 -3.974e-01 -6.600e-03 -3.238e-01  9.800e-03
 -2.000e-04 -6.000e-03]
beta_m99_c04_batch_03 [-1.3000e-03  1.8000e-03 -4.1520e-01 -1.2000e-03  2.4475e+00  4.2000e-03
 -9.9000e-03  6.3000e-03]
beta_m99_c04_batch_04 [ 8.0000e-04 -8.7000e-03  2.5577e+00  3.4000e-03 -3.6280e-01 -2.4000e-03
  1.6000e-03 -6.1000e-03]
study telemetry mean [ 0.1157 -0.3309  0.414   0.3163]
```

### Turn 5 — `execute_in_remote_docker` (`35440 ms` LLM)
- **Rationale**: Exact mixed-axis candidates are tested before fitting any continuous coefficients. If those candidates fail, held-pulse errors will show whether a richer boundary stencil generalizes beyond the pulses used to estimate it.
- **Alternatives Weighed**: `Choose the archived stencil with the lowest aggregate error` (rejected: All three have substantial error on at least one known pulse; aggregate fit would hide this failure.)
- **ATMS NOGOOD DECLARED**: Refuted `Applying any one of the three archived wall stencils unchanged to both axes explains this campaign's tracer pulses.` | Witness: `The worst shift-by-pulse RMSE is 0.19273 for Robin, 0.05425 for sealed, and 0.05726 for periodic; periodic fits pulse 2 much better than pulse 1 at the same shift (0.00171 versus 0.05726).` | Blanket Diagnosis: `Known source weights and calibrated observations expose a pulse-dependent mismatch in the shared chamber response; refitting unknown source weights could mask it.` | Sublated Pivot: `Preserve the prescribed interior upwind stencil, test independent wall laws by axis, and if necessary separate the inward, seam, and leakage terms with held-pulse validation.`

```bash
cd /holdout_task; cat > /tmp/wall_probe.py <<'PY'
import sys, itertools, pickle
import numpy as np
from scipy.sparse import csc_matrix, diags, kron, eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(p['study'],reference_tracer_pulses=p['campaign_diagnostics']))
W=np.array([d['w_pulse'] for d in s['reference_tracer_pulses']]).T
Y=np.array([d['y_pulse_raw'] for d in s['reference_tracer_pulses']]).T
nx,ny=s['nx'],s['ny']; ix,iy=s['sensor_ij'].T; inds=ix+nx*iy
rhs=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1)@W
axes=[]
for q in 'xy':
 k,v=s['kappa_'+q],s['vel_'+q]; n=len(k); h=s['L'+q]/(n-1); i=np.arange(n); left=(i-1)%n; right=(i+1)%n
 dn=(k+k[left])/(2*h*h)+np.maximum(v,0)/h; up=(k+k[right])/(2*h*h)+np.maximum(-v,0)/h
 a=np.zeros((n,n)); a[i,i]=dn+up; a[i,left]=-dn; a[i,right]=-up
 inward=np.array([k[0],k[-1]])/h**2; rob=s['robin_'+q]/h
 cases={'periodic':np.array([[up[0],dn[0],0],[dn[-1],up[-1],0]]),'sealed':np.column_stack([inward,[0,0],[0,0]]),'robin':np.column_stack([inward,[0,0],[rob,rob]])}
 axes.append((a,cases))
ident=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); shifts=s['mu_shifts']
def residual(param):
 mats=[]
 for (a,_),b in zip(axes,np.asarray(param).reshape(2,2,3)):
  a=a.copy(); a[[0,-1],:]=0; a[0,[1,-1,0]]=[-b[0,0],-b[0,1],sum(b[0])]; a[-1,[-2,0,-1]]=[-b[1,0],-b[1,1],sum(b[1])]; mats.append(csc_matrix(a))
 L=kron(eye(ny),mats[0],format='csc')+kron(mats[1],eye(nx),format='csc')+reaction
 return np.stack([splu(L+m*ident).solve(rhs)[inds] for m in shifts])-Y.reshape(len(shifts),len(inds),-1)
rank=[]
for a,b in itertools.product(*[cases for _,cases in axes]):
 par=np.stack([axes[0][1][a],axes[1][1][b]]); r=residual(par); rank.append((np.linalg.norm(r),a,b,par)); print(a,b,'rmse',np.sqrt(np.mean(r*r)), 'max',np.max(np.abs(r)))
best=min(rank,key=lambda v:v[0]); print('best discrete',best[1:]);
for _,cases in axes: print('axis cases',cases)
if best[0]>0.004:
 for hold in [-1,0,1,2]:
  train=np.arange(W.shape[1])!=hold
  fit=least_squares(lambda par:residual(par)[:,:,train].ravel(), best[3].ravel(),diff_step=1e-4,max_nfev=250,ftol=1e-12,gtol=1e-12)
  res=residual(fit.x); print('held pulse',hold,'fit',fit.x.reshape(2,2,3).round(6),'RMSE per shift and pulse',np.sqrt(np.mean(res**2,axis=1)).round(8),'max',np.max(np.abs(res)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `0` (`553 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
periodic periodic rmse 0.023910842542233357 max 0.1208482486323827
periodic sealed rmse 8.829585471045098e-07 max 1.8180442007964537e-06
periodic robin rmse 0.08985676967909627 max 0.23515276074043645
sealed periodic rmse 0.03883974146316868 max 0.19465894627322078
sealed sealed rmse 0.02529045792093289 max 0.13566499395594317
sealed robin rmse 0.08988295492369278 max 0.22252794262805559
robin periodic rmse 0.08646454877526336 max 0.22092831895837808
robin sealed rmse 0.08583959681413281 max 0.226819424378455
robin robin rmse 0.11263049360586647 max 0.25141317063799107
best discrete ('periodic', 'sealed', array([[[14.0284901 , 15.43281227,  0.        ],
        [14.0284901 , 14.65132181,  0.        ]],

       [[15.56539903,  0.        ,  0.        ],
        [ 8.30539903,  0.        ,  0.        ]]]))
axis cases {'periodic': array([[14.0284901 , 15.43281227,  0.        ],
       [14.0284901 , 14.65132181,  0.        ]]), 'sealed': array([[13.53697893,  0.        ,  0.        ],
       [13.53697893,  0.        ,  0.        ]]), 'robin': array([[13.53697893,  0.        ,  6.3927277 ],
       [13.53697893,  0.        ,  6.3927277 ]])}
axis cases {'periodic': array([[15.49187877, 11.93539903,  0.        ],
       [ 8.37891928, 13.1812852 ,  0.        ]]), 'sealed': array([[15.56539903,  0.        ,  0.        ],
       [ 8.30539903,  0.        ,  0.        ]]), 'robin': array([[15.56539903,  0.        ,  6.27415279],
       [ 8.30539903,  0.        ,  6.27415279]])}
```

### Turn 6 — `execute_in_remote_docker` (`49888 ms` LLM)
- **Rationale**: Implement the pulse-supported operator with the verified calibration and the specified ridge-adjusted objective. The public schema check detects integration and shape errors; independent forward-field and KKT checks are still needed before submission.
- **Alternatives Weighed**: `Fit continuous seam and leakage coefficients` (rejected: The exact mixed stencil fits all 288 calibrated pulse readings with RMSE 8.83e-7; no fitted wall coefficients are needed.)

```bash
set -e
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Transport and source inversion for the periodic-x, sealed-y chamber."""

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    a = (kappa + kappa[left]) / (2 * h**2) + np.maximum(velocity, 0) / h
    b = (kappa + kappa[right]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    if not periodic:
        a[0], b[0] = 0, kappa[0] / h**2
        a[-1], b[-1] = kappa[-1] / h**2, 0
    return coo_matrix((np.r_[a + b, -a, -b],
                       (np.tile(i, 3), np.r_[i, left, right])),
                      shape=(n, n)).tocsc()


def _design(s):
    nx, ny = s['nx'], s['ny']
    ax = _axis(s['kappa_x'], s['vel_x'], s['Lx'] / (nx - 1), periodic=True)
    ay = _axis(s['kappa_y'], s['vel_y'], s['Ly'] / (ny - 1), periodic=False)
    reaction = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    rhs = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
    ix, iy = s['sensor_ij'].T
    blocks = []
    for mu in s['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        blocks.append(fields[ix + nx * iy])
    terminal = fields.T.reshape((-1, ny, nx)).transpose(0, 2, 1)
    return np.vstack(blocks), terminal


def _box_simplex(H, c, cap, budget):
    """Minimize 0.5*w.T@H@w + c@w on the capped simplex."""
    n = len(c)
    w = np.full(n, budget / n)
    if budget == 0:
        return w, float(c.min())
    if budget == n * cap:
        return w, float((H @ w + c).max())
    # -1 is a lower bound, +1 an upper bound, 0 a free variable.
    status = np.zeros(n, dtype=int)
    tol = 128 * np.finfo(float).eps * max(1., np.max(np.abs(c)), cap * np.linalg.norm(H, np.inf))
    for _ in range(100 * n):
        g = H @ w + c
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            lower, upper = np.flatnonzero(status == -1), np.flatnonzero(status == 1)
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            status[[lo, hi]] = 0
            continue
        # Profile out the equality multiplier using the free Hessian.
        x, z = np.linalg.solve(H[np.ix_(free, free)],
                               np.column_stack((g[free], np.ones(free.size)))).T
        nu = (x.sum() + budget - w.sum()) / z.sum()
        step = nu * z - x
        alpha, blocker, side = 1., -1, 0
        for j, d in zip(free, step):
            if d < 0 and w[j] + d < 0:
                ratio, bound = -w[j] / d, -1
            elif d > 0 and w[j] + d > cap:
                ratio, bound = (cap - w[j]) / d, 1
            else:
                continue
            if ratio < alpha:
                alpha, blocker, side = max(0., ratio), j, bound
        w[free] += alpha * step
        if blocker >= 0:
            w[blocker] = 0. if side < 0 else cap
            status[blocker] = side
            continue
        g = H @ w + c
        violation = status * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol:
            return w, float(nu)
        status[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    S, terminal = _design(s)
    Z, y = s['Z_nuisance'], s['y_obs_raw']
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (Pz @ S)
    y = y - Z @ (Pz @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S.T @ y), cap, budget)
    residual = S @ w - y
    J = 0.5 * (residual @ residual + ridge * (w @ w))
    g = S.T @ residual + ridge * w
    d = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(d, 0.).tolist(),
            'lam_upper': np.maximum(-d, 0.).tolist(),
            'primal_objective': float(J),
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(d, 0.).sum()),
        },
    }
PY
cat > batch_solver.py <<'PY'
"""Compose the verified sensor calibration with the chamber inversion."""

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`433 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b8166709f2cc31a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b8166709 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`52999 ms` LLM)
- **Rationale**: Independently assemble the dense chamber operator to check the tracer response and terminal field, enumerate feasible active faces to check the public weights, and stress the optimizer at bound-heavy budgets. A forward discrepancy would require revisiting the stencil or indexing; an oracle or certificate discrepancy would require revisiting the optimizer. This probe leaves the solver, validator, and fixture unchanged.
- **Alternatives Weighed**: `Submit on the schema result alone` (rejected: Shape checks cannot catch a transposed field, wrong boundary stencil, or an invalid KKT certificate.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle, itertools
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _box_simplex
from _verified_upstream import calibrate_upstream
with open('public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

def dense_chamber(s):
    nx, ny = s['nx'], s['ny']
    L = np.diag(s['V_reaction_2d'].ravel(order='F'))
    for ix in range(nx):
        for iy in range(ny):
            row = ix + nx*iy
            for axis, i, n, stride in [('x', ix, nx, 1), ('y', iy, ny, nx)]:
                h = s['L'+axis]/(n-1)
                k, v = s['kappa_'+axis], s['vel_'+axis]
                if axis == 'y' and i in (0,n-1):
                    neighbors = [(1 if i == 0 else n-2, k[i]/h**2)]
                else:
                    left, right = (i-1)%n, (i+1)%n
                    neighbors = [(left, (k[i]+k[left])/(2*h**2)+max(v[i],0)/h),
                                 (right, (k[i]+k[right])/(2*h**2)+max(-v[i],0)/h)]
                for j, a in neighbors:
                    L[row,row] += a
                    L[row,row+(j-i)*stride] -= a
    rhs = np.stack([u.ravel(order='F') for u in s['source_modes']], axis=1)
    return L, rhs

def enumerate_faces(H, c, cap, budget):
    n = len(c)
    best = (np.inf, None)
    for status in itertools.product((-1,0,1), repeat=n):
        status = np.asarray(status)
        fixed, free = np.flatnonzero(status), np.flatnonzero(status==0)
        w = np.where(status==1,cap,0.)
        left = budget-w.sum()
        if left < -1e-12 or left > cap*len(free)+1e-12:
            continue
        if len(free):
            K = np.ones((len(free)+1,len(free)+1))
            K[:-1,:-1] = H[np.ix_(free,free)]
            K[-1,-1] = 0
            w[free] = np.linalg.solve(K, np.r_[-c[free]-H[np.ix_(free,fixed)]@w[fixed],left])[:-1]
        if w.min() < -1e-12 or w.max() > cap+1e-12:
            continue
        j = .5*w@H@w+c@w
        if j < best[0]: best = (j,w)
    return best[1]

for item in items:
    study = item['study']
    s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
    out = solve_batch(study)
    direct = solve_downstream(s)
    for key in ('weights','terminal_field'):
        np.testing.assert_array_equal(out[key], direct[key])
    L, rhs = dense_chamber(s)
    nx, ny = s['nx'], s['ny']
    ix, iy = s['sensor_ij'].T
    ident = np.eye(nx*ny)
    blocks = []
    for shift in s['mu_shifts']:
        U = np.linalg.solve(L+shift*ident,rhs)
        blocks.append(U[ix+nx*iy])
    S = np.vstack(blocks)
    pulses = s['reference_tracer_pulses']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    R = (S@W-Y).reshape(len(blocks),len(ix),len(pulses))
    wall = (ix==0)|(ix==nx-1)|(iy==0)|(iy==ny-1)
    worst = max(np.sqrt(np.mean(R[:,q,:]**2,axis=1)).max() for q in (wall,~wall) if q.any())
    assert np.abs(R).max() < 1e-5
    print('periodic-x/sealed-y pulse residual: worst shift/pulse/wall-class RMSE %.4g, max abs %.4g' % (worst,np.abs(R).max()))
    Z, y = s['Z_nuisance'], s['y_obs_raw']
    P = ident[:0,:0] if not len(Z) else np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    S, y = P@S, P@y
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    H, c = S.T@S+ridge*np.eye(S.shape[1]), -S.T@y
    w = out['weights']
    oracle_error = np.max(np.abs(w-enumerate_faces(H,c,cap,budget)))
    assert oracle_error < 1e-10
    field = out['terminal_field'].ravel(order='F')
    field_error = np.max(np.abs(field-U@w))
    pde_residual = np.max(np.abs((L+s['mu_shifts'][-1]*ident)@field-rhs@w))
    assert max(field_error,pde_residual) < 1e-10
    cert = out['kkt_certificate']
    np.testing.assert_array_equal(w,cert['weights'])
    lo, hi, nu = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper']), cert['nu_sum']
    g = H@w+c
    Rw = S@w-y
    J = .5*(Rw@Rw+ridge*(w@w))
    D = J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()
    kkt = max(abs(w.sum()-budget), max(0.,-w.min(),w.max()-cap,-lo.min(),-hi.min()), np.max(np.abs(g-nu-lo+hi)), np.max(np.abs(lo*w)), np.max(np.abs(hi*(cap-w))), abs(J-D), abs(J-cert['primal_objective']), abs(D-cert['dual_lower_bound']))
    assert kkt < 1e-10
    print('public weight oracle max error %.4g; field error %.4g; PDE residual %.4g; max KKT/certificate error %.4g' % (oracle_error,field_error,pde_residual,kkt))

rng = np.random.default_rng(125)
worst = 0.
for case in range(400):
    n = 8
    A = rng.normal(size=(10,n))
    H = A.T@A + .04*np.eye(n) if case%2 else A[:2].T@A + .04*np.eye(n)
    c = rng.normal(size=n)*(1 if case%3 else 25)
    cap = .42
    budget = (0.,cap,3*cap,1.4,n*cap-1e-8,n*cap)[case%6]
    w, nu = _box_simplex(H,c,cap,budget)
    d = H@w+c-nu
    err = max(abs(w.sum()-budget),-w.min(),w.max()-cap,np.max(np.maximum(d,0)*w),np.max(np.maximum(-d,0)*(cap-w)))
    assert err < 1e-10, (case,err)
    worst = max(worst,err)
print('400 box-simplex stress cases: worst feasibility/complementarity error %.4g' % worst)
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `1` (`631 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
periodic-x/sealed-y pulse residual: worst shift/pulse/wall-class RMSE 1.035e-06, max abs 1.818e-06
public weight oracle max error 4.913e-15; field error 1.943e-16; PDE residual 4.885e-15; max KKT/certificate error 6.047e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b8166709f2cc31a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b8166709 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 104, in <module>
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 10 is different from 2)
```

### Turn 8 — `execute_in_remote_docker` (`26203 ms` LLM)
- **Rationale**: The public weight oracle, terminal field, and certificate checks passed. Correct the test-only Gram matrix dimension error and finish the bound-heavy stress check. Select the wall law without each held pulse in turn; a selection change would call the wall identification into question. This step does not alter the deliverables, validator, or fixture.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from downstream_solver import _box_simplex
with contextlib.redirect_stdout(io.StringIO()):
    p = runpy.run_path('/tmp/wall_probe.py')
residual = { (a,b): p['residual'](params) for _,a,b,params in p['rank'] }
ix, iy = p['ix'], p['iy']
wall = (ix == 0) | (ix == p['nx']-1) | (iy == 0) | (iy == p['ny']-1)
worst_pulse = 0.
for held in range(p['W'].shape[1]):
    train = np.arange(p['W'].shape[1]) != held
    selected = min(residual, key=lambda k: np.linalg.norm(residual[k][:,:,train]))
    assert selected == ('periodic', 'sealed'), (held, selected)
    r = residual[selected][:,:,held]
    worst_pulse = max(worst_pulse, *(np.sqrt(np.mean(r[:,q]**2, axis=1)).max() for q in (wall,~wall)))
assert worst_pulse < 1e-5, worst_pulse
print(f'Leave-one-pulse-out selection: 3/3 periodic-x/sealed-y; worst held pulse/shift/wall-class RMSE {worst_pulse:.4g}')
rng = np.random.default_rng(125)
worst = 0.
for case in range(400):
    n, cap = 8, .42
    A = rng.normal(size=(10,n))
    B = A if case % 2 else A[:2]
    H = B.T @ B + .04*np.eye(n)
    c = rng.normal(size=n)*(1 if case%3 else 25)
    budget = (0.,cap,3*cap,1.4,n*cap-1e-8,n*cap)[case%6]
    w, nu = _box_simplex(H,c,cap,budget)
    d = H@w+c-nu
    lo, hi = np.maximum(d,0), np.maximum(-d,0)
    err = max(abs(w.sum()-budget),-w.min(),w.max()-cap,np.max(lo*w),np.max(hi*(cap-w)),abs(d@w-cap*np.minimum(d,0).sum()+nu*(w.sum()-budget)))
    assert err < 1e-10, (case,err)
    worst = max(worst,err)
print(f'400 box-simplex cases: worst feasibility, complementarity or dual-gap error {worst:.4g}')
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`703 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Leave-one-pulse-out selection: 3/3 periodic-x/sealed-y; worst held pulse/shift/wall-class RMSE 1.035e-06
400 box-simplex cases: worst feasibility, complementarity or dual-gap error 2.334e-14
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b8166709f2cc31a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b8166709 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`22504 ms` LLM)
- **Rationale**: The host rejected the previous probe as resolution evidence. This standalone read-only check measures the calibrated tracer residual from the shipped operator rather than a separate probe implementation. A mismatch would indicate that the deliverable does not implement the identified stencil.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _design
from batch_solver import solve_batch

item = pickle.loads(Path('/holdout_task/public_batch.pkl').read_bytes())[0]
study = item['study']
s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
pulses = s['reference_tracer_pulses']
weights = np.column_stack([p['w_pulse'] for p in pulses])
readings = np.column_stack([p['y_pulse_raw'] for p in pulses])
design, _ = _design(s)
ix, iy = s['sensor_ij'].T
wall = (ix == 0) | (ix == s['nx']-1) | (iy == 0) | (iy == s['ny']-1)
errors = (design @ weights - readings).reshape(len(s['mu_shifts']), len(ix), len(pulses))
worst = max(np.sqrt(np.mean(errors[:, group, :]**2, axis=1)).max() for group in (wall, ~wall) if group.any())
assert worst < 1e-5, worst
out = solve_batch(study)
cert = out['kkt_certificate']
gap = abs(cert['primal_objective'] - cert['dual_lower_bound'])
assert gap < 1e-10, gap
print(f'Active downstream operator: worst calibrated pulse/shift/wall-class RMSE {worst:.6g}; composed solver dual gap {gap:.6g}')
PY
```
- **Container Exit Code**: `0` (`386 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Active downstream operator: worst calibrated pulse/shift/wall-class RMSE 1.03482e-06; composed solver dual gap 3.26128e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b8166709f2cc31a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b8166709 (directory).
```

### Turn 10 — `submit_completed_task` (`12501 ms` LLM)
- **Rationale**: The unchanged bundle passed the public schema check, independent weight and terminal-field checks, and isolated execution. Leave-one-pulse-out validation selected the same wall law in all three folds. The host did not accept the separate resolution probe, so no formal nogood closure is claimed.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:267', '/holdout_task/downstream_solver.py': 'FILE:4485'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The periodic-x, sealed-y chamber model passed public schema validation. Worst held-pulse RMSE was 1.035e-6, and worst KKT error across 400 stress cases was 2.334e-14.`

### Turn 11 — `execute_in_remote_docker` (`28634 ms` LLM)
- **Rationale**: Remove the full nuisance subspace without ridge shrinkage, then select the boundary stencil with both a shift and a pulse held out. If another stencil wins a training fold, revisit whether the boundary is identifiable independently of nuisance. The audit writes only to /tmp and checks that the shipped operator matches the selected stencil.
- **Alternatives Weighed**: `Audit classification balance or relay modes` (rejected: The deliverable is a steady PDE inverse problem, with neither classification labels nor relay states. The relevant confounder is overlap between boundary responses and the measured nuisance basis.)

```bash
set -e
cat > /tmp/chamber_nuisance_audit.py <<'PY'
import contextlib
import io
import runpy
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
from downstream_solver import _design

with contextlib.redirect_stdout(io.StringIO()):
    p = runpy.run_path('/tmp/wall_probe.py')
s = p['s']
errors = {(a, b): p['residual'](params) for _, a, b, params in p['rank']}
selected = ('periodic', 'sealed')
Z = s['Z_nuisance'].reshape(*errors[selected].shape[:2], -1)
S, _ = _design(s)
shipped = (S @ p['W'] - p['Y']).reshape(errors[selected].shape)
np.testing.assert_allclose(shipped, errors[selected], rtol=0, atol=1e-12)

def nuisance_free(r, z):
    a = r.reshape(-1, r.shape[-1])
    x = z.reshape(len(a), -1)
    return a - x @ np.linalg.lstsq(x, a, rcond=None)[0]

def rmse(r):
    return np.sqrt(np.mean(r * r))

worst_held = 0.0
min_train_margin = np.inf
min_held_margin = np.inf
for shift in range(shipped.shape[0]):
    tr_s = np.arange(shipped.shape[0]) != shift
    for pulse in range(shipped.shape[2]):
        tr_p = np.arange(shipped.shape[2]) != pulse
        train = {k: rmse(nuisance_free(r[tr_s][:, :, tr_p], Z[tr_s])) for k, r in errors.items()}
        assert min(train, key=train.get) == selected, (shift, pulse, train)
        train_margin = min(v for k, v in train.items() if k != selected) - train[selected]
        held = {k: rmse(nuisance_free(r[shift:shift+1, :, pulse:pulse+1], Z[shift:shift+1])) for k, r in errors.items()}
        held_margin = min(v for k, v in held.items() if k != selected) - held[selected]
        min_train_margin = min(min_train_margin, train_margin)
        min_held_margin = min(min_held_margin, held_margin)
        worst_held = max(worst_held, held[selected])
assert worst_held < 1e-5, worst_held
print(f'Crossed shift/pulse holdout after unregularized nuisance removal: all {shipped.shape[0]*shipped.shape[2]} training folds select periodic-x/sealed-y; worst held RMSE {worst_held:.6g}; minimum RMSE advantage on training folds {min_train_margin:.6g}, on held slices {min_held_margin:.6g}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/chamber_nuisance_audit.py
```
- **Container Exit Code**: `0` (`601 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Crossed shift/pulse holdout after unregularized nuisance removal: all 9 training folds select periodic-x/sealed-y; worst held RMSE 9.46993e-07; minimum RMSE advantage on training folds 0.00392087, on held slices 0.000636625

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b8166709f2cc31a7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b8166709 (directory).
```

### Turn 12 — `submit_completed_task` (`13141 ms` LLM)
- **Rationale**: The unchanged solver bundle passed public schema validation, independent weight and terminal-field checks, and isolated execution. The crossed holdout audit showed that removing the nuisance subspace did not change the selected wall law. The host did not accept formal nogood closure, so none is claimed.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. Public schema validation passed. All nine nuisance-adjusted holdout folds selected periodic x and sealed y, with a worst held-out RMSE of 9.47e-7. Across 400 optimizer stress cases, the worst feasibility, complementarity, or dual-gap error was 2.33e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
