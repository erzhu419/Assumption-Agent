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
| **Turns Used / Max Turns** | `13 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `346,809` (`334,755` / `121,472` / `12,054`) |
| **Total Episode Wall Time (s)** | `380.29s` |

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

### Turn 1 — `execute_in_remote_docker` (`13759 ms` LLM)
- **Rationale**: Read the adapter and input schema first to detect coordinate changes and determine whether wall inference can use public pulses or must rely on historical telemetry.

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'));print('batch',len(x));
for i,a in enumerate(x):
 print('item',i,a.keys(), 'diag',type(a.get('campaign_diagnostics')),len(a.get('campaign_diagnostics') or []));
 for k,v in a['study'].items(): print(k, type(v).__name__, getattr(v,'shape',''),str(v)[:140] if i==0 else '')
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_batch_02_solver.py
-rw-r--r-- 1 root root 5.0K Nov 28 03:56 beta_m100_c03_s00_solver.py
-rw-r--r-- 1 root root  127 Nov 28 03:56 pre_llm_gate_diagnostic.json
batch 1
item 0 dict_keys(['study', 'campaign_diagnostics']) diag <class 'list'> 3
Lx float  2.4
Ly float  2.0
nx int  14
ny int  12
kappa_x ndarray (14,) [0.46193131 0.4954355  0.52699255 0.55476848 0.57714905 0.59283358
 0.60091055 0.60091055 0.59283358 0.57714905 0.55476848 0.52699255
 0.495
kappa_y ndarray (12,) [0.54145852 0.53659768 0.52240895 0.50004181 0.47130832 0.4385363
 0.40438074 0.37160872 0.34287523 0.3205081  0.30631937 0.30145852]
vel_x ndarray (14,) [ 0.35        0.34500413  0.33015913  0.30588881  0.27288602  0.23209293
  0.18467409  0.13198319  0.07552447  0.01690968 -0.04218784 -0.100
vel_y ndarray (12,) [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907 -0.27988581 -0.27272334 -0.25469696 -0.226
robin_x float  1.1134569414466113
robin_y float  1.1384086450100832
V_reaction_2d ndarray (14, 12) [[0.25       0.25       0.25       0.25       0.25       0.25
  0.25       0.25       0.25       0.25       0.25       0.25      ]
 [0.26030
mu_shifts ndarray (3,) [0.15 0.55 1.1 ]
source_modes ndarray (8, 14, 12) [[[3.19191823e-06 6.45795517e-06 8.26589839e-06 ... 1.25864951e-10
   4.13346238e-12 8.58766184e-14]
  [5.32422969e-05 1.07720919e-04 1.3787
sensor_ij ndarray (32, 2) [[10  0]
 [ 6 10]
 [ 2  6]
 [ 8  8]
 [ 8 10]
 [ 6  6]
 [10  2]
 [ 4  8]
 [10  8]
 [ 8  6]
 [ 6  2]
 [ 8  4]
 [10  6]
 [ 4 10]
 [ 2  0]
 [ 8 
row_bank_id ndarray (96,) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 
sensor_calibration_sweeps dict  {'ref_rosette': array([[-1.2       ,  0.        ,  0.        ,  0.        ],
       [-0.4       ,  0.        ,  0.        ,  0.        ],
  
Z_nuisance ndarray (96, 3) [[-1.41101778e+00  1.55395843e-01  1.14339335e+00]
 [ 8.12566418e-01 -8.51632914e-01 -6.98603511e-01]
 [-7.69654876e-01 -1.92209018e+00 -1.8
y_obs_raw ndarray (96,) [-0.30601932  0.79008456  0.5581804   0.8621182   0.06307881  0.76229237
  0.40027435  0.34276241  0.77184851  0.47589112  0.23007205  0.030
aux_manifold_telemetry ndarray (24, 4) [[-1.00704008 -0.88558602 -0.17353109  0.12635924]
 [ 0.48196698  0.7292369  -0.52992972 -0.96837616]
 [-0.74944621 -0.45999828 -1.09616273 
nuisance_ridge float  0.08
source_ridge float  0.04
u_box float  0.42
total_budget float  1.4
```

### Turn 2 — `execute_in_remote_docker` (`15818 ms` LLM)
- **Rationale**: Inspect the calibrated output contract and prior wall stencils before fitting pulse responses. This distinguishes a boundary-model mismatch from a coordinate mismatch and identifies reusable solver components.

```bash
cd /holdout_task && tail -25 _verified_upstream.py && cat _upstream_param_payload.json && cat /tmp/hegel_continual_archive/{archive_manifest,pre_llm_gate_diagnostic}.json && sed -n '1,180p' /tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py && python3 - <<'PY'
import json,pickle,numpy as np
p=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'));print('telemetry structure',type(p), list(p) if isinstance(p,dict) else len(p));
for k,v in (p.items() if isinstance(p,dict) else [('records',p)]):print(k,str(v)[-3800:])
i=pickle.load(open('public_batch.pkl','rb'))[0]
for d in i['campaign_diagnostics']:print({k: (np.shape(v),np.asarray(v).ravel()[:8]) for k,v in d.items()})
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'box simplex quadratic KKT ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`414 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.15523, -0.83377, 0.92215, 0.72712, 2.47344, -0.46373, -0.28058], [-0.27255, -0.19493, -0.38118, 0.85671, 0.86412, 2.59462, -0.19458, 0.11168], [-0.37794, -0.60909, 0.2749, 0.72467, 0.78234, 2.45092, -0.05704, 0.51017], [-0.29273, -0.78112, 0.77077, 0.57766, 0.43717, 2.56812, 0.22749, 0.77202], [-0.39459, -0.77189, 0.86704, 0.40733, -0.00449, 2.52413, 0.44825, 0.84341], [-0.34749, -0.54814, 0.38413, 0.17571, -0.40219, 2.50632, 0.56688, 0.64307], [-0.40533, -0.12376, -0.28714, -0.03565, -0.73655, 2.65986, 0.7148, 0.29447], [-0.39113, 0.26705, -0.7769, -0.2616, -0.86656, 2.59069, 0.85159, -0.11944], [-0.45852, 0.64144, -0.80305, -0.46301, -0.73504, 2.64222, 0.88373, -0.50447], [-0.37686, 0.84195, -0.37853, -0.58233, -0.41559, 2.57231, 0.82591, -0.78974], [-0.35316, 0.81678, 0.325, -0.78025, -0.01507, 2.63128, 0.66515, -0.84159], [-0.32431, 0.60404, 0.74386, -0.82756, 0.44732, 2.66788, 0.57155, -0.5948], [-0.30042, 0.21173, 0.7881, -0.84944, 0.72582, 2.47129, 0.4217, -0.30968], [-0.47997, -0.27165, 0.36365, -0.81459, 0.83142, 2.47261, 0.14357, 0.16559], [-0.44931, -0.59588, -0.31516, -0.75687, 0.75702, 2.48327, 0.02624, 0.55277], [-0.3698, -0.87464, -0.80882, -0.57871, 0.41804, 2.59698, -0.25093, 0.75152], [-0.38081, -0.82133, -0.8067, -0.40917, 0.01458, 2.5539, -0.38155, 0.80382], [-0.41069, -0.57647, -0.37878, -0.18214, -0.39895, 2.52846, -0.58101, 0.72412], [-0.17331, -0.11723, 0.33716, 0.01013, -0.73447, 2.51906, -0.72041, 0.28283]]}]
{'w_pulse': ((8,), array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ])), 'y_pulse_raw': ((96,), array([0.52439431, 0.05170525, 0.09941498, 0.37903726, 0.26353235,
       0.20690985, 0.21489916, 0.24799273]))}
{'w_pulse': ((8,), array([0. , 1. , 0. , 0. , 0.5, 0. , 0. , 0. ])), 'y_pulse_raw': ((96,), array([0.46752131, 0.10274411, 0.15138858, 0.41530858, 0.32703752,
       0.24932624, 0.20764227, 0.32393124]))}
{'w_pulse': ((8,), array([0. , 0. , 1. , 0. , 0. , 0.5, 0. , 0. ])), 'y_pulse_raw': ((96,), array([0.42985465, 0.36873829, 0.30942966, 0.63448685, 0.62309498,
       0.29451359, 0.19980152, 0.83455162]))}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`17275 ms` LLM)
- **Rationale**: Extract only the archived wall definitions and compact telemetry summaries to decide which boundary symmetries and mechanisms to compare against the calibrated pulses.
- **Alternatives Weighed**: `Reuse the last archived solver unchanged` (rejected: The gate reports a downstream failure, and the public pulses can independently test the archived wall assumptions.)

```bash
cd /holdout_task; for f in /tmp/hegel_continual_archive/*solver.py; do echo "$f"; sed -n '1,40p' "$f"; done; python3 - <<'PY'
import json,numpy as np,pickle
p=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print(type(p),list(p) if isinstance(p,dict) else len(p));
for k,v in (p.items() if isinstance(p,dict) else [('records',p)]):
 print(k,type(v),str(v)[:450]);
 if isinstance(v,list):
  for r in v:
   if isinstance(r,dict) and 'facility_telemetry_trace' in r:
    a=np.asarray(r['facility_telemetry_trace']);print({kk:vv for kk,vv in r.items() if kk!='facility_telemetry_trace'},np.round(a.mean(0),3),np.round(a.std(0),3))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -30
```
- **Container Exit Code**: `0` (`443 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_c03_s00'} [ 2.459 -0.008 -0.004  0.006  0.007 -0.413 -0.007  0.004] [0.048 0.601 0.604 0.595 0.603 0.066 0.608 0.593]
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c03_batch_01'} [-3.560e-01  1.100e-02 -2.000e-03  3.000e-03  2.000e-03  2.549e+00
 -6.000e-03  0.000e+00] [0.069 0.594 0.613 0.613 0.611 0.067 0.597 0.596]
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c03_batch_02'} [-0.341 -0.    -0.013  0.001 -0.006 -0.422 -0.002  0.003] [0.07  0.602 0.595 0.593 0.606 0.06  0.594 0.589]
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c03_batch_03'} [ 2.456e+00  3.000e-03 -2.000e-03  0.000e+00  0.000e+00 -3.810e-01
  9.000e-03 -8.000e-03] [0.069 0.595 0.608 0.595 0.586 0.078 0.612 0.593]
{'completed_campaign_index': 5, 'completed_campaign_short_name': 'beta_m100_c03_batch_04'} [-0.377 -0.006 -0.     0.005 -0.003 -0.414 -0.005  0.001] [0.058 0.601 0.603 0.607 0.597 0.06  0.603 0.618]
{'completed_campaign_index': 6, 'completed_campaign_short_name': 'beta_m100_c03_batch_05'} [ 2.447e+00 -3.000e-03  4.000e-03  2.000e-03  1.500e-02 -3.900e-01
 -2.000e-03  3.000e-03] [0.072 0.604 0.599 0.613 0.615 0.071 0.594 0.605]
{'completed_campaign_index': 7, 'completed_campaign_short_name': 'beta_m100_c03_batch_06'} [-3.550e-01  4.000e-03 -6.000e-03 -1.000e-02 -1.200e-02  2.558e+00
 -3.000e-03  2.000e-03] [0.064 0.6   0.593 0.613 0.596 0.062 0.609 0.592]
{'completed_campaign_index': 8, 'completed_campaign_short_name': 'beta_m100_c03_batch_07'} [-3.560e-01  1.100e-02 -2.000e-03  3.000e-03  2.000e-03  2.549e+00
 -6.000e-03  0.000e+00] [0.069 0.594 0.613 0.613 0.611 0.067 0.597 0.596]
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

### Turn 4 — `execute_in_remote_docker` (`37196 ms` LLM)
- **Rationale**: Compare exact open, reflecting, and periodic rows independently at each wall. If none predicts the pulses, held-out pulse errors from generalized rows will show whether the mismatch is reproducible and which stencil coefficients are missing.
- **Alternatives Weighed**: `Assume the historical wall mode persists` (rejected: Current calibrated pulses can test that assumption directly.)

```bash
cd /holdout_task && sed -n '1,19p' /tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py && cat public_asset_index.json && cat >/tmp/probe_walls.py <<'PY'
import sys,pickle,itertools
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
nx,ny=s['nx'],s['ny']; n=nx*ny
ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
z=s['Z_nuisance']; Q=np.linalg.qr(z)[0]; weights=np.array([p['w_pulse'] for p in s['reference_tracer_pulses']]).T
y=np.array([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]).T
rhs=s['source_modes'].transpose(0,2,1).reshape(-1,n).T @ weights
I=eye(n,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'))
def axis(d,p):
 k=s['kappa_'+d];v=s['vel_'+d];h=s['L'+d]/(len(k)-1)
 f=(k[1:]+k[:-1])/(2*h*h); left=f[:-1]+np.maximum(v[1:-1],0)/h; right=f[1:]+np.maximum(-v[1:-1],0)/h
 a=diags((-np.r_[left,p[3]],np.r_[sum(p[:3]),left+right,sum(p[3:])],-np.r_[p[0],right]),[-1,0,1],format='lil')
 a[0,-1]=-p[1];a[-1,0]=-p[4]
 return a.tocsc()
def states(d):
 k=s['kappa_'+d];v=s['vel_'+d];h=s['L'+d]/(len(k)-1);rob=s['robin_'+d]/h
 f=(k[1:]+k[:-1])/(2*h*h); seam=(k[0]+k[-1])/(2*h*h)
 return [(k[0]/h**2,0,rob),(k[-1]/h**2,0,rob)],[(k[0]/h**2,0,0),(k[-1]/h**2,0,0)],[(f[0]+max(-v[0],0)/h,seam+max(v[0],0)/h,0),(f[-1]+max(v[-1],0)/h,seam+max(-v[-1],0)/h,0)]
def pred(p):
 L=(kron(eye(ny),axis('x',p[:6]))+kron(axis('y',p[6:]),eye(nx))+reaction).tocsc()
 return np.concatenate([splu(L+mu*I).solve(rhs)[idx,:] for mu in s['mu_shifts']])
def residual(p,profile=False):
 r=pred(p)-y
 return r-Q@(Q.T@r) if profile else r
sx,sy=[states(d) for d in 'xy']; lookup=[[a[j] for a in t] for t in (sx,sy) for j in range(2)]
print('pulse rms',np.mean(y*y,axis=0)**.5,'nuisance correlations',np.linalg.norm(Q.T@y,axis=0)); print('states',lookup)
sc=[]
for keys in itertools.product(range(3),repeat=4):
 p=np.array([lookup[i][k] for i,k in enumerate(keys)]).ravel();r=residual(p)
 sc.append((np.linalg.norm(r),keys,p,np.sqrt(np.mean(r*r,axis=0))))
for norm,keys,p,r in sorted(sc,key=lambda x:x[0])[:12]: print('discrete',keys,norm,r)
for prof in (False,True):
 for hold in (None,0,1,2):
  cols=[j for j in range(3) if j!=hold];p0=sorted(sc,key=lambda x:x[0])[0][2]
  opt=least_squares(lambda p:residual(p,prof)[:,cols].ravel(),p0,max_nfev=200,gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('free',prof,hold,np.round(opt.x.reshape(-1,3),6).tolist(),'pulse rms',np.sqrt(np.mean(residual(opt.x,prof)**2,axis=0)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_walls.py
```
- **Container Exit Code**: `0` (`1722 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0061078, 0)]]
discrete (2, 2, 1, 1) 2.00554666991784e-05 [1.13123364e-06 1.11665276e-06 1.28965379e-06]
discrete (1, 1, 1, 1) 0.10850079748200082 [0.00344067 0.00098166 0.01047986]
discrete (2, 1, 1, 1) 0.18103859163752267 [0.00277568 0.00106167 0.01823662]
discrete (1, 2, 1, 1) 0.19763087214804947 [0.00495366 0.00078421 0.01953715]
discrete (1, 2, 2, 2) 0.6257266325160332 [0.04866769 0.03194343 0.0262593 ]
discrete (2, 2, 2, 2) 0.6601811217547859 [0.05100088 0.03241279 0.02980456]
discrete (1, 1, 2, 2) 0.6681798322998516 [0.05081971 0.03215965 0.03215251]
discrete (2, 1, 2, 2) 0.7261988358248699 [0.0528104  0.03229568 0.04076069]
discrete (1, 0, 1, 1) 0.8334656407279196 [0.03786982 0.04334408 0.06263594]
discrete (1, 2, 1, 0) 0.9246058829766409 [0.02901924 0.03492556 0.08272398]
discrete (1, 0, 2, 2) 0.9811639582774482 [0.05004608 0.04439299 0.07451575]
discrete (2, 2, 1, 0) 0.983879372923676 [0.02637792 0.03475275 0.09044323]
free False None [[14.046909, 15.452699, 6.2e-05], [14.044508, 14.668634, -4e-05], [16.379674, 1.4e-05, 7e-06], [9.11918, 8e-06, 5e-06]] pulse rms [6.56988045e-07 5.65118505e-07 7.68494619e-07]
free False 0 [[14.046738, 15.454067, 2.8e-05], [14.044578, 14.669439, -2.6e-05], [16.380025, 2e-05, 2.6e-05], [9.119254, 4e-06, 1e-05]] pulse rms [8.45647934e-07 5.57769829e-07 7.54681862e-07]
free False 1 [[14.046222, 15.452219, 4.2e-05], [14.044769, 14.668709, -4.3e-05], [16.380347, -1e-06, 2.4e-05], [9.119289, 5e-06, 1.2e-05]] pulse rms [6.44168397e-07 8.60259660e-07 7.53718991e-07]
free False 2 [[14.047535, 15.451295, 7.1e-05], [14.043729, 14.669373, -7.1e-05], [16.379404, -4.7e-05, 3.8e-05], [9.119463, 8e-06, 2.5e-05]] pulse rms [6.37047201e-07 5.51882462e-07 1.79563497e-06]
free True None [[14.046951, 15.452555, 6.2e-05], [14.044418, 14.668364, -4e-05], [16.379694, 1.3e-05, 6e-06], [9.119184, 8e-06, 5e-06]] pulse rms [6.38005253e-07 5.45763990e-07 7.47552961e-07]
free True 0 [[14.046917, 15.454101, 3.4e-05], [14.044403, 14.669052, -3.1e-05], [16.380051, 2.1e-05, 2.5e-05], [9.119255, 4e-06, 9e-06]] pulse rms [8.22440558e-07 5.38588015e-07 7.33389118e-07]
free True 1 [[14.046328, 15.452141, 4.6e-05], [14.044626, 14.668374, -4.7e-05], [16.380381, -2e-06, 2.3e-05], [9.119292, 5e-06, 1.2e-05]] pulse rms [6.24539090e-07 8.46301427e-07 7.32580319e-07]
free True 2 [[14.047512, 15.450499, 6.5e-05], [14.043701, 14.668645, -6.5e-05], [16.37941, -4.8e-05, 3.7e-05], [9.119444, 8e-06, 2.4e-05]] pulse rms [6.17759244e-07 5.31605538e-07 1.78998198e-06]
```

### Turn 5 — `execute_in_remote_docker` (`58825 ms` LLM)
- **Rationale**: Implement the uniquely supported discrete stencil using study geometry, preserve dynamic upstream calibration, and run the public schema check before independent KKT, terminal-field, and held-out tracer checks.
- **Alternatives Weighed**: `Fit twelve independent real-valued wall coefficients` (rejected: The exact mixed stencil already predicts at the calibration residual scale, and the free model has a worse maximum leave-one-pulse-out error of about 1.80e-6.)
- **ATMS NOGOOD DECLARED**: Refuted `The same boundary stencil applies to both axes.` | Witness: `Across 81 discrete wall assignments, periodic x and reflecting y uniquely matched the three calibrated pulses at RMSE near 1e-6; the runner-up had worst-pulse RMSE 0.01048.` | Blanket Diagnosis: `A single wall mode ties x seam transport to y boundary flux even though the tracer responses distinguish them.` | Sublated Pivot: `Use row-local periodic upwinding at the x seam and zero-leakage one-sided diffusion at the y walls.`

```bash
set -e
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Transport with a periodic x seam and reflecting y walls, in calibrated coordinates."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h, periodic):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    n = len(kappa)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    if periodic:
        seam = (kappa[0] + kappa[-1]) / (2 * h**2)
        left = np.r_[seam, face] + np.maximum(vel, 0) / h
        right = np.r_[face, seam] + np.maximum(-vel, 0) / h
        axis = diags((-left[1:], left + right, -right[:-1]), (-1, 0, 1), format='lil')
        axis[0, n-1] -= left[0]
        axis[n-1, 0] -= right[-1]
    else:
        left = np.r_[face[:-1] + np.maximum(vel[1:-1], 0) / h, kappa[-1] / h**2]
        right = np.r_[kappa[0] / h**2, face[1:] + np.maximum(-vel[1:-1], 0) / h]
        main = np.r_[right[0], left[:-1] + right[1:], left[-1]]
        axis = diags((-left, main, -right), (-1, 0, 1), format='csc')
    return axis.tocsc()


def _responses(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + d], study['vel_' + d], study['L' + d] / (n-1), d == 'x')
              for d, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    ident = eye(nx*ny, format='csc')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T.copy()
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx*sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(shifts) > 1 and len(indices) == np.size(study['y_obs_raw']):
        indices = indices.reshape(len(shifts), -1)
    sampled = []
    for k, mu in enumerate(shifts):
        fields = splu(operator + mu*ident).solve(rhs)
        sampled.append(fields[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(sampled), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, budget):
    """Primal active set for a strictly convex quadratic with a fixed sum."""
    n = len(c)
    if budget <= 0:
        return np.zeros(n), float(np.min(c))
    if budget >= n*cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w + c))
    w = np.full(n, budget/n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    xtol = 64*np.finfo(float).eps * cap * n
    gtol = 128*np.finfo(float).eps * max(1., np.linalg.norm(c, np.inf), cap*np.linalg.norm(H, np.inf))
    for _ in range(30*n + 100):
        free = np.flatnonzero(state == 0)
        g = H @ w + c
        if not len(free):
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[i] >= g[j] - gtol:
                return w, float((g[i] + g[j])/2)
            state[[i, j]] = 0
            continue
        fixed = np.flatnonzero(state != 0)
        target = budget - np.sum(w[fixed])
        a, b = np.linalg.solve(H[np.ix_(free, free)], np.column_stack((
            -c[free] - H[np.ix_(free, fixed)] @ w[fixed], np.ones(len(free))))).T
        nu = (target - np.sum(a))/np.sum(b)
        z = a + nu*b
        delta = z - w[free]
        alpha = np.ones(len(free))
        lo, hi = z < -xtol, z > cap + xtol
        alpha[lo] = -w[free[lo]]/delta[lo]
        alpha[hi] = (cap-w[free[hi]])/delta[hi]
        if np.any(lo | hi):
            hit = np.argmin(alpha)
            w[free] += min(1., max(0., alpha[hit]))*delta
            i = free[hit]
            state[i] = -1 if lo[hit] else 1
            w[i] = 0. if lo[hit] else cap
            continue
        w[free] = np.clip(z, 0., cap)
        g = H @ w + c
        nu = float(np.mean(g[free]))
        violation = state*(g-nu)
        i = np.argmax(violation)
        if violation[i] <= gtol:
            return w, nu
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    S, U_last = _responses(study)
    y = np.asarray(study['y_obs_raw']).ravel()
    Z = np.asarray(study['Z_nuisance'])
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
    data = np.column_stack((S, y))
    data -= Z @ np.linalg.solve(Z.T @ Z + study['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ data)
    design, obs = data[:, :-1], data[:, -1]
    H = design.T @ design + ridge*np.eye(design.shape[1])
    w, nu = _box_simplex(H, -design.T @ obs, cap, budget)
    residual = design @ w - obs
    g = design.T @ residual + ridge*w
    J = float((residual @ residual + ridge*(w @ w))/2)
    d = g-nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': J,
                                'dual_lower_bound': float(J - g @ w + nu*budget + cap*np.minimum(0, d).sum())}}
PY
cat > batch_solver.py <<'PY'
"""Compose the verified sensor transducer with the chamber inverse solver."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`424 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`41434 ms` LLM)
- **Rationale**: An independent dense operator tests the final sparse stencil, sensor ordering, and field reconstruction. Recomputing the KKT certificate tests both objective values, and known optima stress changes in the active box constraints.

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import sys, pickle, itertools
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _box_simplex
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    items = pickle.load(f)
for item in items:
    study = item['study']
    cal = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
    nx,ny = study['nx'], study['ny']
    A = np.diag(study['V_reaction_2d'].ravel(order='F'))
    for iy,ix in itertools.product(range(ny),range(nx)):
        row = ix+nx*iy
        for d,i,n,stride in [('x',ix,nx,1),('y',iy,ny,nx)]:
            k,v,h = study['kappa_'+d],study['vel_'+d][i],study['L'+d]/(n-1)
            if d=='y' and i in (0,n-1):
                neigh = [(1 if i==0 else n-2,k[i]/h**2)]
            else:
                l,r=(i-1)%n,(i+1)%n
                neigh=[(l,(k[l]+k[i])/(2*h*h)+max(v,0)/h),(r,(k[r]+k[i])/(2*h*h)+max(-v,0)/h)]
            for j,a in neigh:
                A[row,row] += a
                A[row,row+(j-i)*stride] -= a
    rhs = np.column_stack([m.ravel(order='F') for m in study['source_modes']])
    sij = study['sensor_ij']
    idx = sij[:,0]+nx*sij[:,1]
    S = []
    for mu in study['mu_shifts']:
        shifted = A + mu*np.eye(nx*ny)
        U = np.linalg.solve(shifted,rhs)
        assert np.max(np.abs(shifted@U-rhs)) < 1e-11
        S.append(U[idx])
    S = np.vstack(S)
    err = np.array([(S@p['w_pulse']-p['y_pulse_raw']).reshape(len(study['mu_shifts']),-1) for p in cal['reference_tracer_pulses']])
    pulse = np.sqrt(np.mean(err**2,axis=(1,2)))
    per_shift = np.sqrt(np.mean(err**2,axis=2))
    assert per_shift.max() < 5e-6
    out = solve_batch(study)
    direct = solve_downstream(cal)
    w = np.asarray(out['weights']); cert = out['kkt_certificate']
    assert np.array_equal(w,direct['weights']) and np.array_equal(w,cert['weights'])
    cap,budget,ridge = [study[k] for k in ('u_box','total_budget','source_ridge')]
    assert w.min() >= 0 and w.max() <= cap and abs(w.sum()-budget) < 1e-12
    expected = (U @ w).reshape((nx,ny),order='F')
    terminal_error = np.max(np.abs(expected-out['terminal_field']))
    assert out['terminal_field'].shape == (nx,ny) and terminal_error < 1e-11
    Z = study['Z_nuisance']
    P = np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+study['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    R = P@(S@w-cal['y_obs_raw'])
    g = S.T@P.T@R+ridge*w
    J = (R@R+ridge*(w@w))/2
    lo,hi,nu = np.asarray(cert['lam_lower']),np.asarray(cert['lam_upper']),cert['nu_sum']
    dual = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    stationarity = np.max(np.abs(g-nu-lo+hi))
    assert stationarity < 1e-11 and min(lo.min(),hi.min()) >= 0
    assert max(np.max(w*lo),np.max((cap-w)*hi)) < 1e-11
    assert abs(J-cert['primal_objective']) < 1e-11 and abs(dual-cert['dual_lower_bound']) < 1e-11 and abs(J-dual) < 1e-11
    print('pulse RMSE',pulse,'worst pulse/shift RMSE',per_shift.max(),'terminal max abs error',terminal_error,'stationarity',stationarity,'duality gap',J-dual)

rng = np.random.default_rng(8)
worst = 0.
for n in (2,8,24):
    for j in range(90):
        B = rng.normal(size=(n,n)); H = B.T@B+0.04*np.eye(n)
        cap = 0.42
        target = rng.uniform(0,cap,n)
        lower = rng.random(n)<.35
        upper = (~lower)&(rng.random(n)<.5)
        target[lower] = 0.; target[upper] = cap
        g = np.zeros(n); g[lower] = rng.uniform(.1,2,lower.sum()); g[upper] = -rng.uniform(.1,2,upper.sum())
        w,nu = _box_simplex(H,g-H@target,cap,float(target.sum()))
        worst = max(worst,np.max(np.abs(w-target)))
assert worst < 1e-10
print('270 constructed KKT optima, worst weights max abs error',worst)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`806 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pulse RMSE [1.13123364e-06 1.11665276e-06 1.28965379e-06] worst pulse/shift RMSE 1.5276571913269954e-06 terminal max abs error 1.3877787807814457e-16 stationarity 3.1710745140856034e-15 duality gap 3.0982161280945775e-15
270 constructed KKT optima, worst weights max abs error 6.106226635438361e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27673 ms` LLM)
- **Rationale**: The schema, field, and KKT checks passed, but the host did not accept the previous structural probe. This separate read-only test checks whether wall selection depends on any one pulse. If the selected stencil changes when a pulse is withheld, reconsider the boundary inference rather than treating the pooled fit as decisive.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import itertools
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/holdout_task')
from downstream_solver import _responses
# Reuse only the setup and stencil definitions from the earlier read-only probe.
p = {}
exec(Path('/tmp/probe_walls.py').read_text().split('sx,sy=')[0], p)
s, y, w = p['s'], p['y'], p['weights']
sx, sy = [p['states'](d) for d in 'xy']
rows = [[state[j] for state in axis] for axis in (sx, sy) for j in range(2)]
choices = list(itertools.product(range(3), repeat=4))
pred = np.stack([p['pred'](np.array([rows[i][k] for i, k in enumerate(key)]).ravel()) for key in choices])
errors = np.sqrt(np.mean((pred - y)**2, axis=1))
S, _ = _responses(s)
worst = 0.
for held_out in range(y.shape[1]):
    train = np.arange(y.shape[1]) != held_out
    scores = np.mean(errors[:, train]**2, axis=1)
    best, runner_up = np.argsort(scores)[:2]
    key = choices[best]
    hold_rmse = float(errors[best, held_out])
    deployed_rmse = float(np.sqrt(np.mean((S @ w[:, held_out] - y[:, held_out])**2)))
    assert key == (2, 2, 1, 1) and hold_rmse < 5e-6
    assert abs(deployed_rmse - hold_rmse) < 1e-12
    worst = max(worst, deployed_rmse)
    print('held-out pulse', held_out, 'training-selected walls', key, 'training runner-up RMSE', np.sqrt(scores[runner_up]), 'deployed held-out RMSE', deployed_rmse)
print('Worst out-of-fold tracer RMSE:', worst)
PY
```
- **Container Exit Code**: `0` (`666 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out pulse 0 training-selected walls (2, 2, 1, 1) training runner-up RMSE 0.00744282262330691 deployed held-out RMSE 1.1312336407293203e-06
held-out pulse 1 training-selected walls (2, 2, 1, 1) training runner-up RMSE 0.007799543234206236 deployed held-out RMSE 1.116652763446899e-06
held-out pulse 2 training-selected walls (2, 2, 1, 1) training runner-up RMSE 0.002101370972055705 deployed held-out RMSE 1.2896537887251233e-06
Worst out-of-fold tracer RMSE: 1.2896537887251233e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 8 — `submit_completed_task` (`13928 ms` LLM)
- **Rationale**: The unchanged artifact passed the public schema check and isolated-process tests. Independent dense reconstruction matched the terminal field within 1.39e-16, and 270 known-optimum cases tested the box-simplex solver with worst weight error 6.11e-16.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:270', '/holdout_task/downstream_solver.py': 'FILE:5202'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema check passed. Worst held-out tracer RMSE was 1.29e-6, and the public KKT duality gap was 3.10e-15.`

### Turn 9 — `execute_in_remote_docker` (`42538 ms` LLM)
- **Rationale**: Test additional wall conventions with each pulse and shift jointly withheld, then repeat selection after within-shift nuisance projection. A changed winner would reopen the stencil choice. Out-of-fold sensor inversion quantifies calibration error independently; the audit writes nothing to the fixtures or deliverables.
- **Alternatives Weighed**: `Run classification or relay-mode checks` (rejected: This is a steady continuous inverse problem with no class labels, relay states, or time evolution. The relevant confounding test is whether nuisance directions change wall selection within shifts.)

```bash
set -e
cat > /tmp/holdout_margin_audit.py <<'PY'
import itertools
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import _fit, _invert
from downstream_solver import _responses
p = {}
exec(Path('/tmp/probe_walls.py').read_text().split('sx,sy=')[0], p)
s, y, w = p['s'], p['y'], p['weights']
raw_study = p['item']['study']
ref = np.asarray(raw_study['sensor_calibration_sweeps']['ref_rosette'])
raw = np.asarray(raw_study['sensor_calibration_sweeps']['raw_rosette'])
cal_err = []
for fold in range(5):
    hold = np.arange(len(raw)) % 5 == fold
    cal_err.append(_invert(raw[hold], _fit(ref[~hold], raw[~hold])) - ref[hold])
cal_err = np.concatenate(cal_err)
print('out-of-fold calibration inverse RMSE per channel:', np.sqrt(np.mean(cal_err**2, axis=0)))
sx, sy = [p['states'](d) for d in 'xy']
rows = [[state[j] for state in axis] for axis in (sx, sy) for j in range(2)]
keys = list(itertools.product(range(3), repeat=4))
coefs = [np.array([rows[i][k] for i, k in enumerate(key)]) for key in keys]
base = coefs[keys.index((2,2,1,1))]
for wall in range(4):
    d, edge = 'xy'[wall//2], wall%2
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k); h = s['L'+d]/(n-1); i, j = (0,1) if edge == 0 else (n-1,n-2)
    face = (k[i]+k[j])/(2*h*h)
    rob = s['robin_'+d]/h
    inward = max(0., (-1 if edge == 0 else 1)*v[i])/h
    outward = max(0., (1 if edge == 0 else -1)*v[i])/h
    seam = (k[0]+k[-1])/(2*h*h)
    base_in, base_seam, base_leak = base[wall]
    for label, row in [('face', (face,base_seam,base_leak)),('double', (2*base_in,base_seam,base_leak)),('leaky', (base_in,base_seam,rob)),('inward_upwind',(face+inward,base_seam,base_leak)),('reversed_upwind',(face+outward,seam+inward if d=='x' else 0,base_leak)),('diffusion_seam',(base_in,seam,base_leak))]:
        a = base.copy(); a[wall] = row
        if any(np.allclose(a, b, rtol=0, atol=1e-12) for b in coefs):
            continue
        keys.append((wall,label)); coefs.append(a)
errors = np.stack([p['pred'](a.ravel())-y for a in coefs])
shift_count = len(s['mu_shifts'])
Z = s['Z_nuisance']
S, _ = _responses(s)
assert np.max(np.abs((S@w-y)-errors[keys.index((2,2,1,1))])) < 1e-12
for profile in (False, True):
    E = errors.reshape(len(keys), shift_count, -1, y.shape[1]).copy()
    if profile:
        for j, z in enumerate(np.array_split(Z, shift_count)):
            q = np.linalg.qr(z, mode='reduced')[0]
            E[:,j] -= q @ (q.T @ E[:,j])
    floors, margins = [], []
    for sh, pul in itertools.product(range(shift_count),range(y.shape[1])):
        shifts = np.arange(shift_count) != sh
        pulses = np.arange(y.shape[1]) != pul
        score = np.mean(E[:,shifts][:,:,:,pulses]**2, axis=(1,2,3))
        i,j = np.argsort(score)[:2]
        floors.append(float(np.sqrt(np.mean(E[i,sh,:,pul]**2))))
        margins.append(float(np.sqrt(score[j])-np.sqrt(score[i])))
        assert keys[i] == (2,2,1,1), (profile,sh,pul,keys[i])
    assert max(floors) < 5e-6
    print('within-shift nuisance profiling:', profile, 'wall candidates:', len(keys), 'worst joint held-out RMSE:', max(floors), 'minimum training RMSE margin:', min(margins))
PY
OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/holdout_margin_audit.py
```
- **Container Exit Code**: `0` (`768 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
out-of-fold calibration inverse RMSE per channel: [0.7525609  0.71222584 0.83899115 0.66901321]
within-shift nuisance profiling: False wall candidates: 96 worst joint held-out RMSE: 1.5276571917474672e-06 minimum training RMSE margin: 3.0455333338054146e-05
within-shift nuisance profiling: True wall candidates: 96 worst joint held-out RMSE: 1.4889849005208312e-06 minimum training RMSE margin: 2.899504757314738e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`20786 ms` LLM)
- **Rationale**: Distinguish a fold-design artifact from unstable sensor inversion by comparing interleaved and shuffled splits with the full-sweep fit. If instability persists under shuffled splits, quantify whether it affects tracer- and study-range readings before drawing conclusions about the composed solver. This check does not modify fixtures or deliverables.

```bash
cd /holdout_task && cat _verified_upstream.py && OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import _fit,_invert
with open('/holdout_task/public_batch.pkl','rb') as f:
    item=pickle.load(f)[0]
s=item['study']['sensor_calibration_sweeps']; x=np.asarray(s['ref_rosette']); y=np.asarray(s['raw_rosette'])
unique,counts=np.unique(x,axis=0,return_counts=True)
print('sweep shape and unique patterns',x.shape,len(unique),'multiplicity range',counts.min(),counts.max()); print('first patterns',x[:12]); print('reference ranges',x.min(0),x.max(0))
full=_fit(x,y); print('full inverse RMSE',np.sqrt(np.mean((_invert(y,full)-x)**2,axis=0)))
rng=np.random.default_rng(8)
for label,folds in [('interleaved',np.arange(len(x))%5),('shuffled',rng.permutation(len(x))%5)]:
    for k in range(5):
        h=folds==k; c=_fit(x[~h],y[~h]); err=_invert(y[h],c)-x[h]
        print(label,k,'RMSE',np.sqrt(np.mean(err**2)),'coefficient change',np.max(np.abs(np.asarray(c)-np.asarray(full))))
PY
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
: dict) -> dict:
    """Invert observations and tracer pulses without modifying the input study."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    result = dict(study)
    result['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    bank = sweeps['single_bank_sweeps']
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
sweep shape and unique patterns (80, 4) 80 multiplicity range 1 1
first patterns [[-1.2  0.   0.   0. ]
 [-0.4  0.   0.   0. ]
 [ 0.5  0.   0.   0. ]
 [ 1.3  0.   0.   0. ]
 [ 0.  -1.2  0.   0. ]
 [ 0.  -0.4  0.   0. ]
 [ 0.   0.5  0.   0. ]
 [ 0.   1.3  0.   0. ]
 [ 0.   0.  -1.2  0. ]
 [ 0.   0.  -0.4  0. ]
 [ 0.   0.   0.5  0. ]
 [ 0.   0.   1.3  0. ]]
reference ranges [-1.20037463 -1.32442596 -1.36443396 -1.37109446] [1.79911098 1.7846524  1.66411788 1.73312487]
full inverse RMSE [0.752561   0.71222625 0.83899108 0.66901377]
interleaved 0 RMSE 0.9993901519549782 coefficient change 3.852362723117282e-06
interleaved 1 RMSE 9.801367331796021e-06 coefficient change 4.406136261181359e-06
interleaved 2 RMSE 9.603525466269151e-06 coefficient change 3.860896556911466e-06
interleaved 3 RMSE 0.9245615681552489 coefficient change 3.716267971428742e-06
interleaved 4 RMSE 0.9632124414510931 coefficient change 4.386031080994002e-06
shuffled 0 RMSE 0.9993937713096357 coefficient change 6.536531945044999e-06
shuffled 1 RMSE 0.9245609757635734 coefficient change 5.25119927935469e-06
shuffled 2 RMSE 0.9632106445106312 coefficient change 5.752157023766746e-06
shuffled 3 RMSE 9.7374011203333e-06 coefficient change 5.5429255851335846e-06
shuffled 4 RMSE 9.121064270976194e-06 coefficient change 4.400300098406085e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`24713 ms` LLM)
- **Rationale**: The full-sweep and cross-fold fits agree, but a few rows may dominate the inverse error. Compare forward and inverse residuals on the same held-out sweep rows. Large inverse errors on clean forward rows would indicate a branch failure worth reporting; large forward errors on those same rows would instead implicate the measurements. Raw tracer residuals test whether the issue reaches the downstream operating range.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import _features, _fit, _invert
from downstream_solver import _responses
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
study = item['study']
sweeps = study['sensor_calibration_sweeps']
x = np.asarray(sweeps['ref_rosette'])
y = np.asarray(sweeps['raw_rosette'])
forward = np.empty(len(x))
inverse = np.empty(len(x))
for fold in range(5):
    hold = np.arange(len(x)) % 5 == fold
    coef = _fit(x[~hold], y[~hold])
    forward[hold] = np.linalg.norm(_features(x[hold]) @ coef - y[hold], axis=1)
    inverse[hold] = np.linalg.norm(_invert(y[hold], coef)-x[hold], axis=1)
inlier = forward < 10*np.median(forward)
print('Calibration rows:', len(x), 'forward outliers:', np.flatnonzero(~inlier).tolist())
print('Worst inverse row error on forward inliers:', float(inverse[inlier].max()), 'minimum forward error among flagged rows:', float(forward[~inlier].min()) if not inlier.all() else None)
print('Rows with clean forward residual but large inverse error:', np.flatnonzero(inlier & (inverse > 1e-3)).tolist())
S, _ = _responses(study)
coef = _fit(x, y)
for j, pulse in enumerate(item['campaign_diagnostics']):
    ref = (S @ pulse['w_pulse']).reshape(-1,4)
    raw = np.asarray(pulse['y_pulse_raw']).reshape(-1,4)
    print('Pulse', j, 'forward RMSE:', float(np.sqrt(np.mean((_features(ref)@coef-raw)**2))), 'inverse RMSE:', float(np.sqrt(np.mean((_invert(raw,coef)-ref)**2))))
PY
```
- **Container Exit Code**: `0` (`453 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Calibration rows: 80 forward outliers: [13, 14, 15]
Worst inverse row error on forward inliers: 3.7340150326362074e-05 minimum forward error among flagged rows: 25.963227276916562
Rows with clean forward residual but large inverse error: []
Pulse 0 forward RMSE: 1.2616325448049983e-06 inverse RMSE: 1.1312336407293203e-06
Pulse 1 forward RMSE: 1.2755530855581635e-06 inverse RMSE: 1.116652763446899e-06
Pulse 2 forward RMSE: 1.4701943338037341e-06 inverse RMSE: 1.2896537887251233e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`19534 ms` LLM)
- **Rationale**: The last receipt concerned calibration outliers, not the unresolved wall choice. Run the existing read-only wall audit directly against the unchanged artifact so the host can assess the relevant evidence without any fixture or solver writes. Reopen the stencil choice only if a held-out fold selects a different wall model.
- **Alternatives Weighed**: `Change the verified upstream calibration` (rejected: The large sweep errors were confined to three contaminated readings. Clean rows showed no inverse branch failure, and tracer inversion remained accurate to 1.29e-6 RMSE.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/holdout_margin_audit.py
```
- **Container Exit Code**: `0` (`795 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
out-of-fold calibration inverse RMSE per channel: [0.7525609  0.71222584 0.83899115 0.66901321]
within-shift nuisance profiling: False wall candidates: 96 worst joint held-out RMSE: 1.5276571917474672e-06 minimum training RMSE margin: 3.0455333338054146e-05
within-shift nuisance profiling: True wall candidates: 96 worst joint held-out RMSE: 1.4889849005208312e-06 minimum training RMSE margin: 2.899504757314738e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=be3fb329ad9f8a51 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_be3fb329 (directory).
```

### Turn 13 — `submit_completed_task` (`14799 ms` LLM)
- **Rationale**: Turn 12 tested 96 boundary candidates on real calibrated tracers without modifying fixtures or deliverables. The deployed stencil won every fold with and without nuisance projection. The same verified artifact passed isolated-process schema, independent terminal-field and KKT checks, and 270 known-optimum solver tests.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema check passed. Every held-out tracer/shift split selected periodic x and reflecting y, with a worst RMSE of 1.53e-6. The public KKT duality gap was 3.10e-15.`

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
