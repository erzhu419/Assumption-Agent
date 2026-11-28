# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c01_batch_02__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `10 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `252,594` (`243,803` / `65,408` / `8,791`) |
| **Total Episode Wall Time (s)** | `285.59s` |

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

### Turn 1 — `execute_in_remote_docker` (`17075 ms` LLM)
- **Rationale**: Inspect the schema, sweep layout, and campaign telemetry before implementing the solver; these determine observation ordering, calibration structure, and whether to replace the baseline boundary rows.

```bash
ls -la /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; python3 - <<'PY'
import json, pathlib, pickle, numpy as np
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
    d=json.loads(p.read_text()); print('\n',p, str(d)[:12500])
items=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(items))
for item in items[:2]:
 print('item keys',list(item)); print('diagnostics',type(item.get('campaign_diagnostics')).__name__)
 for k,v in item['study'].items():
    if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:6])
    elif isinstance(v,dict): print(k,[(a,type(b).__name__,getattr(b,'shape',None),len(b) if isinstance(b,(list,dict)) else None) for a,b in v.items()])
    else: print(k,type(v).__name__,str(v)[:250])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1 | head -60
```
- **Container Exit Code**: `0` (`506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
48, -0.29259, -0.12471, -0.60803, 2.62938, 0.77557], [0.27023, -0.11504, -0.31774, -0.35479, -0.58943, -0.86178, 2.49052, 0.79516], [0.03164, -0.53269, -0.77815, -0.44491, -0.77882, -0.59645, 2.4339, 0.52098], [-0.15558, -0.78663, -0.75361, -0.41538, -0.82665, 0.02863, 2.57929, 0.22295], [-0.34072, -0.86643, -0.354, -0.35944, -0.69775, 0.61617, 2.48836, -0.27897], [-0.54579, -0.66677, 0.30952, -0.40346, -0.29005, 0.84692, 2.58323, -0.66442], [-0.678, -0.28314, 0.75685, -0.30693, 0.13271, 0.61819, 2.63682, -0.78513], [-0.85115, 0.14479, 0.73909, -0.36524, 0.49024, 0.00451, 2.58329, -0.87013], [-0.8407, 0.49247, 0.34203, -0.32326, 0.81966, -0.55594, 2.60438, -0.5887]]}]
items 1
item keys ['study', 'campaign_diagnostics']
diagnostics list
Lx float 2.4
Ly float 2.0
nx int 14
ny int 12
kappa_x (14,) float64 [0.47478108 0.50828527 0.53984232 0.56761825 0.58999882 0.60568335]
kappa_y (12,) float64 [0.50168525 0.4968244  0.48263567 0.46026854 0.43153505 0.39876303]
vel_x (14,) float64 [0.35       0.34500413 0.33015913 0.30588881 0.27288602 0.23209293]
vel_y (12,) float64 [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099]
robin_x float 1.0123957203938798
robin_y float 1.2636561147288357
V_reaction_2d (14, 12) float64 [0.25 0.25 0.25 0.25 0.25 0.25]
mu_shifts (3,) float64 [0.15 0.55 1.1 ]
source_modes (8, 14, 12) float64 [2.19084741e-10 1.83548261e-09 9.72837233e-09 3.26198427e-08
 6.91950948e-08 9.28582559e-08]
sensor_ij (32, 2) int64 [0 0 0 4 4 4]
row_bank_id (96,) int64 [0 1 2 3 0 1]
sensor_calibration_sweeps [('ref_rosette', 'ndarray', (80, 4), None), ('raw_rosette', 'ndarray', (80, 4), None), ('single_bank_sweeps', 'dict', None, 3)]
Z_nuisance (96, 3) float64 [ 1.69388346 -0.00783761  0.03092432  1.39653771 -0.53257087 -0.05347803]
y_obs_raw (96,) float64 [0.27280333 0.15617897 0.44761485 0.13148567 0.20153688 0.17478199]
aux_manifold_telemetry (24, 4) float64 [-2.29034011  1.71427302 -0.10580783  0.35626673  0.75417843  0.02862835]
nuisance_ridge float 0.08
source_ridge float 0.04
u_box float 0.42
total_budget float 1.4
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`17380 ms` LLM)
- **Rationale**: The public item has three tracer pulses. Read the prior solver and compact telemetry to identify candidate wall and calibration models before testing them against the pulses.

```bash
cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /tmp/hegel_continual_archive/*batch_01_solver.py; python3 - <<'PY'
import json,pickle
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
def show(o,d=0):
 if d>4: print(type(o).__name__,len(o) if hasattr(o,'__len__') else o);return
 if isinstance(o,dict):
  for k,v in o.items(): print(' '*d,k,end=': ');show(v,d+1)
 elif isinstance(o,list) and len(o)>6: print('list',len(o),'first',str(o[0])[:180],'last',str(o[-1])[:180])
 elif isinstance(o,list):
  print('list',len(o));
  for v in o: show(v,d+1)
 else: print(o)
show(json.loads(p.read_text())); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; show(i['study']['sensor_calibration_sweeps']['single_bank_sweeps']); print('pulses', [p['w_pulse'] for p in i['campaign_diagnostics']]);print('sensor indices',i['study']['sensor_ij'].tolist())
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`449 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4545455 -0.79090909 -0.53636364 -0.28181818 -0.02727273
  0.22727273  0.48181818  0.73636364  0.99090909  1.24545455  1.5
 -1.3        -1.04545455 -0.79090909 -0.53636364 -0.28181818 -0.02727273
  0.22727273  0.48181818  0.73636364  0.99090909  1.24545455  1.5
 -1.3        -1.04545455 -0.79090909 -0.53636364 -0.28181818 -0.02727273
  0.22727273  0.48181818  0.73636364  0.99090909  1.24545455  1.5
 -1.3        -1.04545455 -0.79090909 -0.53636364 -0.28181818 -0.02727273
  0.22727273  0.48181818  0.73636364  0.99090909  1.24545455  1.5       ]
 raw_readout: [-1.58099879 -1.25772883 -0.93445627 -0.61118883 -0.28791425  0.03537009
  0.35864719  0.68190715  1.00516391  1.32845256  1.65172764  1.97498667
 -1.11300862 -0.9067914  -0.70063188 -0.49447696 -0.28827099 -0.08207942
  0.12408186  0.33026257  0.53645489  0.74263058  0.94880757  1.15501069
 -1.55800973 -1.23728521 -0.91655597 -0.59581989 -0.27509625  0.04563538
  0.36636044  0.68708718  1.00781357  1.3285324   1.64926347  1.97001377
 -1.11601767 -0.90726309 -0.69851971 -0.48982899 -0.28109733 -0.07234982
  0.13637329  0.34509172  0.55381134  0.76254879  0.97127605  1.17998183]
pulses [array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]), array([0. , 1. , 0. , 0. , 0.5, 0. , 0. , 0. ]), array([0. , 0. , 1. , 0. , 0. , 0.5, 0. , 0. ])]
sensor indices [[0, 0], [0, 4], [4, 4], [2, 10], [2, 2], [10, 10], [6, 0], [12, 2], [8, 6], [12, 6], [2, 0], [6, 4], [8, 4], [12, 0], [8, 10], [10, 0], [2, 8], [12, 10], [0, 6], [12, 8], [10, 8], [4, 6], [6, 8], [4, 2], [6, 6], [4, 8], [10, 2], [10, 4], [0, 10], [12, 4], [6, 2], [2, 4]]
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

### Turn 3 — `execute_in_remote_docker` (`20710 ms` LLM)
- **Rationale**: Compare held-out sweep errors before choosing a calibration model, and inspect the archived implementation for the prior wall equations and certificate solver.
- **Alternatives Weighed**: `Reuse the last campaign's reflecting wall rows unchanged` (rejected: The new campaign supplies tracer pulses that can independently test whether those rows still apply.)

```bash
sed -n '32,150p' /tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py; sed -n '1,43p' /tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py; python3 - <<'PY'
import numpy as np,pickle,json
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette'];
for j in [1,2,3]:
 A=np.column_stack([np.ones(len(x))]+[x[:,i]**k for k in range(1,j+1) for i in range(4)]); pred=np.zeros_like(y)
 for k in range(5):
  test=np.arange(len(x))%5==k; pred[test]=A[test]@np.linalg.lstsq(A[~test],y[~test],rcond=None)[0]
 print(j,'sweep out-of-fold per bank RMS',np.mean((pred-y)**2,axis=0)**.5)
 if j==1: print('linear coefficients',np.linalg.lstsq(A,y,rcond=None)[0])
print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],a.mean(0).round(3),a.std(0).round(3))
PY
```
- **Container Exit Code**: `0` (`360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(-1, 0, 1), shape=(len(kappa), len(kappa)), format='csc')


def _transport(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (n - 1), study['robin_' + a])
              for a, n in (('x', nx), ('y', ny))]
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * identity).solve(rhs)
        blocks.append(field[slots])
    return np.vstack(blocks), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    affine = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    observations = np.asarray(study['y_obs_raw'], dtype=float).reshape(-1, 4)
    return np.linalg.solve(affine[1:].T, (observations - affine[0]).T).T.ravel()
1 sweep out-of-fold per bank RMS [9.66268955e-06 1.02865118e-05 1.13827087e-05 1.25239130e-05]
linear coefficients [[ 6.99993212e-02 -5.99996396e-02  8.00011030e-02 -5.00007206e-02]
 [ 1.27000017e+00  1.09967211e-06 -1.39080264e-06  1.28479126e-06]
 [ 2.31770384e-06  8.10002023e-01 -1.46628668e-06 -3.75854324e-06]
 [ 7.05888016e-07 -3.34388068e-07  1.26000020e+00 -9.27460994e-07]
 [-6.15509551e-07 -7.60945419e-07 -1.56636122e-07  8.19998138e-01]]
2 sweep out-of-fold per bank RMS [1.01561207e-05 1.08274213e-05 1.11133619e-05 1.26302669e-05]
3 sweep out-of-fold per bank RMS [1.08500851e-05 1.10927181e-05 1.13402244e-05 1.28947722e-05]
aux mean/std [-0.27898896 -0.30482651 -0.19572754 -0.17152917] [1.11498765 0.94552161 0.90904871 1.0181768 ]
facility_init [ 0.006  0.004  0.003 -0.348 -0.     0.002 -0.37  -0.004] [0.603 0.603 0.608 0.059 0.594 0.614 0.067 0.6  ]
beta_m100_c01_s00 [ 0.009  0.006  0.007  2.425 -0.006 -0.004 -0.371 -0.009] [0.61  0.598 0.602 0.083 0.609 0.6   0.066 0.604]
beta_m100_c01_batch_01 [-0.008  0.007 -0.01  -0.337 -0.004  0.009  2.562 -0.004] [0.602 0.598 0.598 0.056 0.604 0.603 0.055 0.6  ]
```

### Turn 4 — `execute_in_remote_docker` (`37017 ms` LLM)
- **Rationale**: Test boundary stencils directly against tracer pulses, with and without nuisance projection. Per-pulse errors and leave-one-pulse-out continuous fits will distinguish a shared wall mechanism from a fit driven by one tracer.
- **Alternatives Weighed**: `Add nonlinear or cross-bank calibration terms` (rejected: The affine fit is already at the sweep noise floor, cross-bank coefficients are negligible, and higher-degree fits do not improve worst-bank held-out error.)

```bash
cat > /tmp/wall_probe.py <<'PY'
import numpy as np,pickle,itertools,importlib.util
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; p=item['campaign_diagnostics']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; singles=c['single_bank_sweeps']; bank=singles['bank_id']
coef=np.array([np.linalg.lstsq(np.c_[np.ones(len(x)+sum(bank==j)),np.r_[x[:,j],singles['ref_signal'][bank==j]]],np.r_[y[:,j],singles['raw_readout'][bank==j]],rcond=None)[0] for j in range(4)])
Y=np.column_stack([((q['y_pulse_raw'].reshape(-1,4)-coef[:,0])/coef[:,1]).ravel() for q in p]); W=np.column_stack([q['w_pulse'] for q in p]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0]; nx,ny=s['nx'],s['ny']; slots=s['sensor_ij']@np.array([1,nx]); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']])@W
print('pulse RMS by shift/pulse',np.sqrt(np.mean(Y.reshape(3,-1,len(p))**2,axis=1))); print('sensor diagonal sweep',coef)

def axis(a,n,bc):
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); v=s['vel_'+a][1:-1]/h; f=(k[:-1]+k[1:])/(2*h*h); A=diags((np.r_[-f[:-1]-np.maximum(v,0),0],np.r_[0,f[:-1]+f[1:]+abs(v),0],np.r_[0,-f[1:]+np.minimum(v,0)]),(-1,0,1)).toarray(); r=s['robin_'+a]/h
 for i,j,t in [(0,1,bc[0]),(n-1,n-2,bc[1])]:
  d=k[i]/h**2
  if t=='P': A[i,i]=d+k[i]/h**2;A[i,j]=-d;A[i,n-1-i]=-k[i]/h**2
  elif t=='D': A[i,i]=d+r
  else:
   alpha,beta={'R':(1,1),'N':(1,0),'H':(2,0),'B':(2,1),'W':(1,-1)}[t] if isinstance(t,str) else t
   A[i,i]=alpha*d+beta*r; A[i,j]=-alpha*d
 return csc_matrix(A)

def pred(bc):
 A=kron(eye(ny),axis('x',nx,bc[:2]),format='csc')+kron(axis('y',ny,bc[2:]),eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); return np.vstack([splu(A+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])

def residual(bc,proj=False):
 E=pred(bc)-Y
 return E-Q@(Q.T@E) if proj else E

def rms(e): return np.sqrt(np.mean(e*e,axis=0))
for proj in (False,True):
 bcs=list(itertools.product('RNPHBD',repeat=4)); scores=[]
 for b in bcs:
  E=residual(b,proj); scores.append((np.linalg.norm(E),''.join(b),rms(E)))
 for score in sorted(scores)[:8]: print('projected' if proj else 'direct',score)
 for b in ('RRRR','NNNN','PPPP'):
  print(b,rms(residual(b,proj)))
 for held in (-1,0,1,2):
  tr=np.arange(len(p))!=held
  fit=least_squares(lambda t:residual(t.reshape(4,2),proj)[:,tr].ravel(),np.tile([1.,0.5],4),bounds=(np.tile([0.,0.],4),np.tile([20.,20.],4)),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('continuous',proj,held,fit.x.round(5),rms(residual(fit.x.reshape(4,2),proj)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `1` (`3534 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pulse RMS by shift/pulse [[0.27539834 0.27747254 0.27771012]
 [0.15101232 0.150541   0.1501495 ]
 [0.097509   0.09584569 0.09561111]]
sensor diagonal sweep [[ 0.06999939  1.26999988]
 [-0.05999974  0.81000161]
 [ 0.07999974  1.2600007 ]
 [-0.05000103  0.81999867]]
direct (0.3144766717607161, 'PPPP', array([0.00670158, 0.02048199, 0.02378527]))
direct (0.38388694080481833, 'PNPP', array([0.03033271, 0.01861727, 0.01638351]))
direct (0.43051101009760484, 'NNPP', array([0.02845089, 0.02113235, 0.02597293]))
direct (0.44942449346243013, 'HNPP', array([0.02334774, 0.0142145 , 0.03683496]))
direct (0.4589036232580859, 'NHPP', array([0.02965142, 0.01784027, 0.03156248]))
direct (0.4963028333067248, 'HHPP', array([0.02570357, 0.01056671, 0.04234936]))
direct (0.49965684751591793, 'PPNP', array([0.00579665, 0.03109937, 0.03999777]))
direct (0.5216690410460039, 'PPNN', array([0.00929211, 0.03343139, 0.04038286]))
RRRR [0.11268845 0.11997155 0.11561371]
NNNN [0.02983791 0.03448729 0.04191408]
PPPP [0.00670158 0.02048199 0.02378527]
```
- **Stderr (tail)**:
```text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_lsq/least_squares.py", line 832, in least_squares
    f0 = fun_wrapped(x0)
         ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_lsq/least_squares.py", line 830, in fun_wrapped
    return np.atleast_1d(fun(x, *args, **kwargs))
                         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/wall_probe.py", line 39, in <lambda>
    fit=least_squares(lambda t:residual(t.reshape(4,2),proj)[:,tr].ravel(),np.tile([1.,0.5],4),bounds=(np.tile([0.,0.],4),np.tile([20.,20.],4)),gtol=1e-12,ftol=1e-12,xtol=1e-12)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/wall_probe.py", line 26, in residual
    E=pred(bc)-Y
      ^^^^^^^^
  File "/tmp/wall_probe.py", line 23, in pred
    A=kron(eye(ny),axis('x',nx,bc[:2]),format='csc')+kron(axis('y',ny,bc[2:]),eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); return np.vstack([splu(A+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])
                   ^^^^^^^^^^^^^^^^^^^
  File "/tmp/wall_probe.py", line 15, in axis
    if t=='P': A[i,i]=d+k[i]/h**2;A[i,j]=-d;A[i,n-1-i]=-k[i]/h**2
       ^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

### Turn 5 — `execute_in_remote_docker` (`30556 ms` LLM)
- **Rationale**: Compare compact periodic stencils and fit their near and wrap fluxes separately. Held-out pulses will show whether the same coefficients explain different source patterns or merely fit the development traces.
- **Alternatives Weighed**: `Freeze the simple wrap model` (rejected: Its large pulse-dependent residual leaves room for a materially better boundary law.)
- **ATMS NOGOOD DECLARED**: Refuted `Open-wall Robin rows suffice for this campaign.` | Witness: `Robin pulse RMSE ranged from 0.113 to 0.120, while adding opposite-wall coupling reduced it to 0.00670–0.0238.` | Blanket Diagnosis: `Boundary stencils restricted to adjacent cells omit a plausible transport path; the remaining wrap error still requires explanation.` | Sublated Pivot: `Retain the interior upwind stencil and test opposite-wall face coefficients with held-out pulses.`

```bash
sed -i "s/if t=='P':/if isinstance(t,str) and t=='P':/; s/elif t=='D':/elif isinstance(t,str) and t=='D':/" /tmp/wall_probe.py
cat > /tmp/wrap_probe.py <<'PY'
exec(open('/tmp/wall_probe.py').read().split('for proj in (False,True):')[0])
base=axis
def axis(a,n,bc):
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); v=s['vel_'+a]; r=s['robin_'+a]/h
 A=base(a,n,'NN').toarray()
 for i,j,t in [(0,1,bc[0]),(n-1,n-2,bc[1])]:
  p=n-1-i; d=k[i]/h**2; f=(k[i]+k[j])/(2*h*h); fw=(k[i]+k[p])/(2*h*h); vi=(v[i] if i==0 else -v[i])/h
  if isinstance(t,str):
   near,wrap={'row':(d,d),'avg':(f,fw),'mix':(d,fw),'period':(f+max(-vi,0),fw+max(vi,0)),'rowadv':(d+max(-vi,0),d+max(vi,0)),'wraponly':(0,fw),'no':(d,0),'dup':(f,0)}[t]; leak=0
  else:
   near,wrap,leak=t*np.array([d,d,r]);
  A[i,i]=near+wrap+leak;A[i,j]=-near;A[i,p]=-wrap
 return csc_matrix(A)
def score(E):
 return np.sqrt(np.mean(E.reshape(3,-1,len(p))**2,axis=1)) if False else np.sqrt(np.mean(E.reshape(3,-1,W.shape[1])**2,axis=1))
for proj in (False,True):
 print('projection',proj,flush=True)
 types=['row','avg','mix','period','rowadv','wraponly','no','dup']; results=[]
 for bx,by in itertools.product(types,repeat=2):
  E=residual((bx,bx,by,by),proj);results.append((np.linalg.norm(E),bx,by,rms(E)))
 for v in sorted(results,key=lambda r:r[0])[:6]: print(v,flush=True)
 for held in (-1,0,1,2):
  tr=np.arange(W.shape[1])!=held
  fit=least_squares(lambda t:residual(t.reshape(4,3),proj)[:,tr].ravel(),np.tile([1.,1.,0.],4),gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=160)
  E=residual(fit.x.reshape(4,3),proj)
  print('free-stencil fit',held,fit.x.reshape(4,3).round(6),rms(E),score(E),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wrap_probe.py
```
- **Container Exit Code**: `0` (`2541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
stencil fit 2 [[ 1.035210e+00  1.136022e+00 -1.000000e-05]
 [ 1.035383e+00  1.080114e+00  1.000000e-05]
 [ 9.952830e-01  7.609200e-01  3.000000e-06]
 [ 1.009206e+00  1.615880e+00  1.000000e-06]] [7.83870742e-07 7.92491285e-07 9.13820850e-07] [[8.95335741e-07 9.09969275e-07 1.15455123e-06]
 [7.47438296e-07 7.52742844e-07 8.28388927e-07]
 [6.95032319e-07 6.99615207e-07 6.97129036e-07]]
projection True
(1.4484782684317177e-05, 'period', 'period', array([8.54195537e-07, 8.62899490e-07, 8.43364771e-07]))
(0.012883831984791611, 'avg', 'period', array([0.00129425, 0.00011091, 0.0002042 ]))
(0.013751373113755301, 'rowadv', 'period', array([0.00078666, 0.00078127, 0.00086057]))
(0.02248885437953643, 'row', 'period', array([0.00204373, 0.00080467, 0.00066624]))
(0.02248885437953643, 'mix', 'period', array([0.00204373, 0.00080467, 0.00066624]))
(0.036858523140184764, 'period', 'mix', array([0.00059592, 0.00234969, 0.0028767 ]))
free-stencil fit -1 [[ 1.035246e+00  1.136051e+00 -7.000000e-06]
 [ 1.035348e+00  1.080060e+00  8.000000e-06]
 [ 9.952510e-01  7.608800e-01  3.000000e-06]
 [ 1.009227e+00  1.615886e+00  0.000000e+00]] [7.74781600e-07 7.83821438e-07 7.71630763e-07] [[8.99373061e-07 9.14829411e-07 9.07203620e-07]
 [7.42439194e-07 7.51658562e-07 7.36199784e-07]
 [6.63906412e-07 6.64247466e-07 6.49025096e-07]]
free-stencil fit 0 [[ 1.035276e+00  1.135971e+00 -2.000000e-06]
 [ 1.035318e+00  1.079956e+00  3.000000e-06]
 [ 9.952260e-01  7.608670e-01  1.000000e-06]
 [ 1.009254e+00  1.615933e+00  2.000000e-06]] [1.15907317e-06 7.82894551e-07 7.66708597e-07] [[1.58668459e-06 9.13597477e-07 9.02132733e-07]
 [9.57002441e-07 7.53274152e-07 7.31980367e-07]
 [7.72612539e-07 6.60824739e-07 6.43340882e-07]]
free-stencil fit 1 [[ 1.035281e+00  1.136084e+00 -5.000000e-06]
 [ 1.035328e+00  1.080039e+00  7.000000e-06]
 [ 9.952710e-01  7.608910e-01  4.000000e-06]
 [ 1.009192e+00  1.615838e+00 -2.000000e-06]] [7.74401494e-07 8.22622298e-07 7.69957220e-07] [[8.98646865e-07 9.80906322e-07 9.05787924e-07]
 [7.39608571e-07 7.81656098e-07 7.33339346e-07]
 [6.66712827e-07 6.75987327e-07 6.48277721e-07]]
free-stencil fit 2 [[ 1.035201e+00  1.136014e+00 -1.100000e-05]
 [ 1.035398e+00  1.080134e+00  1.000000e-05]
 [ 9.952910e-01  7.609260e-01  4.000000e-06]
 [ 1.009197e+00  1.615865e+00  0.000000e+00]] [7.73197121e-07 7.81835802e-07 1.00787462e-06] [[8.95817982e-07 9.13448245e-07 1.34984299e-06]
 [7.39868951e-07 7.49157906e-07 8.65973569e-07]
 [6.66037117e-07 6.61948941e-07 6.89526947e-07]]
```

### Turn 6 — `execute_in_remote_docker` (`31357 ms` LLM)
- **Rationale**: Implement the pulse-supported wall law and dynamic per-bank calibration, retaining the prior feasible QP algorithm. Run the public schema validator to detect shape or entrypoint mismatches before independent numerical verification.
- **Alternatives Weighed**: `Fit every boundary coefficient independently` (rejected: The exact periodic upwind law reaches the measurement floor on all three tracers without fitted wall coefficients.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Periodic upwind transport and a certified box-simplex source fit."""

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    n = len(kappa)
    row = np.arange(n)
    left = (row - 1) % n
    right = (row + 1) % n
    km = (kappa + kappa[left]) * (0.5 / h**2)
    kp = (kappa + kappa[right]) * (0.5 / h**2)
    west = km + np.maximum(velocity, 0.0) / h
    east = kp + np.maximum(-velocity, 0.0) / h
    return csc_matrix((np.r_[-west, west + east, -east],
                       (np.tile(row, 3), np.r_[left, row, right])),
                      shape=(n, n))


def _transport(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (n - 1))
              for a, n in (('x', nx), ('y', ny))]
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    observations = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * identity).solve(rhs)
        observations.append(field[slots])
    return np.vstack(observations), field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study, raw_readout):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    single = sweeps['single_bank_sweeps']
    offset, gain = np.empty(4), np.empty(4)
    for j in range(4):
        use = np.asarray(single['bank_id']) == j
        x = np.r_[ref[:, j], single['ref_signal'][use]]
        y = np.r_[raw[:, j], single['raw_readout'][use]]
        offset[j], gain[j] = np.linalg.lstsq(
            np.column_stack((np.ones_like(x), x)), y, rcond=None)[0]
    return ((np.asarray(raw_readout).reshape(-1, 4) - offset) / gain).ravel()


PY
sed -n '/^def _box_simplex/,$p' /tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py >> /holdout_task/batch_solver.py
sed -n '/^def _box_simplex/,$p' /holdout_task/batch_solver.py
python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`453 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
F = H[np.ix_(free, free)]
        # Solve for the face minimizer, enforcing the remaining sum exactly.
        rhs = -c[free] - H[free] @ np.where(state == 0, 0.0, w)
        ones = np.ones(len(free))
        a, b = np.linalg.solve(F, np.column_stack((rhs, ones))).T
        nu = (budget - np.sum(w[state != 0]) - a.sum()) / b.sum()
        wf = a + nu * b
        step = wf - w[free]
        hits = np.full(len(free), np.inf)
        lower, upper = wf < -tol, wf > cap + tol
        hits[lower] = -w[free][lower] / step[lower]
        hits[upper] = (cap - w[free][upper]) / step[upper]
        k = int(np.argmin(hits))
        if hits[k] <= 1.0:
            w[free] += np.clip(hits[k], 0.0, 1.0) * step
            state[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0.0 if lower[k] else cap
            continue
        w[free] = np.clip(wf, 0.0, cap)
        g = H @ w + c
        violation = state * (g - nu)
        i = int(np.argmax(violation))
        if violation[i] <= tol:
            return w, float(nu)
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _transport(study)
    y = _calibrate(study, study['y_obs_raw'])
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = study['source_ridge']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4f97650acdd9214 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a4f97650 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38214 ms` LLM)
- **Rationale**: Verify the submitted file in isolation. An independently assembled operator can expose indexing or terminal-shift errors; certificate reconstruction and endpoint budget tests can expose multiplier sign or active-set errors that the schema check cannot.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
worst_pulse = worst_field = worst_gap = worst_complementarity = worst_budget = 0.0
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = item['study']; result = m.solve_batch(s)
    w, u, cert = (result[k] for k in ('weights','terminal_field','kkt_certificate'))
    nx, ny = s['nx'], s['ny']; ii, jj = np.asarray(s['sensor_ij']).T
    operator = np.diag(s['V_reaction_2d'].ravel(order='F'))
    for axis, n, stride in [('x', nx, 1), ('y', ny, nx)]:
        h = s['L'+axis]/(n-1); k = s['kappa_'+axis]; v = s['vel_'+axis]
        for row in range(nx*ny):
            p = row//stride % n
            for direction in (-1, 1):
                q = (p+direction) % n
                coeff = (k[p]+k[q])/(2*h*h)+max(0.,-direction*v[p])/h
                operator[row,row] += coeff
                operator[row,row+(q-p)*stride] -= coeff
    rhs = np.asarray([x.ravel(order='F') for x in s['source_modes']]).T
    blocks = []
    for mu in s['mu_shifts']:
        shifted = operator + mu*np.eye(nx*ny)
        fields = np.linalg.solve(shifted, rhs)
        blocks.append(fields[ii+nx*jj])
    err = np.linalg.norm(shifted@u.ravel(order='F')-rhs@w)/np.linalg.norm(rhs@w)
    worst_field = max(worst_field,err)
    S = np.vstack(blocks)
    for p in item.get('campaign_diagnostics') or []:
        e = (S@p['w_pulse']-m._calibrate(s,p['y_pulse_raw'])).reshape(len(blocks),-1)
        worst_pulse = max(worst_pulse,float(np.sqrt(np.mean(e**2,axis=1)).max()))
    Z = s['Z_nuisance']; r = m._calibrate(s,s['y_obs_raw'])
    X = np.column_stack((S, r)); X -= Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@X)
    A, r = X[:,:-1], X[:,-1]; ridge = s['source_ridge']; g = A.T@(A@w-r)+ridge*w
    J = .5*(np.linalg.norm(A@w-r)**2+ridge*(w@w)); cap = s['u_box']; budget = s['total_budget']
    nu, lo, hi = (cert[k] for k in ('nu_sum','lam_lower','lam_upper'))
    assert np.array_equal(w,cert['weights']) and np.all((w>=0)&(w<=cap))
    assert np.all(lo>=0) and np.all(hi>=0) and np.max(np.abs(g-nu-lo+hi))<1e-10
    assert abs(J-cert['primal_objective'])<1e-10
    dual = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    assert abs(dual-cert['dual_lower_bound'])<1e-10
    worst_gap = max(worst_gap,abs(J-dual))
    worst_complementarity = max(worst_complementarity,np.max(np.abs(lo*w)),np.max(np.abs(hi*(cap-w))))
    worst_budget = max(worst_budget,abs(w.sum()-budget))
print('public worst pulse-by-shift RMSE, field equation relative residual, budget error, gap, complementarity:',worst_pulse,worst_field,worst_budget,worst_gap,worst_complementarity)
assert worst_pulse<2e-6 and max(worst_field,worst_budget,worst_gap,worst_complementarity)<1e-10
rng=np.random.default_rng(498); mx=0.
for n in (4, 8, 16):
    for trial in range(40):
        B=rng.normal(size=(n,n)); H=B.T@B+.04*np.eye(n); c=5*rng.normal(size=n)
        for b in (0., 1e-8, 1.4, float(n//2), n-1e-8, float(n)):
            w,nu=m._box_simplex(H,c,1.,b); g=H@w+c-nu
            e=max(abs(w.sum()-b),np.max(np.maximum(g,0)*w),np.max(np.maximum(-g,0)*(1-w)))
            assert np.all((w>=0)&(w<=1)) and e<1e-9, (n,trial,b,e)
            mx=max(mx,e)
print('720 stressed QPs, worst budget/complementarity residual:',mx)
PY
```
- **Container Exit Code**: `0` (`636 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public worst pulse-by-shift RMSE, field equation relative residual, budget error, gap, complementarity: 9.903091456092953e-07 5.395350083064966e-15 0.0 1.3898604489526178e-14 2.0811051180237145e-16
720 stressed QPs, worst budget/complementarity residual: 1.4210854573093454e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4f97650acdd9214 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a4f97650 (directory).
```

### Turn 8 — `submit_completed_task` (`13463 ms` LLM)
- **Rationale**: The periodic stencil resolved the measured Robin boundary mismatch. The saved candidate passed schema validation, an independent field and certificate check in an isolated Python process, and optimizer stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5191'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema check passed. Worst tracer RMSE across pulses and shifts was 9.90e-7, and the worst budget or complementarity residual across 720 stress-tested quadratic programs was 1.42e-14.`

### Turn 9 — `execute_in_remote_docker` (`37362 ms` LLM)
- **Rationale**: A pooled tracer error could hide a bad sensor bank, and nuisance projection could mask a systematic mismatch. Compare worst bank-by-shift tracer errors before and after projection using calibration fits that exclude the held-out sweeps. Change the artifact only if a challenger improves those slices and materially affects the recovered weights or field.
- **Alternatives Weighed**: `Test relay modes, drift equations, or within-batch classification accuracy` (rejected: This is a steady transport inverse problem with no relay state, drift equation, or class labels. The relevant strata are sensor bank, tracer pulse, and transport shift.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for i, item in enumerate(pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))):
    s = item['study']; c = s['sensor_calibration_sweeps']; b = c['single_bank_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']; n = len(x)
    X = np.zeros((n+len(b['bank_id']), 4)); X[:n] = x
    X[n+np.arange(len(b['bank_id'])), np.asarray(b['bank_id'], int)] = b['ref_signal']
    Y = np.full_like(X, np.nan); Y[:n] = y
    Y[n+np.arange(len(b['bank_id'])), np.asarray(b['bank_id'], int)] = b['raw_readout']
    S, U = m._transport(s); Z = s['Z_nuisance']
    pulses = item.get('campaign_diagnostics') or []
    r = m.solve_batch(s); weights = r['weights']; cert = r['kkt_certificate']
    print('item', i, 'public primal-dual gap', cert['primal_objective']-cert['dual_lower_bound'])
    for model in ('diagonal_affine', 'coupled_affine', 'diagonal_quadratic'):
        raw_oof = np.empty_like(y); pulse_err = []; weight_err = []; field_err = []; nuisance_err = []
        for fold in range(5):
            hold = np.arange(n) % 5 == fold
            coeff = []
            for j in range(4):
                features = {'diagonal_affine': X[:,j:j+1], 'coupled_affine': X, 'diagonal_quadratic': np.c_[X[:,j], X[:,j]**2]}[model]
                D = np.c_[np.ones(len(X)), features]; train = np.r_[~hold, np.ones(len(X)-n, dtype=bool)] & np.isfinite(Y[:,j])
                q = np.linalg.lstsq(D[train],Y[train,j],rcond=None)[0]
                coeff.append(q); raw_oof[hold,j] = D[:n][hold]@q
            C = np.array(coeff).T
            def calibrate(raw):
                d = raw.reshape(-1,4)-C[0]
                if model == 'coupled_affine':
                    return np.linalg.solve(C[1:].T,d.T).T.ravel()
                if model == 'diagonal_quadratic':
                    return (2*d/(C[1]+np.sqrt(C[1]**2+4*C[2]*d))).ravel()
                return (d/C[1]).ravel()
            P = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
            A = S-Z@(P@S); obs = calibrate(s['y_obs_raw']); y_fit = obs-Z@(P@obs)
            H = A.T@A+s['source_ridge']*np.eye(A.shape[1]); w,_ = m._box_simplex(H,-A.T@y_fit,s['u_box'],s['total_budget'])
            weight_err.append(np.linalg.norm(w-weights)/np.linalg.norm(weights))
            field_err.append(np.linalg.norm(np.tensordot(w-weights,U,axes=1))/np.linalg.norm(r['terminal_field']))
            for p in pulses:
                e = calibrate(p['y_pulse_raw'])-S@p['w_pulse']; e_adj = e-Z@(P@e)
                pulse_err.append(np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1,4)**2,axis=1)).max())
                nuisance_err.append(np.sqrt(np.mean(e_adj.reshape(len(s['mu_shifts']),-1,4)**2,axis=1)).max())
        print(model, 'OOF sweep RMSE by bank', np.sqrt(np.mean((raw_oof-y)**2,axis=0)), 'worst bank/shift/pulse RMSE before and after nuisance projection', max(pulse_err,default=float('nan')), max(nuisance_err,default=float('nan')), 'max relative weight/field movement', max(weight_err), max(field_err))
PY
```
- **Container Exit Code**: `0` (`425 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
item 0 public primal-dual gap 6.817463260588852e-16
diagonal_affine OOF sweep RMSE by bank [9.64086454e-06 9.75213827e-06 1.09975583e-05 1.24321352e-05] worst bank/shift/pulse RMSE before and after nuisance projection 2.7146736519029133e-06 2.6945605475694083e-06 max relative weight/field movement 1.3194571430451318e-05 1.7158582304243612e-06
coupled_affine OOF sweep RMSE by bank [9.68115460e-06 1.01803336e-05 1.13991011e-05 1.24722123e-05] worst bank/shift/pulse RMSE before and after nuisance projection 3.6516583838244773e-06 3.610959550157325e-06 max relative weight/field movement 1.140888808821023e-05 3.0243892800688903e-06
diagonal_quadratic OOF sweep RMSE by bank [9.75782638e-06 9.85863166e-06 1.10978252e-05 1.23767859e-05] worst bank/shift/pulse RMSE before and after nuisance projection 3.898703136756416e-06 3.901348439930807e-06 max relative weight/field movement 1.4776806385779548e-05 2.789726227249289e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4f97650acdd9214 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a4f97650 (directory).
```

### Turn 10 — `submit_completed_task` (`14383 ms` LLM)
- **Rationale**: The unchanged candidate passed the isolated field-equation and KKT checks. The targeted calibration audit found no improvement on the worst tracer slices, so no artifact change is warranted.
- **Alternatives Weighed**: `Coupled affine or per-bank quadratic calibration` (rejected: Both increased the worst tracer-slice error, and nuisance projection did not change the ranking.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, the worst tracer RMSE across pulses and shifts was 9.90e-7, and all 720 optimizer stress tests passed. The calibration audit supported retaining the per-bank affine model.`

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
