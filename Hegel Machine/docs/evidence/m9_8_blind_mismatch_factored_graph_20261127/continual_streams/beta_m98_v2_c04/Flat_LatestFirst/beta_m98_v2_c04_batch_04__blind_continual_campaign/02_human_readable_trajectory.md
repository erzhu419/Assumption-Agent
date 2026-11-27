# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c04_batch_04__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `383,449` (`371,767` / `128,000` / `11,682`) |
| **Total Episode Wall Time (s)** | `390.15s` |

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

### Turn 1 — `execute_in_remote_docker` (`15349 ms` LLM)
- **Rationale**: Inspect the legal regime evidence and schema before choosing the boundary operator. The skill lookup checks whether nuisance profiling has a reusable implementation.
- **Alternatives Weighed**: `Copy the latest archived solver unchanged` (rejected: The public gate rejected latest-first reuse; the apparatus log can distinguish recurrence from a new regime.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; python3 - <<'PY'
import pickle
with open('/holdout_task/public_batch.pkl','rb') as f: items=pickle.load(f)
print('Items:',len(items))
for item in items:
 print('Item keys:',list(item),'diagnostics:',type(item.get('campaign_diagnostics')).__name__)
 for k,v in item['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else repr(v) if isinstance(v,(str,int,float,type(None))) else '')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box equality quadratic KKT'
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2834,
        -0.12648,
        -0.86856,
        -1.49371
      ],
      [
        -1.00614,
        -0.38964,
        -0.26156,
        -0.3554,
        -1.0684,
        -0.08427,
        -0.68388,
        -1.16937
      ],
      [
        -0.35511,
        -0.37793,
        -0.09219,
        -0.45192,
        -1.11256,
        0.01201,
        -0.29205,
        -0.90868
      ],
      [
        -0.1803,
        -0.33324,
        -0.09781,
        -0.28228,
        -1.00426,
        -0.17118,
        0.12874,
        -1.29811
      ],
      [
        0.57572,
        -0.47103,
        -0.06708,
        -0.40152,
        -1.14977,
        -0.09363,
        0.0375,
        -1.21522
      ],
      [
        0.78053,
        -0.26643,
        0.26406,
        -0.34482,
        -1.17306,
        0.08379,
        0.62595,
        -1.19666
      ],
      [
        0.81875,
        -0.31027,
        0.07313,
        -0.49361,
        -1.11646,
        -0.2346,
        0.66879,
        -1.40237
      ],
      [
        1.1624,
        -0.38266,
        0.08944,
        -0.37319,
        -1.19915,
        0.09185,
        0.84017,
        -1.25825
      ],
      [
        0.84121,
        -0.26822,
        0.1086,
        -0.37631,
        -1.32892,
        -0.03798,
        0.57064,
        -1.25285
      ]
    ]
  }
]Items: 1
Item keys: ['study', 'campaign_diagnostics'] diagnostics: NoneType
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 1.0657471418243107
robin_y float None 0.9377365279326434
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) 
source_modes ndarray (8, 14, 12) 
sensor_ij ndarray (32, 2) 
row_bank_id ndarray (96,) 
sensor_calibration_sweeps dict None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3) 
y_obs_raw ndarray (96,) 
aux_manifold_telemetry ndarray (24, 4) 
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19043 ms` LLM)
- **Rationale**: Compare the archived stencils and summarize each log entry so the boundary choice can be made from apparatus evidence rather than calibration fit or campaign order.
- **Alternatives Weighed**: `Use the boxed L0 skill` (rejected: The requested convex problem has an equality-constrained mass budget and no L0 penalty.)

```bash
cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json; cd /tmp/hegel_continual_archive; cat beta_m98_v2_c04_batch_03_solver.py; for f in *solver.py; do echo ==== "$f"; diff -u beta_m98_v2_c04_batch_03_solver.py "$f" || test "$?" = 1; done; python3 - <<'PY'
import json,numpy as np,pickle
p=json.load(open('archive_apparatus_log.json')); print('Log:',type(p).__name__)
for k,e in (p.items() if isinstance(p,dict) else enumerate(p)):
 print('Entry',k)
 if isinstance(e,dict):
  for key,v in e.items():
   a=np.asarray(v) if isinstance(v,list) else None
   if a is not None and a.dtype.kind in 'fiu': print(key,a.shape,'mean',a.mean(axis=0).round(4),'std',a.std(axis=0).round(4))
   else: print(key,str(v)[:400])
 else: print(str(e)[:400])
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; print('Calibration:',[(k, type(v).__name__,getattr(v,'shape',None), str(v)[:600] if k=='single_bank_sweeps' else '') for k,v in c.items()]); t=s['aux_manifold_telemetry']; print('Public telemetry',t.mean(axis=0),t.std(axis=0))
PY
```
- **Container Exit Code**: `0` (`385 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
an(axis=0)
-    gain = np.sum(x * (raw - raw.mean(axis=0)), axis=0) / np.sum(x * x, axis=0)
-    offset = raw.mean(axis=0) - gain * ref.mean(axis=0)
-    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
-    return ((obs - offset) / gain).ravel()
+    ref, raw = np.asarray(sweeps['ref_rosette']), np.asarray(sweeps['raw_rosette'])
+    offset, *mix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
+    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
+    return np.linalg.solve(np.asarray(mix).T, (observed - offset).T).T.ravel()
 
 
 def _box_budget(hessian, linear, cap, budget):
Log: list
Entry 0
completed_batch_index 0
completed_short_name beta_m98_v2_c04_s00
facility_telemetry_trace (16, 8) mean [-0.0072  2.4346 -0.676  -0.3809  1.1365 -0.7031 -0.0183  1.178 ] std [0.6327 0.0751 0.1805 0.0893 0.1265 0.1035 0.6648 0.1368]
Entry 1
completed_batch_index 1
completed_short_name beta_m98_v2_c04_batch_01
facility_telemetry_trace (16, 8) mean [ 0.0934 -0.3501 -0.4786 -0.3968 -1.2032 -0.3445 -0.0243 -1.1328] std [0.6479 0.055  0.1164 0.0794 0.1025 0.1305 0.6097 0.15  ]
Entry 2
completed_batch_index 2
completed_short_name beta_m98_v2_c04_batch_02
facility_telemetry_trace (16, 8) mean [ 0.0687 -0.3436 -0.213   2.5216  1.1411 -0.1836  0.0925  1.1502] std [0.7284 0.0726 0.1087 0.0836 0.1561 0.1171 0.6791 0.1447]
Entry 3
completed_batch_index 3
completed_short_name beta_m98_v2_c04_batch_03
facility_telemetry_trace (16, 8) mean [ 7.4700e-02 -3.4680e-01  1.1000e-03 -3.9050e-01 -1.1564e+00 -5.5600e-02
  1.5700e-02 -1.1881e+00] std [0.6961 0.0636 0.1849 0.0629 0.0773 0.1152 0.6303 0.1444]
Calibration: [('ref_rosette', 'ndarray', (80, 4), ''), ('raw_rosette', 'ndarray', (80, 4), ''), ('single_bank_sweeps', 'dict', None, "{'bank_id': array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,\n       3, 3]), 'ref_signal': array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,  0.9,\n        1.4, -1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,\n        0.9,  1.4]), 'raw_readout': array([-1.1584516 , -0.58388634, -0.14215374,  0.2774377 ,  0.78560066,\n       15.99302104, -1.64464425, -0.74775681, -0.05824291,  0.59674288,\n        1.38996829,  2.49421232, -1.68823618, -0.76330364, -0.05224512,\n        0.62319806,  1.44121001,  2.5799701 , -1.11487678, -0.56832234,\n       -0.148129")]
Public telemetry [ 0.11146496 -0.08548183 -0.16320776 -0.2134394 ] [0.69438133 1.14779047 0.83033718 1.14561795]
```

### Turn 3 — `execute_in_remote_docker` (`24380 ms` LLM)
- **Rationale**: The previous output truncated the stencil comparison. Read just the relevant functions and use held-out calibration residuals to determine whether model complexity or isolated contamination explains the errors.
- **Alternatives Weighed**: `Fit calibration from the single-bank sweeps alone` (rejected: Bank 0 has a visibly inconsistent reading, and single-channel measurements cannot identify cross-channel mixing.)

```bash
cd /tmp/hegel_continual_archive; for f in *solver.py; do echo ==== "$f"; sed -n '/^def _axis/,/^def _forward/p; /^def _calibrate/,/^def _box_budget/p' "$f"; grep -E 'ax =|ay =' "$f"; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust Cauchy IRLS regression'; python3 - <<'PY'
import numpy as np,pickle,json
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; r,y=c['ref_rosette'],c['raw_rosette']; X=np.c_[np.ones(len(r)),r]; print('mix with offset',np.linalg.lstsq(X,y,rcond=None)[0].round(5));
for j in range(4):
 for model,x in [('diag',X[:,[0,j+1]]),('full',X),('quadratic',np.c_[X,*[r[:,a]*r[:,b] for a in range(4) for b in range(a,4)]])]:
  q=np.linalg.lstsq(x,y[:,j],rcond=None)[0]; e=y[:,j]-x@q; h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1); print(j,model,'LOO abs quantiles',np.quantile(abs(e/(1-h)),[.5,.9,.95,1]).round(6),'top rows',np.argsort(abs(e))[-4:]);
b=c['single_bank_sweeps']; print('bank sweeps');
for j in range(4):
 a=b['bank_id']==j; print(j,np.c_[b['ref_signal'][a],b['raw_readout'][a]].round(5));
p=json.load(open('archive_apparatus_log.json')); traces=np.array([e['facility_telemetry_trace'] for e in p]); print('latest minus prior means',np.round((traces[-1]-traces[:-1]).mean(axis=1),4)); print('bank layout',s['row_bank_id'].reshape(-1,4)[:8]); print('mu shifts',s['mu_shifts']); print('obs ranges',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.5,1],axis=0));
PY
```
- **Container Exit Code**: `0` (`543 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
mix with offset [[ 0.54706 -0.46046  0.70972 -0.63431]
 [ 0.72885  0.38567 -0.26288  0.03388]
 [-0.34206  1.68775  0.18395  0.10602]
 [ 0.00933 -0.46337  1.73122  0.0598 ]
 [ 0.53989 -0.3236  -0.04339  0.56069]]
0 diag LOO abs quantiles [ 0.596687  1.05482   1.126152 13.287404] top rows [27 13 14 15]
0 full LOO abs quantiles [ 0.557395  1.190174  1.380713 13.324806] top rows [17 15 14 13]
0 quadratic LOO abs quantiles [ 0.826073  2.185907  3.410596 12.654122] top rows [17 15 14 13]
1 diag LOO abs quantiles [ 0.614498  1.113051  1.516485 12.513712] top rows [41 14 13 15]
1 full LOO abs quantiles [ 0.528296  1.15797   1.554632 12.649083] top rows [45 15 14 13]
1 quadratic LOO abs quantiles [ 0.807433  1.950608  2.781661 11.882257] top rows [17 15 14 13]
2 diag LOO abs quantiles [ 0.566342  1.165566  1.701515 12.135016] top rows [67 15 14 13]
2 full LOO abs quantiles [ 0.598835  1.189788  1.507454 12.146191] top rows [49 15 14 13]
2 quadratic LOO abs quantiles [ 0.766759  2.047677  3.154903 11.585252] top rows [17 15 14 13]
3 diag LOO abs quantiles [ 0.476752  1.158054  1.448601 12.985435] top rows [45 15 14 13]
3 full LOO abs quantiles [ 0.51526   1.293478  1.592361 12.989215] top rows [45 15 14 13]
3 quadratic LOO abs quantiles [ 0.823583  2.270074  3.111223 12.223478] top rows [47 15 14 13]
bank sweeps
0 [[-1.1     -1.15845]
 [-0.6     -0.58389]
 [-0.1     -0.14215]
 [ 0.4      0.27744]
 [ 0.9      0.7856 ]
 [ 1.4     15.99302]]
1 [[-1.1     -1.64464]
 [-0.6     -0.74776]
 [-0.1     -0.05824]
 [ 0.4      0.59674]
 [ 0.9      1.38997]
 [ 1.4      2.49421]]
2 [[-1.1     -1.68824]
 [-0.6     -0.7633 ]
 [-0.1     -0.05225]
 [ 0.4      0.6232 ]
 [ 0.9      1.44121]
 [ 1.4      2.57997]]
3 [[-1.1     -1.11488]
 [-0.6     -0.56832]
 [-0.1     -0.14813]
 [ 0.4      0.25098]
 [ 0.9      0.73435]
 [ 1.4      1.40725]]
latest minus prior means [[ 0.0819 -2.7814  0.6771 -0.0096 -2.2929  0.6474  0.034  -2.3661]
 [-0.0187  0.0033  0.4796  0.0063  0.0469  0.2889  0.04   -0.0553]
 [ 0.006  -0.0032  0.2141 -2.9121 -2.2975  0.128  -0.0768 -2.3383]]
bank layout [[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
mu shifts [0.15 0.55 1.1 ]
obs ranges [[-0.24272822 -0.09537634 -0.20281407 -0.20441073]
 [-0.00561593  0.16309277  0.10054351 -0.01445678]
 [ 0.25785599  0.61788569  0.7063332   0.16571259]]
```

### Turn 4 — `execute_in_remote_docker` (`22740 ms` LLM)
- **Rationale**: Separate the three conspicuous calibration rows from systematic curvature before choosing a model. Compare held-out errors on the remaining sweeps and inspect archived forward and certificate code for reusable pieces.
- **Alternatives Weighed**: `Switch boundary stencil along with calibration` (rejected: The latest trace matches batch 01 in the strongly separated telemetry channels; inspect whether batches 01 and 03 already share the sealed-wall stencil.)

```bash
cd /tmp/hegel_continual_archive; grep -A14 '^def _axis' *03_solver.py; sed -n '/^def _forward/,/^def _calibrate/p; /^def _box_budget/,$p' *01_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -18; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['sensor_calibration_sweeps']; x,y=d['ref_rosette'],d['raw_rosette']; print('rows 10..18:', np.c_[x,y][10:19].round(4)); keep=np.ones(len(x),bool);keep[[13,14,15]]=False
for j in range(4):
 for name,A in [('diag',np.c_[np.ones(len(x)),x[:,j]]),('full',np.c_[np.ones(len(x)),x]),('diag cubic',np.c_[np.ones(len(x)),x[:,j],x[:,j]**3]),('full + own cubic',np.c_[np.ones(len(x)),x,x[:,j]**3]),('full + all cubes',np.c_[np.ones(len(x)),x,x**3]),('full + own nonlinear',np.c_[np.ones(len(x)),x,x[:,j]**2,x[:,j]**3])]:
  B=A[keep]; beta=np.linalg.lstsq(B,y[keep,j],rcond=None)[0]; e=(y[keep,j]-B@beta)/(1-np.sum(np.linalg.qr(B,mode='reduced')[0]**2,axis=1)); print(j,name,'LOOCV rms/max',np.sqrt(np.mean(e**2)).round(5),abs(e).max().round(5),'coef',beta.round(5))
PY
```
- **Container Exit Code**: `0` (`519 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
65600e-01  2.50500e-01  7.79700e-01  1.46920e+00
  -5.33100e-01 -4.91000e-02  2.15000e-01]]
0 diag LOOCV rms/max 0.31387 0.83919 coef [-0.0896   1.00635]
0 full LOOCV rms/max 0.14147 0.43624 coef [-0.04141  0.98554 -0.27249 -0.00615  0.16182]
0 diag cubic LOOCV rms/max 0.32015 0.84001 coef [-0.08913  0.9863   0.01305]
0 full + own cubic LOOCV rms/max 0.12217 0.36449 coef [-0.03335  0.80542 -0.29437 -0.00823  0.15918  0.11625]
0 full + all cubes LOOCV rms/max 0.12413 0.34913 coef [-0.03832  0.80876 -0.34763  0.02287  0.11123  0.11764  0.03602 -0.01664
  0.03035]
0 full + own nonlinear LOOCV rms/max 0.12633 0.37602 coef [-0.04059  0.81138 -0.29477 -0.008    0.16013  0.01054  0.11015]
1 diag LOOCV rms/max 0.50977 1.60428 coef [0.11902 1.61941]
1 full LOOCV rms/max 0.26346 0.8717 coef [ 0.10113  0.14066  1.62171 -0.44804  0.03996]
1 diag cubic LOOCV rms/max 0.50323 1.37489 coef [0.10941 1.27574 0.23215]
1 full + own cubic LOOCV rms/max 0.23599 0.606 coef [ 0.08692  0.15575  1.31764 -0.4384   0.03647  0.20581]
1 full + all cubes LOOCV rms/max 0.23827 0.69011 coef [ 0.09499  0.1337   1.34543 -0.30328 -0.03741  0.01917  0.1881  -0.08372
  0.04138]
1 full + own nonlinear LOOCV rms/max 0.24086 0.60554 coef [ 0.07046  0.15419  1.33122 -0.43813  0.0345   0.02481  0.19284]
2 diag LOOCV rms/max 0.49125 1.68171 coef [0.19425 1.7944 ]
2 full LOOCV rms/max 0.30719 1.13931 coef [ 0.17157 -0.02812  0.24736  1.71673 -0.39074]
2 diag cubic LOOCV rms/max 0.4073 0.99338 coef [0.1152  1.11535 0.42486]
2 full + own cubic LOOCV rms/max 0.21605 0.66129 coef [ 0.12396 -0.05476  0.22144  1.21037 -0.34212  0.32139]
2 full + all cubes LOOCV rms/max 0.21758 0.60239 coef [ 0.11905 -0.14448  0.22672  1.22476 -0.43958  0.05965 -0.01247  0.31555
  0.06466]
2 full + own nonlinear LOOCV rms/max 0.20945 0.52591 coef [ 0.06395 -0.05436  0.21522  1.27919 -0.35056  0.09274  0.25986]
3 diag LOOCV rms/max 0.25201 0.96303 coef [-0.1194   0.89778]
3 full LOOCV rms/max 0.15901 0.72312 coef [-0.05831 -0.21737  0.03797  0.07502  0.93107]
3 diag cubic LOOCV rms/max 0.23986 0.73903 coef [-0.14176  0.67327  0.14986]
3 full + own cubic LOOCV rms/max 0.12506 0.45032 coef [-0.0834  -0.21007  0.03242  0.09098  0.70875  0.15064]
3 full + all cubes LOOCV rms/max 0.1311 0.46511 coef [-8.5500e-02 -2.3926e-01  2.7410e-02  6.0980e-02  7.0656e-01  1.7920e-02
 -1.0000e-05  1.9050e-02  1.5380e-01]
3 full + own nonlinear LOOCV rms/max 0.12876 0.46344 coef [-0.07832 -0.20993  0.03327  0.09132  0.70226 -0.0105   0.1569 ]
```

### Turn 5 — `execute_in_remote_docker` (`31689 ms` LLM)
- **Rationale**: Pure-axis sweeps distinguish self-response from cross-talk; complete low-degree terms test whether the existing cubic fit misses interactions. The stencil check resolves what the rejected latest-first solver actually used.
- **Alternatives Weighed**: `Use full affine mixing after outlier rejection` (rejected: Clean held-out errors remain large relative to the observations; pure-axis and mixed sweeps permit identification of the missing nonlinear terms.)
- **ATMS NOGOOD DECLARED**: Refuted `Independent affine calibration can explain current clean rosette sweeps.` | Witness: `At fixed zero input on bank 1, changing only reference bank 2 from 0.5 to 1.3 changes raw bank 1 from -0.1156 to -0.4126. Even full affine calibration leaves worst-bank clean LOO RMS of 0.307.` | Blanket Diagnosis: `The readout depends on other channels, not merely own-bank gain and offset. Structured curvature remains after removing the grossly corrupted rows.` | Sublated Pivot: `Keep the archived sealed-wall candidate separate and identify the sparse nonlinear readout from controlled sweeps.`

```bash
cat >/tmp/rosette_probe.py <<'PY'
import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; keep=np.max(abs(y),axis=1)<8; x,y=x[keep],y[keep]; print('sweep ranges',x.min(0),x.max(0),'rejected',np.where(~keep)[0]); print('pure axis raw'); print(np.c_[x,y][np.count_nonzero(x,axis=1)==1].round(5))
terms=list(itertools.combinations_with_replacement(range(4),2))+list(itertools.combinations_with_replacement(range(4),3)); T=np.column_stack([np.prod(x[:,q],axis=1) for q in terms]); X=np.c_[np.ones(len(x)),x]; print('term order',terms)
for j in range(4):
 base=np.c_[X,x[:,j]**3]; e=y[:,j]-base@np.linalg.lstsq(base,y[:,j],rcond=None)[0]; print('\nbank',j,'base residual worst indices',np.where(keep)[0][np.argsort(abs(e))[-9:]]); print('base residual correlations',sorted(zip(abs(np.corrcoef(np.c_[T,e].T)[-1,:-1]),terms),reverse=True)[:8]);
 for name,A,tt in [('pure cubes + pair products',np.c_[X,x**3,T[:,:10]],[(i,i,i) for i in range(4)]+terms[:10]),('complete degree 3',np.c_[X,T],terms)]:
  coef=np.linalg.lstsq(A,y[:,j],rcond=None)[0]; r=y[:,j]-A@coef; loo=r/(1-(np.linalg.qr(A,mode='reduced')[0]**2).sum(1)); print(name,'LOO rms/max',np.sqrt(np.mean(loo**2)),abs(loo).max(),'fit rms',np.sqrt(np.mean(r**2))); print('linear',coef[:5].round(4),'nonlin',[(t,round(v,4)) for t,v in zip(tt,coef[5:]) if abs(v)>.01]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_probe.py
cd /tmp/hegel_continual_archive; grep -A7 '^def _forward' *03_solver.py
```
- **Container Exit Code**: `0` (`366 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0, 0, 2)), (0.2258507053459704, (1, 3)), (0.21094642915331321, (2, 2, 3)), (0.1890352776936476, (1, 1, 2)), (0.1790645133783057, (2, 3, 3))]
pure cubes + pair products LOO rms/max 1.1159337626093901e-05 3.1125909704558215e-05 fit rms 8.495578011053736e-06
linear [ 0.07    0.1856  1.28   -0.3712 -0.    ] nonlin [((1, 1, 1), 0.2304), ((1, 2), -0.2816)]
complete degree 3 LOO rms/max 1.8224575627851405e-05 8.882712880929965e-05 fit rms 7.482587850574295e-06
linear [ 0.07    0.1856  1.28   -0.3712 -0.    ] nonlin [((1, 2), -0.2816), ((1, 1, 1), 0.2304)]

bank 2 base residual worst indices [33 49 63 52 24 44 61 16 45]
base residual correlations [(0.9262408304577631, (2, 3)), (0.5105151742415112, (1, 2, 3)), (0.3985586762756782, (3, 3)), (0.36850154092895343, (0, 1, 2)), (0.3002213683591455, (1, 3, 3)), (0.2833940109939478, (0, 1, 3)), (0.2734909292861866, (1, 2, 2)), (0.2565921649136462, (0, 0, 2))]
pure cubes + pair products LOO rms/max 1.1550879277788937e-05 2.8420305439241203e-05 fit rms 8.492012348874698e-06
linear [ 0.08    0.      0.1914  1.32   -0.3828] nonlin [((2, 2, 2), 0.2376), ((2, 3), -0.2904)]
complete degree 3 LOO rms/max 1.3915304317433123e-05 4.0600483929451366e-05 fit rms 6.916497355660894e-06
linear [ 0.08   -0.      0.1914  1.32   -0.3828] nonlin [((2, 3), -0.2904), ((2, 2, 2), 0.2376)]

bank 3 base residual worst indices [44 66 28 39 56 24 52 17 45]
base residual correlations [(0.8841354303065335, (0, 3)), (0.427301002303464, (0, 2)), (0.38154254551132466, (0, 1, 2)), (0.3802516523862617, (0, 1, 3)), (0.36306509596809144, (1, 3, 3)), (0.3113103443833539, (1, 2, 3)), (0.29388944407521667, (2, 3, 3)), (0.26357657429644266, (2, 2))]
pure cubes + pair products LOO rms/max 1.0449503943453874e-05 3.291523536239939e-05 fit rms 7.752017508261118e-06
linear [-0.07   -0.2262 -0.      0.1131  0.78  ] nonlin [((3, 3, 3), 0.1404), ((0, 3), -0.1716)]
complete degree 3 LOO rms/max 1.198046744822802e-05 4.0589548115489994e-05 fit rms 5.9392897975893725e-06
linear [-0.07   -0.2262  0.      0.1131  0.78  ] nonlin [((0, 3), -0.1716), ((3, 3, 3), 0.1404)]
def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(reaction, format='csc')
    modes = np.asarray(study['source_modes'])
```

### Turn 6 — `execute_in_remote_docker` (`34961 ms` LLM)
- **Rationale**: Fit the compact rosette model from the current sweeps and reuse the sealed-wall forward operator supported by the apparatus trace. Refit outlier screening within each calibration fold and run the public schema in an isolated Python process.
- **Alternatives Weighed**: `Use the full cubic polynomial` (rejected: The sweeps identify one cubic and one pair interaction per bank; the full cubic basis had worse held-out error.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/beta_m98_v2_c04_batch_01_solver.py').read_text()
a=p.index('def _calibrate('); b=p.index('def _box_budget(',a)
p=p[:a]+'''def _rosette_design(ref, j):
    own = ref[:, j]
    return np.column_stack((np.ones(len(ref)), ref[:, (j-1) % 4], own,
                            ref[:, (j+1) % 4], own**3,
                            own * ref[:, (j+1) % 4]))


def _fit_rosette(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
    # Gross row corruption is much larger than the response to any clean sweep.
    peak = np.max(np.abs(raw), axis=1)
    center = np.median(peak)
    keep = peak <= center + 10 * np.median(np.abs(peak - center))
    return np.array([np.linalg.lstsq(_rosette_design(ref[keep], j), raw[keep, j],
                                     rcond=None)[0] for j in range(4)]), keep


def _calibrate(study):
    coef, _ = _fit_rosette(study)
    offset, prev, gain, nxt, cube, pair = coef.T
    j = np.arange(4)
    jm, jp = (j-1) % 4, (j+1) % 4
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    A = np.diag(gain)
    A[j, jm], A[j, jp] = prev, nxt
    ref = np.linalg.solve(A, (obs - offset).T).T
    for _ in range(20):
        residual = offset + prev * ref[:, jm] + gain * ref + nxt * ref[:, jp]
        residual += cube * ref**3 + pair * ref * ref[:, jp] - obs
        if np.max(np.abs(residual)) < 1e-12:
            return ref.ravel()
        J = np.broadcast_to(A, (len(ref), 4, 4)).copy()
        J[:, j, j] += 3 * cube * ref**2 + pair * ref[:, jp]
        J[:, j, jp] += pair * ref
        ref -= np.linalg.solve(J, residual[..., None])[..., 0]
    raise RuntimeError('Rosette calibration inversion did not converge')


'''+p[b:]
Path('/holdout_task/batch_solver.py').write_text(p)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps'];coef,keep=m._fit_rosette(s); print('rejected sweeps',np.flatnonzero(~keep)); x,y=c['ref_rosette'],c['raw_rosette']; rms=[]
for j in range(4):
 errs=[]
 for i in np.flatnonzero(keep):
  train=np.arange(len(x))!=i; sub=dict(s,sensor_calibration_sweeps={'ref_rosette':x[train],'raw_rosette':y[train]}); b,mask=m._fit_rosette(sub); errs.append(float((m._rosette_design(x[i:i+1],j)@b[j])[0]-y[i,j]));
 rms.append(np.sqrt(np.mean(np.square(errs))))
print('nested clean LOOCV RMS by bank',rms); out=m.solve_batch(s); k=out['kkt_certificate']; print('KKT gap', k['primal_objective']-k['dual_lower_bound']);print('weights',out['weights'])
PY
```
- **Container Exit Code**: `0` (`697 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
rejected sweeps [13 14 15]
nested clean LOOCV RMS by bank [1.1225294723310344e-05, 1.0218924639434315e-05, 1.0593075957882614e-05, 8.759404883325962e-06]
KKT gap 4.996003610813204e-16
weights [0.         0.17465858 0.10646654 0.27887488 0.         0.42
 0.42       0.        ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=558df212c18f7e66 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_558df212 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`39181 ms` LLM)
- **Rationale**: Compare archived wall regimes with identical calibration and held-out sensors. Revisit the stencil if the apparatus trace contradicts the sealed-wall assignment. Independently check the certificate arithmetic and terminal PDE before submission.
- **Alternatives Weighed**: `Submit on the schema receipt and calibration validation` (rejected: Neither check distinguishes the archived wall operators or detects an axis-order error in the terminal field.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util as iu,json,pickle,numpy as np
from pathlib import Path

def load(p):
    spec=iu.spec_from_file_location(p.stem,p); mod=iu.module_from_spec(spec); spec.loader.exec_module(mod); return mod

a=Path('/tmp/hegel_continual_archive'); m=load(Path('/holdout_task/batch_solver.py')); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); Z=s['Z_nuisance']; rr=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; B=s['total_budget']
log=json.load(open(a/'archive_apparatus_log.json')); print('log keys',list(log[-1])); t=np.array([e['facility_telemetry_trace'] for e in log]); print('apparatus distances by channel',np.sqrt(np.mean((t[:-1]-t[-1])**2,axis=1)).round(3))
def fit(S,idx):
    z=Z[idx]; A=S[idx]; b=y[idx]; reg=z.T@z+rr*np.eye(z.shape[1]); A=A-z@np.linalg.solve(reg,z.T@A); b=b-z@np.linalg.solve(reg,z.T@b); return m._box_budget(A.T@A+sr*np.eye(S.shape[1]),-A.T@b,cap,B)[0]
ns=len(s['sensor_ij']); rows=np.arange(len(y));
for p in [Path('/holdout_task/batch_solver.py'),*sorted(a.glob('*solver.py'))]:
    S,U=load(p)._forward(s); w=fit(S,rows); e=y-S@w; gam=np.linalg.solve(Z.T@Z+rr*np.eye(Z.shape[1]),Z.T@e); e-=Z@gam; ee=[]
    for i in range(ns):
        test=rows%ns==i; tr=~test; v=fit(S,tr); z=Z[tr]; gamma=np.linalg.solve(z.T@z+rr*np.eye(z.shape[1]),z.T@(y[tr]-S[tr]@v)); ee.extend(y[test]-S[test]@v-Z[test]@gamma)
    print(p.stem,'projected RMS',np.sqrt(np.mean(e**2)),'sensor-out RMS',np.sqrt(np.mean(np.square(ee))),'max',np.max(np.abs(ee)))
out=m.solve_batch(s); w=out['weights']; k=out['kkt_certificate']; S,U=m._forward(s); P=np.linalg.solve(Z.T@Z+rr*np.eye(Z.shape[1]),Z.T); A=S-Z@(P@S); b=y-Z@(P@y); g=A.T@(A@w-b)+sr*w; l=np.asarray(k['lam_lower']); u=np.asarray(k['lam_upper']); nu=k['nu_sum']; J=.5*np.linalg.norm(A@w-b)**2+.5*sr*(w@w); D=J-g@w+nu*B+cap*np.minimum(0,g-nu).sum(); assert np.max(abs(g-nu-l+u))<1e-10 and min(l.min(),u.min())>=0; assert max(abs(w.sum()-B),-w.min(),w.max()-cap,abs(J-k['primal_objective']),abs(D-k['dual_lower_bound']),abs(J-D),np.max(abs(l*w)),np.max(abs(u*(cap-w))))<1e-10; assert np.allclose(k['weights'],w,rtol=0,atol=1e-12)
# Check the terminal output against the sealed-wall PDE, not its output shape alone.
from scipy.sparse import kron,eye,diags
nx,ny=s['nx'],s['ny']; ax=m._axis(s['kappa_x'],s['vel_x'],s['Lx']); ay=m._axis(s['kappa_y'],s['vel_y'],s['Ly']); assert max(np.max(abs(ax.sum(axis=1))),np.max(abs(ay.sum(axis=1))))<1e-10; L=kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')+s['mu_shifts'][-1]); rhs=np.tensordot(w,s['source_modes'],axes=1).ravel(order='F'); err=np.max(abs(L@out['terminal_field'].ravel(order='F')-rhs)); print('terminal PDE max residual',err,'certificate gap',J-D); assert err<1e-10
PY
```
- **Container Exit Code**: `0` (`566 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
log keys ['completed_batch_index', 'completed_short_name', 'facility_telemetry_trace']
apparatus distances by channel [[1.132 2.783 0.745 0.118 2.298 0.666 1.075 2.375]
 [0.812 0.099 0.523 0.109 0.115 0.346 0.787 0.231]
 [0.472 0.095 0.285 2.914 2.304 0.215 0.402 2.346]]
batch_solver projected RMS 0.09314671023921284 sensor-out RMS 0.09884437219764808 max 0.22909522408987676
beta_m98_v2_c04_batch_01_solver projected RMS 0.09314671023921284 sensor-out RMS 0.09884437219764808 max 0.22909522408987676
beta_m98_v2_c04_batch_02_solver projected RMS 0.007339670406905152 sensor-out RMS 0.007929486588840271 max 0.020545641955745958
beta_m98_v2_c04_batch_03_solver projected RMS 0.1059540491294398 sensor-out RMS 0.1112277225917272 max 0.21923653893952322
beta_m98_v2_c04_s00_solver projected RMS 0.007339670406905152 sensor-out RMS 0.007929486588840271 max 0.020545641955745958
terminal PDE max residual 3.9968028886505635e-15 certificate gap 4.996003610813204e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=558df212c18f7e66 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_558df212 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`46539 ms` LLM)
- **Rationale**: Promote the stencil supported by held-out measurements. Check the exact revised artifact in isolation, including shift holdouts, terminal boundary rows, and the full KKT certificate.
- **Alternatives Weighed**: `Fit new wall coefficients or mix wall operators` (rejected: Two archived implementations agree on the Robin prediction, which reduces held-out error by over an order of magnitude using the supplied coefficients.)
- **ATMS NOGOOD DECLARED**: Refuted `The nearest apparatus trace identifies sealed walls as the active boundary condition.` | Witness: `Under identical nonlinear calibration and fold-local nuisance fitting, archived batch 02 and s00 both achieve 0.00793 sensor-out RMS versus 0.09884 for sealed walls and 0.11123 for periodic walls.` | Blanket Diagnosis: `Whole-trace proximity is not a reliable identifier of the wall operator; calibrated spatial measurements discriminate the three archived operators directly.` | Sublated Pivot: `Retain the fitted rosette and constrained inverse solver, replacing only the sealed-wall forward operator with archived Robin walls.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/batch_solver.py'); s=p.read_text(); r=Path('/tmp/hegel_continual_archive/beta_m98_v2_c04_batch_02_solver.py').read_text()
start=r.index('def _axis('); end=r.index('\ndef ',r.index('def _forward(')+1)
assert s[:s.index('def _axis(')].splitlines()[-4:] == r[:start].splitlines()[-4:] or all(name in s[:s.index('def _axis(')] for name in ('diags','eye','kron','splu'))
p.write_text(s[:s.index('def _axis(')]+r[start:end]+'\n'+s[s.index('def _rosette_design('):])
PY
sed -n '/^def _axis/,/^def _rosette_design/p' /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
S,U=m._forward(s); y=m._calibrate(s); Z=s['Z_nuisance']; rr=s['nuisance_ridge']; sr=s['source_ridge']; c=s['u_box']; B=s['total_budget']; n=len(s['sensor_ij']); q=len(s['mu_shifts']); M=S.shape[1]; rows=np.arange(len(y))
def project(A,b,z):
    T=np.linalg.solve(z.T@z+rr*np.eye(z.shape[1]),z.T)
    return A-z@(T@A),b-z@(T@b)
def predict(test):
    tr=~test; a,b=project(S[tr],y[tr],Z[tr]); w,nu=m._box_budget(a.T@a+sr*np.eye(M),-a.T@b,c,B); z=Z[tr]; gamma=np.linalg.solve(z.T@z+rr*np.eye(z.shape[1]),z.T@(y[tr]-S[tr]@w)); return y[test]-S[test]@w-Z[test]@gamma
es=np.array([predict(rows%n==i) for i in range(n)]); et=[predict(rows//n==k) for k in range(q)]
print('sensor-out pooled RMS, worst-sensor RMS, max ABS:',np.sqrt(np.mean(es**2)),np.max(np.sqrt(np.mean(es**2,axis=1))),np.max(abs(es))); print('shift-out RMS by shift:',[np.sqrt(np.mean(e**2)) for e in et]); assert np.sqrt(np.mean(es**2))<0.01
out=m.solve_batch(s); w=out['weights']; k=out['kkt_certificate']; a,b=project(S,y,Z); e=a@w-b; g=a.T@e+sr*w; lo=np.asarray(k['lam_lower']); hi=np.asarray(k['lam_upper']); nu=k['nu_sum']; J=.5*(e@e+sr*(w@w)); D=J-g@w+nu*B+c*np.minimum(0,g-nu).sum()
assert min(lo.min(),hi.min())>=0
kerr=max(abs(w.sum()-B),-w.min(),w.max()-c,abs(J-k['primal_objective']),abs(D-k['dual_lower_bound']),abs(J-D),np.max(abs(g-nu-lo+hi)),np.max(abs(lo*w)),np.max(abs(hi*(c-w))),np.max(abs(np.asarray(k['weights'])-w)))
assert kerr<1e-10
nx,ny=s['nx'],s['ny']; f=out['terminal_field']; dx=s['Lx']/(nx-1);dy=s['Ly']/(ny-1); vx=s['vel_x'];vy=s['vel_y']; kx=s['kappa_x'];ky=s['kappa_y']; rhs=np.tensordot(w,s['source_modes'],axes=1)
def interior(F,k,v,h):
    dl=F[1:-1]-F[:-2]; dr=F[1:-1]-F[2:]; kl=.5*(k[1:-1]+k[:-2]);kr=.5*(k[1:-1]+k[2:]); return (kl[:,None]/h**2+np.maximum(v[1:-1],0)[:,None]/h)*dl+(kr[:,None]/h**2+np.maximum(-v[1:-1],0)[:,None]/h)*dr
Ax=m._axis(kx,vx,s['Lx'],s['robin_x']); Ay=m._axis(ky,vy,s['Ly'],s['robin_y']); xf=Ax@f; yf=(Ay@f.T).T
assert np.max(abs(interior(f,kx,vx,dx)-xf[1:-1]))<1e-10 and np.max(abs(interior(f.T,ky,vy,dy)-yf.T[1:-1]))<1e-10
for F,AF,kap,h,rob in ((f,xf,kx,dx,s['robin_x']),(f.T,yf.T,ky,dy,s['robin_y'])):
    for i,j in ((0,1),(-1,-2)):
        assert np.max(abs(AF[i]-kap[i]/h**2*(F[i]-F[j])-rob/h*F[i]))<1e-10
perr=np.max(abs(xf+yf+(s['V_reaction_2d']+s['mu_shifts'][-1])*f-rhs)); print('terminal PDE max residual',perr,'certificate max error',kerr,'gap',J-D); assert perr<1e-10
PY
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
def _axis(kappa, velocity, length, robin):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(reaction, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    idx = i + nx * j
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[idx])
    return np.vstack(blocks), fields.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _rosette_design(ref, j):
{"schema_valid": true, "errors": []}
sensor-out pooled RMS, worst-sensor RMS, max ABS: 0.00792948658884027 0.012456342285624342 0.020545641955745958
shift-out RMS by shift: [0.006979167448651695, 0.007993685844317672, 0.008779617181490956]
terminal PDE max residual 2.3314683517128287e-15 certificate max error 4.5102810375396984e-17 gap 4.5102810375396984e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf0051defd340a0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1cf0051d (directory).
```

### Turn 9 — `submit_completed_task` (`17123 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, sensor and shift holdouts, and independent checks of the Robin boundary rows, terminal field, and KKT certificate.
- **Alternatives Weighed**: `Retain the sealed-wall operator` (rejected: With the same calibration, its pooled sensor holdout RMS was 0.09884 versus 0.00793 for Robin walls.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6393'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Worst-sensor holdout RMS was 0.01246, and the terminal PDE residual was 2.33e-15.`

### Turn 10 — `execute_in_remote_docker` (`41514 ms` LLM)
- **Rationale**: Test two failure modes that could change the solver: a missing nonlinear calibration term and different wall regimes on the two axes. Use held-out sweeps for the first and held-out sensors with fold-local nuisance fitting for the second. Leave the verified artifact unchanged during the audit.
- **Alternatives Weighed**: `Run relay-mode and balanced-accuracy checks` (rejected: This continuous tomography study has neither relay states nor class labels. Held-out sensor errors stratified by rosette bank can reveal whether pooled error hides a bank-specific failure.)

```bash
set -e
cat >/tmp/m98_margin_audit.py <<'PY'
import importlib.util as iu
import itertools
import pickle
from pathlib import Path

import numpy as np


def load(path):
    spec = iu.spec_from_file_location(path.stem, path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


root = Path('/holdout_task')
m = load(root / 'batch_solver.py')
with (root / 'public_batch.pkl').open('rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
coef, keep = m._fit_rosette(s)
terms = list(itertools.combinations_with_replacement(range(4), 2)) + list(itertools.combinations_with_replacement(range(4), 3))
poly = np.column_stack([np.prod(x[:, t], axis=1) for t in terms])
errs = {name: [] for name in ('compact', 'full_cubic', 'compact_plus_rational')}
for i in np.flatnonzero(keep):
    train = np.arange(len(x)) != i
    sub = dict(s, sensor_calibration_sweeps={'ref_rosette': x[train], 'raw_rosette': raw[train]})
    _, fold_keep = m._fit_rosette(sub)
    rows = np.flatnonzero(train)[fold_keep]
    ee = {name: [] for name in errs}
    for j in range(4):
        A = m._rosette_design(x, j)
        own, prev, nxt = x[:, j], x[:, (j-1)%4], x[:, (j+1)%4]
        R = [p / (1 + a * q*q) for p in (own, own*nxt) for q in (own-nxt, own+nxt, own-prev, own+prev) for a in (.25, 1.)]
        for name, design in [('compact', A), ('full_cubic', np.c_[np.ones(len(x)), x, poly]), ('compact_plus_rational', np.column_stack([A, *R]))]:
            b = np.linalg.lstsq(design[rows], raw[rows, j], rcond=None)[0]
            ee[name].append(design[i] @ b - raw[i,j])
    for name in errs:
        errs[name].append(ee[name])
for name, e in errs.items():
    print('calibration', name, 'worst-bank LOO RMS', np.max(np.sqrt(np.mean(np.square(e), axis=0))), 'worst absolute', np.max(np.abs(e)))

archive = Path('/tmp/hegel_continual_archive')
sealed = load(next(archive.glob('*01_solver.py')))._axis
periodic = load(next(archive.glob('*03_solver.py')))._axis
robin = m._axis
axes = {'R': robin, 'S': lambda k,v,l,r: sealed(k,v,l), 'P': lambda k,v,l,r: periodic(k,v,l)}
y = m._calibrate(s)
Z = s['Z_nuisance']
n = len(s['sensor_ij'])
idx = np.arange(len(y))
rr = s['nuisance_ridge']
sr = s['source_ridge']
prior = np.eye(Z.shape[1])*rr
metrics = []
for xx, yy in itertools.product(axes, repeat=2):
    calls = iter((axes[xx], axes[yy]))
    m._axis = lambda *args: next(calls)(*args)
    S, _ = m._forward(s)
    errors = np.empty_like(y)
    for i in range(n):
        test = idx % n == i
        tr = ~test
        z, A, b = Z[tr], S[tr], y[tr]
        P = np.linalg.solve(z.T @ z + prior, z.T)
        At, bt = A-z@(P@A), b-z@(P@b)
        w, _ = m._box_budget(At.T@At+sr*np.eye(S.shape[1]), -At.T@bt, s['u_box'], s['total_budget'])
        errors[test] = y[test]-S[test]@w-Z[test]@(P@(b-A@w))
    e = errors.reshape(-1,n)
    rms = np.sqrt(np.mean(e**2))
    worst = np.sqrt(np.mean(e**2,axis=0)).max()
    banks = np.asarray(s['row_bank_id']).ravel()
    bank_rms = [np.sqrt(np.mean(errors[banks==i]**2)) for i in np.unique(banks)]
    metrics.append((rms, xx+yy))
    print('walls', xx+yy, 'pooled', rms, 'worst sensor', worst, 'bank RMS', np.round(bank_rms, 5))
m._axis = robin
print('wall ranking', sorted(metrics))
assert np.isfinite(m.solve_batch(s)['terminal_field']).all()
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`816 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration compact worst-bank LOO RMS 1.1225294723310344e-05 worst absolute 3.0278035602737674e-05
calibration full_cubic worst-bank LOO RMS 1.82245756277245e-05 worst absolute 8.882712881108645e-05
calibration compact_plus_rational worst-bank LOO RMS 1.2959604292549184e-05 worst absolute 4.2933745363349374e-05
walls RR pooled 0.00792948658884027 worst sensor 0.012456342285624318 bank RMS [0.00844 0.00684 0.00783 0.0085 ]
walls RS pooled 0.035845149818814745 worst sensor 0.06669522784897053 bank RMS [0.0461  0.03052 0.03089 0.0336 ]
walls RP pooled 0.03336541912232856 worst sensor 0.06251348060136644 bank RMS [0.04277 0.02965 0.02932 0.02974]
walls SR pooled 0.02550821655200669 worst sensor 0.048587226337393256 bank RMS [0.02374 0.03107 0.02425 0.02203]
walls SS pooled 0.09884437219764808 worst sensor 0.1622813307008789 bank RMS [0.10653 0.10839 0.08644 0.09227]
walls SP pooled 0.10664818378675621 worst sensor 0.15444774089595437 bank RMS [0.11055 0.12166 0.10255 0.0892 ]
walls PR pooled 0.027402606813105346 worst sensor 0.060566925522224806 bank RMS [0.0261  0.02655 0.02788 0.02898]
walls PS pooled 0.10446792363276054 worst sensor 0.1737106340299121 bank RMS [0.11437 0.10239 0.09203 0.1078 ]
walls PP pooled 0.11122772259172718 worst sensor 0.1327167409048768 bank RMS [0.1175  0.11122 0.10532 0.11054]
wall ranking [(0.00792948658884027, 'RR'), (0.02550821655200669, 'SR'), (0.027402606813105346, 'PR'), (0.03336541912232856, 'RP'), (0.035845149818814745, 'RS'), (0.09884437219764808, 'SS'), (0.10446792363276054, 'PS'), (0.10664818378675621, 'SP'), (0.11122772259172718, 'PP')]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf0051defd340a0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1cf0051d (directory).
```

### Turn 11 — `submit_completed_task` (`16496 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, sensor and shift holdouts, and independent checks of the terminal field and KKT certificate. The follow-up audit found no better wall or calibration model.
- **Alternatives Weighed**: `Adopt a mixed wall regime` (rejected: Robin walls on both axes had the lowest holdout RMS in every bank; the best mixed regime had pooled RMS of 0.02551 versus 0.00793.); `Expand the calibration basis` (rejected: Neither the full cubic basis nor the tested rational terms improved worst-bank leave-one-out RMS.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6393'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Worst-sensor holdout RMS was 0.01246, and the terminal PDE residual was 2.33e-15.`

### Turn 12 — `execute_in_remote_docker` (`27421 ms` LLM)
- **Rationale**: Compare fold-local nuisance projection with a no-nuisance diagnostic, reporting worst-rosette and per-bank errors so pooled RMS cannot hide a localized failure. Keep the specified estimator unchanged; a large localized discrepancy would warrant inspecting the calibration or row grouping. Relay modes and classification metrics do not apply to this continuous steady-state study.
- **Alternatives Weighed**: `Repeat the polynomial and mixed-wall sweeps` (rejected: Those challengers were already evaluated on measured holdouts. Full-rosette folds test a distinct failure mode by withholding all four coupled channels together.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util as iu
import pickle
import numpy as np

spec = iu.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = iu.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
S, _ = m._forward(s)
y, Z = m._calibrate(s), s['Z_nuisance']
n = len(s['sensor_ij'])
assert n % 4 == 0 and len(y) % n == 0
site = np.arange(len(y)) % n // 4
ridge = s['source_ridge'] * np.eye(S.shape[1])
prior = s['nuisance_ridge'] * np.eye(Z.shape[1])
bank = np.asarray(s['row_bank_id']).ravel()
for use_nuisance in (False, True):
    residual = np.empty_like(y)
    for i in np.unique(site):
        test = site == i
        tr = ~test
        A, b, z = S[tr], y[tr], Z[tr]
        if use_nuisance:
            P = np.linalg.solve(z.T @ z + prior, z.T)
            At, bt = A - z @ (P @ A), b - z @ (P @ b)
        else:
            At, bt = A, b
        w, _ = m._box_budget(At.T @ At + ridge, -At.T @ bt, s['u_box'], s['total_budget'])
        residual[test] = y[test] - S[test] @ w
        if use_nuisance:
            residual[test] -= Z[test] @ (P @ (b - A @ w))
    rms = lambda e: np.sqrt(np.mean(e**2))
    group_rms = [rms(residual[site == i]) for i in np.unique(site)]
    bank_rms = [rms(residual[bank == i]) for i in np.unique(bank)]
    print('nuisance', use_nuisance, 'whole-rosette holdout RMS', rms(residual), 'worst rosette', max(group_rms), 'bank RMS', bank_rms)
    assert np.isfinite(residual).all()
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nuisance False whole-rosette holdout RMS 0.12305911573300277 worst rosette 0.17768794025143145 bank RMS [0.14239374877544955, 0.10410149911445442, 0.1176825823776404, 0.12494755642541223]
nuisance True whole-rosette holdout RMS 0.00793718049185523 worst rosette 0.010319387096278193 bank RMS [0.008330407673142106, 0.00698993028472668, 0.007791634728121849, 0.008545814619952293]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf0051defd340a0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1cf0051d (directory).
```

### Turn 13 — `submit_completed_task` (`14753 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated schema and independent scientific validation. Whole-rosette holdouts confirmed that nuisance projection improves each bank, and neither mixed wall regimes nor richer calibration improved the relevant holdouts.
- **Alternatives Weighed**: `Remove nuisance projection` (rejected: Omitting it violates the specified objective and increased held-out error in all four banks.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and independent checks of the terminal field and KKT certificate passed. Worst held-out sensor RMS was 0.01246; worst held-out rosette RMS was 0.01032.`

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
