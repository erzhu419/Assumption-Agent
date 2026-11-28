# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_alpha_m100_01_batch_08__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `12 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `339,084` (`328,653` / `114,944` / `10,431`) |
| **Total Episode Wall Time (s)** | `344.76s` |

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

### Turn 1 — `execute_in_remote_docker` (`19364 ms` LLM)
- **Rationale**: Inspect the upstream interface, public schema, and available telemetry before selecting a chamber law. This establishes which measurements are already calibrated and whether the current campaign supports direct validation or requires transfer from earlier campaigns.
- **Alternatives Weighed**: `Use the benchtop law unchanged in deployment` (rejected: Inspect campaign evidence first to determine whether a chamber correction is warranted.)

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py; ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS ridge nuisance profiling' ; cd /holdout_task && python3 - <<'PY'
import json,pickle,os
p='/tmp/hegel_continual_archive/'
for f in ['archive_manifest.json','archive_apparatus_log.json','pre_llm_gate_diagnostic.json']:
 if os.path.exists(p+f):
  d=json.load(open(p+f));print('\n',f,type(d).__name__,list(d) if isinstance(d,dict) else len(d));s=json.dumps(d); print(s if len(s)<10000 else s[:3300]+' ... '+s[-5500:])
b=pickle.load(open('public_batch.pkl','rb')); print('batch',len(b));
def schem(d,lev=0):
 if isinstance(d,dict) and lev<3:
  return {k:schem(v,lev+1) for k,v in d.items()}
 if hasattr(d,'shape'):return (str(d.dtype),d.shape)
 if isinstance(d,(list,tuple,dict)):return (type(d).__name__,len(d))
 return str(d)[:120]
print(schem(b[0]));print(open('public_asset_index.json').read()[:5000])
PY
```
- **Container Exit Code**: `0` (`458 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,
                "type": "ndarray"
              },
              "single_channel_sweeps": {
                "cmd_stress": "ndarray",
                "lab_id": "ndarray",
                "true_stress": "ndarray"
              },
              "true_channel_loads": {
                "dtype": "float64",
                "shape": [
                  80,
                  4
                ],
                "type": "ndarray"
              }
            },
            "eval_times": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "past_dt": {
              "dtype": "float64",
              "shape": [
                6
              ],
              "type": "ndarray"
            },
            "test": {
              "aux_chamber_telemetry": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              },
              "coupon_features_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              },
              "hold_channel_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              },
              "lab_id": {
                "dtype": "int64",
                "shape": [
                  36
                ],
                "type": "ndarray"
              },
              "past_channel_raw": {
                "dtype": "float64",
                "shape": [
                  36,
                  6,
                  4
                ],
                "type": "ndarray"
              },
              "ref_coupons": {
                "dtype": "float64",
                "shape": [
                  36,
                  8
                ],
                "type": "ndarray"
              },
              "strain_0": {
                "dtype": "float64",
                "shape": [
                  36
                ],
                "type": "ndarray"
              },
              "stratum": {
                "dtype": "int64",
                "shape": [
                  36
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "aux_chamber_te
```

### Turn 2 — `execute_in_remote_docker` (`20166 ms` LLM)
- **Rationale**: Read compact summaries to establish the calibrated field names and whether the current campaign has direct chamber observations. If it is blind, the archive entries and facility telemetry determine which transfer hypotheses can be tested.
- **Alternatives Weighed**: `Apply the benchtop law unchanged` (rejected: The first inspection was truncated; the availability of chamber evidence has not yet been established.)

```bash
cd /holdout_task; sed -n '/def calibrate_upstream/,$p' _verified_upstream.py; ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS'; python3 - <<'PY'
import json,pickle,pathlib,numpy as np
p=pathlib.Path('/tmp/hegel_continual_archive'); b=pickle.load(open('public_batch.pkl','rb')); print('batch',len(b));
for item in b:
 d=item['dataset']; print('item',list(item),'diag',None if item.get('campaign_diagnostics') is None else [(k,np.shape(v)) for k,v in item['campaign_diagnostics'].items()]); print('dataset',list(d));
 for k in ('train','test'):
  z=d[k]; print(k,[(q,np.shape(v)) for q,v in z.items()]); print('lab',np.unique(z['lab_id'],return_counts=True),'ref rows',z['ref_coupons'][:2]); print('aux',z.get('aux_chamber_telemetry',[])[:4]);
for name in ('archive_manifest','archive_apparatus_log','pre_llm_gate_diagnostic'):
 f=p/(name+'.json');
 if not f.exists(): continue
 d=json.loads(f.read_text()); print(name, type(d).__name__); print(str(d)[:1000]);
 for k,v in (d.items() if isinstance(d,dict) else [('entries',d)]):
  if isinstance(v,list): print(k,len(v),str(v[-3:])[-4700:]);
  elif isinstance(v,dict): print(k,list(v),str(v)[-2000:]);
PY
```
- **Container Exit Code**: `0` (`359 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.42652, 0.74025, 0.76269, -0.78449, -0.52744, 2.52634, 0.68412], [0.75848, -0.44351, 0.79881, 0.74107, -0.67531, -0.85357, 2.45703, 0.37432], [0.83513, -0.39035, 0.316, 0.83416, -0.42257, -0.64484, 2.60363, -0.12537], [0.87471, -0.36706, -0.30591, 0.77499, -0.01876, -0.02148, 2.577, -0.50415], [0.82231, -0.44045, -0.78663, 0.65462, 0.3979, 0.5478, 2.51758, -0.74953]]}, {'completed_campaign_index': 8, 'completed_campaign_short_name': 'dev_alpha_m100_01_batch_07', 'facility_telemetry_trace': [[-0.3415, -0.49263, 0.44837, 0.77335, -0.18381, -0.79829, 2.56892, 0.8226], [-0.07023, -0.28794, -0.2406, 0.68987, -0.52454, -0.74237, 2.44068, 0.8382], [0.03113, -0.3277, -0.78519, 0.41468, -0.77129, -0.21295, 2.47314, 0.6602], [0.34249, -0.37409, -0.81863, 0.29791, -0.80886, 0.47707, 2.46907, 0.34786], [0.47108, -0.30699, -0.40256, 0.09202, -0.66652, 0.85163, 2.63017, -0.1324], [0.65769, -0.43649, 0.24963, -0.14106, -0.28556, 0.72178, 2.70316, -0.55704], [0.71877, -0.47093, 0.73241, -0.40646, 0.20251, 0.21227, 2.66813, -0.76308], [0.84399, -0.38118, 0.80208, -0.52409, 0.54393, -0.45865, 2.55846, -0.79439], [0.85942, -0.35586, 0.40998, -0.74632, 0.76588, -0.77254, 2.47, -0.66113], [0.79527, -0.35621, -0.26987, -0.84541, 0.80911, -0.73484, 2.64243, -0.32006], [0.68364, -0.12848, -0.73682, -0.87565, 0.62033, -0.21938, 2.4295, 0.06064], [0.57138, -0.29831, -0.83406, -0.78296, 0.3089, 0.4231, 2.75268, 0.49679], [0.40278, -0.25986, -0.38797, -0.78175, -0.12767, 0.82034, 2.65201, 0.79455], [0.13996, -0.36311, 0.27283, -0.63403, -0.55348, 0.71864, 2.59924, 0.8679], [-0.07747, -0.48883, 0.78069, -0.44533, -0.82212, 0.19972, 2.48643, 0.69948], [-0.34884, -0.30589, 0.87162, -0.26276, -0.86108, -0.43406, 2.52434, 0.36938], [-0.46047, -0.45461, 0.44847, -0.0535, -0.65143, -0.81815, 2.44768, -0.09764], [-0.65014, -0.37817, -0.21623, 0.20951, -0.30781, -0.7221, 2.46255, -0.55577], [-0.75746, -0.2905, -0.75426, 0.37219, 0.17669, -0.21598, 2.5697, -0.81214], [-0.86656, -0.30277, -0.77162, 0.58824, 0.5395, 0.45354, 2.51195, -0.83331], [-0.84937, -0.3148, -0.46102, 0.72809, 0.86601, 0.86339, 2.44296, -0.70041], [-0.80208, -0.40442, 0.27943, 0.78987, 0.85523, 0.7381, 2.56707, -0.32279], [-0.63944, -0.39589, 0.76623, 0.84468, 0.66811, 0.16872, 2.47336, 0.10532], [-0.52647, -0.22406, 0.82287, 0.84345, 0.33398, -0.43625, 2.51053, 0.54548]]}]
pre_llm_gate_diagnostic dict
{'pre_llm_gate_accepted': False, 'repair_target': 'downstream_module_solve', 'failure_kind': 'scientific_or_numerical'}
```

### Turn 3 — `execute_in_remote_docker` (`23330 ms` LLM)
- **Rationale**: The previous output was truncated. Read compact summaries to determine whether chamber transfer is identifiable and to isolate historical constitutive code worth testing, without dumping raw telemetry.
- **Alternatives Weighed**: `Use the nearest historical telemetry trace to choose a solver` (rejected: Trace similarity is not evidence of a shared constitutive law until the verified archive branches and observable telemetry differences are understood.)

```bash
cd /holdout_task; python3 - <<'PY'
import json,pickle,pathlib,numpy as np
p=pathlib.Path('/tmp/hegel_continual_archive'); b=pickle.load(open('public_batch.pkl','rb')); print('batch size:',len(b));
for it in b:
 d=it['dataset']; q=it.get('campaign_diagnostics'); print('item keys',list(it),'diagnostics',None if q is None else {k:np.shape(v) for k,v in q.items()}); print('dataset keys',list(d)); print('times',d['eval_times'],'dt',d['past_dt']);
 for name in ('train','test'):
  z=d[name]; a=np.asarray(z.get('aux_chamber_telemetry',[])); print(name,{k:np.shape(v) for k,v in z.items()}, 'aux range/mean/std:',(np.round(np.stack([a.min(0),a.max(0),a.mean(0),a.std(0)]),3) if a.size else None),'aux first',a[:2]); print('reference first',np.asarray(z['ref_coupons'])[:2]);
print((p/'archive_manifest.json').read_text()); log=json.loads((p/'archive_apparatus_log.json').read_text()); print('log structure',[(k,type(v).__name__,len(v) if hasattr(v,'__len__') else v) for k,v in log.items()] if isinstance(log,dict) else len(log)); rows=next((v for v in log.values() if isinstance(v,list) and v and isinstance(v[0],dict)),[]) if isinstance(log,dict) else log;
for r in rows:
 a=np.asarray(r.get('facility_telemetry_trace',[])); print({k:v for k,v in r.items() if k!='facility_telemetry_trace'},a.shape,np.round(a.mean(0),3) if a.size else None)
PY
for f in /tmp/hegel_continual_archive/*solver.py; do echo ==== $f; grep -nE '^(def|[A-Z_]+[[:space:]]*=)|chamber|tau|bend|telemetry' "$f" | tail -42; done
```
- **Container Exit Code**: `0` (`364 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
95:    feature_profile = (fractions * taus) @ kernel + _CHAMBER_LINEAR * times
97:    rate = hold * (theta[:2] @ (1.0 / taus)) + score
101:def solve_batch(dataset: dict) -> dict:
107:    taus, theta = _fit(*_loads(train, bridge), features[:n], dt, times,
111:    increments, rate = _chamber(taus, theta, hold, features[n:], times)
112:    return {'estimated_taus': taus, 'hereditary_weights': theta,
==== /tmp/hegel_continual_archive/dev_alpha_m100_01_batch_02_solver.py
1:"""Calibrated material fit and slope-preserving deployment chamber model."""
6:def _features(train, test):
27:def _design(taus, past, hold, x, dt, t):
28:    a = np.exp(-dt[:, None] / taus)
29:    b = -np.expm1(-dt[:, None] / taus)
35:    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
37:    return curve, np.column_stack((gap / taus, drift))
40:def _fit(past, hold, x, dt, t, curves, rates):
46:    def design(logtaus):
47:        c, r = _design(np.exp(logtaus), past, hold, x, dt, t)
50:    def profile(logtaus):
51:        a = design(logtaus)
75:_CHAMBER_REDISTRIBUTION = np.array([0.5731, 0.7722])
78:def _loads(part):
83:def _chamber(taus, theta, past, hold, x, dt, times):
84:    curve, rate = _design(taus, past, hold, x, dt, times)
86:    zero_slope = taus * -np.expm1(-times[:, None] / taus) - times[:, None]
90:def solve_downstream(cal_item: dict) -> dict:
95:    taus, theta = _fit(*_loads(train), x[:n], dt, times,
98:    increments, rates = _chamber(taus, theta, *_loads(test), x[n:], dt, times)
99:    return {'estimated_taus': taus, 'hereditary_weights': theta,
==== /tmp/hegel_continual_archive/dev_alpha_m100_01_s00_solver.py
6:def _features(train, test):
27:def _bridge(cal):
33:def _loads(part, bridge):
38:def _design(taus, past, hold, x, dt, t):
39:    a = np.exp(-dt[:, None] / taus)
40:    b = -np.expm1(-dt[:, None] / taus)
46:    curve = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / taus),
48:    return curve, np.column_stack((gap / taus, drift))
51:def _fit(past, hold, x, dt, t, curves, rates):
57:    def design(logtaus):
58:        c, r = _design(np.exp(logtaus), past, hold, x, dt, t)
61:    def profile(logtaus):
62:        a = design(logtaus)
84:def solve_batch(dataset):
90:    taus, theta = _fit(*_loads(train, bridge), x[:n], dt, t,
93:    # The initialization log has no paired chamber response for this blind batch.
94:    c, r = _design(taus, *_loads(test, bridge), x[n:], dt, t)
95:    return {'estimated_taus': taus, 'hereditary_weights': theta,
```

### Turn 4 — `execute_in_remote_docker` (`30512 ms` LLM)
- **Rationale**: Fit material parameters on calibrated training data, then use chamber curves and initial rates to test whether deployment shares the same time basis and which load-history terms it needs.
- **Alternatives Weighed**: `Transfer a historical chamber correction without refitting` (rejected: This campaign has 24 direct chamber measurements, so archived laws can be tested instead of assumed.)

```bash
cd /holdout_task; cat /tmp/hegel_continual_archive/dev_alpha_m100_01_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation' | grep -iE 'viscoelastic|hereditary' | cut -c1-320; cat >/tmp/probe08.py <<'PY'
import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import _verified_upstream as up
import dev_alpha_m100_01_batch_02_solver as old
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=up.calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=[d[k] for k in ('train','test','chamber_verification_trials')]; t,dt=d['eval_times'],d['past_dt']; n=len(a['lab_id']); x=old._features(a,b); xc=old._features(a,c)[n:]; p,h=old._loads(c)
tau,th=old._fit(*old._loads(a),x[:n],dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']);print('tau theta',tau,th)
Y=c['strain_curves_obs']-c['strain_0'][:,None]; R=c['instant_rate_obs']; K=-np.expm1(-t[:,None]/tau); B=np.c_[K,t]; s=np.r_[1/tau,1.]; cc,rr=old._design(tau,p,h,xc,dt,t); base=cc@th; rbase=rr@th; dr=rr[:,2:]@th[2:]; g=rr[:,:2]*tau; print('baseline curve,rate RMSE',np.sqrt(np.mean((base-Y)**2)),np.sqrt(np.mean((rbase-R)**2)));print('diag missing',np.isnan(c['coupon_features_raw']).sum(0));print('rate comparison: observed,base,reset,drift',np.round(np.c_[R,rbase,h*(th[:2]/tau).sum()+dr,dr],4)); print('curve singular values',np.linalg.svd(Y,compute_uv=False))
def robust(A,y,scale):
 return least_squares(lambda w:A@w-y,np.linalg.lstsq(A,y,rcond=None)[0],loss='cauchy',f_scale=scale).x
print('entry curve/rate diagnostics')
for i in range(len(Y)):
 z=robust(np.r_[B,s[None]*.15],np.r_[Y[i],R[i]*.15],.0005); print(i,np.round(np.r_[h[i],g[i],dr[i],z,(z-np.r_[g[i]*th[:2],dr[i]]),np.median(abs(Y[i]-B@z)),R[i]-s@z],5))
for name,F in [('base',np.c_[np.ones(len(h)),g,h,dr]),('history',np.c_[np.ones(len(h)),p,h,dr]),('full',np.c_[np.ones(len(h)),p,h,xc])]:
 A=np.einsum('nf,kb->nkfb',F,B).reshape(Y.size,-1); Ar=np.einsum('nf,b->nfb',F,s).reshape(len(Y),-1); z=robust(np.r_[A,.15*Ar],np.r_[Y.ravel(),.15*R],.0005); e=(A@z).reshape(Y.shape)-Y; er=Ar@z-R; print(name,'error quantiles',np.quantile(abs(e),[.5,.9,.95,.99,1]),np.quantile(abs(er),[.5,.9,1]),'coef',np.round(z.reshape(-1,3),5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe08.py
```
- **Container Exit Code**: `0` (`1090 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
10394e+00 -2.37870e-01
  7.90000e-04  2.70000e-04]
17 [ 2.49799e+00  1.71114e+00  2.00847e+00  3.18230e-01  2.08361e+00
  2.36296e+00  6.42100e-02  6.84090e-01  1.17158e+00 -2.54020e-01
  6.20000e-04  4.51000e-03]
18 [ 3.26742e+00  2.71095e+00  2.98932e+00  6.68060e-01  2.77912e+00
  3.60167e+00  1.58280e-01  5.61870e-01  1.82846e+00 -5.09790e-01
  1.34000e-03 -1.40600e-02]
19 [ 3.22610e+00  1.65166e+00  2.70212e+00  2.88190e-01  2.67653e+00
  2.72856e+00  4.49600e-02  1.32566e+00  1.12572e+00 -2.43230e-01
  1.24000e-03 -2.30000e-03]
20 [ 2.85579e+00  1.50504e+00  2.30949e+00  4.48340e-01  2.38832e+00
  2.98114e+00  8.10200e-02  1.15737e+00  1.61120e+00 -3.67320e-01
  9.20000e-04  3.40000e-03]
21 [ 2.28148e+00  1.37071e+00  1.94840e+00  7.14000e-03  1.87341e+00
  1.43443e+00  5.25000e-03  7.52320e-01  2.78680e-01 -1.90000e-03
  1.02000e-03 -3.45000e-03]
22 [ 3.56589e+00  2.68470e+00  2.95778e+00  7.51670e-01  3.01322e+00
  4.38341e+00  1.42460e-01  8.17430e-01  2.62892e+00 -6.09210e-01
  8.00000e-04  6.63000e-03]
23 [ 3.07563e+00  2.62713e+00  2.72130e+00  6.37850e-01  2.60262e+00
  3.32308e+00  1.26610e-01  4.53920e-01  1.70887e+00 -5.11240e-01
  8.60000e-04 -5.08000e-03]
base error quantiles [0.0039677  0.06219388 0.08629661 0.12513322 0.15462302] [0.00600581 0.04216719 0.06745934] coef [[-8.77000e-03 -6.87400e-02 -2.27000e-03]
 [ 4.80000e-04 -7.18000e-03  1.31000e-03]
 [-2.18000e-03 -2.17220e-01  3.42600e-02]
 [ 8.20150e-01  7.95750e-01 -3.25400e-02]
 [ 1.52790e-01  2.83075e+00  2.09410e-01]]
history error quantiles [0.00737311 0.04967316 0.0617421  0.12213436 0.15124271] [0.00802447 0.03627414 0.05956626] coef [[ 8.32000e-03  2.27740e-01 -1.23800e-02]
 [-3.15000e-03  6.46000e-03 -1.57000e-03]
 [-4.33000e-03 -8.19000e-03 -3.00000e-05]
 [-3.91000e-03  1.09900e-02 -5.53000e-03]
 [ 6.00000e-04  9.28000e-03  1.54000e-03]
 [ 4.68000e-03 -5.38400e-02  8.03000e-03]
 [-2.64000e-03 -2.80500e-02 -2.59000e-03]
 [ 8.17600e-01  5.42790e-01  9.20000e-04]
 [ 1.46060e-01  2.70325e+00  2.18670e-01]]
full error quantiles [0.00368207 0.04562719 0.06474711 0.1425867  0.18260984] [0.00382125 0.03235735 0.07076698] coef [[ 0.08815  0.24415  0.22134]
 [-0.00338  0.04038 -0.00592]
 [-0.00765  0.15303 -0.01988]
 [-0.00592  0.13087 -0.0218 ]
 [-0.00759  0.17932 -0.02059]
 [ 0.00387 -0.05671  0.00787]
 [-0.00411  0.13775 -0.0237 ]
 [ 0.81778  0.59833 -0.00995]
 [ 0.03     0.51448  0.04428]
 [-0.01722 -0.38737 -0.01885]
 [ 0.02398  0.0707   0.04818]
 [-0.02869 -0.38292 -0.04453]]
```

### Turn 5 — `execute_in_remote_docker` (`22308 ms` LLM)
- **Rationale**: Subtract the physical reset response, then separate the initial-rate constraint from time-dependent redistribution. This tests whether the remaining chamber response tracks the material drift score, and whether reference alignment could explain the largest apparent deviations.
- **Alternatives Weighed**: `Add more unconstrained linear predictors` (rejected: Raw history and individual coupon features did not remove the structured residual. Examine the response after subtracting reset stress before expanding the basis.)

```bash
sed '/print('\''entry curve\/rate diagnostics'\'')/,$d' /tmp/probe08.py >/tmp/law08.py; cat >>/tmp/law08.py <<'PY'
U=h[:,None]*th[:2]; D=Y-U@K.T; S=R-U@(1/tau)
print('taus theta',tau,th); print('baseline errors',np.sqrt(np.mean((base-Y)**2)),np.sqrt(np.mean((rbase-R)**2))); print('holdout stress range',old._loads(b)[1].min(),old._loads(b)[1].max());print('singular after reset',np.linalg.svd(D,compute_uv=False))
print('drift, reset rate diff, curve residual t[0,3,7,11], slope constrained modal amplitudes (excess fast/slow)');
for i in np.argsort(dr):
 f=robust(B[:,:2]-t[:,None]/tau,D[i]-S[i]*t,.002); print(i,np.round(np.r_[dr[i],S[i],D[i,[0,3,7,11]],f],5))
print('missing features',np.argwhere(np.isnan(c['coupon_features_raw'])));print('strain scale',np.median(abs(Y)), 'rate scale',np.median(abs(R)))
print('fitted diagnostic reference shift',np.max(abs(xc-old._features(a,dict(c, **{k:np.concatenate((c[k],b[k]),axis=0) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}))[n:n+len(h)])))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/law08.py | tail -46
```
- **Container Exit Code**: `0` (`569 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9819e-02 1.12372000e-02 1.09875073e-02 8.77300644e-03
 8.22403665e-03 7.47161384e-03 5.96408239e-03 4.61733580e-03]
taus theta [0.58602128 4.60801471] [ 0.81788856  0.59317809  0.38925988  0.19305049 -0.12839013  0.09432126
 -0.15609568]
baseline errors 0.6572225433734117 1.4262432944422587
holdout stress range 1.958106989124795 4.5286720203633095
singular after reset [1.27476405e+01 1.95737054e-02 1.70334211e-02 1.40620812e-02
 1.26148627e-02 1.14711021e-02 1.05827290e-02 8.74019874e-03
 8.19454789e-03 7.27252465e-03 5.94552296e-03 4.23571170e-03]
drift, reset rate diff, curve residual t[0,3,7,11], slope constrained modal amplitudes (excess fast/slow)
6 [-0.2357  -0.27691 -0.07063 -0.28556 -0.50129 -0.68792 -0.03827 -0.68939]
14 [-0.00123 -0.02794 -0.00711 -0.03013 -0.048   -0.06297 -0.00274 -0.11412]
21 [0.00714 0.03205 0.00816 0.02813 0.06192 0.07541 0.0041  0.09796]
15 [0.05998 0.07877 0.01732 0.08155 0.14713 0.19916 0.00635 0.27747]
13 [0.24155 0.22616 0.05437 0.23103 0.40381 0.55801 0.03089 0.58228]
0 [0.25054 0.24178 0.05742 0.24255 0.43876 0.5935  0.02978 0.69454]
8 [0.27045 0.29581 0.06868 0.30039 0.53525 0.73058 0.03693 0.82339]
16 [0.28001 0.27604 0.06418 0.27968 0.49775 0.67309 0.0346  0.79306]
19 [0.28819 0.28425 0.0674  0.28951 0.51134 0.69057 0.0348  0.84528]
17 [0.31823 0.32913 0.0759  0.32874 0.59467 0.80817 0.04702 0.806  ]
10 [0.31848 0.3158  0.07409 0.31601 0.57006 0.77402 0.04003 0.88785]
12 [0.32613 0.32602 0.07879 0.32864 0.58705 0.79491 0.04471 0.86323]
9 [0.35052 0.3508  0.08193 0.35152 0.62965 0.85901 0.04798 0.92765]
11 [0.37945 0.38167 0.0915  0.38618 0.68861 0.93862 0.05157 0.9991 ]
5 [0.4214  0.42552 0.10127 0.43453 0.77068 1.04857 0.05379 1.18235]
20 [0.44834 0.45351 0.10841 0.45911 0.82626 1.12061 0.05671 1.24647]
7 [0.47389 0.46862 0.11068 0.47749 0.85465 1.15731 0.05501 1.38076]
3 [0.54603 0.53901 0.12509 0.54106 0.97452 1.3217  0.07355 1.41369]
4 [0.57695 0.59037 0.14008 0.6034  1.06787 1.45952 0.07064 1.73368]
23 [0.63785 0.59539 0.14114 0.6014  1.07023 1.4631  0.07993 1.57462]
18 [0.66806 0.68735 0.16382 0.6995  1.25228 1.70496 0.08588 1.89657]
2 [0.67168 0.69171 0.1626  0.70052 1.25572 1.7059  0.08701 1.92834]
22 [0.75167 0.80636 0.18833 0.81464 1.46036 1.98672 0.1061  2.16048]
1 [0.91635 0.97803 0.23006 0.98733 1.76404 2.4021  0.12849 2.6468 ]
missing features [[ 5  0]
 [ 8  1]
 [17  1]
 [22  1]
 [23  3]]
strain scale 3.4173408036513626 rate scale 5.111982901237047
fitted diagnostic reference shift 0.2171086653155685
```

### Turn 6 — `execute_in_remote_docker` (`36191 ms` LLM)
- **Rationale**: Test whether reset plus a shared drift profile predicts held-out coupons. Compare reference-panel conventions and nested fits on lab, missing-feature, and extreme-drift slices; use the simplest score that retains the gains.
- **Alternatives Weighed**: `Fit arbitrary history and feature interactions` (rejected: The reset residual is nearly rank one, so first test a shared temporal profile and use blocked folds to see whether additional score terms are warranted.)
- **ATMS NOGOOD DECLARED**: Refuted `The chamber carries forward the benchtop stress-history state without modification.` | Witness: `The calibrated chamber diagnostics give baseline curve/rate RMSE of 0.6572/1.4262; subtracting the reset stress response instead leaves a nearly rank-one residual (singular values 12.7476 and 0.0196).` | Blanket Diagnosis: `The material spectrum is shared, but the chamber boundary changes how load history and coupon drift enter the response.` | Sublated Pivot: `Test a reset boundary with a low-dimensional, slope-constrained drift response before adding nonlinear predictors.`

```bash
sed '/print('\''taus theta'\'',tau,th);/,$d' /tmp/law08.py >/tmp/rank08.py; cat >>/tmp/rank08.py <<'PY'
B0=np.c_[tau*K,t]; s0=np.ones(3)
print('lab/strata',[(np.unique(z['lab_id'],return_counts=True),np.unique(z['stratum'],return_counts=True)) for z in (a,b,c)])
def fit_response(idx):
    def residual(v):
        frac=np.r_[v[:2],1-v[:2].sum()]; amp=v[2:]; return np.r_[((amp[:,None]*(B0@frac)-D[idx])/.002).ravel(), (amp-S[idx])/.008]
    return least_squares(residual,np.r_[.23,.59,S[idx]],loss='soft_l1',max_nfev=75).x[:2]
def fit_score(idx,frac,score):
    profile=B0@np.r_[frac,1-frac.sum()]; y=np.c_[D,S]; f=np.r_[profile,1.]; sig=np.r_[np.full(len(t),.002),.008]; w=f/sig**2; amp=y@w/(f@w)
    p=least_squares(lambda v: (score[idx]@v-amp[idx])/.013,np.linalg.lstsq(score[idx],amp[idx],rcond=None)[0],loss='soft_l1').x
    return profile,p,amp
nc=len(h); missing=np.any(np.isnan(c['coupon_features_raw']),axis=1)
folds=[('LOO '+str(i),np.arange(nc)==i) for i in range(nc)]+[('lab '+str(i),c['lab_id']==i) for i in np.unique(c['lab_id'])]+[('missing',missing),('low drift',dr<np.quantile(dr,.2)),('high drift',dr>np.quantile(dr,.8))]
fx=old._features(a,dict(c,**{k:np.concatenate((b[k],c[k]),axis=0) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}))[n+len(b['lab_id']):];
print('diagnostic drift conventions: max score errors vs amplitude')
for label,xx in [('train+diag',xc),('all',fx)]:
  frac=fit_response(np.arange(nc)); profile,p,amp=fit_score(np.arange(nc),frac,np.c_[np.ones(nc),xx]); raw=th[2]+xx@th[3:]; print(label,'fractions',np.r_[frac,1-frac.sum()],'feature coefficients',p,'raw drift vs fitted amp',np.quantile(abs(amp-raw),[.5,.9,1]),'exact time basis residual',np.quantile(abs(D-amp[:,None]*profile),[.5,.9,1])); print('amp/drift pairs',np.round(np.c_[raw,amp,amp-raw,missing],4).tolist())
  scores={'intrinsic':raw[:,None],'affine':np.c_[np.ones(nc),raw],'quadratic':np.c_[np.ones(nc),raw,raw**2],'full_features':np.c_[np.ones(nc),xx]}
  for kind,F in scores.items():
    es=[]; rr=[]; slices=[]
    for j,(name,m) in enumerate(folds):
      q,coef,_=fit_score(np.where(~m)[0],fit_response(np.where(~m)[0]),F); A=F[m]@coef; E=abs(A[:,None]*q-D[m]); R0=abs(A-S[m]);
      if j<nc: es.extend(E.ravel()); rr.extend(R0)
      else: slices.append((name,round(np.sqrt(np.mean(E**2)),4),round(np.sqrt(np.mean(R0**2)),4)))
    print(kind,'LOO p90/max curve',np.quantile(es,[.9,1]),'rate',np.quantile(rr,[.9,1]),'blocked RMSE',slices)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rank08.py | tail -15
```
- **Container Exit Code**: `0` (`3059 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
494)]
full_features LOO p90/max curve [0.06351542 0.16763309] rate [0.03823661 0.06961781] blocked RMSE [('lab 0', 0.0209, 0.0122), ('lab 1', 0.0212, 0.0123), ('lab 2', 0.0648, 0.0389), ('lab 3', 0.0642, 0.0397), ('missing', 0.0559, 0.0343), ('low drift', 0.0492, 0.0298), ('high drift', 0.0638, 0.0384)]
all fractions [0.22659303 0.58418078 0.18922619] feature coefficients [ 0.39389233  0.20380403 -0.13295868  0.10154978 -0.15884856] raw drift vs fitted amp [0.01227859 0.05041022 0.05914002] exact time basis residual [0.00155164 0.00401756 0.00828595]
amp/drift pairs [[0.2416, 0.2427, 0.0011, 0.0], [0.9333, 0.9798, 0.0465, 0.0], [0.6721, 0.6949, 0.0229, 0.0], [0.5571, 0.5388, -0.0183, 0.0], [0.5896, 0.5937, 0.0041, 0.0], [0.4423, 0.4279, -0.0143, 1.0], [-0.2367, -0.2787, -0.0419, 0.0], [0.4798, 0.4727, -0.0071, 0.0], [0.271, 0.2983, 0.0273, 1.0], [0.349, 0.3492, 0.0002, 0.0], [0.3183, 0.3162, -0.0021, 0.0], [0.3786, 0.3824, 0.0038, 0.0], [0.3222, 0.3243, 0.0022, 0.0], [0.2365, 0.2263, -0.0102, 0.0], [-0.0019, -0.0264, -0.0245, 0.0], [0.0364, 0.0814, 0.045, 0.0], [0.273, 0.2756, 0.0026, 0.0], [0.3054, 0.3292, 0.0238, 1.0], [0.6868, 0.6934, 0.0066, 0.0], [0.288, 0.2837, -0.0043, 0.0], [0.45, 0.4584, 0.0084, 0.0], [-0.0203, 0.0318, 0.0521, 0.0], [0.7498, 0.8089, 0.0591, 1.0], [0.6494, 0.596, -0.0534, 1.0]]
intrinsic LOO p90/max curve [0.08152483 0.15743263] rate [0.0470529  0.06545812] blocked RMSE [('lab 0', 0.0162, 0.0101), ('lab 1', 0.0312, 0.0191), ('lab 2', 0.0501, 0.0297), ('lab 3', 0.0643, 0.0399), ('missing', 0.063, 0.0386), ('low drift', 0.0604, 0.0368), ('high drift', 0.0681, 0.0412)]
affine LOO p90/max curve [0.07980847 0.17293098] rate [0.05371909 0.07177826] blocked RMSE [('lab 0', 0.0243, 0.0157), ('lab 1', 0.0281, 0.0172), ('lab 2', 0.0833, 0.0504), ('lab 3', 0.0818, 0.0506), ('missing', 0.0651, 0.0401), ('low drift', 0.064, 0.039), ('high drift', 0.0685, 0.0414)]
quadratic LOO p90/max curve [0.10851453 0.35502226] rate [0.06732918 0.14113179] blocked RMSE [('lab 0', 0.0258, 0.0166), ('lab 1', 0.0397, 0.0243), ('lab 2', 0.1219, 0.0748), ('lab 3', 0.0818, 0.0506), ('missing', 0.065, 0.0401), ('low drift', 0.1279, 0.079), ('high drift', 0.1211, 0.0736)]
full_features LOO p90/max curve [0.09143683 0.18912108] rate [0.05396292 0.07838049] blocked RMSE [('lab 0', 0.0266, 0.0166), ('lab 1', 0.0296, 0.0172), ('lab 2', 0.0952, 0.0578), ('lab 3', 0.0812, 0.0503), ('missing', 0.0678, 0.0416), ('low drift', 0.0649, 0.0395), ('high drift', 0.0821, 0.0497)]
```

### Turn 7 — `execute_in_remote_docker` (`33210 ms` LLM)
- **Rationale**: Stage the validated reset law and check the batch interface. Compare reference pools on complete diagnostic coupons by lab; if a consistent alignment convention explains the remaining amplitude errors, it can improve deployment without adding an empirical drift correction.
- **Alternatives Weighed**: `Fit an unconstrained chamber feature score` (rejected: It worsened blocked validation without explaining why residuals change with reference pooling.)

```bash
cd /holdout_task; sed -n '1,130p' _verified_upstream.py; cat batch_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel standardization imputation' | head -12; cp /tmp/hegel_continual_archive/dev_alpha_m100_01_batch_02_solver.py /tmp/base08.py; sed '/^_CHAMBER_REDISTRIBUTION/,$d' /tmp/base08.py > downstream_solver.py; cat >> downstream_solver.py <<'PY'
def _loads(part):
    return (np.sum(part['past_channel_raw'], axis=-1),
            np.sum(part['hold_channel_raw'], axis=-1))


def _chamber(taus, theta, hold, x, times):
    kernel = -np.expm1(-times[:, None] / taus)
    frac = np.array([0.22659303, 0.58418078])
    profile = kernel @ (frac * taus) + (1 - frac.sum()) * times
    score = theta[2] + x @ theta[3:]
    return (hold[:, None] * (kernel @ theta[:2]) + score[:, None] * profile,
            hold * np.sum(theta[:2] / taus) + score)


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    times, dt = cal_item['eval_times'], cal_item['past_dt']
    n = len(train['lab_id'])
    x = _features(train, test)
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, theta = _fit(*_loads(train), x[:n], dt, times, y, train['instant_rate_obs'])
    curves, rates = _chamber(taus, theta, _loads(test)[1], x[n:], times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curves,
            'initial_rate_pred': rates}
PY
sed '/print('\''lab\/strata'\''/d; /print('\''diagnostic drift conventions/,$d' /tmp/rank08.py >/tmp/ref08.py; cat >>/tmp/ref08.py <<'PY'
frac=fit_response(np.arange(nc)); _,_,amp=fit_score(np.arange(nc),frac,np.c_[np.ones(nc),xc]);
for l in np.unique(c['lab_id']):
  m=c['lab_id']==l; print('lab',l,'raw slope/intercept from complete cases', np.polyfit(c['coupon_features_raw'][m&~missing]@th[3:],amp[m&~missing],1));
  for name,parts in [('tr',[a]),('te',[b]),('diag',[c]),('tr+te',[a,b]),('tr+diag',[a,c]),('te+diag',[b,c]),('all',[a,b,c])]:
    panel=np.concatenate([z['ref_coupons'][z['lab_id']==l] for z in parts]); mu=np.median(panel.mean(0)); sd=np.median(panel.std(0)); err=amp[m&~missing]-(th[2]+((c['coupon_features_raw'][m&~missing]-mu)/sd)@th[3:]); print(name,np.round([mu,sd,np.sqrt(np.mean(err**2)),err.mean()],5))
print('sum feature weights',th[3:].sum())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/ref08.py | tail -33
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`833 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
| None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
lab 0 raw slope/intercept from complete cases [0.88243177 0.39376458]
tr [-0.46021  1.15343  0.00509  0.00392]
te [-0.79638  0.89907  0.04792 -0.00643]
diag [-0.80592  1.14103  0.00303  0.00269]
tr+te [-0.47845  1.11206  0.00441  0.00265]
tr+diag [-0.51412  1.22036  0.01418  0.00559]
te+diag [-8.34570e-01  1.07428e+00  1.00400e-02  5.00000e-04]
all [-0.59621  1.14449  0.00383  0.00332]
lab 1 raw slope/intercept from complete cases [1.41565933 0.38976377]
tr [0.53382 0.82176 0.04428 0.01834]
te [0.61079 0.72316 0.00871 0.00558]
diag [0.46878 0.71147 0.00386 0.00322]
tr+te [0.58266 0.78899 0.03339 0.01453]
tr+diag [0.52612 0.78818 0.03301 0.01422]
te+diag [0.55717 0.73577 0.01356 0.00723]
all [0.5897  0.76334 0.02418 0.01121]
lab 2 raw slope/intercept from complete cases [0.770622   0.39289857]
tr [-0.39016  1.26364  0.01196  0.008  ]
te [-0.48823  0.99277  0.11896  0.06209]
diag [-6.39020e-01  1.30609e+00  2.36000e-03  9.70000e-04]
tr+te [-0.3861   1.29691  0.00291  0.00289]
tr+diag [-0.36501  1.39207  0.02468 -0.01034]
te+diag [-0.50259  1.29013  0.00414  0.00365]
all [-0.34006  1.3899   0.0241  -0.01001]
lab 3 raw slope/intercept from complete cases [1.03820038 0.38945664]
tr [0.86806 0.82389 0.03936 0.01813]
te [0.85484 0.70822 0.08236 0.03538]
diag [0.78812 0.96887 0.00244 0.00203]
tr+te [0.8313  0.76956 0.05788 0.02547]
tr+diag [0.79441 0.90768 0.01503 0.00811]
te+diag [0.84048 0.82859 0.03782 0.01743]
all [0.83785 0.84726 0.03205 0.0151 ]
sum feature weights 0.0028859410007626984
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 3.62e-08, "ref_swap_rate_max_diff": 8e-10, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d03f6f8e10b1beb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d03f6f8 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`23321 ms` LLM)
- **Rationale**: Test whether diagnostic-only reference alignment removes the apparent drift-score error, including on missing-feature coupons, while leaving the required runtime pooling unchanged. This distinguishes a validation coordinate mismatch from a missing chamber term.
- **Alternatives Weighed**: `Switch runtime deployment to test-only reference alignment` (rejected: The runtime contract requires train/test pooling, and the staged solver passes the reference-row swap check. The open question is the alignment of the separate diagnostic cohort.)

```bash
sed '/print('\''diagnostic drift conventions/,$d' /tmp/rank08.py >/tmp/diagonly08.py; cat >>/tmp/diagonly08.py <<'PY'
import downstream_solver as dn
for name,xx in [('diag only',dn._features(c,c)[:nc]),('train+diag',xc),('train+test+diag',fx)]:
 raw=th[2]+xx@th[3:]; out=[]
 for label,m in folds:
  frac=fit_response(np.flatnonzero(~m)); q=B0@np.r_[frac,1-frac.sum()]; e=raw[m,None]*q-D[m]; er=raw[m]-S[m]; out.append((label,e,er))
 print(name,'OOF RMSE',*[round(np.sqrt(np.mean(np.concatenate([z[j].ravel() for z in out[:nc]])**2)),6) for j in (1,2)],'blocked',[(k,round(np.sqrt(np.mean(e**2)),6),round(np.sqrt(np.mean(er**2)),6)) for k,e,er in out[nc:]])
print('all-fit fractions',fit_response(np.arange(nc)))
for name,m in [('complete',~missing),('missing',missing)]:
 x0=dn._features(c,c)[:nc]; dr0=th[2]+x0@th[3:]; frac=fit_response(np.arange(nc)); q=B0@np.r_[frac,1-frac.sum()]; print(name,'curve RMS by time',np.round(np.sqrt(np.mean((D[m]-dr0[m,None]*q)**2,axis=0)),6),'rate differences',np.round(S[m]-dr0[m],6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/diagonly08.py | tail -8; sed -n '1,155p' /holdout_task/validate_public_schema.py; sed -n '63,125p' /holdout_task/downstream_solver.py
```
- **Container Exit Code**: `0` (`1332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_invariance(mod.solve_batch, ds, base_out=out)
            if not coupon["valid"]:
                errs.append("pooled_reference_coupon_invariance_failed")
        except (KeyError, TypeError, ValueError, IndexError):
            errs.append("invalid_output_on_reference_exchange")
print(json.dumps({"schema_valid": not errs, "errors": errs, "pooled_reference_check": coupon}, allow_nan=False))
sys.exit(1 if errs else 0)
            return max(1e-8, 1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else 1.0
        scales = np.r_[np.full(nc, scale(err[:nc])), np.full(len(y)-nc, scale(err[nc:]))]
        p = least_squares(lambda v: (design(v[:2]) @ v[2:] - y) / scales,
                          p, bounds=(np.r_[lo, lo, np.full(7, -np.inf)],
                                     np.r_[hi, hi, np.full(7, np.inf)]),
                          loss='soft_l1', f_scale=1.5, max_nfev=120).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


# Diagnostic leave-one-lab-out selects a two-mode, slope-preserving
# redistribution of the coupon drift score. Load memory is unchanged.
def _loads(part):
    return (np.sum(part['past_channel_raw'], axis=-1),
            np.sum(part['hold_channel_raw'], axis=-1))


def _chamber(taus, theta, hold, x, times):
    kernel = -np.expm1(-times[:, None] / taus)
    frac = np.array([0.22659303, 0.58418078])
    profile = kernel @ (frac * taus) + (1 - frac.sum()) * times
    score = theta[2] + x @ theta[3:]
    return (hold[:, None] * (kernel @ theta[:2]) + score[:, None] * profile,
            hold * np.sum(theta[:2] / taus) + score)


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    times, dt = cal_item['eval_times'], cal_item['past_dt']
    n = len(train['lab_id'])
    x = _features(train, test)
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, theta = _fit(*_loads(train), x[:n], dt, times, y, train['instant_rate_obs'])
    curves, rates = _chamber(taus, theta, _loads(test)[1], x[n:], times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curves,
            'initial_rate_pred': rates}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d03f6f8e10b1beb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d03f6f8 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`19173 ms` LLM)
- **Rationale**: Recover the blocked metrics omitted from the previous output and audit the staged feature transform and material fit. A mismatch in imputation or duplicated solver definitions would warrant a code change; otherwise retain the pooled deployment transform and validate the remaining errors by slice.
- **Alternatives Weighed**: `Add a chamber-specific correction for missing features` (rejected: First verify that imputation matches the specified cell-balanced estimator; five incomplete diagnostic rows do not justify a separate deployment law.)

```bash
cd /holdout_task; nl -ba downstream_solver.py | head -110; cat batch_solver.py; OPENBLAS_NUM_THREADS=1 python3 /tmp/diagonly08.py | grep 'OOF RMSE'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable projection' | grep -iE 'viscoelastic|hereditary' | cut -c1-260 | head -10
```
- **Container Exit Code**: `0` (`1430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
frac = np.array([0.22659303, 0.58418078])
    83	    profile = kernel @ (frac * taus) + (1 - frac.sum()) * times
    84	    score = theta[2] + x @ theta[3:]
    85	    return (hold[:, None] * (kernel @ theta[:2]) + score[:, None] * profile,
    86	            hold * np.sum(theta[:2] / taus) + score)
    87	
    88	
    89	def solve_downstream(cal_item: dict) -> dict:
    90	    train, test = cal_item['train'], cal_item['test']
    91	    times, dt = cal_item['eval_times'], cal_item['past_dt']
    92	    n = len(train['lab_id'])
    93	    x = _features(train, test)
    94	    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    95	    taus, theta = _fit(*_loads(train), x[:n], dt, times, y, train['instant_rate_obs'])
    96	    curves, rates = _chamber(taus, theta, _loads(test)[1], x[n:], times)
    97	    return {'estimated_taus': taus, 'hereditary_weights': theta,
    98	            'predicted_strain': test['strain_0'][:, None] + curves,
    99	            'initial_rate_pred': rates}
from __future__ import annotations
import inspect, json
import _verified_upstream as _up
import downstream_solver as _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
diag only OOF RMSE 0.014304 0.008872 blocked [('lab 0', 0.014214, 0.009457), ('lab 1', 0.019434, 0.01067), ('lab 2', 0.011886, 0.008061), ('lab 3', 0.01241, 0.007495), ('missing', 0.029264, 0.018977), ('low drift', 0.002999, 0.001254), ('high drift', 0.014548, 0.009169)]
train+diag OOF RMSE 0.04108 0.024304 blocked [('lab 0', 0.027685, 0.014648), ('lab 1', 0.048079, 0.028582), ('lab 2', 0.052581, 0.031166), ('lab 3', 0.033489, 0.020621), ('missing', 0.056328, 0.033385), ('low drift', 0.044344, 0.026916), ('high drift', 0.072237, 0.043293)]
train+test+diag OOF RMSE 0.044681 0.026948 blocked [('lab 0', 0.018839, 0.010336), ('lab 1', 0.03659, 0.021937), ('lab 2', 0.052919, 0.031369), ('lab 3', 0.056434, 0.034801), ('missing', 0.064302, 0.038943), ('low drift', 0.061187, 0.037244), ('high drift', 0.06822, 0.041262)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d03f6f8e10b1beb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d03f6f8 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`33745 ms` LLM)
- **Rationale**: Correct the stale comment and verify the exact final files in an isolated Python process. Material holdouts test for a lab-dependent fit, while the deployment finite difference independently checks that the reported initial rate is the derivative of the predicted strain.
- **Alternatives Weighed**: `Fit a separate correction for missing-feature coupons` (rejected: The five incomplete diagnostic coupons do not justify departing from the required pooled, cell-balanced imputation.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('downstream_solver.py')
s = p.read_text()
s = s.replace('# Diagnostic leave-one-lab-out selects a two-mode, slope-preserving\n# redistribution of the coupon drift score. Load memory is unchanged.\n', '')
s = s.replace('def _chamber(taus, theta, hold, x, times):\n', 'def _chamber(taus, theta, hold, x, times):\n    # The chamber resets load memory. Its drift profile has unit initial slope.\n')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as bs, downstream_solver as dn
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
d = calibrate_upstream(dict(item['dataset'], chamber_verification_trials=item['campaign_diagnostics']))
a,b,c = (d[k] for k in ('train','test','chamber_verification_trials'))
result = bs.solve_batch(item['dataset'])
tau,w = (result[k] for k in ('estimated_taus','hereditary_weights'))
assert .05 < tau[0] < tau[1] < 25
x = dn._features(a,b)[:len(a['lab_id'])]
p,h = dn._loads(a)
dt,t = d['past_dt'],d['eval_times']
y,r = a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']
for lab in np.unique(a['lab_id']):
    m = a['lab_id'] == lab
    tf,wf = dn._fit(p[~m],h[~m],x[~m],dt,t,y[~m],r[~m])
    D,R = dn._design(tf,p[m],h[m],x[m],dt,t)
    print('material holdout lab',lab,'curve abs p90',round(np.percentile(abs(D@wf-y[m]),90),6),'rate RMSE',round(np.sqrt(np.mean((R@wf-r[m])**2)),6),'taus',tf.round(5),'max weight shift',round(np.max(abs(wf-w)),6))
hc = dn._loads(c)[1]
xc = dn._features(c,c)[:len(hc)]
Y = c['strain_curves_obs']-c['strain_0'][:,None]
R = c['instant_rate_obs']
curve,rate = dn._chamber(tau,w,hc,xc,t)
for label,m in [('all',np.ones(len(hc),bool)),('complete',np.isfinite(c['coupon_features_raw']).all(1)),('missing',np.isnan(c['coupon_features_raw']).any(1))]:
    print('fixed chamber diagnostic',label,'curve/rate RMSE',*(round(np.sqrt(np.mean(v[m]**2)),6) for v in (curve-Y,rate-R)))
hb = dn._loads(b)[1]
xb = dn._features(a,b)[len(a['lab_id']):]
eps = 1e-7
small,derivative = dn._chamber(tau,w,hb,xb,np.array([0.,eps]))
assert np.max(abs(small[:,0])) < 1e-12
err = np.max(abs((small[:,1]-small[:,0])/eps-derivative))
assert err < 1e-5
print('deployment initial-rate finite difference max error',err)
PY
```
- **Container Exit Code**: `1` (`418 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4ba349d4b35884de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4ba349d4 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 60, in <module>
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/holdout_task/batch_solver.py", line 3, in <module>
    import _verified_upstream as _up
ModuleNotFoundError: No module named '_verified_upstream'
```

### Turn 11 — `execute_in_remote_docker` (`32975 ms` LLM)
- **Rationale**: Run the public validator as documented, then test the unchanged artifacts in an isolated process with the task directory explicitly on the import path. Material holdouts detect lab-dependent fits; paired chamber comparisons test whether reset improves both outputs on complete and incomplete coupons; the finite difference detects a mismatch between the strain law and reported initial rate.
- **Alternatives Weighed**: `Change production imports to accommodate direct isolated invocation of the validator` (rejected: The documented invocation already resolves sibling modules; the isolated test can explicitly set the module path without changing either deliverable.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
python3 -I - <<'PY'
import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as bs, downstream_solver as dn
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
d = calibrate_upstream(dict(item['dataset'], chamber_verification_trials=item['campaign_diagnostics']))
a,b,c = (d[k] for k in ('train','test','chamber_verification_trials'))
result = bs.solve_batch(item['dataset'])
tau,w = (result[k] for k in ('estimated_taus','hereditary_weights'))
assert .05 < tau[0] < tau[1] < 25
x = dn._features(a,b)
n = len(a['lab_id'])
p,h = dn._loads(a)
dt,t = d['past_dt'],d['eval_times']
y,r = a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']
for lab in np.unique(a['lab_id']):
    m = a['lab_id'] == lab
    tf,wf = dn._fit(p[~m],h[~m],x[:n][~m],dt,t,y[~m],r[~m])
    D,R = dn._design(tf,p[m],h[m],x[:n][m],dt,t)
    print('material holdout lab',lab,'curve abs p90',round(np.percentile(abs(D@wf-y[m]),90),6),'rate RMSE',round(np.sqrt(np.mean((R@wf-r[m])**2)),6),'taus',tf.round(5),'max weight shift',round(np.max(abs(wf-w)),6))
pc,hc = dn._loads(c)
xc = dn._features(c,c)[:len(hc)]
Y = c['strain_curves_obs']-c['strain_0'][:,None]
R = c['instant_rate_obs']
curve,rate = dn._chamber(tau,w,hc,xc,t)
base,base_rate = dn._design(tau,pc,hc,xc,dt,t)
miss = np.isnan(c['coupon_features_raw']).any(1)
for label,m in [('all',np.ones(len(hc),bool)),('complete',~miss),('missing',miss)]:
    rmse = lambda e: float(np.sqrt(np.mean(e[m]**2)))
    old = (rmse(base@w-Y),rmse(base_rate@w-R))
    new = (rmse(curve-Y),rmse(rate-R))
    print('fixed chamber diagnostics',label,'unchanged vs reset strain/rate RMSE',old,new)
    assert all(i<j for i,j in zip(new,old))
hb = dn._loads(b)[1]
eps = 1e-7
small,derivative = dn._chamber(tau,w,hb,x[n:],np.array([0.,eps]))
np.testing.assert_allclose(derivative,result['initial_rate_pred'],rtol=0,atol=1e-10)
np.testing.assert_allclose(small[:,0],0,rtol=0,atol=1e-12)
err = np.max(abs((small[:,1]-small[:,0])/eps-derivative))
assert err < 1e-5
print('deployment initial-rate finite difference max error',err)
PY
```
- **Container Exit Code**: `0` (`1076 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 3.62e-08, "ref_swap_rate_max_diff": 8e-10, "tol": 1e-05}}
material holdout lab 0 curve abs p90 0.020474 rate RMSE 0.00784 taus [0.58562 4.61219] max weight shift 0.001643
material holdout lab 1 curve abs p90 0.022217 rate RMSE 0.007379 taus [0.58672 4.68127] max weight shift 0.003052
material holdout lab 2 curve abs p90 0.02408 rate RMSE 0.004832 taus [0.58551 4.52702] max weight shift 0.005302
material holdout lab 3 curve abs p90 0.024503 rate RMSE 0.007544 taus [0.58694 4.67116] max weight shift 0.003464
fixed chamber diagnostics all unchanged vs reset strain/rate RMSE (0.6644933392405918, 1.4235097234793512) (0.014445453659327659, 0.008871742188865299)
fixed chamber diagnostics complete unchanged vs reset strain/rate RMSE (0.7266889973767222, 1.5125983194156039) (0.005317558814749194, 0.0021574952859696427)
fixed chamber diagnostics missing unchanged vs reset strain/rate RMSE (0.33578935963587725, 1.0160707205787354) (0.029902692022467703, 0.01897654602347843)
deployment initial-rate finite difference max error 5.60916992853322e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4ba349d4b35884de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4ba349d4 (directory).
```

### Turn 12 — `submit_completed_task` (`13299 ms` LLM)
- **Rationale**: The verified candidate improves both chamber outputs on complete and incomplete coupons, preserves pooled reference alignment, and matches the strain derivative to the reported initial rate within 5.61e-7 in isolated Python.
- **Submitted Final Answer / Summary**: `Both solvers are saved and verified. The public schema and pooled-reference invariance checks passed. The chamber reset reduced diagnostic strain/rate RMSE from 0.6645/1.4235 to 0.0144/0.0089.`

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
