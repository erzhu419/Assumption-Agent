# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_alpha_m100_02_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `10 / 16` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `242,100` (`233,050` / `43,520` / `9,050`) |
| **Total Episode Wall Time (s)** | `283.7s` |

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

### Turn 1 — `execute_in_remote_docker` (`16734 ms` LLM)
- **Rationale**: Inspect the public input layout and telemetry before choosing a calibration or chamber model. This detects missing chamber evidence or incompatible archive models, either of which would change the prediction strategy.

```bash
ls -lh /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference-panel alignment imputation' 2>&1; python3 - <<'PY'
import pickle, numpy as np, json, pathlib
p=pathlib.Path('/tmp/hegel_continual_archive')
for f in sorted(p.glob('*.json')):
 a=json.loads(f.read_text()); print(f.name, type(a).__name__,list(a) if isinstance(a,dict) else len(a)); print(str(a)[-6500:])
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
print('items:',len(x));
def show(d,depth=0):
 for k,v in d.items():
  if isinstance(v,dict):
   print(' '*depth,k); show(v,depth+2)
  else:
   print(' '*depth,k,type(v).__name__,getattr(v,'shape',''),str(v)[:240])
show(x[0])
PY
```
- **Container Exit Code**: `0` (`461 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.28589324 1.33369478 
     instant_rate_obs ndarray (72,) [5.70387784 1.05116913 2.23266634 3.34863804 2.06565493 2.97111073
 5.52661905 2.12468393 3.27695907 3.16782283 4.3933189  4.35728216
 3.04639405 2.84626909 4.61152953 3.03604332 4.62056025 2.46866253
 3.96007074 2.26672938 4.54438269 3.217
     aux_chamber_telemetry ndarray (72, 4) [[ 5.21554419e-01 -2.28432892e+00 -8.85371188e-02 -3.29044921e-01]
 [-7.84641063e-01 -2.18685153e-01  1.67426127e-02 -1.62297452e+00]
 [-7.24453047e-01 -2.64732841e-03  9.87010785e-01  3.74048274e-01]
 [ 4.32821388e-01  4.03712495e-01 -4.61
   test
     lab_id ndarray (36,) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 2 2 2 0 2 3 1 2 2 1 2 0 1 2 2 1 0 3 1 1]
     stratum ndarray (36,) [0 1 1 1 2 2 1 1 0 2 0 0 1 1 2 1 1 2 0 1 2 1 2 2 0 2 0 2 1 2 0 0 1 1 2 0]
     coupon_features_raw ndarray (36, 4) [[-0.00975714  0.11101017 -2.53652501  0.62329863]
 [-0.45784303 -0.46880567  1.81532274  0.30001355]
 [ 1.25645904  0.38995308 -0.24857917 -0.9441847 ]
 [-0.39238823  1.07520728  1.42658145  0.81928091]
 [ 1.54714568 -0.35930813 -1.1059740
     ref_coupons ndarray (36, 8) [[ 0.45029108 -0.93316709 -1.08166944 -0.27470202 -0.6630066  -0.26907788
  -0.76160081 -1.35646814]
 [ 1.20746752  2.09365987 -0.17223505  0.65035423  0.15028806  0.26851883
   0.3476777  -0.35025694]
 [ 2.70687716 -1.06220962  1.06832207 
     past_channel_raw ndarray (36, 6, 4) [[[-5.45577887e-01  6.20815832e-01 -5.07228421e-01  1.08656874e-01]
  [ 5.37997347e-01  2.05913888e-01 -3.70276978e-01  7.45956154e-01]
  [-5.30286971e-02  7.10695601e-02  2.34015076e-01  6.48560870e-01]
  [-1.06122585e-01 -5.22349698e-01  
     hold_channel_raw ndarray (36, 4) [[0.90380949 0.72229631 1.16359745 1.22780888]
 [0.90357257 1.32632957 1.21555648 0.55849845]
 [0.82729339 0.82086044 0.8646408  0.44075614]
 [1.27150885 0.44345433 0.67396919 0.72303542]
 [0.6163333  1.28257851 1.29256372 0.5461358 ]
 [0.6
     strain_0 ndarray (36,) [ 0.03427167  0.09932875  0.06646617  0.1326844   0.03243395 -0.11578171
 -0.111825    0.09911531  0.21655408  0.15599798  0.09368119  0.20170367
  0.08745256 -0.03427642  0.24209258 -0.05972408 -0.00722117  0.20554408
  0.14935419 -0.10252
     aux_chamber_telemetry ndarray (36, 4) [[ 0.42493858  0.20353722  1.35814875  1.99351537]
 [-1.23408442  1.46629526 -0.98456081  2.08595115]
 [-0.96125058 -1.22753474  0.63245482 -1.28064501]
 [ 0.08877898  0.17494145  1.50396815 -1.06750006]
 [-1.76194059  0.37883011  0.5031181
 campaign_diagnostics NoneType  None
```

### Turn 2 — `execute_in_remote_docker` (`18896 ms` LLM)
- **Rationale**: Read the compact telemetry and calibration ranges to determine whether deployment needs a chamber correction and whether calibration must extrapolate over the load history.
- **Alternatives Weighed**: `Use the benchtop law for deployment immediately` (rejected: The chamber labels are absent, but the apparatus log has not yet been evaluated for evidence of a deployment shift.)

```bash
cat /tmp/hegel_continual_archive/archive_apparatus_log.json; cat /holdout_task/public_asset_index.json; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
for k,v in p.items():
 if isinstance(v,dict):
  print(k,[(j,np.shape(w) if not isinstance(w,dict) else list(w)) for j,w in v.items()]); continue
 print(k,np.shape(v),str(v)[:400])
c=p['calibration_load_cells']; x=np.asarray(c['bridge_raw']); y=np.asarray(c['true_channel_loads']); print('sweeps',str(c['single_channel_sweeps'])[:550]); print('calibration range',x.min(0),x.max(0),y.min(0),y.max(0))
for name in ('train','test'):
 s=p[name]; print(name,'history raw ranges',np.min(s['past_channel_raw'],axis=(0,1)),np.max(s['past_channel_raw'],axis=(0,1)),'missing features',np.isnan(s['coupon_features_raw']).sum(0))
PY
```
- **Container Exit Code**: `0` (`438 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ray"
              },
              "strain_curves_obs": {
                "dtype": "float64",
                "shape": [
                  72,
                  12
                ],
                "type": "ndarray"
              },
              "stratum": {
                "dtype": "int64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 3970
    }
  },
  "schema": "hegel_public_asset_index_v1"
}past_dt (6,) [0.35 0.35 0.4  0.45 0.5  0.55]
eval_times (12,) [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
calibration_load_cells [('bridge_raw', (80, 4)), ('true_channel_loads', (80, 4)), ('single_channel_sweeps', ['lab_id', 'cmd_stress', 'true_stress'])]
train [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 8)), ('past_channel_raw', (72, 6, 4)), ('hold_channel_raw', (72, 4)), ('strain_0', (72,)), ('strain_curves_obs', (72, 12)), ('instant_rate_obs', (72,)), ('aux_chamber_telemetry', (72, 4))]
test [('lab_id', (36,)), ('stratum', (36,)), ('coupon_features_raw', (36, 4)), ('ref_coupons', (36, 8)), ('past_channel_raw', (36, 6, 4)), ('hold_channel_raw', (36, 4)), ('strain_0', (36,)), ('aux_chamber_telemetry', (36, 4))]
sweeps {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'cmd_stress': array([-1.59998503, -1.30894168, -1.01793862, -0.72737324, -0.43638198,
       -0.14544952,  0.14547356,  0.43618129,  0.72724567,  1.0182288 ,
        1.30916075,  1.60006869, -1.6000865 , -1.30901453, -1.01822875,
       -0.72727411, -0.43635615, -0.14536266,  0.14547112,  0.43647851,
        0.72738011,  1.01810978,  1.30915463,  1.6000366 , -1.60008846
calibration range [-1.5998429  -1.59987322 -1.59995364 -1.59980361] [1.69992913 1.6999724  1.7000319  1.69991724] [-1.6 -1.6 -1.6 -1.6] [1.7 1.7 1.7 1.7]
train history raw ranges [-0.64403286 -0.6438138  -0.6488514  -0.64239507] [1.1465727  1.13871422 1.14133173 1.13020278] missing features [3 3 3 3]
test history raw ranges [-0.64768326 -0.64475885 -0.64781194 -0.64243676] [1.14518195 1.14987561 1.14725012 1.14247182] missing features [3 3 1 1]
```

### Turn 3 — `execute_in_remote_docker` (`22735 ms` LLM)
- **Rationale**: Read the telemetry metadata without truncation to see whether it specifies a chamber correction. Compare held-out bridge errors to choose the lowest-order calibration that the sweeps support.

```bash
python3 - <<'PY'
import json,pickle,numpy as np,importlib.util
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for row in log:
 for k,v in row.items():
  if k=='facility_telemetry_trace':
   a=np.asarray(v);print(k,a.shape,'tail',a[-2:].tolist())
  else: print(k,v)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells'];x=c['bridge_raw'];y=c['true_channel_loads']; s=c['single_channel_sweeps'];print('sweep residual by lab',[(int(i),float(np.std((s['cmd_stress']-s['true_stress'])[s['lab_id']==i]))) for i in np.unique(s['lab_id'])]);print('unique loads in each of first 8',y[:8].tolist()); print('bridge svd',np.linalg.svd(x-x.mean(0),compute_uv=False));print('bridge similarity rms',np.sqrt(np.mean((x-y)**2,axis=0)));print('scipy available',bool(importlib.util.find_spec('scipy')))
for deg in [1,2,3]:
 from itertools import combinations_with_replacement
 A=np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for p in range(1,deg+1) for j in combinations_with_replacement(range(4),p)]); h=(A@np.linalg.pinv(A));e=(y-h@y)/(1-np.diag(h))[:,None];print('degree',deg,'rank',np.linalg.matrix_rank(A),'loo rms by channel',np.sqrt(np.mean(e**2,axis=0)),'loo total max',np.max(abs(e.sum(1))));
for i in range(4):
 r=np.concatenate([d[k]['ref_coupons'][d[k]['lab_id']==i] for k in ('train','test')]); a=np.concatenate([d[k]['coupon_features_raw'][d[k]['lab_id']==i] for k in ('train','test')]);print('lab',i,'ref col means/std',r.mean(0).round(3),r.std(0).round(3),'raw feat moments',np.nanmean(a,axis=0).round(3),np.nanstd(a,axis=0).round(3))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic' 2>&1 || true
```
- **Container Exit Code**: `0` (`412 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
047, 0.73611, 0.17713, -0.42629, 0.72565, 0.7781]]
sweep residual by lab [(0, 0.00010460535013747777), (1, 6.563915761516974e-05), (2, 0.00010657424978615887), (3, 8.196526425827358e-05)]
unique loads in each of first 8 [[-1.6, 0.0, 0.0, 0.0], [-0.6, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [1.7, 0.0, 0.0, 0.0], [0.0, -1.6, 0.0, 0.0], [0.0, -0.6, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0], [0.0, 1.7, 0.0, 0.0]]
bridge svd [8.27687456 7.47092333 7.32529481 6.79749583]
bridge similarity rms [1.00159136e-04 9.61867942e-05 1.16478632e-04 1.12730290e-04]
scipy available True
degree 1 rank 5 loo rms by channel [1.00077927e-04 9.88375522e-05 1.22416069e-04 1.15490783e-04] loo total max 0.0007842401502718427
degree 2 rank 15 loo rms by channel [1.12372763e-04 9.82262311e-05 1.36821409e-04 1.28917957e-04] loo total max 0.0007709393487610322
degree 3 rank 35 loo rms by channel [0.00016721 0.00015342 0.00017766 0.00017493] loo total max 0.0014598836714978178
lab 0 ref col means/std [-1.128 -0.035 -0.504 -0.423 -0.98  -0.348 -0.188 -0.606] [1.343 1.089 0.792 1.491 0.66  1.27  1.273 1.094] raw feat moments [-0.649 -0.502 -1.235 -0.599] [1.109 1.022 1.229 1.242]
lab 1 ref col means/std [0.774 0.657 0.572 0.448 0.719 0.339 0.759 0.759] [0.667 0.796 1.062 0.875 0.923 0.811 0.981 0.828] raw feat moments [0.562 0.474 0.585 0.522] [0.852 0.878 0.916 1.017]
lab 2 ref col means/std [-0.138 -0.566 -0.777 -0.553 -0.013 -0.014 -0.271 -0.553] [1.799 1.319 1.527 1.588 1.321 1.604 1.468 1.677] raw feat moments [-0.509 -0.417 -0.055 -0.164] [1.304 1.085 1.406 1.549]
lab 3 ref col means/std [0.983 0.527 1.071 0.94  0.689 0.817 0.483 0.844] [0.93  1.039 0.713 0.739 0.909 0.903 0.886 0.75 ] raw feat moments [1.079 0.928 0.794 0.814] [1.028 1.05  0.758 0.979]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 4 — `execute_in_remote_docker` (`56079 ms` LLM)
- **Rationale**: Implement the documented joint feature alignment, calibrated load history, and robust material fit. The first isolated schema run detects import, shape, or reference-split invariance failures that must be fixed before predictive stress testing.
- **Alternatives Weighed**: `Fit a higher-degree bridge calibration` (rejected: The affine bridge has lower aggregate leave-one-out channel error; the cubic fit also increases the worst total-load error.); `Infer a chamber correction from the facility trace` (rejected: The trace supplies no mapping from its eight channels to constitutive parameters, and there are no chamber outcomes or prior solvers.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrate the load bridge and fit the two-branch hereditary material law."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    lab, stratum, refs, raw = (
        np.concatenate((train[key], test[key]), axis=0)
        for key in ('lab_id', 'stratum', 'ref_coupons', 'coupon_features_raw')
    )
    x = np.empty(raw.shape, dtype=float)
    for i in np.unique(lab):
        selected = lab == i
        panel = refs[selected]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[selected] = (raw[selected] - center) / scale
    _, cell, counts = np.unique(
        np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True
    )
    weight = 1.0 / counts[cell.ravel(), None]
    observed = ~np.isnan(x)
    denom = np.sum(weight * observed, axis=0)
    mean = np.divide(np.nansum(weight * x, axis=0), denom,
                     out=np.zeros(x.shape[1]), where=denom > 0)
    x = np.where(observed, x, mean)
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _bridge(calibration):
    raw = np.asarray(calibration['bridge_raw'])
    loads = np.asarray(calibration['true_channel_loads'])
    design = np.column_stack((np.ones(len(raw)), raw))
    return np.linalg.lstsq(design, loads, rcond=None)[0]


def _prepare(part, x, calibration):
    def total(raw):
        return (raw @ calibration[1:] + calibration[0]).sum(axis=-1)

    return (total(part['past_channel_raw']), total(part['hold_channel_raw']),
            np.column_stack((np.ones(len(x)), x)))


def _design(log_tau, samples, dt, times, differentiate=False):
    past, hold, f = samples
    tau = np.exp(log_tau)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    a = tail[:, None] / tau
    b = dt[:, None] / tau
    ea = np.exp(-a)
    kernel = ea * -np.expm1(-b)
    gap = hold[:, None] - past @ kernel

    t = times[:, None] / tau
    growth = -np.expm1(-t)
    curves = np.concatenate((gap[:, None, :] * growth,
                              f[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, f))
    if not differentiate:
        return curves, rate
    dgap = -past @ (a * kernel - b * ea * np.exp(-b))
    dcurves = dgap[:, None, :] * growth - gap[:, None, :] * t * np.exp(-t)
    drate = (dgap - gap) / tau
    return curves, rate, dcurves, drate


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _fit(samples, dt, times, strain, rate):
    y = np.concatenate((strain.ravel(), rate))
    n_curve = strain.size
    bounds = (np.r_[np.log([0.05000001, 0.05000001]), [-np.inf] * 7],
              np.r_[np.log([24.999999, 24.999999]), [np.inf] * 7])

    def matrix(u):
        cur, slope = _design(u, samples, dt, times)
        return np.vstack((cur.reshape(-1, 7), slope))

    grid = np.geomspace(0.085, 20.0, 12)
    best = float('inf')
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            u = np.log([fast, slow])
            a = matrix(u)
            theta = np.linalg.lstsq(a, y, rcond=None)[0]
            err = a @ theta - y
            cost = np.mean(np.minimum(err[:n_curve]**2, 0.25**2))
            cost += np.mean(np.minimum(err[n_curve:]**2, 0.25**2))
            if cost < best:
                best, p = cost, np.r_[u, theta]

    def refine(p, scale, loss, f_scale):
        def residual(q):
            return (matrix(q[:2]) @ q[2:] - y) / scale

        def jac(q):
            cur, slope, dc, dr = _design(q[:2], samples, dt, times, True)
            a = np.vstack((cur.reshape(-1, 7), slope))
            d = np.vstack((dc.reshape(-1, 2), dr)) * q[None, 2:4]
            return np.column_stack((d, a)) / scale[:, None]

        return least_squares(residual, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=f_scale, x_scale='jac', max_nfev=175).x

    def scales(p):
        err = matrix(p[:2]) @ p[2:] - y
        return np.maximum([_mad(err[:n_curve]), _mad(err[n_curve:])], 1e-7)

    def expand(s):
        return np.repeat(s, [n_curve, len(rate)])

    p = refine(p, expand(scales(p)), 'soft_l1', 1.5)
    p = refine(p, expand(scales(p)), 'cauchy', 2.5)
    order = np.argsort(p[:2])
    return np.exp(p[order]), np.r_[p[2 + order], p[4:]]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    x_train, x_test = _features(train, test)
    calibration = _bridge(dataset['calibration_load_cells'])
    samples = _prepare(train, x_train, calibration)
    strain = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(samples, dt, times, strain, train['instant_rate_obs'])
    curves, rate = _design(np.log(taus), _prepare(test, x_test, calibration), dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rate @ weights}
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`498 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 0.0, "ref_swap_rate_max_diff": 0.0, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=efea9c890c902ff7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_efea9c89 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`26609 ms` LLM)
- **Rationale**: Hold out entire labs, strata, and load bands to detect unstable relaxation estimates or errors hidden by pooled metrics. Compare the closed-form history to the specified recurrence and the predicted rate to the short-time strain derivative; any disagreement would require fixing the design before submission.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; dt=d['past_dt']; t=d['eval_times']; x,_=m._features(tr,d['test']); b=m._bridge(d['calibration_load_cells']); s=m._prepare(tr,x,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; out=m.solve_batch(d); u=out['estimated_taus']; w=out['hereditary_weights']; print('full fit',u,w)
for label,a in [('lab',tr['lab_id']),('stratum',tr['stratum']),('hold-load tertile',np.digitize(s[1],np.quantile(s[1],[1/3,2/3])))]:
 for g in np.unique(a):
  test=a==g; fit=~test; tau,weights=m._fit(tuple(h[fit] for h in s),dt,t,y[fit],r[fit]); C,R=m._design(np.log(tau),tuple(h[test] for h in s),dt,t); e=C@weights-y[test]; q=R@weights-r[test]; print(label,g,'n',sum(test),'tau',tau.round(4),'strain median/p90 abs',np.quantile(abs(e),[.5,.9]).round(5),'worst horizon p90',round(np.quantile(abs(e),.9,axis=0).max(),5),'rate median/p90 abs',np.quantile(abs(q),[.5,.9]).round(5))
past, hold, f=s; z=np.zeros((len(r),2)); C,R=m._design(np.log(u),s,dt,t)
for h,p in zip(dt,past.T): z=np.exp(-h/u)*z + (-np.expm1(-h/u))*p[:,None]
assert np.allclose(R[:,:2],(hold[:,None]-z)/u,atol=1e-12)
h=1e-6; Z,_=m._design(np.log(u),s,dt,np.array([h])); delta=np.max(abs((Z[:,0,:]@w)/h-R@w)); print('initial derivative pulse check',delta); assert delta<1e-4
for key,value in out.items(): assert np.isfinite(value).all(), key
assert .05 < u[0] < u[1] < 25
PY
```
- **Container Exit Code**: `0` (`639 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full fit [0.5800778  3.68954745] [ 0.59519039  0.55173674  0.34561312  0.165915   -0.17309528  0.15005955
 -0.09888938]
lab 0 n 13 tau [0.5799 3.7007] strain median/p90 abs [0.00938 0.02195] worst horizon p90 0.03406 rate median/p90 abs [0.00626 0.01001]
lab 1 n 20 tau [0.5791 3.6038] strain median/p90 abs [0.00855 0.02254] worst horizon p90 0.05447 rate median/p90 abs [0.00403 0.00942]
lab 2 n 20 tau [0.5819 3.7709] strain median/p90 abs [0.00673 0.01955] worst horizon p90 0.0464 rate median/p90 abs [0.00459 0.00982]
lab 3 n 19 tau [0.5796 3.6976] strain median/p90 abs [0.00817 0.02287] worst horizon p90 0.08016 rate median/p90 abs [0.00671 0.01268]
stratum 0 n 31 tau [0.5806 3.6878] strain median/p90 abs [0.00803 0.02064] worst horizon p90 0.02259 rate median/p90 abs [0.0058  0.01078]
stratum 1 n 22 tau [0.5797 3.6838] strain median/p90 abs [0.00796 0.02036] worst horizon p90 0.02818 rate median/p90 abs [0.00483 0.01372]
stratum 2 n 19 tau [0.5803 3.707 ] strain median/p90 abs [0.00852 0.02457] worst horizon p90 0.07256 rate median/p90 abs [0.0044 0.0096]
hold-load tertile 0 n 24 tau [0.5788 3.5863] strain median/p90 abs [0.00885 0.02343] worst horizon p90 0.18744 rate median/p90 abs [0.00433 0.00999]
hold-load tertile 1 n 24 tau [0.5801 3.6634] strain median/p90 abs [0.0075  0.02046] worst horizon p90 0.02568 rate median/p90 abs [0.00547 0.01109]
hold-load tertile 2 n 24 tau [0.5808 3.7522] strain median/p90 abs [0.00838 0.02236] worst horizon p90 0.19524 rate median/p90 abs [0.00491 0.01379]
initial derivative pulse check 4.0231220239661525e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=efea9c890c902ff7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_efea9c89 (directory).
```

### Turn 6 — `submit_completed_task` (`15607 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and reference-invariance checks. Held-out benchtop fits produced stable relaxation times, and the closed-form history and initial derivative agreed with the specified recurrence and short-time strain check. These tests verify the implementation, not chamber prediction accuracy.
- **Alternatives Weighed**: `Infer a deployment correction from facility telemetry` (rejected: There are no chamber outcomes, prior verified solvers, or documented mappings from facility telemetry to constitutive parameters.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5073'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and swapping pooled reference rows left predictions unchanged. Deployment uses the benchtop law because the available archive does not identify a chamber correction.`

### Turn 7 — `execute_in_remote_docker` (`35309 ms` LLM)
- **Rationale**: Distinguish a systematic horizon error from a few large sensor residuals, and test whether lab offsets, pairwise interactions, or rational drift terms improve held-out strain and rate. Promote a correction only if it addresses a repeatable bottleneck; this audit cannot establish a chamber law without chamber outcomes.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck/HJB operators, or balanced classification accuracy` (rejected: The documented inputs contain no relay state, stochastic control operator, or class labels. Lab effects can instead be tested directly on continuous residuals.)

```bash
cat > /tmp/audit_material.py <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, _ = m._features(tr, te)
s = m._prepare(tr, x, m._bridge(d['calibration_load_cells']))
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]
r = tr['instant_rate_obs']
load = s[1]
history = load - s[0] @ (dt / dt.sum())
lab = tr['lab_id']
lab_terms = (lab[:,None] == np.unique(lab)).astype(float)
aux = [np.c_[v, v*load, v*history] for v in x.T]
extras = {
    'lab-offset': lab_terms,
    'pairwise': np.column_stack(aux + [load*history, (load+history)**2, (load-history)**2]),
    'rational': np.column_stack([v/(1+q*q) for v in [load, history, *x.T] for q in (load, history)]),
}
for name, group in [('lab',lab), ('stratum',tr['stratum']), ('load',np.digitize(load,np.quantile(load,[1/3,2/3])))]:
    for g in np.unique(group):
        h = group == g
        f = ~h
        tau,w = m._fit(tuple(v[f] for v in s), dt,t,y[f],r[f])
        C,R = m._design(np.log(tau),s,dt,t)
        e, q = y-C@w, r-R@w
        ah = np.abs(e[h])
        j = np.argmax(np.quantile(ah,.9,axis=0))
        print(name,g,'horizon',round(float(t[j]),3),'signed med',round(float(np.median(e[h,j])),4),'p90',round(float(np.quantile(ah[:,j],.9)),4),'count >0.1',int((ah[:,j]>.1).sum()))
        print('  median abs strain,rate:',*(round(float(v),5) for v in [np.median(ah),np.median(np.abs(q[h]))]),end='')
        for label,A in extras.items():
            A = A-A[f].mean(0)
            A /= np.maximum(A[f].std(0),1e-7)
            A[:,A[f].std(0)<1e-6] = 0
            sf,sr=max(m._mad(e[f]),1e-7),max(m._mad(q[f]),1e-7)
            B=(A[f,None,:]*t[None,:,None]).reshape(-1,A.shape[1])/sf
            Z=np.vstack([B,A[f]/sr,np.eye(A.shape[1])*50])
            target=np.r_[e[f].ravel()/sf,q[f]/sr,np.zeros(A.shape[1])]
            w1=least_squares(lambda w: Z@w-target,np.zeros(A.shape[1]),jac=lambda w: Z.copy(),loss='soft_l1',max_nfev=50).x
            corr=A[h]@w1
            changed=np.median(np.abs(e[h]-corr[:,None]*t)),np.median(np.abs(q[h]-corr))
            print(' ',label,*(round(float(v),5) for v in changed),end='')
        print()
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_material.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`944 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 horizon 0.932 signed med 0.0074 p90 0.0341 count >0.1 1
  median abs strain,rate: 0.00938 0.00626  lab-offset 0.00937 0.00626  pairwise 0.00925 0.00657  rational 0.00979 0.00617
lab 1 horizon 1.614 signed med -0.0022 p90 0.0545 count >0.1 2
  median abs strain,rate: 0.00855 0.00403  lab-offset 0.00856 0.00402  pairwise 0.00901 0.00365  rational 0.0087 0.00473
lab 2 horizon 2.636 signed med 0.0012 p90 0.0464 count >0.1 2
  median abs strain,rate: 0.00673 0.00459  lab-offset 0.00671 0.0046  pairwise 0.00675 0.00479  rational 0.00684 0.00456
lab 3 horizon 1.273 signed med 0.0007 p90 0.0802 count >0.1 2
  median abs strain,rate: 0.00817 0.00671  lab-offset 0.00817 0.00671  pairwise 0.00848 0.00595  rational 0.00868 0.00527
stratum 0 horizon 3.318 signed med 0.0021 p90 0.0226 count >0.1 2
  median abs strain,rate: 0.00803 0.0058  lab-offset 0.00804 0.00578  pairwise 0.00852 0.00539  rational 0.00822 0.00538
stratum 1 horizon 2.295 signed med -0.0022 p90 0.0282 count >0.1 2
  median abs strain,rate: 0.00796 0.00483  lab-offset 0.00767 0.0052  pairwise 0.008 0.00428  rational 0.00797 0.00508
stratum 2 horizon 2.636 signed med -0.0025 p90 0.0726 count >0.1 2
  median abs strain,rate: 0.00852 0.0044  lab-offset 0.00862 0.00416  pairwise 0.00858 0.00517  rational 0.00912 0.0045
load 0 horizon 2.977 signed med -0.0057 p90 0.1874 count >0.1 3
  median abs strain,rate: 0.00885 0.00433  lab-offset 0.00902 0.00441  pairwise 0.01113 0.00496  rational 0.01054 0.00502
load 1 horizon 3.659 signed med 0.0009 p90 0.0257 count >0.1 2
  median abs strain,rate: 0.0075 0.00547  lab-offset 0.00736 0.00529  pairwise 0.00772 0.00513  rational 0.00769 0.0059
load 2 horizon 1.273 signed med -0.0032 p90 0.1952 count >0.1 3
  median abs strain,rate: 0.00838 0.00491  lab-offset 0.00848 0.00489  pairwise 0.00927 0.0051  rational 0.00837 0.005
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 0.0, "ref_swap_rate_max_diff": 0.0, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=efea9c890c902ff7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_efea9c89 (directory).
```

### Turn 8 — `submit_completed_task` (`15189 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and pooled-reference invariance checks. The history and initial-rate calculations agreed with the specified recurrence and short-time derivative. The residual audit found no repeatable improvement that justified replacing the verified model.
- **Alternatives Weighed**: `Add lab offsets, pairwise interactions, or rational drift terms` (rejected: None consistently improved held-out strain and rate. The worst horizon had three absolute residuals above 0.1 but a signed median residual of only −0.0032, which does not support a systematic drift correction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5073'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and swapping reference rows between splits left predictions unchanged. Deployment uses the benchtop law because no chamber measurements were available to identify or validate a correction.`

### Turn 9 — `execute_in_remote_docker` (`36739 ms` LLM)
- **Rationale**: The previous audit tested extra drift terms, not whether the robust objective underweights the measured initial rates or whether the fit is stable across different loading histories. Compare alternatives within the same held-out folds and check whether deployment loads extend beyond the observed benchtop range. A consistent improvement would justify changing the objective; unsupported extrapolation would qualify the deployment claim.
- **Alternatives Weighed**: `Repeat the drift-basis or schema audit` (rejected: The unchanged artifact already passed isolated schema verification, and the drift challengers did not consistently improve held-out predictions.)

```bash
cat /holdout_task/public_asset_index.json; cat > /tmp/audit_branch_stability.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
xtr, xte = m._features(tr, te)
b = m._bridge(d['calibration_load_cells'])
s = m._prepare(tr, xtr, b)
s_te = m._prepare(te, xte, b)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
for label, part in [('train', s), ('deployment', s_te)]:
    print(label, 'hold range', np.quantile(part[1], [0, .1, .9, 1]).round(4), 'past range', np.quantile(part[0], [0, .1, .9, 1]).round(4))

def fit_alternative(f, tau, w, loss, rate_weight):
    ss = tuple(v[f] for v in s)
    A, B = m._design(np.log(tau), ss, dt, t)
    sc = max(m._mad(A@w-y[f]), 1e-7)
    sr = max(m._mad(B@w-r[f]), 1e-7)
    def residual(p):
        C,R = m._design(p[:2], ss, dt, t)
        return np.r_[((C@p[2:]-y[f])/sc).ravel(), rate_weight*(R@p[2:]-r[f])/sr]
    p = least_squares(residual, np.r_[np.log(tau),w], bounds=(np.r_[[np.log(.050001)]*2, [-np.inf]*7], np.r_[[np.log(24.9999)]*2,[np.inf]*7]), loss=loss, f_scale=2.5, max_nfev=120).x
    return np.exp(p[:2]), p[2:]

quantiles = np.quantile(s[1], [1/3, 2/3])
groups = [('lab', tr['lab_id']), ('stratum', tr['stratum']), ('hold', np.digitize(s[1], quantiles)), ('history', np.digitize(s[1]-s[0]@(dt/dt.sum()), np.quantile(s[1]-s[0]@(dt/dt.sum()), [1/3,2/3])))]
results = {}
for group, a in groups:
    for k in np.unique(a):
        h = a==k
        f = ~h
        tau,w = m._fit(tuple(v[f] for v in s), dt,t,y[f],r[f])
        fits = {'current': (tau,w), 'soft_l1': fit_alternative(f,tau,w,'soft_l1',1.), 'rate_upweight': fit_alternative(f,tau,w,'cauchy',1.6)}
        for name,(u,v) in fits.items():
            C,R = m._design(np.log(u), tuple(z[h] for z in s),dt,t)
            e,q = abs(C@v-y[h]), abs(R@v-r[h])
            results.setdefault(name,[]).append([np.quantile(e,.9),np.quantile(q,.9),np.quantile(e,.9,axis=0).max(),*np.sort(u)])
        print(group, k, 'baseline taus', np.round(tau,4), 'baseline p90 strain/rate', np.round(results['current'][-1][:2],5))
for name,rows in results.items():
    v = np.asarray(rows)
    print(name, 'worst p90 strain/rate/horizon',v[:,:3].max(0).round(5),'tau interval',v[:,3:].min(0).round(4),v[:,3:].max(0).round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_branch_stability.py
```
- **Container Exit Code**: `0` (`798 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
],
                "type": "ndarray"
              },
              "strain_curves_obs": {
                "dtype": "float64",
                "shape": [
                  72,
                  12
                ],
                "type": "ndarray"
              },
              "stratum": {
                "dtype": "int64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 3970
    }
  },
  "schema": "hegel_public_asset_index_v1"
}train hold range [1.8123 2.6002 4.1741 4.4008] past range [-2.1861 -0.4382  2.3553  3.6077]
deployment hold range [2.6502 2.8798 4.0108 4.4906] past range [-1.7465 -0.462   2.2148  3.7344]
lab 0 baseline taus [0.5799 3.7007] baseline p90 strain/rate [0.02195 0.01001]
lab 1 baseline taus [0.5791 3.6038] baseline p90 strain/rate [0.02254 0.00942]
lab 2 baseline taus [0.5819 3.7709] baseline p90 strain/rate [0.01955 0.00982]
lab 3 baseline taus [0.5796 3.6976] baseline p90 strain/rate [0.02287 0.01268]
stratum 0 baseline taus [0.5806 3.6878] baseline p90 strain/rate [0.02064 0.01078]
stratum 1 baseline taus [0.5797 3.6838] baseline p90 strain/rate [0.02036 0.01372]
stratum 2 baseline taus [0.5803 3.707 ] baseline p90 strain/rate [0.02457 0.0096 ]
hold 0 baseline taus [0.5788 3.5863] baseline p90 strain/rate [0.02343 0.00999]
hold 1 baseline taus [0.5801 3.6634] baseline p90 strain/rate [0.02046 0.01109]
hold 2 baseline taus [0.5808 3.7522] baseline p90 strain/rate [0.02236 0.01379]
history 0 baseline taus [0.5808 3.7186] baseline p90 strain/rate [0.02373 0.00977]
history 1 baseline taus [0.5776 3.583 ] baseline p90 strain/rate [0.02012 0.0121 ]
history 2 baseline taus [0.5822 3.7683] baseline p90 strain/rate [0.02169 0.01249]
current worst p90 strain/rate/horizon [0.02457 0.01379 0.19524] tau interval [0.5776 3.583 ] [0.5822 3.7709]
soft_l1 worst p90 strain/rate/horizon [0.02568 0.01355 0.19548] tau interval [0.5764 3.5241] [0.5827 3.7595]
rate_upweight worst p90 strain/rate/horizon [0.02478 0.01421 0.19516] tau interval [0.5772 3.5415] [0.5817 3.7683]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=efea9c890c902ff7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_efea9c89 (directory).
```

### Turn 10 — `submit_completed_task` (`19550 ms` LLM)
- **Rationale**: The verified artifact passed isolated schema and reference-invariance checks. The closed-form history matched the specified recurrence, and the initial rate matched the short-time strain derivative. Held-out lab, stratum, load, and history slices supported retaining the current fit. These checks validate the benchtop model, not chamber accuracy.
- **Alternatives Weighed**: `Replace Cauchy loss with soft-L1 or upweight initial rates` (rejected: Soft-L1 slightly improved the worst rate error but worsened strain error and broadened the held-out relaxation-time range. Upweighting rates worsened both slice-wide errors.); `Add lab offsets, pairwise interactions, or rational drift terms` (rejected: None consistently improved held-out strain and rate. The worst horizon had three large residuals but a signed median of only −0.0032.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and swapping reference rows between splits left predictions unchanged. The solver uses the benchtop law for blind deployment because the available data do not identify a chamber correction.`

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
