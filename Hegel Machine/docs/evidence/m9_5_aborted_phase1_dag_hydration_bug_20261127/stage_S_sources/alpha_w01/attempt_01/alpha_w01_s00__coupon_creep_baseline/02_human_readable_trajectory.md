# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w01_s00__coupon_creep_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w01_source` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `377,832` (`363,049` / `79,104` / `14,783`) |
| **Total Episode Wall Time (s)** | `431.77s` |

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

### Turn 1 — `execute_in_remote_docker` (`17085 ms` LLM)
- **Rationale**: Inspect array shapes, split layout, calibration coverage, and available helpers before fitting. A mismatch in those inputs would change the alignment, history propagation, or calibration fallback.

```bash
cd /holdout_task && cat validate_public_schema.py && sed -n '1,160p' public_asset_index.json; ls -lah /tmp/hegel_continual_archive/ 2>/dev/null || true; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference panel alignment imputation' || true; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_w01_s00.pkl','rb') as f:d=pickle.load(f)
def show(o,p='',depth=0):
 if isinstance(o,dict) and depth<3:
  for k,v in o.items(): show(v,p+'/'+str(k),depth+1)
 elif isinstance(o,np.ndarray):
  print(p,o.shape,o.dtype,str(o.reshape(-1)[:12]))
 elif isinstance(o,(list,tuple)):
  print(p,type(o).__name__,len(o),str(o[:2])[:500])
  if o and isinstance(o[0],dict) and depth<3: show(o[0],p+'/0',depth+1)
 else: print(p,type(o).__name__,str(o)[:250])
show(d)
PY
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'cmd_stress': array([-6.215057
/0/dataset/past_dt (18,) float64 [0.7823 0.4489 0.4688 0.7627 0.4657 0.3819 0.3589 0.4852 0.5915 0.2131
 0.427  0.3355]
/0/dataset/eval_times (10,) float64 [0.25 0.8  1.35 1.9  2.45 3.   3.55 4.1  4.65 5.2 ]
/0/dataset/yield_stress float 2.95
/0/dataset/calibration_load_cells dict {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'cmd_stress': array([-6.21505785e-01, -2.98912777e-01,  2.28682372e+00, -1.23310750e+00,
        5.8649
/0/dataset/train dict {'lab_id': array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 0, 2, 1, 1,
       3, 1, 1, 0, 3, 0, 2, 1, 2, 1, 3, 3, 2, 3, 0, 1, 0, 1, 2, 3, 3, 0,
       3, 2, 2, 1, 0, 1, 0, 1, 1, 1, 2, 1, 3, 2, 3, 3, 3, 0, 2, 0, 2, 2,
       1, 1, 1, 2, 2
/0/dataset/test dict {'lab_id': array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 3, 3, 1, 1, 2,
       2, 0, 1, 1, 3, 2, 3, 0, 0, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 2, 3,
       0, 0, 3, 2, 2, 3, 2, 2, 0, 3, 2, 2, 1, 2, 2, 1, 3, 3, 2, 0, 2, 2,
       2, 2, 0, 0, 3
/0/gt/taus (2,) float64 [0.522 3.781]
/0/gt/theta (7,) float64 [ 0.663  0.538  0.09   0.16  -0.13   0.11  -0.08 ]
/0/gt/vp_theta (2,) float64 [0. 0.]
/0/gt/stress_gains (4,) float64 [1. 1. 1. 1.]
/0/gt/stress_biases (4,) float64 [0. 0. 0. 0.]
/0/gt/te_clean (80, 10) float64 [-0.91511475 -1.53921694 -1.78923856 -1.90151363 -1.95943445 -1.99293361
 -2.01318815 -2.02473542 -2.02971103 -2.02933828  0.64224696  1.2558122 ]
/0/gt/te_true_rate0 (80,) float64 [-2.7818639   2.67857093 -1.57912988 -0.65093635 -0.54559053  3.63370244
 -0.59846434 -2.11295968  0.28772562  0.63525448 -1.53910437  2.09531607]
/0/gt/te_cross (80,) float64 [5.2        0.18100121 5.2        5.2        5.2        0.05451649
 5.2        5.2        1.11570504 0.38357811 5.2        0.17687767]
/0/gt/te_secant (80,) float64 [-1.71771747  0.95196753 -0.90685144 -3.13432247 -2.16586047  1.97749206
 -2.66273803 -5.90910059  1.7603653   0.63471563  0.8564231   0.10908674]
/0/gt/te_hold_true (80,) float64 [-0.97408061  1.93151489 -1.67174989  0.55701119 -0.42903293  2.02902678
  0.02616972 -0.14025124  0.0353875   0.8734407  -0.33560299  1.985662  ]
/0/gt/te_hist_true (80, 18) float64 [-1.83201538 -2.31897472 -0.2592002  -1.09663848  0.62467323 -0.54119106
  0.7507724  -0.37744122 -1.07096434 -0.56836828  1.58507429  1.88616828]
```

### Turn 2 — `execute_in_remote_docker` (`19084 ms` LLM)
- **Rationale**: The first output omitted nested split fields. Inspect those fields without reading the public truth labels to identify the reference and observation layout; calibration residuals and archive availability determine whether the solver needs robust rejection or a sparse-check fallback.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_w01_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
for split in ('train','test'):
 s=d[split]; print(split,[(k,np.shape(v),str(np.asarray(v).dtype)) for k,v in s.items()]);
 for k in s:
  if 'ref' in k or 'strat' in k or 'time' in k:
   a=np.asarray(s[k]); print(k, a[:12])
 print('feature NaN by column',np.isnan(s['coupon_features_raw']).sum(0))
a=d['calibration_load_cells']
for lab in range(4):
 m=a['lab_id']==lab; x=a['cmd_stress'][m]; y=a['true_stress'][m]; p=np.polyfit(x,y,1); print('cal',lab,len(x),p,np.round(y-np.polyval(p,x),7))
print('top-level keys',list(d))
PY
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s bytes\n' | head -35; else echo 'No continual archive mounted'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation' | head -18
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection' | head -18
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | head -30
```
- **Container Exit Code**: `0` (`561 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8fc36d84a03d897875134",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "Inputs are finite real arrays convertible to float64, with magnitudes that keep intermediate residuals, model evaluations, and least-squares calculations representable. The linear fit requires a nonempty (n,p) design and an (n,) response with n >= p >= 1 and numerical full column rank, including after Cauchy row weighting. Supplied scale and noise_floor must be positive finite scalars.",
        "The relaxation fit requires aligned length-n vectors, n >= 4, strictly increasing nonnegative times measured from the supplied onset, and a predictor numerically independent of the intercept. The onset rate is independently specified and nonnegative, noise_scale is positive, curvature_ratio is nonnegative, 2 <= bins <= n, and 0 < tau_min < tau_max. The saturation branch requires a positive onset rate, a numerically full-rank fitted Jacobian, and a positive slope not active at its lower bound.",
        "The step fit requires aligned length-n vectors and a nonempty integer vector of candidate split indices, each leaving at least three samples on either side. Each candidate design must have full column rank: the predictor cannot be constant within both segments. The scale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
        "fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "fit_anchored_relaxation",
          "signature": "fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None]",
          "doc_summary": "Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing nonnegative times, a predictor independent of the intercept, nonnegative rate and curvature ratio, positive noise scale, 2 <= bins <= n, and ordered positive tau bounds, returning (offset, positive slope,"
        }
      ],
```

### Turn 3 — `execute_in_remote_docker` (`24510 ms` LLM)
- **Rationale**: The previous log was truncated before the calibration and archive results. Compact summaries will distinguish reference layouts and show whether sparse calibration needs archive support; observation missingness determines whether the fit needs masking.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_w01_s00.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
a=d['calibration_load_cells']; np.set_printoptions(precision=5,suppress=True)
for lab in range(4):
    m=a['lab_id']==lab; x=a['cmd_stress'][m]; y=a['true_stress'][m]; p=np.polyfit(x,y,1)
    print('calibration',lab,len(x),np.ptp(x),p,y-np.polyval(p,x))
lab=np.concatenate([d[s]['lab_id'] for s in ('train','test')]); raw=np.concatenate([d[s]['coupon_features_raw'] for s in ('train','test')]); refs=np.concatenate([d[s]['ref_coupons'] for s in ('train','test')]); strata=np.concatenate([d[s]['stratum'] for s in ('train','test')]); print('cells', [np.bincount(strata[lab==i]).tolist() for i in range(4)],'feature missing',np.isnan(raw).sum(0))
for i in range(4):
    r=refs[lab==i]; print('lab',i,'raw mean/std',np.nanmean(raw[lab==i],0),np.nanstd(raw[lab==i],0))
    for layout in (r.reshape(-1,8,4),r.reshape(-1,4,8).transpose(0,2,1)):
        print('ref median of means/stds',np.nanmedian(np.nanmean(layout,axis=1),axis=0),np.nanmedian(np.nanstd(layout,axis=1),axis=0))
print('time',d['eval_times']); print('missing observations',[(k,int(np.isnan(d['train'][k]).sum())) for k in ['instant_rate_obs','strain_curves_obs']])
PY
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%P %s bytes\n' | head -25; else echo 'No continual archive mounted'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation' | cut -c1-225 | head -10
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | grep -m 4 -E 'signature|APPLICABLE|preconditions'
```
- **Container Exit Code**: `0` (`586 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.11563 1.1862  1.07317 1.097  ]
ref median of means/stds [0.47519 0.46834 0.43025 0.41424] [1.15068 1.11721 1.09352 1.10722]
lab 2 raw mean/std [-0.18976 -0.18367 -0.29934 -0.06892] [0.99777 0.96117 0.92813 1.07473]
ref median of means/stds [-0.21345 -0.40335 -0.26979 -0.24669] [0.79606 0.84089 0.78897 0.81873]
ref median of means/stds [-0.20295 -0.29265 -0.31408 -0.27671] [0.80286 0.85684 0.8232  0.77563]
lab 3 raw mean/std [0.74657 0.51442 0.8012  0.97809] [1.57512 1.14311 1.44636 1.58256]
ref median of means/stds [0.75781 0.70334 0.80812 0.89643] [1.22761 1.19727 1.20234 1.24701]
ref median of means/stds [0.81306 0.69421 0.9573  0.76622] [1.30904 1.23093 1.26834 1.23193]
time [0.25 0.8  1.35 1.9  2.45 3.   3.55 4.1  4.65 5.2 ]
missing observations [('instant_rate_obs', 0), ('strain_curves_obs', 0)]
No continual archive mounted
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite 
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in 
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, 
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | No
      "mathematical_preconditions": [
        "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
        "fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
```

### Turn 4 — `execute_in_remote_docker` (`64334 ms` LLM)
- **Rationale**: Implement the exact stress recurrence as exponential history weights and jointly fit the observed curves and onset rates. Schema validation checks the interface; leave-one-lab-out residuals test transfer and distinguish the two plausible reference layouts without using truth labels.
- **Alternatives Weighed**: `Single-branch anchored relaxation helper` (rejected: It assumes a single positive onset response, whereas each coupon has two signed stress contrasts and a shared feature-dependent drift.)

```bash
cd /holdout_task && cat > alpha_w01_s00_solver.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _calibrate(rows):
    """Fit each cell's affine map, rejecting isolated EMI excursions."""
    labs, x, y = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    maps = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        use = (labs == lab) & np.isfinite(x) & np.isfinite(y)
        u, v = x[use], y[use]
        if not len(u):
            continue
        design = np.column_stack((u, np.ones(len(u))))
        i, j = np.triu_indices(len(u), 1)
        delta = u[j] - u[i]
        good = np.abs(delta) > 1e-8
        if not good.any():
            maps[lab, 1] = np.median(v-u)
            continue
        slopes = (v[j[good]] - v[i[good]]) / delta[good]
        offsets = np.median(v - slopes[:, None]*u, axis=1)
        residual = np.abs(v - slopes[:, None]*u - offsets[:, None])
        best = np.argmin(np.median(residual, axis=1))
        r = residual[best]
        keep = r <= max(6*np.median(r), 1e-3)
        if np.ptp(u[keep]) > 1e-8:
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        else:
            maps[lab] = slopes[best], offsets[best]
    return maps


def _align(dataset, feature_major=False):
    tr, te = dataset['train'], dataset['test']
    lab, strata, raw, refs = (np.concatenate([tr[k], te[k]]) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    if feature_major:
        panel = refs.reshape(len(lab), 4, -1).transpose(0, 2, 1)
    else:
        panel = refs.reshape(len(lab), -1, 4)
    count = np.isfinite(panel).sum(axis=1)
    mean = np.divide(np.nansum(panel, axis=1), count,
                     out=np.full(count.shape, np.nan), where=count > 0)
    var = np.divide(np.nansum((panel-mean[:, None])**2, axis=1), count,
                    out=np.full(count.shape, np.nan), where=count > 0)
    aligned = np.empty_like(raw, dtype=float)
    for lab_id in np.unique(lab):
        mask = lab == lab_id
        loc = np.nanmedian(mean[mask], axis=0)
        scale = np.nanmedian(np.sqrt(var[mask]), axis=0)
        aligned[mask] = (raw[mask] - loc) / np.where(scale > 0, scale, 1)
    cells = np.column_stack((lab, strata))
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    means = np.full((int(inv.max()) + 1, raw.shape[1]), np.nan)
    for cell in range(len(means)):
        values = aligned[inv == cell]
        count = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), count, out=means[cell], where=count > 0)
    count = np.isfinite(means).sum(axis=0)
    balanced = np.nansum(means, axis=0) / np.maximum(count, 1)
    fill = means[inv]
    return np.where(np.isfinite(aligned), aligned, np.where(np.isfinite(fill), fill, balanced))


def _inputs(split, x, maps):
    gain, bias = maps[split['lab_id']].T
    return (split['past_stress_cmd']*gain[:, None] + bias[:, None],
            split['stress_hold_cmd']*gain + bias,
            np.column_stack((np.ones(len(x)), x)), np.asarray(split['strain_0']))


class _Law:
    def __init__(self, inputs, dt, times):
        self.hist, self.hold, self.x, self.e0 = inputs
        self.t = np.asarray(times)
        self.tail = np.r_[np.cumsum(np.asarray(dt)[:0:-1])[::-1], 0.]
        self.start = self.tail + dt

    def evaluate(self, p, jac=False):
        tau, a = p[:2], p[2:4]
        late = np.exp(-self.tail[:, None]/tau)
        early = np.exp(-self.start[:, None]/tau)
        delta = self.hold[:, None] - self.hist @ (late - early)
        et = np.exp(-self.t[:, None]/tau)
        shape = -np.expm1(-self.t[:, None]/tau)
        drift = self.x @ p[4:]
        strain = self.e0[:, None] + (delta*a) @ shape.T + drift[:, None]*self.t
        rate = delta @ (a/tau) + drift
        if not jac:
            return strain, rate
        dd = -(self.hist @ (late*self.tail[:, None] - early*self.start[:, None]))/tau**2
        dshape = -et*self.t[:, None]/tau**2
        base = delta[:, None, :]*shape[None, :, :]
        jcurve = np.concatenate(((dd[:, None, :]*shape + delta[:, None, :]*dshape)*a,
                                 base, self.x[:, None, :]*self.t[None, :, None]), axis=2)
        jrate = np.column_stack(((dd/tau-delta/tau**2)*a, delta/tau, self.x))
        return strain, rate, jcurve, jrate


def _fit(law, strain, rate):
    lo = np.r_[[.20, .20], np.full(7, -np.inf)]
    hi = np.r_[[8., 8.], np.full(7, np.inf)]
    def run(p, sc, sr, loss):
        def res(v):
            c, r = law.evaluate(v)
            return np.r_[((c-strain)/sc).ravel(), (r-rate)/sr]
        def jac(v):
            _, _, c, r = law.evaluate(v, True)
            return np.vstack((c.reshape(-1, 9)/sc, r/sr))
        return least_squares(res, p, jac=jac, bounds=(lo, hi), loss=loss,
                             x_scale='jac', max_nfev=100)
    starts = ([.42, 3.0], [.85, 5.8], [.25, 1.6])
    best = min((run(np.r_[tau, .5, .5, np.zeros(5)], .05, .08, 'soft_l1')
                for tau in starts), key=lambda fit: fit.cost)
    c, r = law.evaluate(best.x)
    def scale(residual):
        return max(1e-6, 1.4826*np.median(np.abs(residual-np.median(residual))))
    sc, sr = scale(c-strain), scale(r-rate)
    best = run(best.x, 2.0*sc, 2.0*sr, 'cauchy')
    p = best.x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def solve_alpha_w01_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    maps = _calibrate(dataset['calibration_load_cells'])
    x = _align(dataset)
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), dataset['eval_times']
    model = _Law(_inputs(train, x[:n], maps), dt, t)
    p = _fit(model, train['strain_curves_obs'], train['instant_rate_obs'])
    curve, rate = _Law(_inputs(test, x[n:], maps), dt, t).evaluate(p)
    return dict(estimated_taus=p[:2], hereditary_weights=p[2:],
                viscoplastic_weights=np.zeros(2), predicted_strain=curve,
                initial_rate_pred=rate)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/alpha_cv.py <<'PY'
import importlib.util
import pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_s00_solver.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_alpha_w01_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr=d['train']; labs=tr['lab_id']; dt=d['past_dt']; t=d['eval_times']; y=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; cal=m._calibrate(d['calibration_load_cells']); n=len(labs)
for layout in (False, True):
    x=m._align(d,layout)[:n]; inp=m._inputs(tr,x,cal)
    print('feature_major',layout)
    for lab in range(4):
        fit=labs!=lab; val=~fit
        law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
        p=m._fit(law(fit), y[fit], rate[fit]); c,r=law(val).evaluate(p)
        print(lab,'tau',np.round(p[:2],5),'curve median/p90',np.round(np.quantile(abs(c-y[val]),[.5,.9]),6),'rate median/p90',np.round(np.quantile(abs(r-rate[val]),[.5,.9]),6), 'horizon p90 max', np.quantile(abs(c-y[val]),.9,axis=0).max())
    p=m._fit(m._Law(inp,dt,t),y,rate); print('full',p)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_cv.py
```
- **Container Exit Code**: `0` (`1102 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
feature_major False
0 tau [0.52838 4.45707] curve median/p90 [0.04456  0.706547] rate median/p90 [0.01654  0.045273] horizon p90 max 0.9479199496535948
1 tau [0.527   3.96009] curve median/p90 [0.090076 0.440978] rate median/p90 [0.033582 0.052939] horizon p90 max 0.8009876034329393
2 tau [0.52171 3.71142] curve median/p90 [0.133076 0.311854] rate median/p90 [0.044785 0.063447] horizon p90 max 0.63122368861143
3 tau [0.52756 4.35791] curve median/p90 [0.024309 0.162274] rate median/p90 [0.009892 0.026657] horizon p90 max 0.6686423901355174
full [ 0.52711651  4.12839873  0.67499101  0.54754281  0.08485898  0.13972761
 -0.11188452  0.09529461 -0.06941043]
feature_major True
0 tau [0.52659 3.98061] curve median/p90 [0.025781 0.767207] rate median/p90 [0.010719 0.028641] horizon p90 max 0.9471308260201364
1 tau [0.52118 3.56314] curve median/p90 [0.047624 0.380719] rate median/p90 [0.018526 0.038506] horizon p90 max 0.8427010941247894
2 tau [0.52347 3.86855] curve median/p90 [0.017312 0.254496] rate median/p90 [0.010868 0.025574] horizon p90 max 0.6059434370961831
3 tau [0.52215 3.83171] curve median/p90 [0.051344 0.170258] rate median/p90 [0.016753 0.041477] horizon p90 max 0.7325760543296792
full [ 0.52345946  3.83592662  0.66627796  0.5466458   0.100357    0.14255921
 -0.11712106  0.09632311 -0.06719751]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=87af834120d38a49 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_87af8341 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`23143 ms` LLM)
- **Rationale**: Compare lab-specific drift estimates after subtracting the fitted hereditary response, and inspect the named alignment primitive. If the primitive specifies a different panel layout or fill rule, adopt that contract and retest cross-lab transfer.
- **Alternatives Weighed**: `Select the reference layout on curve p90 alone` (rejected: The signed creep response and onset rate both depend on the same drift, while curve p90 is sensitive to the realized Cauchy glitches.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | grep -E 'doc_summary|signature'; cat > /tmp/alpha_diagnose.py <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
from scipy.optimize import least_squares
for layout in (False, True):
    x=m._align(d,layout)[:n]; inp=m._inputs(tr,x,cal); law=m._Law(inp,dt,t); p=m._fit(law,y,rate)
    c,r=law.evaluate(p); print('layout',layout,'train abs median',np.median(abs(c-y)),np.median(abs(r-rate)))
    raw=tr['coupon_features_raw']; labs=tr['lab_id']; keep=np.isfinite(raw).all(1)
    p[4:]=0; c,r=law.evaluate(p)
    for lab in range(4):
        sel=(labs==lab)&keep
        X=inp[2][sel]; yy=y[sel]-c[sel]; rr=rate[sel]-r[sel]
        dc=np.vstack([(X[:,None,:]*t[None,:,None]).reshape(-1,5),X*3]); dz=np.r_[yy.ravel(),rr*3]
        q=least_squares(lambda w:dc@w-dz,np.zeros(5),loss='cauchy',f_scale=.025).x
        print(lab,'local drift',np.round(q,5),'local residual median',np.median(abs(dc@q-dz)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_diagnose.py
```
- **Container Exit Code**: `0` (`699 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
g.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_reference_panel_balancing",
      "reference_source_txt": null
    }
  ]
}
          "signature": "balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray",
          "doc_summary": "Return positive mean-one weights (n,) from nonempty finite real vectors groups (n,) and strata (n,) with every observed group–stratum combination present."
layout False train abs median 0.043418505566679144 0.018694894962833553
0 local drift [ 0.07708  0.14706 -0.12306  0.1036  -0.07526] local residual median 0.008286043657067713
1 local drift [ 0.06263  0.13949 -0.12022  0.09202 -0.06829] local residual median 0.008456716996501662
2 local drift [ 0.11454  0.13593 -0.11626  0.09215 -0.06964] local residual median 0.00826195404056515
3 local drift [ 0.08905  0.13867 -0.10912  0.09287 -0.06993] local residual median 0.008957304049227455
layout True train abs median 0.022897153806818782 0.009829974357261628
0 local drift [ 0.1069   0.13973 -0.11978  0.09665 -0.07232] local residual median 0.008849654516672203
1 local drift [ 0.09043  0.14425 -0.11351  0.09346 -0.06915] local residual median 0.00906436554396696
2 local drift [ 0.09888  0.13728 -0.1187   0.09657 -0.06594] local residual median 0.008757502260963834
3 local drift [ 0.11458  0.1476  -0.11269  0.09744 -0.06911] local residual median 0.009052400470127342

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=87af834120d38a49 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_87af8341 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29182 ms` LLM)
- **Rationale**: The reference-panel primitive defines column moments across coupons followed by scalar medians within each lab, not moments over reshaped rows. Compare that interpretation and the balanced-fill rules on held-out labs, reporting missing-feature coupons separately so complete rows cannot hide an imputation error.
- **Alternatives Weighed**: `Fit separate drift coefficients by lab` (rejected: This can conceal reference-calibration errors and no longer represents the requested shared seven-weight law.)

```bash
cat > /tmp/alpha_alignment_cv.py <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
lab, strata, raw, ref = (np.concatenate([d[s][k] for s in ('train','test')]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
_, cell, sizes = np.unique(np.column_stack((lab,strata)),axis=0,return_inverse=True,return_counts=True)
for scaling in ('scalar_panel_columns','interleaved','feature_major'):
    x = np.empty_like(raw)
    for j in range(4):
        ix = lab==j
        a = ref[ix]
        if scaling == 'scalar_panel_columns':
            loc = np.nanmedian(np.nanmean(a,axis=0)); sc = np.nanmedian(np.nanstd(a,axis=0))
        else:
            a = a.reshape(-1,8,4) if scaling=='interleaved' else a.reshape(-1,4,8).transpose(0,2,1)
            loc = np.nanmedian(np.nanmean(a,axis=1),axis=0); sc = np.nanmedian(np.nanstd(a,axis=1),axis=0)
        x[ix] = (raw[ix]-loc)/sc
    for fill in ('cell','balanced_rows','balanced_observed_cells'):
        cm = np.array([np.nanmean(x[cell==i],axis=0) for i in range(len(sizes))])
        w = (1/sizes[cell])[:,None]
        wr = np.nansum(w*x,axis=0)/np.sum(w*np.isfinite(x),axis=0)
        aligned = np.where(np.isfinite(x),x,{'cell':cm[cell],'balanced_rows':wr,'balanced_observed_cells':np.nanmean(cm,axis=0)}[fill])
        inp = m._inputs(tr,aligned[:n],cal); metrics=[]
        for j in range(4):
            fit=labs!=j; val=~fit
            law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
            p=m._fit(law(fit),y[fit],rate[fit]); c,r=law(val).evaluate(p)
            ec=abs(c-y[val]); er=abs(r-rate[val]); missing=~np.isfinite(raw[:n][val]).all(1)
            metrics.append([np.median(ec),np.median(ec,axis=0).max(),np.quantile(er,.9),np.median(er[missing]) if missing.any() else 0, p[0],p[1]])
        print(scaling,fill,'worst [curve median,horizon median,rate p90,missing rate median]', np.round(np.max(metrics,axis=0)[:4],6),'tau spans',np.round(np.array(metrics)[:,4:].ptp(axis=0),5))
PY
sed -i 's/np.array(metrics)\[:,4:\].ptp(axis=0)/np.ptp(np.array(metrics)[:,4:],axis=0)/' /tmp/alpha_alignment_cv.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_alignment_cv.py
```
- **Container Exit Code**: `0` (`1868 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scalar_panel_columns cell worst [curve median,horizon median,rate p90,missing rate median] [0.018152 0.033213 0.025299 0.110209] tau spans [0.00158 0.0837 ]
scalar_panel_columns balanced_rows worst [curve median,horizon median,rate p90,missing rate median] [0.017466 0.032677 0.027935 0.103792] tau spans [0.00148 0.07541]
scalar_panel_columns balanced_observed_cells worst [curve median,horizon median,rate p90,missing rate median] [0.017474 0.032681 0.02794  0.103594] tau spans [0.00148 0.07524]
interleaved cell worst [curve median,horizon median,rate p90,missing rate median] [0.133076 0.241904 0.063447 0.104807] tau spans [0.00667 0.74566]
interleaved balanced_rows worst [curve median,horizon median,rate p90,missing rate median] [0.13304  0.242812 0.063944 0.091365] tau spans [0.00665 0.73985]
interleaved balanced_observed_cells worst [curve median,horizon median,rate p90,missing rate median] [0.133074 0.242824 0.063953 0.091635] tau spans [0.00667 0.74111]
feature_major cell worst [curve median,horizon median,rate p90,missing rate median] [0.051344 0.112574 0.041477 0.116052] tau spans [0.00541 0.41747]
feature_major balanced_rows worst [curve median,horizon median,rate p90,missing rate median] [0.05114  0.112195 0.041243 0.127688] tau spans [0.00495 0.38803]
feature_major balanced_observed_cells worst [curve median,horizon median,rate p90,missing rate median] [0.051144 0.112206 0.041238 0.127835] tau spans [0.00493 0.38758]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=87af834120d38a49 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_87af8341 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`34993 ms` LLM)
- **Rationale**: Promote the alignment supported by the reference primitive and held-out labs. A held-out wing tests transfer without labeled coupons from its neighboring lab. Sparse synthetic calibration checks identify which cases need archive support rather than an unstable two-point fit.
- **Alternatives Weighed**: `Impute each feature using only its own lab–stratum cell` (rejected: It does not estimate a column mean balanced across cells and was slightly worse on curve and missing-row metrics.)
- **ATMS NOGOOD DECLARED**: Refuted `The 32 flattened controls must be reshaped into feature-specific reference panels.` | Witness: `Both reshapes worsen cross-lab transfer; the scalar interpretation reduces the worst horizon median from 0.2428 or 0.1122 to 0.0327 with balanced fill.` | Blanket Diagnosis: `The controls measure a shared location and scale for each lab. Reshaping independent reference columns introduced artificial feature-specific offsets and gains.` | Sublated Pivot: `Take control-column moments across coupons, reduce them to scalar medians within each lab, and use cell-balanced column means for missing features.`

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text()
a, b = s.index('def _align('), s.index('\ndef _inputs(')
s = s[:a] + '''def _align(dataset):
    """Remove the shared lab offset and gain, then fill with cell-balanced means."""
    tr, te = dataset['train'], dataset['test']
    lab, strata, raw, refs = (np.concatenate([tr[k], te[k]]) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab_id in np.unique(lab):
        mask = lab == lab_id
        panel = refs[mask]
        count = np.isfinite(panel).sum(axis=0)
        panel = panel[:, count >= 2]
        if panel.shape[1]:
            loc = np.nanmedian(np.nanmean(panel, axis=0))
            scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
            aligned[mask] = (raw[mask] - loc) / (scale if scale > 0 else 1.)
        else:
            aligned[mask] = raw[mask]
    _, cell, size = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = (1. / size[cell])[:, None]
    finite = np.isfinite(aligned)
    den = (weight * finite).sum(axis=0)
    fill = np.divide((weight * np.where(finite, aligned, 0.)).sum(axis=0), den,
                     out=np.zeros(raw.shape[1]), where=den > 0)
    return np.where(finite, aligned, fill)

''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
x=m._align(d)[:n]; inp=m._inputs(tr,x,cal)
for wing in (0,1):
    fit=labs//2!=wing
    law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
    p=m._fit(law(fit),y[fit],rate[fit]); c,r=law(~fit).evaluate(p)
    print('held-out wing',wing,'taus',np.round(p[:2],6),'worst horizon curve median',round(np.median(abs(c-y[~fit]),axis=0).max(),6),'rate p90',round(np.quantile(abs(r-rate[~fit]),.9),6))
p=m._fit(m._Law(inp,dt,t),y,rate); print('full fitted parameters', np.round(p,6))
for npts in (10,5,3,2,1,0):
    u=np.linspace(-2,2,npts) if npts!=1 else np.array([.3]); v=1.13*u-.17
    if npts>=3: v[npts//2]+=1.6
    rows=dict(lab_id=np.full(npts,2),cmd_stress=u,true_stress=v)
    print('synthetic calibration stress: checks',npts,'map',m._calibrate(rows)[2])
PY
grep -inE 'archive|continual|calibrat' public_asset_index.json | cut -c1-250 | head -25; if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s bytes\n' | head -20; fi
```
- **Container Exit Code**: `0` (`772 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held-out wing 0 taus [0.522593 3.744104] worst horizon curve median 0.020041 rate p90 0.021083
held-out wing 1 taus [0.521537 3.726928] worst horizon curve median 0.018781 rate p90 0.025666
full fitted parameters [ 0.521041  3.694478  0.660964  0.533121  0.088461  0.160011 -0.130346
  0.109528 -0.080008]
synthetic calibration stress: checks 10 map [ 1.13 -0.17]
synthetic calibration stress: checks 5 map [ 1.13 -0.17]
synthetic calibration stress: checks 3 map [1.93 1.43]
synthetic calibration stress: checks 2 map [ 1.13 -0.17]
synthetic calibration stress: checks 1 map [ 1.    -0.131]
synthetic calibration stress: checks 0 map [1. 0.]
8:            "calibration_load_cells": {

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5bf47c57b3888a01 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5bf47c57 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`45095 ms` LLM)
- **Rationale**: The public wing results support the creep law and reference convention. The sparse-check stress test exposed a distinct calibration failure, so test the replacement across glitch locations and signs. Archive inventory determines whether a historical same-lab prior can be wired in without guessing its format.
- **Alternatives Weighed**: `Borrow the neighboring lab’s affine map` (rejected: The stated calibration law assigns separate gain and bias to each lab; sharing a wing does not establish that the maps are interchangeable.)
- **ATMS NOGOOD DECLARED**: Refuted `Median residual ranking alone identifies the correct affine map from three checks when one is corrupted.` | Witness: `With synthetic gain 1.13, bias −0.17, and one +1.6 EMI excursion, the three-check estimator returned gain 1.93 and bias 1.43.` | Blanket Diagnosis: `Several pairwise affine fits can have two inliers, so the sparse in-batch checks alone cannot identify the clean pair.` | Sublated Pivot: `Rank sparse fits by inlier count and then by proximity to a prior map; inspect archive availability to establish the source of that prior.`

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text()
s = s.replace('def _calibrate(rows):','def _calibrate(rows, prior=None):',1)
s = s.replace('    maps = np.tile([1., 0.], (4, 1))','    maps = np.tile([1., 0.], (4, 1)) if prior is None else prior.copy()',1)
s = s.replace('            maps[lab, 1] = np.median(v-u)','            maps[lab, 1] = np.median(v-maps[lab, 0]*u)',1)
needle = '        offsets = np.median(v - slopes[:, None]*u, axis=1)'
s = s.replace(needle, '''        if len(u) <= 4:
            offsets = ((v[i[good]]-slopes*u[i[good]]) +
                       (v[j[good]]-slopes*u[j[good]])) / 2
            fits = offsets[:, None] + slopes[:, None]*u
            inlier = np.abs(v-fits) <= 1e-3
            deviation = np.mean((fits-(maps[lab, 0]*u+maps[lab, 1]))**2, axis=1)
            best = np.lexsort((deviation, -inlier.sum(axis=1)))[0]
            keep = inlier[best]
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
            continue
''' + needle, 1)
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
for npts in (10, 5, 4, 3, 2, 1, 0):
    u = np.linspace(-2, 2, npts) if npts != 1 else np.array([.3])
    worst = 0.
    for index in (range(npts) if npts >= 3 else [-1]):
        for sign in (-1, 1):
            v = 1.13*u-.17
            if index >= 0:
                v[index] += sign*1.6
            rows = dict(lab_id=np.full(npts,2), cmd_stress=u, true_stress=v)
            fit = m._calibrate(rows)[2]
            worst = max(worst, np.max(np.abs((fit-[1.13,-.17])*np.array([2.,1.]))))
    print('checks', npts, 'worst coefficient error on stress scale', round(worst, 6))
print('public calibrated maps:', cal.round(6).tolist())
PY
archive=/tmp/hegel_continual_archive; if [ -d "$archive" ]; then echo 'Archive inventory:'; find "$archive" -maxdepth 3 -type f -printf '%P %s bytes\n' | head -25; else echo 'Continual archive not mounted in the public container'; fi
```
- **Container Exit Code**: `0` (`947 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
checks 10 worst coefficient error on stress scale 0.0
checks 5 worst coefficient error on stress scale 0.0
checks 4 worst coefficient error on stress scale 0.0
checks 3 worst coefficient error on stress scale 0.0
checks 2 worst coefficient error on stress scale 0.0
checks 1 worst coefficient error on stress scale 0.26
checks 0 worst coefficient error on stress scale 0.26
public calibrated maps: [[1.000017, -1.6e-05], [0.999989, -3.1e-05], [0.99998, 2.2e-05], [1.000009, 3.9e-05]]
Continual archive not mounted in the public container

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fce5451ce05aceba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fce5451c (directory).
```

### Turn 9 — `execute_in_remote_docker` (`24369 ms` LLM)
- **Rationale**: Check the published asset metadata for an archive format before wiring in a reader. If it specifies none, use a narrow reader for records matching the declared calibration fields and verify the fallback with a temporary fixture.
- **Alternatives Weighed**: `Submit the current solver unchanged` (rejected: The calibration fit accepts a prior, but the public entry point never supplies one. A mounted archive would therefore have no effect.)

```bash
cd /holdout_task && sed -n '1,65p' public_asset_index.json; ls -d /tmp/hegel* /pretrain_task /holdout_task 2>/dev/null; sed -n '1,65p' alpha_w01_s00_solver.py
```
- **Container Exit Code**: `0` (`266 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
- u[i]
        good = np.abs(delta) > 1e-8
        if not good.any():
            maps[lab, 1] = np.median(v-maps[lab, 0]*u)
            continue
        slopes = (v[j[good]] - v[i[good]]) / delta[good]
        if len(u) <= 4:
            offsets = ((v[i[good]]-slopes*u[i[good]]) +
                       (v[j[good]]-slopes*u[j[good]])) / 2
            fits = offsets[:, None] + slopes[:, None]*u
            inlier = np.abs(v-fits) <= 1e-3
            deviation = np.mean((fits-(maps[lab, 0]*u+maps[lab, 1]))**2, axis=1)
            best = np.lexsort((deviation, -inlier.sum(axis=1)))[0]
            keep = inlier[best]
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
            continue
        offsets = np.median(v - slopes[:, None]*u, axis=1)
        residual = np.abs(v - slopes[:, None]*u - offsets[:, None])
        best = np.argmin(np.median(residual, axis=1))
        r = residual[best]
        keep = r <= max(6*np.median(r), 1e-3)
        if np.ptp(u[keep]) > 1e-8:
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        else:
            maps[lab] = slopes[best], offsets[best]
    return maps


def _align(dataset):
    """Remove the shared lab offset and gain, then fill with cell-balanced means."""
    tr, te = dataset['train'], dataset['test']
    lab, strata, raw, refs = (np.concatenate([tr[k], te[k]]) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab_id in np.unique(lab):
        mask = lab == lab_id
        panel = refs[mask]
        count = np.isfinite(panel).sum(axis=0)
        panel = panel[:, count >= 2]
        if panel.shape[1]:
            loc = np.nanmedian(np.nanmean(panel, axis=0))
            scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
            aligned[mask] = (raw[mask] - loc) / (scale if scale > 0 else 1.)
        else:
            aligned[mask] = raw[mask]
    _, cell, size = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = (1. / size[cell])[:, None]
    finite = np.isfinite(aligned)
    den = (weight * finite).sum(axis=0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fce5451ce05aceba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fce5451c (directory).
```

### Turn 10 — `execute_in_remote_docker` (`52082 ms` LLM)
- **Rationale**: No archive format was documented, so read JSON and pickle records containing the declared calibration fields. Verify the entry point using public coupons with re-encoded stresses and rotated sparse-check and EMI patterns. If the outputs change despite unchanged physical stresses, the archive integration needs correction.
- **Alternatives Weighed**: `Use archive calibration for every batch` (rejected: Well-supported current checks should take precedence over historical calibration.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text()
s = 'import json\nimport pickle\nfrom pathlib import Path\n\n' + s
s = s.replace("\ndef _calibrate(rows, prior=None):", "\n_ARCHIVE = Path('/tmp/hegel_continual_archive')\n_CAL_KEYS = ('lab_id', 'cmd_stress', 'true_stress')\n\n\ndef _calibrate(rows, prior=None):", 1)
s = s.replace("    maps = np.tile([1., 0.], (4, 1)) if prior is None else prior.copy()", "    maps = np.tile([1., 0.], (4, 1))\n    historical = np.zeros(4, dtype=bool) if prior is None else np.isfinite(prior).all(axis=1)\n    if prior is not None:\n        maps[historical] = prior[historical]", 1)
s = s.replace("        design = np.column_stack((u, np.ones(len(u))))", "        if historical[lab] and len(u) <= 2:\n            residual = v - (maps[lab, 0]*u + maps[lab, 1])\n            agrees = np.abs(residual) <= 1e-3\n            if agrees.any():\n                maps[lab, 1] += np.median(residual[agrees])\n                continue\n        design = np.column_stack((u, np.ones(len(u))))", 1)
start = s.index('\ndef _align(')
s = s[:start] + '''
def _archive_rows(record):
    # Accept calibration tables and records wrapped like the public dataset.
    if isinstance(record, (list, tuple)):
        for item in reversed(record):
            yield from _archive_rows(item)
    elif isinstance(record, dict):
        table = record.get('dataset', record)
        if isinstance(table, dict):
            table = table.get('calibration_load_cells', table)
            if isinstance(table, dict) and all(key in table for key in _CAL_KEYS):
                yield table


def _load_maps(rows):
    current = _calibrate(rows)
    labs, u, v = (np.asarray(rows[k]) for k in _CAL_KEYS)
    valid = np.isfinite(u) & np.isfinite(v)
    need = set()
    for lab in range(4):
        take = valid & (labs == lab)
        x, y = u[take], v[take]
        slope, bias = current[lab]
        if len(x) <= 4 or np.ptp(x) <= 1e-8 or np.any(np.abs(y - (slope*x + bias)) > 1e-3):
            need.add(lab)
    if not need or not _ARCHIVE.is_dir():
        return current
    prior = np.full((4, 2), np.nan)
    files = (path for path in _ARCHIVE.rglob('*')
             if path.is_file() and path.suffix in ('.json', '.pkl', '.pickle'))
    for path in sorted(files, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            with path.open('rb') as stream:
                record = json.load(stream) if path.suffix == '.json' else pickle.load(stream)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        for table in _archive_rows(record):
            a, x, y = (np.asarray(table[k]) for k in _CAL_KEYS)
            if a.ndim != 1 or a.shape != x.shape or a.shape != y.shape:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            fitted = _calibrate(table)
            for lab in tuple(need):
                mask = finite & (a == lab)
                g, b = fitted[lab]
                inliers = mask & (np.abs(y - (g*x + b)) <= 1e-3)
                if (inliers.sum() >= 3 and inliers.sum() > mask.sum()/2
                        and np.ptp(x[inliers]) > 1e-8):
                    prior[lab] = fitted[lab]
                    need.remove(lab)
            if not need:
                return _calibrate(rows, prior)
    return _calibrate(rows, prior)

''' + s[start:]
s = s.replace("maps = _calibrate(dataset['calibration_load_cells'])", "maps = _load_maps(dataset['calibration_load_cells'])", 1)
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
import json
from pathlib import Path
from tempfile import TemporaryDirectory
baseline = m.solve_alpha_w01_s00(d)
assert np.array_equal(baseline['viscoplastic_weights'], np.zeros(2))
assert .2 <= baseline['estimated_taus'][0] < baseline['estimated_taus'][1] <= 8
assert all(np.isfinite(v).all() for v in baseline.values())
with TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp)
    maps = np.array([[.87,.16],[1.08,-.09],[1.13,-.17],[.95,.08]])
    n = 10
    lab = np.repeat(np.arange(4), n)
    cmd = np.tile(np.linspace(-2,2,n),4)
    true = maps[lab,0]*cmd + maps[lab,1]
    historical = dict(lab_id=lab.tolist(),cmd_stress=cmd.tolist(),true_stress=true.tolist())
    (m._ARCHIVE/'prior.json').write_text(json.dumps([{'dataset': {'calibration_load_cells':historical}}]))
    simulated = d.copy()
    for split in ('train','test'):
        q = simulated[split] = d[split].copy()
        g,b = maps[q['lab_id']].T
        g0,b0 = cal[q['lab_id']].T
        q['past_stress_cmd'] = (g0[:,None]*q['past_stress_cmd']+b0[:,None]-b[:,None])/g[:,None]
        q['stress_hold_cmd'] = (g0*q['stress_hold_cmd']+b0-b)/g
    for k in range(4):
        ids = np.concatenate([j*n+np.array([0,4,9])[:(j+k)%4] for j in range(4)])
        y_sparse = true[ids].copy()
        for j in range(4):
            jj = np.flatnonzero(lab[ids]==j)
            if len(jj)>=2:
                y_sparse[jj[-1]] += 1.6
        simulated['calibration_load_cells'] = dict(lab_id=lab[ids],cmd_stress=cmd[ids],true_stress=y_sparse)
        out = m.solve_alpha_w01_s00(simulated)
        errors = {key: float(np.max(np.abs(out[key]-baseline[key]))) for key in out}
        assert max(errors.values()) < 1e-7, errors
        print('archive sparse/EMI rotation',k,'max output error',max(errors.values()))
PY
```
- **Container Exit Code**: `0` (`988 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
archive sparse/EMI rotation 0 max output error 3.4372504842394846e-12
archive sparse/EMI rotation 1 max output error 2.0825563495918686e-12
archive sparse/EMI rotation 2 max output error 2.644551244657123e-12
archive sparse/EMI rotation 3 max output error 2.9682922786378185e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=578e14bf88086067 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_578e14bf (directory).
```

### Turn 11 — `submit_completed_task` (`12372 ms` LLM)
- **Rationale**: The saved candidate passed the clean-process schema check, held-out wing validation, and end-to-end sparse calibration stress tests. The host verified the final artifact; no further checks are needed for submission.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_w01_s00_solver.py': 'FILE:9097'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_s00_solver.py`. The public schema passed. A simulated archive with sparse and EMI-corrupted load-cell checks changed outputs by at most 3.44×10⁻¹² when physical stresses were held fixed. No live continual archive was mounted in the public container.`

### Turn 12 — `execute_in_remote_docker` (`29562 ms` LLM)
- **Rationale**: Compare the candidate with explicit recurrence and onset differentiation on held-out public coupons; a mismatch would require a formula correction. Inspect signed errors by lab and stress contrasts within each lab to separate lab offsets from cross-lab evidence of missing stress dependence. Only a reproducible pattern would justify testing an extension.
- **Alternatives Weighed**: `Test relay modes, classification recall, or stochastic drift operators` (rejected: The coupon task specifies continuous strain and rate, with no relay state, class labels, or stochastic drift equation.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
from scipy.stats import spearmanr
x=m._align(d)[:n]; inp=m._inputs(tr,x,cal)
rows=[]; diagnostics=[]
for j in range(4):
    fit=labs!=j; val=~fit
    law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
    p=m._fit(law(fit),y[fit],rate[fit]); c,r=law(val).evaluate(p)
    hist,hold,drift_design,e0=(a[val] for a in inp)
    tau=p[:2]; z=np.zeros((val.sum(),2))
    for k,step in enumerate(dt):
        decay=np.exp(-step/tau)
        z=decay*z+(1-decay)*hist[:,k,None]
    contrast=hold[:,None]-z; drift=drift_design@p[4:]
    exact=e0[:,None]+(contrast*p[2:4])@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*t
    onset=contrast@(p[2:4]/tau)+drift
    np.testing.assert_allclose(c,exact,atol=2e-13,rtol=0)
    np.testing.assert_allclose(r,onset,atol=2e-13,rtol=0)
    eps=1e-6; tiny=law(val); tiny.t=np.array([eps]); cc,_=tiny.evaluate(p)
    error=np.max(abs((cc[:,0]-e0)/eps-r)); assert error<1e-5, error
    rc=y[val]-c; rr=rate[val]-r
    positive=t>0
    rd=np.median(rc[:,positive]/t[positive],axis=1)
    complete=np.isfinite(tr['coupon_features_raw'][val]).all(1)
    metrics=[np.median(abs(rc),axis=0).max(),np.quantile(abs(rr),.9),error]
    rows.append(metrics)
    print('lab',j,'worst horizon MedAE, rate p90, derivative error',np.round(metrics,8),'signed residual medians (curve drift, rate), complete cases', np.round([np.median(rd[complete]),np.median(rr[complete])],6))
    q=contrast[complete]
    probes=[q[:,0]+q[:,1], q[:,0]-q[:,1], np.sum(q*q,axis=1), (q[:,0]+q[:,1])/(1+hold[complete]**2)]
    diagnostics.append(np.array([[spearmanr(signal,residual).statistic for signal in probes] for residual in (rd[complete],rr[complete])]))
print('worst lab metrics',np.round(np.max(rows,axis=0),8))
print('within-lab rank correlation by lab and endpoint (sum, difference, squared magnitude, rational stress modulation)')
print(np.round(diagnostics,3))
PY
```
- **Container Exit Code**: `0` (`1109 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
lab 0 worst horizon MedAE, rate p90, derivative error [3.267719e-02 1.491083e-02 4.420000e-06] signed residual medians (curve drift, rate), complete cases [0.002213 0.001243]
lab 1 worst horizon MedAE, rate p90, derivative error [1.210836e-02 2.015817e-02 5.320000e-06] signed residual medians (curve drift, rate), complete cases [0.001307 0.000591]
lab 2 worst horizon MedAE, rate p90, derivative error [1.146146e-02 1.876474e-02 4.730000e-06] signed residual medians (curve drift, rate), complete cases [-0.000447 -0.0024  ]
lab 3 worst horizon MedAE, rate p90, derivative error [2.957215e-02 2.793476e-02 4.310000e-06] signed residual medians (curve drift, rate), complete cases [-0.001495 -0.002012]
worst lab metrics [3.267719e-02 2.793476e-02 5.320000e-06]
within-lab rank correlation by lab and endpoint (sum, difference, squared magnitude, rational stress modulation)
[[[ 0.319 -0.147 -0.075  0.105]
  [ 0.13  -0.026  0.236  0.078]]

 [[-0.119  0.013  0.006 -0.139]
  [-0.133 -0.087  0.053 -0.168]]

 [[-0.115  0.2   -0.17  -0.011]
  [ 0.245  0.249  0.153  0.207]]

 [[-0.286 -0.118 -0.212 -0.249]
  [-0.354 -0.032 -0.046 -0.365]]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=578e14bf88086067 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_578e14bf (directory).
```

### Turn 13 — `submit_completed_task` (`12690 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated schema verification, explicit recurrence and onset checks, held-out lab and wing validation, and end-to-end sparse calibration tests. The residual audit did not justify changing the model.
- **Alternatives Weighed**: `Add nonlinear stress terms or lab-specific drift` (rejected: Held-out residual associations changed sign across labs and did not establish a reproducible correction to the specified law.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_s00_solver.py`. The public schema passed, and the strain and onset predictions matched the explicit hereditary recurrence. With simulated sparse and EMI-corrupted calibration checks, archive fallback preserved all outputs within 3.44×10⁻¹² when physical stresses were unchanged. No live continual archive was mounted.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
