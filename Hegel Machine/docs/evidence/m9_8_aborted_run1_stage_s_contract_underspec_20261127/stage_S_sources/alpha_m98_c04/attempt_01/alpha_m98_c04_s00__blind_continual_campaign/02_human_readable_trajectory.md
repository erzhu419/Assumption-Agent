# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_c04_s00__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_hereditary_memory_and_factored_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `372,540` (`358,623` / `66,816` / `13,917`) |
| **Total Episode Wall Time (s)** | `438.8s` |

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

### Turn 1 — `execute_in_remote_docker` (`16730 ms` LLM)
- **Rationale**: Inspect timing, bridge sweeps, chamber observations, and archive availability before choosing constitutive and transfer models; missing or differently shaped fields would change the solver design.

```bash
cd /holdout_task; ls -la; cat public_asset_index.json validate_public_schema.py; ls -la /tmp/hegel_continual_archive /pretrain_task; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment imputation robust IRLS'; python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def show(d,depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,end=' ')
  if isinstance(v,dict):
   print();show(v,depth+2)
  elif isinstance(v,np.ndarray):
   print(v.shape,v.dtype,repr(v.ravel()[:8]))
  else: print(repr(v)[:240])
show(d)
PY
```
- **Container Exit Code**: `1` (`342 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
```
- **Stderr (tail)**:
```text
ls: cannot access '/tmp/hegel_continual_archive': No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`17391 ms` LLM)
- **Rationale**: Unwrap the campaign list and inspect the time coordinates and feature shapes before defining the hereditary integral; review the skill contracts for compatibility with the specified standardization and joint fit.

```bash
cd /holdout_task; cat validate_public_schema.py; for s in autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$s"; done; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
print('campaign count',len(p))
def show(d,depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,end=' ')
  if isinstance(v,dict):
   print();show(v,depth+2)
  elif isinstance(v,np.ndarray):
   print(v.shape,v.dtype,v.ravel()[:min(12,v.size)])
  else: print(repr(v)[:600])
show(p[0] if isinstance(p,list) else p)
if isinstance(p,list):
 for it in p[1:]:
  d=it.get('dataset',it); print('other campaign', [(k,v) for k,v in it.items() if isinstance(v,(str,int))],list(d),d['chamber_verification_trials'] is not None)
PY
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
41015871 -0.33931893
  0.34291762 -0.54045668  1.71618471  0.91794437 -1.14991009  1.9073998 ]
     ref_coupons ndarray (36, 8) float64 [-0.38042109 -1.6462841  -1.48966265 -1.23607613 -0.68653883 -0.45456618
  0.66472991 -0.39983447 -0.09324013  0.49618597  1.14720043 -1.05688897]
     past_channel_raw ndarray (36, 6, 4) float64 [-0.1288843  -0.27994514  0.54050702  1.1456481  -0.15787091  0.02423193
  0.53514686 -0.51743728  0.10676189  0.56852455  0.37089194 -0.25987292]
     hold_channel_raw ndarray (36, 4) float64 [0.85361453 0.92333082 0.58578313 0.70064589 0.78405704 0.42976047
 0.61562999 1.32721266 0.82740015 1.14562272 1.34931634 1.18776377]
     strain_0 ndarray (36,) float64 [ 0.02641862 -0.00852588 -0.08667475 -0.04008219 -0.06382487 -0.01568699
  0.10179656  0.04462592  0.19726495  0.14120221  0.14466452 -0.02964436]
     aux_chamber_telemetry ndarray (36, 4) float64 [ 0.51286212 -0.56964083  0.87232379 -0.08315554  1.70916831 -0.37326443
 -1.34869873 -1.03828949 -0.89761916  1.17579166 -1.09778944  0.67754142]
   chamber_verification_trials dict 
     lab_id ndarray (18,) int64 [2 0 0 3 0 1 2 1 0 3 1 1]
     stratum ndarray (18,) int64 [0 2 0 2 2 2 0 1 2 2 2 1]
     coupon_features_raw ndarray (18, 4) float64 [ 2.20622181 -0.5967271          nan  1.77588242 -1.96755582 -0.9625032
 -2.03388658 -1.20412177 -0.43435377 -0.68315821  0.25790273 -1.74931894]
     ref_coupons ndarray (18, 8) float64 [-2.1656943  -0.65560068 -0.35719878  0.83194103 -0.43729383 -1.20772051
 -1.88813474  0.39372108  1.05547665  1.63198369 -2.68615654 -0.29342663]
     past_channel_raw ndarray (18, 6, 4) float64 [ 1.07188747  0.47932494 -0.44629593 -0.58984442  0.27024693 -0.34062793
 -0.16540076  0.8384971   0.20180138  1.03226766  0.72793931  0.16939995]
     hold_channel_raw ndarray (18, 4) float64 [1.06572423 0.68527473 0.4675851  0.56201091 0.83527035 0.43791807
 0.6188705  0.7923555  0.56964625 0.49893223 0.61498454 0.81103029]
     strain_0 ndarray (18,) float64 [ 0.12414996  0.02563229  0.16745263 -0.01358368 -0.05058692  0.16859631
 -0.14143006  0.10008253  0.01162863 -0.08521808 -0.00507484 -0.13841829]
     strain_curves_obs ndarray (18, 12) float64 [1.10194648 2.02683231 2.66558593 3.13763431 3.4881883  3.78745629
 4.04138681 4.27402879 4.47579232 4.66833822 4.84786419 5.03150656]
     instant_rate_obs ndarray (18,) float64 [4.5366947  3.80713313 2.81225084 4.25409097 3.24092377 3.97953584
 1.89715163 5.13746801 4.07598125 2.24943582 2.98865027 6.14659139]
```

### Turn 3 — `execute_in_remote_docker` (`17998 ms` LLM)
- **Rationale**: Print the timing and campaign metadata without dumping specimen arrays. The bridge residual determines whether an affine calibration is adequate; the time coordinates and sweep layout determine which hereditary kernels can be tested against the observed curves and rates.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb')); print('campaigns',len(p))
for i,it in enumerate(p):
 d=it['dataset']; print('\n',i,{k:str(v)[:160] for k,v in it.items() if k!='dataset'}); print('keys',list(d))
 for k,v in d.items():
  if not isinstance(v,dict): print(k,str(v)[:700])
 for k in ['train','test','chamber_verification_trials']:
  g=d.get(k); print(k, None if g is None else {f:np.shape(v) for f,v in g.items()})
 c=d['calibration_load_cells']; x,y=c['bridge_raw'],c['true_channel_loads']; A=np.column_stack((np.ones(len(x)),x)); b=np.linalg.lstsq(A,y,rcond=None)[0]; print('bridge affine rmse',np.sqrt(np.mean((A@b-y)**2)),'true channel spread',np.std(y)); print('sweeps', {k:(np.shape(v),np.asarray(v).ravel()[:8]) for k,v in c['single_channel_sweeps'].items()}); print('first calibrated histories',np.round((d['train']['past_channel_raw']@b[1:]+b[0]).sum(-1)[:3],3))
PY
```
- **Container Exit Code**: `0` (`317 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
campaigns 1

 0 {}
keys ['past_dt', 'eval_times', 'calibration_load_cells', 'train', 'test', 'chamber_verification_trials']
past_dt [0.35 0.35 0.4  0.45 0.5  0.55]
eval_times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
train {'lab_id': (72,), 'stratum': (72,), 'coupon_features_raw': (72, 4), 'ref_coupons': (72, 8), 'past_channel_raw': (72, 6, 4), 'hold_channel_raw': (72, 4), 'strain_0': (72,), 'strain_curves_obs': (72, 12), 'instant_rate_obs': (72,), 'aux_chamber_telemetry': (72, 4)}
test {'lab_id': (36,), 'stratum': (36,), 'coupon_features_raw': (36, 4), 'ref_coupons': (36, 8), 'past_channel_raw': (36, 6, 4), 'hold_channel_raw': (36, 4), 'strain_0': (36,), 'aux_chamber_telemetry': (36, 4)}
chamber_verification_trials {'lab_id': (18,), 'stratum': (18,), 'coupon_features_raw': (18, 4), 'ref_coupons': (18, 8), 'past_channel_raw': (18, 6, 4), 'hold_channel_raw': (18, 4), 'strain_0': (18,), 'strain_curves_obs': (18, 12), 'instant_rate_obs': (18,)}
bridge affine rmse 9.4353404964634e-05 true channel spread 0.8651654905398389
sweeps {'lab_id': ((24,), array([0, 0, 0, 0, 0, 0, 1, 1])), 'cmd_stress': ((24,), array([-1.49993043, -0.80003889, -0.20003809,  0.40020482,  1.00022142,
        1.59989719, -1.49988823, -0.80004839])), 'true_stress': ((24,), array([-1.5, -0.8, -0.2,  0.4,  1. ,  1.6, -1.5, -0.8]))}
first calibrated histories [[-0.771 -0.423  2.826  1.786  2.643  1.535]
 [-0.261 -0.628  0.179  1.767 -1.286  1.436]
 [ 2.104  2.752  1.178  1.647 -0.426  0.662]]
```

### Turn 4 — `execute_in_remote_docker` (`51180 ms` LLM)
- **Rationale**: Fit the bridge dynamically and integrate each constant-load history interval exactly. Compare timing and initial-state conventions against both observed outputs; if a different convention materially improves both, validate it out of fold before adopting it.
- **Alternatives Weighed**: `Per-channel nonlinear bridge calibration` (rejected: Affine calibration residuals are already four orders of magnitude below typical channel load variation.); `Single-curve relaxation primitive` (rejected: The task requires two shared relaxation times and a joint fit of curves, initial rates, and coupon drift.)

```bash
cd /holdout_task && cat > batch_solver.py <<'PY'
import numpy as np
from scipy.optimize import minimize


def _bridge(cal):
    raw = cal['bridge_raw']
    return np.linalg.lstsq(np.column_stack((np.ones(len(raw)), raw)),
                           cal['true_channel_loads'].sum(axis=1), rcond=None)[0]


def _align(groups, source):
    refs = np.concatenate([g['ref_coupons'] for g in groups])
    labs = np.concatenate([g['lab_id'] for g in groups])
    stats = {}
    for lab in np.unique(labs):
        ref = refs[labs == lab]
        stats[lab] = (np.median(np.mean(ref, axis=0)), np.median(np.std(ref, axis=0)))
    aligned = []
    for g in groups:
        x = g['coupon_features_raw'].copy()
        for lab in np.unique(g['lab_id']):
            m = g['lab_id'] == lab
            offset, gain = stats[lab]
            x[m] = (x[m] - offset) / gain
        aligned.append(x)
    g = groups[source]
    _, inverse, counts = np.unique(np.column_stack((g['lab_id'], g['stratum'])), axis=0,
                                   return_inverse=True, return_counts=True)
    w = 1 / counts[inverse, None]
    x = aligned[source]
    good = np.isfinite(x)
    fill = np.nansum(w * x, axis=0) / np.sum(w * good, axis=0)
    return [np.where(np.isfinite(x), x, fill) for x in aligned]


def _design(g, x, bridge, dt, t, taus, kind='chronological'):
    past = g['past_channel_raw'] @ bridge[1:] + bridge[0]
    hold = g['hold_channel_raw'] @ bridge[1:] + bridge[0]
    if kind == 'reversed':
        past, dt = past[:, ::-1], dt[::-1]
    age = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
    mass = np.exp(-age[1:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    memory = past @ mass
    if kind == 'first_extended':
        memory += past[:, :1] * np.exp(-age[0] / taus)
    elif kind == 'hold_extended':
        memory += hold[:, None] * np.exp(-age[0] / taus)
    elif kind == 'normalized':
        memory /= -np.expm1(-age[0] / taus)
    amplitude = hold[:, None] - memory
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((amplitude[:, None, :] * -np.expm1(-t[:, None] / taus),
                           drift[:, None, :] * t[None, :, None]), axis=-1)
    rate = np.column_stack((amplitude / taus, drift))
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _obs(g):
    return np.column_stack((g['strain_curves_obs'] - g['strain_0'][:, None], g['instant_rate_obs']))


def _fit(g, x, bridge, dt, t, kind='chronological'):
    y = _obs(g)
    # Third differences largely remove the smooth response, leaving the observation noise.
    dd = np.diff(y[:, :-1], n=3, axis=1)
    sc = max(0.002, np.median(np.abs(dd - np.median(dd))) / (0.67449 * np.sqrt(20)))
    scales = np.r_[np.full(len(t), sc), max(0.012, 2 * sc)]
    target = (y / scales).ravel()

    def profile(logtau, full=False):
        taus = np.sort(np.exp(logtau))
        a = (_design(g, x, bridge, dt, t, taus, kind) / scales[None, :, None]).reshape(-1, 7)
        theta = np.linalg.lstsq(a, target, rcond=None)[0]
        for _ in range(12):
            r = a @ theta - target
            w = 1 / np.sqrt(1 + (r / 2.5)**2)
            update = np.linalg.lstsq(a * w[:, None], target * w, rcond=None)[0]
            if np.linalg.norm(update - theta) < 1e-8 * (1 + np.linalg.norm(theta)):
                theta = update
                break
            theta = update
        loss = np.sum(np.log1p(((a @ theta - target) / 2.5)**2))
        return (taus, theta, scales) if full else loss

    bounds = [(np.log(.050001), np.log(24.999))] * 2
    best = min((minimize(profile, np.log(start), bounds=bounds, method='L-BFGS-B')
                for start in ((.42, 2.8), (.85, 7.), (.18, 1.25))), key=lambda r: r.fun)
    return profile(best.x, True)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    groups = [train, test] if pilot is None else [train, test, pilot]
    x = _align(groups, 0)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    source = 0 if pilot is None else 2
    taus, weights, _ = _fit(groups[source], x[source], bridge, dt, t)
    pred = _design(test, x[1], bridge, dt, t, taus) @ weights
    return dict(estimated_taus=taus, hereditary_weights=weights,
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
PY
cat > /tmp/rheo_probe.py <<'PY'
import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
tr,te,pi=[p[k] for k in ['train','test','chamber_verification_trials']]
dt,t=p['past_dt'],p['eval_times']; b=s._bridge(p['calibration_load_cells']); x=s._align([tr,te,pi],0)
for kind in ('chronological','reversed','first_extended','hold_extended','normalized'):
 for name,g,z in [('train',tr,x[0]),('pilot',pi,x[2])]:
  tau,theta,sc=s._fit(g,z,b,dt,t,kind); r=s._design(g,z,b,dt,t,tau,kind)@theta-s._obs(g)
  print(kind,name,'tau',tau.round(4),'theta',theta.round(5),'median abs',np.median(abs(r[:,:-1])).round(5),np.median(abs(r[:,-1])).round(5),'p90',np.quantile(abs(r[:,:-1]),.9).round(5),np.quantile(abs(r[:,-1]),.9).round(5),'scale',sc[[0,-1]].round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_probe.py
```
- **Container Exit Code**: `0` (`2309 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
chronological train tau [0.655  4.6361] theta [ 0.80703  0.65137  0.30336  0.126   -0.1364   0.17852 -0.13355] median abs 0.01381 0.00723 p90 0.04436 0.02051 scale [0.0168  0.03361]
chronological pilot tau [0.6591 6.1481] theta [ 0.81506  0.88403  0.27483  0.13205 -0.15738  0.20935 -0.16035] median abs 0.015 0.01792 p90 0.29378 0.13702 scale [0.00684 0.01368]
reversed train tau [0.1068 0.9755] theta [ 0.026    0.9307   0.4948   0.12357 -0.1151   0.21607 -0.17575] median abs 0.29993 0.64659 p90 0.96957 1.60101 scale [0.0168  0.03361]
reversed pilot tau [0.6891 7.7607] theta [ 0.77275  0.4009   0.52508  0.10436 -0.12544  0.36654 -0.02628] median abs 0.25305 0.59291 p90 1.24845 1.8446 scale [0.00684 0.01368]
first_extended train tau [0.2807 0.822 ] theta [ 0.04678  0.97107  0.45702  0.13267 -0.1409   0.17388 -0.12792] median abs 0.03711 0.04464 p90 0.14488 0.10148 scale [0.0168  0.03361]
first_extended pilot tau [0.708  3.6395] theta [ 0.9446  -0.03447  0.49238  0.11933 -0.15772  0.22931 -0.15472] median abs 0.03132 0.05095 p90 0.31662 0.15239 scale [0.00684 0.01368]
hold_extended train tau [0.504  1.1188] theta [ 0.36236  0.78798  0.44346  0.12909 -0.13904  0.17379 -0.12784] median abs 0.034 0.03439 p90 0.12489 0.0886 scale [0.0168  0.03361]
hold_extended pilot tau [0.6216 1.4969] theta [ 0.69389  0.44457  0.43241  0.12423 -0.16084  0.20622 -0.16481] median abs 0.02732 0.03873 p90 0.36043 0.14935 scale [0.00684 0.01368]
normalized train tau [0.504  1.1188] theta [ 0.36028  0.71084  0.44346  0.12909 -0.13904  0.17379 -0.12784] median abs 0.034 0.03439 p90 0.12489 0.0886 scale [0.0168  0.03361]
normalized pilot tau [0.6216 1.4969] theta [ 0.68331  0.36629  0.43241  0.12423 -0.16084  0.20622 -0.16481] median abs 0.02732 0.03873 p90 0.36043 0.14935 scale [0.00684 0.01368]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`21615 ms` LLM)
- **Rationale**: A residual proportional to time and present at onset indicates a coupon-level drift error, not an isolated measurement outlier. Check whether those residuals track missing features or lab alignment before changing the chamber fit.
- **Alternatives Weighed**: `Reverse the history or extend its initial state with the first or held load` (rejected: All tested alternatives increased median error on both strain and initial rate in both measurement sets.)

```bash
cat > /tmp/rheo_residual.py <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
for name,g,z in [('train',tr,x[0]),('pilot',pi,x[2])]:
 tau,w,sc=s._fit(g,z,b,dt,t); y=s._obs(g); r=y-s._design(g,z,b,dt,t,tau)@w
 slope=np.median(r[:,:-1]/t,axis=1); nonlinear=np.median(abs(r[:,:-1]-slope[:,None]*t),axis=1)
 print(name,'mad curve/rate',np.median(abs(r),axis=0).round(4),'rate minus slope',np.quantile(abs(r[:,-1]-slope),[.5,.9,1]).round(4))
 for j in (np.arange(len(r)) if name=='pilot' else np.argsort(abs(slope))[-12:]):
  print(j,g['lab_id'][j],g['stratum'][j],''.join('m' if v else '-' for v in np.isnan(g['coupon_features_raw'][j])),np.round([slope[j],r[j,-1],nonlinear[j],r[j,0],r[j,-2]],4))
for lab in np.unique(tr['lab_id']):
 print('ref lab',lab,[(len(a),round(np.median(a.mean(0)),3),round(np.median(a.std(0)),3)) for g in (tr,te,pi) for a in [g['ref_coupons'][g['lab_id']==lab]]]); print('feature nans', [(np.isnan(g['coupon_features_raw'][g['lab_id']==lab]).sum(0)).tolist() for g in (tr,te,pi)])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_residual.py
```
- **Container Exit Code**: `0` (`777 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
6]
45 0 0 -m-- [-0.0122 -0.0322  0.0093 -0.2761 -0.0328]
51 2 1 ---- [-0.0124 -0.0107  0.0071 -0.009  -0.0485]
68 3 0 m--- [-0.0125 -0.0249  0.007   0.013  -0.0597]
34 3 2 m--- [-0.0134 -0.04    0.0113  0.0053 -0.3495]
46 2 0 --m- [-0.0137 -0.0175  0.0088  0.0057 -0.0556]
65 1 2 ---- [ 0.0148  0.0136  0.0105 -0.0199  0.0479]
66 3 2 ---- [0.0155 0.0134 0.0074 0.0033 0.0637]
39 3 0 ---- [ 0.0157  0.0067  0.008  -0.0037  0.0899]
71 0 2 --m- [-0.016  -0.007   0.0051 -0.0183 -0.077 ]
22 1 1 --m- [-0.0204 -0.0003  0.0116 -0.0078 -0.0936]
54 2 1 --m- [-0.0214 -0.0242  0.0087  0.0015 -0.1035]
21 3 1 m--- [-0.0253 -0.0205  0.0088 -0.0113 -0.1022]
pilot mad curve/rate [0.0061 0.0062 0.0118 0.0162 0.0134 0.0158 0.0165 0.0204 0.0177 0.0216
 0.0302 0.0366 0.0179] rate minus slope [0.0072 0.0139 0.019 ]
0 2 0 --m- [-0.002   0.0039  0.0046  0.0069 -0.0301]
1 0 2 ---- [-0.0025  0.0106  0.0047 -0.0003 -0.0218]
2 0 0 ---- [0.0072 0.0131 0.0051 0.0069 0.0125]
3 3 2 ---- [-0.235  -0.216   0.0047 -0.0575 -0.9645]
4 0 2 ---- [-0.0046  0.0005  0.0065 -0.0004 -0.0508]
5 1 2 ---- [ 0.0072  0.0185  0.0054  0.0034 -0.0003]
6 2 0 m--- [0.0027 0.0094 0.0038 0.0036 0.004 ]
7 1 1 --m- [-0.0689 -0.0629  0.0042 -0.0105 -0.2929]
8 0 2 ---- [ 0.0006  0.0116  0.0035  0.0063 -0.005 ]
9 3 2 ---- [-0.1588 -0.1558  0.0053 -0.0392 -0.6408]
10 1 2 ---- [0.0077 0.0173 0.0071 0.0052 0.0021]
11 1 1 ---- [-0.0548 -0.0463  0.0056 -0.0115 -0.2529]
12 2 1 ---- [0.067  0.0759 0.0029 0.0159 0.2495]
13 1 2 ---- [ 0.0039  0.008   0.0042  0.0059 -0.0034]
14 0 2 ---- [-0.0058  0.0019  0.006   0.0046 -0.0431]
15 1 0 ---- [-0.0245 -0.0204  0.0058 -0.0054 -0.1262]
16 0 0 ---- [ 0.0051  0.0211  0.0057  0.002  -0.011 ]
17 2 1 ---m [0.1278 0.129  0.0052 0.0361 0.4905]
ref lab 0 [(24, -0.471, 1.172), (4, -0.723, 1.231), (6, -0.992, 1.069)]
feature nans [[2, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]]
ref lab 1 [(15, 0.498, 0.77), (4, 0.563, 0.689), (6, 0.305, 0.776)]
feature nans [[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
ref lab 2 [(20, -0.388, 1.396), (15, -0.313, 1.308), (4, -0.73, 0.951)]
feature nans [[1, 1, 3, 2], [0, 2, 0, 0], [1, 0, 1, 1]]
ref lab 3 [(13, 0.658, 0.807), (13, 0.886, 0.818), (2, 0.849, 0.426)]
feature nans [[4, 0, 0, 1], [1, 0, 2, 0], [0, 0, 0, 0]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`23050 ms` LLM)
- **Rationale**: Compare drift residuals within lab–stratum cells and against standardized features. Repeated offsets would support a cell correction; variation within a cell would favor testing feature interactions before adding a correction.
- **Alternatives Weighed**: `Increase robustness against isolated curve outliers` (rejected: The large pilot residuals persist across horizons and match the independently observed onset rate, so downweighting isolated points cannot explain them.)

```bash
cat > /tmp/rheo_cells.py <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
for name,g,z in [('bench',tr,x[0]),('chamber',pi,x[2])]:
 tau,w,_=s._fit(g,z,b,dt,t)
 a=s._design(g,z,b,dt,t,tau); r=s._obs(g)-a@w
 slope=np.median(np.column_stack((r[:,:-1]/t,r[:,-1])),axis=1)
 print(name,'residual drift by lab and stratum:')
 for lab in np.unique(g['lab_id']):
  for st in np.unique(g['stratum']):
   idx=np.flatnonzero((g['lab_id']==lab)&(g['stratum']==st))
   print('cell',lab,st,'slopes',slope[idx].round(4),'X',z[idx].round(2).tolist() if name=='chamber' else '')
 print('abs slope by stratum',[(st,np.median(abs(slope[g['stratum']==st])),np.max(abs(slope[g['stratum']==st]))) for st in np.unique(g['stratum'])])
 if 'aux_chamber_telemetry' in g:
  tele=g['aux_chamber_telemetry']; print('telemetry/feature correlations',np.corrcoef(np.column_stack((tele,z,slope)).T)[:4,4:].round(2))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_cells.py
```
- **Container Exit Code**: `0` (`732 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
01   0.0047 -0.0031  0.0023  0.0047 -0.0134 -0.0043 -0.0027  0.0103
  0.0041] X 
cell 0 1 slopes [-0.0001 -0.0071 -0.0102  0.0017 -0.0081 -0.0006  0.0047] X 
cell 0 2 slopes [ 0.001   0.0065 -0.0008  0.0081 -0.0006  0.0065 -0.0159] X 
cell 1 0 slopes [-0.0022 -0.0003] X 
cell 1 1 slopes [ 0.01    0.0019 -0.0176 -0.0054  0.0055] X 
cell 1 2 slopes [ 0.0012 -0.0033 -0.006  -0.0033  0.0003 -0.0051  0.0014  0.0147] X 
cell 2 0 slopes [ 0.0064  0.0103 -0.0047  0.0039  0.003  -0.0139  0.0106 -0.0048] X 
cell 2 1 slopes [ 0.0046  0.0069  0.0044  0.0054 -0.0121 -0.0029 -0.0228] X 
cell 2 2 slopes [ 0.0059 -0.0068  0.0038  0.0037  0.0023] X 
cell 3 0 slopes [ 0.0012  0.0157 -0.0035 -0.0104 -0.0131] X 
cell 3 1 slopes [-0.0067 -0.0251 -0.0073  0.0008] X 
cell 3 2 slopes [ 0.0086  0.0068 -0.0138  0.015 ] X 
abs slope by stratum [(0, 0.0046692715427898435, 0.015660623699438127), (1, 0.005488942426303647, 0.025089733611939835), (2, 0.005511545949675062, 0.015918390349077917)]
telemetry/feature correlations [[ 0.14  0.02  0.15 -0.07  0.01]
 [ 0.07  0.14 -0.06  0.07 -0.02]
 [ 0.16 -0.04  0.04  0.03  0.14]
 [-0.04  0.1  -0.22  0.04 -0.16]]
chamber residual drift by lab and stratum:
cell 0 0 slopes [0.0075 0.0055] X [[0.12, -0.08, 0.67, -0.93], [1.6, -1.39, 0.34, 0.18]]
cell 0 1 slopes [] X []
cell 0 2 slopes [-0.002  -0.0037  0.0006 -0.0051] X [[-1.1, -0.3, -1.16, -0.5], [-0.5, 0.77, 0.18, -1.35], [0.21, 1.31, -0.05, -1.58], [-0.69, 0.76, 0.07, -0.78]]
cell 1 0 slopes [-0.0237] X [[-0.29, 0.28, 0.19, -0.65]]
cell 1 1 slopes [-0.0689 -0.0546] X [[0.47, -1.36, 0.13, 0.16], [0.14, -1.39, 1.71, 0.25]]
cell 1 2 slopes [0.0074 0.0079 0.0045] X [[-0.9, 1.7, -0.04, -0.53], [0.18, 0.09, -0.08, 0.67], [-0.66, -1.0, -0.6, 0.34]]
cell 2 0 slopes [-0.0018  0.0027] X [[1.99, -0.13, 0.13, 1.66], [0.13, -0.38, -1.02, -1.42]]
cell 2 1 slopes [0.067  0.1288] X [[1.04, 0.67, 0.75, -0.75], [-0.41, -1.81, -0.4, 0.18]]
cell 2 2 slopes [] X []
cell 3 0 slopes [] X []
cell 3 1 slopes [] X []
cell 3 2 slopes [-0.2345 -0.158 ] X [[0.42, 1.11, -1.45, -0.78], [-0.25, 1.34, 0.36, 0.28]]
abs slope by stratum [(0, 0.005475221657716808, 0.023685130147157053), (1, 0.06797244380517337, 0.1287765463282559), (2, 0.005103915754737475, 0.23451173361881258)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`33353 ms` LLM)
- **Rationale**: Estimate relaxation independently of baseline drift and inspect whether affected cells show offsets or gain changes. If the projected spectrum agrees with the existing fit, retain the better-conditioned estimate and focus validation on drift transfer.
- **Alternatives Weighed**: `Replace the Prony memory model` (rejected: The observed discrepancy is mostly linear in hold time, whereas the tested history changes degraded both bench and pilot fits.)
- **ATMS NOGOOD DECLARED**: Refuted `A global linear coupon drift accounts for all chamber trials.` | Witness: `Two complete-feature coupons in chamber cell (lab 3, stratum 2) show drift residuals of −0.235 and −0.158, versus a maximum absolute bench drift residual of 0.025; their onset residuals show the same discrepancy.` | Blanket Diagnosis: `The discrepancies are proportional to hold time and agree with independently measured onset rates, concentrating the mismatch in the drift term rather than isolated curve noise.` | Sublated Pivot: `Eliminate coupon-level drift to estimate the common relaxation response, then test additive and multiplicative chamber drift corrections out of fold.`

```bash
cat > /tmp/rheo_transfer.py <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
from scipy.optimize import least_squares

def predict(g,z,tau,w): return s._design(g,z,b,dt,t,tau)@w

def select(g,idx): return {k:v[idx] for k,v in g.items()}

def fit_projected(g,z):
 y=s._obs(g); h=np.r_[t,1.]; pair=np.arange(len(t)-1)
 def unpack(q): return np.exp(q[:2]),q[2:4]
 def fun(q):
  tau,w=unpack(q); r=y-s._design(g,z,b,dt,t,tau)[:,:,:2]@w
  # Adjacent normalized contrasts eliminate each coupon's unknown drift.
  rc=(r[:,pair+1]*t[pair]-r[:,pair]*t[pair+1])/np.hypot(t[pair],t[pair+1]); rate=(r[:,-1]*t[0]-r[:,0])/np.hypot(t[0],1)
  return np.column_stack((rc,rate)).ravel()
 q=least_squares(fun,np.r_[np.log([.65,5]),.8,.8],bounds=([np.log(.05001)]*2+[-10]*2,[np.log(24.999)]*2+[10]*2),loss='cauchy',f_scale=.012,max_nfev=150).x
 return unpack(q)

def drift_obs(g,z,tau,w):
 y=s._obs(g)-s._design(g,z,b,dt,t,tau)[:,:,:2]@w
 return np.median(np.column_stack((y[:,:-1]/t,y[:,-1])),axis=1)

bt,bw,_=s._fit(tr,x[0],b,dt,t)
for nam,g,z in [('bench',tr,x[0]),('pilot',pi,x[2])]:
 tau,w=fit_projected(g,z); do=drift_obs(g,z,tau,w); base=bw[2]+z@bw[3:]; print(nam,'projected tau/weight',tau,w)
 if nam=='pilot':
  for l,u,d,v,zz in zip(g['lab_id'],g['stratum'],do,base,z): print(l,u,'observed, bench, diff, ratio',np.round([d,v,d-v,d/v],4),zz.round(2))
print('test cell counts',[(l,u,int(np.sum((te['lab_id']==l)&(te['stratum']==u)))) for l in np.unique(te['lab_id']) for u in np.unique(te['stratum'])]); print('telemetry train/test summaries:')
for g in [tr,te]:
 a=g['aux_chamber_telemetry'];print('global',a.mean(0).round(3),a.std(0).round(3),'first',a[:4].round(3));print('labs',[(int(l),a[g['lab_id']==l].mean(0).round(3)) for l in np.unique(g['lab_id'])]);print('strata',[(int(l),a[g['stratum']==l].mean(0).round(3)) for l in np.unique(g['stratum'])])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_transfer.py; ls -la /pretrain_task /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`606 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ratio [ 0.2531  0.3536 -0.1006  0.7156] [ 0.21  1.31 -0.05 -1.58]
3 2 observed, bench, diff, ratio [-0.1684  0.1158 -0.2841 -1.4544] [-0.25  1.34  0.36  0.28]
1 2 observed, bench, diff, ratio [ 0.0625  0.2092 -0.1467  0.2987] [ 0.18  0.09 -0.08  0.67]
1 1 observed, bench, diff, ratio [ 0.6483  0.7803 -0.132   0.8308] [ 0.14 -1.39  1.71  0.25]
2 1 observed, bench, diff, ratio [ 0.5647  0.5769 -0.0123  0.9787] [ 1.04  0.67  0.75 -0.75]
1 2 observed, bench, diff, ratio [ 0.0807  0.2042 -0.1235  0.3953] [-0.66 -1.   -0.6   0.34]
0 2 observed, bench, diff, ratio [ 0.1114  0.2286 -0.1173  0.4871] [-0.69  0.76  0.07 -0.78]
1 0 observed, bench, diff, ratio [ 0.1932  0.3492 -0.156   0.5533] [-0.29  0.28  0.19 -0.65]
0 0 observed, bench, diff, ratio [ 0.6096  0.7316 -0.1219  0.8333] [ 1.6  -1.39  0.34  0.18]
2 1 observed, bench, diff, ratio [0.4493 0.4036 0.0457 1.1131] [-0.41 -1.81 -0.4   0.18]
test cell counts [(0, 0, 0), (0, 1, 2), (0, 2, 2), (1, 0, 1), (1, 1, 3), (1, 2, 0), (2, 0, 6), (2, 1, 5), (2, 2, 4), (3, 0, 6), (3, 1, 5), (3, 2, 2)]
telemetry train/test summaries:
global [ 0.17   0.073  0.157 -0.088] [0.979 1.063 1.068 0.801] first [[ 1.241  0.783  0.051  0.507]
 [-0.144  0.855  1.894 -0.165]
 [ 0.307  1.824 -0.935  0.424]
 [ 0.263 -0.357  0.023  0.255]]
labs [(0, array([ 0.16 ,  0.28 ,  0.284, -0.223])), (1, array([0.227, 0.163, 0.382, 0.069])), (2, array([0.088, 0.161, 0.049, 0.21 ])), (3, array([ 0.248, -0.551, -0.171, -0.478]))]
strata [(0, array([ 0.274, -0.006,  0.054, -0.155])), (1, array([-0.024, -0.036, -0.075,  0.021])), (2, array([ 0.246,  0.259,  0.486, -0.123]))]
global [-0.005  0.089  0.112 -0.111] [1.148 1.009 0.925 1.025] first [[ 0.513 -0.57   0.872 -0.083]
 [ 1.709 -0.373 -1.349 -1.038]
 [-0.898  1.176 -1.098  0.678]
 [-2.68   0.511  0.552 -1.422]]
labs [(0, array([-0.466,  0.288,  0.884,  0.683])), (1, array([ 1.352, -0.316,  1.032,  0.063])), (2, array([ 0.079, -0.223, -0.111, -0.551])), (3, array([-0.377,  0.512, -0.151,  0.099]))]
strata [(0, array([-0.152,  0.392, -0.087,  0.158])), (1, array([ 0.302, -0.133,  0.334, -0.325])), (2, array([-0.341,  0.012,  0.02 , -0.148]))]
/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 12:19 ..

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`41290 ms` LLM)
- **Rationale**: Compare global, lab, cell, and partially pooled drift fits under coupon, cell, and lab holdouts. Coupon holdouts measure interpolation, while cell and lab holdouts expose unsupported extrapolation. Adopt a correction only if it improves the observed bottleneck without materially degrading transfer to unseen groups.
- **Alternatives Weighed**: `Adopt the adjacent-contrast spectrum directly` (rejected: The contrast fit shifts the bench slow relaxation time from 4.64 to 2.70, so it is not a stable standalone estimator.); `Use an unpooled lookup for each lab–stratum cell` (rejected: Several test cells have no pilot observations, and other cells have only one or two.)

```bash
cat > /tmp/rheo_cv.py <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
from scipy.optimize import minimize


def source_predict(g,z,tau,w):
    return s._design(g,z,b,dt,t,tau)@w


def basis(g,z,mode):
    l,u=g['lab_id'],g['stratum']
    a=(l[:,None]==np.unique(tr['lab_id'])).astype(float)
    c=(u[:,None]==np.unique(tr['stratum'])).astype(float)
    cell=(a[:,:,None]*c[:,None,:]).reshape(len(l),-1)
    main=np.column_stack((np.ones(len(l)),z))
    if mode=='global': return main,np.array([.10]+[.08]*4)
    if mode=='lab': return np.column_stack((main,a)),np.array([.10]+[.08]*4+[.09]*a.shape[1])
    if mode=='cell': return np.column_stack((main,cell)),np.array([.10]+[.08]*4+[.12]*cell.shape[1])
    return np.column_stack((main,a,c,cell)),np.array([.10]+[.08]*4+[.065]*(a.shape[1]+c.shape[1])+[.10]*cell.shape[1])

bt,bw,_=s._fit(tr,x[0],b,dt,t)
sc=np.r_[np.full(len(t),.009),.014]; h=np.r_[t,1.]

def fit(ix,mode,nuisance=.025):
    z=x[2][ix]; g={k:v[ix] for k,v in pi.items()}
    D,prior=basis(g,z,mode); n=len(z)
    U=np.column_stack((D,np.eye(n)*nuisance))
    U[:,:len(prior)]*=prior
    Q=(U[:,None,:]*(h/sc)[None,:,None]).reshape(-1,U.shape[1]); y=s._obs(g)
    y=(y-(bw[2]+z@bw[3:])[:,None]*h)/sc
    reg=np.diag(np.r_[[0,0],np.ones(U.shape[1])])
    def obj(p,ret=False):
        tau=np.sort(np.exp(p)); A=s._design(g,z,b,dt,t,tau)[:,:,:2]/sc[None,:,None]
        A=np.column_stack((A.reshape(-1,2),Q)); yy=y.ravel()
        beta=np.linalg.lstsq(np.vstack((A,reg)),np.r_[yy,np.zeros(len(reg))],rcond=None)[0]
        for j in range(11):
            r=A@beta-yy; w=1/(1+(r/2.5)**2)
            beta=np.linalg.solve(A.T@(w[:,None]*A)+reg, A.T@(w*yy))
        if ret: return tau,beta[:2],beta[2:2+len(prior)]*prior
        return np.sum(6.25*np.log1p(((A@beta-yy)/2.5)**2))+np.sum((reg@beta)**2)
    bounds=[(np.log(.05001),np.log(24.999))]*2
    p=min((minimize(obj,np.log(start),bounds=bounds,method='L-BFGS-B') for start in [bt,[.65,7.]]),key=lambda r:r.fun).x
    return obj(p,True)

for mode in ['baseline','global','lab','cell','hier']:
    for scheme in ['coupon','cell','lab']:
        tags=np.arange(len(pi['lab_id'])) if scheme=='coupon' else pi['lab_id']*(3 if scheme=='cell' else 1)+(pi['stratum'] if scheme=='cell' else 0)
        pred=np.zeros_like(s._obs(pi))
        for k in np.unique(tags):
            ix,js=tags!=k,tags==k
            g={key:v[js] for key,v in pi.items()}; z=x[2][js]
            if mode=='baseline':
                a,w,_=s._fit({key:v[ix] for key,v in pi.items()},x[2][ix],b,dt,t)
                pred[js]=source_predict(g,z,a,w)
            else:
                a,w,c=fit(ix,mode); D,_=basis(g,z,mode)
                pred[js]=s._design(g,z,b,dt,t,a)[:,:,:2]@w+(bw[2]+z@bw[3:]+D@c)[:,None]*h
        r=abs(pred-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
        print(mode,scheme,'mean',np.round([c.mean(),v.mean()],4),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'max cell mae',round(max(c[(pi['lab_id']*3+pi['stratum'])==i].mean() for i in np.unique(pi['lab_id']*3+pi['stratum'])),4),flush=True)
    if mode!='baseline': print('full fit',fit(np.ones(len(pi['lab_id']),bool),mode),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_cv.py
```
- **Container Exit Code**: `1` (`6305 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline coupon mean [0.1369 0.0626] p90 [0.3253 0.1547] max cell mae 0.4513
baseline cell mean [0.1385 0.0674] p90 [0.3512 0.1604] max cell mae 0.4631
baseline lab mean [0.2258 0.1089] p90 [0.5699 0.2276] max cell mae 0.4631
global coupon mean [0.163  0.0764] p90 [0.4244 0.1769] max cell mae 0.4439
global cell mean [0.1821 0.0857] p90 [0.4497 0.1864] max cell mae 0.4217
global lab mean [0.1796 0.0844] p90 [0.4867 0.1785] max cell mae 0.4217
full fit (array([0.64582496, 2.66388151]), array([0.77137364, 0.41336386]), array([ 0.06310459,  0.00404773, -0.05436236,  0.05266831, -0.04379382]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```
- **Stderr (tail)**:
```text
ze
    res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_lbfgsb_py.py", line 407, in _minimize_lbfgsb
    f, g = func_and_grad(x)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_differentiable_functions.py", line 296, in fun_and_grad
    self._update_fun()
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_differentiable_functions.py", line 262, in _update_fun
    self._update_fun_impl()
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_differentiable_functions.py", line 163, in update_fun
    self.f = fun_wrapped(self.x)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/scipy/optimize/_differentiable_functions.py", line 145, in fun_wrapped
    fx = fun(np.copy(x), *args)
         ^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/rheo_cv.py", line 37, in obj
    beta=np.linalg.solve(A.T@(w[:,None]*A)+reg, A.T@(w*yy))
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/linalg/linalg.py", line 409, in solve
    r = gufunc(a, b, signature=signature, extobj=extobj)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/linalg/linalg.py", line 112, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
```

### Turn 9 — `execute_in_remote_docker` (`20375 ms` LLM)
- **Rationale**: The optimizer can try coincident relaxation times, making the two memory columns identical. A negligible penalty on their weights lets the diagnostic finish. Compare the remaining models on the same folds, and check whether the within-cell slope differences track held stress or baseline drift before choosing an additive or multiplicative correction.
- **Alternatives Weighed**: `Adopt the global regularized challenger` (rejected: It improves lab holdouts but raises coupon-held-out strain P90 from 0.325 to 0.424; the cell-aware models have not yet completed the comparison.)

```bash
sed -i 's/np.r_\[\[0,0\],np.ones(U.shape\[1\])\]/np.r_[[1e-7,1e-7],np.ones(U.shape[1])]/; s/np.vstack((A,reg))/np.vstack((A,np.sqrt(reg)))/; s/np.sum((reg@beta)\*\*2)/np.sum((np.sqrt(reg)@beta)**2)/' /tmp/rheo_cv.py
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_cv.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
tau,w,_=s._fit(pi,x[2],b,dt,t); r=s._obs(pi)-s._design(pi,x[2],b,dt,t,tau)@w
slope=np.median(r[:,:-1]/t,axis=1)
for i in range(len(r)):
 print(int(pi['lab_id'][i]),int(pi['stratum'][i]),np.round([slope[i],pi['hold_channel_raw'][i]@b[1:]+b[0],pi['strain_0'][i],w[2]+x[2][i]@w[3:]],3))
PY
```
- **Container Exit Code**: `0` (`12030 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
b coupon mean [0.1224 0.057 ] p90 [0.292 0.118] max cell mae 0.2667
lab cell mean [0.2069 0.0969] p90 [0.4759 0.1788] max cell mae 0.4631
lab lab mean [0.1833 0.087 ] p90 [0.4964 0.1977] max cell mae 0.4631
full fit (array([0.65630929, 4.95213015]), array([0.80641974, 0.76075419]), array([-0.04510031, -0.02294381, -0.02036425,  0.04153046, -0.01333519,
        0.03109966, -0.00532492,  0.09319075, -0.15549675]))
cell coupon mean [0.0771 0.0353] p90 [0.1735 0.0593] max cell mae 0.1243
cell cell mean [0.1476 0.0697] p90 [0.3802 0.1506] max cell mae 0.4145
cell lab mean [0.1603 0.0754] p90 [0.3909 0.1618] max cell mae 0.4145
full fit (array([0.65375064, 4.07734371]), array([0.79834654, 0.60704118]), array([-0.00128774, -0.00151274, -0.03069954,  0.04267387, -0.02362942,
        0.02467718,  0.        ,  0.01886218,  0.00360045, -0.05437383,
        0.03020605,  0.02097667,  0.10647134,  0.        ,  0.        ,
        0.        , -0.15227438]))
hier coupon mean [0.0781 0.0356] p90 [0.1761 0.0631] max cell mae 0.1279
hier cell mean [0.1698 0.0801] p90 [0.4296 0.1589] max cell mae 0.4249
hier lab mean [0.1871 0.0882] p90 [0.4353 0.1719] max cell mae 0.4249
full fit (array([0.65386901, 4.11091142]), array([0.79872532, 0.61253231]), array([-0.00451176, -0.00288996, -0.02921763,  0.04163043, -0.02247507,
        0.01358484, -0.0023587 ,  0.02822987, -0.04136223,  0.00370481,
        0.00799838, -0.01360941,  0.01223525,  0.        ,  0.01991821,
        0.00363189, -0.05498355,  0.04576892, -0.00709836,  0.07391462,
        0.        ,  0.        ,  0.        , -0.09789877]))
2 0 [-2.000e-03  2.781e+00  1.240e-01  3.200e-01]
0 2 [-2.000e-03  2.684e+00  2.600e-02  1.400e-02]
0 0 [0.007 2.495 0.167 0.592]
3 2 [-0.235  4.078 -0.014 -0.022]
0 2 [-0.005  2.921 -0.051  0.342]
1 2 [ 0.007  3.537  0.169 -0.035]
2 0 [ 0.003  2.459 -0.141  0.366]
1 1 [-0.069  3.799  0.1    0.552]
0 2 [1.000e-03 2.954e+00 1.200e-02 3.400e-01]
3 2 [-0.159  2.451 -0.085  0.061]
1 2 [ 0.008  3.609 -0.005  0.159]
1 1 [-0.055  4.03  -0.138  0.827]
2 1 [ 0.067  3.017 -0.097  0.584]
1 2 [ 0.004  3.033 -0.111  0.165]
0 2 [-0.006  2.918 -0.112  0.203]
1 0 [-0.024  3.927  0.023  0.336]
0 0 [0.005 4.728 0.092 0.748]
2 1 [ 0.128  2.494 -0.128  0.394]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`43257 ms` LLM)
- **Rationale**: Both observed outputs support a cell correction, but the slow branch is sensitive to how much drift freedom the fit allows. Compare fixed and softly pooled spectra, and test whether load or drift-gain interactions improve held-out cells enough to justify adding them.
- **Alternatives Weighed**: `Use lab-level offsets alone` (rejected: They improve prediction within measured cells but raise cell-held-out strain P90 from 0.351 to 0.476.)

```bash
cat > /tmp/rheo_factor.py <<'PY'
exec(open('/tmp/rheo_cv.py').read().split('for mode in [')[0])
from scipy.optimize import least_squares

def prep(g,z):
    l,u=g['lab_id'],g['stratum']; hold=g['hold_channel_raw']@b[1:]+b[0]
    cell=(l[:,None]*3+u[:,None]==np.arange(12)).astype(float)
    return hold,cell

def models(g,z,mode):
    hold,cell=prep(g,z); one=np.ones((len(z),1)); base=bw[2]+z@bw[3:]
    columns=[one,z,cell]; prior=[.1]+[.08]*4+[.12]*12
    if mode=='offset': columns=[one,cell]; prior=[.1]+[.12]*12
    if mode=='load': columns += [(hold[:,None]-3),cell*(hold[:,None]-3)]; prior += [.045]+[.035]*12
    if mode=='gain': columns += [cell*(base[:,None]-.3)]; prior += [.25]*12
    return np.column_stack(columns)*prior

def factor(ix, mode, spec):
    g={k:v[ix] for k,v in pi.items()}; z=x[2][ix]; n=len(z)
    D=models(g,z,mode); U=np.column_stack((D,np.eye(n)*.022)); Q=(U[:,None,:]*(h/sc)[None,:,None]).reshape(-1,U.shape[1]); yy=((s._obs(g)-(bw[2]+z@bw[3:])[:,None]*h)/sc).ravel()
    def obj(p,ret=False):
        tau=np.sort(np.exp(p)); R=(s._design(g,z,b,dt,t,tau)[:,:,:2]/sc[None,:,None]).reshape(-1,2)
        if spec=='fixed':
            A=Q; y=yy-R@bw[:2]; prec=np.ones(A.shape[1]); center=np.zeros(A.shape[1]); regtau=0
        else:
            A=np.column_stack((R,Q)); y=yy; prec=np.r_[[1/.06**2,1/.25**2],np.ones(Q.shape[1])]; center=np.r_[bw[:2],np.zeros(Q.shape[1])]; regtau=np.sum(((p-np.log(bt))/[.12,.35])**2)
        matrix=np.diag(prec); rhs=prec*center
        beta=np.linalg.solve(A.T@A+matrix,A.T@y+rhs)
        for _ in range(20):
            r=A@beta-y; w=1/(1+(r/2.5)**2); new=np.linalg.solve(A.T@(w[:,None]*A)+matrix,A.T@(w*y)+rhs)
            if np.linalg.norm(new-beta)<1e-7: beta=new; break
            beta=new
        if ret: return tau,bw[:2] if spec=='fixed' else beta[:2],beta[:D.shape[1]] if spec=='fixed' else beta[2:2+D.shape[1]]
        return np.sum(6.25*np.log1p(((A@beta-y)/2.5)**2))+np.dot(prec,(beta-center)**2)+regtau
    p=np.log(bt)
    if spec!='fixed': p=min((minimize(obj,q,bounds=[(np.log(.05001),np.log(24.999))]*2,method='L-BFGS-B') for q in (p,np.log([.65,7.]))),key=lambda v:v.fun).x
    return obj(p,True)

def pred(g,z,par,mode):
    tau,w,c=par
    return s._design(g,z,b,dt,t,tau)[:,:,:2]@w+(bw[2]+z@bw[3:]+models(g,z,mode)@c)[:,None]*h
cell=pi['lab_id']*3+pi['stratum']; allix=np.ones(len(cell),bool)
for mode,spec in [('offset','fixed'),('offset','soft'),('cell','fixed'),('cell','soft'),('load','soft'),('gain','soft')]:
    pars=factor(allix,mode,spec); print(mode,spec,'full',np.round(np.r_[pars[0],pars[1]],4),'correction range',np.round((models(pi,x[2],mode)@pars[2]).min(),4),np.round((models(pi,x[2],mode)@pars[2]).max(),4),flush=True)
    for tags in (np.arange(len(cell)),cell,pi['lab_id']):
        yp=np.zeros_like(s._obs(pi))
        for k in np.unique(tags):
            ix=tags==k; yp[ix]=pred({key:v[ix] for key,v in pi.items()},x[2][ix],factor(~ix,mode,spec),mode)
        r=abs(yp-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
        print('folds',len(np.unique(tags)),'mean',np.round([c.mean(),v.mean()],4),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell mean',np.round([max(c[cell==i].mean() for i in np.unique(cell)),max(v[cell==i].mean() for i in np.unique(cell))],4),flush=True)
print('bench params',bt,bw)
R=s._obs(pi)-source_predict(pi,x[2],bt,bw); L=x[2]@bw[3:]+bw[2]; H,_=prep(pi,x[2]); sl=np.median(np.column_stack((R[:,:-1]/t,R[:,-1])),axis=1)
for i in np.argsort(cell): print('bench discrepancy cell',cell[i],np.round([sl[i],L[i],H[i]],4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_factor.py
```
- **Container Exit Code**: `0` (`6611 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rection range -0.2392 0.1352
folds 18 mean [0.0687 0.0332] p90 [0.155  0.0595] worst cell mean [0.121  0.0614]
folds 8 mean [0.1335 0.0636] p90 [0.3442 0.1514] worst cell mean [0.399  0.1891]
folds 4 mean [0.1512 0.0725] p90 [0.387  0.1628] worst cell mean [0.399  0.1891]
cell soft full [0.6546 4.2763 0.8009 0.6378] correction range -0.2425 0.1343
folds 18 mean [0.0748 0.0348] p90 [0.1678 0.059 ] worst cell mean [0.1271 0.0596]
folds 8 mean [0.1412 0.0671] p90 [0.3662 0.1507] worst cell mean [0.4116 0.1921]
folds 4 mean [0.1544 0.0728] p90 [0.3852 0.1623] worst cell mean [0.4116 0.1921]
load soft full [0.6546 4.276  0.8009 0.6378] correction range -0.2536 0.1366
folds 18 mean [0.0575 0.0263] p90 [0.1455 0.0583] worst cell mean [0.1119 0.0499]
folds 8 mean [0.1327 0.0621] p90 [0.356  0.1523] worst cell mean [0.4073 0.1901]
folds 4 mean [0.1464 0.0687] p90 [0.367  0.1584] worst cell mean [0.4073 0.1901]
gain soft full [0.6545 4.2396 0.8006 0.6314] correction range -0.2468 0.1422
folds 18 mean [0.072  0.0345] p90 [0.1802 0.0668] worst cell mean [0.148  0.0669]
folds 8 mean [0.1339 0.0639] p90 [0.3362 0.1529] worst cell mean [0.4324 0.2019]
folds 4 mean [0.1416 0.0673] p90 [0.3826 0.1584] worst cell mean [0.4324 0.2019]
bench params [0.65500978 4.63609879] [ 0.80703122  0.65137105  0.30335985  0.12599707 -0.13639716  0.17852088
 -0.1335525 ]
bench discrepancy cell 0 [0.0591 0.5728 2.4946]
bench discrepancy cell 0 [0.08   0.7316 4.7284]
bench discrepancy cell 2 [0.0262 0.3483 2.9208]
bench discrepancy cell 2 [-0.0194  0.0653  2.6845]
bench discrepancy cell 2 [0.0069 0.2286 2.9185]
bench discrepancy cell 2 [0.025  0.3536 2.9538]
bench discrepancy cell 3 [0.0112 0.3492 3.9266]
bench discrepancy cell 4 [0.0408 0.7803 4.0303]
bench discrepancy cell 4 [-0.0154  0.5495  3.7987]
bench discrepancy cell 5 [0.0035 0.2042 3.0327]
bench discrepancy cell 5 [-0.0047  0.0216  3.5369]
bench discrepancy cell 5 [0.0069 0.2092 3.609 ]
bench discrepancy cell 6 [0.0243 0.3794 2.459 ]
bench discrepancy cell 6 [-0.0221  0.3739  2.7806]
bench discrepancy cell 7 [0.1511 0.4036 2.4944]
bench discrepancy cell 7 [0.1152 0.5769 3.0168]
bench discrepancy cell 11 [-0.18    0.1158  2.4507]
bench discrepancy cell 11 [-0.2552  0.0505  4.0775]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`25011 ms` LLM)
- **Rationale**: Cross the best drift structure with the better-conditioned fixed spectrum. Check test load coverage because sparse cell-specific load slopes should not extrapolate unchecked beyond pilot loads.
- **Alternatives Weighed**: `Use cell-specific drift gains` (rejected: They increased coupon-held-out worst-cell strain MAE from 0.121 to 0.148 and cell-held-out worst-cell MAE from 0.399 to 0.432.)

```bash
cat > /tmp/rheo_loadcheck.py <<'PY'
exec(open('/tmp/rheo_factor.py').read().split("for mode,spec in")[0])
for mode,spec in [('load','fixed'),('cell','fixed'),('load','soft')]:
 pars=factor(allix,mode,spec); print(mode,spec,'drift coefficients',np.round(pars[2]*np.r_[[.1]+[.08]*4+[.12]*12, ([.045]+[.035]*12) if mode=='load' else []],4));
 for tags in (np.arange(len(cell)),cell,pi['lab_id']):
  yp=np.zeros_like(s._obs(pi))
  for k in np.unique(tags):
   ix=tags==k; yp[ix]=pred({key:v[ix] for key,v in pi.items()},x[2][ix],factor(~ix,mode,spec),mode)
  r=abs(yp-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
  print('folds',len(np.unique(tags)),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell',np.round([max(c[cell==i].mean() for i in np.unique(cell)),max(v[cell==i].mean() for i in np.unique(cell))],4))
train_hold=tr['hold_channel_raw']@b[1:]+b[0]
for name,g in [('pilot',pi),('test',te)]:
 hold=g['hold_channel_raw']@b[1:]+b[0]; c=g['lab_id']*3+g['stratum']; print(name,'hold support',[(int(i),np.round([min(hold[c==i]),max(hold[c==i])],2).tolist()) for i in np.unique(c)])
print('train hold median',np.median(train_hold))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_loadcheck.py
```
- **Container Exit Code**: `0` (`2285 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
load fixed drift coefficients [-0.0066  0.     -0.0278  0.0382 -0.0258  0.0207  0.      0.0155  0.0053
 -0.0406  0.0247  0.0168  0.1025  0.      0.      0.     -0.1545 -0.0051
  0.0147  0.      0.0044  0.0004 -0.0032  0.0152 -0.0013 -0.0174  0.
  0.      0.     -0.0158]
folds 18 p90 [0.1496 0.0591] worst cell [0.1143 0.0522]
folds 8 p90 [0.3543 0.1536] worst cell [0.4014 0.1902]
folds 4 p90 [0.3692 0.1602] worst cell [0.4014 0.1902]
cell fixed drift coefficients [-0.0059 -0.0025 -0.0304  0.0433 -0.0231  0.0244  0.      0.0193  0.0012
 -0.0568  0.0283  0.0224  0.1063  0.      0.      0.     -0.1538]
folds 18 p90 [0.155  0.0595] worst cell [0.121  0.0614]
folds 8 p90 [0.3442 0.1514] worst cell [0.399  0.1891]
folds 4 p90 [0.387  0.1628] worst cell [0.399  0.1891]
load soft drift coefficients [-0.0087  0.0006 -0.0279  0.0374 -0.0261  0.0202  0.      0.0139  0.0057
 -0.0397  0.0249  0.015   0.1025  0.      0.      0.     -0.1551 -0.0068
  0.0143  0.      0.0047  0.0005 -0.0035  0.0151 -0.0015 -0.017   0.
  0.      0.     -0.0166]
folds 18 p90 [0.1455 0.0583] worst cell [0.1119 0.0499]
folds 8 p90 [0.356  0.1523] worst cell [0.4073 0.1901]
folds 4 p90 [0.367  0.1584] worst cell [0.4073 0.1901]
pilot hold support [(0, [2.49, 4.73]), (2, [2.68, 2.95]), (3, [3.93, 3.93]), (4, [3.8, 4.03]), (5, [3.03, 3.61]), (6, [2.46, 2.78]), (7, [2.49, 3.02]), (11, [2.45, 4.08])]
test hold support [(1, [3.61, 3.98]), (2, [1.87, 4.68]), (3, [3.97, 3.97]), (4, [2.78, 3.38]), (6, [2.69, 4.13]), (7, [2.38, 4.18]), (8, [3.06, 4.15]), (9, [3.11, 4.0]), (10, [3.06, 4.51]), (11, [3.07, 3.27])]
train hold median 3.4026252252973457

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b729de5384f2a822 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b729de53 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`47480 ms` LLM)
- **Rationale**: Implement the tested chamber correction, clipping only the local load interaction outside observed support. Re-run the actual solver on coupon, cell, and lab holdouts to check whether clipping or integration erases the gains, and verify that the no-pilot path returns finite results with ordered relaxation times.
- **Alternatives Weighed**: `Extrapolate cell-specific load slopes beyond the pilot range` (rejected: Several test cells extend beyond their pilot load range, and some local slopes are supported by only one or two coupons.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('batch_solver.py')
s = p.read_text()
start = s.index('\ndef solve_batch(')
s = s[:start] + '''
def _cell_basis(g, cells):
    return ((g['lab_id'][:, None] == cells[:, 0]) &
            (g['stratum'][:, None] == cells[:, 1])).astype(float)


def _chamber_correction(pilot, xp, test, xt, bridge, dt, t, taus, weights):
    # Fit a shared drift adjustment and shrunk lab/stratum interactions.
    cells = np.unique(np.column_stack((pilot['lab_id'], pilot['stratum'])), axis=0)
    cp, ct = _cell_basis(pilot, cells), _cell_basis(test, cells)
    hp = pilot['hold_channel_raw'] @ bridge[1:] + bridge[0]
    ht = test['hold_channel_raw'] @ bridge[1:] + bridge[0]
    center = np.median(hp)
    local_test = ct * (ht[:, None] - center)
    for j in range(len(cells)):
        seen = hp[cp[:, j] != 0]
        # Beyond a cell's observed range, retain only the shared load slope.
        local_test[:, j] = ct[:, j] * (np.clip(ht, seen.min(), seen.max()) - center)
    n = len(hp)
    basis = np.column_stack((np.full(n, .10), .08*xp, .12*cp,
                             .045*(hp-center), .035*cp*(hp[:, None]-center)))
    # A coupon-specific nuisance slope keeps individual discrepancies from
    # forcing a cell-level correction; it does not transfer to new coupons.
    design = np.column_stack((basis, .022*np.eye(n)))
    h = np.r_[t, 1.]
    r = _obs(pilot) - _design(pilot, xp, bridge, dt, t, taus) @ weights
    dd = np.diff(r[:, :-1], n=3, axis=1)
    noise = max(.009, np.median(np.abs(dd-np.median(dd))) / (.67449*np.sqrt(20)))
    scale = np.r_[np.full(len(t), noise), max(.014, 1.5*noise)]
    a = (design[:, None, :] * (h/scale)[None, :, None]).reshape(r.size, -1)
    y = (r/scale).ravel()
    reg = np.eye(a.shape[1])
    beta = np.linalg.solve(a.T @ a + reg, a.T @ y)
    for _ in range(30):
        w = 1 / (1 + ((a @ beta - y)/2.5)**2)
        new = np.linalg.solve(a.T @ (w[:, None]*a) + reg, a.T @ (w*y))
        if np.linalg.norm(new-beta) < 1e-7:
            beta = new
            break
        beta = new
    weights = weights.copy()
    weights[2:7] += np.r_[.10, np.full(4, .08)] * beta[:5]
    k = 5 + len(cells)
    remaining = .12*(ct @ beta[5:k]) + .045*(ht-center)*beta[k]
    remaining += .035*(local_test @ beta[k+1:k+1+len(cells)])
    return weights, remaining


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    groups = [train, test] if pilot is None else [train, test, pilot]
    x = _align(groups, 0)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    taus, weights, _ = _fit(train, x[0], bridge, dt, t)
    pred = None
    if pilot is not None:
        weights, correction = _chamber_correction(pilot, x[2], test, x[1], bridge, dt, t, taus, weights)
        pred = correction[:, None] * np.r_[t, 1.]
    base = _design(test, x[1], bridge, dt, t, taus) @ weights
    pred = base if pred is None else base + pred
    return dict(estimated_taus=taus, hereditary_weights=weights,
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
'''
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/rheo_candidate_cv.py <<'PY'
exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
lab, st = pi['lab_id'], pi['stratum']
cell = np.unique(np.column_stack((lab, st)),axis=0,return_inverse=True)[1]
for name,tags in [('coupon',np.arange(len(lab))),('cell',cell),('lab',lab)]:
    pred=np.empty_like(s._obs(pi))
    for k in np.unique(tags):
        ix=tags==k
        fit={key:v[~ix] for key,v in pi.items()}
        val={key:v[ix] for key,v in pi.items()}
        d=dict(p, test=val, chamber_verification_trials=fit)
        o=s.solve_batch(d)
        pred[ix]=np.column_stack((o['predicted_strain']-val['strain_0'][:,None],o['initial_rate_pred']))
    r=abs(pred-s._obs(pi)); c,v=r[:,:-1],r[:,-1]
    print(name,'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell MAE',np.round([max(c[cell==k].mean() for k in np.unique(cell)),max(v[cell==k].mean() for k in np.unique(cell))],4))
o=s.solve_batch(dict(p,chamber_verification_trials=None))
assert all(np.all(np.isfinite(v)) for v in o.values())
assert .05 < o['estimated_taus'][0] < o['estimated_taus'][1] < 25
print('blind-path output shapes', {k:v.shape for k,v in o.items()})
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_candidate_cv.py
```
- **Container Exit Code**: `0` (`7396 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
coupon p90 [0.1227 0.046 ] worst cell MAE [0.1068 0.0466]
cell p90 [0.3418 0.149 ] worst cell MAE [0.3985 0.1918]
lab p90 [0.3544 0.1561] worst cell MAE [0.3985 0.1918]
blind-path output shapes {'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (36, 12), 'initial_rate_pred': (36,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ae2d9b6bfea42b34 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ae2d9b6b (directory).
```

### Turn 13 — `submit_completed_task` (`14905 ms` LLM)
- **Rationale**: The exact final solver passed isolated-process schema verification and coupon, cell, and lab holdouts. It improved worst-cell strain MAE from 0.451 to 0.107 on coupon holdouts and from 0.463 to 0.399 on cell holdouts. No archive was available, so the no-pilot path was checked for output validity, not chamber prediction accuracy.
- **Alternatives Weighed**: `Refit the relaxation spectrum from the chamber pilot alone` (rejected: The slow branch was sensitive to drift specification, and the pilot-only baseline had higher worst-cell strain error on every holdout scheme.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, and the no-pilot path returned finite outputs with ordered relaxation times. Worst-cell MAE on held-out chamber labs was 0.399 for strain and 0.192 for initial rate.`

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
