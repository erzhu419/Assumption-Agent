# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_ep01__preloaded_coupon_compliance_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M94-Continual-Binding-World` |
| **Domain** | `Multi-Lab Polymer Coupon Hereditary Creep & Load-Cell Calibration Stream` |
| **Wave / Wormhole Family** | `m94_stream_step_1` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `309,350` (`299,725` / `91,520` / `9,625`) |
| **Total Episode Wall Time (s)** | `428.15s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center. | [Consolidated `m94_src_alpha`]: Cell-balanced imputation corrects for unequal enrollment, but it cannot recover coupon-specific drift: on the three public imputed rows, feature-based initial-rate RMSE was 0.1705 versus 0.0056 for the supplied rate measurements. Separate affine facility calibration from the material law, propagate each relaxation branch with the exact variable-step exponential kernel, and fit the shared coefficients on complete-feature rows using separately MAD-scaled strain and rate residuals; L2 supplies starts rather than the final estimate. At prediction time, any missing feature selects full correction by the finite measured-minus-predicted rate, whereas a complete feature vector selects a 0.25 correction, and neither update changes the hereditary branches. The episode receipt reports zero change under the panel affine gauge, strain MAE 0.009658, and relative relaxation-time error 0.028169; the refactor adds a local identifiability gate requiring the robustly weighted, column-normalized joint Jacobian to have a smallest-to-largest singular-value ratio of at least 1e-8.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Cell-mean imputation alone is sufficient to predict individual coupon drift when features are missing.` (witness: `The three public imputed rows have initial-rate RMSE 0.1705, while their supplied initial-rate observations have RMSE 0.0056.`) -> Pivot `Retain the prescribed alignment and imputation and test a measurement update to the drift term, whose strain contribution is linear in hold time.`
- **`root_A02_nuisance_subspace_and_deconfounding`**: AXIOM A02 (NUISANCE SUBSPACE PROJECTION & IDENTIFIABILITY): When unknown multiplicative gains, batch shifts, random carrier phases, or background geometry changes confound raw observations, never subtract unaligned raw signals or fit pooled confounded shortcuts; separate invariant physical coordinates and project out nuisance subspaces with well-conditioned regularization so local anomalies are not absorbed into nuisance parameters.
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`19366 ms` LLM)
- **Rationale**: The schema and archived solver will show which model components can be reused and which must change for the new calibration, initial memory, and readouts. Inspecting them first avoids guessing at the recurrence or output contract.

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && cat public_asset_index.json && sed -n '1,260p' /tmp/hegel_continual_archive/alpha_s00_solver.py && (PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment imputation' || true)
```
- **Container Exit Code**: `0` (`326 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_calibrated_hereditary_creep@1b21d6d8 | align_coupon_features(raw_features: np.ndarray, reference_coupons: np.ndarray, facility: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align rows using facility reference panels, then impute missing features | REQ: {'field': 'relaxation_time_lower_bound', 'op': '>', 'value': 0}; {'field': 'relaxation_t | --describe autogenous_calibrated_hereditary_creep.align_coupon_features
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
```

### Turn 2 — `execute_in_remote_docker` (`22789 ms` LLM)
- **Rationale**: Inspect the recurrence, missingness, and available test-time observations before fitting. If the archived helper uses a different imputation rule, implement the specified rule directly.

```bash
sed -n '1,250p' /tmp/hegel_continual_archive/alpha_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_calibrated_hereditary_creep.align_coupon_features; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
r=pickle.load(open('public_alpha_ep01.pkl','rb'));print('records',len(r)); d=r[0]['dataset'];print(r[0].keys());
for k,v in d.items():
 for j,a in (v.items() if isinstance(v,dict) else [(k,v)]):
  a=np.asarray(a); print(k,j,a.shape,a.dtype, (np.nanmin(a),np.nanmax(a),np.isnan(a).sum()) if a.dtype.kind in 'fiu' and a.size else '')
PY
```
- **Container Exit Code**: `0` (`652 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.ndarray, complete_features: np.ndarray, fit_rows: np.ndarray, initial_strain: np.ndarray, observed_strain: np.ndarray, observed_rate: np.ndarray, tau_bounds: tuple[float, float], *, rank_rtol: float=1e-08) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Fit two exponential memory branches with zero initial memory | REQ: {'field': 'relaxation_time_lower_bound', 'op': '>', 'value': 0}; {'field': 'relaxation_t | --describe autogenous_calibrated_hereditary_creep.fit_predict_creep"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_calibrated_hereditary_creep",
      "reference_source_txt": null
    }
  ]
}
records 1
dict_keys(['dataset', 'gt'])
past_dt past_dt (22,) float64 (0.097, 1.1659, 0)
eval_times eval_times (12,) float64 (0.18, 6.4, 0)
calibration_load_cells lab_id (40,) int64 (0, 3, 0)
calibration_load_cells cmd_stress (40,) float64 (-2.1686758578622363, 2.2197727763771846, 0)
calibration_load_cells true_stress (40,) float64 (-2.1686540450675658, 2.2197844840398884, 0)
train lab_id (200,) int64 (0, 3, 0)
train stratum (200,) int64 (0, 2, 0)
train coupon_features_raw (200, 4) float64 (-3.3244185915204802, 4.028529417914458, 25)
train ref_coupons (200, 32) float64 (-3.875635975265449, 5.8038854921658976, 258)
train past_stress_cmd (200, 22) float64 (-5.233648602051064, 5.662781841257494, 0)
train strain_0 (200,) float64 (-1.0062689856871054, 1.1478139927451294, 0)
train stress_hold_cmd (200,) float64 (-2.0616405578059793, 2.326141637056111, 0)
train instant_rate_obs (200,) float64 (-9.85133190926465, 8.824402215825836, 0)
train strain_curves_obs (200, 12) float64 (-6.790464725074662, 7.6900143871118765, 0)
train initial_memory_states (200, 2) float64 (-6.383182583867279, 6.069059755092369, 0)
test lab_id (80,) int64 (0, 3, 0)
test stratum (80,) int64 (0, 2, 0)
test coupon_features_raw (80, 4) float64 (-3.4310948889696626, 3.5903846738786798, 8)
test ref_coupons (80, 32) float64 (-4.0598942927411485, 4.800484310239264, 93)
test past_stress_cmd (80, 22) float64 (-4.660490069902025, 4.800597338152122, 0)
test strain_0 (80,) float64 (-0.7416207207417566, 0.9057069330945047, 0)
test stress_hold_cmd (80,) float64 (-2.1107359170130775, 2.2858081124901477, 0)
test instant_rate_obs (80,) float64 (-7.197782537367944, 9.053332222460488, 0)
test initial_memory_states (80, 2) float64 (-6.1919356422419725, 5.548235099020793, 0)
test critical_strain (80,) float64 (-0.4505676389297072, 1.1622627493020963, 0)
```

### Turn 3 — `execute_in_remote_docker` (`19670 ms` LLM)
- **Rationale**: The prior output truncated the regression basis and rate correction. Reading those sections determines what can be retained. Calibration residuals can reveal whether the specified affine law is adequate, and complete-row counts determine whether lab-held-out fits remain supported.
- **Alternatives Weighed**: `Reuse the archived zero-initial-memory fitter unchanged` (rejected: It cannot propagate the supplied initial memory or estimate its effect on the hold.)

```bash
sed -n '35,210p' /tmp/hegel_continual_archive/alpha_s00_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
d=pickle.load(open('public_alpha_ep01.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; p=d['train']; print('history duration',sum(d['past_dt']),'eval times',d['eval_times']);
for i in range(4):
 m=c['lab_id']==i; A=np.c_[c['cmd_stress'][m],np.ones(m.sum())]; b=c['true_stress'][m]; x=np.linalg.lstsq(A,b,rcond=None)[0]; print('loadcell',i,x,np.sqrt(np.mean((A@x-b)**2)))
for i in range(4):
 for j in range(3):
  m=(p['lab_id']==i)&(p['stratum']==j); print('cell',i,j,m.sum(),'complete',np.isfinite(p['coupon_features_raw'][m]).all(1).sum())
PY
```
- **Container Exit Code**: `0` (`624 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ize(p.x, scale, 'cauchy', 2.0) for p in pilots), key=lambda f: f.cost)
    p = result.x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return np.exp(p[:2]), p[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    dtrain = _Design(*_physical_stresses(dataset, train), dt, times, x[:n])
    dtest = _Design(*_physical_stresses(dataset, test), dt, times, x[n:])
    y = np.column_stack([train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']])
    return dtrain, dtest, y, complete[:n]


def _update_drift(pred, observed_rate, complete, times):
    # The rate innovation updates only the coupon's linear drift, not memory.
    # A complete feature vector gives a much stronger prior on that drift.
    rate = np.asarray(observed_rate)
    delta = np.where(np.isfinite(rate), rate - pred[:, -1], 0.0)
    correction = np.where(complete, 0.25, 1.0) * delta
    return pred + correction[:, None] * np.r_[times, 1.0]


def solve_alpha_s00(dataset: dict) -> dict:
    dtrain, dtest, y, complete = _prepare(dataset)
    # Fit the shared law without treating imputed training features as measured.
    tau, theta = _fit(dtrain, y, complete)
    pred = dtest.matrix(np.log(tau)) @ theta
    test = dataset['test']
    if 'instant_rate_obs' in test:
        test_complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
        pred = _update_drift(pred, test['instant_rate_obs'], test_complete, dataset['eval_times'])
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': pred[:, :-1] + np.asarray(test['strain_0'])[:, None],
            'initial_rate_pred': pred[:, -1]}
history duration 10.0055 eval times [0.18   0.249  0.3446 0.4767 0.6595 0.9125 1.2625 1.7467 2.4166 3.3435
 4.6258 6.4   ]
loadcell 0 [9.99938851e-01 3.91667472e-05] 0.0001347378878515103
loadcell 1 [9.99997002e-01 4.04134193e-05] 7.487066459054634e-05
loadcell 2 [9.99986409e-01 1.18590488e-05] 0.0001120028891859394
loadcell 3 [9.99993359e-01 1.92587111e-05] 6.529033672883791e-05
cell 0 0 29 complete 25
cell 0 1 17 complete 14
cell 0 2 11 complete 10
cell 1 0 11 complete 10
cell 1 1 18 complete 15
cell 1 2 9 complete 9
cell 2 0 15 complete 13
cell 2 1 12 complete 10
cell 2 2 26 complete 22
cell 3 0 15 complete 14
cell 3 1 22 complete 19
cell 3 2 15 complete 14
```

### Turn 4 — `execute_in_remote_docker` (`41437 ms` LLM)
- **Rationale**: The history has a closed-form recurrence, but the two preload columns fix branch identity and cannot be swapped after fitting. Held-out residuals will determine whether the archived drift correction should be retained or re-estimated.

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/hegel_continual_archive/alpha_s00_solver.py').read_text().split('def solve_alpha_s00')[0]
s=s.replace('dt, times, x):','dt, times, x, initial):').replace('        self.past = past','        self.initial = np.asarray(initial)\n        self.past = past')
s=s.replace('gap = self.hold[:, None] - self.past @ weights','preload = self.initial * np.exp(-self.dt.sum() / tau)\n        gap = self.hold[:, None] - self.past @ weights - preload').replace('dgap = -self.past @ (v * weights - u * np.exp(-u) * tail_decay)','dgap = -self.past @ (v * weights - u * np.exp(-u) * tail_decay) - preload * (self.dt.sum() / tau)')
s=s.replace('x[:n])',"x[:n], train['initial_memory_states'])").replace('x[n:])',"x[n:], test['initial_memory_states'])")
s=s.replace('        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]','        raise RuntimeError("Fit reversed branch identities")')
Path('/tmp/ep01_baseline.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
import pickle,numpy as np
import ep01_baseline as m
d=pickle.load(open('public_alpha_ep01.pkl','rb'))[0]['dataset']; p=d['train']; des,_,y,complete=m._prepare(d); times=d['eval_times']; labs=p['lab_id'];
err=np.empty_like(y); sig=np.empty_like(y)
for lab in np.unique(labs):
 val=labs==lab; tau,b=m._fit(des,y,complete&~val); pred=(des.matrix(np.log(tau))@b)[val]; sig[val]=pred; err[val]=y[val]-pred; print('fold',lab,'tau',tau,'coef',b,'MAD',m._mad(err[val&complete]))
for label,sel in [('complete',complete),('imputed',~complete)]:
 print(label,'quantiles raw horizon and rate',np.quantile(err[sel][:,[-2,-1]],[0,.1,.5,.9,1],axis=0)); print('rate-horizon LS weight',np.dot(err[sel,-2],err[sel,-1])/np.dot(err[sel,-1],err[sel,-1])/times[-1]);
 for w in (0,.25,.5,.75,1):
  r=err[:,:-1]-w*err[:,[-1]]*times; print(w,'RMSE by lab',[round(np.sqrt(np.mean(r[sel&(labs==i)]**2)),5) for i in range(4)],'end RMSE',np.sqrt(np.mean(r[sel,-1]**2)))
print('complete worst rows',sorted(zip(np.sqrt(np.mean(err[complete,:-1]**2,1)),np.where(complete)[0]),reverse=True)[:12]); print('complete residual median by horizon',np.median(err[complete],0)); print('full fit',m._fit(des,y,complete))
PY
```
- **Container Exit Code**: `0` (`705 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4789677 -0.1183771   0.09820781
 -0.07928633] MAD [0.00955796 0.00899996 0.01156542 0.00937383 0.01262651 0.00965918
 0.0110502  0.00992058 0.01522768 0.01698239 0.01705931 0.03324757
 0.0133403 ]
fold 3 tau [0.44990587 3.29527793] coef [ 0.73963155  0.52113468  0.07990761  0.1461173  -0.11685726  0.09738332
 -0.07819239] MAD [0.01248002 0.01240642 0.01141192 0.00955219 0.01081141 0.01020289
 0.01136796 0.01148003 0.014045   0.01958205 0.01897535 0.02456175
 0.00965761]
complete quantiles raw horizon and rate [[-1.08476556e+00 -3.33248459e-02]
 [-5.65957159e-02 -1.46851514e-02]
 [-8.63449685e-04  8.98034197e-04]
 [ 4.35543404e-02  1.48516104e-02]
 [ 1.10554010e+00  3.64428131e-02]]
rate-horizon LS weight 0.6672252655739288
0 RMSE by lab [0.27124, 0.24305, 0.29444, 0.27836] end RMSE 0.2941231259825491
0.25 RMSE by lab [0.27059, 0.24258, 0.29393, 0.27854] end RMSE 0.29132404748552404
0.5 RMSE by lab [0.27018, 0.24251, 0.29365, 0.27887] end RMSE 0.28980430944143565
0.75 RMSE by lab [0.27001, 0.24286, 0.29359, 0.27936] end RMSE 0.2895840544991815
1 RMSE by lab [0.27009, 0.2436, 0.29376, 0.27999] end RMSE 0.2906662367586242
imputed quantiles raw horizon and rate [[-1.62019539 -0.26692933]
 [-1.31269551 -0.21167886]
 [-0.07725702 -0.02400544]
 [ 0.74955511  0.13430474]
 [ 1.23501667  0.20812095]]
rate-horizon LS weight 0.8949102549216088
0 RMSE by lab [0.40686, 0.39599, 0.46443, 0.32195] end RMSE 0.7810462640157085
0.25 RMSE by lab [0.3231, 0.34705, 0.39231, 0.28478] end RMSE 0.5831563633001103
0.5 RMSE by lab [0.24829, 0.31531, 0.33143, 0.25482] end RMSE 0.39720223235358804
0.75 RMSE by lab [0.19312, 0.30617, 0.28898, 0.23483] end RMSE 0.2512530089783723
1 RMSE by lab [0.17701, 0.32157, 0.27368, 0.22747] end RMSE 0.23695300685884405
complete worst rows [(0.6228990017833642, 77), (0.5096447181186353, 37), (0.5069921396106452, 146), (0.4973973548723702, 132), (0.4928475980520465, 69), (0.49150060517694266, 157), (0.48024813524199766, 140), (0.47871096630301574, 102), (0.47412410096637037, 173), (0.469380769534397, 7), (0.44885190786135015, 62), (0.4442969887747528, 148)]
complete residual median by horizon [-4.26842470e-04 -8.06920652e-04 -1.58177395e-03 -7.02181044e-04
  1.30712635e-03 -4.75447598e-05  1.12689370e-03  7.84597562e-04
  1.10868007e-03  1.87260868e-03  5.99014338e-04 -8.63449685e-04
  8.98034197e-04]
full fit (array([0.44969456, 3.295544  ]), array([ 0.73911682,  0.5224193 ,  0.07964138,  0.14678864, -0.11750458,
        0.09769938, -0.07851951]))
```

### Turn 5 — `execute_in_remote_docker` (`29801 ms` LLM)
- **Rationale**: Coherent residuals growing with time would support revising drift, while isolated or discontinuous trajectory errors would favor robust fitting and validation. The residual traces distinguish these cases using only public training measurements.
- **Alternatives Weighed**: `Fit extra relaxation terms to reduce raw RMSE` (rejected: The current relaxation times are stable across labs, and large trajectory errors coexist with small initial-rate errors. Inspect their shape before changing the constitutive law.)

```bash
cd /holdout_task && cat > /tmp/ep01_residuals.py <<'PY'
import pickle,numpy as np
import ep01_baseline as m
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']; p=d['train']; des,_,y,c=m._prepare(d); t=d['eval_times']; labs=p['lab_id']; err=np.empty_like(y)
for lab in range(4):
 v=labs==lab; tau,b=m._fit(des,y,c&~v); err[v]=y[v]-des.matrix(np.log(tau))[v]@b
r=err[:,:-1]; slope=np.median(r[:,4:]/t[4:],axis=1); end=r[:,-1]; q=np.max(abs(r),axis=1); ix=np.argsort(q[c]); print('largest complete residuals: rank row, trajectory, rate residual');
for i in np.where(c)[0][ix[-10:]]: print(i,labs[i],np.round(err[i],3))
for mask in [c,~c]:
 print('n',mask.sum(),'large errors per horizon',(abs(r[mask])>.10).sum(0),'rate-slope stats',np.quantile(np.column_stack([slope[mask], err[mask,-1], slope[mask]-err[mask,-1]]),[.1,.5,.9],axis=0))
 for w in [0,.25,.5,.75,1]:
  rr=r[mask]-w*err[mask,-1,None]*t; print(w,'MAE',np.mean(abs(rr)), 'median',np.median(abs(rr)),'endpoint MAE',np.mean(abs(rr[:,-1])),'endpoint median',np.median(abs(rr[:,-1])))
print('near thresholds lab x missingness');
for l in range(4): print(l,[(int((mask & (labs==l)).sum()),np.round(np.quantile(abs(end[mask & (labs==l)]),[.5,.8,.95]),3)) for mask in [c,~c]])
PY
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 /tmp/ep01_residuals.py
```
- **Container Exit Code**: `0` (`751 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
006  0.007 -1.141 -0.019  1.024 -0.033 -0.046 -0.047
 -0.064 -0.079 -0.019]
77 1 [-9.000e-03 -1.000e-02  1.000e-03 -1.400e-02 -1.064e+00  3.000e-03
 -1.020e+00  1.144e+00 -6.000e-03 -5.000e-03  1.083e+00 -3.000e-02
  1.000e-02]
61 3 [ 4.000e-03 -6.000e-03  0.000e+00 -8.000e-03 -1.700e-02 -9.000e-03
  1.800e-02  1.157e+00 -8.010e-01 -0.000e+00 -1.000e-03  2.600e-02
  7.000e-03]
62 2 [-4.000e-03  1.169e+00 -1.000e-03 -1.300e-02 -1.100e-02  4.000e-03
  6.000e-03  3.000e-03  6.000e-03 -1.000e-03  6.000e-03  1.025e+00
 -2.000e-03]
n 175 large errors per horizon [13 16 14 17 11 10 10 20 15 14 15 18] rate-slope stats [[-0.00637839 -0.01468515 -0.01241804]
 [ 0.00034198  0.00089803 -0.00027328]
 [ 0.00695917  0.01485161  0.01537902]]
0 MAE 0.08824932904013412 median 0.009390305275714994 endpoint MAE 0.11186415717947146 endpoint median 0.023447852990443163
0.25 MAE 0.08745177638744478 median 0.00915815053884891 endpoint MAE 0.1085120473302857 endpoint median 0.02269601276788711
0.5 MAE 0.08869904330978559 median 0.0098513997026397 endpoint MAE 0.11320396182482934 endpoint median 0.028541350917737865
0.75 MAE 0.09115993864186527 median 0.01101807180902436 endpoint MAE 0.12213572250349376 endpoint median 0.039584215468804196
1 MAE 0.09432459022197588 median 0.012185756123498027 endpoint MAE 0.1332256519409015 endpoint median 0.05451842375081997
n 25 large errors per horizon [ 0  3  2  5  8  8 14 17 20 21 22 23] rate-slope stats [[-0.20625041 -0.21167886 -0.01461803]
 [-0.02054659 -0.02400544 -0.00055528]
 [ 0.11627783  0.13430474  0.01427398]]
0 MAE 0.24190302139181635 median 0.09398768034490713 endpoint MAE 0.623501210271852 endpoint median 0.5070946189127987
0.25 MAE 0.19685021485764684 median 0.06972877058999458 endpoint MAE 0.4752578822746409 endpoint median 0.38512417357590134
0.5 MAE 0.15228140319300926 median 0.047629990972925834 endpoint MAE 0.3291865764346999 endpoint median 0.2815951612759735
0.75 MAE 0.10817213508682187 median 0.02663687930541608 endpoint MAE 0.18311527059475885 endpoint median 0.13859237022405274
1 MAE 0.0824380022010361 median 0.011323989034699 endpoint MAE 0.10094272786038536 endpoint median 0.048919508078974605
near thresholds lab x missingness
0 [(49, array([0.028, 0.045, 0.92 ])), (8, array([0.74 , 1.044, 1.481]))]
1 [(34, array([0.033, 0.068, 0.12 ])), (4, array([0.206, 0.625, 1.021]))]
2 [(45, array([0.019, 0.054, 0.953])), (8, array([0.496, 1.142, 1.592]))]
3 [(47, array([0.016, 0.031, 0.048])), (5, array([0.477, 0.6  , 0.878]))]
```

### Turn 6 — `execute_in_remote_docker` (`51883 ms` LLM)
- **Rationale**: Write the solver and compare adaptive shrinkage with fixed and zero corrections on held-out labs, estimating adaptive weights only from each training fold. Median and capped errors distinguish improvements in drift prediction from the effect of isolated measurement spikes on RMSE.
- **Alternatives Weighed**: `Add relaxation branches to reduce raw curve RMSE` (rejected: The largest residuals are isolated spikes between well-predicted measurements; extra smooth branches would not explain them.)

```bash
cd /holdout_task && cat validate_public_schema.py && cp /tmp/ep01_baseline.py alpha_ep01_solver.py && python3 - <<'PY'
from pathlib import Path
p=Path('alpha_ep01_solver.py'); s=p.read_text(); s=s.replace('Facility calibration and two-branch hereditary creep regression.','Calibrated two-branch hereditary creep with preloaded memory.'); s=s.split('def _update_drift')[0]; s=s.replace('    if p[0] > p[1]:\n        raise RuntimeError("Fit reversed branch identities")\n',''); p.write_text(s)
PY
cat >> alpha_ep01_solver.py <<'PY'
def _rate_weights(residual, complete, times):
    # The initial rate measures the same drift as the slope across hold times.
    # Estimate its reliability from a late-time slope resistant to isolated spikes.
    late = slice(len(times) // 2, None)
    slope = np.median(residual[:, :-1][:, late] / times[late], axis=1)
    noise = _mad((residual[:, -1] - slope)[complete])
    weights = []
    for group in (complete, ~complete):
        # Estimate the variance attributable to drift in the rate innovation.
        spread = _mad(residual[group, -1]) if group.sum() >= 5 else noise
        weights.append(max(0.0, 1.0 - (noise / spread)**2))
    return weights


def _update_drift(pred, observed_rate, complete, times, weights):
    delta = np.asarray(observed_rate) - pred[:, -1]
    correction = np.where(complete, weights[0], weights[1]) * np.where(np.isfinite(delta), delta, 0.0)
    return pred + correction[:, None] * np.r_[times, 1.0]


def _crossing_times(strain_0, curves, critical_strain, times):
    values = np.column_stack([strain_0, curves])
    t = np.r_[0.0, times]
    hit = values >= np.asarray(critical_strain)[:, None]
    first = np.argmax(hit, axis=1)
    result = np.full(len(values), times[-1], dtype=float)
    result[hit[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    lo, hi = values[rows, j - 1], values[rows, j]
    fraction = (np.asarray(critical_strain)[rows] - lo) / (hi - lo)
    result[rows] = t[j - 1] + fraction * (t[j] - t[j - 1])
    return result


def solve_alpha_ep01(dataset: dict) -> dict:
    train, test, y, complete = _prepare(dataset)
    tau, theta = _fit(train, y, complete)
    log_tau = np.log(tau)
    times = np.asarray(dataset['eval_times'])
    weights = _rate_weights(y - train.matrix(log_tau) @ theta, complete, times)
    panel = dataset['test']
    initial = np.asarray(panel['strain_0'])
    test_complete = np.isfinite(panel['coupon_features_raw']).all(axis=1)
    pred = test.matrix(log_tau) @ theta
    if 'instant_rate_obs' in panel:
        pred = _update_drift(pred, panel['instant_rate_obs'], test_complete, times, weights)
    delta = pred[:, :-1]
    curves = initial[:, None] + delta
    return {
        'estimated_taus': tau,
        'predicted_strain': curves,
        'secant_compliance': (curves[:, -1] - initial) / np.maximum(np.abs(test.hold), 0.5),
        'hazard_crossing_times': _crossing_times(initial, curves, panel['critical_strain'], times),
    }
PY
cat > /tmp/ep01_cv.py <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_ep01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_ep01.pkl','rb'))[0]['dataset']; a,_,y,c=m._prepare(d); t=d['eval_times']; labs=d['train']['lab_id'];
for lab in np.unique(labs):
    hold=labs==lab; fit=~hold
    tau,b=m._fit(a,y,fit&c); pred=a.matrix(np.log(tau))@b
    weights=m._rate_weights((y-pred)[fit],c[fit],t)
    print('fold',lab,'tau',tau,'weights',np.round(weights,4))
    for scheme,w in [('none',(0,0)),('fixed',(.25,1)),('adaptive',weights)]:
        r=(m._update_drift(pred,y[:,-1],c,t,w)-y)[:,:-1]; vals=[]
        for sel in (hold&c,hold&~c):
            e=abs(r[sel]); vals.append(np.round([np.sqrt(np.mean(e**2)),np.median(e),np.median(e[:,-1]),np.mean(np.minimum(e,.2))],5))
        print(scheme,vals)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/ep01_cv.py
```
- **Container Exit Code**: `0` (`1070 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_ep01_solver.py']`
- **Stdout (tail)**:
```text
a_ep01(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("predicted_strain", (n_te, k_t)), ("secant_compliance", (n_te,)), ("hazard_crossing_times", (n_te,))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not errs and "estimated_taus" in out:
        ts = np.asarray(out["estimated_taus"], dtype=float)
        if not (0.05 < ts[0] < ts[1] < 25.0): errs.append("taus_not_positive_ordered")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
{"schema_valid": true, "errors": []}
fold 0 tau [0.44983102 3.2816988 ] weights [0.3006 0.9944]
none [array([0.27124, 0.01013, 0.028  , 0.02796]), array([0.40686, 0.12329, 0.73996, 0.12098])]
fixed [array([0.27059, 0.00939, 0.02436, 0.02743]), array([0.17701, 0.01084, 0.03662, 0.02726])]
adaptive [array([0.27049, 0.00944, 0.0259 , 0.02758]), array([0.17682, 0.01038, 0.03407, 0.02634])]
fold 1 tau [0.44973116 3.30289439] weights [0.0493 0.9917]
none [array([0.24305, 0.01053, 0.03272, 0.02672]), array([0.39599, 0.07443, 0.20623, 0.0996 ])]
fixed [array([0.24258, 0.0098 , 0.02875, 0.02407]), array([0.32157, 0.01945, 0.15308, 0.04825])]
adaptive [array([0.24292, 0.01015, 0.03202, 0.02606]), array([0.32068, 0.01938, 0.14689, 0.04753])]
fold 2 tau [0.44979769 3.3055171 ] weights [0.262  0.9955]
none [array([0.29444, 0.00895, 0.01853, 0.02806]), array([0.46443, 0.08884, 0.49617, 0.10737])]
fixed [array([0.29393, 0.00914, 0.02378, 0.0276 ]), array([0.27368, 0.01122, 0.05049, 0.03298])]
adaptive [array([0.29391, 0.00916, 0.02364, 0.02764]), array([0.27367, 0.01112, 0.04887, 0.03258])]
fold 3 tau [0.44990587 3.29527793] weights [0.3991 0.9944]
none [array([0.27836, 0.00885, 0.01573, 0.02526]), array([0.32195, 0.07169, 0.47654, 0.09799])]
fixed [array([0.27854, 0.00866, 0.01426, 0.02547]), array([0.22747, 0.00874, 0.02633, 0.0259 ])]
adaptive [array([0.27872, 0.00917, 0.02006, 0.02647]), array([0.22748, 0.00863, 0.02352, 0.02623])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3aac6eeac15b129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3aac6ee (directory).
```

### Turn 7 — `execute_in_remote_docker` (`48868 ms` LLM)
- **Rationale**: The schema validator checks shapes and finiteness, not readout formulas. Check compliance and first crossings independently, amplify load-cell differences to catch an uncalibrated history or hold, and compare the closed-form preload calculation with a stepwise recurrence to catch indexing errors.
- **Alternatives Weighed**: `Tune rate correction to minimize the largest raw held-out RMSE` (rejected: Isolated strain spikes dominate that metric; adaptive correction already improves worst capped error without weakening the physical model.)

```bash
cd /holdout_task && sed -n '1,39p' alpha_ep01_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_alpha_ep01.pkl', 'rb'))[0]['dataset']
def readouts(ds, out, hold):
    p = ds['test']; times = np.r_[0., ds['eval_times']]
    curve = np.column_stack((p['strain_0'], out['predicted_strain']))
    np.testing.assert_allclose(out['secant_compliance'], (curve[:, -1]-curve[:, 0])/np.maximum(np.abs(hold), .5), rtol=1e-12, atol=1e-12)
    want=[]; cases=[0,0,0]
    for row, crit in zip(curve, p['critical_strain']):
        ix=np.flatnonzero(row>=crit)
        if not len(ix): cases[0]+=1; want.append(times[-1]); continue
        j=ix[0]
        if j==0: cases[1]+=1; want.append(0.); continue
        cases[2]+=1; want.append(np.interp(crit, row[j-1:j+1], times[j-1:j+1]))
    np.testing.assert_allclose(out['hazard_crossing_times'], want, rtol=1e-12, atol=1e-12)
    print('crossing cases [never, initial, interpolated]:', cases, 'stress floor cases:', np.sum(np.abs(hold)<.5))
c=d['calibration_load_cells']; co=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==l], c['true_stress'][c['lab_id']==l],1) for l in range(4)]); p=d['test']; lab=p['lab_id']
out=m.solve_alpha_ep01(d); readouts(d,out,co[lab,0]*p['stress_hold_cmd']+co[lab,1]); print('public taus:',out['estimated_taus'])
# Re-encode the same physical experiments in conspicuously different facility units.
# All outputs must be invariant; failures indicate that history or hold uses command units.
e=copy.deepcopy(d); a=np.array([.63,1.48,.79,1.23]); b=np.array([.47,-.29,.37,-.34]); c=e['calibration_load_cells']; l=c['lab_id']; c['cmd_stress']=a[l]*c['cmd_stress']+b[l]
for key in ('train','test'):
    p=e[key]; l=p['lab_id']; p['past_stress_cmd']=a[l,None]*p['past_stress_cmd']+b[l,None]; p['stress_hold_cmd']=a[l]*p['stress_hold_cmd']+b[l]
re=m.solve_alpha_ep01(e)
for key in out: np.testing.assert_allclose(re[key],out[key],rtol=1e-7,atol=1e-7)
# The closed-form history must agree with an independent stepwise update.
_, des, _, _ = m._prepare(d); tau=out['estimated_taus']; z=des.initial.copy()
for j, dt in enumerate(d['past_dt']):
    decay=np.exp(-dt/tau); z=decay*z+(1-decay)*des.past[:,j,None]
ref=(des.hold[:,None]-z)[:,None,:]*(-np.expm1(-d['eval_times'][:,None]/tau))[None,:,:]
np.testing.assert_allclose(des.matrix(np.log(tau))[:,:-1,:2],ref,rtol=1e-12,atol=1e-12)
print('Facility re-encoding and stepwise preload recurrence passed.')
PY
```
- **Container Exit Code**: `0` (`812 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"""Calibrated two-branch hereditary creep with preloaded memory."""
import numpy as np
from scipy.optimize import least_squares


def _aligned_panels(dataset):
    train, test = dataset['train'], dataset['test']
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for lab in np.unique(labs):
        mask = labs == lab
        r = refs[mask]
        x[mask] = (x[mask] - np.nanmedian(np.nanmean(r, axis=0))) / np.nanmedian(np.nanstd(r, axis=0, ddof=0))
    complete = np.isfinite(x).all(axis=1)
    # Each lab/stratum cell contributes one mean, regardless of its enrollment.
    _, cells = np.unique(np.column_stack([labs, strata]), axis=0, return_inverse=True)
    means = np.stack([np.nanmean(x[cells == c], axis=0) for c in np.unique(cells)])
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    return x, complete


def _physical_stresses(dataset, panel):
    cal = dataset['calibration_load_cells']
    labels = np.asarray(panel['lab_id'])
    past = np.empty_like(panel['past_stress_cmd'], dtype=float)
    hold = np.empty_like(panel['stress_hold_cmd'], dtype=float)
    for lab in np.unique(labels):
        fit = np.asarray(cal['lab_id']) == lab
        cmd = np.asarray(cal['cmd_stress'])[fit]
        gain, bias = np.linalg.lstsq(np.column_stack([cmd, np.ones_like(cmd)]), np.asarray(cal['true_stress'])[fit], rcond=None)[0]
        rows = labels == lab
        past[rows] = gain * np.asarray(panel['past_stress_cmd'])[rows] + bias
        hold[rows] = gain * np.asarray(panel['stress_hold_cmd'])[rows] + bias
    return past, hold


class _Design:
crossing cases [never, initial, interpolated]: [34, 0, 46] stress floor cases: 13
public taus: [0.44969456 3.295544  ]
Facility re-encoding and stepwise preload recurrence passed.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3aac6eeac15b129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3aac6ee (directory).
```

### Turn 8 — `execute_in_remote_docker` (`28484 ms` LLM)
- **Rationale**: The public rows exercised interpolation and censoring but never started at or above the threshold. This targeted test catches an incorrect zero-time result or selection of a later crossing; either failure would require a change to the crossing readout, not the fitted model.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep01_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
# The public rows do not cover an initially met threshold. Check equality at
# t=0 and ensure a later recrossing does not replace the first crossing.
times = np.array([1., 2., 4.])
initial = np.array([0.5, 0., 0., 0.])
curves = np.array([[-1., -2., -3.], [1., 0., 2.], [0.5, 0., 1.], [0., 0.1, 0.4]])
np.testing.assert_allclose(m._crossing_times(initial, curves, np.full(4, 0.5), times), [0., 0.5, 1., 4.])
print('Passed: initial equality, earliest recrossing, exact sample crossing, and no-crossing censoring.')
PY
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Passed: initial equality, earliest recrossing, exact sample crossing, and no-crossing censoring.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3aac6eeac15b129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3aac6ee (directory).
```

### Turn 9 — `submit_completed_task` (`17576 ms` LLM)
- **Rationale**: Leave-one-lab-out validation supports the rate correction. The saved artifact passed isolated-process schema validation and independent tests of both stress conversions, the preload recurrence, the compliance floor, and crossing behavior.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_ep01_solver.py': 'FILE:7781'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_ep01_solver.py`. Worst held-out lab/completeness strain RMSE fell from 0.46443 to 0.32068. The public schema, calibrated stress conversion, preload recurrence, secant compliance, and earliest-crossing checks passed.`

### Turn 10 — `execute_in_remote_docker` (`35360 ms` LLM)
- **Rationale**: Compare each held-out initial-rate residual with an independent late-time slope from the same measured hold. Agreement supports correcting the drift; disagreement supports limiting the rate correction. Signed residuals within lab and stratum can reveal a systematic shift that pooled errors miss.
- **Alternatives Weighed**: `Test classification balance, relay modes, or Fokker–Planck drift` (rejected: The supplied data contain continuous strain measurements and a prescribed hereditary recurrence, not class labels, relay states, or a stochastic drift operator.)

```bash
cd /holdout_task && sed -n '39,126p' alpha_ep01_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_ep01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_ep01.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
p=d['train']; a,_,y,c=m._prepare(d); t=np.asarray(d['eval_times']); lab=np.asarray(p['lab_id']); st=np.asarray(p['stratum']); r=np.empty_like(y)
for l in np.unique(lab):
 v=lab==l; tau,b=m._fit(a,y,c&~v); r[v]=y[v]-a.matrix(np.log(tau))[v]@b
q=np.median(r[:,-7:-1]/t[-6:],axis=1); rate=r[:,-1]; clean=np.abs(r[:,:-1]-q[:,None]*t)<.12
for ok in [True,False]:
 for big in [False,True]:
  sel=(c==ok)&((abs(rate)>.2)==big); print('group complete/large rate',ok,big,'n',sel.sum())
  if sel.any(): print('rate, late slope, discrepancy median absolute',np.round(np.median(abs(np.c_[rate,q,rate-q][sel]),axis=0),4))
for l in np.unique(lab):
 for j in np.unique(st):
  sel=c&(lab==l)&(st==j); print('lab/stratum',l,j,'n',sel.sum(),'median trajectory slope',round(float(np.median(q[sel])),4),'median residual by horizon',np.round(np.nanmedian(np.where(clean[sel],r[sel,:-1],np.nan),axis=0)[[0,5,-1]],4))
print('complete high rate rows: lab, rate innovation, late slope')
print(np.round(np.c_[lab,rate,q][c&(abs(rate)>.2)],4))
PY
```
- **Container Exit Code**: `0` (`815 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
)
    y = np.column_stack([train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']])
    return dtrain, dtest, y, complete[:n]


def _rate_weights(residual, complete, times):
    # The initial rate measures the same drift as the slope across hold times.
    # Estimate its reliability from a late-time slope resistant to isolated spikes.
    late = slice(len(times) // 2, None)
    slope = np.median(residual[:, :-1][:, late] / times[late], axis=1)
    noise = _mad((residual[:, -1] - slope)[complete])
group complete/large rate True False n 175
rate, late slope, discrepancy median absolute [0.0083 0.0034 0.0075]
group complete/large rate True True n 0
group complete/large rate False False n 20
rate, late slope, discrepancy median absolute [0.0661 0.0748 0.005 ]
group complete/large rate False True n 5
rate, late slope, discrepancy median absolute [0.2501 0.2382 0.0119]
lab/stratum 0 0 n 25 median trajectory slope 0.0045 median residual by horizon [-0.003   0.0049  0.0234]
lab/stratum 0 1 n 14 median trajectory slope 0.0033 median residual by horizon [-0.0001  0.0075  0.0215]
lab/stratum 0 2 n 10 median trajectory slope 0.0027 median residual by horizon [ 0.002  -0.003   0.0292]
lab/stratum 1 0 n 10 median trajectory slope 0.0006 median residual by horizon [-0.0016 -0.0016 -0.0063]
lab/stratum 1 1 n 15 median trajectory slope -0.0032 median residual by horizon [ 0.0072 -0.0071 -0.0144]
lab/stratum 1 2 n 9 median trajectory slope -0.0013 median residual by horizon [-0.0008  0.0027 -0.0157]
lab/stratum 2 0 n 13 median trajectory slope -0.0013 median residual by horizon [ 0.0004  0.0028 -0.005 ]
lab/stratum 2 1 n 10 median trajectory slope -0.0015 median residual by horizon [-0.0043 -0.0044 -0.0081]
lab/stratum 2 2 n 22 median trajectory slope -0.0015 median residual by horizon [-0.002  -0.0003 -0.0084]
lab/stratum 3 0 n 14 median trajectory slope -0.0005 median residual by horizon [ 0.0032  0.0011 -0.0008]
lab/stratum 3 1 n 19 median trajectory slope -0.0019 median residual by horizon [ 0.0027 -0.0058 -0.0139]
lab/stratum 3 2 n 14 median trajectory slope -0.0016 median residual by horizon [ 0.0004 -0.003  -0.0118]
complete high rate rows: lab, rate innovation, late slope
[]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3aac6eeac15b129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3aac6ee (directory).
```

### Turn 11 — `execute_in_remote_docker` (`44030 ms` LLM)
- **Rationale**: The held-out residuals support the linear-rate link but leave a small lab-specific slope shift. Compare full rate correction and a compact nonlinear drift basis against the verified solver, fitting corrections inside each fold. Promote a challenger only if it improves the worst supported slices without worsening endpoint compliance.
- **Alternatives Weighed**: `Add lab or stratum intercepts` (rejected: The residual shift seen across lab 0 strata does not identify a transferable stratum effect or a correction for an unseen lab.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_ep01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_ep01.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
a,_,y,c=m._prepare(d); t=np.asarray(d['eval_times']); p=d['train']; lab=p['lab_id']; st=p['stratum']; x,_=m._aligned_panels(d); x=x[:len(y)]; hold=a.hold
# Candidate drift terms all multiply time, so they preserve the hold origin.
cols=[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
cols.extend([hold,hold*np.abs(hold),hold/(1+hold**2)])
for i in range(4): cols.append(x[:,i]/(1+hold**2)-x[:,i])
Z=np.column_stack(cols); late=slice(len(t)//2,None)
rows={k:[] for k in ('adaptive','full_rate','expanded')}
for l in np.unique(lab):
    v=lab==l; fit=~v; tau,b=m._fit(a,y,fit&c); base=a.matrix(np.log(tau))@b
    r=y-base; slope=np.median(r[:,:-1][:,late]/t[late],axis=1)
    w=m._rate_weights(r[fit],c[fit],t)
    # Fit the added physical drift terms on training folds only, using the robust slope readout.
    zz=Z[fit&c]; target=slope[fit&c]; center=np.mean(zz,axis=0); scale=np.std(zz,axis=0); zz=(zz-center)/scale; beta=np.linalg.solve(zz.T@zz+10*np.eye(zz.shape[1]),zz.T@target)
    expanded=base+(((Z-center)/scale)@beta)[:,None]*np.r_[t,1.]
    kwargs={'adaptive':(base,w),'full_rate':(base,(1.,1.)),'expanded':(expanded,m._rate_weights((y-expanded)[fit],c[fit],t))}
    for k,(pred,wk) in kwargs.items():
        e=(m._update_drift(pred,y[:,-1],c,t,wk)-y)[:,:-1]
        for j in np.unique(st[v]):
            for complete in (True,False):
                sel=v&(st==j)&(c==complete)
                if sel.any():
                    r0=e[sel]; end=np.abs(r0[:,-1])/np.maximum(np.abs(hold[sel]),.5)
                    rows[k].append((int(l),int(j),complete,int(sel.sum()),np.sqrt(np.mean(r0**2)),np.mean(np.minimum(abs(r0),.2)),np.median(end)))
    print('held-out lab',l,'rate weights',np.round(w,3),'maximum expanded slope adjustment',round(float(np.max(np.abs((expanded-base)[v,-1]))),5))
for k,z in rows.items():
    for j,label in [(4,'raw strain RMSE'),(5,'capped strain MAE'),(6,'measured endpoint compliance median error')]:
        worst=max(z,key=lambda a:a[j]); supported=max((a for a in z if a[3]>=4),key=lambda a:a[j]); print(k,label,'all',round(worst[j],5),worst[:4],'n>=4',round(supported[j],5),supported[:4])
PY
```
- **Container Exit Code**: `0` (`717 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out lab 0 rate weights [0.301 0.994] maximum expanded slope adjustment 0.00969
held-out lab 1 rate weights [0.049 0.992] maximum expanded slope adjustment 0.0074
held-out lab 2 rate weights [0.262 0.995] maximum expanded slope adjustment 0.01833
held-out lab 3 rate weights [0.399 0.994] maximum expanded slope adjustment 0.00313
adaptive raw strain RMSE all 0.37197 (1, 0, False, 1) n>=4 0.34151 (2, 2, False, 4)
adaptive capped strain MAE all 0.05228 (1, 0, False, 1) n>=4 0.04273 (2, 2, False, 4)
adaptive measured endpoint compliance median error all 0.25041 (1, 1, False, 3) n>=4 0.04886 (2, 2, False, 4)
full_rate raw strain RMSE all 0.37188 (1, 0, False, 1) n>=4 0.34151 (2, 2, False, 4)
full_rate capped strain MAE all 0.0518 (1, 0, False, 1) n>=4 0.04357 (2, 2, False, 4)
full_rate measured endpoint compliance median error all 0.25297 (1, 1, False, 3) n>=4 0.0753 (2, 0, True, 13)
expanded raw strain RMSE all 0.37197 (1, 0, False, 1) n>=4 0.34151 (2, 2, False, 4)
expanded capped strain MAE all 0.05223 (1, 0, False, 1) n>=4 0.04273 (2, 2, False, 4)
expanded measured endpoint compliance median error all 0.25075 (1, 1, False, 3) n>=4 0.04931 (1, 1, True, 15)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3aac6eeac15b129 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3aac6ee (directory).
```

### Turn 12 — `submit_completed_task` (`31634 ms` LLM)
- **Rationale**: The challengers did not improve the worst supported strain slices or endpoint compliance. The unchanged artifact has already passed clean-process verification and independent readout checks.
- **Alternatives Weighed**: `Apply full initial-rate correction to all coupons` (rejected: It did not improve worst supported strain RMSE and increased worst supported median endpoint compliance error from 0.04886 to 0.07530.); `Add nonlinear drift terms` (rejected: They did not improve worst supported strain RMSE or capped MAE and slightly worsened endpoint compliance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_ep01_solver.py`. The solver passed the public schema and independent checks for stress calibration, preload propagation, secant compliance, and earliest threshold crossing. Rate correction reduced worst held-out lab/completeness strain RMSE from 0.46443 to 0.32068.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M94-Continual-Binding-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
