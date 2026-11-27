# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `354,833` (`341,928` / `92,032` / `12,905`) |
| **Total Episode Wall Time (s)** | `426.56s` |

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

### Turn 1 — `execute_in_remote_docker` (`19527 ms` LLM)
- **Rationale**: Inspect the schema and archived implementation to detect input-layout or regime mismatches before reusing code. Summarize public arrays in place; their dimensions and fields determine how to fit the model and construct validation slices.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The task specifies a crystallization plateau and reports that the archived branch binding failed the public gate.)

```bash
set -e
cd /holdout_task
cat validate_public_schema.py
cat public_asset_index.json
ls -lh /tmp/hegel_continual_archive
sed -n '1,250p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' || true
python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_01.pkl','rb') as f: d=pickle.load(f)
def show(x,depth=0):
 p='  '*depth
 print(type(x).__name__,end=' ')
 if isinstance(x,np.ndarray):
  print(x.shape,x.dtype,repr(x.ravel()[:8]))
 elif isinstance(x,dict):
  print(list(x))
  if depth<3:
   for k,v in x.items():
    print(p+'  '+str(k)+':',end=' '); show(v,depth+1)
 elif isinstance(x,(list,tuple)):
  print('len',len(x),repr(x[:1])[:350])
 else: print(repr(x)[:400])
show(d)
PY
```
- **Container Exit Code**: `0` (`471 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
list len 1 [{'dataset': {'past_dt': array([0.6862, 0.3536, 0.4115, 0.5101, 0.4903, 0.7741, 0.4359, 0.5228,
       0.5538, 0.2493, 0.5622, 0.3497, 0.6793, 0.3183, 0.7778, 0.7773,
       0.754 , 0.2048]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shea
```

### Turn 2 — `execute_in_remote_docker` (`22851 ms` LLM)
- **Rationale**: Use calibration residuals to identify whether archive fallback is needed, and count damage-positive rows by lab to establish which slices can identify and validate the two added regime coefficients.
- **Alternatives Weighed**: `Fit the time curves immediately` (rejected: The prior stdout truncated the panel implementation and training target fields; fitting before resolving them risks using the wrong feature coordinates or omitting a measured target.)

```bash
set -e
cat /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation
cd /holdout_task
python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]
print('item keys',list(item)); d=item['dataset']
for k,v in d.items():
 print(k,list(v) if isinstance(v,dict) else (np.shape(v),str(v)[:140]))
 if isinstance(v,dict):
  for j,a in v.items():
   a=np.asarray(a); print(' ',j,a.shape,a.dtype,'examples',str(a.flatten()[:4]),'n_nan',np.isnan(a).sum() if a.dtype.kind in 'fc' else '-')
cal=d['calibration_load_cells'];
for c in range(4):
 m=cal['lab_id']==c; x,y=cal['cmd_stress'][m],cal['true_stress'][m]; coef=np.polyfit(x,y,1); print('cal',c,m.sum(),coef,'max residual',max(abs(y-np.polyval(coef,x))))
for name in ['train','test']:
 s=d[name]; l=s['lab_id']; g=np.array([np.polyfit(cal['cmd_stress'][cal['lab_id']==c],cal['true_stress'][cal['lab_id']==c],1) for c in range(4)])
 p=s['past_stress_cmd']*g[l,0,None]+g[l,1,None]; damage=np.maximum(abs(p)-d['crit_stress'],0)**2@d['past_dt']; print(name,'damage by lab',[(int(c),int(sum(l==c)),int(sum((l==c)&(damage>0))),float(max(damage[l==c]))) for c in np.unique(l)])
PY
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2896 ] n_nan 0
train ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs']
  lab_id (180,) int64 examples [0 1 2 3] n_nan -
  stratum (180,) int64 examples [1 1 0 0] n_nan -
  coupon_features_raw (180, 4) float64 examples [-0.50653748 -1.29092782         nan -0.2012443 ] n_nan 20
  ref_coupons (180, 32) float64 examples [ 0.25229101 -0.61031303 -1.75209797 -2.01246643] n_nan 225
  past_stress_cmd (180, 18) float64 examples [ 1.03377119  1.92662735  0.7413691  -3.87833297] n_nan 0
  stress_hold_cmd (180,) float64 examples [ 0.02404177  1.2106687  -2.19181294 -0.50964194] n_nan 0
  strain_0 (180,) float64 examples [-0.009931 -0.512166 -0.317695  0.023383] n_nan 0
  aux_chamber_telemetry (180, 3) float64 examples [ 1.042811 -0.220458  0.191655  1.604926] n_nan 0
  strain_curves_obs (180, 10) float64 examples [-0.28121657 -0.37651708 -0.34629513 -0.28574706] n_nan 0
  instant_rate_obs (180,) float64 examples [-1.30594803  2.99307484 -2.42893071 -1.06306748] n_nan 0
test ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry']
  lab_id (72,) int64 examples [0 1 2 3] n_nan -
  stratum (72,) int64 examples [2 1 1 0] n_nan -
  coupon_features_raw (72, 4) float64 examples [-0.23841151 -0.51127288  0.39798299  0.51759509] n_nan 4
  ref_coupons (72, 32) float64 examples [-0.34049248 -1.30066948 -0.34558436  0.72083732] n_nan 99
  past_stress_cmd (72, 18) float64 examples [ 0.02760751 -2.16801415  0.21836288 -4.22715836] n_nan 0
  stress_hold_cmd (72,) float64 examples [ 0.81690633 -0.08049237 -2.13346235  0.52966935] n_nan 0
  strain_0 (72,) float64 examples [-0.058637 -0.028608  0.245962 -0.283966] n_nan 0
  aux_chamber_telemetry (72, 3) float64 examples [-2.094208 -1.243715  1.595305  0.825359] n_nan 0
cal 0 10 [ 0.79004725 -0.13998828] max residual 0.00019010847293876587
cal 1 10 [1.30996232 0.16005601] max residual 0.00019176057234515653
cal 2 10 [0.84996132 0.10003695] max residual 0.00012464239328244253
cal 3 10 [ 1.24000154 -0.12996203] max residual 0.00014402498836063593
train damage by lab [(0, 41, 30, 4.981234648028501), (1, 37, 24, 4.90738540332682), (2, 52, 31, 4.665778964870646), (3, 50, 35, 5.007084899559942)]
test damage by lab [(0, 17, 10, 3.8460584871326438), (1, 16, 13, 4.676087058017971), (2, 16, 14, 4.669028433058106), (3, 23, 11, 4.598674428180699)]
```

### Turn 3 — `execute_in_remote_docker` (`53716 ms` LLM)
- **Rationale**: Fit relaxation times by variable projection, solving the nine linear weights exactly at each trial. Compare curve and rate errors separately on held-out labs, damage strata, and horizons to detect a wrong terminal basis or a poor weighting of the measured rate.
- **Alternatives Weighed**: `Fit each time series or lab independently` (rejected: That discards the specified shared parameters and cannot transfer a fit to a held-out lab.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
import ast, pathlib
src=pathlib.Path('/tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py').read_text()
names={'_line','_calibration','_archive_calibration','_align'}
parts=[ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(parts)==4
pathlib.Path('alpha_m97_c02_batch_01_solver.py').write_text('"""Fit the calibrated two-branch crystallization response."""\n\nimport numpy as np\nfrom scipy.optimize import least_squares\n\n\n'+'\n\n\n'.join(parts)+'\n')
print(parts[-1])
PY
cat >> alpha_m97_c02_batch_01_solver.py <<'PY'


def _prepare(data):
    calibration = _calibration(data)
    features = _align(data)
    dt = np.asarray(data['past_dt'])
    t = np.asarray(data['eval_times'])
    blocks = []
    for split, aligned in zip(('train', 'test'), features):
        s = data[split]
        gain, bias = calibration[s['lab_id']].T
        past = s['past_stress_cmd'] * gain[:, None] + bias[:, None]
        hold = s['stress_hold_cmd'] * gain + bias
        damage = (np.maximum(np.abs(past) - data['crit_stress'], 0.0)**2) @ dt
        act = np.tanh(damage) if data['shear_banding_index'] <= 0.50 else np.zeros_like(damage)
        blocks.append((past, hold, np.column_stack((np.ones(len(hold)), aligned)), act))
    return dt, t, blocks


def _basis(taus, dt, t, block, log_plateau=True):
    past, hold, coupons, act = block
    dtau = dt[:, None] / taus
    decay = np.exp(-dtau)
    inject = -np.expm1(-dtau)
    memory = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        memory = decay[s] * memory + inject[s] * past[:, s, None]
    gap = hold[:, None] - memory
    rise = -np.expm1(-t[:, None] / taus)
    terminal = np.where(act[:, None] > 0.0, np.log1p(t), t) if log_plateau else np.broadcast_to(t, (len(hold), len(t)))
    crystallized_hold = act * hold
    crystallized_memory = act * memory[:, 0]
    curves = np.concatenate((gap[:, None] * rise,
                             coupons[:, None] * terminal[:, :, None],
                             (crystallized_hold[:, None] * rise[:, 0])[:, :, None],
                             (crystallized_memory[:, None] * terminal)[:, :, None]), axis=2)
    rates = np.column_stack((gap / taus, coupons, crystallized_hold / taus[0], crystallized_memory))
    return curves, rates


def _fit(dt, t, block, y, rate, rate_weight=1.0, log_plateau=True):
    target = np.concatenate((y.ravel(), rate * rate_weight))

    def project(log_taus, return_weights=False):
        curves, rates = _basis(np.exp(log_taus), dt, t, block, log_plateau)
        design = np.concatenate((curves.reshape(-1, 9), rates * rate_weight), axis=0)
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        if return_weights:
            return weights
        return design @ weights - target

    low = np.log(min(dt.min(), np.min(t[t > 0])) * 0.02)
    high = np.log(max(dt.sum(), t.max()) * 30.0)
    starts = [(0.3, 2.5), (0.7, 6.0), (1.0, 0.25)]
    best = min((least_squares(project, np.clip(np.log(pair), low, high), bounds=(low, high),
                              ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
                for pair in starts), key=lambda fit: fit.cost)
    return np.exp(best.x), project(best.x, return_weights=True)


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    dt, t, (train, test) = _prepare(dataset)
    s = dataset['train']
    y = s['strain_curves_obs'] - s['strain_0'][:, None]
    taus, weights = _fit(dt, t, train, y, s['instant_rate_obs'])
    curves, rates = _basis(taus, dt, t, test)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': dataset['test']['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
d=json.load(open(p))
print('history layout',type(d).__name__,list(d)[:8] if isinstance(d,dict) else len(d))
print(str(d)[:1200])
PY
cat >/tmp/rheo_cv.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; labs=s['lab_id']; active=b[-1]>0
for q, log in [(0.,True),(.25,True),(1.,True),(4.,True),(1.,False)]:
 ec=np.empty_like(y); er=np.empty_like(r); est=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=m._fit(dt,t,tuple(x[tr] for x in b),y[tr],r[tr],q,log); est.append(np.round(tau,5).tolist())
  c, a=m._basis(tau,dt,t,tuple(x[te] for x in b),log); ec[te]=c@w-y[te]; er[te]=a@w-r[te]
 rm=lambda x: np.sqrt(np.mean(x*x))
 print('q,log',q,log,'taus',est,'pooled',rm(ec),rm(er),'worst lab/activation', np.max([[rm(ec[(labs==lab)&(active==on)]),rm(er[(labs==lab)&(active==on)])] for lab in np.unique(labs) for on in (False,True)],axis=0),'worst lab/horizon',max(rm(ec[labs==lab,k]) for lab in np.unique(labs) for k in range(len(t))))
print('all-data parameters',m._fit(dt,t,b,y,r))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_cv.py
```
- **Container Exit Code**: `1` (`576 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7fc5d6c155b32025 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7fc5d6c1 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 9, in <module>
    out = mod.solve_alpha_m97_c02_batch_01(ds)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c02_batch_01_solver.py", line 137, in solve_alpha_m97_c02_batch_01
    dt, t, (train, test) = _prepare(dataset)
                           ^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c02_batch_01_solver.py", line 91, in _prepare
    blocks.append((past, hold, np.column_stack((np.ones(len(hold)), aligned)), act))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/shape_base.py", line 652, in column_stack
    return _nx.concatenate(arrays, 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 180 and the array at index 1 has size 4
```

### Turn 4 — `execute_in_remote_docker` (`28622 ms` LLM)
- **Rationale**: Fix the observed shape mismatch at the alignment boundary. Then compare held-out curve and rate errors across labs, damage states, and horizons to detect terminal-basis mismatch or an unfavorable rate-weighting tradeoff.
- **Alternatives Weighed**: `Align train and test independently` (rejected: The specification requires a shared reference panel. Split the aligned rows only after standardization and imputation.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c02_batch_01_solver.py')
s=p.read_text()
old='    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))'
assert s.count(old)==1
p.write_text(s.replace(old, '    aligned = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))\n    n_train = len(train[\'lab_id\'])\n    return aligned[:n_train], aligned[n_train:]'))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
d=json.load(open(p))
print('history sample',str(d)[:950])
PY
cat >/tmp/rheo_cv.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; labs=s['lab_id']; active=b[-1]>0
rm=lambda x: np.sqrt(np.mean(x*x))
for q, log in [(0.,True),(.25,True),(1.,True),(4.,True),(1.,False)]:
 ec=np.empty_like(y); er=np.empty_like(r); est=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=m._fit(dt,t,tuple(x[tr] for x in b),y[tr],r[tr],q,log); est.append(np.round(tau,5).tolist())
  c,a=m._basis(tau,dt,t,tuple(x[te] for x in b),log); ec[te]=c@w-y[te]; er[te]=a@w-r[te]
 print('rate_weight,log',q,log,'taus',est,'pooled',rm(ec),rm(er),'worst lab/activation',np.max([[rm(ec[(labs==lab)&(active==on)]),rm(er[(labs==lab)&(active==on)])] for lab in np.unique(labs) for on in (False,True)],axis=0),'worst lab/horizon',max(rm(ec[labs==lab,k]) for lab in np.unique(labs) for k in range(len(t))))
print('all-data parameters',m._fit(dt,t,b,y,r))
PY
python3 /tmp/rheo_cv.py
```
- **Container Exit Code**: `0` (`1886 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646936, 0.95752, 2.133164, -0.418978, -0.546283, -1.402876, 1.416566, -0.474167, 0.562697, -1.431478, -1.102292, 0.1, 0.333007, 0.979026, 1.410368, 2.014512, 2.037978, -1.520941, -1.132263, -0.400526, 1.428582, 2.184857, 2.114623, 2.024408, 0.373485, 1.83932, -1.383844, 0.25249, -1.650978, 2.152261, 0.
rate_weight,log 0.0 True taus [[0.47598, 11.47007], [0.57685, 6.17194], [0.35737, 2.03662], [0.38691, 1.92065]] pooled 0.9675824167398395 0.3273841600262443 worst lab/activation [2.27972969 0.4917208 ] worst lab/horizon 4.184955490764345
rate_weight,log 0.25 True taus [[0.48604, 11.36758], [0.56183, 6.05954], [0.41803, 2.17259], [0.43696, 1.99155]] pooled 0.9670435098045048 0.19774804992824827 worst lab/activation [2.27943644 0.26493324] worst lab/horizon 4.184720402634907
rate_weight,log 1.0 True taus [[0.50051, 10.82515], [0.53777, 5.69146], [0.46339, 2.38347], [0.48031, 2.13058]] pooled 0.9664046164930613 0.1476206995927 worst lab/activation [2.27963423 0.19162166] worst lab/horizon 4.184535076097661
rate_weight,log 4.0 True taus [[0.51345, 6.88676], [0.51699, 3.79689], [0.49093, 2.82708], [0.50184, 2.45505]] pooled 0.965825698857249 0.10875458006946025 worst lab/activation [2.29185614 0.13665101] worst lab/horizon 4.190837239333416
rate_weight,log 1.0 False taus [[0.49084, 11.48712], [0.54297, 9.33633], [0.4555, 2.37307], [0.49266, 2.33316]] pooled 0.9532846877307032 0.1819044703185365 worst lab/activation [2.25832895 0.2369899 ] worst lab/horizon 4.166188081174113
all-data parameters (array([0.50353517, 2.95507297]), array([ 0.64984794,  0.51242898,  0.09200022,  0.08933057, -0.08284115,
        0.07738082, -0.03160061,  0.32075028,  0.2456764 ]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=022b97b77b0e3c47 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_022b97b7 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`21427 ms` LLM)
- **Rationale**: Residuals by row, horizon, activation, and missing-feature status will distinguish isolated measurement contamination from a systematic model mismatch. Concentrated errors would justify robust fitting; coherent errors across ordinary rows would instead prompt a review of the basis or alignment.
- **Alternatives Weighed**: `Replace the logarithmic plateau with a linear terminal term` (rejected: The linear alternative barely reduces worst-horizon curve RMSE and worsens worst-slice rate RMSE; the source of the large curve residuals is still unknown.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; dt,t,(b,_)=m._prepare(d)
tau,w=m._fit(dt,t,b,y,r); c,a=m._basis(tau,dt,t,b); e=y-c@w; er=r-a@w
print('absolute curve quantiles by horizon',np.round(np.quantile(abs(e),[.25,.5,.8,.9,.95,.98,1],axis=0),4)); print('rate quantiles',np.quantile(abs(er),[.25,.5,.8,.9,.95,.98,1])); print('largest row residuals: row, lab, damage activation, raw feature missing count, rate residual, curve residuals')
for i in np.argsort(np.linalg.norm(e,axis=1))[-18:][::-1]: print(i,s['lab_id'][i],round(b[-1][i],4),np.isnan(s['coupon_features_raw'][i]).sum(),round(er[i],4),np.round(e[i],3))
print('over 0.5 counts by lab/horizon',[(int(l),np.sum(abs(e[s['lab_id']==l])>.5,axis=0).tolist()) for l in range(4)])
PY
```
- **Container Exit Code**: `0` (`539 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
630e+00 2.52334e+01
  4.51400e+00 3.06450e+00 1.84192e+01 1.74834e+01]]
rate quantiles [0.04005085 0.09021669 0.17610886 0.20452649 0.24385343 0.3007877
 0.37431172]
largest row residuals: row, lab, damage activation, raw feature missing count, rate residual, curve residuals
139 1 0.0 0 -0.0956 [-2.0000e-03 -4.0000e-02 -5.0500e-01 -4.8000e-02 -6.9000e-02  2.5233e+01
 -9.0000e-03  1.0000e-02  2.2000e-02  4.4000e-02]
45 2 0.9848 0 -0.0373 [-1.8000e-02  1.0000e-03 -2.5000e-02 -4.0000e-03  3.2100e-01  6.0000e-03
 -2.5000e-02 -2.6000e-02  1.8419e+01  2.6000e-02]
51 0 0.9972 0 0.1629 [  0.049   0.108  -0.504   0.14    0.189   0.209   0.227   0.217   0.241
 -17.483]
142 0 0.0 0 0.1518 [ 0.036  0.054  1.388  0.043 -0.072 -0.143 -0.246 -0.306 -0.681  7.712]
48 2 0.9976 0 0.1551 [ 0.379  7.768  0.104 -0.31   0.131  0.126  0.104  0.154  0.132  0.141]
26 3 0.9978 0 -0.0541 [ 1.40e-02 -2.00e-03 -2.20e-02 -2.80e-02 -5.50e-02 -6.50e-02 -9.10e-02
 -1.03e-01 -5.59e+00 -1.49e-01]
0 0 0.9991 1 0.1244 [ 0.039  0.147  0.196  0.221  0.267  0.314 -4.514  0.373  0.386  0.432]
53 1 0.9947 0 0.0043 [0.005 0.041 0.065 0.05  4.096 0.088 0.084 0.105 0.1   0.065]
35 3 0.9995 0 0.1276 [ 0.032  0.06   0.149  0.131 -3.694  0.228  0.255  0.283  0.291  0.303]
123 1 0.0 0 0.181 [ 0.046  0.043 -0.004 -0.057 -0.131 -0.199 -0.279 -0.374 -0.433 -3.453]
143 1 0.0 1 -0.0352 [-0.024 -0.061 -0.108 -0.193 -0.24  -0.32  -0.374 -0.473 -0.542 -3.156]
113 3 0.9993 0 0.1198 [ 0.043  0.059  0.127  0.162 -0.172  0.24   0.279  1.775  0.342  2.465]
103 1 0.9979 0 0.0802 [0.016 0.069 0.063 0.062 0.075 0.114 0.08  3.064 0.122 0.115]
127 0 0.0 0 -0.005 [-0.028  0.017 -0.008  3.013  0.561 -0.033 -0.035 -0.021  0.025  0.025]
2 2 0.9986 0 0.1218 [-0.1    0.069  0.085 -2.596  0.135  0.104  0.137  0.122  0.115  0.099]
138 2 0.0 0 0.1428 [ 0.045  0.044  0.33   0.021 -1.757 -0.885 -0.2   -0.271 -0.353 -0.467]
158 2 0.0 0 -0.0432 [ 0.012 -0.056 -0.063 -0.1    1.87  -0.114 -0.15  -0.185 -0.203 -0.26 ]
76 2 0.9989 0 0.0958 [ 0.033  0.067  0.068 -1.636  0.158  0.189  0.239  0.307  0.294  0.334]
over 0.5 counts by lab/horizon [(0, [0, 2, 3, 3, 3, 2, 5, 1, 3, 5]), (1, [1, 1, 2, 2, 1, 3, 0, 2, 2, 5]), (2, [2, 2, 2, 2, 2, 2, 0, 2, 3, 2]), (3, [1, 0, 0, 1, 4, 1, 1, 4, 5, 7])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=022b97b77b0e3c47 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_022b97b7 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`39697 ms` LLM)
- **Rationale**: An isolated strain residual of 25.23 next to residuals near zero points to observation contamination, not a missing relaxation term. Compare robust and ordinary fits on held-out labs without editing the candidate. Retain raw RMSE and report median and 90th-percentile curve errors separately to see whether robust fitting improves typical predictions without worsening the rate.
- **Alternatives Weighed**: `Drop entire stress histories with a large residual` (rejected: The largest spikes are isolated within curves whose neighboring measurements and initial rates remain informative.)

```bash
set -e
export PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1
python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS' >/tmp/rheo_skill_search.txt || true
head -c 3600 /tmp/rheo_skill_search.txt
cat >/tmp/rheo_robust_cv.py <<'PY'
import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m

def robust(dt,t,b,y,r,scale,rate_weight,log=True):
    u=np.r_[y.ravel(),r]
    fixed=np.r_[np.ones(y.size),np.full(r.size,rate_weight)]
    per_curve=np.tile(scale*(1+0.35*t),len(y))
    noise=np.r_[per_curve,np.full(r.size,scale*2)]
    def proj(x,full=False):
        c,a=m._basis(np.exp(x),dt,t,b,log)
        mat=np.concatenate((c.reshape(-1,9),a))
        w=np.linalg.lstsq(mat*wt[:,None],u*wt,rcond=None)[0]
        err=mat@w-u
        return (w,err) if full else err*wt
    wt=fixed
    starts=[np.log([.45,2.7]),np.log([.8,7.])]
    for iteration in range(12):
        opt=min((least_squares(proj,x,bounds=(np.log(.005),np.log(250.)),max_nfev=45) for x in starts),key=lambda f:f.cost)
        w,e=proj(opt.x,True)
        new=fixed/np.hypot(1,e/noise)
        if np.max(np.abs(new-wt))<1e-4: break
        wt=new
        starts=[opt.x]
    return np.exp(opt.x),w

if __name__=='__main__':
    d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; labs=s['lab_id']; active=b[-1]>0
    rm=lambda x: np.sqrt(np.mean(x*x))
    for scale,q,log in [(None,1,True),(.025,2,True),(.06,1,True),(.06,2,True),(.06,4,True),(.14,2,True),(.06,2,False)]:
        fit=(lambda bb,yy,rr:m._fit(dt,t,bb,yy,rr,q,log)) if scale is None else (lambda bb,yy,rr:robust(dt,t,bb,yy,rr,scale,q,log))
        ec=np.empty_like(y); er=np.empty_like(r); est=[]
        for lab in np.unique(labs):
            te=labs==lab; tr=~te
            tau,w=fit(tuple(x[tr] for x in b),y[tr],r[tr]); est.append(tau)
            c,a=m._basis(tau,dt,t,tuple(x[te] for x in b),log); ec[te]=c@w-y[te]; er[te]=a@w-r[te]
        wc=max(np.median(abs(ec[labs==lab,k])) for lab in np.unique(labs) for k in range(len(t)))
        wr=max(rm(er[(labs==lab)&(active==on)]) for lab in np.unique(labs) for on in (False,True))
        wp=max(np.quantile(abs(ec[(labs==lab)&(active==on)]),.90) for lab in np.unique(labs) for on in (False,True))
        wh=max(rm(ec[labs==lab,k]) for lab in np.unique(labs) for k in range(len(t)))
        print(scale,q,log,'worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE',np.round([wc,wh,wp,wr],5),'taus',np.round(est,4).tolist(),flush=True)
        tau,w=fit(b,y,r); print('all',np.round(tau,5),np.round(w,5),flush=True)
PY
python3 /tmp/rheo_robust_cv.py
```
- **Container Exit Code**: `0` (`3696 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
y increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
None 1 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.20346 4.18454 0.48086 0.19162] taus [[0.5005, 10.8251], [0.5378, 5.6915], [0.4634, 2.3835], [0.4803, 2.1306]]
all [0.50354 2.95507] [ 0.64985  0.51243  0.092    0.08933 -0.08284  0.07738 -0.0316   0.32075
  0.24568]
0.025 2 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.04561 4.21689 1.07306 0.08679] taus [[0.5002, 3.7581], [0.5004, 3.8295], [0.501, 3.7643], [0.5005, 3.7352]]
all [0.50046 3.77005] [ 0.67694  0.54623  0.13664  0.15391 -0.12521  0.1152  -0.08702  0.28202
  0.23839]
0.06 1 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.14521 4.20882 0.95063 0.09935] taus [[0.5017, 3.5889], [0.4977, 3.7695], [0.5006, 3.692], [0.489, 2.9876]]
all [0.49913 3.65769] [ 0.67058  0.54199  0.1276   0.14024 -0.11769  0.10981 -0.08005  0.28679
  0.23388]
0.06 2 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.06476 4.21137 0.9868  0.08856] taus [[0.5018, 3.7461], [0.4997, 3.8207], [0.5005, 3.7209], [0.4995, 3.5362]]
all [0.50036 3.72569] [ 0.67581  0.54308  0.13068  0.1451  -0.12037  0.11151 -0.0829   0.28341
  0.23646]
0.06 4 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.05356 4.21514 1.04126 0.08732] taus [[0.5004, 3.7312], [0.5011, 3.8491], [0.5007, 3.7632], [0.5017, 3.6823]]
all [0.50088 3.76108] [ 0.6778   0.54324  0.13419  0.1506  -0.12343  0.11379 -0.08577  0.28223
  0.23778]
0.14 2 True worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.16047 4.18995 0.75833 0.11418] taus [[0.5045, 3.646], [0.4944, 3.6449], [0.4996, 3.5861], [0.4944, 2.9934]]
all [0.49898 3.51639] [ 0.66936  0.53873  0.10622  0.11788 -0.09798  0.09557 -0.05972  0.28532
  0.2326 ]
0.06 2 False worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE [0.09752 4.16195 0.25705 0.21093] taus [[0.497, 3.373], [0.497, 3.3367], [0.5041, 3.8437], [0.5011, 3.5133]]
all [0.49988 3.49924] [ 0.65703  0.54664  0.05768  0.06432 -0.05256  0.04859 -0.03593  0.2927
  0.09652]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=022b97b77b0e3c47 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_022b97b7 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`25664 ms` LLM)
- **Rationale**: Separate residuals by activation, missing coupons, and horizon. If large errors are isolated in time, retain the physical basis and downweight those observations; if they persist across a coupon's curve, review the terminal gate and imputation before changing the fit. Inspect the reused calibration code to identify whether its archive fallback follows the stated triggers.
- **Alternatives Weighed**: `Promote the strongest robust fit immediately` (rejected: The observed tail regression is large enough to check whether it reflects scattered observation spikes or a systematic error on an identifiable subset.)

```bash
set -e
export PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1
python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -c 3000
printf '\n'; sed -n '1,65p' /holdout_task/alpha_m97_c02_batch_01_solver.py
python3 - <<'PY'
import pickle,numpy as np
from rheo_robust_cv import robust,m
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; l=s['lab_id']; on=b[-1]>0; missing=np.any(np.isnan(s['coupon_features_raw']),axis=1)
ta,w=robust(dt,t,b,y,r,.025,2); c,a=m._basis(ta,dt,t,b); e=y-c@w; v=r-a@w
for active in [False,True]:
 for miss in [False,True]:
  z=(on==active)&(missing==miss)
  if not z.any(): continue
  print('active,missing,n',active,miss,z.sum(),'curve abs q50,80,90 by horizon',np.round(np.quantile(abs(e[z]),[.5,.8,.9],axis=0),3),'rate residual q05,50,95',np.round(np.quantile(v[z],[.05,.5,.95]),4))
print('largest terminal residuals: lab activation missing rate, curve')
for i in np.argsort(abs(e[:,-1]))[-14:]: print(i,l[i],round(b[-1][i],3),missing[i],round(v[i],3),np.round(e[i],3))
print('metadata',[(k,str(d[k])[:400]) for k in d if not isinstance(d[k],(dict,np.ndarray))])
PY
```
- **Container Exit Code**: `0` (`687 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.853 0.882 0.987 1.483 1.679]] rate residual q05,50,95 [-0.2339 -0.0472  0.1286]
active,missing,n True False 112 curve abs q50,80,90 by horizon [[0.011 0.01  0.011 0.013 0.013 0.01  0.011 0.013 0.014 0.013]
 [0.025 0.02  0.022 0.024 0.023 0.021 0.02  0.025 0.027 0.028]
 [0.036 0.029 0.031 0.043 0.034 0.028 0.029 0.039 0.038 0.037]] rate residual q05,50,95 [-0.0277 -0.0005  0.0315]
active,missing,n True True 8 curve abs q50,80,90 by horizon [[0.02  0.032 0.068 0.07  0.076 0.143 0.11  0.116 0.12  0.134]
 [0.028 0.097 0.125 0.142 0.143 0.339 0.359 0.215 0.216 0.244]
 [0.046 0.139 0.188 0.202 0.247 0.433 1.73  0.329 0.346 0.365]] rate residual q05,50,95 [-0.2326  0.0191  0.0932]
largest terminal residuals: lab activation missing rate, curve
133 0 0.0 False 0.012 [ 0.018 -0.067 -0.124 -0.292 -0.442 -0.592 -0.747 -0.887 -1.039 -1.232]
124 3 0.0 True -0.268 [-0.092 -0.232 -0.358 -0.487 -0.605  0.456 -0.882 -0.987 -1.112 -1.25 ]
138 2 0.0 False 0.031 [ 0.005 -0.089  0.101 -0.303 -2.173 -1.391 -0.794 -0.952 -1.119 -1.319]
156 3 0.0 False -0.011 [ 0.012 -0.098 -0.194 -0.339 -0.457 -0.604 -0.756 -0.925 -1.101 -1.455]
132 2 0.0 False 0.004 [-0.     0.107  0.234  0.368  0.543  0.716  0.888  1.081  1.247  1.471]
137 2 0.0 True -0.053 [-0.051 -0.156 -0.25  -0.456 -0.659 -0.853 -1.021 -1.251 -1.483 -1.679]
166 0 0.0 False -0.017 [-0.005 -0.106 -0.254 -0.424 -0.589 -0.826 -1.027 -1.228 -1.45  -1.688]
178 1 0.0 False -0.016 [-0.027 -1.037 -0.278 -0.448 -0.657 -0.827 -1.065 -1.286 -1.552 -1.765]
136 2 0.0 False 0.003 [ 0.013 -0.112 -0.281 -0.455 -0.674 -0.879 -1.127 -1.349 -1.586 -1.863]
113 3 0.999 False 0.004 [ 0.011 -0.02   0.004 -0.    -0.373  0.003  0.008  1.472  0.009  2.103]
143 1 0.0 True -0.2 [-0.068 -0.171 -0.277 -0.422 -0.528 -0.669 -0.785 -0.946 -1.077 -3.752]
123 1 0.0 False 0.002 [-1.000e-03 -7.200e-02 -1.840e-01 -3.060e-01 -4.520e-01 -5.950e-01
 -7.530e-01 -9.280e-01 -1.068e+00 -4.168e+00]
142 0 0.0 False 0.009 [-0.015 -0.113  1.1   -0.367 -0.6   -0.785 -1.002 -1.174 -1.661  6.622]
51 0 0.997 False 0.018 [ 1.100e-02  2.200e-02 -6.250e-01 -8.000e-03  1.800e-02  1.800e-02
  2.000e-02 -5.000e-03  6.000e-03 -1.773e+01]
metadata [('crit_stress', '2.65'), ('yield_stress', '2.65'), ('shear_banding_index', '0.12')]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=022b97b77b0e3c47 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_022b97b7 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`30965 ms` LLM)
- **Rationale**: Leave-one-lab-out comparisons will show whether a different terminal basis resolves the coherent zero-damage error without sacrificing the active response or rate prediction. The specified active equations stay fixed in every comparison.
- **Alternatives Weighed**: `Downweight zero-damage curves more aggressively` (rejected: It would hide coherent errors across horizons rather than test the unsupported assumption that zero-damage coupons follow linear terminal flow.)
- **ATMS NOGOOD DECLARED**: Refuted `The current damage-gated linear/logarithmic switch is sufficient to explain the response of complete-panel, zero-damage coupons.` | Witness: `Under robust fitting, 49 inactive complete-panel rows have terminal median absolute strain error 0.497 despite initial-rate residuals concentrated near zero; 112 active complete-panel rows have terminal median 0.013.` | Blanket Diagnosis: `The persistent horizon-dependent residual is present without missing features, and the initial derivative cannot distinguish t from log(1+t). This isolates the unchecked scope of the terminal time basis.` | Sublated Pivot: `Compare linear, intermediate, and logarithmic terminal bases for zero-damage rows while retaining the specified active response exactly.`

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cd /tmp
sed -n '1,60p' /holdout_task/alpha_m97_c02_batch_01_solver.py
cat >/tmp/rheo_plateau_scope.py <<'PY'
import pickle,numpy as np
from rheo_robust_cv import m,robust
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; dt,t,(b,_)=m._prepare(d); s=d['train']; labs=s['lab_id']; on=b[-1]>0; missing=np.isnan(s['coupon_features_raw']).any(axis=1); y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; original=m._basis
rm=lambda x: np.sqrt(np.mean(x*x))
for mix in [0.,.5,1.]:
 def basis(tau,dt,t,b,log=True):
  C,A=original(tau,dt,t,b,True); idx=b[-1]==0
  terminal=(1-mix)*t+mix*np.log1p(t)
  C[idx,:,2:7]=b[2][idx,None,:]*terminal[None,:,None]
  return C,A
 m._basis=basis
 ec=np.empty_like(y); er=np.empty_like(r); params=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=robust(dt,t,tuple(x[tr] for x in b),y[tr],r[tr],.025,2); params.append(tau)
  C,A=basis(tau,dt,t,tuple(x[te] for x in b)); ec[te]=C@w-y[te]; er[te]=A@w-r[te]
 print('inactive log fraction',mix,'taus',np.round(params,4).tolist(),'curve pooled raw RMSE',rm(ec))
 for active in [False,True]:
  group=[(labs==lab)&(on==active) for lab in np.unique(labs)]
  print('active',active,'worst lab median / p90 / rate RMSE',max(np.median(abs(ec[z])) for z in group),max(np.quantile(abs(ec[z]),.9) for z in group),max(rm(er[z]) for z in group),'complete panel worst terminal median',max(np.median(abs(ec[z&~missing,-1])) for z in group))
 tau,w=robust(dt,t,b,y,r,.025,2); print('full',tau,w)
PY
python3 /tmp/rheo_plateau_scope.py
```
- **Container Exit Code**: `0` (`1850 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array([(u, v) for c, u, v in rows(records) if int(c) == lab and np.isfinite(u) and np.isfinite(v)]).reshape(-1, 2)
    if len(pairs) >= 2:
        gain, bias = _line(*pairs.T)
        if len(x):
            bias = np.mean(y - gain*x)
        return gain, bias
    return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)


def _align(data):
    train, test = data['train'], data['test']
inactive log fraction 0.0 taus [[0.5002, 3.7581], [0.5004, 3.8295], [0.501, 3.7643], [0.5005, 3.7352]] curve pooled raw RMSE 0.9838094180950299
active False worst lab median / p90 / rate RMSE 0.2432291675016338 1.0730629083498364 0.08679249982400265 complete panel worst terminal median 0.751759657273249
active True worst lab median / p90 / rate RMSE 0.01630628684587121 0.18016216423189638 0.06503106334195387 complete panel worst terminal median 0.017053754656727538
full [0.50045552 3.77004863] [ 0.67694327  0.5462285   0.13663917  0.15391217 -0.12521414  0.11519789
 -0.08702361  0.28202042  0.23839201]
inactive log fraction 0.5 taus [[0.5004, 3.7441], [0.4999, 3.8019], [0.5007, 3.7488], [0.5001, 3.7301]] curve pooled raw RMSE 0.9553452241712269
active False worst lab median / p90 / rate RMSE 0.12613976094352064 0.5608894830308723 0.08638642573260202 complete panel worst terminal median 0.3698746083648743
active True worst lab median / p90 / rate RMSE 0.017163393049461795 0.181708903565005 0.06524943856295909 complete panel worst terminal median 0.01860747691226816
full [0.50020376 3.75583436] [ 0.67597377  0.54484336  0.1369292   0.1529822  -0.12493615  0.11527483
 -0.08706871  0.28269324  0.23786482]
inactive log fraction 1.0 taus [[0.4996, 3.7189], [0.5002, 3.757], [0.5007, 3.7773], [0.5005, 3.7858]] curve pooled raw RMSE 0.9457154767589052
active False worst lab median / p90 / rate RMSE 0.016556972015111096 0.2637716709954374 0.08674002368228532 complete panel worst terminal median 0.025143380225655765
active True worst lab median / p90 / rate RMSE 0.013856678736779392 0.17366162508682687 0.06282076770397102 complete panel worst terminal median 0.014270552831020744
full [0.50023954 3.75418795] [ 0.6767833   0.5479508   0.1388931   0.15637793 -0.12742595  0.11725568
 -0.08821308  0.28065834  0.23982789]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=022b97b77b0e3c47 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_022b97b7 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`52508 ms` LLM)
- **Rationale**: Promote the logarithmic basis and pointwise robust fit: they reduce the coherent inactive-coupon error while preserving the specified active equations. Re-run the lab holdouts against the exact artifact to catch implementation regressions, then independently check the history propagation and initial derivative for all test rows.
- **Alternatives Weighed**: `Keep a mixed linear/logarithmic terminal term for inactive coupons` (rejected: The mixed basis left worst inactive terminal median error at 0.370 versus 0.025 for the logarithmic basis, without a meaningful rate benefit.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
s = s.replace("    blocks = []\n", "    plateau = data['shear_banding_index'] <= 0.50\n    blocks = []\n")
s = s.replace("if data['shear_banding_index'] <= 0.50 else", "if plateau else")
s = s.replace("aligned)), act))", "aligned)), act, np.full(len(hold), plateau)))")
s = s.replace('def _basis(taus, dt, t, block, log_plateau=True):\n    past, hold, coupons, act = block', 'def _basis(taus, dt, t, block):\n    past, hold, coupons, act, plateau = block')
s = s.replace('    terminal = np.where(act[:, None] > 0.0, np.log1p(t), t) if log_plateau else np.broadcast_to(t, (len(hold), len(t)))', '    # Low shear selects the terminal time law; work damage scales only the two corrections.\n    terminal = np.where(plateau[:, None], np.log1p(t), t)')
a = s.index('def _fit(')
b = s.index('def solve_alpha_m97_c02_batch_01', a)
s = s[:a] + '''def _fit(dt, t, block, y, rate):
    target = np.concatenate((y.ravel(), rate))
    base = np.concatenate((np.ones(y.size), np.full(rate.size, 2.0)))
    # Isolated strain spikes should not outweigh the rest of the same curve.
    scale = np.concatenate((np.tile(0.025 * (1.0 + 0.35 * t), len(y)),
                            np.full(rate.size, 0.05)))
    sqrt_weight = base

    def project(log_taus, return_weights=False):
        curves, rates = _basis(np.exp(log_taus), dt, t, block)
        design = np.concatenate((curves.reshape(-1, 9), rates))
        weights = np.linalg.lstsq(design * sqrt_weight[:, None], target * sqrt_weight,
                                  rcond=None)[0]
        residual = np.einsum('ij,j->i', design, weights) - target
        return (weights, residual) if return_weights else residual * sqrt_weight

    bounds = (np.log(0.005), np.log(250.0))
    starts = np.log([[0.45, 2.7], [0.8, 7.0]])
    for _ in range(12):
        fit = min((least_squares(project, start, bounds=bounds, max_nfev=45)
                   for start in starts), key=lambda result: result.cost)
        weights, residual = project(fit.x, return_weights=True)
        updated = base / np.hypot(1.0, residual / scale)
        if np.max(np.abs(updated - sqrt_weight)) < 1e-4:
            break
        sqrt_weight = updated
        starts = [fit.x]
    return np.exp(fit.x), weights


''' + s[b:]
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
s = d['train']; dt, t, (b, test) = m._prepare(d)
y = s['strain_curves_obs'] - s['strain_0'][:,None]; r = s['instant_rate_obs']; lab = s['lab_id']
on = b[3] > 0; complete = np.isfinite(s['coupon_features_raw']).all(axis=1)
ec = np.empty_like(y); er = np.empty_like(r)
for k in range(4):
    te = lab == k; tr = ~te
    tau, w = m._fit(dt, t, tuple(x[tr] for x in b), y[tr], r[tr])
    c, a = m._basis(tau, dt, t, tuple(x[te] for x in b))
    ec[te] = c @ w - y[te]; er[te] = a @ w - r[te]
    print('held-out lab', k, 'taus', tau.round(5))
groups = [(lab == k) & (on == z) for k in range(4) for z in (False, True)]
rmse = lambda x: np.sqrt(np.mean(x*x))
metrics = (max(np.median(abs(ec[g & complete, -1])) for g in groups if np.any(g & ~on)),
           max(np.quantile(abs(ec[g]), .90) for g in groups), max(rmse(er[g]) for g in groups), rmse(ec))
print('worst inactive complete terminal median, worst lab/regime curve P90 and rate RMSE, pooled raw curve RMSE:', np.round(metrics, 6))
assert np.all(np.asarray(metrics) < [.04, .30, .10, 1.0])
out = m.solve_alpha_m97_c02_batch_01(d)
tau = out['estimated_taus']; hw = out['hereditary_weights']; rw = out['regime_mode_weights']
assert 0 < tau[0] < tau[1] and all(np.isfinite(v).all() for v in out.values())
# Independent accumulated-history formula and derivative check, including the zero-damage case.
past, hold, x, act, _ = test
lags = np.cumsum(dt[::-1])[::-1] - dt
z = past @ ((1-np.exp(-dt[:,None]/tau))*np.exp(-lags[:,None]/tau))
gap = hold[:,None] - z
drift = x @ hw[2:7] + rw[1]*act*z[:,0]
rise = 1-np.exp(-t[:,None]/tau)
amp = hw[:2]*gap; amp[:,0] += rw[0]*act*hold
np.testing.assert_allclose(out['predicted_strain'], d['test']['strain_0'][:,None] + amp @ rise.T + drift[:,None]*np.log1p(t), atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], amp @ (1/tau) + drift, atol=1e-12)
print('contract and independent history/derivative equations verified; full taus', tau.round(6))
PY
```
- **Container Exit Code**: `0` (`1395 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held-out lab 0 taus [0.49964 3.7189 ]
held-out lab 1 taus [0.50025 3.75699]
held-out lab 2 taus [0.50067 3.77727]
held-out lab 3 taus [0.50051 3.78584]
worst inactive complete terminal median, worst lab/regime curve P90 and rate RMSE, pooled raw curve RMSE: [0.025143 0.263772 0.08674  0.945715]
contract and independent history/derivative equations verified; full taus [0.50024  3.754188]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=986a43abf90515f6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_986a43ab (directory).
```

### Turn 10 — `submit_completed_task` (`14969 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, independent history and derivative checks, and leave-one-lab-out stress tests. The host confirmed the previous structural issue was resolved on this exact candidate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_01_solver.py': 'FILE:6874'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema and independent strain and initial-rate checks passed. Worst held-out lab/regime strain P90 absolute error was 0.264, and worst initial-rate RMSE was 0.087.`

### Turn 11 — `execute_in_remote_docker` (`50807 ms` LLM)
- **Rationale**: Compare the saved fit against prespecified nonlinear drift corrections and a fit that excludes imputed rows from parameter estimation while still predicting them. A consistent improvement on held-out labs would justify a fit change; otherwise retain the verified basis. The transient test checks whether an EMI spike can displace a well-supported calibration; failure would call for revising spike detection.
- **Alternatives Weighed**: `Test relay modes, control PDEs, or classification balanced accuracy` (rejected: This batch contains continuous strain and rate measurements, not relays, a control PDE, or class labels. The relevant nuisance check is transfer across labs and signed residuals within labs.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
cat >/tmp/rheo_margin_audit.py <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c02_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
dt, t, (b, _) = m._prepare(d)
s = d['train']
y = s['strain_curves_obs'] - s['strain_0'][:, None]
r = s['instant_rate_obs']
lab = s['lab_id']
complete = np.isfinite(s['coupon_features_raw']).all(axis=1)
active = b[3] > 0
sub = lambda b, mask: tuple(x[mask] for x in b)
rmse = lambda x: np.sqrt(np.mean(x*x))
print('dataset fields', sorted(d), 'train counts by lab (complete, incomplete)', [(int(np.sum((lab == k)&complete)), int(np.sum((lab == k)&~complete))) for k in range(4)])
for method in ('baseline', 'complete-fit', 'nonlinear-drift-probe'):
    ec = np.empty_like(y)
    er = np.empty_like(r)
    folds = []
    for k in range(4):
        te = lab == k
        tr = (lab != k) & complete if method == 'complete-fit' else lab != k
        tau, w = m._fit(dt, t, sub(b, tr), y[tr], r[tr])
        folds.append(tau)
        c, a = m._basis(tau, dt, t, sub(b, te))
        ec[te], er[te] = c @ w - y[te], a @ w - r[te]
        if method == 'nonlinear-drift-probe':
            # A drift correction must act through the same time law and its initial slope.
            # Fit the prespecified directions only on the outer training labs.
            i, j = b[2][:, 1], b[2][:, 2]
            z = np.column_stack(((i-j)**2, (i+j)**2, i/(1+j*j)-i))
            tr = (lab != k) & complete
            ct, at = m._basis(tau, dt, t, sub(b, tr))
            x = z[tr]
            offset = np.median((y[tr] - ct @ w)/np.log1p(t), axis=1)
            slope = r[tr] - at @ w
            beta = np.linalg.lstsq(x, (np.clip(offset, -.08, .08) + np.clip(slope, -.08, .08))/2, rcond=None)[0]
            delta = z[te] @ beta
            ec[te] += delta[:, None]*np.log1p(t)
            er[te] += delta
    groups = [(lab==k)&(active==on) for k in range(4) for on in (False, True)]
    print(method, 'worst complete terminal median, curve P90, rate RMSE, raw curve RMSE', np.round([max(np.median(abs(ec[g&complete,-1])) for g in groups), max(np.quantile(abs(ec[g]), .9) for g in groups), max(rmse(er[g]) for g in groups), rmse(ec)], 6), 'tau fold span', np.round(np.ptp(folds, axis=0), 6), 'complete signed rate medians by lab', np.round([np.median(er[(lab==k)&complete]) for k in range(4)], 6))

# Check that one transient in a real sweep does not move a well-supported in-batch fit.
cal = d['calibration_load_cells']
x, z, channel = (np.asarray(cal[key]) for key in ('cmd_stress', 'true_stress', 'lab_id'))
base = m._calibration(d)
worst = 0.
for k in range(4):
    mask = channel == k
    for row in np.flatnonzero(mask):
        for spike in (-3.1, 3.1):
            noisy = z.copy(); noisy[row] += spike
            got = m._calibration(dict(d, calibration_load_cells=dict(cal, true_stress=noisy)))
            delta = (got[k,0]-base[k,0])*x[mask]+got[k,1]-base[k,1]
            worst = max(worst, float(np.max(abs(delta))))
print('max calibrated stress change after one +/-3.1 transient', worst)
assert worst < .001
PY
python3 -I /tmp/rheo_margin_audit.py
```
- **Container Exit Code**: `0` (`1844 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
dataset fields ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress'] train counts by lab (complete, incomplete) [(36, 5), (33, 4), (47, 5), (45, 5)]
baseline worst complete terminal median, curve P90, rate RMSE, raw curve RMSE [0.025143 0.263772 0.08674  0.945715] tau fold span [0.001033 0.066943] complete signed rate medians by lab [-0.005894  0.005379  0.004179  0.004093]
complete-fit worst complete terminal median, curve P90, rate RMSE, raw curve RMSE [0.0244   0.264774 0.086386 0.945691] tau fold span [0.001559 0.062489] complete signed rate medians by lab [-0.00605   0.004308  0.004421  0.0039  ]
nonlinear-drift-probe worst complete terminal median, curve P90, rate RMSE, raw curve RMSE [0.025345 0.263771 0.086751 0.945704] tau fold span [0.001033 0.066943] complete signed rate medians by lab [-0.00524   0.005003  0.004024  0.004281]
max calibrated stress change after one +/-3.1 transient 7.508421754853778e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=986a43abf90515f6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_986a43ab (directory).
```

### Turn 12 — `submit_completed_task` (`17067 ms` LLM)
- **Rationale**: The saved candidate passed isolated execution and independent formula checks. The focused audit found no material improvement from either challenger, and a single ±3.1 EMI transient shifted calibrated stress by at most 0.0000751 across the tested sweeps. Retain the verified artifact.
- **Alternatives Weighed**: `Exclude imputed rows from fitting or add nonlinear drift terms` (rejected: Neither produced a meaningful held-out improvement; complete-panel signed rate medians were already within 0.006 of zero in every lab.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema and independent strain and initial-rate checks passed. Across held-out labs and damage regimes, the worst strain P90 absolute error was 0.264 and the worst initial-rate RMSE was 0.087.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
