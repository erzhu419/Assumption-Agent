# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `356,433` (`341,880` / `103,168` / `14,553`) |
| **Total Episode Wall Time (s)** | `460.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`21496 ms` LLM)
- **Rationale**: Inspect the schema before coding to catch mismatched field names, nesting, and train/test boundaries. The result determines the preprocessing interface and which measured targets can support held-out validation.
- **Alternatives Weighed**: `Reuse a prior solver unchanged` (rejected: The public gate reports a failed prior branch binding, and the current task specifies a distinct saturated-flow law.)

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && cat public_asset_index.json; ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' 2>&1; python3 - <<'PY'
import pickle
from pathlib import Path
p=Path('/holdout_task/public_alpha_m97_c01_batch_03.pkl')
d=pickle.loads(p.read_bytes())
def walk(d, level=0):
 for k,v in d.items():
  print(' '*level,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''), ('len='+str(len(v))) if isinstance(v,(list,tuple,dict)) else '',repr(v)[:260] if not isinstance(v,dict) else '')
  if isinstance(v,dict) and level<2: walk(v,level+1)
walk(d)
PY
```
- **Container Exit Code**: `1` (`448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "<stdin>", line 6, in walk
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`22214 ms` LLM)
- **Rationale**: Correct the list-versus-dict inspection error. Inspect measured labels and panel conventions before choosing the fitting objective, and check calibration counts and residuals to distinguish direct fits from archive fallback.
- **Alternatives Weighed**: `Use archived gains as defaults` (rejected: Adequate clean in-batch observations take precedence; the per-channel counts and affine residuals determine whether fallback is needed.)

```bash
cd /holdout_task
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation
cat /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
python3 - <<'PY'
import pickle, numpy as np
items=pickle.load(open('public_alpha_m97_c01_batch_03.pkl','rb'))
print('public item count',len(items),'first item keys',list(items[0]))
def walk(d, prefix=''):
 for k,v in d.items():
  if isinstance(v,dict): walk(v,prefix+k+'.'); continue
  a=np.asarray(v)
  print(prefix+k,a.shape,a.dtype, ('nan='+str(np.isnan(a).sum())+' '+str(a.ravel()[:6])) if a.dtype.kind in 'fiu' else str(v)[:100])
for item in items[:2]: walk(item['dataset'])
for item in items:
 d=item['dataset']; c=d['calibration_load_cells']; print('band, times',d['shear_banding_index'],d['eval_times']);
 for ch in range(4):
  m=c['lab_id']==ch; x=c['cmd_stress'][m]; y=c['true_stress'][m]; print('cal',ch,len(x), 'unique commands',len(np.unique(x)), 'max affine residual',np.max(np.abs(y-np.polyval(np.polyfit(x,y,1),x))) if len(np.unique(x))>=2 else None)
PY
head -c 1800 /tmp/hegel_continual_archive/archive_calibration_history.json
```
- **Container Exit Code**: `0` (`555 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5961]
test.stress_hold_cmd (72,) float64 nan=0 [-1.1645903   2.38651818 -1.1693876  -0.4732      0.35694104  0.38480519]
test.strain_0 (72,) float64 nan=0 [ 0.052036 -0.500013 -0.481813  0.101502  0.322033 -0.139833]
test.aux_chamber_telemetry (72, 3) float64 nan=0 [-0.175019  0.782023  1.124864  1.075463  1.483753  1.269029]
band, times 0.82 [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
cal 0 10 unique commands 10 max affine residual 0.0002167704491329303
cal 1 10 unique commands 10 max affine residual 0.00014201764236299041
cal 2 10 unique commands 10 max affine residual 0.00016161625739985475
cal 3 10 unique commands 10 max affine residual 0.00017209599156450817
[
  {
    "episode_short_name": "alpha_m97_c01_s00",
    "lab_id": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "cmd_stress": [
      1.660283,
      -0.418768,
      1.112723,
      -1.62604,
      -0.501169,
      -1.672067,
      0.863389,
      1.008755,
      1.968856,
      1.530083,
      0.979468,
      0.449112,
      1.469886,
      -0.332491,
      -1.532413,
      0.660575,
      -1.672999,
      -1.196407,
      0.934354,
      0.731875,
      1.670914,
      -2.075727,
      1.021513,
      -1.730233,
      -1.965428,
      0.955855,
      0.129182,
      -0.328527,
      -1.550117,
      -1.993809,
      2.131528,
      -2.023748,
      0.667305,
      -0.968757,
      -0.649251,
      -0.022422,
      -1.62562,
      -0.902597,
      1.990079,
      -0.41266
    ],
    "true_stress": [
      1.660305,
      -0.418669,
      1.112724,
      -1.626001,
      -0.501125,
      -1.671976,
      0.863198,
      1.008634,
      1.968816,
      1.530248,
      0.979291,
      0.449215,
      1.470001,
      -0.332446,
      -1.532394,
      0.660518,
      -1.672977,
      -1.196416,
      0.934345,
      0.731869,
      1.670821,
      -2.075755,
      1.021515,
      -1.730423,
      -1.965592,
      0.95586,
      0.129076,
      -0.328478,
      -1.550116,
      -1.993815,
      2.131554,
      -2.023643,
      0.667296,
      -0.968674,
      -0.649069,
      -0.022271,
      -1.625842,
      -0.902458,
      1.990138,
      -0.412675
```

### Turn 3 — `execute_in_remote_docker` (`18660 ms` LLM)
- **Rationale**: The previous log truncated the archived implementation. Read it separately to resolve the panel fill convention and sparse calibration fallback before coding. Cell counts determine whether a complete-grid weighting helper applies; the isolated SciPy import determines whether the solver can use its optimizer.
- **Alternatives Weighed**: `Use archived coefficients on the public batch` (rejected: All four public channels have ten clean in-batch observations, so the specified direct fit takes precedence.)

```bash
sed -n '1,175p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py; python3 -I - <<'PY'
import json, pickle
import numpy as np
import scipy
base='/tmp/hegel_continual_archive/'
a=json.load(open(base+'archive_calibration_history.json'))
for entry in a:
    print('archive', entry.get('episode_short_name'), list(entry), np.bincount(entry['lab_id']).tolist())
print('scipy',scipy.__version__)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
print('past dt range',np.min(d['past_dt']),np.max(d['past_dt']))
for split in ('train','test'):
    s=d[split]; print(split,[(int(l),int(t),int(np.sum((s['lab_id']==l)&(s['stratum']==t)))) for l in np.unique(s['lab_id']) for t in np.unique(s['stratum'])])
PY
```
- **Container Exit Code**: `0` (`385 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dc, dr = _basis(p[:2], dt, t, past, hold, x, True)
        return c.reshape(-1, 7), r, dc.reshape(-1, 2)*p[2:4], dr*p[2:4]

    def func(p, sc, sr):
        c, r, _, _ = get(p)
        return np.r_[(c@p[2:] - yc.ravel())/sc, (r@p[2:] - yr)/sr]

    def jac(p, sc, sr):
        c, r, dc, dr = get(p)
        return np.vstack((np.column_stack((dc, c))/sc, np.column_stack((dr, r))/sr))

    bounds = (np.r_[np.log(max(1e-4, min(dt.min(), t.min())/20))*np.ones(2), np.full(7, -np.inf)],
              np.r_[np.log((dt.sum()+t.max())*20)*np.ones(2), np.full(7, np.inf)])
    best = None
    for ts in ((.26, 1.4), (.55, 3.2), (.9, 5.5)):
        logs = np.log(ts)
        p = np.r_[logs, linear(logs)]
        opts = dict(jac=jac, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=100)
        fit = least_squares(func, p, args=(.04, .05), **opts)
        if best is None or fit.cost < best.cost:
            best = fit
    raw = func(best.x, 1, 1)
    sc = 2.5 * _mad(raw[:yc.size])
    sr = 2.5 * _mad(raw[yc.size:])
    fit = least_squares(func, best.x, args=(sc, sr), **opts)
    p = fit.x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    past, hold, x = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    t, dt = dataset['eval_times'], dataset['past_dt']
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, weights = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])
    c, r = _basis(logtau, dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(logtau), hereditary_weights=weights,
                regime_mode_weights=np.zeros(2), predicted_strain=test['strain_0'][:, None]+c@weights,
                initial_rate_pred=r@weights)
archive alpha_m97_c01_s00 ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'] [10, 10, 10, 10]
archive alpha_m97_c01_batch_01 ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'] [10, 10, 10, 10]
archive alpha_m97_c01_batch_02 ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'] [10, 10, 10, 10]
scipy 1.13.0
past dt range 0.2105 0.7877
train [(0, 0, 12), (0, 1, 13), (0, 2, 10), (1, 0, 12), (1, 1, 25), (1, 2, 12), (2, 0, 16), (2, 1, 15), (2, 2, 26), (3, 0, 8), (3, 1, 12), (3, 2, 19)]
test [(0, 0, 5), (0, 1, 7), (0, 2, 5), (1, 0, 2), (1, 1, 10), (1, 2, 4), (2, 0, 6), (2, 1, 3), (2, 2, 12), (3, 0, 6), (3, 1, 8), (3, 2, 4)]
```

### Turn 4 — `execute_in_remote_docker` (`23009 ms` LLM)
- **Rationale**: Recover the panel-fill code that was truncated in the previous log. The finite counts indicate whether held-out cells need a pooled fill, and the compact coefficient comparison determines whether to pool historical calibration rows or use the latest compatible sweep.
- **Alternatives Weighed**: `Pool historical load-cell rows by default when fallback is needed` (rejected: First check whether the channel gains and biases remained stable across episodes; pooling incompatible calibrations would bias the physical stress inputs.)

```bash
sed -n '/^def _prepare/,/^def _basis/p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py; python3 -I - <<'PY'
import json, pickle, numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
a=json.load(open(p))
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
for e in a+[d['calibration_load_cells']]:
    lab,x,y=[np.asarray(e[k]) for k in ('lab_id','cmd_stress','true_stress')]
    print(e.get('episode_short_name','current'), [np.polyfit(x[lab==c],y[lab==c],1).round(5).tolist() for c in range(4)])
s=d['train']; print('duration',np.sum(d['past_dt']),'reference panel shape',s['ref_coupons'].shape)
for l in range(4):
    for z in np.unique(s['stratum']):
        m=(s['lab_id']==l)&(s['stratum']==z)
        print('finite coupon values',l,z,np.isfinite(s['coupon_features_raw'][m]).sum(0).tolist())
PY
```
- **Container Exit Code**: `0` (`475 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
def _prepare(dataset):
    cal = dataset['calibration_load_cells']
    train, test = dataset['train'], dataset['test']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd']))
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]
    for channel in np.unique(lab):
        take = lab == channel
        rows = cal['lab_id'] == channel
        cmd, phys = cal['cmd_stress'][rows], cal['true_stress'][rows]
        gain, bias = _calibrate(cmd, phys, int(channel))
        past[take] = past[take] * gain + bias
        hold[take] = hold[take] * gain + bias
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[take] = (x[take] - center) / scale
        for level in np.unique(stratum[take]):
            cell = take & (stratum == level)
            values = x[cell]
            x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _basis(logtau, dt, t, past, hold, x, derivative=False):
alpha_m97_c01_s00 [[0.99998, 2e-05], [0.99999, 0.0], [1.00001, -5e-05], [1.0, 5e-05]]
alpha_m97_c01_batch_01 [[1.34, 0.14999], [0.77, -0.16999], [1.21003, -0.12], [0.83001, 0.10997]]
alpha_m97_c01_batch_02 [[1.34003, 0.14998], [0.77001, -0.16998], [1.21, -0.12001], [0.82994, 0.11009]]
current [[1.33999, 0.15], [0.77004, -0.17003], [1.20998, -0.11999], [0.83, 0.11001]]
duration 8.6395 reference panel shape (180, 32)
finite coupon values 0 0 [11, 12, 12, 10]
finite coupon values 0 1 [12, 13, 13, 12]
finite coupon values 0 2 [10, 10, 9, 8]
finite coupon values 1 0 [11, 11, 12, 11]
finite coupon values 1 1 [25, 24, 25, 25]
finite coupon values 1 2 [12, 12, 12, 12]
finite coupon values 2 0 [16, 16, 16, 16]
finite coupon values 2 1 [15, 15, 14, 15]
finite coupon values 2 2 [25, 26, 25, 25]
finite coupon values 3 0 [7, 8, 8, 6]
finite coupon values 3 1 [12, 12, 10, 12]
finite coupon values 3 2 [19, 19, 19, 17]
```

### Turn 5 — `execute_in_remote_docker` (`63683 ms` LLM)
- **Rationale**: Implement the stated constitutive law with shared coefficients for strain and initial rate. Initialize each nonlinear fit by solving the linear weights at fixed relaxation times. Use the two compatible archive sweeps only for sparse or EMI-affected calibration, and run the isolated schema validator to catch shape or non-finite output failures before held-out testing.
- **Alternatives Weighed**: `Sort the fitted relaxation times after optimization` (rejected: The cross-gap term relaxes at tau_0 specifically, so swapping the branches is not a symmetry of the model.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_03_solver.py <<'PY'
"""Calibrate the four rheometers and fit the jammed-slip constitutive law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _prior_calibration(channel):
    # The two immediately preceding sweeps share this campaign's calibration.
    sweeps = json.loads(_ARCHIVE.read_text())[-2:]
    cmd, true = [], []
    for sweep in sweeps:
        rows = np.asarray(sweep['lab_id']) == channel
        cmd.extend(np.asarray(sweep['cmd_stress'])[rows])
        true.extend(np.asarray(sweep['true_stress'])[rows])
    return np.polyfit(cmd, true, 1)


def _calibrate(cmd, true, channel):
    if len(cmd) < 2:
        return _prior_calibration(channel)
    i, j = np.triu_indices(len(cmd), 1)
    distinct = cmd[i] != cmd[j]
    i, j = i[distinct], j[distinct]
    slopes = (true[j] - true[i]) / (cmd[j] - cmd[i])
    offsets = np.median(true[:, None] - cmd[:, None] * slopes, axis=0)
    errors = np.abs(true[:, None] - cmd[:, None] * slopes - offsets)
    clean = errors[:, np.argmin(np.median(errors, axis=0))] <= 3.0
    if not clean.all():
        prior = _prior_calibration(channel)
        if clean.sum() < 4:
            prior_clean = np.abs(true - np.polyval(prior, cmd)) <= 3.0
            if prior_clean.sum() >= clean.sum():
                clean = prior_clean
        if np.unique(cmd[clean]).size < 2:
            return prior
    return np.polyfit(cmd[clean], true[clean], 1)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd']))
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]
    for channel in np.unique(lab):
        rows = cal['lab_id'] == channel
        gain, bias = _calibrate(cal['cmd_stress'][rows], cal['true_stress'][rows], channel)
        m = lab == channel
        past[m] = gain * past[m] + bias
        hold[m] = gain * hold[m] + bias
        panel = refs[m]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (x[m] - center) / scale
        for level in np.unique(stratum[m]):
            cell = m & (stratum == level)
            values = x[cell]
            if np.isnan(values).any():
                x[cell] = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _basis(logtau, dt, t, past, hold, x, jac=False):
    tau = np.exp(logtau)
    after = (np.cumsum(dt[::-1])[::-1] - dt)[:, None] / tau
    step_dt = dt[:, None] / tau
    memory_weights = np.exp(-after) * -np.expm1(-step_dt)
    gap = hold[:, None] - past @ memory_weights
    u = t[:, None] / tau
    step = -np.expm1(-u)
    flow = t / (1.0 + 0.85 * t)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    diff = gap[:, 0] - gap[:, 1]
    c = np.concatenate((gap[:, None, :] * step,
                        x[:, None, :] * flow[None, :, None],
                        (slip[:, None] * flow)[:, :, None],
                        (diff[:, None] * step[:, 0])[:, :, None]), axis=2)
    r = np.column_stack((gap / tau, x, slip, diff / tau[0]))
    if not jac:
        return c, r
    dgap = -past @ (after * memory_weights - step_dt * np.exp(-after - step_dt))
    dstep = -u * np.exp(-u)
    dc = dgap[:, None, :] * step + gap[:, None, :] * dstep
    dr = (dgap - gap) / tau
    dcross = np.stack((dgap[:, 0, None] * step[:, 0] + diff[:, None] * dstep[:, 0],
                       -dgap[:, 1, None] * step[:, 0]), axis=2)
    drcross = np.column_stack((dgap[:, 0] - diff, -dgap[:, 1])) / tau[0]
    return c, r, dc, dr, dcross, drcross


def _fit(dt, t, past, hold, x, yc, yr):
    args = (dt, t, past, hold, x)
    yc = yc.ravel()
    target = np.r_[yc, yr]

    def linear(logs):
        c, r = _basis(logs, *args)
        a = np.vstack((c.reshape(-1, 9), r))
        return np.linalg.lstsq(a, target, rcond=None)[0]

    def residual(p, sc, sr):
        c, r = _basis(p[:2], *args)
        return np.r_[((c @ p[2:]).ravel() - yc) / sc, (r @ p[2:] - yr) / sr]

    def derivative(p, sc, sr):
        c, r, dc, dr, dcross, drcross = _basis(p[:2], *args, jac=True)
        jc = dc * p[2:4] + dcross * p[-1]
        jr = dr * p[2:4] + drcross * p[-1]
        return np.vstack((np.column_stack((jc.reshape(-1, 2), c.reshape(-1, 9))) / sc,
                          np.column_stack((jr, r)) / sr))

    low = np.log(min(dt.min(), t[t > 0].min()) / 20)
    high = np.log((dt.sum() + t.max()) * 20)
    bounds = (np.r_[[low, low], np.full(9, -np.inf)],
              np.r_[[high, high], np.full(9, np.inf)])
    opts = dict(jac=derivative, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=150)
    fits = []
    for pair in ((.26, 1.4), (.55, 3.2), (.9, 5.5), (1.5, .4)):
        logs = np.clip(np.log(pair), low + 1e-6, high - 1e-6)
        fits.append(least_squares(residual, np.r_[logs, linear(logs)], args=(.04, .05), **opts))
    best = min(fits, key=lambda fit: fit.cost)
    raw = residual(best.x, 1.0, 1.0)

    def scale(v):
        return max(1e-7, 2.5 * 1.4826 * np.median(np.abs(v - np.median(v))))

    refined = least_squares(residual, best.x, args=(scale(raw[:yc.size]), scale(raw[yc.size:])), **opts)
    return refined.x[:2], refined.x[2:]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    past, hold, x = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, w = _fit(dt, t, past[:n], hold[:n], x[:n], yc, train['instant_rate_obs'])
    c, r = _basis(logs, dt, t, past[n:], hold[n:], x[n:])
    return dict(estimated_taus=np.exp(logs), hereditary_weights=w[:7], regime_mode_weights=w[7:],
                predicted_strain=test['strain_0'][:, None] + c @ w,
                initial_rate_pred=r @ w)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`557 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0a9c8ef18117bd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0a9c8ef (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31905 ms` LLM)
- **Rationale**: Refit without each lab or stratum's response labels, preserving the specified unlabeled panel alignment. If one fit improves both held-out observables, prefer it; if sparse or EMI-affected calibration deviates materially from the clean fit, revise the fallback.
- **Alternatives Weighed**: `Submit after the shape check alone` (rejected: A model can return valid arrays while missing a held-out lab, stratum, or time point. Comparing robust and ordinary fits on those slices tests whether the robust objective is warranted.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for item in pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb')):
 d=item['dataset']; tr=d['train']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']; past,hold,x=(a[:n] for a in m._prepare(d)); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
 def ols(k, start):
  y=np.r_[yc[k].ravel(),yr[k]]
  def prof(logs, weights=False):
   c,r=m._basis(logs,dt,t,past[k],hold[k],x[k]); a=np.vstack((c.reshape(-1,9),r)); w=np.linalg.lstsq(a,y,rcond=None)[0]; return w if weights else a@w-y
  z=least_squares(prof,start,bounds=(np.log(.01),np.log(250))).x
  return z,prof(z,True)
 def errors(z,w,k):
  c,r=m._basis(z,dt,t,past[k],hold[k],x[k]); return c@w-yc[k],r@w-yr[k]
 allrows=np.ones(n,dtype=bool); wf=[]
 for group in ['all','lab_id','stratum']:
  for val in ([0] if group=='all' else np.unique(tr[group])):
   k=allrows if group=='all' else tr[group]!=val; h=allrows if group=='all' else ~k
   z,w=m._fit(dt,t,past[k],hold[k],x[k],yc[k],yr[k]); zz,ww=ols(k,z)
   for name,a,b in [('robust',z,w),('ols',zz,ww)]:
    ec,er=errors(a,b,h); print(group,val,name,'tau',np.exp(a).round(5),'rmse',np.sqrt(np.mean(ec**2)).round(6),np.sqrt(np.mean(er**2)).round(6),'max-time-rmse',np.sqrt(np.mean(ec**2,axis=0)).max().round(6),'p99',np.quantile(abs(ec),.99).round(5),np.quantile(abs(er),.99).round(5))
    if group=='all': print('weights',b.round(6),'max abs',abs(ec).max().round(5),abs(er).max().round(5))
    elif name=='robust': wf.append((np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(er**2)),np.sqrt(np.mean(ec**2,axis=0)).max()))
 print('worst robust',np.max(wf,axis=0))
 cal=d['calibration_load_cells']; ch=0; use=cal['lab_id']==ch; cx=cal['cmd_stress'][use]; cy=cal['true_stress'][use]; ref=m._calibrate(cx,cy,ch); sparse=m._calibrate(cx[:1],cy[:1],ch); noisy=cy.copy(); noisy[0]+=5; spike=m._calibrate(cx,noisy,ch)
 print('sparse and EMI calibration deviation',abs(ref-sparse),abs(ref-spike)); assert np.max(abs(ref-sparse))<.001 and np.max(abs(ref-spike))<.001
PY
```
- **Container Exit Code**: `0` (`813 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
all 0 robust tau [0.59268 3.8883 ] rmse 0.54515 0.048086 max-time-rmse 1.121853 p99 1.19382 0.21184
weights [ 0.674695  0.549918  0.137468  0.155611 -0.128723  0.115452 -0.087484
  0.317028 -0.213867] max abs 14.97047 0.25761
all 0 ols tau [0.58127 3.90513] rmse 0.543083 0.069569 max-time-rmse 1.111833 p99 1.15611 0.23984
weights [ 0.617174  0.519514  0.145993  0.125428 -0.135555  0.110562 -0.096136
  0.41552  -0.156574] max abs 14.82402 0.26756
lab_id 0 robust tau [0.59175 3.89829] rmse 0.874292 0.054426 max-time-rmse 2.531317 p99 2.6521 0.16261
lab_id 0 ols tau [0.54241 3.55034] rmse 0.874192 0.063147 max-time-rmse 2.528909 p99 2.65467 0.14748
lab_id 1 robust tau [0.59486 3.91646] rmse 0.41756 0.030746 max-time-rmse 0.959025 p99 1.20722 0.10155
lab_id 1 ols tau [0.59258 3.79658] rmse 0.418905 0.076129 max-time-rmse 0.960807 p99 1.17609 0.17907
lab_id 2 robust tau [0.59054 3.86059] rmse 0.547107 0.059039 max-time-rmse 1.486475 p99 0.87229 0.25435
lab_id 2 ols tau [0.61349 4.47771] rmse 0.550479 0.103055 max-time-rmse 1.486309 p99 0.86221 0.26334
lab_id 3 robust tau [0.59313 3.86945] rmse 0.171301 0.045734 max-time-rmse 0.295003 p99 0.93546 0.15576
lab_id 3 ols tau [0.58057 4.02788] rmse 0.183691 0.081968 max-time-rmse 0.295014 p99 1.00212 0.19194
stratum 0 robust tau [0.59175 3.86587] rmse 0.784238 0.044654 max-time-rmse 2.164115 p99 2.07197 0.16202
stratum 0 ols tau [0.56636 3.35148] rmse 0.783154 0.052456 max-time-rmse 2.159809 p99 2.10208 0.15372
stratum 1 robust tau [0.59078 3.85599] rmse 0.263331 0.044221 max-time-rmse 0.702928 p99 1.04192 0.19116
stratum 1 ols tau [0.57071 4.14738] rmse 0.269497 0.092766 max-time-rmse 0.705937 p99 1.06912 0.2635
stratum 2 robust tau [0.59539 3.95707] rmse 0.539153 0.054436 max-time-rmse 1.376745 p99 0.88287 0.25102
stratum 2 ols tau [0.6008  4.26125] rmse 0.542044 0.106258 max-time-rmse 1.375146 p99 0.89969 0.27223
worst robust [0.87429204 0.05903905 2.53131671]
sparse and EMI calibration deviation [2.00185402e-05 1.55722835e-05] [1.21156200e-05 8.73942832e-06]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0a9c8ef18117bd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0a9c8ef (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27258 ms` LLM)
- **Rationale**: Distinguish typical late-time error from a small number of extreme trajectories before changing the physical fit. Exercise the same documented EMI case with sparse in-batch sweeps: if the fitted calibration is pulled toward a spike, revise the fallback rather than changing the constitutive model.
- **Alternatives Weighed**: `Add a correction for high-error strain trajectories` (rejected: The raw RMSE alone cannot distinguish a systematic time-dependent mismatch from corrupted or imputed observations.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for item in pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb')):
 d=item['dataset']; tr=d['train']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']; a,h,x=(v[:n] for v in m._prepare(d)); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; z,w=m._fit(dt,t,a,h,x,y,r); c,q=m._basis(z,dt,t,a,h,x); e=c@w-y; f=q@w-r
 print('times',t.round(4)); print('strain absolute p50',np.quantile(abs(e),.50,axis=0).round(5)); print('strain absolute p90',np.quantile(abs(e),.90,axis=0).round(5)); print('rate absolute p50/p90',np.quantile(abs(f),[.50,.90]).round(5)); print('largest residual trajectories',np.sort(np.max(abs(e),axis=1))[-10:].round(4))
 complete=~np.isnan(tr['coupon_features_raw']).any(axis=1)
 for k in (complete,~complete):
  print('rows',k.sum(),'strain and rate absolute p50/p90',np.quantile(abs(e[k]),[.50,.90]).round(5),np.quantile(abs(f[k]),[.50,.90]).round(5))
 cal=d['calibration_load_cells']
 for npts in (3,4,10):
  for ch in range(4):
   k=cal['lab_id']==ch; cx=cal['cmd_stress'][k]; cy=cal['true_stress'][k]; ref=m._calibrate(cx,cy,ch); worst=0.
   for i in range(npts):
    for delta in (-5.,5.):
     noisy=cy[:npts].copy(); noisy[i]+=delta; est=m._calibrate(cx[:npts],noisy,ch); worst=max(worst,np.max(abs(est-ref)))
   print('EMI points, lab, worst coefficient deviation',npts,ch,round(worst,6))
PY
```
- **Container Exit Code**: `0` (`571 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
strain absolute p50 [0.01278 0.01102 0.01266 0.01282 0.01422 0.0157  0.01501 0.01258 0.01432
 0.01532]
strain absolute p90 [0.04871 0.04844 0.04414 0.05535 0.07606 0.18014 0.05571 0.06231 0.08117
 0.06751]
rate absolute p50/p90 [0.01444 0.06026]
largest residual trajectories [ 2.8968  2.9278  3.1355  3.5423  3.5681  5.1867  5.267   5.6471 11.2046
 14.9705]
rows 157 strain and rate absolute p50/p90 [0.01186 0.03244] [0.01236 0.03135]
rows 23 strain and rate absolute p50/p90 [0.05307 0.2299 ] [0.0806  0.19909]
EMI points, lab, worst coefficient deviation 3 0 0.000198
EMI points, lab, worst coefficient deviation 3 1 1.95605
EMI points, lab, worst coefficient deviation 3 2 3.50203
EMI points, lab, worst coefficient deviation 3 3 0.000381
EMI points, lab, worst coefficient deviation 4 0 7.9e-05
EMI points, lab, worst coefficient deviation 4 1 3.5e-05
EMI points, lab, worst coefficient deviation 4 2 9.8e-05
EMI points, lab, worst coefficient deviation 4 3 0.000276
EMI points, lab, worst coefficient deviation 10 0 3.8e-05
EMI points, lab, worst coefficient deviation 10 1 1.8e-05
EMI points, lab, worst coefficient deviation 10 2 2e-05
EMI points, lab, worst coefficient deviation 10 3 2.4e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0a9c8ef18117bd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0a9c8ef (directory).
```

### Turn 8 — `execute_in_remote_docker` (`46493 ms` LLM)
- **Rationale**: The three-point EMI test exposed an ambiguity in calibration, not a failure of the constitutive law. Test both spike signs just above the specified threshold and at larger amplitudes, across subsets of the measured sweeps. If the exact candidate fails those tests or regresses on held-out strain or rate, revise the calibration before submitting.
- **Alternatives Weighed**: `Always use archived calibration for three-point sweeps` (rejected: A clean three-point sweep identifies the line directly. Archive data is needed when competing lines make spike assignment ambiguous.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
a, b = s.index('def _calibrate('), s.index('\ndef _prepare(')
s = s[:a] + '''def _calibrate(cmd, true, channel):
    if np.unique(cmd).size < 2:
        return _prior_calibration(channel)
    direct = np.polyfit(cmd, true, 1)
    # Clean public sweeps are affine within 0.00022 stress units. A loose
    # consensus tolerance separates clean points from plausible EMI without
    # assuming that the least-squares line is unbiased in a sparse sweep.
    consensus_tol = 0.01
    if np.max(np.abs(true - np.polyval(direct, cmd))) <= consensus_tol:
        return direct
    i, j = np.triu_indices(len(cmd), 1)
    distinct = cmd[i] != cmd[j]
    i, j = i[distinct], j[distinct]
    slopes = (true[j] - true[i]) / (cmd[j] - cmd[i])
    offsets = true[i] - cmd[i] * slopes
    errors = np.abs(true[:, None] - cmd[:, None] * slopes - offsets)
    support = np.sum(errors <= consensus_tol, axis=0)
    best = np.argmax(support)
    clean = errors[:, best] <= consensus_tol
    if clean.sum() >= 4:
        return np.polyfit(cmd[clean], true[clean], 1)
    # A noisy three-point sweep cannot distinguish competing two-point lines.
    prior = _prior_calibration(channel)
    clean = np.abs(true - np.polyval(prior, cmd)) <= 3.0
    if clean.all():
        return direct
    if np.unique(cmd[clean]).size >= 2:
        return np.polyfit(cmd[clean], true[clean], 1)
    return prior

''' + s[b:]
p.write_text(s)
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for item in pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb')):
 d=item['dataset']; cal=d['calibration_load_cells']; worst={}
 for ch in range(4):
  k=cal['lab_id']==ch; x=cal['cmd_stress'][k]; y=cal['true_stress'][k]; ref=np.polyfit(x,y,1)
  for n in (0,1,3,4,10):
   if n<2:
    worst[n]=max(worst.get(n,0),np.max(abs(m._calibrate(x[:n],y[:n],ch)-ref))); continue
   for subset in itertools.combinations(range(len(x)),n):
    ii=np.array(subset); a,b=x[ii],y[ii];
    assert np.allclose(m._calibrate(a,b,ch),np.polyfit(a,b,1))
    for j in range(n):
     clean=np.arange(n)!=j; expected=np.polyfit(a[clean],b[clean],1)
     for delta in (-5.,-3.01,3.01,5.):
      z=b.copy(); z[j]+=delta
      worst[n]=max(worst.get(n,0),np.max(abs(m._calibrate(a,z,ch)-expected)))
 print('maximum calibration coefficient error by sweep size',worst)
 assert max(worst.values()) < 0.001
 tr=d['train']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']; past,hold,x=(a[:n] for a in m._prepare(d)); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; scores=[]
 for group in ('lab_id','stratum'):
  for level in np.unique(tr[group]):
   k=tr[group]!=level; z,w=m._fit(dt,t,past[k],hold[k],x[k],yc[k],yr[k]); c,r=m._basis(z,dt,t,past[~k],hold[~k],x[~k]); ec=c@w-yc[~k]; er=r@w-yr[~k]; scores.append([np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(er**2)),np.quantile(abs(ec),.5),np.quantile(abs(er),.5)])
 worst_cv=np.max(scores,axis=0); print('worst held-out strain/rate RMSE and median absolute errors',worst_cv)
 assert np.all(worst_cv[:2] < [0.875,0.060])
PY
```
- **Container Exit Code**: `1` (`6985 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
maximum calibration coefficient error by sweep size {0: 4.0460389440277744e-05, 1: 4.0460389440277744e-05, 3: 4.629901590997216, 4: 0, 10: 0}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e4c31ccc3602a6bb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_e4c31ccc (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 21, in <module>
AssertionError
```

### Turn 9 — `execute_in_remote_docker` (`58982 ms` LLM)
- **Rationale**: The three-point failure survived the residual-only shortcut, so test whether a broad, campaign-specific plausibility check catches it without altering clean fits. If a spike still passes, the test prints the exact channel and subset for diagnosis. The isolated run also checks that the edit preserves held-out accuracy and that both predicted observables match the specified recurrence and initial derivative.
- **Alternatives Weighed**: `Use archived calibration for every sparse sweep` (rejected: Two or three clean points with distinct commands can determine the current calibration; test that the solver still uses those points directly.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
old = '''    if np.max(np.abs(true - np.polyval(direct, cmd))) <= consensus_tol:
        return direct
'''
new = '''    # In a sparse sweep, two nearby commands can let an EMI spike at a
    # third command hide behind a nearly exact but implausible affine fit.
    # The compatible campaign sweeps have gains 0.77–1.34 and |bias| < 0.18.
    # Use a generous plausibility range only to flag sparse fits, never to
    # clip or replace a calibration supported by four clean observations.
    plausible_sparse_fit = 0.3 < direct[0] < 2.0 and abs(direct[1]) < 0.6
    if (len(cmd) >= 4 or plausible_sparse_fit) and np.max(np.abs(true - np.polyval(direct, cmd))) <= consensus_tol:
        return direct
'''
assert old in s
p.write_text(s.replace(old, new, 1))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for item in pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb')):
 d=item['dataset']; cal=d['calibration_load_cells']; worst={}; witness={}
 for ch in range(4):
  k=cal['lab_id']==ch; x=cal['cmd_stress'][k]; y=cal['true_stress'][k]; ref=np.polyfit(x,y,1); prior=m._prior_calibration(ch)
  for n in (0,1,2,3,4,10):
   if n<2:
    worst[n]=max(worst.get(n,0),np.max(abs(m._calibrate(x[:n],y[:n],ch)-ref))); continue
   for subset in itertools.combinations(range(len(x)),n):
    ii=np.array(subset); a,b=x[ii],y[ii]
    assert np.allclose(m._calibrate(a,b,ch),np.polyfit(a,b,1))
    for j in range(n):
     clean=np.arange(n)!=j; expected=prior if n==2 else np.polyfit(a[clean],b[clean],1)
     for delta in (-5.,-3.01,3.01,5.):
      obs=b.copy(); obs[j]+=delta
      if n==2:
       line=np.polyfit(a,obs,1)
       if 0.3 < line[0] < 2 and abs(line[1]) < .6: continue
      err=np.max(abs(m._calibrate(a,obs,ch)-expected))
      if err > worst.get(n,-1): worst[n]=err; witness[n]=(ch,subset,j,delta)
 print('maximum calibration coefficient error by sweep size (two points: implausible fits only)',worst)
 if max(worst.values()) >= .001: print('failing calibration witnesses',witness)
 assert max(worst.values()) < .001
 tr=d['train']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']; past,hold,x=(a[:n] for a in m._prepare(d)); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; scores=[]
 for group in ('lab_id','stratum'):
  for level in np.unique(tr[group]):
   k=tr[group]!=level; z,w=m._fit(dt,t,past[k],hold[k],x[k],yc[k],yr[k]); c,r=m._basis(z,dt,t,past[~k],hold[~k],x[~k]); ec=c@w-yc[~k]; er=r@w-yr[~k]; scores.append([np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(er**2)),np.median(abs(ec)),np.median(abs(er))])
 print('worst held-out strain/rate RMSE and median absolute errors',np.max(scores,axis=0))
 assert np.all(np.max(scores,axis=0)[:2] < [.875,.060])
 tau=np.exp(z); memory=np.zeros((n,2))
 for v,step in zip(past.T,dt):
  f=np.exp(-step/tau); memory=f*memory+(1-f)*v[:,None]
 g=hold[:,None]-memory; flow=x@w[2:7]+w[7]*np.sign(hold)*np.sqrt(abs(hold)); cross=w[8]*(g[:,0]-g[:,1]);
 curve=(g*w[:2])@(-np.expm1(-t[:,None]/tau)).T+flow[:,None]*(t/(1+.85*t))+cross[:,None]*(-np.expm1(-t/tau[0])); rate=(g*w[:2]/tau).sum(axis=1)+flow+cross/tau[0]
 c,r=m._basis(z,dt,t,past,hold,x); errors=(np.max(abs(c@w-curve)),np.max(abs(r@w-rate))); print('direct recurrence and analytic initial-rate discrepancies',errors); assert max(errors)<1e-12
PY
```
- **Container Exit Code**: `0` (`7601 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
maximum calibration coefficient error by sweep size (two points: implausible fits only) {0: 4.0460389440277744e-05, 1: 4.0460389440277744e-05, 2: 0.0, 3: 0.0, 4: 0.0, 10: 0.0}
worst held-out strain/rate RMSE and median absolute errors [0.87429204 0.05903905 0.01630093 0.01729783]
direct recurrence and analytic initial-rate discrepancies (8.881784197001252e-16, 1.3322676295501878e-15)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8a7f5236ebd907ed (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_8a7f5236 (directory).
```

### Turn 10 — `submit_completed_task` (`15296 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema validation and sparse and EMI calibration stress tests. Its strain and rate calculations matched an independent implementation of the recurrence and analytic derivative to within 1.4e-15.
- **Alternatives Weighed**: `Replace the robust objective with ordinary least squares` (rejected: Ordinary least squares raised worst held-out initial-rate RMSE from 0.05904 to 0.10626 without materially improving worst held-out strain RMSE.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_03_solver.py': 'FILE:7275'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed. Across held-out labs and strata, worst RMSE was 0.8743 for strain and 0.05904 for initial rate; worst median absolute errors were 0.01630 and 0.01730, respectively.`

### Turn 11 — `execute_in_remote_docker` (`34267 ms` LLM)
- **Rationale**: The previous calibration stress test excluded two-point spikes that produced plausible coefficients. Test those cases using measured command pairs, and quantify historical calibration agreement before deciding whether to change the fallback. Check signed residuals on complete-feature rows by held-out lab to distinguish systematic model bias from isolated trajectory errors.
- **Alternatives Weighed**: `Add generic drift operators, relay modes, or balanced-accuracy checks` (rejected: This is a continuous strain-and-rate problem with a specified rational flow term and cross-gap difference, not a relay, diffusion, or classification task. Held-out lab and stratum residuals test the relevant nuisance structure.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/rheology_margin_audit.py <<'PY'
import importlib.util
import itertools
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.loads((root / 'public_alpha_m97_c01_batch_03.pkl').read_bytes()):
    d = item['dataset']
    assert d['shear_banding_index'] > .50
    tr = d['train']
    n = len(tr['lab_id'])
    a, h, x = (v[:n] for v in m._prepare(d))
    dt, t = d['past_dt'], d['eval_times']
    yc = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    yr = tr['instant_rate_obs']
    for lab in np.unique(tr['lab_id']):
        k = tr['lab_id'] != lab
        z, w = m._fit(dt, t, a[k], h[k], x[k], yc[k], yr[k])
        c, r = m._basis(z, dt, t, a[~k], h[~k], x[~k])
        ec, er = c @ w - yc[~k], r @ w - yr[~k]
        complete = np.isfinite(tr['coupon_features_raw'][~k]).all(axis=1)
        print('held-out lab', lab, 'complete-feature signed median strain by time', np.round(np.median(ec[complete], axis=0), 4), 'signed median rate', round(np.median(er[complete]), 4))
        for level in np.unique(tr['stratum'][~k]):
            j = tr['stratum'][~k] == level
            print('  stratum', level, 'rows', j.sum(), 'median absolute strain/rate', np.round([np.median(abs(ec[j])), np.median(abs(er[j]))], 5))
    cal = d['calibration_load_cells']
    for lab in range(4):
        k = cal['lab_id'] == lab
        a, b = cal['cmd_stress'][k], cal['true_stress'][k]
        prior = m._prior_calibration(lab)
        drift = np.max(abs(b - np.polyval(prior, a)))
        worst = (0., None)
        missed = 0
        for pair in itertools.combinations(range(len(a)), 2):
            x, y = a[list(pair)], b[list(pair)]
            for j, delta in itertools.product(range(2), (-5., -3.01, 3.01, 5.)):
                obs = y.copy()
                obs[j] += delta
                coef = m._calibrate(x, obs, lab)
                err = np.max(abs(coef - prior))
                missed += err > .001
                if err > worst[0]:
                    worst = (err, (pair, j, delta, coef.round(6).tolist()))
        print('lab', lab, 'clean archive prediction max residual', round(drift, 6), 'missed two-point spikes', missed, 'worst', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_margin_audit.py
```
- **Container Exit Code**: `0` (`1315 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held-out lab 0 complete-feature signed median strain by time [-0.0033 -0.0002  0.0076  0.0028 -0.0016 -0.011  -0.0063  0.0005  0.0052
 -0.004 ] signed median rate -0.0014
  stratum 0 rows 12 median absolute strain/rate [0.01689 0.01436]
  stratum 1 rows 13 median absolute strain/rate [0.01232 0.00895]
  stratum 2 rows 10 median absolute strain/rate [0.01729 0.01038]
held-out lab 1 complete-feature signed median strain by time [ 0.0024 -0.0029  0.0002  0.0039  0.0013 -0.0018 -0.0057 -0.0024 -0.0061
 -0.0029] signed median rate -0.0034
  stratum 0 rows 12 median absolute strain/rate [0.01717 0.0227 ]
  stratum 1 rows 25 median absolute strain/rate [0.01092 0.0151 ]
  stratum 2 rows 12 median absolute strain/rate [0.01338 0.00977]
held-out lab 2 complete-feature signed median strain by time [ 0.0063 -0.0019  0.0012  0.0032  0.0008  0.0022 -0.0004 -0.0012 -0.0037
  0.0005] signed median rate 0.0016
  stratum 0 rows 16 median absolute strain/rate [0.01229 0.01058]
  stratum 1 rows 15 median absolute strain/rate [0.01216 0.01631]
  stratum 2 rows 26 median absolute strain/rate [0.01332 0.01744]
held-out lab 3 complete-feature signed median strain by time [-0.0011 -0.0035  0.0002  0.001  -0.0042  0.0056  0.0009  0.0033  0.0044
  0.0047] signed median rate -0.0034
  stratum 0 rows 8 median absolute strain/rate [0.0178  0.03184]
  stratum 1 rows 12 median absolute strain/rate [0.01511 0.01077]
  stratum 2 rows 19 median absolute strain/rate [0.01548 0.01684]
lab 0 clean archive prediction max residual 0.000239 missed two-point spikes 0 worst (0.0, None)
lab 1 clean archive prediction max residual 0.000178 missed two-point spikes 5 worst (1.2011568582252288, ((6, 9), 1, 3.01, [1.971163, 0.340932]))
lab 2 clean archive prediction max residual 0.000208 missed two-point spikes 0 worst (0.0, None)
lab 3 clean archive prediction max residual 0.00019 missed two-point spikes 0 worst (0.0, None)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8a7f5236ebd907ed (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_8a7f5236 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`34824 ms` LLM)
- **Rationale**: The audit showed that plausible gain and bias cannot distinguish a clean two-point line from one containing a spike. Test the corrected branch on every measured pair and both spike signs, and compare all five public outputs with the verified prior artifact to detect unintended changes to the clean-data path.
- **Alternatives Weighed**: `Always replace a two-point fit with historical calibration` (rejected: Two clean points identify the in-batch affine fit. Use the prefix sweep only as the independent reference for detecting a spike and as the fallback when a spike leaves fewer than two usable points.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
old = '''    direct = np.polyfit(cmd, true, 1)
    # Clean public sweeps'''
new = '''    direct = np.polyfit(cmd, true, 1)
    if len(cmd) == 2:
        # Two points give zero affine residual even if one is an EMI spike.
        # Compare against the compatible prefix sweep to detect the spike,
        # but retain the in-batch fit when neither point exceeds the threshold.
        prior = _prior_calibration(channel)
        if np.any(np.abs(true - np.polyval(prior, cmd)) > 3.0):
            return prior
        return direct
    # Clean public sweeps'''
assert old in s
p.write_text(s.replace(old, new, 1))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import itertools
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
name = 'alpha_m97_c01_batch_03_solver.py'
def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
m = load(root / name)
previous = list(Path('/tmp/.hegel_candidate_archive/v03_8a7f5236').rglob(name))
assert len(previous) == 1, previous
old = load(previous[0])
count = 0
for item in pickle.loads((root / 'public_alpha_m97_c01_batch_03.pkl').read_bytes()):
    d = item['dataset']
    cal = d['calibration_load_cells']
    for ch in range(4):
        rows = cal['lab_id'] == ch
        x, y = cal['cmd_stress'][rows], cal['true_stress'][rows]
        prior = m._prior_calibration(ch)
        for i, j in itertools.combinations(range(len(x)), 2):
            a, b = x[[i,j]], y[[i,j]]
            np.testing.assert_array_equal(m._calibrate(a,b,ch), np.polyfit(a,b,1))
            for k, delta in itertools.product(range(2), (-5., -3.01, 3.01, 5.)):
                noisy = b.copy()
                noisy[k] += delta
                np.testing.assert_array_equal(m._calibrate(a,noisy,ch), prior)
                count += 1
    out = m.solve_alpha_m97_c01_batch_03(d)
    ref = old.solve_alpha_m97_c01_batch_03(d)
    for key in ref:
        np.testing.assert_array_equal(out[key], ref[key], err_msg=key)
print('Two-point EMI trials passed:', count, '; clean two-point fits retained; all five public outputs unchanged.')
PY
```
- **Container Exit Code**: `0` (`1300 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Two-point EMI trials passed: 1440 ; clean two-point fits retained; all five public outputs unchanged.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=69f7f1f44d6c049b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_69f7f1f4 (directory).
```

### Turn 13 — `submit_completed_task` (`16434 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema validation. The calibration fix passed the expanded EMI tests and left all five public outputs unchanged from the candidate already verified against held-out labs, held-out strata, and the independent recurrence.
- **Alternatives Weighed**: `Keep the two-point coefficient plausibility heuristic` (rejected: It missed five injected spikes on measured command pairs. The prefix-sweep comparison detected those spikes without replacing clean two-point fits.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema and all 1,440 two-point EMI stress tests passed. Worst held-out RMSE across labs and strata was 0.8743 for strain and 0.05904 for initial rate; worst median absolute errors were 0.01630 and 0.01730, respectively.`

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
