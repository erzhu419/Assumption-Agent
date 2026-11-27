# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `313,551` (`302,458` / `83,328` / `11,093`) |
| **Total Episode Wall Time (s)** | `439.42s` |

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

### Turn 1 — `execute_in_remote_docker` (`19383 ms` LLM)
- **Rationale**: Inspect the public schema and available measurements before choosing the fit and validation slices. If a channel lacks usable calibration, inspect the permitted prefix history for that channel.
- **Alternatives Weighed**: `Reuse the prior batch solver unchanged` (rejected: The public gate reports that the prior binding failed; the current batch specifies a distinct jammed-slip response.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle
from pathlib import Path
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))
def outline(d, prefix=''):
 for k,v in d.items():
  if isinstance(v,dict):
   print(prefix,k,'dict'); outline(v,prefix+'  ')
  elif hasattr(v,'shape'): print(prefix,k,v.shape,v.dtype,repr(v.ravel()[:5]))
  elif isinstance(v,(list,tuple)): print(prefix,k,type(v).__name__,len(v),repr(v[:1])[:450])
  else: print(prefix,k,repr(v)[:350])
outline(p)
print('\nArchive:', [x.name for x in Path('/tmp/hegel_continual_archive').glob('*')])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation' 2>&1 | head -45
```
- **Container Exit Code**: `0` (`1064 ms`, timed_out=`False`)
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
  File "<stdin>", line 11, in <module>
  File "<stdin>", line 5, in outline
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`20757 ms` LLM)
- **Rationale**: Inspect one public item and the documented primitives to determine the observation layout and whether any channel qualifies for calibration fallback. Use the prior solver only to clarify shared input conventions, not to select the current regime.
- **Alternatives Weighed**: `Implement against inferred field names` (rejected: The first inspection stopped at the list wrapper; the nested training and calibration fields have not been read.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
from pathlib import Path
p=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))
def outline(d, prefix=''):
 for k,v in d.items():
  if isinstance(v,dict): print(prefix,k,'dict'); outline(v,prefix+'  ')
  elif hasattr(v,'shape'): print(prefix,k,v.shape,v.dtype,repr(v.ravel()[:6]))
  elif isinstance(v,(list,tuple)): print(prefix,k,type(v).__name__,len(v),repr(v[:1])[:450])
  else: print(prefix,k,repr(v)[:350])
print('public items:',len(p))
outline(p[0])
for a in p: print('item context:', {k:v for k,v in a.items() if k!='dataset' and isinstance(v,(str,int,float))})
print('archive files:', [x.name for x in Path('/tmp/hegel_continual_archive').glob('*')])
PY
cat public_asset_index.json
sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
for f in align_reference_panel balanced_cell_weights weighted_column_fill; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.$f; done
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation
```
- **Container Exit Code**: `0` (`523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "fit_anchored_relaxation",
          "signature": "fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None]",
          "doc_summary": "Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing nonnegative times, a predictor independent of the intercept, nonnegative rate and curvature ratio, positive noise scale, 2 <= bins <= n, and ordered positive tau bounds, returning (offset, positive slope,"
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_trend_decomposition",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`17357 ms` LLM)
- **Rationale**: The previous output was truncated before the shared preprocessing code. Read that code to resolve the cell-balanced imputation convention, and check current calibration residuals to decide whether any archive fallback is needed.
- **Alternatives Weighed**: `Use the archive calibration for all channels` (rejected: Use direct fits unless the public rows show a channel meets the stated sparse-data or EMI fallback criterion.)

```bash
sed -n '1,170p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']; c=x['calibration_load_cells']
for i in range(4):
 m=c['lab_id']==i; a=c['cmd_stress'][m]; b=c['true_stress'][m]; p=np.polyfit(a,b,1); print('calibration',i,len(a),p,'max fit residual',max(abs(b-np.polyval(p,a))))
for name in ('train','test'):
 d=x[name]; print(name,'missing coupons by column',np.isnan(d['coupon_features_raw']).sum(axis=0),'missing ref count',np.isnan(d['ref_coupons']).sum())
print('scipy',__import__('scipy').__version__)
PY
```
- **Container Exit Code**: `0` (`1002 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, np.column_stack((gap/taus, features))


def _mad(x):
    return max(1e-6, 1.4826*np.median(np.abs(x-np.median(x))))


def _fit(past, hold, features, dt, t, y, rates):
    ok = np.isfinite(y)
    rok = np.isfinite(rates)
    target = np.r_[y[ok], rates[rok]]
    count = ok.sum()
    def matrix(logtau):
        a, b = _design(np.exp(logtau), past, hold, features, dt, t)
        return np.concatenate((a[ok], b[rok]), axis=0)
    def noise(res):
        s1 = _mad(res[:count]); s2 = _mad(res[count:]) if rok.any() else s1
        return np.r_[np.full(count, s1), np.full(rok.sum(), s2)]
    tmin = min(t[t > 0].min(), dt[dt > 0].min())/20
    tmax = max(t.max(), dt.sum())*20
    bounds = (np.r_[np.repeat(np.log(tmin), 2), np.full(7, -np.inf)],
              np.r_[np.repeat(np.log(tmax), 2), np.full(7, np.inf)])
    def optimize(start, scale, loss):
        def residual(p):
            return (matrix(p[:2]) @ p[2:] - target)/scale
        return least_squares(residual, start, bounds=bounds, loss=loss, x_scale='jac',
                             max_nfev=200)
    seeds = t.max()*np.array([[.05,.28], [.08,.65], [.13,1.6], [.24,.65], [.025,2.]])
    starts = []
    for tau in seeds:
        logtau = np.log(tau)
        mat = matrix(logtau)
        w = np.linalg.lstsq(mat, target, rcond=None)[0]
        starts.append(np.r_[logtau, w])
    scale = noise(matrix(starts[0][:2]) @ starts[0][2:] - target)
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda fit: fit.cost)
    scale = 2.0*noise(pilot.fun*scale)
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda fit: fit.cost)
    scale = 2.0*noise(fit.fun*scale)
    p = optimize(fit.x, scale, 'cauchy').x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
    past, hold, features, baseline = _prepare(dataset)
    t = np.asarray(dataset['eval_times'])
    dt = np.asarray(dataset['past_dt'])
    train = dataset['train']
calibration 0 10 [1.28996286 0.11997817] max fit residual 0.00017199813083901283
calibration 1 10 [ 0.8199772  -0.14996294] max fit residual 0.0002451248283934504
calibration 2 10 [1.16999799 0.14000869] max fit residual 0.00017239041523975374
calibration 3 10 [ 0.76000329 -0.11003612] max fit residual 0.00015032266651648918
train missing coupons by column [6 4 8 4] missing ref count 239
test missing coupons by column [3 3 2 0] missing ref count 101
scipy 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`17573 ms` LLM)
- **Rationale**: Recover the exact panel-imputation convention and archive record format. The former affects prediction on public rows with missing features; the latter determines how the solver handles the specified sparse or EMI-affected channels.
- **Alternatives Weighed**: `Use archived calibration on the public batch` (rejected: Each channel has ten clean in-batch points, so none qualifies for fallback.)

```bash
sed -n '1,116p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; python3 - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/archive_calibration_history.json'); x=json.loads(p.read_text()); print('calibration archive format:',type(x).__name__,repr(x)[:1400])
PY
```
- **Container Exit Code**: `0` (`330 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - center)/scale
    cells = np.column_stack((lab, stratum))
    means = []
    for cell in np.unique(cells, axis=0):
        values = aligned[np.all(cells == cell, axis=1)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
    baseline = joined('strain_0')
    return past, hold, np.column_stack((np.ones(len(lab)), aligned)), baseline


def _design(taus, past, hold, features, dt, t):
    # Each history sample contributes its exact final-time memory weight.
calibration archive format: list [{'episode_short_name': 'alpha_m97_c03_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.893607, -0.483769, -1.39264, 1.611445, -1.810815, -1.870654, 1.48082, 1.268821, -0.114529, 0.252739, -1.204476, -0.664616, -0.154159, -0.613549, -1.259125, -2.011382, 2.18392, -0.23172, 1.98347, -2.000907, -0.486537, -0.139492, -0.631342, -1.874772, -0.708257, 2.13352, 0.386634, 1.10423, -1.407517, 1.70372, 1.521295, 1.987385, 1.659545, -0.294044, 0.359165, -0.017198, 1.319848, 0.959274, 1.866895, -0.480321], 'true_stress': [0.893713, -0.483774, -1.392656, 1.611564, -1.810667, -1.870622, 1.480773, 1.268837, -0.114695, 0.252685, -1.204516, -0.664704, -0.154113, -0.613571, -1.259248, -2.011128, 2.183777, -0.231673, 1.983509, -2.001075, -0.486729, -0.139381, -0.631253, -1.874747, -0.708398, 2.133538, 0.386675, 1.104402, -1.407596, 1.703866, 1.521337, 1.987399, 1.659449, -0.294029, 0.359147, -0.017404, 1.319703, 0.959321, 1.866928, -0.480472]}, {'episode_short_name': 'alpha_m97_c03_batch_01', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [2.15143, 1.121894, -0.424697, -1.698186, 1.597457, -0.341977, 0.304259, 1.741536, -2.045051, 1.452617, 1.72849, -1.533376, -0.721054, -1.917953, -0.632766, 1
```

### Turn 5 — `execute_in_remote_docker` (`66916 ms` LLM)
- **Rationale**: Implement the specified response and its analytic initial derivative, with separate robust residual scales for strain and rate. Run the isolated schema validator now so any import, shape, or nonfinite-output failure is fixed before grouped prediction validation.
- **Alternatives Weighed**: `Sort the fitted time constants after optimization` (rejected: The slip contrast relaxes on branch 0, so swapping the branches changes the fitted response.)

```bash
cat > /holdout_task/alpha_m97_c03_batch_03_solver.py <<'PY'
"""Estimate two-branch polymer memory and jammed slip from calibrated measurements."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _calibration_rows(sweep, channel):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    keep = (np.asarray(sweep['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _clean_sweep(x, y):
    """Find a clean affine sweep if ordinary least squares flags an anomaly."""
    fit = np.polyfit(x, y, 1)
    if np.max(np.abs(y - np.polyval(fit, x))) <= .01 or len(x) < 3:
        return fit, np.ones(len(x), dtype=bool), False
    candidates = []
    for i, j in zip(*np.triu_indices(len(x), k=1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        residual = np.abs((y - y[i]) - gain * (x - x[i]))
        keep = residual <= .01
        if keep.sum() >= 2 and np.any(residual > 3.):
            candidates.append(keep)
    if not candidates:
        return fit, np.ones(len(x), dtype=bool), False
    count = max(s.sum() for s in candidates)
    best = np.unique([s for s in candidates if s.sum() == count], axis=0)
    if len(best) != 1:
        return fit, np.ones(len(x), dtype=bool), True
    keep = best[0]
    return np.polyfit(x[keep], y[keep], 1), keep, True


def _archive_fit(channel):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        x, y = _calibration_rows(sweep, channel)
        if len(x) >= 2 and np.ptp(x) > 0:
            fit, keep, emi = _clean_sweep(x, y)
            if not emi or (not keep.all() and keep.sum() >= 4):
                return fit
    raise ValueError(f'No usable archived load-cell calibration for channel {channel}')


def _calibrate(sweep):
    fits = []
    for channel in range(4):
        x, y = _calibration_rows(sweep, channel)
        if len(x) < 2:
            fit = _archive_fit(channel)
        else:
            fit, keep, emi = _clean_sweep(x, y)
            if emi and (keep.all() or keep.sum() < 4):
                fit = _archive_fit(channel)
                keep = np.abs(y - np.polyval(fit, x)) <= 3.
                if keep.sum() >= 4:
                    fit = np.polyfit(x[keep], y[keep], 1)
        fits.append(fit)
    return np.asarray(fits)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def joined(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = joined('lab_id'), joined('stratum')
    ref, aligned = joined('ref_coupons'), joined('coupon_features_raw').astype(float)
    for channel in np.unique(lab):
        rows = lab == channel
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - center) / scale
    cells = np.column_stack((lab, stratum))
    means = []
    for cell in np.unique(cells, axis=0):
        values = aligned[np.all(cells == cell, axis=1)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
    drift = np.column_stack((np.ones(len(lab)), aligned))
    return past, hold, drift, joined('strain_0')


def _design(taus, past, hold, drift, dt, times):
    # Closed form of the zero-initialized history recurrence.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (np.exp(-later[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gaps = hold[:, None] - memory
    relax = -np.expm1(-times[:, None] / taus)
    flow = times / (1. + .85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    contrast = gaps[:, 0] - gaps[:, 1]
    curves = np.concatenate((gaps[:, None, :] * relax[None, :, :],
                             drift[:, None, :] * flow[None, :, None],
                             slip[:, None, None] * flow[None, :, None],
                             contrast[:, None, None] * relax[None, :, :1]), axis=2)
    rates = np.column_stack((gaps / taus, drift, slip, contrast / taus[0]))
    return curves, rates


def _fit(past, hold, drift, dt, times, curves, rates):
    ok = np.isfinite(curves)
    rate_ok = np.isfinite(rates)
    target = np.r_[curves[ok], rates[rate_ok]]
    n_curve = ok.sum()

    def matrix(logtau):
        a, b = _design(np.exp(logtau), past, hold, drift, dt, times)
        return np.vstack((a[ok], b[rate_ok]))

    def scales(residual):
        def mad(values):
            return max(1e-6, 1.4826 * np.median(np.abs(values - np.median(values))))
        curve = mad(residual[:n_curve])
        rate = mad(residual[n_curve:]) if rate_ok.any() else curve
        return np.r_[np.full(n_curve, curve), np.full(rate_ok.sum(), rate)]

    lower = np.log(min(times[times > 0].min(), dt[dt > 0].min()) / 20.)
    upper = np.log(20. * max(times.max(), dt.sum()))
    bounds = (np.r_[lower, lower, np.full(9, -np.inf)],
              np.r_[upper, upper, np.full(9, np.inf)])
    seeds = times.max() * np.array([[.05, .28], [.08, .65], [.13, 1.6], [.24, .65], [.025, 2.]])
    starts = []
    # The slip relaxation is attached to branch 0: never sort a fitted pair.
    for tau in np.vstack((seeds, seeds[:, ::-1])):
        logtau = np.clip(np.log(tau), lower, upper)
        weights = np.linalg.lstsq(matrix(logtau), target, rcond=None)[0]
        starts.append(np.r_[logtau, weights])

    def residual(p):
        return matrix(p[:2]) @ p[2:] - target

    def optimize(p, scale, loss):
        return least_squares(lambda x: residual(x) / scale, p, bounds=bounds,
                             loss=loss, x_scale='jac', max_nfev=150)

    scale = scales(residual(min(starts, key=lambda p: np.linalg.norm(residual(p)))))
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda fit: fit.cost)
    scale = 2. * scales(residual(pilot.x))
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda f: f.cost)
    fit = optimize(fit.x, 2. * scales(residual(fit.x)), 'cauchy')
    return np.exp(fit.x[:2]), fit.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    past, hold, drift, baseline = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    train = dataset['train']
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], drift[:n], dt, times,
                         np.asarray(train['strain_curves_obs']) - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']))
    curves, rates = _design(taus, past[n:], hold[n:], drift[n:], dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': baseline[n:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`2441 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c0c165a6af85e1b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c0c165a6 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`44971 ms` LLM)
- **Rationale**: Refit within each lab and stratum holdout to measure transfer; check missing-coupon and stress-sign slices for concentrated bias. Compare predictions against an independent memory recurrence and derivative, and exercise the documented sparse and EMI calibration paths. Failures would identify whether to change the fit, alignment, response implementation, or fallback.
- **Alternatives Weighed**: `Submit after schema validation alone` (rejected: Shape and finiteness checks cannot detect a wrong derivative, unstable branch estimates, or errors concentrated in one lab, stratum, or horizon.)

```bash
cat > /tmp/verify_batch03.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb'))[0]['dataset']
train = d['train']
n = len(train['lab_id'])
lab = np.asarray(train['lab_id'])
stratum = np.asarray(train['stratum'])
missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)
y, rate = train['strain_curves_obs'], train['instant_rate_obs']
past, hold, drift, baseline = m._prepare(d)
dt, times = d['past_dt'], d['eval_times']
full = m.solve_alpha_m97_c03_batch_03(d)
tau = full['estimated_taus']
w = np.r_[full['hereditary_weights'], full['regime_mode_weights']]
print('full taus', tau, 'weights', w)
assert np.all(tau > 0)
z = np.zeros((len(past), 2))
for j, step in enumerate(dt):
    a = np.exp(-step / tau)
    z = a * z + (1-a) * past[:, j, None]
g0, g1 = (hold[:, None] - z).T
v = w[2] + drift[:, 1:] @ w[3:7] + w[7] * np.sign(hold) * np.sqrt(np.abs(hold))
ref_rate = (w[0]*g0 + w[8]*(g0-g1))/tau[0] + w[1]*g1/tau[1] + v
ref_curve = baseline[:, None] + (w[0]*g0 + w[8]*(g0-g1))[:, None]*(-np.expm1(-times/tau[0])) + (w[1]*g1)[:, None]*(-np.expm1(-times/tau[1])) + v[:, None]*times/(1+.85*times)
np.testing.assert_allclose(full['predicted_strain'], ref_curve[n:], atol=1e-12)
np.testing.assert_allclose(full['initial_rate_pred'], ref_rate[n:], atol=1e-12)
check_curve, _ = m._design(tau, past[n:], hold[n:], drift[n:], dt, np.array([0., 1e-7]))
np.testing.assert_allclose(check_curve[:, 0] @ w, 0., atol=1e-12)
np.testing.assert_allclose(((check_curve[:, 1]-check_curve[:, 0]) @ w)/1e-7, ref_rate[n:], atol=1e-5, rtol=1e-5)

def error(x):
    return 'rmse %.6g, medAE %.6g, q90AE %.6g' % (np.sqrt(np.nanmean(x*x)), np.nanmedian(np.abs(x)), np.nanquantile(np.abs(x), .9))

for group, labels in [('lab', lab), ('stratum', stratum)]:
    ec, er = np.empty_like(y), np.empty_like(rate)
    for label in np.unique(labels):
        val = labels == label
        fit, pred = np.flatnonzero(~val), np.flatnonzero(val)
        t, coef = m._fit(past[fit], hold[fit], drift[fit], dt, times, y[fit]-baseline[fit,None], rate[fit])
        a, b = m._design(t, past[pred], hold[pred], drift[pred], dt, times)
        ec[val], er[val] = baseline[pred,None] + a @ coef - y[val], b @ coef - rate[val]
        print(group, label, 'taus', t, 'regime', coef[7:], 'curve', error(ec[val]), 'rate', error(er[val]))
    print(group, 'worst horizon', np.argmax(np.nanmean(ec*ec, axis=0)), error(ec[:, np.argmax(np.nanmean(ec*ec, axis=0))]))
    for name, mask in [('missing coupon', missing), ('complete coupon', ~missing), ('negative hold', hold[:n] < 0), ('nonnegative hold', hold[:n] >= 0)]:
        print(group, name, 'n', mask.sum(), 'curve', error(ec[mask]), 'rate', error(er[mask]))

cal = d['calibration_load_cells']
base = m._calibrate(cal)
for channel in range(4):
    idx = np.flatnonzero(cal['lab_id'] == channel)
    struck = {key: np.array(values, copy=True) for key, values in cal.items()}
    struck['true_stress'][idx[0]] += 4.5
    np.testing.assert_allclose(m._calibrate(struck), base, rtol=0., atol=2e-4)
    keep = np.ones(len(cal['lab_id']), dtype=bool)
    keep[idx[1:]] = False
    sparse = m._calibrate({key: values[keep] for key, values in cal.items()})
    np.testing.assert_allclose(sparse[np.arange(4) != channel], base[np.arange(4) != channel])
    print('sparse channel', channel, 'archive', sparse[channel], 'current', base[channel])
print('Independent recurrence, initial derivative, and calibration path checks passed.')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch03.py
```
- **Container Exit Code**: `0` (`15020 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rmse 0.457357, medAE 0.0115593, q90AE 0.0347961 rate rmse 0.0235955, medAE 0.0113653, q90AE 0.0285402
lab worst horizon 4 rmse 4.81581, medAE 0.0139102, q90AE 0.164716
lab missing coupon n 20 curve rmse 0.501606, medAE 0.0773422, q90AE 0.266214 rate rmse 0.169786, medAE 0.0974033, q90AE 0.267005
lab complete coupon n 160 curve rmse 1.7061, medAE 0.0110012, q90AE 0.0295361 rate rmse 0.017563, medAE 0.0108159, q90AE 0.0285387
lab negative hold n 98 curve rmse 0.67324, medAE 0.0120011, q90AE 0.0413649 rate rmse 0.0410564, medAE 0.014184, q90AE 0.0386322
lab nonnegative hold n 82 curve rmse 2.28018, medAE 0.0126968, q90AE 0.149243 rate rmse 0.0749558, medAE 0.0113246, q90AE 0.0666199
stratum 0 taus [0.53123423 3.96667329] regime [ 0.3121489  -0.21838132] curve rmse 2.75183, medAE 0.0117878, q90AE 0.122588 rate rmse 0.0602469, medAE 0.0132214, q90AE 0.0515778
stratum 1 taus [0.53168022 3.9840473 ] regime [ 0.31101834 -0.21585494] curve rmse 0.426479, medAE 0.0127676, q90AE 0.0924969 rate rmse 0.0756373, medAE 0.0111097, q90AE 0.0448788
stratum 2 taus [0.53310875 4.07467646] regime [ 0.30329371 -0.22147095] curve rmse 0.701688, medAE 0.0115249, q90AE 0.0343826 rate rmse 0.0222582, medAE 0.0127774, q90AE 0.0315021
stratum worst horizon 4 rmse 4.81591, medAE 0.01351, q90AE 0.166203
stratum missing coupon n 20 curve rmse 0.501667, medAE 0.0757014, q90AE 0.268426 rate rmse 0.170265, medAE 0.0966119, q90AE 0.266859
stratum complete coupon n 160 curve rmse 1.70613, medAE 0.01089, q90AE 0.0294089 rate rmse 0.0174262, medAE 0.0106901, q90AE 0.0288807
stratum negative hold n 98 curve rmse 0.67323, medAE 0.0119519, q90AE 0.042909 rate rmse 0.0411242, medAE 0.0139258, q90AE 0.0370155
stratum nonnegative hold n 82 curve rmse 2.28023, medAE 0.0126382, q90AE 0.149252 rate rmse 0.0751142, medAE 0.0114329, q90AE 0.0673376
sparse channel 0 archive [1.28996334 0.11998329] current [1.28996286 0.11997817]
sparse channel 1 archive [ 0.82003341 -0.15000563] current [ 0.8199772  -0.14996294]
sparse channel 2 archive [1.16998653 0.13997717] current [1.16999799 0.14000869]
sparse channel 3 archive [ 0.75996642 -0.11003498] current [ 0.76000329 -0.11003612]
Independent recurrence, initial derivative, and calibration path checks passed.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c0c165a6af85e1b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c0c165a6 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`26703 ms` LLM)
- **Rationale**: Compare the largest held-out strain errors with the rest of the same trajectories and their independently measured initial rates. Isolated jumps support retaining the robust fit; sustained errors would warrant revisiting the model or preprocessing.
- **Alternatives Weighed**: `Change the model to reduce raw worst-horizon RMSE` (rejected: The large gap between RMSE and median error calls for inspecting the largest errors before changing the specified physics.)

```bash
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/verify_batch03.py').read_text().split('\ncal = ')[0]
s += '''
    if group == 'lab':
        ranked = np.argsort(np.nan_to_num(np.abs(ec), nan=-1).ravel())[::-1][:6]
        for row, col in zip(*np.unravel_index(ranked, ec.shape)):
            print('largest strain residual', 'lab', lab[row], 'stratum', stratum[row], 'horizon', col, 'observed', round(y[row,col], 4), 'predicted', round(y[row,col]+ec[row,col], 4), 'same-row residuals', np.round(ec[row], 3), 'rate residual', round(er[row], 4))
'''
Path('/tmp/verify_batch03_residuals.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch03_residuals.py | grep -E 'largest strain residual|^full taus'
```
- **Container Exit Code**: `0` (`13178 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full taus [0.53185614 4.00562371] weights [ 0.6779529   0.55198201  0.13907094  0.15718644 -0.12773427  0.11838727
largest strain residual lab 2 stratum 0 horizon 4 observed 66.5288 predicted 2.1647 same-row residuals [-4.0000e-03 -3.0000e-03 -1.1000e-02 -2.3000e-02 -6.4364e+01  3.3000e-02
largest strain residual lab 0 stratum 2 horizon 9 observed 11.8566 predicted -2.2768 same-row residuals [ 1.0000e-03 -9.0000e-03 -7.0000e-03  8.0000e-03  8.0000e-03  1.3000e-02
largest strain residual lab 1 stratum 0 horizon 8 observed -10.907 predicted 0.5218 same-row residuals [-1.8000e-02 -1.5000e-02  9.0000e-03  8.0000e-03  2.5000e-02 -1.4000e-02
largest strain residual lab 3 stratum 2 horizon 3 observed 7.05 predicted 2.0559 same-row residuals [ 5.000e-03 -1.600e-02  1.600e-02 -4.994e+00  1.000e-02  6.000e-03
largest strain residual lab 0 stratum 1 horizon 6 observed 5.8941 predicted 1.0978 same-row residuals [-0.023 -0.058 -0.052 -0.023 -0.051 -0.063 -4.796 -0.055 -0.021 -0.065] rate residual -0.0692
largest strain residual lab 0 stratum 1 horizon 2 observed -6.9166 predicted -2.2863 same-row residuals [ 2.00e-03 -6.00e-03  4.63e+00 -6.00e-03 -2.10e-02  3.00e-02  9.00e-03

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c0c165a6af85e1b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c0c165a6 (directory).
```

### Turn 8 — `submit_completed_task` (`22466 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema and physics checks. Lab and stratum holdouts recovered similar time constants, and the largest strain deviations were isolated within otherwise well-predicted trajectories.
- **Alternatives Weighed**: `Refit to reduce raw worst-horizon RMSE` (rejected: The largest residual was 64.36 while adjacent errors on the same trajectory were about 0.02 and 0.03. Fitting that spike would distort the smooth response.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:7414'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The isolated schema validator, independent memory and initial-rate checks, and sparse and EMI calibration tests passed. The worst held-out strain horizon had a median absolute error of 0.014; isolated observation spikes drove its RMSE to 4.816. Missing-coupon rows had the highest initial-rate RMSE (0.170).`

### Turn 9 — `execute_in_remote_docker` (`47692 ms` LLM)
- **Rationale**: Verify the two constitutive terms independently and measure what removing them costs on real held-out lab measurements, with parameters fitted only on the other labs. Separate complete and missing coupons to see whether systematic lab offsets persist when predictors are observed. An identity failure would require a basis fix; consistent signed errors on complete coupons would justify revisiting alignment.
- **Alternatives Weighed**: `Run relay-mode, stochastic-control, or balanced-accuracy tests` (rejected: The observed outcomes are continuous strain and initial rate. The applicable checks are the rational flow and branch-gap difference identities, plus signed residuals within held-out labs.)

```bash
cat > /tmp/audit_batch03.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb'))[0]['dataset']
tr = d['train']
n = len(tr['lab_id'])
dt, t = d['past_dt'], d['eval_times']
p, h, x, b = [a[:n] for a in m._prepare(d)]
y, r = tr['strain_curves_obs'] - b[:, None], tr['instant_rate_obs']
missing = ~np.isfinite(tr['coupon_features_raw']).all(axis=1)
lab = np.asarray(tr['lab_id'])
for field in ('train', 'test'):
    print(field, 'fields', sorted(d[field]))
print('dataset fields', sorted(d), 'shear banding index', d['shear_banding_index'])
for g in np.unique(lab):
    print('lab', g, 'rows', (lab == g).sum(), 'missing', ((lab == g) & missing).sum(), 'positive stress fraction', np.mean(h[lab == g] > 0))

modes = ('specified', 'linear flow', 'no branch contrast', 'no signed slip')
err = {name: [np.empty_like(y), np.empty_like(r)] for name in modes}
for g in np.unique(lab):
    train, val = lab != g, lab == g
    tau, w = m._fit(p[train], h[train], x[train], dt, t, y[train], r[train])
    a, q = m._design(tau, p, h, x, dt, t)
    # Audit the two nontrivial constitutive bases separately, not merely their sum.
    np.testing.assert_allclose(a[:, :, 2], t[None, :] / (1 + .85*t[None, :]) * np.ones((n, 1)))
    np.testing.assert_allclose(q[:, 2], 1.)
    decay = -np.expm1(-t/tau[0])
    gap_difference = (q[:, 0] * tau[0] - q[:, 1] * tau[1])
    np.testing.assert_allclose(a[:, :, 8], gap_difference[:, None] * decay[None, :], atol=1e-12)
    np.testing.assert_allclose(q[:, 8], gap_difference / tau[0], atol=1e-12)
    yc = train[:, None] & np.isfinite(y)
    yr = train & np.isfinite(r)
    targets = np.r_[y[yc], r[yr]]
    residual = np.r_[(a @ w - y)[yc], (q @ w - r)[yr]]
    def mad(v):
        return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))
    scale = np.r_[np.full(yc.sum(), 2*mad(residual[:yc.sum()])), np.full(yr.sum(), 2*mad(residual[yc.sum():]))]
    for name in modes:
        A, Q = a.copy(), q.copy()
        cols = np.arange(9)
        if name == 'linear flow':
            A[:, :, 2:8] *= (1 + .85*t[None, :, None])
        if name == 'no branch contrast':
            cols = cols[cols != 8]
        if name == 'no signed slip':
            cols = cols[cols != 7]
        A, Q = A[:, :, cols], Q[:, cols]
        design = np.vstack((A[yc], Q[yr]))
        fit = w if name == 'specified' else least_squares(lambda v: (design @ v - targets)/scale, w[cols], loss='cauchy', x_scale='jac').x
        err[name][0][val] = (A @ fit - y)[val]
        err[name][1][val] = (Q @ fit - r)[val]

def med(a):
    return float(np.nanmedian(np.abs(a)))
for name, (e, f) in err.items():
    print(name, 'pooled curve/rate medAE', round(med(e), 5), round(med(f), 5), 'worst lab curve/rate medAE', round(max(med(e[lab == g]) for g in np.unique(lab)), 5), round(max(med(f[lab == g]) for g in np.unique(lab)), 5))
e, f = err['specified']
for g in np.unique(lab):
    for mask, label in ((~missing, 'complete'), (missing, 'missing')):
        rows = (lab == g) & mask
        if rows.any():
            print('held-out lab', g, label, 'n', rows.sum(), 'signed curve/rate medians', np.round([np.nanmedian(e[rows]), np.nanmedian(f[rows])], 5), 'rate medAE', round(med(f[rows]), 5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_batch03.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`9121 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
train fields ['aux_chamber_telemetry', 'coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd']
test fields ['aux_chamber_telemetry', 'coupon_features_raw', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'stratum', 'stress_hold_cmd']
dataset fields ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress'] shear banding index 0.82
lab 0 rows 49 missing 9 positive stress fraction 0.4897959183673469
lab 1 rows 47 missing 6 positive stress fraction 0.44680851063829785
lab 2 rows 45 missing 4 positive stress fraction 0.4444444444444444
lab 3 rows 39 missing 1 positive stress fraction 0.4358974358974359
specified pooled curve/rate medAE 0.01238 0.0127 worst lab curve/rate medAE 0.01354 0.01607
linear flow pooled curve/rate medAE 0.04676 0.18633 worst lab curve/rate medAE 0.05253 0.20947
no branch contrast pooled curve/rate medAE 0.0464 0.1632 worst lab curve/rate medAE 0.04885 0.20148
no signed slip pooled curve/rate medAE 0.06249 0.08254 worst lab curve/rate medAE 0.07353 0.10102
held-out lab 0 complete n 40 signed curve/rate medians [-0.00025 -0.00016] rate medAE 0.01169
held-out lab 0 missing n 9 signed curve/rate medians [0.01129 0.04225] rate medAE 0.04355
held-out lab 1 complete n 41 signed curve/rate medians [ 0.00145 -0.00281] rate medAE 0.00864
held-out lab 1 missing n 6 signed curve/rate medians [0.02967 0.07478] rate medAE 0.15561
held-out lab 2 complete n 41 signed curve/rate medians [-0.00221 -0.00158] rate medAE 0.01455
held-out lab 2 missing n 4 signed curve/rate medians [0.08594 0.14276] rate medAE 0.14276
held-out lab 3 complete n 38 signed curve/rate medians [ 4.00e-05 -1.62e-03] rate medAE 0.01112
held-out lab 3 missing n 1 signed curve/rate medians [-0.08319 -0.09985] rate medAE 0.09985
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c0c165a6af85e1b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c0c165a6 (directory).
```

### Turn 10 — `submit_completed_task` (`21529 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution. Grouped validation supports the specified rational flow and branch-gap contrast, and the independent recurrence and derivative checks confirm their implementation. The largest raw strain errors were isolated within otherwise well-predicted trajectories, so fitting them would distort the smooth response.
- **Alternatives Weighed**: `Simplify the constitutive response` (rejected: Linearizing terminal flow, dropping the branch-gap contrast, or dropping signed slip each increased held-out strain and initial-rate median errors.); `Add a lab-specific residual correction` (rejected: Complete-coupon rows had absolute signed median residuals below 0.003 for both outputs in every held-out lab; the larger errors were concentrated on rows with missing coupons.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:7414'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The isolated schema validator, independent memory and initial-rate checks, and sparse and EMI calibration tests passed. In leave-one-lab-out validation, the worst lab’s median absolute errors were 0.0135 for strain and 0.0161 for initial rate. Isolated measurement spikes drove the worst strain-horizon RMSE to 4.816; missing-coupon rows were the hardest initial-rate slice (RMSE 0.170).`

### Turn 11 — `execute_in_remote_docker` (`30354 ms` LLM)
- **Rationale**: The audit already verified the rational-flow and branch-gap terms. This paired comparison tests whether fitting on imputed rows materially limits accuracy: both candidates use the same prescribed prediction-time alignment, and only training-row eligibility changes. Promote a change only if it improves the missing-coupon bottleneck across both holdout schemes without materially worsening complete-coupon predictions.
- **Alternatives Weighed**: `Change the prescribed panel imputation or add a lab-specific correction` (rejected: Cell-balanced imputation is part of the contract, and complete-coupon signed median residuals were below 0.003 in every held-out lab.)

```bash
cat > /tmp/batch03_missing_margin.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
n = len(tr['lab_id'])
p, h, x, b = (a[:n] for a in m._prepare(d))
y = tr['strain_curves_obs'] - b[:, None]
r = tr['instant_rate_obs']
lab = np.asarray(tr['lab_id'])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for grouping in ('lab_id', 'stratum'):
    groups = np.asarray(tr[grouping])
    for complete_only in (False, True):
        ec, er = np.empty_like(y), np.empty_like(r)
        fitted = []
        for g in np.unique(groups):
            val = groups == g
            fit = ~val & (~missing if complete_only else True)
            tau, weights = m._fit(p[fit], h[fit], x[fit], d['past_dt'], d['eval_times'], y[fit], r[fit])
            a, q = m._design(tau, p[val], h[val], x[val], d['past_dt'], d['eval_times'])
            ec[val], er[val] = a @ weights - y[val], q @ weights - r[val]
            fitted.append(np.r_[tau, weights])
        med = lambda a: np.nanmedian(np.abs(a))
        rms = lambda a: np.sqrt(np.nanmean(a*a))
        print(grouping, 'complete_only', complete_only,
              'missing curve medAE / rate medAE / rate RMSE', np.round([med(ec[missing]), med(er[missing]), rms(er[missing])], 6),
              'complete worst-group curve / rate medAE', np.round([max(med(ec[(groups == g) & ~missing]) for g in np.unique(groups)), max(med(er[(groups == g) & ~missing]) for g in np.unique(groups))], 6),
              'tau range', np.round(np.ptp(fitted, axis=0)[:2], 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/batch03_missing_margin.py
```
- **Container Exit Code**: `0` (`19235 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab_id complete_only False missing curve medAE / rate medAE / rate RMSE [0.077342 0.097403 0.169786] complete worst-group curve / rate medAE [0.011063 0.014555] tau range [0.000708 0.053937]
lab_id complete_only True missing curve medAE / rate medAE / rate RMSE [0.078697 0.098094 0.170512] complete worst-group curve / rate medAE [0.011595 0.014071] tau range [0.000752 0.025783]
stratum complete_only False missing curve medAE / rate medAE / rate RMSE [0.075701 0.096612 0.170265] complete worst-group curve / rate medAE [0.011658 0.011677] tau range [0.001875 0.108003]
stratum complete_only True missing curve medAE / rate medAE / rate RMSE [0.077741 0.098129 0.170771] complete worst-group curve / rate medAE [0.011442 0.011778] tau range [0.000613 0.039776]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c0c165a6af85e1b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c0c165a6 (directory).
```

### Turn 12 — `submit_completed_task` (`18651 ms` LLM)
- **Rationale**: Retain the verified solver. The constitutive ablations worsened held-out predictions, complete-coupon residuals showed no material lab offset, and the final paired comparison found no prediction gain on the limiting slice.
- **Alternatives Weighed**: `Train only on complete coupons` (rejected: It narrowed the fitted slow time constant’s range across folds but did not improve missing-coupon strain or rate errors in either holdout scheme.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The isolated schema check, independent memory and initial-rate checks, and sparse and EMI calibration tests passed. The worst leave-one-lab-out median absolute errors were 0.0135 for strain and 0.0161 for initial rate. Missing-coupon rows remained the hardest rate slice (RMSE 0.170); training only on complete coupons did not improve it.`

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
