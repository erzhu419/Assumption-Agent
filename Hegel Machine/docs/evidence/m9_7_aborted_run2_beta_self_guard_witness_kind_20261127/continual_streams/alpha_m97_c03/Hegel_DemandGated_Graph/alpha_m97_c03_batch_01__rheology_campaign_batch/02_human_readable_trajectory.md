# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `418,748` (`404,779` / `118,656` / `13,969`) |
| **Total Episode Wall Time (s)** | `473.21s` |

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

### Turn 1 — `execute_in_remote_docker` (`17230 ms` LLM)
- **Rationale**: Inspect the schema and prior implementation before coding: a mismatch in the label layout, calibration rules, or observation operator changes which components can be reused. The skill lookup checks whether panel alignment or relaxation primitives are already available.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: Its prior branch failed the public gate, so only components that match the current regime should be reused.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && wc -l /tmp/hegel_continual_archive/* && sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment cell-balanced imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
l_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```

### Turn 2 — `execute_in_remote_docker` (`19480 ms` LLM)
- **Rationale**: The earlier output truncated the archived implementation. Inspect it directly and print compact data summaries to locate missing features, confirm the calibration fallback format, and detect any public channel that needs spike handling before fitting.

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))
print('items',len(p),'item keys',list(p[0]))
for item in p:
 d=item['dataset']; print('top keys',list(d),'dt',d['past_dt'],'time',d['eval_times'])
 for section in ('train','test','calibration_load_cells'):
  s=d[section]; print(section)
  for k,v in s.items():
   if isinstance(v,dict): print(k,{a:np.shape(b) for a,b in v.items()});continue
   x=np.asarray(v); print(k,x.shape,x.dtype, 'missing='+str(np.isnan(x).sum()) if x.dtype.kind=='f' else '')
  print('lab counts',np.unique(s['lab_id'],return_counts=True))
 c=d['calibration_load_cells']
 for j in range(4):
  m=c['lab_id']==j; x,y=c['cmd_stress'][m],c['true_stress'][m]; co=np.polyfit(x,y,1); print('calibration',j,co,'max residual',max(abs(y-np.polyval(co,x))))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history sample',str(h)[:1800])
PY
```
- **Container Exit Code**: `0` (`492 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.2965 0.6942 0.5869 0.3435
 0.3902 0.5082 0.3785 0.7877 0.3776 0.2105 0.2875 0.2881] time [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
train
lab_id (180,) int64 
stratum (180,) int64 
coupon_features_raw (180, 4) float64 missing=23
ref_coupons (180, 32) float64 missing=238
past_stress_cmd (180, 18) float64 missing=0
stress_hold_cmd (180,) float64 missing=0
strain_0 (180,) float64 missing=0
aux_chamber_telemetry (180, 3) float64 missing=0
strain_curves_obs (180, 10) float64 missing=0
instant_rate_obs (180,) float64 missing=0
lab counts (array([0, 1, 2, 3]), array([35, 49, 57, 39]))
test
lab_id (72,) int64 
stratum (72,) int64 
coupon_features_raw (72, 4) float64 missing=8
ref_coupons (72, 32) float64 missing=95
past_stress_cmd (72, 18) float64 missing=0
stress_hold_cmd (72,) float64 missing=0
strain_0 (72,) float64 missing=0
aux_chamber_telemetry (72, 3) float64 missing=0
lab counts (array([0, 1, 2, 3]), array([17, 16, 21, 18]))
calibration_load_cells
lab_id (40,) int64 
cmd_stress (40,) float64 missing=0
true_stress (40,) float64 missing=0
lab counts (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
calibration 0 [1.28998908 0.12000143] max residual 0.00021677044913304133
calibration 1 [ 0.82003896 -0.15002605] max residual 0.0001420176423634345
calibration 2 [1.16998093 0.14000924] max residual 0.0001616162574000768
calibration 3 [ 0.76000147 -0.10998819] max residual 0.0001720959915643052
history sample [{'episode_short_name': 'alpha_m97_c03_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.893607, -0.483769, -1.39264, 1.611445, -1.810815, -1.870654, 1.48082, 1.268821, -0.114529, 0.252739, -1.204476, -0.664616, -0.154159, -0.613549, -1.259125, -2.011382, 2.18392, -0.23172, 1.98347, -2.000907, -0.486537, -0.139492, -0.631342, -1.874772, -0.708257, 2.13352, 0.386634, 1.10423, -1.407517, 1.70372, 1.521295, 1.987385, 1.659545, -0.294044, 0.359165, -0.017198, 1.319848, 0.959274, 1.866895, -0.480321], 'true_stress': [0.893713, -0.483774, -1.392656, 1.611564, -1.810667, -1.870622, 1.480773, 1.268837, -0.114695, 0.252685, -1.204516, -0.664704, -0.154113, -0.613571, -1.259248, -2.011128, 2.183777, -0.231673, 1.983509, -2.001075, -0.486729, -0.139381, -0.631253, -1.874747, -0.708398, 2.133538, 0.386675, 1.104402, -1.407596, 1.703866, 1.521337, 1.987399, 1.659449, -0.294029, 0.359147, -0.017404, 1.319703, 0.959321, 1.866928, -0.480472]}]
```

### Turn 3 — `execute_in_remote_docker` (`20259 ms` LLM)
- **Rationale**: The archived alignment was truncated and the treatment of zero-damage rows affects the observation operator. Recover the exact panel convention and count damage states by lab to decide whether the inactive branch needs separate fitting and validation.

```bash
sed -n '/^def _align/,/^def _design/p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=p['dataset']; g=p['gt'];print('gt fields for validation:', {k:np.shape(v) for k,v in g.items()})
c=d['calibration_load_cells']; ab=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==j],c['true_stress'][c['lab_id']==j],1) for j in range(4)])
for split in ('train','test'):
 s=d[split]; lab=s['lab_id']; x=s['past_stress_cmd']*ab[lab,0,None]+ab[lab,1,None]; damage=np.maximum(abs(x)-d['crit_stress'],0)**2 @ d['past_dt']; print(split,'damage quantiles',np.quantile(damage,[0,.1,.5,.9,1]),'inactive/total per lab',[(sum((lab==j)&(damage==0)),sum(lab==j)) for j in range(4)],'stratum counts',np.unique(s['stratum'],return_counts=True))
PY
```
- **Container Exit Code**: `0` (`504 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _design(split, x, gain, bias, dt, t):
gt fields for validation: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
train damage quantiles [0.         0.         3.36574325 4.67106028 5.45569795] inactive/total per lab [(11, 35), (17, 49), (21, 57), (11, 39)] stratum counts (array([0, 1, 2]), array([48, 65, 67]))
test damage quantiles [0.         0.         3.25925347 4.3940671  5.47233208] inactive/total per lab [(7, 17), (4, 16), (9, 21), (4, 18)] stratum counts (array([0, 1, 2]), array([19, 28, 25]))
```

### Turn 4 — `execute_in_remote_docker` (`61683 ms` LLM)
- **Rationale**: Fit the nine linear weights conditional on the two relaxation times using training observations only. Score the resulting artifact separately on clean strain and initial rate in each lab and damage state; systematic errors on inactive rows would trigger a review of their flow law.
- **Alternatives Weighed**: `Apply logarithmic flow to every row` (rejected: A third of the public rows have zero damage, and the task specifies the plateau only for positive damage.)

```bash
set -e
out=/holdout_task/alpha_m97_c03_batch_01_solver.py
{ printf '%s\n' '"""Calibrated two-branch hereditary model for the rheology campaign."""' 'import json' 'import numpy as np' 'from scipy.optimize import least_squares' "_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'"; sed -n '/^def _align(/,/^def _design(/p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py | sed '$d'; } > "$out"
cat >> "$out" <<'PY'

def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _archive_line(history, lab):
    for sweep in reversed(history):
        use = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[use]
        y = np.asarray(sweep['true_stress'])[use]
        if len(x) >= 2 and np.ptp(x) > 0:
            return _line(x, y)
    return np.array([1., 0.])


def _calibrate(cal):
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        use = cal['lab_id'] == lab
        x, y = cal['cmd_stress'][use], cal['true_stress'][use]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            distinct = dx != 0
            if distinct.any():
                a = np.median((y[j] - y[i])[distinct] / dx[distinct])
                b = np.median(y - a*x)
                clean = np.abs(y - (a*x + b)) <= 3.
        needs_history = len(x) < 2 or not clean.all()
        if needs_history and history is None:
            try:
                with open(_ARCHIVE) as handle:
                    history = json.load(handle)
            except FileNotFoundError:
                history = []
        if clean.sum() >= 4 or (not needs_history and clean.sum() >= 2):
            result[lab] = _line(x[clean], y[clean])
        else:
            result[lab] = _archive_line(history or [], lab)
    return result


def _design(split, aligned, calibration, dt, time, critical, banding):
    gain, bias = calibration[split['lab_id']].T
    stress = gain[:, None]*split['past_stress_cmd'] + bias[:, None]
    hold = gain*split['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(stress)-critical, 0.)**2 @ dt
    active = (damage > 0.) & (banding <= .50)
    act = np.where(active, np.tanh(damage), 0.)
    flow = np.where(active[:, None], np.log1p(time), time)
    covariates = np.column_stack((np.ones(len(hold)), aligned))
    elapsed_after = np.cumsum(dt[::-1])[::-1] - dt

    def build(log_tau):
        tau = np.exp(log_tau)
        memory_kernel = -np.expm1(-dt[:, None]/tau) * np.exp(-elapsed_after[:, None]/tau)
        z = stress @ memory_kernel
        gap = hold[:, None] - z
        rise = -np.expm1(-time[:, None]/tau)
        amplitude = act*hold
        log_memory = act*z[:, 0]
        n, k = len(hold), len(time)
        curve = np.empty((n, k, 9))
        curve[:, :, :2] = gap[:, None, :]*rise[None, :, :]
        curve[:, :, 2:7] = flow[:, :, None]*covariates[:, None, :]
        curve[:, :, 7] = amplitude[:, None]*rise[None, :, 0]
        curve[:, :, 8] = log_memory[:, None]*flow
        rate = np.column_stack((gap/tau, covariates, amplitude/tau[0], log_memory))
        return curve, rate
    return build


def _fit(build, strain, rate, dt, time):
    strain = strain.ravel()
    # Estimate relative measurement precision from the two training residual blocks.
    def profile(log_tau, weight):
        curve, instant = build(log_tau)
        matrix = np.vstack((curve.reshape(-1, 9), weight*instant))
        obs = np.concatenate((strain, weight*rate))
        beta = np.linalg.lstsq(matrix, obs, rcond=None)[0]
        return matrix @ beta - obs, beta

    lower = max(np.min(dt)*.025, 1e-4)
    upper = 30*max(np.sum(dt), np.max(time))
    bounds = np.log([lower, upper])
    starts = ((.3, 2.0), (.65, 4.5), (.15, 1.0), (1.4, 8.0))
    def run(start, weight):
        return least_squares(lambda q: profile(q, weight)[0], start, bounds=bounds,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=160)
    weight = .5
    fits = [run(np.log(np.clip(start, lower*1.01, upper*.99)), weight) for start in starts]
    ordered = [fit for fit in fits if fit.x[0] < fit.x[1]]
    best = min(ordered or fits, key=lambda fit: fit.cost)
    for _ in range(2):
        residual, _ = profile(best.x, weight)
        def mad(x):
            return np.median(np.abs(x-np.median(x)))
        new_weight = np.clip(mad(residual[:len(strain)]) / max(mad(residual[len(strain):]/weight), 1e-12), .03, 30.)
        updated = run(best.x, new_weight)
        if best.x[0] < best.x[1] and updated.x[0] >= updated.x[1]:
            break
        best, weight = updated, new_weight
    return np.exp(best.x), profile(best.x, weight)[1]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    calibration = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    params = (calibration, dt, time, dataset['crit_stress'], dataset['shear_banding_index'])
    build = _design(train, x_train, *params)
    taus, beta = _fit(build, train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'], dt, time)
    curve, rate = _design(test, x_test, *params)(np.log(taus))
    return {'estimated_taus': taus, 'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:], 'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c03_batch_01_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c03_batch_01(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'fit',o[k],'truth',g[j],'maximum abs error',max(abs(o[k]-g[j])))
lab=d['test']['lab_id']; act=(np.maximum(abs(g['te_hist_true'])-d['crit_stress'],0)**2 @ d['past_dt'])>0
for l in range(4):
 for a in (False,True):
  use=(lab==l)&(act==a)
  print('lab/active/n',l,a,sum(use),'curve RMSE',np.sqrt(np.mean((o['predicted_strain'][use]-g['te_clean'][use])**2)),'rate RMSE',np.sqrt(np.mean((o['initial_rate_pred'][use]-g['te_true_rate0'][use])**2)))
PY
```
- **Container Exit Code**: `0` (`1234 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus fit [0.55493983 2.15514345] truth [0.568 4.113] maximum abs error 1.957856545712958
hereditary_weights fit [ 0.70587509  0.54994263  0.09291859  0.06181145 -0.08362619  0.05850462
 -0.03735361] truth [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] maximum abs error 0.09818854910250703
regime_mode_weights fit [0.0868404  0.26625818] truth [0.28 0.24] maximum abs error 0.19315959586284065
lab/active/n 0 False 7 curve RMSE 0.2233234187230063 rate RMSE 0.24111378421685928
lab/active/n 0 True 10 curve RMSE 0.16366425222368997 rate RMSE 0.26523439585192293
lab/active/n 1 False 4 curve RMSE 0.21454334099452482 rate RMSE 0.17720380241704262
lab/active/n 1 True 12 curve RMSE 0.22913768486844147 rate RMSE 0.1947246672314101
lab/active/n 2 False 9 curve RMSE 0.21389196634085741 rate RMSE 0.2567324931278572
lab/active/n 2 True 12 curve RMSE 0.18104101109714585 rate RMSE 0.20494943382629977
lab/active/n 3 False 4 curve RMSE 0.18689536174707325 rate RMSE 0.2168767816725233
lab/active/n 3 True 14 curve RMSE 0.180950145448074 rate RMSE 0.22403363704804471

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8fcb2532a8a2caf1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8fcb2532 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`31134 ms` LLM)
- **Rationale**: Use public ground truth only in a diagnostic, not in the solver. Comparing reconstructed physical stress, the initial slope, and the curve after removing the hereditary terms distinguishes calibration and drift errors from a wrong time basis.
- **Alternatives Weighed**: `Tune optimizer starts or residual weights first` (rejected: The parameter and prediction errors are large enough that an operator mismatch should be ruled out before tuning.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; s=importlib.util.spec_from_file_location('m',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0];d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; xtr,x=m._align(tr,te); dt,t=d['past_dt'],d['eval_times']; cal=m._calibrate(d['calibration_load_cells']); tau=g['taus']; beta=np.r_[g['theta'],g['vp_theta']]; lab=te['lab_id']; past=g['te_hist_true']; hold=g['te_hold_true']; print('yield/critical',d['yield_stress'],d['crit_stress']); print('physical stress max reconstruction errors',np.max(abs(past-(te['past_stress_cmd']*cal[lab,0,None]+cal[lab,1,None]))),np.max(abs(hold-(te['stress_hold_cmd']*cal[lab,0]+cal[lab,1])))); D=np.maximum(abs(past)-d['crit_stress'],0)**2 @dt; a=np.tanh(D); active=D>0; z=np.zeros((len(lab),2));
for h,ds in zip(past.T,dt): z=np.exp(-ds/tau)*z+(-np.expm1(-ds/tau))*h[:,None]
gap=hold[:,None]-z; rise=-np.expm1(-t[:,None]/tau); expo=(gap*beta[:2])@rise.T+beta[7]*(a*hold)[:,None]*rise[:,0]; init=(gap*beta[:2]/tau).sum(1)+beta[7]*a*hold/tau[0]; y=g['te_clean']-te['strain_0'][:,None]-expo; r=g['te_true_rate0']-init; drift=beta[2]+x@beta[3:7]+beta[8]*a*z[:,0]; rm=lambda q:np.sqrt(np.mean(q*q)); print('truth-coefficient rate rms',[(str(b),rm((r-drift)[active==b])) for b in (False,True)]);
for b in (False,True):
 mask=active==b; print('state',b,'rate difference quantiles',np.quantile((r-drift)[mask],[0,.25,.5,.75,1]));
 for name,ph in [('linear',t),('log',np.log1p(t)),('sqrt',2*(np.sqrt(1+t)-1)),('none',np.zeros_like(t))]: print(name,'rms using measured initial slope',rm(y[mask]-r[mask,None]*ph),'rms using given drift',rm(y[mask]-drift[mask,None]*ph));
for j in range(4):
 ref=np.r_[tr['ref_coupons'][tr['lab_id']==j],te['ref_coupons'][lab==j]]; f=lab==j; print('lab',j,'panel median mean/sd',np.median(np.nanmean(ref,0)),np.median(np.nanstd(ref,0)),'feature mean/sd',np.mean(x[f],0),np.std(x[f],0),'slope residual mean/sd',np.mean((r-drift)[f]),np.std((r-drift)[f]))
print('hold/damage/slope offsets samples',np.c_[hold,D,r-drift][:16]);
PY
```
- **Container Exit Code**: `0` (`563 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
259  0.00206324  0.13929309]
linear rms using measured initial slope 0.6101335058093961 rms using given drift 0.6427319212958598
log rms using measured initial slope 3.831935385182209e-16 rms using given drift 0.04443224707337248
sqrt rms using measured initial slope 0.21825473353374916 rms using given drift 0.24208822790632348
none rms using measured initial slope 0.42399660339909667 rms using given drift 0.42399660339909667
lab 0 panel median mean/sd -0.6699936988674082 0.797640169599114 feature mean/sd [ 0.21492207 -0.04046799 -0.39921125  0.10329199] [0.93403749 0.98902395 1.02187034 0.81069083] slope residual mean/sd -0.0015620318927729215 0.022035478452098474
lab 1 panel median mean/sd 0.4724206820805475 1.278508107751232 feature mean/sd [-0.13778604  0.06358068  0.20045    -0.12101356] [1.29881077 1.00993106 0.93801973 1.41107297] slope residual mean/sd 0.008787170367937105 0.03373945239459148
lab 2 panel median mean/sd -0.2837997579004786 0.9269109576118877 feature mean/sd [0.59424224 0.13372143 0.08729444 0.18343702] [0.88395753 0.92010962 1.04785001 0.94528551] slope residual mean/sd -0.013452410459210553 0.0320682499585547
lab 3 panel median mean/sd 0.7364494311861511 1.312178982785337 feature mean/sd [-0.04459494 -0.3107345  -0.34173864 -0.15776705] [1.11874371 0.70946617 1.00863881 1.1251077 ] slope residual mean/sd -0.00307865626215388 0.018813660177408514
hold/damage/slope offsets samples [[-1.41055100e+00  3.05795779e+00 -9.69597797e-03]
 [ 1.66761900e+00  4.96279428e+00  1.20246113e-04]
 [-1.53495900e+00  4.16982350e+00 -5.54985876e-03]
 [-2.82756000e-01  3.53040798e+00  9.03221880e-03]
 [ 6.28301000e-01  3.32443296e+00  9.32398288e-03]
 [ 1.26300000e-01  4.11618912e+00 -5.87320861e-05]
 [-2.36510000e-02  5.01345650e+00  2.91534210e-03]
 [-1.66167700e+00  3.96919880e+00 -7.97376587e-03]
 [ 9.13996000e-01  2.88150214e+00 -3.22800601e-03]
 [ 9.48916000e-01  3.38872521e+00 -1.82194564e-05]
 [-1.99368400e+00  4.02813470e+00  1.45321533e-04]
 [-1.14234100e+00  4.09170869e+00 -1.98604983e-02]
 [-1.37676100e+00  3.91843965e+00  4.51964397e-04]
 [-1.74577000e-01  2.15669346e+00 -3.63731325e-04]
 [-7.00292000e-01  3.63015589e+00 -4.36001940e-03]
 [-7.49390000e-02  4.46582219e+00  8.35704778e-03]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8fcb2532a8a2caf1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8fcb2532 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32042 ms` LLM)
- **Rationale**: The clean curves isolate the incorrect basis without changing calibration or optimizer settings. Refit the corrected operator and compare clean test slices; training residuals split by feature missingness will show whether imputation is the next bottleneck.
- **Alternatives Weighed**: `Keep linear flow at zero damage and absorb the mismatch into fitted weights` (rejected: Changing amplitudes cannot reproduce the observed logarithmic time dependence at the same initial slope.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero damage selects the prior linear time basis even in the low-banding campaign.` | Witness: `On the 24 zero-damage public test rows, the clean curve after subtracting hereditary terms matches the inferred initial slope times log1p(t) to 2.34e-16 RMS, versus 0.5978 RMS with the linear basis.` | Blanket Diagnosis: `Damage was incorrectly used to switch both the terminal-flow time basis and the two network amplitudes.` | Sublated Pivot: `Use logarithmic flow throughout the low-banding regime, while retaining the damage gate on both network amplitudes.`

```bash
set -e
sed -i 's/flow = np.where(active\[:, None\], np.log1p(time), time)/# Low banding selects the time basis; damage gates the two network terms.\n    flow = np.log1p(time)[None, :] if banding <= .50 else time[None, :]/' /holdout_task/alpha_m97_c03_batch_01_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/check_batch01.py <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c03_batch_01_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c03_batch_01(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(k,'fit',o[k],'max abs error',np.max(abs(o[k]-g[j])))
lab=d['test']['lab_id']; act=(np.maximum(abs(g['te_hist_true'])-d['crit_stress'],0)**2 @ d['past_dt'])>0
missing=np.any(~np.isfinite(d['test']['coupon_features_raw']),axis=1)
ec=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; rm=lambda e:np.sqrt(np.mean(e*e))
for name,use in [(f'lab {l} active {a}',(lab==l)&(act==a)) for l in range(4) for a in (False,True)]+[(f'missing {v}',missing==v) for v in (False,True)]:
 print(name,'n',sum(use),'curve/rate RMSE',rm(ec[use]),rm(er[use]))
tr,te=d['train'],d['test']; x,_=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); c,r=m._design(tr,x,cal,d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index'])(np.log(o['estimated_taus'])); b=np.r_[o['hereditary_weights'],o['regime_mode_weights']]
for name,e in [('curve',c@b+tr['strain_0'][:,None]-tr['strain_curves_obs']),('rate',r@b-tr['instant_rate_obs'])]:
 mask=np.any(~np.isfinite(tr['coupon_features_raw']),axis=1)
 for missing_rows in (False,True):
  z=e[mask==missing_rows]; print('train',name,'missing',missing_rows,'abs residual q50/q90/q99/max',np.quantile(abs(z),[.5,.9,.99,1]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch01.py
```
- **Container Exit Code**: `0` (`928 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus fit [0.55034886 2.14190343] max abs error 1.9710965679825305
hereditary_weights fit [ 0.69714883  0.5495197   0.16100388  0.12400572 -0.12809292  0.1200281
 -0.06333073] max abs error 0.0359942839613336
regime_mode_weights fit [0.0784326  0.27391565] max abs error 0.20156739722790185
lab 0 active False n 7 curve/rate RMSE 0.1554124347525414 0.22985801240484047
lab 0 active True n 10 curve/rate RMSE 0.13516439650463521 0.27644308239335214
lab 1 active False n 4 curve/rate RMSE 0.0859821340397424 0.07893420783054587
lab 1 active True n 12 curve/rate RMSE 0.10416426925511682 0.2209825535667297
lab 2 active False n 9 curve/rate RMSE 0.16741321436059362 0.23477820325306192
lab 2 active True n 12 curve/rate RMSE 0.13009268658990036 0.21386521400641664
lab 3 active False n 4 curve/rate RMSE 0.08151950374170511 0.1563127772707841
lab 3 active True n 14 curve/rate RMSE 0.09863245167513789 0.20691549381363722
missing False n 65 curve/rate RMSE 0.1229345786619884 0.2180985933658261
missing True n 7 curve/rate RMSE 0.1450679067316999 0.23760594473845956
train curve missing False abs residual q50/q90/q99/max [ 0.08778264  0.2828679   3.61599926 75.9435908 ]
train curve missing True abs residual q50/q90/q99/max [ 0.12079462  0.45492158  0.9615375  20.8288156 ]
train rate missing False abs residual q50/q90/q99/max [0.15995582 0.36058297 0.48196644 0.50851393]
train rate missing True abs residual q50/q90/q99/max [0.13853014 0.31885794 0.54743819 0.55459885]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2a2f132858389266 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2a2f1328 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`36960 ms` LLM)
- **Rationale**: The corrected basis improves strain predictions, but training curve residuals still reach 75.94. Fit with a robust loss and separate residual scales for curves and rates, using complete coupons to avoid propagating imputation error into the shared parameters. Compare all five outputs against the existing public validation baseline.
- **Alternatives Weighed**: `Drop entire coupons with large strain residuals` (rejected: The largest discrepancies occur in individual strain observations; robust fitting can downweight them without discarding the coupon's other measurements.)

```bash
set -e
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection' | head -18
out=/holdout_task/alpha_m97_c03_batch_01_solver.py
python3 - "$out" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); text = p.read_text()
a = text.index('def _fit('); b = text.index('def solve_alpha', a)
text = text[:a] + '''def _fit(build, strain, rate, dt, time, complete):
    # Imputed covariates add a shared drift error to every observation of a coupon.
    # Fit the shared law on measured covariates; alignment still imputes at prediction.
    use = complete if np.count_nonzero(complete) >= 18 else np.ones(len(rate), dtype=bool)
    y_curve = strain[use].ravel()
    y_rate = rate[use]
    lower = max(np.min(dt)*.025, 1e-4)
    upper = 30*max(np.sum(dt), np.max(time))
    bounds = (np.r_[np.full(2, np.log(lower)), np.full(9, -np.inf)],
              np.r_[np.full(2, np.log(upper)), np.full(9, np.inf)])

    def blocks(q):
        curve, instant = build(q)
        return curve[use].reshape(-1, 9), instant[use]

    def residual(p):
        curve, instant = blocks(p[:2])
        return curve @ p[2:] - y_curve, instant @ p[2:] - y_rate

    def run(p, scale):
        def objective(v):
            c, r = residual(v)
            return np.r_[c/scale[0], r/scale[1]]
        return least_squares(objective, p, bounds=bounds, loss='soft_l1',
                             x_scale='jac', max_nfev=160)

    scale = np.array([.02, .02])
    fits = []
    for start in ((.3, 2.), (.65, 4.5), (.15, 1.), (1.4, 8.)):
        q = np.log(np.clip(start, lower*1.01, upper*.99))
        c, r = blocks(q)
        beta = np.linalg.lstsq(np.vstack((c, r)), np.r_[y_curve, y_rate], rcond=None)[0]
        fits.append(run(np.r_[q, beta], scale))
    best = min([fit for fit in fits if fit.x[0] < fit.x[1]] or fits,
               key=lambda fit: fit.cost)
    for _ in range(2):
        scale = np.array([max(1.4826*np.median(np.abs(v-np.median(v))), 1e-6)
                          for v in residual(best.x)])
        updated = run(best.x, scale)
        if best.x[0] < best.x[1] and updated.x[0] >= updated.x[1]:
            break
        best = updated
    return np.exp(best.x[:2]), best.x[2:]


''' + text[b:]
text = text.replace("train['instant_rate_obs'], dt, time)", "train['instant_rate_obs'], dt, time,\n                      np.isfinite(train['coupon_features_raw']).all(axis=1))")
p.write_text(text)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch01.py
```
- **Container Exit Code**: `0` (`1771 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
ull-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
{"schema_valid": true, "errors": []}
estimated_taus fit [0.56788516 4.02941561] max abs error 0.08358439417230024
hereditary_weights fit [ 0.6778154   0.54432897  0.13845738  0.15680113 -0.12803643  0.1162538
 -0.08855162] max abs error 0.0056710316976325
regime_mode_weights fit [0.28139012 0.239199  ] max abs error 0.001390116658031737
lab 0 active False n 7 curve/rate RMSE 0.025468147505180824 0.01840974568421881
lab 0 active True n 10 curve/rate RMSE 0.026934147374077202 0.020404488045613636
lab 1 active False n 4 curve/rate RMSE 0.01250266454963075 0.009656868183435429
lab 1 active True n 12 curve/rate RMSE 0.05226381388441632 0.04019104561427029
lab 2 active False n 9 curve/rate RMSE 0.0037498023803514855 0.004293248710488049
lab 2 active True n 12 curve/rate RMSE 0.05905031784690701 0.046050351787081896
lab 3 active False n 4 curve/rate RMSE 0.014403639728875251 0.01094632340126886
lab 3 active True n 14 curve/rate RMSE 0.01926965953691681 0.014197796106651207
missing False n 65 curve/rate RMSE 0.010899934136736887 0.008236551391282144
missing True n 7 curve/rate RMSE 0.11050133760493613 0.08518714899563576
train curve missing False abs residual q50/q90/q99/max [1.19871710e-02 3.92739989e-02 3.61067744e+00 7.61120149e+01]
train curve missing True abs residual q50/q90/q99/max [ 0.08395646  0.3335866   0.86504315 20.84438461]
train rate missing False abs residual q50/q90/q99/max [0.01259985 0.03504021 0.05077986 0.06221349]
train rate missing True abs residual q50/q90/q99/max [0.07558426 0.19397902 0.29804353 0.30076328]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe6993b8c928afca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fe6993b8 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`31653 ms` LLM)
- **Rationale**: Residuals as large as 76 still exert bounded but nonzero influence under soft-L1. Compare a Cauchy refinement using leave-one-lab-out fits, with scales estimated inside each fold. Keep the verified artifact unchanged unless the challenger improves recovery without worsening held-lab predictions.
- **Alternatives Weighed**: `Replace the specified cell-balanced imputation` (rejected: Most remaining error is concentrated in incomplete coupons, but the task prescribes how to fill missing features.)

```bash
set -e
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -35
cat > /tmp/compare_batch01.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'; s=importlib.util.spec_from_file_location('m',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; x,xt=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']; args=cal,dt,t,d['crit_stress'],d['shear_banding_index']; build=m._design(tr,x,*args); test=m._design(te,xt,*args); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; lab=tr['lab_id']; good=np.isfinite(tr['coupon_features_raw']).all(1)
rm=lambda v: np.sqrt(np.mean(v*v))
for kind in ('baseline','cauchy'):
 def fit(use):
  b=lambda q: tuple(v[use] for v in build(q)); tau,beta=m._fit(b,yc[use],yr[use],dt,t,good[use]); q=np.r_[np.log(tau),beta]
  if kind=='baseline': return q
  sub=use & good
  def residual(q):
   c,r=build(q[:2]); return (c[sub]@q[2:]-yc[sub]).ravel(), r[sub]@q[2:]-yr[sub]
  scale=[max(1.4826*np.median(abs(e-np.median(e))),1e-6) for e in residual(q)]
  new=least_squares(lambda v: np.concatenate([e/a for e,a in zip(residual(v),scale)]),q,loss='cauchy',x_scale='jac').x
  return new if new[0]<new[1] else q
 print(kind)
 for j in range(4):
  q=fit(lab!=j); c,r=build(q[:2]); u=(lab==j)&good; ec=c[u]@q[2:]-yc[u]; er=r[u]@q[2:]-yr[u]; print('held lab',j,'curve median abs / rate RMSE:',np.median(abs(ec)),rm(er),'tau',np.exp(q[:2]))
 q=fit(np.ones(len(lab),bool)); c,r=test(q[:2]); ec=c@q[2:]+te['strain_0'][:,None]-g['te_clean']; er=r@q[2:]-g['te_true_rate0']; missing=~np.isfinite(te['coupon_features_raw']).all(1); print('tau',np.exp(q[:2]),'max coefficient error',np.max(abs(q[2:]-np.r_[g['theta'],g['vp_theta']])),'complete/incomplete strain and rate RMSE',[(rm(ec[u]),rm(er[u])) for u in (~missing,missing)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/compare_batch01.py
```
- **Container Exit Code**: `0` (`4059 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
baseline
held lab 0 curve median abs / rate RMSE: 0.010883777373800751 0.019026570109890957 tau [0.56792008 4.03986225]
held lab 1 curve median abs / rate RMSE: 0.012272457063896569 0.01666197701169974 tau [0.56763216 4.02748864]
held lab 2 curve median abs / rate RMSE: 0.01173656583234231 0.022501405344474153 tau [0.5676652  4.01648446]
held lab 3 curve median abs / rate RMSE: 0.01812438084424846 0.025986157775946202 tau [0.56848462 4.04015495]
tau [0.56788516 4.02941561] max coefficient error 0.0056710316976325 complete/incomplete strain and rate RMSE [(0.010899934136736887, 0.008236551391282144), (0.11050133760493613, 0.08518714899563576)]
cauchy
held lab 0 curve median abs / rate RMSE: 0.011001518231502638 0.01940097503740285 tau [0.56740568 4.06073754]
held lab 1 curve median abs / rate RMSE: 0.012396308727933791 0.016269484956438807 tau [0.56738545 4.0454668 ]
held lab 2 curve median abs / rate RMSE: 0.011241214545570255 0.021899920277244078 tau [0.56688919 4.01893941]
held lab 3 curve median abs / rate RMSE: 0.017999291039277998 0.025768861883332293 tau [0.56784587 4.04557203]
tau [0.56733753 4.03884778] max coefficient error 0.004361358493505274 complete/incomplete strain and rate RMSE [(0.010899668990330753, 0.008312785552853733), (0.11115091806177821, 0.08536404556067373)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe6993b8c928afca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fe6993b8 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`26000 ms` LLM)
- **Rationale**: The clean public sweep has not exercised the documented sparse and EMI paths. Inject a qualifying spike at the ends and center of actual channel sweeps; if the spike is absorbed into the fitted line, or a recoverable in-batch fit is replaced by stale archive values, revise calibration before submission.
- **Alternatives Weighed**: `Promote the Cauchy refinement` (rejected: Its small improvement in coefficient recovery and worst held-lab error did not improve either clean-target metric on incomplete coupons.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
p='/holdout_task/'; s=importlib.util.spec_from_file_location('m',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; h=json.load(open(m._ARCHIVE)); print('sweeps',len(h),[[(j,sum(np.asarray(k['lab_id'])==j)) for j in range(4)] for k in h]); truth=m._calibrate(c)
for j in range(4):
 idx=np.flatnonzero(c['lab_id']==j); idx=idx[np.argsort(c['cmd_stress'][idx])]; other=np.flatnonzero(c['lab_id']!=j); print('channel',j,'points',len(idx),'current',truth[j],'prefix',m._archive_line(h,j));
 for n in (1,3,5,len(idx)):
  sel=idx[np.linspace(0,len(idx)-1,n,dtype=int)]; chosen=np.r_[other,sel]; b={k:np.array(v[chosen],copy=True) for k,v in c.items()};
  if n==1:
   print(' one point fallback',m._calibrate(b)[j]); continue
  for locus in (0,n//2,n-1):
   q={k:v.copy() for k,v in b.items()}; q['true_stress'][len(other)+locus]+=4.; a=m._calibrate(q)[j]; print(' spike n/location',n,locus,'max calibrated stress deviation',round(float(np.max(abs(np.polyval(a-truth[j],c['cmd_stress'][idx])))),5));
PY
sed -n '/^def _align/,/^def _line/p' /holdout_task/alpha_m97_c03_batch_01_solver.py
```
- **Container Exit Code**: `0` (`616 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/location 10 5 max calibrated stress deviation 1e-05
 spike n/location 10 9 max calibrated stress deviation 5e-05
channel 3 points 10 current [ 0.76000147 -0.10998819] prefix [ 1.00004220e+00 -8.39804234e-05]
 one point fallback [ 1.00004220e+00 -8.39804234e-05]
 spike n/location 3 0 max calibrated stress deviation 0.63754
 spike n/location 3 1 max calibrated stress deviation 0.63754
 spike n/location 3 2 max calibrated stress deviation 3.8801
 spike n/location 5 0 max calibrated stress deviation 6e-05
 spike n/location 5 2 max calibrated stress deviation 8e-05
 spike n/location 5 4 max calibrated stress deviation 0.00039
 spike n/location 10 0 max calibrated stress deviation 2e-05
 spike n/location 10 5 max calibrated stress deviation 0.0
 spike n/location 10 9 max calibrated stress deviation 2e-05
def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])



def _line(x, y):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe6993b8c928afca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fe6993b8 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`42692 ms` LLM)
- **Rationale**: Test both signs of a qualifying EMI spike at every sampled position, as well as the one-point fallback and clean in-batch fits. If any supported sweep still misses, revise the calibration before submission; public slice scores check that the change leaves clean-sweep predictions intact.
- **Alternatives Weighed**: `Always use the archived line on short sweeps` (rejected: The archived gains are near unity, while the current clean gains range from about 0.76 to 1.29. Clean in-batch points are more informative.)
- **ATMS NOGOOD DECLARED**: Refuted `A median pairwise slope exposes a qualifying EMI spike even in three-point calibration sweeps.` | Witness: `A +4 stress spike at an endpoint of a three-point sweep drawn from the public calibration rows produced up to 3.8801 stress units of reconstruction error.` | Blanket Diagnosis: `With three samples, a contaminated endpoint affects two of the three pairwise slopes and can mask its own residual.` | Sublated Pivot: `Score affine lines by clean-point consensus; where short spiked sweeps admit competing lines, use the archived line to distinguish them.`

```bash
set -e
out=/holdout_task/alpha_m97_c03_batch_01_solver.py
python3 - "$out" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text()
a = s.index('def _calibrate('); b = s.index('def _design(', a)
s = s[:a] + '''def _calibrate(cal):
    result = np.empty((4, 2))
    history = None

    def prior(lab):
        nonlocal history
        if history is None:
            try:
                with open(_ARCHIVE) as handle:
                    history = json.load(handle)
            except FileNotFoundError:
                history = []
        return _archive_line(history, lab)

    for lab in range(4):
        use = cal['lab_id'] == lab
        x, y = cal['cmd_stress'][use], cal['true_stress'][use]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2:
            result[lab] = prior(lab)
            continue
        # A spike at the end of a short sweep can rotate a median-slope fit.
        # Score pair lines by tight affine consensus before testing for EMI.
        pairs = []
        peak = 0.
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            for u, v in zip(i, j):
                if x[v] == x[u]:
                    continue
                gain = (y[v]-y[u])/(x[v]-x[u])
                prediction = y[u] + gain*(x-x[u])
                error = np.abs(prediction-y)
                # The load-cell sweeps are resolved below 0.001 stress units.
                inliers = error <= .003
                count = np.count_nonzero(inliers)
                if count > peak:
                    peak, pairs = count, []
                if count == peak:
                    pairs.append((inliers, np.max(error)))
        if pairs and peak < len(x) and max(error for _, error in pairs) > 3.:
            archived = prior(lab)
            archive_prediction = np.polyval(archived, x)
            candidates = [_line(x[mask], y[mask]) for mask, error in pairs if error > 3.]
            result[lab] = min(candidates, key=lambda ab: np.mean((np.polyval(ab, x)-archive_prediction)**2))
        else:
            result[lab] = _line(x, y)
    return result


''' + s[b:]; p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch01.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
from unittest.mock import patch
p='/holdout_task/'; s=importlib.util.spec_from_file_location('m',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; h=json.load(open(m._ARCHIVE)); clean=m._calibrate(c)
for n in (1,2,3,5,10):
 errs=[]
 for lab in range(4):
  idx=np.flatnonzero(c['lab_id']==lab); idx=idx[np.argsort(c['cmd_stress'][idx])]; other=np.flatnonzero(c['lab_id']!=lab); sel=idx[np.linspace(0,len(idx)-1,n,dtype=int)]; rows=np.r_[other,sel]; sub={k:np.array(v[rows],copy=True) for k,v in c.items()}
  if n==1:
   assert np.allclose(m._calibrate(sub)[lab],m._archive_line(h,lab)); continue
  with patch.object(m,'_archive_line',side_effect=AssertionError('unexpected archive fallback')):
   errs.append(np.max(abs(np.polyval(m._calibrate(sub)[lab]-clean[lab],c['cmd_stress'][idx]))))
  if n>=3:
   for locus in range(n):
    for emi in (-4., 4.):
     z={k:v.copy() for k,v in sub.items()}; z['true_stress'][len(other)+locus]+=emi
     errs.append(np.max(abs(np.polyval(m._calibrate(z)[lab]-clean[lab],c['cmd_stress'][idx]))))
 if errs:
  print('sweep size',n,'worst reconstructed stress deviation',max(errs)); assert max(errs)<.002
PY
```
- **Container Exit Code**: `0` (`2552 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus fit [0.56788516 4.02941561] max abs error 0.08358439417230024
hereditary_weights fit [ 0.6778154   0.54432897  0.13845738  0.15680113 -0.12803643  0.1162538
 -0.08855162] max abs error 0.0056710316976325
regime_mode_weights fit [0.28139012 0.239199  ] max abs error 0.001390116658031737
lab 0 active False n 7 curve/rate RMSE 0.025468147505180824 0.01840974568421881
lab 0 active True n 10 curve/rate RMSE 0.026934147374077202 0.020404488045613636
lab 1 active False n 4 curve/rate RMSE 0.01250266454963075 0.009656868183435429
lab 1 active True n 12 curve/rate RMSE 0.05226381388441632 0.04019104561427029
lab 2 active False n 9 curve/rate RMSE 0.0037498023803514855 0.004293248710488049
lab 2 active True n 12 curve/rate RMSE 0.05905031784690701 0.046050351787081896
lab 3 active False n 4 curve/rate RMSE 0.014403639728875251 0.01094632340126886
lab 3 active True n 14 curve/rate RMSE 0.01926965953691681 0.014197796106651207
missing False n 65 curve/rate RMSE 0.010899934136736887 0.008236551391282144
missing True n 7 curve/rate RMSE 0.11050133760493613 0.08518714899563576
train curve missing False abs residual q50/q90/q99/max [1.19871710e-02 3.92739989e-02 3.61067744e+00 7.61120149e+01]
train curve missing True abs residual q50/q90/q99/max [ 0.08395646  0.3335866   0.86504315 20.84438461]
train rate missing False abs residual q50/q90/q99/max [0.01259985 0.03504021 0.05077986 0.06221349]
train rate missing True abs residual q50/q90/q99/max [0.07558426 0.19397902 0.29804353 0.30076328]
sweep size 2 worst reconstructed stress deviation 0.00015420412099802179
sweep size 3 worst reconstructed stress deviation 0.0002370689128114028
sweep size 5 worst reconstructed stress deviation 0.000386977544829973
sweep size 10 worst reconstructed stress deviation 0.00010379325277916484

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac54a030e1be6023 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_ac54a030 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`24724 ms` LLM)
- **Rationale**: The channel-level stress test passed. This end-to-end check tests whether simultaneous three-point EMI sweeps amplify small calibration errors through the damage gate or the fit. If any returned field moves materially, trace that sensitivity before submitting; otherwise retain the verified artifact.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; c=d['calibration_load_cells']; solve=m.solve_alpha_m97_c03_batch_01
baseline=solve(d); ref=m._calibrate(c); keys=list(baseline); max_diff=np.zeros(len(keys)); stress_error=0.
rows=[]
for lab in range(4):
 j=np.flatnonzero(c['lab_id']==lab); j=j[np.argsort(c['cmd_stress'][j])]; rows.extend(j[[0,len(j)//2,-1]])
for location in range(3):
 for sign in (-1,1):
  sweep={k:v[rows].copy() for k,v in c.items()}; sweep['true_stress'][location::3]+=sign*4.
  case=dict(d,calibration_load_cells=sweep); pred=solve(case); estimate=m._calibrate(sweep)
  for lab in range(4):
   x=c['cmd_stress'][c['lab_id']==lab]; stress_error=max(stress_error,np.max(abs(np.polyval(estimate[lab]-ref[lab],x))))
  max_diff=np.maximum(max_diff,[np.max(abs(pred[k]-baseline[k])) for k in keys])
print('simultaneous 3-point EMI sweeps, worst stress deviation:',stress_error)
print('worst absolute changes by output:',dict(zip(keys,max_diff)))
assert stress_error<.002 and np.all(max_diff<np.array([.02,.005,.005,.005,.005]))
print('clean public strain MAE:',np.mean(abs(baseline['predicted_strain']-g['te_clean'])))
PY
```
- **Container Exit Code**: `0` (`4512 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
simultaneous 3-point EMI sweeps, worst stress deviation: 0.00048083410048437974
worst absolute changes by output: {'estimated_taus': 0.00013753461821419677, 'hereditary_weights': 4.3987394385380796e-05, 'regime_mode_weights': 3.3587542005886206e-05, 'predicted_strain': 0.0002904740872105016, 'initial_rate_pred': 0.00036081338021087817}
clean public strain MAE: 0.014226021376067849

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac54a030e1be6023 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_ac54a030 (directory).
```

### Turn 12 — `submit_completed_task` (`14752 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification. Simultaneous three-point EMI stress tests changed each of the five outputs by less than 0.00037 in absolute terms, and both structural findings are resolved.
- **Alternatives Weighed**: `Use the Cauchy refinement` (rejected: It slightly improved coefficient recovery but worsened both prediction metrics on the highest-error slice.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:8194'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema check passed, and clean-test strain MAE was 0.01423. Coupons with missing features had the largest errors: strain RMSE 0.1105 and initial-rate RMSE 0.0852.`

### Turn 13 — `execute_in_remote_docker` (`34356 ms` LLM)
- **Rationale**: The independent stepwise recurrence would expose a sign, coordinate, time-basis, or missing-term error hidden by the matrix formulation. Signed leave-one-lab-out residuals would expose a lab offset hidden by pooled error. If either check reveals a material discrepancy, investigate that mechanism; otherwise retain the verified solver.
- **Alternatives Weighed**: `Test relay modes, HJB operators, or within-batch classification accuracy` (rejected: This task has continuous rheology outcomes and no relay, control PDE, or class labels. The applicable checks are exact recurrence agreement and conditional residual bias across labs.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; o=m.solve_alpha_m97_c03_batch_01(d)
x,xt=m._align(tr,te); dt,t=d['past_dt'],d['eval_times']; cal=m._calibrate(d['calibration_load_cells']); tau=o['estimated_taus']; hw=o['hereditary_weights']; rw=o['regime_mode_weights']; gain,bias=cal[te['lab_id']].T
past=te['past_stress_cmd']*gain[:,None]+bias[:,None]; hold=te['stress_hold_cmd']*gain+bias
z=np.zeros((len(hold),2)); damage=np.zeros(len(hold))
for stress,step in zip(past.T,dt):
    decay=np.exp(-step/tau)
    z=decay*z+(1-decay)*stress[:,None]
    damage+=np.maximum(np.abs(stress)-d['crit_stress'],0)**2*step
act=np.tanh(damage); amp=hw[:2]*(hold[:,None]-z); amp[:,0]+=rw[0]*act*hold
slope=hw[2]+xt@hw[3:]+rw[1]*act*z[:,0]
assert d['shear_banding_index']<=.50
curve=te['strain_0'][:,None]+amp@(1-np.exp(-t[None,:]/tau[:,None]))+slope[:,None]*np.log1p(t)
rate=(amp/tau).sum(axis=1)+slope
for label,mask in [('zero damage',damage==0),('positive damage',damage>0)]:
    assert np.any(mask)
    handoff=max(np.max(abs(curve[mask]-o['predicted_strain'][mask])),np.max(abs(rate[mask]-o['initial_rate_pred'][mask])))
    print(label,'count',mask.sum(),'maximum independent recurrence discrepancy',handoff); assert handoff<1e-10
lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; build=m._design(tr,x,cal,dt,t,d['crit_stress'],d['shear_banding_index'])
for j in np.unique(lab):
    fit=lab!=j; use=(lab==j)&complete
    q,b=m._fit(lambda v:tuple(a[fit] for a in build(v)),yc[fit],yr[fit],dt,t,complete[fit]); c,r=build(np.log(q)); ec=(c[use]@b-yc[use]); er=r[use]@b-yr[use]
    print('held lab',j,'curve/rate signed median',np.median(ec),np.median(er),'rate RMSE',np.sqrt(np.mean(er**2)))
print('pooled public strain MAE',np.mean(abs(curve-g['te_clean'])),'relative tau errors',abs(tau/g['taus']-1))
PY
```
- **Container Exit Code**: `0` (`2716 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
zero damage count 24 maximum independent recurrence discrepancy 8.881784197001252e-16
positive damage count 48 maximum independent recurrence discrepancy 1.3322676295501878e-15
held lab 0 curve/rate signed median -0.0011047884072930003 -0.0008086357481773554 rate RMSE 0.019026570109890957
held lab 1 curve/rate signed median -0.0034148902388822233 -0.002179314539696664 rate RMSE 0.01666197701169974
held lab 2 curve/rate signed median 0.0003962652561642771 -0.010062435692231197 rate RMSE 0.022501405344474153
held lab 3 curve/rate signed median 0.005188700836797677 -0.0047920089724889925 rate RMSE 0.025986157775946202
pooled public strain MAE 0.014226021376067852 relative tau errors [0.00020218 0.020322  ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac54a030e1be6023 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_ac54a030 (directory).
```

### Turn 14 — `submit_completed_task` (`13955 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation. An independent recurrence matched both damage states within 1.4e-15, simultaneous EMI shifted each output by less than 0.00037, and worst held-out-lab rate RMSE was 0.0260. Both structural findings are resolved.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema passed, and test strain MAE was 0.01423. Coupons with missing features were the hardest slice: strain RMSE was 0.1105 and initial-rate RMSE was 0.0852.`

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
