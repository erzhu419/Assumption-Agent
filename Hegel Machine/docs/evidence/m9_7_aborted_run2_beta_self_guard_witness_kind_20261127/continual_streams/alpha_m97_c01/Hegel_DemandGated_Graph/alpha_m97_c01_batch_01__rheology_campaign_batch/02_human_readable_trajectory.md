# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `422,821` (`407,406` / `108,032` / `15,415`) |
| **Total Episode Wall Time (s)** | `496.96s` |

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

### Turn 1 — `execute_in_remote_docker` (`17915 ms` LLM)
- **Rationale**: Inspect the public schema and archived implementation to detect mismatched units, panel conventions, or phase formulas before fitting; any mismatch determines which archived components can be reused.

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation'; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb') as f:d=pickle.load(f)
for k,v in d.items():
 print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None))
 if isinstance(v,dict):
  for a,b in v.items(): print(' ',a,type(b),getattr(b,'shape',None),str(b)[:160] if np.size(b)<25 else str(np.asarray(b).ravel()[:5]))
 elif isinstance(v,(list,tuple)):print(' ',len(v),repr(v[:2])[:500])
 elif isinstance(v,np.ndarray):print(' ',repr(v.ravel()[:8]))
 else: print(' ',repr(v)[:400])
PY
```
- **Container Exit Code**: `1` (`402 ms`, timed_out=`False`)
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
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18419 ms` LLM)
- **Rationale**: Read the corrected dataset path within the public list and inspect the archived alignment code. The calibration residuals determine whether this batch needs history fallback; the panel and observation fields determine the alignment and fitting design.

```bash
sed -n '1,230p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill; python3 - <<'PY'
import pickle,numpy as np,json
p=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))
print('items',len(p))
for item in p:
 print('item keys',list(item)); d=item['dataset']
 for k,v in d.items():
  print(k, {a:(np.shape(b),str(np.asarray(b).dtype),str(np.asarray(b).ravel()[:3])) for a,b in v.items()} if isinstance(v,dict) else str(v)[:240])
 c=d['calibration_load_cells']
 for i in range(4):
  m=c['lab_id']==i; x=c['cmd_stress'][m]; y=c['true_stress'][m]; ab=np.polyfit(x,y,1);print('cal',i,len(x),ab,np.max(abs(y-np.polyval(ab,x))))
for fn in ['archive_manifest.json','archive_calibration_history.json']:
 a=json.load(open('/tmp/hegel_continual_archive/'+fn));print(fn,str(a)[:2200])
PY
```
- **Container Exit Code**: `0` (`501 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
loat64', '[-0.94468  -0.248683 -0.856825]'), 'strain_curves_obs': ((180, 10), 'float64', '[-0.30682153 -0.00513339  0.11082238]'), 'instant_rate_obs': ((180,), 'float64', '[ 1.11099833 -0.66347914 -0.3031002 ]')}
test {'lab_id': ((72,), 'int64', '[0 1 2]'), 'stratum': ((72,), 'int64', '[0 0 1]'), 'coupon_features_raw': ((72, 4), 'float64', '[-1.34597117 -1.27212031  1.0461071 ]'), 'ref_coupons': ((72, 32), 'float64', '[ 0.07822747 -0.22232973 -1.10442901]'), 'past_stress_cmd': ((72, 18), 'float64', '[-1.0782928  -0.36112687 -1.45929886]'), 'stress_hold_cmd': ((72,), 'float64', '[-0.29222313  2.18481558 -1.35448264]'), 'strain_0': ((72,), 'float64', '[ 0.513179  0.68354  -0.21838 ]'), 'aux_chamber_telemetry': ((72, 3), 'float64', '[ 1.223187 -0.042873 -0.05421 ]')}
cal 0 10 [1.33999935 0.14999378] 0.00015647060919077305
cal 1 10 [ 0.77000406 -0.16999413] 0.00018822176288391113
cal 2 10 [ 1.21002713 -0.12000366] 0.0001037212625434325
cal 3 10 [0.83000992 0.10997266] 8.725097943051985e-05
archive_manifest.json [{'episode_short_name': 'alpha_m97_c01_s00', 'task_id': 'alpha_m97_c01_s00__rheology_campaign_baseline', 'entrypoint': 'solve_alpha_m97_c01_s00', 'output_schema_keys': ['estimated_taus', 'hereditary_weights', 'initial_rate_pred', 'predicted_strain', 'regime_mode_weights'], 'input_coordinate_contract': 'raw_measured', 'solver_path': '/tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py'}]
archive_calibration_history.json [{'episode_short_name': 'alpha_m97_c01_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.660283, -0.418768, 1.112723, -1.62604, -0.501169, -1.672067, 0.863389, 1.008755, 1.968856, 1.530083, 0.979468, 0.449112, 1.469886, -0.332491, -1.532413, 0.660575, -1.672999, -1.196407, 0.934354, 0.731875, 1.670914, -2.075727, 1.021513, -1.730233, -1.965428, 0.955855, 0.129182, -0.328527, -1.550117, -1.993809, 2.131528, -2.023748, 0.667305, -0.968757, -0.649251, -0.022422, -1.62562, -0.902597, 1.990079, -0.41266], 'true_stress': [1.660305, -0.418669, 1.112724, -1.626001, -0.501125, -1.671976, 0.863198, 1.008634, 1.968816, 1.530248, 0.979291, 0.449215, 1.470001, -0.332446, -1.532394, 0.660518, -1.672977, -1.196416, 0.934345, 0.731869, 1.670821, -2.075755, 1.021515, -1.730423, -1.965592, 0.95586, 0.129076, -0.328478, -1.550116, -1.993815, 2.131554, -2.023643, 0.667296, -0.968674, -0.649069, -0.022271, -1.625842, -0.902458, 1.990138, -0.412675]}]
```

### Turn 3 — `execute_in_remote_docker` (`19790 ms` LLM)
- **Rationale**: Confirm the archived panel convention before reusing it. Count activated samples per lab to detect whether any leave-one-lab-out fold would lack support for estimating the regime weights.
- **Alternatives Weighed**: `Pool historical and current calibration sweeps` (rejected: All four current channels have ten clean points, and historical gains differ materially.)

```bash
grep -n '^def\|^class' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; sed -n '55,145p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb') as f: items=pickle.load(f)
print('public datasets:',len(items))
for item in items:
 d=item['dataset']; c=d['calibration_load_cells']; a=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==j],c['true_stress'][c['lab_id']==j],1) for j in range(4)])
 for k in ('train','test'):
  s=d[k]; lab=s['lab_id']; p=s['past_stress_cmd']*a[lab,0,None]+a[lab,1,None]; D=np.maximum(0,abs(p)-d['crit_stress'])**2@d['past_dt']; print(k,'max D',D.max(),'active by lab',[int(sum((D>0)&(lab==j))) for j in range(4)],'missing features',np.isnan(s['coupon_features_raw']).sum(0),'missing refs',np.isnan(s['ref_coupons']).sum())
 print('strain/rate SD:',np.std(d['train']['strain_curves_obs'],axis=0),np.std(d['train']['instant_rate_obs']))
PY
```
- **Container Exit Code**: `0` (`465 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
times, past, hold, features):
        self.dt = np.asarray(dt)
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.t = np.asarray(times)
        self.past = past
        self.hold = hold
        self.x = np.column_stack((np.ones(len(hold)), features))

    def design(self, log_taus):
        tau = np.exp(log_taus)
        u, a, t = (z[:, None] / tau for z in (self.dt, self.after, self.t))
        ed, ea, et = np.exp(-u), np.exp(-a), np.exp(-t)
        step, response = -np.expm1(-u), -np.expm1(-t)
        gap = self.hold[:, None] - self.past @ (ea * step)
        dgap = -self.past @ (ea * (a * step - u * ed))
        n, k = len(gap), len(self.t)
        design = np.empty((n, k + 1, 7))
        deriv = np.empty((n, k + 1, 2))
        design[:, :-1, :2] = gap[:, None, :] * response
        design[:, -1, :2] = gap / tau
        design[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        design[:, -1, 2:] = self.x
        deriv[:, :-1] = dgap[:, None, :] * response - gap[:, None, :] * (t * et)
        deriv[:, -1] = (dgap - gap) / tau
        return design, deriv


def _mad(values):
    return max(1.4826 * np.median(np.abs(values - np.median(values))), 1e-8)


def _fit(model, y):
    cache = [None, None, None]

    def evaluate(theta, scale):
        if cache[0] is None or not np.array_equal(theta[:2], cache[0]):
            cache[:] = [theta[:2].copy(), *model.design(theta[:2])]
        design, deriv = cache[1:]
        return (design @ theta[2:] - y) / scale, design, deriv

    def fun(theta, scale):
        return evaluate(theta, scale)[0].ravel()

    def jac(theta, scale):
        _, design, deriv = evaluate(theta, scale)
        j = np.concatenate((deriv * theta[2:4], design), axis=2)
        return (j / np.asarray(scale)[..., None]).reshape(-1, 9)

    t_min, t_max = model.t[model.t > 0].min(), model.t.max()
    low, high = np.log(t_min / 25), np.log(t_max * 20)
    bounds = (np.r_[low, low, np.full(7, -np.inf)],
              np.r_[high, high, np.full(7, np.inf)])
    seeds = np.log(t_max * np.array([[.07, .5], [.025, .3], [.14, 1.5], [.3, 2.5]]))
public datasets: 1
train max D 5.285981812113418 active by lab [19, 36, 28, 37] missing features [4 4 5 6] missing refs 238
test max D 4.814115326079685 active by lab [13, 14, 13, 8] missing features [1 1 1 1] missing refs 96
strain/rate SD: [0.79479227 0.99881693 1.18776265 1.3152022  2.14584314 2.47918683
 1.53971191 1.60959738 1.63975135 1.7163256 ] 1.9164809921152814
```

### Turn 4 — `execute_in_remote_docker` (`79940 ms` LLM)
- **Rationale**: Implement the exact response and initial-rate equations, then check the public schema and analytic Jacobian. A derivative discrepancy would invalidate the optimizer; the observed residual distribution will guide the held-out comparisons.
- **Alternatives Weighed**: `Reuse the archived linear terminal basis` (rejected: The prescribed activated regime requires log(1+t), with additional damage-modulated memory terms.); `Use plain least squares for the observation fit` (rejected: The unusually high variance at two adjacent horizons warrants a robust pilot fit and per-horizon residual diagnostics.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_01_solver.py <<'PY'
"""Calibrated two-branch creep with a damage-gated crystallization plateau."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    labs = np.asarray(rows['lab_id'])
    cmd, true = (np.asarray(rows[k], float) for k in ('cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        m = (labs == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[m], true[m]
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            keep = x[i] != x[j]
            i, j = i[keep], j[keep]
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            errs = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
            if len(errs):
                score = np.median(errs, axis=1)
                best = np.argmin(score)
                core = errs[best] <= max(8 * score[best], 1e-10)
                a, b = _line(x[core], y[core])
                clean = np.abs(a * x + b - y) <= 3.0
                x, y = x[clean], y[clean]
        if len(x) < 2:
            if history is None:
                path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
                with path.open() as handle:
                    history = json.load(handle)
            for sweep in reversed(history):
                m = np.asarray(sweep['lab_id']) == lab
                hx, hy = (np.asarray(sweep[k])[m] for k in ('cmd_stress', 'true_stress'))
                if len(hx) >= 2:
                    x, y = hx, hy
                    break
        result[lab] = _line(x, y)
    return result


def _inputs(dataset):
    def stack(key):
        return np.concatenate([dataset[part][key] for part in ('train', 'test')])

    lab, stratum = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    x = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel
        center = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        x[m] = (raw[m] - center) / scale
    means = []
    for channel, group in np.unique(np.column_stack((lab, stratum)), axis=0):
        v = x[(lab == channel) & (stratum == group)]
        count = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, x, stack('strain_0')


class _Model:
    def __init__(self, dataset, past, hold, x):
        self.dt = np.asarray(dataset['past_dt'])
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.]
        self.t = np.asarray(dataset['eval_times'])
        self.past, self.hold = past, hold
        self.x = np.column_stack((np.ones(len(x)), x))
        damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.) ** 2 @ self.dt
        self.active = (dataset['shear_banding_index'] <= .50) & (damage > 0)
        self.act = np.where(self.active, np.tanh(damage), 0.)
        phi = np.where(self.active[:, None], np.log1p(self.t), self.t)
        self.terminal = np.concatenate((phi, np.ones((len(x), 1))), axis=1)
        self.drift_basis = self.terminal[:, :, None] * self.x[:, None, :]

    def design(self, theta, plateau=True):
        tau = np.exp(theta)
        u, a, t = (v[:, None] / tau for v in (self.dt, self.after, self.t))
        step, decay = -np.expm1(-u), np.exp(-a)
        z = self.past @ (decay * step)
        dz = self.past @ (decay * (a * step - u * np.exp(-u)))
        gap = self.hold[:, None] - z
        h = np.vstack((-np.expm1(-t), 1 / tau))
        dh = np.vstack((-t * np.exp(-t), -1 / tau))
        d = np.zeros((len(z), len(h), 9))
        dp = np.zeros((len(z), len(h), 4))
        d[:, :, :2] = gap[:, None, :] * h
        dp[:, :, :2] = gap[:, None, :] * dh - dz[:, None, :] * h
        if plateau:
            d[:, :, 2:7] = self.drift_basis
            d[:, :, 7] = (self.act * self.hold)[:, None] * h[:, 0]
            d[:, :, 8] = (self.act * z[:, 0])[:, None] * self.terminal
            dp[:, :, 2] = (self.act * self.hold)[:, None] * dh[:, 0]
            dp[:, :, 3] = (self.act * dz[:, 0])[:, None] * self.terminal
        else:
            d[:, :, 2:7] = np.r_[self.t, 1][None, :, None] * self.x[:, None, :]
        return d, dp


def _fit(model, y, plateau=True):
    count = 9 if plateau and np.any(model.active) else 7
    cached = None
    d = dp = None
    scale = 1.

    def evaluate(theta):
        nonlocal cached, d, dp
        if cached is None or not np.array_equal(cached, theta[:2]):
            cached = theta[:2].copy()
            d, dp = model.design(cached, plateau)
            d = d[:, :, :count]
        return d

    def residual(theta):
        return ((evaluate(theta) @ theta[2:] - y) / scale).ravel()

    def jacobian(theta):
        design = evaluate(theta)
        j_tau = dp[:, :, :2] * theta[2:4]
        if count == 9:
            j_tau[:, :, 0] += dp[:, :, 2:] @ theta[9:11]
        return (np.concatenate((j_tau, design), axis=2) /
                np.asarray(scale)[..., None]).reshape(-1, count + 2)

    tmax = model.t.max()
    low, high = model.t[model.t > 0].min() / 25, 20 * tmax
    bounds = (np.r_[np.log([low, low]), np.full(count, -np.inf)],
              np.r_[np.log([high, high]), np.full(count, np.inf)])
    seeds = np.log(tmax * np.array([[.065, .42], [.025, .28], [.15, 1.1], [.3, 2.5], [.42, .065]]))
    best = None
    for seed in seeds:
        w0 = np.linalg.lstsq(evaluate(seed).reshape(-1, count), y.ravel(), rcond=None)[0]
        fit = least_squares(residual, np.r_[seed, w0], jac=jacobian, bounds=bounds,
                            loss='soft_l1', f_scale=.015, x_scale='jac', max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    err = (evaluate(best.x) @ best.x[2:] - y)
    center = np.median(err, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(err - center), axis=0), 1e-7)[None, :]
    fit = least_squares(residual, best.x, jac=jacobian, bounds=bounds,
                        loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    tau = np.exp(fit.x[:2])
    weights = np.zeros(9)
    weights[:count] = fit.x[2:]
    if count == 7 and tau[0] > tau[1]:
        tau = tau[::-1]
        weights[:2] = weights[1::-1]
    return tau, weights


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train = dataset['train']
    n = len(train['lab_id'])
    past, hold, x, strain0 = _inputs(dataset)
    target = np.column_stack((train['strain_curves_obs'] - strain0[:n, None],
                              train['instant_rate_obs']))
    tau, w = _fit(_Model(dataset, past[:n], hold[:n], x[:n]), target)
    design, _ = _Model(dataset, past[n:], hold[n:], x[n:]).design(np.log(tau))
    pred = design @ w
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': strain0[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; past,hold,x,s0=m._inputs(d); n=len(d['train']['lab_id']); y=np.column_stack((d['train']['strain_curves_obs']-s0[:n,None],d['train']['instant_rate_obs'])); model=m._Model(d,past[:n],hold[:n],x[:n]); t,w=m._fit(model,y); D,dp=model.design(np.log(t)); err=abs(D@w-y); print('tau',t,'weights',w);print('absolute residual percentiles by horizon and rate',np.quantile(err,[.5,.9,.95,.98,1],axis=0).round(5)); print('largest sample residuals',np.sort(err.max(axis=1))[-12:]);
for j in range(2):
 th=np.log(t); step=np.eye(2)[j]*1e-5; fd=((model.design(th+step)[0]-model.design(th-step)[0])@w)/(2e-5); analytic=dp[:,:,j]*w[j]+(dp[:,:,2:]@w[7:] if j==0 else 0); print('Jacobian max discrepancy',j,np.max(abs(fd-analytic)))
PY
```
- **Container Exit Code**: `0` (`863 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
tau [0.61533898 3.21908213] weights [ 0.67761741  0.51798142  0.10044314  0.12141781 -0.10611715  0.08977721
 -0.07047445  0.27849342  0.29639656]
absolute residual percentiles by horizon and rate [[1.610000e-02 2.444000e-02 4.036000e-02 6.154000e-02 7.488000e-02
  9.761000e-02 1.082400e-01 1.177300e-01 1.347800e-01 1.501300e-01
  5.052000e-02]
 [6.487000e-02 1.478200e-01 1.438000e-01 2.011700e-01 3.096400e-01
  4.309400e-01 4.385000e-01 6.319400e-01 5.963800e-01 7.384800e-01
  1.267500e-01]
 [4.050400e-01 2.993100e-01 2.007000e-01 3.162500e-01 4.317200e-01
  7.300800e-01 5.975500e-01 1.017380e+00 8.631200e-01 1.132020e+00
  1.512000e-01]
 [1.283900e+00 5.033700e-01 5.418700e-01 7.675700e-01 9.364500e-01
  4.286210e+00 9.415100e-01 1.631260e+00 1.415750e+00 1.679240e+00
  1.825900e-01]
 [7.034220e+00 2.907280e+00 1.173310e+00 1.359500e+00 2.311396e+01
  2.061962e+01 3.435740e+00 5.092660e+00 6.488530e+00 7.225890e+00
  2.723000e-01]]
largest sample residuals [ 3.4357367   3.92326822  4.78741438  4.81086009  5.092664    6.48853478
  7.03421998  7.22589372 10.49443418 15.74825084 20.61962066 23.11395882]
Jacobian max discrepancy 0 1.2143397398745037e-10
Jacobian max discrepancy 1 3.302169648833342e-11

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fae7a66b7d7b91ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fae7a66b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`30494 ms` LLM)
- **Rationale**: Leave one lab out and score activated and inactive rows separately for strain and rate. Compare the prescribed design with diagnostic alternatives to distinguish a misread plateau convention from observation outliers or lab-specific residual bias.

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS multi-output residual'; cat > /tmp/rheo_cv.py <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; past,hold,x,s0=m._inputs(d); labs=d['train']['lab_id']; n=len(labs); past,hold,x=past[:n],hold[:n],x[:n]; y=np.column_stack((d['train']['strain_curves_obs']-s0[:n,None],d['train']['instant_rate_obs'])); groups=d['train']['stratum']; damage=np.maximum(abs(past)-d['crit_stress'],0)**2@d['past_dt'];
def model(mask,mode):
 z=m._Model(d,past[mask],hold[mask],x[mask]);
 if mode in ('global_log','linear_drift'):
  z.terminal[:,:-1]=np.log1p(z.t) if mode=='global_log' else z.t
  z.drift_basis=z.terminal[:,:,None]*z.x[:,None,:]
  if mode=='linear_drift':z.terminal[:,:-1]=np.log1p(z.t)
 return z
for mode in ('linear','gated','global_log','linear_drift'):
 pred=np.empty_like(y); params=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=m._fit(model(tr,mode),y[tr],plateau=mode!='linear'); params.append(np.r_[tau,w]); pred[te]=model(te,mode).design(np.log(tau),plateau=mode!='linear')[0]@w
 r=pred-y; a=abs(r); print('\n',mode,'fold tau and rw',np.round(np.asarray(params)[:,[0,1,-2,-1]],5)); print('pooled MAE, med strain/rate',a[:,:-1].mean(),a[:,-1].mean(),np.median(a[:,:-1]),np.median(a[:,-1])); print('horizon median',np.round(np.median(a,axis=0),5));
 for lab in np.unique(labs):
  for active in (False,True):
   mask=(labs==lab)&((damage>0)==active); b=a[mask]; print('lab,active,n,median strain/rate,90p strain/rate, late bias',lab,active,len(b),*[round(v,5) for v in (np.median(b[:,:-1]),np.median(b[:,-1]),np.quantile(b[:,:-1],.9),np.quantile(b[:,-1],.9),np.median(r[mask,-2]))])
 if mode=='gated':
  print('gated median signed residual by lab and stratum:');
  for lab in np.unique(labs):
   for g in np.unique(groups):
    q=(labs==lab)&(groups==g); print(lab,g,q.sum(),np.median(r[q],axis=0).round(4))
  np.save('/tmp/rheo_gated_oof.npy',r)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_cv.py
```
- **Container Exit Code**: `0` (`1370 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
07
horizon median [0.01062 0.01375 0.01313 0.0107  0.01346 0.01394 0.0136  0.0162  0.01275
 0.01379 0.01459]
lab,active,n,median strain/rate,90p strain/rate, late bias 0 False 17 0.01289 0.00877 0.03552 0.02751 -0.00456
lab,active,n,median strain/rate,90p strain/rate, late bias 0 True 19 0.01724 0.01433 0.29288 0.17227 -0.00516
lab,active,n,median strain/rate,90p strain/rate, late bias 1 False 14 0.01653 0.01868 0.13987 0.09243 0.00283
lab,active,n,median strain/rate,90p strain/rate, late bias 1 True 36 0.01451 0.01957 0.17554 0.06153 -0.00015
lab,active,n,median strain/rate,90p strain/rate, late bias 2 False 16 0.01076 0.01229 0.03095 0.02959 -0.00435
lab,active,n,median strain/rate,90p strain/rate, late bias 2 True 28 0.01114 0.00627 0.28262 0.02839 -0.00255
lab,active,n,median strain/rate,90p strain/rate, late bias 3 False 13 0.01018 0.01512 0.03849 0.03895 -0.00499
lab,active,n,median strain/rate,90p strain/rate, late bias 3 True 37 0.01281 0.0168 0.06959 0.03426 0.00227

 linear_drift fold tau and rw [[0.62208 3.8799  0.28798 0.23869]
 [0.61438 3.87433 0.27946 0.24558]
 [0.61614 3.72304 0.2875  0.24556]
 [0.61813 3.88832 0.28069 0.24675]]
pooled MAE, med strain/rate 0.16278601187090253 0.1458009409145114 0.04446936805246149 0.1250509473121872
horizon median [0.02929 0.06115 0.06806 0.06305 0.05051 0.03889 0.02207 0.02031 0.05252
 0.07024 0.12505]
lab,active,n,median strain/rate,90p strain/rate, late bias 0 False 17 0.04027 0.11406 0.1622 0.2764 0.05366
lab,active,n,median strain/rate,90p strain/rate, late bias 0 True 19 0.04944 0.1303 0.32179 0.23817 0.00689
lab,active,n,median strain/rate,90p strain/rate, late bias 1 False 14 0.05311 0.11151 0.13988 0.17509 0.0589
lab,active,n,median strain/rate,90p strain/rate, late bias 1 True 36 0.04833 0.13728 0.22436 0.36044 0.04046
lab,active,n,median strain/rate,90p strain/rate, late bias 2 False 16 0.03537 0.10829 0.15939 0.2915 0.06089
lab,active,n,median strain/rate,90p strain/rate, late bias 2 True 28 0.04591 0.12617 0.32864 0.27759 0.06551
lab,active,n,median strain/rate,90p strain/rate, late bias 3 False 13 0.03851 0.11696 0.14202 0.34382 0.0462
lab,active,n,median strain/rate,90p strain/rate, late bias 3 True 37 0.04374 0.13349 0.13066 0.23188 0.05321

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fae7a66b7d7b91ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fae7a66b (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29726 ms` LLM)
- **Rationale**: Recover comparable out-of-fold slice metrics from the truncated log and determine whether residual tails come from missing coupon features or isolated curve observations. That distinction determines whether to revisit the panel convention or keep the existing robust fit.
- **Alternatives Weighed**: `Keep the shared drift linear and apply log(1+t) only to the damage correction` (rejected: Leave-one-lab-out pooled median rate error was 0.12505 versus 0.01459 with the phase-wide clock.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero specimen damage switches the terminal clock back to linear even in the low-shear plateau campaign.` | Witness: `With the same calibrated inputs, active-only logarithmic clock gave 0.1501 final-horizon median error in sample; a phase-wide logarithmic clock gave 0.01379 at that horizon even with leave-one-lab-out fitting, with late median bias no larger than 0.00516 across activation-by-lab slices.` | Blanket Diagnosis: `Specimen activation was incorrectly allowed to switch the shared drift basis as well as the damage-specific terms.` | Sublated Pivot: `Use a logarithmic clock throughout the low-shear campaign and retain zero damage-specific corrections when D_cryst is zero.`

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -45; cat /holdout_task/public_asset_index.json | head -125; OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_cv.py | awk '/^ (linear|gated|global_log|linear_drift)$/ {print; p=1} /pooled MAE|horizon median/ {print} /^lab,active/ && p {print} /gated median signed/ {p=0}'; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; past,hold,x,s0=m._inputs(d); tr=d['train']; n=len(tr['lab_id']); y=np.column_stack((tr['strain_curves_obs']-s0[:n,None],tr['instant_rate_obs'])); z=m._Model(d,past[:n],hold[:n],x[:n]); z.terminal[:,:-1]=np.log1p(z.t); z.drift_basis=z.terminal[:,:,None]*z.x[:,None,:]; tau,w=m._fit(z,y); r=z.design(np.log(tau))[0]@w-y; miss=np.isnan(tr['coupon_features_raw']).any(1); print('global_log taus and weights',tau,w)
for active in (False,True):
 for missing in (False,True):
  b=abs(r[(z.active==active)&(miss==missing)]); print('active/missing/n',active,missing,len(b),'median 90p strain',np.quantile(b[:,:-1],[.5,.9]),'median 90p rate',np.quantile(b[:,-1],[.5,.9]))
print('large strain residuals by horizon',np.sum(abs(r[:,:-1])>.25,axis=0));print('samples with multiple large strain errors',np.bincount(np.sum(abs(r[:,:-1])>.25,axis=1)))
PY
```
- **Container Exit Code**: `0` (`1752 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"type": "ndarray"
              },
              "stress_hold_cmd": {
                "dtype": "float64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "aux_chamber_telemetry": {
                "dtype": "float64",
                "shape": [
                  180,
                  3
                ],
                "type": "ndarray"
              },
              "coupon_features_raw": {
                "dtype": "float64",
                "shape": [
                  180,
                  4
                ],
                "type": "ndarray"
              },
pooled MAE, med strain/rate 0.27725385156183563 0.3206968988855535 0.15061305925147928 0.24435971685409608
horizon median [0.0593  0.11767 0.13773 0.14566 0.1615  0.17536 0.18315 0.18248 0.18411
pooled MAE, med strain/rate 0.2300759022169782 0.06518460365360686 0.07783877528350258 0.04929011519081252
horizon median [0.0179  0.03066 0.05223 0.06932 0.09398 0.11685 0.11569 0.13043 0.14138
pooled MAE, med strain/rate 0.13109791582109015 0.02443141270261902 0.012976445600033325 0.014588957533762707
horizon median [0.01062 0.01375 0.01313 0.0107  0.01346 0.01394 0.0136  0.0162  0.01275
pooled MAE, med strain/rate 0.16278601187090253 0.1458009409145114 0.04446936805246149 0.1250509473121872
horizon median [0.02929 0.06115 0.06806 0.06305 0.05051 0.03889 0.02207 0.02031 0.05252
global_log taus and weights [0.61278825 3.69022857] [ 0.67823678  0.55271244  0.13976353  0.15676009 -0.12694661  0.11757215
 -0.08865851  0.28111826  0.24146184]
active/missing/n False False 54 median 90p strain [0.01032134 0.02996165] median 90p rate [0.01216739 0.02865572]
active/missing/n False True 6 median 90p strain [0.09831297 0.17460558] median 90p rate [0.08674036 0.12781754]
active/missing/n True False 107 median 90p strain [0.01169457 0.03560784] median 90p rate [0.01308552 0.02780179]
active/missing/n True True 13 median 90p strain [0.11139004 0.4217625 ] median 90p rate [0.0862376  0.26691676]
large strain residuals by horizon [11 11  8 11 11 18 12 15 11 13]
samples with multiple large strain errors [102  53  21   0   1   0   0   2   1]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fae7a66b7d7b91ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fae7a66b (directory).
```

### Turn 7 — `execute_in_remote_docker` (`49830 ms` LLM)
- **Rationale**: Promote the clock that reduced held-out error and late bias. Verify the exact artifact against an independent memory recurrence on both sides of the shear cutoff, then check whether any held-out lab or stratum regresses on either output.
- **Alternatives Weighed**: `Change the required cell-balanced imputation` (rejected: Missing coupons show larger residuals, but the imputation convention is specified and no validated replacement is needed.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_01_solver.py')
s = p.read_text()
old = '        phi = np.where(self.active[:, None], np.log1p(self.t), self.t)'
new = '        # Shear selects the phase clock; specimen damage gates only the extra memory terms.\n        clock = np.log1p(self.t) if dataset[\'shear_banding_index\'] <= .50 else self.t\n        phi = np.broadcast_to(clock, (len(x), len(self.t)))'
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl', 'rb'))[0]['dataset']
out = m.solve_alpha_m97_c01_batch_01(d)
past, hold, x, s0 = m._inputs(d)
tr = d['train']; lab = tr['lab_id']; n = len(lab)
y = np.column_stack((tr['strain_curves_obs'] - s0[:n, None], tr['instant_rate_obs']))

def direct(tau, hw, rw, phase, p, h, v):
    z = np.zeros((len(h), 2)); dose = np.zeros(len(h))
    for j, dt in enumerate(d['past_dt']):
        e = np.exp(-dt / tau)
        z = e * z + (1 - e) * p[:, j, None]
        dose += np.maximum(0, abs(p[:, j]) - d['crit_stress'])**2 * dt
    act = np.tanh(dose) if phase <= .5 else np.zeros_like(dose)
    amp = (h[:, None] - z) * hw[:2]
    amp[:, 0] += rw[0] * act * h
    drift = hw[2] + v @ hw[3:] + rw[1] * act * z[:, 0]
    t = d['eval_times']; clock = np.log1p(t) if phase <= .5 else t
    curve = amp @ (1 - np.exp(-t[None, :] / tau[:, None])) + drift[:, None] * clock
    return curve, amp @ (1 / tau) + drift

tau, hw, rw = [out[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights')]
print('public shear, taus, weights:', d['shear_banding_index'], tau, hw, rw)
for phase in (.50, np.nextafter(.50, 1), d['shear_banding_index']):
    q = dict(d, shear_banding_index=phase)
    model = m._Model(q, past[n:], hold[n:], x[n:])
    expected, rate = direct(tau, hw, rw, phase, past[n:], hold[n:], x[n:])
    z = model.design(np.log(tau))[0] @ np.r_[hw, rw]
    np.testing.assert_allclose(z, np.column_stack((expected, rate)), rtol=1e-11, atol=1e-12)
    print('shear/damage gated rows:', phase, int(model.active.sum()), 'of', len(model.active))
expected, rate = direct(tau, hw, rw, d['shear_banding_index'], past[n:], hold[n:], x[n:])
np.testing.assert_allclose(out['predicted_strain'], expected + s0[n:, None], atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], rate, atol=1e-12)
active = m._Model(d, past[:n], hold[:n], x[:n]).active
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for name, groups in [('lab', lab), ('stratum', tr['stratum'])]:
    r = np.empty_like(y); params = []
    for g in np.unique(groups):
        a = groups != g; b = ~a
        t, w = m._fit(m._Model(d, past[:n][a], hold[:n][a], x[:n][a]), y[a])
        r[b] = m._Model(d, past[:n][b], hold[:n][b], x[:n][b]).design(np.log(t))[0] @ w - y[b]
        params.append(t)
    err = abs(r)
    slices = [(groups == g) & (active == a) for g in np.unique(groups) for a in (False, True)]
    worst = np.max([[np.median(err[f, :-1]), np.median(err[f, -1]), np.median(err[f, -2]), abs(np.median(r[f, -2]))] for f in slices if f.any()], axis=0)
    print(name, 'worst slice median strain/rate/final strain and absolute final bias:', worst.round(5), 'pooled median strain/rate:', np.median(err[:, :-1]), np.median(err[:, -1]), 'tau ranges:', np.ptp(params, axis=0))
    for miss in (False, True):
        print('missing:', miss, 'n:', (missing == miss).sum(), 'median strain/rate:', np.median(err[missing == miss, :-1]), np.median(err[missing == miss, -1]))
PY
```
- **Container Exit Code**: `0` (`1346 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public shear, taus, weights: 0.12 [0.61278825 3.69022857] [ 0.67823678  0.55271244  0.13976353  0.15676009 -0.12694661  0.11757215
 -0.08865851] [0.28111826 0.24146184]
shear/damage gated rows: 0.5 48 of 72
shear/damage gated rows: 0.5000000000000001 0 of 72
shear/damage gated rows: 0.12 48 of 72
lab worst slice median strain/rate/final strain and absolute final bias: [0.01724 0.01957 0.01873 0.00516] pooled median strain/rate: 0.012976445600033325 0.014588957533762707 tau ranges: [0.00162028 0.06558372]
missing: False n: 161 median strain/rate: 0.01144263253705996 0.013748074276416444
missing: True n: 19 median strain/rate: 0.10818481366045962 0.08244084304520483
stratum worst slice median strain/rate/final strain and absolute final bias: [0.01459 0.01578 0.01597 0.00956] pooled median strain/rate: 0.012903137888177041 0.01461544129295378 tau ranges: [0.00120002 0.10746158]
missing: False n: 161 median strain/rate: 0.01139605085451656 0.013012512751895855
missing: True n: 19 median strain/rate: 0.11055302747180551 0.09289250186526088

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=335511fd911094bf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_335511fd (directory).
```

### Turn 8 — `submit_completed_task` (`17757 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, an independent recurrence check across the shear cutoff, and held-out lab and stratum tests. The host verified the scientific checks against the submitted artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:7648'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. The public schema validator and independent strain and initial-rate checks passed. Worst leave-one-lab-out median errors across lab-by-activation slices were 0.01724 for strain and 0.01957 for initial rate. Missing coupon features remain the largest error slice: in leave-one-stratum-out validation, their median errors were 0.11055 and 0.09289, respectively.`

### Turn 9 — `execute_in_remote_docker` (`39192 ms` LLM)
- **Rationale**: Exercise the documented sparse and EMI calibration paths using measured sweep rows. A failed coefficient check would call for a calibration fix; systematic held-out bias on complete coupons would call for revisiting lab alignment rather than changing the prescribed imputation.
- **Alternatives Weighed**: `Add rational, relay, or stochastic drift terms` (rejected: The supplied rheology law contains none of these mechanisms, and the independent recurrence already verifies the two memory branches and terminal clock.); `Score within-batch balanced accuracy` (rejected: The outputs are continuous strain and rate, not class labels; signed held-out residuals within lab and activation strata are the relevant nuisance diagnostic.)

```bash
set -e
cd /holdout_task
cat > /tmp/rheo_margin_audit.py <<'PY'
import importlib.util, json, pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; lab = c['lab_id']; cmd = c['cmd_stress']; true = c['true_stress']
ref = np.array([np.polyfit(cmd[lab==j], true[lab==j], 1) for j in range(4)])
np.testing.assert_allclose(m._calibrate(d), ref, atol=1e-12)
worst_stress = 0.
for j in range(4):
    ids = np.flatnonzero(lab == j)
    for sign in (-1, 1):
        keep = np.concatenate((np.flatnonzero(lab != j), ids[:5]))
        pert = {k: np.asarray(v)[keep].copy() for k,v in c.items()}
        pert['true_stress'][-1] += sign * 4.
        expected = np.polyfit(cmd[ids[:4]], true[ids[:4]], 1)
        est = m._calibrate(dict(d, calibration_load_cells=pert))
        np.testing.assert_allclose(est[j], expected, atol=1e-12)
        others = np.arange(4) != j
        np.testing.assert_allclose(est[others], ref[others], atol=1e-12)
        for split in ('train','test'):
            data = d[split]; q = data['lab_id'] == j
            vals = np.r_[data['past_stress_cmd'][q].ravel(), data['stress_hold_cmd'][q]]
            worst_stress = max(worst_stress, np.max(abs(np.polyval(est[j]-ref[j], vals))))
print('Four clean calibration points plus either spike sign, worst stress shift from full sweep:', worst_stress)
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)
for j in range(4):
    idx = np.flatnonzero(lab == j)
    keep = (lab != j); keep[idx[0]] = True
    sparse = {k: np.asarray(v)[keep] for k,v in c.items()}
    sweep = next(h for h in reversed(history) if np.count_nonzero(np.asarray(h['lab_id']) == j) >= 2)
    q = np.asarray(sweep['lab_id']) == j
    expected = np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1)
    np.testing.assert_allclose(m._calibrate(dict(d, calibration_load_cells=sparse))[j], expected, atol=1e-12)
print('One-point history fallback verified for each channel')
past,hold,x,s0 = m._inputs(d); tr = d['train']; n = len(tr['lab_id']); y = np.c_[tr['strain_curves_obs']-s0[:n,None], tr['instant_rate_obs']]
active = m._Model(d,past[:n],hold[:n],x[:n]).active
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
r = np.empty_like(y)
for j in range(4):
    te = tr['lab_id']==j; ti = ~te
    def model(mask):
        return m._Model(d,past[:n][mask],hold[:n][mask],x[:n][mask])
    tau,w = m._fit(model(ti),y[ti]); r[te] = model(te).design(np.log(tau))[0] @ w-y[te]
slices = [(tr['lab_id']==j)&(active==gate)&complete for j in range(4) for gate in (False,True)]
print('Complete-coupon lab-by-activation worst absolute signed median bias, each horizon then rate:', np.max([abs(np.median(r[q],axis=0)) for q in slices],axis=0).round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Four clean calibration points plus either spike sign, worst stress shift from full sweep: 0.00036974818179833234
One-point history fallback verified for each channel
Complete-coupon lab-by-activation worst absolute signed median bias, each horizon then rate: [0.00311 0.00959 0.00953 0.00544 0.00783 0.01191 0.01021 0.01422 0.00897
 0.00638 0.0108 ]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=335511fd911094bf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_335511fd (directory).
```

### Turn 10 — `submit_completed_task` (`16354 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, independent response checks on both sides of the shear cutoff, and held-out lab and stratum validation. The margin audit verified one-point history fallback for all four channels and found a worst stress shift of 0.000370 after fitting four clean points in the presence of an EMI spike. No further change is supported by the audit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:7648'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. The public schema, independent strain and initial-rate calculations, and sparse and spiked calibration checks all passed. Worst leave-one-lab-out median errors across lab-by-activation slices were 0.01724 for strain and 0.01957 for initial rate. Missing coupon features remain the largest observed error source: median errors on those rows were 0.11055 and 0.09289 in leave-one-stratum-out validation.`

### Turn 11 — `execute_in_remote_docker` (`24404 ms` LLM)
- **Rationale**: The previous calibration audit covered one point without EMI and five points with EMI, but not their overlap. With two observations an affine line always fits both, so an EMI spike can evade residual screening. Test how much that distorts real held-out command stresses and inspect the archive policy before deciding whether a sparse-only calibration change is warranted.
- **Alternatives Weighed**: `Add new constitutive terms or classifier diagnostics` (rejected: The independent response and host panel checks already cover the prescribed two-branch law; the task has no relay state or class labels.)

```bash
set -e
sed -n '10,62p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py
cat > /tmp/rheo_sparse_probe.py <<'PY'
import importlib.util, pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; p = d['test']
for lab in range(4):
    ids = np.flatnonzero(c['lab_id']==lab)
    vals = np.r_[p['past_stress_cmd'][p['lab_id']==lab].ravel(), p['stress_hold_cmd'][p['lab_id']==lab]]
    baseline = m._calibrate(d)[lab]
    for size in (2, 3, 5):
        worst = 0.
        for j in range(size):
            for sign in (-1, 1):
                keep = np.r_[np.flatnonzero(c['lab_id']!=lab), ids[:size]]
                sweeps = {k: np.asarray(v)[keep].copy() for k,v in c.items()}
                sweeps['true_stress'][-size+j] += sign*4
                est = m._calibrate(dict(d, calibration_load_cells=sweeps))[lab]
                worst = max(worst, np.max(np.abs(np.polyval(est-baseline, vals))))
        print('channel, rows including EMI, worst physical stress shift:', lab, size, round(worst, 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_sparse_probe.py
```
- **Container Exit Code**: `0` (`593 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in = min(0.01, 8 * np.median(residual[core]))
            clean = residual <= 3.0 - margin
            emi = np.any(residual > 3.0)
        if (len(x) < 2 or emi) and clean.sum() < 4 and archive_path.is_file():
            with archive_path.open() as f:
                archive = json.load(f)
            historic = archive.get('calibration_load_cells', archive) if isinstance(archive, dict) else archive
            if isinstance(historic, dict) and 'lab_id' in historic:
                historic = [dict(zip(('lab_id', 'cmd_stress', 'true_stress'), r)) for r in zip(historic['lab_id'], historic['cmd_stress'], historic['true_stress'])]
            if isinstance(historic, list) and clean.sum() < 4:
                points = [(r['cmd_stress'], r['true_stress']) for r in historic
                          if {'lab_id', 'cmd_stress', 'true_stress'} <= r.keys() and int(r['lab_id']) == channel]
                if points:
                    xx, yy = np.asarray(points, float).T
                    x, y = np.r_[x[clean], xx], np.r_[y[clean], yy]
                    clean = np.ones(x.size, dtype=bool)
        x, y = x[clean], y[clean]
        if x.size >= 2 and np.ptp(x) > 0:
            coeff[channel] = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
        else:
            coeff[channel] = 1.0, np.median(y - x) if x.size else 0.0
    return coeff


channel, rows including EMI, worst physical stress shift: 0 2 439.360581
channel, rows including EMI, worst physical stress shift: 0 3 439.360581
channel, rows including EMI, worst physical stress shift: 0 5 0.00019
channel, rows including EMI, worst physical stress shift: 1 2 12.861907
channel, rows including EMI, worst physical stress shift: 1 3 12.671687
channel, rows including EMI, worst physical stress shift: 1 5 0.000527
channel, rows including EMI, worst physical stress shift: 2 2 15.306482
channel, rows including EMI, worst physical stress shift: 2 3 15.306268
channel, rows including EMI, worst physical stress shift: 2 5 0.00037
channel, rows including EMI, worst physical stress shift: 3 2 16.919192
channel, rows including EMI, worst physical stress shift: 3 3 17.765441
channel, rows including EMI, worst physical stress shift: 3 5 0.000147

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=335511fd911094bf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_335511fd (directory).
```

### Turn 12 — `execute_in_remote_docker` (`40601 ms` LLM)
- **Rationale**: Measure clean departures from nominal and historical lines, and count missed spikes near and above the cutoff. These results determine whether a sparse-only trigger can catch the observed failure without replacing clean in-batch fits.
- **Alternatives Weighed**: `Always blend historical and current sweeps` (rejected: This would override the clean in-batch fit and could import stale gains into the public predictions.); `Keep the residual-only filter for all sweep sizes` (rejected: Two distinct points exactly determine a line, so the fit can absorb an EMI spike without a residual.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import collections, json, pickle
import numpy as np
p='/holdout_task/public_alpha_m97_c01_batch_01.pkl'
with open(p,'rb') as f: d=pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: h=json.load(f)
c=d['calibration_load_cells']; labs=c['lab_id']; x=c['cmd_stress']; y=c['true_stress']
for j in range(4):
    ids=np.flatnonzero(labs==j); xx=x[ids]; yy=y[ids]
    sweep=next(s for s in reversed(h) if sum(np.asarray(s['lab_id'])==j)>=2)
    q=np.asarray(sweep['lab_id'])==j
    old=np.polyfit(np.asarray(sweep['cmd_stress'])[q],np.asarray(sweep['true_stress'])[q],1)
    err=yy-np.polyval(old,xx)
    print('channel',j,'archive line',np.round(old,6),'max clean departure from nominal/archive',round(max(abs(yy-xx)),6),round(max(abs(err)),6))
    for delta in (3.01,3.25,4.):
        misses=collections.Counter()
        for size in (2,3):
            for i in range(size):
                for sign in (-1,1):
                    if abs((yy-xx)[i]+sign*delta)<=3: misses['nominal']+=1
                    if abs(err[i]+sign*delta)<=3: misses['archive']+=1
        print(' spike',delta,'missed of 10 by nominal/archive',misses['nominal'],misses['archive'])
    pairs=np.triu_indices(len(xx),1); gap=np.ptp(xx)
    print(' smallest cmd separation/span',float(np.min(abs(xx[pairs[0]]-xx[pairs[1]]))),float(gap))
PY
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
channel 0 archive line [9.99976e-01 2.00000e-05] max clean departure from nominal/archive 0.791724 0.791749
 spike 3.01 missed of 10 by nominal/archive 5 5
 spike 3.25 missed of 10 by nominal/archive 5 5
 spike 4.0 missed of 10 by nominal/archive 0 0
 smallest cmd separation/span 0.023981999999999948 3.891897
channel 1 archive line [9.99993e-01 5.00000e-06] max clean departure from nominal/archive 0.520488 0.520483
 spike 3.01 missed of 10 by nominal/archive 5 5
 spike 3.25 missed of 10 by nominal/archive 2 2
 spike 4.0 missed of 10 by nominal/archive 0 0
 smallest cmd separation/span 0.022413000000000016 3.2981030000000002
channel 2 archive line [ 1.000012e+00 -4.600000e-05] max clean departure from nominal/archive 0.331732 0.331752
 spike 3.01 missed of 10 by nominal/archive 5 5
 spike 3.25 missed of 10 by nominal/archive 2 2
 spike 4.0 missed of 10 by nominal/archive 0 0
 smallest cmd separation/span 0.005259000000000014 2.9880809999999998
channel 3 archive line [1.000005e+00 5.100000e-05] max clean departure from nominal/archive 0.391463 0.39142
 spike 3.01 missed of 10 by nominal/archive 3 3
 spike 3.25 missed of 10 by nominal/archive 0 0
 spike 4.0 missed of 10 by nominal/archive 0 0
 smallest cmd separation/span 0.05906699999999998 3.37308

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=335511fd911094bf (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_335511fd (directory).
```

### Turn 13 — `execute_in_remote_docker` (`41845 ms` LLM)
- **Rationale**: Limit the change to two- and three-point sweeps. The measured clean offsets are at most 0.792, so a single spike larger than 3.0 remains separated by the 1.5 cutoff even when its sign opposes the clean offset. Verify that clean fits stay unchanged, history is used only when fewer than two clean points remain, and the exact artifact still passes the public schema.
- **Alternatives Weighed**: `Use a 3.0 cutoff on deviations from nominal stress` (rejected: Measured clean offsets can partly cancel a near-threshold spike, so this cutoff missed spikes of either sign in the sparse stress test.); `Use historical coefficients for all sparse sweeps` (rejected: The measured historical gains differ from current gains; two clean current points still identify the affine calibration.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_01_solver.py')
s = p.read_text()
old = '''        x, y = cmd[m], true[m]
        if len(x) >= 3:'''
new = '''        x, y = cmd[m], true[m]
        if 2 <= len(x) <= 3:
            # A sparse affine fit can absorb a spike entirely. In this
            # campaign clean command-to-true corrections are below half
            # the EMI cutoff, which separates a single spike from its peers.
            suspect = np.abs(y - x) > 1.5
            if suspect.sum() == 1:
                x, y = x[~suspect], y[~suspect]
        if len(x) >= 3:'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
cat > /tmp/rheo_sparse_verify.py <<'PY'
import importlib.util, itertools, json, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)
c = d['calibration_load_cells']; labs = c['lab_id']; x = c['cmd_stress']; y = c['true_stress']
ref = np.array([np.polyfit(x[labs == j], y[labs == j], 1) for j in range(4)])
np.testing.assert_allclose(m._calibrate(d), ref, atol=1e-12)
worst = {2: 0., 3: 0.}; count = 0
for j in range(4):
    other = np.flatnonzero(labs != j); rows = np.flatnonzero(labs == j)
    sweep = next(h for h in reversed(history) if np.count_nonzero(np.asarray(h['lab_id']) == j) >= 2)
    q = np.asarray(sweep['lab_id']) == j
    fallback = np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1)
    test = d['test']; q = test['lab_id'] == j
    vals = np.r_[test['past_stress_cmd'][q].ravel(), test['stress_hold_cmd'][q]]
    for size in (2, 3):
        for combo in itertools.combinations(rows, size):
            keep = np.r_[other, combo]; clean = {k: np.asarray(v)[keep].copy() for k, v in c.items()}
            np.testing.assert_allclose(m._calibrate(dict(d, calibration_load_cells=clean))[j], np.polyfit(x[list(combo)], y[list(combo)], 1), atol=1e-10)
            for k, bad in enumerate(combo):
                good = [i for i in combo if i != bad]
                expected = fallback if size == 2 else np.polyfit(x[good], y[good], 1)
                for delta in (-4., -3.0001, 3.0001, 4.):
                    corrupted = dict(clean, true_stress=clean['true_stress'].copy())
                    corrupted['true_stress'][len(other)+k] += delta
                    actual = m._calibrate(dict(d, calibration_load_cells=corrupted))
                    np.testing.assert_allclose(actual[j], expected, atol=1e-10)
                    np.testing.assert_allclose(actual[np.arange(4) != j], ref[np.arange(4) != j], atol=1e-12)
                    worst[size] = max(worst[size], np.max(abs(np.polyval(actual[j]-ref[j], vals))))
                    count += 1
print('Sparse EMI cases passed:', count, 'worst stress shift by sweep size:', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_sparse_verify.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`6803 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
Sparse EMI cases passed: 7200 worst stress shift by sweep size: {2: 1.4036968168830355, 3: 0.06249218822230581}
Four clean calibration points plus either spike sign, worst stress shift from full sweep: 0.00036974818179833234
One-point history fallback verified for each channel
Complete-coupon lab-by-activation worst absolute signed median bias, each horizon then rate: [0.00311 0.00959 0.00953 0.00544 0.00783 0.01191 0.01021 0.01422 0.00897
 0.00638 0.0108 ]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=22ca90d97b0db1b9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_22ca90d9 (directory).
```

### Turn 14 — `submit_completed_task` (`15116 ms` LLM)
- **Rationale**: The verified candidate preserves clean in-batch calibration, rejects the tested sparse spikes, and uses history when fewer than two clean points remain. The remaining stress shift in that fallback reflects the difference between archived and current calibration.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. The isolated public schema check passed. The sparse calibration fix passed all 7,200 tested EMI cases without changing the clean public fit. Missing coupon features remain the largest observed error slice: leave-one-stratum-out median errors were 0.11055 for strain and 0.09289 for initial rate.`

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
