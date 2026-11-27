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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `332,087` (`319,987` / `103,168` / `12,100`) |
| **Total Episode Wall Time (s)** | `397.77s` |

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

### Turn 1 — `execute_in_remote_docker` (`16521 ms` LLM)
- **Rationale**: Inspect the public schema and compact data samples to identify calibration and panel conventions before fitting the specified physical model.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The current batch specifies a different terminal-flow basis; only compatible calibration and alignment logic should carry forward.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c01_batch_03.pkl','rb'))
def show(x,path='',d=0):
 print(path,type(x).__name__,getattr(x,'shape',''),getattr(x,'dtype',''), (len(x) if isinstance(x,(dict,list,tuple)) else '') )
 if isinstance(x,dict) and d<3:
  for k,v in x.items(): show(v,path+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)) and len(x): print(' sample:',str(x[0])[:550])
 elif isinstance(x,np.ndarray): print(' sample:',str(x[:2])[:350])
 elif np.isscalar(x): print(' value:',x)
show(p)
PY
ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`480 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```

### Turn 2 — `execute_in_remote_docker` (`16595 ms` LLM)
- **Rationale**: Inspect the nested measurement keys and predecessor preprocessing to detect mismatches in panel layout, targets, and calibration fallback before implementing the fit.
- **Alternatives Weighed**: `Carry forward the predecessor's complete model` (rejected: Its calibration and panel conventions may transfer, but the current task specifies a different terminal-flow basis.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c01_batch_03.pkl','rb'))[0]
def view(d,depth=0):
 for k,v in d.items():
  if isinstance(v,dict):
   print(' '*depth,k); view(v,depth+2)
  else:
   a=np.asarray(v); print(' '*depth,k,a.shape,a.dtype,repr(a.ravel()[:5]))
view(p)
PY
cat /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
cat /tmp/hegel_continual_archive/archive_manifest.json
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation
```
- **Container Exit Code**: `0` (`497 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`19474 ms` LLM)
- **Rationale**: The previous output truncated the predecessor code. This focused read recovers its alignment convention, shows where missing entries occur, and tests whether any observed calibration channel needs the archive fallback.
- **Alternatives Weighed**: `Use the reference-panel primitive directly` (rejected: The task requires medians of per-reference-column moments and lab-by-stratum imputation; confirm those operations against the predecessor before choosing an implementation.)

```bash
sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('history layout:',str(h)[:800]); d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; print('missing:', {s:{k:int(np.isnan(d[s][k]).sum()) for k in ('coupon_features_raw','ref_coupons')} for s in ('train','test')}); c=d['calibration_load_cells'];
for j in range(4):
 m=c['lab_id']==j; x=c['cmd_stress'][m]; y=c['true_stress'][m]; print('calibration',j,len(x),np.round(y-np.polyval(np.polyfit(x,y,1),x),6))
PY
```
- **Container Exit Code**: `0` (`427 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
past
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
history layout: [{'episode_short_name': 'alpha_m97_c01_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.660283, -0.418768, 1.112723, -1.62604, -0.501169, -1.672067, 0.863389, 1.008755, 1.968856, 1.530083, 0.979468, 0.449112, 1.469886, -0.332491, -1.532413, 0.660575, -1.672999, -1.196407, 0.934354, 0.731875, 1.670914, -2.075727, 1.021513, -1.730233, -1.965428, 0.955855, 0.129182, -0.328527, -1.550117, -1.993809, 2.131528, -2.023748, 0.667305, -0.968757, -0.649251, -0.022422, -1.62562, -0.902597, 1.990079, -0.41266], 'true_stress': [1.660305, -0.418669, 1.112724, -1.626001, -0.501125, -1.671976, 0.863198, 1.008634, 1.968816, 1.530248, 0.979291, 0.449215, 1.470001, -0.332446, -1.532394, 0.660518, -1.6729
missing: {'train': {'coupon_features_raw': 23, 'ref_coupons': 238}, 'test': {'coupon_features_raw': 8, 'ref_coupons': 95}}
calibration 0 10 [-9.90e-05  4.30e-05 -1.28e-04  9.00e-05  8.00e-05  2.17e-04  9.00e-06
 -9.00e-05 -1.54e-04  3.30e-05]
calibration 1 10 [ 4.00e-06  3.40e-05 -1.20e-05 -7.40e-05 -9.70e-05  2.80e-05  1.13e-04
  1.01e-04  4.60e-05 -1.42e-04]
calibration 2 10 [-5.70e-05 -1.19e-04  7.40e-05  2.60e-05  4.00e-05 -9.20e-05 -1.10e-05
  6.00e-06  1.62e-04 -2.90e-05]
calibration 3 10 [ 4.20e-05  1.70e-05 -3.70e-05 -1.40e-05 -1.30e-05  9.00e-06 -2.00e-05
  4.20e-05 -1.72e-04  1.47e-04]
```

### Turn 4 — `execute_in_remote_docker` (`63488 ms` LLM)
- **Rationale**: All four public channels have ten clean calibration rows, so the solver fits each channel in batch. This step implements the specified basis and checks the output contract in an isolated process before measuring held-out strain and rate errors.
- **Alternatives Weighed**: `Fit strain curves alone` (rejected: Measured initial rates independently constrain the short-time response and are a required output.); `Use the bundled single-relaxation primitive` (rejected: It does not implement two history-dependent branches with the prescribed slip and gap-contrast terms.)

```bash
set -e
cd /holdout_task
cat > alpha_m97_c01_batch_03_solver.py <<'PY'
"""Fit the calibrated two-branch jammed-slip rheology model."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    params = np.empty((4, 2))
    history = None
    for c in range(4):
        mask = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[mask], true[mask]
        clean = np.ones(len(x), dtype=bool)
        emi = False
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            errors = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
            # With one spike and three rows, all pairs have zero median error.
            # Resolve ties by the deviation from the median pair slope.
            typical = np.median(errors, axis=1)
            tied = np.flatnonzero(typical <= typical.min() + 1e-12)
            best = tied[np.argmin(np.abs(slopes[tied] - np.median(slopes)))]
            core = errors[best] <= 8 * typical[best] + 1e-12
            residual = np.abs(np.polyval(_line(x[core], y[core]), x) - y)
            emi = np.any(residual > 3.0)
            clean = residual <= 3.0
        xc, yc = x[clean], y[clean]
        if len(xc) >= 4 or (len(xc) >= 2 and not emi):
            params[c] = _line(xc, yc)
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for sweep in reversed(history):
            m = np.asarray(sweep['lab_id']) == c
            hx, hy = (np.asarray(sweep[k])[m] for k in ('cmd_stress', 'true_stress'))
            if len(hx) >= 2:
                old = _line(hx, hy)
                break
        if len(xc) >= 2:
            params[c] = _line(xc, yc)
        elif len(xc) == 1:
            params[c] = old[0], yc[0] - old[0] * xc[0]
        else:
            params[c] = old
    return params


def _prepare(dataset):
    def stack(name):
        return np.concatenate([dataset[s][name] for s in ('train', 'test')])

    lab, strata = stack('lab_id'), stack('stratum')
    ref, aligned = stack('ref_coupons'), stack('coupon_features_raw')
    for c in range(4):
        m = lab == c
        if np.any(m):
            center = np.nanmedian(np.nanmean(ref[m], axis=0))
            scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
            aligned[m] = (aligned[m] - center) / scale
    cells = []
    for c, s in np.unique(np.column_stack((lab, strata)), axis=0):
        a = aligned[(lab == c) & (strata == s)]
        count = np.sum(np.isfinite(a), axis=0)
        cells.append(np.divide(np.nansum(a, axis=0), count,
                               out=np.full(a.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(cells, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dt, times, past, hold, features):
        self.dt = np.asarray(dt)
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.times = np.asarray(times)
        self.past = past
        self.hold = hold
        base = np.column_stack((np.ones(len(hold)), features,
                                np.sign(hold) * np.sqrt(np.abs(hold))))
        phi = self.times / (1.0 + 0.85 * self.times)
        self.fixed = base[:, None, :] * np.r_[phi, 1.0][None, :, None]

    def design(self, taus):
        taus = np.asarray(taus)
        memory = -np.expm1(-self.dt[:, None] / taus) * np.exp(-self.after[:, None] / taus)
        gap = self.hold[:, None] - self.past @ memory
        kernel = np.vstack((-np.expm1(-self.times[:, None] / taus), 1.0 / taus))
        prony = gap[:, None, :] * kernel
        contrast = (gap[:, 0] - gap[:, 1])[:, None] * kernel[None, :, 0]
        return np.concatenate((prony, self.fixed, contrast[..., None]), axis=2)


def _fit(model, y, mask=None, rate_weight=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    valid = np.isfinite(y) & mask
    target = y[valid]
    n_curves = y.shape[1] - 1

    def run(weight, starts):
        scale = np.broadcast_to(np.r_[np.ones(n_curves), weight], y.shape)[valid]

        def evaluate(log_taus):
            design = model.design(np.exp(log_taus))[valid] * scale[:, None]
            coef = np.linalg.lstsq(design, target * scale, rcond=None)[0]
            resid = np.einsum('ij,j->i', design, coef) - target * scale
            return resid, coef

        opt = min((least_squares(lambda x: evaluate(x)[0], start,
                                 bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),
                                 max_nfev=100, ftol=1e-10) for start in starts),
                  key=lambda o: o.cost)
        return np.exp(opt.x), evaluate(opt.x)[1]

    starts = np.log([[0.45, 4.0], [0.2, 2.0], [1.0, 10.0]])
    tau, coef = run(1.0 if rate_weight is None else rate_weight, starts)
    if rate_weight is None:
        r = np.einsum('ntp,p->nt', model.design(tau), coef) - y

        def mad(x):
            return max(np.median(np.abs(x - np.median(x))), 1e-8)

        if np.any(valid[:, -1]):
            w = np.clip(mad(r[:, :-1][valid[:, :-1]]) / mad(r[:, -1][valid[:, -1]]), 0.1, 10.0)
            tau, coef = run(w, [np.log(tau)])
    return tau, coef


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    past, hold, aligned, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    y = np.column_stack((train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs']))
    tau, coef = _fit(_Model(dt, times, past[:n], hold[:n], aligned[:n]), y)
    design = _Model(dt, times, past[n:], hold[n:], aligned[n:]).design(tau)
    prediction = np.einsum('ntp,p->nt', design, coef)
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], regime_mode_weights=coef[7:],
                predicted_strain=prediction[:, :-1] + strain0[n:, None],
                initial_rate_pred=prediction[:, -1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
sed -n '34,73p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
```
- **Container Exit Code**: `0` (`653 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
            design = np.column_stack((x, np.ones(x.size)))
            provisional = np.linalg.lstsq(design[core], y[core], rcond=None)[0]
            residual = np.abs(design @ provisional - y)
            # Allow for calibration noise in deciding which near-cutoff rows
            # to exclude, without relaxing the trigger for archive lookup.
            margin = min(0.01, 8 * np.median(residual[core]))
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


def _inputs(dataset):
    splits = (dataset['train'], dataset['test'])

    def stack(key):
        return np.concatenate([s[key] for s in splits], axis=0)

    lab, strata = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=372e7d4d46d0fbd8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_372e7d4d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27697 ms` LLM)
- **Rationale**: Measure held-out errors by lab, horizon, stress sign, and coupon missingness. Compare prefix and current calibration coefficients to see whether the latest sweep is a reliable fallback when current measurements are sparse.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: Finite outputs of the correct shape do not show that the fitted model generalizes across labs or predicts initial rates accurately.)

```bash
cat > /tmp/b03_validate.py <<'PY'
import importlib.util,json,pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('b03',root+'alpha_m97_c01_batch_03_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
a=d['train']; n=len(a['lab_id']); times=d['eval_times']; p,h,x,z=s._prepare(d)
m=s._Model(d['past_dt'],times,p[:n],h[:n],x[:n]); y=np.column_stack((a['strain_curves_obs']-z[:n,None],a['instant_rate_obs']))
rmse=lambda r: float(np.sqrt(np.nanmean(r*r)))
tau,b=s._fit(m,y); full=m.design(tau)@b-y
print('full fit',tau,b,'residual RMSE by time then rate',np.sqrt(np.nanmean(full**2,axis=0)).round(5),'max residual',np.nanmax(abs(full),axis=0).round(5))
r=np.empty_like(y)
for lab in np.unique(a['lab_id']):
 hold=a['lab_id']==lab
 t,c=s._fit(m,y,np.broadcast_to(~hold[:,None],y.shape)); r[hold]=(m.design(t)@c-y)[hold]
 print('LOLO lab',lab,'tau',t.round(5),'curve RMSE',round(rmse(r[hold,:-1]),6),'initial-rate RMSE',round(rmse(r[hold,-1]),6),'max weight shift',round(max(abs(c-b)),6))
missing=np.isnan(a['coupon_features_raw']).any(axis=1)
for name,sel in [('missing coupon',missing),('complete coupon',~missing),('negative hold',h[:n]<0),('nonnegative hold',h[:n]>=0)]:
 print(name,'n',sum(sel),'early/late/rate RMSE',*[round(rmse(q),6) for q in (r[sel,:3],r[sel,7:-1],r[sel,-1])])
for j in (0,len(times)-1,len(times)):
 mask=np.ones_like(y,dtype=bool);mask[:,j]=False
 t,c=s._fit(m,y,mask); print('withheld horizon',j,'RMSE',round(rmse((m.design(t)@c-y)[:,j]),6),'taus',t.round(5))
base=s._calibrate(d); print('in-batch calib',base.round(6))
for sweep in json.loads(s._HISTORY.read_text()):
 ll,xx,yy=(np.asarray(sweep[k]) for k in ('lab_id','cmd_stress','true_stress'))
 print('prefix',sweep.get('episode_short_name'),np.array([s._line(xx[ll==c],yy[ll==c]) for c in range(4)]).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b03_validate.py
```
- **Container Exit Code**: `0` (`872 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full fit [0.58244441 3.90380135] [ 0.61375459  0.51899414  0.14601774  0.12229425 -0.13560913  0.11146808
 -0.0951333   0.42177895 -0.1515578 ] residual RMSE by time then rate [0.57278 0.4566  0.26881 0.1776  0.12054 0.45758 0.14901 0.18896 0.89009
 1.11248 0.07307] max residual [ 5.28384  5.14783  3.10137  1.26415  1.18646  5.58933  1.06173  2.05163
 11.16308 14.83034  0.29964]
LOLO lab 0 tau [0.53023 3.48051] curve RMSE 0.875154 initial-rate RMSE 0.068844 max weight shift 0.108367
LOLO lab 1 tau [0.59882 3.83754] curve RMSE 0.419472 initial-rate RMSE 0.083574 max weight shift 0.025833
LOLO lab 2 tau [0.61741 4.48861] curve RMSE 0.551047 initial-rate RMSE 0.109761 max weight shift 0.07514
LOLO lab 3 tau [0.58316 4.0206 ] curve RMSE 0.183924 initial-rate RMSE 0.084429 max weight shift 0.015868
missing coupon n 23 early/late/rate RMSE 0.639952 1.846347 0.128865
complete coupon n 157 early/late/rate RMSE 0.418382 0.55064 0.083357
negative hold n 87 early/late/rate RMSE 0.595942 0.713438 0.091851
nonnegative hold n 93 early/late/rate RMSE 0.254074 0.937439 0.089133
withheld horizon 0 RMSE 0.572958 taus [0.57556 3.88458]
withheld horizon 9 RMSE 1.120916 taus [0.56051 3.79582]
withheld horizon 10 RMSE 0.087617 taus [0.59236 3.96891]
in-batch calib [[ 1.339989  0.150001]
 [ 0.770039 -0.170026]
 [ 1.209981 -0.119991]
 [ 0.830001  0.110012]]
prefix alpha_m97_c01_s00 [[ 9.999760e-01  2.000000e-05]
 [ 9.999930e-01  5.000000e-06]
 [ 1.000012e+00 -4.600000e-05]
 [ 1.000005e+00  5.100000e-05]]
prefix alpha_m97_c01_batch_01 [[ 1.339999  0.149994]
 [ 0.770004 -0.169994]
 [ 1.210027 -0.120003]
 [ 0.83001   0.109973]]
prefix alpha_m97_c01_batch_02 [[ 1.34003   0.149977]
 [ 0.770006 -0.169977]
 [ 1.209998 -0.120013]
 [ 0.829939  0.110089]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=372e7d4d46d0fbd8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_372e7d4d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31955 ms` LLM)
- **Rationale**: Compare Huber and Cauchy fits against least squares with residual scales estimated inside each training fold. Promote a robust fit only if it improves typical held-out strain error without sacrificing initial-rate accuracy.
- **Alternatives Weighed**: `Change the constitutive equation` (rejected: Individual strain residuals reach 14.83, but that alone does not establish a missing physical term. Test whether limiting their influence improves held-out predictions first.)

```bash
cat > /tmp/b03_robust.py <<'PY'
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/b03_validate.py').read())
from scipy.optimize import least_squares

def robust(model, y, mask, kind):
    valid = mask & np.isfinite(y)
    t, c = s._fit(model, y, mask)
    residual = model.design(t) @ c - y
    sigma = []
    for j in range(y.shape[1]):
        v = residual[:, j][valid[:, j]]
        sigma.append(max(1.4826 * np.median(abs(v - np.median(v))), 1e-6) if len(v) else 1.)
    sigma = np.asarray(sigma)
    obs = y[valid]
    base = 1 / np.broadcast_to(sigma, y.shape)[valid]
    x = np.log(t)
    pred = model.design(t) @ c
    for _ in range(12):
        u = (pred[valid]-obs)*base
        w = 1 / np.sqrt(1 + (u / 2.5)**2) if kind == 'cauchy' else np.minimum(1, np.sqrt(1.5 / np.maximum(np.abs(u), 1e-10)))
        scale = base * w
        target = obs * scale
        def evaluate(v):
            D = model.design(np.exp(v))[valid] * scale[:, None]
            b = np.linalg.lstsq(D, target, rcond=None)[0]
            return np.einsum('ij,j->i',D,b)-target,b
        opt = least_squares(lambda v: evaluate(v)[0], x, bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),max_nfev=40)
        b = evaluate(opt.x)[1]
        newpred = model.design(np.exp(opt.x)) @ b
        if np.max(abs(newpred-pred)) < 1e-7:
            return np.exp(opt.x), b
        x,pred = opt.x,newpred
    return np.exp(x), b

print('absolute baseline residual quantiles by output', np.nanquantile(abs(full),[0.50,0.80,0.90,0.95,0.99],axis=0).round(4), sep='\n')
for kind in ['baseline','huber','cauchy']:
    fit = s._fit if kind == 'baseline' else lambda m,y,mask:robust(m,y,mask,kind)
    t,c=fit(m,y,np.ones_like(y,dtype=bool)); print(kind,'full tau/coef',t.round(5),c.round(5))
    for lab in np.unique(a['lab_id']):
        held=a['lab_id']==lab
        t,c=fit(m,y,np.broadcast_to(~held[:,None],y.shape)); e=(m.design(t)@c-y)[held]
        print(kind,int(lab),'tau',t.round(5),'curve median/p80/MAE',np.nanquantile(abs(e[:,:-1]),[.50,.80]).round(5),round(np.nanmean(abs(e[:,:-1])),5),'rate RMSE',round(rmse(e[:,-1]),5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b03_robust.py
```
- **Container Exit Code**: `0` (`1687 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
absolute baseline residual quantiles by output
[[0.0179 0.0236 0.0308 0.036  0.037  0.0414 0.0381 0.0394 0.0436 0.042
  0.0457]
 [0.0343 0.0485 0.0643 0.0689 0.0754 0.096  0.0736 0.0736 0.0801 0.08
  0.0809]
 [0.0502 0.0728 0.0901 0.0972 0.1075 0.2114 0.1013 0.1078 0.1277 0.1389
  0.11  ]
 [0.196  0.1863 0.149  0.1427 0.1479 0.3186 0.1676 0.1483 0.3068 0.1918
  0.1422]
 [3.5415 1.5095 1.066  1.0582 0.4079 1.3075 0.7367 0.6124 2.4127 0.7227
  0.2201]]
baseline full tau/coef [0.58244 3.9038 ] [ 0.61375  0.51899  0.14602  0.12229 -0.13561  0.11147 -0.09513  0.42178
 -0.15156]
baseline 0 tau [0.53023 3.48051] curve median/p80/MAE [0.02973 0.06484] 0.1332 rate RMSE 0.06884
baseline 1 tau [0.59882 3.83754] curve median/p80/MAE [0.03563 0.08161] 0.0983 rate RMSE 0.08357
baseline 2 tau [0.61741 4.48861] curve median/p80/MAE [0.04665 0.12916] 0.11484 rate RMSE 0.10976
baseline 3 tau [0.58316 4.0206 ] curve median/p80/MAE [0.04042 0.08869] 0.08124 rate RMSE 0.08443
huber full tau/coef [0.59279 3.83384] [ 0.67002  0.54244  0.13793  0.15374 -0.12852  0.11625 -0.08729  0.32565
 -0.21176]
huber 0 tau [0.59079 3.8589 ] curve median/p80/MAE [0.01472 0.0398 ] 0.12258 rate RMSE 0.05129
huber 1 tau [0.59544 3.88515] curve median/p80/MAE [0.01252 0.02462] 0.07305 rate RMSE 0.03492
huber 2 tau [0.59241 3.86954] curve median/p80/MAE [0.01535 0.03221] 0.07808 rate RMSE 0.064
huber 3 tau [0.594   3.76849] curve median/p80/MAE [0.01656 0.03386] 0.0541 rate RMSE 0.04451
cauchy full tau/coef [0.59318 3.84394] [ 0.67225  0.54287  0.13752  0.15427 -0.12838  0.11626 -0.08723  0.32326
 -0.21388]
cauchy 0 tau [0.59182 3.87079] curve median/p80/MAE [0.01477 0.03938] 0.12248 rate RMSE 0.05134
cauchy 1 tau [0.59592 3.8999 ] curve median/p80/MAE [0.01269 0.02466] 0.07291 rate RMSE 0.03424
cauchy 2 tau [0.59151 3.85759] curve median/p80/MAE [0.01472 0.0304 ] 0.07756 rate RMSE 0.0638
cauchy 3 tau [0.59416 3.78467] curve median/p80/MAE [0.01691 0.0337 ] 0.05412 rate RMSE 0.04443

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=372e7d4d46d0fbd8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_372e7d4d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`29171 ms` LLM)
- **Rationale**: Promote the tested robust estimator without changing the physical basis. Rerun held-out lab and horizon checks on the actual deliverable to catch discrepancies between the experiment and integrated implementation.
- **Alternatives Weighed**: `Huber weighting` (rejected: It was close, but Cauchy gave lower worst-lab curve MAE and rate RMSE, and slightly less variation in the fitted relaxation times across labs.)

```bash
cd /holdout_task && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy IRLS separable nonlinear least squares variable projection' | head -12; cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
s = s.replace('def _fit(model, y, mask=None, rate_weight=None):','def _initial_fit(model, y, mask=None, rate_weight=None):')
idx = s.index('\ndef solve_alpha_m97_c01_batch_03')
s = s[:idx] + '''

def _fit(model, y, mask=None):
    valid = np.isfinite(y)
    if mask is not None:
        valid &= mask
    tau, coef = _initial_fit(model, y, mask)
    prediction = model.design(tau) @ coef
    residual = prediction - y
    sigma = np.ones(y.shape[1])
    for j in range(y.shape[1]):
        v = residual[valid[:, j], j]
        if v.size:
            sigma[j] = max(1.4826 * np.median(np.abs(v - np.median(v))), 1e-6)
    obs = y[valid]
    base = np.broadcast_to(1.0 / sigma, y.shape)[valid]
    log_tau = np.log(tau)
    for _ in range(12):
        u = (prediction[valid] - obs) * base
        scale = base / np.hypot(1.0, u / 2.5)
        target = obs * scale

        def evaluate(v):
            design = model.design(np.exp(v))[valid] * scale[:, None]
            b = np.linalg.lstsq(design, target, rcond=None)[0]
            return np.einsum('ij,j->i', design, b) - target, b

        opt = least_squares(lambda v: evaluate(v)[0], log_tau,
                            bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),
                            max_nfev=40)
        log_tau = opt.x
        tau, coef = np.exp(log_tau), evaluate(log_tau)[1]
        updated = model.design(tau) @ coef
        if np.max(np.abs(updated - prediction)) < 1e-7:
            break
        prediction = updated
    return tau, coef

''' + s[idx:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b03_validate.py && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1324 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
full fit [0.59317591 3.84393513] [ 0.67224571  0.54286504  0.13751981  0.15426573 -0.12838179  0.11626361
 -0.08722808  0.32326006 -0.21387616] residual RMSE by time then rate [0.57214 0.45792 0.26946 0.17456 0.11218 0.45861 0.14457 0.18225 0.89228
 1.12264 0.04888] max residual [ 5.26653  5.17198  3.13556  1.31437  1.19975  5.64635  1.05691  2.00236
 11.2033  14.98017  0.3054 ]
LOLO lab 0 tau [0.59182 3.87079] curve RMSE 0.874908 initial-rate RMSE 0.051344 max weight shift 0.006886
LOLO lab 1 tau [0.59592 3.8999 ] curve RMSE 0.417651 initial-rate RMSE 0.034237 max weight shift 0.004372
LOLO lab 2 tau [0.59151 3.85759] curve RMSE 0.547138 initial-rate RMSE 0.063798 max weight shift 0.009972
LOLO lab 3 tau [0.59416 3.78467] curve RMSE 0.171665 initial-rate RMSE 0.044426 max weight shift 0.005738
missing coupon n 23 early/late/rate RMSE 0.64126 1.847403 0.128675
complete coupon n 157 early/late/rate RMSE 0.415727 0.546787 0.02227
negative hold n 87 early/late/rate RMSE 0.593671 0.709803 0.048491
nonnegative hold n 93 early/late/rate RMSE 0.252499 0.936727 0.052272
withheld horizon 0 RMSE 0.57214 taus [0.5937  3.85804]
withheld horizon 9 RMSE 1.122758 taus [0.59364 3.85677]
withheld horizon 10 RMSE 0.04934 taus [0.59035 3.82528]
in-batch calib [[ 1.339989  0.150001]
 [ 0.770039 -0.170026]
 [ 1.209981 -0.119991]
 [ 0.830001  0.110012]]
prefix alpha_m97_c01_s00 [[ 9.999760e-01  2.000000e-05]
 [ 9.999930e-01  5.000000e-06]
 [ 1.000012e+00 -4.600000e-05]
 [ 1.000005e+00  5.100000e-05]]
prefix alpha_m97_c01_batch_01 [[ 1.339999  0.149994]
 [ 0.770004 -0.169994]
 [ 1.210027 -0.120003]
 [ 0.83001   0.109973]]
prefix alpha_m97_c01_batch_02 [[ 1.34003   0.149977]
 [ 0.770006 -0.169977]
 [ 1.209998 -0.120013]
 [ 0.829939  0.110089]]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=908a1ad73d16b337 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_908a1ad7 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`23449 ms` LLM)
- **Rationale**: Exercise the calibration fallback using public sweep rows. If the sparse-spike fit departs materially from the clean line, use the archive to resolve point ambiguity before refitting. Check the predecessor’s panel convention and whether large residuals are isolated or persist across a trajectory before changing the estimator.
- **Alternatives Weighed**: `Submit the schema-valid robust fit immediately` (rejected: The public batch has ten clean calibration points per channel, so it does not exercise the required sparse and EMI calibration paths.)

```bash
set -e
sed -n '62,112p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -30
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib,io,copy
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/b03_validate.py').read())
print('panel shape',d['train']['ref_coupons'].shape)
for lab in range(4):
 rows=a['lab_id']==lab
 e=full[rows,:-1]
 print('lab',lab,'large residuals per row',np.bincount(np.sum(abs(e)>0.3,axis=1)), 'by horizon',np.sum(abs(e)>0.3,axis=0))
base=s._calibrate(d)
for lab in range(4):
 for kind in ('absent','single','spike','sparse_spike'):
  q=copy.deepcopy(d); c=q['calibration_load_cells']; ix=np.flatnonzero(c['lab_id']==lab); keep=np.ones(len(c['lab_id']),bool)
  if kind=='absent': keep[ix]=False
  if kind=='single': keep[ix[1:]]=False
  if kind in ('spike','sparse_spike'): c['true_stress'][ix[0]]+=5.
  if kind=='sparse_spike': keep[ix[3:]]=False
  q['calibration_load_cells']={k:v[keep] for k,v in c.items()}
  err=np.max(abs(s._calibrate(q)[lab]-base[lab])); print('calibration stress',lab,kind,round(err,6))
PY
```
- **Container Exit Code**: `0` (`1081 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
on",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "abc35d3defaecd9adc5e1091c1b53ab480677bd3aac9cbe7e691e6bd41e49062",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "scale > 0 finite scalar or broadcastable array; loss in {'huber', 'cauchy'}.",
        "weights >= 0 when provided; sensitivity must have nonzero weighted norm above gain_floor for identifiable slope."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "propagate_linear_basis": "APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)",
        "profile_weighted_affine": "APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)",
        "robust_residual_weights": "APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)"
      },
      "functions": [
        {
          "function": "robust_residual_weights",
          "signature": "robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray",
          "doc_summary": "For residuals of shape S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."
        }
      ],
      "index_cards": [
panel shape (180, 32)
lab 0 large residuals per row [21 11  3] by horizon [3 1 0 1 1 5 2 1 2 1]
lab 1 large residuals per row [33 14  2] by horizon [5 2 1 0 1 4 1 1 2 1]
lab 2 large residuals per row [42 11  2  0  2] by horizon [1 3 2 2 0 1 4 3 4 3]
lab 3 large residuals per row [27 10  1  1] by horizon [0 1 2 2 2 3 0 1 2 2]
calibration stress 0 absent 4.1e-05
calibration stress 0 single 0.000187
calibration stress 0 spike 1.2e-05
calibration stress 0 sparse_spike 1.940909
calibration stress 1 absent 4.9e-05
calibration stress 1 single 6e-05
calibration stress 1 spike 0.0
calibration stress 1 sparse_spike 1.956065
calibration stress 2 absent 2.2e-05
calibration stress 2 single 4.6e-05
calibration stress 2 spike 6e-06
calibration stress 2 sparse_spike 3.50203
calibration stress 3 absent 7.7e-05
calibration stress 3 single 6.3e-05
calibration stress 3 spike 4e-06
calibration stress 3 sparse_spike 15.363212

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=908a1ad73d16b337 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_908a1ad7 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`49492 ms` LLM)
- **Rationale**: Keep the verified constitutive fit unchanged. Use the historical sweep to identify the bad row only when in-batch calibration is sparse or shows a possible EMI excursion. Test both spike signs and all positions in a three-row sweep, then independently check the memory recurrence and both prediction formulas on the final artifact.
- **Alternatives Weighed**: `Reject contaminated rows by three-point consensus alone` (rejected: Each candidate line fits two points exactly; the observed sparse-spike tests produced coefficient errors as large as 15.36.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
s = s.replace('''            # With one spike and three rows, all pairs have zero median error.
            # Resolve ties by the deviation from the median pair slope.''', '''            # A three-point sweep cannot identify a bad row by pair consensus.
            # Inconsistent triples with a possible EMI excursion use the prefix
            # sweep to identify the contaminated point, not to fit clean rows.''')
s = s.replace('''            clean = residual <= 3.0
        xc, yc = x[clean], y[clean]
        if len(xc) >= 4 or (len(xc) >= 2 and not emi):''', '''            clean = residual <= 3.0
            if len(x) == 3:
                worst_by_pair = errors.max(axis=1)
                emi = worst_by_pair.min() > 0.01 and worst_by_pair.max() > 3.0
        xc, yc = x[clean], y[clean]
        if len(xc) >= 4 or (len(x) >= 2 and not emi):
            if not emi:
                xc, yc = x, y''')
s = s.replace('''        if len(xc) >= 2:
            params[c] = _line(xc, yc)
        elif len(xc) == 1:''', '''        if len(x) < 4 or emi:
            clean = np.abs(np.polyval(old, x) - y) <= 3.0
            xc, yc = x[clean], y[clean]
        if len(xc) >= 2:
            params[c] = _line(xc, yc)
        elif len(xc) == 1:''')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('b03','alpha_m97_c01_batch_03_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('public_alpha_m97_c01_batch_03.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
base=s._calibrate(d)
worst={}
for lab in range(4):
    for kind in ('absent','single','clean_three','spike','sparse_spike'):
        for shift in ((-5.,5.) if 'spike' in kind else (0.,)):
            for position in (range(3) if kind=='sparse_spike' else (0,)):
                q=copy.deepcopy(d); c=q['calibration_load_cells']; ix=np.flatnonzero(c['lab_id']==lab)
                keep=np.ones(len(c['lab_id']),bool)
                n={'absent':0,'single':1,'clean_three':3,'spike':len(ix),'sparse_spike':3}[kind]
                keep[ix[n:]]=False
                c['true_stress'][ix[position]]+=shift
                q['calibration_load_cells']={k:v[keep] for k,v in c.items()}
                err=float(np.max(abs(s._calibrate(q)[lab]-base[lab])))
                worst[kind]=max(worst.get(kind,0.),err)
print('max calibration coefficient deviation:',worst)
assert max(worst.values())<0.001
out=s.solve_alpha_m97_c01_batch_03(d)
past,hold,X,z0=s._prepare(d); n=len(d['train']['lab_id']); past,hold,X,z0=past[n:],hold[n:],X[n:],z0[n:]
tau=out['estimated_taus']; hw=out['hereditary_weights']; rw=out['regime_mode_weights']; t=d['eval_times']
z=np.zeros((len(hold),2))
for j,dt in enumerate(d['past_dt']):
    a=np.exp(-dt/tau); z=a*z+(1-a)*past[:,j,None]
gap=hold[:,None]-z
flow=hw[2]+X@hw[3:7]+rw[0]*np.sign(hold)*np.sqrt(np.abs(hold))
prony=1-np.exp(-t[:,None]/tau)
curve=z0[:,None]+(gap*hw[:2])@prony.T+flow[:,None]*t/(1+0.85*t)+rw[1]*np.diff(gap[:,::-1],axis=1)*prony[:,0]
rate=(gap*hw[:2])@(1/tau)+flow+rw[1]*(gap[:,0]-gap[:,1])/tau[0]
err=max(np.max(abs(out['predicted_strain']-curve)),np.max(abs(out['initial_rate_pred']-rate)))
print('max strain/rate difference vs direct recurrence and formula:',err)
assert err<1e-12
PY
```
- **Container Exit Code**: `0` (`958 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
max calibration coefficient deviation: {'absent': 7.735259867347466e-05, 'single': 0.00018710334187657662, 'clean_three': 0.00027556800680979265, 'spike': 1.2115620030561658e-05, 'sparse_spike': 0.0004929715553510655}
max strain/rate difference vs direct recurrence and formula: 1.3322676295501878e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b4b7f8771393c838 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_b4b7f877 (directory).
```

### Turn 10 — `submit_completed_task` (`21962 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification and an independent recurrence check. Sparse and EMI calibration tests across all four channels kept the maximum coefficient deviation below 0.0005. The calibration change leaves the clean public fit unchanged.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_03_solver.py': 'FILE:8343'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed, and both predictions matched the direct constitutive formulas to within 1.4e-15. Across held-out labs, the worst median absolute strain error was 0.01691 and the worst initial-rate RMSE was 0.06380.`

### Turn 11 — `execute_in_remote_docker` (`43216 ms` LLM)
- **Rationale**: Test whether plausible omitted terms improve prediction on unseen labs. Probe four- and five-row calibration sweeps and spikes just above the specified threshold, which the previous checks did not cover. A material calibration error would warrant a targeted fix.
- **Alternatives Weighed**: `Run relay, Fokker–Planck, or classification tests` (rejected: This dataset contains continuous strain and rate measurements, with no relay state, diffusion operator, or class labels. Held-out lab errors and signed residuals test the relevant nuisance effects.)

```bash
set -e
cat > /tmp/b03_gate_audit.py <<'PY'
import contextlib, copy, io
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/b03_validate.py').read())

# All candidate corrections are trained on other labs, then scored separately
# on the held-out trajectories and initial rates.
gap = np.broadcast_to(h[:n, None], (n, 2)).copy()
state = np.zeros_like(gap)
for j, dt in enumerate(d['past_dt']):
    w = np.exp(-dt / tau)
    state = w * state + (1 - w) * p[:n, j, None]
gap -= state
stress = h[:n]
X = x[:n]
phi = times / (1 + 0.85 * times)
flow = np.r_[phi, 1.]
contrast = np.r_[-np.expm1(-times / tau[0]), 1 / tau[0]]
missing = np.isnan(a['coupon_features_raw']).any(axis=1)
material = np.column_stack([X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)] + [X[:, i] * X[:, j] * X[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)])
corrections = {
    'prescribed': np.empty((*y.shape, 0)),
    'nonlinear material pairs/triples': material[:, None, :] * flow[None, :, None],
    'extra rational stress slip': np.column_stack([stress / (1 + stress**2), np.sign(stress)*np.sqrt(abs(stress))/(1+stress**2)])[:, None, :] * flow[None, :, None],
    'second contrast kernel': (gap[:, 0] - gap[:, 1])[:, None, None] * np.r_[-np.expm1(-times / tau[1]), 1 / tau[1]][None, :, None],
    'missingness diagnostic': missing[:, None, None] * flow[None, :, None],
}
for name, A in corrections.items():
    metrics = []
    for lab in np.unique(a['lab_id']):
        held = a['lab_id'] == lab
        mask = np.broadcast_to(~held[:, None], y.shape)
        t, b = s._fit(m, y, mask)
        err = m.design(t) @ b - y
        if A.shape[-1]:
            keep = mask & np.isfinite(err)
            scales = np.nanmedian(np.abs(np.where(mask, err, np.nan)), axis=0) * 1.4826
            w = np.broadcast_to(np.maximum(scales, 1e-6), y.shape)[keep]
            z = least_squares(lambda coeff: (A[keep] @ coeff + err[keep])/w,
                              np.zeros(A.shape[-1]), loss='cauchy', f_scale=2.5).x
            err = err + A @ z
        e = err[held]
        metrics.append((np.nanmedian(abs(e[:,:-1])), rmse(e[:,-1]), np.nanmedian(e[:,-1])))
    scores = np.asarray(metrics)
    print(name, 'worst median strain / rate RMSE', np.round(scores[:,:2].max(axis=0), 6), 'rate bias by lab', np.round(scores[:,2], 6))

base = s._calibrate(d)
worst = {}
sweep = d['calibration_load_cells']
for lab in range(4):
    rows = np.flatnonzero(sweep['lab_id'] == lab)
    for count in (2, 3, 4, 5, 10):
        shifts = (0.,) if count == 2 else (0., -3.01, 3.01, -5., 5.)
        for amount in shifts:
            for position in (range(count) if amount else (0,)):
                c = {k:v.copy() for k,v in sweep.items()}
                c['true_stress'][rows[position]] += amount
                keep = np.ones(len(c['lab_id']), dtype=bool)
                keep[rows[count:]] = False
                c = {k:v[keep] for k,v in c.items()}
                err = float(np.max(abs(s._calibrate(dict(d, calibration_load_cells=c))[lab]-base[lab])))
                key = (count, abs(amount))
                if key not in worst or err > worst[key][0]:
                    worst[key] = (err, lab, position, amount)
for key, v in sorted(worst.items()):
    print('calibration count / spike magnitude', key, 'worst error / lab / position / signed spike', tuple(round(c,7) for c in v))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b03_gate_audit.py
```
- **Container Exit Code**: `0` (`2813 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
prescribed worst median strain / rate RMSE [0.016914 0.063798] rate bias by lab [ 0.000996 -0.002879 -0.003152 -0.005654]
nonlinear material pairs/triples worst median strain / rate RMSE [0.018025 0.065215] rate bias by lab [ 0.001172 -0.002459 -0.001633 -0.006711]
extra rational stress slip worst median strain / rate RMSE [0.016388 0.063733] rate bias by lab [ 0.000332 -0.00058  -0.002298 -0.003615]
second contrast kernel worst median strain / rate RMSE [0.016686 0.063799] rate bias by lab [ 0.000767 -0.003018 -0.003117 -0.0064  ]
missingness diagnostic worst median strain / rate RMSE [0.016092 0.065114] rate bias by lab [ 0.000996 -0.002879 -0.003152 -0.007931]
calibration count / spike magnitude (2, 0.0) worst error / lab / position / signed spike (0.000381, 3, 0, 0.0)
calibration count / spike magnitude (3, 0.0) worst error / lab / position / signed spike (0.0002756, 3, 0, 0.0)
calibration count / spike magnitude (3, 3.01) worst error / lab / position / signed spike (0.000493, 2, 0, -3.01)
calibration count / spike magnitude (3, 5.0) worst error / lab / position / signed spike (0.000493, 2, 0, -5.0)
calibration count / spike magnitude (4, 0.0) worst error / lab / position / signed spike (2.74e-05, 0, 0, 0.0)
calibration count / spike magnitude (4, 3.01) worst error / lab / position / signed spike (0.0002756, 3, 3, -3.01)
calibration count / spike magnitude (4, 5.0) worst error / lab / position / signed spike (0.0002756, 3, 3, -5.0)
calibration count / spike magnitude (5, 0.0) worst error / lab / position / signed spike (2.32e-05, 1, 0, 0.0)
calibration count / spike magnitude (5, 3.01) worst error / lab / position / signed spike (7.05e-05, 2, 0, -3.01)
calibration count / spike magnitude (5, 5.0) worst error / lab / position / signed spike (7.05e-05, 2, 0, -5.0)
calibration count / spike magnitude (10, 0.0) worst error / lab / position / signed spike (0.0, 0, 0, 0.0)
calibration count / spike magnitude (10, 3.01) worst error / lab / position / signed spike (3.82e-05, 0, 8, -3.01)
calibration count / spike magnitude (10, 5.0) worst error / lab / position / signed spike (3.82e-05, 0, 8, -5.0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b4b7f8771393c838 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_b4b7f877 (directory).
```

### Turn 12 — `submit_completed_task` (`19113 ms` LLM)
- **Rationale**: The verified artifact satisfies the output contract and reproduces both specified prediction formulas. The focused audit found no material calibration failure just above the EMI threshold and no convincing evidence for an additional constitutive term.
- **Alternatives Weighed**: `Add a rational stress-slip term` (rejected: The diagnostic reduced worst-lab median strain error by only 0.000526 and initial-rate RMSE by 0.000065, insufficient evidence to depart from the specified equation.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The isolated public schema check passed, and strain and initial-rate predictions matched the direct formulas within 1.4e-15. Across held-out labs, the worst median absolute strain error was 0.01691 and the worst initial-rate RMSE was 0.06380. Sparse and EMI calibration tests kept coefficient deviations below 0.0005.`

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
