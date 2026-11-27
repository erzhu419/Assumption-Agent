# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `328,416` (`315,534` / `92,032` / `12,882`) |
| **Total Episode Wall Time (s)** | `449.85s` |

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

### Turn 1 — `execute_in_remote_docker` (`15405 ms` LLM)
- **Rationale**: Inspect the public schema and a small sample to detect field, axis, and split mismatches before implementing the estimator.

```bash
cd /holdout_task; cat validate_public_schema.py; cat public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS reference panel alignment imputation viscoelastic relaxation' 2>&1; python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c03_s00.pkl','rb') as f:d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''), str(v)[:280] if not isinstance(v,np.ndarray) else str(v[:2])[:250]);
 if isinstance(v,dict):
  for a,b in v.items():print(' ',a,type(b).__name__,getattr(b,'shape',''),str(b)[:140])
PY
```
- **Container Exit Code**: `1` (`364 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18355 ms` LLM)
- **Rationale**: Read the dataset from the list entry identified by the validator and inspect panel axes, missingness, and stratum counts. These determine the alignment and imputation operators; the skill descriptions determine whether existing primitives match those operators.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,scipy
with open('public_alpha_m97_c03_s00.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset'];print('scipy',scipy.__version__,'public item keys',list(item));
for k,v in d.items():
 print(k, type(v).__name__,getattr(v,'shape',''),repr(v)[:220] if not isinstance(v,dict) else '')
 if isinstance(v,dict):
  for a,b in v.items():
   x=np.asarray(b);print(' ',a,x.shape,x.dtype,'first',str(x[:2] if x.ndim else x)[:240], 'nonfinite',np.sum(~np.isfinite(x)) if np.issubdtype(x.dtype,np.number) else '-')
   if a in ('lab_id','stratum','stratum_id'): print(' counts',np.unique(x,return_counts=True))
   if a=='ref_coupons' and x.ndim>1: print(' ref moments',np.nanmean(x,axis=0),np.nanstd(x,axis=0))
PY
for card in autogenous_prim_robust_trend_decomposition.cauchy_linear_fit autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$card"; done
```
- **Container Exit Code**: `0` (`476 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nempty real matrix of shape (n, p) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
        "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
        "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
      },
      "functions": [
        {
          "function": "weighted_column_fill",
          "signature": "weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray",
          "doc_summary": "Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."
        }
      ],
      "index_cards": [
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_reference_panel_balancing",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`21799 ms` LLM)
- **Rationale**: The previous log truncated the strain fields. Confirm their shapes and missingness before building the residual, and check affine residuals to determine whether the public data need the EMI fallback. Inspect only the archive format so ordinary calibration stays entirely in batch.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c03_s00.pkl','rb'))[0];d=item['dataset']
for k in ('train','test'):
 print(k)
 for n,v in d[k].items():
  a=np.asarray(v);print(' ',n,a.shape,a.dtype,'nonfinite',np.count_nonzero(~np.isfinite(a)), 'range', (float(np.nanmin(a)),float(np.nanmax(a))))
 print('  cell counts',np.unique(np.stack([d[k]['lab_id'],d[k]['stratum']],axis=1),axis=0,return_counts=True)[1])
print('GT validation fields only:',[(k,np.shape(v)) for k,v in item['gt'].items()])
c=d['calibration_load_cells']
for lab in range(4):
 ix=c['lab_id']==lab;x=c['cmd_stress'][ix];y=c['true_stress'][ix];p=np.polyfit(x,y,1)
 print('calibration',lab,'max_affine_residual',np.max(np.abs(y-np.polyval(p,x))))
PY
if test -f /tmp/hegel_continual_archive/archive_calibration_history.json; then python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json';a=json.load(open(p))
print('Archive format:', type(a).__name__, list(a)[:8] if isinstance(a,dict) else ('records',len(a)))
if isinstance(a,dict):
 for k,v in list(a.items())[:8]:print(k,type(v).__name__,str(v)[:220])
elif a:print(str(a[0])[:350])
PY
fi
```
- **Container Exit Code**: `0` (`359 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
train
  lab_id (180,) int64 nonfinite 0 range (0.0, 3.0)
  stratum (180,) int64 nonfinite 0 range (0.0, 2.0)
  coupon_features_raw (180, 4) float64 nonfinite 21 range (-3.9916993615681347, 5.112926091068586)
  ref_coupons (180, 32) float64 nonfinite 231 range (-4.7771952683568015, 5.076790267783817)
  past_stress_cmd (180, 18) float64 nonfinite 0 range (-2.067, 2.067)
  stress_hold_cmd (180,) float64 nonfinite 0 range (-1.99111, 1.989577)
  strain_0 (180,) float64 nonfinite 0 range (-1.113888, 0.837982)
  aux_chamber_telemetry (180, 3) float64 nonfinite 0 range (-3.105885, 3.106324)
  strain_curves_obs (180, 10) float64 nonfinite 0 range (-18.705006627425867, 9.69996514417255)
  instant_rate_obs (180,) float64 nonfinite 0 range (-3.9463767231913693, 3.751797619229004)
  cell counts [18 17 11  9 24 13 17  8 17 16 14 16]
test
  lab_id (72,) int64 nonfinite 0 range (0.0, 3.0)
  stratum (72,) int64 nonfinite 0 range (0.0, 2.0)
  coupon_features_raw (72, 4) float64 nonfinite 6 range (-2.7152891013326466, 5.910442579049255)
  ref_coupons (72, 32) float64 nonfinite 93 range (-3.3783768434540407, 5.145925555832124)
  past_stress_cmd (72, 18) float64 nonfinite 0 range (-2.067, 2.067)
  stress_hold_cmd (72,) float64 nonfinite 0 range (-1.990892, 1.988736)
  strain_0 (72,) float64 nonfinite 0 range (-0.814206, 0.727098)
  aux_chamber_telemetry (72, 3) float64 nonfinite 0 range (-3.295369, 3.166989)
  cell counts [ 4  7  3  3  8  6  6  4 10  8  7  6]
GT validation fields only: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
calibration 0 max_affine_residual 0.00017956711800549974
calibration 1 max_affine_residual 0.00025393585360999893
calibration 2 max_affine_residual 0.000189510249081426
calibration 3 max_affine_residual 0.00012149675200370455
```

### Turn 4 — `execute_in_remote_docker` (`67182 ms` LLM)
- **Rationale**: Implement the closed-form history weights and fit only to training observations. The isolated validator catches interface failures; independent public truth then checks whether the inferred weights and relaxation times reproduce both outputs across horizons and lab–stratum cells.
- **Alternatives Weighed**: `Ordinary least squares alone` (rejected: Cauchy glitches can dominate the relaxation fit; least squares serves only to initialize it.)

```bash
cat > /holdout_task/alpha_m97_c03_s00_solver.py <<'PY'
"""Calibrated two-branch Prony model for terminal polymer flow."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _archive_calibration(lab):
    try:
        with open(_ARCHIVE) as stream:
            history = json.load(stream)
    except FileNotFoundError:
        return None
    if isinstance(history, dict):
        if 'stress_gains' in history and 'stress_biases' in history:
            return np.array([history['stress_gains'][lab], history['stress_biases'][lab]])
        history = history.get('records', history.get(str(lab), []))
    if isinstance(history, dict):
        history = [history]
    if history and isinstance(history[0], (list, tuple, int, float)):
        return np.asarray(history, dtype=float) if np.shape(history) == (2,) else None
    fits = [[r['gain'], r['bias']] for r in history
            if int(r.get('lab_id', lab)) == lab and 'gain' in r and 'bias' in r]
    return np.median(fits, axis=0) if fits else None


def _calibrate(cal):
    result = []
    for lab in range(4):
        rows = np.asarray(cal['lab_id']) == lab
        x = np.asarray(cal['cmd_stress'])[rows]
        y = np.asarray(cal['true_stress'])[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        emi = False
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                offset = np.median(y - slope*x)
                clean = np.abs(y - (slope*x + offset)) <= 3.0
                emi = not clean.all()
                x, y = x[clean], y[clean]
        archived = _archive_calibration(lab) if len(x) < 2 or emi else None
        if len(x) >= 4 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        elif archived is not None:
            fit = archived
        elif len(x) >= 2 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        else:
            fit = (1.0, np.mean(y-x) if len(x) else 0.0)
        result.append(fit)
    return np.asarray(result)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab, stratum = joined('lab_id'), joined('stratum')
    ref = joined('ref_coupons')
    aligned = joined('coupon_features_raw').astype(float)
    for c in np.unique(lab):
        rows = lab == c
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
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
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (np.exp(-trailing[:, None]/taus) * -np.expm1(-dt[:, None]/taus))
    gap = hold[:, None] - memory
    prony = gap[:, None, :] * -np.expm1(-t[None, :, None]/taus)
    flow = features[:, None, :] * t[None, :, None]
    return np.concatenate((prony, flow), axis=2), np.column_stack((gap/taus, features))


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
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, t,
                         train['strain_curves_obs'] - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']))
    a, b = _design(taus, past[n:], hold[n:], features[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': baseline[n:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np,time
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c03_s00_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_s00.pkl','rb'))[0];d=i['dataset'];g=i['gt'];start=time.monotonic(); o=m.solve_alpha_m97_c03_s00(d);print('runtime',time.monotonic()-start)
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta')]:print(a,np.round(o[a],7),'truth',g[b])
for a,b in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=o[a]-g[b]; print(a,'RMSE',np.sqrt(np.mean(e**2)),'worst time',np.sqrt(np.mean(e**2,axis=0)).max(),'worst cell',max(np.sqrt(np.mean(e[(d['test']['lab_id']==l)&(d['test']['stratum']==s)]**2)) for l in range(4) for s in range(3)))
PY
```
- **Container Exit Code**: `0` (`2383 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.8502572008874267
estimated_taus [0.5998444 4.0692677] truth [0.6   4.081]
hereditary_weights [ 0.6878669  0.5525896  0.1394993  0.1686569 -0.1393592  0.1190643
 -0.0896503] truth [ 0.69  0.55  0.14  0.17 -0.14  0.12 -0.09]
predicted_strain RMSE 0.10210208675271701 worst time 0.16780176992359666 worst cell 0.39242375568522625
initial_rate_pred RMSE 0.032484593059631875 worst time 0.032484593059631875 worst cell 0.12644901420986568

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```

### Turn 5 — `execute_in_remote_docker` (`24857 ms` LLM)
- **Rationale**: Separate complete and missing coupons, and replace fitted parameters with public reference values only in diagnostics. This will distinguish fitting error from alignment or imputation error and show which component is worth changing.
- **Alternatives Weighed**: `Retune the robust optimizer immediately` (rejected: The fitted coefficients already closely match the public reference; first determine whether exact coefficients remove the prediction error.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c03_s00_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with open(p+'public_alpha_m97_c03_s00.pkl','rb') as f:i=pickle.load(f)[0]
d=i['dataset'];g=i['gt'];o=m.solve_alpha_m97_c03_s00(d); test=d['test']; n=len(d['train']['lab_id']); t=d['eval_times']; e=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; missing=np.isnan(test['coupon_features_raw']); past,hold,f,base=m._prepare(d)
for name,mask in [('complete',~missing.any(1)),('missing',missing.any(1))]:
 print(name,mask.sum(),'curve RMSE',np.sqrt(np.mean(e[mask]**2)),'rate RMSE',np.sqrt(np.mean(er[mask]**2)))
for j in np.where(missing.any(1))[0]:
 l,s=test['lab_id'][j],test['stratum'][j]; cell=(test['lab_id']==l)&(test['stratum']==s)
 print('missing row',j,'lab/stratum',l,s,'columns',np.where(missing[j])[0],'drift-error from curve/rate',np.dot(t,e[j])/np.dot(t,t),er[j])
a,b=m._design(g['taus'],past[n:],hold[n:],f[n:],d['past_dt'],t); ec=base[n:,None]+a@g['theta']-g['te_clean']; rc=b@g['theta']-g['te_true_rate0']; print('oracle-parameter RMSE',np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(rc**2)))
print('max calibrated stress deviation',np.max(np.abs(past[n:]-g['te_hist_true'])),np.max(np.abs(hold[n:]-g['te_hold_true']))); print('max train/test calibrated history magnitude',np.max(np.abs(past)))
PY
```
- **Container Exit Code**: `0` (`1288 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete 66 curve RMSE 0.007427947210522975 rate RMSE 0.004286948623328194
missing 6 curve RMSE 0.3528329846169809 rate RMSE 0.1116280798970805
missing row 5 lab/stratum 1 0 columns [1] drift-error from curve/rate 0.204091972900165 0.20727270983265944
missing row 11 lab/stratum 3 0 columns [1] drift-error from curve/rate -0.06037074499618498 -0.06120452933785758
missing row 13 lab/stratum 1 0 columns [3] drift-error from curve/rate -0.06714167260969868 -0.0704811268406127
missing row 27 lab/stratum 3 1 columns [2] drift-error from curve/rate 0.06575936601907217 0.06355496034243935
missing row 31 lab/stratum 2 0 columns [2] drift-error from curve/rate 0.0673934010267832 0.06460719438461804
missing row 54 lab/stratum 1 1 columns [0] drift-error from curve/rate 0.1263420595735474 0.12196756488963789
oracle-parameter RMSE 0.10175548427221649 0.03216722731095101
max calibrated stress deviation 0.00017162815781412988 0.00016435587143703856
max train/test calibrated history magnitude 2.0671716281578143

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29259 ms` LLM)
- **Rationale**: Compare fitting policies on held-out measured labs without selecting against public truth. Check the exact history operator at the relaxation bounds and time the isolated validator without a thread override; a failure would call for a numerical or runtime fix before submission.
- **Alternatives Weighed**: `Replace cell-balanced imputation with a learned predictor` (rejected: The task specifies cell-balanced column means, and exact reference parameters barely change the missing-coupon error.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c03_s00_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with open(p+'public_alpha_m97_c03_s00.pkl','rb') as f:d=pickle.load(f)[0]['dataset']
tr=d['train']; n=len(tr['lab_id']); t=d['eval_times']; dt=d['past_dt']; past,hold,x,base=(v[:n] for v in m._prepare(d)); y=tr['strain_curves_obs']-base[:,None]; rates=tr['instant_rate_obs']; complete=np.isfinite(tr['coupon_features_raw']).all(1)
for lab in range(4):
 val=tr['lab_id']==lab
 for method,fitrows in [('all',~val),('complete',~val & complete)]:
  tau,w=m._fit(past[fitrows],hold[fitrows],x[fitrows],dt,t,y[fitrows],rates[fitrows]); a,b=m._design(tau,past[val],hold[val],x[val],dt,t); ec=np.abs((a@w)-y[val]); er=np.abs((b@w)-rates[val]); print(lab,method,'curve median abs by horizon',np.round(np.median(ec,axis=0),4),'rate median abs',round(np.median(er),4),'missing rate median abs',round(np.median(er[~complete[val]]),4),'taus',np.round(tau,4))
for tau in [np.array([.6,4.]),np.array([dt.min()/20,dt.sum()*20])]:
 z=np.zeros((n,2));
 for j,delta in enumerate(dt):z=np.exp(-delta/tau)*z-np.expm1(-delta/tau)*past[:,j,None]
 a,b=m._design(tau,past,hold,x,dt,t); np.testing.assert_allclose(b[:,:2],(hold[:,None]-z)/tau,rtol=1e-12,atol=1e-12)
print('history operator agrees with sequential recurrence')
PY
/usr/bin/time -f 'validator wall seconds: %e' timeout 30s python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `127` (`5750 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 all curve median abs by horizon [0.0146 0.0084 0.0134 0.0121 0.0102 0.0143 0.0114 0.0119 0.0122 0.0138] rate median abs 0.0113 missing rate median abs 0.1654 taus [0.6007 4.1289]
0 complete curve median abs by horizon [0.0144 0.0086 0.0137 0.0116 0.0106 0.0143 0.0107 0.0126 0.012  0.0142] rate median abs 0.0114 missing rate median abs 0.1658 taus [0.6014 4.1677]
1 all curve median abs by horizon [0.0141 0.0109 0.0133 0.0134 0.0157 0.016  0.0186 0.0162 0.0126 0.0179] rate median abs 0.0154 missing rate median abs 0.1425 taus [0.6003 4.1021]
1 complete curve median abs by horizon [0.0139 0.0109 0.0138 0.0136 0.0152 0.0166 0.019  0.0164 0.0128 0.0168] rate median abs 0.0156 missing rate median abs 0.1421 taus [0.6003 4.0946]
2 all curve median abs by horizon [0.0105 0.0109 0.0139 0.0108 0.0154 0.0171 0.0136 0.0101 0.0136 0.02  ] rate median abs 0.0137 missing rate median abs 0.1412 taus [0.5988 4.0243]
2 complete curve median abs by horizon [0.0106 0.0109 0.0141 0.0108 0.015  0.0174 0.0133 0.0109 0.0124 0.0218] rate median abs 0.0131 missing rate median abs 0.1413 taus [0.5998 4.0882]
3 all curve median abs by horizon [0.015  0.0101 0.0147 0.0155 0.0119 0.0157 0.0179 0.0238 0.0187 0.021 ] rate median abs 0.0108 missing rate median abs 0.1123 taus [0.5996 4.0241]
3 complete curve median abs by horizon [0.015  0.01   0.0147 0.0159 0.0121 0.016  0.0174 0.0236 0.0186 0.0203] rate median abs 0.0106 missing rate median abs 0.1128 taus [0.6004 4.0589]
history operator agrees with sequential recurrence

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```
- **Stderr (tail)**:
```text
bash: line 16: /usr/bin/time: No such file or directory
```

### Turn 7 — `execute_in_remote_docker` (`15060 ms` LLM)
- **Rationale**: The recurrence and held-out lab checks passed. The default-thread timing check did not run because /usr/bin/time was unavailable; use Bash's built-in timer. A timeout would trigger a runtime fix before submission.
- **Alternatives Weighed**: `Exclude incomplete training coupons` (rejected: Held-out lab errors did not improve consistently, and worst-horizon median absolute error increased from 0.0238 to 0.0240.)

```bash
time timeout 30s python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1410 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```
- **Stderr (tail)**:
```text
real	0m1.100s
user	0m0.946s
sys	0m0.070s
```

### Turn 8 — `submit_completed_task` (`14935 ms` LLM)
- **Rationale**: The isolated validator passed on the final artifact, the history operator matched the sequential recurrence, and held-out lab checks supported retaining all training rows. Substituting reference parameters barely reduced public test error, so optimizer retuning would not address the dominant missing-feature error.
- **Alternatives Weighed**: `Exclude training rows with missing coupon features` (rejected: It did not consistently improve held-out lab errors and slightly worsened the worst-horizon strain error.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_s00_solver.py': 'FILE:6563'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c03_s00_solver.py`. The public schema validator passed in 1.1 s. Worst leave-one-lab-out median absolute errors were 0.0238 for strain and 0.0154 for initial rate. Missing coupon features remain the main source of public test error.`

### Turn 9 — `execute_in_remote_docker` (`30955 ms` LLM)
- **Rationale**: Compare aligned and raw features on the same complete coupons in held-out labs so missingness does not confound the comparison. Check the calibrated regime and signed residual drift to identify whether a failure calls for revisiting calibration, alignment, or the flow law. Balanced accuracy does not apply to these continuous outcomes.
- **Alternatives Weighed**: `Fit additional rational, stochastic-drift, or relay terms` (rejected: The specified linear-flow law has no relay or stochastic state. First check whether measured residuals show a repeatable stress-sign effect or remaining lab offset.)

```bash
cat >/tmp/audit_rheology.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p+'public_alpha_m97_c03_s00.pkl', 'rb') as stream:
    d = pickle.load(stream)[0]['dataset']
tr = d['train']
lab, stratum = tr['lab_id'], tr['stratum']
n = len(lab)
t, dt = d['eval_times'], d['past_dt']
past, hold, x, base = (v[:n] for v in m._prepare(d))
y, rates = tr['strain_curves_obs']-base[:, None], tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('Regime: max calibrated |history|, critical stress, banding index:', np.abs(past).max(), d['crit_stress'], d['shear_banding_index'])
# Audit alignment by comparing against raw features with the same physical model.
raw = tr['coupon_features_raw']
models = {'aligned': x, 'raw': np.column_stack((np.ones(n), raw))}
for label, features in models.items():
    err = np.full_like(y, np.nan)
    rate_err = np.full(n, np.nan)
    for c in np.unique(lab):
        fit = complete & (lab != c)
        val = complete & (lab == c)
        taus, w = m._fit(past[fit], hold[fit], features[fit], dt, t, y[fit], rates[fit])
        a, b = m._design(taus, past[val], hold[val], features[val], dt, t)
        err[val] = y[val] - a @ w
        rate_err[val] = rates[val] - b @ w
    cells = [complete & (lab == c) & (stratum == s) for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    print(label, 'worst cell/horizon MedAE', max(np.median(np.abs(err[v]),axis=0).max() for v in cells if v.any()), 'worst cell rate MedAE', max(np.median(np.abs(rate_err[v])) for v in cells if v.any()))
    if label == 'aligned':
        for c in np.unique(lab):
            v = complete & (lab == c)
            drift_error = np.median(err[v]/t, axis=1)
            signed = [np.median(drift_error[q]) for q in [hold[v]<0, hold[v]>=0]]
            print('lab', c, 'median residual drift by hold sign', np.round(signed,5), 'median rate residual', round(float(np.median(rate_err[v])),5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_rheology.py
```
- **Container Exit Code**: `0` (`5199 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Regime: max calibrated |history|, critical stress, banding index: 2.0671716281578143 2.65 0.12
aligned worst cell/horizon MedAE 0.029279312824640358 worst cell rate MedAE 0.015925799372132773
lab 0 median residual drift by hold sign [-0.00045 -0.00154] median rate residual -0.00338
lab 1 median residual drift by hold sign [-0.0007  -0.00258] median rate residual -0.0008
lab 2 median residual drift by hold sign [-0.00017  0.00034] median rate residual -0.00286
lab 3 median residual drift by hold sign [0.00068 0.00202] median rate residual -0.00057
raw worst cell/horizon MedAE 0.5878005561396251 worst cell rate MedAE 0.12123977189620472

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```

### Turn 10 — `execute_in_remote_docker` (`29798 ms` LLM)
- **Rationale**: The real-measurement audit supports the alignment and flow law. The public batch has ten clean calibration points per channel, so it does not exercise the documented fallback. Thin the measured rows and inject spikes just above the threshold to detect coefficient shifts that would justify a calibration-only fix.
- **Alternatives Weighed**: `Add nonlinear flow terms` (rejected: The public data are inside the specified linear regime, and held-out residual drift shows no consistent stress-sign effect.)

```bash
python3 -I - <<'PY'
import importlib.util, pickle, tempfile, pathlib, json
import numpy as np
p = pathlib.Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (p/'public_alpha_m97_c03_s00.pkl').open('rb') as stream:
    d = pickle.load(stream)[0]['dataset']
cal = d['calibration_load_cells']
nominal = m._calibrate(cal)
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp)/'archive_calibration_history.json'
    path.write_text(json.dumps({'stress_gains': nominal[:,0].tolist(), 'stress_biases': nominal[:,1].tolist()}))
    m._ARCHIVE = path
    idx = np.flatnonzero(cal['lab_id'] == 0)
    for count in (0, 1, 2, 3, 4, 10):
        keep = (cal['lab_id'] != 0) | np.isin(np.arange(len(cal['lab_id'])), idx[:count])
        trial = {k: v[keep].copy() for k,v in cal.items()}
        err = np.max(np.abs(m._calibrate(trial)[0] - nominal[0]))
        print('clean rows',count,'max coefficient shift',round(float(err),7))
    for count in (3, 4, 5, 10):
        keep = (cal['lab_id'] != 0) | np.isin(np.arange(len(cal['lab_id'])), idx[:count])
        errors = []
        for spike in (-4., 4.):
            for j in range(count):
                trial = {k: v[keep].copy() for k,v in cal.items()}
                trial['true_stress'][np.flatnonzero(trial['lab_id'] == 0)[j]] += spike
                errors.append(np.max(np.abs(m._calibrate(trial)[0] - nominal[0])))
        print('rows including one EMI spike',count,'worst coefficient shift',round(float(max(errors)),7))
PY
```
- **Container Exit Code**: `0` (`518 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clean rows 0 max coefficient shift 0.0
clean rows 1 max coefficient shift 0.0
clean rows 2 max coefficient shift 8.39e-05
clean rows 3 max coefficient shift 5.91e-05
clean rows 4 max coefficient shift 5.47e-05
clean rows 10 max coefficient shift 0.0
rows including one EMI spike 3 worst coefficient shift 1.937233
rows including one EMI spike 4 worst coefficient shift 1.1781425
rows including one EMI spike 5 worst coefficient shift 7.87e-05
rows including one EMI spike 10 worst coefficient shift 2.02e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed5c7dfefa005d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed5c7df (directory).
```

### Turn 11 — `execute_in_remote_docker` (`44808 ms` LLM)
- **Rationale**: A single spike can tilt a sparse fit enough to evade residual screening. Test leave-one-out hypotheses only when the full sparse fit is inconsistent with the observed calibration precision, and use archive agreement to resolve ambiguous candidates. Stress across all four measured channels to detect missed spikes or unnecessary archive calls before accepting the edit.
- **Alternatives Weighed**: `Always use archive calibration on sparse channels` (rejected: Clean two- and three-reading channels are identifiable in batch and do not meet the archive-use criteria.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_s00_solver.py')
s = p.read_text()
start = s.index('        archived = _archive_calibration(lab) if len(x) < 2 or emi else None')
s = s[:start] + '''        archived = None
        if not emi and 3 <= len(x) <= 4 and np.ptp(x) > 0:
            # A spike can rotate a sparse fit enough to mask its own residual.
            # Do not extrapolate from every pair in an already clean batch.
            fit_all = np.polyfit(x, y, 1)
            if np.max(np.abs(y - np.polyval(fit_all, x))) > 0.01:
                candidates = []
                for j in range(len(x)):
                    keep = np.arange(len(x)) != j
                    if np.ptp(x[keep]) == 0:
                        continue
                    fit_j = np.polyfit(x[keep], y[keep], 1)
                    err = np.abs(y - np.polyval(fit_j, x))
                    if err[j] > 3.0 and np.max(err[keep]) <= 0.01:
                        candidates.append(j)
                if candidates:
                    prior = _archive_calibration(lab)
                    if prior is not None:
                        err = np.abs(y - np.polyval(prior, x))
                        j = np.argmax(err)
                        if j in candidates and err[j] > 3.0 and np.max(np.delete(err, j)) <= 0.01:
                            archived = prior
                    elif len(candidates) == 1:
                        keep = np.arange(len(x)) != candidates[0]
                        x, y = x[keep], y[keep]
        if len(x) < 2 or emi:
            archived = _archive_calibration(lab)
''' + s[start:].split('\n', 1)[1]
p.write_text(s)
PY
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_alpha_m97_c03_s00.pkl','rb') as f: item = pickle.load(f)[0]
d = item['dataset']; cal = d['calibration_load_cells']; nominal = m._calibrate(cal)
calls = []
def archive(lab):
    calls.append(lab)
    return nominal[lab]
m._archive_calibration = archive
for count in (0,1,2,3,4,5,10):
    clean_max = spike_max = 0.0
    for lab in range(4):
        ix = np.flatnonzero(cal['lab_id'] == lab)
        keep = (cal['lab_id'] != lab) | np.isin(np.arange(len(cal['lab_id'])), ix[:count])
        trial = {k: v[keep].copy() for k,v in cal.items()}
        calls.clear()
        clean_max = max(clean_max, np.max(np.abs(m._calibrate(trial)[lab] - nominal[lab])))
        assert calls == ([lab] if count < 2 else []), (count,lab,calls)
        if count >= 3:
            for row in np.flatnonzero(trial['lab_id'] == lab):
                for delta in (-4., -3.05, 3.05, 4.):
                    bad = {k: v.copy() for k,v in trial.items()}
                    bad['true_stress'][row] += delta
                    error = np.max(np.abs(m._calibrate(bad)[lab] - nominal[lab]))
                    if error >= 0.01: print('miss',count,lab,row,delta,round(error,6))
                    spike_max = max(spike_max, error)
    print('rows',count,'clean max',clean_max,'EMI max',spike_max)
    assert clean_max < 0.002 and spike_max < 0.002
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1611 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
rows 0 clean max 0.0 EMI max 0.0
rows 1 clean max 0.0 EMI max 0.0
rows 2 clean max 0.000829267744907547 EMI max 0.0
rows 3 clean max 0.0001521807529185626 EMI max 0.0
miss 4 0 2 -4.0 1.178083
miss 4 0 2 -3.05 0.898281
miss 4 0 2 3.05 0.898341
miss 4 0 2 4.0 1.178143
miss 4 0 3 -4.0 1.063413
miss 4 0 3 -3.05 0.81084
miss 4 0 3 3.05 0.810949
miss 4 0 3 4.0 1.063523
miss 4 1 10 -4.0 3.933513
miss 4 1 10 -3.05 2.999326
miss 4 1 10 3.05 2.999141
miss 4 1 10 4.0 3.933328
miss 4 1 12 -4.0 3.643087
miss 4 1 12 -3.05 2.777832
miss 4 1 12 3.05 2.778017
miss 4 1 12 4.0 3.643273
miss 4 2 21 -4.0 2.173987
miss 4 2 21 -3.05 1.657668
miss 4 2 21 3.05 1.657648
miss 4 2 21 4.0 2.173967
miss 4 2 23 -4.0 2.543375
miss 4 2 23 -3.05 1.939313
miss 4 2 23 3.05 1.9394
miss 4 2 23 4.0 2.543462
miss 4 3 31 -4.0 0.971678
miss 4 3 31 -3.05 0.740917
miss 4 3 31 3.05 0.740807
miss 4 3 31 4.0 0.971567
miss 4 3 33 -4.0 3.329201
miss 4 3 33 -3.05 2.538493
miss 4 3 33 3.05 2.538681
miss 4 3 33 4.0 3.329388
rows 4 clean max 9.352035211272388e-05 EMI max 3.9335131430008987
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4d8df7fc7f885ed2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4d8df7fc (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 30, in <module>
AssertionError
```

### Turn 12 — `execute_in_remote_docker` (`75037 ms` LLM)
- **Rationale**: The previous edit could identify a spike but still count it as a clean point, letting the four-row fit override the archive with contaminated data. Remove the identified reading first. Re-run the failed stress test across all channels and verify the exact artifact in an isolated process; a failure will distinguish residual screening misses from a schema or runtime regression.
- **Alternatives Weighed**: `Always prefer archive coefficients when EMI is suspected` (rejected: The task requires an in-batch fit whenever at least four clean points remain.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_s00_solver.py')
s = p.read_text()
a = s.index('        emi = False')
b = s.index('        result.append(fit)', a)
s = s[:a] + '''        archived = _archive_calibration(lab) if len(x) < 2 else None
        if len(x) >= 3 and np.ptp(x) > 0:
            # Screen against a clean leave-one-out fit: the spike can tilt
            # the full fit enough to hide its own residual.
            full_error = np.abs(y - np.polyval(np.polyfit(x, y, 1), x))
            if full_error.max() > 0.01:
                candidates = []
                for j in range(len(x)):
                    keep = np.arange(len(x)) != j
                    if np.ptp(x[keep]) == 0:
                        continue
                    residual = np.abs(y - np.polyval(np.polyfit(x[keep], y[keep], 1), x))
                    if residual[j] > 3.0 and residual[keep].max() <= 0.01:
                        candidates.append(j)
                if candidates:
                    if len(candidates) > 1 or len(x) <= 4:
                        archived = _archive_calibration(lab)
                    spike = candidates[0] if len(candidates) == 1 else None
                    if spike is None and archived is not None:
                        residual = np.abs(y - np.polyval(archived, x))
                        j = int(np.argmax(residual))
                        if j in candidates and residual[j] > 3.0:
                            spike = j
                    if spike is not None:
                        keep = np.arange(len(x)) != spike
                        x, y = x[keep], y[keep]
                else:
                    i, j = np.triu_indices(len(x), 1)
                    dx = x[j] - x[i]
                    use = dx != 0
                    slope = np.median((y[j] - y[i])[use] / dx[use])
                    offset = np.median(y - slope*x)
                    clean = np.abs(y - (slope*x + offset)) <= 3.0
                    if not clean.all():
                        archived = _archive_calibration(lab)
                        x, y = x[clean], y[clean]
        if len(x) >= 4 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        elif archived is not None:
            fit = archived
        elif len(x) >= 2 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        else:
            fit = (1.0, np.mean(y-x) if len(x) else 0.0)
''' + s[b:]
p.write_text(s)
PY
python3 -I - <<'PY'
import importlib.util, pickle, tempfile, json
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', p/'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (p/'public_alpha_m97_c03_s00.pkl').open('rb') as f: d = pickle.load(f)[0]['dataset']
cal = d['calibration_load_cells']; nominal = m._calibrate(cal)
worst = 0.
with tempfile.TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp)/'archive_calibration_history.json'
    m._ARCHIVE.write_text(json.dumps({'stress_gains': nominal[:,0].tolist(), 'stress_biases': nominal[:,1].tolist()}))
    read = m._archive_calibration; calls = []
    def tracked(lab):
        calls.append(lab)
        return read(lab)
    m._archive_calibration = tracked
    for count in (0,1,2,3,4,5,10):
        clean_max = spike_max = 0.
        for lab in range(4):
            ix = np.flatnonzero(cal['lab_id'] == lab)
            keep = (cal['lab_id'] != lab) | np.isin(np.arange(len(cal['lab_id'])), ix[:count])
            trial = {k: v[keep] for k,v in cal.items()}
            calls.clear()
            fit = m._calibrate(trial)
            assert calls == ([lab] if count < 2 else []), (count, lab, calls)
            clean_max = max(clean_max, np.max(np.abs(fit[lab]-nominal[lab])))
            if count >= 3:
                for j in np.flatnonzero(trial['lab_id'] == lab):
                    for spike in (-4., -3.05, 3.05, 4.):
                        bad = dict(trial, true_stress=trial['true_stress'].copy())
                        bad['true_stress'][j] += spike
                        shift = np.max(np.abs(m._calibrate(bad)[lab]-nominal[lab]))
                        if shift > 0.002:
                            print('miss', count, lab, j, spike, shift)
                        spike_max = max(spike_max, shift)
        print('rows', count, 'clean max', clean_max, 'EMI max', spike_max)
        worst = max(worst, clean_max, spike_max)
    assert worst < 0.002, worst
PY
time timeout 30s python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1713 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
rows 0 clean max 0.0 EMI max 0.0
rows 1 clean max 0.0 EMI max 0.0
rows 2 clean max 0.000829267744907547 EMI max 0.0
rows 3 clean max 0.0001521807529185626 EMI max 0.0
rows 4 clean max 9.352035211272388e-05 EMI max 0.0
rows 5 clean max 0.00012032028737740674 EMI max 0.000250902871512082
rows 10 clean max 0.0 EMI max 3.6459933799434395e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0a87e544f35bb7db (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_0a87e544 (directory).
```
- **Stderr (tail)**:
```text
real	0m1.009s
user	0m0.899s
sys	0m0.057s
```

### Turn 13 — `submit_completed_task` (`15461 ms` LLM)
- **Rationale**: The final candidate passed the isolated schema check and the sparse and EMI calibration tests. Held-out lab measurements support reference-panel alignment, and substituting reference parameters did not materially reduce the missing-coupon error.
- **Alternatives Weighed**: `Prefer archived coefficients whenever a spike is detected` (rejected: The specification requires fitting in batch when at least four clean readings remain.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c03_s00_solver.py`. The isolated public validator passed in 1.0 s. Across all four channels, the largest calibration coefficient shift under the tested EMI spikes was 0.000251. Missing coupon features remain the main source of public test prediction error.`

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
