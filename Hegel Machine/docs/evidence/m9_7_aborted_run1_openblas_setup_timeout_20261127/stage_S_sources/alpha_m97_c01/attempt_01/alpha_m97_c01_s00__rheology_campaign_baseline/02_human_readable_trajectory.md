# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `388,603` (`372,709` / `49,408` / `15,894`) |
| **Total Episode Wall Time (s)** | `490.66s` |

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

### Turn 1 — `execute_in_remote_docker` (`17995 ms` LLM)
- **Rationale**: Inspect it in place to determine the split layout, calibration and reference-panel representation, and whether baseline strain is supplied or must be inferred. Those findings determine how to construct the fit and validation folds.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c01_s00.pkl','rb') as f: d=pickle.load(f)
def describe(x,depth=0):
 if isinstance(x,dict):
  return {k:describe(v,depth+1) for k,v in x.items()} if depth < 2 else list(x)
 if isinstance(x,np.ndarray):
  return (x.shape,str(x.dtype),x[:2].tolist() if x.ndim and x.size and x[0].size<=12 else None)
 if isinstance(x,(tuple,list)):
  return (len(x),[describe(v,depth+1) for v in x[:2]])
 return repr(x)[:180]
for k,v in d.items(): print(k,describe(v))
PY
ls -l /tmp/hegel_continual_archive 2>/dev/null || true
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation; Cauchy robust IRLS; viscoelastic relaxation' 2>&1 | head -50
```
- **Container Exit Code**: `0` (`680 ms`, timed_out=`False`)
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
  File "<stdin>", line 11, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`17836 ms` LLM)
- **Rationale**: The first inspection treated the list of cases as a dictionary. Read the cases in place to determine the calibration layout, panel grouping, and strain anchors; compare the discovered primitives with those requirements before choosing an implementation.

```bash
cd /holdout_task && cat public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))
print('cases',len(items))
def show(x,p='',depth=0):
 if isinstance(x,dict) and depth<5:
  for k,v in x.items(): show(v,p+'/'+str(k),depth+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,x.ravel()[:10].tolist())
 elif isinstance(x,(list,tuple)):
  print(p,len(x),repr(x[:2])[:350])
 else: print(p,repr(x)[:180])
for i,item in enumerate(items):
 print('CASE',i,list(item)); show(item)
PY
ls -l /tmp/hegel_continual_archive 2>/dev/null || true
for f in autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$f"; done
```
- **Container Exit Code**: `0` (`667 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`24369 ms` LLM)
- **Rationale**: The previous output was truncated. Compact summaries will reveal whether calibration needs the EMI fallback, which cells can supply missing-feature means, and what time and strain scales the robust fit must handle.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,importlib.util,os
items=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb')); print('cases',len(items),'scipy',bool(importlib.util.find_spec('scipy')))
for it in items:
 d=it['dataset']; print('item keys',list(it),'dataset keys',list(d)); print('time',d['eval_times'],'dt',np.round(d['past_dt'],3),'crit',d['crit_stress'],'banding',d['shear_banding_index']); print('split shapes',{s:{k:np.shape(v) for k,v in d[s].items()} for s in ['train','test']}); c=d['calibration_load_cells'];
 for lab in np.unique(c['lab_id']):
  m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('cal',lab,len(x),p,'res',np.round(y-np.polyval(p,x),3))
 for s in ['train','test']:
  a=d[s]; print(s,'cells',np.unique(np.column_stack([a['lab_id'],a['stratum']]),axis=0,return_counts=True)[1], 'missing',np.isnan(a['coupon_features_raw']).sum(axis=0),'past range',np.min(a['past_stress_cmd']),np.max(a['past_stress_cmd']));
  for lab in np.unique(a['lab_id']):
   m=a['lab_id']==lab; r=a['ref_coupons'][m]; x=a['coupon_features_raw'][m]; print(lab,'panel medloc/scale',round(np.median(np.nanmean(r,axis=0)),3),round(np.median(np.nanstd(r,axis=0)),3),'x mean',np.round(np.nanmean(x,axis=0),3))
 if 'strain_curves_obs' in d['train']:
  a=d['train']; z=a['strain_curves_obs']-a['strain_0'][:,None]; print('strain delta quantiles by time',np.round(np.nanquantile(z,[0,.1,.5,.9,1],axis=0),4)); print('obs missing',np.isnan(z).sum())
print('archive contents',os.listdir('/tmp/hegel_continual_archive') if os.path.isdir('/tmp/hegel_continual_archive') else 'absent')
PY
```
- **Container Exit Code**: `0` (`393 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
6 0.666 0.22  0.478 0.248 0.784 0.259 0.388 0.274 0.421
 0.534 0.679 0.691 0.435 0.492 0.418] crit 2.65 banding 0.12
split shapes {'train': {'lab_id': (180,), 'stratum': (180,), 'coupon_features_raw': (180, 4), 'ref_coupons': (180, 32), 'past_stress_cmd': (180, 18), 'stress_hold_cmd': (180,), 'strain_0': (180,), 'aux_chamber_telemetry': (180, 3), 'strain_curves_obs': (180, 10), 'instant_rate_obs': (180,)}, 'test': {'lab_id': (72,), 'stratum': (72,), 'coupon_features_raw': (72, 4), 'ref_coupons': (72, 32), 'past_stress_cmd': (72, 18), 'stress_hold_cmd': (72,), 'strain_0': (72,), 'aux_chamber_telemetry': (72, 3)}}
cal 0 10 [9.99975548e-01 2.04663470e-05] res [ 0.  0.  0. -0.  0.  0. -0. -0. -0.  0.]
cal 1 10 [9.99993103e-01 4.97280839e-06] res [-0.  0.  0.  0.  0. -0.  0. -0. -0. -0.]
cal 2 10 [ 1.00001216e+00 -4.56445486e-05] res [-0.  0.  0. -0. -0.  0. -0.  0.  0.  0.]
cal 3 10 [1.00000478e+00 5.05462111e-05] res [-0.  0. -0.  0.  0.  0. -0.  0. -0. -0.]
train cells [16 19 13 12 19 13 19 10 18 13 16 12] missing [7 4 4 5] past range -2.067 2.067
0 panel medloc/scale -0.615 0.807 x mean [-0.762 -0.512 -0.961 -0.55 ]
1 panel medloc/scale 0.484 1.203 x mean [0.655 0.853 0.344 0.402]
2 panel medloc/scale -0.297 0.931 x mean [-0.311 -0.419 -0.316 -0.231]
3 panel medloc/scale 0.869 1.332 x mean [1.219 0.721 1.076 0.554]
test cells [ 6  9  3  4  9  3  7  1 13  5  3  9] missing [1 4 0 2] past range -2.067 2.067
0 panel medloc/scale -0.701 0.791 x mean [-0.637 -0.56  -0.668 -0.923]
1 panel medloc/scale 0.462 1.236 x mean [ 0.319  0.333  0.599 -0.12 ]
2 panel medloc/scale -0.256 0.918 x mean [-0.271  0.041 -0.311  0.142]
3 panel medloc/scale 0.819 1.315 x mean [1.34  1.031 0.915 0.567]
strain delta quantiles by time [[-1.992000e+00 -1.622100e+00 -2.380600e+00 -2.486700e+00 -1.848670e+01
  -3.761900e+00 -3.428500e+00 -1.169650e+01 -1.303400e+01 -9.229100e+00]
 [-6.028000e-01 -1.117500e+00 -1.286000e+00 -1.373200e+00 -1.503500e+00
  -1.737500e+00 -1.566400e+00 -1.672700e+00 -1.717300e+00 -1.918600e+00]
 [ 2.160000e-02  1.553000e-01  1.953000e-01  3.108000e-01  3.582000e-01
   4.486000e-01  4.831000e-01  5.972000e-01  4.703000e-01  7.490000e-01]
 [ 6.370000e-01  1.280700e+00  1.601600e+00  1.913000e+00  2.210600e+00
   2.293000e+00  2.392800e+00  2.818300e+00  2.968600e+00  3.111300e+00]
 [ 8.928000e-01  1.516580e+01  2.603500e+00  2.938100e+00  1.883465e+02
   3.614600e+00  5.588700e+00  4.508000e+00  5.411300e+00  5.353700e+00]]
obs missing 0
archive contents absent
```

### Turn 4 — `execute_in_remote_docker` (`79707 ms` LLM)
- **Rationale**: Fit the specified recurrence and linear terminal flow without using telemetry or test labels. Held-lab and per-horizon errors will reveal whether the fit transfers across calibrated channels; residuals against the independent rate observations will show whether that endpoint needs additional attention.
- **Alternatives Weighed**: `Fit squared error on all strain readings` (rejected: Observed excursions reach 188 strain units, so squared error would give glitches substantial leverage over the time constants.)

```bash
cat > /holdout_task/alpha_m97_c01_s00_solver.py <<'PY'
"""Calibrate and fit the two-branch linear-flow rheology model."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(lab):
    if not _ARCHIVE.is_file():
        return None
    history = json.loads(_ARCHIVE.read_text())
    if isinstance(history, dict):
        history = history.get('calibration_load_cells', history)
        if 'lab_id' in history:
            history = [dict(zip(history, row)) for row in zip(*history.values())]
        else:
            record = history.get(str(lab))
            history = [record] if isinstance(record, dict) else (record or [])
    rows = [r for r in history if int(r.get('lab_id', lab)) == lab]
    affine = [(r['gain'], r['bias']) for r in rows if 'gain' in r and 'bias' in r]
    if affine:
        return np.median(affine, axis=0)
    pairs = [(r['cmd_stress'], r['true_stress']) for r in rows if 'cmd_stress' in r and 'true_stress' in r]
    if len(pairs) >= 2:
        x, y = np.asarray(pairs).T
        if np.ptp(x) > 0:
            return np.polyfit(x, y, 1)
    return None


def _calibrate(c):
    result = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        mask = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][mask], c['true_stress'][mask]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            # A repeated-median line detects gross calibration errors before OLS.
            dx = x[:, None] - x
            dy = y[:, None] - y
            with np.errstate(divide='ignore', invalid='ignore'):
                slopes = [np.median(yi[xi != 0] / xi[xi != 0]) for xi, yi in zip(dx, dy)]
            gain = np.median(slopes)
            bias = np.median(y - gain*x)
            good = np.abs(y - gain*x - bias) <= 3.0
            spikes = not np.all(good)
            if spikes:
                x, y = x[good], y[good]
        historical = _historical_calibration(lab) if len(x) < 2 or spikes else None
        if len(x) >= 2 and np.ptp(x) > 0 and (len(x) >= 4 or historical is None):
            result[lab] = np.polyfit(x, y, 1)
        elif historical is not None:
            result[lab] = historical
        elif len(x):
            result[lab, 1] = np.median(y - x)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    train_size = len(train['lab_id'])
    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab = join('lab_id')
    strata = join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x = join('coupon_features_raw')
    ref = join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        eligible = np.isfinite(panel).sum(axis=0) >= 2
        panel = panel[:, eligible]
        loc = np.median(np.nanmean(panel, axis=0))
        std = np.median(np.nanstd(panel, axis=0))
        x[sel] = (x[sel] - loc)/std
    # Give each observed lab/stratum cell equal total weight.
    _, cell, count = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True, return_counts=True)
    w = 1.0/count[cell, None]
    valid = np.isfinite(x)
    denom = (valid*w).sum(axis=0)
    fill = np.divide((np.where(valid, x, 0.)*w).sum(axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 0)
    x = np.where(valid, x, fill)
    return (past, hold, x), train_size


def _design(logtau, data, dt, time):
    past, hold, x = data
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for width, stress in zip(dt, past.T):
        u = width/tau
        a = np.exp(-u)
        dz = a*dz + (u*a)*(z-stress[:, None])
        z = a*z - np.expm1(-u)*stress[:, None]
    gap = hold[:, None] - z
    u = time[:, None]/tau
    f = -np.expm1(-u)
    a = np.empty((len(hold), len(time), 7))
    a[:, :, :2] = gap[:, None, :]*f
    a[:, :, 2] = time
    a[:, :, 3:] = time[None, :, None]*x[:, None, :]
    jac = -dz[:, None, :]*f - gap[:, None, :]*u*np.exp(-u)
    return a, jac, gap


def _fit(data, dt, time, delta):
    good = np.isfinite(delta)
    y = delta[good]
    # Coarse profile over ordered times, then refine the best separated seeds.
    lo = np.log(max(np.min(dt)*0.12, 1e-4))
    hi = np.log(max(np.max(time), np.sum(dt))*12.)
    grid = np.geomspace(max(np.min(dt)*0.7, 1e-4), max(np.max(time), np.sum(dt))*1.25, 8)
    seeds = []
    scale_y = max(np.median(np.abs(y - np.median(y)))*1.4826, 1e-6)
    for i, short in enumerate(grid[:-1]):
        for long in grid[i+1:]:
            logs = np.log([short, long])
            a = _design(logs, data, dt, time)[0][good]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for factor in (0.20, 0.04):
                resid = y-a@b
                w = 1./np.hypot(1., resid/(scale_y*factor))
                b = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
            score = np.median(np.abs(y-a@b))
            seeds.append((score, np.r_[logs, b]))
    seeds.sort(key=lambda item: item[0])
    best, runner = seeds[0], next((s for s in seeds[1:] if np.linalg.norm(s[1][:2]-seeds[0][1][:2]) > 0.9), seeds[1])
    def residual(p):
        return _design(p[:2], data, dt, time)[0][good]@p[2:] - y
    def jacobian(p):
        a, jac, _ = _design(p[:2], data, dt, time)
        return np.column_stack((jac[good]*p[2:4], a[good]))
    bounds = (np.r_[lo, lo, np.full(7, -np.inf)], np.r_[hi, hi, np.full(7, np.inf)])
    scale = max(1.2*best[0], 1e-7*scale_y)
    solutions = []
    for _, initial in (best, runner):
        p = initial
        for factor in (4., 1.):
            opt = least_squares(residual, p, jac=jacobian, bounds=bounds, x_scale='jac', loss='cauchy', f_scale=scale*factor, max_nfev=120)
            p = opt.x
        solutions.append((opt.cost, p))
    params = min(solutions, key=lambda item: item[0])[1]
    order = np.argsort(params[:2])
    return params[:2][order], np.r_[params[2:4][order], params[4:]]


def _predict(logtau, hw, data, dt, time):
    a, _, gap = _design(logtau, data, dt, time)
    rate = gap@(hw[:2]/np.exp(logtau)) + hw[2] + data[2]@hw[3:7]
    return a@hw, rate


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    dt, times = dataset['past_dt'], dataset['eval_times']
    data, n = _prepare(dataset)
    train, test = dataset['train'], dataset['test']
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, weights = _fit(tuple(v[:n] for v in data), dt, times, delta)
    strain, rate = _predict(logs, weights, tuple(v[n:] for v in data), dt, times)
    return {'estimated_taus': np.exp(logs), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_s00_solver as m
it=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]; d=it['dataset']; g=it['gt']; r=m.solve_alpha_m97_c01_s00(d)
print('gt fields', {k:np.shape(v) for k,v in g.items()})
print('fit',r['estimated_taus'],r['hereditary_weights']); tr=d['train']; all_data,n=m._prepare(d); data=tuple(v[:n] for v in all_data); dt,t=d['past_dt'], d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]
a,b=m._predict(np.log(r['estimated_taus']),r['hereditary_weights'],data,dt,t)
print('fit noise quantiles strain,rate',np.quantile(abs(a-y),[.25,.5,.75,.90,.95]), np.quantile(abs(b-tr['instant_rate_obs']),[.25,.5,.75,.90,.95]))
for lab in np.unique(tr['lab_id']):
 hold=tr['lab_id']==lab; params=m._fit(tuple(v[~hold] for v in data),dt,t,y[~hold]); a,b=m._predict(*params,tuple(v[hold] for v in data),dt,t)
 e=np.abs(a-y[hold]); er=np.abs(b-tr['instant_rate_obs'][hold]); print('held lab',lab,'taus',np.exp(params[0]),'strain med/p90',np.quantile(e,[.5,.9]), 'worst horizon med',np.median(e,axis=0).max(),'rate med/p90',np.quantile(er,[.5,.9]))
PY
```
- **Container Exit Code**: `0` (`1042 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
gt fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
fit [0.54791613 3.75415075] [ 0.69516952  0.54866375  0.13889015  0.16798342 -0.13828147  0.117804
 -0.0890369 ]
fit noise quantiles strain,rate [0.00863822 0.01820816 0.03802236 0.21790063 0.59568547] [0.00718977 0.0168223  0.02759552 0.04469872 0.11053191]
held lab 0 taus [0.54372659 3.7198632 ] strain med/p90 [0.02499171 0.14025603] worst horizon med 0.061803651856243924 rate med/p90 [0.01542942 0.03673475]
held lab 1 taus [0.54871143 3.68711148] strain med/p90 [0.02571772 0.28552368] worst horizon med 0.04969600644255062 rate med/p90 [0.01622319 0.05118986]
held lab 2 taus [0.55128808 3.81412783] strain med/p90 [0.01988654 0.31211916] worst horizon med 0.03451334001267181 rate med/p90 [0.01726468 0.03463089]
held lab 3 taus [0.54991592 3.75847904] strain med/p90 [0.02531539 0.19616958] worst horizon med 0.05230852702543487 rate med/p90 [0.02106446 0.04622824]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=46271427d0e5ff2a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_46271427 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`30803 ms` LLM)
- **Rationale**: The strain-only solver passes the public schema. Compare it with a joint strain-and-rate fit on the same held-lab folds, estimating both noise scales within each training fold. Incorporate rate observations only if they improve held-lab rate prediction without materially worsening the strain endpoint.
- **Alternatives Weighed**: `Fit a separate rate model` (rejected: Its predictions could disagree with the required time-zero derivative of the strain model.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
import alpha_m97_c01_s00_solver as m

d = pickle.load(open('public_alpha_m97_c01_s00.pkl', 'rb'))[0]['dataset']
tr = d['train']
dt, t = d['past_dt'], d['eval_times']
aligned, n = m._prepare(d)
x = tuple(v[:n] for v in aligned)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']

def solve_joint(data, yc, yr, logs, hw):
    a, b = m._predict(logs, hw, data, dt, t)
    def scale(e):
        return max(1e-7, np.median(np.abs(e - np.median(e)))*1.4826)
    sc = scale(a-yc)
    sr = scale(b-yr)
    def resid(p):
        c, v = m._predict(p[:2], p[2:], data, dt, t)
        return np.r_[((c-yc)/sc).ravel(), (v-yr)/sr]
    lo = np.log(max(np.min(dt)*0.12, 1e-4))
    hi = np.log(max(np.max(t), np.sum(dt))*12.)
    opt = least_squares(resid, np.r_[logs, hw], bounds=(np.r_[lo, lo, np.full(7, -np.inf)], np.r_[hi, hi, np.full(7, np.inf)]), x_scale='jac', loss='cauchy', max_nfev=100)
    return opt.x[:2], opt.x[2:]

for lab in np.unique(tr['lab_id']):
    keep = tr['lab_id'] != lab
    a, b = (tuple(v[mask] for v in x) for mask in (keep, ~keep))
    base = m._fit(a, dt, t, y[keep])
    joint = solve_joint(a, y[keep], r[keep], *base)
    for name, p in [('strain', base), ('joint', joint)]:
        yc, yr = m._predict(*p, b, dt, t)
        ec, er = np.abs(yc-y[~keep]), np.abs(yr-r[~keep])
        print(lab, name, 'taus', np.round(np.exp(p[0]),5), 'curve median/worst-horizon median', np.round([np.median(ec), np.median(ec,axis=0).max()],6), 'rate median/p90',np.round(np.quantile(er,[.5,.9]),6))
PY
```
- **Container Exit Code**: `0` (`756 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 strain taus [0.54373 3.71986] curve median/worst-horizon median [0.024992 0.061804] rate median/p90 [0.015429 0.036735]
0 joint taus [0.54518 3.74225] curve median/worst-horizon median [0.025281 0.061126] rate median/p90 [0.015466 0.034943]
1 strain taus [0.54871 3.68711] curve median/worst-horizon median [0.025718 0.049696] rate median/p90 [0.016223 0.05119 ]
1 joint taus [0.54763 3.66958] curve median/worst-horizon median [0.025546 0.049433] rate median/p90 [0.018346 0.049456]
2 strain taus [0.55129 3.81413] curve median/worst-horizon median [0.019887 0.034513] rate median/p90 [0.017265 0.034631]
2 joint taus [0.55059 3.80892] curve median/worst-horizon median [0.019431 0.033306] rate median/p90 [0.017086 0.03404 ]
3 strain taus [0.54992 3.75848] curve median/worst-horizon median [0.025315 0.052309] rate median/p90 [0.021064 0.046228]
3 joint taus [0.54717 3.71923] curve median/worst-horizon median [0.02469  0.050412] rate median/p90 [0.02143  0.043804]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=46271427d0e5ff2a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_46271427 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`38816 ms` LLM)
- **Rationale**: Promote the joint fit without introducing a separate rate model. Verify the exact artifact in isolation and check the time-zero derivative, public reference errors, and missing-coupon slice; a derivative mismatch would require a formula correction, while a large slice error would warrant inspecting alignment and imputation.
- **Alternatives Weighed**: `Retain the strain-only fit` (rejected: The joint fit improved worst-horizon strain error and rate p90 in every held-lab fold; worst-lab rate median increased only from 0.02106 to 0.02143.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_s00_solver.py')
s = p.read_text().replace('def _fit(data, dt, time, delta):', 'def _fit(data, dt, time, delta, rate_obs=None):')
needle = '    order = np.argsort(params[:2])\n'
assert s.count(needle) == 1
s = s.replace(needle, '''    if rate_obs is not None:
        rate_good = np.isfinite(rate_obs)
        if np.any(rate_good):
            _, rate = _predict(params[:2], params[2:], data, dt, time)

            def noise_scale(error):
                return max(1e-7, 1.4826*np.median(np.abs(error - np.median(error))))

            curve_scale = noise_scale(residual(params))
            rate_scale = noise_scale((rate - rate_obs)[rate_good])

            def joint_residual(p):
                curve, rate = _predict(p[:2], p[2:], data, dt, time)
                return np.r_[((curve - delta)[good])/curve_scale,
                             ((rate - rate_obs)[rate_good])/rate_scale]

            params = least_squares(joint_residual, params, bounds=bounds,
                                   x_scale='jac', loss='cauchy', max_nfev=100).x
''' + needle)
old = '    logs, weights = _fit(tuple(v[:n] for v in data), dt, times, delta)'
assert s.count(old) == 1
s = s.replace(old, '''    logs, weights = _fit(tuple(v[:n] for v in data), dt, times, delta,
                         train.get('instant_rate_obs'))''')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for it in pickle.load(open('/holdout_task/public_alpha_m97_c01_s00.pkl', 'rb')):
    d, truth = it['dataset'], it['gt']
    b = m.solve_alpha_m97_c01_s00(d)
    assert all(np.isfinite(v).all() for v in b.values()) and np.all(np.diff(b['estimated_taus']) > 0)
    assert np.array_equal(b['regime_mode_weights'], np.zeros(2))
    te = d['test']; aligned, n = m._prepare(d); data = tuple(v[n:] for v in aligned)
    logs, hw = np.log(b['estimated_taus']), b['hereditary_weights']
    value, _ = m._predict(logs, hw, data, d['past_dt'], np.array([0., 1e-6]))
    np.testing.assert_allclose(value[:, 0], 0.)
    np.testing.assert_allclose(value[:, 1]/1e-6, b['initial_rate_pred'], atol=1e-5, rtol=0)
    error = b['predicted_strain'] - truth['te_clean']; rate_error = b['initial_rate_pred'] - truth['te_true_rate0']
    rmse = lambda a: np.sqrt(np.mean(a*a))
    print('tau relative error', np.round(b['estimated_taus']/truth['taus']-1, 6), 'weight max abs error', np.max(np.abs(hw-truth['theta'])))
    print('strain/rate RMSE', rmse(error), rmse(rate_error), 'worst horizon RMSE', np.sqrt(np.mean(error**2, axis=0)).max())
    for group, masks in [('lab', [(str(lab), te['lab_id']==lab) for lab in np.unique(te['lab_id'])]), ('coupon', [('missing', np.isnan(te['coupon_features_raw']).any(axis=1)), ('complete', np.isfinite(te['coupon_features_raw']).all(axis=1))])]:
        for key, mask in masks:
            if mask.any(): print(group, key, 'n', mask.sum(), 'strain/rate RMSE', rmse(error[mask]), rmse(rate_error[mask]))
PY
```
- **Container Exit Code**: `0` (`920 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
tau relative error [0.001851 0.037316] weight max abs error 0.004616626359409959
strain/rate RMSE 0.140703783323529 0.044005179056491275 worst horizon RMSE 0.231451034456022
lab 0 n 18 strain/rate RMSE 0.13338940144439115 0.04189922200798974
lab 1 n 16 strain/rate RMSE 0.09680098073494363 0.029465529228061317
lab 2 n 21 strain/rate RMSE 0.07861828263564487 0.02466930896066139
lab 3 n 17 strain/rate RMSE 0.22035137061965993 0.06909220627106065
coupon missing n 7 strain/rate RMSE 0.44515124650592125 0.13897331343655
coupon complete n 65 strain/rate RMSE 0.02427613843690416 0.008066782315742834

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=707fc8abd4543e08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_707fc8ab (directory).
```

### Turn 7 — `execute_in_remote_docker` (`31846 ms` LLM)
- **Rationale**: The missing-coupon gap could reflect irreducible feature uncertainty or a fit distorted by noisy horizons and imputed training rows. Compare horizon weighting and complete-row fitting on held-lab observations, with scales estimated inside each fold, before changing the solver.
- **Alternatives Weighed**: `Fit alternative imputations against public test truth` (rejected: That would use held-out targets to choose preprocessing; the documented imputation rule should instead be checked directly.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
import alpha_m97_c01_s00_solver as m

d=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]['dataset']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; data,n=m._prepare(d); data=tuple(a[:n] for a in data)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
lo=np.log(max(np.min(dt)*.12,1e-4)); hi=np.log(max(np.max(t),np.sum(dt))*12)
bounds=(np.r_[lo,lo,np.full(7,-np.inf)],np.r_[hi,hi,np.full(7,np.inf)])
def take(mask): return tuple(v[mask] for v in data)
def scale(e): return np.maximum(1e-7,1.4826*np.median(np.abs(e-np.median(e,axis=0)),axis=0))
results={}
for lab in np.unique(tr['lab_id']):
    hold=tr['lab_id']==lab; train=~hold
    base=m._fit(take(train),dt,t,y[train],rate[train]); p0=np.r_[base[0],base[1]]
    trusted=train&complete
    yp,rp=m._predict(*base,take(trusted),dt,t)
    horizon=scale(yp-y[trusted]); rate_scale=scale(rp-rate[trusted]); print('lab',lab,'complete-row training noise by horizon',np.round(horizon,4),'rate',round(rate_scale,4))
    for label in ['baseline','per_horizon','complete_only','per_horizon_complete']:
        p=p0
        if label!='baseline':
            fit=trusted if 'complete' in label else train
            c_scale=horizon if 'horizon' in label else np.median(horizon)
            def residual(p):
                a,b=m._predict(p[:2],p[2:],take(fit),dt,t)
                return np.r_[((a-y[fit])/c_scale).ravel(),(b-rate[fit])/rate_scale]
            p=least_squares(residual,p0,bounds=bounds,x_scale='jac',loss='cauchy',max_nfev=100).x
        a,b=m._predict(p[:2],p[2:],take(hold),dt,t); c=complete[hold]
        ec,er=np.abs(a-y[hold]),np.abs(b-rate[hold]); s=[np.median(ec,axis=0).max(),np.quantile(er,.9),np.median(ec[c],axis=0).max(),np.quantile(er[c],.9),np.median(ec[~c]),np.median(er[~c])]
        results.setdefault(label,[]).append(s); print(label,np.round(np.exp(p[:2]),5),np.round(s,6))
for label,v in results.items(): print('WORST',label,np.round(np.max(v,axis=0),6))
for key in ['train','test']:
    print(key,'missing by lab',[(int(l),int(np.sum((d[key]['lab_id']==l)&np.isnan(d[key]['coupon_features_raw']).any(axis=1)))) for l in np.unique(d[key]['lab_id'])])
PY
```
- **Container Exit Code**: `0` (`1014 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.0219 0.0284 0.025  0.0247] rate 0.0225
baseline [0.54518 3.74225] [0.061126 0.034943 0.057785 0.030336 0.092944 0.02033 ]
per_horizon [0.5452 3.7359] [0.060719 0.034947 0.057588 0.030226 0.092966 0.020326]
complete_only [0.54557 3.74409] [0.061938 0.035266 0.057788 0.031154 0.091201 0.020983]
per_horizon_complete [0.54558 3.73529] [0.06117  0.035261 0.057409 0.03107  0.090733 0.020891]
lab 1 complete-row training noise by horizon [0.017  0.0173 0.0187 0.019  0.0207 0.0208 0.0246 0.0217 0.0283 0.0283] rate 0.0202
baseline [0.54763 3.66958] [0.049433 0.049456 0.042356 0.034779 0.233774 0.099551]
per_horizon [0.54744 3.66828] [0.048781 0.049178 0.041958 0.034529 0.234084 0.099563]
complete_only [0.54733 3.6778 ] [0.048882 0.049673 0.04283  0.03426  0.234208 0.099739]
per_horizon_complete [0.5473  3.68265] [0.047192 0.049258 0.042235 0.034152 0.234687 0.099744]
lab 2 complete-row training noise by horizon [0.0198 0.0191 0.0194 0.019  0.0248 0.0209 0.025  0.0292 0.0299 0.0329] rate 0.0206
baseline [0.55059 3.80892] [0.033306 0.03404  0.028679 0.028526 0.440514 0.15015 ]
per_horizon [0.55033 3.80595] [0.030711 0.033879 0.028994 0.028033 0.440116 0.150097]
complete_only [0.55094 3.84136] [0.035718 0.034353 0.03063  0.028882 0.44067  0.150176]
per_horizon_complete [0.55078 3.84081] [0.032278 0.034089 0.029535 0.028432 0.440349 0.150141]
lab 3 complete-row training noise by horizon [0.0181 0.0175 0.0207 0.0184 0.0214 0.0194 0.0229 0.0268 0.0305 0.0275] rate 0.0181
baseline [0.54717 3.71923] [0.050412 0.043804 0.046313 0.041768 0.419897 0.176604]
per_horizon [0.54681 3.70919] [0.049689 0.043363 0.045744 0.041584 0.418812 0.176436]
complete_only [0.54765 3.74434] [0.05078  0.044516 0.046069 0.042195 0.4203   0.17591 ]
per_horizon_complete [0.54754 3.74047] [0.048921 0.044236 0.045115 0.041995 0.418519 0.1757  ]
WORST baseline [0.061126 0.049456 0.057785 0.041768 0.440514 0.176604]
WORST per_horizon [0.060719 0.049178 0.057588 0.041584 0.440116 0.176436]
WORST complete_only [0.061938 0.049673 0.057788 0.042195 0.44067  0.17591 ]
WORST per_horizon_complete [0.06117  0.049258 0.057409 0.041995 0.440349 0.1757  ]
train missing by lab [(0, 6), (1, 6), (2, 4), (3, 4)]
test missing by lab [(0, 2), (1, 1), (2, 3), (3, 1)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=707fc8abd4543e08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_707fc8ab (directory).
```

### Turn 8 — `execute_in_remote_docker` (`32731 ms` LLM)
- **Rationale**: Use the supplied reference parameters as a diagnostic, not to fit the solver: if the missing-coupon gap persists, changing the estimator will not address it. Exercise the documented sparse and EMI calibration branches, which the public schema run does not cover; a failure would require a calibration fix before submission.
- **Alternatives Weighed**: `Add per-horizon loss scaling` (rejected: It changes worst held-lab horizon median error by only 0.0004 and does not materially improve the missing-coupon slice.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
it = pickle.load(open('/holdout_task/public_alpha_m97_c01_s00.pkl','rb'))[0]
d, g = it['dataset'], it['gt']
aligned, n = m._prepare(d)
a = tuple(v[n:] for v in aligned)
print('public calibration max abs errors:', np.max(np.abs(a[0]-g['te_hist_true'])), np.max(np.abs(a[1]-g['te_hold_true'])))
y, r = m._predict(np.log(g['taus']), g['theta'], a, d['past_dt'], d['eval_times'])
y += d['test']['strain_0'][:,None]
for missing in (False, True):
    rows = np.isnan(d['test']['coupon_features_raw']).any(axis=1) == missing
    print('reference parameters, missing:', missing, 'strain and rate RMSE:', np.sqrt(np.mean((y[rows]-g['te_clean'][rows])**2)), np.sqrt(np.mean((r[rows]-g['te_true_rate0'][rows])**2)))
with tempfile.TemporaryDirectory() as td:
    m._ARCHIVE = Path(td)/'archive_calibration_history.json'
    m._ARCHIVE.write_text('{"1":{"gain":1.25,"bias":-0.1}}')
    lab = np.repeat([0, 2, 3], 7)
    x = np.tile(np.linspace(-2, 2, 7), 3)
    expected = np.array([[0.90, 0.12], [1.25, -0.1], [1.05, 0.2], [0.95, -0.08]])
    y = expected[lab, 0]*x + expected[lab, 1]
    y[0] += 12
    np.testing.assert_allclose(m._calibrate({'lab_id':lab, 'cmd_stress':x, 'true_stress':y}), expected, atol=1e-12)
    m._ARCHIVE.write_text('not JSON')
    lab = np.repeat(np.arange(4), 7)
    x = np.tile(np.linspace(-2, 2, 7), 4)
    y = expected[lab, 0]*x + expected[lab, 1]
    np.testing.assert_allclose(m._calibrate({'lab_id':lab, 'cmd_stress':x, 'true_stress':y}), expected, atol=1e-12)
print('Calibration stress cases passed: EMI, absent in-batch channel, and no archive access for clean calibration.')
PY
```
- **Container Exit Code**: `0` (`452 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public calibration max abs errors: 7.100918360203323e-05 6.791443310216749e-05
reference parameters, missing: False strain and rate RMSE: 0.02876607209205059 0.009092030832438997
reference parameters, missing: True strain and rate RMSE: 0.43609948815367094 0.13785187748465136
Calibration stress cases passed: EMI, absent in-batch channel, and no archive access for clean calibration.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=707fc8abd4543e08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_707fc8ab (directory).
```

### Turn 9 — `submit_completed_task` (`16296 ms` LLM)
- **Rationale**: The exact artifact passed isolated verification, including the sparse-channel and EMI calibration cases. Substituting the supplied reference parameters still left a missing-coupon strain RMSE of 0.4361, showing that parameter estimation is not the main source of that gap.
- **Alternatives Weighed**: `Add per-horizon loss scaling` (rejected: It improved the worst held-lab horizon median by only 0.0004 and did not materially improve the missing-coupon slice.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_s00_solver.py': 'FILE:7930'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema, time-zero derivative, and calibration stress tests passed. Public strain RMSE was 0.0243 for complete coupons and 0.4452 for coupons with missing features.`

### Turn 10 — `execute_in_remote_docker` (`43984 ms` LLM)
- **Rationale**: Compare the recurrence with its closed form, verify the linear-regime bounds, and test whether nonlinear terms or training-lab nuisance slopes materially improve held-lab predictions. A recurrence mismatch calls for a formula fix; a consistent held-lab improvement calls for investigating the corresponding preprocessing or model assumption.
- **Alternatives Weighed**: `Apply classification, relay-mode, or Fokker–Planck checks` (rejected: This task specifies continuous strain and rate under a deterministic creep law, with no class labels, relay states, or probability-density dynamics.)

```bash
cat > /tmp/rheology_structure_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_s00.pkl','rb'))[0]['dataset']
tr = d['train']
dt,t = d['past_dt'],d['eval_times']
all_data,n = m._prepare(d)
past,hold,x = (v[:n] for v in all_data)
print('regime: banding',d['shear_banding_index'],'max calibrated |past|',np.max(np.abs(all_data[0])),'critical',d['crit_stress'])
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
# Constitutive alternatives in the same coordinates; each correction vanishes at t=0.
def make_basis(logs):
    a,_,gap = m._design(logs,(past,hold,x),dt,t)
    taus = np.exp(logs)
    expected_gap = []
    for tau in taus:
        elapsed_after = np.cumsum(dt[::-1])[::-1]-dt
        expected_gap.append(hold - past @ (np.exp(-elapsed_after/tau)*(-np.expm1(-dt/tau))))
    np.testing.assert_allclose(gap, np.column_stack(expected_gap), atol=2e-14, rtol=0)
    s=(gap[:,0]-gap[:,1]); q=(gap[:,0]+gap[:,1]); h=hold
    terms = [s*q,h*s,h*q, s*q*h, h**2, h/(1+s*s)-h]
    terms += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
    c = np.column_stack(terms)
    rate = np.column_stack((gap/taus, np.ones(len(x)), x))
    return a,rate,a,t[None,:,None]*c[:,None,:],c

def scales(e):
    return max(1e-7, 1.4826*np.median(np.abs(e-np.median(e))))

def refit(a,b,y0,r0,keep,theta):
    def residual(p):
        return np.r_[((a[keep]@p-y[keep])/y0).ravel(),(b[keep]@p-r[keep])/r0]
    return least_squares(residual,theta,loss='cauchy',x_scale='jac',max_nfev=100).x

scores={}
for l in np.unique(lab):
    keep = lab!=l
    te = lab==l
    logs,w = m._fit(tuple(v[keep] for v in (past,hold,x)),dt,t,y[keep],r[keep])
    a,b,_,c,cr = make_basis(logs)
    clean = keep & complete
    sc,sr = scales(a[clean]@w-y[clean]),scales(b[clean]@w-r[clean])
    ac,bc = np.concatenate((a,c),axis=-1),np.column_stack((b,cr))
    # Only training labs get fixed-effect slopes; the withheld lab receives the physical slope.
    u = np.unique(lab[keep]); dummies = (lab[:,None]==u[None,1:]).astype(float)
    an = np.concatenate((a,t[None,:,None]*dummies[:,None,:]),axis=-1)
    bn = np.column_stack((b,dummies))
    wn = refit(an,bn,sc,sr,keep,np.r_[w,np.zeros(dummies.shape[1])])
    wc = refit(ac,bc,sc,sr,keep,np.r_[w,np.zeros(cr.shape[1])])
    signed = np.median((a[te]@w-y[te])[complete[te]],axis=0)
    print('lab',l,'complete-coupon median residual first,last:',np.round(signed[[0,-1]],6),'training nuisance slope range:',np.round(np.ptp(np.r_[0.,wn[7:]]),6))
    for label,A,B,p in [('baseline',a,b,w),('nonlinear',ac,bc,wc),('lab_nuisance',a,b,wn[:7])]:
        e = np.abs(A[te]@p-y[te]); er = np.abs(B[te]@p-r[te]); ok=complete[te]
        metrics=[np.median(e,axis=0).max(),np.quantile(er,.9),np.median(e[ok],axis=0).max(),np.quantile(er[ok],.9)]
        scores.setdefault(label,[]).append(metrics)
        print(label,np.round(metrics,6))
for label,rows in scores.items():
    print('WORST',label,np.round(np.max(rows,axis=0),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_structure_audit.py
```
- **Container Exit Code**: `0` (`891 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
regime: banding 0.12 max calibrated |past| 2.0670707734505207 critical 2.65
lab 0 complete-coupon median residual first,last: [0.003431 0.008822] training nuisance slope range: 0.002522
baseline [0.061126 0.034943 0.057785 0.030336]
nonlinear [0.058144 0.035177 0.052843 0.027145]
lab_nuisance [0.061906 0.035666 0.060385 0.031363]
lab 1 complete-coupon median residual first,last: [-0.003763  0.004872] training nuisance slope range: 0.003584
baseline [0.049433 0.049456 0.042356 0.034779]
nonlinear [0.04415  0.049195 0.038613 0.034657]
lab_nuisance [0.051179 0.051665 0.044366 0.035251]
lab 2 complete-coupon median residual first,last: [0.002919 0.004196] training nuisance slope range: 0.002458
baseline [0.033306 0.03404  0.028679 0.028526]
nonlinear [0.028339 0.036293 0.02804  0.026214]
lab_nuisance [0.034015 0.033696 0.028225 0.029598]
lab 3 complete-coupon median residual first,last: [0.000254 0.010021] training nuisance slope range: 0.001536
baseline [0.050412 0.043804 0.046313 0.041768]
nonlinear [0.047297 0.042702 0.039199 0.037236]
lab_nuisance [0.055553 0.045326 0.052705 0.040797]
WORST baseline [0.061126 0.049456 0.057785 0.041768]
WORST nonlinear [0.058144 0.049195 0.052843 0.037236]
WORST lab_nuisance [0.061906 0.051665 0.060385 0.040797]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=707fc8abd4543e08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_707fc8ab (directory).
```

### Turn 11 — `submit_completed_task` (`18178 ms` LLM)
- **Rationale**: The recurrence matched its independent closed form, and the unchanged artifact passed isolated verification. The audit found no calibration or model change that materially improves the worst slices while preserving the specified constitutive law.
- **Alternatives Weighed**: `Add nonlinear flow terms` (rejected: They lowered the worst held-lab horizon median from 0.0611 to 0.0581 but depart from the specified linear-flow law. The public measurements are within both linear-regime bounds.); `Add lab-specific drift slopes` (rejected: They worsened both the worst held-lab horizon median and the worst held-lab rate p90 after calibration and alignment.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_s00_solver.py': 'FILE:7930'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema, time-zero derivative, and sparse-channel and EMI calibration tests passed. Public strain RMSE was 0.0243 for complete coupons and 0.4452 for coupons with missing features; using the reference parameters still gave 0.4361 on the missing-feature slice.`

### Turn 12 — `execute_in_remote_docker` (`42486 ms` LLM)
- **Rationale**: The reference-parameter diagnostic ruled out parameter error as the main source of the missing-coupon gap, but did not test the imputation rule. Compare cell-normalization choices on actual missing rows and intentionally masked measured features, without using public test targets to choose a method. Change the solver only if the evidence supports a specification-consistent improvement.
- **Alternatives Weighed**: `Repeat the nonlinear and lab-specific drift audit` (rejected: The recurrence already matched its closed form, lab-specific drift worsened the principal held-lab errors, and nonlinear flow departs from the specified law.)

```bash
cat > /tmp/rheology_fill_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_s00.pkl','rb'))[0]['dataset']
tr = d['train']
raw, panel, lab, stratum = [np.concatenate([d[s][k] for s in ('train','test')]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')]
aligned, n = m._prepare(d)
past, hold, reference = aligned
x = raw.copy()
for l in np.unique(lab):
    sel = lab==l
    p = panel[sel]
    p = p[:,np.isfinite(p).sum(axis=0)>=2]
    x[sel] = (x[sel]-np.median(np.nanmean(p,axis=0)))/np.median(np.nanstd(p,axis=0))
keys, cell = np.unique(np.c_[lab,stratum],axis=0,return_inverse=True)
def fill(x, method):
    means = np.array([np.nanmean(x[cell==j],axis=0) for j in range(len(keys))])
    equal = np.nanmean(means,axis=0)
    if method == 'row_weight':
        counts=np.bincount(cell)
        return np.nansum(x/counts[cell,None],axis=0)/np.sum(np.isfinite(x)/counts[cell,None],axis=0)
    if method == 'equal_observed_cell':
        return equal
    return np.where(np.isfinite(means[cell]),means[cell],equal)
np.testing.assert_allclose(reference,np.where(np.isfinite(x),x,fill(x,'row_weight')))
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
dt,t = d['past_dt'],d['eval_times']
missing = np.isnan(x[:n]).any(axis=1)
methods=('row_weight','equal_observed_cell','local_cell')
scores={method:[] for method in methods}
for l in np.unique(lab[:n]):
    keep = lab[:n]!=l
    h = ~keep
    # Reconstruct one masked feature at a time on held-lab complete rows.
    # The actual missing-row evaluation below also checks the fit end to end.
    complete = np.flatnonzero(h & ~missing)
    for method in methods:
        a = np.where(np.isfinite(x),x,fill(x,method))
        data = (past[:n],hold[:n],a[:n])
        logs,w=m._fit(tuple(v[keep] for v in data),dt,t,y[keep],r[keep])
        c,b=m._predict(logs,w,tuple(v[h] for v in data),dt,t)
        e,er=np.abs(c-y[h]),np.abs(b-r[h])
        synthetic=[]
        for j in range(x.shape[1]):
            for i in complete:
                masked=x.copy()
                masked[i,j]=np.nan
                f=fill(masked,method)
                synthetic.append((f[j] if f.ndim==1 else f[i,j])-x[i,j])
        row=[np.median(e,axis=0).max(),np.quantile(er,.9),np.median(e[missing[h]]),np.median(er[missing[h]]),np.sqrt(np.mean(np.square(synthetic)))]
        scores[method].append(row)
        print('lab',l,method,np.round(row,6))
for method,rows in scores.items():
    print('WORST',method,np.round(np.max(rows,axis=0),6))
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_fill_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1965 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 row_weight [0.061126 0.034943 0.092944 0.02033  0.984164]
lab 0 equal_observed_cell [0.061118 0.034942 0.093906 0.020576 0.983901]
lab 0 local_cell [0.061917 0.040954 0.118703 0.030082 0.984427]
lab 1 row_weight [0.049433 0.049456 0.233774 0.099551 1.02946 ]
lab 1 equal_observed_cell [0.04942  0.049285 0.233046 0.099416 1.029704]
lab 1 local_cell [0.049255 0.055155 0.173271 0.091452 1.058344]
lab 2 row_weight [0.033306 0.03404  0.440514 0.15015  1.031369]
lab 2 equal_observed_cell [0.033285 0.034027 0.44024  0.149995 1.031292]
lab 2 local_cell [0.034893 0.032249 0.309576 0.09823  1.020936]
lab 3 row_weight [0.050412 0.043804 0.419897 0.176604 1.030629]
lab 3 equal_observed_cell [0.050324 0.043797 0.419197 0.176583 1.030495]
lab 3 local_cell [0.049646 0.047319 0.425411 0.174228 0.995828]
WORST row_weight [0.061126 0.049456 0.440514 0.176604 1.031369]
WORST equal_observed_cell [0.061118 0.049285 0.44024  0.176583 1.031292]
WORST local_cell [0.061917 0.055155 0.425411 0.174228 1.058344]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=707fc8abd4543e08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_707fc8ab (directory).
```

### Turn 13 — `execute_in_remote_docker` (`39432 ms` LLM)
- **Rationale**: Implement equal weighting of observed cell means directly. The held-lab audit found no material regression from this correction. Verify that the final artifact implements the grouping rule and retains the public schema, time-zero derivative, and slice accuracy.
- **Alternatives Weighed**: `Keep inverse cell-size row weights` (rejected: Missing entries reduce a cell's contribution to that feature, so observed cell means do not receive equal weight.); `Fill from the coupon's own lab and stratum` (rejected: It worsened worst held-lab rate p90 from 0.04946 to 0.05516 and was less accurate on deliberately masked measurements.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_s00_solver.py')
s = p.read_text()
start = s.index('    # Give each observed lab/stratum cell equal total weight.')
end = s.index('    x = np.where(valid, x, fill)', start)
s = s[:start] + '''    # Each cell with an observation contributes one mean per feature.
    cells, cell = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True)
    valid = np.isfinite(x)
    sums = np.zeros((len(cells), x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(valid, x, 0.))
    np.add.at(counts, cell, valid)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    observed_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), observed_cells,
                     out=np.zeros(x.shape[1]), where=observed_cells > 0)
''' + s[end:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
it = pickle.load(open('public_alpha_m97_c01_s00.pkl', 'rb'))[0]
d, g = it['dataset'], it['gt']
data, n = m._prepare(d)
raw, ref, lab, strata = [np.concatenate([d[k][name] for k in ('train','test')]) for name in ('coupon_features_raw','ref_coupons','lab_id','stratum')]
x = raw.copy()
for l in np.unique(lab):
    sel = lab == l
    panel = ref[sel]; panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
    x[sel] = (x[sel] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0))
keys, cell = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True)
means = np.array([np.nanmean(x[cell == i], axis=0) for i in range(len(keys))])
np.testing.assert_allclose(data[2], np.where(np.isfinite(x), x, np.nanmean(means, axis=0)))
out = m.solve_alpha_m97_c01_s00(d)
assert all(np.isfinite(v).all() for v in out.values())
assert np.all(np.diff(out['estimated_taus']) > 0)
assert np.array_equal(out['regime_mode_weights'], np.zeros(2))
logs, w = np.log(out['estimated_taus']), out['hereditary_weights']
test_data = tuple(v[n:] for v in data)
short, _ = m._predict(logs,w,test_data,d['past_dt'],np.array([0.,1e-6]))
np.testing.assert_allclose(short[:,0],0.)
np.testing.assert_allclose(short[:,1]/1e-6,out['initial_rate_pred'],atol=1e-5,rtol=0.)
err = out['predicted_strain']-g['te_clean']; rate_err = out['initial_rate_pred']-g['te_true_rate0']
print('max relative tau error', np.max(np.abs(out['estimated_taus']/g['taus']-1)), 'max absolute weight error',np.max(np.abs(w-g['theta'])))
print('worst horizon strain RMSE',np.sqrt(np.mean(err**2,axis=0)).max())
missing = np.isnan(raw[n:]).any(axis=1)
for name, mask in [('complete',~missing),('missing',missing)]+[(f'lab {l}',lab[n:]==l) for l in np.unique(lab[n:])]:
    print(name, 'strain/rate RMSE',np.sqrt(np.mean(err[mask]**2)),np.sqrt(np.mean(rate_err[mask]**2)))
PY
```
- **Container Exit Code**: `0` (`970 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
max relative tau error 0.0372735082903044 max absolute weight error 0.004625598700016886
worst horizon strain RMSE 0.23159478716959553
complete strain/rate RMSE 0.024283649547854894 0.008067991425432363
missing strain/rate RMSE 0.44542980581289576 0.13905862864918736
lab 0 strain/rate RMSE 0.13375656031933092 0.04201393107437055
lab 1 strain/rate RMSE 0.09696973373570347 0.02951776917923684
lab 2 strain/rate RMSE 0.0788470410516579 0.02474197698511727
lab 3 strain/rate RMSE 0.22017979875078653 0.06903656602143164

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2c9364dd7cd219a0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_2c9364dd (directory).
```

### Turn 14 — `submit_completed_task` (`15890 ms` LLM)
- **Rationale**: The final artifact passed isolated verification of the schema, alignment, and time-zero derivative. Equal weighting implements the specified cell balance without a material validation regression, and the recurrence matched its independent closed form.
- **Alternatives Weighed**: `Retain inverse cell-size row weights` (rejected: Missing entries would reduce a cell's contribution to that feature instead of giving each observed cell mean equal weight.); `Add lab-specific drift slopes` (rejected: They worsened both principal worst-lab error metrics.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema, time-zero derivative, and sparse-channel and EMI calibration checks passed. Public strain RMSE was 0.0243 for complete coupons and 0.4454 for coupons with missing features.`

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
