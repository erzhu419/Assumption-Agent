# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m96_w02_s00__coupon_creep_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M96-Autonomous-Structure-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_m96_w02_source` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `371,304` (`355,191` / `83,328` / `16,113`) |
| **Total Episode Wall Time (s)** | `495.55s` |

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

### Turn 1 — `execute_in_remote_docker` (`18046 ms` LLM)
- **Rationale**: Inspect the public input and validator so the implementation uses the actual split, shapes, and calibration conventions. Check archive and primitive availability before deciding how to handle sparse calibration checks and strain glitches.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np,os
with open('public_alpha_m96_w02_s00.pkl','rb') as f: d=pickle.load(f)
def brief(d,depth=0):
 for k,v in d.items():
  print('  '*depth,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''), (str(v)[:200] if np.isscalar(v) else (len(v) if hasattr(v,'__len__') else '')))
  if isinstance(v,dict): brief(v,depth+1)
  if isinstance(v,(tuple,list)) and v: print('  '*depth,'first',str(v[0])[:350])
brief(d)
p='/tmp/hegel_continual_archive'; print('Archive:',os.listdir(p) if os.path.isdir(p) else 'not mounted')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference panel alignment imputation' 2>&1 | head -35
```
- **Container Exit Code**: `0` (`686 ms`, timed_out=`False`)
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
  File "<stdin>", line 8, in <module>
  File "<stdin>", line 4, in brief
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`20481 ms` LLM)
- **Rationale**: Identify the actual split layout, missing-feature pattern and archive format. Review whether the available primitives match the required per-lab alignment and cell-balanced imputation before implementing them.
- **Alternatives Weighed**: `Implement against assumed dictionary nesting` (rejected: The first inspection encountered a list at the root; inspect its entries and the validator call before defining the solver's input handling.)

```bash
cd /holdout_task && ls -lh && tail -42 validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np,os,json
with open('public_alpha_m96_w02_s00.pkl','rb') as f: d=pickle.load(f)
def brief(v,p='root',depth=0):
 if isinstance(v,dict):
  print(p,'dict',list(v));
  for k,x in v.items(): brief(x,p+'.'+str(k),depth+1)
 elif isinstance(v,(tuple,list)):
  print(p,type(v).__name__,len(v))
  if v and depth<3: brief(v[0],p+'[0]',depth+1)
 elif isinstance(v,np.ndarray):
  a=v.ravel();print(p,v.shape,v.dtype,'sample',a[:8], 'nonfinite',np.count_nonzero(~np.isfinite(a)) if np.issubdtype(v.dtype,np.number) else '-')
 else: print(p,repr(v)[:250])
brief(d)
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
if os.path.exists(p):
 print('archive size',os.path.getsize(p));
 with open(p) as f: print(f.read(3000))
else: print('calibration archive absent')
PY
for skill in autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$skill" | head -28; done
```
- **Container Exit Code**: `0` (`617 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
erence_panel_balancing.weighted_column_fill",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_reference_panel_balancing",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "a0bd4b961c67e2ccad723d94cee31eee33e031d5925b39c96b71ace643383209",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "Alignment accepts real numeric arrays of shapes (n, p) and (n, q), with p, q > 0, and a finite real group vector of shape (n,). Nonfinite panel entries are treated as missing. Each group must have at least one control with two finite observations, a finite median control mean, and a finite, strictly positive median control population standard deviation. A constant control can contribute; a zero median gain raises ValueError.",
        "Interpreting alignment as removal of a nuisance transform assumes a shared additive offset and positive multiplicative gain for signals and controls within each group. The control panel must have comparable baseline location and spread across groups, independent of sample composition. These scientific assumptions cannot be verified by shape or moment checks.",
        "Cell weighting requires two nonempty, finite real vectors of matching shape (n,). Every combination of an observed group and an observed stratum must be present. If there are K such cells, each row in a cell of size m receives weight n/(K*m).",
        "Column fill requires a nonempty real matrix of shape (n, p) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
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
```

### Turn 3 — `execute_in_remote_docker` (`20457 ms` LLM)
- **Rationale**: The last output truncated the dataset summary. Inspect only the fields needed to distinguish sparse calibration from outliers, map the training observations, and confirm the optimizer is available.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, os, numpy as np
p=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))
print('public cases',len(p))
for c in p:
 d=c['dataset']; print('case fields',list(c),'dataset fields',list(d))
 for name in ['train','test','calibration_load_cells','eval_times','yield_stress']:
  v=d.get(name); print(name,type(v).__name__)
  if isinstance(v,dict):
   for k,x in v.items():
    a=np.asarray(x); print(' ',k,a.shape,a.dtype,a.ravel()[:5])
   if 'lab_id' in v: print(' counts:',np.unique(v['lab_id'],return_counts=True))
   if name=='calibration_load_cells':
    for lab in np.unique(v['lab_id']):
     m=np.asarray(v['lab_id'])==lab; print('  lab',lab,np.c_[v['cmd_stress'],v['true_stress']][m].tolist()[:25])
  elif isinstance(v,list): print('  len',len(v),'first',str(v[:2])[:350])
  else: print(' ',repr(v)[:250])
f='/tmp/hegel_continual_archive/archive_calibration_history.json'; print('archive:',open(f).read(2200) if os.path.isfile(f) else 'absent')
import scipy; print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
07 -0.09280564]
  stress_hold_cmd (80,) float64 [ 0.49370627  1.44768973 -2.02670874  0.26665289  0.96238665]
  instant_rate_obs (80,) float64 [ 2.54421578 -0.46576313 -3.27283474  2.74142386  1.57616249]
  residual_plastic_state (80,) float64 [0. 0. 0. 0. 0.]
 counts: (array([0, 1, 2, 3]), array([18, 21, 16, 25]))
calibration_load_cells dict
  lab_id (40,) int64 [0 0 0 0 0]
  cmd_stress (40,) float64 [ 0.10243588  2.27153697 -0.39013684  1.66300299  1.19246503]
  true_stress (40,) float64 [ 0.10251766  2.27145135 -0.39029502  1.66310826  1.19256097]
 counts: (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
  lab 0 [[0.10243588173962292, 0.10251765576701409], [2.271536967323592, 2.2714513459120087], [-0.3901368350209129, -0.3902950198881394], [1.663002993098691, 1.6631082615931536], [1.1924650251026074, 1.192560968994837], [0.07025838599986622, 0.0703005044258471], [-0.5626504952663804, -0.5626307858680979], [-0.7079133152907002, -0.7076695930136058], [1.5209126610404016, 1.5208581435425084], [2.2640157855944807, 2.2641097473795457]]
  lab 1 [[-1.7685669447438546, -1.7681501943898525], [0.5080096882565717, 0.5079922704649497], [-1.707781306160187, -1.7078470085328372], [2.302804253252617, 2.302691160039904], [2.138134082387346, 2.1382257236733224], [2.258919510070635, 2.2588505930914664], [-0.9642175754315669, -0.964211777000023], [-0.7482860387089001, -0.7483930300244426], [-0.4956348959442587, -0.4954736095593176], [-1.1682527741478606, -1.168259084861551]]
  lab 2 [[0.8848863790636887, 0.8848480717180784], [-0.3319641887978175, -0.33206043987611433], [-0.08617708515324507, -0.08637874600552291], [2.162353751433418, 2.162327591332684], [-1.9078114759214164, -1.9078615855141707], [0.6981514941851676, 0.6983337996791484], [0.6810710681413794, 0.681358283943802], [1.8946795510041552, 1.8945202402220442], [0.3151599329308117, 0.31528704319495143], [0.08761534070466714, 0.08755322193331827]]
  lab 3 [[-0.4747891157018027, -0.4748044031630739], [-1.783433395958736, -1.7835026947890595], [0.7247468440085938, 0.7247566881418807], [1.895774031987525, 1.8957772323019697], [-0.31270076973547445, -0.3129277114411474], [0.789909565367859, 0.7898916066070741], [-1.741794097307289, -1.7416982468185147], [0.22873170228886508, 0.22864946534181102], [2.1009066009447723, 2.1009118330553718], [1.1928607071086703, 1.1931447814071001]]
eval_times ndarray
  array([0.25, 0.8 , 1.35, 1.9 , 2.45, 3.  , 3.55, 4.1 , 4.65, 5.2 ])
yield_stress float
  2.95
archive: absent
scipy 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`74698 ms` LLM)
- **Rationale**: Implement the specified recurrence and its analytic derivative, with robust fits for calibration spikes and strain glitches. The first clean-process run checks the output contract; public reference fields and onset residuals will identify which slices need further validation.
- **Alternatives Weighed**: `Treat the measured onset rate as a test prediction` (rejected: The contract calls for the derivative of the calibrated hold law; use training rates as observations and compare test rates as a diagnostic.)

```bash
cd /holdout_task && cat > alpha_m96_w02_s00_solver.py <<'PY'
"""Calibrated sub-yield creep fit for the multi-wing coupon stream."""
import numpy as np
from scipy.optimize import least_squares


def _stress_calibration(checks):
    lab = np.asarray(checks['lab_id'])
    cmd = np.asarray(checks['cmd_stress'], float)
    true = np.asarray(checks['true_stress'], float)
    params = np.tile([1., 0.], (4, 1))
    for c in range(4):
        use = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[use], true[use]
        if not len(x):
            continue
        i, j = np.triu_indices(len(x), 1)
        distinct = abs(x[i] - x[j]) > 1e-8
        i, j = i[distinct], j[distinct]
        if not len(i):
            params[c, 1] = np.median(y - x)
            continue
        slopes = (y[i] - y[j]) / (x[i] - x[j])
        errors = y[None, :] - slopes[:, None] * x[None, :]
        intercepts = np.median(errors, axis=1)
        errors -= intercepts[:, None]
        best = np.argmin(np.median(abs(errors), axis=1))
        cutoff = max(7e-4, 4.5 * 1.4826 * np.median(abs(errors[best])))
        inliers = abs(errors[best]) <= cutoff
        params[c] = np.linalg.lstsq(np.column_stack((x[inliers], np.ones(sum(inliers)))), y[inliers], rcond=None)[0]
    return params


def _align(train, test):
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    stratum = np.concatenate((train['stratum'], test['stratum']))
    x = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0))
        x[use] = (x[use] - offset) / gain
    # Give each observed lab/stratum cell equal row mass before excluding missing values.
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    valid = np.isfinite(x)
    mass = np.sum(w * valid, axis=0)
    fill = np.divide(np.sum(w * np.where(valid, x, 0.), axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0.)
    return np.where(valid, x, fill)


def _inputs(data, x, calibration):
    gain, bias = calibration[np.asarray(data['lab_id'])].T
    past = gain[:, None] * data['past_stress_cmd'] + bias[:, None]
    hold = gain * data['stress_hold_cmd'] + bias
    return past, hold, np.column_stack((np.ones(len(x)), x))


class _Design:
    def __init__(self, dt, t, inputs):
        self.dt, self.t = np.asarray(dt), np.asarray(t)
        self.past, self.hold, self.x = inputs
        self.base = np.empty((len(self.x), len(t) + 1, 7))
        self.base[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:] = self.x

    def make(self, taus, derivative=False):
        a = self.base.copy()
        if derivative:
            da = np.empty(a.shape[:2] + (2,))
        for m, tau in enumerate(taus):
            z = np.zeros(len(self.x))
            dz = np.zeros_like(z)
            e = np.exp(-self.dt / tau)
            f = -np.expm1(-self.dt / tau)
            de = e * self.dt / tau**2
            for j in range(len(e)):
                if derivative:
                    dz = e[j] * dz + de[j] * (z - self.past[:, j])
                z = e[j] * z + f[j] * self.past[:, j]
            gap = self.hold - z
            h = -np.expm1(-self.t / tau)
            a[:, :-1, m] = gap[:, None] * h
            a[:, -1, m] = gap / tau
            if derivative:
                dh = -np.exp(-self.t / tau) * self.t / tau**2
                da[:, :-1, m] = gap[:, None] * dh - dz[:, None] * h
                da[:, -1, m] = -dz / tau - gap / tau**2
        return (a, da) if derivative else a


def _fit(design, curves, initial_rate):
    y = np.column_stack((curves, initial_rate))
    def scales(r):
        def mad(a):
            return 1.4826 * np.median(abs(a - np.median(a)))
        s = max(1e-7, mad(r[:, :-1]))
        return np.r_[np.full(y.shape[1] - 1, s), max(1e-7, mad(r[:, -1]))]

    def linear_start(a):
        a = a.reshape(-1, 7)
        yflat = y.ravel()
        b = np.linalg.lstsq(a, yflat, rcond=None)[0]
        for _ in range(3):
            r = (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)
            s = 2.5 * scales(r)
            w = (1. / np.hypot(s, r)).ravel()
            b = np.linalg.lstsq(a * w[:, None], yflat * w, rcond=None)[0]
        return b, (np.einsum('ij,j->i', a, b) - yflat).reshape(y.shape)

    seeds = []
    grid = (.22, .38, .68, 1.20, 2.15, 3.8, 6.5, 7.9)
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            b, r = linear_start(design.make((t1, t2)))
            seeds.append((np.r_[t1, t2, b], r))
    pilot = min(seeds, key=lambda sr: np.median(abs(sr[1][:, :-1])))
    scale = 2.5 * scales(pilot[1])
    bounds = (np.r_[.20, .20, np.full(7, -np.inf)], np.r_[8., 8., np.full(7, np.inf)])

    def score(r):
        return np.sum(np.log1p((r / scale)**2))

    def optimize(p):
        def residual(q):
            a = design.make(q[:2])
            return ((np.einsum('ntj,j->nt', a, q[2:]) - y) / scale).ravel()

        def jac(q):
            a, da = design.make(q[:2], derivative=True)
            j = np.concatenate((da * q[2:4], a), axis=2)
            return (j / scale[None, :, None]).reshape(-1, 9)

        return least_squares(residual, p, jac=jac, bounds=bounds, loss='cauchy',
                             tr_solver='exact', x_scale='jac', max_nfev=120,
                             ftol=2e-9, xtol=2e-9, gtol=2e-7)

    seeds.sort(key=lambda sr: score(sr[1]))
    fitted = min((optimize(p) for p, _ in seeds[:4]), key=lambda result: result.cost)
    scale = 2.5 * scales(fitted.fun.reshape(y.shape) * scale)
    p = optimize(fitted.x).x
    if p[0] > p[1]:
        p[[0, 1, 2, 3]] = p[[1, 0, 3, 2]]
    if p[0] == p[1]:
        if p[1] < 8.:
            p[1] = np.nextafter(p[1], np.inf)
        else:
            p[0] = np.nextafter(p[0], -np.inf)
    return p[:2], p[2:]


def solve_alpha_m96_w02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    cal = _stress_calibration(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    dtr = _Design(dt, t, _inputs(train, x[:n], cal))
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, theta = _fit(dtr, curves, train['instant_rate_obs'])
    dte = _Design(dt, t, _inputs(test, x[n:], cal))
    pred = np.einsum('ntj,j->nt', dte.make(tau), theta)
    return dict(estimated_taus=tau, hereditary_weights=theta,
                viscoplastic_weights=np.zeros(2),
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,time
import alpha_m96_w02_s00_solver as s
case=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d=case['dataset']; g=case['gt']
print('dt',np.asarray(d['past_dt']).shape,'missing train/test',*[np.isnan(v['coupon_features_raw']).sum(axis=0) for v in (d['train'],d['test'])]); print('GT fields',[(k,np.shape(v)) for k,v in g.items()]);
a=time.perf_counter(); o=s.solve_alpha_m96_w02_s00(d); print('runtime',time.perf_counter()-a)
for k,v in o.items():
 if v.ndim==1 and len(v)<8: print(k,v)
 if k in g and np.shape(v)==np.shape(g[k]): print('  RMSE',np.sqrt(np.mean((v-g[k])**2)))
print('test rate obs minus pred, abs quantiles',np.quantile(abs(d['test']['instant_rate_obs']-o['initial_rate_pred']),[.25,.5,.75,.9,.95,.99,1]))
PY
```
- **Container Exit Code**: `0` (`853 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
dt (18,) missing train/test [6 2 4 4] [2 4 0 3]
GT fields [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
runtime 0.07656293991021812
estimated_taus [0.54979587 3.99767435]
hereditary_weights [ 0.70563325  0.52887556  0.09121502  0.15780708 -0.12853684  0.10859704
 -0.07898253]
viscoplastic_weights [0. 0.]
test rate obs minus pred, abs quantiles [0.00292086 0.00597425 0.01136478 0.02667872 0.08997843 0.14870712
 0.15687117]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=851578da00250aac (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_851578da (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27423 ms` LLM)
- **Rationale**: Check whether the rate residual comes from missing features, a lab-specific calibration error, or unstable fitted parameters. Score each output against its public reference and refit with each lab held out to test transfer independently of the public test labels.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m96_w02_s00_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; out=s.solve_alpha_m96_w02_s00(d)
rmse=lambda a: np.sqrt(np.mean(np.square(a)))
e=out['predicted_strain']-g['te_clean']; r=out['initial_rate_pred']-g['te_true_rate0']; tr,te=d['train'],d['test']
print('taus abs errors',abs(out['estimated_taus']-g['taus']),'theta abs errors',abs(out['hereditary_weights']-g['theta']),'vp exact',np.array_equal(out['viscoplastic_weights'],g['vp_theta']))
assert .20<=out['estimated_taus'][0]<out['estimated_taus'][1]<=8 and np.array_equal(out['viscoplastic_weights'],g['vp_theta'])
for label,m in [(f'lab {j}',te['lab_id']==j) for j in range(4)]+[(f'stratum {j}',te['stratum']==j) for j in np.unique(te['stratum'])]+[('missing',np.isnan(te['coupon_features_raw']).any(axis=1)),('complete',~np.isnan(te['coupon_features_raw']).any(axis=1))]:
 print('test',label,'n',sum(m),'strain/rate RMSE',rmse(e[m]),rmse(r[m]))
print('horizon test RMSE',np.sqrt(np.mean(e**2,axis=0)))
cal=s._stress_calibration(d['calibration_load_cells']); hist,hold,_=s._inputs(te,s._align(tr,te)[len(tr['lab_id']):],cal)
print('stress max abs error hist/hold',np.max(abs(hist-g['te_hist_true'])),np.max(abs(hold-g['te_hold_true'])))
for j in range(4):
 m=tr['lab_id']==j
 fold={**d,'train':{k:v[~m] for k,v in tr.items()},'test':{k:v[m] for k,v in tr.items() if k!='strain_curves_obs'}}
 o=s.solve_alpha_m96_w02_s00(fold); q=o['predicted_strain']-tr['strain_curves_obs'][m]; z=o['initial_rate_pred']-tr['instant_rate_obs'][m]
 print('leave lab out',j,'taus',o['estimated_taus'],'strain/rate MAE medians',np.median(abs(q)),np.median(abs(z)),'endpoints',np.median(abs(q[:,[0,-1]]),axis=0))
PY
```
- **Container Exit Code**: `0` (`814 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
taus abs errors [0.00079587 0.03467435] theta abs errors [0.00163325 0.00087556 0.00121502 0.00219292 0.00146316 0.00140296
 0.00101747] vp exact True
test lab 0 n 18 strain/rate RMSE 0.10777752899108009 0.03440286315294982
test lab 1 n 21 strain/rate RMSE 0.11909757604195048 0.03779167497843205
test lab 2 n 16 strain/rate RMSE 0.09603360227016434 0.030374056960169364
test lab 3 n 25 strain/rate RMSE 0.07080948913972093 0.022528619905787077
test stratum 0 n 26 strain/rate RMSE 0.0719893459233227 0.022912611762200025
test stratum 1 n 34 strain/rate RMSE 0.1281028992531601 0.04072677475870576
test stratum 2 n 20 strain/rate RMSE 0.06601876152091801 0.020858791144359897
test missing n 9 strain/rate RMSE 0.2939867322209308 0.09341488438774635
test complete n 71 strain/rate RMSE 0.005323159797127179 0.0017164742432530724
horizon test RMSE [0.00784236 0.02508796 0.04232469 0.05955898 0.07679473 0.09403295
 0.11127344 0.12851571 0.14575926 0.16300367]
stress max abs error hist/hold 0.00011658517255908052 0.00010323840522463712
leave lab out 0 taus [0.550302   4.05794372] strain/rate MAE medians 0.01136901712315641 0.007688185542010034 endpoints [0.00844322 0.01639019]
leave lab out 1 taus [0.55024974 4.0478922 ] strain/rate MAE medians 0.010887942926047216 0.006862086493796049 endpoints [0.00825398 0.02315654]
leave lab out 2 taus [0.55010206 4.02347419] strain/rate MAE medians 0.016007478703394203 0.01211193914151154 endpoints [0.00790066 0.03658367]
leave lab out 3 taus [0.54995648 3.99796967] strain/rate MAE medians 0.01593282180356595 0.005941725267016584 endpoints [0.00694631 0.03057669]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=851578da00250aac (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_851578da (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32130 ms` LLM)
- **Rationale**: The supplied test onset measurement may identify the scalar drift lost to imputation without changing the reference alignment or shared fit. Test that prediction on real and deliberately masked held-out coupons before editing the solver; retain mean-only prediction if onset conditioning does not improve held-out strain.
- **Alternatives Weighed**: `Refit the hereditary times and weights` (rejected: The complete-feature slice already fits well, and the error on incomplete rows grows in proportion to time, implicating the linear drift rather than the hereditary branches.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; t=d['eval_times']; rmse=lambda e: np.sqrt(np.mean(e**2))
def audit(name,o,b,truth,ref):
 missing=np.isnan(b['coupon_features_raw']).any(axis=1); available=missing & np.isfinite(b['instant_rate_obs']); correction=np.where(available,b['instant_rate_obs']-o['initial_rate_pred'],0.)
 ep=o['predicted_strain']-truth; ec=ep+correction[:,None]*t; er=o['initial_rate_pred']-ref; rc=er+correction
 print(name,'missing',sum(missing),'max correction',np.max(abs(correction)), 'missing strain median error',np.median(abs(ep[missing])),np.median(abs(ec[missing])),'missing rate RMSE',rmse(er[missing]),rmse(rc[missing]), 'missing late strain median error',np.median(abs(ep[missing,-1])),np.median(abs(ec[missing,-1])))
 return ep,ec,er,rc
for j in range(4):
 m=tr['lab_id']==j; b={k:v[m] for k,v in tr.items() if k!='strain_curves_obs'}; fold=dict(d,train={k:v[~m] for k,v in tr.items()},test=b)
 o=s.solve_alpha_m96_w02_s00(fold); audit(f'held-out lab {j}',o,b,tr['strain_curves_obs'][m],b['instant_rate_obs'])
 # Mask one known feature per complete held-out coupon; keep the same measured curves.
 m0=~np.isnan(b['coupon_features_raw']).any(axis=1); v={k:a[m0] for k,a in b.items()}; v['coupon_features_raw']=v['coupon_features_raw'].copy(); v['coupon_features_raw'][np.arange(sum(m0)),(np.arange(sum(m0))+j)%4]=np.nan
 q=s.solve_alpha_m96_w02_s00(dict(fold,test=v)); audit(f'held-out lab {j}, masked',q,v,tr['strain_curves_obs'][m][m0],v['instant_rate_obs'])
o=s.solve_alpha_m96_w02_s00(d); ep,ec,er,rc=audit('clean public',o,te,g['te_clean'],g['te_true_rate0']); missing=np.isnan(te['coupon_features_raw']).any(axis=1)
print('clean missing strain RMSE',rmse(ep[missing]),rmse(ec[missing]),'late horizon RMSE',rmse(ep[:,-1]),rmse(ec[:,-1]))
print('clean missing rate noise max',np.max(abs(te['instant_rate_obs'][missing]-g['te_true_rate0'][missing])))
PY
```
- **Container Exit Code**: `0` (`1126 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0132789 0.0 missing late strain median error 0.5973720403421869 0.05557123187073765
held-out lab 0, masked missing 47 max correction 0.23194793845792416 missing strain median error 0.15342064596350757 0.017002273801559564 missing rate RMSE 0.09963507414093438 0.0 missing late strain median error 0.3366226163078827 0.04518155664003386
held-out lab 1 missing 2 max correction 0.1973376390679682 missing strain median error 0.09564220487261621 0.01921080060865743 missing rate RMSE 0.13957294589148753 0.0 missing late strain median error 0.5798312175128753 0.05539916554058677
held-out lab 1, masked missing 47 max correction 0.34343638373698937 missing strain median error 0.2153787495992427 0.016863391183119143 missing rate RMSE 0.1295882044763103 0.0 missing late strain median error 0.5706313640689271 0.03313697569763774
held-out lab 2 missing 4 max correction 0.231006136669752 missing strain median error 0.10346897773135179 0.0206540535457157 missing rate RMSE 0.11876999394425229 0.0 missing late strain median error 0.14033217960552816 0.01964484850293846
held-out lab 2, masked missing 43 max correction 0.3437545119331811 missing strain median error 0.19492453050781633 0.018161479454281704 missing rate RMSE 0.12399097882837878 0.0 missing late strain median error 0.4843388621730922 0.04575784508356129
held-out lab 3 missing 3 max correction 0.1828441663394278 missing strain median error 0.1745483142929472 0.017012529198286545 missing rate RMSE 0.11150689590272962 0.0 missing late strain median error 0.36038391189886165 0.02952825166197559
held-out lab 3, masked missing 47 max correction 0.3136284616058521 missing strain median error 0.2272244827435903 0.01819620075761212 missing rate RMSE 0.12955340192364093 0.0 missing late strain median error 0.45352023805137964 0.034340785871794566
clean public missing 9 max correction 0.15687117315753263 missing strain median error 0.16831866303538248 0.007196135175035784 missing rate RMSE 0.09341488438774635 0.007483770504305796 missing late strain median error 0.4060302929834011 0.01884935032306112
clean missing strain RMSE 0.2939867322209308 0.023460481139638938 late horizon RMSE 0.16300366686840692 0.015412209338270751
clean missing rate noise max 0.013955557296820409

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=851578da00250aac (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_851578da (directory).
```

### Turn 7 — `execute_in_remote_docker` (`39590 ms` LLM)
- **Rationale**: Promote the targeted correction after it improved strain predictions in all four held-out labs. Verify the exact artifact preserves complete rows, applies the correction only to linear drift, and still produces finite predictions without onset measurements. Check whether the named calibration archive is available before deciding how to handle sparse spot checks.
- **Alternatives Weighed**: `Condition every test coupon on onset` (rejected: The complete-feature rate RMSE is already 0.00172; using the noisier measured onset there would inject avoidable strain error.)
- **ATMS NOGOOD DECLARED**: Refuted `Cell-balanced mean imputation alone recovers coupon-specific linear drift when a feature is missing.` | Witness: `Complete-feature strain RMSE was 0.00532, but missing-feature RMSE was 0.294; conditioning on onset reduced real missing-feature strain median absolute error in each of four held-out labs.` | Blanket Diagnosis: `Missing features leave a coupon-specific linear drift unresolved. The resulting error grows linearly with time, while the calibrated hereditary branches remain predictive.` | Sublated Pivot: `Keep reference alignment and the shared hereditary fit; reconstruct only the unresolved drift from finite measured onset on incomplete test rows.`

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m96_w02_s00_solver.py')
s = p.read_text()
old = "    pred = np.einsum('ntj,j->nt', dte.make(tau), theta)\n    return dict(estimated_taus=tau, hereditary_weights=theta,"
new = "    pred = np.einsum('ntj,j->nt', dte.make(tau), theta)\n    # Missing features affect only the linear drift. The measured onset\n    # identifies that drift after the hereditary onset has been accounted for.\n    if 'instant_rate_obs' in test:\n        onset = np.asarray(test['instant_rate_obs'])\n        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(onset)\n        drift_correction = onset[use] - pred[use, -1]\n        pred[use, :-1] += drift_correction[:, None] * t\n        pred[use, -1] += drift_correction\n    return dict(estimated_taus=tau, hereditary_weights=theta,"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
path='/holdout_task/alpha_m96_w02_s00_solver.py'
spec=importlib.util.spec_from_file_location('solver',path); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']
o=s.solve_alpha_m96_w02_s00(d)
rmse=lambda e: np.sqrt(np.mean(e**2))
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; missing=np.isnan(te['coupon_features_raw']).any(axis=1)
for name,m in [('missing',missing),('complete',~missing)]+[(f'lab {j}',te['lab_id']==j) for j in range(4)]+[(f'stratum {j}',te['stratum']==j) for j in np.unique(te['stratum'])]:
 print(name,'strain/rate RMSE',rmse(e[m]),rmse(r[m]))
print('worst horizon RMSE',max(np.sqrt(np.mean(e**2,axis=0))))
assert rmse(e[missing]) < .035 and rmse(r[missing]) < .012 and rmse(e[~missing]) < .007
assert np.max(abs(o['estimated_taus']-g['taus'])) < .06 and np.max(abs(o['hereditary_weights']-g['theta'])) < .004
assert np.array_equal(o['viscoplastic_weights'],g['vp_theta'])
no_onset={k:v for k,v in te.items() if k!='instant_rate_obs'}; fallback=s.solve_alpha_m96_w02_s00(dict(d,test=no_onset))
for k in ['predicted_strain','initial_rate_pred']:
 assert np.all(np.isfinite(fallback[k])) and np.array_equal(o[k][~missing],fallback[k][~missing])
np.testing.assert_allclose(o['predicted_strain']-fallback['predicted_strain'],(o['initial_rate_pred']-fallback['initial_rate_pred'])[:,None]*d['eval_times'],atol=5e-15)
for j in range(4):
 m=tr['lab_id']==j; test={k:v[m] for k,v in tr.items() if k!='strain_curves_obs'}; held=s.solve_alpha_m96_w02_s00(dict(d,train={k:v[~m] for k,v in tr.items()},test=test)); miss=np.isnan(test['coupon_features_raw']).any(axis=1)
 q=np.median(abs(held['predicted_strain'][miss]-tr['strain_curves_obs'][m][miss])); print('held out lab',j,'missing strain median abs error',q); assert q<.03
PY
ls -d /tmp/hegel_continual_archive /pretrain_task 2>/dev/null; grep -inE 'archive|spot check|EMI' public_asset_index.json | cut -c1-250 | head -15
```
- **Container Exit Code**: `0` (`1241 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
missing strain/rate RMSE 0.023460481139638938 0.007483770504305796
complete strain/rate RMSE 0.005323159797127179 0.0017164742432530724
lab 0 strain/rate RMSE 0.005370307359396204 0.001650606477547554
lab 1 strain/rate RMSE 0.013171174639567901 0.004203604985680502
lab 2 strain/rate RMSE 0.00983974269055473 0.003215202942753041
lab 3 strain/rate RMSE 0.007082589511675654 0.002260352514415351
stratum 0 strain/rate RMSE 0.004815891779037501 0.0015578949610759625
stratum 1 strain/rate RMSE 0.01342610889814314 0.004284534611271459
stratum 2 strain/rate RMSE 0.003417154990068073 0.0011401432710579648
worst horizon RMSE 0.015412209338270751
held out lab 0 missing strain median abs error 0.016939237130229795
held out lab 1 missing strain median abs error 0.019210800608657375
held out lab 2 missing strain median abs error 0.02065405354571559
held out lab 3 missing strain median abs error 0.017012529198286552
/pretrain_task

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d47ed7248ea41af7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d47ed724 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`26302 ms` LLM)
- **Rationale**: The held-out strain results verify the drift correction. Stress-test calibration separately: if sparse or spiked checks cause material errors, use archive evidence where available rather than changing the already validated creep fit.
- **Alternatives Weighed**: `Submit the verified nominal fit now` (rejected: The task explicitly names sparse checks and Wing B spikes; neither calibration slice has been measured yet.)

```bash
cd /holdout_task && sed -n '1,160p' public_asset_index.json && find /pretrain_task /tmp/hegel_continual_archive -maxdepth 3 -type f \( -iname '*calib*' -o -iname '*manifest*' \) -printf '%p %s bytes\n' 2>/dev/null | head -25; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; b=d['calibration_load_cells']; labs=np.asarray(b['lab_id']); target=np.column_stack((g['stress_gains'],g['stress_biases'])); print('calibration counts',np.bincount(labs,minlength=4));
for keep in (None, 3, 2, 1):
 mask=np.ones(len(labs),dtype=bool)
 if keep is not None:
  for lab in (2,3):
   idx=np.flatnonzero(labs==lab); order=idx[np.argsort(np.asarray(b['cmd_stress'])[idx])]; mask[idx]=False; mask[order[np.linspace(0,len(order)-1,keep).round().astype(int)]]=True
 q={k:np.asarray(v)[mask].copy() for k,v in b.items()};
 if keep is None:
  for lab in (2,3):
   idx=np.flatnonzero(q['lab_id']==lab); q['true_stress'][idx[:2]] += [0.24,-0.31]
 p=s._stress_calibration(q); o=s.solve_alpha_m96_w02_s00(dict(d,calibration_load_cells=q)); m=d['test']['lab_id']>=2
 print('Wing B checks', 'full + 2 spikes/lab' if keep is None else keep, 'max gain/bias errors',np.max(abs(p[2:]-target[2:]),axis=0),'strain/rate RMSE',*[np.sqrt(np.mean((o[k][m]-g[r][m])**2)) for k,r in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]])
PY
```
- **Container Exit Code**: `0` (`772 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
              "stratum": {
                "dtype": "int64",
                "shape": [
                  80
                ],
                "type": "ndarray"
              },
              "stress_hold_cmd": {
                "dtype": "float64",
                "shape": [
                  80
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "coupon_features_raw": {
                "dtype": "float64",
                "shape": [
                  200,
                  4
                ],
                "type": "ndarray"
              },
              "instant_rate_obs": {
                "dtype": "float64",
                "shape": [
                  200
                ],
                "type": "ndarray"
              },
              "lab_id": {
                "dtype": "int64",
                "shape": [
                  200
                ],
                "type": "ndarray"
              },
              "past_stress_cmd": {
                "dtype": "float64",
                "shape": [
                  200,
                  18
                ],
                "type": "ndarray"
              },
              "ref_coupons": {
                "dtype": "float64",
                "shape": [
                  200,
                  32
                ],
                "type": "ndarray"
              },
              "strain_0": {
                "dtype": "float64",
                "shape": [
                  200
                ],
                "type": "ndarray"
              },
              "strain_curves_obs": {
                "dtype": "float64",
calibration counts [10 10 10 10]
Wing B checks full + 2 spikes/lab max gain/bias errors [1.76788905e-05 9.33620190e-06] strain/rate RMSE 0.008273050425718414 0.0026753191246958984
Wing B checks 3 max gain/bias errors [1.88688415e-05 5.22033069e-05] strain/rate RMSE 0.008271404028887703 0.0026750836738545837
Wing B checks 2 max gain/bias errors [1.91875431e-05 3.88837301e-05] strain/rate RMSE 0.00827233804402074 0.0026747755274886824
Wing B checks 1 max gain/bias errors [0.00000000e+00 6.92988303e-05] strain/rate RMSE 0.008282715681292787 0.0026812527184691035

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d47ed7248ea41af7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d47ed724 (directory).
```

### Turn 9 — `submit_completed_task` (`16895 ms` LLM)
- **Rationale**: The exact artifact passed isolated schema verification and improved incomplete-coupon strain predictions in all four held-out labs without changing complete-row predictions. Sparse and spiked Wing B checks kept strain RMSE below 0.0083 on the public stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m96_w02_s00_solver.py': 'FILE:7398'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m96_w02_s00_solver.py`. The public schema validator passed. Using measured onset to correct drift on incomplete coupons reduced public strain RMSE from 0.294 to 0.0235. The calibration archive was unavailable; sparse and spiked checks were tested using the public data, but cross-epoch calibration was not verified.`

### Turn 10 — `execute_in_remote_docker` (`38881 ms` LLM)
- **Rationale**: Cross-check the recurrence and onset derivative against an independent closed convolution; a discrepancy would require correcting the physical design. Re-express the same measurements in nonidentity load-cell coordinates because the original sparse stress test used unit gains and could not expose a unit-gain fallback failure. A material one-check degradation would warrant evaluating whether the training response can identify relative gain without an archive.
- **Alternatives Weighed**: `Add rational terms or test relay modes and classification balance` (rejected: This is continuous, sub-yield regression with a specified hereditary law; there are no relay modes or classification labels. Lab-specific signed residuals are the relevant nuisance diagnostic.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
p='/holdout_task/alpha_m96_w02_s00_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; t=d['eval_times']; o=s.solve_alpha_m96_w02_s00(d)
rmse=lambda x: np.sqrt(np.mean(np.square(x)))
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; miss=np.isnan(te['coupon_features_raw']).any(axis=1)
for lab in range(4):
 m=(te['lab_id']==lab)&~miss
 print('complete lab',lab,'strain/rate RMSE',rmse(e[m]),rmse(r[m]),'signed late/rate mean',e[m,-1].mean(),r[m].mean())
# Independently express the history as a closed convolution of pairwise edge differences.
cal=s._stress_calibration(d['calibration_load_cells']); dt=d['past_dt']; past,hold,x=s._inputs(te,s._align(tr,te)[len(tr['lab_id']):],cal)
tau=o['estimated_taus']; theta=o['hereditary_weights']; edges=np.r_[0.,np.cumsum(dt)]; w=np.exp(-(edges[-1]-edges[1:])[:,None]/tau)-np.exp(-(edges[-1]-edges[:-1])[:,None]/tau)
gap=hold[:,None]-past@w; transient=gap*theta[:2]; drift=x@theta[2:]; prior=transient@(1/tau)+drift; drift[miss]+=te['instant_rate_obs'][miss]-prior[miss]
np.testing.assert_allclose(o['predicted_strain'],te['strain_0'][:,None]+transient@(-np.expm1(-t[None,:]/tau[:,None]))+drift[:,None]*t,rtol=0,atol=1e-12)
np.testing.assert_allclose(o['initial_rate_pred'],transient@(1/tau)+drift,rtol=0,atol=1e-12)
print('independent convolution and analytic onset agree')
# Invert a nonidentity Wing B load-cell gain in the command coordinates without changing any physical measurements or response labels.
cd=d['calibration_load_cells']; q={k:np.array(v,copy=True) for k,v in cd.items()}; recoded={}
for name,b in [('train',tr),('test',te)]:
 recoded[name]=dict(b,past_stress_cmd=b['past_stress_cmd'].copy(),stress_hold_cmd=b['stress_hold_cmd'].copy())
for lab,a,bias in [(2,1.12,.06),(3,.91,-.04)]:
 m=q['lab_id']==lab; q['cmd_stress'][m]=(q['cmd_stress'][m]-bias)/a
 for z in recoded.values():
  m=z['lab_id']==lab
  for k in ['past_stress_cmd','stress_hold_cmd']: z[k][m]=(z[k][m]-bias)/a
for n in (10,2,1):
 m=np.ones(len(q['lab_id']),bool)
 if n<10:
  for lab in (2,3):
   idx=np.flatnonzero(q['lab_id']==lab); idx=idx[np.argsort(q['cmd_stress'][idx])]; m[idx]=False; m[idx[[0,-1] if n==2 else [len(idx)//2]]]=True
 z=s.solve_alpha_m96_w02_s00(dict(d,**recoded,calibration_load_cells={k:v[m] for k,v in q.items()})); wing=te['lab_id']>=2
 print('nonidentity wing B',n,'checks per lab: strain/rate RMSE',rmse(z['predicted_strain'][wing]-g['te_clean'][wing]),rmse(z['initial_rate_pred'][wing]-g['te_true_rate0'][wing]))
 if n==10:
  np.testing.assert_allclose(z['predicted_strain'],o['predicted_strain'],rtol=0,atol=1e-8)
PY
```
- **Container Exit Code**: `0` (`1028 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
complete lab 0 strain/rate RMSE 0.005399692350555928 0.0016888438476462641 signed late/rate mean 0.0021236483679623903 0.0003484976994574722
complete lab 1 strain/rate RMSE 0.003604362234534293 0.001221032224509998 signed late/rate mean -0.003054414941541559 -0.0005138016517634521
complete lab 2 strain/rate RMSE 0.0035432489361650167 0.001222227610961236 signed late/rate mean 0.005319305917772148 0.0009953367042478995
complete lab 3 strain/rate RMSE 0.006976905335913392 0.0022306117518565787 signed late/rate mean -0.004310191894824542 -0.000845032180454893
independent convolution and analytic onset agree
nonidentity wing B 10 checks per lab: strain/rate RMSE 0.008268667622392364 0.002673858619704825
nonidentity wing B 2 checks per lab: strain/rate RMSE 0.008272338044020599 0.0026747755274889014
nonidentity wing B 1 checks per lab: strain/rate RMSE 0.29932996142427404 0.2480215565195149

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d47ed7248ea41af7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d47ed724 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`60526 ms` LLM)
- **Rationale**: Test the exact anchored basis on the same real responses under different command codings and spot locations, and use within-lab held-out predictions to check that calibration transfers beyond the coupons used to estimate it. Promote only if those checks show a material improvement without degrading the anchored labs.
- **Alternatives Weighed**: `Borrow a recent lab gain from the calibration archive` (rejected: The archive is not mounted in this run.); `Recalibrate every lab from its creep response` (rejected: Labs with adequate checks already have independent calibration; letting creep move those calibrations would weaken identification.)
- **ATMS NOGOOD DECLARED**: Refuted `A single check can be reconciled by fitting only an offset and fixing the lab gain to one.` | Witness: `Under equivalent nonidentity command coordinates, two checks preserved Wing B strain RMSE at 0.00827; one check raised it to 0.299.` | Blanket Diagnosis: `A single check fixes stress at its command coordinate, leaving the gain multiplying the command contrast unresolved. The exact hereditary design exposes that gain through stress variation, relative to labs with identified calibration.` | Sublated Pivot: `Keep the affine anchor exact and estimate the remaining gain from training responses, using adequately calibrated labs to initialize the shared hereditary law.`

```bash
cd /holdout_task && cat > /tmp/alpha_sparse_probe.py <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('/holdout_task/public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']

def joint(d):
    tr,te=d['train'],d['test']; lab=tr['lab_id']; checks=d['calibration_load_cells']; cl=checks['lab_id']; cmd=checks['cmd_stress']; val=checks['true_stress']; cal=s._stress_calibration(checks)
    sparse=[]; anchors=[]
    for j in range(4):
        m=(cl==j)&np.isfinite(cmd)&np.isfinite(val)
        if m.any() and np.ptp(cmd[m])<1e-8 and np.any(lab==j):
            sparse.append(j); anchors.append(np.median(cmd[m])); cal[j]=[1.,np.median(val[m]-cmd[m])]
    x=s._align(tr,te); n=len(lab); dt,t=d['past_dt'],d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; fit=~np.isin(lab,sparse)
    if not fit.any(): return s.solve_alpha_m96_w02_s00(d),cal
    base=s._inputs(tr,x[:n],cal); base_design=s._Design(dt,t,base)
    pilot=s._Design(dt,t,tuple(a[fit] for a in base)); tau,theta=s._fit(pilot,y[fit],rate[fit]); target=np.column_stack((y,rate)); r=(base_design.make(tau)@theta-target)[fit]
    def mad(a): return 1.4826*np.median(abs(a-np.median(a)))
    scale=2.5*np.r_[np.full(len(t),max(1e-7,mad(r[:,:-1]))),max(1e-7,mad(r[:,-1]))]
    # At the known check (x*,y*), sigma(x)=y*+gain*(x-x*).
    # Design each lab's exact derivative with respect to gain.
    contrasts=[]
    for j,a in zip(sparse,anchors):
        m=lab==j
        contrasts.append(s._Design(dt,t,(tr['past_stress_cmd'][m]-a,tr['stress_hold_cmd'][m]-a,base[2][m])))
    def error(p):
        a=base_design.make(p[:2]);
        for k,(j,contrast) in enumerate(zip(sparse,contrasts)):
            a[lab==j,:,:2]+=(np.exp(p[9+k])-1)*contrast.make(p[:2])[:,:,:2]
        return ((a@p[2:9]-target)/scale).ravel()
    p=least_squares(error,np.r_[tau,theta,np.zeros(len(sparse))],bounds=(np.r_[.2,.2,np.full(7+len(sparse),-np.inf)],np.r_[8.,8.,np.full(7+len(sparse),np.inf)]),loss='cauchy',x_scale='jac',max_nfev=150,ftol=1e-9).x
    for j,a,q in zip(sparse,anchors,p[9:]):
        gain=np.exp(q); cal[j,1]-=(gain-1)*a; cal[j,0]=gain
    f=s._Design(dt,t,s._inputs(te,x[n:],cal)).make(p[:2])@p[2:9]
    if 'instant_rate_obs' in te:
        onset=te['instant_rate_obs']; m=np.isnan(te['coupon_features_raw']).any(axis=1)&np.isfinite(onset); f[m,:-1]+=(onset[m]-f[m,-1])[:,None]*t; f[m,-1]=onset[m]
    return dict(predicted_strain=f[:,:-1]+te['strain_0'][:,None],initial_rate_pred=f[:,-1],estimated_taus=p[:2],hereditary_weights=p[2:9]),cal

def recode(d, pair, spot):
    q={k:v.copy() for k,v in d['calibration_load_cells'].items()}; recoded={}
    for name in ('train','test'):
        b=d[name]; recoded[name]=dict(b,past_stress_cmd=b['past_stress_cmd'].copy(),stress_hold_cmd=b['stress_hold_cmd'].copy())
    mask=np.ones(len(q['lab_id']),bool)
    for lab,a,bias in pair:
        m=q['lab_id']==lab; q['cmd_stress'][m]=(q['cmd_stress'][m]-bias)/a
        for z in recoded.values():
            sel=z['lab_id']==lab
            for k in ('past_stress_cmd','stress_hold_cmd'): z[k][sel]=(z[k][sel]-bias)/a
        idx=np.flatnonzero(m); idx=idx[np.argsort(q['cmd_stress'][idx])]; mask[idx]=False; mask[idx[spot]]=True
    return dict(d,**recoded,calibration_load_cells={k:v[mask] for k,v in q.items()})

if __name__=='__main__':
    rmse=lambda x: np.sqrt(np.mean(x*x)); wing=te['lab_id']>=2
    for pair in [[(2,1.,0.),(3,1.,0.)],[(2,1.12,.06),(3,.91,-.04)],[(2,.80,-.03),(3,1.25,.07)]]:
        for spot in (0,5,-1):
            q=recode(d,pair,spot); old=s.solve_alpha_m96_w02_s00(q); new,cal=joint(q)
            errors=lambda o,m: (rmse(o['predicted_strain'][m]-g['te_clean'][m]),rmse(o['initial_rate_pred'][m]-g['te_true_rate0'][m]))
            print('gain',tuple(a for j,a,b in pair),'spot',spot,'wing baseline',errors(old,wing),'joint',errors(new,wing),'anchor wing',errors(new,~wing),'estimated gains',cal[2:,0],flush=True)
    pair=[(2,1.12,.06),(3,.91,-.04)]
    for fold in range(3):
        hold=(np.arange(len(tr['lab_id']))//4)%3==fold
        ds=dict(d,train={k:v[~hold] for k,v in tr.items()},test={k:v[hold] for k,v in tr.items() if k!='strain_curves_obs'})
        ds=recode(ds,pair,5); o=s.solve_alpha_m96_w02_s00(ds); z,cal=joint(ds); m=ds['test']['lab_id']>=2
        print('within-lab fold',fold,'wing B strain MdAE',[np.median(abs(v['predicted_strain'][m]-tr['strain_curves_obs'][hold][m])) for v in (o,z)],'gain',cal[2:,0],flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_sparse_probe.py
```
- **Container Exit Code**: `0` (`2490 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3917, 0.0032951266296324445) estimated gains [1.00081344 0.9998254 ]
gain (1.0, 1.0) spot 5 wing baseline (0.008284056956267228, 0.002681661628328376) joint (0.008126272000501096, 0.0030960124370037418) anchor wing (0.010324091014791878, 0.003294364945831649) estimated gains [1.00084003 0.99977179]
gain (1.0, 1.0) spot -1 wing baseline (0.008282646938302805, 0.0026811727324207828) joint (0.008129488455390417, 0.0031217060583736927) anchor wing (0.010323957261638261, 0.0032939862962575216) estimated gains [1.00085397 0.99974353]
gain (1.12, 0.91) spot 0 wing baseline (0.2987150828558193, 0.24799375785357303) joint (0.008115829832783606, 0.0030474137280329166) anchor wing (0.010324238319395796, 0.0032951264140388885) estimated gains [1.12091105 0.90984111]
gain (1.12, 0.91) spot 5 wing baseline (0.29932996142427404, 0.2480215565195149) joint (0.008126274345195816, 0.0030960134305790944) anchor wing (0.010324091056045983, 0.0032943649582603683) estimated gains [1.12094084 0.90979233]
gain (1.12, 0.91) spot -1 wing baseline (0.2997109135776883, 0.24806890370097923) joint (0.008129492661233325, 0.003121708140788743) anchor wing (0.010323957363706025, 0.0032939863157914803) estimated gains [1.12095645 0.90976661]
gain (0.8, 1.25) spot 0 wing baseline (0.656100139166222, 0.5450873126297604) joint (0.008115829600880198, 0.0030474155490437527) anchor wing (0.010324238295620744, 0.0032951266328269968) estimated gains [0.80065075 1.24978174]
gain (0.8, 1.25) spot 5 wing baseline (0.6583727349818519, 0.5453753992946113) joint (0.008126274417419884, 0.0030960137374304807) anchor wing (0.010324091046763052, 0.0032943648994347456) estimated gains [0.80067203 1.24971474]
gain (0.8, 1.25) spot -1 wing baseline (0.6600799593381622, 0.5456285253273074) joint (0.008129490955584675, 0.0031217068750353937) anchor wing (0.010323957329390603, 0.0032939862767082427) estimated gains [0.80068318 1.24967941]
within-lab fold 0 wing B strain MdAE [0.11447729827772257, 0.015417254229484735] gain [1.12198798 0.90821178]
within-lab fold 1 wing B strain MdAE [0.11913394123603527, 0.010675770726749911] gain [1.1191995  0.90842113]
within-lab fold 2 wing B strain MdAE [0.08898840081668483, 0.012958128305441685] gain [1.12129745 0.91122121]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d47ed7248ea41af7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d47ed724 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`55459 ms` LLM)
- **Rationale**: Promote the tested sparse-calibration fit. If the integrated artifact changes nominal outputs, loses coordinate invariance, or fails to reproduce the held-out improvement, correct the integration before submission.
- **Alternatives Weighed**: `Re-estimate gains in every batch` (rejected: Only singleton checks leave a gain unresolved. The nominal fit and calibrations with identified slopes should remain unchanged.)

```bash
set -e
cd /holdout_task
cp alpha_m96_w02_s00_solver.py /tmp/alpha_before_sparse.py
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m96_w02_s00_solver.py')
s = p.read_text()
old = 'def solve_alpha_m96_w02_s00(dataset: dict) -> dict:\n'
new = '''def _fit_sparse_calibration(train, checks, x, dt, t, cal, curves):
    """Complete a singleton affine check using the training creep response."""
    lab = np.asarray(train['lab_id'])
    cmd = np.asarray(checks['cmd_stress'])
    true = np.asarray(checks['true_stress'])
    check_lab = np.asarray(checks['lab_id'])
    sparse = []
    identified = []
    for c in np.unique(lab):
        use = (check_lab == c) & np.isfinite(cmd) & np.isfinite(true)
        if np.any(use):
            if np.ptp(cmd[use]) > 1e-8:
                identified.append(c)
            else:
                sparse.append((c, np.median(cmd[use])))

    rates = train['instant_rate_obs']
    base = _inputs(train, x, cal)
    design = _Design(dt, t, base)
    if not sparse or not identified:
        return _fit(design, curves, rates)

    pilot = np.isin(lab, identified)
    tau, theta = _fit(_Design(dt, t, tuple(a[pilot] for a in base)),
                      curves[pilot], rates[pilot])
    target = np.column_stack((curves, rates))
    r = (design.make(tau) @ theta - target)[pilot]

    def mad(a):
        return max(1e-7, 1.4826 * np.median(abs(a - np.median(a))))

    scale = 2.5 * np.r_[np.full(len(t), mad(r[:, :-1])), mad(r[:, -1])]
    contrasts = []
    for c, anchor in sparse:
        m = lab == c
        contrast = _Design(dt, t, (train['past_stress_cmd'][m] - anchor,
                                   train['stress_hold_cmd'][m] - anchor, base[2][m]))
        contrasts.append((m, contrast))

    # Stress at the check remains fixed; only the gain on command differences moves.
    # Calibrated labs set the absolute scale shared by the two hereditary branches.
    def residual(p):
        a = design.make(p[:2])
        for q, (m, contrast) in zip(p[9:], contrasts):
            a[m, :, :2] += np.expm1(q) * contrast.make(p[:2])[:, :, :2]
        return ((a @ p[2:9] - target) / scale).ravel()

    free = np.full(7 + len(sparse), np.inf)
    p = least_squares(residual, np.r_[tau, theta, np.zeros(len(sparse))],
                      bounds=(np.r_[.2, .2, -free], np.r_[8., 8., free]),
                      loss='cauchy', x_scale='jac', max_nfev=150, ftol=1e-9).x
    for (c, anchor), q in zip(sparse, p[9:]):
        cal[c, 0] = np.exp(q)
        cal[c, 1] -= np.expm1(q) * anchor
    return _order_branches(p[:9])


def solve_alpha_m96_w02_s00(dataset: dict) -> dict:
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '    if p[0] > p[1]:\n'
assert s.count(old) == 1
s = s.replace(old, '    return _order_branches(p)\n\n\ndef _order_branches(p):\n' + old)
old = "    dtr = _Design(dt, t, _inputs(train, x[:n], cal))\n    curves = train['strain_curves_obs'] - train['strain_0'][:, None]\n    tau, theta = _fit(dtr, curves, train['instant_rate_obs'])"
assert s.count(old) == 1
s = s.replace(old, "    curves = train['strain_curves_obs'] - train['strain_0'][:, None]\n    tau, theta = _fit_sparse_calibration(\n        train, dataset['calibration_load_cells'], x[:n], dt, t, cal, curves\n    )")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
probe=load('probe','/tmp/alpha_sparse_probe.py'); old=load('old','/tmp/alpha_before_sparse.py'); new=probe.s
d,g,tr,te=probe.d,probe.g,probe.tr,probe.te
base=old.solve_alpha_m96_w02_s00(d); nominal=new.solve_alpha_m96_w02_s00(d)
for k in base: np.testing.assert_array_equal(nominal[k],base[k])
print('Nominal outputs unchanged in every field')
rmse=lambda x: np.sqrt(np.mean(x*x)); wing=te['lab_id']>=2; worst=np.zeros((2,2)); equiv=0.
for spot in (0,5,-1):
    ref=new.solve_alpha_m96_w02_s00(probe.recode(d,[(2,1.,0.),(3,1.,0.)],spot))
    for pair in ([(2,1.12,.06),(3,.91,-.04)],[(2,.8,-.03),(3,1.25,.07)]):
        q=probe.recode(d,pair,spot); o=new.solve_alpha_m96_w02_s00(q)
        for k in ref:
            equiv=max(equiv,float(np.max(abs(o[k]-ref[k])))); np.testing.assert_allclose(o[k],ref[k],rtol=0,atol=3e-6)
        for i,m in enumerate((wing,~wing)):
            worst[i]=np.maximum(worst[i],[rmse(o['predicted_strain'][m]-g['te_clean'][m]),rmse(o['initial_rate_pred'][m]-g['te_true_rate0'][m])])
        assert .2<=o['estimated_taus'][0]<o['estimated_taus'][1]<=8 and np.array_equal(o['viscoplastic_weights'],g['vp_theta'])
print('Max output difference across equivalent singleton coordinates:',equiv)
print('Worst sparse Wing B/A strain and onset RMSE:',worst.tolist())
assert np.all(worst < [[.010,.004],[.013,.004]])
for fold in range(3):
    hold=(np.arange(len(tr['lab_id']))//4)%3==fold
    q=probe.recode(dict(d,train={k:v[~hold] for k,v in tr.items()},test={k:v[hold] for k,v in tr.items() if k!='strain_curves_obs'}),[(2,1.12,.06),(3,.91,-.04)],5)
    m=q['test']['lab_id']>=2; y=tr['strain_curves_obs'][hold][m]
    errs=[np.median(abs(s.solve_alpha_m96_w02_s00(q)['predicted_strain'][m]-y)) for s in (old,new)]
    print('Within-lab fold',fold,'Wing B strain median absolute error, old/new:',errs)
    assert errs[1]<.02 and errs[1]<.25*errs[0]
PY
```
- **Container Exit Code**: `0` (`2323 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Nominal outputs unchanged in every field
Max output difference across equivalent singleton coordinates: 6.688298448409569e-08
Worst sparse Wing B/A strain and onset RMSE: [[0.008129492650245117, 0.0031217081305441037], [0.010324238321900046, 0.003295126638026271]]
Within-lab fold 0 Wing B strain median absolute error, old/new: [0.11447729827772257, 0.015417254261768688]
Within-lab fold 1 Wing B strain median absolute error, old/new: [0.11913394123603527, 0.010675770627666947]
Within-lab fold 2 Wing B strain median absolute error, old/new: [0.08898840081668483, 0.0129581287609149]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eadc6c2b09604035 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_eadc6c2b (directory).
```

### Turn 13 — `submit_completed_task` (`15453 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification and improved all three sparse-calibration held-out folds. Nominal outputs were unchanged, and equivalent singleton command coordinates changed outputs by at most 6.69e-8.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m96_w02_s00_solver.py`. The public schema validator passed. Onset conditioning reduced missing-feature strain RMSE from 0.294 to 0.0235. For recoded Wing B labs with one calibration check each, the anchored fit reduced worst-case strain RMSE from 0.660 to 0.00813 without changing nominal outputs. Cross-epoch calibration was not verified because the archive was unavailable.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M96-Autonomous-Structure-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
