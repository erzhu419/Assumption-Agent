# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_v2_c01_s00__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_hereditary_memory_and_factored_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `206,213` (`198,347` / `94,592` / `7,866`) |
| **Total Episode Wall Time (s)** | `254.55s` |

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

### Turn 1 — `execute_in_remote_docker` (`17786 ms` LLM)
- **Rationale**: Inspect the schema and archive inventory before fitting: an absent archive implies the requested baseline fallback, while available campaign evidence can support a separate chamber model. Print only field summaries and small samples.
- **Alternatives Weighed**: `Implement the benchtop law for deployment immediately` (rejected: An available archive or chamber measurement may establish a different deployment law.)

```bash
cd /holdout_task; cat validate_public_schema.py; cat public_asset_index.json; ls -la /tmp/hegel_continual_archive/ 2>/dev/null; find /tmp/hegel_continual_archive/ -maxdepth 2 -type f -printf '%p %s\n' 2>/dev/null | head -60; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference-panel alignment imputation' 2>&1; python3 - <<'PY'
import pickle
import numpy as np
items=pickle.load(open('public_batch.pkl','rb'))
print('items:',len(items))
def inspect(x,depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print('  '*depth,k,end=': ')
   if isinstance(v,dict): print(); inspect(v,depth+1)
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype,v.flat[:min(v.size,6)])
   elif isinstance(v,(list,tuple)): print(type(v).__name__,len(v),str(v[:1])[:240])
   else: print(repr(v)[:300])
inspect(items[0])
PY
```
- **Container Exit Code**: `0` (`420 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ecomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
items: 1
 dataset: 
   past_dt: (6,) float64 [0.35 0.35 0.4  0.45 0.5  0.55]
   eval_times: (12,) float64 [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545]
   calibration_load_cells: 
     bridge_raw: (80, 4) float64 [-1.59997081e+00  4.00689783e-05  5.49078467e-05  1.14220221e-04
 -6.00079346e-01  2.74929997e-05]
     true_channel_loads: (80, 4) float64 [-1.6  0.   0.   0.  -0.6  0. ]
     single_channel_sweeps: 
       lab_id: (24,) int64 [0 0 0 0 0 0]
       cmd_stress: (24,) float64 [-1.50006605 -0.80002768 -0.19986659  0.4000817   0.99973451  1.6000585 ]
       true_stress: (24,) float64 [-1.5 -0.8 -0.2  0.4  1.   1.6]
   train: 
     lab_id: (72,) int64 [3 1 2 0 3 3]
     stratum: (72,) int64 [0 2 1 0 1 1]
     coupon_features_raw: (72, 4) float64 [1.21590406 1.35939071 2.22557928 1.17485867        nan 1.18467476]
     ref_coupons: (72, 8) float64 [0.78575856 0.43564525 2.44339855 2.35885172 0.69041243 0.80655121]
     past_channel_raw: (72, 6, 4) float64 [ 0.10924772 -0.06689769 -0.26836843  0.79975515  0.31935466 -0.49595129]
     hold_channel_raw: (72, 4) float64 [0.76688848 0.51214959 0.90831406 1.14313476 0.77203607 0.37625921]
     strain_0: (72,) float64 [-0.04341426 -0.11861684 -0.04359393 -0.06632571 -0.05011796  0.07950224]
     strain_curves_obs: (72, 12) float64 [0.6763363  1.36793576 1.88337601 2.29425193 2.6548604  2.93015027]
     instant_rate_obs: (72,) float64 [3.20498414 1.5574968  2.1356212  3.13976177 3.94717341 2.52701628]
     aux_chamber_telemetry: (72, 4) float64 [ 1.47033609  0.3024811   0.11949447  1.28569688  0.46079816 -0.50058905]
   test: 
     lab_id: (36,) int64 [1 3 2 3 1 2]
     stratum: (36,) int64 [1 2 1 2 1 2]
     coupon_features_raw: (36, 4) float64 [-0.02016159  1.51907906  0.23330673  1.3282453   0.14364975  0.41886771]
     ref_coupons: (36, 8) float64 [ 1.78610587  0.80298481  0.33832328 -0.64617152 -0.38467991  0.35855022]
     past_channel_raw: (36, 6, 4) float64 [ 0.99104913 -0.12718804  1.07287858 -0.41762204  1.12672546  1.01037696]
     hold_channel_raw: (36, 4) float64 [0.35430892 0.66201029 0.9226882  1.04545231 0.74689822 0.42310072]
     strain_0: (36,) float64 [ 0.13015459  0.24261431  0.03400766  0.07318248 -0.05774535  0.10944476]
     aux_chamber_telemetry: (36, 4) float64 [-0.59432625  2.02855014  1.14799154 -0.82396726 -1.34555314 -1.76565125]
 campaign_diagnostics: None
```

### Turn 2 — `execute_in_remote_docker` (`21493 ms` LLM)
- **Rationale**: The prior inspection was truncated. Confirm archive availability, determine how the eight reference columns behave within each lab, and compare affine calibration residuals with the raw bridge discrepancy before choosing the fitting model.
- **Alternatives Weighed**: `Fit chamber effects using auxiliary telemetry alone` (rejected: Telemetry without labeled chamber outputs or archive evidence cannot identify the deployment response.)

```bash
cd /holdout_task; ls -la /tmp/hegel_continual_archive 2>/dev/null || echo 'NO ARCHIVE'; find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%p %s\n' 2>/dev/null | head -30; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit 2>&1 | head -35; python3 - <<'PY'
import numpy as n,pickle,scipy
x=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; print('scipy',scipy.__version__)
c=x['calibration_load_cells']; u,v=c['bridge_raw'],c['true_channel_loads']; print('calibration affine vs identity rms',n.std(n.c_[u,n.ones(len(u))]@n.linalg.lstsq(n.c_[u,n.ones(len(u))],v,rcond=None)[0]-v), n.std(u-v)); print('axis rows',n.sum(n.count_nonzero(v,axis=1)<=1))
for k in ['train','test']:
 d=x[k]; print(k)
 for i in n.unique(d['lab_id']):
  j=d['lab_id']==i; r=d['ref_coupons'][j]; f=d['coupon_features_raw'][j]; print(i,len(r),'nan',n.isnan(f).sum(0),'ref means/std',r.mean(0).round(3),r.std(0).round(3),'feature means/std',n.nanmean(f,0).round(3),n.nanstd(f,0).round(3))
PY
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step"
      ],
scipy 1.13.0
calibration affine vs identity rms 9.760876451545361e-05 0.0001023746229584971
axis rows 16
train
0 13 nan [2 0 0 0] ref means/std [-0.816 -0.645 -1.353 -0.704 -0.783 -1.034  0.257 -0.91 ] [1.161 1.439 1.161 1.096 1.028 1.297 1.466 1.154] feature means/std [ 0.25  -0.876 -0.816 -0.013] [1.19  1.107 1.362 1.15 ]
1 25 nan [3 0 4 0] ref means/std [0.592 0.659 0.403 0.312 0.429 0.585 0.384 0.133] [0.597 0.819 0.767 0.944 0.961 0.718 0.961 0.693] feature means/std [0.265 0.439 0.287 0.635] [0.863 0.851 0.638 0.75 ]
2 13 nan [1 1 1 2] ref means/std [-0.358 -0.467 -0.084 -0.524 -0.695  0.103 -0.491 -0.397] [0.784 0.89  1.555 1.23  1.659 1.336 1.645 1.226] feature means/std [-0.325  0.267 -0.396 -0.212] [1.23  1.173 1.342 1.48 ]
3 21 nan [0 3 1 2] ref means/std [0.985 0.698 0.855 1.13  0.654 1.015 0.835 0.534] [1.027 0.829 0.697 0.612 0.753 0.921 0.744 0.967] feature means/std [0.725 0.776 0.795 0.886] [0.729 0.653 0.855 0.779]
test
0 7 nan [2 0 0 0] ref means/std [-0.554 -0.025  0.047 -0.594 -0.943 -0.609 -0.721 -0.351] [0.974 1.355 1.055 1.576 0.922 1.364 1.204 1.074] feature means/std [-1.349 -0.372 -0.151 -0.474] [0.807 1.345 0.876 0.745]
1 13 nan [0 2 0 2] ref means/std [1.016 0.638 0.077 0.57  0.74  0.948 0.48  0.765] [0.934 0.717 0.916 0.68  0.784 0.753 0.874 0.972] feature means/std [0.527 0.13  0.663 0.705] [0.767 0.843 0.755 0.718]
2 6 nan [0 0 0 1] ref means/std [-0.553 -0.115 -0.98  -0.037 -0.817  0.256 -0.444  0.301] [0.88  1.022 0.401 0.948 0.829 0.946 1.094 1.304] feature means/std [-0.362 -1.079 -0.746 -0.632] [1.046 1.113 1.41  1.087]
3 10 nan [0 2 0 1] ref means/std [0.814 0.652 0.358 0.917 0.407 1.144 0.496 0.819] [0.897 0.859 0.766 0.798 1.024 0.885 0.612 0.89 ] feature means/std [0.664 0.511 0.736 0.826] [0.848 0.954 1.222 0.258]
```

### Turn 3 — `execute_in_remote_docker` (`62358 ms` LLM)
- **Rationale**: Fit the measured bridge dynamically, impose the exact hereditary recurrence and onset derivative, and validate both response channels on unseen lab labels. With no archive or diagnostic outcomes, the deployment prediction uses the specified benchtop fallback.
- **Alternatives Weighed**: `Use a single loss scale or the unmodified robust-fit primitive` (rejected: Curve and rate residuals can have different noise levels, and the lifetime fit requires a joint nonlinear model.)

```bash
cd /holdout_task && cat > batch_solver.py <<'PY'
"""Calibrate the load bridge and fit a shared two-branch relaxation spectrum."""
import numpy as np
from scipy.optimize import least_squares


def _axial_load_map(calibration):
    # The calibration experiments prescribe channel loads and measure bridge output.
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    bridge = np.asarray(calibration['bridge_raw'], dtype=float)
    coeff = np.linalg.lstsq(np.column_stack((loads, np.ones(len(loads)))), bridge, rcond=None)[0]
    return coeff[-1], np.linalg.solve(coeff[:4], np.ones(4))


def _features(train, test):
    groups = (train, test)
    raw, ref, lab, strata = (np.concatenate([d[k] for d in groups], axis=0) for k in
                            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty(raw.shape, dtype=float)
    for key in np.unique(lab):
        rows = lab == key
        panel = ref[rows]
        center = np.median(np.mean(panel, axis=0))
        scale = np.median(np.std(panel, axis=0, ddof=0))
        x[rows] = (raw[rows] - center) / (scale if scale > 0 else 1.)
    _, cell, counts = np.unique(np.column_stack((lab, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    present = np.isfinite(x)
    denom = np.sum(w * present, axis=0)
    center = np.sum(w * np.where(present, x, 0.), axis=0) / np.where(denom > 0, denom, 1.)
    x = np.where(present, x, center)
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _specimen(group, x, bridge_map):
    offset, gain = bridge_map
    return ((group['past_channel_raw'] - offset) @ gain,
            (group['hold_channel_raw'] - offset) @ gain,
            np.column_stack((np.ones(len(x)), x)))


def _branches(specimen, dt, times, taus, derivatives=False):
    past, hold, _ = specimen
    u = dt[:, None] / taus
    decay = np.exp(-u)
    increment = -np.expm1(-u)
    state = np.zeros((len(hold), 2))
    dstate = np.zeros_like(state)
    for j in range(len(dt)):
        stress = past[:, j, None]
        if derivatives:
            dstate = decay[j] * (dstate + u[j] * (state - stress))
        state = decay[j] * state + increment[j] * stress
    gap = hold[:, None] - state
    v = times[:, None] / taus
    relax = -np.expm1(-v)
    curves = gap[:, None, :] * relax
    rates = gap / taus
    if not derivatives:
        return curves, rates
    dcurves = -dstate[:, None, :] * relax - gap[:, None, :] * v * np.exp(-v)
    drates = -dstate / taus - rates
    return curves, rates, dcurves, drates


def _fit(specimen, dt, times, curves_obs, rate_obs):
    n, k = curves_obs.shape
    drift = specimen[2]
    fixed = np.vstack(((times[None, :, None] * drift[:, None, :]).reshape(n*k, 5), drift))
    obs = np.concatenate((curves_obs.ravel(), rate_obs))
    # Curve and rate channels have separate noise scales. Rates count as
    # independent observations rather than as a coupon-level penalty.
    group = np.arange(len(obs)) >= n*k
    floor = 1e-6 * max(1., np.median(np.abs(obs)))

    def design(tau):
        c, r = _branches(specimen, dt, times, tau)
        return np.column_stack((np.vstack((c.reshape(n*k, 2), r)), fixed))

    def mad(err):
        return max(floor, 1.4826 * np.median(np.abs(err - np.median(err))))

    grid = np.geomspace(.065, 23., 12)
    seeds = []
    for j, a in enumerate(grid[:-1]):
        for b in grid[j+1:]:
            mat = design((a, b))
            coef = np.linalg.lstsq(mat, obs, rcond=None)[0]
            err = mat @ coef - obs
            seeds.append((mad(err[~group]) + .35 * mad(err[group]), a, b, coef))
    seeds.sort(key=lambda s: s[0])
    _, a, b, coef = seeds[0]
    e = design((a, b)) @ coef - obs
    scales = np.where(group, mad(e[group]), mad(e[~group]))
    # Equal bounds on branch lifetimes preserve label symmetry; sort only at output.
    bounds = (np.r_[np.log([.05000001]*2), [-np.inf]*7],
              np.r_[np.log([24.999999]*2), [np.inf]*7])

    def fun(p):
        return (design(np.exp(p[:2])) @ p[2:] - obs) / scales

    def jac(p):
        c, r, dc, dr = _branches(specimen, dt, times, np.exp(p[:2]), derivatives=True)
        var = np.vstack((c.reshape(n*k, 2), r))
        dvar = np.vstack((dc.reshape(n*k, 2), dr)) * p[2:4]
        return np.column_stack((dvar, var, fixed)) / scales[:, None]

    fits = [least_squares(fun, np.r_[np.log([a,b]), coef], jac=jac, bounds=bounds,
                          loss='soft_l1', x_scale='jac', max_nfev=80)
            for _, a, b, coef in seeds[:4]]
    best = min(fits, key=lambda fit: fit.cost)
    for _ in range(2):
        e = design(np.exp(best.x[:2])) @ best.x[2:] - obs
        scales = np.where(group, mad(e[group]), mad(e[~group]))
        best = least_squares(fun, best.x, jac=jac, bounds=bounds,
                             loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    tau = np.exp(best.x[:2]); theta = best.x[2:].copy()
    order = np.argsort(tau)
    tau = tau[order]; theta[:2] = theta[:2][order]
    return tau, theta


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    xtr, xte = _features(train, test)
    bridge = _axial_load_map(dataset['calibration_load_cells'])
    tr = _specimen(train, xtr, bridge)
    tau, theta = _fit(tr, dt, times, train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])
    # This blind campaign supplies neither chamber responses nor a lineage archive.
    te = _specimen(test, xte, bridge)
    curves, rates = _branches(te, dt, times, tau)
    drift = te[2] @ theta[2:]
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curves @ theta[:2] + drift[:, None] * times,
            'initial_rate_pred': rates @ theta[:2] + drift}
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import batch_solver as m
from collections import Counter
import numpy as np
D=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']
a=m.solve_batch(D); print('tau',a['estimated_taus'],'theta',a['hereditary_weights'])
print('curves/rates range',[(k,np.min(a[k]),np.max(a[k])) for k in ['predicted_strain','initial_rate_pred']])
for lab in np.unique(D['train']['lab_id']):
 t=D['train']; mask=t['lab_id']==lab; s=lambda sel:{k:v[sel] for k,v in t.items()}; d=dict(D,train=s(~mask),test={k:np.concatenate([s(mask)[k],v]) for k,v in D['test'].items()}); p=m.solve_batch(d); n=mask.sum(); c=p['predicted_strain'][:n]-t['strain_curves_obs'][mask]; r=p['initial_rate_pred'][:n]-t['instant_rate_obs'][mask]; print('lab',lab,'tau',p['estimated_taus'],'curve abs median/p90',np.quantile(abs(c),[.5,.9]),'rate abs median/p90',np.quantile(abs(r),[.5,.9]))
PY
```
- **Container Exit Code**: `0` (`1349 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
tau [0.65737009 4.42553321] theta [ 0.56836014  0.58079929  0.31376931  0.13958976 -0.13562909  0.11833551
 -0.10259935]
curves/rates range [('predicted_strain', 0.0832253276303724, 6.902118636644201), ('initial_rate_pred', 0.7535604823674845, 4.310735389852304)]
lab 0 tau [0.66072654 4.53950747] curve abs median/p90 [0.0072156  0.01938512] rate abs median/p90 [0.00683592 0.01305291]
lab 1 tau [0.65713979 4.41041834] curve abs median/p90 [0.00805843 0.02339795] rate abs median/p90 [0.00654825 0.01304185]
lab 2 tau [0.65746889 4.48854113] curve abs median/p90 [0.00816171 0.01873348] rate abs median/p90 [0.00754068 0.01504155]
lab 3 tau [0.65448093 4.28273401] curve abs median/p90 [0.00923541 0.02346054] rate abs median/p90 [0.00632523 0.0136522 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5aa0f7be4000ac5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5aa0f7b (directory).
```

### Turn 4 — `execute_in_remote_docker` (`30260 ms` LLM)
- **Rationale**: The public schema and lab holdouts passed. Verify the exact artifact in isolated Python; bridge-coordinate invariance would catch an incorrect calibration under cross-talk, and the derivative check would catch an onset-rate implementation inconsistent with the reported strain.
- **Alternatives Weighed**: `Infer a chamber correction from unlabeled telemetry` (rejected: Neither diagnostics nor an archive is available, so the correction cannot be validated and the task specifies the benchtop fallback.)

```bash
set -e; cd /holdout_task; export OPENBLAS_NUM_THREADS=1; python3 -I /holdout_task/validate_public_schema.py; python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; a=m.solve_batch(d)
# Reexpress the same measured stresses in a bridge with cross-talk and offsets.
M=np.array([[1.2,.12,0,-.06],[.1,.85,.15,0],[0,-.16,1.08,.13],[.16,0,.1,.95]])
b=np.array([.11,-.08,.13,.06]); transform=lambda u:u@M+b
c=d['calibration_load_cells']; e=dict(d,calibration_load_cells=dict(c,bridge_raw=transform(c['bridge_raw'])))
for key in ('train','test'):
 g=d[key]; e[key]=dict(g,past_channel_raw=transform(g['past_channel_raw']),hold_channel_raw=transform(g['hold_channel_raw']))
z=m.solve_batch(e)
assert .05<a['estimated_taus'][0]<a['estimated_taus'][1]<25.
for key in a:
 err=float(np.max(np.abs(a[key]-z[key]))); print('bridge coordinate invariance',key,err); assert err<1e-6
# Check onset rate against the right derivative of the reported strain law.
h=1e-6; tiny=m.solve_batch(dict(d,eval_times=d['eval_times'].copy()))
tr,te=m._features(d['train'],d['test']); sample=m._specimen(d['test'],te,m._axial_load_map(c)); past,hold,drift=sample
state=np.zeros((len(hold),2)); tau=a['estimated_taus']; q=a['hereditary_weights']
for duration,stress in zip(d['past_dt'],past.T):
 decay=np.exp(-duration/tau); state=state*decay+stress[:,None]*(1-decay)
def response(t):
 return ((hold[:,None]-state)*(-np.expm1(-t/tau)))@q[:2]+t*(drift@q[2:])
curve=np.column_stack([response(t) for t in d['eval_times']])+d['test']['strain_0'][:,None]
rate=(4*response(h)-response(2*h))/(2*h)
print('curve reconstruction max error',float(np.max(abs(curve-a['predicted_strain'])))); print('onset derivative max error',float(np.max(abs(rate-a['initial_rate_pred']))))
np.testing.assert_allclose(curve,a['predicted_strain'],atol=1e-10,rtol=0)
np.testing.assert_allclose(rate,a['initial_rate_pred'],atol=1e-8,rtol=0)
PY
```
- **Container Exit Code**: `0` (`974 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
bridge coordinate invariance estimated_taus 1.149391692933932e-11
bridge coordinate invariance hereditary_weights 3.852917984659143e-12
bridge coordinate invariance predicted_strain 3.97193389289896e-12
bridge coordinate invariance initial_rate_pred 2.3723245590190345e-12
curve reconstruction max error 8.881784197001252e-16
onset derivative max error 2.7347013542566856e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5aa0f7be4000ac5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5aa0f7b (directory).
```

### Turn 5 — `submit_completed_task` (`13454 ms` LLM)
- **Rationale**: The archived candidate passed isolated Python schema verification, leave-one-lab-out evaluation, and both physical consistency checks.
- **Alternatives Weighed**: `Fit a chamber-specific correction` (rejected: No labeled chamber measurements or lineage archive were available to identify or validate it.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5969'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema, bridge calibration invariance, and onset-rate consistency checks passed. Worst held-out lab p90 absolute errors were 0.02346 for strain and 0.01504 for onset rate. Deployment uses the requested benchtop fallback because no chamber diagnostics or archive were available.`

### Turn 6 — `execute_in_remote_docker` (`29758 ms` LLM)
- **Rationale**: A consistent held-out gain from squared and pairwise bridge terms would justify revisiting affine calibration. Persistent signed error across time within a held-out lab would justify testing the feature alignment. The audit uses real labeled measurements and leaves the verified artifact unchanged.
- **Alternatives Weighed**: `Add rational constitutive terms, relay modes, or balance classification metrics` (rejected: The supplied responses are continuous and obey a specified benchtop recurrence, with no relay observations or class labels. The relevant checks are bridge nonlinearity and lab-specific residual bias.)

```bash
set -e; cd /holdout_task; export OPENBLAS_NUM_THREADS=1; python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; u,v=c['bridge_raw'],c['true_channel_loads']; pair=np.triu_indices(4)
def basis(x,degree):
    a=np.c_[np.ones(len(x)),x]
    return np.c_[a,x[:,pair[0]]*x[:,pair[1]]] if degree==2 else a
fold=np.empty(len(u),int); fold[np.random.default_rng(801).permutation(len(u))]=np.arange(len(u))%5
for degree in (1,2):
    x=basis(u,degree); errs=[]
    for j in range(5):
        tr=fold!=j; te=~tr; coeff=np.linalg.lstsq(x[tr],v[tr].sum(1),rcond=None)[0]; errs.append(np.sqrt(np.mean((x[te]@coeff-v[te].sum(1))**2)))
    print('held-out axial calibration RMSE by fold, degree',degree,np.round(errs,8))
t=d['train']; lab=t['lab_id']; times=d['eval_times']; n=len(lab); curves=np.empty_like(t['strain_curves_obs']); rates=np.empty(n)
for j in np.unique(lab):
    te=lab==j; tr=~te
    pred=m.solve_batch(dict(d,train={k:v[tr] for k,v in t.items()},test={k:np.concatenate((v[te],d['test'][k]),axis=0) for k,v in t.items() if k in d['test']})); curves[te]=pred['predicted_strain'][:sum(te)]; rates[te]=pred['initial_rate_pred'][:sum(te)]
ce=curves-t['strain_curves_obs']; re=rates-t['instant_rate_obs']; idx=np.array_split(np.argsort(times),3)
for j in np.unique(lab):
    rows=lab==j
    print('lab',j,'n',sum(rows),'early/mid/late signed strain median', [round(float(np.median(ce[rows][:,q])),6) for q in idx], 'early/mid/late abs strain p90',[round(float(np.quantile(abs(ce[rows][:,q]),.9)),6) for q in idx],'signed rate median',round(float(np.median(re[rows])),6))
PY
```
- **Container Exit Code**: `0` (`717 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out axial calibration RMSE by fold, degree 1 [0.00013058 0.00014941 0.00025651 0.00019356 0.00019788]
held-out axial calibration RMSE by fold, degree 2 [0.00023787 0.0001798  0.00026927 0.00017484 0.00019852]
lab 0 n 13 early/mid/late signed strain median [0.002584, 0.003024, -0.000478] early/mid/late abs strain p90 [0.020656, 0.022207, 0.015881] signed rate median -0.005721
lab 1 n 25 early/mid/late signed strain median [0.000546, -0.000407, 0.001395] early/mid/late abs strain p90 [0.022873, 0.020158, 0.023807] signed rate median 0.00086
lab 2 n 13 early/mid/late signed strain median [-0.004358, -0.004634, 0.000172] early/mid/late abs strain p90 [0.022211, 0.018185, 0.017142] signed rate median -0.001021
lab 3 n 21 early/mid/late signed strain median [-7.2e-05, 0.001641, -0.000216] early/mid/late abs strain p90 [0.020289, 0.025273, 0.021639] signed rate median 0.003695

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5aa0f7be4000ac5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5aa0f7b (directory).
```

### Turn 7 — `submit_completed_task` (`15960 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification. The focused audit found no consistent benefit from nonlinear calibration and no persistent signed strain bias across all time windows in any held-out lab.
- **Alternatives Weighed**: `Add quadratic and pairwise bridge terms` (rejected: They worsened calibration RMSE in four of five held-out folds, including the worst fold.); `Fit a chamber-specific correction` (rejected: No chamber response measurements or lineage archive were available to identify or validate one.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5969'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema, bridge calibration invariance, and onset-rate consistency checks passed. The worst held-out p90 absolute errors were 0.02527 for strain across lab-by-time slices and 0.01504 for onset rate across labs. With no chamber diagnostics or lineage archive available, deployment uses the specified benchtop fallback; the error estimates apply to benchtop data only.`

### Turn 8 — `execute_in_remote_docker` (`26816 ms` LLM)
- **Rationale**: Hold out entire load ranges to detect calibration errors that random folds can miss. Select rational pairwise terms only within training folds. A consistent reduction in high-load error would justify reassessing calibration; relay and classification checks do not apply to the observed continuous responses.
- **Alternatives Weighed**: `Repeat the random-fold quadratic audit` (rejected: It already favored affine calibration and did not isolate the highest-load extrapolation slice.)

```bash
set -e; cd /holdout_task; export OPENBLAS_NUM_THREADS=1; python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from itertools import combinations
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; u=np.asarray(c['bridge_raw']); v=np.asarray(c['true_channel_loads']); y=v.sum(1)
# Contiguous slices by load norm probe transfer beyond the interior, unlike random folds.
folds=np.array_split(np.argsort(np.linalg.norm(v,axis=1)),5)
forms=[(i,j,sign,a) for i,j in combinations(range(4),2) for sign in (-1,1) for a in (.001,.01,.1)]
def predict(tr,te,form):
    if form is None:
        offset,gain=m._axial_load_map({'bridge_raw':u[tr],'true_channel_loads':v[tr]})
        return (u[te]-offset)@gain
    center=u[tr].mean(0); scale=u[tr].std(0); scale=np.where(scale>0,scale,1.)
    i,j,sign,a=form
    def basis(rows):
        z=(u[rows]-center)/scale
        return np.c_[np.ones(len(rows)),z]/(1+a*(z[:,i]+sign*z[:,j])**2)[:,None]
    return basis(te)@np.linalg.lstsq(basis(tr),y[tr],rcond=None)[0]
for idx,te in enumerate(folds):
    tr=np.concatenate([f for j,f in enumerate(folds) if j!=idx]); inner=np.array_split(tr,4)
    def score(form):
        return max(np.sqrt(np.mean((predict(np.concatenate([f for j,f in enumerate(inner) if j!=i]),h,form)-y[h])**2)) for i,h in enumerate(inner))
    form=min(forms,key=score)
    print('load-norm slice',idx,'n',len(te),'axial RMSE affine / nested rational',*[float(np.sqrt(np.mean((predict(tr,te,f)-y[te])**2))) for f in (None,form)])
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`749 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
load-norm slice 0 n 16 axial RMSE affine / nested rational 0.0002121577452099484 0.0006136205952260035
load-norm slice 1 n 16 axial RMSE affine / nested rational 0.00015252379990381325 0.0015124450958238537
load-norm slice 2 n 16 axial RMSE affine / nested rational 0.0002081807478190055 0.0030111913533731544
load-norm slice 3 n 16 axial RMSE affine / nested rational 0.00020606457436143366 0.0036552737941634183
load-norm slice 4 n 16 axial RMSE affine / nested rational 0.0001778469245342535 0.011189383387911415
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5aa0f7be4000ac5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5aa0f7b (directory).
```

### Turn 9 — `submit_completed_task` (`13835 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, and the load-range stress test showed no benefit from replacing affine calibration.
- **Alternatives Weighed**: `Load-dependent rational bridge calibration` (rejected: Nested selection produced higher axial-load RMSE than affine calibration in all five held-out load ranges.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema, bridge calibration invariance, and onset-rate consistency checks passed. Worst held-out p90 absolute errors were 0.02527 for strain across lab-by-time slices and 0.01504 for onset rate across labs. These errors apply to benchtop data only. With no chamber diagnostics or lineage archive available, deployment uses the specified benchtop fallback.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
