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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `341,968` (`329,482` / `98,816` / `12,486`) |
| **Total Episode Wall Time (s)** | `411.73s` |

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

### Turn 1 — `execute_in_remote_docker` (`17206 ms` LLM)
- **Rationale**: Inspect the public schema and compact array summaries to catch axis or identifier mismatches before fitting; look up reusable robust-fitting and panel-alignment primitives.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS reference-panel alignment imputation viscoelastic relaxation' ; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))
def show(x,depth=0):
 if isinstance(x,dict) and depth<3:
  for k,v in x.items():
   print('  '*depth,k,end=': ')
   if isinstance(v,dict): print();show(v,depth+1)
   else: show(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,repr(x.ravel()[:10]))
 elif isinstance(x,(list,tuple)): print(type(x).__name__,len(x),repr(x[:2])[:400])
 else: print(repr(x)[:300])
show(p)
PY
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
list 1 [{'dataset': {'past_dt': array([0.3146, 0.3466, 0.7365, 0.6657, 0.2202, 0.478 , 0.2481, 0.7844,
       0.2593, 0.3877, 0.2743, 0.4214, 0.5335, 0.679 , 0.691 , 0.4346,
       0.4923, 0.4181]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shear_banding_index': 0.12, 'calibration_load_cells':
```

### Turn 2 — `execute_in_remote_docker` (`18940 ms` LLM)
- **Rationale**: Inspect split fields and missingness to identify the reference-panel and initial-strain layouts. Check per-channel affine residuals to decide whether the specified EMI or sparse-calibration fallback is needed, and verify whether the reusable primitives match the documented alignment.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0];print('item keys',p.keys());d=p['dataset']
for k,v in d.items():
 print(k, end=': ')
 if isinstance(v,dict):
  print()
  for n,a in v.items():
   x=np.asarray(a);print(' ',n,x.shape,x.dtype,repr(x.flatten()[:8]),'nonfinite',int((~np.isfinite(x)).sum()) if x.dtype.kind in 'iuf' else '-')
 else: print(repr(v)[:350])
c=d['calibration_load_cells'];
for g in np.unique(c['lab_id']):
 i=c['lab_id']==g;x=c['cmd_stress'][i];y=c['true_stress'][i];b=np.polyfit(x,y,1);print('cal',g,len(x),b,'max resid',max(abs(y-np.polyval(b,x))))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights
```
- **Container Exit Code**: `0` (`540 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
loss in {'huber', 'cauchy'}.",
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
        "autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_affine_projection",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`54675 ms` LLM)
- **Rationale**: Build the specified model using the exact closed form of the discrete memory recurrence and the observed onset rates. Check shapes and calibration residuals first; substantial residuals would trigger archive fallback, and public reference discrepancies would motivate targeted slice validation.
- **Alternatives Weighed**: `Fit strain curves by ordinary least squares` (rejected: Documented Cauchy glitches can distort short-time curvature and the inferred relaxation times.); `Use the generic relaxation skill directly` (rejected: Its scalar anchored relaxation does not represent the joint two-branch memory and onset-rate constraints.)

```bash
cd /holdout_task && cat > alpha_m97_c01_s00_solver.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


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
        gain, bias = np.polyfit(cmd, phys, 1)
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
    tau = np.exp(logtau)
    end = np.cumsum(dt[::-1])[::-1] - dt
    a = end[:, None] / tau
    b = dt[:, None] / tau
    decay = np.exp(-a)
    w = decay * -np.expm1(-b)
    gap = hold[:, None] - past @ w
    u = t[:, None] / tau
    step = -np.expm1(-u)
    curves = np.concatenate((gap[:, None, :] * step, x[:, None, :] * t[None, :, None]), axis=2)
    rates = np.column_stack((gap / tau, x))
    if not derivative:
        return curves, rates
    dgap = -past @ (a * w - b * np.exp(-a-b))
    dc = dgap[:, None, :] * step - gap[:, None, :] * u * np.exp(-u)
    dr = (dgap - gap) / tau
    return curves, rates, dc, dr


def _mad(a):
    return max(1e-7, 1.4826 * np.median(np.abs(a - np.median(a))))


def _fit(dt, t, past, hold, x, yc, yr):
    def linear(logtau):
        c, r = _basis(logtau, dt, t, past, hold, x)
        cm = c.reshape(-1, 7)
        return np.linalg.lstsq(np.vstack((cm, r)), np.r_[yc.ravel(), yr], rcond=None)[0]

    def get(p):
        c, r, dc, dr = _basis(p[:2], dt, t, past, hold, x, True)
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
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_s00_solver as s
item=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0];d=item['dataset'];g=item['gt'];y=s.solve_alpha_m97_c01_s00(d)
print('test keys',d['test'].keys()); print('gt keys',g.keys()); print('params',y['estimated_taus'],y['hereditary_weights'])
for k,v in g.items():
 a=np.asarray(v);print(k,a.shape,repr(a.ravel()[:8])[:220]);
c=d['calibration_load_cells']
for i in range(4):
 sel=c['lab_id']==i;x=c['cmd_stress'][sel];z=c['true_stress'][sel];p=np.polyfit(x,z,1);print('cal',i,len(x),p,np.max(abs(z-np.polyval(p,x))))
PY
```
- **Container Exit Code**: `0` (`722 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
test keys dict_keys(['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry'])
gt keys dict_keys(['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true'])
params [0.54905694 3.77038479] [ 0.69583235  0.55031183  0.13897409  0.16750122 -0.13800429  0.11741156
 -0.08909816]
taus (2,) array([0.547, 3.626])
theta (7,) array([ 0.69,  0.55,  0.14,  0.17, -0.14,  0.12, -0.09])
vp_theta (2,) array([0., 0.])
stress_gains (4,) array([1., 1., 1., 1.])
stress_biases (4,) array([0., 0., 0., 0.])
te_clean (72, 10) array([0.65400724, 0.9166957 , 1.15637535, 1.39123827, 1.62760612,
       1.86724638, 2.11054554, 2.3571915 ])
te_true_rate0 (72,) array([ 0.62033984,  0.22576952,  0.48881491, -0.9817484 ,  1.51431831,
       -0.24538023,  0.41588179,  2.50878782])
te_hist_true (72, 18) array([ 0.80635594, -1.53020249,  0.2444242 ,  0.03545663, -0.01728959,
       -0.01265362,  2.067     ,  1.04485693])
te_hold_true (72,) array([-0.573845,  0.418512, -0.566401,  0.710061,  0.766983, -0.378641,
       -0.260673,  1.461534])
cal 0 10 [9.99975548e-01 2.04663470e-05] 0.00019043123953488283
cal 1 10 [9.99993103e-01 4.97280839e-06] 0.00017519098003559996
cal 2 10 [ 1.00001216e+00 -4.56445486e-05] 0.0001234997387544734
cal 3 10 [1.00000478e+00 5.05462111e-05] 0.0002649288187752674

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76cb786d42aaf509 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76cb786d (directory).
```

### Turn 4 — `execute_in_remote_docker` (`30371 ms` LLM)
- **Rationale**: Replay the public reference parameters to separate alignment error from fit error, then measure clean prediction errors by lab, stratum, missingness, and horizon. Hold out each lab's response measurements to test transfer across channels, and inspect only the archive layout needed for the documented fallback.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: Finite outputs do not establish accuracy for held-out channels or coupons with missing features, and the specified calibration fallback is not implemented yet.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_s00_solver as s
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; y=s.solve_alpha_m97_c01_s00(d)
pa,h,x=s._prepare(d); tr,te=d['train'],d['test']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
rmse=lambda z: np.sqrt(np.mean(z*z))
print('max abs physical stress',abs(pa).max()); print('relative tau errors',y['estimated_taus']/g['taus']-1, 'weight max absolute error',np.max(abs(y['hereditary_weights']-g['theta'])))
c,r=s._basis(np.log(g['taus']),dt,t,g['te_hist_true'],g['te_hold_true'],x[n:]); dc=te['strain_0'][:,None]+c@g['theta']-g['te_clean']; dr=r@g['theta']-g['te_true_rate0']; print('reference replay RMSE/max',rmse(dc),abs(dc).max(),rmse(dr),abs(dr).max())
ec=y['predicted_strain']-g['te_clean']; er=y['initial_rate_pred']-g['te_true_rate0']; m=np.isnan(te['coupon_features_raw']).any(axis=1)
print('public test RMSE overall, missing features, per horizon',rmse(ec),rmse(er),rmse(ec[m]),rmse(er[m]),np.sqrt(np.mean(ec*ec,axis=0)))
for lab in np.unique(te['lab_id']):
 for st in np.unique(te['stratum']):
  i=(te['lab_id']==lab)&(te['stratum']==st)
  if i.any(): print('public test lab/stratum',lab,st,i.sum(),round(rmse(ec[i]),5),round(rmse(er[i]),5))
for lab in np.unique(tr['lab_id']):
 i=tr['lab_id']!=lab; j=~i; tl,w=s._fit(dt,t,pa[:n][i],h[:n][i],x[:n][i],yc[i],yr[i]); c,r=s._basis(tl,dt,t,pa[:n][j],h[:n][j],x[:n][j]); a=c@w-yc[j]; b=r@w-yr[j]; print('OOF lab',lab,'tau',np.exp(tl),'curve/rate MAE median',np.median(abs(a)),np.median(abs(b)),'worst horizon median AE',np.median(abs(a),axis=0).max())
PY
python3 - <<'PY'
from pathlib import Path
import json
p=Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
if p.is_file():
 a=json.loads(p.read_text()); print('archive layout:',type(a).__name__,list(a)[:10] if isinstance(a,dict) else len(a)); print('archive sample:',repr(a)[:1300])
else: print('calibration archive not mounted')
PY
```
- **Container Exit Code**: `0` (`696 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
max abs physical stress 2.0670707734505207
relative tau errors [0.0037604 0.0398193] weight max absolute error 0.00583234606097538
reference replay RMSE/max 0.1694422447776063 1.8038596231450617 0.05356092442925188 0.3468960813740507
public test RMSE overall, missing features, per horizon 0.17178333888301894 0.053867540221721716 0.5461364358332674 0.17117750060157333 [0.01617012 0.04559678 0.07513494 0.10473446 0.13437038 0.16401101
 0.19365774 0.22329615 0.2529361  0.2825666 ]
public test lab/stratum 0 0 6 0.02375 0.0082
public test lab/stratum 0 1 9 0.23379 0.07313
public test lab/stratum 0 2 3 0.02185 0.00822
public test lab/stratum 1 0 4 0.01113 0.00403
public test lab/stratum 1 1 9 0.02043 0.00668
public test lab/stratum 1 2 3 0.31131 0.09725
public test lab/stratum 2 0 7 0.02737 0.00893
public test lab/stratum 2 1 1 0.00795 0.00219
public test lab/stratum 2 2 13 0.08126 0.0252
public test lab/stratum 3 0 5 0.49741 0.15623
public test lab/stratum 3 1 3 0.01902 0.00696
public test lab/stratum 3 2 9 0.02011 0.00616
OOF lab 0 tau [0.54529645 3.71762438] curve/rate MAE median 0.02516539335269863 0.01490442402107639 worst horizon median AE 0.05930750036667343
OOF lab 1 tau [0.5489129  3.69685388] curve/rate MAE median 0.02328468976962217 0.016869399297014642 worst horizon median AE 0.04609069149384226
OOF lab 2 tau [0.55158502 3.82007826] curve/rate MAE median 0.01882026629951298 0.0175792239632641 worst horizon median AE 0.029300513145641416
OOF lab 3 tau [0.54812252 3.73271927] curve/rate MAE median 0.023194752377911076 0.020974024266023683 worst horizon median AE 0.04425239901757172
calibration archive not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76cb786d42aaf509 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76cb786d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`33439 ms` LLM)
- **Rationale**: Use public labels only for diagnostic replay: test whether the error is exactly a terminal-drift offset and whether a documented mean convention explains it. If none does, retain the specified imputation and quantify the resulting limitation rather than fitting to public test labels.
- **Alternatives Weighed**: `Retune the Prony fit immediately` (rejected: Public reference replay reproduces almost all of the prediction error, so optimizer tuning is unlikely to fix the main discrepancy.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c01_s00_solver as s
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); t=d['eval_times']; _,_,x=s._prepare(d)
c,r=s._basis(np.log(g['taus']),d['past_dt'],t,g['te_hist_true'],g['te_hold_true'],x[n:]); pred=r@g['theta']; delta=g['te_true_rate0']-pred
print('reference curve/rate residual identity max:', np.max(abs(g['te_clean']-(te['strain_0'][:,None]+c@g['theta']+delta[:,None]*t))))
raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); lab=np.r_[tr['lab_id'],te['lab_id']]; st=np.r_[tr['stratum'],te['stratum']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); missing=np.isnan(raw); complete=~missing[n:].any(1); print('complete-row max rate residual:',abs(delta[complete]).max())
print('row lab stratum col current implied zero labmean allsame-stratum traincell raw-inferred')
for j in np.flatnonzero(~complete):
 i=n+j; l=lab[i]; a=st[i]; m=missing[i]; panel=ref[lab==l]; center=np.nanmedian(np.nanmean(panel,axis=0)); scale=np.nanmedian(np.nanstd(panel,axis=0)); aligned=(raw-center)/scale; w=g['theta'][3:]; current=x[i,1:][m]@w[m]; implied=current+delta[j]; print(j,l,a,np.flatnonzero(m),np.round([current,implied,0,np.nanmean(aligned[lab==l],axis=0)[m]@w[m],np.nanmean(aligned[st==a],axis=0)[m]@w[m],np.nanmean(aligned[(lab==l)&(st==a)&(np.arange(len(lab))<n)],axis=0)[m]@w[m]],6), np.round(center+scale*implied/w[m],6) if m.sum()==1 else '-')
for k in range(4):
 print('raw feature',k,'covariance/correlation', np.round(np.ma.corrcoef(np.ma.masked_invalid(raw),rowvar=False).filled(np.nan)[k],3))
print('reference-vs-feature correlations by lab')
for l in range(4):
 q=lab==l; corr=np.ma.corrcoef(np.ma.masked_invalid(np.c_[raw[q],ref[q]]),rowvar=False).filled(np.nan)[:4,4:]; print(l, np.round(np.nanmax(abs(corr),axis=1),3))
PY
ls -la /tmp/hegel_continual_archive /pretrain_task 2>/dev/null || true
```
- **Container Exit Code**: `0` (`999 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
reference curve/rate residual identity max: 7.993605777301127e-15
complete-row max rate residual: 0.04129352320202706
row lab stratum col current implied zero labmean allsame-stratum traincell raw-inferred
0 0 1 [0] [-0.024422  0.093684  0.       -0.022076  0.161529 -0.015719] [-0.162466]
3 3 0 [1] [ 0.054215 -0.292681  0.        0.006627  0.107553  0.064533] [3.696374]
12 0 1 [3] [ 0.037208 -0.14241   0.        0.003278 -0.045318  0.026982] [0.694223]
25 1 2 [1] [-0.062142  0.09937   0.       -0.029556  0.018361 -0.070255] [-0.421197]
34 2 2 [1] [-0.001538  0.088797  0.        0.002086 -0.083922  0.007957] [-0.866733]
35 2 2 [3] [-0.034659 -0.033792  0.       -0.014449 -0.054136 -0.018946] [0.082619]
52 2 0 [1] [ 0.036979  0.017459  0.        0.002086 -0.015844  0.07774 ] [-0.387647]
raw feature 0 covariance/correlation [1.    0.286 0.343 0.271]
raw feature 1 covariance/correlation [0.286 1.    0.266 0.227]
raw feature 2 covariance/correlation [0.343 0.266 1.    0.17 ]
raw feature 3 covariance/correlation [0.271 0.227 0.17  1.   ]
reference-vs-feature correlations by lab
0 [0.241 0.308 0.348 0.289]
1 [0.246 0.256 0.361 0.306]
2 [0.286 0.257 0.271 0.373]
3 [0.265 0.341 0.383 0.346]
/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 08:39 ..

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76cb786d42aaf509 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76cb786d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`19599 ms` LLM)
- **Rationale**: The reference curve residual equals the rate residual times elapsed time to numerical precision, so changing the hereditary model would not address it. Check the provided panel primitive for a balancing convention the implementation may have missed and the asset index for the archive format before implementing the outstanding calibration fallback.
- **Alternatives Weighed**: `Infer missing coupon values from public target strains` (rejected: That would use test labels and violate the requested lab–stratum mean imputation.)

```bash
cd /holdout_task && grep -inE -C 2 'archive|calibration|imput|panel|runtime' public_asset_index.json; PYTHONPATH=/tmp python3 - <<'PY'
import inspect
from hegel_wormhole_skills.autogenous_prim_reference_panel_balancing import align_reference_panel,weighted_column_fill
for fn in (align_reference_panel,weighted_column_fill):
 print(inspect.getsource(fn))
PY
```
- **Container Exit Code**: `0` (`416 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
roups, return_inverse=True)
    result = np.full(values.shape, np.nan)
    for (code, level) in enumerate(levels):
        rows = codes == code
        panel = controls[rows]
        observed = np.isfinite(panel)
        counts = observed.sum(axis=0)
        keep = counts >= 2
        if not keep.any():
            raise ValueError(f'group {level!r} has no control with two finite observations')
        (panel, observed, counts) = (panel[:, keep], observed[:, keep], counts[keep])
        means = np.where(observed, panel, 0.0).sum(axis=0) / counts
        residuals = np.where(observed, panel, means) - means
        spreads = np.sqrt(np.sum(residuals ** 2, axis=0) / counts)
        (offset, gain) = (np.median(means), np.median(spreads))
        if not np.isfinite(offset) or not np.isfinite(gain) or gain <= 0:
            raise ValueError(f'group {level!r} requires a finite offset and positive finite gain')
        block = values[rows]
        result[rows] = (np.where(np.isfinite(block), block, np.nan) - offset) / gain
    return result

@track_wormhole_skill("autogenous_prim_reference_panel_balancing")
def weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."""
    values = _real_array(values, 'values').astype(float, copy=False)
    weights = _real_array(weights, 'weights').astype(float, copy=False)
    if values.ndim != 2 or not values.shape[0] or (not values.shape[1]):
        raise ValueError('values must be a nonempty matrix')
    if weights.shape != (values.shape[0],):
        raise ValueError('weights must be a vector matching value rows')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('weights must be finite and strictly positive')
    observed = np.isfinite(values)
    support = weights @ observed
    numerator = weights @ np.where(observed, values, 0.0)
    if not np.isfinite(support).all() or not np.isfinite(numerator).all():
        raise ValueError('weighted column sums must be finite')
    return np.divide(numerator, support, out=np.zeros(values.shape[1]), where=support > 0)


[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76cb786d42aaf509 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76cb786d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`41992 ms` LLM)
- **Rationale**: Implement the missing calibration branches without changing the clean-channel fit. The isolated checks would catch an unnecessary archive read, failure to recover sparse-channel coefficients, or leakage of an EMI spike into the fitted calibration.
- **Alternatives Weighed**: `Replace lab–stratum imputation to reduce public test error` (rejected: The measured discrepancy is a terminal-drift offset, and the task explicitly requires lab–stratum mean imputation.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_s00_solver.py')
s = p.read_text().replace('import numpy as np', 'import json\nfrom pathlib import Path\n\nimport numpy as np', 1)
start = '''_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _CALIBRATION_ARCHIVE.is_file():
        return None
    archive = json.loads(_CALIBRATION_ARCHIVE.read_text())
    if 'stress_gains' in archive and 'stress_biases' in archive:
        return float(archive['stress_gains'][channel]), float(archive['stress_biases'][channel])
    channels = archive.get('channels', archive)
    record = channels.get(str(channel)) if isinstance(channels, dict) else channels[channel]
    if record is None:
        return None
    return float(record['gain']), float(record['bias'])


def _calibrate(cmd, phys, channel):
    if len(cmd) >= 3:
        i, j = np.triu_indices(len(cmd), 1)
        dx = cmd[j] - cmd[i]
        keep = dx != 0
        slopes = (phys[j[keep]] - phys[i[keep]]) / dx[keep]
        offsets = np.median(phys[:, None] - cmd[:, None] * slopes, axis=0)
        error = np.abs(phys[:, None] - cmd[:, None] * slopes - offsets)
        robust = np.argmin(np.median(error, axis=0))
        clean = error[:, robust] <= 3.0
    else:
        clean = np.ones(len(cmd), dtype=bool)
    if len(cmd) < 2 or not clean.all():
        prior = _archived_calibration(channel)
        if prior is not None and len(cmd) >= 2:
            prior_clean = np.abs(phys - (prior[0] * cmd + prior[1])) <= 3.0
            if prior_clean.sum() >= clean.sum():
                clean = prior_clean
    else:
        prior = None
    if np.unique(cmd[clean]).size >= 2:
        return np.polyfit(cmd[clean], phys[clean], 1)
    if prior is not None:
        return prior
    raise ValueError(f'Insufficient load-cell calibration for channel {channel}')


'''
s = s.replace('def _prepare(dataset):', start + 'def _prepare(dataset):', 1)
s = s.replace('gain, bias = np.polyfit(cmd, phys, 1)', 'gain, bias = _calibrate(cmd, phys, int(channel))', 1)
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, tempfile, json
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
x=np.linspace(-2,2,8); y=1.2*x-.25
with tempfile.TemporaryDirectory() as td:
    m._CALIBRATION_ARCHIVE=Path(td)/'archive.json'
    np.testing.assert_allclose(m._calibrate(x,y,2),[1.2,-.25])
    spike=y.copy(); spike[-1]+=8
    np.testing.assert_allclose(m._calibrate(x,spike,2),[1.2,-.25])
    m._CALIBRATION_ARCHIVE.write_text(json.dumps({'channels':{'2':{'gain':1.2,'bias':-.25}}}))
    for n in [0,1]:
        np.testing.assert_allclose(m._calibrate(x[:n],y[:n],2),[1.2,-.25])
    np.testing.assert_allclose(m._calibrate(x,spike,2),[1.2,-.25])
    m._CALIBRATION_ARCHIVE.write_text('unreadable')
    np.testing.assert_allclose(m._calibrate(x,y,2),[1.2,-.25])
print('Calibration checks passed: clean bypass, sparse archive fallback, and EMI rejection with and without archive.')
PY
```
- **Container Exit Code**: `0` (`861 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Calibration checks passed: clean bypass, sparse archive fallback, and EMI rejection with and without archive.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=83c5a4fd89df0e1e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_83c5a4fd (directory).
```

### Turn 8 — `execute_in_remote_docker` (`29527 ms` LLM)
- **Rationale**: Verify the final artifact’s prediction slices in an isolated process. If imputed training inputs are distorting the fit despite the robust loss, excluding those rows should improve the worst held-out-lab metrics without changing test imputation. Otherwise retain the verified candidate.
- **Alternatives Weighed**: `Change the required lab–stratum imputation` (rejected: The benchmark specifies the imputation rule, and public target values cannot be used to fill missing inputs.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c01_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); pa,h,x=s._prepare(d); t,dt=d['eval_times'],d['past_dt']; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; observed=~np.isnan(tr['coupon_features_raw']).any(axis=1); print('complete training coupons',observed.sum(),n)
y=s.solve_alpha_m97_c01_s00(d); np.testing.assert_array_equal(y['regime_mode_weights'],np.zeros(2)); ec=y['predicted_strain']-g['te_clean']; er=y['initial_rate_pred']-g['te_true_rate0']; rmse=lambda a: np.sqrt(np.mean(a*a));
for label,i in [('all',np.ones(len(er),dtype=bool)),('complete',~np.isnan(te['coupon_features_raw']).any(axis=1)),('imputed',np.isnan(te['coupon_features_raw']).any(axis=1))]: print('public slice',label,'strain/rate RMSE',rmse(ec[i]),rmse(er[i]))
for mode in ['all','complete']:
 metrics=[]
 for lab in np.unique(tr['lab_id']):
  val=tr['lab_id']==lab; fit=~val if mode=='all' else ~val & observed; z,w=s._fit(dt,t,pa[:n][fit],h[:n][fit],x[:n][fit],yc[fit],yr[fit]); c,r=s._basis(z,dt,t,pa[:n][val],h[:n][val],x[:n][val]); a=np.abs(c@w-yc[val]); b=np.abs(r@w-yr[val]); metrics.append([np.median(a,axis=0).max(),np.median(b),*np.exp(z)]); print('OOF',mode,int(lab),np.round(metrics[-1],6))
 print('worst OOF horizon/rate MedAE',mode,np.max(np.array(metrics)[:,:2],axis=0))
PY
```
- **Container Exit Code**: `0` (`1086 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete training coupons 160 180
public slice all strain/rate RMSE 0.17178333888301894 0.053867540221721716
public slice complete strain/rate RMSE 0.02380370736115653 0.007657172666986189
public slice imputed strain/rate RMSE 0.5461364358332674 0.17117750060157333
OOF all 0 [0.059308 0.014904 0.545296 3.717624]
OOF all 1 [0.046091 0.016869 0.548913 3.696854]
OOF all 2 [0.029301 0.017579 0.551585 3.820078]
OOF all 3 [0.044252 0.020974 0.548123 3.732719]
worst OOF horizon/rate MedAE all [0.0593075  0.02097402]
OOF complete 0 [0.058569 0.015594 0.545695 3.728512]
OOF complete 1 [0.04438  0.015936 0.548725 3.719735]
OOF complete 2 [0.027378 0.017294 0.551654 3.863987]
OOF complete 3 [0.043877 0.021411 0.549136 3.793023]
worst OOF horizon/rate MedAE complete [0.05856882 0.02141098]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=83c5a4fd89df0e1e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_83c5a4fd (directory).
```

### Turn 9 — `execute_in_remote_docker` (`35887 ms` LLM)
- **Rationale**: A slightly offset archive can classify a just-over-threshold EMI spike as clean and override a well-supported in-batch fit. Enforce the specified precedence and test that case alongside sparse fallback and public prediction slices on the edited artifact.
- **Alternatives Weighed**: `Exclude imputed training coupons` (rejected: Worst horizon median absolute error improved by 0.00074, but worst rate median absolute error worsened by 0.00044; this does not address the missing-feature drift error.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_s00_solver.py')
s = p.read_text()
old = 'if prior is not None and len(cmd) >= 2:'
assert s.count(old) == 1
p.write_text(s.replace(old, 'if prior is not None and len(cmd) >= 2 and clean.sum() < 4:'))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
x = np.linspace(-2, 2, 8); y = 1.2*x-.25
with tempfile.TemporaryDirectory() as td:
    s._CALIBRATION_ARCHIVE = Path(td)/'archive.json'
    s._CALIBRATION_ARCHIVE.write_text(json.dumps({'channels': {'2': {'gain': 1.2, 'bias': -.05}}}))
    spike = y.copy(); spike[-1] += 3.1
    np.testing.assert_allclose(s._calibrate(x, spike, 2), [1.2, -.25])
    np.testing.assert_allclose(s._calibrate(x[:1], y[:1], 2), [1.2, -.05])
    s._CALIBRATION_ARCHIVE.write_text('unreadable')
    np.testing.assert_allclose(s._calibrate(x, y, 2), [1.2, -.25])
p = pickle.load(open('public_alpha_m97_c01_s00.pkl', 'rb'))[0]
d, g = p['dataset'], p['gt']; out = s.solve_alpha_m97_c01_s00(d)
np.testing.assert_array_equal(out['regime_mode_weights'], np.zeros(2))
imputed = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
for key, ref, worst in [('predicted_strain', 'te_clean', .547), ('initial_rate_pred', 'te_true_rate0', .172)]:
    e = out[key] - g[ref]
    rmse = [np.sqrt(np.mean(e[i]**2)) for i in (~imputed, imputed)]
    assert max(rmse) < worst, (key, rmse)
    print(key, 'complete/imputed RMSE:', rmse)
print('Calibration precedence, sparse fallback, and clean archive bypass passed.')
PY
```
- **Container Exit Code**: `0` (`827 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
predicted_strain complete/imputed RMSE: [0.02380370736115653, 0.5461364358332674]
initial_rate_pred complete/imputed RMSE: [0.007657172666986189, 0.17117750060157333]
Calibration precedence, sparse fallback, and clean archive bypass passed.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e77991532fe4a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9e779915 (directory).
```

### Turn 10 — `submit_completed_task` (`16686 ms` LLM)
- **Rationale**: The final artifact passed isolated schema, calibration-precedence, and prediction-slice checks. Replaying the reference parameters localized the remaining discrepancy to terminal drift, so retuning the hereditary branches would not address the main error.
- **Alternatives Weighed**: `Exclude imputed training coupons` (rejected: It slightly improved the worst held-out horizon error but worsened the worst rate error and did not address the missing-input drift discrepancy.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_s00_solver.py': 'FILE:6002'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema and calibration checks passed. Public strain RMSE was 0.0238 for complete coupons and 0.5461 for coupons requiring imputation.`

### Turn 11 — `execute_in_remote_docker` (`36747 ms` LLM)
- **Rationale**: Compare the solver against an independently stepped recurrence to catch sign, time-index, or derivative errors. Check telemetry invariance and signed held-out-lab residuals for nuisance leakage. Use the public reference only to diagnose whether the remaining complete-coupon drift error points to a panel-alignment mismatch, not to fit a correction.
- **Alternatives Weighed**: `Add rational terms, relay states, or classification metrics` (rejected: This is continuous regression in a specified linear-flow regime, with no relay states or class labels. The applicable checks are recurrence equivalence and signed residuals within held-out labs.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p = pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]
d,g = p['dataset'],p['gt']; tr,te = d['train'],d['test']; t,dt = d['eval_times'],d['past_dt']
past,hold,x = s._prepare(d); out = s.solve_alpha_m97_c01_s00(d)
tau = out['estimated_taus']; w = out['hereditary_weights']; n = len(tr['lab_id'])
z = np.zeros((len(hold),2))
for j, step in enumerate(dt):
    a = np.exp(-step/tau)
    z = a*z + (1-a)*past[:,j,None]
gap = hold[:,None]-z
drift = x@w[2:]
strain = (gap[n:]*w[:2])@(-np.expm1(-t[None,:]/tau[:,None])) + drift[n:,None]*t + te['strain_0'][:,None]
rate = (gap[n:]*w[:2])@(1/tau) + drift[n:]
np.testing.assert_allclose(out['predicted_strain'],strain,atol=1e-12,rtol=0)
np.testing.assert_allclose(out['initial_rate_pred'],rate,atol=1e-12,rtol=0)
assert np.max(np.abs(past)) < d['crit_stress'] and d['shear_banding_index'] <= .50
print('exact recurrence, envelope, and derivative agree; stress margin:',d['crit_stress']-np.max(np.abs(past)))
q = dict(d)
for k in ('train','test'):
    q[k] = dict(d[k]); q[k].pop('aux_chamber_telemetry',None)
without = s.solve_alpha_m97_c01_s00(q)
for k in out:
    np.testing.assert_array_equal(out[k],without[k])
print('all five outputs invariant to removing uncoupled telemetry')
c,r = s._basis(np.log(g['taus']),dt,t,g['te_hist_true'],g['te_hold_true'],x[n:]); theta=g['theta']; delta=g['te_true_rate0']-r@theta
clean = ~np.isnan(te['coupon_features_raw']).any(axis=1)
for lab in np.unique(te['lab_id']):
    j = (te['lab_id']==lab)&clean
    signal = x[n:][j,1:]@theta[3:]
    scale,off=np.linalg.lstsq(np.c_[signal,np.ones(j.sum())],delta[j],rcond=None)[0]
    print('complete lab',lab,'reference drift RMSE before/after affine diagnostic',np.sqrt(np.mean(delta[j]**2)),np.sqrt(np.mean((delta[j]-scale*signal-off)**2)),'slope/offset residual',scale,off)
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    j=tr['lab_id']==lab; i=~j
    logs,coef=s._fit(dt,t,past[:n][i],hold[:n][i],x[:n][i],yc[i],yr[i]); c,r=s._basis(logs,dt,t,past[:n][j],hold[:n][j],x[:n][j]); e=c@coef-yc[j]; er=r@coef-yr[j]
    print('held-out lab',lab,'median rate residual',np.median(er),'median final-horizon residual',np.median(e[:,-1]),'curve/rate median AE',np.median(np.abs(e)),np.median(np.abs(er)))
PY
```
- **Container Exit Code**: `0` (`863 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
exact recurrence, envelope, and derivative agree; stress margin: 0.5829292265494792
all five outputs invariant to removing uncoupled telemetry
complete lab 0 reference drift RMSE before/after affine diagnostic 0.003824166130985694 3.700350406554467e-16 slope/offset residual 0.013086264124972768 0.0021755700814920596
complete lab 1 reference drift RMSE before/after affine diagnostic 0.009898263120366702 4.918701211948971e-16 slope/offset residual -0.03545538206187612 -0.0011662137017653992
complete lab 2 reference drift RMSE before/after affine diagnostic 0.0013232578498554206 5.367946697648364e-16 slope/offset residual 0.00021273517602826556 -0.0013019496021385514
complete lab 3 reference drift RMSE before/after affine diagnostic 0.015082101063698975 4.443521622008104e-16 slope/offset residual -0.04905664403806269 0.003100822436036397
held-out lab 0 median rate residual -0.004297659261346731 median final-horizon residual 0.008929150178178524 curve/rate median AE 0.02516539335269863 0.01490442402107639
held-out lab 1 median rate residual 0.0048632398029922475 median final-horizon residual 0.008235434392268703 curve/rate median AE 0.02328468976962217 0.016869399297014642
held-out lab 2 median rate residual 0.0032288836235307983 median final-horizon residual 0.0045989927156890875 curve/rate median AE 0.01882026629951298 0.0175792239632641
held-out lab 3 median rate residual -0.009319109403928855 median final-horizon residual 0.0052608454644724745 curve/rate median AE 0.023194752377911076 0.020974024266023683

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e77991532fe4a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9e779915 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`24620 ms` LLM)
- **Rationale**: On complete coupons, an affine transformation explains the remaining reference drift discrepancy to numerical precision in every lab. Compare its diagnostic center and scale with reference-panel summaries calculated without target values. Change the solver only if the match identifies a convention consistent with the task.
- **Alternatives Weighed**: `Fit lab-specific drift corrections to the public test targets` (rejected: That would leak target values into the estimator rather than resolve the panel-alignment convention.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c01_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; raw=te['coupon_features_raw']; theta=g['theta']; t=d['eval_times']; dt=d['past_dt']; n=len(tr['lab_id']); _,_,x=s._prepare(d)
_,r=s._basis(np.log(g['taus']),dt,t,g['te_hist_true'],g['te_hold_true'],x[n:]); drift=g['te_true_rate0']-r[:,:2]@theta[:2]-theta[2]
for lab in range(4):
 a=tr['ref_coupons'][tr['lab_id']==lab]; b=te['ref_coupons'][te['lab_id']==lab]; both=np.vstack((a,b)); j=(te['lab_id']==lab)&np.isfinite(raw).all(1); slope,offset=np.linalg.lstsq(np.c_[raw[j]@theta[3:],np.ones(j.sum())],drift[j],rcond=None)[0]; signature=np.array([-offset/slope/theta[3:].sum(),1/slope]); print('lab',lab,'diagnostic center/scale',signature)
 for key,panel in [('pooled',both),('train',a),('test',b),('unique',np.unique(both,axis=0))]:
  print(key,np.array([np.nanmedian(np.nanmean(panel,axis=0)),np.nanmedian(np.nanstd(panel,axis=0))])-signature)
 print('equal-split stats',.5*sum(np.array([np.nanmedian(np.nanmean(q,axis=0)),np.nanmedian(np.nanstd(q,axis=0))]) for q in (a,b))-signature)
PY
```
- **Container Exit Code**: `0` (`503 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 diagnostic center/scale [-0.65  0.82]
pooled [0.02973279 0.01073074]
train [ 0.03542765 -0.01282475]
test [-0.05118562 -0.02929701]
unique [0.02973279 0.01073074]
equal-split stats [-0.00787899 -0.02106088]
lab 1 diagnostic center/scale [0.48 1.28]
pooled [-0.02487923 -0.04538289]
train [ 0.00442655 -0.07670025]
test [-0.01770982 -0.04449535]
unique [-0.02487923 -0.04538289]
equal-split stats [-0.00664164 -0.0605978 ]
lab 2 diagnostic center/scale [-0.25  0.94]
pooled [-0.02039721  0.00019997]
train [-0.04724362 -0.00942371]
test [-0.00552683 -0.02227747]
unique [-0.02039721  0.00019997]
equal-split stats [-0.02638523 -0.01585059]
lab 3 diagnostic center/scale [0.8  1.42]
pooled [ 0.07338613 -0.06966043]
train [ 0.06915662 -0.08838295]
test [ 0.01918342 -0.10526367]
unique [ 0.07338613 -0.06966043]
equal-split stats [ 0.04417002 -0.09682331]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e77991532fe4a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9e779915 (directory).
```

### Turn 13 — `submit_completed_task` (`17742 ms` LLM)
- **Rationale**: The saved artifact passed isolated validation. Independent recurrence replay matched the predictions, all five outputs were invariant to uncoupled telemetry, and the observed stresses stayed below the critical threshold. The main remaining error is on coupons requiring imputation; the audit did not identify a better alignment consistent with the specified input-only rule.
- **Alternatives Weighed**: `Replace pooled panel statistics with a lab-specific affine correction` (rejected: The alternative panel summaries did not reproduce the reference alignment, and fitting the correction from public test targets would leak labels.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema, calibration fallback, and exact memory recurrence checks passed. Public strain RMSE was 0.0238 for complete coupons and 0.5461 for coupons requiring the specified imputation.`

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
