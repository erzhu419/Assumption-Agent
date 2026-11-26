# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w03_batch_04__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w03_step_04` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `474,208` (`459,466` / `127,872` / `14,742`) |
| **Total Episode Wall Time (s)** | `467.12s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_10_split_preloaded_creep_transfer`**: [BIFURCATED SUB-REGIME `candidate_clean_command_span_over_comman >= 0.5`]: Pooling raw coupon features across labs confounds their affine measurement scales, and even cell-balanced imputation cannot recover specimen-specific drift: the public missing-feature rows had strain RMSE 0.4227 versus 0.00458 on complete rows, with residual slopes correlated 0.998 with initial-rate residuals. Align each lab to its own reference panel, average the observed lab–stratum cell means with equal weight per feature, and fit the exact preloaded two-mode recurrence with a soft-L1 pilot and Cauchy refinement on separately scaled strain and rate residuals; a relative Jacobian singular-value threshold of 1e-8 rejects locally rank-deficient fits. On imputed rows with a finite independent initial-rate measurement, add the rate residual times hold time without changing either hereditary amplitude; the public probe of this correction reported strain MAE 0.004857 and relative relaxation-time error 0.002965. Accept an independent current-batch calibration only when at least four checks and a strict majority pass the residual gate and span at least 0.5 of the characteristic command scale. Otherwise compare nominal and same-lab historical lines against the current checks: archived gains of 0.779–1.239 did not persist reliably, and two endpoints fit exactly even if one is an EMI glitch, so this refactor requires prior corroboration or redundant current checks before changing the slope; cross-epoch persistence and unseen-regime extrapolation remain unverified. (Counter-regime `candidate_clean_command_span_over_comman` warning: Below the span gate, calibration is not independently supported by a clean majority sweep. Select among nominal and historical lines using the current checks; a two-point line without corroboration cannot distinguish a gain change from EMI. The separate hold-prediction correction applies only to rows with imputed features and a finite independent initial rate.)
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Cell-balanced feature imputation alone supplies the coupon-specific drift accurately enough for hold prediction.` (witness: `Public missing-feature rows have strain RMSE 0.4227 versus 0.00458 on complete rows. Their residual is almost entirely linear in time and its slope correlates 0.998 with the initial-rate residual.`) -> Pivot `Keep the calibrated two-mode recurrence and cell-balanced alignment. Correct only the drift on imputed rows with the measured initial rate minus the fitted initial rate.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The latest supported historical load-cell gains and biases persist into the current batch whenever its sweeps are sparse.` (witness: `Current gains are near 1, whereas archived gains range from 0.779 to 1.239. Thinning the current checks on one wing caused deviations up to 0.698 stress units even with two clean endpoints.`) -> Pivot `Treat the archived and nominal lines as alternatives, use current checks to select between them, and update the slope when clean checks span enough stress.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15902 ms` LLM)
- **Rationale**: Inspect the public observations and archived solver interfaces to identify the inherited feature basis and determine whether current calibration can stand on its own.

```bash
cd /holdout_task; cat validate_public_schema.py; python3 validate_public_schema.py; ls -lh /tmp/hegel_continual_archive/; python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_w03_batch_04.pkl','rb') as f: d=pickle.load(f)
def show(d,depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,getattr(v,'shape',''), (str(v)[:350] if not isinstance(v,dict) else '')) if not isinstance(v,np.ndarray) else print(' '*depth,k,v.shape,v.dtype,v.ravel()[:5]);
  if isinstance(v,dict) and depth<3: show(v, depth+1)
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation robust IRLS' 2>&1 | head -30; for x in /tmp/hegel_continual_archive/*solver.py; do echo ==== "$x"; grep -nE '^def |^class |^import |^from ' "$x"; done
```
- **Container Exit Code**: `0` (`773 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
togenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
==== /tmp/hegel_continual_archive/alpha_w03_batch_01_solver.py
2:from __future__ import annotations
3:import base64, copy, types
4:import numpy as np
14:def solve_alpha_w03_batch_01(dataset: dict) -> dict:
==== /tmp/hegel_continual_archive/alpha_w03_batch_02_solver.py
2:from __future__ import annotations
3:import base64, copy, types
4:import numpy as np
14:def solve_alpha_w03_batch_02(dataset: dict) -> dict:
==== /tmp/hegel_continual_archive/alpha_w03_batch_03_solver.py
2:import json
3:from pathlib import Path
5:import numpy as np
6:from scipy.optimize import least_squares
9:def _line(x, y):
30:def _checks(calibration, lab):
37:def _supported_line(x, y):
49:def _archive_calibration(labs):
64:def _sparse_line(x, y, historical):
91:def _prep(d):
125:def _design(tau, x, h, p, dt, t, initial):
148:def _fit(x, h, p, dt, t, initial, y, rate):
199:def _crossings(strain0, curves, critical, times):
214:def solve_alpha_w03_batch_03(dataset: dict) -> dict:
==== /tmp/hegel_continual_archive/alpha_w03_s00_solver.py
2:import json
3:import pickle
4:from pathlib import Path
6:import numpy as np
7:from scipy.optimize import least_squares
10:def _line(x, y):
31:def _checks(calibration, lab):
38:def _supported_line(x, y):
50:def _archive_calibration(labs):
81:def _sparse_line(x, y, prior):
98:def _prep(d):
132:def _design(tau, x, h, p, dt, t):
151:def _fit(x, h, p, dt, t, y, rate):
202:def solve_alpha_w03_s00(dataset: dict) -> dict:
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 6, in <module>
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 991, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1128, in get_code
  File "<frozen importlib._bootstrap_external>", line 1186, in get_data
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/alpha_w03_batch_04_solver.py'
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`14834 ms` LLM)
- **Rationale**: Read the dataset inside its list wrapper and inspect the prior constitutive basis before coding; a changed reference-panel layout or observation convention would change preprocessing or the strain and rate equations.

```bash
cd /holdout_task
python3 - <<'PY'
import pickle,numpy as np
z=pickle.load(open('public_alpha_w03_batch_04.pkl','rb'));print('cases',len(z),'case keys',list(z[0]));d=z[0]['dataset']
def show(a,indent=''):
 for k,v in a.items():
  if isinstance(v,dict):
   print(indent+k);show(v,indent+'  ')
  elif isinstance(v,np.ndarray): print(indent+k,v.shape,v.dtype,v.ravel()[:4])
  else: print(indent+k,type(v).__name__,str(v)[:230])
show(d)
PY
sed -n '1,198p' /tmp/hegel_continual_archive/alpha_w03_batch_03_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_preloaded_creep_transfer.align_panel_covariates | head -30
```
- **Container Exit Code**: `0` (`578 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ts(u):
            if cache[0] is None or not np.array_equal(cache[0], u[:2]):
                cache[:] = [u[:2].copy(), *design(decode(u))]
            return cache[1:]
        def fun(u):
            a, _ = parts(u)
            return (a @ u[2:] - obs)/scales
        def jac(u):
            a, d = parts(u)
            d = d*u[2:4]
            j = np.column_stack((d[:, 0] + (1.-u[1])*d[:, 1], (8.-u[0])*d[:, 1], a))
            return j/scales[:, None]
        return least_squares(fun, v, jac=jac, loss=loss, x_scale='jac',
                             bounds=([.2, 1e-6] + [-np.inf]*7, [7.9999, 1.] + [np.inf]*7),
                             max_nfev=150, ftol=2e-10)

    def spread(r):
        return max(1e-6, 1.4826*np.median(np.abs(r - np.median(r))))

    def scales_for(res, factor):
        return np.r_[np.full(n*k, factor*spread(res[:n*k])), np.full(n, factor*spread(res[n*k:]))]

    starts = []
    for lo, hi in ((.24, 1.4), (.4, 2.8), (.75, 4.0), (1.4, 6.5), (2.9, 7.6)):
        a, _ = design([lo, hi])
        coef = np.linalg.lstsq(a, obs, rcond=None)[0]
        ini = np.r_[lo, (hi-lo)/(8.-lo), coef]
        starts.append(ini)
    pilot = starts[1]
    a, _ = design(decode(pilot))
    scales = scales_for(a @ pilot[2:] - obs, 2.5)
    coarse = min((optimize(v, scales, 'soft_l1') for v in starts), key=lambda r: r.cost)
    a, _ = design(decode(coarse.x))
    scales = scales_for(a @ coarse.x[2:] - obs, 2.8)
    result = optimize(coarse.x, scales, 'cauchy')
    return decode(result.x), result.x[2:]


{
  "ok": true,
  "target": "autogenous_preloaded_creep_transfer.align_panel_covariates",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_preloaded_creep_transfer",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "7ccb7445b3f7244a2aae605ef4bc1923982291c4e7bdee10b4167f6a5096737d",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "observations_per_contributing_reference_column",
          "op": ">=",
          "value": 2
        },
        {
          "field": "median_reference_population_standard_deviation",
          "op": ">",
          "value": 0
        },
        {
          "field": "maximum_applied_absolute_stress_over_yield_stress",
          "op": "<",
          "value": 1
        },
        {
          "field": "lower_over_upper_relaxation_time_bound",
```

### Turn 3 — `execute_in_remote_docker` (`16102 ms` LLM)
- **Rationale**: Recover the exact inherited alignment and memory conventions, and measure calibration support per lab. Use archive calibration only where current checks cannot identify a reliable line.
- **Alternatives Weighed**: `Copy the previous batch solver unchanged` (rejected: It omits both plastic basis terms required by the current constitutive contract.)

```bash
sed -n '30,148p;214,260p' /tmp/hegel_continual_archive/alpha_w03_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; a=json.load(open(p));print('archive layout',str(a)[:950]);d=pickle.load(open('/holdout_task/public_alpha_w03_batch_04.pkl','rb'))[0]['dataset'];c=d['calibration_load_cells'];
for l in range(4):
 m=c['lab_id']==l;x=c['cmd_stress'][m];y=c['true_stress'][m];b=np.polyfit(x,y,1);print(l,len(x),np.ptp(x),b,np.round(y-np.polyval(b,x),5));
for k in ('train','test'):
 s=d[k];print(k,'feature missing',np.isnan(s['coupon_features_raw']).sum(axis=0),'reference missing',np.isnan(s['ref_coupons']).sum(),'strata',np.unique(s['stratum']),'rate finite',np.isfinite(s['instant_rate_obs']).sum())
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel | head -45
```
- **Container Exit Code**: `0` (`526 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
        "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
        "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
      },
      "functions": [
        {
          "function": "align_reference_panel",
          "signature": "align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray",
          "doc_summary": "Return aligned real values (n, p) using real controls (n, q) and finite real groups (n,), with nonempty columns and a supported positive control gain per group."
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

### Turn 4 — `execute_in_remote_docker` (`16463 ms` LLM)
- **Rationale**: Recover the inherited strain baseline and memory equations from the truncated output. Compact per-lab residuals will show where current checks need outlier rejection or historical support, rather than pooling across batches by default.

```bash
sed -n '91,147p;214,255p' /tmp/hegel_continual_archive/alpha_w03_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/holdout_task/public_alpha_w03_batch_04.pkl';d=pickle.load(open(p,'rb'))[0]['dataset'];c=d['calibration_load_cells'];hist=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'));print('archive batch row counts',[len(z['lab_id']) for z in hist]);
for l in range(4):
 m=c['lab_id']==l;x=c['cmd_stress'][m];y=c['true_stress'][m];A=np.c_[x,np.ones(len(x))];b=np.linalg.lstsq(A,y,rcond=None)[0];print('calibration',l,len(x),round(np.ptp(x),3),np.round(b,5),np.round(np.sort(y-A@b),5))
for s in ('train','test'):
 z=d[s];print(s,'feature NaNs',np.isnan(z['coupon_features_raw']).sum(axis=0),'reference NaNs',np.isnan(z['ref_coupons']).sum(), 'plastic state nonzero',np.count_nonzero(z.get('residual_plastic_state',[])))
PY
```
- **Container Exit Code**: `0` (`592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
initial*np.exp(-dt.sum()/tau)
    gap -= preload
    dgap -= preload*dt.sum()/tau**2
    q = t[:, None]/tau
    shape = -np.expm1(-q)
    dshape = -np.exp(-q)*q/tau
    curves = np.concatenate((gap[:, None, :]*shape, x[:, None, :]*t[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x))
    dc = dgap[:, None, :]*shape + gap[:, None, :]*dshape
    dr = (dgap - gap/tau)/tau
    return curves, rates, dc, dr


def solve_alpha_w03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, h, p = _prep(dataset)
    tau, theta = _fit(x[:n], h[:n], p[:n], dt, t, train['initial_memory_states'],
                      train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    curves, rates, _, _ = _design(tau, x[n:], h[n:], p[n:], dt, t, test['initial_memory_states'])
    increment = curves @ theta
    # The hold derivative at zero is the hereditary rate plus constant drift.
    # When a coupon covariate was imputed, use the independently measured
    # initial rate to correct that coupon's drift, not its memory amplitudes.
    missing = np.isnan(test['coupon_features_raw']).any(axis=1)
    rate = np.asarray(test['instant_rate_obs'])
    use = missing & np.isfinite(rate)
    increment[use] += (rate[use] - rates[use] @ theta)[:, None] * t
    strain0 = np.asarray(test['strain_0'])
    predicted = strain0[:, None] + increment
    return dict(estimated_taus=tau, predicted_strain=predicted,
                secant_compliance=(predicted[:, -1]-strain0)/np.maximum(np.abs(h[n:]), .5),
                hazard_crossing_times=_crossings(strain0, predicted, test['critical_strain'], t))
archive batch row counts [40, 40, 40, 40]
calibration 0 10 3.438 [ 0.80255 -0.13684] [-1.2e-04 -1.0e-04 -6.0e-05 -5.0e-05 -4.0e-05  4.0e-05  4.0e-05  5.0e-05
  6.0e-05  1.9e-04]
calibration 1 10 3.553 [1.20092 0.13169] [-1.2e-04 -7.0e-05 -6.0e-05 -3.0e-05  2.0e-05  3.0e-05  4.0e-05  4.0e-05
  4.0e-05  1.0e-04]
calibration 2 10 3.954 [ 0.77912 -0.13493] [-1.3e-04 -1.0e-04 -7.0e-05 -3.0e-05 -3.0e-05 -0.0e+00  3.0e-05  1.1e-04
  1.2e-04  1.2e-04]
calibration 3 10 3.971 [1.2392  0.10523] [-2.5e-04 -6.0e-05 -6.0e-05  0.0e+00  1.0e-05  4.0e-05  6.0e-05  7.0e-05
  7.0e-05  1.1e-04]
train feature NaNs [10  5  1  3] reference NaNs 250 plastic state nonzero 0
test feature NaNs [2 2 2 4] reference NaNs 105 plastic state nonzero 0
```

### Turn 5 — `execute_in_remote_docker` (`61310 ms` LLM)
- **Rationale**: Implement the exact inherited memory recurrence and both required threshold terms. Compare the unified and linear laws on held-out labs to detect missing constitutive terms, and test rate anchoring separately to see whether it helps curve prediction.
- **Alternatives Weighed**: `Use the measured test initial rate to correct every predicted curve` (rejected: Rate noise would accumulate over the hold; first measure the effect on held-out curves, especially when coupon features are missing.)

```bash
set -e
cd /holdout_task
solver=alpha_w03_batch_04_solver.py
{ printf '%s\n' '"""Calibrated two-branch Prony creep with thresholded viscoplasticity."""'; sed -n '2,124p' /tmp/hegel_continual_archive/alpha_w03_batch_03_solver.py; } > "$solver"
cat >> "$solver" <<'PY'
def _overstress(h, p, dt, yield_stress):
    hold = np.maximum(np.abs(h) - yield_stress, 0.)**2 * np.sign(h)
    work = np.maximum(np.abs(p) - yield_stress, 0.)**2 @ dt
    return hold, work * h


def _design(tau, x, h, p, dt, t, hold, work):
    tau = np.asarray(tau)
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, b = after[:, None] / tau, dt[:, None] / tau
    decay, jump = np.exp(-a), -np.expm1(-b)
    gap = h[:, None] - p @ (decay * jump)
    dgap = -(p @ (decay * (a * jump - b * np.exp(-b)) / tau))
    q = t[:, None] / tau
    shape = -np.expm1(-q)
    dshape = -np.exp(-q) * q / tau
    c = np.concatenate((gap[:, None, :] * shape,
                        x[:, None, :] * t[None, :, None],
                        (hold[:, None] * t)[..., None],
                        (work[:, None] * shape[:, 0])[..., None]), axis=2)
    r = np.column_stack((gap / tau, x, hold, work / tau[0]))
    dc = dgap[:, None, :] * shape + gap[:, None, :] * dshape
    dr = (dgap - gap / tau) / tau
    dw = work[:, None] * dshape[:, 0]
    return c, r, dc, dr, dw, -work / tau[0]**2


def _fit(x, h, p, dt, t, hold, work, y, rate):
    n, k = y.shape
    obs = np.r_[y.ravel(), rate]
    valid = np.isfinite(obs)
    obs = obs[valid]
    active = np.r_[np.ones(7, dtype=bool), np.any(hold), np.any(work)]

    def design(tau):
        c, r, dc, dr, dw, wr = _design(tau, x, h, p, dt, t, hold, work)
        a = np.vstack((c.reshape(-1, 9), r))[valid][:, active]
        d = np.vstack((dc.reshape(-1, 2), dr))[valid]
        return a, d, np.r_[dw.ravel(), wr][valid]

    def decode(v):
        return np.array([v[0], v[0] + (24.99 - v[0]) * v[1]])

    def optimize(v, scales, loss):
        cache = [None, None]

        def parts(u):
            if cache[0] is None or not np.array_equal(cache[0], u[:2]):
                cache[:] = [u[:2].copy(), design(decode(u))]
            return cache[1]

        def fun(u):
            a, _, _ = parts(u)
            return (a @ u[2:] - obs) / scales

        def jac(u):
            a, d, w = parts(u)
            j = d * u[2:4]
            if active[-1]:
                j[:, 0] += w * u[-1]
            return np.column_stack((j[:, 0] + (1. - u[1]) * j[:, 1],
                                    (24.99 - u[0]) * j[:, 1], a)) / scales[:, None]

        return least_squares(fun, v, jac=jac, loss=loss, x_scale='jac',
                             bounds=([.051, 1e-6] + [-np.inf] * active.sum(),
                                     [24.98, 1.] + [np.inf] * active.sum()),
                             max_nfev=150, ftol=1e-10)

    is_rate = np.r_[np.zeros(n*k, dtype=bool), np.ones(n, dtype=bool)][valid]

    def scales_for(res, factor):
        def mad(z):
            return max(1e-6, 1.4826 * np.median(np.abs(z - np.median(z)))) if len(z) else 1.
        return factor * np.where(is_rate, mad(res[is_rate]), mad(res[~is_rate]))

    starts = []
    for lo, hi in ((.24, 1.4), (.45, 2.7), (.9, 4.5), (1.6, 7.), (3., 13.)):
        a, _, _ = design([lo, hi])
        coef = np.linalg.lstsq(a, obs, rcond=None)[0]
        starts.append((np.r_[lo, (hi-lo)/(24.99-lo), coef], a @ coef - obs))
    pilot = min(starts, key=lambda s: np.median(np.abs(s[1][~is_rate])))
    scales = scales_for(pilot[1], 2.5)
    coarse = min((optimize(v, scales, 'soft_l1') for v, _ in starts), key=lambda r: r.cost)
    a, _, _ = design(decode(coarse.x))
    scales = scales_for(a @ coarse.x[2:] - obs, 2.8)
    result = optimize(coarse.x, scales, 'cauchy')
    weights = np.zeros(9)
    weights[active] = result.x[2:]
    return decode(result.x), weights


def solve_alpha_w03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, h, p = _prep(dataset)
    hold, work = _overstress(h, p, dt, dataset['yield_stress'])
    tau, weights = _fit(x[:n], h[:n], p[:n], dt, t, hold[:n], work[:n],
                        train['strain_curves_obs'] - train['strain_0'][:, None],
                        train['instant_rate_obs'])
    curves, rates, *_ = _design(tau, x[n:], h[n:], p[n:], dt, t, hold[n:], work[n:])
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=test['strain_0'][:, None] + curves @ weights,
                initial_rate_pred=rates @ weights)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/creep04_oof.py <<'PY'
import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_batch_04_solver.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_alpha_w03_batch_04.pkl','rb'))[0]['dataset'];tr=d['train'];n=len(tr['lab_id']);t=d['eval_times'];dt=d['past_dt'];lab=tr['lab_id'];x,h,p=[a[:n] for a in s._prep(d)];hold,work=s._overstress(h,p,dt,d['yield_stress']);y=tr['strain_curves_obs']-tr['strain_0'][:,None];r=tr['instant_rate_obs'];miss=np.isnan(tr['coupon_features_raw']).any(axis=1)
for linear in (False, True):
 cy=np.zeros_like(y);rr=np.zeros_like(r)
 ho,wo=(np.zeros_like(hold),np.zeros_like(work)) if linear else (hold,work)
 for ell in sorted(set(lab)):
  a=lab!=ell;b=~a
  tau,w=s._fit(x[a],h[a],p[a],dt,t,ho[a],wo[a],y[a],r[a]);c,v,*_=s._design(tau,x[b],h[b],p[b],dt,t,ho[b],wo[b]);cy[b]=c@w;rr[b]=v@w
  print('linear' if linear else 'unified','lab',ell,'tau',tau.round(5),'weights',w.round(5),'RMSE',np.sqrt(np.mean((cy[b]-y[b])**2)),np.sqrt(np.mean((rr[b]-r[b])**2)))
 for title,m in [('complete',~miss),('missing',miss),('sub-yield',(hold==0)&(work==0)),('hold over yield',hold!=0),('history-only over yield',(hold==0)&(work!=0))]:
  if m.any():
   e=cy[m]-y[m];dr=r[m]-rr[m];print(title,m.sum(),'strain by horizon',np.sqrt(np.mean(e**2,axis=0)).round(5),'rate',np.sqrt(np.mean(dr**2)).round(5),'strain correction 1',np.sqrt(np.mean((e+dr[:,None]*t)**2,axis=0)).round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/creep04_oof.py
```
- **Container Exit Code**: `0` (`1068 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
43018 0.33592 0.55555 0.4451  0.42696 0.71445 0.52342
 0.69702] rate 0.11388 strain correction 1 [0.00698 0.54041 0.37909 0.22353 0.43018 0.30845 0.2464  0.3824  0.03857
 0.35307]
sub-yield 100 strain by horizon [0.22044 0.35128 0.26236 0.30267 0.2438  0.22102 0.30486 0.34134 0.36772
 0.31205] rate 0.03214 strain correction 1 [0.22024 0.3539  0.25962 0.29626 0.23089 0.21092 0.28682 0.27708 0.33712
 0.26682]
hold over yield 100 strain by horizon [0.26312 0.35204 0.38445 0.23828 0.31834 0.29202 0.29501 0.30931 0.2956
 0.25059] rate 0.03938 strain correction 1 [0.2622  0.3519  0.37737 0.22249 0.28862 0.27509 0.28207 0.27527 0.24021
 0.16526]
linear lab 0 tau [ 0.24713 11.23554] weights [ 0.27519  4.50671  0.10893  0.17714 -0.13823  0.10103 -0.06613  0.
  0.     ] RMSE 0.754064948390197 1.3787793991570918
linear lab 1 tau [0.26403 8.39388] weights [ 0.25026  3.69011  0.09073  0.17715 -0.13777  0.10346 -0.06856  0.
  0.     ] RMSE 0.8535981748594943 1.7259692024594977
linear lab 2 tau [ 0.26578 10.29514] weights [ 0.23565  4.12788  0.08284  0.17721 -0.14427  0.12561 -0.06218  0.
  0.     ] RMSE 0.8004185585768783 1.58251609155039
linear lab 3 tau [ 0.28784 12.88929] weights [ 0.2929   4.76799  0.07483  0.18751 -0.13486  0.10527 -0.07763  0.
  0.     ] RMSE 0.6446883104778172 1.3340018812912533
complete 182 strain by horizon [0.40502 0.70496 0.78823 0.75756 0.72686 0.73449 0.73324 0.76477 0.83869
 0.8957 ] rate 1.44983 strain correction 1 [0.25615 0.65599 1.30669 2.14009 2.98536 3.86209 4.7399  5.60094 6.48282
 7.31961]
missing 18 strain by horizon [0.39864 0.91889 0.88707 0.9771  0.87815 0.89442 1.001   1.0638  0.94259
 1.05336] rate 1.9527 strain correction 1 [0.09527 0.99989 1.88699 2.79761 4.15617 5.15151 6.22066 7.48926 8.65706
 9.81396]
sub-yield 100 strain by horizon [0.22393 0.3718  0.29407 0.38597 0.36079 0.39861 0.50895 0.60748 0.73625
 0.80261] rate 0.2434 strain correction 1 [0.21913 0.41973 0.40624 0.51957 0.56008 0.62865 0.79415 0.81214 0.97797
 1.06057]
hold over yield 100 strain by horizon [0.52633 0.95825 1.08901 1.03313 0.98498 0.98333 0.9486  0.94823 0.94765
 1.00782] rate 2.11016 strain correction 1 [ 0.27024  0.88711  1.893    3.07806  4.36075  5.615    6.87197  8.15664
  9.43518 10.66402]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ad9eba9c23f5fe3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ad9eba9c (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27410 ms` LLM)
- **Rationale**: The unified law passes the schema and substantially improves held-out labs. Compare median and tail strain errors before deciding whether to use the independent test-rate measurement on missing-feature rows; inspect historical calibration changes before deciding how much to trust sparse-check fallback.
- **ATMS NOGOOD DECLARED**: Refuted `The inherited two-branch linear basis suffices in the mixed-stress campaign.` | Witness: `With the same calibrated inputs, the linear basis gave 2.110 initial-rate RMSE above hold yield, versus 0.0394 after adding the two threshold terms.` | Blanket Diagnosis: `Without signed hold flow and accumulated plastic work, the linear fit distorts relaxation times and amplitudes to absorb above-yield deformation.` | Sublated Pivot: `Keep the inherited Prony recurrence and add the two thresholded terms, including their derivatives at zero.`

```bash
set -e
sed -n '7,40p' /holdout_task/validate_public_schema.py
cat > /tmp/creep04_diag.py <<'PY'
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
import json
history=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for tab in history+[d['calibration_load_cells']]:
 print('sweep', [None if (b:=s._supported_line(*s._checks(tab,l))) is None else b.round(5).tolist() for l in range(4)])
curve=np.empty_like(y);rate=np.empty_like(r);coef=np.empty((n,9)); imputed=np.isnan(tr['coupon_features_raw'])
for ell in sorted(set(lab)):
 a=lab!=ell;b=~a
 tau,w=s._fit(x[a],h[a],p[a],dt,t,hold[a],work[a],y[a],r[a]);c,v,*_=s._design(tau,x[b],h[b],p[b],dt,t,hold[b],work[b]);curve[b]=c@w;rate[b]=v@w;coef[b]=w
for label,m in [('complete',~miss),('missing',miss),('missing wing A',miss&(lab<2)),('missing wing B',miss&(lab>=2))]:
 print(label,'n',m.sum(),'rate abs quantiles',np.quantile(abs(rate[m]-r[m]),[.5,.8,.95,1]).round(5))
 for alpha in (0,.8,.95,1):
  e=curve[m]-y[m]+alpha*(r[m]-rate[m])[:,None]*t
  print(alpha,'curve abs p50,p80,p90,p95,max',np.quantile(abs(e),[.5,.8,.9,.95,1]).round(5),'capped RMSE',np.sqrt(np.mean(np.minimum(e**2,.05**2),axis=0)).round(5))
print('complete max per lab',[(int(i),round(np.max(abs(rate[(lab==i)&~miss]-r[(lab==i)&~miss])),5)) for i in np.unique(lab)])
print('missing rate correction candidates',np.c_[lab[miss],imputed[miss].astype(int),np.round((r-rate)[miss],4)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/creep04_diag.py
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.16981 0.89934 1.20443] capped RMSE [0.0086  0.03384 0.03816 0.04051 0.0457  0.04842 0.04834 0.05    0.04974
 0.05   ]
0.95 curve abs p50,p80,p90,p95,max [0.01909 0.04208 0.05989 0.86193 1.14481] capped RMSE [0.00652 0.0287  0.02394 0.01613 0.02597 0.02558 0.0351  0.03252 0.03146
 0.0384 ]
1 curve abs p50,p80,p90,p95,max [0.01109 0.02981 0.0493  0.85186 1.16871] capped RMSE [0.00641 0.02808 0.02205 0.01306 0.01929 0.02141 0.02842 0.02291 0.02786
 0.03209]
missing wing B n 5 rate abs quantiles [0.07843 0.09981 0.15295 0.17066]
0 curve abs p50,p80,p90,p95,max [0.15662 0.50036 1.10638 1.28902 1.80558] capped RMSE [0.02598 0.04475 0.04162 0.0409  0.04814 0.04456 0.04563 0.04537 0.04496
 0.04582]
0.8 curve abs p50,p80,p90,p95,max [0.05296 0.13146 0.98839 1.11134 1.19763] capped RMSE [0.01102 0.03081 0.0386  0.03917 0.04108 0.04382 0.04338 0.04672 0.04623
 0.04963]
0.95 curve abs p50,p80,p90,p95,max [0.03147 0.10809 0.96023 1.05119 1.19311] capped RMSE [0.00881 0.02683 0.0332  0.03155 0.04017 0.03769 0.03534 0.04163 0.04202
 0.04376]
1 curve abs p50,p80,p90,p95,max [0.03145 0.11311 0.94632 1.04961 1.19569] capped RMSE [0.00826 0.02641 0.03176 0.02968 0.03964 0.03735 0.03382 0.04245 0.0385
 0.04333]
complete max per lab [(0, 0.0209), (1, 0.03453), (2, 0.04226), (3, 0.02345)]
missing rate correction candidates [[ 0.      1.      0.      1.      0.     -0.1787]
 [ 1.      1.      0.      0.      0.      0.0905]
 [ 3.      0.      1.      0.      0.     -0.0784]
 [ 3.      0.      1.      0.      0.      0.0211]
 [ 2.      0.      0.      0.      1.     -0.0821]
 [ 0.      0.      0.      0.      1.      0.1233]
 [ 0.      1.      0.      0.      0.      0.0997]
 [ 0.      0.      1.      0.      0.      0.0807]
 [ 1.      1.      0.      0.      0.     -0.1322]
 [ 1.      1.      0.      0.      0.     -0.047 ]
 [ 0.      1.      0.      0.      0.     -0.1934]
 [ 2.      1.      0.      0.      0.      0.1707]
 [ 1.      0.      0.      0.      1.     -0.0611]
 [ 0.      1.      0.      0.      0.     -0.0445]
 [ 0.      1.      0.      0.      0.     -0.1554]
 [ 0.      0.      1.      0.      0.     -0.1403]
 [ 1.      1.      0.      0.      0.      0.112 ]
 [ 3.      0.      1.      0.      0.     -0.0048]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ad9eba9c23f5fe3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ad9eba9c (directory).
```

### Turn 7 — `execute_in_remote_docker` (`24778 ms` LLM)
- **Rationale**: Check whether test rates are actually supplied before changing prediction, and thin or contaminate real calibration sweeps to see whether the latest-only archive fallback misses the earlier nonnominal calibration that matches this batch.
- **Alternatives Weighed**: `Apply rate anchoring to all coupons` (rejected: Held-out median strain error increased from 0.0099 to 0.0200 on complete-feature rows, so any anchoring should be limited to missing-feature rows with an available test rate.)

```bash
set -e
sed -n '1,94p' /holdout_task/alpha_w03_batch_04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
base='/holdout_task/'
sp=importlib.util.spec_from_file_location('s',base+'alpha_w03_batch_04_solver.py'); s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
d=pickle.load(open(base+'public_alpha_w03_batch_04.pkl','rb'))[0]['dataset']
for split in ('train','test'):
 q=d[split];print(split,'keys',sorted(q));print('rates available',np.isfinite(q.get('instant_rate_obs',[])).sum(),'missing feature rows by lab',np.bincount(q['lab_id'][np.isnan(q['coupon_features_raw']).any(axis=1)],minlength=4))
c=d['calibration_load_cells']; rng=np.random.default_rng(404)
for lab in range(4):
 x,y=s._checks(c,lab);ref=s._supported_line(x,y);prior=s._archive_calibration([lab]).get(lab);print('lab',lab,'latest archival prior',prior)
 for label in ('one spot','two random spots','two wide spots','one EMI'):
  errors=[]
  for i in range(40):
   ids=rng.choice(len(x),2,replace=False) if label=='two random spots' else ([np.argmin(x),np.argmax(x)] if label=='two wide spots' else ([i%len(x)] if label=='one spot' else np.arange(len(x))))
   xx=x[ids];yy=y[ids].copy();
   if label=='one EMI': yy[i%len(x)]+=(-1)**i*.35
   fit=s._supported_line(xx,yy)
   if fit is None: fit=s._sparse_line(xx,yy,prior)
   errors.append(np.max(np.abs((fit[0]-ref[0])*np.array([-3.,3.])+fit[1]-ref[1])))
  print(label, 'max physical stress deviation',round(max(errors),6))
PY
```
- **Container Exit Code**: `0` (`770 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s[k]
    delta = residuals[k]
    close = np.abs(delta) < tol
    if close.sum() >= 2 and np.ptp(x[close]) >= 0.5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if len(x) >= 2 and np.ptp(x) >= 0.5:
        # Permit a new calibration when the current checks form a coherent
        # line, even if neither old nor nominal parameters explain it.
        fit = _line(x, y)
        if np.max(np.abs(y - (fit[0]*x + fit[1]))) <= 0.002:
            return fit
    return gain, bias


def _prep(d):
    c = d['calibration_load_cells']
    lab = np.concatenate([d[k]['lab_id'] for k in ('train', 'test')])
    strata = np.concatenate([d[k]['stratum'] for k in ('train', 'test')])
train keys ['coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd']
rates available 200 missing feature rows by lab [8 5 2 3]
test keys ['coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'residual_plastic_state', 'strain_0', 'stratum', 'stress_hold_cmd']
rates available 80 missing feature rows by lab [2 4 0 4]
lab 0 latest archival prior [1.00003921e+00 1.74615955e-05]
one spot max physical stress deviation 0.729202
two random spots max physical stress deviation 0.729202
two wide spots max physical stress deviation 9.3e-05
one EMI max physical stress deviation 0.000151
lab 1 latest archival prior [ 1.00003065e+00 -1.76921923e-05]
one spot max physical stress deviation 0.734447
two random spots max physical stress deviation 0.734447
two wide spots max physical stress deviation 4.1e-05
one EMI max physical stress deviation 5.4e-05
lab 2 latest archival prior [ 9.99982073e-01 -7.35421522e-06]
one spot max physical stress deviation 0.797576
two random spots max physical stress deviation 0.797576
two wide spots max physical stress deviation 0.000122
one EMI max physical stress deviation 5.4e-05
lab 3 latest archival prior [1.00004087e+00 3.06525896e-05]
one spot max physical stress deviation 0.822849
two random spots max physical stress deviation 0.822849
two wide spots max physical stress deviation 5.8e-05
one EMI max physical stress deviation 5.1e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ad9eba9c23f5fe3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ad9eba9c (directory).
```

### Turn 8 — `execute_in_remote_docker` (`51661 ms` LLM)
- **Rationale**: Promote the sparse-calibration fix and the missing-only drift correction. Replay sparse and EMI-contaminated versions of the actual sweeps; if stress remains biased, revise archive selection before accepting the candidate.
- **Alternatives Weighed**: `Pool all historical rows into one calibration line` (rejected: Pooling would average distinct instrument regimes and bias physical stress near yield.)
- **ATMS NOGOOD DECLARED**: Refuted `The most recent historical calibration is the relevant prior when current load-cell checks are sparse.` | Witness: `Three historical sweeps are nominal, but the earlier offset sweep matches all four in-batch labs. With one actual check per lab, the latest-only fallback produced physical stress errors up to 0.823 at commands ±3.` | Blanket Diagnosis: `Calibration regimes recur rather than persist monotonically, so recency can substitute the wrong instrument response for the current one.` | Sublated Pivot: `Keep supported current fits; use current spots to select among historical sweeps and defer ambiguous cases to the companion lab.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w03_batch_04_solver.py')
s = p.read_text()
a = s.index('def _archive_calibration(')
b = s.index('def _prep(', a)
s = s[:a] + '''def _archive_calibration(checks, fits):
    """Select archived lines by current agreement, borrowing evidence within a wing."""
    path = _ARCHIVE / 'archive_calibration_history.json'
    nominal = np.array([1., 0.])
    sweeps = [{lab: nominal for lab in checks}]
    if path.exists():
        with path.open() as stream:
            for table in json.load(stream):
                sweep = {lab: _supported_line(*_checks(table, lab)) for lab in checks}
                sweeps.append({lab: fit for lab, fit in sweep.items() if fit is not None})

    def disagreement(lab, line):
        if fits[lab] is not None:
            # A supported current slope is more informative than a narrow check.
            dg, db = fits[lab] - line
            residual = np.abs(np.array([db - dg, db + dg]))
        else:
            x, y = checks[lab]
            residual = np.abs(y - (line[0]*x + line[1]))
        return np.minimum(residual, .02).mean() if len(residual) else 0.

    priors = {}
    for lab in checks:
        if fits[lab] is not None:
            continue
        peer = lab ^ 1
        scores = [(disagreement(lab, sweep[lab]),
                   disagreement(peer, sweep[peer]) if peer in sweep else .02,
                   sweep[lab]) for sweep in sweeps if lab in sweep]
        local_best = min(a for a, _, _ in scores)
        # Treat sub-millistress differences as ties when a spot does not
        # distinguish two lines, then use the companion lab's checks.
        priors[lab] = min((v for v in scores if v[0] <= local_best + .001),
                          key=lambda v: (v[1], v[0]))[2]
    return priors


def _sparse_line(x, y, prior):
    gain, bias = prior
    delta = y - (gain*x + bias)
    close = np.abs(delta) < .02
    if close.sum() >= 2 and np.ptp(x[close]) >= .5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if len(x) >= 2 and np.ptp(x) >= .5:
        fit = _line(x, y)
        if np.max(np.abs(y - (fit[0]*x + fit[1]))) <= .002:
            return fit
    return prior


''' + s[b:]
s = s.replace("checks = {i: _checks(c, i) for i in np.unique(lab)}", "checks = {i: _checks(c, i) for i in range(4)}")
s = s.replace("    sparse = [i for i in fits if fits[i] is None]\n    priors = _archive_calibration(sparse) if sparse else {}", "    priors = _archive_calibration(checks, fits) if any(v is None for v in fits.values()) else {}")
s = s.replace('''    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=test['strain_0'][:, None] + curves @ weights,
                initial_rate_pred=rates @ weights)''', '''    increment = curves @ weights
    initial_rate = rates @ weights
    # Missing covariates affect only the constant creep drift. Its observed
    # initial-rate innovation can recover their combined contribution without
    # changing either hereditary memory or the fitted material coefficients.
    measured = test.get('instant_rate_obs')
    if measured is not None:
        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(measured)
        increment[use] += (measured[use] - initial_rate[use])[:, None] * t
        initial_rate[use] = measured[use]
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=test['strain_0'][:, None] + increment,
                initial_rate_pred=initial_rate)''')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/creep04_stress_cal.py <<'PY'
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
c=d['calibration_load_cells']; full={l:s._checks(c,l) for l in range(4)}; ref={l:s._supported_line(*full[l]) for l in full}
rng=np.random.default_rng(404)
for label in ('one spot','two random spots','two wide spots','one EMI','no spots on one lab','one spot across wing'):
 errors=[]
 for ell in full:
  x,y=full[ell]
  for i in range(40):
   checks=full.copy()
   if label=='one spot': ids=[i%len(x)]
   elif label=='two random spots': ids=rng.choice(len(x),2,replace=False)
   elif label=='two wide spots': ids=[np.argmin(x),np.argmax(x)]
   elif label=='one EMI': ids=np.arange(len(x))
   elif label=='no spots on one lab': ids=[]
   else:
    ids=[i%len(x)]
    peer=ell ^ 1
    checks[peer]=tuple(v[:0] for v in full[peer])
   xx, yy=x[ids],y[ids].copy()
   if label=='one EMI': yy[i%len(x)]+=(-1)**i*.35
   checks[ell]=xx,yy
   fits={l:s._supported_line(*xy) for l,xy in checks.items()}
   prior=s._archive_calibration(checks,fits) if any(v is None for v in fits.values()) else {}
   for l in (ell,ell^1):
    fit=fits[l] if fits[l] is not None else s._sparse_line(*checks[l],prior[l]); delta=fit-ref[l]
    errors.append(np.max(np.abs(delta[0]*np.array([-3.,3.])+delta[1])))
 print(label,'worst physical stress deviation',round(max(errors),6))
 assert max(errors)<.005, (label,max(errors))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/creep04_stress_cal.py
```
- **Container Exit Code**: `0` (`8431 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
one spot worst physical stress deviation 0.000375
two random spots worst physical stress deviation 0.001384
two wide spots worst physical stress deviation 0.000122
one EMI worst physical stress deviation 0.000151
no spots on one lab worst physical stress deviation 0.00017
one spot across wing worst physical stress deviation 0.000375

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f214ac36692c1357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f214ac36 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`30462 ms` LLM)
- **Rationale**: Sparse and EMI replay supports resolving the calibration failure. Check the unobserved history-only branch and the exact sub-yield limit separately: a sign or derivative failure would require a basis correction. Measure the test-rate correction and confirm it does not alter complete-feature outputs.

```bash
set -e
cd /holdout_task
python3 - <<'PY'
import json
x=json.load(open('public_asset_index.json'))
print('Asset index keys:', list(x) if isinstance(x,dict) else type(x).__name__)
for line in json.dumps(x,indent=2).splitlines():
 if any(word in line.lower() for word in ('innovation','rate','yield','tolerance','calibration','imput')): print(line[:240])
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
sy=d['yield_stress']; hc=np.array([sy,-sy,0.,sy+.4,-sy-.4,.6,-.6]); pc=np.zeros((len(hc), len(dt))); pc[5,2]=sy+.5; pc[6,2]=-sy-.5
ho,wo=s._overstress(hc,pc,dt,sy)
np.testing.assert_allclose(ho,[0,0,0,.16,-.16,0,0],atol=1e-14)
np.testing.assert_allclose(wo,[0,0,0,0,0,.25*dt[2]*.6,-.25*dt[2]*.6],atol=1e-14)
tau=np.array([.62,4.1]); weights=np.array([.7,.57,.08,.15,-.13,.11,-.09,.25,.17]); xc=x[:len(hc)]
c,v,*_=s._design(tau,xc,hc,pc,dt,np.array([0.,1e-6]),ho,wo)
np.testing.assert_allclose(c[:,0,:],0,atol=1e-14)
np.testing.assert_allclose((c[:,1,:]@weights)/1e-6,v@weights,atol=4e-6)
h0=np.clip(h,-sy,sy); p0=np.clip(p,-sy,sy); a,b=s._overstress(h0,p0,dt,sy)
c0,v0,*_=s._design(tau,x,h0,p0,dt,t,a,b)
t0,w0=s._fit(x,h0,p0,dt,t,a,b,c0@weights,v0@weights)
np.testing.assert_array_equal(w0[7:],0); np.testing.assert_allclose(t0,tau,atol=1e-5)
print('Signed hold, history-only work, zero-time derivative, and sub-yield fit passed')
base=s.solve_alpha_w03_batch_04(d); te=d['test']; mask=np.isnan(te['coupon_features_raw']).any(axis=1)
xb,hb,pb=s._prep(d); hh,ww=s._overstress(hb,pb,dt,sy)
c1,v1,*_=s._design(base['estimated_taus'],xb[n:],hb[n:],pb[n:],dt,t,hh[n:],ww[n:]); w=np.r_[base['hereditary_weights'],base['viscoplastic_weights']]
innovation=te['instant_rate_obs']-v1@w
print('Missing-feature absolute rate innovation by wing:', [round(float(np.max(abs(innovation[mask&(te['lab_id']//2==j)]))),6) for j in (0,1)])
np.testing.assert_allclose(base['initial_rate_pred'][~mask],(v1@w)[~mask]); np.testing.assert_allclose(base['predicted_strain'][~mask],(te['strain_0'][:,None]+c1@w)[~mask]); print('Complete-feature outputs retain constitutive predictions')
PY
```
- **Container Exit Code**: `0` (`561 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Asset index keys: ['files', 'schema']
            "calibration_load_cells": {
              "instant_rate_obs": {
              "instant_rate_obs": {
            "yield_stress": 1.52
            "te_true_rate0": {
Signed hold, history-only work, zero-time derivative, and sub-yield fit passed
Missing-feature absolute rate innovation by wing: [0.275889, 0.227652]
Complete-feature outputs retain constitutive predictions

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f214ac36692c1357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f214ac36 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`21334 ms` LLM)
- **Rationale**: Check whether the pooled imputation gain hides a lab-level regression or whether complete test coupons expose a rate offset that would invalidate anchoring. Either failure would warrant changing the correction; otherwise retain the verified calibration and constitutive basis.

```bash
set -e
sed -n '98,140p' /holdout_task/alpha_w03_batch_04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
curve=np.empty_like(y); rate=np.empty_like(r)
for ell in np.unique(lab):
 a=lab!=ell; b=~a
 tau,w=s._fit(x[a],h[a],p[a],dt,t,hold[a],work[a],y[a],r[a]); c,v,*_=s._design(tau,x[b],h[b],p[b],dt,t,hold[b],work[b]); curve[b]=c@w; rate[b]=v@w
for ell in np.unique(lab):
 m=(lab==ell)&miss
 if not m.any(): continue
 e=curve[m]-y[m]; innovation=(r-rate)[m]
 print('missing lab',ell,'n',m.sum(),'median absolute strain error before/after',round(float(np.median(abs(e))),5),round(float(np.median(abs(e+innovation[:,None]*t))),5))
# The held-out lab supplies no outcomes for estimating the missing-feature correction.
# Check that the test rate exposes no evidence of a lab-wide measurement offset.
base=s.solve_alpha_w03_batch_04(d); te=d['test']; complete=~np.isnan(te['coupon_features_raw']).any(axis=1)
for ell in np.unique(te['lab_id']):
 m=(te['lab_id']==ell)&complete
 e=te['instant_rate_obs'][m]-base['initial_rate_pred'][m]
 print('test complete lab',ell,'rate innovation median, max abs',round(float(np.median(e)),6),round(float(np.max(abs(e))),6))
PY
```
- **Container Exit Code**: `0` (`625 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
st')]).astype(float)
    p = np.concatenate([d[k]['past_stress_cmd'] for k in ('train', 'test')]).astype(float)
    checks = {i: _checks(c, i) for i in range(4)}
    fits = {i: _supported_line(*xy) for i, xy in checks.items()}
    priors = _archive_calibration(checks, fits) if any(v is None for v in fits.values()) else {}
    for i, fit in fits.items():
        m = lab == i
        g, b = fit if fit is not None else _sparse_line(*checks[i], priors.get(i))
        h[m] = g*h[m] + b
        p[m] = g*p[m] + b
        rc = ref[m]
        usable = np.isfinite(rc).sum(axis=0) >= 2
        rc = rc[:, usable]
        if rc.size:
            center = np.median(np.nanmean(rc, axis=0))
            scale = np.median(np.nanstd(rc, axis=0, ddof=0))
            x[m] = (x[m] - center) / (scale if scale > 0 else 1.)
    # Give each observed lab/stratum cell equal influence, separately per column.
    cells = [x[(lab == i) & (strata == j)] for i, j in np.unique(np.column_stack((lab, strata)), axis=0)]
    sums = np.array([np.nansum(a, axis=0) for a in cells])
    counts = np.array([np.isfinite(a).sum(axis=0) for a in cells])
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    good = (counts > 0).sum(axis=0)
    fill = np.divide(means.sum(axis=0), good, out=np.zeros(x.shape[1]), where=good > 0)
    return np.column_stack((np.ones(len(h)), np.where(np.isfinite(x), x, fill))), h, p


def _overstress(h, p, dt, yield_stress):
    hold = np.maximum(np.abs(h) - yield_stress, 0.)**2 * np.sign(h)
    work = np.maximum(np.abs(p) - yield_stress, 0.)**2 @ dt
    return hold, work * h


def _design(tau, x, h, p, dt, t, hold, work):
missing lab 0 n 8 median absolute strain error before/after 0.35334 0.01357
missing lab 1 n 5 median absolute strain error before/after 0.24571 0.00932
missing lab 2 n 2 median absolute strain error before/after 0.33222 0.03013
missing lab 3 n 3 median absolute strain error before/after 0.05162 0.03145
test complete lab 0 rate innovation median, max abs 0.003892 0.018065
test complete lab 1 rate innovation median, max abs 0.005585 0.025803
test complete lab 2 rate innovation median, max abs 0.003446 0.018012
test complete lab 3 rate innovation median, max abs 0.000211 0.020103

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f214ac36692c1357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f214ac36 (directory).
```

### Turn 11 — `submit_completed_task` (`12144 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema verification, signed-yield and zero-time derivative checks, and sparse and EMI calibration replays. The missing-feature correction improved every held-out lab without altering complete-feature predictions.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_w03_batch_04_solver.py': 'FILE:10506'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w03_batch_04_solver.py`. The solver passed the public schema and constitutive checks. Worst stress deviation in sparse and EMI calibration replays was 0.001384, and missing-feature median strain error improved in all four held-out labs.`

### Turn 12 — `execute_in_remote_docker` (`33355 ms` LLM)
- **Rationale**: Test whether interactions or rational stress terms reduce held-out error on complete coupons, where missing features cannot confound the result. Replay sparse checks and EMI together: if two checks let one spike override a corroborated archive line, tighten that fallback before submission.
- **Alternatives Weighed**: `Test relay modes, HJB drift, or classification balance` (rejected: The benchmark has continuous strain and rate outputs, with no relay, stochastic control state, or class labels. Lab- and stratum-specific signed residuals are the relevant confounding check.)

```bash
set -e
cd /holdout_task
cat > /tmp/creep04_margin.py <<'PY'
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
from itertools import combinations
# Use complete-feature coupons to separate constitutive residuals from missing-feature drift.
# Probe fixed alternatives, estimating all additional amplitudes off the held-out lab.
scale=d['yield_stress']; stress=h/scale
pairs=np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(1,5),2)])
rat=np.column_stack((stress/(1+stress**2), hold/(1+stress**2),work/(1+stress**2)))
extras={'interactions':pairs,'rational stress':rat}
for ell in np.unique(lab):
 a=(lab!=ell)&~miss; b=(lab==ell)&~miss
 tau,w=s._fit(x[a],h[a],p[a],dt,t,hold[a],work[a],y[a],r[a]); c,v,*_=s._design(tau,x,h,p,dt,t,hold,work)
 ey=y-c@w; er=r-v@w
 good=np.abs(ey[a])<.04; rr=np.abs(er[a])<.04
 print('lab',ell,'baseline median abs curve/rate',np.round([np.median(abs(ey[b])),np.median(abs(er[b]))],6))
 for name,g in extras.items():
  gcurve=g[:,None,:]*t[None,:,None]
  coef=np.linalg.lstsq(np.vstack((gcurve[a][good],g[a][rr])),np.r_[ey[a][good],er[a][rr]],rcond=None)[0]
  print(' ',name,'median abs curve/rate',np.round([np.median(abs(ey[b]-gcurve[b]@coef)),np.median(abs(er[b]-g[b]@coef))],6))
 print('  signed median rate residual by stratum',[(int(j),round(float(np.median(er[b & (tr['stratum']==j)])),6)) for j in np.unique(tr['stratum'][b])])
# Replay occasional interference on the same real spots used in a sparse check.
cal=d['calibration_load_cells']; full={i:s._checks(cal,i) for i in range(4)}
ref={i:s._supported_line(*xy) for i,xy in full.items()}
for m in (1,2):
 errors=[]
 for ell,(xx,yy) in full.items():
  for ids in combinations(range(len(xx)),m):
   if m==2 and np.ptp(xx[list(ids)])<.5: continue
   for idx in range(m):
    for spike in (-.35,.35):
     checks=full.copy(); cx,cy=xx[list(ids)],yy[list(ids)].copy(); cy[idx]+=spike; checks[ell]=(cx,cy)
     fits=ref.copy(); fits[ell]=None
     prior=s._archive_calibration(checks,fits)[ell]
     delta=s._sparse_line(cx,cy,prior)-ref[ell]
     errors.append((float(np.max(abs(delta[0]*np.array([-3.,3.])+delta[1]))),ell,ids,idx,spike))
 print('sparse EMI spots',m,'worst (stress error, lab, row ids, contaminated index, spike)',max(errors),'above 0.005',sum(z[0]>.005 for z in errors),'/',len(errors))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/creep04_margin.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`6176 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 baseline median abs curve/rate [0.007708 0.006651]
  interactions median abs curve/rate [0.007466 0.006962]
  rational stress median abs curve/rate [0.007738 0.006803]
  signed median rate residual by stratum [(0, 0.003385), (1, -0.000746), (2, 0.001861)]
lab 1 baseline median abs curve/rate [0.011565 0.00862 ]
  interactions median abs curve/rate [0.011811 0.008572]
  rational stress median abs curve/rate [0.011839 0.008636]
  signed median rate residual by stratum [(0, 0.001729), (1, 0.001293), (2, 0.005843)]
lab 2 baseline median abs curve/rate [0.013469 0.008895]
  interactions median abs curve/rate [0.01413  0.008484]
  rational stress median abs curve/rate [0.013546 0.009007]
  signed median rate residual by stratum [(0, 0.002525), (1, -0.00326), (2, 0.002869)]
lab 3 baseline median abs curve/rate [0.008155 0.007962]
  interactions median abs curve/rate [0.00878  0.007634]
  rational stress median abs curve/rate [0.008033 0.007802]
  signed median rate residual by stratum [(0, -0.003922), (1, 0.00488), (2, -0.002137)]
sparse EMI spots 1 worst (stress error, lab, row ids, contaminated index, spike) (0.8071753051089254, 3, (5,), 0, 0.35) above 0.005 2 / 80
sparse EMI spots 2 worst (stress error, lab, row ids, contaminated index, spike) (0.8599758695903948, 3, (5, 6), 0, 0.35) above 0.005 1 / 604
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f214ac36692c1357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f214ac36 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`34628 ms` LLM)
- **Rationale**: Re-run the combined sparse-plus-EMI replay to see whether a contaminated spot can still override independent calibration evidence. Compare against the clean sparse replays so the fix does not trade EMI resistance for a regression on valid checks.
- **Alternatives Weighed**: `Extend the constitutive basis with interaction or rational stress terms` (rejected: Neither extension improved the worst held-out complete-feature strain median; lab-by-stratum signed residual medians also showed no consistent offset.)
- **ATMS NOGOOD DECLARED**: Refuted `Local residual ranking reliably identifies the archived calibration regime when only one or two load-cell checks are available.` | Witness: `On replayed public load-cell checks, one EMI spike caused stress deviations above 0.005 in 2 of 80 one-spot cases and 1 of 604 two-spot cases, with a maximum of 0.860 at commands ±3.` | Blanket Diagnosis: `An interference spike can make the wrong archived line look locally better, even when the companion lab independently identifies the correct regime.` | Sublated Pivot: `Keep supported current fits and multi-spot local agreement; when local evidence reduces to one spot, prioritize a corroborating companion-lab fit and only refit using checks consistent with the selected line.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w03_batch_04_solver.py')
s = p.read_text()
a = s.index('    priors = {}\n', s.index('def _archive_calibration'))
b = s.index('\n\ndef _prep(', a)
s = s[:a] + '''    priors = {}
    for lab in checks:
        if fits[lab] is not None:
            continue
        peer = lab ^ 1
        x, y = checks[lab]
        candidates = []
        for sweep in sweeps:
            if lab not in sweep:
                continue
            line = sweep[lab]
            residual = np.abs(y - (line[0]*x + line[1]))
            support = np.count_nonzero(residual <= .002)
            peer_error = disagreement(peer, sweep[peer]) if peer in sweep else .02
            candidates.append((line, support, peer_error, disagreement(lab, line)))
        # Two matching checks can identify a local regime. A lone matching
        # check cannot overrule a clean, independently fitted companion lab.
        local_support = max(v[1] for v in candidates)
        if local_support >= 2:
            pool = [v for v in candidates if v[1] == local_support]
            chosen = min(pool, key=lambda v: (v[2], v[3]))
        else:
            peer_best = min(v[2] for v in candidates)
            if fits[peer] is not None and peer_best <= .002:
                pool = [v for v in candidates if v[2] <= peer_best + .001]
                chosen = min(pool, key=lambda v: (-v[1], v[3], v[2]))
            else:
                local_best = min(v[3] for v in candidates)
                pool = [v for v in candidates if v[3] <= local_best + .001]
                chosen = min(pool, key=lambda v: (-v[1], v[2], v[3]))
        priors[lab] = chosen[0], fits[peer] is not None and chosen[2] <= .002
    return priors


def _sparse_line(x, y, prior):
    (gain, bias), corroborated = prior
    delta = y - (gain*x + bias)
    close = np.abs(delta) <= .002
    if close.sum() >= 2 and np.ptp(x[close]) >= .5:
        return _line(x[close], y[close])
    if close.any():
        return gain, bias + np.median(delta[close])
    if not corroborated and len(x) >= 2 and np.ptp(x) >= .5:
        fit = _line(x, y)
        if np.max(np.abs(y - (fit[0]*x + fit[1]))) <= .002:
            return fit
    return gain, bias
''' + s[b:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/creep04_stress_cal.py
python3 -I /tmp/creep04_margin.py
```
- **Container Exit Code**: `0` (`13728 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
one spot worst physical stress deviation 0.000375
two random spots worst physical stress deviation 0.001384
two wide spots worst physical stress deviation 0.000122
one EMI worst physical stress deviation 0.000151
no spots on one lab worst physical stress deviation 0.00017
one spot across wing worst physical stress deviation 0.000375
lab 0 baseline median abs curve/rate [0.007708 0.006651]
  interactions median abs curve/rate [0.007466 0.006962]
  rational stress median abs curve/rate [0.007738 0.006803]
  signed median rate residual by stratum [(0, 0.003385), (1, -0.000746), (2, 0.001861)]
lab 1 baseline median abs curve/rate [0.011565 0.00862 ]
  interactions median abs curve/rate [0.011811 0.008572]
  rational stress median abs curve/rate [0.011839 0.008636]
  signed median rate residual by stratum [(0, 0.001729), (1, 0.001293), (2, 0.005843)]
lab 2 baseline median abs curve/rate [0.013469 0.008895]
  interactions median abs curve/rate [0.01413  0.008484]
  rational stress median abs curve/rate [0.013546 0.009007]
  signed median rate residual by stratum [(0, 0.002525), (1, -0.00326), (2, 0.002869)]
lab 3 baseline median abs curve/rate [0.008155 0.007962]
  interactions median abs curve/rate [0.00878  0.007634]
  rational stress median abs curve/rate [0.008033 0.007802]
  signed median rate residual by stratum [(0, -0.003922), (1, 0.00488), (2, -0.002137)]
sparse EMI spots 1 worst (stress error, lab, row ids, contaminated index, spike) (0.0001702375022077096, 3, (9,), 0, 0.35) above 0.005 0 / 80
sparse EMI spots 2 worst (stress error, lab, row ids, contaminated index, spike) (0.00037517581665290345, 3, (8, 9), 1, 0.35) above 0.005 0 / 604

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b2a240e093a05a3e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_b2a240e0 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`20952 ms` LLM)
- **Rationale**: The calibration replay established that companion-lab corroboration fixes the stress error. Run the two previously worst cases through the full solver in an isolated process to detect whether the remaining calibration perturbation is amplified by thresholding or refitting; revise the fallback only if that occurs.
- **Alternatives Weighed**: `Change the constitutive basis` (rejected: The held-out interaction and rational-term challengers did not improve the worst complete-feature strain slice; the measured failure was in calibration.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import copy
exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
solve = s.solve_alpha_w03_batch_04
base = solve(d)
worst = dict.fromkeys(base, 0.)
for rows in ([5], [5, 6]):
    challenge = copy.deepcopy(d)
    tab = challenge['calibration_load_cells']
    wing_lab = np.flatnonzero(tab['lab_id'] == 3)
    tab['true_stress'][wing_lab[5]] += .35
    keep = (tab['lab_id'] != 3) | np.isin(np.arange(len(tab['lab_id'])), wing_lab[rows])
    challenge['calibration_load_cells'] = {k: np.asarray(v)[keep] for k, v in tab.items()}
    out = solve(challenge)
    err = {k: float(np.max(np.abs(out[k] - base[k]))) for k in base}
    print('lab 3 retained rows', rows, 'maximum change by output', err)
    for k in worst:
        worst[k] = max(worst[k], err[k])
assert worst['predicted_strain'] < .01 and worst['initial_rate_pred'] < .005, worst
assert np.max(np.abs(base['estimated_taus'])) > 0 and worst['estimated_taus'] < .02, worst
assert max(worst['hereditary_weights'], worst['viscoplastic_weights']) < .005, worst
PY
```
- **Container Exit Code**: `0` (`583 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 3 retained rows [5] maximum change by output {'estimated_taus': 5.160723148023294e-05, 'hereditary_weights': 4.578255246578955e-05, 'viscoplastic_weights': 2.4058955337746823e-05, 'predicted_strain': 0.003544473154412753, 'initial_rate_pred': 0.0007352021981872348}
lab 3 retained rows [5, 6] maximum change by output {'estimated_taus': 0.00038383403868991195, 'hereditary_weights': 4.7688091435293956e-05, 'viscoplastic_weights': 3.423646156991422e-05, 'predicted_strain': 0.005514935263019272, 'initial_rate_pred': 0.001139492539119935}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b2a240e093a05a3e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_b2a240e0 (directory).
```

### Turn 15 — `submit_completed_task` (`14069 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification and clean sparse calibration replays. The combined sparse-and-EMI replays showed that companion-lab corroboration fixed the observed calibration failure, and isolated end-to-end checks showed that the previously worst cases stayed within the validation margins for all five outputs.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w03_batch_04_solver.py`. The solver passed the public schema and constitutive checks. Across 684 sparse calibration replays with EMI, the worst stress deviation fell from 0.860 to 0.000375. Replaying the two previously worst cases through the full solver changed predicted strain by at most 0.00552 and initial rate by at most 0.00114.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
