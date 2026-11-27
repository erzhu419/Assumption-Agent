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
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `444,813` (`431,504` / `125,440` / `13,309`) |
| **Total Episode Wall Time (s)** | `545.32s` |

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

### Turn 1 — `execute_in_remote_docker` (`11495 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver for phase-specific assumptions before reusing any code; a mismatch would require a separate fit for the current regime.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py
```
- **Container Exit Code**: `0` (`263 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
```

### Turn 2 — `execute_in_remote_docker` (`19099 ms` LLM)
- **Rationale**: Inspect the alignment convention and summarize calibration residuals and missing observations. EMI or sparse calibration would trigger archive use; missing targets would require masking in the fit.

```bash
sed -n '1,105p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; head -c 2400 /tmp/hegel_continual_archive/archive_calibration_history.json; echo; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation robust IRLS' 2>&1; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb')); print('items',len(items))
for i in items:
 d=i['dataset']; print('keys',i.keys(),d.keys()); print('grid',d['eval_times'],d['past_dt'],'banding',d['shear_banding_index']);
 for k in ['train','test','calibration_load_cells']:
  print(k,[(f,np.shape(a),int(np.isnan(a).sum()) if np.asarray(a).dtype.kind=='f' else None) for f,a in d[k].items()]);
 c=d['calibration_load_cells'];
 for lab in range(4):
  ix=c['lab_id']==lab; x=c['cmd_stress'][ix]; y=c['true_stress'][ix]; p=np.polyfit(x,y,1) if len(x)>=2 else None; print('cal',lab,len(x),p,np.round(y-np.polyval(p,x),5) if p is not None else '')
PY
```
- **Container Exit Code**: `0` (`444 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
items 1
keys dict_keys(['dataset', 'gt']) dict_keys(['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test'])
grid [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] [0.4917 0.5023 0.5442 0.7304 0.4565 0.765  0.2965 0.6942 0.5869 0.3435
 0.3902 0.5082 0.3785 0.7877 0.3776 0.2105 0.2875 0.2881] banding 0.12
train [('lab_id', (180,), None), ('stratum', (180,), None), ('coupon_features_raw', (180, 4), 23), ('ref_coupons', (180, 32), 238), ('past_stress_cmd', (180, 18), 0), ('stress_hold_cmd', (180,), 0), ('strain_0', (180,), 0), ('aux_chamber_telemetry', (180, 3), 0), ('strain_curves_obs', (180, 10), 0), ('instant_rate_obs', (180,), 0)]
test [('lab_id', (72,), None), ('stratum', (72,), None), ('coupon_features_raw', (72, 4), 8), ('ref_coupons', (72, 32), 95), ('past_stress_cmd', (72, 18), 0), ('stress_hold_cmd', (72,), 0), ('strain_0', (72,), 0), ('aux_chamber_telemetry', (72, 3), 0)]
calibration_load_cells [('lab_id', (40,), None), ('cmd_stress', (40,), 0), ('true_stress', (40,), 0)]
cal 0 10 [1.28998908 0.12000143] [-1.0e-04  4.0e-05 -1.3e-04  9.0e-05  8.0e-05  2.2e-04  1.0e-05 -9.0e-05
 -1.5e-04  3.0e-05]
cal 1 10 [ 0.82003896 -0.15002605] [ 0.0e+00  3.0e-05 -1.0e-05 -7.0e-05 -1.0e-04  3.0e-05  1.1e-04  1.0e-04
  5.0e-05 -1.4e-04]
cal 2 10 [1.16998093 0.14000924] [-6.0e-05 -1.2e-04  7.0e-05  3.0e-05  4.0e-05 -9.0e-05 -1.0e-05  1.0e-05
  1.6e-04 -3.0e-05]
cal 3 10 [ 0.76000147 -0.10998819] [ 4.0e-05  2.0e-05 -4.0e-05 -1.0e-05 -1.0e-05  1.0e-05 -2.0e-05  4.0e-05
 -1.7e-04  1.5e-04]
```

### Turn 3 — `execute_in_remote_docker` (`18439 ms` LLM)
- **Rationale**: Resolve how reference statistics and cell-balanced fills are computed before fitting; an alignment mismatch would contaminate both the drift coefficients and held-out predictions. Identify which public truth fields are available for independent verification, not for fitting.
- **Alternatives Weighed**: `Use the archived stress gains for the public batch` (rejected: All four channels have ten clean public observations, so the specified in-batch fit takes precedence.)

```bash
sed -n '45,110p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; head -c 1400 /tmp/hegel_continual_archive/archive_calibration_history.json; echo; for f in autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$f" | head -c 3200; echo; done; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; g=item['gt']; d=item['dataset']; print('truth schema',[(k,np.shape(v)) for k,v in g.items()] if isinstance(g,dict) else type(g));
for split in ('train','test'):
 s=d[split]; print(split,'lab-stratum counts',np.unique(np.column_stack((s['lab_id'],s['stratum'])),axis=0,return_counts=True)); print('reference finite counts',np.sum(np.isfinite(s['ref_coupons']),axis=0));
PY
```
- **Container Exit Code**: `0` (`431 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
        "fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "cauchy_linear_fit",
          "signature": "cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float]",
          "doc_summary": "Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank before and after weighting, and positive scale if supplied and positive floor, returning (coefficients (p,), floored residual MAD)."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
        "autogenous_pri
truth schema [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
train lab-stratum counts (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([12, 13, 10, 12, 25, 12, 16, 15, 26,  8, 12, 19]))
reference finite counts [174 171 174 163 174 170 175 176 175 173 175 175 169 174 176 172 171 176
 176 171 175 175 172 169 174 169 172 172 166 168 178 172]
test lab-stratum counts (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([ 5,  7,  5,  2, 10,  4,  6,  3, 12,  6,  8,  4]))
reference finite counts [70 69 70 70 70 68 69 70 69 67 69 68 70 69 71 70 66 69 70 69 71 69 69 69
 66 67 71 65 70 71 68 70]
```

### Turn 4 — `execute_in_remote_docker` (`50031 ms` LLM)
- **Rationale**: Fit both observation channels using the specified strain equation and its analytic initial derivative. Compare against public truth only after fitting; parameter errors or lab-specific prediction errors would distinguish branch identification and alignment failures.
- **Alternatives Weighed**: `Reuse the prior linear-flow design unchanged` (rejected: It lacks the logarithmic clock and the two fast-branch crystallization terms required by this batch.)

```bash
set -e
cd /holdout_task
f=alpha_m97_c03_batch_01_solver.py
sed '/^def _design/,$d' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py | sed '1c\"""Calibrated two-branch Prony model for the crystallization campaign."""' > "$f"
cat >> "$f" <<'PY'
def _design(taus, past, hold, features, dt, t, damage):
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (np.exp(-trailing[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gap = hold[:, None] - memory
    relax = -np.expm1(-t[:, None] / taus)
    act = np.tanh(damage)
    active = damage > 0
    clock = np.where(active[:, None], np.log1p(t), t)
    extra = act * hold
    offset = act * memory[:, 0]
    curves = np.concatenate((gap[:, None, :] * relax,
                             features[:, None, :] * clock[:, :, None],
                             (extra[:, None] * relax[:, 0])[:, :, None],
                             (offset[:, None] * clock)[:, :, None]), axis=2)
    rates = np.column_stack((gap / taus, features, extra / taus[0], offset))
    return curves, rates


def _fit(past, hold, features, dt, t, damage, y, rates):
    valid_y, valid_r = np.isfinite(y), np.isfinite(rates)
    target = np.r_[y[valid_y], rates[valid_r]]
    n_curve = valid_y.sum()
    n_rate = valid_r.sum()
    k = 9 if np.any(damage > 0) else 7

    def matrix(log_tau):
        a, b = _design(np.exp(log_tau), past, hold, features, dt, t, damage)
        return np.vstack((a[valid_y, :k], b[valid_r, :k]))

    def scales(residual):
        def mad(x):
            return max(1e-6, 1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.
        return np.repeat([mad(residual[:n_curve]), mad(residual[n_curve:])], [n_curve, n_rate])

    lo = np.log(min(t[t > 0].min(), dt[dt > 0].min()) / 20)
    hi = np.log(max(t.max(), dt.sum()) * 20)
    bounds = (np.r_[lo, lo, np.full(k, -np.inf)], np.r_[hi, hi, np.full(k, np.inf)])

    def raw_residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - target

    def optimize(p, scale, loss):
        return least_squares(lambda q: raw_residual(q) / scale, p, bounds=bounds,
                             loss=loss, x_scale='jac', tr_solver='lsmr', max_nfev=130)

    starts = []
    for seed in ((.09, .48), (.045, .95), (.19, .7), (.13, 1.65), (.48, .09)):
        lt = np.log(t.max() * np.asarray(seed)).clip(lo + 1e-8, hi - 1e-8)
        starts.append(np.r_[lt, np.linalg.lstsq(matrix(lt), target, rcond=None)[0]])
    scale = scales(raw_residual(starts[0]))
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda f: f.cost)
    scale = 2 * scales(raw_residual(pilot.x))
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda f: f.cost)
    fit = optimize(fit.x, 2 * scales(raw_residual(fit.x)), 'cauchy')
    taus, weights = np.exp(fit.x[:2]), np.pad(fit.x[2:], (0, 9-k))
    if k == 7 and taus[0] > taus[1]:
        taus = taus[::-1]
        weights[:2] = weights[:2][::-1]
    return taus, weights


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    past, hold, features, baseline = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    damage = np.maximum(0, np.abs(past) - dataset['crit_stress'])**2 @ dt
    if dataset['shear_banding_index'] > .50:
        damage[:] = 0
    train = dataset['train']
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, t, damage[:n],
                         train['strain_curves_obs'] - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']))
    a, b = _design(taus, past[n:], hold[n:], features[n:], dt, t, damage[n:])
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': baseline[n:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 - <<'PY'
import pickle,numpy as np,time
from alpha_m97_c03_batch_01_solver import solve_alpha_m97_c03_batch_01,_prepare
x=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=x['dataset'],x['gt']
s=time.monotonic(); o=solve_alpha_m97_c03_batch_01(d); print('seconds',time.monotonic()-s)
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(a,np.round(o[a],7),np.round(g[b],7))
for a,b in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=o[a]-g[b]; print(a,'rmse',np.sqrt(np.mean(e**2)),'max',np.max(abs(e)),'lab rmses',[np.sqrt(np.mean(e[d['test']['lab_id']==c]**2)) for c in range(4)])
p,h,_,_=_prepare(d); active=(np.maximum(0,abs(p)-d['crit_stress'])**2@d['past_dt'])>0; print('active counts',np.bincount(active[:180]),np.bincount(active[180:]))
PY
```
- **Container Exit Code**: `0` (`6607 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 2.9478934730868787
estimated_taus [0.5739377 3.8967272] [0.568 4.113]
hereditary_weights [ 0.6843929  0.529457   0.1008878  0.1186501 -0.1179786  0.081732
 -0.0589822] [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
regime_mode_weights [0.2876195 0.2339289] [0.28 0.24]
predicted_strain rmse 0.22710136573094275 max 1.5976637344054119 lab rmses [0.19145951045185527, 0.2918585748705488, 0.2364491142284004, 0.17531920614461546]
initial_rate_pred rmse 0.07825725243981624 max 0.22464832290386605 lab rmses [0.0604625117703501, 0.10910254452134258, 0.07715984261118414, 0.059308399532584606]
active counts [ 60 120] [24 48]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5524c5df2a53d01c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5524c5df (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25640 ms` LLM)
- **Rationale**: Stratify public verification residuals by damage and missing coupons, then use the published reference parameters only for diagnosis. A time-dependent error on complete rows would implicate the response basis; an error confined to missing rows would implicate imputation.
- **Alternatives Weighed**: `Retune the robust optimizer immediately` (rejected: That would not resolve an incorrect clock or alignment rule; first isolate the systematic residual using the public verification data.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_01_solver as m
x=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=x['dataset'],x['gt']; p,h,f,b=m._prepare(d); n=len(d['train']['lab_id']); t=d['eval_times']; dt=d['past_dt']; weights=np.r_[g['theta'],g['vp_theta']]; p=np.vstack((p[:n],g['te_hist_true'])); h=np.r_[h[:n],g['te_hold_true']]; dmg=np.maximum(0,abs(p)-d['crit_stress'])**2@dt
a,r=m._design(g['taus'],p,h,f,dt,t,dmg); err=b[n:,None]+a[n:]@weights-g['te_clean']; er=r[n:]@weights-g['te_true_rate0']; missing=np.isnan(d['test']['coupon_features_raw']).any(axis=1)
for active in [False,True]:
 for miss in [False,True]:
  mask=(dmg[n:]>0)==active; mask &= missing==miss
  if not mask.any(): continue
  print('active, missing, count',active,miss,mask.sum(),'curve RMSE by t',np.sqrt(np.mean(err[mask]**2,0)).round(5),'rate',np.sqrt(np.mean(er[mask]**2))); print('error examples',err[mask][:3].round(5),er[mask][:3].round(5));
base=b[n:,None]+a[n:,:,:2]@weights[:2]+a[n:,:,7:]@weights[7:]; flow=g['te_clean']-base; drift=f[n:]@weights[2:7]; print('inactive inferred clock median',np.median(flow[(dmg[n:]==0)&~missing]/drift[(dmg[n:]==0)&~missing,None],axis=0)); print('linear',t,'log',np.log1p(t));
for grp,sl in [('train',slice(None,n)),('test',slice(n,None))]:
 ix=(dmg[sl]==0); print(grp,'inactive hold and maxpast quantiles',np.quantile(abs(h[sl][ix]),[0,.25,.5,.75,1]), np.quantile(abs(p[sl][ix]).max(1),[0,.25,.5,.75,1]),'yield',d['yield_stress'])
PY
```
- **Container Exit Code**: `0` (`523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
09326 0.13379 0.17699 0.22218 0.2689  0.31686
  0.3658 ]
 [0.01235 0.07183 0.15808 0.25996 0.37215 0.49161 0.61651 0.74558 0.87804
  1.0132 ]
 [0.01795 0.10447 0.22996 0.3782  0.54145 0.71527 0.89702 1.08485 1.27759
  1.47427]] [0.00059 0.00435 0.00624]
active, missing, count False True 1 curve RMSE by t [0.02591 0.08563 0.15454 0.22881 0.30661 0.3869  0.46906 0.55265 0.6374
 0.72308] rate 0.07373951561661052
error examples [[-0.02591 -0.08563 -0.15454 -0.22881 -0.30661 -0.3869  -0.46906 -0.55265
  -0.6374  -0.72308]] [-0.07374]
active, missing, count True False 42 curve RMSE by t [0.00316 0.00738 0.0105  0.01298 0.01503 0.01678 0.01832 0.01967 0.02089
 0.022  ] rate 0.012059262270186987
error examples [[ 2.540e-03  5.940e-03  8.440e-03  1.043e-02  1.209e-02  1.350e-02
   1.473e-02  1.582e-02  1.680e-02  1.769e-02]
 [-3.000e-05 -7.000e-05 -1.000e-04 -1.300e-04 -1.500e-04 -1.700e-04
  -1.800e-04 -2.000e-04 -2.100e-04 -2.200e-04]
 [ 1.460e-03  3.400e-03  4.830e-03  5.970e-03  6.920e-03  7.720e-03
   8.430e-03  9.050e-03  9.620e-03  1.013e-02]] [ 0.0097  -0.00012  0.00555]
active, missing, count True True 6 curve RMSE by t [0.02391 0.0558  0.07938 0.09809 0.11361 0.12687 0.13844 0.1487  0.15793
 0.16631] rate 0.09115040079762791
error examples [[ 0.0198   0.04621  0.06573  0.08123  0.09408  0.10506  0.11464  0.12314
   0.13078  0.13772]
 [-0.04083 -0.09526 -0.13552 -0.16746 -0.19396 -0.21659 -0.23635 -0.25387
  -0.26963 -0.28393]
 [ 0.02573  0.06003  0.08539  0.10552  0.12222  0.13648  0.14893  0.15997
   0.1699   0.17891]] [ 0.07548 -0.15562  0.09806]
inactive inferred clock median [0.25852773 0.60320253 0.85809887 1.06039191 1.22817388 1.37147639
 1.49658166 1.60755857 1.70730983 1.79786902]
linear [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] log [0.26236426 0.61215402 0.87083301 1.07612807 1.24639991 1.39182902
 1.51879084 1.63141465 1.7326462  1.82454929]
train inactive hold and maxpast quantiles [0.0442358  0.5509356  1.02999871 1.43597082 1.98175386] [1.59951623 1.77656396 1.84309829 1.89154649 1.90727757] yield 2.65
test inactive hold and maxpast quantiles [0.186788  0.682277  0.922647  1.3062255 1.932947 ] [1.63334232 1.80363933 1.84823048 1.87714907 1.90545667] yield 2.65

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5524c5df2a53d01c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5524c5df (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34719 ms` LLM)
- **Rationale**: Measure the clock correction separately on damaged, undamaged, complete, and missing-coupon rows. Training residuals on complete versus missing rows will show whether the remaining error warrants further work on alignment or imputation.
- **Alternatives Weighed**: `Keep linear flow and change coupon alignment` (rejected: An alignment correction changes drift amplitude but cannot turn a linear time basis into the observed logarithmic curvature.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero damage switches the low-banding campaign’s drift clock back to linear time.` | Witness: `On 23 complete undamaged public rows, the inferred clock ends at 1.798, close to log1p(5.2) = 1.825 and far from 5.2; substituting the reference parameters into the linear-clock model gives final strain RMSE 1.116.` | Blanket Diagnosis: `The solver applied the sample damage gate to the clock as well as to the two damage-dependent amplitudes.` | Sublated Pivot: `Use the logarithmic clock throughout the low-banding campaign and preserve the damage dependence of the two correction amplitudes.`

```bash
set -e
cd /holdout_task
sed -n '76,115p' alpha_m97_c03_batch_01_solver.py
sed -i -e 's/def _design(taus, past, hold, features, dt, t, damage):/def _design(taus, past, hold, features, dt, t, damage, plateau):/' -e '/    active = damage > 0/d' -e 's/clock = np.where(active\[:, None\], np.log1p(t), t)/clock = np.log1p(t)[None, :] if plateau else t[None, :]/' -e 's/def _fit(past, hold, features, dt, t, damage, y, rates):/def _fit(past, hold, features, dt, t, damage, y, rates, plateau):/' -e 's/past, hold, features, dt, t, damage)$/past, hold, features, dt, t, damage, plateau)/' -e "s/if dataset\['shear_banding_index'\] > .50:/plateau = dataset['shear_banding_index'] <= .50\n    if not plateau:/" -e "s/np.asarray(train\['instant_rate_obs'\]))/np.asarray(train['instant_rate_obs']), plateau)/" -e 's/damage\[n:\])$/damage[n:], plateau)/' alpha_m97_c03_batch_01_solver.py
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_01_solver as m
item=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=item['dataset'],item['gt']; out=m.solve_alpha_m97_c03_batch_01(d)
for key,ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(key,out[key],g[ref]);
p,h,f,b=m._prepare(d); n=len(d['train']['lab_id']); dt,t=d['past_dt'],d['eval_times']; dmg=np.maximum(0,abs(p)-d['crit_stress'])**2@dt
miss=np.isnan(d['test']['coupon_features_raw']).any(1); lab=d['test']['lab_id']; masks={'all':np.ones(len(lab),bool),'missing':miss,'complete':~miss,'damaged':dmg[n:]>0,'undamaged':dmg[n:]==0,**{f'lab{c}':lab==c for c in range(4)}}
for key,ref in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=out[key]-g[ref]; print(key,{k:(float(np.sqrt(np.mean(e[i]**2))),float(np.max(abs(e[i])))) for k,i in masks.items()})
a,r=m._design(out['estimated_taus'],p[:n],h[:n],f[:n],dt,t,dmg[:n],True); w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; err=b[:n,None]+a@w-d['train']['strain_curves_obs']; er=r@w-d['train']['instant_rate_obs']; missing=np.isnan(d['train']['coupon_features_raw']).any(1)
for label,i in [('complete',~missing),('missing',missing)]: print('train',label,'curve and rate absolute residual quantiles',np.quantile(abs(err[i]),[.25,.5,.9,.99]),np.quantile(abs(er[i]),[.25,.5,.9,.99]));
PY
```
- **Container Exit Code**: `0` (`7232 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
    baseline = joined('strain_0')
    return past, hold, np.column_stack((np.ones(len(lab)), aligned)), baseline


def _design(taus, past, hold, features, dt, t, damage):
{"schema_valid": true, "errors": []}
estimated_taus [0.56683462 3.95879779] [0.568 4.113]
hereditary_weights [ 0.67490365  0.54231973  0.13823235  0.15665472 -0.12837941  0.11571591
 -0.08861017] [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
regime_mode_weights [0.28213852 0.23890293] [0.28 0.24]
predicted_strain {'all': (0.037181178223170394, 0.2776881171062149), 'missing': (0.11444208641858192, 0.2776881171062149), 'complete': (0.010994224361825589, 0.05938841978553855), 'damaged': (0.043317377637122505, 0.2776881171062149), 'undamaged': (0.019862769984957264, 0.12032571481516685), 'lab0': (0.03709808341094933, 0.1747588386216552), 'lab1': (0.051333200121030285, 0.2776881171062149), 'lab2': (0.033095969643696994, 0.15591079663552032), 'lab3': (0.024692961481508863, 0.10259811073754313)}
initial_rate_pred {'all': (0.0286736572259425, 0.15410138370605878), 'missing': (0.08810153934571711, 0.15410138370605878), 'complete': (0.008650204000783588, 0.027306856432059767), 'damaged': (0.033530031878076565, 0.15410138370605878), 'undamaged': (0.014765154270879526, 0.06192924595277827), 'lab0': (0.02805929100704612, 0.0962669843441244), 'lab1': (0.03973014414347473, 0.15410138370605878), 'lab2': (0.026318207898355525, 0.08964556301621607), 'lab3': (0.01827414184197368, 0.05543153738553808)}
train complete curve and rate absolute residual quantiles [0.00544048 0.01190466 0.03850095 3.61141554] [0.00648568 0.01294541 0.03474032 0.05054146]
train missing curve and rate absolute residual quantiles [0.03299053 0.0670292  0.37297197 0.86432474] [0.02120765 0.06094217 0.18532052 0.34055864]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eeb4365d7ee97dd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_eeb4365d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28198 ms` LLM)
- **Rationale**: Use public reference parameters and stresses only for diagnosis. Compare the magnitude of the missing-coupon errors with plausible interpretations of the imputation rule; if those shifts are small, focus on fitting stability and calibration fallbacks rather than changing the imputer.
- **Alternatives Weighed**: `Tune imputation to public holdout truth` (rejected: Public truth can isolate the residual source, but the solver must use the specified cell-balanced estimator.)

```bash
cd /holdout_task && export OPENBLAS_NUM_THREADS=1 && python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_01_solver as m
items=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb')); print('public cases:',len(items)); d,g=items[0]['dataset'],items[0]['gt']; n=len(d['train']['lab_id']); p,h,f,b=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; lab=np.r_[d['train']['lab_id'],d['test']['lab_id']]; st=np.r_[d['train']['stratum'],d['test']['stratum']]; raw=np.vstack([d[k]['coupon_features_raw'] for k in ('train','test')]); miss=np.isnan(raw); p[n:],h[n:]=g['te_hist_true'],g['te_hold_true']; dmg=np.maximum(0,abs(p)-d['crit_stress'])**2@dt; a,r=m._design(g['taus'],p,h,f,dt,t,dmg,True); w=np.r_[g['theta'],g['vp_theta']]; e=r[n:]@w-g['te_true_rate0']; print('rate error with public reference parameters and stresses, complete by lab:',[(int(c),float(e[(lab[n:]==c)&~miss[n:].any(1)].mean()),float(np.sqrt(np.mean(e[(lab[n:]==c)&~miss[n:].any(1)]**2)))) for c in range(4)]); print('missing: row lab stratum missing columns current rate error cell-mean delta stratum-balanced delta');
for i in np.flatnonzero(miss.any(1)&(np.arange(len(lab))>=n)):
 cols=miss[i]; v=np.where(miss,np.nan,f[:,1:]); cell=np.nanmean(v[(lab==lab[i])&(st==st[i])],axis=0); strat=np.mean([np.nanmean(v[(lab==c)&(st==st[i])],axis=0) for c in range(4)],axis=0); print(i-n,lab[i],st[i],np.flatnonzero(cols),round(e[i-n],5),round((cell[cols]-f[i,1:][cols])@w[3:7][cols],5),round((strat[cols]-f[i,1:][cols])@w[3:7][cols],5))
for sname,s in [('train',slice(None,n)),('test',slice(n,None))]: print(sname,'complete undamaged terminal oracle rmse',np.sqrt(np.mean((b[s,None]+a[s]@w-g['te_clean'])[(dmg[s]==0)&~miss[s].any(1),-1]**2)) if sname=='test' else 'not scored', 'missing counts',miss[s].sum(0));
PY
```
- **Container Exit Code**: `0` (`438 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public cases: 1
rate error with public reference parameters and stresses, complete by lab: [(0, 0.0011897490931395804, 0.006081403433865572), (1, 0.0003681372768782976, 0.0005855240612964955), (2, 0.0034102739967118126, 0.004361160161045212), (3, 0.0023159498149589028, 0.019226733078809982)]
missing: row lab stratum missing columns current rate error cell-mean delta stratum-balanced delta
16 2 1 [0] 0.07548 0.00752 -0.01568
18 1 1 [1] -0.15562 0.01632 0.01282
27 0 2 [3] 0.09806 -0.03164 -0.03361
44 1 1 [0] -0.00906 0.00261 -0.01568
45 3 2 [0 3] -0.05093 0.06697 0.02752
47 2 2 [2] 0.08744 0.04727 0.02795
62 0 1 [3] -0.07374 0.01603 0.01498
train complete undamaged terminal oracle rmse not scored missing counts [ 5  2  5 11]
test complete undamaged terminal oracle rmse 0.012602926264471235 missing counts [3 1 1 3]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eeb4365d7ee97dd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_eeb4365d (directory).
```

### Turn 8 — `execute_in_remote_docker` (`24798 ms` LLM)
- **Rationale**: Compare fits with and without imputed training rows on complete rows from each held-out lab. Change the fit only if it improves the worst held-out slice. Inspect the calibration fallback before stress-testing sparse sweeps and EMI.
- **Alternatives Weighed**: `Replace the specified imputation rule` (rejected: The tested mean variants do not consistently explain the missing-coupon residuals; retain the prescribed estimator for prediction.)

```bash
cd /holdout_task && sed -n '1,82p' alpha_m97_c03_batch_01_solver.py && head -c 1800 /tmp/hegel_continual_archive/archive_calibration_history.json && echo && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_m97_c03_batch_01_solver as m
item=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=item['dataset']; s=d['train']; p,h,f,b=m._prepare(d); n=len(s['lab_id']); t,dt=d['eval_times'],d['past_dt']; dmg=np.maximum(0,abs(p)-d['crit_stress'])**2@dt; y=s['strain_curves_obs']-b[:n,None]; r=s['instant_rate_obs']; complete=np.isfinite(s['coupon_features_raw']).all(1); lab=s['lab_id'];
def fit(mask): return m._fit(p[:n][mask],h[:n][mask],f[:n][mask],dt,t,dmg[:n][mask],y[mask],r[mask],True)
for leave in [-1,0,1,2,3]:
    validation=(lab==leave)&complete
    for use_missing in [True,False]:
        train=(lab!=leave)&(use_missing|complete)
        tau,w=fit(train)
        if leave==-1: print('all labs',use_missing,'tau',tau,'weights',w.round(5)); continue
        a,j=m._design(tau,p[:n][validation],h[:n][validation],f[:n][validation],dt,t,dmg[:n][validation],True)
        ec=np.abs(a@w-y[validation]); er=np.abs(j@w-r[validation]); print('LOLO',leave,use_missing,'tau',tau.round(5),'curve abs median/p85',np.nanquantile(ec,[.5,.85]).round(5),'rate abs median/p85',np.nanquantile(er,[.5,.85]).round(5))
PY
```
- **Container Exit Code**: `0` (`29710 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2,
      2.18392,
      -0.23172,
      1.98347,
      -2.000907,
      -0.486537,
      -0.139492,
      -0.631342,
      -1.874772,
      -0.708257,
      2.13352,
      0.386634,
      1.10423,
      -1.407517,
      1.70372,
      1.521295,
      1.987385,
      1.659545,
      -0.294044,
      0.359165,
      -0.017198,
      1.319848,
      0.959274,
      1.866895,
      -0.480321
    ],
    "true_stress": [
      0.893713,
      -0.483774,
      -1.392656,
      1.611564,
      -1.810667,
      -1.870622,
      1.480773,
      1.268837,
      -0.114695,
      0.252685,
      -1.204516,
      -0.664704,
      -0.154113,
      -0.613571,
      -1.259248,
      -2.011128,
      2.183777,
      -0.231673,
      1.983509,
      -2.001075,
      -0.486729,
      -0.139381,
      -0.631253,
      -1.874747,
      -0.708398,
      2.133538,
      0.386675,
      1.104402,
      -1.407596,
      1.703866,
      1.521337,
      1.987399,
      1.659449,
      -0.294029,
      0.359147,
      -0.017404,
      1.319703,
      0.959321,
      1.866928,
      -0.480472
   
all labs True tau [0.56683462 3.95879779] weights [ 0.6749   0.54232  0.13823  0.15665 -0.12838  0.11572 -0.08861  0.28214
  0.2389 ]
all labs False tau [0.56757434 4.03391525] weights [ 0.67719  0.54489  0.13847  0.15693 -0.12783  0.11644 -0.08855  0.2816
  0.23884]
LOLO 0 True tau [0.5671  3.96473] curve abs median/p85 [0.01108 0.02775] rate abs median/p85 [0.01299 0.02955]
LOLO 0 False tau [0.56759 4.04837] curve abs median/p85 [0.01129 0.02761] rate abs median/p85 [0.01242 0.02978]
LOLO 1 True tau [0.56616 3.91763] curve abs median/p85 [0.01271 0.02952] rate abs median/p85 [0.01124 0.02324]
LOLO 1 False tau [0.56736 4.02803] curve abs median/p85 [0.01225 0.02916] rate abs median/p85 [0.01005 0.02394]
LOLO 2 True tau [0.56664 3.99731] curve abs median/p85 [0.01222 0.02918] rate abs median/p85 [0.01414 0.03045]
LOLO 2 False tau [0.56718 4.03498] curve abs median/p85 [0.01164 0.02796] rate abs median/p85 [0.01358 0.02987]
LOLO 3 True tau [0.56812 4.02672] curve abs median/p85 [0.01794 0.04696] rate abs median/p85 [0.01297 0.04053]
LOLO 3 False tau [0.56825 4.05078] curve abs median/p85 [0.01784 0.04792] rate abs median/p85 [0.0123  0.04105]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eeb4365d7ee97dd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_eeb4365d (directory).
```

### Turn 9 — `execute_in_remote_docker` (`38680 ms` LLM)
- **Rationale**: The archive contains raw sweeps, while the inherited reader appears to expect fitted coefficients. Removing all but one point per lab will show whether it silently falls back to unit gain; injecting a signed spike at each observed command will show whether the EMI filter needs a change.
- **Alternatives Weighed**: `Switch immediately to complete-only training` (rejected: It moves the slow time constant closer to the public reference, but slightly worsens both worst held-out-lab tail errors. Resolve the calibration fallback first.)

```bash
cd /holdout_task && sed -n '1,90p' alpha_m97_c03_batch_01_solver.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json,pickle,numpy as np
import alpha_m97_c03_batch_01_solver as m
history=json.load(open(m._ARCHIVE)); print('archive record schemas:',[(r.get('episode_short_name'),list(r)) for r in history]); d=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; base=m._calibrate(c)
for lab in range(4):
    ii=np.flatnonzero(c['lab_id']==lab)
    keep=(c['lab_id']!=lab); keep[ii[0]]=True
    sparse={k:v[keep] for k,v in c.items()}
    print('lab',lab,'archive parser',m._archive_calibration(lab),'sparse fit',m._calibrate(sparse)[lab],'in-batch fit',base[lab]);
    errors=[]
    for j in ii:
        for delta in (-4.,4.):
            spike={k:v.copy() for k,v in c.items()}; spike['true_stress'][j]+=delta
            errors.append(np.max(abs(m._calibrate(spike)[lab]-base[lab])))
    print('single EMI worst coefficient change',max(errors))
PY
```
- **Container Exit Code**: `1` (`511 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.01:
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
        result.append(fit)
    return np.asarray(result)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab, stratum = joined('lab_id'), joined('stratum')
archive record schemas: [('alpha_m97_c03_s00', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=eeb4365d7ee97dd5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_eeb4365d (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "/holdout_task/alpha_m97_c03_batch_01_solver.py", line 26, in _archive_calibration
    if int(r.get('lab_id', lab)) == lab and 'gain' in r and 'bias' in r]
       ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'
```

### Turn 10 — `execute_in_remote_docker` (`33528 ms` LLM)
- **Rationale**: The sparse path crashes because the archive stores arrays of raw sweep measurements rather than scalar fitted coefficients. Fit the archived measurements by channel, then verify that zero- and one-point inputs use that fit and that a signed EMI spike does not displace a clean in-batch fit.
- **Alternatives Weighed**: `Reuse the archived calibration for clean in-batch channels` (rejected: The task prioritizes clean in-batch fits, and prefix calibration may differ from the current campaign.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_01_solver.py')
s = p.read_text()
a, b = s.index('def _archive_calibration('), s.index('def _calibrate(')
s = s[:a] + '''def _archive_calibration(lab):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    fits = []
    for sweep in history:
        rows = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'], dtype=float)[rows]
        y = np.asarray(sweep['true_stress'], dtype=float)[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size >= 2 and np.ptp(x) > 0:
            fits.append(np.polyfit(x, y, 1))
    return np.median(fits, axis=0) if fits else None


''' + s[b:]
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=item['dataset']; c=d['calibration_load_cells']; base=m._calibrate(c); history=json.load(open(m._ARCHIVE)); archived=history[-1]
for lab in range(4):
    ii=np.flatnonzero(c['lab_id']==lab); ha=np.asarray(archived['lab_id'])==lab; expected=np.polyfit(np.asarray(archived['cmd_stress'])[ha],np.asarray(archived['true_stress'])[ha],1)
    np.testing.assert_allclose(m._archive_calibration(lab),expected,atol=1e-12)
    for keep_one in [False,True]:
        keep=c['lab_id']!=lab; keep[ii[0]]=keep_one
        np.testing.assert_allclose(m._calibrate({k:v[keep] for k,v in c.items()})[lab],expected,atol=1e-12)
    worst=0.
    for j in ii:
        for delta in (-4.,4.):
            spike={k:v.copy() for k,v in c.items()}; spike['true_stress'][j]+=delta
            worst=max(worst,np.max(np.abs(m._calibrate(spike)[lab]-base[lab])))
    assert worst < 3e-4, (lab,worst)
    print('lab',lab,'prefix fit',expected,'current fit',base[lab],'worst EMI coefficient change',worst)
PY
```
- **Container Exit Code**: `0` (`4364 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
lab 0 prefix fit [9.99996399e-01 1.32406004e-05] current fit [1.28998908 0.12000143] worst EMI coefficient change 3.818153686449555e-05
lab 1 prefix fit [ 9.99987895e-01 -2.46087894e-05] current fit [ 0.82003896 -0.15002605] worst EMI coefficient change 1.8460903045997012e-05
lab 2 prefix fit [1.00004377e+00 1.86490255e-05] current fit [1.16998093 0.14000924] worst EMI coefficient change 1.9925208780219306e-05
lab 3 prefix fit [ 1.00004220e+00 -8.39804234e-05] current fit [ 0.76000147 -0.10998819] worst EMI coefficient change 2.4296443576904414e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8f76e9bae84788 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_3c8f76e9 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`36202 ms` LLM)
- **Rationale**: Check the final artifact against an independent stepwise memory recurrence and the analytic rate for both clocks. A mismatch would require correcting the time basis or initial derivative; the slice metrics separate those failures from the known missing-coupon error.

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
sed -n '/^def _prepare/,/^def _design/p' alpha_m97_c03_batch_01_solver.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item = pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]
d, g = item['dataset'], item['gt']
o = m.solve_alpha_m97_c03_batch_01(d)
p,h,f,b = m._prepare(d)
n = len(d['train']['lab_id']); t, dt = d['eval_times'], d['past_dt']
damage = np.maximum(0, np.abs(p[n:])-d['crit_stress'])**2 @ dt
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
tau = o['estimated_taus']
assert np.all(tau > 0)
mem = np.zeros((len(h)-n,2))
for j, step in enumerate(dt):
    decay = np.exp(-step/tau)
    mem = decay*mem + (1-decay)*p[n:,j,None]
gap = h[n:,None]-mem
act = np.tanh(damage)
drift = f[n:] @ w[2:7]
prony = (gap*w[:2]) @ (-np.expm1(-t[:,None]/tau)).T
for plateau in (True, False):
    a,r = m._design(tau,p[n:],h[n:],f[n:],dt,t,damage if plateau else np.zeros_like(damage),plateau)
    relax = -np.expm1(-t/tau[0]); clock = np.log1p(t) if plateau else t
    strain = b[n:,None] + prony + drift[:,None]*clock
    rate = gap @ (w[:2]/tau) + drift
    if plateau:
        strain += (w[7]*act*h[n:])[:,None]*relax + (w[8]*act*mem[:,0])[:,None]*clock
        rate += w[7]*act*h[n:]/tau[0]+w[8]*act*mem[:,0]
        np.testing.assert_allclose(o['predicted_strain'],strain,atol=1e-11)
        np.testing.assert_allclose(o['initial_rate_pred'],rate,atol=1e-11)
    np.testing.assert_allclose(b[n:,None]+a@w,strain,atol=1e-11)
    np.testing.assert_allclose(r@w,rate,atol=1e-11)
lab=d['test']['lab_id']; missing=~np.isfinite(d['test']['coupon_features_raw']).all(1)
slices={'all':np.ones(len(lab),bool),'damaged':damage>0,'undamaged':damage==0,'complete':~missing,'missing':missing,**{f'lab{i}':lab==i for i in range(4)}}
for key,truth in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    err=o[key]-g[truth]; assert np.isfinite(err).all()
    print(key,'slice RMSE',{k:round(np.sqrt(np.mean(err[v]**2)),6) for k,v in slices.items()}); print(key,'abs max',float(np.max(abs(err))))
for key,truth in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
    assert np.isfinite(o[key]).all()
    print(key,'max relative error',float(np.max(abs(o[key]/g[truth]-1))))
print('Independent recurrence, strain, and rate checks passed for both clocks.')
PY
```
- **Container Exit Code**: `0` (`3958 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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


def _design(taus, past, hold, features, dt, t, damage, plateau):
predicted_strain slice RMSE {'all': 0.037181, 'damaged': 0.043317, 'undamaged': 0.019863, 'complete': 0.010994, 'missing': 0.114442, 'lab0': 0.037098, 'lab1': 0.051333, 'lab2': 0.033096, 'lab3': 0.024693}
predicted_strain abs max 0.2776881171062149
initial_rate_pred slice RMSE {'all': 0.028674, 'damaged': 0.03353, 'undamaged': 0.014765, 'complete': 0.00865, 'missing': 0.088102, 'lab0': 0.028059, 'lab1': 0.03973, 'lab2': 0.026318, 'lab3': 0.018274}
initial_rate_pred abs max 0.15410138370605878
estimated_taus max relative error 0.03749142068701039
hereditary_weights max relative error 0.03570071298447719
regime_mode_weights max relative error 0.007637558394833377
Independent recurrence, strain, and rate checks passed for both clocks.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8f76e9bae84788 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_3c8f76e9 (directory).
```

### Turn 12 — `submit_completed_task` (`14809 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated-process verification. The measured worst slice is reported without tuning the prescribed imputation rule to public test outcomes.
- **Alternatives Weighed**: `Fit only on rows with complete coupon features` (rejected: It slightly worsened the worst held-out-lab 85th-percentile errors for both strain and initial rate.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:8372'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The solver passed the public schema, sparse and EMI calibration checks, and independent strain and initial-rate checks. Public RMSE was 0.0372 for strain and 0.0287 for initial rate; rows with missing coupon features had the largest errors.`

### Turn 13 — `execute_in_remote_docker` (`39368 ms` LLM)
- **Rationale**: A repeatable worst-lab improvement from pairwise or rational drift would justify investigating a missing term. Failure under per-lab unit changes would point to calibration or panel alignment. Neither probe uses public test truth to fit or select coefficients.
- **Alternatives Weighed**: `Test relay modes, HJB drift, or classification balance` (rejected: The public rheology data contain no relay states, stochastic control targets, or class labels. Held-out lab and within-lab stratum residuals test the relevant nuisance effects.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
cat >/tmp/alpha_m97_c03_audit.py <<'PY'
import copy, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0]['dataset']; s=d['train']; n=len(s['lab_id']); t,dt=d['eval_times'],d['past_dt']; p,h,f,b=m._prepare(d); damage=np.maximum(0,abs(p)-d['crit_stress'])**2@dt; y=s['strain_curves_obs']-b[:n,None]; r=s['instant_rate_obs']; complete=np.isfinite(s['coupon_features_raw']).all(1); lab=s['lab_id']
# Fit challenger coefficients exclusively on each fold's training labs. They change only the log-clock amplitude and its matching initial derivative.
for leave in range(4):
    tr=lab!=leave; va=(lab==leave)&complete; idx=np.flatnonzero(va)
    tau,w=m._fit(p[:n][tr],h[:n][tr],f[:n][tr],dt,t,damage[:n][tr],y[tr],r[tr],True)
    A,B=m._design(tau,p[:n],h[:n],f[:n],dt,t,damage[:n],True); ec=y-A@w; er=r-B@w
    z=f[:n,1:]; diff=p[:n,-1]-h[:n]; pair=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)])
    rational=np.column_stack([h[:n]/(1+diff**2),diff/(1+h[:n]**2),np.tanh(damage[:n])*diff/(1+diff**2)])
    options={'specified':(ec,er)}
    for name,F in [('pairwise',pair),('rational',rational)]:
        C=F[:,None,:]*np.log1p(t)[None,:,None]; iy=tr[:,None]&complete[:,None]&np.isfinite(ec); ir=tr&complete&np.isfinite(er)
        scales=[max(1e-6,1.4826*np.median(abs(v-np.median(v)))) for v in [ec[iy],er[ir]]]
        M=np.vstack([C[iy]/scales[0],F[ir]/scales[1]]); q=np.r_[ec[iy]/scales[0],er[ir]/scales[1]]
        coeff=least_squares(lambda x:M@x-q,np.zeros(F.shape[1]),loss='cauchy',f_scale=2).x
        options[name]=(ec-C@coeff,er-F@coeff)
    for name,(c,e) in options.items():
        print('held lab',leave,name,'strain/rate p85',np.nanquantile(abs(c[idx]),.85),np.nanquantile(abs(e[idx]),.85),'signed rate medians by stratum',[round(float(np.nanmedian(e[idx][s['stratum'][idx]==g])),5) for g in np.unique(s['stratum'][idx])])
# Unit and origin changes per lab must leave aligned inputs and outputs unchanged.
d2=copy.deepcopy(d); gain=np.array([1.4,.65,2.,.8]); shift=np.array([-.9,1.3,.5,-2.]); cg=np.array([.8,1.3,.7,1.1]); cb=np.array([.6,-.9,.3,.2]); c=d2['calibration_load_cells']; j=c['lab_id']; c['cmd_stress']=cg[j]*c['cmd_stress']+cb[j]
for key in ('train','test'):
    a=d2[key]; j=a['lab_id']; a['ref_coupons']=gain[j,None]*a['ref_coupons']+shift[j,None]; a['coupon_features_raw']=gain[j,None]*a['coupon_features_raw']+shift[j,None]; a['past_stress_cmd']=cg[j,None]*a['past_stress_cmd']+cb[j,None]; a['stress_hold_cmd']=cg[j]*a['stress_hold_cmd']+cb[j]
for a,c in zip(m._prepare(d),m._prepare(d2)): np.testing.assert_allclose(a,c,atol=1e-11,rtol=0)
a,c=m.solve_alpha_m97_c03_batch_01(d),m.solve_alpha_m97_c03_batch_01(d2)
for k in a:
    print('lab affine invariance',k,float(np.max(abs(a[k]-c[k])))); np.testing.assert_allclose(a[k],c[k],atol=1e-5,rtol=0)
PY
python3 -I /tmp/alpha_m97_c03_audit.py
```
- **Container Exit Code**: `1` (`17506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 specified strain/rate p85 0.02774894989028564 0.02954730195600118 signed rate medians by stratum [0.00688, 0.00115, -0.0145]
held lab 0 pairwise strain/rate p85 0.02900690404333079 0.028638873416032353 signed rate medians by stratum [0.00658, -0.00161, -0.01145]
held lab 0 rational strain/rate p85 0.028049570020028704 0.02913018664297687 signed rate medians by stratum [0.00624, 0.00109, -0.01513]
held lab 1 specified strain/rate p85 0.029524231896222965 0.023242930085302246 signed rate medians by stratum [-0.00546, 0.00408, 0.00472]
held lab 1 pairwise strain/rate p85 0.031089024870042902 0.02273434863323695 signed rate medians by stratum [-0.00588, 0.00261, 0.00369]
held lab 1 rational strain/rate p85 0.029910761299009613 0.024502353581975795 signed rate medians by stratum [-0.00359, 0.00377, 0.00542]
held lab 2 specified strain/rate p85 0.02917642988257091 0.030451461614220843 signed rate medians by stratum [0.00291, 0.01353, 0.01043]
held lab 2 pairwise strain/rate p85 0.030925527849622214 0.029450728484459003 signed rate medians by stratum [-0.00013, 0.01101, 0.00627]
held lab 2 rational strain/rate p85 0.029283855768604122 0.0304741203218983 signed rate medians by stratum [0.00275, 0.01388, 0.01004]
held lab 3 specified strain/rate p85 0.046963914612165834 0.04052519086580651 signed rate medians by stratum [0.00881, -0.00719, 0.00488]
held lab 3 pairwise strain/rate p85 0.0470942166348709 0.041384968412314656 signed rate medians by stratum [0.00877, -0.00754, 0.00352]
held lab 3 rational strain/rate p85 0.047297367561728185 0.04035204634613037 signed rate medians by stratum [0.00906, -0.00835, 0.0052]
lab affine invariance estimated_taus 0.0285149020340123

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8f76e9bae84788 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_3c8f76e9 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/alpha_m97_c03_audit.py", line 29, in <module>
    print('lab affine invariance',k,float(np.max(abs(a[k]-c[k])))); np.testing.assert_allclose(a[k],c[k],atol=1e-5,rtol=0)
                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 1504, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/usr/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 797, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0, atol=1e-05

Mismatched elements: 2 / 2 (100%)
Max absolute difference: 0.0285149
Max relative difference: 0.00715141
 x: array([0.566835, 3.958798])
 y: array([0.566949, 3.987313])
```

### Turn 14 — `execute_in_remote_docker` (`25530 ms` LLM)
- **Rationale**: Test whether more accurate derivatives and linear subproblem solves eliminate the observed unit sensitivity without degrading public prediction or parameter recovery. The same probe checks all five outputs; the baseline comparison guards against trading away prediction accuracy for numerical invariance.
- **Alternatives Weighed**: `Add pairwise or rational drift terms` (rejected: Neither improved the worst held-out-lab strain error, and the observed unit sensitivity arises after preprocessing.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
cp alpha_m97_c03_batch_01_solver.py /tmp/alpha_m97_v03.py
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c03_batch_01_solver.py'); s=p.read_text(); old="loss=loss, x_scale='jac', tr_solver='lsmr', max_nfev=130)"; new="loss=loss, jac='cs', x_scale='jac', tr_solver='exact',\n                             ftol=1e-10, gtol=1e-10, xtol=1e-10, max_nfev=180)"; assert old in s; p.write_text(s.replace(old,new))
PY
python3 -I validate_public_schema.py
python3 -I /tmp/alpha_m97_c03_audit.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np,time
x=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=x['dataset'],x['gt']; miss=np.isnan(d['test']['coupon_features_raw']).any(1)
for p in ('/tmp/alpha_m97_v03.py','alpha_m97_c03_batch_01_solver.py'):
 spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); t=time.monotonic(); o=m.solve_alpha_m97_c03_batch_01(d); print(p,'seconds',round(time.monotonic()-t,3))
 for key,ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(key,'max relative error',np.max(abs(o[key]/g[ref]-1)))
 for key,ref in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
  e=o[key]-g[ref]; print(key,'pooled / complete / missing RMSE',[round(float(np.sqrt(np.mean(v**2))),6) for v in (e,e[~miss],e[miss])])
PY
```
- **Container Exit Code**: `0` (`9808 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
4263035 0.023470812773846994 signed rate medians by stratum [-0.00536, 0.00364, 0.00464]
held lab 1 pairwise strain/rate p85 0.030985251703751018 0.02323115513220463 signed rate medians by stratum [-0.00577, 0.00275, 0.00359]
held lab 1 rational strain/rate p85 0.029716697871666475 0.024687496791513427 signed rate medians by stratum [-0.00349, 0.00367, 0.00533]
held lab 2 specified strain/rate p85 0.029134010837416506 0.030398899485746776 signed rate medians by stratum [0.00287, 0.01357, 0.01029]
held lab 2 pairwise strain/rate p85 0.03103109020815202 0.02950440171847659 signed rate medians by stratum [-0.00042, 0.01096, 0.00628]
held lab 2 rational strain/rate p85 0.029263949254336095 0.03042508465440967 signed rate medians by stratum [0.00283, 0.01389, 0.0099]
held lab 3 specified strain/rate p85 0.04717229503012167 0.04050090780574039 signed rate medians by stratum [0.0082, -0.00695, 0.00508]
held lab 3 pairwise strain/rate p85 0.0473383862945248 0.04204825732064341 signed rate medians by stratum [0.00814, -0.00734, 0.00386]
held lab 3 rational strain/rate p85 0.047182238167704144 0.04030537832335841 signed rate medians by stratum [0.00844, -0.00806, 0.00533]
lab affine invariance estimated_taus 2.4195356829181947e-10
lab affine invariance hereditary_weights 7.2026828945581656e-12
lab affine invariance regime_mode_weights 1.666222715357435e-12
lab affine invariance predicted_strain 2.7305491201445875e-11
lab affine invariance initial_rate_pred 1.779554281711171e-11
/tmp/alpha_m97_v03.py seconds 3.27
estimated_taus max relative error 0.03749142068701039
hereditary_weights max relative error 0.03570071298447719
regime_mode_weights max relative error 0.007637558394833377
predicted_strain pooled / complete / missing RMSE [0.037181, 0.010994, 0.114442]
initial_rate_pred pooled / complete / missing RMSE [0.028674, 0.00865, 0.088102]
alpha_m97_c03_batch_01_solver.py seconds 0.49
estimated_taus max relative error 0.02901625482507697
hereditary_weights max relative error 0.035554089940209366
regime_mode_weights max relative error 0.0068534496828089875
predicted_strain pooled / complete / missing RMSE [0.037149, 0.010857, 0.114457]
initial_rate_pred pooled / complete / missing RMSE [0.028576, 0.00853, 0.087885]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fb51bd53db928b1a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_fb51bd53 (directory).
```

### Turn 15 — `submit_completed_task` (`15296 ms` LLM)
- **Rationale**: The final artifact passed isolated-process verification. The optimizer improved numerical invariance and pooled prediction accuracy; the 0.000015 increase in missing-coupon strain RMSE did not outweigh those gains.
- **Alternatives Weighed**: `Keep the previous optimizer` (rejected: Equivalent lab units shifted a fitted time constant by 0.0285; the final fit reduced the shift to 2.42e-10 and improved public time-constant recovery.); `Add pairwise or rational drift terms` (rejected: Neither improved worst held-out-lab strain error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. Public RMSE was 0.0371 for strain and 0.0286 for initial rate. Missing coupon features remained the largest error source. The final solver passed isolated-process schema validation and the lab-unit invariance audit.`

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
