# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_w01_batch_01__tracer_survey_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Tracer Reaction-Transport & Continual Sensor Tomography Stream` |
| **Wave / Wormhole Family** | `beta_w01_step_01` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `434,366` (`418,703` / `116,736` / `15,663`) |
| **Total Episode Wall Time (s)** | `479.36s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_10_calibrated_reaction_transpor`**: The episode refuted two seemingly interchangeable choices: square-root ridge profiling missed the reference observation by 0.282, whereas direct ridge-residual profiling matched it within 4.44e-15, and omitting the boundary inflow diagonal changed the forward matrix by up to 1.04e-3. The validated mechanism combines a harmonic-diffusion, cell-centered upwind stencil with separable shifted solves, then minimizes the squared ridge residual on a capped simplex and certifies the result with a convex dual lower bound. Fit affine calibration independently for each bank when at least two reference levels differ by more than 1e-10 times the bank's maximum absolute reference level; otherwise transfer predictions from independently calibrated banks and require resolved prediction variation and positive raw–prediction covariance before estimating gain. The supplied host probe reported relative field and weight errors of 2e-08 and 5e-08 and a KKT duality gap of 5.1e-09. Limits: sparse-bank transfer depends on the trusted-bank model predicting the missing bank, the episode did not establish cross-epoch persistence or unseen-regime extrapolation, and the bundled synthetic tests were not executed here.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `A square-root ridge profile or an interchangeable upwind discretization reproduces the published system.` (witness: `The square-root profile misses the reference observation by 0.282, while the matched residual projection differs by 4.44e-15; omitting the boundary inflow diagonal changes the forward matrix by up to 1.04e-3.`) -> Pivot `Use the exact verified residual projection and cell-centered upwind stencil.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`13245 ms` LLM)
- **Rationale**: Inspect the validator, public shapes, and archived solver to catch interface or stencil mismatches before reuse; keep the public data in the container.

```bash
cd /holdout_task && cat validate_public_schema.py public_asset_index.json && ls -lh /tmp/hegel_continual_archive/ && sed -n '1,280p' /tmp/hegel_continual_archive/beta_w01_s00_solver.py && python3 - <<'PY'
import pickle
with open('public_beta_w01_batch_01.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:300] if not hasattr(v,'shape') else str(v.flat[:4]))
PY
```
- **Container Exit Code**: `1` (`525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
if g[j] <= g[i] + eps:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            free = np.array([i, j])
        fixed = np.flatnonzero(active)
        count = len(free)
        kkt = np.ones((count + 1, count + 1))
        kkt[:count, :count] = hessian[np.ix_(free, free)]
        kkt[-1, -1] = 0.
        rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        solution = np.linalg.solve(kkt, rhs)
        goal, nu = solution[:-1], float(-solution[-1])
        direction = goal - w[free]
        crossing = np.ones(count)
        lower, upper = goal < -eps, goal > cap + eps
        crossing[lower] = -w[free[lower]] / direction[lower]
        crossing[upper] = (cap - w[free[upper]]) / direction[upper]
        hit = int(np.argmin(crossing))
        if crossing[hit] < 1.:
            w[free] += max(0., crossing[hit]) * direction
            index = free[hit]
            active[index] = -1 if lower[hit] else 1
            w[index] = 0. if lower[hit] else cap
            continue
        w[free] = np.clip(goal, 0., cap)
        violation = active * (hessian @ w + linear - nu)
        index = int(np.argmax(violation))
        if violation[index] <= eps:
            return w, nu
        active[index] = 0
    raise RuntimeError('Box-simplex working set did not converge')


def solve_beta_w01_s00(study: dict) -> dict:
    design, terminal = _responses(study)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrated_observations(study, design, nuisance)
    profiled = _profile(nuisance,
                        np.column_stack((design, y)), float(study['nuisance_ridge']))
    s, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    hessian = s.T @ s + ridge * np.eye(s.shape[1])
    w, nu = _box_simplex(hessian, -(s.T @ r), cap, budget)
    residual = s @ w - r
    g = s.T @ residual + ridge * w
    lower, upper = np.maximum(g - nu, 0.), np.maximum(nu - g, 0.)
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = primal - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': terminal @ w,
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': float(primal), 'dual_lower_bound': float(dual)}}
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`15260 ms` LLM)
- **Rationale**: The public pickle is a list, so inspect individual studies and compare their fields against the archived transport and profiling code. Differences in grid layout or ridge convention would change which routines can be reused.

```bash
cd /holdout_task; cat public_asset_index.json; tail -65 validate_public_schema.py; sed -n '1,155p' /tmp/hegel_continual_archive/beta_w01_s00_solver.py; python3 - <<'PY'
import pickle, numpy as np
p=pickle.load(open('public_beta_w01_batch_01.pkl','rb')); print(type(p).__name__,len(p))
for d in p[:2]:
 print(type(d).__name__)
 for k,v in d.items(): print(k,type(v).__name__,v.shape if hasattr(v,'shape') else '',repr(v)[:220])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`608 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ecution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
        "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
        "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
      },
      "functions": [
        {
          "function": "fit_ridge_nuisance",
          "signature": "fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
        }
      ],
      "index_cards": [
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`16255 ms` LLM)
- **Rationale**: The previous output was truncated. Read just the archived transport and calibration code, the validator entry point, and compact study diagnostics. A nonseparable reaction field would change the transport solver; sparse sweeps would make the archive calibration format relevant.

```bash
cd /holdout_task && sed -n '1,140p' /tmp/hegel_continual_archive/beta_w01_s00_solver.py && tail -42 validate_public_schema.py && python3 - <<'PY'
import pickle,json,numpy as np
p=pickle.load(open('public_beta_w01_batch_01.pkl','rb'));print('studies',len(p));s=p[0]['study'];print('keys/shapes',[(k,getattr(v,'shape',type(v).__name__)) for k,v in s.items()]);print('sweep layout',repr(s['sensor_calibration_sweeps'])[:850]);
for i,x in enumerate(p):
 s=x['study'];V=s['V_reaction_2d'];print(i,'budgets',s['budget_grid'],'separability residual',np.max(np.abs(V-V[:,:1]-V[:1,:]+V[0,0])))
for name in ('archive_manifest','archive_calibration_history'):
 d=json.load(open('/tmp/hegel_continual_archive/'+name+'.json'));print(name,type(d).__name__,str(d)[:1400])
PY
```
- **Container Exit Code**: `0` (`493 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'budget_grid', (3,))]
sweep layout {'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([ 1.1896123 , -0.98209235,  0.91053054, -0.89497536,  1.07058723,
        0.98275033,  0.58502467,  0.15796057, -0.52387114,  0.33412638,
        1.26443253,  1.53042415,  1.05790902, -0.88922764, -0.33819462,
        0.7676414 ,  1.20251432,  0.18352191,  0.06347938,  1.06982141,
        0.51639138,  1.33181595, -0.51581372, -1.13428185, -0.67880119,
       -1.07246521,  1.54527101,  1.22809838, -0.50125809, -0.42140327,
        1.53918959,  0.00705909, -0.15454843,  1.2114941 , -1.1530756 ,
        1.0281231 ,  0.55592031,  1.39091745,  0.20735857, -0.90388505,
        0.0753373 , -0.05626123,  1.14546704,  0.3225242 ,  1.20629427,
        0
0 budgets [1.4 2.6 3.8] separability residual 2.220446049250313e-16
archive_manifest list [{'episode_short_name': 'beta_w01_s00', 'task_id': 'beta_w01_s00__tracer_tomography_baseline', 'entrypoint': 'solve_beta_w01_s00', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/beta_w01_s00_solver.py'}]
archive_calibration_history list [{'episode_short_name': 'beta_w01_s00', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [-0.394794, 0.749202, -0.92172, -0.573561, 0.911677, 1.071939, 0.763828, -0.538027, -0.987248, 0.704763, 0.552736, 0.697699, -1.146367, -0.82039, -0.43132, 0.103308, -0.753514, 0.490399, 0.285573, 1.446642, -0.419722, 0.099837, -0.710198, 0.603609, -0.325496, 0.916142, 1.559094, 1.230692, 1.57791, -0.269765, 0.584184, 1.488076, 0.281891, -0.970375, 0.213444, 1.543352, 0.441655, -0.099229, 0.290376, 1.572457, 0.273822, -1.0943, -0.328338, -1.130808, 0.29169, -0.829249, 0.993043, -0.623942], 'raw_readout': [-0.394794, 0.749202, -0.92172, -0.573561, 0.911677, 1.071939, 0.763828, -0.538027, -0.987248, 0.704763, 0.552736, 0.697699, -1.146367, -0.82039, -0.43132, 0.103308, -0.753514, 0.490399, 0.285573, 1.446642, -0.419722, 0.099837, -0.710198, 0.603609, -0.325496, 0.916142, 1.559094, 1.230692, 1.57791, -0.269766, 0.584184, 1.488076, 0.281891, -0.970375, 0.213444, 1.543352, 0.441655, -0.099229, 0.290376, 1.572457, 0.273822, -1.0943, -0.328338, -1.130808, 0.29169, -0.829249, 0.993043, -0.623942]}]
```

### Turn 4 — `execute_in_remote_docker` (`33722 ms` LLM)
- **Rationale**: Implement the three-budget interface around the archived stencil and active-set routines. The isolated public validator will identify any interface, certificate, or response-convention mismatch that requires a targeted fix.
- **Alternatives Weighed**: `Run the archived solver separately for each budget` (rejected: The response matrix and nuisance projection are shared; rebuilding them would repeat the transport solves.)

```bash
set -e
cd /holdout_task
archive=/tmp/hegel_continual_archive/beta_w01_s00_solver.py
sed -n '/^def _calibrated_observations/,/^def _box_simplex/p' "$archive"
sed '/^def solve_beta_w01_s00/,$d' "$archive" > beta_w01_batch_01_solver.py
cat >> beta_w01_batch_01_solver.py <<'PY'
def solve_beta_w01_batch_01(study: dict) -> dict:
    """Recover one set of source weights and certificate per campaign budget."""
    budgets = np.asarray(study['budget_grid'], dtype=float)
    # The sparse-calibration routine uses one nominal source budget.
    campaign = dict(study, total_budget=float(budgets[1]))
    design, terminal = _responses(campaign)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    y = _calibrated_observations(campaign, design, nuisance)
    profiled = _profile(nuisance, np.column_stack((design, y)),
                        float(study['nuisance_ridge']))
    s, r = profiled[:, :-1], profiled[:, -1]
    ridge, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = s.T @ s + ridge * np.eye(s.shape[1])
    linear = -(s.T @ r)
    weights, certs = [], []
    for budget in budgets:
        w, nu = _box_simplex(hessian, linear, cap, float(budget))
        residual = s @ w - r
        g = s.T @ residual + ridge * w
        lower = np.maximum(g - nu, 0.)
        upper = np.maximum(nu - g, 0.)
        primal = 0.5 * (residual @ residual + ridge * (w @ w))
        dual = primal - g @ w + nu * budget - cap * upper.sum()
        weights.append(w)
        certs.append({'weights': w.copy(), 'nu_sum': nu,
                      'lam_lower': lower, 'lam_upper': upper,
                      'primal_objective': float(primal),
                      'dual_lower_bound': float(dual)})
    return {'weights_by_budget': np.stack(weights),
            'terminal_field_mid_budget': (terminal @ weights[1]).reshape(study['nx'], study['ny']),
            'kkt_certificates': certs}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
d, r = p[:, :-1], p[:, -1]
    h = d.T @ d + float(study['source_ridge']) * np.eye(d.shape[1])
    w, _ = _box_simplex(h, -(d.T @ r), float(study['u_box']), float(study['total_budget']))
    coeff = np.linalg.lstsq(z, y[keep] - design[keep] @ w, rcond=None)[0]
    prediction = design @ w + nuisance @ coeff
    for bank, (x, r) in sparse.items():
        take = rows == bank
        pred, measurement = prediction[take], observed[take]
        if x.size:
            anchor, raw_anchor = np.median(x), np.median(r)
        else:
            anchor, raw_anchor = pred.mean(), measurement.mean()
        pred = pred - anchor
        measurement = measurement - raw_anchor
        energy = pred @ pred
        cross = pred @ measurement
        if energy > 0. and cross > 0.:
            y[take] = anchor + measurement * (energy / cross)
    return y


def _responses(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'],
                   study['reaction_x'] + v.mean(axis=1))
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'],
                   study['reaction_y'] + v.mean(axis=0) - v.mean())
    operator = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    ident = eye(nx * ny, format='csc')
    rhs = sources.reshape(m, -1).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    sensor = ix * ny + iy
    rows = []
    for shift in study['mu_shifts']:
        fields = splu(operator + float(shift) * ident).solve(rhs)
        rows.append(fields[sensor])
    return np.vstack(rows), fields.reshape(nx, ny, m)


def _profile(z, values, ridge):
    if z.shape[1] == 0:
        return values
    q, singular, _ = np.linalg.svd(z, full_matrices=False)
    if ridge == 0:
        q = q[:, singular > np.finfo(float).eps * max(z.shape) * singular[0]]
        return values - q @ (q.T @ values)
    shrink = singular**2 / (singular**2 + ridge)
    return values - q @ (shrink[:, None] * (q.T @ values))


def _box_simplex(hessian, linear, cap, budget):
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba3861e39fe628b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba3861e (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27987 ms` LLM)
- **Rationale**: Check all three budgets and the terminal field against the public reference. Then mask sweeps by wing and inject known gain changes and EMI spikes to see whether sparse calibration needs to prefer the historical reference over transport-based extrapolation.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/beta_w01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
vsrc=open('validate_public_schema.py').read(); start=vsrc.index('def verify_box'); end=vsrc.index('\n    # Unconditional',start); print(vsrc[end:end+1250])
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
 s=item['study']; gt=item['gt']; S=gt['S_tilde']; r=gt['r_y']; lm=s['source_ridge']; cap=s['u_box']; rows=s['row_bank_id']; sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; d,t=m._responses(s); y=m._calibrated_observations(s,d,s['Z_nuisance']); print('bank affine fits', [m._fit_bank(sw['ref_signal'][bank==b],sw['raw_readout'][bank==b]) for b in range(4)]); print('design/target max mismatch',np.max(np.abs(m._profile(s['Z_nuisance'],d,s['nuisance_ridge'])-S)),np.max(np.abs(m._profile(s['Z_nuisance'],y[:,None],s['nuisance_ridge']).ravel()-r)))
 def audit(name, st):
  out=m.solve_beta_w01_batch_01(st); w=out['weights_by_budget']; gaps=[]; objectives=[]
  for wi, b, c in zip(w,s['budget_grid'],out['kkt_certificates']):
   rr=S@wi-r; g=S.T@rr+lm*wi; nu=c['nu_sum']; gaps.append(float(g@wi-nu*b-cap*np.minimum(0,g-nu).sum())); objectives.append(float(.5*(rr@rr+lm*(wi@wi))-c['primal_objective']))
  e=out['terminal_field_mid_budget']-gt['terminal_field_mid_budget']; print(name,'gaps',gaps,'objective discrepancy',max(map(abs,objectives)),'weight_linf',np.max(np.abs(w-gt['weights_by_budget'])),'field_linf',np.max(np.abs(e)))
 audit('public',s)
 gains=np.array([1.06,.96,1.18,.85]); offsets=np.array([.02,-.03,.06,-.04]); changed=copy.deepcopy(s); changed['y_obs_raw']=gains[rows]*y+offsets[rows]; changed['sensor_calibration_sweeps']['raw_readout']=gains[bank]*sw['ref_signal']+offsets[bank]; audit('full affine change',changed)
 for b in range(4): changed['sensor_calibration_sweeps']['raw_readout'][np.flatnonzero(bank==b)[0]]+=.75
 audit('one EMI spike per bank',changed)
 for wing in (0,1):
  for mode in ('unchanged','shifted'):
   trial=copy.deepcopy(s if mode=='unchanged' else changed); take=bank//2!=wing
   for b in (2*wing,2*wing+1): take[np.flatnonzero(bank==b)[1]]=True
   trial['sensor_calibration_sweeps']={k:v[take] for k,v in trial['sensor_calibration_sweeps'].items()}; audit(f'one spot per bank wing {wing}, {mode}',trial)
PY
```
- **Container Exit Code**: `0` (`615 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1e-9 or gap > gap_tol:
        return {"valid": False, "reason": f"duality_gap_too_large:{gap:.3e}", "gap": float(gap)}
    return {"valid": True, "reason": "ok", "true_primal": float(j_true), "verified_dual_lb": float(l_kkt), "gap": float(max(0.0, gap))}

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_w01_batch_01_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importl
bank affine fits [(1.0000000026487041, 2.5938503652120043e-09), (1.000000007677515, -3.989360840264717e-09), (0.9999999978830053, -3.638942598007235e-09), (0.9999999969221983, 7.960834364272503e-10)]
design/target max mismatch 7.476658181460039e-16 2.4594292824531294e-08
public gaps [1.4329757753658856e-09, 2.910734653305269e-09, 4.3884938044636e-09] objective discrepancy 2.8101995452889383e-09 weight_linf 1.042398428685587e-08 field_linf 2.537707353283958e-10
full affine change gaps [1.4329757528144804e-09, 2.910734587385777e-09, 4.388493721196873e-09] objective discrepancy 2.8101994620222115e-09 weight_linf 1.0423984397878172e-08 field_linf 2.5377073706311926e-10
one EMI spike per bank gaps [1.4329757701617152e-09, 2.9107346394274813e-09, 4.388493762830237e-09] objective discrepancy 2.81019972570018e-09 weight_linf 1.0423984397878172e-08 field_linf 2.5377074139992795e-10
one spot per bank wing 0, unchanged gaps [0.005409101290550956, 0.004977440599307549, 0.00454577990806438] objective discrepancy 0.0004775470004267715 weight_linf 0.005111310816952125 field_linf 0.00013402043541781844
one spot per bank wing 0, shifted gaps [0.005409126620715035, 0.004977463906424481, 0.0045458011921343106] objective discrepancy 0.0004775480920249549 weight_linf 0.005111323168019599 field_linf 0.0001340209062149489
one spot per bank wing 1, unchanged gaps [0.00020889827780885438, 0.0002151155396931722, 0.00022133280157774632] objective discrepancy 0.0001107604267744089 weight_linf 0.0016756435132106173 field_linf 3.328800199834184e-05
one spot per bank wing 1, shifted gaps [0.0002088959961195052, 0.0002151141355402628, 0.00022133227496123897] objective discrepancy 0.00011076265286212583 weight_linf 0.0016756574686923875 field_linf 3.32882300088759e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba3861e39fe628b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba3861e (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31133 ms` LLM)
- **Rationale**: First measure observation-model residuals and joint identifiability on the public wing jackknifes. If the joint fit is ill-conditioned or model residuals bias the gains, favor the archival prior rather than promoting the unconstrained fit.
- **Alternatives Weighed**: `Always use archival gains for spot-only banks` (rejected: This handles unchanged gains but cannot recover a gain change from a single spot check without information from the observation model.)
- **ATMS NOGOOD DECLARED**: Refuted `A ridge-regularized fit on the other wing predicts observation levels accurately enough to calibrate a wing with one spot check per bank.` | Witness: `Masking the measured public sweeps to one spot per Wing A bank raises the worst independently evaluated KKT gap to 5.41e-3 without changing the gains, exceeding the 2e-4 public tolerance.` | Blanket Diagnosis: `The sequential fit treats estimated source and nuisance contributions as known, so their prediction error can be absorbed into the inferred bank gains.` | Sublated Pivot: `Test joint estimation using spot checks to anchor the inverse affine map, with archival calibration as a fallback when the data cannot identify gains.`

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','beta_w01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
 s=item['study']; d,_=m._responses(s); z=s['Z_nuisance']; a=np.column_stack((d,z)); rows=s['row_bank_id']; sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; x=sw['ref_signal']; y=m._calibrated_observations(s,d,z)
 fit=np.linalg.lstsq(a,y,rcond=None)[0]; print('physical fit RMSE',np.std(a@fit-y),'weight sum',sum(fit[:d.shape[1]]),'weight range',fit[:d.shape[1]].min(),fit[:d.shape[1]].max())
 for wing in (0,1):
  for gains, offsets in [(np.ones(4),np.zeros(4)),(np.array([1.06,.96,1.18,.85]),np.array([.02,-.03,.06,-.04]))]:
   raw=y*gains[rows]+offsets[rows]; reads=x*gains[bank]+offsets[bank]; sparse=[2*wing,2*wing+1]; base=y.copy(); cols=[]
   for b in sparse:
    k=np.flatnonzero(bank==b)[1]; mask=rows==b; col=np.zeros(len(y)); col[mask]=raw[mask]-reads[k]; cols.append(col); base[mask]=x[k]
   X=np.column_stack((a,-np.column_stack(cols))); coef,_,rank,sv=np.linalg.lstsq(X,base,rcond=None); cal=base+np.column_stack(cols)@coef[-2:]; print('wing',wing,'true gains',gains[sparse],'joint gains',1/coef[-2:],'rank',rank,'min sv',sv[-1],'cal max err',np.max(np.abs(cal-y)),'joint RMSE',np.std(X@coef-base))
 print('public archive-shaped sparse diagnostics done')
PY
```
- **Container Exit Code**: `0` (`469 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
physical fit RMSE 0.010811473033012473 weight sum 2.7687665454345436 weight range -3.9817108216923116 3.960409190364479
wing 0 true gains [1. 1.] joint gains [1.00089889 1.00077976] rank 21 min sv 0.0015916521149504585 cal max err 0.004855862988189408 joint RMSE 0.010793511853518248
wing 0 true gains [1.06 0.96] joint gains [1.06095283 0.96074857] rank 21 min sv 0.0015916521151159806 cal max err 0.004855862988191184 joint RMSE 0.010793511853518244
wing 1 true gains [1. 1.] joint gains [1.00035156 1.00111246] rank 21 min sv 0.0016771726043989673 cal max err 0.005088925550084777 joint RMSE 0.010772451987862471
wing 1 true gains [1.18 0.85] joint gains [1.18041484 0.85094559] rank 21 min sv 0.0016771726044583857 cal max err 0.005088925550081225 joint RMSE 0.010772451987862489
public archive-shaped sparse diagnostics done

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba3861e39fe628b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba3861e (directory).
```

### Turn 7 — `execute_in_remote_docker` (`36015 ms` LLM)
- **Rationale**: Rotate the retained spot checks on each wing and compare unconstrained, box-constrained, and regularized joint fits against archived and known gains. The independent KKT and stationarity margins will determine whether source constraints improve calibration and when the archive is preferable.
- **Alternatives Weighed**: `Adopt unconstrained joint calibration immediately` (rejected: It estimates negative source weights on the public data; the resulting gain error must be compared with physically bounded fits before it is used.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import lsq_linear
sp=importlib.util.spec_from_file_location('solver','beta_w01_batch_01_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
 s,gt=item['study'],item['gt']; d,t=m._responses(s); z=s['Z_nuisance']; a=np.column_stack((d,z)); rows=s['row_bank_id']; sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; x=sw['ref_signal']; y=m._calibrated_observations(s,d,z); M=d.shape[1]; cap=s['u_box']; ridge=s['source_ridge']; print('cap/ridges',cap,ridge,s['nuisance_ridge']); S,r=gt['S_tilde'],gt['r_y']; h=S.T@S+ridge*np.eye(M)
 def score(cal):
  rc=m._profile(z,cal[:,None],s['nuisance_ridge']).ravel(); G=[]; R=[]; W=[]
  for bg in s['budget_grid']:
   w,nu=m._box_simplex(h,-S.T@rc,cap,bg); g=h@w-S.T@r; gc=h@w-S.T@rc; G.append(g@w-nu*bg-cap*np.minimum(g-nu,0).sum()); R.append(max(abs(g-gc))); W.append(w)
  return max(G),max(R),np.max(abs(np.array(W)-gt['weights_by_budget']))
 for wing in (0,1):
  sparse=[2*wing,2*wing+1]; ids=[np.flatnonzero(bank==b) for b in sparse]; vals={k:[] for k in ['archive','joint','box','box_ridge','known']}
  for i in range(min(map(len,ids))):
   gains=np.array([1.06,.96,1.18,.85]); offsets=np.array([.02,-.03,.06,-.04]); raw=gains[rows]*y+offsets[rows]; reads=gains[bank]*x+offsets[bank]; base=y.copy(); cols=np.zeros((len(y),2));
   for j,b in enumerate(sparse):
    k=ids[j][i]; take=rows==b; cols[take,j]=raw[take]-reads[k]; base[take]=x[k]
   X=np.column_stack((a,-cols)); lo=np.r_[np.zeros(M),np.full(z.shape[1],-np.inf),[0,0]]; hi=np.r_[np.full(M,cap),np.full(z.shape[1]+2,np.inf)]; beta=np.linalg.lstsq(X,base,rcond=None)[0]; box=lsq_linear(X,base,bounds=(lo,hi),tol=1e-12).x; reg=np.zeros((M,X.shape[1])); reg[:,:M]=np.sqrt(ridge)*np.eye(M); box_reg=lsq_linear(np.vstack((X,reg)),np.r_[base,np.zeros(M)],bounds=(lo,hi),tol=1e-12).x
   for k,inv in [('archive',np.ones(2)),('joint',beta[-2:]),('box',box[-2:]),('box_ridge',box_reg[-2:]),('known',1/gains[sparse])]: vals[k].append(score(base+cols@inv))
  print('wing',wing,'worst across spot locations (gap, stationarity, weights)',{k:np.max(v,axis=0).tolist() for k,v in vals.items()}); print('wing',wing,'median across spot locations',{k:np.median(v,axis=0).tolist() for k,v in vals.items()})
PY
```
- **Container Exit Code**: `0` (`670 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cap/ridges 1.0 0.04 0.65
wing 0 worst across spot locations (gap, stationarity, weights) {'archive': [0.5588976968070156, 0.07663973528190242, 0.564330896182329], 'joint': [0.00843742842232627, 0.0010698468297721947, 0.007188769148293195], 'box': [0.0038497238396683012, 0.0009453973568685319, 0.00837302725994363], 'box_ridge': [0.027255423065763562, 0.0035065303204955545, 0.023114906576746523], 'known': [4.388493943241478e-09, 1.7572020921674536e-09, 1.0423980401075283e-08]}
wing 0 median across spot locations {'archive': [0.11816411675494323, 0.028576262698283433, 0.2787302305731247], 'joint': [0.0017311937764657448, 0.0005747148192661901, 0.0053025634692556115], 'box': [0.0021482120569224702, 0.0005483530839234356, 0.005628233515701214], 'box_ridge': [0.013587266018952755, 0.0018224582213078825, 0.011500099573486508], 'known': [4.38849364486904e-09, 1.7572020496667284e-09, 1.042397998474165e-08]}
wing 1 worst across spot locations (gap, stationarity, weights) {'archive': [1.3168731491242185, 0.16442684068006128, 0.5652155357027633], 'joint': [0.007126936657479661, 0.0007025009566606136, 0.003968872565783904], 'box': [0.007196221020129356, 0.0007098312003716578, 0.0032825532620835984], 'box_ridge': [0.022249352903212616, 0.002147147728031807, 0.007275917851495062], 'known': [4.3884937767080245e-09, 1.757202097371624e-09, 1.0423980345564132e-08]}
wing 1 median across spot locations {'archive': [0.30748715176925123, 0.07169478389696468, 0.3352177175685885], 'joint': [0.0015326526662286372, 0.0003085774790785754, 0.0018133305651915725], 'box': [0.0018849161969681444, 0.0003812874920551372, 0.0019624889328610645], 'box_ridge': [0.009370526183061329, 0.0010768998077039724, 0.005249776185153828], 'known': [4.3884935685412074e-09, 1.7572020401257493e-09, 1.0423980026375013e-08]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba3861e39fe628b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba3861e (directory).
```

### Turn 8 — `execute_in_remote_docker` (`20267 ms` LLM)
- **Rationale**: Inspect historical sweep fits by bank and their metadata, without dumping the arrays. Whether the archive contains matching or incompatible calibration versions determines when to reuse archived gains and when to estimate a drift from the current observations.
- **Alternatives Weighed**: `Promote box-constrained joint estimation as the default sparse calibration` (rejected: Its worst gap is still 7.20e-3. One spot check constrains gain and offset together, and measurement noise limits the precision recoverable from transport observations alone.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import json,numpy as np
from pathlib import Path
root=Path('/tmp/hegel_continual_archive')
for name in ('archive_manifest.json','archive_calibration_history.json','pre_llm_gate_diagnostic.json'):
 obj=json.loads((root/name).read_text())
 def summarize(x,level=0):
  if isinstance(x,dict):
   if {'bank_id','ref_signal','raw_readout'}<=x.keys():
    b=np.asarray(x['bank_id']); r=np.asarray(x['ref_signal']); y=np.asarray(x['raw_readout']); fits={}
    for i in np.unique(b):
     take=b==i; a=r[take]; v=y[take]; fits[str(i)]=(len(a),float(np.ptp(a)),np.polyfit(a,v,1).tolist() if len(a)>1 and np.ptp(a)>0 else (a.tolist(),v.tolist()))
    return {'sweep_fits':fits,'metadata':{k:summarize(v,level+1) for k,v in x.items() if k not in ('bank_id','ref_signal','raw_readout')}}
   return {k:summarize(v,level+1) for k,v in x.items()} if level<6 else list(x)
  if isinstance(x,list): return [summarize(v,level+1) for v in x[:8]]+([f'{len(x)} total entries'] if len(x)>8 else [])
  return x
 print(name,json.dumps(summarize(obj))[:3200])
PY
sed -n '/^def _fit_bank/,/^def _calibrated_observations/p' beta_w01_batch_01_solver.py
```
- **Container Exit Code**: `0` (`504 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sweep_fits": {"0": [12, 2.059187, [1.0000000000000004, -1.531327719511087e-16]], "1": [12, 2.593009, [1.0000000000000002, 9.242827980962757e-17]], "2": [12, 2.548285, [1.0000001104493583, -1.5539370678733572e-07]], "3": [12, 2.703265, [1.0, -5.0482089962061506e-17]]}, "metadata": {"episode_short_name": "beta_w01_s00"}}]
pre_llm_gate_diagnostic.json {"arm": "Full_Mechanism_Graph", "pre_llm_reason": "interface_adaptation_required", "parameter_payload": {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0], "wing_versions": {"wing_a": 1, "wing_b": 1}}, "guard_evaluations": [{"binding_id": "bind_beta_w01_c0", "branch_tag": "C0_base", "parameter_version": 1, "guard_result": {"status": "TRUE", "state": "TRUE", "measured_value": 2.0434255615483102e-16, "statistic": 2.0434255615483102e-16, "observable_kind": "nonseparable_2d_additive_residual_ratio", "units": "dimensionless", "normalization": "identity", "applies_to_component_id": "separable_kronecker_transport", "guard_spec": {"applies_to_component_id": "separable_kronecker_transport", "field": "nonseparable_2d_additive_residual_ratio", "guard_id": "guard_beta_w01_c0", "normalization": "identity", "observable_kind": "nonseparable_2d_additive_residual_ratio", "op": "<=", "units": "dimensionless", "value": 1e-06}, "guard_receipt_sha256": "be4527a465bfb445b9c6afa1"}, "schema_compatible": false}]}
def _fit_bank(x, y, prior=(1., 0.)):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    independent = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x), initial=0.))
    if not independent.any():
        gain, offset = prior
        return gain, float(np.median(y - gain * x)) if len(x) else offset
    gain = float(np.median((y[j[independent]] - y[i[independent]]) / dx[independent]))
    residual = y - gain * x
    offset = float(np.median(residual))
    deviation = np.abs(residual - offset)
    scale = 1.4826 * np.median(deviation)
    keep = deviation <= max(4.5 * scale, 1e-10)
    if np.ptp(x[keep]) > 0 if keep.any() else False:
        gain, offset = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
    return (gain, offset) if gain > 0 else prior


def _calibrated_observations(study, design, nuisance):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba3861e39fe628b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba3861e (directory).
```

### Turn 9 — `execute_in_remote_docker` (`41448 ms` LLM)
- **Rationale**: Historical sweeps support near-identity calibration on both wings. Retain it when spot checks agree, and require evidence from the observations before changing it when a spot disagrees. The isolated validator checks for regressions on the public campaign; wing jackknifes are still needed to verify the sparse path.
- **Alternatives Weighed**: `Fit new gains whenever sweeps are sparse` (rejected: The public joint fits absorbed observation noise into the gains even when the true calibration was unchanged.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_w01_batch_01_solver.py')
s = p.read_text().replace('import numpy as np', 'import json\nfrom pathlib import Path\n\nimport numpy as np', 1)
a, b = s.index('def _calibrated_observations('), s.index('def _profile(')
s = s[:a] + '''def _historical_bank_fits():
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    fits = [(1., 0.)] * 4
    if path.is_file():
        for sweep in json.loads(path.read_text()):
            banks = np.asarray(sweep['bank_id'])
            ref = np.asarray(sweep['ref_signal'], dtype=float)
            raw = np.asarray(sweep['raw_readout'], dtype=float)
            for bank in range(4):
                use = (banks == bank) & np.isfinite(ref) & np.isfinite(raw)
                if use.any() and np.ptp(ref[use]) > 1e-10:
                    fits[bank] = _fit_bank(ref[use], raw[use], fits[bank])
    return fits


def _calibrated_observations(study, design, nuisance):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    rows = np.asarray(study['row_bank_id'], dtype=int).ravel()
    observed = np.asarray(study['y_obs_raw'], dtype=float).ravel()
    y = observed.copy()
    sparse = []
    priors = None
    for bank in range(4):
        use = (banks == bank) & np.isfinite(ref) & np.isfinite(raw)
        x, r = ref[use], raw[use]
        take = rows == bank
        if len(x) >= 2 and np.ptp(x) > 1e-10:
            gain, offset = _fit_bank(x, r)
        else:
            if priors is None:
                priors = _historical_bank_fits()
            gain, offset = priors[bank]
            if take.any() and len(x):
                xb, rb = float(np.median(x)), float(np.median(r))
                # A compatible spot verifies continuity but does not identify a new slope.
                if abs(rb - (gain * xb + offset)) > 2e-6:
                    sparse.append((take, xb, rb, gain))
        y[take] = (observed[take] - offset) / gain
    if not sparse:
        return y

    # For disagreeing spots, remove the shared source and nuisance subspace
    # before deciding whether the archived calibration needs to change.
    physics = np.column_stack((design, nuisance))
    q, singular, _ = np.linalg.svd(physics, full_matrices=False)
    q = q[:, singular > np.finfo(float).eps * max(physics.shape) * singular[0]]

    def project(values):
        return values - q @ (q.T @ values)

    base = y.copy()
    delta = np.zeros((len(y), len(sparse)))
    for j, (take, xb, rb, gain) in enumerate(sparse):
        base[take] = xb + (observed[take] - rb) / gain
        delta[take, j] = observed[take] - rb
    unchanged_residual = project(y)
    anchored_residual = project(base)
    projected_delta = project(delta)
    correction, _, rank, _ = np.linalg.lstsq(projected_delta, -anchored_residual, rcond=None)
    # Require improvement beyond what a single extra coefficient would fit from noise.
    dof = max(1, len(y) - q.shape[1] - rank)
    corrected_residual = anchored_residual + projected_delta @ correction
    loss = corrected_residual @ corrected_residual
    threshold = 9 * loss / dof
    if rank == len(sparse) and np.all(correction + [1 / row[3] for row in sparse] > 0):
        if min(unchanged_residual @ unchanged_residual, anchored_residual @ anchored_residual) - loss > threshold * rank:
            return base + delta @ correction
    if unchanged_residual @ unchanged_residual - anchored_residual @ anchored_residual > threshold * len(sparse):
        return base
    return y


''' + s[b:]
s = s.replace("    # The sparse-calibration routine uses one nominal source budget.\n    campaign = dict(study, total_budget=float(budgets[1]))\n    design, terminal = _responses(campaign)", "    design, terminal = _responses(study)").replace('_calibrated_observations(campaign, design, nuisance)', '_calibrated_observations(study, design, nuisance)')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`413 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b2ed23e57b89941b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b2ed23e5 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 64, in <module>
    out = mod.solve_beta_w01_batch_01(st)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/beta_w01_batch_01_solver.py", line 178, in solve_beta_w01_batch_01
    design, terminal = _responses(study)
                       ^^^^^^^^^^
NameError: name '_responses' is not defined
```

### Turn 10 — `execute_in_remote_docker` (`34391 ms` LLM)
- **Rationale**: The calibration edit accidentally removed the archived response builder. Restore it without changing the stencil, then verify the public interface and rotate sparse spot checks on both wings to distinguish a fixed continuity case from the remaining gain-drift limitation.

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
import ast
from pathlib import Path
p = Path('beta_w01_batch_01_solver.py')
s = p.read_text()
archive = Path('/tmp/hegel_continual_archive/beta_w01_s00_solver.py').read_text()
node = next(n for n in ast.parse(archive).body if isinstance(n, ast.FunctionDef) and n.name == '_responses')
assert not any(isinstance(n, ast.FunctionDef) and n.name == '_responses' for n in ast.parse(s).body)
p.write_text(s.replace('def _profile(', ast.get_source_segment(archive, node) + '\n\ndef _profile(', 1))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']
    sw = s['sensor_calibration_sweeps']; bank = sw['bank_id']; rows = s['row_bank_id']; ref = sw['ref_signal']
    d, t = m._responses(s); z = s['Z_nuisance']; ridge = s['source_ridge']; cap = s['u_box']; S, r = gt['S_tilde'], gt['r_y']
    y = m._calibrated_observations(s,d,z)
    def audit(trial):
        yc = m._calibrated_observations(trial,d,z)
        sc = m._profile(z,np.column_stack((d,yc)),s['nuisance_ridge']); dc, rc = sc[:,:-1], sc[:,-1]
        h = dc.T @ dc + ridge*np.eye(dc.shape[1]); q = -dc.T @ rc
        gaps=[]; stat=[]; obj=[]; weights=[]
        for b in s['budget_grid']:
            w,nu = m._box_simplex(h,q,cap,b); weights.append(w)
            rr = S@w-r; rr_cal = dc@w-rc; g = S.T@rr+ridge*w; gc = dc.T@rr_cal+ridge*w
            gaps.append(g@w-nu*b-cap*np.minimum(g-nu,0).sum()); stat.append(np.max(np.abs(g-gc))); obj.append(abs(.5*(rr@rr-rr_cal@rr_cal)))
        w = np.stack(weights)
        return np.array([max(gaps),max(stat),max(obj),np.max(np.abs(w-gt['weights_by_budget'])),np.max(np.abs((t@w[1]).reshape(s['nx'],s['ny'])-gt['terminal_field_mid_budget']))])
    print('metric order: KKT gap, stationarity error, objective error, weights Linf, field Linf')
    print('public',audit(s))
    gains=np.array([1.06,.96,1.18,.85]); offsets=np.array([.02,-.03,.06,-.04]); full=copy.deepcopy(s)
    full['y_obs_raw']=gains[rows]*y+offsets[rows]; full['sensor_calibration_sweeps']['raw_readout']=gains[bank]*ref+offsets[bank]
    emi=copy.deepcopy(full)
    for b in range(4): emi['sensor_calibration_sweeps']['raw_readout'][np.flatnonzero(bank==b)[0]]+=.75
    print('full affine plus EMI',audit(emi))
    for wing in (0,1):
        ids=[np.flatnonzero(bank==b) for b in (2*wing,2*wing+1)]
        for mode, source in [('unchanged',s),('shifted',full)]:
            vals=[]
            for i in range(max(map(len,ids))):
                take=(bank//2!=wing); take[[v[i%len(v)] for v in ids]]=True
                trial=dict(source,sensor_calibration_sweeps={k:v[take] for k,v in source['sensor_calibration_sweeps'].items()})
                vals.append(audit(trial))
            worst=np.max(vals,axis=0); print('wing',wing,mode,worst)
            if mode=='unchanged': assert worst[0]<2e-4 and worst[1]<1e-4 and worst[2]<1e-6
PY
```
- **Container Exit Code**: `0` (`665 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
metric order: KKT gap, stationarity error, objective error, weights Linf, field Linf
public [4.38849380e-09 1.75720241e-09 2.81019955e-09 1.04239843e-08
 2.53770735e-10]
full affine plus EMI [4.38849376e-09 1.75720240e-09 2.81019973e-09 1.04239844e-08
 2.53770741e-10]
wing 0 unchanged [3.02464891e-09 1.23485770e-09 1.72530035e-09 9.51566155e-09
 1.16102012e-10]
wing 0 shifted [0.00843743 0.00106985 0.00069009 0.00718877 0.00017993]
wing 1 unchanged [2.38198625e-09 9.25227006e-10 1.08489789e-09 1.52749082e-08
 3.33392243e-10]
wing 1 shifted [7.12693666e-03 7.02500957e-04 8.69615857e-04 3.96887257e-03
 9.32161523e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=626f38f71520ec71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_626f38f7 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`27925 ms` LLM)
- **Rationale**: With three reference points, a spike at either end can affect two of the three pairwise slopes. Check whether that failure changes the certificates before adding archive support for small sweeps.
- **Alternatives Weighed**: `Keep tuning gain estimates for a shifted bank with a single spot` (rejected: A single reference cannot distinguish gain from offset; observation noise already limits the joint estimate.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','beta_w01_batch_01_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
 s,gt=item['study'],item['gt']; d,_=m._responses(s); z=s['Z_nuisance']; sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; x=sw['ref_signal']; rows=s['row_bank_id']; y=m._calibrated_observations(s,d,z); S,r=gt['S_tilde'],gt['r_y']; cap=s['u_box']; ridge=s['source_ridge']; h=S.T@S+ridge*np.eye(S.shape[1])
 for mode in ('unchanged','shifted'):
  gains,offsets=(np.ones(4),np.zeros(4)) if mode=='unchanged' else (np.array([1.06,.96,1.18,.85]),np.array([.02,-.03,.06,-.04])); worst=np.zeros(3)
  for b in range(4):
   ids=np.flatnonzero(bank==b); ids=ids[np.argsort(x[ids])]; picks=ids[[0,len(ids)//2,-1]]; take=(bank!=b); take[picks]=True
   for spike in [-1,*picks]:
    yy=gains[bank]*x+offsets[bank];
    if spike>=0: yy[spike]+=.75
    st=dict(s,y_obs_raw=gains[rows]*y+offsets[rows],sensor_calibration_sweeps=dict(bank_id=bank[take],ref_signal=x[take],raw_readout=yy[take])); yc=m._calibrated_observations(st,d,z); rc=m._profile(z,yc[:,None],s['nuisance_ridge']).ravel()
    for bg in s['budget_grid']:
     w,nu=m._box_simplex(h,-S.T@rc,cap,bg); rr=S@w-r; g=S.T@rr+ridge*w; worst=np.maximum(worst,[g@w-nu*bg-cap*np.minimum(g-nu,0).sum(),np.max(np.abs(S.T@(rc-r))),abs(.5*(rr@rr-(S@w-rc)@(S@w-rc)))])
  print(mode,'three spots, at most one EMI; worst gap, stationarity, objective error:',worst)
PY
```
- **Container Exit Code**: `0` (`494 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
unchanged three spots, at most one EMI; worst gap, stationarity, objective error: [ 2.2461357   0.36674504 36.29927433]
shifted three spots, at most one EMI; worst gap, stationarity, objective error: [ 2.4346955   0.403299   60.09292362]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=626f38f71520ec71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_626f38f7 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`48030 ms` LLM)
- **Rationale**: Test whether the clean affine line is recoverable from the remaining pairs. Check both wings with clean and corrupted short sweeps, and retain the public and single-spot comparisons to detect regressions.
- **Alternatives Weighed**: `Use historical calibration for every inconsistent short sweep` (rejected: It would discard genuine gain changes even when two clean current references determine the updated affine map.)
- **ATMS NOGOOD DECLARED**: Refuted `The median of pairwise slopes is enough to reject one EMI spike in a three-point sweep.` | Witness: `Retaining three public reference levels and injecting one spike raised the worst KKT gap to 2.25 with unchanged calibration and 2.43 with shifted calibration.` | Blanket Diagnosis: `One corrupted point affects two of the three slopes, so the median can discard the only clean pair.` | Sublated Pivot: `Keep the candidate affine lines and use the residual outside the shared source and nuisance subspace to select among them.`

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
from pathlib import Path
p = Path('beta_w01_batch_01_solver.py')
s = p.read_text().replace('import json\n', 'import json\nfrom itertools import product\n', 1)
s = s.replace('    sparse = []\n    priors = None', '    sparse = []\n    short_sweeps = []\n    priors = None', 1)
s = s.replace('            gain, offset = _fit_bank(x, r)\n        else:', '''            gain, offset = _fit_bank(x, r)
            if take.any() and 3 <= len(x) <= 4 and np.ptp(r - gain * x) > 4e-6:
                # With one spike, at least one of these pairs still gives the
                # clean affine line. The short sweep alone cannot select it.
                i, j = np.triu_indices(len(x), 1)
                dx, dr = x[j] - x[i], r[j] - r[i]
                good = (np.abs(dx) > 1e-10) & (dx * dr > 0)
                slopes = dr[good] / dx[good]
                intercepts = r[i[good]] - slopes * x[i[good]]
                if slopes.size:
                    short_sweeps.append((take, slopes, intercepts))
        else:''', 1)
s = s.replace('    if not sparse:\n        return y', '    if not sparse and not short_sweeps:\n        return y', 1)
s = s.replace('    base = y.copy()\n    delta', '''    if short_sweeps:
        baseline = y.copy()
        shifts = []
        for take, slopes, intercepts in short_sweeps:
            corrected = (observed[take, None] - intercepts) / slopes
            shift = np.zeros((len(y), len(slopes)))
            shift[take] = corrected - baseline[take, None]
            shifts.append(shift)
        residual = project(baseline)
        projected = [project(shift).T for shift in shifts]
        best_loss = np.inf
        best = None
        for indices in product(*(range(len(group)) for group in projected)):
            trial = residual + sum(group[i] for group, i in zip(projected, indices))
            loss = trial @ trial
            if loss < best_loss:
                best_loss, best = loss, indices
        y = baseline + sum(group[:, i] for group, i in zip(shifts, best))
    if not sparse:
        return y

    base = y.copy()
    delta''', 1)
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/beta_short_sweep_audit.py <<'PY'
import importlib.util, pickle
import numpy as np
sp = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for item in pickle.load(open('/holdout_task/public_beta_w01_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']; d,t = m._responses(s); z=s['Z_nuisance']
    sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; x=sw['ref_signal']; rows=s['row_bank_id']; y=m._calibrated_observations(s,d,z)
    S,r=gt['S_tilde'],gt['r_y']; cap=s['u_box']; ridge=s['source_ridge']; h=S.T@S+ridge*np.eye(S.shape[1])
    def audit(trial):
        rc=m._profile(z,m._calibrated_observations(trial,d,z)[:,None],s['nuisance_ridge']).ravel()
        result=[]
        for i,b in enumerate(s['budget_grid']):
            w,nu=m._box_simplex(h,-S.T@rc,cap,b); rr=S@w-r; g=S.T@rr+ridge*w
            result.append([g@w-nu*b-cap*np.minimum(g-nu,0).sum(), np.max(np.abs(S.T@(rc-r))), abs(.5*(rr@rr-(S@w-rc)@(S@w-rc))), np.max(np.abs(w-gt['weights_by_budget'][i])), np.max(np.abs((t@w).reshape(s['nx'],s['ny'])-gt['terminal_field_mid_budget'])) if i==1 else 0.])
        return np.max(result,axis=0)
    print('metrics: KKT gap, stationarity error, objective error, weight Linf, middle field Linf')
    print('public',audit(s))
    for label,gains,offsets in [('unchanged',np.ones(4),np.zeros(4)),('shifted',np.array([1.06,.96,1.18,.85]),np.array([.02,-.03,.06,-.04]))]:
        ids=[np.flatnonzero(bank==b) for b in range(4)]
        ids=[i[np.argsort(x[i])] for i in ids]
        for k in (1,3,4):
            vals=[]
            for wing in (0,1):
                group=ids[2*wing:2*wing+2]
                layouts=([[ind[j%len(ind)] for ind in group] for j in range(max(map(len,group)))] if k==1 else [[ind[np.linspace(0,len(ind)-1,k,dtype=int)] for ind in group]])
                for picks in layouts:
                    pick=np.array(picks).ravel(); take=(bank//2!=wing); take[pick]=True
                    for spike in [-1,*pick] if k>1 else [-1]:
                        yy=gains[bank]*x+offsets[bank];
                        if spike>=0: yy[spike]+=.75
                        trial=dict(s,y_obs_raw=gains[rows]*y+offsets[rows],sensor_calibration_sweeps=dict(bank_id=bank[take],ref_signal=x[take],raw_readout=yy[take])); vals.append(audit(trial))
            print(label,k,'points per sparse bank',np.max(vals,axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/beta_short_sweep_audit.py
```
- **Container Exit Code**: `0` (`827 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
metrics: KKT gap, stationarity error, objective error, weight Linf, middle field Linf
public [4.38849362e-09 1.75720205e-09 2.81020039e-09 1.04239800e-08
 2.53770715e-10]
unchanged 1 points per sparse bank [4.38849360e-09 1.75720205e-09 2.81020023e-09 1.04239801e-08
 2.53770716e-10]
unchanged 3 points per sparse bank [4.38849378e-09 1.75720209e-09 2.81020072e-09 1.04239802e-08
 2.53770720e-10]
unchanged 4 points per sparse bank [4.38849350e-09 1.75720203e-09 2.81020061e-09 1.04239805e-08
 2.53770726e-10]
shifted 1 points per sparse bank [0.00990852 0.00126646 0.00080603 0.00821444 0.00021186]
shifted 3 points per sparse bank [4.38849375e-09 1.75720210e-09 2.81020051e-09 1.04239809e-08
 2.53770736e-10]
shifted 4 points per sparse bank [4.38849400e-09 1.75720212e-09 2.81020072e-09 1.04239811e-08
 2.53770735e-10]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2318f0a373161bff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_2318f0a3 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`34270 ms` LLM)
- **Rationale**: Verify the emitted certificates and middle-budget field through the actual entry point, including simultaneous spikes on both banks of either wing. Test offset-only drift separately: unlike gain-and-offset drift, it is identifiable from one spot if the archived gain still holds, so failure there would justify a targeted change.
- **Alternatives Weighed**: `Tune the joint gain estimate further using synthetic single-spot gain shifts` (rejected: A lone spot cannot independently identify gain and offset, and the previous joint fits miss certificate tolerances even when they reduce weight error.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

solver = load('solver', '/holdout_task/beta_w01_batch_01_solver.py')
validator = load('validator', '/holdout_task/validate_public_schema.py')
for item in pickle.load(open('/holdout_task/public_beta_w01_batch_01.pkl', 'rb')):
    s, gt = item['study'], item['gt']
    sw = s['sensor_calibration_sweeps']; bank = sw['bank_id']; x = sw['ref_signal']; rows = s['row_bank_id']
    design, _ = solver._responses(s)
    y = solver._calibrated_observations(s, design, s['Z_nuisance'])
    ids = [np.flatnonzero(bank == b) for b in range(4)]
    ids = [ind[np.argsort(x[ind])] for ind in ids]
    def audit(st):
        out = solver.solve_beta_w01_batch_01(st)
        w = out['weights_by_budget']; certs = out['kkt_certificates']
        verdicts = [validator.verify_box_simplex_kkt_certificate(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], budget, cert) for budget, cert in zip(s['budget_grid'], certs)]
        assert len(certs) == 3 and all(np.allclose(wi, c['weights'], rtol=0, atol=1e-12) for wi,c in zip(w,certs))
        field = out['terminal_field_mid_budget']; assert field.shape == (s['nx'], s['ny'])
        return [max(v['gap'] for v in verdicts), np.max(np.abs(w - gt['weights_by_budget'])), np.max(np.abs(field - gt['terminal_field_mid_budget']))], [v.get('reason') for v in verdicts if not v['valid']]
    metrics, failures = audit(s)
    print('public (gap, weight Linf, field Linf):', metrics, failures)
    assert not failures
    for label, gains, offsets, count in [('continuity', np.ones(4), np.zeros(4), 1), ('offset-only', np.ones(4), np.array([.02,-.03,.06,-.04]), 1), ('affine + EMI', np.array([1.06,.96,1.18,.85]), np.array([.02,-.03,.06,-.04]), 3)]:
        for wing in (0,1):
            stats = []; reasons = set()
            for frac in (.15,.5,.85):
                group = ids[2*wing:2*wing+2]
                picks = [ind[[0, int(frac*(len(ind)-1)), -1]] if count == 3 else ind[[int(frac*(len(ind)-1))]] for ind in group]
                keep = bank//2 != wing; keep[np.concatenate(picks)] = True
                for k in range(count):
                    reads = gains[bank]*x + offsets[bank]
                    if count == 3:
                        reads[picks[0][k]] += .75
                        reads[picks[1][2-k]] -= .75
                    st = dict(s, y_obs_raw=gains[rows]*y+offsets[rows], sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=x[keep],raw_readout=reads[keep]))
                    result, bad = audit(st); stats.append(result); reasons.update(bad)
            worst = np.max(stats, axis=0)
            print(label, 'wing', wing, 'worst (gap, weight Linf, field Linf):', worst, 'failures:', sorted(reasons))
            if label != 'offset-only':
                assert not reasons and worst[1] < 1e-6 and worst[2] < 1e-6
PY
```
- **Container Exit Code**: `0` (`517 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2318f0a373161bff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_2318f0a3 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`31040 ms` LLM)
- **Rationale**: The previous validator import terminated before the custom audit ran. Run only its independent certificate function so the audit reaches both unresolved slices. If offset-only drift fails, inspect the archive-gain anchoring rule; if simultaneous spikes fail, inspect the affine candidate selection.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import ast, importlib.util, pickle
from pathlib import Path
import numpy as np
p = Path('validate_public_schema.py')
node = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate')
import math
exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'))
sp = importlib.util.spec_from_file_location('solver','beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for item in pickle.load(open('public_beta_w01_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']
    sw = s['sensor_calibration_sweeps']; bank = sw['bank_id']; x = sw['ref_signal']; rows = s['row_bank_id']
    d, _ = m._responses(s); y = m._calibrated_observations(s,d,s['Z_nuisance'])
    def audit(st):
        o=m.solve_beta_w01_batch_01(st); w=o['weights_by_budget']; f=o['terminal_field_mid_budget']; certs=o['kkt_certificates']
        assert w.shape==(3,d.shape[1]) and f.shape==(s['nx'],s['ny']) and len(certs)==3
        result=[]; errors=[]
        for wi,b,c in zip(w,s['budget_grid'],certs):
            assert np.array_equal(wi,c['weights'])
            v=verify_box_simplex_kkt_certificate(gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],b,c)
            if not v['valid']: errors.append(v['reason'])
            rr=gt['S_tilde']@wi-gt['r_y']; g=gt['S_tilde'].T@rr+s['source_ridge']*wi; nu=c['nu_sum']
            result.append(g@wi-nu*b-s['u_box']*np.minimum(0,g-nu).sum())
        return [max(result),np.max(abs(w-gt['weights_by_budget'])),np.max(abs(f-gt['terminal_field_mid_budget']))],errors
    metric, errors = audit(s); print('public gap/weight/field Linf:',metric,errors); assert not errors
    ids=[np.flatnonzero(bank==b) for b in range(4)]; ids=[i[np.argsort(x[i])] for i in ids]
    for mode,n,gain in [('offset-only',1,np.ones(4)),('shifted with dual EMI',3,np.array([1.06,.96,1.18,.85]))]:
        offset=np.array([.02,-.03,.06,-.04]); raw=gain[rows]*y+offset[rows]; true=gain[bank]*x+offset[bank]
        for wing in (0,1):
            metrics=[]; errors=[]
            for frac in (.15,.5,.85):
                picks=[]
                for i in ids[2*wing:2*wing+2]:
                    j=int(frac*(len(i)-1)); picks.append(i[[j] if n==1 else [0,j,-1]])
                keep=bank//2!=wing; keep[np.concatenate(picks)]=True
                for k in range(n):
                    read=true.copy()
                    if n==3:
                        read[picks[0][k]]+=.75; read[picks[1][2-k]]-=.75
                    st=dict(s,y_obs_raw=raw,sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=x[keep],raw_readout=read[keep]))
                    metric,bad=audit(st); metrics.append(metric); errors.extend(bad)
            print(mode,'wing',wing,'worst gap/weight/field Linf:',np.max(metrics,axis=0),'failed certificates:',len(errors),'example:',errors[:1])
PY
```
- **Container Exit Code**: `0` (`1168 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public gap/weight/field Linf: [4.3884938044636e-09, 1.042398428685587e-08, 2.537707353283958e-10] []
offset-only wing 0 worst gap/weight/field Linf: [4.38849412e-09 1.04239847e-08 2.53770751e-10] failed certificates: 0 example: []
offset-only wing 1 worst gap/weight/field Linf: [4.38849418e-09 1.04239836e-08 2.53770719e-10] failed certificates: 0 example: []
shifted with dual EMI wing 0 worst gap/weight/field Linf: [4.38849573e-09 1.04239848e-08 2.53770750e-10] failed certificates: 0 example: []
shifted with dual EMI wing 1 worst gap/weight/field Linf: [4.38849590e-09 1.04239843e-08 2.53770734e-10] failed certificates: 0 example: []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2318f0a373161bff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_2318f0a3 (directory).
```

### Turn 15 — `submit_completed_task` (`15833 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation and independent entry-point checks for all three certificates and the middle-budget field, including both wings. Further tuning would target the known single-reference identifiability limit rather than fix a regression on verified slices.
- **Alternatives Weighed**: `Further tune joint gain and offset estimation from one reference point per bank` (rejected: One reference does not independently identify both parameters, and the tested transport-based estimates did not recover them to certificate tolerance.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_w01_batch_01_solver.py`. The public validator passed for all three budgets. Independent tests on both wings passed for single-spot offset drift and simultaneous EMI spikes in three-point sweeps; the worst KKT gap across those tests was 4.39e-9. Limitation: when both gain and offset changed and only one reference point per bank was available, the worst stress-test gap was 9.91e-3.`

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
