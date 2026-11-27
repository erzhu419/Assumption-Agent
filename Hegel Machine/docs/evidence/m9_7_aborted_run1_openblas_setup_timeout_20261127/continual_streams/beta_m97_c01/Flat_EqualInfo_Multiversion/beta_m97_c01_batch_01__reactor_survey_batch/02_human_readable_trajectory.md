# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c01_batch_01__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c01_step_01` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `235,625` (`227,809` / `85,504` / `7,816`) |
| **Total Episode Wall Time (s)** | `301.81s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16250 ms` LLM)
- **Rationale**: Inspect the public contract and prior solver to identify which discretization and data conventions carry over and which need to change. Check whether a verified nuisance-profiling or KKT primitive is available before implementing one.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; sed -n '1,300p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT' 2>&1 | head -35
```
- **Container Exit Code**: `0` (`368 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
= int(np.argmin(ratios))
        if ratios[k] < 1:
            w[free] += max(0.0, ratios[k]) * direction
            active[free[k]] = -1 if below[k] else 1
            w[free[k]] = 0.0 if below[k] else upper
            continue

        w[free] = np.clip(candidate, 0.0, upper)
        nu = -face[-1]
        grad = hessian @ w - linear
        violation = np.where(lo, nu - grad, np.where(hi, grad - nu, 0.0))
        k = int(np.argmax(violation))
        if violation[k] <= tol:
            return w, float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_s00(study: dict) -> dict:
    if (study['recirculation_loop_mode'] != 0
            or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('This solver implements the open-Robin regime')
    target = _calibrated_target(study)
    sensor, field = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    stacked = np.column_stack((target, sensor))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = stacked - z @ np.linalg.solve(gram, z.T @ stacked)
    target, sensor = projected[:, 0], projected[:, 1:]
    ridge, upper, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(sensor, target, ridge, upper, budget)
    residual = sensor @ w - target
    grad = sensor.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = grad - nu
    lower_bound = objective - grad @ w + nu * budget + upper * np.minimum(reduced, 0.0).sum()
    cert = {
        'weights': w.tolist(), 'nu_sum': nu,
        'lam_lower': np.maximum(reduced, 0.0).tolist(),
        'lam_upper': np.maximum(-reduced, 0.0).tolist(),
        'primal_objective': float(objective), 'dual_lower_bound': float(lower_bound),
    }
    terminal = (field @ w).reshape((study['nx'], study['ny']), order='F')
    return {'weights': w, 'terminal_field': terminal, 'kkt_certificate': cert}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18297 ms` LLM)
- **Rationale**: Recover the forward-operator code omitted from the truncated log and inspect public sweep coverage to determine which banks need history. Confirm the verifier's multiplier convention before reusing the archived active-set solver.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: Its open-Robin boundary condition does not implement the specified fouled regime.)

```bash
sed -n '1,110p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; sed -n '42,150p' /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 2>&1 | head -25; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_beta_m97_c01_batch_01.pkl','rb') as f: s=pickle.load(f)
for k,v in s.items():
 if isinstance(v,dict): print(k,{key:np.shape(val) for key,val in v.items()})
 elif np.isscalar(v): print(k,v)
 else: print(k,type(v).__name__,np.shape(v))
print('sweeps:',s['sensor_calibration_sweeps'])
print('v_mem:',np.square(np.maximum(0,np.asarray(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation'])).sum())
PY
```
- **Container Exit Code**: `1` (`616 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2,
      2,
      2,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "ref_signal": [
      0.274267,
      1.963912,
      0.496022,
      0.377988,
      -0.945951,
      1.664435,
      0.401932,
      -0.662038,
      0.388942,
      1.214001,
      1.652959,
      -1.263181,
      -0.543278,
      -0.369797,
      1.237839,
      0.438772,
      -1.241952,
      -0.735744,
      -0.176883,
      -0.618214,
      1.038437,
      1.692369,
      -1.265494,
      0.684674,
      0.420674,
      0.210506,
      -1.142051,
      1.942963,
      1.367292,
      -1.490556,
      0.401698,
      1.188278,
      0.978018,
      0.006246,
      -0.237646,
      -1.398028,
      0.293942,
      1.3537,
      0.562188,
      -0.989909,
      -0.642173,
      -0.377959,
      -0.95691,
      -0.953174,
      1.419985,
      -0.938988,
      0.959744,
      -0.942318
    ],
    "raw_readout": [
      0.274267,
      1.963912,
      0.496022,
      0.377988,
      -0.945951,
      1.664435,
      0.401932,
      -0.662038,
      0.388942,
      1.214001,
      1.652959,
      -1.263181,
      -0.543278,
      -0.369797,
      1.237839,
      0.438772,
      -1.241952,
      -0.735744,
      -0.176883,
      -0.618214,
      1.038437,
      1.692369,
      -1.265494,
      0.684674,
      0.420674,
      0.210506,
      -1.142051,
      1.942963,
      1.367292,
      -1.490556,
      0.401698,
      1.188278,
      0.978018,
      0.006246,
      -0.237646,
      -1.398028,
      0.293942,
      1.3537,
      0.562188,
      -0.989909,
      -0.642173,
      -0.377959,
      -0.95691,
      -0.953174,
      1.419985,
      -0.938988,
      0.959744,
      -0.942318
    ]
  }
]{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 3 — `execute_in_remote_docker` (`18562 ms` LLM)
- **Rationale**: Read the missing forward assembly and unpack the public list correctly. The per-bank sweep rows determine whether direct fitting, spike removal, or archived calibration is needed.
- **Alternatives Weighed**: `Reuse the archived calibration fallback unchanged` (rejected: It expects a per-bank parameter map, but the supplied history contains episode sweep records that must be fitted.)

```bash
sed -n '55,103p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; sed -n '40,110p' /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_beta_m97_c01_batch_01.pkl','rb') as f: a=pickle.load(f)
print(type(a).__name__,len(a))
for e in a:
 print('entry keys',list(e) if isinstance(e,dict) else type(e).__name__)
 if not isinstance(e,dict): continue
 s=next((d for d in [e,*e.values()] if isinstance(d,dict) and 'sensor_calibration_sweeps' in d),None)
 if s is None: continue
 print({k:(v if np.isscalar(v) else list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()})
 sw=s['sensor_calibration_sweeps']
 for b in range(4):
  m=np.asarray(sw['bank_id'])==b
  print(b,np.round(np.c_[np.asarray(sw['ref_signal'])[m],np.asarray(sw['raw_readout'])[m]],6).tolist())
 print('v_mem',np.square(np.maximum(0,np.asarray(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation'])).sum())
PY
```
- **Container Exit Code**: `0` (`542 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pe != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out: errs.append("missing:kkt_certificate")
    elif not errs:
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
        if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
        elif float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("w_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
list 1
entry keys ['study', 'gt']
{'Lx': 1.5, 'Ly': 1.3, 'nx': 18, 'ny': 16, 'kappa_x': (18,), 'vel_x': (18,), 'robin_x': 0.85, 'reaction_x': (18,), 'kappa_y': (16,), 'vel_y': (16,), 'robin_y': 0.75, 'reaction_y': (16,), 'V_reaction_2d': (18, 16), 'recirculation_loop_mode': 0.0, 'past_vortex_circulation_trace': (16,), 'critical_vortex_circulation': 0.5, 'mu_shifts': (4,), 'source_modes': (12, 18, 16), 'sensor_ij': (36, 2), 'row_bank_id': (144,), 'sensor_calibration_sweeps': ['bank_id', 'ref_signal', 'raw_readout'], 'Z_nuisance': (144, 5), 'y_obs_raw': (144,), 'aux_manifold_telemetry': (144, 3), 'nuisance_ridge': 0.65, 'source_ridge': 0.04, 'u_box': 1.0, 'total_budget': 2.6}
0 [[-1.083514, -1.279403], [1.436139, 2.021342], [0.476672, 0.76444], [-1.379075, -1.666588], [1.124264, 1.612786], [-1.008754, -1.181468], [-0.545696, -0.574862], [1.159503, 1.658949], [0.821322, 1.215932], [-0.214967, -0.141607], [-0.784196, -0.887297], [-1.266671, -1.519339]]
1 [[0.900673, 0.581532], [-1.346184, -1.193485], [1.985429, 1.438489], [0.582765, 0.330384], [1.624365, 1.153248], [0.687592, 0.413198], [1.322749, 0.914972], [-0.009779, -0.137726], [-0.761243, -0.731382], [-0.999244, -0.919403], [-0.954807, -0.884298], [1.151292, 0.779521]]
2 [[0.387301, 0.372507], [-0.283784, -0.446217], [-1.382342, -1.786457], [0.966675, 1.079344], [1.48697, 1.714104], [0.484664, 0.49129], [-1.125168, -1.472705], [-0.379082, -0.56248], [2.114219, 2.479347], [0.406528, 0.395964], [-0.92497, -1.228463], [-0.682303, -0.93241]]
3 [[-0.888164, -0.634939], [-0.355165, -0.18189], [-0.133275, 0.006716], [0.752048, 0.759241], [2.189532, 1.981102], [0.629129, 0.65476], [2.079442, 1.887526], [-0.055504, 0.072822], [0.834782, 0.829565], [-1.335165, -1.01489], [0.520207, 0.562176], [1.01805, 0.985342]]
v_mem 0.7179268595150001
```

### Turn 4 — `execute_in_remote_docker` (`47372 ms` LLM)
- **Rationale**: Build the solver using the archived spatial discretization and active-set method, changing the boundary rows, wall sink, and calibration history handling as required. Compare the projected sensor matrix and target independently to distinguish a forward-model mismatch from a calibration mismatch before stress-testing fallback slices.
- **Alternatives Weighed**: `Use archived calibration parameters for the public study` (rejected: Every public bank has 12 clean in-batch points; the archived identity calibration does not represent those sweeps.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py').read_text()
src = src.replace('Open-Robin reactor tomography and box-simplex KKT inversion.', 'Calibrated multi-boundary reactor tomography and box-simplex KKT inversion.')
a = src.index('def _calibrated_target(')
b = src.index('def _axis_operator(', a)
src = src[:a] + '''def _archived_line(history, bank):
    for sweep in reversed(history):
        mask = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'], dtype=float)[mask]
        y = np.asarray(sweep['raw_readout'], dtype=float)[mask]
        if x.size >= 2 and np.ptp(x) > 0:
            return _fit_line(x, y)
    raise ValueError(f'No archived calibration sweep for bank {bank}')


def _calibrated_target(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'])
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    result = np.empty_like(obs)
    history = None
    for bank in range(4):
        x, y = ref[banks == bank], raw[banks == bank]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            # Select the affine line with the smallest median absolute residual
            # before applying the EMI cutoff; a spike cannot pull an OLS fit.
            i, j = np.triu_indices(x.size, 1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            residual = y[None, :] - y[i, None] - slopes[:, None] * (x[None, :] - x[i, None])
            if slopes.size:
                residual -= np.median(residual, axis=1, keepdims=True)
                best = np.argmin(np.median(np.abs(residual), axis=1))
                clean = np.abs(residual[best]) <= 3.0
        if clean.sum() >= 4 or (x.size >= 2 and np.all(clean)):
            gain, offset = _fit_line(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_CALIBRATION_ARCHIVE.read_text())
            gain, offset = _archived_line(history, bank)
        rows = row_banks == bank
        result[rows] = (obs[rows] - offset) / gain
    return result


def _vortex_memory(study):
    excess = np.asarray(study['past_vortex_circulation_trace'], dtype=float) - study['critical_vortex_circulation']
    return float(np.square(np.maximum(excess, 0.0)).sum())


''' + src[b:]
a = src.index('    ax = _axis_operator(', src.index('def _forward('))
b = src.index('    operator = ', a)
src = src[:a] + '''    lx, ly = study['Lx'], study['Ly']
    memory = _vortex_memory(study)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0.0
    ax = _axis_operator(study['kappa_x'], study['vel_x'], lx, 0.0 if sealed else study['robin_x'])
    ay = _axis_operator(study['kappa_y'], study['vel_y'], ly, 0.0 if sealed else study['robin_y'])
    potential = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        potential = potential + (0.95 + 0.55 * np.tanh(memory)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    potential = potential.ravel(order='F')
''' + src[b:]
a = src.index('def solve_beta_m97_c01_s00(')
b = src.index('    target = _calibrated_target(study)', a)
src = src[:a] + 'def solve_beta_m97_c01_batch_01(study: dict) -> dict:\n' + src[b:]
Path('/holdout_task/beta_m97_c01_batch_01_solver.py').write_text(src)
PY
python3 -I /holdout_task/validate_public_schema.py || true
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'beta_m97_c01_batch_01_solver.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for item in pickle.load(open(p+'public_beta_m97_c01_batch_01.pkl','rb')):
 s,g=item['study'],item['gt']; o=mod.solve_beta_m97_c01_batch_01(s)
 raw, field=mod._forward(s); y=mod._calibrated_target(s); z=s['Z_nuisance']
 B=np.column_stack((y,raw)); P=B-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@B)
 print('reference keys/shapes', {k:np.shape(v) for k,v in g.items()})
 print('projected max abs: S',np.max(np.abs(g['S_tilde']-P[:,1:])), 'y',np.max(np.abs(g['r_y']-P[:,0])))
 for k in ('weights','terminal_field'):
  if k in g: print(k,'max absolute error',np.max(np.abs(g[k]-o[k])))
 c=o['kkt_certificate']; print('reported KKT gap',c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`1471 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
reference keys/shapes {'S_tilde': (144, 12), 'r_y': (144,), 'w_opt': (12,), 'cert_opt': (), 'terminal_field_A': (18, 16)}
projected max abs: S 1.6653345369377348e-16 y 8.782890481562333e-08
reported KKT gap 6.938893903907228e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e08605f76c9c469 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4e08605f (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32026 ms` LLM)
- **Rationale**: Check the named weight and field references, then stress each bank with EMI and missing sweeps. Failures would isolate calibration or archive handling; budget-endpoint failures would isolate the constrained optimizer.

```bash
set -e
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
from copy import deepcopy
from pathlib import Path
import importlib.util, pickle
import numpy as np
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'beta_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
e = pickle.loads((p/'public_beta_m97_c01_batch_01.pkl').read_bytes())[0]
s,g = e['study'],e['gt']
base = m.solve_beta_m97_c01_batch_01(s)
print('public weights max abs error:', np.max(np.abs(base['weights']-g['w_opt'])))
print('public terminal field max abs difference vs A:', np.max(np.abs(base['terminal_field']-g['terminal_field_A'])))
assert np.max(np.abs(base['weights']-g['w_opt'])) < 1e-5
text = (p/'validate_public_schema.py').read_text().split('\n_d = ')[0]
ns={}; exec(compile(text,'public_certificate_checker','exec'),ns)
verify=ns['verify_box_simplex_kkt_certificate']
def check(st, out, label):
    ref=verify(g['S_tilde'],g['r_y'],st['source_ridge'],st['u_box'],st['total_budget'],out['kkt_certificate'])
    print(label,'reference KKT:',ref['reason'],'gap:',ref['gap'])
    assert ref['valid'] and np.max(np.abs(out['weights']-out['kkt_certificate']['weights'])) < 1e-12
check(s,base,'public')
archive=m._CALIBRATION_ARCHIVE
m._CALIBRATION_ARCHIVE=Path('/nonexistent-calibration-history')
for bank in range(4):
    st=deepcopy(s); sw=st['sensor_calibration_sweeps']; idx=np.flatnonzero(np.asarray(sw['bank_id'])==bank)
    for j, spike in zip(idx[:4], [4.0,-6.0,9.0,-4.5]): sw['raw_readout'][j] += spike
    st['aux_manifold_telemetry']=np.full_like(st['aux_manifold_telemetry'],1e6)
    out=m.solve_beta_m97_c01_batch_01(st); check(st,out,f'EMI bank {bank}')
    assert np.max(np.abs(base['terminal_field']-out['terminal_field'])) < 1e-5
m._CALIBRATION_ARCHIVE=archive
for bank,keep in enumerate([0,1,0,1]):
    st=deepcopy(s); sw=st['sensor_calibration_sweeps']; b=np.asarray(sw['bank_id']); mask=(b!=bank); mask[np.flatnonzero(b==bank)[:keep]]=True
    for k in sw: sw[k]=np.asarray(sw[k])[mask]
    rows=np.asarray(st['row_bank_id'])==bank
    st['y_obs_raw'][rows]=m._calibrated_target(s)[rows]
    out=m.solve_beta_m97_c01_batch_01(st); check(st,out,f'archive bank {bank}, rows={keep}')
    np.testing.assert_allclose(out['terminal_field'],base['terminal_field'],atol=1e-12)
for budget in [0,1,2.6,6,11.99,12]:
    st=dict(s,total_budget=budget); check(st,m.solve_beta_m97_c01_batch_01(st),f'budget {budget}')
PY
```
- **Container Exit Code**: `0` (`2085 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public weights max abs error: 3.652812085863655e-07
public terminal field max abs difference vs A: 2.9225960776479987e-08
public reference KKT: ok gap: 4.830668848855579e-07
EMI bank 0 reference KKT: ok gap: 3.9440781693031446e-07
EMI bank 1 reference KKT: ok gap: 5.040621090211805e-07
EMI bank 2 reference KKT: ok gap: 4.897481798611658e-07
EMI bank 3 reference KKT: ok gap: 3.6709643084706367e-07
archive bank 0, rows=0 reference KKT: ok gap: 4.830668855343445e-07
archive bank 1, rows=1 reference KKT: ok gap: 4.83066884948008e-07
archive bank 2, rows=0 reference KKT: ok gap: 4.830668840771768e-07
archive bank 3, rows=1 reference KKT: ok gap: 4.830668852880138e-07
budget 0 reference KKT: ok gap: 1.5209356506173943e-07
budget 1 reference KKT: ok gap: 3.089820149182465e-07
budget 2.6 reference KKT: ok gap: 4.830668848855579e-07
budget 6 reference KKT: ok gap: 3.9066122350939203e-07
budget 11.99 reference KKT: ok gap: 7.888201203343215e-10
budget 12 reference KKT: ok gap: 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e08605f76c9c469 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4e08605f (directory).
```

### Turn 6 — `submit_completed_task` (`13312 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated-process validation, matched the public weight and field references, and met the certificate tolerance in every tested slice.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:8120'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Maximum absolute errors were 3.65e-7 for weights and 2.92e-8 for the terminal field. The worst KKT gap across the EMI, missing-calibration, and budget stress tests was 5.04e-7, below the 2e-4 tolerance.`

### Turn 7 — `execute_in_remote_docker` (`36706 ms` LLM)
- **Rationale**: The public case verifies the fouled branch but not whether sealing is correctly disabled in loop mode 1 or at zero vortex memory. An independent dense stencil using the measured reactor inputs will distinguish a gating, boundary, or field-indexing failure from a passing implementation without changing the artifact.
- **Alternatives Weighed**: `Test classification confounding or add unrelated basis terms` (rejected: The task is a continuous inverse problem with a specified transport operator and no class labels. Bank calibration and ridge nuisance projection already have direct reference checks.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'beta_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (p/'public_beta_m97_c01_batch_01.pkl').open('rb') as f:
    s = pickle.load(f)[0]['study']

def axis(k, v, length, loss):
    n = len(k)
    h = length/(n-1)
    a = np.zeros((n, n))
    for i, j in ((0, 1), (n-1, n-2)):
        a[i, i] = k[i]/h**2 + loss/h
        a[i, j] = -k[i]/h**2
    for i in range(1, n-1):
        for j, upwind in ((i-1, max(v[i], 0)), (i+1, max(-v[i], 0))):
            a[i, j] = -(k[i]+k[j])/(2*h**2) - upwind/h
        a[i, i] = -a[i, i-1]-a[i, i+1]
    return a

nx, ny = s['nx'], s['ny']
lx, ly = s['Lx'], s['Ly']
threshold = s['critical_vortex_circulation']
observed = np.asarray(s['past_vortex_circulation_trace'])
on_threshold = np.minimum(observed, threshold)
rhs = s['source_modes'].reshape(12, -1).T
si, sj = np.asarray(s['sensor_ij']).T
for mode in (0, 1):
    for label, trace in [('observed', observed), ('capped', on_threshold)]:
        st = dict(s, recirculation_loop_mode=mode, past_vortex_circulation_trace=trace)
        memory = sum((float(v)-threshold)**2 for v in trace if v > threshold)
        sealed = mode == 0 and memory > 0
        ax = axis(s['kappa_x'], s['vel_x'], lx, 0 if sealed else s['robin_x'])
        ay = axis(s['kappa_y'], s['vel_y'], ly, 0 if sealed else s['robin_y'])
        reaction = np.array(s['V_reaction_2d'], dtype=float, copy=True)
        if sealed:
            np.testing.assert_allclose(ax.sum(1), 0, atol=1e-12)
            np.testing.assert_allclose(ay.sum(1), 0, atol=1e-12)
            for i in range(nx):
                t = i/(nx-1)
                wall = np.exp(-16*t*t)+np.exp(-16*(1-t)**2)
                for j in range(ny):
                    reaction[i,j] += (0.95+0.55*np.tanh(memory))*wall*(0.6+0.4*np.sin(np.pi*j/(ny-1)))
        op = np.kron(ax, np.eye(ny)) + np.kron(np.eye(nx), ay) + np.diag(reaction.ravel())
        matrices = [np.linalg.solve(op+mu*np.eye(nx*ny), rhs) for mu in st['mu_shifts']]
        expected_s = np.vstack([a[si*ny+sj] for a in matrices])
        s_calc, _ = m._forward(st)
        out = m.solve_beta_m97_c01_batch_01(st)
        expected_field = (matrices[-1]@out['weights']).reshape(nx, ny)
        ds = np.max(np.abs(s_calc-expected_s))
        df = np.max(np.abs(out['terminal_field']-expected_field))
        print(f'mode={mode} trace={label} sealed={sealed}: design max error={ds:.3e}, field max error={df:.3e}')
        assert max(ds, df) < 1e-12
PY
```
- **Container Exit Code**: `0` (`635 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mode=0 trace=observed sealed=True: design max error=1.665e-16, field max error=3.469e-17
mode=0 trace=capped sealed=False: design max error=6.939e-17, field max error=4.163e-17
mode=1 trace=observed sealed=False: design max error=6.939e-17, field max error=4.163e-17
mode=1 trace=capped sealed=False: design max error=6.939e-17, field max error=4.163e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e08605f76c9c469 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4e08605f (directory).
```

### Turn 8 — `submit_completed_task` (`14468 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process schema and certificate checks. The independent dense audit confirmed that sealing occurs only in loop mode 0 with positive vortex memory; no remaining failure warrants an edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:8120'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Maximum absolute errors were 3.65e-7 for weights and 2.92e-8 for the terminal field. The worst KKT gap across EMI, missing-sweep, and budget tests was 5.04e-7, below the 2e-4 tolerance. An independent dense reference also matched the forward model in both loop modes, with and without vortex memory.`

### Turn 9 — `execute_in_remote_docker` (`30678 ms` LLM)
- **Rationale**: Leave out each real calibration reading in turn and evaluate the resulting certificate against the unchanged public reference. A failure would identify a bank whose affine calibration depends excessively on one reading and warrants inspection; passing would close that margin without changing the verified solver.
- **Alternatives Weighed**: `Repeat the loop-mode audit or fit a batch-centered classifier` (rejected: The independent stencil already covers both loop modes and both memory states. This inverse problem has no class labels; bank-specific calibration is the relevant check.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'beta_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with (p/'public_beta_m97_c01_batch_01.pkl').open('rb') as f:
    case = pickle.load(f)[0]
s, g = case['study'], case['gt']
z = s['Z_nuisance']; lam = s['nuisance_ridge']
def project(y):
    return y - z @ np.linalg.solve(z.T @ z + lam*np.eye(z.shape[1]), z.T @ y)
ns = {}; exec(compile((p/'validate_public_schema.py').read_text().split('\n_d = ')[0], 'public_certificate_checker', 'exec'), ns)
verify = ns['verify_box_simplex_kkt_certificate']
sw = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
banks = sw['bank_id']; row_banks = np.asarray(s['row_bank_id'])
worst_gap = (-np.inf, None); worst_target = (-np.inf, None); failures = []
for i in range(len(banks)):
    mask = np.arange(len(banks)) != i
    st = dict(s, sensor_calibration_sweeps={k:v[mask] for k,v in sw.items()})
    out = m.solve_beta_m97_c01_batch_01(st)
    cert = verify(g['S_tilde'], g['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    dy = project(m._calibrated_target(st)) - g['r_y']
    bank = int(banks[i]); label = (bank, i)
    err = float(np.max(np.abs(dy[row_banks == bank])))
    if err > worst_target[0]: worst_target = (err, label)
    if cert['gap'] > worst_gap[0]: worst_gap = (cert['gap'], label)
    if not cert['valid']: failures.append((label, cert['reason']))
print('Real sweep holdouts:', len(banks), 'worst held-out bank projected target error:', worst_target, 'worst reference KKT gap:', worst_gap, 'failures:', failures)
assert not failures
PY
```
- **Container Exit Code**: `0` (`726 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Real sweep holdouts: 48 worst held-out bank projected target error: (1.1749732919952294e-07, (1, 14)) worst reference KKT gap: (5.466081512382759e-07, (3, 46)) failures: []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e08605f76c9c469 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4e08605f (directory).
```

### Turn 10 — `submit_completed_task` (`13989 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process validation, independent forward-operator checks, and certificate tests covering EMI, missing sweeps, budget endpoints, and every real calibration-row holdout. No tested slice warrants an edit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Maximum absolute errors were 3.65e-7 for weights and 2.92e-8 for the terminal field. The worst KKT gap across all tested slices, including 48 calibration holdouts, was 5.47e-7 against a 2e-4 tolerance.`

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
