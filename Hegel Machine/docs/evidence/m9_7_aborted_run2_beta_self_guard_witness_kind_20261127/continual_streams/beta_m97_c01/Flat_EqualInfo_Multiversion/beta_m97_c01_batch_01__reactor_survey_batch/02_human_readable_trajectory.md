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
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `271,755` (`263,128` / `69,760` / `8,627`) |
| **Total Episode Wall Time (s)** | `300.89s` |

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

### Turn 1 — `execute_in_remote_docker` (`18241 ms` LLM)
- **Rationale**: Inspect the public contract and archived solver to detect any mismatch in input keys, sensor stacking, or operator conventions. Those findings determine which parts can be reused and which must change.
- **Alternatives Weighed**: `Write the forward operator from scratch` (rejected: The verified predecessor can establish sensor ordering, discretization, and boundary conventions before any changes.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json; sed -n '1,240p' /holdout_task/validate_public_schema.py; sed -n '1,300p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1
```
- **Container Exit Code**: `0` (`376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 if low[hit] else cap
                continue
            w[free] = np.clip(target, 0, cap)
        else:
            g = H @ w - c
            low, high = np.flatnonzero(side < 0), np.flatnonzero(side > 0)
            i, j = low[g[low].argmin()], high[g[high].argmax()]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            side[[i, j]] = 0
            continue
        g = H @ w - c
        violations = side * (g - nu)
        release = int(violations.argmax())
        if violations[release] <= 1e-12:
            return w, nu
        side[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_s00(study: dict) -> dict:
    if study['recirculation_loop_mode'] != 0 or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']:
        raise ValueError('Only the open-outflow Robin regime is specified')
    y = _calibrate(study)
    sensors, terminal_modes = _forward(study)
    z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((y, sensors))
    projected = data - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    r, s = projected[:, 0], projected[:, 1:]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), s.T @ r, cap, budget)
    error = s @ w - r
    g = s.T @ error + ridge * w
    primal = float(0.5 * (error @ error + ridge * (w @ w)))
    delta = g - nu
    dual = min(primal, primal - float(g @ w) + nu * budget + cap * float(np.minimum(delta, 0).sum()))
    return {
        'weights': w,
        'terminal_field': (terminal_modes @ w).reshape((study['nx'], study['ny']), order='F'),
        'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                            'lam_lower': np.maximum(delta, 0).tolist(),
                            'lam_upper': np.maximum(-delta, 0).tolist(),
                            'primal_objective': primal, 'dual_lower_bound': dual},
    }
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19745 ms` LLM)
- **Rationale**: Recover the forward discretization and validator details omitted from the previous log. Inspect calibration coverage and the history format to determine which banks need fallback and how to test that path.

```bash
sed -n '1,105p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; sed -n '1,180p' /holdout_task/validate_public_schema.py; head -c 550 /tmp/hegel_continual_archive/archive_calibration_history.json; echo; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_beta_m97_c01_batch_01.pkl','rb'))
print(type(p),len(p)); print(list(p) if isinstance(p,dict) else list(p[0]));
if isinstance(p,dict): p=p.get('cases',p.get('studies',[p]))
for i,c in enumerate(p):
 if not isinstance(c,dict) or 'study' not in c: print(i,type(c)); continue
 s=c['study']; b=s['sensor_calibration_sweeps']; print(i,s['recirculation_loop_mode'],float(np.maximum(0,np.array(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation']).dot(np.maximum(0,np.array(s['past_vortex_circulation_trace'])-s['critical_vortex_circulation']))),b if i==0 else len(b))
PY
```
- **Container Exit Code**: `0` (`402 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
udget"], out["kkt_certificate"], gap_tol=2e-4)
        if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
        elif float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("w_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
[
  {
    "episode_short_name": "beta_m97_c01_s00",
    "bank_id": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
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
      1
<class 'list'> 1
['study', 'gt']
0 0.0 0.7179268595150001 {'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([-1.083514,  1.436139,  0.476672, -1.379075,  1.124264, -1.008754,
       -0.545696,  1.159503,  0.821322, -0.214967, -0.784196, -1.266671,
        0.900673, -1.346184,  1.985429,  0.582765,  1.624365,  0.687592,
        1.322749, -0.009779, -0.761243, -0.999244, -0.954807,  1.151292,
        0.387301, -0.283784, -1.382342,  0.966675,  1.48697 ,  0.484664,
       -1.125168, -0.379082,  2.114219,  0.406528, -0.92497 , -0.682303,
       -0.888164, -0.355165, -0.133275,  0.752048,  2.189532,  0.629129,
        2.079442, -0.055504,  0.834782, -1.335165,  0.520207,  1.01805 ]), 'raw_readout': array([-1.27940331,  2.02134213,  0.76444045, -1.66658821,  1.61278588,
       -1.18146772, -0.57486169,  1.65894877,  1.21593174, -0.14160686,
       -0.88729661, -1.51933912,  0.58153175, -1.19348541,  1.43848888,
        0.33038441,  1.15324843,  0.41319784,  0.91497183, -0.1377255 ,
       -0.73138198, -0.91940275, -0.88429754,  0.77952072,  0.3725073 ,
       -0.44621655, -1.78645703,  1.07934357,  1.71410356,  0.49129027,
       -1.47270499, -0.56248004,  2.47934717,  0.3959643 , -1.22846318,
       -0.93240976, -0.63493915, -0.18189029,  0.00671621,  0.75924087,
        1.98110217,  0.65475967,  1.88752566,  0.07282165,  0.8295648 ,
       -1.01489022,  0.56217583,  0.98534243])}
```

### Turn 3 — `execute_in_remote_docker` (`17269 ms` LLM)
- **Rationale**: Read the omitted operator and verifier lines to avoid changing the wrong boundary axis or using the wrong KKT sign convention. Confirm the archived sweep fields so sparse-bank fallback can fit the prior measurements directly.
- **Alternatives Weighed**: `Copy the archived calibration path unchanged` (rejected: It expects a bank-indexed dictionary, but the history file begins with a list of prior sweep records.)

```bash
sed -n '56,110p' /tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py; sed -n '1,57p' /holdout_task/validate_public_schema.py; python3 - <<'PY'
import json,pickle
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
a=json.load(open(p)); print('History records:',[(r['episode_short_name'],list(r),len(r['bank_id'])) for r in a]); s=pickle.load(open('/holdout_task/public_beta_m97_c01_batch_01.pkl','rb'))[0]['study']; print('Study keys:',sorted(s))
PY
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ll(np.isfinite(w)) and np.all(np.isfinite(ll)) and np.all(np.isfinite(lu)) and math.isfinite(nu) and math.isfinite(p_rep) and math.isfinite(d_rep)):
        return {"valid": False, "reason": "non_finite_values", "gap": 1e9}
    u_b = float(u_box)
    bt = float(b_tot)
    lw = float(lam_w)
    if float(np.min(w)) < -feas_tol or float(np.max(w)) > u_b + feas_tol:
        return {"valid": False, "reason": "box_violation", "gap": 1e9}
    if abs(float(np.sum(w)) - bt) > feas_tol * max(1.0, bt):
        return {"valid": False, "reason": f"budget_sum_violation:{abs(float(np.sum(w)) - bt):.3e}", "gap": 1e9}
    if float(np.min(ll)) < -feas_tol or float(np.min(lu)) < -feas_tol:
        return {"valid": False, "reason": "negative_bound_multipliers", "gap": 1e9}

    w_feas = np.clip(w, 0.0, u_b)
    res = S @ w_feas - ry
    j_true = 0.5 * float(res @ res) + 0.5 * lw * float(w_feas @ w_feas)
    g = S.T @ res + lw * w_feas
    # Unconditional convex supporting-hyperplane + box-Lagrange lower bound valid for ANY `(w_feas, nu)`:
    l_kkt = j_true - float(g @ w_feas) + nu * bt + u_b * float(np.sum(np.minimum(0.0, g - nu)))
    stat_res = float(np.max(np.abs(g - nu - ll + lu)))
    if abs(p_rep - j_true) > max(gap_tol, 1e-6 * max(1.0, abs(j_true))):
        return {"valid": False, "reason": f"reported_primal_mismatch:{abs(p_rep - j_true):.3e}", "gap": 1e9}
    if d_rep > l_kkt + gap_tol:
        return {"valid": False, "reason": f"unsound_dual_lower_bound:{d_rep:.8f}>{l_kkt:.8f}", "gap": 1e9}
    if stat_res > 1e-4:
        return {"valid": False, "reason": f"kkt_stationarity_violation:{stat_res:.3e}", "gap": 1e9}
    eff_lb = min(d_rep, l_kkt)
    gap = j_true - eff_lb
    if gap < -1e-9 or gap > gap_tol:
        return {"valid": False, "reason": f"duality_gap_too_large:{gap:.3e}", "gap": float(gap)}
    return {"valid": True, "reason": "ok", "true_primal": float(j_true), "verified_dual_lb": float(l_kkt), "gap": float(max(0.0, gap))}

History records: [('beta_m97_c01_s00', ['episode_short_name', 'bank_id', 'ref_signal', 'raw_readout'], 48)]
Study keys: ['Lx', 'Ly', 'V_reaction_2d', 'Z_nuisance', 'aux_manifold_telemetry', 'critical_vortex_circulation', 'kappa_x', 'kappa_y', 'mu_shifts', 'nuisance_ridge', 'nx', 'ny', 'past_vortex_circulation_trace', 'reaction_x', 'reaction_y', 'recirculation_loop_mode', 'robin_x', 'robin_y', 'row_bank_id', 'sensor_calibration_sweeps', 'sensor_ij', 'source_modes', 'source_ridge', 'total_budget', 'u_box', 'vel_x', 'vel_y', 'y_obs_raw']
```

### Turn 4 — `execute_in_remote_docker` (`44624 ms` LLM)
- **Rationale**: Build the solver from the archived discretization and active-set method. Compare each boundary convention against the public projected forward matrix and terminal field; verify the calibrated observations separately so a calibration error cannot be mistaken for an operator error.
- **Alternatives Weighed**: `Remove Robin loss on both axes` (rejected: The sorption mask identifies the x-walls; the public forward matrix will distinguish x-only sealing from sealing both axes.)

```bash
set -e
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py').read_text()
src = src.replace('Open-Robin reactor tomography', 'Calibrated reaction-transport tomography')
a = src.index('    for b in range(4):')
b = src.index('    gain, offset =', a)
src = src[:a] + '''    history = None
    for b in range(4):
        mask = banks == b
        x, y = ref[mask], raw[mask]
        clean = _clean_calibration(x, y)
        has_emi = np.unique(x).size >= 2 and not np.all(clean)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or not has_emi):
            coeffs[b] = _affine(x[clean], y[clean])
            continue
        if x.size < 2 or has_emi:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            for record in reversed(history):
                mask = np.asarray(record['bank_id']) == b
                hx = np.asarray(record['ref_signal'])[mask]
                hy = np.asarray(record['raw_readout'])[mask]
                keep = _clean_calibration(hx, hy)
                if np.unique(hx[keep]).size >= 2:
                    coeffs[b] = _affine(hx[keep], hy[keep])
                    break
            else:
                raise ValueError(f'No usable archived calibration for bank {b}')
        else:
            raise ValueError(f'Bank {b} needs two distinct calibration references')
''' + src[b:]
a = src.index('    ax, ay =', src.index('def _forward'))
b = src.index('    op =', a)
src = src[:a] + '''    lx, ly = study['Lx'], study['Ly']
    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                        - study['critical_vortex_circulation'])
    memory = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx)
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], ly)
    reaction = np.asarray(study['V_reaction_2d'])
    if sealed:
        x = np.linspace(0.0, lx, nx)[:, None]
        y = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(x / (0.25 * lx))**2) + np.exp(-((lx - x) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(memory)) * wall * (0.6 + 0.4 * np.sin(np.pi * y / ly))
    reaction = reaction.ravel(order='F')
''' + src[b:]
a = src.index('def solve_beta_m97_c01_s00(')
b = src.index('    y = _calibrate(study)', a)
src = src[:a] + 'def solve_beta_m97_c01_batch_01(study: dict) -> dict:\n' + src[b:]
Path('/holdout_task/beta_m97_c01_batch_01_solver.py').write_text(src)
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',base+'beta_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open(base+'public_beta_m97_c01_batch_01.pkl','rb'))[0]; st,gt=item['study'],item['gt']; z=st['Z_nuisance']; proj=lambda a:a-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
print('r_y max error:',np.max(np.abs(proj(m._calibrate(st))-gt['r_y'])))
for sx,sy in [(True,False),(True,True),(False,True),(False,False)]:
 axis=m._axis; count=[0]
 def choose(k,v,r,l):
  i=count[0]; count[0]+=1; return axis(k,v,0.0 if (sx,sy)[i] else st[('robin_x','robin_y')[i]],l)
 m._axis=choose
 S,F=m._forward(st); m._axis=axis
 print('sealed x/y',sx,sy,'S_tilde max error',np.max(np.abs(proj(S)-gt['S_tilde'])),'field max error at reference weights',np.max(np.abs((F@gt['w_opt']).reshape((st['nx'],st['ny']),order='F')-gt['terminal_field_A'])))
out=m.solve_beta_m97_c01_batch_01(st); print('output max errors',np.max(np.abs(out['weights']-gt['w_opt'])),np.max(np.abs(out['terminal_field']-gt['terminal_field_A']))); print('reported gap',out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`691 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{
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
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
r_y max error: 8.782890525971254e-08
sealed x/y True False S_tilde max error 0.031082372977105636 field max error at reference weights 0.017544950036994023
sealed x/y True True S_tilde max error 1.1796119636642288e-16 field max error at reference weights 4.85722573273506e-17
sealed x/y False True S_tilde max error 0.02519759923375522 field max error at reference weights 0.01901541825160822
sealed x/y False False S_tilde max error 0.03757185783447866 field max error at reference weights 0.025043063478499758
output max errors 0.10722209197386587 0.02050392813755553
reported gap 5.204170427930421e-17
{"schema_valid": false, "errors": ["kkt_invalid:reported_primal_mismatch:4.195e-03"]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7a17428e4270da87 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7a17428e (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32606 ms` LLM)
- **Rationale**: Apply the boundary correction and verify the operator, calibrated observations, weights, terminal field, and certificate separately. Compare archived and current bank fits to determine how to build meaningful fallback stress tests without assuming calibration is stationary.
- **ATMS NOGOOD DECLARED**: Refuted `The x-localized sorption mask implies that only the x-axis loses its Robin leakage.` | Witness: `X-only sealing gave a projected-operator max error of 0.03108; sealing both axes reproduced the operator and reference-weight terminal field to 1.18e-16 and 4.86e-17, respectively.` | Blanket Diagnosis: `Sorption localization and boundary leakage are separate operator terms; the x-wall sink does not determine which endpoint rows are sealed.` | Sublated Pivot: `Remove Robin leakage on both axes in the fouled regime, preserving the x-localized sink and archived interior discretization.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/beta_m97_c01_batch_01_solver.py')
s = p.read_text()
old = "    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], ly)"
assert s.count(old) == 1
p.write_text(s.replace(old, "    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], ly)"))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, json
import numpy as np
base = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', base+'beta_m97_c01_batch_01_solver.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load(open(base+'public_beta_m97_c01_batch_01.pkl','rb')):
    s, g = item['study'], item['gt']; out = m.solve_beta_m97_c01_batch_01(s)
    S, _ = m._forward(s); z = s['Z_nuisance']; proj = lambda a: a-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
    ds, dy = proj(S)-g['S_tilde'], proj(m._calibrate(s))-g['r_y']
    for b in range(4):
        rows = s['row_bank_id']==b
        print('bank', b, 'projected signal/operator max errors', np.max(np.abs(dy[rows])), np.max(np.abs(ds[rows])))
    for key, ref in [('weights','w_opt'), ('terminal_field','terminal_field_A')]:
        err = np.max(np.abs(out[key]-g[ref])); print(key,'max error',err); assert err < 1e-5
    c = out['kkt_certificate']; print('certificate gap', c['primal_objective']-c['dual_lower_bound'])
    assert np.max(np.abs(ds)) < 1e-10 and np.max(np.abs(dy)) < 1e-6
    prior = json.loads(m._ARCHIVE.read_text())[-1]; current=s['sensor_calibration_sweeps']
    for b in range(4):
        fits=[]
        for a in (current, prior):
            rows=np.asarray(a['bank_id'])==b
            fits.append(m._affine(np.asarray(a['ref_signal'])[rows],np.asarray(a['raw_readout'])[rows]))
        print('bank',b,'current minus archive coefficients',fits[0]-fits[1])
PY
```
- **Container Exit Code**: `0` (`753 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
bank 0 projected signal/operator max errors 1.8763309517222382e-08 1.1796119636642288e-16
bank 1 projected signal/operator max errors 8.17940397634942e-08 9.71445146547012e-17
bank 2 projected signal/operator max errors 7.798636492140076e-08 9.71445146547012e-17
bank 3 projected signal/operator max errors 8.782890525971254e-08 1.1102230246251565e-16
weights max error 3.6528120894718796e-07
terminal_field max error 2.922596074872441e-08
certificate gap 1.3183898417423734e-16
bank 0 current minus archive coefficients [0.30999998 0.14000001]
bank 1 current minus archive coefficients [-0.20999997 -0.12999998]
bank 2 current minus archive coefficients [ 0.22       -0.09999993]
bank 3 current minus archive coefficients [-0.15000004  0.12000003]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9634199c8230b3bd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9634199c (directory).
```

### Turn 6 — `execute_in_remote_docker` (`36264 ms` LLM)
- **Rationale**: The public case verifies the fouled regime with dense calibration. Stress the untested branches: a forward mismatch would require correcting the mode gate, a calibration mismatch would require changing archive selection or EMI rejection, and a budget failure would require correcting the active-set solve.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, json, pickle
import numpy as np

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = '/holdout_task/'
m = load(base + 'beta_m97_c01_batch_01_solver.py', 'candidate')
a = load('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py', 'archive')
item = pickle.load(open(base + 'public_beta_m97_c01_batch_01.pkl', 'rb'))[0]
s = item['study']
for mode, trace in [(1, s['past_vortex_circulation_trace']), (0, np.full_like(s['past_vortex_circulation_trace'], s['critical_vortex_circulation']))]:
    case = dict(s, recirculation_loop_mode=mode, past_vortex_circulation_trace=trace)
    for actual, ref in zip(m._forward(case), a._forward(case)):
        np.testing.assert_allclose(actual, ref, atol=1e-14, rtol=0)
print('Open-mode and zero-memory boundary slices match archived forward solver')
record = json.loads(m._ARCHIVE.read_text())[-1]
observed = np.linspace(-0.4, 0.6, len(s['row_bank_id']))
sweeps = s['sensor_calibration_sweeps']
for condition in ('missing', 'one_point', 'three_with_emi', 'many_with_emi'):
    worst = 0.0
    for b in range(4):
        case = copy.deepcopy(s)
        keep = np.asarray(sweeps['bank_id']) != b
        x = {'missing': [], 'one_point': [0.0], 'three_with_emi': [-1.0, 0.0, 1.0], 'many_with_emi': np.linspace(-2.0, 2.0, 11)}[condition]
        xx = np.asarray(x, dtype=float)
        gains, offsets = [], []
        for j in range(4):
            data = record if j == b and condition != 'many_with_emi' else sweeps
            rows = np.asarray(data['bank_id']) == j
            gain, offset = m._affine(np.asarray(data['ref_signal'])[rows], np.asarray(data['raw_readout'])[rows])
            gains.append(gain)
            offsets.append(offset)
        yy = gains[b]*xx + offsets[b]
        if condition.endswith('emi'):
            yy[len(yy)//2] += 8.0
        case['sensor_calibration_sweeps'] = dict(bank_id=np.r_[np.asarray(sweeps['bank_id'])[keep], np.full(len(xx), b, dtype=int)], ref_signal=np.r_[np.asarray(sweeps['ref_signal'])[keep], xx], raw_readout=np.r_[np.asarray(sweeps['raw_readout'])[keep], yy])
        ids = s['row_bank_id']
        case['y_obs_raw'] = np.asarray(gains)[ids]*observed + np.asarray(offsets)[ids]
        worst = max(worst, float(np.max(np.abs(m._calibrate(case)-observed))))
    print(condition, 'worst calibration error', worst)
    assert worst < 1e-12
for budget in (0.0, s['u_box'], 3*s['u_box'], 12*s['u_box']):
    case = dict(s, total_budget=budget)
    out = m.solve_beta_m97_c01_batch_01(case)
    w = np.asarray(out['weights'])
    assert abs(w.sum()-budget) < 1e-10 and w.min() > -1e-12 and w.max() < s['u_box']+1e-12
    cert = out['kkt_certificate']
    assert abs(cert['primal_objective']-cert['dual_lower_bound']) < 1e-10
print('Box-simplex endpoint and integer-capacity budgets passed')
PY
```
- **Container Exit Code**: `0` (`637 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Open-mode and zero-memory boundary slices match archived forward solver
missing worst calibration error 1.1102230246251565e-16
one_point worst calibration error 1.1102230246251565e-16
three_with_emi worst calibration error 1.1102230246251565e-16
many_with_emi worst calibration error 3.885780586188048e-16
Box-simplex endpoint and integer-capacity budgets passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9634199c8230b3bd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9634199c (directory).
```

### Turn 7 — `submit_completed_task` (`14227 ms` LLM)
- **Rationale**: The exact final artifact passed isolated public validation and targeted stress tests for sparse and contaminated calibration, open and zero-memory boundary modes, and endpoint and integer-capacity budgets. The host receipt is bound to the submitted artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:7711'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Maximum absolute errors were 3.65e-7 for source weights and 2.92e-8 for the terminal field. Calibration fallback, boundary-mode switching, and box-simplex budget stress tests also passed.`

### Turn 8 — `execute_in_remote_docker` (`27128 ms` LLM)
- **Rationale**: Leave out each real calibration measurement to detect a bank fit that depends on one point, then measure the effect on both deliverables. A failing slice would identify the calibration bank to investigate. The earlier counterfactual tests already covered both recirculation modes and the zero-memory boundary.
- **Alternatives Weighed**: `Add nonlinear terms or optimize a pooled classification score` (rejected: This is a continuous inverse problem with an explicitly specified operator. The corrected projected matrix already matches the public reference to roundoff, and there are no class labels.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'beta_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.load(open(p + 'public_beta_m97_c01_batch_01.pkl', 'rb')):
    s, g = item['study'], item['gt']
    sweeps = {k: np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id', 'ref_signal', 'raw_readout')}
    z = np.asarray(s['Z_nuisance'])
    y = m._calibrate(s)
    profiled = y - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ y)
    for b in range(4):
        ii = np.flatnonzero(sweeps['bank_id'] == b)
        rows = np.asarray(s['row_bank_id']) == b
        errors = []
        for i in ii:
            keep = np.arange(len(sweeps['bank_id'])) != i
            case = dict(s, sensor_calibration_sweeps={k: v[keep] for k, v in sweeps.items()})
            train = ii[ii != i]
            gain, offset = m._affine(sweeps['ref_signal'][train], sweeps['raw_readout'][train])
            out = m.solve_beta_m97_c01_batch_01(case)
            errors.append([abs(gain * sweeps['ref_signal'][i] + offset - sweeps['raw_readout'][i]), np.max(np.abs(np.asarray(out['weights']) - g['w_opt'])), np.max(np.abs(np.asarray(out['terminal_field']) - g['terminal_field_A']))])
        worst = np.max(errors, axis=0)
        print(f'bank {b}: {len(ii)} real sweeps; max held-out raw, weights, terminal errors: {worst}; max profiled observation error: {np.max(np.abs(profiled[rows] - g["r_y"][rows])):.3g}')
        assert np.all(worst < 1e-5), (b, worst)
PY
```
- **Container Exit Code**: `0` (`940 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
bank 0: 12 real sweeps; max held-out raw, weights, terminal errors: [1.85267526e-07 4.00045933e-07 2.97798811e-08]; max profiled observation error: 1.88e-08
bank 1: 12 real sweeps; max held-out raw, weights, terminal errors: [1.57613863e-07 4.05419315e-07 3.22155392e-08]; max profiled observation error: 8.18e-08
bank 2: 12 real sweeps; max held-out raw, weights, terminal errors: [1.95326948e-07 4.04748598e-07 3.27312189e-08]; max profiled observation error: 7.8e-08
bank 3: 12 real sweeps; max held-out raw, weights, terminal errors: [2.34730528e-07 4.20785076e-07 3.24363456e-08]; max profiled observation error: 8.78e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9634199c8230b3bd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9634199c (directory).
```

### Turn 9 — `submit_completed_task` (`15033 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated validation, and the within-bank audit found no calibration point that materially affects either output. The independent certificate and targeted stress tests cover the remaining applicable contract checks.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c01_batch_01_solver.py': 'FILE:7711'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Across all 48 leave-one-sweep-out tests, the maximum absolute errors were 4.21e-7 for source weights and 3.28e-8 for the terminal field. Boundary switching, sparse and EMI-contaminated calibration, and box-simplex budget stress tests also passed.`

### Turn 10 — `execute_in_remote_docker` (`34463 ms` LLM)
- **Rationale**: Compare the active-set result with an independent constrained optimizer and quantify the effect of skipping nuisance profiling on the measured bank slices. An optimizer discrepancy would trigger a KKT implementation review; a nonzero sealed endpoint row sum would trigger a stencil correction.
- **Alternatives Weighed**: `Apply within-batch balanced accuracy or introduce nonlinear drift terms` (rejected: The measured problem is continuous, with an explicit nuisance design and transport operator. A nuisance ablation and endpoint conservation check test the relevant assumptions directly.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p+'beta_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load(open(p+'public_beta_m97_c01_batch_01.pkl', 'rb')):
    s,g = item['study'],item['gt']; z = np.asarray(s['Z_nuisance']); y = m._calibrate(s); S,F = m._forward(s)
    lam = s['nuisance_ridge']; D = z.T @ z + lam*np.eye(z.shape[1]); P = np.eye(len(y)) - z @ np.linalg.solve(D,z.T)
    assert np.max(np.abs(P @ S-g['S_tilde'])) < 1e-10
    assert np.max(np.abs(P @ y-g['r_y'])) < 1e-6
    def solve(M, r):
        ridge=s['source_ridge']; u=s['u_box']; B=s['total_budget']
        obj=lambda w: 0.5*(np.linalg.norm(M@w-r)**2+ridge*(w@w))
        jac=lambda w: M.T@(M@w-r)+ridge*w
        n=M.shape[1]
        result=minimize(obj,np.full(n,B/n),jac=jac,bounds=[(0,u)]*n,constraints=[dict(type='eq',fun=lambda w:w.sum()-B,jac=lambda w:np.ones(n))],method='SLSQP',options=dict(ftol=1e-14,maxiter=500))
        assert result.success, result.message
        return result.x
    out=m.solve_beta_m97_c01_batch_01(s); ref=solve(P@S,P@y); naive=solve(S,y)
    err=np.max(np.abs(out['weights']-ref)); print('independent profiled optimizer max weight delta:',err); assert err < 1e-5
    for label,w in [('profiled',ref),('unprofiled',naive)]:
        bank=[float(np.linalg.norm(((P@S)@w-P@y)[np.asarray(s['row_bank_id'])==b])) for b in range(4)]
        print(label,'max weight error to public reference:',np.max(np.abs(w-g['w_opt'])),'profiled residual norm per bank:',bank)
    for ax in ('x','y'):
        A = m._axis(s['kappa_'+ax],s['vel_'+ax],0.0,s['L'+ax]); ends=np.asarray(A.sum(axis=1)).ravel()[[0,-1]]
        np.testing.assert_allclose(ends,0.,atol=1e-12,rtol=0)
    print('Zero-flux endpoint row sums verified on both measured axes')
PY
```
- **Container Exit Code**: `0` (`588 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent profiled optimizer max weight delta: 1.3552966771079866e-07
profiled max weight error to public reference: 3.775711328313225e-07 profiled residual norm per bank: [0.03894238377964199, 0.061751673139695665, 0.03711600994224386, 0.04074576183930883]
unprofiled max weight error to public reference: 0.8611623416493099 profiled residual norm per bank: [0.21996614291264888, 0.13433371785003245, 0.13321395764112753, 0.17883324798567707]
Zero-flux endpoint row sums verified on both measured axes

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9634199c8230b3bd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9634199c (directory).
```

### Turn 11 — `submit_completed_task` (`14841 ms` LLM)
- **Rationale**: The final artifact passed isolated validation. An independent optimizer agreed within 1.36e-7 in source weights, and both sealed axes satisfied the zero-flux endpoint check. The latest audit found no defect requiring an edit.
- **Alternatives Weighed**: `Skip nuisance profiling` (rejected: The measured ablation increased source-weight error and worsened the residual norm in every sensor bank.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_batch_01_solver.py`. The public validator passed. Across 48 leave-one-sweep-out tests, maximum absolute errors were 4.21e-7 for source weights and 3.27e-8 for the terminal field. Calibration fallback, boundary switching, and box-simplex budget stress tests also passed.`

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
