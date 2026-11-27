# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c02_s00__reactor_tomography_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c02_source` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `268,445` (`258,757` / `65,408` / `9,688`) |
| **Total Episode Wall Time (s)** | `318.88s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`root_A03_ill_posed_regularization_and_certificates`**: AXIOM A03 (HEAVY-TAILED REGULARIZATION & DUAL CERTIFICATES): Under outliers, stage drift, or ill-posed combinatorial landscapes, replace naive L2 least squares with heavy-tailed robust losses (Huber/Cauchy/MAD) or exact variable projection, and verify optimality/safety margins via dual bounds or worst-corner simulation.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15337 ms` LLM)
- **Rationale**: Inspect the public schema, array layout, and supported calibration fallback before implementing; a mismatch in sensor stacking or field ordering would change the forward model.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_beta_m97_c02_s00.pkl','rb'))
for k,v in s.items():
 print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:275])
PY
ls -l /tmp/hegel_continual_archive 2>/dev/null; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator | STATUS: MISMATCHED_BOUNDARY_OR_TENSOR(zero_ghost_scalar_dirichlet_only)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`15715 ms` LLM)
- **Rationale**: Inspect the study inside the public list to resolve the sensor and source layouts, and check the ridge primitive’s contract before choosing an implementation.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
studies=pickle.load(open('public_beta_m97_c02_s00.pkl','rb'))
print('studies',len(studies))
for k,v in studies[0].items():
 print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:(750 if ('sensor' in k or 'calibration' in k) else 170)])
PY
sed -n '51,150p' /holdout_task/validate_public_schema.py
ls -l /tmp/hegel_continual_archive 2>/dev/null
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`395 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`19220 ms` LLM)
- **Rationale**: The prior inspection stopped at the wrapper rather than the study fields. Reading the nested field shapes and validator sign convention will determine sensor stacking, boundary-grid implementation, and certificate multipliers; archive availability determines how to test calibration fallback.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
item=pickle.load(open('public_beta_m97_c02_s00.pkl','rb'))[0]
for name, d in item.items():
 print(name)
 for k,v in d.items():
  if isinstance(v,np.ndarray):
   print(k,v.shape,v.dtype,repr(v) if v.size<=25 else ('range',v.min(),v.max(), 'first',v.ravel()[:5]))
  elif isinstance(v,(list,tuple)):
   print(k, type(v).__name__,len(v),repr(v)[:600])
  elif isinstance(v,dict):
   print(k,{p:(getattr(q,'shape',None),repr(q)[:260]) for p,q in v.items()})
  else: print(k,repr(v)[:240])
PY
sed -n '32,55p' validate_public_schema.py
if test -f /tmp/hegel_continual_archive/archive_calibration_history.json; then head -c 1400 /tmp/hegel_continual_archive/archive_calibration_history.json; else echo 'No calibration archive mounted'; fi
python3 -I -c 'import scipy; print("SciPy",scipy.__version__)'
```
- **Container Exit Code**: `0` (`474 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,
       0.1962148 , 0.19982431])
cert_opt {'weights': (None, '[0.24349145775048647, 0.33998039113126294, 0.14898826172890828, 0.12667697308479964, 0.383895035601229, 0.2706317879961699, 0.14517414379017055, 0.1730898138233597, 0.20928997979959976, 0.16274303892623132, 0.19621480148020454, 0.19982431488757776]'), 'nu_sum': (None, '0.00740069553106712'), 'lam_lower': (None, '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0408340855860843e-17, 6.938893903907228e-18, 6.938893903907228e-18, 6.938893903907228e-18, 1.3877787807814457e-17]'), 'lam_upper': (None, '[1.3877787807814457e-17, 0.0, 2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'), 'primal_objective': (None, '0.017155323171798267'), 'dual_lower_bound': (None, '0.01715532317179823')}
terminal_field_A (18, 16) float64 ('range', 0.0015798871346810932, 0.03815335154288428, 'first', array([0.00157989, 0.00278903, 0.00407259, 0.0052697 , 0.00625795]))
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
No calibration archive mounted
SciPy 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`69785 ms` LLM)
- **Rationale**: Compare the design, residualized observations, weights, and terminal field independently with the public reference so a discrepancy identifies which stage to fix.
- **Alternatives Weighed**: `Use the ridge profiling primitive` (rejected: The specified five-column normal equation handles the observations and all forward columns in one solve, without constructing the primitive’s unused whitener.)

```bash
cd /holdout_task && cat > beta_m97_c02_s00_solver.py <<'PY'
"""Open-Robin reaction–transport tomography and certified source inversion."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    parameters = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, k=1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            if i.size:
                # Median of pointwise median slopes, with centered intercept.
                slopes = (y[j] - y[i]) / (x[j] - x[i])
                medians = [np.median(slopes[(i == k) | (j == k)]) for k in range(x.size)]
                gain = np.median(medians)
                offset = np.median(y - gain * x)
                clean = np.abs(y - (gain * x + offset)) <= 3.0
        if x.size < 2 or not clean.all():
            if history is None:
                try:
                    history = json.loads(_ARCHIVE.read_text())
                except FileNotFoundError:
                    history = {}
        if np.ptp(x[clean]) > 0 if clean.sum() >= 2 else False:
            parameters[b] = _line(x[clean], y[clean])
        else:
            prior = (history or {}).get(str(b))
            if prior is None:
                raise ValueError(f'Bank {b} has no identifiable in-batch calibration or archived calibration')
            parameters[b] = prior['gain'], prior['offset']
    gain, offset = parameters[np.asarray(study['row_bank_id'])].T
    return (np.asarray(study['y_obs_raw']) - offset) / gain


def _axis(kappa, velocity, robin, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    main = np.empty(kappa.size)
    main[1:-1] = face[:-1] + face[1:] + np.abs(velocity[1:-1]) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main[0], main[-1] = -upper[0] + robin / h, -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'] / (ny - 1))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    operator += diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc')
    identity = eye(nx * ny, format='csc')
    sources = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij']).T
    rows = ix + nx * iy
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(sources)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields


def _box_simplex(hessian, linear, cap, budget):
    """Primal active set; the equality multiplier uses the gradient sign."""
    n = linear.size
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w - linear
        return w, float(g.min() if budget == 0 else g.max())
    active = np.zeros(n, dtype=int)
    for _ in range(20 * n**2):
        free = np.flatnonzero(active == 0)
        g = hessian @ w - linear
        if not free.size:
            lo = np.flatnonzero(active < 0)
            hi = np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue
        k = free.size
        kkt = np.ones((k + 1, k + 1))
        kkt[:k, :k] = hessian[np.ix_(free, free)]
        kkt[:k, k] = -1
        kkt[k, k] = 0
        p = np.flatnonzero(active != 0)
        sol = np.linalg.solve(kkt, np.r_[linear[free] - hessian[np.ix_(free, p)] @ w[p], budget - w[p].sum()])
        trial, nu = sol[:-1], float(sol[-1])
        direction = trial - w[free]
        steps = np.full(k, np.inf)
        lower = trial < -1e-13
        upper = trial > cap + 1e-13
        steps[lower] = -w[free[lower]] / direction[lower]
        steps[upper] = (cap - w[free[upper]]) / direction[upper]
        hit = np.argmin(steps)
        if steps[hit] < 1:
            w[free] += steps[hit] * direction
            i = free[hit]
            active[i], w[i] = (-1, 0.0) if lower[hit] else (1, cap)
            continue
        w[free] = np.clip(trial, 0, cap)
        g = hessian @ w - linear
        violation = active * (g - nu)
        i = np.argmax(violation)
        if violation[i] <= 1e-12:
            return w, nu
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c02_s00(study: dict) -> dict:
    raw_design, terminal_basis = _forward(study)
    z = np.asarray(study['Z_nuisance'])
    stacked = np.column_stack((_calibrate(study), raw_design))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    residualized = stacked - z @ np.linalg.solve(gram, z.T @ stacked)
    target, design = residualized[:, 0], residualized[:, 1:]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ target, cap, budget)
    residual = design @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = design.T @ residual + ridge * w
    dual = objective - g @ w + nu * budget + cap * np.minimum(0, g - nu).sum()
    certificate = dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(0, g - nu),
                       lam_upper=np.maximum(0, nu - g), primal_objective=objective,
                       dual_lower_bound=float(dual))
    return dict(weights=w, terminal_field=(terminal_basis @ w).reshape((study['nx'], study['ny']), order='F'),
                kkt_certificate=certificate)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','beta_m97_c02_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('public_beta_m97_c02_s00.pkl','rb'))[0]; s,g=d['study'],d['gt']; out=m.solve_beta_m97_c02_s00(s)
z=s['Z_nuisance']; r=np.column_stack((m._calibrate(s),m._forward(s)[0])); r-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@r)
for name,a,b in [('design',r[:,1:],g['S_tilde']),('target',r[:,0],g['r_y']),('weights',out['weights'],g['w_opt']),('terminal',out['terminal_field'],g['terminal_field_A'])]: print(name,'max absolute error',np.max(np.abs(a-b)))
print('certificate gap',out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
print('calibration rows per bank',np.bincount(s['sensor_calibration_sweeps']['bank_id']))
PY
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
design max absolute error 8.326672684688674e-17
target max absolute error 1.0501607938806501e-07
weights max absolute error 2.1489762833115655e-07
terminal max absolute error 1.5003001722913023e-08
certificate gap 6.245004513516506e-17
calibration rows per bank [12 12 12 12]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e1109e102a5a933 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9e1109e1 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`40964 ms` LLM)
- **Rationale**: The public case has no active weight bounds and all four banks have full sweeps. Stress the budget endpoints, simultaneous lower and upper bounds, and calibration with both missing rows and EMI spikes in an isolated Python process. Any failure will identify whether the active set, certificate, or calibration fallback needs revision.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
validator_source = Path('validate_public_schema.py').read_text().split('\n_d =', 1)[0]
namespace = {}
exec(validator_source, namespace)
verify = namespace['verify_box_simplex_kkt_certificate']
item = pickle.load(open('public_beta_m97_c02_s00.pkl', 'rb'))[0]
s = item['study']
z = s['Z_nuisance']
raw_design, basis = m._forward(s)
def profile(a):
    return a - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ a)

print('Public-study budget stress:')
worst = 0.0
for budget in [0., 1e-6, .4, 1., 2.6, 6., 11.9, 12.]:
    st = dict(s, total_budget=budget)
    out = m.solve_beta_m97_c02_s00(st)
    w = out['weights']
    chk = verify(item['gt']['S_tilde'], item['gt']['r_y'], s['source_ridge'], s['u_box'], budget, out['kkt_certificate'])
    assert chk['valid'], (budget, chk)
    assert np.max(np.abs(out['terminal_field'].ravel(order='F') - basis @ w)) < 1e-12
    worst = max(worst, chk['gap'])
    print(f'  budget={budget:g}: lower={np.count_nonzero(w == 0)} upper={np.count_nonzero(w == 1)} gap={chk["gap"]:.3g}')

rng = np.random.default_rng(42)
max_gap = 0.0
joint = 0
for k in range(400):
    n = 12
    a = rng.normal(size=(15, n))
    h = a.T @ a + .04 * np.eye(n)
    c = rng.normal(size=n) * rng.choice([.01, 1., 100.])
    cap = float(rng.choice([.15, 1., 5.]))
    budget = cap * (rng.integers(1, n) if k % 2 else rng.uniform(.001, n - .001))
    w, nu = m._box_simplex(h, c, cap, budget)
    g = h @ w - c
    gap = g @ w - nu * budget - cap * np.minimum(0, g - nu).sum()
    assert w.min() >= 0 and w.max() <= cap and abs(w.sum() - budget) < 1e-10 and abs(gap) < 1e-9, (k, budget, gap)
    joint += bool(np.any(w == 0) and np.any(w == cap))
    max_gap = max(max_gap, abs(gap))
assert joint > 0
print(f'Random QPs: {joint}/400 had both bound types; max absolute gap {max_gap:.3g}')

bank = np.repeat(np.arange(4), 7)
ref = np.tile(np.linspace(-1., 1., 7), 4)
gain = np.array([.8, 1.1, 1.4, 1.7])
offset = np.array([-.3, .1, -.15, .65])
sweep_raw = gain[bank] * ref + offset[bank]
sweep_raw[15] += 3.1
sweep_raw[19] -= 3.2
keep = (bank != 1) | (ref == 0)
rows = s['row_bank_id']
target = raw_design @ np.linspace(.05, .38, 12) + z @ np.arange(5) * .07
st = dict(s, y_obs_raw=gain[rows] * target + offset[rows], sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=sweep_raw[keep]))
with tempfile.TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp) / 'archive.json'
    m._ARCHIVE.write_text(json.dumps({'1': {'gain': float(gain[1]), 'offset': float(offset[1])}}))
    err = np.max(np.abs(m._calibrate(st) - target))
    assert err < 1e-12, err
    out = m.solve_beta_m97_c02_s00(st)
    chk = verify(profile(raw_design), profile(target), s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'], gap_tol=1e-9)
    assert chk['valid'], chk
    print(f'Sparse-bank archive + two EMI spikes: max calibration error {err:.3g}, gap {chk["gap"]:.3g}')
print(f'Worst public-reference stress gap {worst:.3g}')
PY
```
- **Container Exit Code**: `0` (`716 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public-study budget stress:
  budget=0: lower=12 upper=0 gap=0
  budget=1e-06: lower=11 upper=0 gap=3.68e-14
  budget=0.4: lower=9 upper=0 gap=1.48e-08
  budget=1: lower=6 upper=0 gap=3.52e-08
  budget=2.6: lower=0 upper=0 gap=6.08e-08
  budget=6: lower=0 upper=0 gap=9.51e-08
  budget=11.9: lower=0 upper=11 gap=1.68e-07
  budget=12: lower=0 upper=12 gap=1.72e-07
Random QPs: 241/400 had both bound types; max absolute gap 3.64e-12
Sparse-bank archive + two EMI spikes: max calibration error 2.22e-16, gap 3.75e-16
Worst public-reference stress gap 1.72e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e1109e102a5a933 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9e1109e1 (directory).
```

### Turn 6 — `submit_completed_task` (`14846 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification, independent comparisons against the public weights and terminal field, and stress tests covering both bound types and calibration fallback. No tested slice exceeded its tolerance.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c02_s00_solver.py': 'FILE:6563'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c02_s00_solver.py`. The public validator passed. Maximum absolute error was 2.15e-7 for the source weights and 1.50e-8 for the terminal field. Stress tests also passed for active bounds, budget endpoints, sparse calibration, and EMI spikes.`

### Turn 7 — `execute_in_remote_docker` (`32511 ms` LLM)
- **Rationale**: Independently apply the documented stencil to the returned terminal field: a failure would identify a boundary, spacing, or layout error. Compare residualized observations by bank to detect calibration errors that could be hidden in a pooled metric.
- **Alternatives Weighed**: `Test relay modes or within-batch classification accuracy` (rejected: The specified study is an open-Robin inverse problem with continuous measurements, no relay state, and no class labels. Calibration can instead be checked separately for each sensor bank.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c02_s00_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
item = pickle.load(open('public_beta_m97_c02_s00.pkl', 'rb'))[0]
s, gt = item['study'], item['gt']
print('Operating regime:', s['recirculation_loop_mode'], max(s['past_vortex_circulation_trace']), s['critical_vortex_circulation'])
assert s['recirculation_loop_mode'] == 0 and max(s['past_vortex_circulation_trace']) <= s['critical_vortex_circulation']
out = solver.solve_beta_m97_c02_s00(s)
w, u = out['weights'], out['terminal_field']
rhs = np.einsum('m,mij->ij', w, s['source_modes'])
def apply_axis(f, axis):
    k, v, rob, length = (s[key + '_' + axis] for key in ('kappa', 'vel', 'robin', 'L'))
    h = length / (len(k)-1)
    p = np.empty_like(f)
    p[0] = k[0]*(f[0]-f[1])/h**2 + rob*f[0]/h
    p[-1] = k[-1]*(f[-1]-f[-2])/h**2 + rob*f[-1]/h
    for i in range(1,len(k)-1):
        left, right = f[i]-f[i-1], f[i]-f[i+1]
        p[i] = ((k[i]+k[i-1])*left + (k[i]+k[i+1])*right)/(2*h*h) + (max(v[i],0)*left + max(-v[i],0)*right)/h
    return p
pde = np.max(np.abs(apply_axis(u,'x') + apply_axis(u.T,'y').T + (s['V_reaction_2d'] + s['mu_shifts'][-1])*u - rhs))
assert pde < 1e-12, pde
z = s['Z_nuisance']
y = solver._calibrate(s)
r = y - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T @ y)
calibration = s['sensor_calibration_sweeps']
print('Terminal stencil max residual:', pde)
for b in range(4):
    sel = calibration['bank_id'] == b
    x, raw = calibration['ref_signal'][sel], calibration['raw_readout'][sel]
    gain, offset = solver._line(x,raw)
    bank_error = np.max(np.abs((r - gt['r_y'])[s['row_bank_id']==b]))
    sweep_error = np.max(np.abs(raw-gain*x-offset))
    print(f'Bank {b}: sweep residual {sweep_error:.3g}, reference residualized error {bank_error:.3g}')
    assert bank_error < 1e-6, (b, bank_error)
assert np.array_equal(w, out['kkt_certificate']['weights'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`591 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Operating regime: 0.0 0.25658 0.5
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e1109e102a5a933 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9e1109e1 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
  File "<stdin>", line 14, in apply_axis
  File "<stdin>", line 14, in <genexpr>
KeyError: 'L_x'
```

### Turn 8 — `execute_in_remote_docker` (`29161 ms` LLM)
- **Rationale**: The independent audit stopped on a test-script key typo, not a solver failure. Use the actual Lx and Ly keys and fail immediately on errors. Separate boundary and interior residuals will localize any stencil mismatch; per-bank observation errors will identify any calibration discrepancy hidden by pooling.

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c02_s00_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
for item in pickle.load(open('public_beta_m97_c02_s00.pkl', 'rb')):
    s, gt = item['study'], item['gt']
    assert s['recirculation_loop_mode'] == 0 and max(s['past_vortex_circulation_trace']) <= s['critical_vortex_circulation']
    out = solver.solve_beta_m97_c02_s00(s)
    w, u = out['weights'], out['terminal_field']
    rhs = np.einsum('m,mij->ij', w, s['source_modes'])
    def apply_axis(f, axis):
        k, v, rob = (s[key + '_' + axis] for key in ('kappa', 'vel', 'robin'))
        h = s['L' + axis] / (len(k) - 1)
        p = np.empty_like(f)
        p[0] = k[0] * (f[0] - f[1]) / h**2 + rob * f[0] / h
        p[-1] = k[-1] * (f[-1] - f[-2]) / h**2 + rob * f[-1] / h
        for i in range(1, len(k) - 1):
            left, right = f[i] - f[i-1], f[i] - f[i+1]
            p[i] = ((k[i] + k[i-1]) * left + (k[i] + k[i+1]) * right) / (2*h*h) + (max(v[i], 0)*left + max(-v[i], 0)*right) / h
        return p
    e = apply_axis(u, 'x') + apply_axis(u.T, 'y').T + (s['V_reaction_2d'] + s['mu_shifts'][-1]) * u - rhs
    boundary = np.ones(u.shape, dtype=bool)
    boundary[1:-1, 1:-1] = False
    for name, mask in [('boundary', boundary), ('interior', ~boundary)]:
        err = np.max(np.abs(e[mask]))
        print(f'Terminal {name} stencil residual: {err:.3g}')
        assert err < 1e-12, (name, err)
    z = s['Z_nuisance']
    y = solver._calibrate(s)
    r = y - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ y)
    for bank in range(4):
        mask = s['row_bank_id'] == bank
        err = np.max(np.abs((r - gt['r_y'])[mask]))
        print(f'Bank {bank} max residualized observation error: {err:.3g}')
        assert err < 1e-6, (bank, err)
    assert np.array_equal(w, out['kkt_certificate']['weights'])
PY
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`605 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Terminal boundary stencil residual: 2.15e-16
Terminal interior stencil residual: 9.71e-16
Bank 0 max residualized observation error: 1.05e-07
Bank 1 max residualized observation error: 7.66e-08
Bank 2 max residualized observation error: 4.68e-08
Bank 3 max residualized observation error: 2.62e-08
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e1109e102a5a933 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9e1109e1 (directory).
```

### Turn 9 — `submit_completed_task` (`13979 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated verification, including active bounds, budget endpoints, sparse calibration, and EMI stress tests. Independent stencil residuals stayed below 1e-15 at both boundary and interior points, and the worst per-bank observation error was 1.05e-7.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c02_s00_solver.py': 'FILE:6563'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_s00_solver.py`. The public schema validator passed. Maximum absolute errors were 2.15e-7 for the source weights and 1.50e-8 for the terminal field. The worst KKT certificate gap across budget stress tests was 1.72e-7, below the 1e-6 tolerance.`

### Turn 10 — `execute_in_remote_docker` (`27378 ms` LLM)
- **Rationale**: Use measured sweeps to quantify leave-one-row-out stability, then inject just-over-threshold EMI at each retained reference level. Five-point sweeps leave exactly four clean rows, the stated direct-fit threshold. If these cases expose spike masking, revise the spike detector while preserving the direct affine fit on clean data.
- **Alternatives Weighed**: `Repeat the terminal stencil and budget sweeps` (rejected: Both already passed on the verified artifact; neither detects a calibration spike masked by a high-leverage reference point.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('public_beta_m97_c02_s00.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']
base = m._calibrate(s)
for b in range(4):
    ix = np.flatnonzero(c['bank_id'] == b)
    rows = s['row_bank_id'] == b
    loo = []
    for i in ix:
        keep = np.arange(len(c['bank_id'])) != i
        out = m._calibrate(dict(s, sensor_calibration_sweeps={k: v[keep] for k,v in c.items()}))
        loo.append(np.max(np.abs(out[rows] - base[rows])))
    err = []
    for n in (5, 12):
        subset = ix[np.linspace(0, len(ix)-1, n, dtype=int)]
        keep = (c['bank_id'] != b) | np.isin(np.arange(len(c['bank_id'])), subset)
        clean = {k: v[keep] for k,v in c.items()}
        control = m._calibrate(dict(s, sensor_calibration_sweeps=clean))
        worst = (0., -1, 0.)
        for j, spike in itertools.product(np.flatnonzero(clean['bank_id'] == b), (-3.01, 3.01)):
            raw = clean['raw_readout'].copy()
            raw[j] += spike
            out = m._calibrate(dict(s, sensor_calibration_sweeps=dict(clean, raw_readout=raw)))
            worst = max(worst, (float(np.max(np.abs(out[rows] - control[rows]))), int(j), spike))
        err.append((n, worst))
    print(f'bank {b}: leave-one-sweep-out max={max(loo):.3g}; injected EMI max (sweep count, (error, row, spike)): {err}')
index = json.load(open('public_asset_index.json'))
for line in json.dumps(index, indent=2).splitlines():
    if any(word in line.lower() for word in ('calibration', 'archive', 'emi')):
        print(line.strip())
PY
```
- **Container Exit Code**: `0` (`700 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank 0: leave-one-sweep-out max=9.7e-08; injected EMI max (sweep count, (error, row, spike)): [(5, (2.1229107804998648e-07, 3, 3.01)), (12, (9.699685921127355e-08, 3, 3.01))]
bank 1: leave-one-sweep-out max=3.76e-08; injected EMI max (sweep count, (error, row, spike)): [(5, (9.22699654459791e-08, 16, 3.01)), (12, (3.756086641715228e-08, 16, 3.01))]
bank 2: leave-one-sweep-out max=8e-08; injected EMI max (sweep count, (error, row, spike)): [(5, (3.6382192725170626e-07, 27, 3.01)), (12, (8.003840878956225e-08, 32, 3.01))]
bank 3: leave-one-sweep-out max=3.49e-08; injected EMI max (sweep count, (error, row, spike)): [(5, (1.443695045733051e-07, 37, 3.01)), (12, (3.492306133168199e-08, 38, 3.01))]
"sensor_calibration_sweeps": {

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e1109e102a5a933 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9e1109e1 (directory).
```

### Turn 11 — `submit_completed_task` (`15453 ms` LLM)
- **Rationale**: The unchanged, verified artifact passed isolated execution, independent stencil and per-bank checks, budget and active-bound stress tests, and the latest EMI audit at minimum clean support. The audit found no failure that warrants changing the solver.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_s00_solver.py`. The public validator passed. Maximum absolute errors were 2.15e-7 for the source weights and 1.50e-8 for the terminal field. The worst KKT gap across budget stress tests was 1.72e-7, below the 1e-6 tolerance. All four banks passed the EMI stress test with just four clean calibration points remaining.`

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
