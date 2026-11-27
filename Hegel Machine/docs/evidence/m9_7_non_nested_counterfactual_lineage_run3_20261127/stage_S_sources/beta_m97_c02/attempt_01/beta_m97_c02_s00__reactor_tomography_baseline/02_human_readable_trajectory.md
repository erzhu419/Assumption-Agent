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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `243,022` (`233,113` / `65,408` / `9,909`) |
| **Total Episode Wall Time (s)** | `315.75s` |

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

### Turn 1 — `execute_in_remote_docker` (`18869 ms` LLM)
- **Rationale**: Inspect the public schema and array shapes before coding: a mismatch in sensor stacking, mode layout, or certificate fields would change the implementation. Check archive availability without reading calibration history before a fallback is warranted.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 -I - <<'PY'
import pickle
with open('public_beta_m97_c02_s00.pkl','rb') as f:
    study=pickle.load(f)
for k,v in study.items():
    if hasattr(v,'shape'):
        print(k,type(v).__name__,v.shape,v.dtype,repr(v.ravel()[:5]))
    elif isinstance(v,dict):
        print(k,{a:(type(b).__name__,getattr(b,'shape',None),len(b) if hasattr(b,'__len__') else None) for a,b in v.items()})
    elif isinstance(v,(list,tuple)):
        print(k,type(v).__name__,len(v),repr(v[:1])[:450])
    else:
        print(k,repr(v)[:450])
PY
if [ -d /tmp/hegel_wormhole_skills ]; then PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; fi
ls -l /tmp/hegel_continual_archive/archive_calibration_history.json 2>/dev/null || true
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19032 ms` LLM)
- **Rationale**: Inspect the study dictionaries inside the list and read the rest of the verifier. The array layout determines how to stack sensor responses, and the verifier determines the multiplier sign and certificate tolerances.

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
import numpy as np
with open('public_beta_m97_c02_s00.pkl','rb') as f:
    data=pickle.load(f)
print('container:',type(data).__name__,len(data))
def show(x, depth=0):
    pad='  '*depth
    if isinstance(x,dict):
        for k,v in x.items():
            print(pad,k,end=': ')
            if isinstance(v,(dict,list,tuple)):
                print(type(v).__name__,len(v))
                show(v,depth+1)
            elif isinstance(v,np.ndarray):
                print(v.shape,v.dtype,repr(v.ravel()[:12]))
            else:
                print(repr(v)[:200])
    elif isinstance(x,(list,tuple)) and len(x):
        if isinstance(x[0],(list,tuple,dict)):
            show(x[0],depth+1)
        else:
            print(pad,repr(x[:8])[:200])
show(data)
PY
sed -n '44,160p' validate_public_schema.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
ls -l /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`24318 ms` LLM)
- **Rationale**: The previous output truncated the source constraints and verifier. Check those fields to avoid guessing the input contract, and use per-bank residuals and counts to determine whether any public bank needs the archive fallback.
- **Alternatives Weighed**: `Use the general ridge-profiling primitive` (rejected: The specified projection requires only one small linear solve shared by the observations and all forward columns.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
import numpy as np
item = pickle.load(open('public_beta_m97_c02_s00.pkl', 'rb'))[0]
s = item['study']
print('study keys:', sorted(s))
for k in ('nuisance_ridge', 'source_ridge', 'u_box', 'total_budget', 'recirculation_loop_mode', 'critical_vortex_circulation'):
    if k in s:
        print(k, s[k])
print('max circulation:', max(s['past_vortex_circulation_trace']))
for k, v in item.get('gt', {}).items():
    print('reference:', k, np.shape(v))
c = s['sensor_calibration_sweeps']
for bank in range(4):
    keep = c['bank_id'] == bank
    x, y = c['ref_signal'][keep], c['raw_readout'][keep]
    if x.size >= 2:
        A = np.column_stack((x, np.ones_like(x)))
        p = np.linalg.lstsq(A, y, rcond=None)[0]
        print('bank:', bank, 'n:', x.size, 'fit:', p, 'max abs residual:', max(abs(y - A @ p)))
    else:
        print('bank:', bank, 'n:', x.size)
PY
sed -n '42,78p' validate_public_schema.py
if [ -f /tmp/hegel_continual_archive/archive_calibration_history.json ]; then echo 'Calibration history is available'; else echo 'No calibration history in this container'; fi
```
- **Container Exit Code**: `0` (`374 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
it: [ 9.99999988e-01 -7.05374990e-09] max abs residual: 2.184302421159856e-07
bank: 3 n: 12 fit: [ 1.00000002e+00 -6.34178188e-09] max abs residual: 1.4953273541618017e-07
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

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c02_s00_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c02_s00.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c02_s00(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out: errs.append("missing:kkt_certificate")
    elif not isinstance(out["kkt_certificate"], dict) or not {"weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"} <= set(out["kkt_certificate"].keys()):
        errs.append("bad_kkt_certificate_keys")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
No calibration history in this container
```

### Turn 4 — `execute_in_remote_docker` (`77654 ms` LLM)
- **Rationale**: Implement the specified model and check every numerical output against the public reference. Comparing the projected design and observations separately will distinguish a forward-indexing error from a calibration or projection error if the reference certificate fails.
- **Alternatives Weighed**: `Use a generic constrained optimizer` (rejected: An active-set solve on the 12 weights gives the equality multiplier directly and supports verification of the dual lower bound.)

```bash
set -e
cd /holdout_task
cat > beta_m97_c02_s00_solver.py <<'PY'
"""Calibrated open-Robin reactor tomography with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(bank):
    if not _HISTORY.is_file():
        return None
    with _HISTORY.open() as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get('calibration_history', records.get('history', records))
    if isinstance(records, dict):
        records = records.get(str(bank), [])
        if isinstance(records, dict):
            records = [records]
    for row in reversed(records):
        if int(row.get('bank_id', bank)) == bank and 'gain' in row and 'offset' in row:
            return float(row['gain']), float(row['offset'])
    return None


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    banks, x_all, y_all = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    gains, offsets = np.empty(4), np.empty(4)
    for bank in range(4):
        x, y = x_all[banks == bank], y_all[banks == bank]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, 1)
            distinct = x[i] != x[j]
            if np.any(distinct):
                i, j = i[distinct], j[distinct]
                slope = np.median((y[i] - y[j]) / (x[i] - x[j]))
                delta = y - slope * x
                clean = np.abs(delta - np.median(delta)) <= 3.0
        prior = _historical_calibration(bank) if x.size < 2 or not np.all(clean) else None
        if (clean.sum() >= 4 or prior is None) and np.unique(x[clean]).size >= 2:
            x, y = x[clean], y[clean]
            gain, offset = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
        elif prior is not None:
            gain, offset = prior
        else:
            raise ValueError(f'Insufficient calibration data for bank {bank}')
        gains[bank], offsets[bank] = gain, offset
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[rows]) / gains[rows]


def _axis(kappa, velocity, robin, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    mat = np.zeros((n, n))
    face = (kappa[1:] + kappa[:-1]) / (2.0 * h * h)
    i = np.arange(1, n - 1)
    v = velocity[i] / h
    mat[i, i - 1] = -face[:-1] - np.maximum(v, 0)
    mat[i, i + 1] = -face[1:] - np.maximum(-v, 0)
    mat[i, i] = face[:-1] + face[1:] + abs(v)
    mat[0, 0] = kappa[0] / (h * h) + robin / h
    mat[0, 1] = -kappa[0] / (h * h)
    mat[-1, -1] = kappa[-1] / (h * h) + robin / h
    mat[-1, -2] = -kappa[-1] / (h * h)
    return mat


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'] / (ny - 1))
    operator = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx))
    diag = np.diag_indices(nx * ny)
    operator[diag] += np.asarray(study['V_reaction_2d']).ravel(order='F')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = ix + nx * iy
    blocks = []
    for mu in study['mu_shifts']:
        shifted = operator.copy()
        shifted[diag] += mu
        fields = np.linalg.solve(shifted, rhs)
        blocks.append(fields[sensor_rows])
    return np.vstack(blocks), fields


def _box_simplex(design, target, ridge, upper, budget):
    n = design.shape[1]
    hess = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    if budget == 0:
        state[:] = -1
    elif budget == n * upper:
        state[:] = 1
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        low, high = state < 0, state > 0
        if free.size:
            # Minimize exactly on the current face with its remaining budget.
            hff = hess[np.ix_(free, free)]
            kkt = np.ones((free.size + 1, free.size + 1))
            kkt[:-1, :-1], kkt[:-1, -1], kkt[-1, -1] = hff, -1, 0
            rhs = np.r_[linear[free] - upper * hess[np.ix_(free, high)].sum(axis=1), budget - upper * high.sum()]
            solution = np.linalg.solve(kkt, rhs)
            candidate, nu = solution[:-1], solution[-1]
            direction = candidate - w[free]
            below, above = candidate < 0, candidate > upper
            if np.any(below | above):
                alpha = np.full(free.size, np.inf)
                alpha[below] = -w[free[below]] / direction[below]
                alpha[above] = (upper - w[free[above]]) / direction[above]
                hit = np.argmin(alpha)
                w[free] = np.clip(w[free] + alpha[hit] * direction, 0, upper)
                state[free[hit]] = -1 if below[hit] else 1
                w[free[hit]] = 0 if below[hit] else upper
                continue
            w[free] = candidate
        else:
            gradient = hess @ w - linear
            a = np.min(gradient[low], initial=np.inf)
            b = np.max(gradient[high], initial=-np.inf)
            if b > a + 1e-12:
                state[np.flatnonzero(low)[np.argmin(gradient[low])]] = 0
                state[np.flatnonzero(high)[np.argmax(gradient[high])]] = 0
                continue
            nu = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
        reduced = hess @ w - linear - nu
        violation = state * reduced
        if np.max(violation) <= 1e-12:
            return w, float(nu)
        state[np.argmax(violation)] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c02_s00(study: dict) -> dict:
    if study['recirculation_loop_mode'] != 0 or np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']:
        raise ValueError('Study is outside the open-Robin regime')
    calibrated = _calibrate(study)
    forward, fields = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((calibrated, forward))
    values -= z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)
    target, design = values[:, 0], values[:, 1:]
    ridge, upper, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(design, target, ridge, upper, budget)
    residual = design @ w - target
    primal = 0.5 * float(residual @ residual + ridge * (w @ w))
    reduced = design.T @ residual + ridge * w - nu
    lower, upper_mult = np.maximum(reduced, 0), np.maximum(-reduced, 0)
    # Equivalent to the supporting-hyperplane bound, without cancellation in the gap.
    gap = float(lower @ w + upper_mult @ (upper - w) + nu * (w.sum() - budget))
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=lower, lam_upper=upper_mult,
                primal_objective=primal, dual_lower_bound=primal - max(0.0, gap))
    shape = int(study['nx']), int(study['ny'])
    return dict(weights=w, terminal_field=(fields @ w).reshape(shape, order='F'), kkt_certificate=cert)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pathlib, pickle
import numpy as np
p=pathlib.Path.cwd()
spec=importlib.util.spec_from_file_location('solver',p/'beta_m97_c02_s00_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
study_gt=pickle.loads((p/'public_beta_m97_c02_s00.pkl').read_bytes())[0]; s,g=study_gt['study'],study_gt['gt']
out=m.solve_beta_m97_c02_s00(s)
stack, fields=m._forward(s); y=m._calibrate(s); z=s['Z_nuisance']; a=np.column_stack((y,stack)); a-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
for name,pred,ref in [('design',a[:,1:],g['S_tilde']),('observations',a[:,0],g['r_y']),('weights',out['weights'],g['w_opt']),('field',out['terminal_field'],g['terminal_field_A'])]:
    print(name,'max abs:',float(np.max(np.abs(pred-ref))),'rel norm:',float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)))
ns={}; exec((p/'validate_public_schema.py').read_text().split('\n_d =')[0],ns)
print('reference certificate:',ns['verify_box_simplex_kkt_certificate'](g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']))
print('own certificate:',ns['verify_box_simplex_kkt_certificate'](a[:,1:],a[:,0],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']))
PY
```
- **Container Exit Code**: `0` (`465 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
design max abs: 7.632783294297951e-17 rel norm: 6.079020245808513e-16
observations max abs: 1.0501607983215422e-07 rel norm: 1.1141297716472125e-06
weights max abs: 2.1489762802584522e-07 rel norm: 5.328395721943562e-07
field max abs: 1.5003001729851917e-08 rel norm: 3.164304743787599e-07
reference certificate: {'valid': True, 'reason': 'ok', 'true_primal': 0.017155323171807582, 'verified_dual_lb': 0.017155262395883303, 'gap': 6.077592427933265e-08}
own certificate: {'valid': True, 'reason': 'ok', 'true_primal': 0.017155328264771265, 'verified_dual_lb': 0.017155328264771216, 'gap': 4.85722573273506e-17}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc34768bef7a4de1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc34768b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`40693 ms` LLM)
- **Rationale**: Stress the verified artifact in an isolated Python process. A calibration failure would identify the branch to fix; a certificate failure would identify the budget and active-set regime. The tests reuse public inputs without fetching additional data.
- **Alternatives Weighed**: `Submit after the public reference check` (rejected: The public study has four clean banks and one source budget, so it does not exercise EMI rejection, sparse-bank fallback, or fully saturated constraints.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, json, pathlib, pickle, tempfile
import numpy as np
p = pathlib.Path.cwd()
spec = importlib.util.spec_from_file_location('solver', p/'beta_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}; exec((p/'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
item = pickle.loads((p/'public_beta_m97_c02_s00.pkl').read_bytes())[0]
s, gt = item['study'], item['gt']
base = m.solve_beta_m97_c02_s00(s)
calibrated = m._calibrate(s)
gain, offset = np.array([0.6, 1.4, 1.8, 0.85]), np.array([-0.5, 0.2, -0.9, 0.4])
full = copy.deepcopy(s)
c = full['sensor_calibration_sweeps']; bank = c['bank_id']; rows = full['row_bank_id']
c['raw_readout'] = gain[bank]*c['ref_signal'] + offset[bank]
full['y_obs_raw'] = gain[rows]*calibrated + offset[rows]
full['aux_manifold_telemetry'] = None
worst_cal = worst_field = worst_weights = 0.0
with tempfile.TemporaryDirectory() as d:
    history = pathlib.Path(d)/'history.json'
    m._HISTORY = history
    for label in ('clean', 'EMI', 'sparse archive'):
        candidate = copy.deepcopy(full)
        sweeps = candidate['sensor_calibration_sweeps']
        if label == 'EMI':
            sweeps['raw_readout'][np.flatnonzero(bank == 1)[4]] += 3.1
        if label == 'sparse archive':
            keep = bank != 2
            keep[np.flatnonzero(bank == 2)[0]] = True
            candidate['sensor_calibration_sweeps'] = {k:v[keep] for k,v in sweeps.items()}
            history.write_text(json.dumps({'2': {'gain': gain[2], 'offset': offset[2]}}))
        if label == 'clean':
            history.write_text('not JSON: clean banks must not read this')
        elif label == 'EMI':
            history.unlink()
        out = m.solve_beta_m97_c02_s00(candidate)
        ecal = np.max(np.abs(m._calibrate(candidate)-calibrated))
        ew = np.max(np.abs(out['weights']-base['weights']))
        ef = np.max(np.abs(out['terminal_field']-base['terminal_field']))
        print(label, 'calibration/weights/field max error:', ecal, ew, ef)
        assert max(ecal, ew, ef) < 1e-12
        worst_cal, worst_weights, worst_field = max(worst_cal,ecal), max(worst_weights,ew), max(worst_field,ef)

stack, fields = m._forward(s)
z = s['Z_nuisance']; gram = z.T@z + s['nuisance_ridge']*np.eye(z.shape[1])
design = stack - z@np.linalg.solve(gram,z.T@stack)
rng = np.random.default_rng(97)
max_gap = 0.0
faces = set()
for budget in (0.0, 1e-8, 1.0, 2.0, 2.6, 6.0, 11.0, 12.0-1e-8, 12.0):
    for scale in (0.1, 3.0, 30.0):
        for _ in range(8):
            perturbed = dict(s, total_budget=budget)
            y = scale*rng.standard_normal(rows.size)
            perturbed['y_obs_raw'] = y
            yc = m._calibrate(perturbed)
            target = yc - z@np.linalg.solve(gram,z.T@yc)
            out = m.solve_beta_m97_c02_s00(perturbed)
            c = out['kkt_certificate']; w = out['weights']
            result = verify(design,target,s['source_ridge'],s['u_box'],budget,c)
            assert result['valid'], (budget,scale,result)
            assert np.array_equal(w,c['weights'])
            assert np.max(np.abs(out['terminal_field']-(fields@w).reshape((s['nx'],s['ny']),order='F'))) < 1e-13
            faces.add((int(sum(w < 1e-11)),int(sum(w > 1-1e-11))))
            max_gap = max(max_gap,result['gap'])
print('216 active-set stress cases; distinct lower/upper active counts:',len(faces),'worst certificate gap:',max_gap)
assert len(faces)>10
PY
```
- **Container Exit Code**: `0` (`1952 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clean calibration/weights/field max error: 1.3322676295501878e-15 2.6367796834847468e-15 1.1102230246251565e-16
EMI calibration/weights/field max error: 1.3322676295501878e-15 2.4147350785597155e-15 1.0408340855860843e-16
sparse archive calibration/weights/field max error: 1.3322676295501878e-15 3.0531133177191805e-15 1.3183898417423734e-16
216 active-set stress cases; distinct lower/upper active counts: 30 worst certificate gap: 1.4551915228366852e-11

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc34768bef7a4de1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc34768b (directory).
```

### Turn 6 — `submit_completed_task` (`14829 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process checks against the public numerical reference, plus calibration perturbations and budget stress tests spanning 30 active-bound patterns. No further change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c02_s00_solver.py': 'FILE:7379'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c02_s00_solver.py`. The public schema passed, the terminal-field maximum absolute error was 1.50e-8, and the public KKT certificate gap was 6.08e-8 against a 1e-6 tolerance. All 216 active-set stress cases passed, along with simulated EMI and sparse-bank calibration tests.`

### Turn 7 — `execute_in_remote_docker` (`35671 ms` LLM)
- **Rationale**: Check the terminal PDE directly on the grid, separating wall and interior residuals to locate any stencil or indexing error. Split the public comparison by bank and shift to expose localized calibration or stacking errors that a pooled norm could hide. Change the solver only if a slice fails.
- **Alternatives Weighed**: `Apply relay, drift, or classification diagnostics` (rejected: This task specifies a linear open-Robin boundary-value problem and continuous source inversion; it has no relay states, stochastic drift operator, or classification labels.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np
p = Path('beta_m97_c02_s00_solver.py')
spec = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c02_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
out = m.solve_beta_m97_c02_s00(s)
w = out['weights']
assert np.isfinite(w).all() and 0 <= w.min() <= w.max() <= s['u_box']
assert abs(w.sum() - s['total_budget']) < 1e-7
# Check the terminal PDE directly in grid coordinates, independently of the matrix assembly.
u = out['terminal_field']
def apply_axis(f, k, v, robin, h):
    a = np.empty_like(f)
    dl = f[1:-1] - f[:-2]
    dr = f[1:-1] - f[2:]
    a[1:-1] = ((k[1:-1] + k[:-2])[:,None] * dl + (k[1:-1] + k[2:])[:,None] * dr) / (2*h*h)
    a[1:-1] += (np.maximum(v[1:-1],0)[:,None]*dl + np.maximum(-v[1:-1],0)[:,None]*dr)/h
    a[0] = k[0]*(f[0]-f[1])/h**2 + robin*f[0]/h
    a[-1] = k[-1]*(f[-1]-f[-2])/h**2 + robin*f[-1]/h
    return a
ax = apply_axis(u,s['kappa_x'],s['vel_x'],s['robin_x'],s['Lx']/(s['nx']-1))
ay = apply_axis(u.T,s['kappa_y'],s['vel_y'],s['robin_y'],s['Ly']/(s['ny']-1)).T
rhs = np.einsum('m,mij->ij',w,s['source_modes'])
res = ax + ay + (s['V_reaction_2d']+s['mu_shifts'][-1])*u - rhs
regions = {'interior': res[1:-1,1:-1], 'x walls': res[[0,-1],:], 'y walls': res[:,[0,-1]]}
for name, a in regions.items():
    err = np.max(np.abs(a))
    print(name, 'terminal PDE max residual:', err)
    assert err < 1e-10
S, _ = m._forward(s)
z = s['Z_nuisance']
A = np.column_stack((m._calibrate(s), S))
A -= z @ np.linalg.solve(z.T@z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T@A)
for bank in range(4):
    err = np.max(np.abs(A[s['row_bank_id']==bank,0]-gt['r_y'][s['row_bank_id']==bank]))
    print('bank', bank, 'projected observation max error:', err)
    assert err < 1e-6
for i, idx in enumerate(np.split(np.arange(len(A)), len(s['mu_shifts']))):
    err = np.max(np.abs(A[idx,1:]-gt['S_tilde'][idx]))
    print('shift', i, 'projected forward max error:', err)
    assert err < 1e-12
ns = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], ns)
cert = ns['verify_box_simplex_kkt_certificate'](gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate'])
print('public certificate:', cert)
assert cert['valid']
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
interior terminal PDE max residual: 8.049116928532385e-16
x walls terminal PDE max residual: 2.636779683484747e-16
y walls terminal PDE max residual: 1.8041124150158794e-16
bank 0 projected observation max error: 1.0501607983215422e-07
bank 1 projected observation max error: 7.661794398927668e-08
bank 2 projected observation max error: 4.6758460436890914e-08
bank 3 projected observation max error: 2.623003814150593e-08
shift 0 projected forward max error: 5.551115123125783e-17
shift 1 projected forward max error: 6.245004513516506e-17
shift 2 projected forward max error: 7.632783294297951e-17
shift 3 projected forward max error: 3.469446951953614e-17
public certificate: {'valid': True, 'reason': 'ok', 'true_primal': 0.017155323171807582, 'verified_dual_lb': 0.017155262395883303, 'gap': 6.077592427933265e-08}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc34768bef7a4de1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc34768b (directory).
```

### Turn 8 — `submit_completed_task` (`14651 ms` LLM)
- **Rationale**: Independent wall and interior stencil checks and per-bank and per-shift comparisons passed in an isolated Python process. The same verified artifact passed calibration and active-set stress tests, so no further change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c02_s00_solver.py': 'FILE:7379'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_s00_solver.py`. The public schema passed, the terminal-field maximum absolute error was 1.50e-8, and the public KKT gap was 6.08e-8 (tolerance: 1e-6). All 216 active-set stress cases passed, along with simulated EMI and sparse-bank calibration tests.`

### Turn 9 — `execute_in_remote_docker` (`30873 ms` LLM)
- **Rationale**: The independent stencil audit settled the algebraic checks relevant to this linear PDE. Leave one real calibration measurement out at a time to detect whether a single sweep point controls a bank fit enough to break the downstream certificate. If a slice fails, examine that bank’s calibration leverage before changing the transport model.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('solver', Path('beta_m97_c02_s00_solver.py'))
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('public_beta_m97_c02_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
sweep = s['sensor_calibration_sweeps']
banks = np.asarray(sweep['bank_id'])
ns = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
worst_gap = worst_field = 0.0
for bank in range(4):
    indices = np.flatnonzero(banks == bank)
    if len(indices) < 5:
        print(f'bank {bank}: {len(indices)} sweeps; cannot withhold a row while retaining four training points')
        continue
    gap = field = holdout = 0.0
    for i in indices:
        keep = np.arange(banks.size) != i
        trimmed = {k: np.asarray(v)[keep] for k, v in sweep.items()}
        same = trimmed['bank_id'] == bank
        x = trimmed['ref_signal'][same]
        fit = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), trimmed['raw_readout'][same], rcond=None)[0]
        holdout = max(holdout, abs(fit[0]*sweep['ref_signal'][i] + fit[1] - sweep['raw_readout'][i]))
        out = solver.solve_beta_m97_c02_s00(dict(s, sensor_calibration_sweeps=trimmed))
        result = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
        assert result['valid'], (bank, int(i), result)
        gap = max(gap, result['gap'])
        field = max(field, np.max(np.abs(out['terminal_field'] - gt['terminal_field_A'])))
    print(f'bank {bank}: {len(indices)} held-out sweeps, worst raw error {holdout:.3g}, reference KKT gap {gap:.3g}, field max error {field:.3g}')
    worst_gap, worst_field = max(worst_gap, gap), max(worst_field, field)
print(f'worst held-out calibration slice: reference KKT gap {worst_gap:.6g}, field max error {worst_field:.6g}')
PY
```
- **Container Exit Code**: `0` (`661 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank 0: 12 held-out sweeps, worst raw error 2.62e-07, reference KKT gap 8.36e-08, field max error 1.99e-08
bank 1: 12 held-out sweeps, worst raw error 1.66e-07, reference KKT gap 7.34e-08, field max error 1.62e-08
bank 2: 12 held-out sweeps, worst raw error 2.42e-07, reference KKT gap 8.78e-08, field max error 1.86e-08
bank 3: 12 held-out sweeps, worst raw error 1.63e-07, reference KKT gap 7.32e-08, field max error 1.72e-08
worst held-out calibration slice: reference KKT gap 8.78301e-08, field max error 1.99191e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc34768bef7a4de1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc34768b (directory).
```

### Turn 10 — `submit_completed_task` (`16227 ms` LLM)
- **Rationale**: Every bank passed leave-one-sweep-out validation, and independent interior and wall stencil checks passed. The saved artifact is unchanged from the version verified in isolated Python runs; no further edit is warranted.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_s00_solver.py`. The public schema passed, and the terminal field matched the reference to a maximum absolute error of 1.50e-8. All 216 active-set stress cases and 48 calibration holdouts passed. The worst holdout KKT gap was 8.78e-8, below the 1e-6 tolerance.`

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
