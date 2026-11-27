# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c01_s00__reactor_tomography_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c01_source` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `244,542` (`234,668` / `69,760` / `9,874`) |
| **Total Episode Wall Time (s)** | `315.96s` |

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

### Turn 1 — `execute_in_remote_docker` (`16013 ms` LLM)
- **Rationale**: Inspect the public schema and sampled array shapes to resolve coordinate, stacking, and calibration conventions before implementing the solver; check whether a compatible profiling primitive and calibration archive are available.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
s=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))
for k,v in s.items():
 print(k,type(v).__name__,getattr(v,'shape',None),str(v)[:350] if not isinstance(v,np.ndarray) else (v.dtype, v.ravel()[:5]))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; ls -l /tmp/hegel_continual_archive/
```
- **Container Exit Code**: `2` (`419 ms`, timed_out=`False`)
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
ls: cannot access '/tmp/hegel_continual_archive/': No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`19639 ms` LLM)
- **Rationale**: The public pickle is a list of cases, so inspect its first study directly. The field shapes and validator will determine the sensor stacking and certificate conventions; per-case calibration counts and circulation maxima will show which branches need public validation.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
cases=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))
print('case count:', len(cases))
s=cases[0]['study']
for k,v in s.items():
    if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:4])
    else: print(k,type(v).__name__,str(v)[:450])
for i,c in enumerate(cases):
    s=c['study']; cal=s['sensor_calibration_sweeps']; print('slice',i,'mode',s['recirculation_loop_mode'],'vortex max',max(s['past_vortex_circulation_trace']),'calibration structure',type(cal).__name__,len(cal))
    if isinstance(cal,dict): print({k:np.asarray(v).shape for k,v in cal.items()})
    elif len(cal) and isinstance(cal[0],dict): print({b:sum(row['bank_id']==b for row in cal) for b in range(4)})
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`17240 ms` LLM)
- **Rationale**: The previous log truncated the forward-input shapes and certificate checks. Recover those details to avoid guessing the sensor layout or multiplier sign. Measure the per-bank affine residuals to determine whether the public case exercises the EMI fallback.

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
import numpy as np
c=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]
s=c['study']; gt=c['gt']; cal=s['sensor_calibration_sweeps']
for k,v in s.items():
 a=np.asarray(v) if not isinstance(v,dict) else None
 print(k, (a.shape,a.dtype,a.ravel()[:3]) if a is not None else list(v))
for b in range(4):
 m=cal['bank_id']==b; x=cal['ref_signal'][m]; y=cal['raw_readout'][m]; p=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; print('cal',b,len(x),p,np.max(abs(y-np.c_[x,np.ones(len(x))]@p)))
print('gt weights',gt['w_opt']); print('gt nu',gt['cert_opt']['nu_sum'])
PY
sed -n '44,120p' validate_public_schema.py
```
- **Container Exit Code**: `0` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
64 0.35352301 0.16684891 0.1463776  0.40370263 0.19659516
 0.14651712 0.17912997 0.17041277 0.11822251 0.19300278 0.1813399 ]
gt nu 0.006181370093102531
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
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c01_s00_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c01_s00.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c01_s00(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out: errs.append("missing:kkt_certificate")
    elif not errs:
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
        if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
        elif float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("w_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 4 — `execute_in_remote_docker` (`71858 ms` LLM)
- **Rationale**: Implement the specified operator and certificate. Compare the projected inputs, weights, and terminal field separately against the public reference so any discrepancy identifies which component needs correction.
- **Alternatives Weighed**: `Use a general-purpose constrained optimizer` (rejected: Exact face solves directly recover the equality multiplier and allow a tight certificate.)

```bash
cd /holdout_task && cat > beta_m97_c01_s00_solver.py <<'PY'
"""Open-Robin reactor tomography with a box-simplex KKT certificate."""
from pathlib import Path
import json

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_calibration(x, y):
    # Two clean reference levels determine an affine line. Select the line
    # supported by the most sweep points before applying the EMI threshold.
    best = None
    for i in range(x.size):
        for j in range(i):
            if x[i] == x[j]:
                continue
            slope = (y[i] - y[j]) / (x[i] - x[j])
            residual = np.abs(y - (y[j] + slope * (x - x[j])))
            mask = residual <= 3.0
            score = (int(mask.sum()), -float(np.median(residual)))
            if best is None or score > best[0]:
                best = score, mask
    return best[1] if best is not None else np.zeros(x.size, dtype=bool)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, ref, raw = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    coeffs = np.empty((4, 2))
    for b in range(4):
        x, y = ref[banks == b], raw[banks == b]
        clean = _clean_calibration(x, y)
        has_emi = clean.sum() < x.size and np.unique(x).size >= 2
        prior = None
        if (x.size < 2 or has_emi) and _ARCHIVE.is_file():
            record = json.loads(_ARCHIVE.read_text()).get(str(b))
            if record is not None:
                prior = np.array([record['gain'], record['offset']])
                if has_emi:
                    clean = np.abs(y - (prior[0] * x + prior[1])) <= 3.0
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or prior is None):
            coeffs[b] = _affine(x[clean], y[clean])
        elif prior is not None:
            coeffs[b] = prior
        else:
            raise ValueError(f'Bank {b} needs two distinct clean references or an archived calibration')
    gain, offset = coeffs[np.asarray(study['row_bank_id'])].T
    return (np.asarray(study['y_obs_raw']).reshape(-1) - offset) / gain


def _axis(kappa, velocity, robin, length):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    face = (kappa[:-1] + kappa[1:]) / (2 * h * h)
    diag = np.empty(n)
    diag[1:-1] = face[:-1] + face[1:] + np.abs(velocity[1:-1]) / h
    diag[[0, -1]] = kappa[[0, -1]] / (h * h) + robin / h
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / (h * h)
    upper[0] = -kappa[0] / (h * h)
    return diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = (_axis(study[f'kappa_{a}'], study[f'vel_{a}'], study[f'robin_{a}'], study[f'L{a}']) for a in ('x', 'y'))
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[i + nx * j])
    return np.vstack(blocks), fields


def _box_simplex(H, c, cap, budget):
    """Solve strictly convex QP by exact minimization on successive faces."""
    n = c.size
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = H @ w - c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    side = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(100 * n):
        free = np.flatnonzero(side == 0)
        fixed = np.flatnonzero(side)
        if free.size:
            m = free.size
            system = np.ones((m + 1, m + 1))
            system[:m, :m] = H[np.ix_(free, free)]
            system[:m, m] = -1
            system[m, m] = 0
            rhs = np.r_[c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
            sol = np.linalg.solve(system, rhs)
            target, nu = sol[:-1], float(sol[-1])
            step = target - w[free]
            crossings = np.full(m, np.inf)
            low = target < -1e-13
            high = target > cap + 1e-13
            crossings[low] = -w[free[low]] / step[low]
            crossings[high] = (cap - w[free[high]]) / step[high]
            hit = int(crossings.argmin())
            if crossings[hit] < 1:
                w[free] += max(0.0, crossings[hit]) * step
                side[free[hit]] = -1 if low[hit] else 1
                w[free[hit]] = 0 if low[hit] else cap
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
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
sp=importlib.util.spec_from_file_location('candidate','beta_m97_c01_s00_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for case in pickle.load(open('public_beta_m97_c01_s00.pkl','rb')):
 s,gt=case['study'],case['gt']; o=m.solve_beta_m97_c01_s00(s); raw, f=m._forward(s); z=s['Z_nuisance']; proj=lambda a: a-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
 for name,a,b in [('S',proj(raw),gt['S_tilde']),('r_y',proj(m._calibrate(s)),gt['r_y']),('weights',o['weights'],gt['w_opt']),('terminal',o['terminal_field'],gt['terminal_field_A'])]:
  print(name,'max_abs',np.max(np.abs(a-b)),'relative_l2',np.linalg.norm(a-b)/np.linalg.norm(b))
 print('own gap',o['kkt_certificate']['primal_objective']-o['kkt_certificate']['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
S max_abs 6.245004513516506e-17 relative_l2 6.306782057412317e-16
r_y max_abs 7.150326286620157e-08 relative_l2 7.976578681535388e-07
weights max_abs 2.2727349924256401e-07 relative_l2 5.899364395524786e-07
terminal max_abs 1.7855715184200704e-08 relative_l2 3.076586925357644e-07
own gap 7.632783294297951e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c1ebb5cb870a69d0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c1ebb5cb (directory).
```

### Turn 5 — `execute_in_remote_docker` (`35876 ms` LLM)
- **Rationale**: The only public case has clean near-identity calibration and interior weights. Check the documented sparse-calibration and EMI branches, including whether an endpoint spike just above 3.0 can be masked by the current line fit. Independently check KKT conditions for saturated weights and extreme budgets.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
sp=importlib.util.spec_from_file_location('candidate','beta_m97_c01_s00_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
a=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]['study']
rng=np.random.default_rng(559)
worst=0.; active=0
for i in range(240):
    A=rng.normal(size=(24,12)); H=A.T@A+0.04*np.eye(12); c=rng.normal(size=12)*35
    b=[0., 1., 2.6, 4., 11.7, 12.][i%6]
    w,nu=m._box_simplex(H,c,1.,b); d=H@w-c-nu
    violation=max(abs(w.sum()-b), max(0.,-w.min(), w.max()-1.), np.max(np.where(w<1e-9,np.maximum(-d,0),np.where(w>1-1e-9,np.maximum(d,0),abs(d)))))
    worst=max(worst,violation); active+=bool(np.any((w==0)|(w==1)))
print('QP max KKT residual',worst,'active-box slices',active,flush=True)
assert worst<1e-8
s=copy.deepcopy(a); S,F=m._forward(s); z=s['Z_nuisance']; truth=np.zeros(12); truth[[0,1,2]]=[1,1,.6]
gain=np.array([.72,1.27,1.13,.89]); off=np.array([.3,-.4,.15,-.25]); rb=s['row_bank_id']; y=S@truth+z@rng.normal(size=5); s['source_ridge']=.00001
s['y_obs_raw']=gain[rb]*y+off[rb]; cb=np.repeat(np.arange(4),4); ref=np.tile([-1.5,-.5,.5,1.5],4); raw=gain[cb]*ref+off[cb]; s['sensor_calibration_sweeps']=dict(bank_id=cb,ref_signal=ref,raw_readout=raw.copy()); s['past_vortex_circulation_trace'][0]=s['critical_vortex_circulation']; base=m.solve_beta_m97_c01_s00(s)
cal_error=np.max(abs(m._calibrate(s)-y)); print('nonidentity calibration max error',cal_error,flush=True); assert cal_error<1e-12
with tempfile.TemporaryDirectory() as tmp:
    m._ARCHIVE=Path(tmp)/'archive_calibration_history.json'; m._ARCHIVE.write_text(json.dumps({str(i):dict(gain=gain[i],offset=off[i]) for i in range(4)}))
    sparse=copy.deepcopy(s); keep=(cb!=1)|(ref==-1.5); sparse['sensor_calibration_sweeps']={k:v[keep] for k,v in sparse['sensor_calibration_sweeps'].items()}; got=m.solve_beta_m97_c01_s00(sparse); sparse_err=np.max(abs(got['weights']-base['weights'])); print('single-row archive weight error',sparse_err,flush=True); assert sparse_err<1e-10
    s['sensor_calibration_sweeps']['raw_readout'][3]+=3.1; got=m.solve_beta_m97_c01_s00(s); err=np.max(abs(got['weights']-base['weights'])); cal_err=np.max(abs(m._calibrate(s)-y)); print('3.1-unit EMI weight and calibration max errors',err,cal_err,flush=True); assert cal_err<1e-10 and err<1e-9
PY
```
- **Container Exit Code**: `1` (`520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
QP max KKT residual 1.4210854715202004e-14 active-box slices 240
nonidentity calibration max error 1.7763568394002505e-15
single-row archive weight error 3.219646771412954e-15
3.1-unit EMI weight and calibration max errors 0.9995541977978474 1.7204652688915512

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c1ebb5cb870a69d0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c1ebb5cb (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
AssertionError
```

### Turn 6 — `execute_in_remote_docker` (`47262 ms` LLM)
- **Rationale**: Change only calibration, since the public forward operator and active-bound KKT checks passed. Revalidate the public outputs and test both spike signs near the threshold, every sweep position, and archive fallback for missing or single-row banks.
- **Alternatives Weighed**: `Lower the EMI threshold` (rejected: The threshold is specified; the error is applying it to a fit that already absorbed the spike.)
- **ATMS NOGOOD DECLARED**: Refuted `Maximizing the count within the EMI tolerance identifies the clean calibration line before spike detection.` | Witness: `With four sweep points, one 3.1-unit endpoint spike escaped detection; maximum calibration error was 1.7205 and maximum weight error was 0.9996.` | Blanket Diagnosis: `A tilted affine fit absorbed the single-bank spike into its slope and offset, biasing all observations from that bank.` | Sublated Pivot: `Choose the calibration line by median residual before thresholding; use the resulting clean in-batch fit when at least four clean points remain.`

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
from pathlib import Path
p = Path('beta_m97_c01_s00_solver.py')
s = p.read_text()
s = s.replace('    # Two clean reference levels determine an affine line. Select the line\n    # supported by the most sweep points before applying the EMI threshold.', '    # Estimate the clean line before applying the EMI threshold. Counting\n    # points within 3 units first can reward a tilted line that hides a spike.')
s = s.replace('score = (int(mask.sum()), -float(np.median(residual)))', 'score = (-float(np.median(residual)), int(mask.sum()))')
s = s.replace('                if has_emi:\n                    clean = np.abs(y - (prior[0] * x + prior[1])) <= 3.0\n', '')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
sp = importlib.util.spec_from_file_location('candidate', 'beta_m97_c01_s00_solver.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
case = pickle.load(open('public_beta_m97_c01_s00.pkl', 'rb'))[0]
s = case['study']; gt = case['gt']; out = m.solve_beta_m97_c01_s00(s)
we = np.max(abs(out['weights']-gt['w_opt'])); fe = np.max(abs(out['terminal_field']-gt['terminal_field_A']))
print('Public max weight and terminal errors:', we, fe)
assert we < 3e-7 and fe < 2e-8
s = copy.deepcopy(s)
S, F = m._forward(s); rng = np.random.default_rng(559)
w = np.r_[1., 1., .6, np.zeros(9)]
y = S @ w + s['Z_nuisance'] @ rng.normal(size=5)
gain = np.array([.72, 1.27, 1.13, .89]); off = np.array([.3, -.4, .15, -.25]); rb = s['row_bank_id']
s['y_obs_raw'] = gain[rb]*y + off[rb]; s['source_ridge'] = 1e-5
s['past_vortex_circulation_trace'][0] = s['critical_vortex_circulation']
worst = np.zeros(3); count = 0
with tempfile.TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp)/'archive_calibration_history.json'
    for n in (4, 5, 12):
        cb = np.repeat(np.arange(4), n); ref = np.tile(np.linspace(-1.5, 1.5, n), 4); raw = gain[cb]*ref+off[cb]
        s['sensor_calibration_sweeps'] = dict(bank_id=cb, ref_signal=ref, raw_readout=raw.copy())
        base = m.solve_beta_m97_c01_s00(s)
        # Three clean rows exercise archive fallback; four or more must use
        # in-batch calibration even if archive gains and offsets have drifted.
        drift = 0 if n == 4 else .19
        m._ARCHIVE.write_text(json.dumps({str(i):dict(gain=gain[i]+drift, offset=off[i]-drift) for i in range(4)}))
        for k in range(len(raw)):
            for spike in (-9., -3.0001, 3.0001, 9.):
                s['sensor_calibration_sweeps']['raw_readout'] = raw.copy()
                s['sensor_calibration_sweeps']['raw_readout'][k] += spike
                got = m.solve_beta_m97_c01_s00(s)
                errors = [np.max(abs(m._calibrate(s)-y)), *(np.max(abs(got[key]-base[key])) for key in ('weights', 'terminal_field'))]
                worst = np.maximum(worst, errors); count += 1
    m._ARCHIVE.write_text(json.dumps({str(i):dict(gain=gain[i],offset=off[i]) for i in range(4)}))
    for rows in (0, 1):
        for bank in range(4):
            keep = (cb != bank) | ((np.arange(cb.size) % n) < rows)
            s['sensor_calibration_sweeps'] = dict(bank_id=cb[keep], ref_signal=ref[keep], raw_readout=raw[keep])
            got = m.solve_beta_m97_c01_s00(s)
            worst = np.maximum(worst, [np.max(abs(m._calibrate(s)-y)), *(np.max(abs(got[key]-base[key])) for key in ('weights', 'terminal_field'))]); count += 1
print('Calibration stress slices:', count, 'max observation, weight, field errors:', worst)
assert np.all(worst < [1e-12, 1e-9, 1e-10])
PY
```
- **Container Exit Code**: `0` (`4426 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public max weight and terminal errors: 2.2727349924256401e-07 1.7855715184200704e-08
Calibration stress slices: 344 max observation, weight, field errors: [3.55271368e-15 2.16493490e-14 7.77156117e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=88d74104227685f1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_88d74104 (directory).
```

### Turn 7 — `submit_completed_task` (`13673 ms` LLM)
- **Rationale**: The final artifact passed isolated public validation and corrected the documented EMI failure without changing the public output errors. Archive fallback and active box constraints passed their targeted stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:6884'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c01_s00_solver.py`. The public validator passed. Maximum public errors were 2.27e-7 for weights and 1.79e-8 for the terminal field. All 344 calibration stress cases and 240 box-simplex KKT stress cases passed.`

### Turn 8 — `execute_in_remote_docker` (`37461 ms` LLM)
- **Rationale**: Use the public measurements to check for boundary, ordering, and shift errors that could survive a shape check, and for bank-specific calibration errors that a pooled metric could hide. If a discrepancy appears, correct the implicated component; otherwise retain the verified candidate.
- **Alternatives Weighed**: `Test relay modes or compute within-batch classification accuracy` (rejected: The task specifies an open-Robin inverse problem, not relay switching or classification; the relevant grouped diagnostic is projected observation error by sensor bank.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for case in pickle.load(open('public_beta_m97_c01_s00.pkl', 'rb')):
    s, gt = case['study'], case['gt']
    out = m.solve_beta_m97_c01_s00(s)
    nx, ny = s['nx'], s['ny']
    v = s['V_reaction_2d']
    axes = [(s[f'kappa_{a}'], s[f'vel_{a}'], s[f'robin_{a}'], s[f'L{a}'] / (n-1)) for a, n in zip('xy', (nx, ny))]
    dense = np.zeros((nx*ny, nx*ny))
    balance = np.empty(nx*ny)
    for j in range(ny):
        for i in range(nx):
            row = i + nx*j
            dense[row, row] = balance[row] = v[i,j]
            for q, stride, (kap, vel, rob, h) in zip((i,j), (1,nx), axes):
                if q == 0 or q == len(kap)-1:
                    direction = 1 if q == 0 else -1
                    d = kap[q]/h**2
                    dense[row,row] += d + rob/h
                    dense[row,row+direction*stride] -= d
                    balance[row] += rob/h
                else:
                    for direction in (-1,1):
                        d = (kap[q]+kap[q+direction])/(2*h**2) + max(-direction*vel[q],0)/h
                        dense[row,row] += d
                        dense[row,row+direction*stride] -= d
    balance_err = np.max(abs(dense.sum(axis=1)-balance))
    assert balance_err < 1e-12
    rhs = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
    sensor_i, sensor_j = s['sensor_ij'].T
    blocks = []
    for shift in s['mu_shifts']:
        f = np.linalg.solve(dense + shift*np.eye(nx*ny), rhs)
        blocks.append(f[sensor_i+nx*sensor_j])
    stack, _ = m._forward(s)
    stack_err = np.max(abs(np.vstack(blocks)-stack))
    assert stack_err < 1e-12
    terminal_err = np.max(abs((f@out['weights']).reshape((nx,ny),order='F')-out['terminal_field']))
    assert terminal_err < 1e-12
    z = s['Z_nuisance']
    y = m._calibrate(s)
    r = y - z @ np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]), z.T@y)
    bank_errors = {b: float(np.max(abs((r-gt['r_y'])[s['row_bank_id']==b]))) for b in range(4)}
    assert max(bank_errors.values()) < 1e-6
    cert = out['kkt_certificate']; w = out['weights']; g = gt['S_tilde'].T@(gt['S_tilde']@w-gt['r_y'])+s['source_ridge']*w
    stationarity = np.max(abs(g-cert['nu_sum']-np.asarray(cert['lam_lower'])+np.asarray(cert['lam_upper'])))
    assert stationarity < 1e-6
    print('Robin row-balance error:', balance_err, '| dense comparison, all shifts:', stack_err, '| terminal field:', terminal_err)
    print('Projected observation max error by bank:', bank_errors, '| reference KKT stationarity:', stationarity)
    print('Public regime:', s['recirculation_loop_mode'], '| circulation margin:', s['critical_vortex_circulation']-max(s['past_vortex_circulation_trace']))
PY
```
- **Container Exit Code**: `0` (`615 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Robin row-balance error: 1.1213252548714081e-14 | dense comparison, all shifts: 8.326672684688674e-17 | terminal field: 4.85722573273506e-17
Projected observation max error by bank: {0: 5.883198372913512e-08, 1: 7.150326286620157e-08, 2: 4.652717500164272e-08, 3: 5.371307620904986e-08} | reference KKT stationarity: 6.366350567468121e-08
Public regime: 0.0 | circulation margin: 0.18641000000000002

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=88d74104227685f1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_88d74104 (directory).
```

### Turn 9 — `submit_completed_task` (`13397 ms` LLM)
- **Rationale**: Isolated validation of the unchanged final artifact confirmed the forward stencil, all four calibration banks, and the KKT certificate. The EMI failure is resolved, and the requested open-outflow regime has been verified.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:6884'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The public validator passed, with maximum absolute errors of 2.27e-7 for weights and 1.79e-8 for the terminal field. All 344 calibration and 240 box-simplex stress cases passed.`

### Turn 10 — `submit_completed_task` (`15503 ms` LLM)
- **Rationale**: The unchanged artifact already passed the requested clean-process audit. Independent stencil assembly checked boundary and coordinate conventions, and bank-wise residuals checked that pooled error did not hide a calibration failure: the worst bank residual was 7.15e-8. Relay modes and classification metrics do not apply to this inverse problem.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py` for the specified open-outflow Robin regime. The public validator passed. Maximum absolute errors were 2.27e-7 for the weights and 1.79e-8 for the terminal field. All 344 calibration stress cases and 240 box-simplex KKT stress cases passed.`

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
