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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `299,482` (`288,081` / `94,464` / `11,401`) |
| **Total Episode Wall Time (s)** | `364.27s` |

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

### Turn 1 — `execute_in_remote_docker` (`17598 ms` LLM)
- **Rationale**: Inspect the schema and compact array summaries to detect ordering or interface mismatches before implementation, and check whether a reusable ridge-profiling primitive is available.
- **Alternatives Weighed**: `Implement immediately from the equation summary` (rejected: The actual sensor layout, source dictionary, and validator conventions determine how the forward rows and certificate must be assembled.)

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c01_s00.pkl','rb') as f: s=pickle.load(f)
for k,v in s.items():
    if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:8])
    elif isinstance(v,(list,tuple)): print(k,type(v).__name__,len(v),repr(v[:1])[:500])
    elif isinstance(v,dict): print(k,[(x,getattr(y,'shape',type(y).__name__)) for x,y in v.items()])
    else: print(k,repr(v)[:500])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; ls -l /tmp/hegel_continual_archive
```
- **Container Exit Code**: `2` (`426 ms`, timed_out=`False`)
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
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
ls: cannot access '/tmp/hegel_continual_archive': No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`17169 ms` LLM)
- **Rationale**: Read the public studies from their list wrapper and inspect the validator and compact case summaries to resolve field names, sensor layout, and calibration coverage before implementation.
- **Alternatives Weighed**: `Infer sensor stacking from array dimensions alone` (rejected: A wrong row order can pass shape validation while invalidating the inversion.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_beta_m97_c01_s00.pkl','rb') as f:
    cases = pickle.load(f)
print('case count:',len(cases))
for i,c in enumerate(cases):
    print('case',i,'keys',list(c))
    for k,v in c['study'].items():
        if isinstance(v,np.ndarray):
            print(k,v.shape,v.dtype,v.ravel()[:6])
        else:
            print(k,repr(v)[:650])
    print('reference weights:',c['gt']['w_opt'])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`484 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`16608 ms` LLM)
- **Rationale**: Print compact field layouts, calibration rows, and only the relevant validator lines to resolve sensor ordering, archive fallback needs, and the certificate convention before implementation.
- **Alternatives Weighed**: `Write the solver from the truncated schema` (rejected: Missing sensor sampling details could force an incorrect forward matrix, and missing certificate conventions could reverse the equality multiplier sign.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
cs=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))
for i,c in enumerate(cs):
 s=c['study']; print('case',i,'gt keys',list(c['gt']))
 for k,v in s.items():
  if isinstance(v,dict): print(k,[(j,np.shape(a),str(a)[:160]) for j,a in v.items()]); continue
  if isinstance(v,(list,tuple)):
   print(k,'len',len(v),'first',repr(v[0])[:250] if len(v) else None)
   if 'calibration' in k:
    for q in v: print(' ',repr(q)[:260])
   continue
  if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:3]); continue
  print(k,repr(v)[:200])
PY
sed -n '45,100p' validate_public_schema.py
```
- **Container Exit Code**: `0` (`363 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dout', (48,), '[ 0.27426682  1.96391203  0.49602187  0.37798807 -0.94595092  1.66443524\n  0.40193199 -0.66203787  0.38894191  1.21400081  1.65295903 -1.26318102\n -0.54327801 -')]
Z_nuisance (144, 5) float64 [ 1.58901831  0.97499717 -1.89738988]
y_obs_raw (144,) float64 [ 2.37072802 -0.37833314 -0.28587477]
aux_manifold_telemetry (144, 3) float64 [-1.217029 -0.62667  -0.47297 ]
nuisance_ridge 0.65
source_ridge 0.04
u_box 1.0
total_budget 2.6
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
    elif not isinstance(out["kkt_certificate"], dict) or not {"weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"} <= set(out["kkt_certificate"].keys()):
        errs.append("bad_kkt_certificate_keys")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 4 — `execute_in_remote_docker` (`74308 ms` LLM)
- **Rationale**: Implement the solver, then compare the forward matrix, calibrated residual, weights, and terminal field independently. A forward-matrix mismatch points to the discretization or stacking; a residual mismatch points to calibration; an invalid certificate points to the optimizer or multiplier convention.
- **Alternatives Weighed**: `Use a generic constrained optimizer` (rejected: The small quadratic admits direct active-set solves that identify tight bounds and yield the equality multiplier for the certificate.); `Import the ridge-profiling primitive` (rejected: One multi-right-hand-side solve implements the specified five-column ridge projection directly without computing an unused whitener.)

```bash
cd /holdout_task && cat > beta_m97_c01_s00_solver.py <<'PY'
"""Open-Robin reactor tomography with a box-simplex optimality certificate."""

from pathlib import Path
import json

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _archived_affine(bank):
    history = json.loads(_CALIBRATION_ARCHIVE.read_text())
    if isinstance(history, dict):
        history = history.get('banks', history.get('calibration_history', history))
    if isinstance(history, dict) and str(bank) in history:
        rows = history[str(bank)]
        if isinstance(rows, dict):
            rows = [rows]
    else:
        rows = [row for row in history if int(row['bank_id']) == bank]
    if not rows:
        raise ValueError(f'No archived calibration for bank {bank}')
    if 'gain' in rows[-1] and 'offset' in rows[-1]:
        return np.array([rows[-1]['gain'], rows[-1]['offset']], dtype=float)
    return _affine(np.array([r['ref_signal'] for r in rows]).ravel(),
                   np.array([r['raw_readout'] for r in rows]).ravel())


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    y = np.empty(len(row_banks))
    for bank in np.unique(row_banks):
        x, z = ref[banks == bank], raw[banks == bank]
        if len(x) < 2:
            gain, offset = _archived_affine(bank)
        else:
            gain, offset = _affine(x, z)
            if len(x) >= 3:
                i, j = np.triu_indices(len(x), 1)
                distinct = x[i] != x[j]
                i, j = i[distinct], j[distinct]
                if len(i):
                    slope = np.median((z[i] - z[j]) / (x[i] - x[j]))
                    intercept = np.median(z - slope * x)
                    clean = np.abs(z - slope * x - intercept) <= 3.0
                    if not np.all(clean):
                        if clean.sum() >= 4:
                            gain, offset = _affine(x[clean], z[clean])
                        else:
                            gain, offset = _archived_affine(bank)
        take = row_banks == bank
        y[take] = (np.asarray(study['y_obs_raw'])[take] - offset) / gain
    return y


def _axis_operator(length, kappa, velocity, robin):
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    h = length / (n - 1)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    v = np.asarray(velocity, dtype=float) / h
    lower = -face - np.maximum(v[1:], 0)
    upper = -face - np.maximum(-v[:-1], 0)
    main = np.empty(n)
    main[1:-1] = face[:-1] + face[1:] + np.abs(v[1:-1])
    main[[0, -1]] = kappa[[0, -1]] / h**2 + robin / h
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    axes = [_axis_operator(study['L' + d], study['kappa_' + d],
                           study['vel_' + d], study['robin_' + d]) for d in 'xy']
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx))
          + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = modes.transpose(0, 2, 1).reshape(m, nx * ny).T
    sensor_i, sensor_j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = sensor_i + nx * sensor_j
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(op + mu * identity).solve(rhs)
        blocks.append(fields[sensor_rows])
    return np.vstack(blocks), fields


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        w.fill(0 if budget == 0 else cap)
        g = H @ w - c
        return w, float(np.min(g) if budget == 0 else np.max(g))
    # Active state: -1 is lower, +1 is upper, 0 is free.
    state = np.zeros(n, dtype=int)
    for _ in range(20 * n**2):
        f = np.flatnonzero(state == 0)
        a = np.flatnonzero(state != 0)
        if not len(f):
            g = H @ w - c
            low, high = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = low[np.argmin(g[low])], high[np.argmax(g[high])]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            state[[i, j]] = 0
            continue
        k = len(f)
        K = np.empty((k + 1, k + 1))
        K[:k, :k] = H[np.ix_(f, f)]
        K[:k, k], K[k, :k], K[k, k] = -1, 1, 0
        sol = np.linalg.solve(K, np.r_[c[f] - H[np.ix_(f, a)] @ w[a],
                                         budget - w[a].sum()])
        target, nu = sol[:-1], sol[-1]
        delta = target - w[f]
        below, above = target < -1e-13, target > cap + 1e-13
        if np.any(below | above):
            step = np.full(k, np.inf)
            step[below] = -w[f[below]] / delta[below]
            step[above] = (cap - w[f[above]]) / delta[above]
            hit = np.argmin(step)
            w[f] = np.clip(w[f] + step[hit] * delta, 0, cap)
            state[f[hit]] = -1 if below[hit] else 1
            w[f[hit]] = 0 if below[hit] else cap
            continue
        w[f] = np.clip(target, 0, cap)
        violation = state * (H @ w - c - nu)
        release = np.argmax(violation)
        if violation[release] <= 1e-12:
            return w, float(nu)
        state[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_s00(study: dict) -> dict:
    """Recover source weights and the last-shift field for the open reactor."""
    if (study['recirculation_loop_mode'] != 0 or
        np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('This solver requires the open Robin reactor regime')
    y = _calibrated_observations(study)
    S, fields = _forward(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((y, S))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residualized = values - Z @ np.linalg.solve(gram, Z.T @ values)
    y, S = residualized[:, 0], residualized[:, 1:]
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(S, y, ridge, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    primal = float(0.5 * (r @ r + ridge * (w @ w)))
    shifted_g = g - nu
    # Evaluate the tangent bound in gap form to avoid large cancellations.
    gap = np.maximum(shifted_g, 0) @ w + np.maximum(-shifted_g, 0) @ (cap - w)
    certificate = {'weights': w.copy(), 'nu_sum': nu,
                   'lam_lower': np.maximum(shifted_g, 0),
                   'lam_upper': np.maximum(-shifted_g, 0),
                   'primal_objective': primal, 'dual_lower_bound': float(primal - gap)}
    field = (fields @ w).reshape(study['source_modes'].shape[1:], order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': certificate}
PY
python3 -I /holdout_task/validate_public_schema.py && python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
def load(name):
 s=importlib.util.spec_from_file_location(name,p+name+'.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m=load('beta_m97_c01_s00_solver'); v=load('validate_public_schema')
for item in pickle.load(open(p+'public_beta_m97_c01_s00.pkl','rb')):
 s,gt=item['study'],item['gt']; out=m.solve_beta_m97_c01_s00(s)
 y=m._calibrated_observations(s); S,_=m._forward(s); Z=s['Z_nuisance']; A=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); P=lambda X:X-Z@np.linalg.solve(A,Z.T@X)
 print('max errors S, y, w, field:',*[float(np.max(np.abs(x-z))) for x,z in [(P(S),gt['S_tilde']),(P(y),gt['r_y']),(out['weights'],gt['w_opt']),(out['terminal_field'],gt['terminal_field_A'])]])
 print(v.verify_box_simplex_kkt_certificate(gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']))
PY
```
- **Container Exit Code**: `0` (`526 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e08cbb266d27fd2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e08cbb2 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`21953 ms` LLM)
- **Rationale**: Importing the validator ran its CLI and exited before the numerical comparisons. Load the independent verifier without the CLI, then compare each output to the public reference. A forward-matrix mismatch calls for inspecting discretization or stacking; an observation-residual mismatch calls for inspecting calibration or nuisance projection.
- **Alternatives Weighed**: `Submit on the schema pass` (rejected: The schema validator checks shapes and keys, not numerical agreement or certificate validity.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from pathlib import Path
spec=importlib.util.spec_from_file_location('solver', Path('beta_m97_c01_s00_solver.py').resolve())
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# Load the verifier function without running the validator's CLI, which calls sys.exit.
ns={}; exec(Path('validate_public_schema.py').read_text().split('\n_d =',1)[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
for i,item in enumerate(pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))):
 s,gt=item['study'],item['gt']; out=m.solve_beta_m97_c01_s00(s)
 y=m._calibrated_observations(s); S,_=m._forward(s); Z=s['Z_nuisance']; A=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); P=lambda X:X-Z@np.linalg.solve(A,Z.T@X)
 for name,x,z in [('S_tilde',P(S),gt['S_tilde']),('r_y',P(y),gt['r_y']),('weights',out['weights'],gt['w_opt']),('terminal_field',out['terminal_field'],gt['terminal_field_A'])]:
  e=np.abs(x-z); print(i,name,'max_abs',float(e.max()),'worst_shift_max_abs', e.reshape(len(s['mu_shifts']),-1).max(axis=1) if name in ('S_tilde','r_y') else 'n/a')
 print(i,'independent cert',verify(gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']))
PY
```
- **Container Exit Code**: `0` (`415 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 S_tilde max_abs 6.245004513516506e-17 worst_shift_max_abs [4.16333634e-17 5.55111512e-17 6.24500451e-17 4.85722573e-17]
0 r_y max_abs 7.150326286620157e-08 worst_shift_max_abs [5.88319837e-08 6.00221393e-08 5.54610349e-08 7.15032629e-08]
0 weights max_abs 2.2727349924256401e-07 worst_shift_max_abs n/a
0 terminal_field max_abs 1.7855715184200704e-08 worst_shift_max_abs n/a
0 independent cert {'valid': True, 'reason': 'ok', 'true_primal': 0.01925475754670141, 'verified_dual_lb': 0.0192546744947323, 'gap': 8.305196911098656e-08}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e08cbb266d27fd2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e08cbb2 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34279 ms` LLM)
- **Rationale**: Perturb the public study with nontrivial bank gains, an EMI spike, and a sparse bank to detect calibration failures without changing the expected reconstruction. Test mixed and endpoint budgets to detect active-set feasibility or multiplier errors; revise only the component that fails.
- **Alternatives Weighed**: `Submit on the public numerical pass` (rejected: The public case has four well-sampled banks and interior source weights, so it does not exercise archive fallback, EMI handling, or active bounds.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
p=Path('beta_m97_c01_s00_solver.py').resolve()
spec=importlib.util.spec_from_file_location('solver', p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec(Path('validate_public_schema.py').read_text().split('\n_d =',1)[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
st=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]['study']
base=m.solve_beta_m97_c01_s00(st)
y=m._calibrated_observations(st)
sw=st['sensor_calibration_sweeps']; bank=sw['bank_id']; rows=st['row_bank_id']
gains=np.array([0.73,1.53,0.91,1.21]); offsets=np.array([0.4,-0.85,0.32,-0.14]); raw=gains[bank]*sw['ref_signal']+offsets[bank]
raw[np.flatnonzero(bank==2)[5]]+=3.05
keep=(bank!=3); keep[np.flatnonzero(bank==3)[0]]=True
var=dict(st, y_obs_raw=gains[rows]*y+offsets[rows], aux_manifold_telemetry=np.zeros_like(st['aux_manifold_telemetry']), sensor_calibration_sweeps={'bank_id':bank[keep],'ref_signal':sw['ref_signal'][keep],'raw_readout':raw[keep]})
with tempfile.TemporaryDirectory() as d:
    m._CALIBRATION_ARCHIVE=Path(d)/'archive_calibration_history.json'
    m._CALIBRATION_ARCHIVE.write_text(json.dumps([{'bank_id':3,'gain':gains[3],'offset':offsets[3]}]))
    out=m.solve_beta_m97_c01_s00(var)
    err=max(np.max(np.abs(out[k]-base[k])) for k in ('weights','terminal_field'))
    print('combined affine calibration, EMI, sparse bank, auxiliary invariance: max output change',err)
    assert err < 1e-10
rng=np.random.default_rng(472)
worst=0.; bound_count=0
for i in range(200):
    S=rng.normal(size=(24,12)); y=10*rng.normal(size=24); cap=0.8; ridge=0.04
    budget=[0.,0.13,2.4,3.7,9.57,12*cap][i%6]
    w,nu=m._box_simplex(S,y,ridge,cap,budget)
    assert np.min(w)>=0 and np.max(w)<=cap and abs(w.sum()-budget)<1e-10
    r=S@w-y; g=S.T@r+ridge*w-nu; ll=np.maximum(g,0); lu=np.maximum(-g,0)
    J=float(0.5*(r@r+ridge*(w@w))); gap=float(ll@w+lu@(cap-w)); worst=max(worst,gap)
    if 0<budget<12*cap and np.any(w==0) and np.any(w==cap): bound_count+=1
    check=verify(S,y,ridge,cap,budget,dict(weights=w,nu_sum=nu,lam_lower=ll,lam_upper=lu,primal_objective=J,dual_lower_bound=J-gap))
    assert check['valid'], (i,check)
assert bound_count>0
print('200 active-set problems: worst certificate gap',worst,'mixed-bound cases',bound_count)
PY
```
- **Container Exit Code**: `0` (`524 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
combined affine calibration, EMI, sparse bank, auxiliary invariance: max output change 4.829470157119431e-15
200 active-set problems: worst certificate gap 2.9345427426207555e-14 mixed-bound cases 66

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e08cbb266d27fd2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e08cbb2 (directory).
```

### Turn 7 — `submit_completed_task` (`16217 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema and public-reference checks. Targeted stress tests exercised EMI filtering, sparse-bank archive fallback, and mixed and endpoint box constraints without exposing a failure that warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:7389'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The public terminal field matched within 1.79e-8, and the independently verified KKT gap was 8.31e-8, below the 1e-6 tolerance. Calibration stress tests and all 200 box-simplex stress cases passed.`

### Turn 8 — `execute_in_remote_docker` (`27102 ms` LLM)
- **Rationale**: The previous EMI stress test used a well-sampled bank. At exactly four sweep rows, one spike contaminates half the pairwise slopes and could mask the archive trigger. Test both spike signs at every retained position on the public reference grids; if recovery fails, revise spike detection without changing the verified forward model or optimizer.
- **Alternatives Weighed**: `Test relay modes or within-cohort classification accuracy` (rejected: The contract specifies one open Robin regime and a continuous inversion, not relay dynamics or labeled cohorts. The public forward matrix already matches across every shift to 6.25e-17.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('solver',Path('beta_m97_c01_s00_solver.py').resolve()); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; bank=np.asarray(sw['bank_id']); ref=np.asarray(sw['ref_signal']); rows=np.asarray(s['row_bank_id']); y=m._calibrated_observations(s)
gain=np.array([.73,1.53,.91,1.21]); offset=np.array([.4,-.85,.32,-.14]); raw=gain[bank]*ref+offset[bank]
with tempfile.TemporaryDirectory() as d:
 m._CALIBRATION_ARCHIVE=Path(d)/'archive_calibration_history.json'; m._CALIBRATION_ARCHIVE.write_text(json.dumps([dict(bank_id=b,gain=gain[b],offset=offset[b]) for b in range(4)]))
 for n in (4,5):
  errors=[]
  for b in range(4):
   take=np.flatnonzero(bank==b); subset=take[np.linspace(0,len(take)-1,n,dtype=int)]; keep=bank!=b; keep[subset]=True
   for ix in subset:
    for spike in (-3.05,0.,3.05):
     x=raw.copy(); x[ix]+=spike
     trial=dict(s,y_obs_raw=gain[rows]*y+offset[rows],sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=ref[keep],raw_readout=x[keep]))
     err=float(np.max(np.abs(m._calibrated_observations(trial)-y))); errors.append((err,b,int(ix),spike))
  print('sweep rows',n,'failures',sum(e[0]>1e-10 for e in errors),'of',len(errors),'worst (error, bank, row, spike)',max(errors))
PY
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sweep rows 4 failures 16 of 48 worst (error, bank, row, spike) (136.4456377711008, 1, 23, -3.05)
sweep rows 5 failures 0 of 60 worst (error, bank, row, spike) (3.1086244689504383e-15, 2, 26, 3.05)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e08cbb266d27fd2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e08cbb2 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`49694 ms` LLM)
- **Rationale**: Fix the failing calibration slice without changing the verified transport or optimizer. Check archive calls and end-to-end output on one-, four-, and five-row sweeps, then independently verify that the isolated solver still passes the public certificate check.
- **Alternatives Weighed**: `Always use the archive for four-row banks` (rejected: Four clean in-batch points must be fitted directly, and a clean bank should not depend on the archive.)
- **ATMS NOGOOD DECLARED**: Refuted `The median of pairwise slopes reliably detects a single threshold-crossing EMI spike in a four-row calibration sweep.` | Witness: `On public reference grids, the estimator failed 16 of 48 four-row perturbations; the worst was a negative 3.05 spike in bank 1.` | Blanket Diagnosis: `One contaminated row affects three of six slopes, so the median can absorb the spike into the fitted gain and fail to trigger archive fallback.` | Sublated Pivot: `Fit each three-row subset, test the held-out residual from the best-fitting subset, and use the archive if it exceeds the EMI threshold.`

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c01_s00_solver.py')
s = p.read_text()
old = '''            if len(x) >= 3:
                i, j = np.triu_indices(len(x), 1)'''
new = '''            if len(x) == 4:
                # One spike contaminates half the pairwise slopes. Fit each
                # three-row subset and test the omitted row of the best fit.
                best_error = np.inf
                for omitted in range(4):
                    kept = np.arange(4) != omitted
                    trial_gain, trial_offset = _affine(x[kept], z[kept])
                    residual = z - (trial_gain * x + trial_offset)
                    error = residual[kept] @ residual[kept]
                    if error < best_error:
                        best_error = error
                        omitted_residual = residual[omitted]
                if abs(omitted_residual) > 3.0:
                    gain, offset = _archived_affine(bank)
            elif len(x) >= 3:
                i, j = np.triu_indices(len(x), 1)'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('solver',Path('beta_m97_c01_s00_solver.py').resolve()); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec(Path('validate_public_schema.py').read_text().split('\n_d =',1)[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
for item in pickle.load(open('public_beta_m97_c01_s00.pkl','rb')):
 s,gt=item['study'],item['gt']; base=m.solve_beta_m97_c01_s00(s)
 audit=verify(gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],base['kkt_certificate']); assert audit['valid'],audit
 print('public certificate gap',audit['gap'],'max weight error',np.max(np.abs(base['weights']-gt['w_opt'])),'max field error',np.max(np.abs(base['terminal_field']-gt['terminal_field_A'])))
 sw=s['sensor_calibration_sweeps']; bank=np.asarray(sw['bank_id']); ref=np.asarray(sw['ref_signal']); rows=np.asarray(s['row_bank_id']); y=m._calibrated_observations(s)
 gain=np.array([.73,1.53,.91,1.21]); offset=np.array([.4,-.85,.32,-.14]); raw=gain[bank]*ref+offset[bank]
 archive_reader=m._archived_affine; calls=[]
 def tracked(b):
  calls.append(b)
  return archive_reader(b)
 m._archived_affine=tracked
 with tempfile.TemporaryDirectory() as d:
  m._CALIBRATION_ARCHIVE=Path(d)/'archive_calibration_history.json'; m._CALIBRATION_ARCHIVE.write_text(json.dumps([dict(bank_id=b,gain=gain[b],offset=offset[b]) for b in range(4)]))
  for n in (1,4,5):
   errors=[]; output_errors=[]
   for b in range(4):
    take=np.flatnonzero(bank==b); subset=take[np.linspace(0,len(take)-1,n,dtype=int)]; keep=bank!=b; keep[subset]=True
    for ix in subset:
     for spike in ((0.,) if n==1 else (-3.05,0.,3.05)):
      x=raw.copy(); x[ix]+=spike
      trial=dict(s,y_obs_raw=gain[rows]*y+offset[rows],sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=ref[keep],raw_readout=x[keep]))
      calls.clear(); errors.append(float(np.max(np.abs(m._calibrated_observations(trial)-y))))
      expected=[b] if n==1 or (n==4 and spike!=0) else []
      assert calls==expected,(n,b,ix,spike,calls,expected)
      out=m.solve_beta_m97_c01_s00(trial)
      output_errors.append(max(float(np.max(np.abs(out[k]-base[k]))) for k in ('weights','terminal_field')))
   print('sweep rows',n,'trials',len(errors),'max calibrated error',max(errors),'max weight/field change',max(output_errors))
   assert max(errors)<1e-10 and max(output_errors)<1e-10
 m._archived_affine=archive_reader
PY
```
- **Container Exit Code**: `0` (`1094 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public certificate gap 8.305196911098656e-08 max weight error 2.2727349924256401e-07 max field error 1.7855715184200704e-08
sweep rows 1 trials 4 max calibrated error 2.6645352591003757e-15 max weight/field change 5.190292640122607e-15
sweep rows 4 trials 48 max calibrated error 2.6645352591003757e-15 max weight/field change 6.356026815979021e-15
sweep rows 5 trials 60 max calibrated error 3.1086244689504383e-15 max weight/field change 6.689093723366568e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f929bfb8a0dc5c74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f929bfb8 (directory).
```

### Turn 10 — `submit_completed_task` (`14147 ms` LLM)
- **Rationale**: The final saved artifact passed isolated schema and numerical verification. The calibration fix eliminated all 16 previously observed four-row failures, preserved direct fitting for clean banks, and left the verified transport and optimizer unchanged.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:8153'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The public KKT gap was 8.31e-8, below the 1e-6 tolerance, and the maximum terminal-field error was 1.79e-8. The calibration fix passed all 48 four-row trials, and all 200 box-simplex stress cases passed.`

### Turn 11 — `execute_in_remote_docker` (`31050 ms` LLM)
- **Rationale**: Hold out each real calibration row and propagate the refit through the solver. A concentrated bank error would motivate a calibration change; a forward mismatch would instead point to stencil assembly or row ordering. The independent certificate check tests whether any held-out refit consumes the public tolerance margin.
- **Alternatives Weighed**: `Fit relay dynamics or compute within-batch classification scores` (rejected: The contract specifies open Robin transport and continuous observations. The relevant nuisance checks are held-out affine calibration errors and residual agreement within each bank.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', Path('beta_m97_c01_s00_solver.py').resolve())
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns = {}; exec(Path('validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
for i, item in enumerate(pickle.load(open('public_beta_m97_c01_s00.pkl', 'rb'))):
    s, gt = item['study'], item['gt']
    base = m.solve_beta_m97_c01_s00(s)
    sw = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    bank = sw['bank_id']; rows = np.asarray(s['row_bank_id'])
    y = m._calibrated_observations(s); S, _ = m._forward(s)
    Z = s['Z_nuisance']; A = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def project(v): return v - Z @ np.linalg.solve(A, Z.T @ v)
    err = np.abs(project(S) - gt['S_tilde'])
    print('case', i, 'forward max error by shift:', err.reshape(len(s['mu_shifts']), -1).max(axis=1))
    assert err.max() < 1e-12
    for b in np.unique(bank):
        indices = np.flatnonzero(bank == b)
        if len(indices) < 5: continue
        held = []; weights = []; fields = []; gaps = []
        for j in indices:
            keep = np.arange(len(bank)) != j
            train = keep & (bank == b)
            gain, offset = np.linalg.lstsq(np.column_stack([sw['ref_signal'][train], np.ones(train.sum())]), sw['raw_readout'][train], rcond=None)[0]
            held.append(sw['raw_readout'][j] - gain * sw['ref_signal'][j] - offset)
            trial = dict(s, sensor_calibration_sweeps={k: v[keep] for k, v in sw.items()})
            out = m.solve_beta_m97_c01_s00(trial)
            weights.append(np.max(np.abs(out['weights'] - base['weights'])))
            fields.append(np.max(np.abs(out['terminal_field'] - base['terminal_field'])))
            check = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
            assert check['valid'], (i, b, j, check)
            gaps.append(check['gap'])
        print('bank', int(b), 'folds', len(indices), 'held-out raw max/bias', np.max(np.abs(held)), np.mean(held), 'public residual max', np.max(np.abs((project(y)-gt['r_y'])[rows == b])), 'max weight/field change', max(weights), max(fields), 'worst independent KKT gap', max(gaps))
PY
```
- **Container Exit Code**: `0` (`623 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 forward max error by shift: [4.16333634e-17 5.55111512e-17 6.24500451e-17 4.85722573e-17]
bank 0 folds 12 held-out raw max/bias 3.0207692529975594e-07 6.25974790543736e-09 public residual max 5.883198372913512e-08 max weight/field change 1.3154655292835216e-07 6.5550535069403004e-09 worst independent KKT gap 1.0670401430490362e-07
bank 1 folds 12 held-out raw max/bias 1.7688976710295838e-07 -1.207949492876402e-10 public residual max 7.150326286620157e-08 max weight/field change 7.768480853043158e-08 5.524707410303753e-09 worst independent KKT gap 9.989416903388726e-08
bank 2 folds 12 held-out raw max/bias 2.298255774810004e-07 -2.731543053300759e-11 public residual max 4.652717500164272e-08 max weight/field change 1.0107606118747725e-07 6.4360302434729455e-09 worst independent KKT gap 1.0808126138864282e-07
bank 3 folds 12 held-out raw max/bias 1.4337553467889908e-07 -2.5323651564316547e-11 public residual max 5.371307620904986e-08 max weight/field change 8.445603283391279e-08 4.453340231302949e-09 worst independent KKT gap 9.386217272866482e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f929bfb8a0dc5c74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f929bfb8 (directory).
```

### Turn 12 — `submit_completed_task` (`13994 ms` LLM)
- **Rationale**: The saved solver passed isolated schema and numerical verification. Stress tests covered sparse calibration, four-row EMI, and mixed and endpoint box constraints. The held-out bank audit found no failure that warrants another edit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The maximum public terminal-field error was 1.79e-8. Across 48 leave-one-sweep-out folds, the worst independently verified KKT gap was 1.08e-7, below the 1e-6 tolerance.`

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
