# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c03_s00__reactor_tomography_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c03_source` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `241,692` (`231,993` / `20,352` / `9,699`) |
| **Total Episode Wall Time (s)** | `334.35s` |

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

### Turn 1 — `execute_in_remote_docker` (`15896 ms` LLM)
- **Rationale**: Read the public contract and array shapes first to resolve sensor stacking and grid ordering before implementing the forward model and certificate.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f: s=pickle.load(f)
for k,v in s.items():
 if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:5])
 elif isinstance(v,(list,tuple)): print(k,type(v).__name__,len(v),str(v[:1])[:500])
 elif isinstance(v,dict): print(k,{a:(type(b).__name__,getattr(b,'shape',None)) for a,b in v.items()})
 else: print(k,repr(v)[:300])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -35
```
- **Container Exit Code**: `0` (`553 ms`, timed_out=`False`)
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
```

### Turn 2 — `execute_in_remote_docker` (`17544 ms` LLM)
- **Rationale**: The public pickle wraps the study in a list. Inspect the inner schema and validator entry point to resolve array orientation, sensor stacking, and calibration layout before implementing the solver.

```bash
cd /holdout_task && tail -50 validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f:
    cases = pickle.load(f)
def show(x, depth=0):
    pad='  '*depth
    if isinstance(x, dict):
        for k,v in x.items():
            print(pad+str(k)+':',end=' ')
            if isinstance(v,dict):
                print(); show(v,depth+1)
            elif isinstance(v,np.ndarray): print(v.shape,v.dtype,v.ravel()[:5])
            elif isinstance(v,(list,tuple)):
                print(type(v).__name__,len(v))
                if v and depth<3: show(v[0],depth+1)
            else: print(repr(v)[:180])
    else: print(pad+repr(x)[:240])
print('Studies:',len(cases)); show(cases[0])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -45
```
- **Container Exit Code**: `0` (`517 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`17127 ms` LLM)
- **Rationale**: The prior output truncated the forward inputs. Print their names and shapes compactly to resolve sensor stacking, and check the archive layout for the specified calibration fallback.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_beta_m97_c03_s00.pkl','rb'))[0]['study']
for k,v in s.items():
 if k=='sensor_calibration_sweeps':
  print(k,type(v).__name__,str(v)[:1500]);continue
 if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:6])
 else: print(k,str(v)[:250])
PY
sed -n '66,130p' validate_public_schema.py
p=/tmp/hegel_continual_archive/archive_calibration_history.json; if test -f "$p"; then python3 - "$p" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1])); print('archive layout:', list(obj) if isinstance(obj,dict) else (len(obj),list(obj[0]) if obj else []))
PY
fi
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,  0.69839 , -0.260974, -0.723373,
       -0.09476 , -0.5241  , -1.493419, -1.382773,  1.855897,  0.114249,
        1.853062, -0.77891 ,  1.464848, -0.952095,  1.253344, -1.324626,
        0.447921, -0.495095,  2.182426,  2.194405, -1.195034,  1.353202,
        0.467707, -1.264416,  0.593127,  0.596117,  1.467343, -1.056263,
        0.211325, -1.250988,  0.51585 ,  1.236978, -0.96195 ,  0.611761,
        1.710794, -0.837501,  1.007049, -0.081748,  1.502034,  1.660641]), 'raw_readout': array([ 1.26434822, -1.21995788,  0.25652201,  1.34746497,  0.56935394,
        1.81106495, -0.01981812,  0.89386805, -0.78079004,  0.69838983,
       -0.26097386, -0.7233731 , -0.09476009, -0.5241    , -1.4934189 ,
       -1.38277309,  1.85589698,  0.11424894,  1.85306177, -0.77890977,
        1.46484806, -0.95209503,  1.25334409, -1.32462611,  0.44792081,
       -0.49509508,  2.1824259 ,  2.19440484, -1.19503395,  1.35320205,
        0.46770698, -1.26441593,  0.59312714,  0.59611709,  1.46734296,
       -1.056263  ,  0.21132507, -1.25098802,  0.51584999,  1.23697786,
       -0.96194983,  0.61176099,  1.71079387, -0.83750103,  1.00704902,
       -0.081748  ,  1.50203397,  1.66064113])}
Z_nuisance (144, 5) float64 [1.65056309 1.35686558 1.18924549 0.9594841  0.90625412 0.77897528]
y_obs_raw (144,) float64 [-0.50716915  0.6420841   1.66057403 -3.04709463  1.25126415  2.00059958]
aux_manifold_telemetry (144, 3) float64 [ 0.257286 -1.376092 -1.531445  1.062927  0.872685 -0.983499]
nuisance_ridge 0.65
source_ridge 0.04
u_box 1.0
total_budget 2.6
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

### Turn 4 — `execute_in_remote_docker` (`74750 ms` LLM)
- **Rationale**: Compare projected inputs separately from source weights and terminal field to distinguish grid or stacking mistakes from calibration and optimization errors.
- **Alternatives Weighed**: `Use the general ridge-profiling primitive` (rejected: The specified projection is a single five-column solve for all right-hand sides; it needs neither the primitive's SVD whitener nor an extra runtime import.)

```bash
cd /holdout_task && cat > beta_m97_c03_s00_solver.py <<'PY'
"""Open-Robin reaction-transport tomography and box-simplex inversion."""
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _fit_affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gains = np.empty(4)
    offsets = np.empty(4)
    history = None
    for bank in range(4):
        mask = banks == bank
        x, y = ref[mask], raw[mask]
        spikes = np.zeros(x.size, dtype=bool)
        if x.size >= 3:
            # Repeated-median line keeps an isolated spike out of the reference fit.
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            offset = np.median(y - gain * x)
            spikes = np.abs(y - gain * x - offset) > 3.0
        clean = ~spikes
        prior = None
        if x.size < 2 or np.any(spikes):
            if history is None:
                try:
                    with open(_CALIBRATION_ARCHIVE) as f:
                        history = json.load(f)
                except FileNotFoundError:
                    history = {}
            prior = _archived_affine(history, bank)
        if clean.sum() >= 4 or (clean.sum() >= 2 and prior is None):
            gains[bank], offsets[bank] = _fit_affine(x[clean], y[clean])
        elif prior is not None:
            gains[bank], offsets[bank] = prior
        else:
            raise ValueError(f'Insufficient calibration data for bank {bank}')
    b = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw']) - offsets[b]) / gains[b]


def _archived_affine(history, bank):
    if isinstance(history, dict):
        if 'gain' in history and 'offset' in history:
            gain, offset = history['gain'], history['offset']
            if isinstance(gain, (list, dict)):
                return gain[str(bank) if isinstance(gain, dict) else bank], offset[str(bank) if isinstance(offset, dict) else bank]
            if history.get('bank_id') == bank:
                return gain, offset
        entry = history.get(str(bank))
        if isinstance(entry, dict) and 'gain' in entry and 'offset' in entry:
            return entry['gain'], entry['offset']
        records = history.get('banks', history.get('history', []))
        return _archived_affine(records, bank) if records else None
    for record in reversed(history):
        if isinstance(record, dict) and record.get('bank_id') == bank:
            return record['gain'], record['offset']
    return None


def _axis_operator(kap, vel, rob, h):
    kap, vel = np.asarray(kap), np.asarray(vel)
    n = kap.size
    face = (kap[1:] + kap[:-1]) / (2 * h**2)
    lower = -face - np.maximum(vel[1:], 0) / h
    upper = -face - np.maximum(-vel[:-1], 0) / h
    diag = np.empty(n)
    diag[1:-1] = face[:-1] + face[1:] + np.abs(vel[1:-1]) / h
    upper[0], lower[-1] = -kap[0] / h**2, -kap[-1] / h**2
    diag[0], diag[-1] = -upper[0] + rob / h, -lower[-1] + rob / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis_operator(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))
    ay = _axis_operator(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'] / (ny - 1))
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    modes = np.column_stack([mode.ravel(order='F') for mode in np.asarray(study['source_modes'])])
    i, j = np.asarray(study['sensor_ij']).T
    sensor_idx = i + nx * j
    sampled = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(modes)
        sampled.append(fields[sensor_idx])
    return np.vstack(sampled), fields


def _box_simplex(S, y, ridge, upper, budget):
    """Solve the small QP by feasible moves to equality-constrained face minima."""
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    b = S.T @ y
    if budget == 0 or budget == m * upper:
        w = np.full(m, budget / m)
        g = H @ w - b
        return w, float(np.min(g) if budget == 0 else np.max(g))

    w = np.full(m, budget / m)
    active = np.zeros(m, dtype=int)  # -1 at zero, +1 at upper
    for _ in range(100 * m):
        free = np.flatnonzero(active == 0)
        if not free.size:
            g = H @ w - b
            lo = np.flatnonzero(active < 0)
            hi = np.flatnonzero(active > 0)
            enter, leave = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[enter] >= g[leave] - 1e-12:
                return w, float((g[enter] + g[leave]) / 2)
            active[[enter, leave]] = 0
            continue

        fixed = np.flatnonzero(active)
        rhs = b[free] - H[np.ix_(free, fixed)] @ w[fixed]
        hf = H[np.ix_(free, free)]
        base, unit = np.linalg.solve(hf, np.column_stack((rhs, np.ones(free.size)))).T
        nu = (budget - w[fixed].sum() - base.sum()) / unit.sum()
        target = base + nu * unit
        if np.any(target < -1e-13) or np.any(target > upper + 1e-13):
            move = target - w[free]
            a = np.full(free.size, np.inf)
            lower = target < -1e-13
            higher = target > upper + 1e-13
            a[lower] = -w[free[lower]] / move[lower]
            a[higher] = (upper - w[free[higher]]) / move[higher]
            hit = np.argmin(a)
            w[free] += a[hit] * move
            active[free[hit]] = -1 if lower[hit] else 1
            w[free[hit]] = 0.0 if lower[hit] else upper
            continue

        w[free] = np.clip(target, 0.0, upper)
        if not fixed.size:
            return w, float(nu)
        g = H @ w - b
        violation = active[fixed] * (g[fixed] - nu)
        release = np.argmax(violation)
        if violation[release] <= 1e-12:
            return w, float(nu)
        active[fixed[release]] = 0
    raise RuntimeError('Box-simplex active-set iteration did not converge')


def solve_beta_m97_c03_s00(study: dict) -> dict:
    S, fields = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((y, S))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    y, S = projected[:, 0], projected[:, 1:]
    ridge, upper, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, y, ridge, upper, budget)
    residual = S @ w - y
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = S.T @ residual + ridge * w
    lower = primal - g @ w + nu * budget + upper * np.minimum(0.0, g - nu).sum()
    cert = dict(weights=w, nu_sum=nu, lam_lower=np.maximum(g - nu, 0),
                lam_upper=np.maximum(nu - g, 0), primal_objective=primal,
                dual_lower_bound=float(lower))
    terminal = (fields @ w).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=w, terminal_field=terminal, kkt_certificate=cert)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','beta_m97_c03_s00_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_beta_m97_c03_s00.pkl','rb') as f: item=pickle.load(f)[0]
s,g=item['study'],item['gt']; out=m.solve_beta_m97_c03_s00(s)
a,F=m._forward(s); z=s['Z_nuisance']; d=np.column_stack((m._calibrate(s),a)); p=d-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@d)
for name,x,y in [('S_tilde',p[:,1:],g['S_tilde']),('r_y',p[:,0],g['r_y']),('weights',out['weights'],g['w_opt']),('terminal',out['terminal_field'],g['terminal_field_A'])]: print(name,'maxabs',np.max(np.abs(x-y)))
c=out['kkt_certificate']; print('gap',c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`552 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
S_tilde maxabs 8.326672684688674e-17
r_y maxabs 6.839283095416704e-08
weights maxabs 3.983972646226519e-08
terminal maxabs 1.4750419601028408e-09
gap 3.469446951953614e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0342b3cca7ca37aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0342b3cc (directory).
```

### Turn 5 — `execute_in_remote_docker` (`47905 ms` LLM)
- **Rationale**: The public case does not exercise EMI or sparse calibration, reversed upwinding, or constrained QP faces. Test each with a known reference so any failure identifies which part of the solver to fix.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
ns = {'__name__': 'certificate_check'}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']

# Exercise calibration with known gains, an isolated EMI spike, and a missing bank.
rng = np.random.default_rng(9703)
gains, offsets = np.array([1.4, .7, 2., .8]), np.array([-.3, .2, .9, -1.])
rids = s['row_bank_id']
y = rng.normal(size=rids.size)
base = dict(s, y_obs_raw=y * gains[rids] + offsets[rids])
bank = np.repeat(np.arange(4), 12)
x = np.tile(np.linspace(-2, 2, 12), 4)
raw = gains[bank] * x + offsets[bank]
with tempfile.TemporaryDirectory() as td:
    archive = Path(td) / 'history.json'
    m._CALIBRATION_ARCHIVE = str(archive)
    cases = []
    cases.append(('clean', np.ones(bank.size, dtype=bool), raw.copy()))
    cases.append(('EMI', np.ones(bank.size, dtype=bool), raw.copy()))
    cases[-1][2][17] += 4.2
    cases.append(('one_row', (bank != 2) | (np.arange(bank.size) == 26), raw.copy()))
    cases.append(('zero_rows', bank != 3, raw.copy()))
    cases.append(('two_clean', (bank != 0) | (np.arange(bank.size) < 2), raw.copy()))
    for label, keep, v in cases:
        if label in ('EMI', 'one_row', 'zero_rows'):
            archive.write_text(json.dumps({str(i): {'gain': float(g), 'offset': float(o)} for i, (g, o) in enumerate(zip(gains, offsets))}))
        else:
            archive.write_text('invalid JSON: clean sweeps must not consult the archive')
        st = dict(base, sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=x[keep], raw_readout=v[keep]))
        err = np.max(np.abs(m._calibrate(st) - y))
        assert err < 1e-12, (label, err)
        print('calibration', label, f'{err:.3e}')

# A row-wise dense reference catches a reversed upwind direction or indexing error.
small = dict(s, nx=4, ny=3, kappa_x=np.array([.12,.08,.15,.10]), kappa_y=np.array([.10,.14,.09]), vel_x=np.array([-.2,-.3,.25,.2]), vel_y=np.array([.1,-.23,-.2]), V_reaction_2d=rng.uniform(.2,1.,(4,3)), source_modes=rng.normal(size=(12,4,3)), sensor_ij=np.array([[2,0],[0,2],[3,1]]))
L = np.diag(small['V_reaction_2d'].ravel(order='F'))
for j in range(3):
    for i in range(4):
        row = i + 4*j
        for axis, k, n, step in [('x',i,4,1),('y',j,3,4)]:
            h = small['L'+axis]/(n-1)
            kap = small['kappa_'+axis]
            v = small['vel_'+axis][k]
            if k == 0 or k == n-1:
                d = kap[k]/h**2
                L[row,row] += d + small['robin_'+axis]/h
                L[row,row+step*(1 if k == 0 else -1)] -= d
            else:
                for e in (-1,1):
                    d = (kap[k]+kap[k+e])/(2*h*h) + max(-e*v,0)/h
                    L[row,row] += d
                    L[row,row+e*step] -= d
rhs = np.array([a.ravel(order='F') for a in small['source_modes']]).T
ref = [np.linalg.solve(L+mu*np.eye(12), rhs) for mu in small['mu_shifts']]
A,F = m._forward(small)
err = max(np.max(np.abs(A - np.vstack([a[[2,8,7]] for a in ref]))), np.max(np.abs(F-ref[-1])))
assert err < 1e-12, err
print('signed-upwind forward max error', f'{err:.3e}')

# Independently verify optimality where lower bounds, upper bounds, or all constraints bind.
worst = 0.
for trial in range(200):
    D = rng.normal(size=(36, 12))
    if trial % 3 == 0:
        D[:, 6:] = D[:, :6]
    y = rng.normal(size=36) * (0.03, 2., 20.)[trial % 3]
    upper = (0.2, 1., 2.3)[trial % 3]
    budget = upper * (0., 0.01, 1., 2.6, 6., 11.99, 12.)[trial % 7]
    ridge = (0.0001, .04, .4)[trial % 3]
    w, nu = m._box_simplex(D, y, ridge, upper, budget)
    r = D @ w - y
    g = D.T @ r + ridge*w
    p = 0.5 * (r @ r + ridge*(w @ w))
    cert = dict(weights=w, nu_sum=nu, lam_lower=np.maximum(g-nu,0), lam_upper=np.maximum(nu-g,0), primal_objective=p, dual_lower_bound=p-g@w+nu*budget+upper*np.minimum(0.,g-nu).sum())
    chk = verify(D, y, ridge, upper, budget, cert, gap_tol=1e-8)
    assert chk['valid'], (trial, chk)
    worst = max(worst, chk['gap'])
print('200 box-simplex cases; worst verified gap', f'{worst:.3e}')
PY
```
- **Container Exit Code**: `0` (`573 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration clean 1.332e-15
calibration EMI 1.332e-15
calibration one_row 1.332e-15
calibration zero_rows 1.332e-15
calibration two_clean 1.332e-15
signed-upwind forward max error 1.665e-16
200 box-simplex cases; worst verified gap 1.819e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0342b3cca7ca37aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0342b3cc (directory).
```

### Turn 6 — `submit_completed_task` (`13930 ms` LLM)
- **Rationale**: The saved artifact passed isolated public validation and targeted checks for missing or spiking calibration rows, signed upwinding, and active box constraints.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:7605'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c03_s00_solver.py`. The isolated public validator passed. Maximum absolute errors were 3.98e-8 for the weights and 1.48e-9 for the terminal field. Calibration fallback and signed upwinding checks passed; the worst verified KKT gap across 200 stress cases was 1.82e-12.`

### Turn 7 — `execute_in_remote_docker` (`32667 ms` LLM)
- **Rationale**: Compare projected design and response against the public reference by shift and bank. A design mismatch would direct a stencil or stacking correction; a bank-specific response mismatch would direct a calibration correction. The audit leaves the artifact unchanged.
- **Alternatives Weighed**: `Test relay modes or within-batch classification accuracy` (rejected: The contract defines open-loop tomography and a continuous inverse problem, not relay control or classification. Bank-wise residuals test the relevant calibration failure.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    cases = pickle.load(f)
worst = dict(S=0., y=0., w=0., field=0., gap=0.)
for case in cases:
    s, gt = case['study'], case['gt']
    trace_max = np.max(s['past_vortex_circulation_trace'])
    assert s['recirculation_loop_mode'] == 0 and trace_max <= s['critical_vortex_circulation']
    sw = s['sensor_calibration_sweeps']
    print('open Robin circulation margin', s['critical_vortex_circulation'] - trace_max)
    for b in range(4):
        keep = sw['bank_id'] == b
        x, y = sw['ref_signal'][keep], sw['raw_readout'][keep]
        gain, offset = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
        fit_err = np.max(np.abs(y - gain*x - offset))
        print('bank', b, 'sweep count', len(x), 'fit max error', fit_err)
    A, _ = m._forward(s)
    Z = s['Z_nuisance']
    D = np.column_stack((m._calibrate(s), A))
    P = D - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ D)
    eS = np.max(np.abs(P[:, 1:] - gt['S_tilde']), axis=1)
    ey = np.abs(P[:, 0] - gt['r_y'])
    n_sensor = len(s['sensor_ij'])
    for t, mu in enumerate(s['mu_shifts']):
        idx = slice(t*n_sensor, (t+1)*n_sensor)
        print('shift', mu, 'design max error', max(eS[idx]), 'response max error', max(ey[idx]))
    for b in range(4):
        idx = s['row_bank_id'] == b
        print('bank', b, 'projected response max error', max(ey[idx]))
    out = m.solve_beta_m97_c03_s00(s)
    c = out['kkt_certificate']
    metrics = dict(S=max(eS), y=max(ey), w=np.max(np.abs(out['weights']-gt['w_opt'])), field=np.max(np.abs(out['terminal_field']-gt['terminal_field_A'])), gap=abs(c['primal_objective']-c['dual_lower_bound']))
    assert metrics['S'] < 1e-12 and metrics['y'] < 1e-6, metrics
    assert metrics['w'] < 1e-6 and metrics['field'] < 1e-7 and metrics['gap'] < 1e-8, metrics
    for k in worst:
        worst[k] = max(worst[k], metrics[k])
print('Worst public errors across', len(cases), 'study/studies:', worst)
PY
```
- **Container Exit Code**: `0` (`571 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
open Robin circulation margin 0.16185300000000002
bank 0 sweep count 12 fit max error 2.2827795472316252e-07
bank 1 sweep count 12 fit max error 2.2650622012374903e-07
bank 2 sweep count 12 fit max error 1.697776854961604e-07
bank 3 sweep count 12 fit max error 1.6093999207069708e-07
shift 0.0 design max error 5.551115123125783e-17 response max error 4.3155998774579984e-08
shift 0.22 design max error 6.245004513516506e-17 response max error 4.503123651744545e-08
shift 0.52 design max error 8.326672684688674e-17 response max error 3.321036556869217e-08
shift 0.9 design max error 5.898059818321144e-17 response max error 6.839283095416704e-08
bank 0 projected response max error 6.839283095416704e-08
bank 1 projected response max error 2.6572357203491492e-08
bank 2 projected response max error 4.0179266136419756e-08
bank 3 projected response max error 3.8122303980259176e-08
Worst public errors across 1 study/studies: {'S': 8.326672684688674e-17, 'y': 6.839283095416704e-08, 'w': 3.983972646226519e-08, 'field': 1.4750419601028408e-09, 'gap': 3.469446951953614e-17}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0342b3cca7ca37aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0342b3cc (directory).
```

### Turn 8 — `submit_completed_task` (`15027 ms` LLM)
- **Rationale**: The public measurements confirmed the open-Robin regime and agreement with reference outputs across all banks and shifts. Targeted stress tests passed for sparse and spiking calibration sweeps, signed upwinding, and active box constraints. The verified artifact is unchanged.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:7605'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public validator passed in isolated Python. Maximum absolute errors were 3.98e-8 for the source weights and 1.48e-9 for the terminal field. The worst verified KKT gap across 200 stress cases was 1.82e-12.`

### Turn 9 — `execute_in_remote_docker` (`32626 ms` LLM)
- **Rationale**: A large leave-one-out sweep residual or weight change would warrant revisiting calibration; a nonzero interior response to a constant field would warrant correcting the stencil. The unprojected control measures the effect of the specified nuisance correction on the public case.
- **Alternatives Weighed**: `Test relay modes and within-batch classification accuracy` (rejected: The specified branch is open Robin and the output is continuous. Stencil closure and an unprojected inversion are the relevant checks.)

```bash
cat >/tmp/audit_beta_m97_c03_s00.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_beta_m97_c03_s00.pkl').open('rb') as f:
    cases = pickle.load(f)
worst_holdout = worst_shift = worst_control = worst_robin = 0.
for case in cases:
    s, gt = case['study'], case['gt']
    sw = s['sensor_calibration_sweeps']
    S, _ = m._forward(s)
    Z = s['Z_nuisance']
    gram = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def project(a):
        return a - Z @ np.linalg.solve(gram, Z.T @ a)
    y = m._calibrate(s)
    ps = project(S)
    args = tuple(s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    baseline = m.solve_beta_m97_c03_s00(s)['weights']
    control, _ = m._box_simplex(S, y, *args)
    worst_control = max(worst_control, np.max(np.abs(control - gt['w_opt'])))
    for bank in range(4):
        ids = np.flatnonzero(sw['bank_id'] == bank)
        x, raw = sw['ref_signal'][ids], sw['raw_readout'][ids]
        for i in range(ids.size):
            keep = np.arange(ids.size) != i
            gain, offset = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), raw[keep], rcond=None)[0]
            worst_holdout = max(worst_holdout, abs(raw[i] - (gain*x[i] + offset)))
            y_jk = y.copy()
            mask = s['row_bank_id'] == bank
            y_jk[mask] = (s['y_obs_raw'][mask] - offset) / gain
            w, _ = m._box_simplex(ps, project(y_jk), *args)
            worst_shift = max(worst_shift, np.max(np.abs(w - baseline)))
    for axis in ('x', 'y'):
        n = s['n' + axis]
        h = s['L' + axis] / (n - 1)
        rob = s['robin_' + axis]
        A = m._axis_operator(s['kappa_' + axis], s['vel_' + axis], rob, h)
        expected = np.zeros(n)
        expected[[0, -1]] = rob / h
        worst_robin = max(worst_robin, np.max(np.abs(A @ np.ones(n) - expected)))
print('Worst held-out calibration residual:', worst_holdout)
print('Worst weight change from omitting one measured sweep row:', worst_shift)
print('Worst weight error with nuisance projection omitted (negative control):', worst_control)
print('Worst constant-field axis stencil error:', worst_robin)
assert worst_holdout < 1e-6 and worst_shift < 1e-6 and worst_robin < 1e-12
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_beta_m97_c03_s00.py
```
- **Container Exit Code**: `0` (`433 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst held-out calibration residual: 2.763915443182441e-07
Worst weight change from omitting one measured sweep row: 1.3618453226227345e-07
Worst weight error with nuisance projection omitted (negative control): 0.8464388622889786
Worst constant-field axis stencil error: 3.552713678800501e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0342b3cca7ca37aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0342b3cc (directory).
```

### Turn 10 — `submit_completed_task` (`15075 ms` LLM)
- **Rationale**: The unchanged artifact passed the public validator and targeted stress checks. Leaving out one calibration sweep row changed the weights by at most 1.36e-7, and the constant-field stencil matched the Robin boundary response to 3.55e-15. The audit found no reason to change the solver.
- **Alternatives Weighed**: `Omit nuisance projection` (rejected: The public negative control increased maximum source-weight error to 0.846.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public validator passed in isolated Python. Maximum absolute errors were 3.98e-8 for the source weights and 1.48e-9 for the terminal field. The worst verified KKT gap across 200 stress cases was 1.82e-12.`

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
