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
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `266,065` (`254,971` / `72,192` / `11,094`) |
| **Total Episode Wall Time (s)** | `403.34s` |

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

### Turn 1 — `execute_in_remote_docker` (`16347 ms` LLM)
- **Rationale**: Inspect the public schema and array layouts first; a mismatch would change how sensor rows, source modes, and the terminal field are assembled. Check whether a nuisance-profiling primitive is available before implementing it.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f: s=pickle.load(f)
print('TOP',type(s))
for k,v in s.items():
 print(k,type(v).__name__,getattr(v,'shape',''),str(v)[:310] if not isinstance(v,np.ndarray) or v.size<=20 else str(v.ravel()[:5]))
PY
ls -l /tmp/hegel_continual_archive /tmp/hegel_wormhole_skills 2>/dev/null || true
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -40
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`18033 ms` LLM)
- **Rationale**: The pickle is a list of cases, not a study dictionary. Inspect the individual study layout and validator to avoid misordering sensor rows or source modes, and check the calibration archive format before implementing its gated fallback.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f: cases=pickle.load(f)
print('case count:',len(cases))
for i,c in enumerate(cases):
 s=c['study']; print('case',i,'mode',s.get('recirculation_loop_mode'),'trace max',np.max(s['past_vortex_circulation_trace']),'critical',s['critical_vortex_circulation'])
 if i==0:
  for k,v in s.items():
   if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:8])
   else: print(k,type(v).__name__,repr(v)[:850])
PY
if test -f /tmp/hegel_continual_archive/archive_calibration_history.json; then head -c 4200 /tmp/hegel_continual_archive/archive_calibration_history.json; echo; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -60
```
- **Container Exit Code**: `0` (`391 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`18830 ms` LLM)
- **Rationale**: The earlier output omitted the source and sensor layouts and part of the certificate verifier. Recover those details with a compact inspection; they determine row assembly and the equality-multiplier sign. Bank counts establish whether the public cases exercise the archive fallback.

```bash
cd /holdout_task && sed -n '43,160p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f: cases=pickle.load(f)
for k,v in cases[0]['study'].items():
 if isinstance(v,dict): print(k,[(a,np.shape(b)) for a,b in v.items()])
 elif isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:4])
 else: print(k,str(v)[:160])
for i,c in enumerate(cases):
 s=c['study']; d=s['sensor_calibration_sweeps']; b=np.asarray(d['bank_id']); x=np.asarray(d['ref_signal']); y=np.asarray(d['raw_readout']); print('case',i,'mode',s['recirculation_loop_mode'],'trace_max',max(s['past_vortex_circulation_trace']),'calibration',[(int(sum(b==j)),round(float(y[b==j].min()),3),round(float(y[b==j].max()),3)) if np.any(b==j) else (0,) for j in range(4)])
PY
p=/tmp/hegel_continual_archive/archive_calibration_history.json; if test -f "$p"; then head -c 1800 "$p"; echo; else echo 'Calibration archive not mounted'; fi
```
- **Container Exit Code**: `0` (`362 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c03_s00(st)
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
Lx 1.5
Ly 1.3
nx 18
ny 16
kappa_x (18,) float64 [0.12       0.11957433 0.11831181 0.11625543]
vel_x (18,) float64 [0.24       0.23647059 0.23294118 0.22941176]
robin_x 0.85
reaction_x (18,) float64 [0.3        0.30055363 0.30221453 0.3049827 ]
kappa_y (16,) float64 [0.085      0.08915823 0.09313473 0.09675571]
vel_y (16,) float64 [0.2        0.20266667 0.20533333 0.208     ]
robin_y 0.75
reaction_y (16,) float64 [0.4        0.39923307 0.39694066 0.39314791]
V_reaction_2d (18, 16) float64 [0.7        0.69923307 0.69694066 0.69314791]
recirculation_loop_mode 0.0
past_vortex_circulation_trace (16,) float64 [0.286695 0.304501 0.31505  0.114439]
critical_vortex_circulation 0.5
mu_shifts (4,) float64 [0.   0.22 0.52 0.9 ]
source_modes (12, 18, 16) float64 [0.00810691 0.01819605 0.03146544 0.0419204 ]
sensor_ij (36, 2) int64 [6 2 2 8]
row_bank_id (144,) int64 [0 1 2 3]
sensor_calibration_sweeps [('bank_id', (48,)), ('ref_signal', (48,)), ('raw_readout', (48,))]
Z_nuisance (144, 5) float64 [1.65056309 1.35686558 1.18924549 0.9594841 ]
y_obs_raw (144,) float64 [-0.50716915  0.6420841   1.66057403 -3.04709463]
aux_manifold_telemetry (144, 3) float64 [ 0.257286 -1.376092 -1.531445  1.062927]
nuisance_ridge 0.65
source_ridge 0.04
u_box 1.0
total_budget 2.6
case 0 mode 0.0 trace_max 0.338147 calibration [(12, -1.22, 1.811), (12, -1.493, 1.856), (12, -1.264, 2.194), (12, -1.251, 1.711)]
Calibration archive not mounted
```

### Turn 4 — `execute_in_remote_docker` (`65254 ms` LLM)
- **Rationale**: Implement the specified operators and multiplier convention, then compare intermediate arrays to the public reference so a discrepancy identifies which stage needs correction. Test both plausible sensor stack orders rather than inferring order from final weights alone.
- **Alternatives Weighed**: `Use the nuisance profiling skill directly` (rejected: It computes an additional Gram whitener that this solver does not need; a single batched linear solve implements the specified residualization.)

```bash
cd /holdout_task && cat > beta_m97_c03_s00_solver.py <<'PY'
"""Open-outflow reaction–transport tomography and box-simplex inversion."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gain_offset = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        if x.size >= 2:
            fit = _affine(x, y)
            clean = np.ones(x.size, dtype=bool)
            if x.size >= 3:
                # A robust pilot identifies EMI without mistaking large reference signals for spikes.
                ii, jj = np.triu_indices(x.size, 1)
                valid = x[ii] != x[jj]
                slopes = (y[jj[valid]] - y[ii[valid]]) / (x[jj[valid]] - x[ii[valid]])
                if slopes.size:
                    a = np.median(slopes)
                    d = np.median(y - a*x)
                    clean = np.abs(y - (a*x + d)) <= 3.0
            if clean.all() or clean.sum() >= 4:
                gain_offset[b] = _affine(x[clean], y[clean]) if not clean.all() else fit
                continue
        else:
            clean = np.ones(x.size, dtype=bool)
        if history is None and _ARCHIVE.is_file():
            history = json.loads(_ARCHIVE.read_text())
        if history is not None:
            entries = history.get('banks', history) if isinstance(history, dict) else history
            if isinstance(entries, dict):
                entry = entries.get(str(b))
            else:
                entry = next((e for e in reversed(entries) if e['bank_id'] == b), None)
            if entry is not None:
                gain_offset[b] = entry['gain'], entry['offset']
                continue
        if clean.sum() >= 2:
            gain_offset[b] = _affine(x[clean], y[clean])
        else:
            raise ValueError(f'Insufficient calibration data for bank {b}')
    gain, offset = gain_offset[np.asarray(study['row_bank_id'], dtype=int)].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


def _axis(kap, vel, length, robin):
    kap, vel = np.asarray(kap), np.asarray(vel)
    h = length / (kap.size - 1)
    faces = (kap[:-1] + kap[1:]) / (2*h*h)
    lo = -faces - np.maximum(vel[1:], 0)/h
    hi = -faces - np.maximum(-vel[:-1], 0)/h
    diag = np.empty(kap.size)
    diag[1:-1] = faces[1:] + faces[:-1] + np.abs(vel[1:-1])/h
    hi[0] = -kap[0]/h**2
    lo[-1] = -kap[-1]/h**2
    diag[0], diag[-1] = -hi[0] + robin/h, -lo[-1] + robin/h
    return diags((lo, diag, hi), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    operator = kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(np.ravel(study['V_reaction_2d'], order='F'))
    ident = eye(nx*ny, format='csc')
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T
    i, j = np.asarray(study['sensor_ij']).T
    rows = i + nx*j
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu((operator + mu*ident).tocsc()).solve(modes)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields


def _invert(s, y, ridge, upper, budget):
    n = s.shape[1]
    gram = s.T @ s + ridge*np.eye(n)
    rhs = s.T @ y
    w = np.full(n, budget/n)
    # Keep primal feasibility while adding blocking bounds and release bounds with the wrong KKT sign.
    active = np.zeros(n, dtype=int)
    for _ in range(20*n*n):
        free = active == 0
        g = gram @ w - rhs
        if free.any():
            mat = gram[np.ix_(free, free)]
            inv_g, inv_one = np.linalg.solve(mat, np.column_stack((g[free], np.ones(free.sum())))).T
            nu = inv_g.sum()/inv_one.sum()
            direction = nu*inv_one - inv_g
            if np.max(np.abs(direction)) > 1e-12:
                ratio = np.full(direction.size, np.inf)
                up = direction > 1e-13
                down = direction < -1e-13
                wf = w[free]
                ratio[up] = (upper - wf[up])/direction[up]
                ratio[down] = -wf[down]/direction[down]
                block = np.argmin(ratio)
                step = min(1.0, max(0.0, ratio[block]))
                w[free] = np.clip(wf + step*direction, 0, upper)
                if step < 1.0:
                    idx = np.flatnonzero(free)[block]
                    active[idx] = 1 if up[block] else -1
                    w[idx] = upper if up[block] else 0.0
                continue
        else:
            lo = np.max(g[active == 1], initial=-np.inf)
            hi = np.min(g[active == -1], initial=np.inf)
            nu = np.clip(np.mean(g), min(lo, hi), max(lo, hi))
        violation = active*(g - nu)
        idx = np.argmax(violation)
        if violation[idx] <= 1e-11:
            break
        active[idx] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')
    res = s @ w - y
    grad = s.T @ res + ridge*w
    slack = grad - nu
    obj = 0.5*(res @ res + ridge*(w @ w))
    dual = obj - grad @ w + nu*budget + upper*np.minimum(slack, 0).sum()
    return w, dict(weights=w.copy(), nu_sum=float(nu), lam_lower=np.maximum(slack, 0),
                   lam_upper=np.maximum(-slack, 0), primal_objective=float(obj),
                   dual_lower_bound=float(min(obj, dual)))


def solve_beta_m97_c03_s00(study: dict) -> dict:
    if study['recirculation_loop_mode'] != 0 or np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']:
        raise ValueError('Study is outside the open-outflow Robin regime')
    sensors, terminal = _forward(study)
    data = np.column_stack((_calibrate(study), sensors))
    z = np.asarray(study['Z_nuisance'])
    gram_z = z.T @ z + study['nuisance_ridge']*np.eye(z.shape[1])
    profiled = data - z @ np.linalg.solve(gram_z, z.T @ data)
    w, cert = _invert(profiled[:, 1:], profiled[:, 0], study['source_ridge'], study['u_box'], study['total_budget'])
    field = (terminal @ w).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=w, terminal_field=field, kkt_certificate=cert)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import runpy,pickle,numpy as np
m=runpy.run_path('beta_m97_c03_s00_solver.py')
for item in pickle.load(open('public_beta_m97_c03_s00.pkl','rb')):
 s,g=item['study'],item['gt']; z=s['Z_nuisance']; p=lambda a:a-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
 x,_=m['_forward'](s); xc=x.reshape(len(s['mu_shifts']),len(s['sensor_ij']),-1).transpose(1,0,2).reshape(x.shape)
 o=m['solve_beta_m97_c03_s00'](s)
 print('max abs: S shift-major',np.max(abs(p(x)-g['S_tilde'])),'S sensor-major',np.max(abs(p(xc)-g['S_tilde'])),'ry',np.max(abs(p(m['_calibrate'](s))-g['r_y'])),'weights',np.max(abs(o['weights']-g['w_opt'])),'terminal',np.max(abs(o['terminal_field']-g['terminal_field_A'])),'internal gap',o['kkt_certificate']['primal_objective']-o['kkt_certificate']['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`584 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
max abs: S shift-major 6.938893903907228e-17 S sensor-major 0.06273677422121082 ry 6.839283095416704e-08 weights 3.9839727072887854e-08 terminal 1.4750419843889695e-09 internal gap 7.632783294297951e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99c1a082a19276ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99c1a082 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`33184 ms` LLM)
- **Rationale**: The public case does not exercise EMI, sparse-bank fallback, or budgets at the box endpoints. A calibration failure would require changing the filtering or archive gating; a certificate failure would isolate the active-set or multiplier logic.

```bash
cd /holdout_task && python3 -I - <<'PY'
import json, pickle, runpy, tempfile
from pathlib import Path
import numpy as np
m = runpy.run_path('beta_m97_c03_s00_solver.py')
solve = m['solve_beta_m97_c03_s00']
with open('public_beta_m97_c03_s00.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
base = solve(s)
d = s['sensor_calibration_sweeps']; bank = d['bank_id']; k = np.flatnonzero(bank == 2)[0]
raw = d['raw_readout'].copy(); raw[k] += 5
out = solve(dict(s, sensor_calibration_sweeps=dict(d, raw_readout=raw)))
emi = np.max(np.abs(out['weights'] - base['weights']))
assert emi < 1e-6, emi
with tempfile.TemporaryDirectory() as td:
    archive = Path(td)/'history.json'
    solve.__globals__['_ARCHIVE'] = archive
    keep = (bank != 2); keep[k] = True
    gain, offset = m['_affine'](d['ref_signal'][bank == 2], d['raw_readout'][bank == 2])
    archive.write_text(json.dumps({'banks': {'2': {'gain': gain, 'offset': offset}}}))
    out = solve(dict(s, sensor_calibration_sweeps={name: vals[keep] for name, vals in d.items()}))
    sparse = np.max(np.abs(out['weights'] - base['weights']))
    assert sparse < 1e-12, sparse
    archive.write_text('not needed for clean banks')
    assert np.array_equal(solve(s)['weights'], base['weights'])
perturbed = solve(dict(s, aux_manifold_telemetry=1000*s['Z_nuisance'][:, :3]))
assert np.array_equal(perturbed['terminal_field'], base['terminal_field'])
rng = np.random.default_rng(731)
worst_gap = worst_budget = worst_comp = 0.0
for scale in [0.03, 1., 12.]:
    for budget in [0., 1e-8, 1., 2.6, 6., 12.-1e-8, 12.]:
        for _ in range(12):
            A, y = rng.normal(size=(36, 12))*scale, rng.normal(size=36)*scale
            w, c = m['_invert'](A, y, 0.04, 1., budget)
            r = A @ w - y; g = A.T @ r + 0.04*w
            p = 0.5*(r@r + 0.04*(w@w)); nu = c['nu_sum']
            lb = p - g@w + nu*budget + np.minimum(g-nu, 0).sum()
            assert np.all((w >= 0) & (w <= 1))
            assert np.max(np.abs(g-nu-c['lam_lower']+c['lam_upper'])) < 1e-10
            assert abs(c['primal_objective']-p) < 1e-10 and c['dual_lower_bound'] <= lb+1e-10
            worst_budget = max(worst_budget, abs(w.sum()-budget))
            worst_gap = max(worst_gap, p-c['dual_lower_bound'])
            worst_comp = max(worst_comp, np.max(w*c['lam_lower']), np.max((1-w)*c['lam_upper']))
assert worst_budget < 1e-9 and worst_gap < 1e-7 and worst_comp < 1e-7, (worst_budget,worst_gap,worst_comp)
print(f'EMI weight drift={emi:.3g}; archive weight drift={sparse:.3g}; 252 QPs worst budget error={worst_budget:.3g}, gap={worst_gap:.3g}, complementarity={worst_comp:.3g}')
PY
```
- **Container Exit Code**: `0` (`681 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
EMI weight drift=6.32e-08; archive weight drift=0; 252 QPs worst budget error=1.78e-15, gap=2.18e-11, complementarity=5.46e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99c1a082a19276ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99c1a082 (directory).
```

### Turn 6 — `submit_completed_task` (`14617 ms` LLM)
- **Rationale**: The final artifact passed the public validator in an isolated Python process. Separate checks confirmed the terminal field against the public reference and exercised calibration fallback, budget endpoints, and KKT certification.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:6611'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public validator passed. Maximum absolute errors were 3.98e-8 for source weights and 1.48e-9 for the terminal field. EMI and simulated archive fallback checks passed; the worst KKT duality gap across 252 stress cases was 2.18e-11.`

### Turn 7 — `execute_in_remote_docker` (`36947 ms` LLM)
- **Rationale**: An independent stencil residual can catch a sign, boundary, or coordinate error that a shape check cannot. The measured per-bank and unprofiled comparisons quantify the effect of pooling or omitting nuisance removal; the mode and circulation checks verify the stated regime boundary.
- **Alternatives Weighed**: `Fit nonlinear basis terms or evaluate within-batch balanced accuracy` (rejected: The task specifies a linear stencil and continuous measurements, with no classification labels. Direct stencil residuals and per-bank calibration errors test the relevant failure modes.)

```bash
set -e
cd /holdout_task
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f:
    cases = pickle.load(f)
m = runpy.run_path('beta_m97_c03_s00_solver.py')
solve = m['solve_beta_m97_c03_s00']
for item in cases:
    s, gt = item['study'], item['gt']
    out = solve(s)
    u, w = out['terminal_field'], out['weights']
    rhs = np.einsum('m,mij->ij', w, s['source_modes'])
    residual = (s['V_reaction_2d'] + s['mu_shifts'][-1])*u - rhs
    for axis, key in enumerate(('x','y')):
        v = np.moveaxis(u, axis, 0)
        r = np.moveaxis(residual, axis, 0)
        k, a, rob = (s[p+'_'+key] for p in ('kappa','vel','robin'))
        h = s['L'+key]/(len(k)-1)
        r[0] += k[0]*(v[0]-v[1])/h**2 + rob*v[0]/h
        r[-1] += k[-1]*(v[-1]-v[-2])/h**2 + rob*v[-1]/h
        for i in range(1,len(k)-1):
            left, right = v[i]-v[i-1], v[i]-v[i+1]
            r[i] += ((k[i-1]+k[i])*left+(k[i]+k[i+1])*right)/(2*h*h)
            r[i] += (max(a[i],0)*left+max(-a[i],0)*right)/h
    defect = np.max(np.abs(residual))
    field_error = np.max(np.abs(u-gt['terminal_field_A']))
    weight_error = np.max(np.abs(w-gt['w_opt']))
    assert defect < 1e-10 and field_error < 1e-7 and weight_error < 1e-6
    d = s['sensor_calibration_sweeps']; x, y = d['ref_signal'], d['raw_readout']; bank = d['bank_id']
    pooled = m['_affine'](x,y)
    bank_report = []
    for b in range(4):
        is_b = bank == b
        fit = m['_affine'](x[is_b],y[is_b])
        bank_report.append(tuple(float(np.sqrt(np.mean((y[is_b] - (a*x[is_b]+c))**2))) for a,c in (fit,pooled)))
    assert all(local < 1e-5 for local, _ in bank_report), bank_report
    S, _ = m['_forward'](s)
    unprofiled, _ = m['_invert'](S,m['_calibrate'](s),s['source_ridge'],s['u_box'],s['total_budget'])
    critical = s['critical_vortex_circulation']
    trace = np.array([critical])
    np.testing.assert_array_equal(solve(dict(s,past_vortex_circulation_trace=trace))['weights'], w)
    for change in ({'recirculation_loop_mode':1}, {'past_vortex_circulation_trace':np.array([np.nextafter(critical,np.inf)])}):
        try:
            solve(dict(s,**change))
        except ValueError as exc:
            assert 'outside' in str(exc)
        else:
            raise AssertionError('Out-of-regime study was accepted')
    print(f'independent stencil defect={defect:.3g}, field error={field_error:.3g}, weight error={weight_error:.3g}, unprofiled weight error={np.max(np.abs(unprofiled-gt["w_opt"])):.3g}')
    print('Bank calibration RMSE (local, pooled):',bank_report)
PY
```
- **Container Exit Code**: `0` (`679 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
independent stencil defect=7.63e-16, field error=1.48e-09, weight error=3.98e-08, unprofiled weight error=0.846
Bank calibration RMSE (local, pooled): [(1.1091508622448057e-07, 1.1216481413341495e-07), (1.1232122670783792e-07, 1.1303665979213014e-07), (8.916697901045768e-08, 9.049516808127773e-08), (8.293940562614721e-08, 8.410565394922424e-08)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99c1a082a19276ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99c1a082 (directory).
```

### Turn 8 — `submit_completed_task` (`15192 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, the independent stencil and regime-boundary checks, and calibration and KKT stress tests.
- **Alternatives Weighed**: `Omit nuisance residualization` (rejected: On the public study, omission raised the maximum weight error from 3.98e-8 to 0.846.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:6611'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public validator passed. Maximum absolute errors were 3.98e-8 for source weights and 1.48e-9 for the terminal field. Across 252 optimization stress cases, the worst KKT duality gap was 2.18e-11. EMI handling and sparse-bank calibration fallback passed with simulated inputs.`

### Turn 9 — `execute_in_remote_docker` (`31815 ms` LLM)
- **Rationale**: Subsample each measured bank sweep and inject one over-threshold spike. This tests whether a high-leverage reading can hide the spike and prevent the archive fallback. If it can, change the spike detector rather than the verified transport or inversion.
- **Alternatives Weighed**: `Repeat the stencil and regime checks` (rejected: The independent stencil and both regime branches already passed. The previous EMI test used a full sweep, not a minimally sampled bank.)

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
import itertools, json, pickle, runpy, tempfile
from pathlib import Path
import numpy as np
m = runpy.run_path('beta_m97_c03_s00_solver.py')
cal = m['_calibrate']
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    cases = pickle.load(f)
with tempfile.TemporaryDirectory() as td:
    archive = Path(td) / 'calibration.json'
    cal.__globals__['_ARCHIVE'] = archive
    for item in cases:
        s = item['study']; d = s['sensor_calibration_sweeps']
        bank, x, y = (d[key] for key in ('bank_id', 'ref_signal', 'raw_readout'))
        archive.write_text(json.dumps({'banks': {str(b): dict(zip(('gain','offset'), m['_affine'](x[bank == b], y[bank == b]))) for b in range(4)}}))
        baseline = cal(s)
        for b in range(4):
            ids = np.flatnonzero(bank == b); others = np.flatnonzero(bank != b)
            rows = s['row_bank_id'] == b
            clean_worst = 0.; emi_worst = 0.; bad = 0; count = 0; witness = None
            for group in itertools.combinations(ids, 4):
                if np.ptp(x[list(group)]) < 0.3:
                    continue
                keep = np.r_[others, group]
                sweeps = {k: v[keep] for k,v in d.items()}
                ret = cal(dict(s, sensor_calibration_sweeps=sweeps))
                clean_worst = max(clean_worst, np.max(np.abs(ret[rows] - baseline[rows])))
                for j in range(4):
                    for spike in (-5., 5.):
                        corrupt = sweeps['raw_readout'].copy(); corrupt[len(others)+j] += spike
                        ret = cal(dict(s, sensor_calibration_sweeps=dict(sweeps, raw_readout=corrupt)))
                        delta = np.max(np.abs(ret[rows] - baseline[rows]))
                        count += 1; bad += delta > 1e-4
                        if delta > emi_worst:
                            emi_worst = delta; witness = (tuple(map(int, group)), int(group[j]), spike)
            print(f'bank={b} clean four-point worst drift={clean_worst:.3g}; sparse EMI misses={bad}/{count}, worst drift={emi_worst:.3g}, witness={witness}')
PY
```
- **Container Exit Code**: `0` (`7604 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank=0 clean four-point worst drift=1.8e-06; sparse EMI misses=1292/3960, worst drift=60.8, witness=((3, 5, 8, 11), 8, 5.0)
bank=1 clean four-point worst drift=2.01e-06; sparse EMI misses=1304/3960, worst drift=5.54e+03, witness=((16, 20, 21, 23), 16, -5.0)
bank=2 clean four-point worst drift=7.33e-07; sparse EMI misses=1142/3952, worst drift=603, witness=((26, 31, 34, 35), 26, -5.0)
bank=3 clean four-point worst drift=1.33e-06; sparse EMI misses=1264/3960, worst drift=1.67e+03, witness=((37, 42, 43, 46), 42, -5.0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99c1a082a19276ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99c1a082 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`56428 ms` LLM)
- **Rationale**: Retest the failed four-point subsets, including injections just above 3.0. Independently check that sparse banks use the archive and that a bank with four surviving clean rows never reads it.
- **Alternatives Weighed**: `Lower the EMI cutoff` (rejected: That changes the specified 3.0 threshold rather than correcting the leverage bias in the fit used to measure the residual.)
- **ATMS NOGOOD DECLARED**: Refuted `A median of pairwise slopes reliably detects a single over-threshold spike in a four-row calibration sweep.` | Witness: `One corrupted row affected three of the six pairwise slopes; measured-subset injections produced 1,304 misses in bank 1 alone and calibrated-signal drift as high as 5.54e3.` | Blanket Diagnosis: `The spike can shift the pilot enough to hide its own residual, so the solver skips the archive and inverts a corrupted affine gain.` | Sublated Pivot: `Compare the four three-row fits before thresholding the omitted point; retain direct in-batch fitting when at least four clean rows remain.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_s00_solver.py')
s = p.read_text()
anchor = 'def _calibrate(study):'
assert anchor in s
s = s.replace(anchor, '''def _clean_sweep(x, y):
    clean = np.ones(x.size, dtype=bool)
    if x.size >= 4:
        # Fit each one-point deletion. At four rows a single spike contaminates
        # half the pair slopes, but its deletion leaves three consistent rows.
        best_error = np.inf
        for i in range(x.size):
            keep = np.arange(x.size) != i
            if np.ptp(x[keep]) == 0:
                continue
            a, d = _affine(x[keep], y[keep])
            residual = y - (a*x + d)
            error = residual[keep] @ residual[keep]
            if error < best_error:
                best_error, omitted, best_residual = error, i, residual
        if np.isfinite(best_error):
            clean = np.abs(best_residual) <= 3.0
            if not clean[omitted] and clean.sum() == x.size - 1:
                return clean
    if x.size >= 3:
        # Use a robust pilot if no single deletion explains the contamination.
        ii, jj = np.triu_indices(x.size, 1)
        valid = x[ii] != x[jj]
        slopes = (y[jj[valid]] - y[ii[valid]]) / (x[jj[valid]] - x[ii[valid]])
        if slopes.size:
            a = np.median(slopes)
            d = np.median(y - a*x)
            return np.abs(y - (a*x + d)) <= 3.0
    return np.ones(x.size, dtype=bool)


def _calibrate(study):''', 1)
start = s.index('            fit = _affine(x, y)')
end = s.index('                continue\n', start) + len('                continue\n')
s = s[:start] + '''            clean = _clean_sweep(x, y)
            if clean.all() or clean.sum() >= 4:
                gain_offset[b] = _affine(x[clean], y[clean])
                continue
''' + s[end:]
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import itertools, json, pickle, runpy, tempfile
from pathlib import Path
import numpy as np
m = runpy.run_path('beta_m97_c03_s00_solver.py')
cal, solve = m['_calibrate'], m['solve_beta_m97_c03_s00']
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    cases = pickle.load(f)
with tempfile.TemporaryDirectory() as td:
    archive = Path(td) / 'calibration.json'
    cal.__globals__['_ARCHIVE'] = archive
    for item in cases:
        s, gt = item['study'], item['gt']; d = s['sensor_calibration_sweeps']
        bank, x, y = (d[key] for key in ('bank_id', 'ref_signal', 'raw_readout'))
        baseline = cal(s)
        out = solve(s)
        we = np.max(np.abs(out['weights']-gt['w_opt'])); fe = np.max(np.abs(out['terminal_field']-gt['terminal_field_A']))
        assert we < 1e-6 and fe < 1e-7, (we, fe)
        archive.write_text(json.dumps({'banks': {str(b): dict(zip(('gain','offset'), m['_affine'](x[bank == b], y[bank == b]))) for b in range(4)}}))
        for b in range(4):
            ids = np.flatnonzero(bank == b); others = np.flatnonzero(bank != b)
            rows = s['row_bank_id'] == b
            max_clean = max_emi = 0.; misses = count = 0
            for group in itertools.combinations(ids, 4):
                if np.ptp(x[list(group)]) < 0.3:
                    continue
                keep = np.r_[others, group]; sweeps = {k: v[keep] for k,v in d.items()}
                max_clean = max(max_clean, np.max(np.abs(cal(dict(s, sensor_calibration_sweeps=sweeps))[rows]-baseline[rows])))
                for j, spike in itertools.product(range(4), (-5., -3.0001, 3.0001, 5.)):
                    raw = sweeps['raw_readout'].copy(); raw[len(others)+j] += spike
                    error = np.max(np.abs(cal(dict(s, sensor_calibration_sweeps=dict(sweeps, raw_readout=raw)))[rows]-baseline[rows]))
                    max_emi = max(max_emi, error); misses += error > 1e-4; count += 1
            print(f'bank={b} clean drift={max_clean:.3g}, EMI misses={misses}/{count}, worst EMI drift={max_emi:.3g}', flush=True)
            assert max_clean < 1e-4 and misses == 0
            for selected in ([], [ids[0]]):
                keep = np.r_[others, np.array(selected, dtype=int)]
                np.testing.assert_array_equal(cal(dict(s, sensor_calibration_sweeps={k: v[keep] for k,v in d.items()}))[rows], baseline[rows])
        archive.write_text('must not be read when at least four clean rows remain')
        for b in range(4):
            ids = np.flatnonzero(bank == b); keep = np.r_[np.flatnonzero(bank != b), ids[[0, 3, 6, 9, 11]]]
            sweeps = {k: v[keep].copy() for k,v in d.items()}; sweeps['raw_readout'][-1] += 5
            np.testing.assert_allclose(cal(dict(s, sensor_calibration_sweeps=sweeps)), baseline, atol=1e-5, rtol=0)
        print(f'public weight error={we:.3g}, field error={fe:.3g}; archive gating checked at zero, one, and four surviving rows')
PY
```
- **Container Exit Code**: `0` (`50169 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
bank=0 clean drift=1.8e-06, EMI misses=0/7920, worst EMI drift=0
bank=1 clean drift=2.01e-06, EMI misses=0/7920, worst EMI drift=0
bank=2 clean drift=7.33e-07, EMI misses=0/7904, worst EMI drift=0
bank=3 clean drift=1.33e-06, EMI misses=0/7920, worst EMI drift=0
public weight error=3.98e-08, field error=1.48e-09; archive gating checked at zero, one, and four surviving rows

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07c7084cc96f71dd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_07c7084c (directory).
```

### Turn 11 — `submit_completed_task` (`15345 ms` LLM)
- **Rationale**: Isolated verification of the revised artifact reproduced the public accuracy and eliminated the demonstrated four-reading EMI failure, including injections just above the specified cutoff. The transport and inversion code did not change.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c03_s00_solver.py`. The public validator passed. Maximum absolute errors were 3.98e-8 for source weights and 1.48e-9 for the terminal field. The revised calibration detected all 31,664 injected single-spike cases in four-reading sweeps, and simulated archive fallback passed. The worst KKT duality gap across 252 stress cases was 2.18e-11.`

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
