# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_w03_batch_04__tracer_survey_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Tracer Reaction-Transport & Continual Sensor Tomography Stream` |
| **Wave / Wormhole Family** | `beta_w03_step_04` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `428,816` (`416,370` / `123,520` / `12,446`) |
| **Total Episode Wall Time (s)** | `403.35s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_11_split_separable_transport_inversio`**: [BIFURCATED SUB-REGIME `numerically_distinct_reference_levels_pe > 1`]: Treating the two axis reactions as a substitute for the explicit 2D potential failed: their maximum discrepancy was 0.61204165, and the validated stencil instead adds each independent axis reaction to its own diagonal alongside the corresponding component of the separable potential. Diagonal symmetrization of the harmonic-diffusion, upwind, Robin operators lets all shifted solves reuse the same modal source projections; the reusable solver accepts an additive potential only within a relative residual of 1e-10. Fit positive affine gains bank by bank using a median pairwise slope and inlier least squares rather than pooling banks or fitting all sweep points by raw L2: two numerically distinct reference levels are necessary for a direct estimate, whereas a spot check fixes only the offset conditional on a supplied gain. Apply the ridge residual-maker to both the design and observations, then solve each capped-simplex problem and certify that same objective with a convex tangent lower bound. The public host probe measured relative field and weight errors of 1e-8 and 5e-8 and a maximum KKT duality gap of 3.01e-8; it did not establish the persistence of calibration priors across epochs. (Counter-regime `numerically_distinct_reference_levels_pe` warning: At one level, use a caller-supplied positive gain and update only the offset; at zero levels, use the supplied gain and offset. More than one level permits a direct fit but does not make a contaminated sweep identifiable.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The supplied reaction_x and reaction_y vectors can be added directly to recover V_reaction_2d.` (witness: `The public study gives max |V_reaction_2d - reaction_x[:,None] - reaction_y[None,:]| = 0.6120416490895675.`) -> Pivot `Test the grid's mixed differences, then compare transport conventions against the public design. Retain the axis flux stencils and use the verified reaction representation.`
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`10830 ms` LLM)
- **Rationale**: Inspect the schema and prior solver before implementing: a stencil-sign, terminal-field, or certificate mismatch would change the implementation.

```bash
cd /holdout_task && ls -l && cat public_asset_index.json validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && grep -nE '^def |ridge|certificate|terminal|upwind' /tmp/hegel_continual_archive/beta_w03_batch_03_solver.py
```
- **Container Exit Code**: `0` (`259 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
total 68K
-rw-r--r-- 1 root root 8.4K Nov 26 18:18 archive_calibration_history.json
-rw-r--r-- 1 root root 1.4K Nov 26 18:18 archive_manifest.json
-rw-r--r-- 1 root root  12K Nov 26 18:18 beta_w03_batch_01_solver.py
-rw-r--r-- 1 root root  16K Nov 26 18:18 beta_w03_batch_02_solver.py
-rw-r--r-- 1 root root 7.1K Nov 26 18:18 beta_w03_batch_03_solver.py
-rw-r--r-- 1 root root 8.2K Nov 26 18:18 beta_w03_s00_solver.py
-rw-r--r-- 1 root root  156 Nov 26 18:18 pre_llm_gate_diagnostic.json
1:"""Separable finite-volume tomography with bank calibration and a box-simplex certificate."""
6:def _build_1d(L, n, kappa, vel, robin, diag_add):
18:def _axis_modes(operator):
19:    # D^-1 A D is symmetric: upwind changes the two off-diagonals differently.
28:def _responses(study):
44:def _fit_bank(x, y, prior):
63:def _calibrate(study):
93:def _project(v, cap, budget):
104:def _box_simplex(S, y, ridge, cap, budget):
106:    H, f = S.T @ S + ridge * np.eye(m), S.T @ y
135:    g = S.T @ residual + ridge * w
139:    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
145:def _linear_minimum(g, cap, budget):
150:def solve_beta_w03_batch_03(study: dict) -> dict:
159:        denominator = squared + study['nuisance_ridge']
162:    pairs = [_box_simplex(Y[:, :-1], Y[:, -1], study['source_ridge'], study['u_box'], float(b))
164:    weights, certificates = zip(*pairs)
165:    terminal = np.einsum('m,mxy->xy', weights[1], U[-1])
166:    return {'weights_by_budget': np.stack(weights), 'terminal_field_mid_budget': terminal,
167:            'kkt_certificates': list(certificates)}
```

### Turn 2 — `execute_in_remote_docker` (`14324 ms` LLM)
- **Rationale**: Check the archived observation and nuisance conventions, certificate tolerances, and per-bank sweep coverage before coding. Sparse or inconsistent sweeps would change how the calibration prior is used.

```bash
cat /tmp/hegel_continual_archive/beta_w03_batch_03_solver.py; sed -n '1,76p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling robust reference calibration box simplex' 2>&1; python3 - <<'PY'
import pickle, numpy as np, json
p='/holdout_task/public_beta_w03_batch_04.pkl'
items=pickle.load(open(p,'rb')); print('items:',len(items))
for item in items:
 s=item['study']; print({k:(v.shape if isinstance(v,np.ndarray) else v if np.isscalar(v) else (type(v).__name__,len(v))) for k,v in s.items()}); print('sweeps',s['sensor_calibration_sweeps']);
print('archive format:',str(json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')))[:1400])
PY
```
- **Container Exit Code**: `0` (`589 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
16534144, -0.56985451, -0.62599341,
        0.21318491, -0.68650801, -0.19488637,  0.95286967,  0.80201412,
       -0.84167445, -0.06885218, -0.8849101 ,  1.20630625, -1.02256115,
        1.00125616, -0.92883059,  0.36820511,  0.63865171,  1.01234529,
       -0.93971323,  0.81974861,  1.34230924, -0.25049604,  0.04115477,
       -0.72629715, -0.8279292 ,  1.30860868]), 'raw_readout': array([ 0.8310618 , -0.22543621, -0.82819762,  0.39352429,  0.38528165,
       -0.80918962, -0.53123614,  0.39445598,  0.24820241, -0.83489009,
        0.50644851,  0.04732398,  1.42784328,  0.0937792 ,  1.1248975 ,
        0.54869504,  1.81092117, -1.33166743, -0.85333081,  0.90279573,
        0.58958141,  1.77942856,  1.58993673, -0.56604423, -0.55223676,
        0.03846086, -0.59483298, -0.24878052,  0.55912497,  0.45293774,
       -0.70405465, -0.16006505, -0.73448821,  0.73751896, -0.8313808 ,
        0.5931842 , -1.01903521,  0.55128594,  0.87871562,  1.33114642,
       -1.0322108 ,  1.09796963,  1.73063379, -0.19777556,  0.15532607,
       -0.77382797, -0.89687388,  1.68983253])}
archive format: [{'episode_short_name': 'beta_w03_s00', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [0.245653, -1.113811, 0.027254, 1.414102, 0.553402, -0.851579, 0.787671, -0.107799, 0.245738, -1.146026, 1.513209, -0.674392, 1.014084, 1.462201, 0.58645, 1.180725, -0.030848, -0.07461, 1.459047, -0.786925, 0.667998, -0.583994, -1.037639, 1.45851, 0.767074, 0.882849, -1.013551, 1.380532, -0.597517, -0.777696, -0.317067, -0.439673, 0.957118, 0.162144, 0.898025, 0.76185, 0.156971, 1.234544, -0.333414, -0.75002, -0.825523, 0.165567, 0.864059, 0.667182, -0.850595, 1.292118, -0.513195, -0.069646], 'raw_readout': [0.245653, -1.113811, 0.027254, 1.414102, 0.553402, -0.851579, 0.787671, -0.107799, 0.245738, -1.146026, 1.513209, -0.674392, 1.014084, 1.462201, 0.58645, 1.180725, -0.030848, -0.07461, 1.459047, -0.786925, 0.667998, -0.583994, -1.037639, 1.45851, 0.767074, 0.882849, -1.013551, 1.380532, -0.597517, -0.777696, -0.317067, -0.439673, 0.957118, 0.162144, 0.898025, 0.76185, 0.156971, 1.234544, -0.333414, -0.75002, -0.825523, 0.165567, 0.864059, 0.667182, -0.850595, 1.292118, -0.513195, -0.069646]}, {'episode_short_name': 'beta_w03_batch_01', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3
```

### Turn 3 — `execute_in_remote_docker` (`17932 ms` LLM)
- **Rationale**: Recover the exact nuisance and certificate conventions. Measure cross-coupling and bank-level calibration residuals to decide whether separability or any archived calibration can be reused.

```bash
sed -n '44,170p' /tmp/hegel_continual_archive/beta_w03_batch_03_solver.py; sed -n '1,57p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; python3 - <<'PY'
import json,pickle,numpy as np
p='/holdout_task/public_beta_w03_batch_04.pkl'; items=pickle.load(open(p,'rb')); print('Public cases:',len(items)); hist=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for s in hist+[dict(item['study']['sensor_calibration_sweeps'],episode_short_name='current') for item in items]:
 print(s['episode_short_name'],end=': ')
 b,x,y=map(np.asarray,(s['bank_id'],s['ref_signal'],s['raw_readout']))
 for k in range(4):
  a,c=x[b==k],y[b==k]; i,j=np.triu_indices(len(a),1); good=np.abs(a[j]-a[i])>1e-10
  if good.any():
   g=np.median((c[j[good]]-c[i[good]])/(a[j[good]]-a[i[good]])); o=np.median(c-g*a); print((k,len(a),round(g,6),round(o,6),np.round(np.sort(abs(c-g*a-o))[-3:],7).tolist()),end=' ')
  else: print((k,len(a)),end=' ')
 print()
for item in items:
 v=item['study']['V_reaction_2d']; print('Nonseparable fraction:',np.linalg.norm(v-v.mean(0)-v.mean(1)[:,None]+v.mean())/np.linalg.norm(v))
PY
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ion.calibrate_affine_banks
autogenous_separable_transport_inversion@211159ab | fit_box_simplex(design: np.ndarray, observations: np.ndarray, nuisance: np.ndarray, cap: float, budget: float, *, nuisance_ridge: float=0.0, source_ridge: float=0.0, gap_rtol: float=1e-11, max_iter: int=10000) -> tuple[np.ndarray, dict[str, float | np.ndarray]] | Fit nonnegative capped weights with fixed sum and certify the objective | REQ: {'field': 'min_axis_length', 'op': '>', 'value': 0}; {'field': 'min_cell_diffusivity', ' | --describe autogenous_separable_transport_inversion.fit_box_simplex
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
Public cases: 1
beta_w03_s00: (0, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (1, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (2, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (3, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) 
beta_w03_batch_01: (0, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (1, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (2, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (3, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) 
beta_w03_batch_02: (0, 12, 0.7434, -0.1418, [5e-07, 7e-07, 1e-06]) (1, 12, 1.2425, 0.142, [4e-07, 4e-07, 5e-07]) (2, 12, 0.7039, -0.1116, [4e-07, 4e-07, 5e-07]) (3, 12, 1.2107, 0.1055, [3e-07, 7e-07, 8e-07]) 
beta_w03_batch_03: (0, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (1, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (2, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) (3, 12, 1.0, 0.0, [0.0, 0.0, 0.0]) 
current: (0, 12, 0.7434, -0.1418, [0.0, 0.0, 0.0]) (1, 12, 1.2425, 0.142, [0.0, 0.0, 0.0]) (2, 12, 0.7039, -0.1116, [0.0, 0.0, 0.0]) (3, 12, 1.2107, 0.1055, [0.0, 0.0, 0.0]) 
Nonseparable fraction: 0.5413482328861372
```

### Turn 4 — `execute_in_remote_docker` (`19168 ms` LLM)
- **Rationale**: Recover the observation row order, ridge profiling, and certificate sign conventions from the exact source; these determine whether a correct forward solve passes the public validator. Check the located primitives for a compatible implementation before reusing them.
- **Alternatives Weighed**: `Use the most recent archive calibration as the default` (rejected: Batch 03 has unit gains and zero offsets, whereas all four observed batch-04 sweeps agree with batch 02.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived additive decomposition into x-only and y-only reaction terms is adequate for batch 04.` | Witness: `The best additive decomposition leaves a relative Frobenius residual of 0.541348, above the specified 0.001 threshold.` | Blanket Diagnosis: `Axis averaging removes the spatial cross-coupling term, and gain or nuisance fitting cannot restore it to the transport operator.` | Sublated Pivot: `Retain the archived 1D flux stencil, assemble the full 2D reaction diagonal, and apply the specified 1.45 diffusivity multiplier.`

```bash
sed -n '93,167p' /tmp/hegel_continual_archive/beta_w03_batch_03_solver.py; sed -n '1,58p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_separable_transport_inversion.fit_box_simplex | head -65; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_separable_transport_inversion.calibrate_affine_banks | head -55; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -40
```
- **Container Exit Code**: `0` (`350 ms`, timed_out=`False`)
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

### Turn 5 — `execute_in_remote_docker` (`52974 ms` LLM)
- **Rationale**: Assemble the specified coupled stencil and preserve the archived box-simplex certificate convention. The isolated public validator will catch observation ordering, profiling, or certificate errors before testing terminal fields and sparse calibration.
- **Alternatives Weighed**: `Reuse the complete separable inversion primitive` (rejected: Its forward operator cannot represent the measured cross-coupling, and nuisance profiling needs the square root of the ridge Schur complement rather than ordinary residuals.)

```bash
set -e
cd /holdout_task
archive=/tmp/hegel_continual_archive/beta_w03_batch_03_solver.py
sed -n '150,166p' "$archive"
cat > beta_w03_batch_04_solver.py <<'PY'
"""Coupled finite-volume inversion for the batch-04 tracer survey."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


PY
sed -n '/^def _build_1d/,/^def _axis_modes/{ /^def _axis_modes/!p; }' "$archive" >> beta_w03_batch_04_solver.py
cat >> beta_w03_batch_04_solver.py <<'PY'
def _responses(study):
    nx, ny = study['nx'], study['ny']
    axes = [_build_1d(study['L' + a], study['n' + a],
                      1.45 * np.asarray(study['kappa_' + a]),
                      np.asarray(study['vel_' + a]), study['robin_' + a],
                      np.asarray(study['reaction_' + a])) for a in 'xy']
    base = kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc')
    potential = np.asarray(study['V_reaction_2d']).ravel()
    rhs = np.asarray(study['source_modes'], dtype=float).reshape(-1, nx * ny).T
    return np.stack([splu(base + diags(potential + shift, format='csc')).solve(rhs).T
                     .reshape(-1, nx, ny) for shift in study['mu_shifts']])


def _sweeps(record):
    banks, ref, raw = (np.asarray(record[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    return [(ref[banks == b], raw[banks == b]) for b in range(4)]


def _fit_sweep(x, y, prior):
    """Fit the current affine line, with the archived line breaking sparse ties."""
    gain, offset = prior
    if not x.size:
        return prior
    i, j = np.triu_indices(x.size, 1)
    span = x[j] - x[i]
    distinct = np.abs(span) > 1e-8
    i, j, span = i[distinct], j[distinct], span[distinct]
    slopes = (y[j] - y[i]) / span
    models = [(gain, offset), (gain, np.median(y - gain * x))]
    models.extend((a, y[k] - a * x[k]) for a, k in zip(slopes, i) if a > 0)
    med = np.median(slopes) if slopes.size else gain
    resid = y - med * x
    mad = np.median(np.abs(resid - np.median(resid)))
    # Do not let sparse, contaminated rows inflate the consensus radius.
    tol = np.clip(4 * mad, 2e-6, 0.02)
    def score(model):
        a, b = model
        err = np.abs(y - a * x - b)
        return ((err <= tol).sum(), -np.mean(np.minimum(err, tol)),
                -abs(a - gain) - abs(b - offset))
    best = max(models, key=score)
    keep = np.abs(y - best[0] * x - best[1]) <= tol
    if keep.sum() >= 3 and np.ptp(x[keep]) > 1e-8:
        a, b = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
        if a > 0:
            return a, b
    if np.max(np.abs(y - gain * x - offset)) <= 2e-6:
        return gain, np.median(y - gain * x)
    if x.size == 1 or (x.size == 2 and np.min(np.abs(y - gain * x - offset)) <= 2e-6):
        return prior
    return best


def _calibrate(study):
    samples = _sweeps(study['sensor_calibration_sweeps'])
    priors = [(1., 0.)] * 4
    if any(x.size < 4 or np.ptp(x) <= 1e-8 for x, _ in samples):
        path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        history = [[_fit_sweep(x, y, (1., 0.)) for x, y in _sweeps(record)]
                   for record in json.loads(path.read_text())] if path.is_file() else []
        if history:
            def loss(profile, banks):
                errors = [np.median(np.abs(y - profile[b][0] * x - profile[b][1]))
                          for b in banks for x, y in [samples[b]] if x.size]
                return sum(errors) / len(errors) if errors else 0.
            for wing in ((0, 1), (2, 3)):
                profile = min(reversed(history), key=lambda p: (loss(p, wing), loss(p, range(4))))
                for b in wing:
                    priors[b] = profile[b]
    gain, offset = np.asarray([_fit_sweep(x, y, prior) for (x, y), prior in zip(samples, priors)]).T
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw']) - offset[rows]) / gain[rows]


def _profile(values, nuisance, ridge):
    basis, sigma, _ = np.linalg.svd(nuisance, full_matrices=False)
    square = sigma ** 2
    denom = square + ridge
    retained = np.sqrt(np.divide(ridge, denom, out=np.ones_like(square), where=denom > 0))
    return values - basis @ ((1 - retained[:, None]) * (basis.T @ values))


PY
sed -n '/^def _project/,/^def solve_beta_w03_batch_03/{ /^def solve_beta_w03_batch_03/!p; }' "$archive" >> beta_w03_batch_04_solver.py
cat >> beta_w03_batch_04_solver.py <<'PY'
def solve_beta_w03_batch_04(study: dict) -> dict:
    responses = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    design = responses[:, :, i, j].transpose(0, 2, 1).reshape(-1, responses.shape[1])
    profiled = _profile(np.column_stack((design, _calibrate(study))),
                        study['Z_nuisance'], study['nuisance_ridge'])
    weights, certificate = _box_simplex(profiled[:, :-1], profiled[:, -1], study['source_ridge'],
                                       study['u_box'], study['total_budget'])
    terminal = np.einsum('m,mxy->xy', weights, responses[-1])
    return {'weights': weights, 'terminal_field': terminal, 'kkt_certificate': certificate}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`504 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
def solve_beta_w03_batch_03(study: dict) -> dict:
    U = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    S = U[:, :, i, j].transpose(0, 2, 1).reshape(-1, U.shape[1])
    Y = np.column_stack((S, _calibrate(study)))
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    if Z.shape[1]:
        Q, sigma, _ = np.linalg.svd(Z, full_matrices=False)
        squared = sigma**2
        denominator = squared + study['nuisance_ridge']
        shrink = np.divide(squared, denominator, out=np.zeros_like(squared), where=denominator > 0)
        Y -= Q @ (shrink[:, None] * (Q.T @ Y))
    pairs = [_box_simplex(Y[:, :-1], Y[:, -1], study['source_ridge'], study['u_box'], float(b))
             for b in study['budget_grid']]
    weights, certificates = zip(*pairs)
    terminal = np.einsum('m,mxy->xy', weights[1], U[-1])
    return {'weights_by_budget': np.stack(weights), 'terminal_field_mid_budget': terminal,
{"schema_valid": false, "errors": ["kkt_invalid:reported_primal_mismatch:1.466e+00"]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=caaa09093ddb52a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_caaa0909 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`24904 ms` LLM)
- **Rationale**: Compare the public reference design and observations separately. If the archived ridge residual transform matches, change profiling; if neither transform matches on the specified operator, investigate the stencil or observation ordering before changing the certificate.
- **Alternatives Weighed**: `Switch immediately to the archived residual transform` (rejected: The primal mismatch alone does not distinguish a profiling convention mismatch from a forward-model mismatch.)

```bash
set -e
cd /holdout_task
cat validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy,pickle,numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
f=runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
a=pickle.load(open('public_beta_w03_batch_04.pkl','rb'))[0]; s,g=a['study'],a['gt']
print('gt keys:',{k:getattr(v,'shape',type(v).__name__) for k,v in g.items()})
out=f['solve_beta_w03_batch_04'](s); print('reported cert scalars:',{k:v for k,v in out['kkt_certificate'].items() if np.isscalar(v)}); print('reference cert scalars:',{k:v for k,v in g['cert_opt'].items() if np.isscalar(v)})
z=s['Z_nuisance']; P=np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
i,j=s['sensor_ij'].T; y=f['_calibrate'](s); nx,ny=s['nx'],s['ny']; ref=g['S_tilde']; yr=g.get('y_tilde')
if yr is not None: print('y relative error raw, P, sqrtP:',[np.linalg.norm(q-yr)/np.linalg.norm(yr) for q in (y,P@y,f['_profile'](y[:,None],z,s['nuisance_ridge'])[:,0])])
for scale in (1.,1.45):
 A,B=[f['_build_1d'](s['L'+k],s['n'+k],scale*s['kappa_'+k],s['vel_'+k],s['robin_'+k],s['reaction_'+k]) for k in 'xy']; base=kron(A,eye(ny),format='csc')+kron(eye(nx),B,format='csc'); V=s['V_reaction_2d'];
 for vp in (V,V.mean(0)+V.mean(1)[:,None]-V.mean(),np.zeros_like(V)):
  U=np.stack([splu(base+diags(vp.ravel()+mu,format='csc')).solve(s['source_modes'].reshape(12,-1).T).T.reshape(12,nx,ny) for mu in s['mu_shifts']]); S=U[:,:,i,j].transpose(0,2,1).reshape(-1,12)
  print('scale / potential defect / design errors raw, P, sqrtP:',scale, np.linalg.norm(vp-V),[np.linalg.norm(q-ref)/np.linalg.norm(ref) for q in (S,P@S,f['_profile'](S,z,s['nuisance_ridge']))])
PY
```
- **Container Exit Code**: `0` (`618 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
beta_w03_batch_04.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_w03_batch_04(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
gt keys: {'S_tilde': (288, 12), 'r_y': (288,), 'w_opt': (12,), 'cert_opt': 'dict', 'terminal_field_A': (40, 36)}
reported cert scalars: {'nu_sum': 0.008363044505971192, 'primal_objective': 1.499091118572636, 'dual_lower_bound': 1.4990911185726359}
reference cert scalars: {'nu_sum': 0.008555934924422548, 'primal_objective': 0.033157566471961525, 'dual_lower_bound': 0.033157566471961504}
scale / potential defect / design errors raw, P, sqrtP: 1.0 0.0 [0.2133040707110568, 0.17492719968711418, 0.175019333590784]
scale / potential defect / design errors raw, P, sqrtP: 1.0 38.27059804135258 [0.17269693846836576, 0.12858949235344339, 0.12870079214795457]
scale / potential defect / design errors raw, P, sqrtP: 1.0 70.69497176949704 [0.8286321063752942, 0.8094169454263435, 0.8094603820574021]
scale / potential defect / design errors raw, P, sqrtP: 1.45 0.0 [0.10231803833773492, 7.115004702461674e-15, 0.0047204560793336495]
scale / potential defect / design errors raw, P, sqrtP: 1.45 38.27059804135258 [0.11844326654347072, 0.06665417150746847, 0.06680639936417447]
scale / potential defect / design errors raw, P, sqrtP: 1.45 70.69497176949704 [0.5739299706867255, 0.5550917022204562, 0.5551339470681685]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=caaa09093ddb52a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_caaa0909 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`24084 ms` LLM)
- **Rationale**: Apply the isolated profiling correction. The validator checks the certificate against public reference quantities; separate comparisons will show whether observation calibration or terminal-field construction still needs work.
- **Alternatives Weighed**: `Change the stencil or certificate independently of profiling` (rejected: The existing stencil reproduces the public design to roundoff with the ridge-residual transform.)
- **ATMS NOGOOD DECLARED**: Refuted `The certified objective uses square-root Schur-complement whitening.` | Witness: `For the specified full operator, the square-root transform misses the public design by 4.720e-3 relative, while the archived ridge-residual transform matches to 7.115e-15.` | Blanket Diagnosis: `The square-root transform represents a different ridge objective; the report squares residuals after applying the ridge-residual transform itself.` | Sublated Pivot: `Keep the coupled FVM and replace only the profiling transform, then verify all outputs independently.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_w03_batch_04_solver.py')
s = p.read_text()
a = s.index('    retained = np.sqrt(')
b = s.index('\n\n\n', a)
s = s[:a] + '''    # The report's objective uses squared ridge-adjusted residuals.
    shrink = np.divide(square, denom, out=np.zeros_like(square), where=denom > 0)
    return values - basis @ (shrink[:, None] * (basis.T @ values))''' + s[b:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
sample = pickle.load(open('/holdout_task/public_beta_w03_batch_04.pkl','rb'))[0]
s, gt = sample['study'], sample['gt']
out = f['solve_beta_w03_batch_04'](s)
i,j = s['sensor_ij'].T
U = f['_responses'](s)
S = U[:,:,i,j].transpose(0,2,1).reshape(-1,U.shape[1])
Y = f['_profile'](np.column_stack((S,f['_calibrate'](s))),s['Z_nuisance'],s['nuisance_ridge'])
for name, estimate, ref in [('design',Y[:,:-1],gt['S_tilde']),('observations',Y[:,-1],gt['r_y']),('weights',out['weights'],gt['w_opt']),('terminal',out['terminal_field'],gt['terminal_field_A'])]:
    error = np.linalg.norm(estimate-ref)/np.linalg.norm(ref)
    print(name,'relative_error',error)
    assert error < 1e-6, (name,error)
c = out['kkt_certificate']
print('certificate gap',c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`608 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
design relative_error 7.078399940919918e-15
observations relative_error 6.649740711424818e-07
weights relative_error 4.9076124608975445e-08
terminal relative_error 3.039165037452239e-08
certificate gap 6.938893903907228e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1082d61197d38eb4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1082d611 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`27857 ms` LLM)
- **Rationale**: The exact candidate passes on complete sweeps. Withhold reference rows and add wing-local EMI to test whether bank calibration fails under the stated sparse and contaminated regimes; if it does, revise the archive selection or affine fit rather than the verified transport operator.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
from copy import deepcopy
import numpy as np

f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
with open('/holdout_task/public_beta_w03_batch_04.pkl', 'rb') as fp:
    item = pickle.load(fp)[0]
study, gt = item['study'], item['gt']
key = 'sensor_calibration_sweeps'
bank, ref, raw = (np.asarray(study[key][k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
worst = np.zeros(3)

def check(label, keep, spikes=()):
    global worst
    case = deepcopy(study)
    shifted = raw.copy()
    for i, delta in spikes:
        shifted[i] += delta
    case[key] = dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=shifted[keep])
    out = f['solve_beta_w03_batch_04'](case)
    w, t = out['weights'], out['terminal_field']
    c = out['kkt_certificate']
    S, y = gt['S_tilde'], gt['r_y']
    r = S @ w - y
    g = S.T @ r + study['source_ridge'] * w
    p = 0.5 * (r @ r + study['source_ridge'] * (w @ w))
    nu = c['nu_sum']
    lb = p - g @ w + nu * study['total_budget'] + study['u_box'] * np.minimum(g - nu, 0).sum()
    scores = [np.linalg.norm(w - gt['w_opt']) / np.linalg.norm(gt['w_opt']), np.linalg.norm(t - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A']), p - min(lb, c['dual_lower_bound'])]
    worst = np.maximum(worst, scores)
    assert abs(c['primal_objective'] - p) < 2e-4 and scores[2] < 2e-4 and max(scores[:2]) < 1e-5, (label, scores)

for i in range(len(bank)):
    check(f'leave reference row {i} out', np.arange(len(bank)) != i)
for wing in ((0, 1), (2, 3)):
    idx = [np.flatnonzero(bank == b) for b in wing]
    check(f'EMI wing {wing}', np.ones(len(bank), dtype=bool), [(j, d) for ids in idx for j, d in [(ids[np.argmin(ref[ids])], -0.18), (ids[np.argmax(ref[ids])], 0.18)]])
    for missing, anchor in (idx, idx[::-1]):
        keep = ~np.isin(bank, wing)
        keep[anchor[np.argmax(np.abs(ref[anchor]))]] = True
        check(f'missing bank {bank[missing[0]]}, one wing spot check', keep)
print('Worst across 54 public-derived calibration slices (weight rel L2, field rel L2, external KKT gap):', worst.tolist())
PY
```
- **Container Exit Code**: `0` (`2372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst across 54 public-derived calibration slices (weight rel L2, field rel L2, external KKT gap): [1.0210435454251054e-06, 3.7778000237981915e-07, 8.686102185087585e-07]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1082d61197d38eb4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1082d611 (directory).
```

### Turn 9 — `submit_completed_task` (`13265 ms` LLM)
- **Rationale**: The unchanged candidate passed isolated public validation, independent reference comparisons, and stress tests for sparse calibration and wing-local interference. Both structural findings are resolved on the verified artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_w03_batch_04_solver.py': 'FILE:7674'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_w03_batch_04_solver.py`. It passed the public validator and 54 calibration stress tests. Worst-case relative errors were 1.02e-6 for weights and 3.78e-7 for the terminal field; the largest independently evaluated KKT gap was 8.69e-7.`

### Turn 10 — `execute_in_remote_docker` (`32363 ms` LLM)
- **Rationale**: The prior stress tests treated sparsity and EMI separately. Their combination can make a single bad point choose the wrong archived calibration; if that happens, revise archive selection rather than the verified transport model. The terminal residual separately detects an incorrect terminal shift or source combination.
- **Alternatives Weighed**: `Test relay modes, drift equations, or classification confounding` (rejected: This steady tracer inversion has no relay modes, time-dependent drift model, or classification labels. Its relevant nuisance test is whether a contaminated spot check overrides stronger calibration evidence from the other wing.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
from scipy.sparse import diags, eye, kron
f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
with open('public_beta_w03_batch_04.pkl', 'rb') as fp:
    item = pickle.load(fp)[0]
s, gt = item['study'], item['gt']
nx, ny = s['nx'], s['ny']
axes = [f['_build_1d'](s['L'+a], s['n'+a], 1.45*s['kappa_'+a], s['vel_'+a], s['robin_'+a], s['reaction_'+a]) for a in 'xy']
base = kron(axes[0], eye(ny)) + kron(eye(nx), axes[1])
original = f['solve_beta_w03_batch_04'](s)
rhs = original['weights'] @ s['source_modes'].reshape(len(original['weights']), -1)
A = base + diags(s['V_reaction_2d'].ravel() + s['mu_shifts'][-1])
res = np.linalg.norm(A @ original['terminal_field'].ravel() - rhs) / np.linalg.norm(rhs)
print('Terminal weighted-source FVM residual:', res)
assert res < 1e-10
B, X, Y = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
S, y = gt['S_tilde'], gt['r_y']
worst = {}
for wing in ((0,1), (2,3)):
    for target in wing:
        ids = np.flatnonzero(B == target)
        anchor = ids[np.argmax(np.abs(X[ids]))]
        for regime, delta in [('missing wing', 0.), ('one clean spot', 0.), ('one EMI spot', -.18), ('one EMI spot', .18)]:
            mask = ~np.isin(B, wing)
            raw = Y.copy()
            if regime != 'missing wing':
                mask[anchor] = True
                raw[anchor] += delta
            case = dict(s, sensor_calibration_sweeps=dict(bank_id=B[mask], ref_signal=X[mask], raw_readout=raw[mask]))
            out = f['solve_beta_w03_batch_04'](case)
            w, c = out['weights'], out['kkt_certificate']
            r = S @ w - y
            g = S.T @ r + s['source_ridge'] * w
            p = .5 * (r @ r + s['source_ridge'] * (w @ w))
            nu = c['nu_sum']
            lb = p - g @ w + nu * s['total_budget'] + s['u_box'] * np.minimum(g - nu, 0).sum()
            metrics = [np.linalg.norm(w-gt['w_opt'])/np.linalg.norm(gt['w_opt']), np.linalg.norm(out['terminal_field']-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']), abs(c['primal_objective']-p), p-min(lb,c['dual_lower_bound']), np.max(np.abs(g-nu-np.asarray(c['lam_lower'])+np.asarray(c['lam_upper'])))]
            worst[regime] = np.maximum(worst.get(regime, np.zeros(len(metrics))), metrics)
for regime, metrics in worst.items():
    print(regime, '(weight rel, field rel, primal error, external gap, stationarity):', metrics.tolist())
PY
```
- **Container Exit Code**: `0` (`1293 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Terminal weighted-source FVM residual: 1.8512452281622332e-14
missing wing (weight rel, field rel, primal error, external gap, stationarity): [8.780153003094224e-07, 4.2297385701642794e-07, 1.1459323454032777e-07, 6.318108785319243e-07, 8.344004547543304e-08]
one clean spot (weight rel, field rel, primal error, external gap, stationarity): [1.0210435454251054e-06, 3.7778000237981915e-07, 1.2637880136834312e-07, 8.686102185087585e-07, 1.051966206158661e-07]
one EMI spot (weight rel, field rel, primal error, external gap, stationarity): [8.780153003094224e-07, 4.2297385701642794e-07, 1.1459323454032777e-07, 6.318108785319243e-07, 8.344004547543304e-08]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1082d61197d38eb4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1082d611 (directory).
```

### Turn 11 — `submit_completed_task` (`13502 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated public validation, independent output comparisons, and combined sparse-wing and EMI stress. The terminal field satisfied the weighted-source finite-volume equation to a relative residual of 1.85e-14. The focused audit found no failure that warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/beta_w03_batch_04_solver.py': 'FILE:7674'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_w03_batch_04_solver.py`. It passed the public validator in isolated Python and stress tests covering missing calibration sweeps and EMI spikes. Across the tested slices, the worst relative errors were 1.02e-6 for source weights and 4.23e-7 for the terminal field. The largest independently evaluated KKT gap was 8.69e-7, below the 1e-6 tolerance.`

### Turn 12 — `execute_in_remote_docker` (`27822 ms` LLM)
- **Rationale**: The prior combined test sampled only large-reference anchors. An EMI spike at another measured reference value could favor an incompatible archive profile. If that exhausts the certificate margin, change how sparse wing evidence is weighed against the intact wing; the verified transport stencil does not need another check.

```bash
set -e
cd /holdout_task
cat > /tmp/beta_w03_batch_04_margin.py <<'PY'
import pickle, runpy
import numpy as np

f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
with open('/holdout_task/public_beta_w03_batch_04.pkl', 'rb') as fp:
    item = pickle.load(fp)[0]
s, gt = item['study'], item['gt']
bank, ref, raw = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
rows = np.asarray(s['row_bank_id'])
S, y = gt['S_tilde'], gt['r_y']
base = f['_calibrate'](s)
records = []
for b in range(4):
    wing = b // 2
    ids = np.flatnonzero(bank == b)
    mask = (bank // 2 != wing)
    for anchor in ids:
        keep = mask.copy()
        keep[anchor] = True
        for delta in (-0.18, 0., 0.18):
            altered = raw.copy()
            altered[anchor] += delta
            case = dict(s, sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=altered[keep]))
            error = np.max(np.abs(f['_calibrate'](case)[rows // 2 == wing] - base[rows // 2 == wing]))
            records.append((error, b, int(anchor), delta, case))
for regime, selected in [('clean spot', [r for r in records if r[3] == 0]), ('EMI spot', [r for r in records if r[3] != 0])]:
    error, b, anchor, delta, case = max(selected, key=lambda r: r[0])
    out = f['solve_beta_w03_batch_04'](case)
    w, c = out['weights'], out['kkt_certificate']
    r = S @ w - y
    grad = S.T @ r + s['source_ridge'] * w
    p = .5 * (r @ r + s['source_ridge'] * (w @ w))
    nu = c['nu_sum']
    lb = p - grad @ w + nu * s['total_budget'] + s['u_box'] * np.minimum(grad - nu, 0).sum()
    print(regime, 'slices', len(selected), 'bank/anchor/EMI', (b, anchor, delta), 'max corrected observation drift', error, 'weight relative error', np.linalg.norm(w-gt['w_opt'])/np.linalg.norm(gt['w_opt']), 'field relative error', np.linalg.norm(out['terminal_field']-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']), 'external gap', p-min(lb,c['dual_lower_bound']))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/beta_w03_batch_04_margin.py
```
- **Container Exit Code**: `0` (`2363 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clean spot slices 48 bank/anchor/EMI (0, 0, 0.0) max corrected observation drift 1.4750900936277844e-06 weight relative error 1.0210435454251054e-06 field relative error 3.7778000237981915e-07 external gap 2.8387519203365574e-07
EMI spot slices 96 bank/anchor/EMI (2, 24, -0.18) max corrected observation drift 1.954227217060776 weight relative error 0.7947360121144039 field relative error 0.32284708920451977 external gap 0.12220605376828621

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1082d61197d38eb4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_1082d611 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`26310 ms` LLM)
- **Rationale**: Change only archive selection and rerun the exact witnessed failure and the public gate. A remaining above-tolerance gap would call for inspecting affine fit residuals or archive precision, not changing the verified transport stencil.
- **Alternatives Weighed**: `Always force both wings to use the same historical profile` (rejected: That would discard corroborated local calibration evidence when the wings differ.)
- **ATMS NOGOOD DECLARED**: Refuted `A lone calibration spot is reliable enough to select a wing's archive profile ahead of an intact bank's sweep.` | Witness: `Adding a −0.18 EMI spike to bank 2's measured anchor 24 caused a 1.95 maximum corrected-observation drift, 0.795 relative weight error, and 0.122 external KKT gap.` | Blanket Diagnosis: `The sparse wing's single contaminated measurement dominated the profile ranking despite corroborating measurements from the intact wing. The affine calibration model and transport stencil were not the source of the failure.` | Sublated Pivot: `Give local matches priority only when at least two rows corroborate the profile; otherwise rank profiles by agreement across the batch.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_w03_batch_04_solver.py')
s = p.read_text()
a = s.index('            def loss(profile, banks):')
b = s.index('                for b in wing:', a)
s = s[:a] + '''            def support(profile, banks):
                return sum(np.count_nonzero(np.abs(y - profile[b][0] * x - profile[b][1]) <= 2e-6)
                           for b in banks for x, y in [samples[b]])
            for wing in ((0, 1), (2, 3)):
                # One spot can agree by chance after EMI. Only corroborated
                # local matches take precedence over evidence across the batch.
                def score(profile):
                    local = support(profile, wing)
                    return (local if local >= 2 else 0, support(profile, range(4)))
                profile = max(reversed(history), key=score)
''' + s[b:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/beta_w03_batch_04_margin.py
```
- **Container Exit Code**: `0` (`2811 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
clean spot slices 48 bank/anchor/EMI (0, 0, 0.0) max corrected observation drift 1.4750900936277844e-06 weight relative error 1.0210435454251054e-06 field relative error 3.7778000237981915e-07 external gap 2.8387519203365574e-07
EMI spot slices 96 bank/anchor/EMI (0, 0, -0.18) max corrected observation drift 1.229032215555037e-06 weight relative error 6.800106623997032e-07 field relative error 3.865082638266074e-07 external gap 4.9189899160861916e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6ba2d418861aaf2a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6ba2d418 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`28794 ms` LLM)
- **Rationale**: The previous audit solved only the slice with maximum observation drift, which need not maximize weight, field, or certificate error. Solve all 144 slices and replay the exact failure before submitting; if any gap exceeds tolerance, the reported bank and anchor will identify where to inspect calibration.

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle, runpy
import numpy as np

f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
with open('public_beta_w03_batch_04.pkl', 'rb') as fp:
    item = pickle.load(fp)[0]
s, gt = item['study'], item['gt']
bank, ref, raw = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
S, y = gt['S_tilde'], gt['r_y']
regimes = {'clean': [], 'EMI': []}
for anchor, b in enumerate(bank):
    keep = bank // 2 != b // 2
    keep[anchor] = True
    for delta in (0., -.18, .18):
        altered = raw.copy()
        altered[anchor] += delta
        case = dict(s, sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=altered[keep]))
        out = f['solve_beta_w03_batch_04'](case)
        w, c = out['weights'], out['kkt_certificate']
        r = S @ w - y
        g = S.T @ r + s['source_ridge'] * w
        p = .5 * (r @ r + s['source_ridge'] * (w @ w))
        nu = c['nu_sum']
        lb = p - g @ w + nu * s['total_budget'] + s['u_box'] * np.minimum(g - nu, 0).sum()
        metrics = [np.linalg.norm(w - gt['w_opt']) / np.linalg.norm(gt['w_opt']), np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A']), abs(c['primal_objective'] - p), p - min(lb, c['dual_lower_bound'])]
        regimes['EMI' if delta else 'clean'].append((metrics, (int(b), anchor, delta)))
        if anchor == 24 and b == 2 and delta == -.18:
            print('Exact EMI regression witness (weight, field, primal mismatch, external gap):', metrics)
        assert abs(w.sum() - s['total_budget']) <= 1e-7 and w.min() >= -1e-7 and w.max() <= s['u_box'] + 1e-7
for regime, records in regimes.items():
    worst = [max((m[i], label) for m, label in records) for i in range(4)]
    print(regime, len(records), 'slices; per-metric maximum and (bank, anchor, spike):', worst)
    assert worst[3][0] <= 1e-6 and worst[2][0] <= 1e-6 and max(worst[0][0], worst[1][0]) <= 1e-5
PY
```
- **Container Exit Code**: `0` (`6539 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Exact EMI regression witness (weight, field, primal mismatch, external gap): [8.780153003094224e-07, 4.2297385701642794e-07, 1.1459323454032777e-07, 6.318108785319243e-07]
clean 48 slices; per-metric maximum and (bank, anchor, spike): [(1.3738124608167096e-06, (0, 2, 0.0)), (7.287284585134929e-07, (3, 36, 0.0)), (1.2695116750199853e-07, (3, 47, 0.0)), (8.801097636929045e-07, (3, 47, 0.0))]
EMI 96 slices; per-metric maximum and (bank, anchor, spike): [(8.780153003094224e-07, (3, 47, 0.18)), (4.2297385701642794e-07, (3, 47, 0.18)), (1.1459323454032777e-07, (3, 47, 0.18)), (6.318108785319243e-07, (3, 47, 0.18))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6ba2d418861aaf2a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6ba2d418 (directory).
```

### Turn 15 — `submit_completed_task` (`12856 ms` LLM)
- **Rationale**: The final artifact passed isolated public validation, independent certificate checks across all 144 stress cases, and replay of the exact EMI failure. All recorded structural findings are resolved.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_w03_batch_04_solver.py`. The solver passed the public validator in isolated Python and all 144 calibration stress cases. The worst relative errors were 1.37e-6 for source weights and 7.29e-7 for the terminal field. The largest independently evaluated KKT gap was 8.80e-7, below the 1e-6 tolerance.`

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
