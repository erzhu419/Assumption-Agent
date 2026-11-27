# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c01_batch_01__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `233,831` (`225,654` / `82,304` / `8,177`) |
| **Total Episode Wall Time (s)** | `286.01s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15697 ms` LLM)
- **Rationale**: Identify the active boundary regime, the callable contract, and any reusable nuisance-projection primitive before implementing the inverse solve.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The public gate flags a downstream mismatch; inspect the apparatus log and solver before choosing components to reuse.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py /holdout_task/public_asset_index.json; cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/archive_apparatus_log.json; wc -l /tmp/hegel_continual_archive/*.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling capped simplex quadratic programming'
```
- **Container Exit Code**: `0` (`368 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`23004 ms` LLM)
- **Rationale**: The previous output was truncated. Inspect the prior solver and summarize the log and input shapes to determine which boundary and calibration components can be reused.
- **Alternatives Weighed**: `Infer wall conditions from the production observation alone` (rejected: Unknown source weights and nuisance responses can absorb wall-model error; the apparatus log provides independent evidence.)

```bash
cat /tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -25; python3 - <<'PY'
import json, pickle, numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p)); print('log:',type(a).__name__)
for e in (a if isinstance(a,list) else a.values()):
 if isinstance(e,dict):
  for k,v in e.items(): print(k, repr(v)[:350], np.shape(v) if isinstance(v,list) else '')
 else: print(repr(e)[:350])
print('index:',str(json.load(open('/holdout_task/public_asset_index.json')))[:2400])
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 print('item',[(k,type(v).__name__) for k,v in item.items()]); print('diagnostics',str(item.get('campaign_diagnostics'))[:150]);
 for k,v in item['study'].items():
  if isinstance(v,dict): print(k, [(kk,type(vv).__name__,getattr(vv,'shape',None)) for kk,vv in v.items()])
  else: print(k, type(v).__name__, getattr(v,'shape',None), repr(v)[:100] if not isinstance(v,np.ndarray) else (float(np.min(v)),float(np.max(v))))
PY
```
- **Container Exit Code**: `0` (`546 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'ndarray'}, 'kappa_x': {'dtype': 'float64', 'shape': [14], 'type': 'ndarray'}, 'kappa_y': {'dtype': 'float64', 'shape': [12], 'type': 'ndarray'}, 'mu_shifts': {'dtype': 'float64', 'shape': [3], 'type': 'ndarray'}, 'nuisance_ridge': 0.08, 'nx': 14, 'ny': 12, 'robin_x': 0.8942552148014796, 'robin_y': 1.163864943166582, 'row_bank_id': {'dtype': 'int64', 'shape': [96], 'type': 'ndarray'}, 'sensor_calibration_sweeps': {'raw_rosette': {'dtype': 'float64', 'shape': [80, 4], 'type': 'ndarray'}, 'ref_rosette': {'dtype': 'float64', 'shape': [80, 4], 'type': 'ndarray'}, 'single_bank_sweeps': {'bank_id': 'ndarray', 'raw_readout': 'ndarray', 'ref_signal': 'ndarray'}}, 'sensor_ij': {'dtype': 'int64', 'shape': [32, 2], 'type': 'ndarray'}, 'source_modes': {'dtype': 'float64', 'shape': [8, 14, 12], 'type': 'ndarray'}, 'source_ridge': 0.04, 'total_budget': 1.4, 'u_box': 0.42, 'vel_x': {'dtype': 'float64', 'shape': [14], 'type': 'ndarray'}, 'vel_y': {'dtype': 'float64', 'shape': [12], 'type': 'ndarray'}, 'y_obs_raw': {'dtype': 'float64', 'shape': [96], 'type': 'ndarray'}}}, 'len': 1, 'type': 'list'}}, '/holdout_task/validate_public_schema.py': {'bytes': 1970}}, 'schema': 'hegel_public_asset_index_v1'}
item [('study', 'dict'), ('campaign_diagnostics', 'list')]
diagnostics [{'w_pulse': array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]), 'y_pulse_raw': array([ 0.42137666,  0.21632042,  0.38722578,  0.25459585,  0.433499  ,
 
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) (0.4503444980474903, 0.5893237404212179)
kappa_y ndarray (12,) (0.33243569115070254, 0.5724356911507026)
vel_x ndarray (14,) (-0.20572483830236554, 0.35)
vel_y ndarray (12,) (-0.279885813991893, -0.0)
robin_x float None 0.8942552148014796
robin_y float None 1.163864943166582
V_reaction_2d ndarray (14, 12) (0.25, 0.42738476356834465)
mu_shifts ndarray (3,) (0.15, 1.1)
source_modes ndarray (8, 14, 12) (1.9454012254756794e-14, 0.9899685283867117)
sensor_ij ndarray (32, 2) (0.0, 12.0)
row_bank_id ndarray (96,) (0.0, 3.0)
sensor_calibration_sweeps [('ref_rosette', 'ndarray', (80, 4)), ('raw_rosette', 'ndarray', (80, 4)), ('single_bank_sweeps', 'dict', None)]
Z_nuisance ndarray (96, 3) (-2.2881079651862244, 2.985201300405091)
y_obs_raw ndarray (96,) (-0.1428059132172546, 0.4151802167628829)
aux_manifold_telemetry ndarray (24, 4) (-2.3716313552199875, 1.8300137159846437)
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
```

### Turn 3 — `execute_in_remote_docker` (`26866 ms` LLM)
- **Rationale**: Measure calibration error on paired references and boundary error on known-source pulses before changing either component; inspect the archived optimizer to see whether it can be reused.

```bash
python3 - <<'PY'
import pickle,numpy as np,runpy
m=runpy.run_path('/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py'); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; X=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(X,y,rcond=None)[0]; print('affine',B.round(6),'RMSE',np.sqrt(np.mean((X@B-y)**2,axis=0))); print('banks sequence',s['row_bank_id'][:40]); print('single-bank shapes and head',[(k,v.shape,v[:4]) for k,v in sw['single_bank_sweeps'].items()]); S,U=m['_forward'](s); print('telemetry',s['aux_manifold_telemetry'][:7]);
for p in i['campaign_diagnostics']:
 t=dict(s,y_obs_raw=p['y_pulse_raw']); z=m['_calibrate'](t); f=S@p['w_pulse']; d=(z-f).reshape(3,-1); print('pulse',p['w_pulse'],'calibrated',z[:8],'baseline',f[:8],'shift RMSE',np.sqrt(np.mean(d*d,axis=1)), 'means',d.mean(axis=1))
PY
sed -n '50,140p' /tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
active[[lo, hi]] = 0
            continue

        h = hessian[np.ix_(free, free)]
        system = np.ones((len(free) + 1, len(free) + 1))
        system[:-1, :-1] = h
        system[:-1, -1] = -1
        system[-1, -1] = 0
        direction_nu = np.linalg.solve(system, np.r_[-g[free], budget - w.sum()])
        direction, nu = direction_nu[:-1], float(direction_nu[-1])
        # Only truncate when the full equality-constrained step crosses a bound.
        limits = np.full(len(free), np.inf)
        down = (direction < 0) & (w[free] + direction < -1e-14 * cap)
        up = (direction > 0) & (w[free] + direction > cap + 1e-14 * cap)
        limits[down] = -w[free[down]] / direction[down]
        limits[up] = (cap - w[free[up]]) / direction[up]
        hit = int(np.argmin(limits))
        if limits[hit] < 1:
            w[free] += max(0., limits[hit]) * direction
            idx = free[hit]
            active[idx] = 1 if up[hit] else -1
            w[idx] = cap if up[hit] else 0.
            continue

        w[free] = np.clip(w[free] + direction, 0, cap)
        g = hessian @ w + linear
        violation = active * (g - nu)
        release = int(np.argmax(violation))
        if violation[release] <= tol:
            return w, nu
        active[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    data = np.column_stack((design, y))
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    s, y = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex_qp(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }
```

### Turn 4 — `execute_in_remote_docker` (`36488 ms` LLM)
- **Rationale**: Compare reflecting and periodic stencils first, then use an edge-coefficient fit with one pulse held out to identify any missing coupling before implementing the chamber model.
- **Alternatives Weighed**: `Fit a nonlinear rosette model` (rejected: The affine fit residual is already about four orders of magnitude smaller than the largest tracer discrepancy.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived open-wall Robin boundary describes the active chamber.` | Witness: `Both known-source tracer pulses disagree systematically after calibration; their lowest-shift mean residuals are 0.218 and 0.149.` | Blanket Diagnosis: `Paired calibration sweeps isolate the sensor transform, and known tracer weights remove production source ambiguity.` | Sublated Pivot: `Keep the upwind interior and affine calibration; compare discrete wall closures and fit edge coefficients only to diagnose missing stencil terms.`

```bash
cat > /tmp/boundary_probe.py <<'PY'
import pickle, runpy
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
old=runpy.run_path('/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; p=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
W=np.column_stack([a['w_pulse'] for a in p]); Y=np.column_stack([old['_calibrate'](dict(s,y_obs_raw=a['y_pulse_raw'])) for a in p]).reshape(3,len(rows),-1)
R=np.stack([x.ravel(order='F') for x in s['source_modes']],axis=1)@W
base=[]; edge=[]; face=[]; robin=[]
for ax,n in [('x',nx),('y',ny)]:
 h=s['L'+ax]/(n-1); k=s['kappa_'+ax]; rob=s['robin_'+ax]; base.append(old['_axis'](k,s['vel_'+ax],h,rob).toarray()); edge.extend(k[[0,-1]]/h**2); face.extend(0.5*(k[[0,-1]]+k[[1,-2]])/h**2); robin.extend([rob/h]*2)
edge,face,robin=map(np.array,(edge,face,robin)); ident=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def predict(a,b,c):
 axes=[x.copy() for x in base]
 for k,axis in enumerate(axes):
  for j,(i,inn,opp) in enumerate([(0,1,-1),(-1,-2,0)],start=2*k):
   axis[i,i]=a[j]+b[j]+c[j]; axis[i,inn]=-a[j]; axis[i,opp]=-c[j]
 L=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+reaction
 return np.array([splu(L+mu*ident).solve(R)[rows] for mu in s['mu_shifts']])
for name,a,b,c in [('robin',edge,robin,edge*0),('reflecting kappa',edge,edge*0,edge*0),('reflecting face',face,edge*0,edge*0),('doubled reflecting kappa',2*edge,edge*0,edge*0),('doubled reflecting face',2*face,edge*0,edge*0),('periodic kappa',edge,edge*0,edge),('periodic face',face,edge*0,np.repeat([edge[:2].mean(),edge[2:].mean()],2))]:
 d=predict(a,b,c)-Y; print(name, np.sqrt(np.mean(d*d,axis=1)).round(7).tolist())
for free in (8,12):
 for train in [np.ones(Y.shape,dtype=bool),np.indices(Y.shape)[2]==0,np.indices(Y.shape)[2]==1]:
  def fun(x): return predict(x[:4]*edge,x[4:8]*robin,x[8:]*edge if free==12 else edge*0)-Y
  x0=np.r_[np.ones(4),np.full(4,0.05),np.full(free-8,0.05)]; lo=np.r_[np.full(4,0.01),np.full(4,-0.18),np.full(free-8,-0.05)]; hi=np.full(free,8.)
  opt=least_squares(lambda x: fun(x)[train],x0,bounds=(lo,hi),gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=180)
  print('fit',free,train.sum(),opt.x.round(6),'rms by shift/pulse',np.sqrt(np.mean(fun(opt.x)**2,axis=1)).round(8).tolist())
print('edge/face/robin',edge,face,robin)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/boundary_probe.py
```
- **Container Exit Code**: `0` (`1856 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
robin [[0.2212845, 0.1536641], [0.0867129, 0.0678414], [0.0387311, 0.0344556]]
reflecting kappa [[9e-07, 7e-07], [6e-07, 5e-07], [5e-07, 5e-07]]
reflecting face [[0.0017007, 0.0011761], [0.0008371, 0.0005803], [0.0004732, 0.0003202]]
doubled reflecting kappa [[0.0460498, 0.036504], [0.0226517, 0.0190603], [0.0126086, 0.0116518]]
doubled reflecting face [[0.0471659, 0.0372935], [0.0232217, 0.0194508], [0.0129415, 0.0118593]]
periodic kappa [[0.0451432, 0.0424119], [0.0294095, 0.0313666], [0.0199204, 0.0223552]]
periodic face [[0.0520678, 0.049888], [0.0295946, 0.0310174], [0.0189958, 0.0208336]]
fit 8 192 [ 9.999960e-01  9.999830e-01  1.000009e+00  9.999970e-01  0.000000e+00
 -1.000000e-06 -0.000000e+00 -1.000000e-06] rms by shift/pulse [[7.9e-07, 6.3e-07], [5.7e-07, 5.1e-07], [4.9e-07, 4.6e-07]]
fit 8 96 [ 1.000007e+00  1.000063e+00  1.000009e+00  9.999470e-01  2.000000e-06
  8.000000e-06 -2.000000e-06 -2.000000e-06] rms by shift/pulse [[7.2e-07, 1.02e-06], [5.3e-07, 7.8e-07], [4.9e-07, 6.6e-07]]
fit 8 96 [ 1.000002e+00  9.999480e-01  9.999990e-01  1.000067e+00  1.000000e-06
 -5.000000e-06 -1.000000e-06  5.000000e-06] rms by shift/pulse [[2.48e-06, 5.8e-07], [7.9e-07, 4.8e-07], [5.5e-07, 4.5e-07]]
fit 12 192 [ 1.000007e+00  1.000007e+00  9.999970e-01  9.999750e-01  2.000000e-06
  2.000000e-06 -2.000000e-06 -2.000000e-06 -0.000000e+00 -0.000000e+00
  1.000000e-06  1.000000e-06] rms by shift/pulse [[7.6e-07, 6.4e-07], [5.5e-07, 4.9e-07], [4.9e-07, 4.5e-07]]
fit 12 96 [ 9.999850e-01  1.000245e+00  9.998950e-01  9.998390e-01  1.000000e-06
 -1.000000e-05  2.000000e-06  1.300000e-05 -3.000000e-06 -2.300000e-05
  8.000000e-06 -2.900000e-05] rms by shift/pulse [[6.3e-07, 1.739e-05], [4.6e-07, 6.6e-06], [4.2e-07, 3.22e-06]]
fit 12 96 [ 1.000052e+00  9.999360e-01  9.999890e-01  9.982920e-01 -5.000000e-06
  3.000000e-06  2.000000e-06 -5.000000e-06 -1.000000e-05 -1.000000e-05
 -2.000000e-06  5.900000e-05] rms by shift/pulse [[4.517e-05, 5e-07], [2.222e-05, 4e-07], [1.262e-05, 3.8e-07]]
edge/face/robin [13.21323267 13.21323267 17.31617966 10.05617966] [13.70474383 13.70474383 17.2426594  10.12969991] [4.84388241 4.84388241 6.40125719 6.40125719]
```

### Turn 5 — `execute_in_remote_docker` (`38811 ms` LLM)
- **Rationale**: Implement the reflecting stencil that matches both tracer pulses and retain dynamic calibration and the specified inverse objective. An independent dense solve checks wall indexing and terminal-field ordering; a separate optimizer and KKT checks test the weights and certificate.
- **Alternatives Weighed**: `Fit independent coefficients for each wall` (rejected: The exact reflecting stencil predicts both pulses better than coefficients fitted on one pulse and evaluated on the other.)

```bash
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py').read_text()
src = src.replace('with calibrated observations', 'with reflecting walls, calibrated observations').replace('def _axis(kappa, velocity, h, robin):', 'def _axis(kappa, velocity, h):\n    """Row-local upwind interior and zero-flux wall rows."""').replace('edge[0] + robin / h, left + right,\n                                       edge[1] + robin / h', 'edge[0], left + right, edge[1]').replace(", study['robin_x'])", ')').replace(", study['robin_y'])", ')')
Path('/holdout_task/batch_solver.py').write_text(src)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
from scipy.optimize import minimize
m = runpy.run_path('/holdout_task/batch_solver.py')
i = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = i['study']; out = m['solve_batch'](s)
nx, ny = s['nx'], s['ny']; I = np.eye(nx*ny)
def axis(k, v, h):
    A = np.zeros((len(k),len(k)))
    for j in range(len(k)):
        for side in (-1,1):
            q = j+side
            if 0 <= q < len(k):
                a = k[j]/h**2 if j in (0,len(k)-1) else (k[j]+k[q])/(2*h**2)+max(-side*v[j],0)/h
                A[j,q] = -a
                A[j,j] += a
    return A
ax, ay = (axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1)) for d in 'xy')
L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
rows = s['sensor_ij'] @ [1,nx]
R = np.column_stack([a.ravel(order='F') for a in s['source_modes']])
fields = [np.linalg.solve(L+mu*I,R) for mu in s['mu_shifts']]
S = np.concatenate([F[rows] for F in fields])
worst = 0.
for p in i['campaign_diagnostics']:
    obs = m['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw']))
    errors = (S@p['w_pulse']-obs).reshape(len(fields),-1)
    worst = max(worst,np.sqrt(np.mean(errors**2,axis=1)).max())
assert worst < 2e-6, worst
z = s['Z_nuisance']; P = np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
D,y = P@S,P@m['_calibrate'](s)
w = out['weights']; c = out['kkt_certificate']; ridge = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']
def J(w):
    r=D@w-y
    return .5*(r@r+ridge*(w@w))
def grad(w): return D.T@(D@w-y)+ridge*w
ref = minimize(J,np.full(len(w),budget/len(w)),jac=grad,bounds=[(0,cap)]*len(w),constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(len(w))}],method='SLSQP',options={'ftol':1e-14})
assert ref.success and np.max(abs(ref.x-w)) < 1e-6
assert np.max(abs(np.asarray(c['weights'])-w)) < 1e-12
assert abs(J(w)-c['primal_objective']) < 1e-12
lower,upper = np.asarray(c['lam_lower']),np.asarray(c['lam_upper'])
assert np.max(abs(grad(w)-c['nu_sum']-lower+upper)) < 1e-12
assert max(np.max(abs(w*lower)),np.max(abs((cap-w)*upper))) < 1e-12
assert abs(c['primal_objective']-c['dual_lower_bound']) < 1e-12
terminal_error = np.max(abs(out['terminal_field']-(fields[-1]@w).reshape(nx,ny,order='F')))
assert terminal_error < 1e-12
print(f'worst pulse/shift RMSE={worst:.3g}; max terminal error={terminal_error:.3g}; independent QP weight error={np.max(abs(ref.x-w)):.3g}; duality gap={c["primal_objective"]-c["dual_lower_bound"]:.3g}')
PY
```
- **Container Exit Code**: `0` (`607 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
worst pulse/shift RMSE=8.52e-07; max terminal error=1.11e-16; independent QP weight error=7.45e-09; duality gap=5.55e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d95f09ace061155 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d95f09a (directory).
```

### Turn 6 — `submit_completed_task` (`12770 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, matched the independent transport and optimization checks, and received host verification for its field, weights, and KKT certificate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5296'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The solver uses reflecting walls and fits sensor calibration from each study’s sweeps. Public schema validation passed, and the worst tracer pulse/shift RMSE was 8.52e-7.`

### Turn 7 — `execute_in_remote_docker` (`28576 ms` LLM)
- **Rationale**: Check whether pooled tracer error hides a weak sensor bank and whether nuisance projection masks it. Compare full and diagonal affine calibration on held-out rosettes and independent bank sweeps; change the candidate only if the comparison exposes a meaningful calibration or transport deficit.
- **Alternatives Weighed**: `Add drift, relay, or classification models` (rejected: The supplied data describe steady transport and continuous observations, with no relay state, time series, or class labels. Sensor bank and shift are the relevant real-data slices.)

```bash
sed -n '1,55p' /holdout_task/batch_solver.py; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py')
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; bank=c['single_bank_sweeps']; S,U=m['_forward'](s); z=s['Z_nuisance']; P=np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
 def fit(idx,kind):
  if kind=='full':
   b=np.linalg.lstsq(np.c_[np.ones(sum(idx)),x[idx]],y[idx],rcond=None)[0]; return b[0],b[1:]
  b=np.array([np.linalg.lstsq(np.c_[np.ones(sum(idx)),x[idx,j]],y[idx,j],rcond=None)[0] for j in range(4)]); return b[:,0],np.diag(b[:,1])
 def rmse(a,axis): return np.sqrt(np.mean(a*a,axis=axis))
 print('forward constant-mode balance: reaction range',s['V_reaction_2d'].min(),s['V_reaction_2d'].max())
 predictions={}
 for kind in ('full','diagonal'):
  scores=[]
  for fold in range(5):
   tr=np.arange(len(x))%5!=fold; a,B=fit(tr,kind); scores.extend(rmse(y[~tr]-a-x[~tr]@B,0))
  a,B=fit(np.ones(len(x),bool),kind); j=bank['bank_id']; r=bank['raw_readout']-a[j]-bank['ref_signal']*B[j,j]
  print(kind,'worst held-out calibration RMSE',max(scores),'worst independent single-bank RMSE',max(rmse(r[j==i],0) for i in range(4)))
  worst=projected=0
  for p in item['campaign_diagnostics'] or []:
   obs=(p['y_pulse_raw'].reshape(-1,4)-a)@np.linalg.inv(B); d=S@p['w_pulse']-obs.ravel(); shape=(len(s['mu_shifts']),-1,4)
   worst=max(worst,rmse(d.reshape(shape),1).max()); projected=max(projected,rmse((P@d).reshape(shape),1).max())
  print(kind,'worst pulse/shift/bank raw-calibrated RMSE',worst,'after nuisance projection',projected)
  obs=((s['y_obs_raw'].reshape(-1,4)-a)@np.linalg.inv(B)).ravel(); A=P@S; b=P@obs; w,nu=m['_box_simplex_qp'](A.T@A+s['source_ridge']*np.eye(A.shape[1]),-A.T@b,s['u_box'],s['total_budget']); predictions[kind]=w
 print('weight max change across affine calibration forms',np.max(np.abs(predictions['full']-predictions['diagonal'])))
PY
```
- **Container Exit Code**: `0` (`610 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
csc',
    )


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.stack([mode.ravel(order='F') for mode in study['source_modes']], axis=1)
    ij = np.asarray(study['sensor_ij'])
    sensor_rows = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    observations = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        observations.append(fields[sensor_rows])
    terminal_modes = np.stack([field.reshape((nx, ny), order='F') for field in fields.T])
    return np.vstack(observations), terminal_modes


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    affine = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(affine[1:].T, (observed - affine[0]).T).T.ravel()


def _box_simplex_qp(hessian, linear, cap, budget):
    """Feasible active set; -1 and +1 mark lower and upper bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w.fill(cap)
forward constant-mode balance: reaction range 0.25 0.42738476356834465
full worst held-out calibration RMSE 1.2814459912883847e-05 worst independent single-bank RMSE 1.1831331438117373e-05
full worst pulse/shift/bank raw-calibrated RMSE 1.0711268160507524e-06 after nuisance projection 1.0198789243776439e-06
diagonal worst held-out calibration RMSE 1.2122628940663725e-05 worst independent single-bank RMSE 1.1775533909583145e-05
diagonal worst pulse/shift/bank raw-calibrated RMSE 1.1223124141026601e-06 after nuisance projection 1.0932592266087275e-06
weight max change across affine calibration forms 2.3832572472459734e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d95f09ace061155 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d95f09a (directory).
```

### Turn 8 — `submit_completed_task` (`14999 ms` LLM)
- **Rationale**: Tracer agreement held across all sensor banks before nuisance projection, so projection was not hiding a weak bank. The unchanged candidate passed isolated schema validation and independent checks of the terminal field, weights, and KKT certificate.
- **Alternatives Weighed**: `Restrict calibration to diagonal gains` (rejected: It slightly improved held-out sweep RMSE but worsened the worst tracer slice and changed weights by at most 2.38e-6; there was no material gain.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5296'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The solver uses reflecting walls, fits sensor calibration from each study’s sweeps, and returns source weights, the terminal field, and a KKT certificate. Public schema validation passed; the worst tracer RMSE across pulses, shifts, and sensor banks was 1.07e-6 before nuisance projection.`

### Turn 9 — `execute_in_remote_docker` (`42494 ms` LLM)
- **Rationale**: Test whether pairwise or rational calibration terms improve held-out sweeps, and whether the inverse solver or certificate fails when real observations are paired with budgets that force different active bounds. A material calibration gain would warrant tracer validation; a feasibility or certificate failure would warrant an optimizer fix.
- **Alternatives Weighed**: `Apply batch classification, relay-state, or Fokker–Planck/HJB tests` (rejected: This study provides continuous measurements from steady transport, with no class labels, relay state, or time-dependent data. The earlier sensor-bank audit already tested whether nuisance projection concealed a weak bank.)

```bash
cat > /tmp/m98_margin_audit.py <<'PY'
import pickle
import runpy
import numpy as np
from scipy.optimize import minimize

m = runpy.run_path('/holdout_task/batch_solver.py')
i = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = i['study']
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
base = np.c_[np.ones(len(x)), x]
pairs = np.column_stack([f for j in range(4) for k in range(j+1,4) for f in (x[:,j]*x[:,k], (x[:,j]-x[:,k])**2)])
for label, design in [('affine',base), ('pairwise',np.c_[base,pairs]), ('rational',np.c_[base,np.column_stack([x/(1+a*x*x) for a in (.1,1.)])])]:
    errs = []
    for fold in range(5):
        tr = np.arange(len(x)) % 5 != fold
        coef = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
        errs.append(np.mean((design[~tr]@coef-y[~tr])**2,axis=0)**.5)
    print(label, 'worst held-out bank/fold RMSE',np.max(errs), 'pooled RMSE',np.mean(np.asarray(errs)**2)**.5)

for d in 'xy':
    k = s['kappa_'+d]; h = s['L'+d]/(s['n'+d]-1)
    A = m['_axis'](k,s['vel_'+d],h)
    assert np.linalg.norm(A@np.ones(len(k)),np.inf) < 1e-12
    assert np.allclose(A.diagonal()[[0,-1]], k[[0,-1]]/h**2,rtol=1e-14,atol=0)
print('Reflecting row balance and endpoint-kappa wall coefficients verified')

S,U = m['_forward'](s)
Z = s['Z_nuisance']; P = np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
D = P@S; ridge = s['source_ridge']; n = D.shape[1]; cap = s['u_box']
cases = [('production', s['y_obs_raw'])] + [(f'tracer_{j}',p['y_pulse_raw']) for j,p in enumerate(i['campaign_diagnostics'] or [])]
max_gap=max_err=max_field=0.; patterns=set()
for label,obs in cases:
    for budget in (0., .01*cap, cap, s['total_budget'], (n-1)*cap, (n-.01)*cap, n*cap):
        t = dict(s,y_obs_raw=obs,total_budget=budget)
        b = P@m['_calibrate'](t)
        def J(w):
            r = D@w-b
            return .5*(r@r+ridge*(w@w))
        def grad(w): return D.T@(D@w-b)+ridge*w
        out = m['solve_batch'](t); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); g=grad(w)
        assert abs(w.sum()-budget) < 1e-11 and w.min() >= -1e-12 and w.max() <= cap+1e-12
        assert np.max(np.abs(g-nu-lo+hi)) < 1e-11
        assert max(np.max(np.abs(lo*w)),np.max(np.abs(hi*(cap-w)))) < 1e-11
        dual=J(w)-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
        assert max(abs(c['primal_objective']-J(w)),abs(c['dual_lower_bound']-dual)) < 1e-11
        gap=abs(J(w)-dual); assert gap < 1e-11
        max_gap=max(max_gap,gap)
        patterns.add((int(sum(w<1e-10)),int(sum(w>cap-1e-10))))
        field=np.einsum('i,ijk->jk',w,U); max_field=max(max_field,np.max(np.abs(field-out['terminal_field'])))
        if 0<budget<n*cap:
            ref=minimize(J,np.full(n,budget/n),jac=grad,bounds=[(0,cap)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(n)}],method='SLSQP',options={'ftol':1e-14,'maxiter':200})
            err=np.max(np.abs(ref.x-w)); assert ref.success and err < 1e-6,(label,budget,ref.message,err)
            max_err=max(max_err,err)
assert max_field < 1e-12
print(f'{7*len(cases)} real-observation budget cases, {len(patterns)} active-bound patterns; worst weight error {max_err:.3g}, dual gap {max_gap:.3g}, field assembly error {max_field:.3g}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_margin_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`794 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine worst held-out bank/fold RMSE 1.2814459912879136e-05 pooled RMSE 1.0188079207788443e-05
pairwise worst held-out bank/fold RMSE 2.0214795156125165e-05 pooled RMSE 1.229395989831855e-05
rational worst held-out bank/fold RMSE 1.4653626970822766e-05 pooled RMSE 1.1307026870586861e-05
Reflecting row balance and endpoint-kappa wall coefficients verified
21 real-observation budget cases, 11 active-bound patterns; worst weight error 2.88e-08, dual gap 2.22e-15, field assembly error 5.55e-17
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d95f09ace061155 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d95f09a (directory).
```

### Turn 10 — `submit_completed_task` (`16958 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, independent transport and optimization checks, and 21 budget stress cases covering 11 active-bound patterns. The host verification receipt remains bound to the submitted artifact.
- **Alternatives Weighed**: `Add pairwise or rational calibration terms` (rejected: Both increased worst held-out calibration RMSE relative to the affine fit.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It fits sensor calibration from each study’s sweeps and returns source weights, the terminal field, and a KKT certificate. Public schema validation passed. The worst tracer RMSE across pulses, shifts, and sensor banks was 1.07e-6; the largest duality gap across 21 budget stress cases was 2.22e-15.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
