# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m99_c03_batch_01__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `454,212` (`440,100` / `212,992` / `14,112`) |
| **Total Episode Wall Time (s)** | `418.63s` |

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

### Turn 1 — `execute_in_remote_docker` (`15999 ms` LLM)
- **Rationale**: Inspect the public schema, prior solver, and prefix apparatus log to distinguish a changed boundary operator from a calibration or optimization error. That distinction determines which part of the solver to revise.
- **Alternatives Weighed**: `Copy the prior solver unchanged` (rejected: The public gate rejected the current solve, and the chamber setting and calibration can change between campaigns.)

```bash
ls -lh /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json; python3 - <<'PY'
import json,pickle,pathlib,numpy as np
p=pathlib.Path('/tmp/hegel_continual_archive/archive_apparatus_log.json'); d=json.loads(p.read_text()); print('APPARATUS',str(d)[-13000:]); x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; print('ITEM',x.keys(),type(x.get('campaign_diagnostics'))); 
def show(x,indent=''):
 for k,v in x.items():
  if isinstance(v,dict): print(indent,k);show(v,indent+'  ')
  elif hasattr(v,'shape'): print(indent,k,v.shape,v.dtype,np.asarray(v).ravel()[:4])
  else: print(indent,k,type(v).__name__,str(v)[:240])
show(x['study'])
PY
sed -n '1,260p' /tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
polished = np.where(hi, cap, 0.0)
        count = int(free.sum())
        a = np.ones(count)
        kkt = np.block([[H[np.ix_(free, free)] / scale, -a[:, None]],
                        [a[None, :], np.zeros((1, 1))]])
        rhs = np.r_[-(H @ polished + c)[free] / scale, budget - polished.sum()]
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        polished[free] = solution[:-1]
        nu = solution[-1] * scale
        reduced = H @ polished + c - nu
        if (np.all((polished >= 0) & (polished <= cap))
                and np.all(reduced[lo] >= -1e-11 * scale)
                and np.all(reduced[hi] <= 1e-11 * scale)):
            return polished, float(nu)
    g = H @ w + c
    if np.any(free):
        nu = np.mean(g[free])
    else:
        nu = 0.5 * (g[hi].max() + g[lo].min())
    return w, float(nu)


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrated_observations(study)
    Z = study['Z_nuisance']
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = np.column_stack((S, y))
    projected -= Z @ np.linalg.solve(gram, Z.T @ projected)
    design, target = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(design.T @ target), cap, budget)
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    reduced = g - nu
    lower = np.maximum(reduced, 0)
    upper = np.maximum(-reduced, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * upper.sum())
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`17239 ms` LLM)
- **Rationale**: Read the missing forward and calibration code and distinguish actual tracer pulses from index metadata. If pulses are absent, the apparatus log selects the boundary candidates; if they are present, held-out pulse residuals can test those candidates directly.
- **Alternatives Weighed**: `Implement the open-wall baseline immediately` (rejected: The first output was truncated before the actual diagnostics and apparatus settings, so the prior gate failure cannot yet be localized.)

```bash
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
d=json.load(open(p)); print('LOG TYPE',type(d).__name__); print(json.dumps(d,separators=(',',':'))[-8500:]); x=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('ITEM COUNT',len(x));
for i,item in enumerate(x):
 s=item['study']; diag=item.get('campaign_diagnostics'); print('ITEM',i,'diagnostics',None if diag is None else len(diag),'study keys',list(s)); print('scalars',{k:v for k,v in s.items() if np.isscalar(v)}); c=s['sensor_calibration_sweeps']; print('calibration',[(k,getattr(v,'shape',None),type(v).__name__) for k,v in c.items()]); print('single banks',str(c.get('single_bank_sweeps'))[:350]);
PY
sed -n '1,72p' /tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py
cat /holdout_task/validate_public_schema.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -25
```
- **Container Exit Code**: `0` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le == 0:
        return np.full(n, budget / n), 0.0
    result = minimize(lambda w: ((0.5 * w @ H @ w + c @ w) / scale,
                                 (H @ w + c) / scale),
                      np.full(n, budget / n), jac=True, method='SLSQP',
                      bounds=[(0, cap)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                    'jac': lambda w: ones}],
                      options={'ftol': 1e-14, 'maxiter': 500})
    w = result.x
    tol = 1e-9 * cap
    lo, hi = w <= tol, w >= cap - tol
    free = ~(lo | hi)
    if np.any(free):
        # Remove the numerical optimization tolerance on the identified face.
import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
m_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (m_m,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not isinstance(out.get("kkt_certificate"), dict): errs.append("missing_or_bad:kkt_certificate")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
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
```

### Turn 3 — `execute_in_remote_docker` (`24706 ms` LLM)
- **Rationale**: Check whether the archived operator omits a boundary term and whether the calibration is organized by bank. The actual pulse availability determines whether to test boundary candidates on held-out tracers or infer the setting from the prefix log.
- **Alternatives Weighed**: `Dump the full archive again` (rejected: The previous output truncated the boundary code and study summary; compact telemetry statistics and targeted source lines retain the evidence needed for the next fit.)

```bash
sed -n '1,57p' /tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for a in json.load(open(p)):
 print('LOG', {k:v for k,v in a.items() if k!='facility_telemetry_trace'}, 'median',np.round(np.median(a['facility_telemetry_trace'],axis=0),4))
x=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(x))
for a in x:
 s=a['study']; d=a.get('campaign_diagnostics'); print('diagnostics',None if d is None else len(d)); print('keys',list(s)); print('scalars',{k:v for k,v in s.items() if np.isscalar(v)}); c=s['sensor_calibration_sweeps']; print('sweep layout',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in c.items()]); print('bank preview',repr(c.get('single_bank_sweeps'))[:420]); print('aux median',np.median(s['aux_manifold_telemetry'],axis=0)); print('banks',np.unique(s['row_bank_id'],return_counts=True))
PY
```
- **Container Exit Code**: `0` (`347 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
i, j = study['sensor_ij'].T
    sample = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sample])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _calibrated_observations(study):
    sweep = study['sensor_calibration_sweeps']
    ref = sweep['ref_rosette']
    offset, *mixing = np.linalg.lstsq(np.c_[np.ones(len(ref)), ref],
                                      sweep['raw_rosette'], rcond=None)[0]
    raw = study['y_obs_raw'].reshape(-1, 4)
    return np.linalg.solve(np.asarray(mixing).T, (raw - offset).T).T.ravel()


def _box_simplex(H, c, cap, budget):
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(c.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((H @ w + c).max())
    ones = np.ones(n)
    scale = max(np.linalg.norm(H, ord=np.inf), np.linalg.norm(c, ord=np.inf))
LOG {'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'} median [-0.3753 -0.0161 -0.028   0.0396  0.0229 -0.4096  0.0049 -0.0116]
LOG {'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m99_c03_s00'} median [ 2.4299  0.0187 -0.0401  0.015  -0.0379 -0.3836  0.0407 -0.0055]
items 1
diagnostics 3
keys ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
scalars {'Lx': 2.4, 'Ly': 2.0, 'nx': 14, 'ny': 12, 'robin_x': 1.0240161106792895, 'robin_y': 1.1758119992635399, 'nuisance_ridge': 0.08, 'source_ridge': 0.04, 'u_box': 0.42, 'total_budget': 1.4}
sweep layout [('ref_rosette', 'ndarray', (80, 4)), ('raw_rosette', 'ndarray', (80, 4)), ('single_bank_sweeps', 'dict', None)]
bank preview {'bank_id': array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
       3, 3]), 'ref_signal': array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,  0.9,
        1.4, -1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,
        0.9,  1.4]), 'raw_readout': array([-1.33699303, -0.70199401, -0.06700262,  0.56798762,  1.20300451,
        1.83800794, -0.96300307, -0.54799776, -0.132
aux median [-0.00463159 -0.06102515 -0.00285803  0.07125616]
banks (array([0, 1, 2, 3]), array([24, 24, 24, 24]))
```

### Turn 4 — `execute_in_remote_docker` (`27085 ms` LLM)
- **Rationale**: Calibration cross-validation tests whether a nonlinear sensor inverse is needed. Baseline tracer residuals before and after nuisance removal test whether the remaining discrepancy calls for a changed wall operator.
- **Alternatives Weighed**: `Fit wall and sensor parameters jointly to the main observation` (rejected: Known-input calibration sweeps and tracer pulses can identify these components separately without confounding them with unknown source weights.)

```bash
cat > /tmp/probe_m99.py <<'PY'
import importlib.util,pickle,numpy as np
np.set_printoptions(precision=6,suppress=True)
p='/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py'
sp=importlib.util.spec_from_file_location('prior',p); prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; C=np.c_[np.ones(len(ref)),ref]; coef=np.linalg.lstsq(C,raw,rcond=None)[0]; dr=raw-C@coef
print('affine coef',coef,'cal rms',np.sqrt(np.mean(dr**2,axis=0)), 'max',np.max(abs(dr),axis=0))
features=[ref[:,i]*ref[:,j] for i in range(4) for j in range(i,4)]; C2=np.column_stack([C]+features); print('quadratic coefficients',np.linalg.lstsq(C2,raw,rcond=None)[0][5:].T); print('LOO rms linear/quadratic',*[np.sqrt(np.mean(((raw-X@np.linalg.lstsq(X,raw,rcond=None)[0])/(1-np.sum(np.linalg.qr(X)[0]**2,axis=1))[:,None])**2,axis=0)) for X in [C,C2]])
bank=cal['single_bank_sweeps']; print('single-bank keys',list(bank));
for i in range(4):
 ix=bank['bank_id']==i; print('bank',i,np.polynomial.polynomial.polyfit(bank['ref_signal'][ix],bank['raw_readout'][ix],2))
print('row ordering',s['row_bank_id'][:32], 'sensor_ij',s['sensor_ij']); S,U=prior._forward(s); W=np.column_stack([p['w_pulse'] for p in pulses]); R=np.column_stack([p['y_pulse_raw'] for p in pulses]); Y=np.column_stack([prior._calibrated_observations({**s,'y_obs_raw':r}) for r in R.T]); E=Y-S@W
print('pulses',W.T); print('pulse RMS',np.sqrt(np.mean(E**2,axis=0)),'raw calibrated ranges',Y.min(),Y.max(),'forward ranges', (S@W).min(),(S@W).max());print('error by row',E[:32]); print('pulse affinity with nuisance',np.sqrt(np.mean((E-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],E,rcond=None)[0])**2,axis=0)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_m99.py
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
09 0.000008 0.000009 0.000009] max [0.00002  0.000021 0.000021 0.000027]
quadratic coefficients [[ 0.000002 -0.000001 -0.000001 -0.000002 -0.        0.000004  0.
  -0.000002  0.000001  0.000001]
 [ 0.000001 -0.       -0.000001 -0.000001  0.000003 -0.000001  0.000002
   0.000001 -0.000001  0.000002]
 [ 0.000001 -0.000001 -0.000002 -0.000002 -0.000001 -0.000002  0.000002
   0.       -0.000002 -0.000003]
 [ 0.000001  0.000002  0.000003  0.000001 -0.       -0.000002 -0.000002
   0.000002 -0.000003  0.      ]]
LOO rms linear/quadratic [0.00001  0.000009 0.00001  0.00001 ] [0.000011 0.00001  0.000011 0.00001 ]
single-bank keys ['bank_id', 'ref_signal', 'raw_readout']
bank 0 [0.059996 1.269997 0.000009]
bank 1 [-0.049996  0.829997 -0.000009]
bank 2 [-0.070009  0.780004  0.000006]
bank 3 [0.079995 1.309996 0.000007]
row ordering [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3] sensor_ij [[12 10]
 [ 6  6]
 [ 6  0]
 [ 6  4]
 [ 2 10]
 [ 0  0]
 [ 8  2]
 [ 0  8]
 [ 0  4]
 [ 4  4]
 [12  2]
 [ 6  8]
 [10  4]
 [ 8 10]
 [ 8  6]
 [ 2  4]
 [ 8  8]
 [12  4]
 [ 2  2]
 [ 2  0]
 [ 8  0]
 [ 8  4]
 [12  8]
 [10  8]
 [ 2  6]
 [12  6]
 [10  2]
 [ 6  2]
 [ 4  6]
 [ 4  8]
 [ 4  2]
 [ 0  6]]
pulses [[1.  0.  0.  0.5 0.  0.  0.  0. ]
 [0.  1.  0.  0.  0.5 0.  0.  0. ]
 [0.  0.  1.  0.  0.  0.5 0.  0. ]]
pulse RMS [0.138217 0.120817 0.093447] raw calibrated ranges 0.015542723452265237 0.46861385441018577 forward ranges 0.005511559349495761 0.2386922390212155
error by row [[0.226823 0.30244  0.123226]
 [0.196572 0.166393 0.107988]
 [0.198122 0.18192  0.202361]
 [0.182593 0.15617  0.125916]
 [0.30048  0.195178 0.096398]
 [0.259741 0.183612 0.155895]
 [0.169264 0.165885 0.174817]
 [0.308298 0.185712 0.104811]
 [0.279361 0.177921 0.130923]
 [0.203848 0.155212 0.119297]
 [0.167177 0.194676 0.232963]
 [0.225307 0.191218 0.099606]
 [0.166566 0.18076  0.161852]
 [0.25632  0.261104 0.107398]
 [0.183757 0.178373 0.117884]
 [0.234897 0.161865 0.120362]
 [0.211002 0.208341 0.106679]
 [0.172553 0.205702 0.195684]
 [0.229123 0.16529  0.137427]
 [0.239918 0.178877 0.158411]
 [0.182387 0.184869 0.231554]
 [0.170171 0.164534 0.139805]
 [0.204162 0.262279 0.134764]
 [0.202378 0.230662 0.118165]
 [0.249278 0.166059 0.106913]
 [0.1855   0.229576 0.161251]
 [0.164126 0.177358 0.199605]
 [0.18229  0.160243 0.155321]
 [0.217641 0.162038 0.104309]
 [0.243773 0.179981 0.096668]
 [0.202602 0.15999  0.142129]
 [0.298336 0.180697 0.117038]]
pulse affinity with nuisance [0.137495 0.120061 0.093165]
```

### Turn 5 — `execute_in_remote_docker` (`38486 ms` LLM)
- **Rationale**: Compare canonical boundary conditions first, then fit a boundary-row basis that nests them. If a simpler model predicts all held-out pulses at the calibration floor, use it; if none does, inspect the residual by shift before changing the interior operator.
- **Alternatives Weighed**: `Use a flexible correction fitted directly to the observed signals` (rejected: It would not specify the terminal field or establish the chamber operator on a held-out tracer.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived open-wall Robin operator predicts the current chamber's tracer response.` | Witness: `Calibrated pulse RMS errors are 0.13822, 0.12082, and 0.09345; nuisance projection barely reduces them. Affine calibration already predicts held-out sweeps to about 1e-5, and quadratic terms do not improve it.` | Blanket Diagnosis: `Known-input tracer errors persist after separating sensor calibration and the measured nuisance span, localizing the failure to the forward model.` | Sublated Pivot: `Retain the interior transport and affine sensor model; test boundary leakage, inward coupling, and opposite-wall coupling on held-out pulses.`

```bash
cat > /tmp/walls_m99.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; d=a['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; N=nx*ny
w=np.column_stack([p['w_pulse'] for p in d]); y=np.column_stack([m._calibrated_observations({**s,'y_obs_raw':p['y_pulse_raw']}) for p in d]); rhs=np.column_stack([q.ravel(order='F') for q in s['source_modes']])@w
i,j=s['sensor_ij'].T; pick=i+nx*j; I=sparse.eye(N,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
axes=[]
for axis in 'xy':
 k,v,r,L=[s[p+axis] for p in ('kappa_','vel_','robin_','L')]; h=L/(len(k)-1); A=m._axis_operator(k,v,r,L).tolil(); axes.append((A,k,h,r))
def pred(p):
 mats=[]
 for ia,(base,k,h,r) in enumerate(axes):
  A=base.copy()
  for side,(i,j,opp) in enumerate([(0,1,-1),(-1,-2,0)]):
   diff,leak,wrap=np.asarray(p).reshape(2,2,3)[ia,side]; v=k[i]/h**2
   A[i,i]=v*(diff+wrap)+r/h*leak; A[i,j]=-v*diff; A[i,opp]=-v*wrap
  mats.append(A.tocsc())
 A=sparse.kron(sparse.eye(ny),mats[0],format='csc')+sparse.kron(mats[1],sparse.eye(nx),format='csc')+V
 return np.vstack([splu(A+shift*I).solve(rhs)[pick] for shift in s['mu_shifts']])
def rms(e):return np.sqrt(np.mean(e**2,axis=0))
def report(name,p):
 e=pred(p)-y; print(name,'params',np.round(np.array(p).reshape(4,3),6).tolist(),'rms',np.round(rms(e),8),'shift rms',np.round(rms(e.reshape(3,-1).T),8),flush=True)
for name,p in [('robin',[1,1,0]*4),('reflect',[1,0,0]*4),('mirror',[2,0,0]*4),('periodic',[1,0,1]*4),('clamped-decoupled',[0,0,0]*4)]: report(name,p)
models={
 'leak-only':(np.tile([1.,0.,0.],4),[1,4,7,10]),
 'reflect-diff':(np.tile([1.,0.,0.],4),[0,3,6,9]),
 'diff-leak':(np.tile([1.,0.,0.],4),[0,1,3,4,6,7,9,10]),
 'diff-leak-wrap':(np.tile([1.,0.,0.],4),list(range(12))),
}
def fit(base,ind,cols,profile=False):
 p=base.copy()
 def fun(z):
  p[ind]=z; e=(pred(p)-y)[:,cols]; e=e-Q@(Q.T@e) if profile else e
  return e.ravel()
 opt=least_squares(fun,p[ind],max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
 p[ind]=opt.x
 return p
for name,(base,ind) in models.items():
 p=fit(base,ind,[0,1,2]); report(name,p); print('leave-one-pulse-out rms', [round(float(rms((pred(fit(base,ind,[i for i in range(3) if i!=j]))-y)[:,j])),8) for j in range(3)])
 if name=='diff-leak-wrap': report('profiled all',fit(base,ind,[0,1,2],True))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/walls_m99.py
```
- **Container Exit Code**: `0` (`1330 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
robin params [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]] rms [0.1382168  0.12081736 0.09344732] shift rms [0.18832223 0.07578554 0.03499155]
reflect params [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]] rms [1.15e-06 1.11e-06 1.08e-06] shift rms [1.25e-06 1.07e-06 1.00e-06]
mirror params [[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]] rms [0.03195797 0.02887035 0.02154832] shift rms [0.04199725 0.02054456 0.01154516]
periodic params [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]] rms [0.03301801 0.02783103 0.03496082] shift rms [0.04348598 0.02872603 0.01925613]
clamped-decoupled params [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] rms [0.15445527 0.13841641 0.10684491] shift rms [0.20913894 0.09214827 0.0469145 ]
leak-only params [[1.0, -1e-06, 0.0], [1.0, 1e-06, 0.0], [1.0, 3e-06, 0.0], [1.0, 0.0, 0.0]] rms [9.8e-07 9.3e-07 8.9e-07] shift rms [9.3e-07 9.2e-07 9.4e-07]
leave-one-pulse-out rms [1.02e-06, 9.3e-07, 8.9e-07]
reflect-diff params [[0.999999, 0.0, 0.0], [0.999977, 0.0, 0.0], [0.999962, 0.0, 0.0], [1.000003, 0.0, 0.0]] rms [9.4e-07 8.6e-07 8.3e-07] shift rms [9.2e-07 8.5e-07 8.6e-07]
leave-one-pulse-out rms [9.6e-07, 8.6e-07, 8.3e-07]
diff-leak params [[0.99998, -1e-06, 0.0], [0.999973, -1e-06, 0.0], [0.999943, -1e-06, 0.0], [0.999992, -0.0, 0.0]] rms [9.1e-07 8.4e-07 8.2e-07] shift rms [9.1e-07 8.4e-07 8.3e-07]
leave-one-pulse-out rms [9.2e-07, 8.5e-07, 8.3e-07]
diff-leak-wrap params [[0.999967, -3e-06, 0.0], [0.999959, -2e-06, -0.0], [0.999952, -0.0, -0.0], [1.000006, 1e-06, -1e-06]] rms [9.0e-07 8.3e-07 8.2e-07] shift rms [9.0e-07 8.3e-07 8.2e-07]
leave-one-pulse-out rms [1.66e-06, 8.5e-07, 3.33e-06]
profiled all params [[0.999967, -3e-06, 0.0], [0.999958, -2e-06, -0.0], [0.99995, -1e-06, -0.0], [1.000009, 1e-06, -1e-06]] rms [9.0e-07 8.3e-07 8.2e-07] shift rms [9.0e-07 8.3e-07 8.2e-07]
```

### Turn 6 — `execute_in_remote_docker` (`20907 ms` LLM)
- **Rationale**: Remove only the Robin leakage term: the resulting exact stencil predicts every public tracer near the calibration error floor. The isolated schema run checks the installed artifact for integration failures; it does not yet verify the terminal field or KKT certificate.
- **Alternatives Weighed**: `Fit separate diffusion, leakage, and wraparound parameters at each wall` (rejected: The fitted values are consistent with exact reflecting rows, and the full model worsens worst held-out pulse RMS to 3.33e-6.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py')
s = p.read_text()
s = s.replace('"""Forward transport and box-simplex source inversion."""', '"""Reflecting-wall chamber transport and box-simplex source inversion."""')
s = s.replace('def _axis_operator(kappa, velocity, robin, length):', 'def _axis_operator(kappa, velocity, length):')
s = s.replace('    diag = np.r_[kappa[0] / h**2 + robin / h,\n', '    # Reflecting walls retain inward diffusion but have no Robin leakage.\n    diag = np.r_[kappa[0] / h**2,\n')
s = s.replace('                 kappa[-1] / h**2 + robin / h]', '                 kappa[-1] / h**2]')
s = s.replace("                             study['robin_' + a], study['L' + a]) for a in 'xy']", "                             study['L' + a]) for a in 'xy']")
assert 'robin' not in s
Path('/holdout_task/batch_solver.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
sed -n '50,180p' /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`539 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
* cap
    lo, hi = w <= tol, w >= cap - tol
    free = ~(lo | hi)
    if np.any(free):
        # Remove the numerical optimization tolerance on the identified face.
        polished = np.where(hi, cap, 0.0)
        count = int(free.sum())
        a = np.ones(count)
        kkt = np.block([[H[np.ix_(free, free)] / scale, -a[:, None]],
                        [a[None, :], np.zeros((1, 1))]])
        rhs = np.r_[-(H @ polished + c)[free] / scale, budget - polished.sum()]
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        polished[free] = solution[:-1]
        nu = solution[-1] * scale
        reduced = H @ polished + c - nu
        if (np.all((polished >= 0) & (polished <= cap))
                and np.all(reduced[lo] >= -1e-11 * scale)
                and np.all(reduced[hi] <= 1e-11 * scale)):
            return polished, float(nu)
    g = H @ w + c
    if np.any(free):
        nu = np.mean(g[free])
    else:
        nu = 0.5 * (g[hi].max() + g[lo].min())
    return w, float(nu)


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrated_observations(study)
    Z = study['Z_nuisance']
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = np.column_stack((S, y))
    projected -= Z @ np.linalg.solve(gram, Z.T @ projected)
    design, target = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(design.T @ target), cap, budget)
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    reduced = g - nu
    lower = np.maximum(reduced, 0)
    upper = np.maximum(-reduced, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * upper.sum())
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a144c92e460a3ddb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a144c92e (directory).
```

### Turn 7 — `execute_in_remote_docker` (`37943 ms` LLM)
- **Rationale**: The schema and host probe passed. Independently reconstruct the dense forward problem and projected objective to detect a terminal layout error or certificate inconsistency, and exercise saturated budgets to detect an active-bound optimizer failure. A failure would identify which component to revise.
- **Alternatives Weighed**: `Retune the boundary from the host-reported gap` (rejected: Exact reflecting rows already explain the measured tracers at the calibration floor; first determine whether the gap appears under the independently reconstructed objective.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle
import numpy as np
p = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(p)
p.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
x = m.solve_batch(s)
def axis(k, v, L):
    n = len(k); h = L/(n-1); A = np.zeros((n,n))
    for i in range(1,n-1):
        A[i,i-1] = -(k[i]+k[i-1])/(2*h*h)-max(v[i],0)/h
        A[i,i+1] = -(k[i]+k[i+1])/(2*h*h)-max(-v[i],0)/h
        A[i,i] = -A[i,i-1]-A[i,i+1]
    for i,j in ((0,1),(n-1,n-2)):
        A[i,i] = k[i]/h**2; A[i,j] = -A[i,i]
    return A
nx,ny=s['nx'],s['ny']
Ax,Ay=[axis(s['kappa_'+a],s['vel_'+a],s['L'+a]) for a in 'xy']
A=np.kron(np.eye(ny),Ax)+np.kron(Ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
I=np.eye(len(A)); B=np.column_stack([q.ravel(order='F') for q in s['source_modes']]); i,j=s['sensor_ij'].T
fields=[np.linalg.solve(A+mu*I,B) for mu in s['mu_shifts']]; S=np.vstack([U[i+nx*j] for U in fields])
ref=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; C=np.linalg.lstsq(np.column_stack((np.ones(len(ref)),ref)),raw,rcond=None)[0]
def cal(raw): return ((raw.reshape(-1,4)-C[0])@np.linalg.inv(C[1:])).ravel()
E=np.stack([cal(d['y_pulse_raw'])-S@d['w_pulse'] for d in item['campaign_diagnostics']]).reshape(len(item['campaign_diagnostics']),len(fields),-1)
print('worst pulse-by-shift RMS',np.sqrt(np.mean(E**2,axis=2)).max())
Z=s['Z_nuisance']; P=np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
D,y=P@S,P@cal(s['y_obs_raw']); r=s['source_ridge']; H=D.T@D+r*np.eye(D.shape[1]); c=-D.T@y
cap,b=s['u_box'],s['total_budget']; w=np.asarray(x['weights']); cert=x['kkt_certificate']; nu=cert['nu_sum']; lo=np.array(cert['lam_lower']); hi=np.array(cert['lam_upper']); g=H@w+c
obj=(np.linalg.norm(D@w-y)**2+r*(w@w))/2
terminal=x['terminal_field'].ravel(order='F'); terminal_res=np.linalg.norm((A+s['mu_shifts'][-1]*I)@terminal-B@w)/np.linalg.norm(B@w)
print('public terminal equation relative residual',terminal_res,'KKT gap',obj-cert['dual_lower_bound'],'stationarity',np.max(np.abs(g-nu-lo+hi)))
assert terminal_res<1e-11 and np.max(np.abs(g-nu-lo+hi))<1e-10
assert np.array_equal(w,cert['weights']) and abs(w.sum()-b)<1e-10 and np.min(w)>=-1e-12 and np.max(w)<=cap+1e-12
assert np.min(lo)>=0 and np.min(hi)>=0 and max(abs(lo@w),abs(hi@(cap-w)))<1e-10
assert abs(cert['primal_objective']-obj)<1e-10 and abs(obj-cert['dual_lower_bound'])<1e-10
worst=0.
for budget in (0.,cap,2*cap,b,len(w)*cap):
    for linear in (c, c+np.linspace(-2,2,len(w)), c-np.linspace(-2,2,len(w))):
        q,v=m._box_simplex(H,linear,cap,budget); grad=H@q+linear
        gap=grad@q-v*budget-cap*np.minimum(grad-v,0).sum()
        worst=max(worst,abs(gap)); assert abs(gap)<1e-8 and abs(q.sum()-budget)<1e-10 and q.min()>=-1e-12 and q.max()<=cap+1e-12
print('worst gap over 15 budget/active-bound slices',worst)
PY
```
- **Container Exit Code**: `0` (`450 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
worst pulse-by-shift RMS 1.3137623648402021e-06
public terminal equation relative residual 4.076264784376566e-15 KKT gap 6.737665980693919e-15 stationarity 2.005340338229189e-15
worst gap over 15 budget/active-bound slices 4.3298697960381105e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a144c92e460a3ddb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a144c92e (directory).
```

### Turn 8 — `submit_completed_task` (`13257 ms` LLM)
- **Rationale**: The exact saved candidate passed the isolated schema check, the host probe, and an independent reconstruction of the terminal field and KKT certificate. Stress tests covered empty, saturated, and mixed-bound budgets without exposing an optimizer failure.
- **Alternatives Weighed**: `Fit separate coefficients for each wall` (rejected: The exact reflecting stencil matches the public tracers near the calibration error floor; the more flexible model worsened worst held-out pulse error.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5100'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The solver uses reflecting walls, fits sensor calibration from each study’s sweeps, and returns the source weights, terminal field, and KKT certificate. The public schema passed. The worst tracer RMS across pulses and shifts was 1.31e-6, and the largest absolute KKT gap across 15 budget and active-bound stress cases was 4.33e-15.`

### Turn 9 — `execute_in_remote_docker` (`41064 ms` LLM)
- **Rationale**: Use held-out calibration sweeps and independent tracer readings to test whether cross-channel, pairwise, three-way, or rational terms improve the worst measured bank and shift. If a challenger improves both, test it against the end-to-end field and KKT metrics; otherwise retain the verified solver.
- **Alternatives Weighed**: `Test classifier balance, relay states, or stochastic drift` (rejected: The chamber data have continuous signals and a steady-state transport equation, not classification labels or relay states. Bank-wise tracer errors and the projected source spectrum test the applicable nuisance and identifiability risks.)

```bash
cat > /tmp/audit_m99_calibration.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

p = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(p)
p.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

def fit(kind, x, y):
    if kind == 'bank-diagonal':
        coef = np.array([np.linalg.lstsq(np.c_[np.ones(len(x)), x[:, j]], y[:, j], rcond=None)[0] for j in range(4)])
        return lambda v: coef[:, 0] + v * coef[:, 1]
    if kind == 'bank-rational':
        params = []
        for j in range(4):
            v, z = x[:, j], y[:, j]
            b, a = np.linalg.lstsq(np.c_[np.ones(len(v)), v], z, rcond=None)[0]
            q = least_squares(lambda q: q[0] + q[1]*v/(1+q[2]*v*v)-z, [b,a,0.], bounds=([-np.inf,-np.inf,0.],[np.inf,np.inf,np.inf]), gtol=1e-13).x
            params.append(q)
        b, a, d = np.asarray(params).T
        return lambda v: b + a*v/(1+d*v*v)
    def features(v):
        terms = [np.ones(len(v)), *v.T]
        if kind == 'interactions':
            terms += [v[:,i]*v[:,j] for i in range(4) for j in range(i,4)]
            terms += [v[:,i]*v[:,j]*v[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]
        return np.array(terms).T
    coef = np.linalg.lstsq(features(x), y, rcond=None)[0]
    return lambda v: features(v) @ coef

for item in items:
    s = item['study']; cal = s['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    S, _ = m._forward(s)
    expected = np.stack([(S @ d['w_pulse']).reshape(-1,4) for d in item['campaign_diagnostics']])
    observed = np.stack([d['y_pulse_raw'].reshape(-1,4) for d in item['campaign_diagnostics']])
    folds = np.array_split(np.arange(len(x)), 8)
    for kind in ('bank-diagonal', 'affine', 'interactions', 'bank-rational'):
        err = np.empty_like(y)
        for hold in folds:
            keep = np.setdiff1d(np.arange(len(x)), hold)
            err[hold] = fit(kind, x[keep], y[keep])(x[hold])-y[hold]
        pred = fit(kind, x, y)(expected.reshape(-1,4)).reshape(expected.shape)
        rms = lambda e, axis: np.sqrt(np.mean(e**2,axis=axis))
        p = (pred-observed).reshape(len(expected),len(s['mu_shifts']),-1,4)
        print(kind, 'sweep worst fold/channel', format(max(rms(err[h],0).max() for h in folds),'.4g'), 'pulse worst pulse/shift/bank', format(rms(p,2).max(),'.4g'), 'pulse bank RMS', rms(p,(0,1,2)))
    banks = cal['single_bank_sweeps']; z = s['Z_nuisance']
    op = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    ep = np.stack([m._calibrated_observations({**s,'y_obs_raw':d['y_pulse_raw']})-S@d['w_pulse'] for d in item['campaign_diagnostics']])
    print('full affine calibrated pulse worst bank RMS before/after nuisance', *(np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)).max() for e in (ep,ep-(z@(op@ep.T)).T)))
    print('response/nuisance spectral fraction', np.linalg.norm(z@(op@S),2)/np.linalg.norm(S,2), 'projected design relative singular spectrum', (lambda t:t/t[0])(np.linalg.svd(S-z@(op@S),compute_uv=False)))
    for j in range(4):
        i = banks['bank_id']==j
        a = np.polyfit(banks['ref_signal'][i], banks['raw_readout'][i], 1)
        print('bank', j, 'independent sweep slope/intercept minus rosette', a-np.polyfit(x[:,j],y[:,j],1))
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_m99_calibration.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`783 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank-diagonal sweep worst fold/channel 1.381e-05 pulse worst pulse/shift/bank 1.559e-06 pulse bank RMS [9.66646125e-07 3.76245170e-07 1.47153166e-06 4.84033452e-07]
affine sweep worst fold/channel 1.472e-05 pulse worst pulse/shift/bank 1.88e-06 pulse bank RMS [9.76086442e-07 4.14446726e-07 1.54709885e-06 5.18727944e-07]
interactions sweep worst fold/channel 1.754e-05 pulse worst pulse/shift/bank 5.997e-06 pulse bank RMS [1.06988963e-06 5.93874928e-06 3.15568414e-06 9.95931252e-07]
bank-rational sweep worst fold/channel 1.379e-05 pulse worst pulse/shift/bank 2.961e-06 pulse bank RMS [2.38509173e-06 3.76245166e-07 1.78668785e-06 4.84033488e-07]
full affine calibrated pulse worst bank RMS before/after nuisance 1.9834577245908285e-06 1.978684359387432e-06
response/nuisance spectral fraction 0.08753005552223803 projected design relative singular spectrum [1.         0.30417197 0.25087412 0.12521708 0.080468   0.04094647
 0.03303272 0.01874902]
bank 0 independent sweep slope/intercept minus rosette [5.64280231e-07 6.70128883e-07]
bank 1 independent sweep slope/intercept minus rosette [-7.33143013e-06 -2.01564924e-06]
bank 2 independent sweep slope/intercept minus rosette [ 5.53457107e-06 -5.86422359e-06]
bank 3 independent sweep slope/intercept minus rosette [-4.87250064e-06  3.34038593e-07]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a144c92e460a3ddb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a144c92e (directory).
```

### Turn 10 — `execute_in_remote_docker` (`42814 ms` LLM)
- **Rationale**: Promote the simpler calibration only if its improvement holds after inversion and the independently reconstructed objective still validates the certificate. The reflecting stencil and optimizer are unchanged; the isolated schema check verifies the installed candidate.
- **Alternatives Weighed**: `Retain cross-channel terms or introduce nonlinear interactions` (rejected: Neither improved worst-bank tracer prediction, and the independent single-bank sweeps support the simpler gain-and-offset model.)

```bash
set -e
python3 - <<'PY'
import ast
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
node = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == '_calibrated_observations')
lines = s.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno] = ['''def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    ref_mean, raw_mean = ref.mean(axis=0), raw.mean(axis=0)
    centered = ref - ref_mean
    # This campaign's banks respond independently; fit their gains and offsets.
    gain = np.sum(centered * (raw - raw_mean), axis=0) / np.sum(centered**2, axis=0)
    measured = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return ((measured - raw_mean) / gain + ref_mean).ravel()
''']
Path('/tmp/m99_diagonal.py').write_text(''.join(lines))
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
old = load('old', '/holdout_task/batch_solver.py')
new = load('new', '/tmp/m99_diagonal.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']; S, U = old._forward(s)
    def worst(mod):
        e = np.stack([mod._calibrated_observations({**s, 'y_obs_raw':p['y_pulse_raw']})-S@p['w_pulse'] for p in item['campaign_diagnostics']])
        e = e.reshape(len(e), len(s['mu_shifts']), -1, 4)
        return np.sqrt(np.mean(e**2, axis=2)).max()
    print('Worst calibrated pulse/shift/bank RMS, full affine -> bank diagonal:', worst(old), worst(new))
    assert worst(new) < worst(old)
    a, b = old.solve_batch(s), new.solve_batch(s)
    for key in ('weights', 'terminal_field'):
        print(key, 'relative change', np.linalg.norm(b[key]-a[key])/np.linalg.norm(a[key]))
    cal = s['sensor_calibration_sweeps']; ref, raw = cal['ref_rosette'], cal['raw_rosette']
    gains, offsets = np.asarray([np.polyfit(ref[:,j], raw[:,j], 1) for j in range(4)]).T
    y = ((s['y_obs_raw'].reshape(-1,4)-offsets)/gains).ravel()
    Z = s['Z_nuisance']; P = np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    w = b['weights']; cap = s['u_box']; budget = s['total_budget']; cert = b['kkt_certificate']
    D = P@S; e = D@w-P@y; g = D.T@e+s['source_ridge']*w
    obj = (e@e+s['source_ridge']*(w@w))/2
    nu = cert['nu_sum']; lo = np.asarray(cert['lam_lower']); hi = np.asarray(cert['lam_upper'])
    stationarity = np.max(np.abs(g-nu-lo+hi)); gap = abs(obj-cert['dual_lower_bound'])
    print('Independent KKT stationarity and absolute gap:', stationarity, gap)
    assert max(stationarity, gap, abs(obj-cert['primal_objective']), abs(w.sum()-budget), abs(lo@w), abs(hi@(cap-w))) < 1e-10
    assert np.array_equal(w, cert['weights']) and np.min(w)>=-1e-12 and np.max(w)<=cap+1e-12 and min(lo.min(),hi.min())>=0
    np.testing.assert_allclose(b['terminal_field'], np.tensordot(w,U,axes=(0,0)), atol=1e-12, rtol=0)
PY
cp /tmp/m99_diagonal.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`813 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
Worst calibrated pulse/shift/bank RMS, full affine -> bank diagonal: 2.4108460722148644e-06 1.9981469861476835e-06
weights relative change 7.658012775081816e-07
terminal_field relative change 1.2779574360775433e-07
Independent KKT stationarity and absolute gap: 7.511352650979575e-16 7.520026268359459e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=830fa452b5ed21b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_830fa452 (directory).
```

### Turn 11 — `submit_completed_task` (`13882 ms` LLM)
- **Rationale**: The installed candidate passed the isolated schema check, and independent recomputation confirmed its terminal field and KKT certificate. The calibration change improved the measured bottleneck without changing the reflecting stencil or the optimizer already tested across 15 budget and active-bound cases.
- **Alternatives Weighed**: `Retain full affine calibration or add nonlinear terms` (rejected: The bank-diagonal fit improved both worst held-out sweep error and worst tracer error; interactions and rational terms worsened the latter.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5242'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It uses reflecting walls and fits each sensor bank’s gain and offset from the study’s calibration sweeps. The public schema passed. Worst calibrated tracer RMS across pulses, shifts, and banks was 2.00e-6, and the independently recomputed public KKT gap was 7.52e-15.`

### Turn 12 — `execute_in_remote_docker` (`36917 ms` LLM)
- **Rationale**: Pooling the independent bench sweeps could reduce calibration variance or introduce bias; the known-input tracers distinguish those outcomes. Holding out one measured rosette at a time and refitting the nuisance projection tests whether a single sensor group masks weak source identifiability or a certificate failure.
- **Alternatives Weighed**: `Repeat the nonlinear calibration and classifier audits` (rejected: Held-out sweeps and independent tracers already tested the nonlinear terms. The continuous chamber measurements have no classification labels or relay modes.)

```bash
set -e
cat > /tmp/m99_sensor_holdout.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    S, U = m._forward(s)
    sweeps = s['sensor_calibration_sweeps']
    bank = sweeps['single_bank_sweeps']
    xf, yf = sweeps['ref_rosette'], sweeps['raw_rosette']
    coef = []
    for j in range(4):
        take = bank['bank_id'] == j
        coef.append(np.polyfit(np.r_[xf[:, j], bank['ref_signal'][take]], np.r_[yf[:, j], bank['raw_readout'][take]], 1))
    gain, offset = np.array(coef).T
    observed = np.array([p['y_pulse_raw'] for p in item['campaign_diagnostics']])
    expected = np.array([S @ p['w_pulse'] for p in item['campaign_diagnostics']])
    active = np.array([m._calibrated_observations({**s, 'y_obs_raw': raw}) for raw in observed])
    pooled = (((observed.reshape(len(observed), -1, 4) - offset) / gain)).reshape(observed.shape)
    def worst(v):
        e = (v - expected).reshape(len(observed), len(s['mu_shifts']), -1, 4)
        return np.sqrt(np.mean(e**2, axis=2)).max()
    print('Worst tracer pulse/shift/bank RMS, rosette-only versus pooled bench sweeps:', worst(active), worst(pooled))

    y = m._calibrated_observations(s)
    n_sensors = len(s['sensor_ij'])
    assert n_sensors % 4 == 0
    base = m.solve_batch(s)['weights']
    gaps, residuals, weight_changes, design_ratios = [], [], [], []
    for start in range(0, n_sensors, 4):
        keep_sensor = np.ones(n_sensors, dtype=bool)
        keep_sensor[start:start+4] = False
        keep = np.tile(keep_sensor, len(s['mu_shifts']))
        Z = s['Z_nuisance'][keep]
        obs = np.column_stack((S[keep], y[keep]))
        obs -= Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ obs)
        D, r = obs[:, :-1], obs[:, -1]
        sing = np.linalg.svd(D, compute_uv=False)
        design_ratios.append(sing[-1]/sing[0])
        h = {**s, 'sensor_ij': s['sensor_ij'][keep_sensor], 'Z_nuisance': Z, 'y_obs_raw': s['y_obs_raw'][keep]}
        if 'row_bank_id' in s:
            h['row_bank_id'] = s['row_bank_id'][keep]
        out = m.solve_batch(h)
        w, cert = out['weights'], out['kkt_certificate']
        e = D @ w - r
        obj = (e @ e + s['source_ridge']*(w @ w))/2
        g = D.T @ e + s['source_ridge']*w
        err = g - cert['nu_sum'] - np.array(cert['lam_lower']) + np.array(cert['lam_upper'])
        gaps.append(abs(obj - cert['dual_lower_bound']))
        residuals.append(np.max(np.abs(err)))
        weight_changes.append(np.linalg.norm(w-base)/np.linalg.norm(base))
        assert abs(w.sum()-s['total_budget']) < 1e-10 and min(w) >= -1e-12 and max(w) <= s['u_box']+1e-12
        assert gaps[-1] < 1e-9 and residuals[-1] < 1e-9
    print('Held-out rosettes:', len(gaps), 'worst KKT gap:', max(gaps), 'worst stationarity residual:', max(residuals), 'minimum projected singular ratio:', min(design_ratios), 'maximum relative weight change:', max(weight_changes))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m99_sensor_holdout.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`711 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst tracer pulse/shift/bank RMS, rosette-only versus pooled bench sweeps: 1.9981469861476835e-06 1.6181834216987962e-06
Held-out rosettes: 8 worst KKT gap: 1.2415415917566008e-14 worst stationarity residual: 0.0 minimum projected singular ratio: 0.014760429305380572 maximum relative weight change: 0.027694507988986983
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=830fa452b5ed21b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_830fa452 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`38667 ms` LLM)
- **Rationale**: Install the pooled fit and test the exact artifact in isolated Python. The independent objective check detects calibration or certificate inconsistencies, and the rosette holdouts detect sensitivity to a single sensor group. The stencil and optimizer remain unchanged.
- **Alternatives Weighed**: `Keep rosette-only calibration` (rejected: Pooling the two bench sweep sets reduced worst tracer error by 19% on measurements not used to fit the calibration.)

```bash
set -e
cp /holdout_task/batch_solver.py /tmp/m99_rosette_only.py
python3 - <<'PY'
import ast
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
f = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == '_calibrated_observations')
lines = s.splitlines(keepends=True)
lines[f.lineno-1:f.end_lineno] = ['''def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    banks = sweeps['single_bank_sweeps']
    gain, offset = np.empty(4), np.empty(4)
    for j in range(4):
        take = banks['bank_id'] == j
        x = np.r_[ref[:, j], banks['ref_signal'][take]]
        y = np.r_[raw[:, j], banks['raw_readout'][take]]
        xc = x - x.mean()
        gain[j] = (xc @ (y - y.mean())) / (xc @ xc)
        offset[j] = y.mean() - gain[j] * x.mean()
    measured = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return ((measured - offset) / gain).ravel()
''']
p.write_text(''.join(lines))
PY
if ! (set -e; OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py; OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m99_sensor_holdout.py; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = load('solver', '/holdout_task/batch_solver.py')
old = load('baseline', '/tmp/m99_rosette_only.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']; S, U = m._forward(s)
    def worst(mod):
        E = np.stack([mod._calibrated_observations({**s, 'y_obs_raw': d['y_pulse_raw']}) - S @ d['w_pulse'] for d in item['campaign_diagnostics']])
        E = E.reshape(len(E), len(s['mu_shifts']), -1, 4)
        return np.sqrt(np.mean(E**2, axis=2)).max()
    print('Worst pulse/shift/bank RMS, baseline to installed:', worst(old), worst(m))
    assert worst(m) < worst(old)
    a, b = old.solve_batch(s), m.solve_batch(s)
    for key in ('weights', 'terminal_field'):
        print(key, 'relative change', np.linalg.norm(b[key]-a[key])/np.linalg.norm(a[key]))
    cal = s['sensor_calibration_sweeps']; bank = cal['single_bank_sweeps']; coeff = []
    for j in range(4):
        t = bank['bank_id'] == j
        coeff.append(np.polyfit(np.r_[cal['ref_rosette'][:,j], bank['ref_signal'][t]], np.r_[cal['raw_rosette'][:,j], bank['raw_readout'][t]], 1))
    gain, offset = np.asarray(coeff).T
    y = ((s['y_obs_raw'].reshape(-1,4)-offset)/gain).ravel()
    Z = s['Z_nuisance']; P = np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    D = P@S; w = b['weights']; e = D@w-P@y; ridge = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']
    g = D.T@e+ridge*w; obj = (e@e+ridge*(w@w))/2
    c = b['kkt_certificate']; nu = c['nu_sum']; lo = np.asarray(c['lam_lower']); hi = np.asarray(c['lam_upper'])
    gap = abs(obj-c['dual_lower_bound']); station = np.max(np.abs(g-nu-lo+hi))
    print('Independent public KKT gap and stationarity:', gap, station)
    assert np.array_equal(w, c['weights']) and w.min() >= -1e-12 and w.max() <= cap+1e-12 and min(lo.min(), hi.min()) >= 0
    assert max(gap, station, abs(w.sum()-budget), abs(lo@w), abs(hi@(cap-w)), abs(obj-c['primal_objective'])) < 1e-10
    np.testing.assert_allclose(b['terminal_field'], np.tensordot(w, U, axes=(0,0)), atol=1e-12, rtol=0)
PY
); then cp /tmp/m99_rosette_only.py /holdout_task/batch_solver.py; exit 1; fi
```
- **Container Exit Code**: `0` (`1009 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Worst tracer pulse/shift/bank RMS, rosette-only versus pooled bench sweeps: 1.6181834218095656e-06 1.6181834216987962e-06
Held-out rosettes: 8 worst KKT gap: 1.1539380562197721e-14 worst stationarity residual: 0.0 minimum projected singular ratio: 0.014760429305380572 maximum relative weight change: 0.027694420173391725
Worst pulse/shift/bank RMS, baseline to installed: 1.9981469861476835e-06 1.6181834218095656e-06
weights relative change 1.0656913955892476e-06
terminal_field relative change 1.9063249083371858e-07
Independent public KKT gap and stationarity: 7.11757042193284e-15 3.478120569333498e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ad89a313d69c3459 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_ad89a313 (directory).
```

### Turn 14 — `submit_completed_task` (`14278 ms` LLM)
- **Rationale**: The installed candidate passed the isolated schema check and independent checks of the terminal field and KKT certificate. The measured calibration improvement held on tracer data not used in the fit, with no regression in the rosette holdouts.
- **Alternatives Weighed**: `Retain rosette-only calibration` (rejected: Pooling both sweep sets lowered the worst independent tracer error by 19%, and all eight rosette holdouts passed the KKT checks.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It uses reflecting walls, fits each sensor bank from both calibration sweep sets, and returns source weights, the terminal field, and a KKT certificate. The public schema passed. Worst tracer RMS across pulses, shifts, and banks improved from 2.00e-6 to 1.62e-6; the worst independently recomputed KKT gap across eight leave-one-rosette-out tests was 1.15e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
