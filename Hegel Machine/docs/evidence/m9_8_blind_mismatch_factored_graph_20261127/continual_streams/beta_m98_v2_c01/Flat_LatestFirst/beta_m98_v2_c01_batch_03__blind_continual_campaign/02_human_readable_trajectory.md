# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c01_batch_03__blind_continual_campaign`

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
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `257,463` (`249,255` / `100,224` / `8,208`) |
| **Total Episode Wall Time (s)** | `275.62s` |

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

### Turn 1 — `execute_in_remote_docker` (`16678 ms` LLM)
- **Rationale**: Inspect the diagnostics status, apparatus log, study layout, and validator to determine which boundary evidence is available and what the solver must return before choosing an implementation.
- **Alternatives Weighed**: `Reuse the most recent archived solver unchanged` (rejected: Archive order does not establish the active boundary regime, and calibration must be fitted from the current sweeps.)

```bash
ls -l /tmp/hegel_continual_archive /holdout_task; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle
with open('/holdout_task/public_batch.pkl','rb') as f: batch=pickle.load(f)
print('batch length:',len(batch))
for item in batch:
 print('item:',list(item),'diagnostics present:',item.get('campaign_diagnostics') is not None)
 for k,v in item['study'].items():
  print(k,type(v).__name__,getattr(v,'shape',None), list(v) if isinstance(v,dict) else (v if isinstance(v,(int,float,str)) else ''))
PY
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
M, nx, ny = np.asarray(st["source_modes"]).shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out or not isinstance(out["kkt_certificate"], dict): errs.append("missing:kkt_certificate")
    else:
        kc = out["kkt_certificate"]
        for kk in ("weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"):
            if kk not in kc: errs.append(f"missing_kkt:{kk}")
        if not errs:
            kw = np.asarray(kc["weights"], dtype=float); ll = np.asarray(kc["lam_lower"], dtype=float); lu = np.asarray(kc["lam_upper"], dtype=float)
            po = float(kc["primal_objective"]); dl = float(kc["dual_lower_bound"])
            if np.min(kw) < -1e-6 or np.max(kw) > float(st["u_box"]) + 1e-6: errs.append("kkt_box_violation")
            if abs(float(np.sum(kw)) - float(st["total_budget"])) > 1e-5: errs.append("kkt_budget_violation")
            if np.min(ll) < -1e-6 or np.min(lu) < -1e-6: errs.append("kkt_neg_multiplier")
            if dl < -1e-7 or po < dl - 1e-6 or (po - dl) > 2e-4: errs.append(f"kkt_duality_gap_or_sign:{po:.6f}_vs_{dl:.6f}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
batch length: 1
item: ['study', 'campaign_diagnostics'] diagnostics present: True
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 0.8942552148014796
robin_y float None 1.163864943166582
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) 
source_modes ndarray (8, 14, 12) 
sensor_ij ndarray (32, 2) 
row_bank_id ndarray (96,) 
sensor_calibration_sweeps dict None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3) 
y_obs_raw ndarray (96,) 
aux_manifold_telemetry ndarray (24, 4) 
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
```

### Turn 2 — `execute_in_remote_docker` (`19552 ms` LLM)
- **Rationale**: Compare archived implementations to identify distinct boundary and calibration candidates. Inspect pulse and sweep layouts so the next step can test those candidates on tracer data independently of the inverse fit.
- **Alternatives Weighed**: `Select the boundary operator from archived telemetry alone` (rejected: The current public study has diagnostic pulses, which directly test the forward operator.)

```bash
cd /tmp/hegel_continual_archive; cat archive_manifest.json; diff -u beta_m98_v2_c01_s00_solver.py beta_m98_v2_c01_batch_02_solver.py || true; diff -u beta_m98_v2_c01_batch_02_solver.py beta_m98_v2_c01_batch_01_solver.py || true; cat beta_m98_v2_c01_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -20; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; print('pulse count',len(ds)); print('first pulse fields',[(k,np.shape(v)) for k,v in ds[0].items()]); print('sweeps',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in s['sensor_calibration_sweeps'].items()]); print('bank pattern',s['row_bank_id'][:32]); print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('pulse raw norms',[float(np.linalg.norm(x['y_pulse_raw'])) for x in ds])
PY
```
- **Container Exit Code**: `0` (`411 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
direction
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
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
pulse count 2
first pulse fields [('w_pulse', (8,)), ('y_pulse_raw', (96,))]
sweeps [('ref_rosette', (80, 4)), ('raw_rosette', (80, 4)), ('single_bank_sweeps', ['bank_id', 'ref_signal', 'raw_readout'])]
bank pattern [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
aux mean/std [-0.27406232 -0.08612061  0.04319386 -0.3247204 ] [0.81493315 0.83845502 0.92584971 0.77864814]
pulse raw norms [2.2196408107042447, 2.1702914584263913]
```

### Turn 3 — `execute_in_remote_docker` (`21076 ms` LLM)
- **Rationale**: Compare direct and nuisance-profiled pulse residuals for both archived operators. If neither fits, use the sweep diagnostics to distinguish calibration mismatch from missing boundary terms before expanding the model.
- **Alternatives Weighed**: `Fit a flexible boundary model immediately` (rejected: First measure whether either archived operator already explains both pulses and whether calibration error could account for the mismatch.)

```bash
sed -n '1,61p' /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -20; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import importlib.util, pickle, numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; z=s['Z_nuisance']; q=np.linalg.qr(z)[0]
for n in ('01','02'):
 path=f'/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_{n}_solver.py'; spec=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); S,U=m._forward(s)
 for d in ds:
  sc=dict(s,y_obs_raw=d['y_pulse_raw']); y=m._calibrate(sc); a=S@d['w_pulse']; r=a-y; print(n,'rmse direct/profiled:',np.linalg.norm(r)/np.sqrt(len(r)),np.linalg.norm(r-q@(q.T@r))/np.sqrt(len(r)), 'signal rms',np.linalg.norm(a)/np.sqrt(len(r)))
cs=s['sensor_calibration_sweeps']; x=cs['ref_rosette']; y=cs['raw_rosette']; b=cs['single_bank_sweeps']; print('ref/raw ranges',np.min(x,axis=0),np.max(x,axis=0),np.min(y,axis=0),np.max(y,axis=0)); print('bank fields',[(k,np.shape(v),np.asarray(v).ravel()[:6]) for k,v in b.items()]); a=np.c_[np.ones(len(x)),x]; coef=np.linalg.lstsq(a,y,rcond=None)[0]; print('affine coeffs and RMS',coef,np.sqrt(np.mean((a@coef-y)**2,axis=0)))
PY
```
- **Container Exit Code**: `0` (`604 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
p(hessian, linear, cap, budget):
    """Feasible active set; -1 and +1 mark lower and upper bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w.fill(cap)
        return w, float((hessian @ w + linear).max())
    tol = 1e-11 * max(1.0, np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 100):
        g = hessian @ w + linear
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active == -1)
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
01 rmse direct/profiled: 0.016089379439839124 0.015855986959091767 signal rms 0.19632097905472287
01 rmse direct/profiled: 0.016769063348767415 0.016297232855133056 signal rms 0.19898333317229897
02 rmse direct/profiled: 0.11817780011815184 0.11633685304742658 signal rms 0.08752464833946058
02 rmse direct/profiled: 0.10749028946567739 0.10592520447747933 signal rms 0.10031205610035863
ref/raw ranges [-1.30368096 -1.38332161 -1.28097056 -1.37296195] [1.76550608 1.79820209 1.79700096 1.7954184 ] [-1.60175858 -1.1766569  -1.54120904 -1.18955387] [2.35751486 1.36856941 2.30627334 1.44020266]
bank fields [('bank_id', (24,), array([0, 0, 0, 0, 0, 0])), ('ref_signal', (24,), array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4])), ('raw_readout', (24,), array([-1.3389874 , -0.69400756, -0.04899154,  0.59600999,  1.24098865,
        1.8859931 ]))]
affine coeffs and RMS [[ 7.99998267e-02 -7.00019918e-02  6.00011898e-02 -5.00009491e-02]
 [ 1.29000060e+00 -2.44595072e-07 -2.68503376e-07  2.71816701e-06]
 [-2.12352592e-07  8.00002771e-01 -3.32064249e-07  1.78836886e-06]
 [-8.56350385e-07  2.29021367e-07  1.24999780e+00 -1.08135761e-06]
 [ 8.65396371e-07 -1.91178964e-07  1.09392060e-07  8.30001139e-01]] [9.80879566e-06 1.05245360e-05 8.98263119e-06 7.90435584e-06]
```

### Turn 4 — `execute_in_remote_docker` (`36425 ms` LLM)
- **Rationale**: Test constrained wall-row variants and compare fits in both pulse-holdout directions. Keep added terms only if they improve prediction on the pulse not used for fitting.
- **Alternatives Weighed**: `Adopt reflecting walls unchanged` (rejected: Their remaining pulse error is large enough to test a missing wall term before freezing the stencil.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived open-wall Robin stencil describes the current campaign.` | Witness: `Both direct pulse residuals exceed 0.107 RMSE with Robin walls and fall below 0.017 with reflecting walls, while the affine calibration sweep residual is near 1e-5.` | Blanket Diagnosis: `The discrepancy localizes to the forward model rather than affine sensor calibration or the profiled nuisance subspace.` | Sublated Pivot: `Retain the upwind interior and compare discrete wall conductance and leakage conventions on held-out pulses.`

```bash
cat > /tmp/wall_fit.py <<'PY'
import pickle, importlib.util, numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; Y=np.stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds],axis=1); W=np.stack([d['w_pulse'] for d in ds],axis=1)
x,y=s['nx'],s['ny']; I=eye(x*y,format='csc'); ij=s['sensor_ij']; rows=ij[:,0]+x*ij[:,1]; rhs=np.stack([v.ravel(order='F') for v in np.tensordot(W.T,s['source_modes'],axes=1)],axis=1)
ks=[s['kappa_'+a] for a in 'xy']; vs=[s['vel_'+a] for a in 'xy']; hs=[s['Lx']/(x-1),s['Ly']/(y-1)]; rs=[s['robin_'+a] for a in 'xy']; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(c,l,v=(0,0,0,0), face=False):
 A=[]
 for j,(k,vv,h,r) in enumerate(zip(ks,vs,hs,rs)):
  a=m._axis(k,vv,h,0).tolil()
  for t,i in enumerate((0,len(k)-1)):
   u=i+(1 if t==0 else -1); kk=.5*(k[i]+k[u]) if face else k[i]; e=c[2*j+t]*kk/h**2+v[2*j+t]*abs(vv[i])/h
   a[i,u]=-e; a[i,i]=e+l[2*j+t]*r/h
  A.append(a.tocsc())
 L=kron(eye(y),A[0],format='csc')+kron(A[1],eye(x),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
def err(a): return np.sqrt(np.mean(np.square(a-Y),axis=0))
for label,c,l,v,face in [('reflect',1,0,(0,)*4,False),('twice',2,0,(0,)*4,False),('half',.5,0,(0,)*4,False),('face',1,0,(0,)*4,True),('twice-face',2,0,(0,)*4,True),('reflect-velocity',1,0,(1,)*4,False),('open',1,1,(0,)*4,False)]:
 a=pred([c]*4,[l]*4,v,face);print(label,err(a),'shift RMS',np.sqrt(np.mean((a-Y).reshape(3,32,2)**2,axis=1)).round(6).tolist())
models=[('shared-wall', [1.,0.],lambda b:([b[0]]*4,[b[1]]*4)),('axis-wall',[1.,1.,0.,0.],lambda b:(np.repeat(b[:2],2),np.repeat(b[2:],2))),('four-wall',[1.]*4+[0.]*4,lambda b:(b[:4],b[4:]))]
for label,start,par in models:
 for tr in ([0],[1],[0,1]):
  fit=least_squares(lambda b:(pred(*par(b))-Y)[:,tr].ravel(),start,bounds=([.05]*(len(start)//2)+[-.2]*(len(start)//2),[8.]*(len(start)//2)+[2.]*(len(start)//2)),max_nfev=100)
  print(label,tr,np.round(fit.x,6),err(pred(*par(fit.x))))
q=np.load('/dev/null') if False else None
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_fit.py
python3 - <<'PY'
import json,numpy as np
p=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus summaries');
for e in p:
 a=np.asarray(e['facility_telemetry_trace']); print(e['completed_short_name'],a.shape,np.round(a.mean(0),3),np.round(a.std(0),3))
PY
```
- **Container Exit Code**: `0` (`1885 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
reflect [0.01608938 0.01676906] shift RMS [[0.021499, 0.024196], [0.014317, 0.013687], [0.010462, 0.008416]]
twice [0.03922685 0.0456864 ] shift RMS [[0.059131, 0.069182], [0.028768, 0.03356], [0.017092, 0.018691]]
half [0.03718855 0.03486038] shift RMS [[0.055044, 0.05305], [0.028925, 0.025515], [0.016807, 0.01343]]
face [0.01647166 0.01748111] shift RMS [[0.022205, 0.025316], [0.01449, 0.014161], [0.010533, 0.008679]]
twice-face [0.03982039 0.04639304] shift RMS [[0.060046, 0.070246], [0.029189, 0.034086], [0.017307, 0.018989]]
reflect-velocity [0.01855658 0.02076529] shift RMS [[0.025778, 0.030449], [0.015671, 0.016402], [0.011088, 0.009871]]
open [0.1181778  0.10749029] shift RMS [[0.187276, 0.171866], [0.075198, 0.065859], [0.034219, 0.028054]]
shared-wall [0] [0.958756 0.004425] [0.01510779 0.01417718]
shared-wall [1] [ 0.818576 -0.001954] [0.01561302 0.0133192 ]
shared-wall [0, 1] [ 0.852317 -0.001293] [0.01528842 0.01345369]
axis-wall [0] [ 0.649617  1.422096 -0.037928  0.058341] [0.01480382 0.01439866]
axis-wall [1] [ 0.79753   0.883345  0.051961 -0.037695] [0.01664585 0.01247706]
axis-wall [0, 1] [ 0.592839  1.249366 -0.014388  0.014092] [0.01518513 0.01298837]
four-wall [0] [ 3.021924  2.614563  0.794399  0.368154  0.267645  0.091329 -0.076583
 -0.022631] [0.0108794  0.01104718]
four-wall [1] [ 5.283710e-01  8.000000e+00  1.465385e+00  7.859350e-01  3.542000e-03
 -9.926100e-02  8.000000e-05  9.114000e-03] [0.01485415 0.00142527]
four-wall [0, 1] [ 5.902480e-01  2.064291e+00  1.904497e+00  5.928230e-01 -5.800000e-05
 -3.111400e-02 -9.000000e-04  8.400000e-05] [0.01186832 0.00499672]
apparatus summaries
beta_m98_v2_c01_s00 (16, 8) [-0.01   1.15   2.459 -0.003  1.134 -0.456 -0.029  1.17 ] [0.604 0.119 0.072 0.669 0.12  0.073 0.58  0.148]
beta_m98_v2_c01_batch_01 (16, 8) [ 0.015 -1.169 -0.386  0.068 -1.149 -0.391  0.062 -1.166] [0.599 0.134 0.099 0.651 0.133 0.07  0.643 0.095]
beta_m98_v2_c01_batch_02 (16, 8) [ 0.066  1.098 -0.351  0.078  1.142  2.571  0.089  1.158] [0.642 0.121 0.062 0.694 0.069 0.087 0.675 0.136]
```

### Turn 5 — `execute_in_remote_docker` (`34839 ms` LLM)
- **Rationale**: Test whether opposite-edge coupling or mixed wall conditions explain the residual on both pulses without fitted wall coefficients. Use shift-resolved errors and a held-out calibration sweep to decide whether to retain a discrete stencil or investigate the remaining discrepancy.
- **Alternatives Weighed**: `Freeze the four-wall continuous fit` (rejected: Single-pulse fits yield materially different wall coefficients, so the fitted parameters do not yet establish the stencil.)

```bash
sed '/^for label,c,l,v,face/,$d' /tmp/wall_fit.py > /tmp/wall_discrete.py
cat >> /tmp/wall_discrete.py <<'PY'
from itertools import product
q=np.linalg.qr(s['Z_nuisance'])[0]
def axis(j, kind):
 k,v,h,r=ks[j],vs[j],hs[j],rs[j]; n=len(k); A=m._axis(k,v,h,0).tolil()
 for i,u,t in ((0,1,-1),(n-1,n-2,1)):
  if kind in ('R','RL','RR'):
   if kind=='R' or (kind=='RL' and i==0) or (kind=='RR' and i!=0): A[i,i]+=r/h
   continue
  if kind=='N': continue
  if kind=='D':
   A[i,i]=2*k[i]/h**2; A[i,u]=-2*k[i]/h**2; continue
  if kind=='E':
   A[i,i]=k[i]/h**2+r/h; A[i,u]=-k[i]/h**2; continue
  if kind=='I':
   A[i,i]=2*k[i]/h**2; continue
  w=n-1-i
  if kind in ('Pfull','Pdiff','Pseam','Pavg','Pshift','Pdouble'):
   inner=(k[i]+k[u])*.5/h**2 if kind in ('Pfull','Pdiff','Pavg') else k[i]/h**2
   seam=(k[i]+k[w])*.5/h**2 if kind in ('Pfull','Pdiff','Pavg','Pseam','Pdouble') else k[i]/h**2
   if kind=='Pfull': inner+=max(t*v[i],0)/h; seam+=max(-t*v[i],0)/h
   if kind=='Pavg': inner+=max(t*v[i],0)/h; seam+=max(-t*v[i],0)/h; seam*=.5
   if kind=='Pdouble': seam*=2
   if kind=='Pshift': w=n-2 if i==0 else 1
   A[i,u]=-inner; A[i,w]=-seam; A[i,i]=inner+seam
 return A.tocsc()
def eval_pair(ax,ay):
 L=kron(eye(y),ax,format='csc')+kron(ay,eye(x),format='csc')+V
 a=np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']]); r=a-Y
 return err(a),np.sqrt(np.mean((r-q@(q.T@r))**2,axis=0)),np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1))
kinds=['N','R','RL','RR','D','I','Pfull','Pdiff','Pseam','Pavg','Pshift','Pdouble']
axes=[{k:axis(j,k) for k in kinds} for j in range(2)]
res=[]
for a,b in product(kinds,repeat=2):
 e,f,sh=eval_pair(axes[0][a],axes[1][b]); res.append((max(e),a,b,e,f,sh))
for _,a,b,e,f,sh in sorted(res,key=lambda row:row[0])[:12]: print(a,b,'direct',e,'profiled',f,'shift',sh.round(7).tolist())
for j in range(2): print('axis',j,'k edges',ks[j][[0,1,-2,-1]],'velocity edges',vs[j][[0,1,-2,-1]])
ref=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; a=np.c_[np.ones(len(ref)),ref]; test=np.arange(len(ref))%5==0; coef=np.linalg.lstsq(a[~test],raw[~test],rcond=None)[0]; print('held-sweep rmse by channel',np.sqrt(np.mean((a[test]@coef-raw[test])**2,axis=0))); print('pulse weights',W.T)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_discrete.py
```
- **Container Exit Code**: `0` (`778 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Pfull Pfull direct [1.05463490e-06 1.07459164e-06] profiled [1.03887152e-06 1.05678793e-06] shift [[8e-07, 9e-07], [1.1e-06, 1.1e-06], [1.2e-06, 1.2e-06]]
Pseam Pseam direct [0.00045849 0.0002403 ] profiled [0.00044148 0.00023655] shift [[0.0006331, 0.0003758], [0.0003827, 0.0001478], [0.0002888, 0.0001008]]
Pseam Pdiff direct [0.00058874 0.00032594] profiled [0.00056789 0.00032035] shift [[0.0008452, 0.0005148], [0.0004629, 0.0001964], [0.0003335, 0.0001229]]
Pseam Pfull direct [0.0006774  0.00021111] profiled [0.00066365 0.00020806] shift [[0.0010202, 0.000305], [0.0005062, 0.000173], [0.0002821, 0.0001037]]
Pdiff Pfull direct [0.00023128 0.00072173] profiled [0.00022965 0.0007086 ] shift [[0.0002901, 0.0010828], [0.0002136, 0.0005394], [0.0001752, 0.0003149]]
Pdouble Pdiff direct [0.0007321 0.0007267] profiled [0.00069729 0.00070586] shift [[0.0009517, 0.0008494], [0.0006434, 0.0007213], [0.0005369, 0.0005852]]
Pdouble Pseam direct [0.00063135 0.00073314] profiled [0.00060163 0.00071274] shift [[0.000772, 0.0008537], [0.0005853, 0.0007319], [0.0005072, 0.0005899]]
Pdouble Pfull direct [0.00081563 0.00086723] profiled [0.00080708 0.00084778] shift [[0.001142, 0.0011075], [0.0006693, 0.0008065], [0.0004936, 0.0006158]]
Pfull Pseam direct [0.0009829  0.00039619] profiled [0.00095746 0.00038915] shift [[0.0015177, 0.0006334], [0.0006741, 0.0002358], [0.0003747, 0.0001186]]
Pdiff Pseam direct [0.0011184  0.00109137] profiled [0.00109597 0.00107337] shift [[0.00171, 0.00169], [0.0007882, 0.0007447], [0.0004551, 0.0004031]]
Pfull Pdiff direct [0.00114609 0.00049024] profiled [0.00111632 0.00048137] shift [[0.0017666, 0.0007796], [0.0007908, 0.0002999], [0.0004406, 0.0001525]]
Pseam Pdouble direct [0.00122396 0.00041128] profiled [0.00119709 0.00040815] shift [[0.0014368, 0.0005346], [0.00119, 0.000371], [0.0010069, 0.0002898]]
axis 0 k edges [0.4503445  0.48384869 0.48384869 0.4503445 ] velocity edges [ 0.35        0.34500413 -0.15511704 -0.20572484]
axis 1 k edges [0.57243569 0.56757485 0.33729653 0.33243569] velocity edges [-0.         -0.05560533 -0.25469696 -0.22652476]
held-sweep rmse by channel [1.29908417e-05 1.17420965e-05 9.07793312e-06 8.64641724e-06]
pulse weights [[1.  0.  0.  0.5 0.  0.  0.  0. ]
 [0.  1.  0.  0.  0.5 0.  0.  0. ]]
```

### Turn 6 — `execute_in_remote_docker` (`36391 ms` LLM)
- **Rationale**: Promote the parameter-free boundary stencil supported by both tracer pulses. Check the saved solver against the public schema, each pulse and shift, and the inverse problem’s feasibility, terminal field, and KKT certificate.
- **Alternatives Weighed**: `Periodic diffusion without seam advection` (rejected: It leaves pulse residuals hundreds of times larger than the full periodic upwind stencil.)
- **ATMS NOGOOD DECLARED**: Refuted `Local reflecting boundary rows suffice for this campaign.` | Witness: `Reflecting walls yield worst-pulse RMSE 0.01677; periodic upwind rows reduce it to 1.075e-6 without fitted boundary parameters.` | Blanket Diagnosis: `The missing terms are the opposite-edge diffusion and row-local upwind couplings on both axes.` | Sublated Pivot: `Apply the same discrete transport stencil at every grid point using modular neighbors; retain dynamic sensor calibration and the constrained inverse solver.`

```bash
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py').read_text()
start = src.index('def _axis(')
end = src.index('def _forward(', start)
src = src[:start] + '''def _axis(kappa, velocity, h):
    """Row-local upwind transport with periodic neighbors on both ends."""
    left = (kappa + np.roll(kappa, 1)) * (0.5 / h**2) + np.maximum(velocity, 0) / h
    right = (kappa + np.roll(kappa, -1)) * (0.5 / h**2) + np.maximum(-velocity, 0) / h
    i = np.arange(len(kappa))
    n = len(i)
    return csc_matrix((np.r_[-left, left + right, -right],
                       (np.tile(i, 3), np.r_[(i - 1) % n, i, (i + 1) % n])),
                      shape=(n, n))


''' + src[end:]
src = src.replace('with calibrated observations', 'with periodic boundaries, calibrated observations').replace('import diags, eye, kron', 'import csc_matrix, diags, eye, kron')
for a in 'xy':
    src = src.replace(f", study['robin_{a}'])", ')')
Path('/holdout_task/batch_solver.py').write_text(src)
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for i,item in enumerate(pickle.load(open('/holdout_task/public_batch.pkl','rb'))):
 s=item['study']; S,U=m._forward(s); z=s['Z_nuisance']; raw=m._calibrate(s); proj=lambda a:a-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a); A,b=proj(S),proj(raw)
 for d in item.get('campaign_diagnostics') or []:
  rr=(S@d['w_pulse']-m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw']))).reshape(len(s['mu_shifts']),-1)
  er=np.sqrt(np.mean(rr**2,axis=1)); print('item',i,'pulse shift RMSE:',er); assert max(er)<2e-5
 o=m.solve_batch(s); w=o['weights']; c=o['kkt_certificate']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; r=A@w-b; g=A.T@r+ridge*w; lo=np.asarray(c['lam_lower']); up=np.asarray(c['lam_upper']); nu=c['nu_sum']; J=.5*(r@r+ridge*(w@w)); gap=J-c['dual_lower_bound'];
 print('weights:',w,'budget error:',abs(w.sum()-budget),'duality gap:',gap,'stationarity:',np.max(abs(g-nu-lo+up)))
 assert np.allclose(w,c['weights'],atol=1e-12,rtol=0) and abs(w.sum()-budget)<1e-11 and np.all((w>=0)&(w<=cap)); assert np.all(lo>=0) and np.all(up>=0) and np.max(abs(g-nu-lo+up))<1e-10 and max(np.max(lo*w),np.max(up*(cap-w)))<1e-10; assert abs(J-c['primal_objective'])<1e-10 and abs(gap)<1e-10
 assert np.allclose(o['terminal_field'],np.tensordot(w,U,axes=1),rtol=0,atol=1e-12)
PY
```
- **Container Exit Code**: `0` (`705 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
item 0 pulse shift RMSE: [8.39987991e-07 1.07587831e-06 1.21394825e-06]
item 0 pulse shift RMSE: [8.69533035e-07 1.09643722e-06 1.22718348e-06]
weights: [0.01232882 0.23841249 0.         0.         0.42       0.33326338
 0.         0.39599532] budget error: 4.440892098500626e-16 duality gap: 2.220446049250313e-16 stationarity: 8.881784197001252e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2b04478c23f77dc5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2b04478c (directory).
```

### Turn 7 — `submit_completed_task` (`13320 ms` LLM)
- **Rationale**: The periodic stencil explains both tracer pulses across all shifts without fitted wall parameters. The saved artifact passed isolated schema, feasibility, and certificate checks, and the host independently verified the field, weights, and certificate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5360'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst tracer-pulse RMSE across all shifts was 1.23e-6. Independent verification measured relative errors of 9.10e-7 for the terminal field and 5.59e-6 for the weights.`

### Turn 8 — `execute_in_remote_docker` (`37176 ms` LLM)
- **Rationale**: An independent stencil and seam/interior pulse split can expose boundary errors hidden by aggregate RMSE; a failure would reopen stencil selection. Calibration folds bound how far sweep uncertainty can move the inverse objective; a large bound would warrant revisiting calibration before submission.
- **Alternatives Weighed**: `Test classification bias and relay modes` (rejected: This study has continuous tracer readings, no labeled cohorts, and no relay states. The measured nuisance subspace and withheld calibration rows are the relevant stress slices.)

```bash
cat > /tmp/audit_periodic.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def axis(k, v, h):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        for step in (-1,1):
            j = (i+step)%n
            c = (k[i]+k[j])/(2*h*h) + max(-step*v[i],0)/h
            a[i,j] -= c
            a[i,i] += c
    return a

for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = item['study']; nx, ny = s['nx'],s['ny']
    axes = [axis(s['kappa_'+a], s['vel_'+a], s['L'+a]/(s['n'+a]-1)) for a in 'xy']
    for a in axes:
        np.testing.assert_allclose(a@np.ones(len(a)),0,atol=1e-12)
        assert a[0,-1]<0 and a[-1,0]<0
    L = np.kron(np.eye(ny),axes[0])+np.kron(axes[1],np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    ij = s['sensor_ij']; ii = ij[:,0]+nx*ij[:,1]; modes = s['source_modes']; rhs = np.column_stack([u.ravel(order='F') for u in modes]); I = np.eye(nx*ny)
    fields = [np.linalg.solve(L+mu*I,rhs) for mu in s['mu_shifts']]
    S = np.vstack([u[ii] for u in fields]); S0,_ = m._forward(s)
    print('independent forward max discrepancy:',np.max(abs(S0-S)))
    np.testing.assert_allclose(S0,S,rtol=0,atol=1e-12)
    edge = (ij[:,0]==0)|(ij[:,0]==nx-1)|(ij[:,1]==0)|(ij[:,1]==ny-1)
    sweeps = s['sensor_calibration_sweeps']; ref,raw = sweeps['ref_rosette'],sweeps['raw_rosette']; design = np.c_[np.ones(len(ref)),ref]
    z = s['Z_nuisance']; p = np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); A = p@S
    out = m.solve_batch(s); w = out['weights']; cap = s['u_box']; ridge = s['source_ridge']; wt = fields[-1]@w
    np.testing.assert_allclose(out['terminal_field'],wt.reshape((nx,ny),order='F'),rtol=0,atol=1e-12)
    def calibrate(raw_vector,coef):
        return np.linalg.solve(coef[1:].T,(raw_vector.reshape(-1,4)-coef[0]).T).T.ravel()
    cal_err=[]; pulse_errors=[]; gaps=[]
    for fold in range(5):
        test = np.arange(len(ref))%5==fold
        coef = np.linalg.lstsq(design[~test],raw[~test],rcond=None)[0]
        cal_err.extend(np.sqrt(np.mean((design[test]@coef-raw[test])**2,axis=0)))
        for pulse in item.get('campaign_diagnostics') or []:
            residual = (S@pulse['w_pulse']-calibrate(pulse['y_pulse_raw'],coef)).reshape(-1,len(ii))
            pulse_errors.extend(float(e) for mask in (edge,~edge) if mask.any() for e in np.sqrt(np.mean(residual[:,mask]**2,axis=1)))
        y = calibrate(s['y_obs_raw'],coef); g = A.T@(A@w-p@y)+ridge*w
        linear = np.zeros(len(w)); remaining = s['total_budget']
        for i in np.argsort(g):
            linear[i] = min(remaining,cap); remaining -= linear[i]
        gaps.append(g@(w-linear))
    print('worst held-out calibration channel RMSE:',max(cal_err))
    print('worst pulse/shift/spatial-slice RMSE:',max(pulse_errors),'edge sensors:',int(edge.sum()))
    print('worst calibration-fold objective-gap bound:',max(gaps))
    print('nuisance attenuation operator norm:',np.linalg.norm(p@np.linalg.qr(z)[0],2))
    assert max(cal_err)<3e-5 and max(pulse_errors)<2e-5 and max(gaps)<2e-5
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_periodic.py
```
- **Container Exit Code**: `0` (`459 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent forward max discrepancy: 5.551115123125783e-16
worst held-out calibration channel RMSE: 1.357350898574377e-05
worst pulse/shift/spatial-slice RMSE: 2.131219645205073e-06 edge sensors: 11
worst calibration-fold objective-gap bound: 3.6021698887580524e-07
nuisance attenuation operator norm: 0.0010245248692024999

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2b04478c23f77dc5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2b04478c (directory).
```

### Turn 9 — `submit_completed_task` (`14229 ms` LLM)
- **Rationale**: The independent dense stencil agreed with the saved solver to 5.55e-16. The worst calibration-fold objective-gap bound was 3.60e-7, and the host independently verified the weights, terminal field, and certificate. No further changes are warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5360'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and KKT checks passed. Worst tracer-pulse RMSE across calibration folds, shifts, and sensor regions was 2.13e-6.`

### Turn 10 — `submit_completed_task` (`17763 ms` LLM)
- **Rationale**: The independent dense stencil agreed with the saved solver to 5.55e-16, and the worst calibration-fold objective-gap bound was 3.60e-7. The host independently verified the field, weights, and certificate for the unchanged artifact. Repeating those checks would not change the submission decision.
- **Alternatives Weighed**: `Run additional relay or cohort-classification checks` (rejected: The tomography study has no relay states or classification cohorts. The relevant boundary, calibration, and nuisance checks already ran on the saved solver.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and independent KKT checks passed. Worst tracer-pulse RMSE across held-out calibration folds, shifts, and sensor regions was 2.13e-6.`

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
