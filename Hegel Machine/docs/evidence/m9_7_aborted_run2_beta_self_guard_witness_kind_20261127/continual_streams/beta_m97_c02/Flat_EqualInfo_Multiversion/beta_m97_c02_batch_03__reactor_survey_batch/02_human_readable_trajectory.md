# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c02_batch_03__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c02_step_03` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `297,823` (`289,105` / `85,760` / `8,718`) |
| **Total Episode Wall Time (s)** | `312.38s` |

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

### Turn 1 — `execute_in_remote_docker` (`15269 ms` LLM)
- **Rationale**: Inspect the public contract and prior implementation before editing: mismatched grid, sensor-stack, or certificate conventions would change the solver implementation. The skill lookup checks whether an existing profiling or KKT primitive can replace new code.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The new periodic boundary and reaction modulation require explicit changes to the forward operator.)

```bash
cd /holdout_task && cat public_asset_index.json && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && sed -n '1,270p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`395 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
]
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


def solve_beta_m97_c02_batch_01(study: dict) -> dict:
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
No matching skill cards found.
```

### Turn 2 — `execute_in_remote_docker` (`18598 ms` LLM)
- **Rationale**: Read the validator and archived operator directly, and summarize the calibration rows without dumping the study arrays. This detects conventions that would change the forward model or bank fitting before implementation.
- **Alternatives Weighed**: `Implement from the truncated transcript` (rejected: The omitted operator and sweep formats determine boundary indexing, archive fallback, and the certificate convention.)

```bash
cd /holdout_task && cat validate_public_schema.py && wc -l /tmp/hegel_continual_archive/* && sed -n '1,180p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py && python3 - <<'PY'
import pickle,json,numpy as np
p=pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb')); print('payload:',type(p),len(p))
if isinstance(p,dict): print('payload keys:',list(p)); p=[p] if 'study' in p else next(iter(p.values()))
for i,c in enumerate(p):
 s=c['study']; q=s['sensor_calibration_sweeps']; print(i,'study',[(k, getattr(v,'shape',v if np.isscalar(v) else type(v).__name__)) for k,v in s.items()]); print('sweeps',repr(q)[:2500]); print('weights',c.get('gt',{}).get('w_opt'))
 if i>=4: break
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive head',repr(h)[:1700])
PY
```
- **Container Exit Code**: `0` (`462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8655, -0.18291523,
        1.38426384,  1.22230043,  0.39810433,  1.31656903, -0.88792304,
        1.09701998, -1.1266033 , -0.97346918, -1.79231193, -0.83130317,
       -1.54489438,  2.5830024 ,  2.8814544 , -1.05622191, -1.83819539,
        0.66803269,  0.91270065,  2.41531614, -1.17898885,  1.57784107,
       -0.91231133, -0.18447169,  0.28418531, -1.17237947, -0.88826131,
       -0.93492238, -0.54441872,  0.27338717, -0.80135299, -0.46611313,
        0.81635759,  0.78643449,  0.19715051, -0.51399964, -1.20312103,
       -0.09252017,  0.84610943, -0.18815762,  0.73233617, -0.45332299,
       -1.63826628, -1.14439699, -1.85061221])}
weights [0.27720364 0.43932746 0.21324021 0.15535484 0.25288942 0.19391384
 0.14229485 0.18940363 0.20291836 0.12335882 0.19390701 0.21618794]
archive head [{'episode_short_name': 'beta_m97_c02_s00', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [0.268167, 1.973482, 0.155493, -0.991373, 0.005186, 0.883223, 0.855335, -0.111767, 2.179932, 0.515458, 2.19924, 1.8819, 0.790972, -1.447989, -0.050426, 1.319399, 2.029099, -0.88645, 0.286097, -0.734514, 0.486405, -1.23754, -1.240911, 1.652451, 2.155713, 0.453713, 0.912999, 1.858977, -1.367223, 1.234135, -0.671947, -0.037994, -1.441484, 1.234848, 2.043594, 2.149632, 1.492805, -1.165341, -0.972774, 0.868599, 0.633375, 1.932977, 0.643473, 0.809337, 0.674338, 1.857391, -0.593005, -0.394796], 'raw_readout': [0.268167, 1.973482, 0.155493, -0.991373, 0.005186, 0.883223, 0.855335, -0.111767, 2.179932, 0.515458, 2.19924, 1.8819, 0.790972, -1.447989, -0.050426, 1.319399, 2.029099, -0.88645, 0.286097, -0.734514, 0.486405, -1.23754, -1.240911, 1.652451, 2.155713, 0.453713, 0.912999, 1.858977, -1.367223, 1.234135, -0.671947, -0.037994, -1.441484, 1.234848, 2.043594, 2.149632, 1.492805, -1.165341, -0.972774, 0.868599, 0.633375, 1.932977, 0.643473, 0.809337, 0.674338, 1.857391, -0.593005, -0.394796]}, {'episode_short_name': 'beta_m97_c02_batch_01', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [-0.791143, -0.216611, -1.13301, -1.30817, 1.713448, 0.875702, 0.943663, -0.29371, 1.475187, 1.843625, 0.932592, 2.112826, 0.840203, -0.917936, -1.382269, -1.346584, -0.366771, -0.228184, 1.263794, 1.49297, 1.843164, 0.64334, 2.147042, -0.014648, 1.758375, 1.79159
```

### Turn 3 — `execute_in_remote_docker` (`21177 ms` LLM)
- **Rationale**: The prior output omitted the forward stencil. Read only that section and summarize the sweep fits: endpoint conventions determine the periodic edit, while historical fit differences determine whether fallback should pool episodes or use the latest clean sweep.
- **Alternatives Weighed**: `Pool historical and in-batch calibration sweeps` (rejected: The task requires fitting clean banks from in-batch data; historical sweeps are reserved for sparse or EMI-affected banks.)

```bash
cd /holdout_task && sed -n '1,105p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py && tail -45 validate_public_schema.py && python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))
if isinstance(p,dict): p=[p] if 'study' in p else next(iter(p.values()))
def profile(q):
 b,x,y=(np.asarray(q[k]) for k in ('bank_id','ref_signal','raw_readout'))
 out=[]
 for i in range(4):
  m=b==i; a=np.column_stack((x[m],np.ones(m.sum()))); v=np.linalg.lstsq(a,y[m],rcond=None)[0]
  out.append((int(m.sum()),v.round(8).tolist(),round(float(np.max(np.abs(a@v-y[m]),initial=0)),8)))
 return out
for c in p:
 s=c['study']; print('loop',s['recirculation_loop_mode'],'keys',list(s),'sweeps',profile(s['sensor_calibration_sweeps']))
for q in json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')): print(q['episode_short_name'],profile(q))
PY
```
- **Container Exit Code**: `0` (`441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ath.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c02_batch_03_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c02_batch_03.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c02_batch_03(st)
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
loop 0.85 keys ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'vel_x', 'robin_x', 'reaction_x', 'kappa_y', 'vel_y', 'robin_y', 'reaction_y', 'V_reaction_2d', 'recirculation_loop_mode', 'past_vortex_circulation_trace', 'critical_vortex_circulation', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget'] sweeps [(12, [0.81000001, -0.15], 1.4e-07), (12, [1.32999997, 0.13999999], 1.9e-07), (12, [0.85999999, 0.11000002], 1.5e-07), (12, [1.20000002, -0.11999996], 2.3e-07)]
beta_m97_c02_s00 [(12, [1.0, -0.0], 0.0), (12, [1.0, 0.0], 0.0), (12, [1.0, -0.0], 0.0), (12, [1.0, 0.0], 0.0)]
beta_m97_c02_batch_01 [(12, [0.81000007, -0.14999997], 4.1e-07), (12, [1.33000005, 0.13999991], 5.4e-07), (12, [0.86000006, 0.11000008], 6.2e-07), (12, [1.2, -0.11999997], 4.3e-07)]
beta_m97_c02_batch_02 [(12, [0.81000013, -0.15000002], 4.3e-07), (12, [1.32999984, 0.13999997], 2.7e-07), (12, [0.86000012, 0.10999987], 6.2e-07), (12, [1.20000007, -0.12000011], 5.5e-07)]
```

### Turn 4 — `execute_in_remote_docker` (`17397 ms` LLM)
- **Rationale**: Read only the missing forward assembly and summarize the relevant inputs. This identifies which grid spacing, reaction terms, and sensor ordering the implementation should preserve before testing it against the public forward matrix and terminal field.
- **Alternatives Weighed**: `Carry the prior vortex-memory term into periodic mode without inspecting it` (rejected: The archived forward assembly was truncated, so doing so could introduce an extra reaction term relative to the specified periodic operator.)

```bash
sed -n '65,125p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))[0]['study']
for k in ('reaction_x','reaction_y','past_vortex_circulation_trace','mu_shifts'):
 v=np.asarray(s[k]); print(k, 'shape',v.shape,'range',(v.min(),v.max()))
PY
```
- **Container Exit Code**: `0` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study['Ly']
    excess = np.maximum(0, np.asarray(study['past_vortex_circulation_trace'])
                        - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx / (nx - 1))
    # Fouling seals both boundary axes; the sorption mask is localized in x.
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], ly / (ny - 1))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    reaction = np.asarray(study['V_reaction_2d'])
    if sealed:
        xs = np.linspace(0, lx, nx)[:, None]
        ys = np.linspace(0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator += diags(reaction.ravel(order='F'), format='csc')
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
reaction_x shape (18,) range (0.3, 0.45999999999999996)
reaction_y shape (16,) range (0.26, 0.4)
past_vortex_circulation_trace shape (16,) range (0.083915, 0.331506)
mu_shifts shape (4,) range (0.0, 0.9)
```

### Turn 5 — `execute_in_remote_docker` (`46937 ms` LLM)
- **Rationale**: Build the solver and compare the profiled design and observations separately before assessing the weights, terminal field, and certificate. A design mismatch points to the transport stencil or grid; a target mismatch points to calibration or nuisance projection.
- **Alternatives Weighed**: `Switch to an endpoint-exclusive periodic grid immediately` (rejected: The archived study uses endpoint-inclusive coordinates; compare the forward matrix against the public reference before changing the spacing or coordinates.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py').read_text()
p = p.replace('Reaction–transport tomography with vortex-memory fouling and certified source inversion.', 'Bank-calibrated reaction–transport tomography with certified source inversion.')
p = p.replace('diags, eye, kron', 'coo_matrix, diags, eye, kron').replace('solve_beta_m97_c02_batch_02', 'solve_beta_m97_c02_batch_03')
a = p.index('        if clean.sum() >= 2')
b = p.index('    gain, offset = parameters', a)
p = p[:a] + '''        if x.size >= 2 and clean.all():
            parameters[bank] = _line(x, y)
            continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        for sweep in reversed(history):
            ref, raw = _bank_rows(sweep, bank)
            keep = _clean_rows(ref, raw)
            if keep.sum() >= 2 and np.ptp(ref[keep]) > 0:
                prior_gain, prior_offset = _line(ref[keep], raw[keep])
                break
        else:
            raise ValueError(f'Bank {bank} lacks an identifiable calibration')
        clean = np.abs(y - (prior_gain * x + prior_offset)) <= 3.0
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            parameters[bank] = _line(x[clean], y[clean])
        else:
            parameters[bank] = prior_gain, prior_offset
''' + p[b:]
a = p.index('def _forward')
p = p[:a] + '''def _periodic_axis(kappa, velocity, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    i = np.arange(kappa.size)
    im, ip = (i - 1) % i.size, (i + 1) % i.size
    face_m = (kappa + kappa[im]) / (2 * h**2)
    face_p = (kappa + kappa[ip]) / (2 * h**2)
    lower = -face_m - np.maximum(velocity, 0) / h
    upper = -face_p - np.maximum(-velocity, 0) / h
    main = face_m + face_p + np.abs(velocity) / h
    return coo_matrix((np.r_[lower, upper, main], (np.tile(i, 3), np.r_[im, ip, i])),
                      shape=(i.size, i.size)).tocsc()


''' + p[a:]
a = p.index('    excess =', p.index('def _forward'))
b = p.index('    operator += diags', a)
p = p[:a] + '''    loop = study['recirculation_loop_mode']
    hx, hy = lx / (nx - 1), ly / (ny - 1)
    xs = np.linspace(0, lx, nx)[:, None]
    ys = np.linspace(0, ly, ny)[None, :]
    reaction = np.asarray(study['V_reaction_2d'])
    if loop > 0.50:
        ax = _periodic_axis(study['kappa_x'], study['vel_x'], hx)
        ay = _periodic_axis(study['kappa_y'], study['vel_y'], hy)
        reaction = reaction + 0.75 * loop * np.cos(2 * np.pi * xs / lx) * np.cos(2 * np.pi * ys / ly)
    else:
        excess = np.maximum(0, np.asarray(study['past_vortex_circulation_trace'])
                            - study['critical_vortex_circulation'])
        v_mem = float(excess @ excess)
        sealed = loop == 0 and v_mem > 0
        ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], hx)
        ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], hy)
        if sealed:
            wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
            reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
''' + p[b:]
Path('beta_m97_c02_batch_03_solver.py').write_text(p)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/beta_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for i,c in enumerate(pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))):
 s,g=c['study'],c['gt']; z=s['Z_nuisance']; gram=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); profile=lambda a:a-z@np.linalg.solve(gram,z.T@a)
 A,_=m._forward(s); out=m.solve_beta_m97_c02_batch_03(s); cert=out['kkt_certificate']; grad=g['S_tilde'].T@(g['S_tilde']@out['weights']-g['r_y'])+s['source_ridge']*out['weights'];
 for k,v,ref in [('profiled_design',profile(A),g['S_tilde']),('profiled_target',profile(m._calibrate(s)),g['r_y']),('weights',out['weights'],g['w_opt']),('terminal_field',out['terminal_field'],g['terminal_field_A'])]:
  print(i,k,'max_abs',np.max(np.abs(v-ref)))
 print(i,'reference_stationarity',np.max(np.abs(grad-cert['nu_sum']-cert['lam_lower']+cert['lam_upper'])),'self_gap',cert['primal_objective']-cert['dual_lower_bound'])
PY
```
- **Container Exit Code**: `1` (`665 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6315512ae26e8fee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6315512a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 64, in <module>
    out = mod.solve_beta_m97_c02_batch_03(st)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module '_cand' has no attribute 'solve_beta_m97_c02_batch_03'. Did you mean: 'solve_beta_m97_c02_batch_01'?
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: module 'candidate' has no attribute 'solve_beta_m97_c02_batch_03'. Did you mean: 'solve_beta_m97_c02_batch_01'?
```

### Turn 6 — `execute_in_remote_docker` (`22188 ms` LLM)
- **Rationale**: Rename the existing function without changing its numerics. Compare the design and target separately so a public-reference mismatch can be traced to transport or calibration before revising the solver.
- **Alternatives Weighed**: `Change the transport stencil before rerunning validation` (rejected: The traceback identifies an export-name mismatch, not a numerical failure.)

```bash
set -e
cd /holdout_task
sed -i 's/^def solve_beta_m97_c02_batch_01(/def solve_beta_m97_c02_batch_03(/' beta_m97_c02_batch_03_solver.py
python3 -I validate_public_schema.py || true
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/beta_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for i,c in enumerate(pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))):
 s,g=c['study'],c['gt']; z=s['Z_nuisance']; gram=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); profile=lambda a:a-z@np.linalg.solve(gram,z.T@a)
 A,_=m._forward(s); out=m.solve_beta_m97_c02_batch_03(s); cert=out['kkt_certificate']; w=out['weights']; grad=g['S_tilde'].T@(g['S_tilde']@w-g['r_y'])+s['source_ridge']*w
 for k,v,ref in [('design',profile(A),g['S_tilde']),('target',profile(m._calibrate(s)),g['r_y']),('weights',w,g['w_opt']),('terminal_field',out['terminal_field'],g.get('terminal_field_A'))]:
  if ref is not None: print(i,k,'max_abs',np.max(np.abs(v-ref)))
  else: print('available reference keys',list(g))
 print(i,'reference_stationarity',np.max(np.abs(grad-cert['nu_sum']-cert['lam_lower']+cert['lam_upper'])),'self_gap',cert['primal_objective']-cert['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`647 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
0 design max_abs 1.3877787807814457e-16
0 target max_abs 7.420397940549606e-08
0 weights max_abs 2.650539188964185e-07
0 terminal_field max_abs 1.3292430045552805e-08
0 reference_stationarity 6.130937901804856e-08 self_gap 1.942890293094024e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f006df1576de80f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7f006df1 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30546 ms` LLM)
- **Rationale**: Stress the exact verified artifact using the public study with individual bank rows removed or spiked, then exercise the loop threshold and budget limits. Calibration drift would call for revising bank selection; a feasibility or gap failure would isolate the constrained optimizer.
- **Alternatives Weighed**: `Submit on public agreement alone` (rejected: The public case does not exercise sparse-bank fallback, EMI rejection, or budgets at the box-simplex limits.)

```bash
set -e
cd /holdout_task
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))[0]['study']
q = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}; b = q['bank_id']
baseline = m.solve_beta_m97_c02_batch_03(s); y0 = m._calibrate(s)
worst = np.zeros(3); count = 0
for bank in range(4):
    rows = np.flatnonzero(b == bank)
    variants = [{k: v[keep] for k,v in q.items()} for keep in [b != bank, (b != bank) | (np.arange(len(b)) == rows[0])]]
    for i in rows:
        for spike in (-30., -3.01, 3.01, 30.):
            raw = q['raw_readout'].astype(float).copy(); raw[i] += spike
            variants.append(dict(q, raw_readout=raw))
    for sweeps in variants:
        changed = dict(s, sensor_calibration_sweeps=sweeps)
        out = m.solve_beta_m97_c02_batch_03(changed)
        worst = np.maximum(worst, [np.max(abs(m._calibrate(changed)-y0)), np.max(abs(out['weights']-baseline['weights'])), np.max(abs(out['terminal_field']-baseline['terminal_field']))]); count += 1
print('calibration stress slices:', count, 'worst calibrated-target, weight, and field delta:', worst)
assert np.all(worst < [2e-6, 1e-5, 1e-6])
worst_gap = 0.
for loop in (0., 0.50, 0.50001, 0.85):
    for frac in (0., 0.001, 0.5, 0.999, 1.):
        study = dict(s, recirculation_loop_mode=loop, total_budget=frac*s['u_box']*len(baseline['weights']))
        out = m.solve_beta_m97_c02_batch_03(study); w = out['weights']; c = out['kkt_certificate']
        gap = c['primal_objective'] - c['dual_lower_bound']; worst_gap = max(worst_gap, gap)
        assert np.all(np.isfinite(out['terminal_field'])) and w.min() >= -1e-10 and w.max() <= study['u_box']+1e-10
        assert abs(w.sum()-study['total_budget']) < 1e-10 and -1e-9 <= gap < 1e-8, (loop, frac, gap)
print('loop/budget stress slices: 20; worst certified gap:', worst_gap)
PY
```
- **Container Exit Code**: `0` (`2163 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration stress slices: 200 worst calibrated-target, weight, and field delta: [4.95394203e-07 6.36409487e-07 3.55223240e-08]
loop/budget stress slices: 20; worst certified gap: 1.7763568394002505e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f006df1576de80f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7f006df1 (directory).
```

### Turn 8 — `submit_completed_task` (`13994 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python validation. Sparse-bank and EMI tests shifted weights by at most 6.37e-7, and the loop-threshold and budget-limit tests preserved feasibility.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_03_solver.py': 'FILE:8122'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public validator passed, along with 200 calibration stress cases and 20 loop-mode and budget stress cases.`

### Turn 9 — `execute_in_remote_docker` (`27854 ms` LLM)
- **Rationale**: The stencil checks detect an omitted wrap face or a sign error that would warrant correcting the operator. Recomputing the objective and dual bound from the public reference matrices tests whether calibration error materially narrows the certificate margin.
- **Alternatives Weighed**: `Run classification, relay, or stochastic-control diagnostics` (rejected: The study specifies continuous reaction–transport measurements and a convex inverse problem, not class labels, relay states, or a Fokker–Planck/HJB model.)

```bash
set -e
cd /holdout_task
sed -n '1,145p' beta_m97_c02_batch_03_solver.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for i, row in enumerate(pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))):
    s, g = row['study'], row['gt']
    for axis in ('x','y'):
        k, v = (np.asarray(s[key+'_'+axis]) for key in ('kappa','vel'))
        h = s['L'+axis]/(len(k)-1)
        a = m._periodic_axis(k, v, h).toarray()
        err = np.max(np.abs(a.sum(axis=1)))
        endpoints = np.array([-(k[0]+k[-1])/(2*h*h)-max(v[0],0)/h, -(k[0]+k[-1])/(2*h*h)-max(-v[-1],0)/h])
        wrap = np.max(np.abs(a[[0,-1],[-1,0]] - endpoints))
        print(i, axis, 'constant-null residual', err, 'wrap-face residual', wrap)
        assert max(err, wrap) < 1e-10
    out = m.solve_beta_m97_c02_batch_03(s); c = out['kkt_certificate']; w = out['weights']; a, y = g['S_tilde'], g['r_y']
    r = a@w-y; grad = a.T@r+s['source_ridge']*w
    obj = (r@r+s['source_ridge']*(w@w))/2
    lb = obj-grad@w+c['nu_sum']*s['total_budget']+s['u_box']*np.minimum(grad-c['nu_sum'],0).sum()
    cert_error = max(abs(obj-c['primal_objective']), abs(lb-c['dual_lower_bound']))
    print(i, 'reference KKT gap', obj-lb, 'certificate value deviation', cert_error)
    assert -1e-10 <= obj-lb < 1e-6 and cert_error < 1e-6
PY
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lx) * np.cos(2 * np.pi * ys / ly)
    else:
        excess = np.maximum(0, np.asarray(study['past_vortex_circulation_trace'])
                            - study['critical_vortex_circulation'])
        v_mem = float(excess @ excess)
        sealed = loop == 0 and v_mem > 0
        ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], hx)
        ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], hy)
        if sealed:
            wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
            reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    operator += diags(reaction.ravel(order='F'), format='csc')
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
0 x constant-null residual 3.552713678800501e-15 wrap-face residual 0.0
0 y constant-null residual 3.552713678800501e-15 wrap-face residual 0.0
0 reference KKT gap 3.761793598879015e-07 certificate value deviation 3.77755897298937e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f006df1576de80f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7f006df1 (directory).
```

### Turn 10 — `submit_completed_task` (`14875 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python validation. The audit independently confirmed both periodic wrap coefficients and the public certificate margin. Sparse-bank, EMI, loop-threshold, and budget-limit tests found no failure that warranted another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_03_solver.py': 'FILE:8122'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public validator and all 220 stress cases passed. The independently recomputed public KKT gap was 3.76e-7, below the 1e-6 tolerance.`

### Turn 11 — `execute_in_remote_docker` (`38017 ms` LLM)
- **Rationale**: Test two previously uncovered slices using the public measurements: simultaneous archive fallback for all four banks and a spike with exactly four clean points remaining. If the reference certificate fails, the slice identifies whether to revisit historical calibration or spike rejection.
- **Alternatives Weighed**: `Repeat the stencil and schema checks` (rejected: Both passed on the unchanged artifact; neither tests whether archive fallback or EMI rejection can invalidate the certificate against the original measurements.)

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
row = pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))[0]
s, g = row['study'], row['gt']
q = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}; banks = q['bank_id']
A, y = g['S_tilde'], g['r_y']; ridge, budget, cap = (s[k] for k in ('source_ridge','total_budget','u_box'))
worst = {}; failures = []
def audit(label, mask, raw):
    study = dict(s, sensor_calibration_sweeps={k:(raw if k == 'raw_readout' else v)[mask] for k,v in q.items()})
    out = m.solve_beta_m97_c02_batch_03(study); w = out['weights']; cert = out['kkt_certificate']
    r = A@w-y; grad = A.T@r+ridge*w
    obj = 0.5*(r@r+ridge*(w@w)); nu = cert['nu_sum']
    lb = obj-grad@w+nu*budget+cap*np.minimum(grad-nu,0).sum()
    metric = np.array([obj-lb, max(abs(obj-cert['primal_objective']),abs(lb-cert['dual_lower_bound'])), np.max(abs(w-g['w_opt']))])
    worst[label] = np.maximum(worst.get(label, np.zeros(3)), metric)
    if max(metric[:2]) >= 1e-6: failures.append((label, metric.tolist()))
raw = q['raw_readout']
audit('all_banks_missing', np.zeros(banks.size, bool), raw)
for b in range(4):
    i = np.flatnonzero(banks == b); i = i[np.argsort(q['ref_signal'][i])]
    clean = i[[0,3,7,-1]]
    for j in np.setdiff1d(i,clean):
        keep = (banks != b) | np.isin(np.arange(banks.size), np.r_[clean,j])
        for spike in (-3.01,3.01,-30.,30.):
            changed = raw.copy(); changed[j] += spike
            audit('four_clean_plus_emi', keep, changed)
for label, vals in worst.items(): print(label, 'worst reference gap, certificate value error, weight error:', vals)
print('reference margin failures:', len(failures), failures[:3])
assert not failures
PY
```
- **Container Exit Code**: `0` (`1205 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
all_banks_missing worst reference gap, certificate value error, weight error: [5.18849571e-07 5.12944492e-07 1.09285964e-06]
four_clean_plus_emi worst reference gap, certificate value error, weight error: [5.38155747e-07 5.37902807e-07 3.39008598e-07]
reference margin failures: 0 []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f006df1576de80f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_7f006df1 (directory).
```

### Turn 12 — `submit_completed_task` (`15341 ms` LLM)
- **Rationale**: The verified artifact passed isolated Python validation. Simultaneous archive fallback and EMI with exactly four clean points both kept the reference gap and certificate value error below tolerance, so no further edit was warranted.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public validator passed. In the final calibration stress audit, the worst independently recomputed KKT gap was 5.38e-7, below the 1e-6 tolerance.`

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
