# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a04_batch_04__blind_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 14` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `231,908` (`223,525` / `93,440` / `8,383`) |
| **Total Episode Wall Time (s)** | `281.93s` |

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

### Turn 1 — `execute_in_remote_docker` (`13993 ms` LLM)
- **Rationale**: Inspect the contract and compact data summaries to identify applicable archived solvers and avoid fitting operators the measurements do not support.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py; for f in /tmp/hegel_continual_archive/{archive_manifest,pre_llm_gate_diagnostic}.json; do echo "$f"; wc -c "$f"; head -c 11000 "$f"; printf '\n'; done; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for k,v in s.items():
 for n,a in (v.items() if isinstance(v,dict) else [('',v)]):
  a=np.array(a); print(k,n,a.shape,'range',np.min(a),np.max(a),'first',a[:2])
print('times',s['eval_times'],'telemetry',s['telemetry_vec'],'summary',s['train']['summary'])
PY
```
- **Container Exit Code**: `0` (`414 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
415742,
    "public_cv_traj_mae": 0.380069
  },
  "dag_runtime_trace": null,
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2460.07
}
eval_times  (8,) range 0.25 3.0 first [0.25       0.64285714]
panel_a u (48, 4) range -1.2728363323595389 1.2725156075464146 first [[-1.14291032  1.05316519  0.6342555  -0.39637768]
 [ 0.9899207   1.0060293   0.76263922  0.43129634]]
panel_a v (48, 4) range -6.739128174250137 6.486106731768872 first [[-1.13845994  1.06918504  0.77402011 -0.33570211]
 [ 1.0030877   1.0149549   0.9112152   0.37451212]]
panel_b u (24, 4) range -1.2164403239384511 1.2487159844581965 first [[ 0.12972486 -0.53824609 -0.48871708 -0.49304151]
 [ 0.41386639 -0.79580877 -0.64065095  0.87961343]]
panel_b v (24, 4) range -1.2794443332635654 1.6317150891160246 first [[-0.28584934 -0.21897578  0.47942181 -0.12341956]
 [-0.21122428  0.99963969  0.76470077  0.01650578]]
panel_c u (24, 4) range -1.2148264831176894 1.2427686037240622 first [[ 0.11552766 -0.65133644  1.06443121 -0.13929231]
 [-1.07646673  0.30656862  0.2464646  -0.59591807]]
panel_c v1 (24, 4) range -1.5376900305478758 1.2069641275441767 first [[-0.0294816   0.03693657  0.35688304 -0.75941714]
 [ 0.43443878  0.00516072  0.59675927  0.39883872]]
panel_c v2 (24, 4) range -1.5708520987721053 1.412641423195929 first [[-1.5708521   0.57716577  0.03093686  0.01067249]
 [ 0.55959376 -1.05165292  0.89655308 -0.15042254]]
train x (72, 4) range -1.3960945057348222 1.3060529917278205 first [[ 0.60099781  0.98979477  0.32374958 -0.67515358]
 [-0.61945012  1.03607219  0.06749066  0.56416601]]
train y (72,) range -2.535613487289088 1.797967736352752 first [ 0.64825714 -0.17301134]
train traj (72, 8) range -3.030905591779976 3.9339512077218064 first [[-0.19124195 -0.36231749 -0.46886423 -0.53105028 -0.55643417 -0.59509054
  -0.59865486 -0.62282105]
 [-0.08117371 -0.14234478 -0.15632966 -0.14670616 -0.13434223 -0.11514748
  -0.10035931 -0.0857855 ]]
train summary (4,) range -0.2122390424809773 1.5736688855314385 first [0.45409888 0.80527411]
test x (36, 4) range -1.4454080043038167 1.2862666745433975 first [[-5.09330902e-01 -1.34205767e+00 -8.39075298e-01 -7.98501057e-01]
 [-1.19943560e-04  9.69332726e-01 -4.06639388e-01  4.26151598e-01]]
telemetry_vec  (4,) range 0.1 0.2 first [0.1 0.2]
times [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ] telemetry [0.1 0.2 0.2 0.2] summary [ 0.45409888  0.80527411 -0.21223904  1.57366889]
```

### Turn 2 — `execute_in_remote_docker` (`15199 ms` LLM)
- **Rationale**: The prior candidate missed both prediction metrics. Identify reusable operator fits and confirm the validation contract before choosing which structural hypotheses to test.

```bash
ls -lR /tmp/hegel_continual_archive | head -100; python3 - <<'PY'
import json
m=json.load(open('/tmp/hegel_continual_archive/archive_manifest.json')); print(type(m),len(m))
for a in m:
 print({k:v for k,v in a.items() if not isinstance(v,dict)})
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS reference-panel alignment' 2>&1 | head -30; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`362 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pickle, sys
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    st = pickle.load(fh)[0]["study"]
out = mod.solve_batch(copy.deepcopy(st))
n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        yh = np.asarray(out["y_hat"], dtype=float)
        sp = np.asarray(out["state_pred"], dtype=float)
        dv = np.asarray(out["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([
            float(np.mean(st["train"]["y"][:sp_idx])),
            float(np.std(st["train"]["y"][:sp_idx])),
            float(np.mean(st["train"]["traj"][:sp_idx, -1])),
            float(np.std(st["train"]["traj"][:sp_idx, -1])),
        ], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv = mod.solve_batch(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], dtype=float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], dtype=float) - vtr)))
    if ym > 0.038:
        errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038:
        errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
```

### Turn 3 — `execute_in_remote_docker` (`14962 ms` LLM)
- **Rationale**: The four small source files can reveal which panel transformations and temporal bases are worth testing separately on the public 52/20 split.
- **Alternatives Weighed**: `Reuse an archived solver unchanged` (rejected: The prior gate failed both outputs and did not identify which operators transfer to this study.)

```bash
for f in /tmp/hegel_continual_archive/*_solver.py; do printf '\n=== %s ===\n' "$f"; cat "$f"; done
```
- **Container Exit Code**: `0` (`275 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, (cols+1) % 4] = coupling*(x-1)
        step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
        x -= np.clip(step, -.4, .4)
    return x


def _operator(panel, x, branch):
    u, v = panel['u'], panel[branch]
    if branch == 'v1':
        def basis(a, p):
            return np.column_stack((_with_bias(a), a/(1+p[0]*np.abs(a)), np.sin(p[1]*a)))
        start, bounds = (.8, 4.5), ((.05, .1), (5., 9.))
    else:
        def basis(a, p):
            return np.column_stack((_with_bias(a), np.cos(p[0]*a), np.cos(p[1]*a)))
        start, bounds = (1.1, 3.9), ((.3, 1.5), (2., 8.))

    def residual(p):
        a = basis(u, p)
        return (a @ _fit(a, v) - v).ravel()

    p = least_squares(residual, start, bounds=bounds, gtol=1e-12).x
    return basis(x, p) @ _fit(basis(u, p), v)


def _design(study):
    raw = np.concatenate((study['train']['x'], study['test']['x']))
    x = _calibrate(study['panel_a'], raw)
    c = _operator(study['panel_c'], x, 'v1')
    d = _operator(study['panel_c'], x, 'v2')
    matched = (c*d).sum(axis=1)
    scalar = np.column_stack((_with_bias(c), d, matched))
    fast = np.column_stack((c, matched))
    slow = np.column_stack((d, c[:, 0]*d[:, 1]))
    return scalar, fast, slow, x


def _trajectory(fast, slow, t, targets):
    """Fit an exponential fast response and a rational slow response jointly."""
    n, width = len(targets), len(t)
    values = targets.ravel()

    def design(f, g, rates):
        rise = -np.expm1(-rates[0]*t)
        tail = t/(1 + rates[1]*t)
        return np.column_stack(((f[:, None, :]*rise[None, :, None]).reshape(len(f)*width, f.shape[1]),
                                (g[:, None, :]*tail[None, :, None]).reshape(len(g)*width, g.shape[1])))

    def residual(rates):
        a = design(fast[:n], slow[:n], rates)
        return a @ _fit(a, values) - values

    rates = least_squares(residual, [1.8, .45], bounds=([.05, .001], [6., 2.]), gtol=1e-10).x
    coeff = _fit(design(fast[:n], slow[:n], rates), values)
    return (design(fast[n:], slow[n:], rates) @ coeff).reshape(len(fast)-n, width)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, fast, slow, _ = _design(study)
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    state = _trajectory(fast, slow, study['eval_times'], train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
```

### Turn 4 — `execute_in_remote_docker` (`29868 ms` LLM)
- **Rationale**: Panel residuals will distinguish transferable operator bases from family changes; trajectory rank and temporal residuals will determine whether a two-mode readout is appropriate.

```bash
sed -n '1,115p' /tmp/hegel_continual_archive/*_01_solver.py; sed -n '1,115p' /tmp/hegel_continual_archive/*_02_solver.py; sed -n '1,20p' /tmp/hegel_continual_archive/*_03_solver.py; cat >/tmp/probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
p=s['panel_a'];u,v=p['u'],p['v'];base=np.ones(len(u));nx=np.roll(u,-1,axis=1)
keep=np.ones(len(u),bool)
for i in range(8):
 f=np.stack([np.ones_like(u),u,u**3,(u-1)*nx],axis=-1); h=np.column_stack([f[:,j]@fit(f[keep,j],v[keep,j]) for j in range(4)]);r=np.linalg.norm(v-h,axis=1);keep=r<max(.001,5*np.median(r))
print('cal cubic',keep.sum(),np.round(np.quantile(r[keep],[0,.5,1]),6));print('cal coeff',np.array([fit(f[keep,j],v[keep,j]) for j in range(4)]))
for tag in ['panel_b','panel_c']:
 u=s[tag]['u']; base=np.ones(len(u));
 for name,v in s[tag].items():
  if name=='u':continue
  def check(parts):
   f=np.column_stack(parts);return '%.5g'%np.sqrt(np.mean((f@fit(f,v)-v)**2))
  print(tag,name,[(k,check([base,u]+z)) for k,z in [('linear',[]),('cube',[u**3]),('square',[u**2]),('poly3',[u**2,u**3]),('full2',[[u[:,i]*u[:,j] for i,j in []],np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)])][1:]),('sin31',[np.sin(3.1*u),np.sin(6.2*u)]),('cos',[np.cos(1.1*u),np.cos(3.9*u)]),('ratSin',[u/(1+.8*abs(u)),np.sin(4.5*u)])]])
t=s['eval_times']; z=s['train']['traj']; print('svd',np.linalg.svd(z,full_matrices=False)[1]); print('svd centered',np.linalg.svd(z-z.mean(0),full_matrices=False)[1]);print('y-tr corr',np.corrcoef(np.c_[s['train']['y'],z].T)[0]);
for family in range(4):
 def tb(p):
  e=-np.expm1(-p[0]*t); b=[t/(1+p[1]*t),-np.expm1(-p[1]*t),np.sin(p[1]*t),t*np.exp(-p[1]*t)][family];return np.array([e,b]).T
 def res(p):
  a=tb(p);return (a@fit(a,z.T)-z.T).ravel()
 q=least_squares(res,[1.5,.5],bounds=([.01,.001],[10,5]));print('time',family,q.x,np.sqrt(np.mean(q.fun**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`472 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
hts[:width], h @ weights[width:]))
        f, j = _fit(amplitudes, values)
    return np.outer(g_test @ weights[:width], f) + np.outer(h_test @ weights[width:], j)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, g, h = _design(study)
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    state = _trajectory(g[:n], h[:n], train['traj'], g[n:], h[n:])
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
"""Calibrate the raw coordinates and fit the two panel-C readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _with_bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_basis(x):
    return np.stack((np.ones_like(x), x, x**3,
                     (x - 1) * np.roll(x, -1, axis=1)), axis=2)


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
cal cubic 45 [5.20e-05 1.89e-04 3.96e-04]
cal coeff [[ 9.17137141e-03  1.00398565e+00  1.42886166e-05  2.10551196e-06]
 [-1.44431626e-01  1.15243765e+00  2.16909395e-05  2.22797885e-07]
 [ 9.59197255e-02  1.06901887e+00  6.27322362e-05  4.34944521e-06]
 [ 4.43677827e-03  8.58130502e-01  1.81387438e-05 -4.67019800e-06]]
panel_b v [('linear', '9.0054e-05'), ('cube', '7.9969e-05'), ('square', '8.0352e-05'), ('poly3', '6.6854e-05'), ('full2', '6.2224e-05'), ('sin31', '6.7579e-05'), ('cos', '6.8436e-05'), ('ratSin', '6.5485e-05')]
panel_c v1 [('linear', '0.41869'), ('cube', '0.36785'), ('square', '0.30923'), ('poly3', '0.25523'), ('full2', '6.8426e-05'), ('sin31', '0.32546'), ('cos', '0.26716'), ('ratSin', '0.30117')]
panel_c v2 [('linear', '0.61338'), ('cube', '0.48863'), ('square', '0.14544'), ('poly3', '0.11921'), ('full2', '0.11431'), ('sin31', '0.35364'), ('cos', '0.01971'), ('ratSin', '0.38947')]
svd [28.44421438  2.04694728  0.0503262   0.04742875  0.04406328  0.03602907
  0.03577707  0.03305995]
svd centered [28.24173704  2.0001557   0.05000311  0.04660947  0.04403061  0.03599407
  0.03534547  0.03305992]
y-tr corr [ 1.         -0.15384464 -0.1347745  -0.12022257 -0.10697904 -0.0980955
 -0.08947002 -0.08385201 -0.0786197 ]
time 0 [1.82829708 0.41586961] 0.00435259126628398
time 1 [1.82458589 0.42220048] 0.004393432498337677
time 2 [1.75649208 0.38856722] 0.006055795939602422
time 3 [1.72731586 1.02789948] 0.00631409543910097
```

### Turn 5 — `execute_in_remote_docker` (`35144 ms` LLM)
- **Rationale**: Compare compact cosine and interaction grammars against panel measurements and the held-out scalar and trajectory targets. Report the temporal amplitudes separately to locate any remaining readout mismatch.
- **Alternatives Weighed**: `Retain cubic and cross-coordinate calibration` (rejected: Those coefficients are at the calibration noise scale on the 45 inlier rows.)

```bash
cat >/tmp/discover.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def bias(a): return np.c_[np.ones(len(a)),a]
def quad(a): return np.column_stack([a[:,i]*a[:,j] for i,j in cwr(range(a.shape[1]),2)])
def calibrate(p,x):
 u,v=p['u'],p['v']; k=np.ones(len(u),bool)
 for _ in range(8):
  a,b=u[k],v[k]; ac=a-a.mean(0); w=(ac*(b-b.mean(0))).sum(0)/(ac**2).sum(0); q=b.mean(0)-a.mean(0)*w; e=np.linalg.norm(u*w+q-v,axis=1); nk=e<max(.0006,5*np.median(e))
  if np.array_equal(k,nk): break
  k=nk
 return (x-q)/w
x=calibrate(s['panel_a'],s['train']['x']); u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
f1=lambda a:bias(np.c_[a,quad(a)])
c1=fit(f1(u),v1); print('v1 coeff',np.round(c1,4).T); c=f1(x)@c1
bs=[('cos common',lambda a,p:np.c_[bias(a),np.cos(p[0]*a)],[1.25]),('cos separate',lambda a,p:np.c_[bias(a),np.cos(p*a)],[1.25]*4),('cos2 common',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.cos(p[1]*a)],[1.1,3.9]),('cos harmonic',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.cos(2*p[0]*a)],[1.25]),('cos sin',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),np.sin(p[1]*a)],[1.25,3]),('cos quad',lambda a,p:np.c_[bias(a),np.cos(p[0]*a),quad(a)],[1.25])]
bp=s['panel_b']; b=bias(x)@fit(bias(bp['u']),bp['v']); T=np.c_[-np.expm1(-1.8283*t),t/(1+.41587*t)]; targets=np.c_[y,z,fit(T,z.T).T]
for name,f,p in bs:
 def residual(p):
  a=f(u,p);return (a@fit(a,v2)-v2).ravel()
 q=least_squares(residual,p,bounds=(.15,8),gtol=1e-12); d=f(x,q.x)@fit(f(u,q.x),v2); print(name,np.round(q.x,5),np.sqrt(np.mean(q.fun**2)))
 xy=[('cd',np.c_[c,d]),('cd-match',np.c_[c,d,(c*d).sum(1)]),('cd-pair',np.c_[c,d,c*d,c[:,0]*d[:,1]]),('c',c),('d',d),('cd-cross',np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(-1,16)]),('cd-full',np.c_[c,d,quad(np.c_[c,d])]),('bcd',np.c_[b,c,d,(c*d).sum(1),c[:,0]*d[:,1]]),('xcd',np.c_[x,c,d,(c*d).sum(1),c[:,0]*d[:,1]])]
 for nm,a in xy:
  A=bias(a); err=np.abs(A[52:]@fit(A[:52],targets[:52])-targets[52:]).mean(0);print(nm,np.round([err[0],err[1:9].mean(),err[-2],err[-1]],5),end='; ')
 print()
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/discover.py
```
- **Container Exit Code**: `0` (`499 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0000e-04]]
cos common [2.39997] 8.784631506939206e-05
cd [0.21328 0.14546 0.17088 0.07338]; cd-match [0.00366 0.06621 0.00908 0.07408]; cd-pair [0.00425 0.00439 0.00904 0.00844]; c [0.67812 0.7748  0.1686  0.90342]; d [0.40653 0.3192  0.36047 0.09659]; cd-cross [0.0063  0.00692 0.01101 0.0108 ]; cd-full [0.024   0.02123 0.03807 0.03927]; bcd [0.00316 0.00465 0.00992 0.00928]; xcd [0.00316 0.00465 0.00992 0.00928]; 
cos separate [2.39984 2.39998 2.40015 2.40001] 8.603050237022439e-05
cd [0.21328 0.14546 0.17088 0.07339]; cd-match [0.00366 0.06621 0.00908 0.07408]; cd-pair [0.00426 0.00439 0.00904 0.00843]; c [0.67812 0.7748  0.1686  0.90342]; d [0.40653 0.3192  0.36047 0.0966 ]; cd-cross [0.00629 0.00692 0.01101 0.01079]; cd-full [0.02399 0.02121 0.03807 0.03919]; bcd [0.00316 0.00464 0.00992 0.00928]; xcd [0.00316 0.00464 0.00992 0.00928]; 
cos2 common [2.39978 4.83372] 7.140919437563701e-05
cd [0.21328 0.1455  0.17088 0.07342]; cd-match [0.00366 0.06625 0.00908 0.07413]; cd-pair [0.00426 0.00439 0.00905 0.00848]; c [0.67812 0.7748  0.1686  0.90342]; d [0.40653 0.31923 0.36049 0.09663]; cd-cross [0.00627 0.00693 0.01101 0.01081]; cd-full [0.02396 0.02119 0.03808 0.03917]; bcd [0.00314 0.00464 0.00993 0.00932]; xcd [0.00314 0.00464 0.00993 0.00932]; 
cos harmonic [1.20123] 7.282511835175129e-05
cd [0.21328 0.14549 0.17089 0.0734 ]; cd-match [0.00364 0.06624 0.00908 0.07411]; cd-pair [0.00425 0.00439 0.00904 0.00844]; c [0.67812 0.7748  0.1686  0.90342]; d [0.4065  0.31922 0.36049 0.09659]; cd-cross [0.00625 0.00691 0.01101 0.01077]; cd-full [0.02403 0.02119 0.03804 0.03911]; bcd [0.00314 0.00464 0.00993 0.00929]; xcd [0.00314 0.00464 0.00993 0.00929]; 
cos sin [2.40001 2.895  ] 6.816678409845551e-05
cd [0.2133  0.14545 0.17088 0.07337]; cd-match [0.00366 0.06623 0.0091  0.07411]; cd-pair [0.00429 0.0044  0.00905 0.0084 ]; c [0.67812 0.7748  0.1686  0.90342]; d [0.40652 0.3192  0.36048 0.09662]; cd-cross [0.00622 0.00691 0.01101 0.01081]; cd-full [0.02389 0.02136 0.03808 0.03956]; bcd [0.00313 0.00464 0.00993 0.00924]; xcd [0.00313 0.00464 0.00993 0.00924]; 
cos quad [2.39973] 6.0209944584783776e-05
cd [0.2133  0.14548 0.17088 0.07335]; cd-match [0.00367 0.06616 0.00909 0.07404]; cd-pair [0.00429 0.0044  0.00906 0.00849]; c [0.67812 0.7748  0.1686  0.90342]; d [0.40657 0.31911 0.36047 0.09647]; cd-cross [0.0062  0.00692 0.01103 0.01087]; cd-full [0.02393 0.02124 0.03815 0.03942]; bcd [0.00313 0.00464 0.00993 0.0093 ]; xcd [0.00313 0.00464 0.00993 0.0093 ];
```

### Turn 6 — `execute_in_remote_docker` (`52834 ms` LLM)
- **Rationale**: Fit the temporal rates using only each split's training trajectories. Check the public gate and coordinate-tail holdouts to determine whether the compact readout generalizes beyond the split used to discover it.
- **Alternatives Weighed**: `Fit every quadratic term in the two branch outputs` (rejected: The larger basis raised exploratory held-out MAEs to roughly 0.024 for scalar and 0.021 for trajectory.)
- **ATMS NOGOOD DECLARED**: Refuted `Additive panel-C branch terms suffice for both readouts.` | Witness: `On the 52/20 split, additive scalar and trajectory MAEs were 0.21328 and 0.14546. Adding the matched interaction cut scalar MAE to 0.00366, but trajectory MAE remained 0.06621 until the branch cross terms were allowed.` | Blanket Diagnosis: `The panel operators are identified near measurement noise, but their additive span omits the bilinear products needed downstream.` | Sublated Pivot: `Retain the measured branches, add the matched product to the scalar and fast trajectory readouts, and give the slow trajectory readout its own cross product.`

```bash
cat >/holdout_task/batch_solver.py <<'PY'
"""Recover the calibrated panel-C branches and their scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        a, b = u[keep], v[keep]
        center = a - a.mean(axis=0)
        scale = (center * (b - b.mean(axis=0))).sum(axis=0) / (center**2).sum(axis=0)
        offset = b.mean(axis=0) - a.mean(axis=0) * scale
        error = np.linalg.norm(u * scale + offset - v, axis=1)
        revised = error < max(0.0006, 5 * np.median(error))
        if np.array_equal(keep, revised):
            break
        keep = revised
    return (raw - offset) / scale


def _bilinear(x):
    a, b, c, d = x.T
    return _bias(np.column_stack((a*b, a*d, b*c, c*d)))


def _cosine(x, frequency):
    return _bias(np.column_stack((x, np.cos(frequency * x))))


def _operators(panel, x):
    u = panel['u']
    c = _bilinear(x) @ _fit(_bilinear(u), panel['v1'])
    v = panel['v2']

    def residual(p):
        a = _cosine(u, p[0])
        return (a @ _fit(a, v) - v).ravel()

    frequency = least_squares(residual, [1.5], bounds=(0.15, 8), gtol=1e-12).x[0]
    d = _cosine(x, frequency) @ _fit(_cosine(u, frequency), v)
    return c, d


def _temporal_design(g, h, t, rates):
    rise = -np.expm1(-rates[0] * t)
    tail = t / (1 + rates[1] * t)
    return np.concatenate((g[:, None, :] * rise[None, :, None],
                           h[:, None, :] * tail[None, :, None]), axis=2)


def _fit_trajectory(g, h, t, target):
    width = g.shape[1] + h.shape[1]
    n = len(target)
    values = target.ravel()

    def matrix(rates):
        return _temporal_design(g[:n], h[:n], t, rates).reshape(-1, width)

    def residual(rates):
        a = matrix(rates)
        return a @ _fit(a, values) - values

    rates = least_squares(residual, [1.8, 0.45], bounds=([0.05, 0.001], [6, 2]), gtol=1e-10).x
    return rates, _fit(matrix(rates), values)


def _features(study, raw):
    x = _calibrate(study['panel_a'], raw)
    c, d = _operators(study['panel_c'], x)
    matched = np.sum(c*d, axis=1)
    cross = c[:, 0]*d[:, 1]
    scalar = _bias(np.column_stack((c, d, matched)))
    fast = _bias(np.column_stack((c, matched)))
    slow = _bias(np.column_stack((d, cross)))
    return scalar, fast, slow


def solve_batch(study: dict) -> dict:
    train = study['train']
    t = study['eval_times']
    n = len(train['x'])
    raw = np.concatenate((train['x'], study['test']['x']))
    scalar, fast, slow = _features(study, raw)
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    rates, weights = _fit_trajectory(fast, slow, t, train['traj'])
    state = _temporal_design(fast[n:], slow[n:], t, rates) @ weights
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
PY
cat >/tmp/stress_batch.py <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj')); t=s['eval_times']; f,g,h=m._features(s,x)
i=np.arange(len(x)); rng=np.random.default_rng(835)
splits=[('public',i[52:])]+[(f'random{j}',rng.permutation(i)[:16]) for j in range(12)]
x_cal=m._calibrate(s['panel_a'],x)
splits.append(('large_norm',np.argsort(np.linalg.norm(x_cal,axis=1))[-16:]))
for j in range(4):
 order=np.argsort(x_cal[:,j]); splits.extend([(f'x{j}_low',order[:16]),(f'x{j}_high',order[-16:])])
metrics=[]
for name,b in splits:
 a=np.setdiff1d(i,b); local={**s,'train':{'x':x[a],'y':y[a],'traj':z[a]},'test':{'x':x[b]}}
 out=m.solve_batch(local); ey=np.abs(out['y_hat']-y[b]); ez=np.abs(out['state_pred']-z[b]); vals=np.r_[ey.mean(),ez.mean(),ez.mean(0).max()]; metrics.append(vals); print(name,*np.round(vals,6))
print('worst',np.max(metrics,axis=0)); print('scalar weights',np.round(m._fit(f,y),5)); rates,w=m._fit_trajectory(g,h,t,z); print('trajectory rates',rates,'weights',np.round(w,5))
PY
export OPENBLAS_NUM_THREADS=1
python3 /holdout_task/validate_public_schema.py && python3 -I /tmp/stress_batch.py
```
- **Container Exit Code**: `0` (`850 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public 0.003636 0.004066 0.005951
random0 0.004873 0.004109 0.004988
random1 0.003962 0.003656 0.004451
random2 0.004828 0.003989 0.004835
random3 0.00404 0.004136 0.004495
random4 0.006027 0.003978 0.004573
random5 0.005179 0.00394 0.0044
random6 0.006242 0.003682 0.00521
random7 0.005035 0.004055 0.004819
random8 0.005096 0.004289 0.005288
random9 0.006437 0.004047 0.004963
random10 0.004345 0.004376 0.005421
random11 0.005201 0.00431 0.005462
large_norm 0.005934 0.004503 0.005627
x0_low 0.005629 0.003766 0.005201
x0_high 0.00724 0.004439 0.005411
x1_low 0.006465 0.00384 0.005197
x1_high 0.006712 0.00395 0.004513
x2_low 0.005575 0.004158 0.005251
x2_high 0.006065 0.004364 0.005438
x3_low 0.005375 0.004351 0.005559
x3_high 0.008083 0.004147 0.005504
worst [0.00808318 0.00450316 0.00595111]
scalar weights [-4.2000e-04 -4.9228e-01  4.1569e-01 -6.1935e-01 -6.2668e-01 -5.9056e-01
 -5.8417e-01  3.9798e-01 -4.6444e-01  4.2028e-01]
trajectory rates [1.82134265 0.40144403] weights [-7.7000e-04  5.5710e-01  4.7480e-01  5.4428e-01  5.4047e-01  3.4968e-01
  6.7000e-04  8.4729e-01 -8.2101e-01  7.5213e-01  8.2317e-01 -2.5183e-01]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9b679e160f65b50 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9b679e1 (directory).
```

### Turn 7 — `submit_completed_task` (`13271 ms` LLM)
- **Rationale**: The verified artifact passes the public contract, and the isolated Python stress run confirms that the compact interaction model stays below tolerance on random and coordinate-tail holdouts.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:3065'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed. Held-out MAE was 0.00364 for the scalar output and 0.00407 for trajectories, both below 0.038. Across 22 validation splits, worst MAEs were 0.00808 and 0.00450, respectively.`

### Turn 8 — `execute_in_remote_docker` (`36816 ms` LLM)
- **Rationale**: The audit checks whether omitted auxiliary terms, cross-branch products, or rational attenuation improve both outputs on measured tail slices. Pairwise linear sums and differences are already in the branch span; the full cross-product audit covers their bilinear combinations. Keep the verified solver unless an alternative fixes a measurable error.
- **Alternatives Weighed**: `Test relay modes, drift operators, or within-batch classification` (rejected: The study supplies continuous regression targets and no relay modes or batch labels. These checks would not distinguish candidate readouts.)

```bash
cat >/tmp/audit_readout.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
x = m._calibrate(s['panel_a'], tr['x'])
c, d = m._operators(s['panel_c'], x)
f, g, h = m._features(s, tr['x'])
p = s['panel_b']
b = m._bias(x) @ m._fit(m._bias(p['u']), p['v'])
cd = (c[:, :, None] * d[:, None, :]).reshape(len(x), -1)
i = np.arange(len(x))
splits = [i[52:], np.argsort(np.linalg.norm(x, axis=1))[-16:]]
for j in range(4):
    order = np.argsort(x[:, j]); splits.extend((order[:16], order[-16:]))
metrics = {k: [] for k in ('current', 'affine_auxiliary', 'full_cross', 'rational')}
strengths = []
for val in splits:
    fit = np.setdiff1d(i, val)
    rates, _ = m._fit_trajectory(g[fit], h[fit], s['eval_times'], tr['traj'][fit])
    design = m._temporal_design(g, h, s['eval_times'], rates)
    basis = m._temporal_design(np.ones((len(x), 1)), np.ones((len(x), 1)), s['eval_times'], rates)
    z = tr['traj']; y = tr['y']
    def errors(a, t, gain):
        a = a * gain[:, None]; t = t * gain[:, None, None]
        yp = a @ m._fit(a[fit], y[fit]); zp = t @ m._fit(t[fit].reshape(-1, t.shape[-1]), z[fit].ravel())
        return (yp-y), (zp-z)
    cross_time = np.concatenate((design, basis[:, :, :1]*cd[:, None, :], basis[:, :, 1:]*cd[:, None, :]), axis=-1)
    aux_time = np.concatenate((design, basis[:, :, :1]*b[:, None, :], basis[:, :, 1:]*b[:, None, :]), axis=-1)
    q = np.mean(x**2, axis=1)
    gain = lambda a: 1 / (1 + a*q)
    def objective(a):
        ey, ez = errors(f, design, gain(a))
        return np.mean(ey[fit]**2) + np.mean(ez[fit]**2)
    opt = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-8})
    a = opt.x if opt.fun < objective(0) else 0.
    strengths.append(a)
    for name, fa, ta, gainval in [('current',f,design,gain(0)),('affine_auxiliary',np.c_[f,b],aux_time,gain(0)),('full_cross',np.c_[f,cd],cross_time,gain(0)),('rational',f,design,gain(a))]:
        ey, ez = errors(fa,ta,gainval)
        metrics[name].append([np.abs(ey[val]).mean(), np.abs(ez[val]).mean()])
for name, scores in metrics.items():
    print(name, 'public', np.round(scores[0],6), 'worst',np.round(np.max(scores,axis=0),6))
print('rational strength range', min(strengths), max(strengths))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /tmp/audit_readout.py
```
- **Container Exit Code**: `0` (`536 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
current public [0.003636 0.004066] worst [0.008083 0.004503]
affine_auxiliary public [0.003141 0.003995] worst [0.014184 0.004889]
full_cross public [0.006299 0.005261] worst [0.014532 0.006519]
rational public [0.003636 0.004066] worst [0.008083 0.004503]
rational strength range 0.0 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9b679e160f65b50 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9b679e1 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`27529 ms` LLM)
- **Rationale**: The schema check establishes delivery, but not which interactions matter. A paired ablation on measured tail slices, with temporal rates refit for each reduced model, tests whether the improvement persists beyond the discovery split. If removing a term does not hurt across these slices, narrow the structural claim.
- **Alternatives Weighed**: `Expand the readout with auxiliary, full-cross, or rational terms` (rejected: The completed audit found worse tail errors for the expanded bases and fitted zero rational attenuation on every audited split.)

```bash
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
x, y, z = (s['train'][k] for k in ('x', 'y', 'traj'))
t = s['eval_times']
xc = m._calibrate(s['panel_a'], x)
f, g, h = m._features(s, x)
i = np.arange(len(x))
splits = [('public', i[52:]), ('large_norm', np.argsort(np.linalg.norm(xc, axis=1))[-16:])]
for j in range(x.shape[1]):
    order = np.argsort(xc[:, j])
    splits.extend([(f'x{j}_low', order[:16]), (f'x{j}_high', order[-16:])])
scores = []
for name, b in splits:
    a = np.setdiff1d(i, b)
    out = m.solve_batch({**s, 'train': {'x': x[a], 'y': y[a], 'traj': z[a]}, 'test': {'x': x[b]}})
    err = [np.mean(abs(out['y_hat']-y[b])), np.mean(abs(out['state_pred']-z[b]))]
    scalar = f[:, :-1]
    yp = scalar[b] @ m._fit(scalar[a], y[a])
    err.append(np.mean(abs(yp-y[b])))
    for gg, hh in [(g[:, :-1], h[:, :-1]), (g, h[:, :-1])]:
        rates, w = m._fit_trajectory(gg[a], hh[a], t, z[a])
        pred = m._temporal_design(gg[b], hh[b], t, rates) @ w
        err.append(np.mean(abs(pred-z[b])))
    scores.append(err)
    print(name, np.round(err, 6))
scores = np.asarray(scores)
print('columns: solver_y, solver_traj, additive_y, additive_traj, no_slow_cross_traj')
print('worst MAEs:', np.round(scores.max(0), 6))
print('minimum paired MAE increases:', np.round((scores[:, 2:] - scores[:, [0, 1, 1]]).min(0), 6))
assert scores[:, :2].max() <= .038
PY
```
- **Container Exit Code**: `0` (`653 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public [0.003636 0.004066 0.213293 0.143822 0.068238]
large_norm [0.005934 0.004503 0.490791 0.267562 0.178618]
x0_low [0.005629 0.003766 0.258777 0.165014 0.069027]
x0_high [0.00724  0.004439 0.382471 0.223732 0.088724]
x1_low [0.006465 0.00384  0.207423 0.156309 0.063857]
x1_high [0.006712 0.00395  0.249596 0.15432  0.06179 ]
x2_low [0.005575 0.004158 0.169818 0.107669 0.047628]
x2_high [0.006065 0.004364 0.323402 0.254555 0.115683]
x3_low [0.005375 0.004351 0.263862 0.15173  0.076512]
x3_high [0.008083 0.004147 0.488057 0.310267 0.122461]
columns: solver_y, solver_traj, additive_y, additive_traj, no_slow_cross_traj
worst MAEs: [0.008083 0.004503 0.490791 0.310267 0.178618]
minimum paired MAE increases: [0.164243 0.103511 0.04347 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9b679e160f65b50 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9b679e1 (directory).
```

### Turn 10 — `submit_completed_task` (`15791 ms` LLM)
- **Rationale**: The unchanged artifact passed the public validator and isolated Python stress tests. Refitting the reduced models confirmed that removing the matched scalar interaction or the slow trajectory cross term increased held-out error on every audited split.
- **Alternatives Weighed**: `Add auxiliary-panel, full cross-product, or rational terms` (rejected: The auxiliary and full cross-product expansions worsened tail errors, and the fitted rational correction was zero on every audited split.)
- **Submitted Final Answer / Summary**: `Saved and verified `/holdout_task/batch_solver.py`. The public schema check passed. On the required 52/20 split, MAE was 0.00364 for scalar predictions and 0.00407 for trajectories, both below 0.038. Across 22 validation splits, the worst MAEs were 0.00808 and 0.00450, respectively.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel_M10_1_UnknownCut_DAG_Discovery",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
