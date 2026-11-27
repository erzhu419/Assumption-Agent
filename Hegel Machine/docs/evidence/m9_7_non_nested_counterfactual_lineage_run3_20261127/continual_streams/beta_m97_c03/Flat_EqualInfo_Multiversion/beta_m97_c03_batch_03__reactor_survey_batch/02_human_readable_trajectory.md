# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c03_batch_03__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c03_step_03` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `256,051` (`248,809` / `65,408` / `7,242`) |
| **Total Episode Wall Time (s)** | `271.56s` |

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

### Turn 1 — `execute_in_remote_docker` (`13274 ms` LLM)
- **Rationale**: Inspect the public contract and archived function boundaries to identify which operator conventions carry forward and where periodic closure needs to change.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; grep -nE '^def |^import |^from ' /tmp/hegel_continual_archive/*solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`315 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_c03_batch_01_solver.py:88:def _forward(study):
/tmp/hegel_continual_archive/beta_m97_c03_batch_01_solver.py:117:def _box_simplex(H, c, cap, budget):
/tmp/hegel_continual_archive/beta_m97_c03_batch_01_solver.py:175:def solve_beta_m97_c03_batch_01(study: dict) -> dict:
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:2:import json
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:3:from pathlib import Path
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:5:import numpy as np
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:6:from scipy.sparse import diags, eye, kron
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:7:from scipy.sparse.linalg import splu
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:13:def _archived_calibration(bank):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:25:def _line_fit(x, y):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:29:def _calibrate(study):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:73:def _axis(kappa, velocity, robin, length):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:88:def _forward(study):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:117:def _box_simplex(H, c, cap, budget):
/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py:175:def solve_beta_m97_c03_batch_01(study: dict) -> dict:
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:2:import json
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:3:from pathlib import Path
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:5:import numpy as np
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:6:from scipy.sparse import diags, eye, kron
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:7:from scipy.sparse.linalg import splu
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:13:def _archived_calibration(bank):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:37:def _line_fit(x, y):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:41:def _calibrate(study):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:85:def _axis(kappa, velocity, robin, length):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:100:def _forward(study):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:117:def _box_simplex(H, c, cap, budget):
/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py:175:def solve_beta_m97_c03_s00(study: dict) -> dict:
No matching skill cards found.
```

### Turn 2 — `execute_in_remote_docker` (`16968 ms` LLM)
- **Rationale**: Read the archived stencil, actual array layout, calibration history format, and verifier conventions before adapting the solver. A mismatch would change the operator assembly, calibration fallback, or certificate sign.

```bash
cat /tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py; sed -n '38,130p' /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle, numpy as np, json
with open('/holdout_task/public_beta_m97_c03_batch_03.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,v.ravel()[:5])
 elif isinstance(v,dict): print(k,{a:(np.shape(b),str(b)[:140]) for a,b in v.items()})
 else: print(k,type(v).__name__,str(v)[:900])
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history header',str(h)[:1350])
PY
```
- **Container Exit Code**: `1` (`372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
['ny']), order='F'),
            'kkt_certificate': certificate}

    w_feas = np.clip(w, 0.0, u_b)
    res = S @ w_feas - ry
    j_true = 0.5 * float(res @ res) + 0.5 * lw * float(w_feas @ w_feas)
    g = S.T @ res + lw * w_feas
    # Unconditional convex supporting-hyperplane + box-Lagrange lower bound valid for ANY `(w_feas, nu)`:
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
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c03_batch_03_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c03_batch_03.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c03_batch_03(st)
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
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 3 — `execute_in_remote_docker` (`20062 ms` LLM)
- **Rationale**: The validator identifies the pickle as a list of cases. Inspect the study arrays, bank coverage, and archived stencil to catch axis-order or calibration-fallback mismatches before implementing the periodic branch.

```bash
sed -n '38,175p' /tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
items=pickle.load(open('/holdout_task/public_beta_m97_c03_batch_03.pkl','rb')); print('cases:',len(items))
d=items[0]['study']; print('item and gt keys:',list(items[0]),list(items[0]['gt']))
for k,v in d.items():
 if isinstance(v,dict): print(k,{a:np.asarray(b).tolist() if np.size(b)<100 else np.shape(b) for a,b in v.items()})
 else:
  a=np.asarray(v); print(k,a.shape,a.dtype,a.ravel()[:5])
for i,item in enumerate(items):
 s=item['study']; b=np.asarray(s['sensor_calibration_sweeps']['bank_id'],int); print('case',i,'loop',s.get('recirculation_loop_mode'),'bank counts',np.bincount(b,minlength=4),'gt shapes',{k:np.shape(v) for k,v in item['gt'].items()})
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; h=json.load(open(p)); print('history',type(h).__name__,len(h),str(h[-1])[:600] if isinstance(h,list) else list(h))
PY
```
- **Container Exit Code**: `0` (`366 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.256419, 1.842579, -1.490331, 1.224235, 0.054897, 0.807965, 1.166814, 1.412796, -0.008659, -0.926917, 1.725776, -1.403503, 0.883847, 1.197932, 0.834109, 2.047369, 1.473478, 2.082333, 1.136001, 0.729196, -0.93955, -1.041375, -0.822857, -1.237617, 0.822121, -0.80934, -1.475581, 2.03562, -0.237924, 1.189686, -0.610196, -0.442137, 0.923241, -1.362116], 'raw_readout': [-0.26362256619358315, 0.7773686566623506, -0.2530727285542995, 0.11369582961533867, -0.15916133460677379, 0.6603710577042242, 1.8939525305726894, 1.7102217090840761, -0.9834789969363162, 1.0040534519630255, 0.7018213004178022, 1.418421302272297, 0.03203917651666098, -1.0204796968344427, -1.140006852778825, 1.277211836731324, -1.3224581066125074, 0.7949031922716564, -0.11718026734157458, 0.47021277470569856, 0.7501148294669384, 0.9419810017644806, -0.16675413208102638, -0.8829952812349008, 1.9464157195404617, -1.7461335706986827, 0.952939440992923, 1.3235598999025409, 0.8942486134288148, 2.3258955179869827, 1.6487039773426657, 2.3671529644696605, 1.250481097010041, 0.7704511708818431, -1.1986690531730737, -1.3188223556065566, -0.5811998105698122, -0.9295980923067484, 0.8005817645961878, -0.5698454993589662, -1.1294880554242597, 1.8199208710562076, -0.08985625153335036, 1.109336388582672, -0.4025645886679745, -0.2613950441104426, 0.8855224783635902, -1.0341776065756827]}
Z_nuisance (144, 5) float64 [ 1.44769575 -0.01274111 -1.60876955 -1.12110408  0.50586486]
y_obs_raw (144,) float64 [1.33039537 0.3298372  1.33439099 2.70671655 3.91317101]
aux_manifold_telemetry (144, 3) float64 [ 0.030155  0.340821 -1.082168 -0.700223  1.042604]
nuisance_ridge () float64 [0.65]
source_ridge () float64 [0.04]
u_box () float64 [1.]
total_budget () float64 [2.6]
case 0 loop 0.85 bank counts [12 12 12 12] gt shapes {'S_tilde': (144, 12), 'r_y': (144,), 'w_opt': (12,), 'cert_opt': (), 'terminal_field_A': (18, 16)}
history list 3 {'episode_short_name': 'beta_m97_c03_batch_02', 'bank_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'ref_signal': [0.503978, -0.972172, 1.380469, 0.476666, 1.830948, 0.007321, 0.03793, 1.732305, -0.402397, 0.963124, 1.161219, -0.716898, 1.368098, 1.292142, 0.362033, -1.259075, 2.185431, -1.442889, -0.438504, -0.07154, 0.68655, 0.207293, 1.403433, 1.95928, 1.046374, 1.137845, -0.45338, 1.698665, 2.100953, 1.507679, 0.29002, -0.916485, 1.221373, 1.218839, 2.170608, -0.743832, -1.177921, 1.691
```

### Turn 4 — `execute_in_remote_docker` (`16584 ms` LLM)
- **Rationale**: Recover the forward assembly and active-set routine omitted by the truncated output. The assembly determines how to preserve shift stacking and sensor indexing while replacing the boundary stencil.

```bash
sed -n '88,175p' /tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py; python3 - <<'PY'
import pickle, numpy as np
p='/holdout_task/public_beta_m97_c03_batch_03.pkl'; o=pickle.load(open(p,'rb')); print('public cases:',len(o)); s=o[0]['study']; print('study keys:',list(s)); print('grid:',[(k,s[k]) for k in ('nx','ny','Lx','Ly') if k in s]); print('circulation range:',np.min(s['past_vortex_circulation_trace']),np.max(s['past_vortex_circulation_trace'])); print('reference objective:',o[0]['gt']['cert_opt'].get('primal_objective'))
PY
```
- **Container Exit Code**: `0` (`709 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.zeros(n), float(np.min(c))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w + c))

    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)
    tol = 1e-12 * max(1.0, np.max(np.abs(c)), cap * np.max(np.abs(H)))
    for _ in range(100 * n):
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            g = H @ w + c
            lower = np.flatnonzero(status < 0)
            upper = np.flatnonzero(status > 0)
            l = lower[np.argmin(g[lower])]
            u = upper[np.argmax(g[upper])]
            if g[l] >= g[u] - tol:
                return w, float((g[l] + g[u]) / 2)
            status[[l, u]] = 0
            continue

        fixed = np.flatnonzero(status)
        m = free.size
        K = np.zeros((m + 1, m + 1))
        K[:m, :m] = H[np.ix_(free, free)]
        K[:m, m] = K[m, :m] = -1.0
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed],
                    w[fixed].sum() - budget]
        sol = np.linalg.solve(K, rhs)
        target, nu = sol[:-1], float(sol[-1])
        outside = (target < 0) | (target > cap)
        if np.any(outside):
            f = free[outside]
            bounds = np.where(target[outside] < 0, 0.0, cap)
            direction = target - w[free]
            steps = (bounds - w[f]) / direction[outside]
            first = int(np.argmin(steps))
            w[free] += steps[first] * direction
            w[f[first]] = bounds[first]
            status[f[first]] = -1 if bounds[first] == 0 else 1
            continue

        w[free] = target
        reduced = H @ w + c - nu
        violations = status * reduced
        release = int(np.argmax(violations))
        if violations[release] <= tol:
            return w, nu
        status[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c03_batch_01(study: dict) -> dict:
public cases: 1
study keys: ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'vel_x', 'robin_x', 'reaction_x', 'kappa_y', 'vel_y', 'robin_y', 'reaction_y', 'V_reaction_2d', 'recirculation_loop_mode', 'past_vortex_circulation_trace', 'critical_vortex_circulation', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
grid: [('nx', 18), ('ny', 16), ('Lx', 1.5), ('Ly', 1.3)]
circulation range: 0.069568 0.335299
reference objective: 0.02210908714119178
```

### Turn 5 — `execute_in_remote_docker` (`34474 ms` LLM)
- **Rationale**: Preserve the archived calibration, nuisance projection, and active-set solve while adding periodic closure. Compare each numerical output against the public reference: a sensor-matrix mismatch points to the stencil or grid convention, while an isolated observation mismatch points to calibration.

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/beta_m97_c03_batch_02_solver.py')
s = p.read_text().replace('Reaction-transport tomography with vortex-memory wall sorption.', 'Calibrated reaction-transport inversion with periodic recirculation.').replace('from scipy.sparse import diags, eye, kron', 'from scipy.sparse import coo_matrix, diags, eye, kron').replace('def solve_beta_m97_c03_batch_01(', 'def solve_beta_m97_c03_batch_03(')
pos = s.index('def _forward(')
s = s[:pos] + '''def _periodic_axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    lower = -(kappa + kappa[im]) / (2 * h**2) - np.maximum(velocity, 0) / h
    upper = -(kappa + kappa[ip]) / (2 * h**2) - np.maximum(-velocity, 0) / h
    return coo_matrix((np.r_[lower, -lower - upper, upper],
                       (np.tile(i, 3), np.r_[im, i, ip])), shape=(n, n)).tocsc()


''' + s[pos:]
a = s.index("    ax, ay = [_axis(", s.index('def _forward('))
b = s.index('        lx, ly', a)
s = s[:a] + '''    loop = float(study['recirculation_loop_mode'])
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if loop > 0.50:
        ax, ay = [_periodic_axis(np.asarray(study['kappa_' + a]),
                                np.asarray(study['vel_' + a]), study['L' + a])
                  for a in ('x', 'y')]
        x_phase = np.linspace(0., 2 * np.pi, nx)[:, None]
        y_phase = np.linspace(0., 2 * np.pi, ny)[None, :]
        reaction = reaction + 0.75 * loop * np.cos(x_phase) * np.cos(y_phase)
    else:
        ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                        0. if sealed else study['robin_' + a], study['L' + a])
                  for a in ('x', 'y')]
    if sealed:
''' + s[b:]
Path('/holdout_task/beta_m97_c03_batch_03_solver.py').write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
from pathlib import Path
import importlib.util, pickle, numpy as np
root=Path('/holdout_task'); spec=importlib.util.spec_from_file_location('solver',root/'beta_m97_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns={}; exec((root/'validate_public_schema.py').read_text().split('\n_d =')[0],ns)
for item in pickle.load((root/'public_beta_m97_c03_batch_03.pkl').open('rb')):
 s,g=item['study'],item['gt']; y=m._calibrate(s); S,_=m._forward(s); Z=s['Z_nuisance']; T=np.column_stack((y,S)); T-=Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@T); out=m.solve_beta_m97_c03_batch_03(s)
 for name,a,b in [('r_y',T[:,0],g['r_y']),('S_tilde',T[:,1:],g['S_tilde']),('weights',out['weights'],g['w_opt']),('terminal',out['terminal_field'],g['terminal_field_A'])]: print(name,'max_abs',np.max(np.abs(a-b)),'relative_l2',np.linalg.norm(a-b)/np.linalg.norm(b))
 print('KKT against public reference', ns['verify_box_simplex_kkt_certificate'](g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']))
PY
```
- **Container Exit Code**: `0` (`752 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
r_y max_abs 1.2342505240070523e-07 relative_l2 7.213312061825235e-07
S_tilde max_abs 1.5265566588595902e-16 relative_l2 9.993114003404028e-16
weights max_abs 5.080723844141932e-07 relative_l2 8.881599545948852e-07
terminal max_abs 2.594308316089533e-08 relative_l2 2.625495777395669e-07
KKT against public reference {'valid': True, 'reason': 'ok', 'true_primal': 0.022109087141212645, 'verified_dual_lb': 0.022108684612550625, 'gap': 4.0252866202045157e-07}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f33a2551c1765974 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f33a2551 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37109 ms` LLM)
- **Rationale**: The public case passes but exercises only one mode and a well-populated calibration sweep. Stress threshold switching, sparse and spiked calibration, and saturated simplex budgets; any failure identifies the branch that needs correction.

```bash
python3 -I - <<'PY'
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import importlib.util, pickle, numpy as np
root=Path('/holdout_task'); archive=Path('/tmp/hegel_continual_archive')
def load(path):
 spec=importlib.util.spec_from_file_location(path.stem,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load(root/'beta_m97_c03_batch_03_solver.py'); prev=load(archive/'beta_m97_c03_batch_02_solver.py')
ns={}; exec((root/'validate_public_schema.py').read_text().split('\n_d =')[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
max_gap=0.; max_cal=0.; max_branch=0.; n=0
for item in pickle.load((root/'public_beta_m97_c03_batch_03.pkl').open('rb')):
 s,g=item['study'],item['gt']; y=m._calibrate(s); w=s['sensor_calibration_sweeps']; b=np.asarray(w['bank_id']); raw=np.asarray(w['raw_readout']); rb=np.asarray(s['row_bank_id']); ix=np.flatnonzero(b==0)
 for j in ix[[np.argmin(np.asarray(w['ref_signal'])[ix]),np.argmax(np.asarray(w['ref_signal'])[ix])]]:
  for spike in (-8.,8.):
   t=deepcopy(s); t['sensor_calibration_sweeps']['raw_readout']=raw.copy(); t['sensor_calibration_sweeps']['raw_readout'][j]+=spike
   with patch.object(m,'_archived_calibration',side_effect=AssertionError('history read despite enough clean points')):
    err=np.max(np.abs(m._calibrate(t)-y)); max_cal=max(max_cal,err); assert err<1e-6
 for bank in np.unique(b):
  mask=(b!=bank); mask[np.flatnonzero(b==bank)[0]]=True; t=deepcopy(s)
  t['sensor_calibration_sweeps']={k:np.asarray(v)[mask] for k,v in w.items()}
  gain,offset=m._archived_calibration(bank); expected=y.copy(); expected[rb==bank]=(s['y_obs_raw'][rb==bank]-offset)/gain
  err=np.max(np.abs(m._calibrate(t)-expected)); max_cal=max(max_cal,err); assert err<1e-10
 for loop, high in [(0.5,False),(0.,False),(0.,True),(np.nextafter(0.5,1.),True),(0.85,False)]:
  t=deepcopy(s); t['recirculation_loop_mode']=loop
  if high: t['past_vortex_circulation_trace']=np.asarray(t['past_vortex_circulation_trace'])+0.8
  S,F=m._forward(t)
  if loop<=0.5:
   oldS,oldF=prev._forward(t); err=max(np.max(np.abs(S-oldS)),np.max(np.abs(F-oldF))); max_branch=max(max_branch,err); assert err<1e-12
  else:
   for axis in 'xy':
    A=m._periodic_axis(t['kappa_'+axis],t['vel_'+axis],t['L'+axis]); assert A[0,-1]<0 and A[-1,0]<0; assert np.max(np.abs(A.sum(axis=1)))<1e-12
  Z=t['Z_nuisance']; dat=np.column_stack((m._calibrate(t),S)); dat-=Z@np.linalg.solve(Z.T@Z+t['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@dat); ry,St=dat[:,0],dat[:,1:]
  for budget in (0., 1e-8, t['u_box'], s['total_budget'], 6*t['u_box'], 11*t['u_box']+0.99*t['u_box'], 12*t['u_box']):
   t['total_budget']=budget; out=m.solve_beta_m97_c03_batch_03(t); c=out['kkt_certificate']; ww=out['weights']; assert np.array_equal(ww,c['weights']); assert abs(ww.sum()-budget)<1e-9; assert np.allclose((F@ww).reshape(t['nx'],t['ny'],order='F'),out['terminal_field']); rec=verify(St,ry,t['source_ridge'],t['u_box'],budget,c); assert rec['valid'],(loop,budget,rec); max_gap=max(max_gap,rec['gap']); n+=1
print(f'{n} mode/budget certificates; worst gap={max_gap:.3e}; worst calibration slice error={max_cal:.3e}; archived nonperiodic branch max diff={max_branch:.3e}')
PY
```
- **Container Exit Code**: `0` (`731 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
35 mode/budget certificates; worst gap=8.882e-16; worst calibration slice error=6.177e-08; archived nonperiodic branch max diff=0.000e+00

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f33a2551c1765974 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f33a2551 (directory).
```

### Turn 7 — `submit_completed_task` (`14171 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated-process validation. Stress tests covered the mode threshold, sparse and spiked calibration, and simplex budget endpoints; the nonperiodic branches matched the archived solver exactly.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_03_solver.py': 'FILE:9285'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema and KKT checks passed, as did all 35 mode and budget stress cases.`

### Turn 8 — `execute_in_remote_docker` (`30118 ms` LLM)
- **Rationale**: Hold out each real calibration point and check projection agreement separately by bank. A bank-local failure would trigger inspection of its affine fit or nuisance projection rather than an unrelated model change.
- **Alternatives Weighed**: `Balanced-accuracy or relay-state audit` (rejected: This is a continuous inverse problem with no class labels or relay state; both applicable boundary branches have already been stress-tested.); `Add nonlinear calibration or stochastic drift terms` (rejected: The specified model is affine calibration and deterministic reaction transport; the projected forward matrix already matches the reference to 1.53e-16.)

```bash
python3 -I - <<'PY'
from pathlib import Path
import importlib.util, pickle, numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'beta_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load((root/'public_beta_m97_c03_batch_03.pkl').open('rb')):
    s, gt = item['study'], item['gt']
    sw = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    b,x,y = (sw[k] for k in ('bank_id','ref_signal','raw_readout'))
    loo = np.empty_like(y, dtype=float)
    for j in range(len(y)):
        take = (b == b[j]) & (np.arange(len(b)) != j)
        gain,offset = m._line_fit(x[take],y[take]); loo[j] = y[j] - (gain*x[j]+offset)
    print('Worst held-out sweep error by bank:', {int(i):float(np.max(np.abs(loo[b==i]))) for i in np.unique(b)})
    S,F = m._forward(s); Z = np.asarray(s['Z_nuisance']); rb = np.asarray(s['row_bank_id']); cal = m._calibrate(s)
    T = np.column_stack([cal,S]); T -= Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T@T)
    w = m.solve_beta_m97_c03_batch_03(s)['weights']
    print('Worst observation-map error by bank:', {int(i):float(np.max(np.abs(T[rb==i,0]-gt['r_y'][rb==i]))) for i in np.unique(rb)})
    print('Projected forward-map max error:', np.max(np.abs(T[:,1:]-gt['S_tilde'])))
    print('Worst bank residual RMS (before/after nuisance projection):', [max(np.sqrt(np.mean(r[rb==i]**2)) for i in np.unique(rb)) for r in [cal-S@w,T[:,0]-T[:,1:]@w]])
    print('Terminal max error:', np.max(np.abs((F@w).reshape((s['nx'],s['ny']),order='F')-gt['terminal_field_A'])))
    assert np.max(np.abs(loo)) < 1e-5
    assert np.max(np.abs(T[:,0]-gt['r_y'])) < 1e-6 and np.max(np.abs(T[:,1:]-gt['S_tilde'])) < 1e-12
PY
```
- **Container Exit Code**: `0` (`1626 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst held-out sweep error by bank: {0: 1.3338129431494927e-07, 1: 2.1248822990038718e-07, 2: 2.03274786336749e-07, 3: 2.1884847045683387e-07}
Worst observation-map error by bank: {0: 1.2342505240070523e-07, 1: 3.8843867289362066e-08, 2: 1.0322044374788675e-07, 3: 8.559578201783324e-08}
Projected forward-map max error: 1.5265566588595902e-16
Worst bank residual RMS (before/after nuisance projection): [2.043546003615152, 0.012166218983419233]
Terminal max error: 2.594308316089533e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f33a2551c1765974 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f33a2551 (directory).
```

### Turn 9 — `submit_completed_task` (`14051 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process validation. The projected forward map matched the public reference within 1.53e-16, and the held-out calibration audit found no bank that needed a different model.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_m97_c03_batch_03_solver.py': 'FILE:9285'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema and KKT checks passed, as did all 35 mode and budget stress cases. The maximum public weight error was 5.08e-7.`

### Turn 10 — `execute_in_remote_docker` (`31297 ms` LLM)
- **Rationale**: Use subsets of the measured calibration sweeps to check whether clean sparse fits accidentally consult the archive or amplify rounding at narrow reference spans. Separately, perturb inputs that should not affect the periodic branch; any output change would identify unintended coupling.
- **Alternatives Weighed**: `Repeat the public reference and leave-one-out audits` (rejected: Those checks already passed on the unchanged artifact; they did not exercise clean two- and three-point sweeps.)

```bash
python3 -I - <<'PY'
import importlib.util, itertools, pickle
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
root=Path('/holdout_task'); p=root/'beta_m97_c03_batch_03_solver.py'
spec=importlib.util.spec_from_file_location('solver', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load((root/'public_beta_m97_c03_batch_03.pkl').open('rb')):
    s,g=item['study'],item['gt']; sw={k:np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}; b,x=sw['bank_id'],sw['ref_signal']; rb=np.asarray(s['row_bank_id']); full=m._calibrate(s)
    for n in (2,3):
        worst=(0.,None,None); count=0
        with patch.object(m,'_archived_calibration',side_effect=AssertionError('archive read on a clean sparse sweep')):
            for bank in np.unique(b):
                for ix in itertools.combinations(np.flatnonzero(b==bank),n):
                    span=float(np.ptp(x[list(ix)]))
                    if not span: continue
                    mask=b!=bank; mask[list(ix)]=True
                    out=m._calibrate(dict(s,sensor_calibration_sweeps={k:v[mask] for k,v in sw.items()})); e=float(np.max(np.abs(out[rb==bank]-full[rb==bank]))); count+=1
                    if e>worst[0]: worst=(e,int(bank),span)
        print(f'{n}-point clean sweeps: {count} slices; worst (calibrated observation delta, bank, reference span)={worst}')
    base=m.solve_beta_m97_c03_batch_03(s)
    altered=deepcopy(s); altered.pop('aux_manifold_telemetry')
    if s['recirculation_loop_mode']>0.5:
        altered['robin_x']=0.; altered['robin_y']=0.; altered['past_vortex_circulation_trace']=np.full_like(s['past_vortex_circulation_trace'],10.)
    out=m.solve_beta_m97_c03_batch_03(altered)
    for k in ('weights','terminal_field'): np.testing.assert_array_equal(base[k],out[k])
    print('Weights and terminal field unchanged with auxiliary telemetry removed and inactive wall inputs perturbed')
PY
```
- **Container Exit Code**: `0` (`1100 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2-point clean sweeps: 264 slices; worst (calibrated observation delta, bank, reference span)=(2.0847007260904604e-05, 0, 0.008307000000000009)
3-point clean sweeps: 880 slices; worst (calibrated observation delta, bank, reference span)=(6.0813363167433465e-06, 1, 0.24598199999999992)
Weights and terminal field unchanged with auxiliary telemetry removed and inactive wall inputs perturbed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f33a2551c1765974 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f33a2551 (directory).
```

### Turn 11 — `submit_completed_task` (`17488 ms` LLM)
- **Rationale**: The same artifact passed isolated-process verification. The latest audit confirmed that clean sparse sweeps do not read the archive and that unused telemetry and inactive wall inputs do not change periodic outputs.
- **Alternatives Weighed**: `Use archived calibration for narrow but clean two-point sweeps` (rejected: The specification requires fitting those sweeps from in-batch data; the sparse-sweep audit measured sensitivity but did not establish a contract failure.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_batch_03_solver.py`. The public schema and KKT checks passed, as did all 35 mode and budget stress cases. The maximum public weight error was 5.08e-7.`

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
