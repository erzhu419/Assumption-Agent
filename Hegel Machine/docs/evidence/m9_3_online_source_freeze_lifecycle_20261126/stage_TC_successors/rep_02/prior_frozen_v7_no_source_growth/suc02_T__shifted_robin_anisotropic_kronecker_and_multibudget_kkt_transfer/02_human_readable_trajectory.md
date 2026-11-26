# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Anisotropic Reversed-Advection Kronecker Sylvester Transport & Multi-Budget KKT Pareto Certification` |
| **Wave / Wormhole Family** | `m93_stage_T_transfer` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `212,138` (`203,453` / `58,624` / `8,685`) |
| **Total Episode Wall Time (s)** | `316.67s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`21935 ms` LLM)
- **Rationale**: Inspect the public schema and compact study summaries to establish stencil conventions, shapes, and certificate fields before implementing; schema mismatches would change the assembly and validation code.

```bash
ls -lah /holdout_task; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle
import numpy as np
import scipy
with open('/holdout_task/public_suc02_T_studies.pkl','rb') as f:
    data=pickle.load(f)
print('scipy',scipy.__version__,'root',type(data).__name__)
def brief(v,depth=0):
    if isinstance(v,dict) and depth<3:
        for k,x in list(v.items())[:40]:
            print('  '*depth,k,end=': ')
            brief(x,depth+1)
    elif isinstance(v,(list,tuple)):
        print(type(v).__name__,len(v))
        if v and depth<3: brief(v[0],depth+1)
    elif isinstance(v,np.ndarray):
        print(v.shape,v.dtype,v.ravel()[:6])
    else: print(repr(v)[:200])
brief(data)
PY
ls -ld /pretrain_task /tmp/hegel_wormhole_skills 2>/dev/null || :
```
- **Container Exit Code**: `0` (`636 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    certs = out.get("kkt_certificates")
    if not isinstance(certs, list) or len(certs) != B: errs.append("bad_certs_list")
    else:
        for bi, bg in enumerate(st["budget_grid"]):
            chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi])
            if not chk["valid"]: errs.append(f"kkt_b{bi}:{chk['reason']}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
scipy 1.13.0 root list
list 1
   study:      Lx: 2.2
     Ly: 1.9
     nx: 48
     ny: 36
     kappa_x: (48,) float64 [0.5937878  0.54113151 0.62243384 0.59735773 0.60769725 0.56408046]
     vel_x: (48,) float64 [-0.56102563 -0.57892246 -0.59635865 -0.61303584 -0.62866869 -0.64298972]
     robin_x: 0.75
     kappa_y: (36,) float64 [0.40923628 0.3810096  0.37890501 0.4096731  0.43267575 0.40080399]
     vel_y: (36,) float64 [0.36843761 0.37191853 0.3787746  0.38879751 0.40168272 0.41703871]
     robin_y: 2.05
     mu_shifts: (12,) float64 [0.08   0.1125 0.1582 0.2224 0.3128 0.4398]
     source_modes: (16, 48, 36) float64 [1.12563459e-13 1.85957134e-13 2.88401066e-13 4.19903486e-13
 5.73945562e-13 7.36479272e-13]
     sensor_ij: (36, 2) int64 [32  7 36 17 41  7]
     Z_nuisance: (432, 5) float64 [ 1.02996265 -0.64220332  0.12484201  1.3724282  -0.9718141   0.22758282]
     y_obs: (432,) float64 [ 0.22543437 -1.25089027  0.77746149 -0.12710204 -0.87374164  0.33568503]
     nuisance_ridge: 0.65
     source_ridge: 0.04
     u_box: 1.0
     budget_grid: (3,) float64 [1.4 2.6 3.8]
   gt:      Ax: (48, 48) float64 [ 297.69193024 -281.78862346    0.            0.            0.
    0.        ]
     By: (36, 36) float64 [ 182.95720958 -141.66925148    0.            0.            0.
    0.        ]
     S_tilde: (432, 16) float64 [0.01920258 0.00017545 0.00932164 0.00452536 0.00210249 0.00382289]
     r_y: (432,) float64 [-0.00080592 -0.00043301 -0.01047027  0.00709256  0.0057192   0.01881688]
     weights_by_budget: (3, 16) float64 [1.54591719e-17 1.27793061e-01 9.59144585e-02 4.78493717e-02
 1.64133922e-01 1.68565914e-17]
     certs_by_budget: list 3
     terminal_field_mid_budget: (48, 36) float64 [0.00052723 0.00068274 0.00085471 0.00103221 0.00121002 0.00140265]
drwxr-xr-x 2 root   root   4096 Nov 23 09:31 /pretrain_task
drwxr-xr-x 2 ubuntu ubuntu 4096 Nov 26 08:47 /tmp/hegel_wormhole_skills
```

### Turn 2 — `execute_in_remote_docker` (`33799 ms` LLM)
- **Rationale**: Compare the full axis matrices and both plausible ridge conventions to the supplied references. Any discrepancy changes the corresponding assembly formula before implementing the solver.
- **Alternatives Weighed**: `Assume a standard Robin stencil and ridge projection` (rejected: Different endpoint and profiling conventions can produce correct-looking fields while invalidating the certificate against the public reference.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; ls -lah /pretrain_task; sed -n '30,68p' /holdout_task/validate_public_schema.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
from scipy.linalg import eigh
item=pickle.load(open('/holdout_task/public_suc02_T_studies.pkl','rb'))[0]; s,g=item['study'],item['gt']
for d,key in [('x','Ax'),('y','By')]:
 k,v,r=(s[f'{a}_{d}'] for a in ('kappa','vel','robin')); n=len(k); h=s[f'L{d}']/n; f=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2; p=np.maximum(v,0)/h; q=np.maximum(-v,0)/h; bi=r/h/(1+r*h/(2*k[[0,-1]])); diag=np.r_[0,f]+np.r_[f,0]+p+q; diag[[0,-1]]+=bi; a=np.diag(diag)+np.diag(-f-p[1:],-1)+np.diag(-f-q[:-1],1)
 print(d,'stencil max error',np.max(abs(a-g[key])),'boundary diagonals',diag[[0,-1]],g[key].diagonal()[[0,-1]],'boundary residual', (a-g[key]).diagonal()[[0,-1]])
z=s['Z_nuisance']; y=s['y_obs']; lam=s['nuisance_ridge']; p=y-z@np.linalg.solve(z.T@z+lam*np.eye(z.shape[1]),z.T@y); q,d,_=np.linalg.svd(z,full_matrices=False); t=y-q@((1-np.sqrt(lam/(d*d+lam)))*(q.T@y)); print('profile residual max errors: residualizer',np.max(abs(p-g['r_y'])),'sqrt residualizer',np.max(abs(t-g['r_y']))); print('reference cert scalars', [{k:v for k,v in c.items() if np.ndim(v)==0} for c in g['certs_by_budget']])
PY
```
- **Container Exit Code**: `0` (`469 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "multibudget_kronecker_kkt_transfer_solver.py")
spec = importlib.util.spec_from_file_location("cand_suc02_T", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_suc02_T_studies.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_multibudget_kronecker_transfer(st)
M, nx, ny = st["source_modes"].shape
B = len(st["budget_grid"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
x stencil max error 0.0 boundary diagonals [297.69193024 156.52335361] [297.69193024 156.52335361] boundary residual [0. 0.]
y stencil max error 2.842170943040401e-14 boundary diagonals [182.95720958 183.28076386] [182.95720958 183.28076386] boundary residual [2.84217094e-14 2.84217094e-14]
profile residual max errors: residualizer 1.7763568394002505e-15 sqrt residualizer 0.1286710594787821
reference cert scalars [{'nu_sum': -0.009426901506605953, 'primal_objective': 0.03906431692465978, 'dual_lower_bound': 0.03906431692465975}, {'nu_sum': 0.009527983302115323, 'primal_objective': 0.03932836821873202, 'dual_lower_bound': 0.03932836821873199}, {'nu_sum': 0.02791499255736784, 'primal_objective': 0.06179415373442193, 'dual_lower_bound': 0.06179415373442188}]
```

### Turn 3 — `execute_in_remote_docker` (`61816 ms` LLM)
- **Rationale**: Implement the verified stencil and profiling conventions, then run the public certificate validator. A validation failure would distinguish a transport or profiling mismatch from a defect in the budget solve.
- **Alternatives Weighed**: `Factor a full 2D operator separately at each shift` (rejected: The signed tridiagonal axes are diagonally similar to symmetric matrices, so two eigendecompositions can be reused for every source and shift.)
- **ATMS NOGOOD DECLARED**: Refuted `The square-root ridge residualizer can represent this contract's profiled data.` | Witness: `Public r_y max error was 0.128671 for the square-root transform and 1.78e-15 for the direct residualizer.` | Blanket Diagnosis: `The benchmark transforms the design and observations with the ridge residualizer itself before forming the QP.` | Sublated Pivot: `Apply the verified direct transform jointly to all design columns and the observation vector.`

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -45
cat > /holdout_task/multibudget_kronecker_kkt_transfer_solver.py <<'PY'
"""Separable shifted transport and certified box-simplex budget optimization."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh_tridiagonal


def _axis_modes(length, kappa, velocity, robin):
    """Return eigenvalues and left/right transforms in the physical grid basis."""
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    faces = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    positive = np.maximum(v, 0.0) / h
    negative = np.maximum(-v, 0.0) / h
    diagonal = np.r_[0.0, faces] + np.r_[faces, 0.0] + positive + negative
    diagonal[0] += robin / h / (1 + robin * h / (2 * k[0]))
    diagonal[-1] += robin / h / (1 + robin * h / (2 * k[-1]))
    lower = faces + positive[1:]
    upper = faces + negative[:-1]
    # D^{-1} A D is symmetric when D[i+1]/D[i] = sqrt(lower[i]/upper[i]).
    log_d = np.r_[0.0, np.cumsum(0.5 * np.log(lower / upper))]
    d = np.exp(log_d - 0.5 * (log_d.max() + log_d.min()))
    eig, q = eigh_tridiagonal(diagonal, -np.sqrt(lower * upper))
    return eig, q.T / d, d[:, None] * q


def _responses(study):
    axes = [_axis_modes(study['L' + a], study['kappa_' + a],
                        study['vel_' + a], study['robin_' + a]) for a in ('x', 'y')]
    (ex, lx, rx), (ey, ly, ry) = axes
    sources = np.asarray(study['source_modes'], dtype=float)
    src_modes = lx @ sources @ ly.T
    eigen_sums = ex[:, None] + ey[None, :]
    si, sj = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = rx @ (src_modes / (eigen_sums + shift)) @ ry.T
        blocks.append(fields[:, si, sj].T)
    return np.concatenate(blocks, axis=0), fields


def _box_simplex(hessian, linear, cap, total):
    """Feasible active-set method: hit bounds, then release wrong-sign bounds."""
    n = linear.size
    if total == 0:
        return np.zeros(n), float(np.min(linear))
    if total == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(hessian @ w + linear))

    w = np.full(n, total / n)
    active = np.zeros(n, dtype=np.int8)  # -1: lower; +1: upper
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        grad = hessian @ w + linear
        if free.size == 0:
            lo = np.flatnonzero(active == -1)
            hi = np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j] <= grad[i] + 2e-13:
                return w, float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue

        fixed = np.flatnonzero(active != 0)
        rhs = linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed]
        factor = cho_factor(hessian[np.ix_(free, free)])
        x, s = cho_solve(factor, np.column_stack((-rhs, np.ones(free.size)))).T
        remaining = total - w[fixed].sum()
        nu = (remaining - x.sum()) / s.sum()
        candidate = x + nu * s
        candidate += (remaining - candidate.sum()) / free.size
        direction = candidate - w[free]
        crossing = np.full(free.size, np.inf)
        down = (candidate < 0) & (direction < 0)
        up = (candidate > cap) & (direction > 0)
        crossing[down] = -w[free[down]] / direction[down]
        crossing[up] = (cap - w[free[up]]) / direction[up]
        j = int(np.argmin(crossing))
        if crossing[j] < 1 and (candidate[j] < -2e-14 or candidate[j] > cap + 2e-14):
            w[free] = np.clip(w[free] + max(0.0, crossing[j]) * direction, 0, cap)
            active[free[j]] = 1 if up[j] else -1
            w[free[j]] = cap if up[j] else 0.0
            continue

        w[free] = np.clip(candidate, 0, cap)
        grad = hessian @ w + linear
        nu = float(np.mean(grad[free]))
        violation = active * (grad - nu)
        j = int(np.argmax(violation))
        if violation[j] <= 2e-13:
            return w, nu
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    stacked, terminal = _responses(study)
    nuisance = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((stacked, study['y_obs']))
    ridge = float(study['nuisance_ridge'])
    # The contract uses the ridge residualizer itself, not its square root.
    gram = nuisance.T @ nuisance + ridge * np.eye(nuisance.shape[1])
    profiled = values - nuisance @ cho_solve(cho_factor(gram), nuisance.T @ values)
    design, target = profiled[:, :-1], profiled[:, -1]
    lam, cap = float(study['source_ridge']), float(study['u_box'])
    hessian = design.T @ design + lam * np.eye(design.shape[1])
    linear = -(design.T @ target)
    certificates = []
    for total in map(float, study['budget_grid']):
        w, nu = _box_simplex(hessian, linear, cap, total)
        residual = design @ w - target
        primal = 0.5 * float(residual @ residual + lam * (w @ w))
        g = design.T @ residual + lam * w
        lower = np.maximum(g - nu, 0.0)
        upper = np.maximum(nu - g, 0.0)
        # Same tangential lower bound, evaluated via the complementarity gap.
        gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - total)
        certificates.append({'weights': w, 'nu_sum': nu, 'lam_lower': lower,
                             'lam_upper': upper, 'primal_objective': primal,
                             'dual_lower_bound': primal - max(0.0, float(gap))})
    weights = np.stack([cert['weights'] for cert in certificates])
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal),
            'kkt_certificates': certificates}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`526 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/multibudget_kronecker_kkt_transfer_solver.py']`
- **Stdout (tail)**:
```text
: "pretraining_only_l1_primitive_sublation",
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
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1d6a3a51ee08aef2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1d6a3a51 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`39039 ms` LLM)
- **Rationale**: The schema checks field shape but not field accuracy. Compare every shift against independent sparse solves to detect sensor ordering or transport errors, including reversed signs. Stress near-endpoint and vertex budgets with an independent linear oracle to detect active-set or certificate failures.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'multibudget_kronecker_kkt_transfer_solver.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
with open(p+'public_suc02_T_studies.pkl','rb') as f: item=pickle.load(f)[0]
s,g=item['study'],item['gt']
t=time.perf_counter(); out=mod.solve_multibudget_kronecker_transfer(s); print('public solve seconds',time.perf_counter()-t)
for k in ('weights_by_budget','terminal_field_mid_budget'):
    err=np.max(np.abs(out[k]-g[k])); print(k,'max abs error',err); assert err<1e-8
exec(open(p+'validate_public_schema.py').read().split('\n_d =',1)[0])
checks=[verify_box_simplex_kkt_certificate(g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],b,c) for b,c in zip(s['budget_grid'],out['kkt_certificates'])]
assert all(c['valid'] for c in checks), checks
print('public gaps', [c['gap'] for c in checks])

def reversed_axis(a):
    h=s['L'+a]/len(s['kappa_'+a]); k=s['kappa_'+a]; v=-s['vel_'+a]; r=s['robin_'+a]; n=len(k); A=np.zeros((n,n))
    for i in range(n):
        for j in (i-1,i+1):
            if 0<=j<n:
                d=2/(1/k[i]+1/k[j])/h**2
                A[i,i]+=d; A[i,j]-=d
        j=i-1 if v[i]>0 else i+1
        d=abs(v[i])/h; A[i,i]+=d
        if 0<=j<n: A[i,j]-=d
        if i in (0,n-1): A[i,i]+=1/(h/r+h*h/(2*k[i]))
    return A
worst_s=worst_f=0.
for study,ax,by in [(s,g['Ax'],g['By']), (dict(s,vel_x=-s['vel_x'],vel_y=-s['vel_y'],mu_shifts=s['mu_shifts'][::-1]),reversed_axis('x'),reversed_axis('y'))]:
    nx,ny=study['nx'],study['ny']; a=sp.kronsum(sp.csc_matrix(by),sp.csc_matrix(ax),format='csc'); eye=sp.eye(nx*ny,format='csc'); sources=study['source_modes'].reshape(-1,nx*ny).T
    ref=[]; si,sj=study['sensor_ij'].T
    for mu in study['mu_shifts']:
        fields=splu(a+float(mu)*eye).solve(sources).T.reshape(-1,nx,ny)
        ref.append(fields[:,si,sj].T)
    stack,terminal=mod._responses(study)
    worst_s=max(worst_s,np.max(abs(stack-np.vstack(ref)))); worst_f=max(worst_f,np.max(abs(terminal-fields)))
print('sparse reference worst sensor and terminal discrepancies',worst_s,worst_f); assert max(worst_s,worst_f)<1e-10
rng=np.random.default_rng(218); worst_gap=worst_feas=0.
for case in range(180):
    n=16; rank=(1,5,16)[case%3]; D=rng.normal(size=(rank,n)); H=D.T@D+(.04 if case%2 else 1e-4)*np.eye(n); q=rng.normal(size=n)*(100 if case%4==0 else .1); cap=(.3,1.,2.)[case%3]
    if case%9==0: H=np.eye(n); q=np.r_[np.full(n//2,-50.),np.full(n//2,50.)]
    for b in (0.,cap*1e-8,cap*.75,cap*8,cap*8.3,cap*(n-1e-8),cap*n):
        w,nu=mod._box_simplex(H,q,cap,b); grad=H@w+q
        # Greedy linear minimizer verifies the tangential bound independently of nu.
        v=np.zeros(n); order=np.argsort(grad); v[order]=np.clip(b-cap*np.arange(n),0,cap)
        gap=float(grad@(w-v)); dual_gap=grad@w-nu*b-cap*np.minimum(grad-nu,0).sum(); feas=max(abs(w.sum()-b),-min(w),max(w)-cap)
        worst_gap=max(worst_gap,gap,dual_gap); worst_feas=max(worst_feas,feas)
        assert -1e-9<=gap<=1e-8 and -1e-9<=dual_gap<=1e-8 and feas<1e-10,(case,b,gap,dual_gap,feas)
print('1260 budget slices: worst gap',worst_gap,'worst feasibility error',worst_feas)
PY
```
- **Container Exit Code**: `0` (`1408 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public solve seconds 0.010750379879027605
weights_by_budget max abs error 1.1282641487753153e-14
terminal_field_mid_budget max abs error 2.0643209364124004e-16
public gaps [6.938893903907228e-16, 1.6653345369377348e-15, 3.3306690738754696e-15]
sparse reference worst sensor and terminal discrepancies 7.129713486264677e-16 5.273559366969494e-16
1260 budget slices: worst gap 1.5890577742538878e-11 worst feasibility error 3.552713678800501e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1d6a3a51ee08aef2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1d6a3a51 (directory).
```

### Turn 5 — `submit_completed_task` (`13869 ms` LLM)
- **Rationale**: The saved artifact passed the public schema in an isolated Python process, matched independent sparse solves across all shifts with both tested advection directions, and met gap and feasibility tolerances at interior, vertex, and near-endpoint budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/multibudget_kronecker_kkt_transfer_solver.py': 'FILE:5683'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed. Independent sparse solves matched the sensor and terminal fields within 7.13e-16, and the worst KKT gap across 1,260 budget stress cases was 1.59e-11, below the 1e-6 limit.`

### Turn 6 — `execute_in_remote_docker` (`36822 ms` LLM)
- **Rationale**: Audit task-specific failure modes on every supplied study: full-grid residuals reveal shift or boundary errors, augmented least squares independently checks nuisance profiling, and cross-budget supporting lines check the multiplier sign. The study keys establish whether batch or relay checks apply.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from pathlib import Path
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'multibudget_kronecker_kkt_transfer_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (p/'public_suc02_T_studies.pkl').open('rb') as f:
    items = pickle.load(f)
print('public studies:', len(items), '; input keys:', sorted(items[0]['study']))
worst = dict(design=0., target=0., pde=0., envelope=0., weight=0., field=0., gap=0.)
for item in items:
    s, gt = item['study'], item['gt']
    A, B = gt['Ax'], gt['By']
    modes = [m._axis_modes(s['L'+axis], s['kappa_'+axis], s['vel_'+axis], s['robin_'+axis]) for axis in ('x','y')]
    (ex,lx,rx),(ey,ly,ry) = modes
    F = s['source_modes']
    coeff = lx @ F @ ly.T
    min_sep = np.inf
    for shift in s['mu_shifts']:
        denom = ex[:,None] + ey[None,:] + shift
        min_sep = min(min_sep, np.min(np.abs(denom)))
        U = rx @ (coeff / denom) @ ry.T
        err = np.max(np.abs(A @ U + U @ B.T + shift*U - F)) / np.max(np.abs(F))
        worst['pde'] = max(worst['pde'], err)
    print('minimum shifted eigenvalue magnitude:', min_sep, '; axis similarity scale ratios:', [float(np.linalg.norm(R,axis=1).max()/np.linalg.norm(R,axis=1).min()) for _,_,R in modes])
    stacked, _ = m._responses(s)
    z = s['Z_nuisance']
    # Independently profile via augmented least squares, not normal equations.
    V = np.column_stack((stacked, s['y_obs']))
    Z_aug = np.vstack((z, np.sqrt(s['nuisance_ridge'])*np.eye(z.shape[1])))
    V_aug = np.vstack((V, np.zeros((z.shape[1], V.shape[1]))))
    P = V - z @ np.linalg.lstsq(Z_aug,V_aug,rcond=None)[0]
    worst['design'] = max(worst['design'], np.max(np.abs(P[:,:-1]-gt['S_tilde'])))
    worst['target'] = max(worst['target'], np.max(np.abs(P[:,-1]-gt['r_y'])))
    out = m.solve_multibudget_kronecker_transfer(s)
    for key, metric in [('weights_by_budget','weight'),('terminal_field_mid_budget','field')]:
        worst[metric] = max(worst[metric],np.max(np.abs(out[key]-gt[key])))
    grid = np.asarray(s['budget_grid'])
    cert = out['kkt_certificates']
    for a, ca in zip(grid, cert):
        worst['gap'] = max(worst['gap'],ca['primal_objective']-ca['dual_lower_bound'])
        # A budget multiplier gives a supporting line for the value function at every other budget.
        for b, cb in zip(grid, cert):
            v = ca['dual_lower_bound'] + ca['nu_sum']*(b-a) - cb['primal_objective']
            worst['envelope'] = max(worst['envelope'], v)
print('worst public audit metrics:', worst)
assert max(worst['design'],worst['target'],worst['pde'],worst['envelope']) < 1e-10, worst
assert max(worst['weight'],worst['field']) < 1e-8 and worst['gap'] <= 1e-6, worst
PY
```
- **Container Exit Code**: `0` (`511 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public studies: 1 ; input keys: ['Lx', 'Ly', 'Z_nuisance', 'budget_grid', 'kappa_x', 'kappa_y', 'mu_shifts', 'nuisance_ridge', 'nx', 'ny', 'robin_x', 'robin_y', 'sensor_ij', 'source_modes', 'source_ridge', 'u_box', 'vel_x', 'vel_y', 'y_obs']
minimum shifted eigenvalue magnitude: 1.6323633660401091 ; axis similarity scale ratios: [4.194738798866187, 2.5067016705517378]
worst public audit metrics: {'design': 7.945033519973776e-16, 'target': 3.9968028886505635e-15, 'pde': 2.8450819496672173e-13, 'envelope': 0.0, 'weight': 1.1282641487753153e-14, 'field': 2.0643209364124004e-16, 'gap': 8.326672684688674e-17}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1d6a3a51ee08aef2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1d6a3a51 (directory).
```

### Turn 7 — `submit_completed_task` (`14199 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification, independent sparse solves with reversed advection, and an audit that found no cross-budget supporting-line violations. The public study has no batch or relay inputs, and the task-specific checks found no defect requiring a revision.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/multibudget_kronecker_kkt_transfer_solver.py': 'FILE:5683'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed, and the mid-budget terminal field matched the reference within 2.07e-16. The worst KKT gap across 1,260 budget stress cases was 1.59e-11, below the 1e-6 limit.`

### Turn 8 — `execute_in_remote_docker` (`37297 ms` LLM)
- **Rationale**: Use exact transformations of the public study to test what the earlier audit did not: geometry scaling, sensor-to-observation alignment, and nuisance-basis invariance. A discrepancy would identify geometry assembly, stacking, or profiling as needing a targeted check; batch and relay tests do not apply to this study.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
from pathlib import Path
import pickle
import numpy as np
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'multibudget_kronecker_kkt_transfer_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (p/'public_suc02_T_studies.pkl').open('rb') as f:
    items = pickle.load(f)
rng = np.random.default_rng(461)
worst_w = worst_f = worst_gap = 0.0
for item in items:
    s, gt = item['study'], item['gt']
    count = len(s['sensor_ij'])
    order = rng.permutation(count)
    rows = (np.arange(len(s['mu_shifts']))[:, None]*count + order).ravel()
    z = s['Z_nuisance']
    q = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))[0]
    for scale in (0.55, 1.9):
        # A, B, shifts and forcing all scale by 1/scale**2, preserving U.
        t = dict(s, Lx=s['Lx']*scale, Ly=s['Ly']*scale,
                 vel_x=s['vel_x']/scale, vel_y=s['vel_y']/scale,
                 robin_x=s['robin_x']/scale, robin_y=s['robin_y']/scale,
                 mu_shifts=s['mu_shifts']/scale**2, source_modes=s['source_modes']/scale**2,
                 sensor_ij=s['sensor_ij'][order], y_obs=s['y_obs'][rows], Z_nuisance=z[rows]@q)
        out = m.solve_multibudget_kronecker_transfer(t)
        worst_w = max(worst_w, np.max(np.abs(out['weights_by_budget']-gt['weights_by_budget'])))
        worst_f = max(worst_f, np.max(np.abs(out['terminal_field_mid_budget']-gt['terminal_field_mid_budget'])))
        design, target = gt['S_tilde'], gt['r_y']
        for b, w, cert in zip(s['budget_grid'], out['weights_by_budget'], out['kkt_certificates'], strict=True):
            r = design@w-target
            lam, cap, nu = s['source_ridge'], s['u_box'], cert['nu_sum']
            g = design.T@r+lam*w
            primal = 0.5*(r@r+lam*(w@w))
            dual = primal-g@w+nu*b+cap*np.minimum(g-nu, 0).sum()
            gap = primal-dual
            worst_gap = max(worst_gap, gap)
            assert abs(w.sum()-b)<1e-10 and w.min()>=-1e-10 and w.max()<=cap+1e-10
            assert -1e-10 <= gap <= 1e-6
            assert abs(cert['primal_objective']-primal)<1e-10 and abs(cert['dual_lower_bound']-dual)<1e-10
print(f'Geometry, sensor ordering and nuisance rotation: max weight error={worst_w:.3e}, max field error={worst_f:.3e}, max independently recomputed gap={worst_gap:.3e}')
assert max(worst_w, worst_f)<1e-8
PY
```
- **Container Exit Code**: `0` (`489 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Geometry, sensor ordering and nuisance rotation: max weight error=1.438e-14, max field error=2.394e-16, max independently recomputed gap=9.354e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1d6a3a51ee08aef2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1d6a3a51 (directory).
```

### Turn 9 — `submit_completed_task` (`13621 ms` LLM)
- **Rationale**: The unchanged saved artifact passed isolated-process verification, independent sparse solves with reversed advection, and budget and invariance stress tests. No tested slice calls for a revision.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py`. All three public KKT certificates passed. The worst KKT gap across 1,260 budget stress cases was 1.59e-11, below the 1e-6 limit. Geometry rescaling, sensor reordering, and nuisance-basis rotation preserved the reference weights within 1.44e-14 and the terminal field within 2.40e-16.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/multibudget_kronecker_kkt_transfer_solver.py": "FILE:5683"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_weights_rel_l2": 0.0,
    "max_field_rel_l2": 0.0,
    "max_kkt_gap": 0.0,
    "student_elapsed_sec": 1.6018,
    "instance_diagnostics": [
      {
        "seed": 99611,
        "w_rel_l2": 0.0,
        "field_rel_l2": 0.0,
        "max_kkt_gap": 0.0
      },
      {
        "seed": 99612,
        "w_rel_l2": 0.0,
        "field_rel_l2": 0.0,
        "max_kkt_gap": 0.0
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.043198,
      "import_cpu_sec": 0.039754,
      "candidate_call_wall_sec": 0.006294,
      "candidate_call_cpu_sec": 0.00579,
      "ipc_and_codec_wall_sec": 0.193742,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 1.488139,
      "verifier_and_gt_cpu_sec": 1.44025,
      "total_supervisor_wall_sec": 1.731374
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 1.6018, \"instance_diagnostics\": [{\"seed\": 99611, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}, {\"seed\": 99612, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_ffb5806b499b7e88973fdb6c6e2301c8__:{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 1.6018, \"instance_diagnostics\": [{\"seed\": 99611, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}, {\"seed\": 99612, \"w_rel_l2\": 0.0, \"field_rel_l2\": 0.0, \"max_kkt_gap\": 0.0}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.043198, \"import_cpu_sec\": 0.039754, \"candidate_call_wall_sec\": 0.006294, \"candidate_call_cpu_sec\": 0.00579, \"ipc_and_codec_wall_sec\": 0.193742, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 1.488139, \"verifier_and_gt_cpu_sec\": 1.44025, \"total_supervisor_wall_sec\": 1.731374}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 8074.2,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
