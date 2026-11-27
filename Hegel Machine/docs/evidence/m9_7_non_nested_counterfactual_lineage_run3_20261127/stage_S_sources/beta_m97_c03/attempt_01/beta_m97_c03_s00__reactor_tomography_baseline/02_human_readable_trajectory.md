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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `332,519` (`318,864` / `90,112` / `13,655`) |
| **Total Episode Wall Time (s)** | `437.63s` |

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

### Turn 1 — `execute_in_remote_docker` (`18019 ms` LLM)
- **Rationale**: Inspect the public schema before coding: source and sensor array layouts determine the forward assembly, and calibration coverage determines whether the archive fallback is needed.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_beta_m97_c03_s00.pkl','rb') as f:
    data=pickle.load(f)
def show(x, depth=0):
    if isinstance(x,dict) and depth < 3:
        for k,v in x.items():
            print('  '*depth+str(k),end=': ')
            if isinstance(v,dict): print()
            show(v,depth+1)
    elif isinstance(x,np.ndarray):
        print(x.shape,x.dtype, x.tolist() if x.size<=15 else x.ravel()[:5])
    elif isinstance(x,(list,tuple)):
        print(type(x).__name__,len(x),repr(x[:1])[:400])
    else: print(repr(x)[:500])
show(data)
PY
ls -l /tmp/hegel_continual_archive 2>/dev/null || true
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1 | head -25
```
- **Container Exit Code**: `0` (`379 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator | STATUS: MISMATCHED_BOUNDARY_OR_TENSOR(zero_ghost_scalar_dirichlet_only)
```

### Turn 2 — `execute_in_remote_docker` (`16073 ms` LLM)
- **Rationale**: The previous output truncated the source and sensor layouts. Read those fields directly to avoid assembling a transposed forward model, and check calibration coverage to determine whether the archive fallback is needed.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_beta_m97_c03_s00.pkl','rb'))[0]; s=p['study']
for k,v in s.items():
 if isinstance(v,np.ndarray): print(k,v.shape,v.dtype, v.ravel()[:12])
 elif k=='sensor_calibration_sweeps':
  print(k,type(v),str(v)[:3000])
 else: print(k,type(v).__name__,str(v)[:250])
print('reference weights',p['gt']['w_opt'])
PY
ls /tmp/hegel_continual_archive 2>/dev/null || true
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -35
```
- **Container Exit Code**: `0` (`413 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
able",
  "execution_available": true,
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
```

### Turn 3 — `execute_in_remote_docker` (`20601 ms` LLM)
- **Rationale**: The preceding output omitted the forward-array names and dimensions. Print only the needed layout and per-bank fit summaries, and resolve the multiplier sign against the supplied numerical reference before implementing the certificate.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_beta_m97_c03_s00.pkl','rb'))[0]; s=p['study']
for k,v in s.items():
 if k=='sensor_calibration_sweeps':
  print(k,[(a,np.shape(b)) for a,b in v.items()]); b=np.asarray(v['bank_id']); x=np.asarray(v['ref_signal']); y=np.asarray(v['raw_readout']);
  for i in range(4):
   X=np.column_stack((x[b==i],np.ones(sum(b==i)))); c=np.linalg.lstsq(X,y[b==i],rcond=None)[0]; print(i,len(X),c,np.max(abs(y[b==i]-X@c),initial=0.))
 elif isinstance(v,np.ndarray): print(k,v.shape,v.dtype)
 elif isinstance(v,(list,tuple)): print(k,len(v),type(v[0]).__name__ if v else None,np.shape(v) if v and not isinstance(v[0],dict) else '', list(v[0]) if v and isinstance(v[0],dict) else '')
 else: print(k,v)
w=p['gt']['w_opt']; print('reference gradient',p['gt']['S_tilde'].T@(p['gt']['S_tilde']@w-p['gt']['r_y'])+s['source_ridge']*w); print('reference nu',p['gt']['cert_opt']['nu_sum'])
PY
sed -n '45,120p' validate_public_schema.py
if [ -f /tmp/hegel_continual_archive/archive_calibration_history.json ]; then head -c 2000 /tmp/hegel_continual_archive/archive_calibration_history.json; echo; fi
```
- **Container Exit Code**: `0` (`364 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-2.38253569e-09] 1.6977768546588834e-07
3 12 [9.99999972e-01 1.33748665e-08] 1.6093999222199784e-07
Z_nuisance (144, 5) float64
y_obs_raw (144,) float64
aux_manifold_telemetry (144, 3) float64
nuisance_ridge 0.65
source_ridge 0.04
u_box 1.0
total_budget 2.6
reference gradient [0.00769253 0.00769253 0.00769253 0.00769253 0.00769253 0.00769253
 0.00769253 0.00769253 0.00769253 0.00769253 0.00769253 0.00769253]
reference nu 0.0076925287566829426
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
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c03_s00_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c03_s00.pkl"), "rb"))[0]
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
    elif not isinstance(out["kkt_certificate"], dict) or not {"weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"} <= set(out["kkt_certificate"].keys()):
        errs.append("bad_kkt_certificate_keys")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 4 — `execute_in_remote_docker` (`33536 ms` LLM)
- **Rationale**: The public study supplies axis reaction arrays as well as a 2D reaction term, and the sensor row order is not explicit. A small comparison against the supplied forward and terminal references will determine both conventions before they are committed to the solver.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import itertools, pickle
import numpy as np
from scipy.linalg import solve
p=pickle.load(open('public_beta_m97_c03_s00.pkl','rb'))[0]; s=p['study']; gt=p['gt']
Z=s['Z_nuisance']; G=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); residual=lambda A:A-Z@solve(G,Z.T@A,assume_a='pos')
nx,ny=s['nx'],s['ny']; modes=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
def axis(a, spacing, react):
 k,v,r=s['kappa_'+a],s['vel_'+a],s['reaction_'+a]; n=len(k); h=s['L'+a]/(n-spacing); A=np.zeros((n,n)); i=np.arange(1,n-1); km=(k[i]+k[i-1])/2; kp=(k[i]+k[i+1])/2
 A[i,i-1]=-km/h**2-np.maximum(v[i],0)/h; A[i,i+1]=-kp/h**2-np.maximum(-v[i],0)/h; A[i,i]=-A[i,i-1]-A[i,i+1]
 A[0,0]=k[0]/h**2+s['robin_'+a]/h; A[0,1]=-k[0]/h**2; A[-1,-1]=k[-1]/h**2+s['robin_'+a]/h; A[-1,-2]=-k[-1]/h**2
 if react: A[i,i]+=r[i]
 if react==2: A[0,0]+=r[0]; A[-1,-1]+=r[-1]
 return A
results=[]
for spacing, react in itertools.product((1,0),(0,1,2)):
 L=np.kron(np.eye(ny),axis('x',spacing,react))+np.kron(axis('y',spacing,react),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F')); U=[solve(L+mu*np.eye(nx*ny),modes) for mu in s['mu_shifts']]; T=np.stack([u[idx] for u in U]);
 for order,A in [('shift-major',T.reshape(-1,modes.shape[1])),('sensor-major',T.transpose(1,0,2).reshape(-1,modes.shape[1]))]:
  F=(U[-1]@gt['w_opt']).reshape((nx,ny),order='F'); results.append((np.max(abs(residual(A)-gt['S_tilde'])),np.max(abs(F-gt['terminal_field_A'])),spacing,react,order))
print('max sensor error, max terminal error, grid denominator correction, axis reaction setting, row order:'); print(*sorted(results)[:5],sep='\n')
c=s['sensor_calibration_sweeps']; coeff=np.array([np.linalg.lstsq(np.column_stack((c['ref_signal'][c['bank_id']==i],np.ones(sum(c['bank_id']==i)))),c['raw_readout'][c['bank_id']==i],rcond=None)[0] for i in range(4)]); g,o=coeff[s['row_bank_id']].T
for name,y in [('affine fit',(s['y_obs_raw']-o)/g),('identity',s['y_obs_raw'])]: print(name,'max target error',np.max(abs(residual(y)-gt['r_y'])))
PY
ls -ld /tmp/hegel_continual_archive /tmp/hegel_continual_archive/archive_calibration_history.json 2>/dev/null || true
```
- **Container Exit Code**: `0` (`481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
max sensor error, max terminal error, grid denominator correction, axis reaction setting, row order:
(7.632783294297951e-17, 4.85722573273506e-17, 1, 0, 'shift-major')
(0.005316122423986297, 0.001560999358113796, 0, 0, 'shift-major')
(0.0159638158675027, 0.008782411223686542, 1, 1, 'shift-major')
(0.016184531743050783, 0.008806630463110112, 1, 2, 'shift-major')
(0.01953687480325815, 0.009483130878350204, 0, 1, 'shift-major')
affine fit max target error 6.839283006598862e-08
identity max target error 8.881784197001252e-16
```

### Turn 5 — `execute_in_remote_docker` (`69773 ms` LLM)
- **Rationale**: Implement the reference-supported operator and a feasible active-set inversion, preserving the required affine calibration even for near-identity banks. The isolated schema run checks importability and output shapes; quantitative certificate and active-bound checks are still needed.
- **Alternatives Weighed**: `Add the supplied axis reaction arrays to the stencil` (rejected: The prescribed 2D reaction term alone reproduces the public forward matrix and terminal field to numerical precision.)

```bash
cd /holdout_task && cat > beta_m97_c03_s00_solver.py <<'PY'
"""Open-Robin reaction-transport tomography and box-simplex inversion."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(bank):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    if isinstance(history, dict) and 'banks' in history:
        history = history['banks']
    if isinstance(history, dict) and 'bank_id' in history:
        rows = [dict(bank_id=b, gain=g, offset=o) for b, g, o in
                zip(history['bank_id'], history['gain'], history['offset'])]
        row = next((r for r in reversed(rows) if int(r['bank_id']) == bank), None)
    elif isinstance(history, list):
        row = next((r for r in reversed(history) if int(r['bank_id']) == bank), None)
    else:
        row = history.get(str(bank))
        if isinstance(row, list) and row and isinstance(row[0], dict):
            row = row[-1]
    if row is None:
        return None
    if isinstance(row, dict):
        return float(row['gain']), float(row['offset'])
    return tuple(map(float, row))


def _line_fit(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    x = np.asarray(sweeps['ref_signal'], dtype=float)
    y = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    coefficients = np.empty((4, 2))
    for bank in np.unique(row_banks):
        xb, yb = x[banks == bank], y[banks == bank]
        clean = np.ones(xb.size, dtype=bool)
        # A median-slope fit prevents a single EMI spike from tilting the
        # line used to detect it. Fit the final line by least squares.
        if xb.size >= 3:
            i, j = np.triu_indices(xb.size, 1)
            dx = xb[j] - xb[i]
            distinct = dx != 0
            if np.any(distinct):
                slopes = (yb[j[distinct]] - yb[i[distinct]]) / dx[distinct]
                slope = np.median(slopes)
                residual = yb - slope * xb
                clean = np.abs(residual - np.median(residual)) <= 3.0
        if xb.size < 2 or not np.all(clean):
            archived = _archived_calibration(bank)
            if archived is not None:
                coefficients[bank] = archived
                continue
        if np.unique(xb[clean]).size < 2:
            raise ValueError(f'Insufficient calibration data for sensor bank {bank}')
        coefficients[bank] = _line_fit(xb[clean], yb[clean])
    gain, offset = coefficients[row_banks].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


def _axis(kappa, velocity, robin, length):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    main = np.empty(n)
    main[1:-1] = -lower[:-1] - upper[1:]
    main[0] = -upper[0] + robin / h
    main[-1] = -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                    study['robin_' + a], study['L' + a]) for a in ('x', 'y')]
    op = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
          + diags(np.ravel(study['V_reaction_2d'], order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_rows = i + nx * j
    identity = eye(nx * ny, format='csc')
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * identity).solve(modes)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field


def _box_simplex(H, c, cap, budget):
    """Feasible active-set method; -1 and +1 mark lower and upper bounds."""
    n = len(c)
    if not 0 <= budget <= n * cap:
        raise ValueError('Source budget is outside the feasible box')
    if budget == 0:
        return np.zeros(n), float(np.min(c))
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


def solve_beta_m97_c03_s00(study: dict) -> dict:
    if (study['recirculation_loop_mode'] != 0 or
            np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('This solver requires the open outflow Robin regime')
    y = _calibrate(study)
    sensors, terminal = _forward(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    data = np.column_stack((y, sensors))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    data -= Z @ np.linalg.solve(gram, Z.T @ data)
    r, S = data[:, 0], data[:, 1:]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(S.T @ r), cap, budget)
    residual = S @ w - r
    gradient = S.T @ residual + ridge * w
    reduced = gradient - nu
    ll = np.where(w == 0, np.maximum(reduced, 0), 0.)
    lu = np.where(w == cap, np.maximum(-reduced, 0), 0.)
    primal = float((residual @ residual + ridge * (w @ w)) / 2)
    lower = primal - gradient @ w + nu * budget + cap * np.minimum(0, reduced).sum()
    certificate = {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': ll.tolist(),
                   'lam_upper': lu.tolist(), 'primal_objective': primal,
                   'dual_lower_bound': min(primal, float(lower))}
    return {'weights': w, 'terminal_field': (terminal @ w).reshape((study['nx'], study['ny']), order='F'),
            'kkt_certificate': certificate}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`447 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=40a67ceef710f947 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_40a67cee (directory).
```

### Turn 6 — `execute_in_remote_docker` (`41576 ms` LLM)
- **Rationale**: The schema check cannot detect inaccurate weights, fields, or certificates. Compare all three against the public references, then stress the calibration fallback and active bounds that the public example does not exercise. A slice failure would identify which part of the solver needs revision.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np

p = Path.cwd()
spec = importlib.util.spec_from_file_location('candidate', p/'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# Load the independent certificate verifier without executing the schema CLI.
ns = {}
exec((p/'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
items = pickle.load(open(p/'public_beta_m97_c03_s00.pkl','rb'))
errors = np.zeros(3)
for item in items:
    s, gt = item['study'], item['gt']
    out = m.solve_beta_m97_c03_s00(s)
    result = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    assert result['valid'], result
    errors = np.maximum(errors, [np.max(abs(out['weights']-gt['w_opt'])), np.max(abs(out['terminal_field']-gt['terminal_field_A'])), result['gap']])
    np.testing.assert_array_equal(out['weights'], out['kkt_certificate']['weights'])
assert max(errors) < 1e-6, errors
print('public worst max weight error, max field error, verified gap:', errors)

# Force nonidentity calibration, archive-only banks, and isolated EMI in complete sweeps.
s = items[0]['study']
c = s['sensor_calibration_sweeps']
b, x = c['bank_id'], c['ref_signal']
rb = s['row_bank_id']
gain, offset = np.array([.72,1.28,1.53,.88]), np.array([.31,-.4,.19,.55])
with tempfile.TemporaryDirectory() as directory:
    m._ARCHIVE = Path(directory)/'calibration.json'
    for case in ('clean', 'one row', 'no rows', 'EMI'):
        for bank in range(4):
            raw = (gain * x[:,None] + offset)[np.arange(x.size), b]
            where = np.flatnonzero(b == bank)
            keep = np.ones(b.size, dtype=bool)
            if case == 'one row': keep[where[1:]] = False
            if case == 'no rows': keep[where] = False
            if case == 'EMI': raw[where[0]] += 8
            # A deliberately stale archive must never override other clean banks.
            history = {str(i): {'gain': (gain[i] if i == bank and case != 'clean' else 3), 'offset': (offset[i] if i == bank and case != 'clean' else 2)} for i in range(4)}
            m._ARCHIVE.write_text(json.dumps(history))
            modified = dict(s, sensor_calibration_sweeps=dict(bank_id=b[keep], ref_signal=x[keep], raw_readout=raw[keep]), y_obs_raw=gain[rb]*s['y_obs_raw']+offset[rb])
            cal_error = np.max(abs(m._calibrate(modified)-s['y_obs_raw']))
            assert cal_error < 1e-12, (case, bank, cal_error)
            out = m.solve_beta_m97_c03_s00(modified)
            np.testing.assert_allclose(out['weights'], items[0]['gt']['w_opt'], rtol=0, atol=1e-10)
print('all four banks passed clean, sparse, missing, and isolated-EMI calibration slices')

# Probe active-set release, integer-budget vertices and budget endpoints.
rng = np.random.default_rng(1107)
worst = np.zeros(3)
count = np.zeros(3,dtype=int)
for t in range(800):
    n = 12
    A = rng.normal(size=(n if t%2 else 3,n))
    H = A.T@A + (1e-4 if t%3 else .04)*np.eye(n)
    c = rng.normal(size=n)*(30 if t%4 else 1)
    cap = 0.4 if t%2 else 1.
    budget = [0, n*cap, cap, 2*cap, (n-1)*cap, .05*cap, (n-.05)*cap, rng.uniform(0,n*cap)][t%8]
    w, nu = m._box_simplex(H,c,cap,budget)
    g = H@w+c
    lb_gap = g@w - nu*budget - cap*np.minimum(g-nu,0).sum()
    stat = np.where(w==0, np.minimum(g-nu,0), np.where(w==cap,np.maximum(g-nu,0),g-nu))
    worst = np.maximum(worst,[max(-w.min(),w.max()-cap,abs(w.sum()-budget)),abs(lb_gap),np.max(abs(stat))])
    count += [np.any(w==0), np.any(w==cap), np.all((w==0)|(w==cap))]
assert max(worst) < 1e-8, worst
print('800 QPs: worst feasibility, absolute gap, stationarity:', worst, '; lower/upper/all-bound cases:', count)
PY
```
- **Container Exit Code**: `0` (`927 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public worst max weight error, max field error, verified gap: [3.98397272e-08 1.47504199e-09 6.58119472e-09]
all four banks passed clean, sparse, missing, and isolated-EMI calibration slices
800 QPs: worst feasibility, absolute gap, stationarity: [5.21804822e-15 3.41060513e-13 1.42108547e-14] ; lower/upper/all-bound cases: [520 562 346]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=40a67ceef710f947 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_40a67cee (directory).
```

### Turn 7 — `submit_completed_task` (`14089 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema and independent certificate verification, plus stress tests for sparse and EMI-affected calibration and active source bounds.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:7789'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public schema and KKT checks passed. Maximum public errors were 3.98e-8 for weights and 1.48e-9 for the terminal field. Calibration stress tests passed for all four banks, and 800 box-simplex stress cases passed.`

### Turn 8 — `execute_in_remote_docker` (`28804 ms` LLM)
- **Rationale**: The previous EMI test supplied a correct archive for the affected bank, so it could not detect an archive overriding four or more clean current points. Inject one spike into each public bank and supply stale archive coefficients. If calibration changes materially, make the clean in-batch fit take precedence while preserving the sparse-bank fallback.
- **Alternatives Weighed**: `Test alternative drift laws, relay modes, or within-batch classification accuracy` (rejected: The specified study uses a fixed open-Robin operator and continuous observations, with no relay modes or classification labels.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
items = pickle.load(open('public_beta_m97_c03_s00.pkl', 'rb'))
forward_err = projection_effect = 0.0
emi_errors = []
for case in items:
    s, gt = case['study'], case['gt']
    S, _ = m._forward(s)
    Z = s['Z_nuisance']
    y = m._calibrate(s)
    B = np.column_stack((y, S))
    R = B - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ B)
    forward_err = max(forward_err, np.max(abs(R[:, 1:] - gt['S_tilde'])))
    projection_effect = max(projection_effect, np.max(abs(B - R)))
    sweep = s['sensor_calibration_sweeps']
    with tempfile.TemporaryDirectory() as d:
        m._ARCHIVE = Path(d) / 'calibration.json'
        for bank in range(4):
            rows = np.flatnonzero(sweep['bank_id'] == bank)
            assert rows.size >= 5
            raw = sweep['raw_readout'].copy()
            raw[rows[0]] += 8.0
            m._ARCHIVE.write_text(json.dumps({str(bank): {'gain': 1.1, 'offset': 0.2}}))
            changed = dict(s, sensor_calibration_sweeps=dict(sweep, raw_readout=raw))
            sel = s['row_bank_id'] == bank
            emi_errors.append(float(np.max(abs(m._calibrate(changed)[sel] - y[sel]))))
print('Worst public projected forward error:', forward_err)
print('Largest nuisance correction on public data:', projection_effect)
print('Stale-archive calibration error with at least four clean in-batch points per bank:', emi_errors)
assert forward_err < 1e-12
assert max(emi_errors) < 1e-6, 'A stale archive overrides adequate clean in-batch calibration after an isolated EMI spike'
PY
```
- **Container Exit Code**: `1` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst public projected forward error: 8.326672684688674e-17
Largest nuisance correction on public data: 4.231459169520146
Stale-archive calibration error with at least four clean in-batch points per bank: [0.5732917489699783, 0.3636909143675122, 0.46578052525942715, 0.4182699775327734]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=40a67ceef710f947 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_40a67cee (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 36, in <module>
AssertionError: A stale archive overrides adequate clean in-batch calibration after an isolated EMI spike
```

### Turn 9 — `execute_in_remote_docker` (`40173 ms` LLM)
- **Rationale**: Fix calibration precedence without changing the verified operator or inversion. Check both sides of the four-clean-point threshold and verify that the corrected calibration preserves public weights, terminal fields, and KKT certificates.
- **Alternatives Weighed**: `Always override an EMI-affected sweep with archive coefficients` (rejected: A stale archive caused large calibration errors even when the current sweep retained enough clean points.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_s00_solver.py')
s = p.read_text()
old = '        if xb.size < 2 or not np.all(clean):\n'
new = ('        # Prefer four or more clean current points to historical coefficients.\n'
       '        if xb.size < 2 or (not np.all(clean) and np.count_nonzero(clean) < 4):\n')
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np

p = Path.cwd()
spec = importlib.util.spec_from_file_location('candidate', p/'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((p/'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (p/'public_beta_m97_c03_s00.pkl').open('rb') as f:
    items = pickle.load(f)
public_worst = np.zeros(3)
slices = {}
for item in items:
    s, gt = item['study'], item['gt']
    out = m.solve_beta_m97_c03_s00(s)
    checked = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    assert checked['valid'], checked
    np.testing.assert_array_equal(out['weights'], out['kkt_certificate']['weights'])
    public_worst = np.maximum(public_worst, [np.max(abs(out['weights'] - gt['w_opt'])), np.max(abs(out['terminal_field'] - gt['terminal_field_A'])), abs(checked['gap'])])
    x = s['sensor_calibration_sweeps']['ref_signal']
    b = s['sensor_calibration_sweeps']['bank_id']
    rb = s['row_bank_id']
    gain, offset = np.array([.72, 1.28, 1.53, .88]), np.array([.31, -.4, .19, .55])
    with tempfile.TemporaryDirectory() as d:
        m._ARCHIVE = Path(d)/'calibration.json'
        for bank in range(4):
            rows = np.flatnonzero(b == bank)
            for case in ('clean', '0 points', '1 point', '2 points', '3 points', '3 clean plus EMI', '4 clean plus EMI', 'full plus EMI', 'negative EMI'):
                raw = gain[b]*x + offset[b]
                keep = np.ones(len(x), dtype=bool)
                if case.endswith('points') or case == '1 point':
                    keep[rows[int(case[0]):]] = False
                elif 'EMI' in case:
                    if case[0].isdigit():
                        keep[rows[int(case[0])+1:]] = False
                    raw[rows[0]] += -8 if case == 'negative EMI' else 8
                use_archive = case in ('0 points', '1 point', '3 clean plus EMI')
                m._ARCHIVE.write_text(json.dumps({str(i): {'gain': gain[i] if use_archive and i == bank else 1.1, 'offset': offset[i] if use_archive and i == bank else .2} for i in range(4)}))
                changed = dict(s, sensor_calibration_sweeps=dict(bank_id=b[keep], ref_signal=x[keep], raw_readout=raw[keep]), y_obs_raw=gain[rb]*s['y_obs_raw']+offset[rb])
                err = np.max(abs(m._calibrate(changed) - s['y_obs_raw']))
                assert err < 1e-12, (case, bank, err)
                slices[case] = max(slices.get(case, 0), err)
                actual = m.solve_beta_m97_c03_s00(changed)
                np.testing.assert_allclose(actual['weights'], gt['w_opt'], rtol=0, atol=1e-10)
                np.testing.assert_allclose(actual['terminal_field'], gt['terminal_field_A'], rtol=0, atol=1e-10)
                checked = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], actual['kkt_certificate'])
                assert checked['valid'], (case, bank, checked)
assert max(public_worst) < 1e-6, public_worst
print('Public worst weight error, field error, KKT gap:', public_worst)
print('Worst calibration errors across four banks by slice:', slices)
PY
```
- **Container Exit Code**: `0` (`847 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public worst weight error, field error, KKT gap: [3.98397272e-08 1.47504199e-09 6.58119472e-09]
Worst calibration errors across four banks by slice: {'clean': 1.3322676295501878e-15, '0 points': 1.3322676295501878e-15, '1 point': 1.3322676295501878e-15, '2 points': 3.1086244689504383e-15, '3 points': 2.6645352591003757e-15, '3 clean plus EMI': 1.3322676295501878e-15, '4 clean plus EMI': 3.552713678800501e-15, 'full plus EMI': 1.3322676295501878e-15, 'negative EMI': 1.3322676295501878e-15}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d3661362ec291239 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d3661362 (directory).
```

### Turn 10 — `submit_completed_task` (`15760 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and independent KKT verification. Stress tests on both sides of the four-clean-point threshold confirmed that sparse banks still use the archive and stale values no longer override adequate current measurements.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/beta_m97_c03_s00_solver.py': 'FILE:7902'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The solver now prefers four or more clean in-batch calibration points over stale archive values. The public schema and KKT checks passed, with maximum errors of 3.98e-8 for weights and 1.48e-9 for the terminal field.`

### Turn 11 — `execute_in_remote_docker` (`34097 ms` LLM)
- **Rationale**: The prior EMI test used a large spike on one fixed row. A spike just above 3.0 can tilt a four-point median-slope fit enough to hide itself. Test that failure on subsets of the public measurements; if confirmed, change spike detection for sparse sweeps without changing the forward model or inversion.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    items = pickle.load(f)
worst = 0.0
failures = 0
first = None
with tempfile.TemporaryDirectory() as d:
    m._ARCHIVE = Path(d) / 'calibration.json'
    for item in items:
        s, gt = item['study'], item['gt']
        sw = s['sensor_calibration_sweeps']
        b, x, raw = (np.asarray(sw[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        expected = m._calibrate(s)
        for bank in range(4):
            rows = np.flatnonzero(b == bank)
            g, o = m._line_fit(x[rows], raw[rows])
            m._ARCHIVE.write_text(json.dumps({str(bank): {'gain': float(g), 'offset': float(o)}}))
            for subset in itertools.combinations(rows, 4):
                keep = b != bank
                keep[list(subset)] = True
                for j in (subset[0], subset[-1]):
                    for spike in (-3.01, 3.01):
                        y = raw.copy()
                        y[j] += spike
                        trial = dict(s, sensor_calibration_sweeps=dict(bank_id=b[keep], ref_signal=x[keep], raw_readout=y[keep]))
                        err = float(np.max(np.abs(m._calibrate(trial) - expected)))
                        worst = max(worst, err)
                        if err > 1e-6:
                            failures += 1
                            if first is None:
                                first = (bank, tuple(map(int, subset)), int(j), spike, err)
print('Near-threshold EMI on sparse public sweeps: worst calibrated error', worst, 'failures', failures, 'first', first)
assert failures == 0, 'Sparse sweep masks a qualifying EMI spike and bypasses the archive'
PY
```
- **Container Exit Code**: `1` (`3603 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Near-threshold EMI on sparse public sweeps: worst calibrated error 18621.93508112139 failures 3804 first (0, (0, 1, 2, 3), 3, -3.01, 9.63042386050093)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d3661362ec291239 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d3661362 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 38, in <module>
AssertionError: Sparse sweep masks a qualifying EMI spike and bypasses the archive
```

### Turn 12 — `execute_in_remote_docker` (`47909 ms` LLM)
- **Rationale**: Score candidate calibration lines by their agreement with a majority of the sweep before applying the 3.0 cutoff. Verify that above-threshold spikes trigger the sparse-bank fallback, adequate clean data still override stale history, and a below-threshold perturbation does not trigger archive use.
- **Alternatives Weighed**: `Lower the EMI cutoff` (rejected: That would change the specified 3.0 threshold without fixing the leverage bias in the preliminary fit.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_m97_c03_s00_solver.py')
s = p.read_text()
a = s.index('        # A median-slope fit')
b = s.index('        # Prefer four', a)
s = s[:a] + '''        if xb.size >= 3:
            i, j = np.triu_indices(xb.size, 1)
            distinct = xb[j] != xb[i]
            i, j = i[distinct], j[distinct]
            if i.size:
                slopes = (yb[j] - yb[i]) / (xb[j] - xb[i])
                if xb.size >= 4:
                    # Score complete lines by their residuals, not the slopes alone:
                    # a spike at the edge of a sparse sweep has high leverage.
                    offsets = yb[i] - slopes * xb[i]
                    errors = np.abs(yb - slopes[:, None] * xb - offsets[:, None])
                    best = np.argmin(np.median(errors, axis=1))
                    core = np.argsort(errors[best])[:xb.size // 2 + 1]
                    if np.unique(xb[core]).size >= 2:
                        slope, offset = _line_fit(xb[core], yb[core])
                    else:
                        slope, offset = slopes[best], offsets[best]
                else:
                    slope = np.median(slopes)
                    offset = np.median(yb - slope * xb)
                clean = np.abs(yb - (slope * xb + offset)) <= 3.0
''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'beta_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with open('public_beta_m97_c03_s00.pkl', 'rb') as f:
    items = pickle.load(f)
worst = dict(sparse_emi=0., adequate_emi=0., below_threshold=0., missing=0.)
public = np.zeros(3)
count = 0
with tempfile.TemporaryDirectory() as d:
    m._ARCHIVE = Path(d) / 'calibration.json'
    for item in items:
        s, gt = item['study'], item['gt']
        out = m.solve_beta_m97_c03_s00(s)
        v = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
        assert v['valid'], v
        public = np.maximum(public, [np.max(abs(out['weights']-gt['w_opt'])), np.max(abs(out['terminal_field']-gt['terminal_field_A'])), abs(v['gap'])])
        sw = s['sensor_calibration_sweeps']
        b, x, raw = (np.asarray(sw[k]) for k in ('bank_id','ref_signal','raw_readout'))
        expected = m._calibrate(s)
        for bank in range(4):
            rows = np.flatnonzero(b == bank)
            g, o = m._line_fit(x[rows], raw[rows])
            ok = {str(bank): {'gain':float(g), 'offset':float(o)}}
            stale = {str(bank): {'gain':1.1, 'offset':.2}}
            def check(name, selection, spike_row=None, spike=0., target=expected):
                global count
                keep = b != bank
                keep[list(selection)] = True
                y = raw.copy()
                if spike_row is not None:
                    y[spike_row] += spike
                trial = dict(s, sensor_calibration_sweeps=dict(bank_id=b[keep], ref_signal=x[keep], raw_readout=y[keep]))
                err = float(np.max(np.abs(m._calibrate(trial) - target)))
                worst[name] = max(worst[name], err)
                count += 1
            m._ARCHIVE.write_text(json.dumps(ok))
            for selection in itertools.combinations(rows, 4):
                for j in selection:
                    for spike in (-3.01, 3.01):
                        check('sparse_emi', selection, j, spike)
            check('missing', [])
            check('missing', rows[:1])
            m._ARCHIVE.write_text(json.dumps(stale))
            for selection in (rows, rows[:5], rows[-5:]):
                for j in selection:
                    for spike in (-3.01, 3.01):
                        clean = selection[selection != j]
                        gg, oo = m._line_fit(x[clean], raw[clean])
                        target = expected.copy()
                        mask = s['row_bank_id'] == bank
                        target[mask] = (s['y_obs_raw'][mask]-oo)/gg
                        check('adequate_emi', selection, j, spike, target)
            selection = rows[:4]
            j = selection[0]
            measurement = raw[selection].copy()
            measurement[0] += 2.99
            gg, oo = m._line_fit(x[selection], measurement)
            target = expected.copy()
            mask = s['row_bank_id'] == bank
            target[mask] = (s['y_obs_raw'][mask]-oo)/gg
            check('below_threshold', selection, j, 2.99, target)
print('Worst calibration errors over', count, 'checks:', worst)
print('Public worst weight error, terminal-field error and KKT gap:', public)
assert max(worst.values()) < 1e-12, worst
assert max(public) < 1e-6, public
PY
```
- **Container Exit Code**: `0` (`11115 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Worst calibration errors over 16028 checks: {'sparse_emi': 0.0, 'adequate_emi': 0.0, 'below_threshold': 0.0, 'missing': 0.0}
Public worst weight error, terminal-field error and KKT gap: [3.98397272e-08 1.47504199e-09 6.58119472e-09]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f69a64d736f3f536 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_f69a64d7 (directory).
```

### Turn 13 — `submit_completed_task` (`14849 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and independent certificate verification. The revised calibration caught the near-threshold spikes that defeated the previous screen, preserved the sparse-bank fallback, and kept adequate clean measurements ahead of stale archive values without worsening public accuracy.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c03_s00_solver.py`. The public schema and independent KKT checks passed. Maximum public errors were 3.98e-8 for weights and 1.48e-9 for the terminal field. All 16,028 calibration checks passed.`

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
