# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c01_s00__reactor_tomography_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c01_source` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `215,663` (`206,873` / `24,704` / `8,790`) |
| **Total Episode Wall Time (s)** | `327.16s` |

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

### Turn 1 — `execute_in_remote_docker` (`15324 ms` LLM)
- **Rationale**: Inspect the schema and dimensions before implementing so sensor orientation, source ordering, and archive availability are not guessed.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_beta_m97_c01_s00.pkl','rb') as f: s=pickle.load(f)
print('root:',type(s))
for k,v in s.items():
 print(k,type(v).__name__,getattr(v,'shape',None),str(v)[:230] if not isinstance(v,np.ndarray) or v.size<20 else f'dtype={v.dtype}')
PY
ls -la /tmp/hegel_continual_archive 2>/dev/null; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic program'
```
- **Container Exit Code**: `0` (`576 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator | STATUS: MISMATCHED_BOUNDARY_OR_TENSOR(zero_ghost_scalar_dirichlet_only)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19011 ms` LLM)
- **Rationale**: The first inspection stopped at the list root. Read the nested studies and verifier to resolve matrix layout and calibration inputs before implementing.
- **Alternatives Weighed**: `Implement from array dimensions alone` (rejected: Dimensions do not identify shift stacking, source field ordering, or the calibration record format.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_beta_m97_c01_s00.pkl','rb') as f: rows=pickle.load(f)
print('cases:',len(rows)); print('entry keys:',list(rows[0]))
for k,v in rows[0]['study'].items():
 print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:170] if not isinstance(v,np.ndarray) or v.size<20 else '')
for i,row in enumerate(rows):
 s=row['study']; c=s['sensor_calibration_sweeps']; print('case',i,'mode',s['recirculation_loop_mode'],'vortex_max',max(s['past_vortex_circulation_trace']),'sweeps',str(c)[:350])
PY
cat validate_public_schema.py
ls -la /tmp/hegel_continual_archive /holdout_task 2>/dev/null
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`35987 ms` LLM)
- **Rationale**: Check the forward model and calibrated target separately before inversion. A mismatch in either would otherwise be obscured by the fitted weights.
- **Alternatives Weighed**: `Assume the axis reaction arrays are unused` (rejected: The public study includes both axis and 2D reaction arrays; comparing the resulting sensor matrices and fields will distinguish the conventions.)

```bash
cd /holdout_task && sed -n '1,150p' validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import itertools,pickle,numpy as np
from scipy.linalg import solve
row=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]; s,g=row['study'],row['gt']; nx,ny=s['nx'],s['ny']; n=nx*ny; z=s['Z_nuisance']; gram=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); proj=lambda a:a-z@solve(gram,z.T@a,assume_a='pos'); ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; print('sensor ranges',ij.min(0),ij.max(0)); print('bank pattern first 48',s['row_bank_id'][:48]); print('V versus axis reaction sum',np.linalg.norm(s['V_reaction_2d']-s['reaction_x'][:,None]-s['reaction_y'][None,:])); c=s['sensor_calibration_sweeps']; b=s['row_bank_id']; y=np.empty(len(b));
for k in range(4):
 m=c['bank_id']==k; x=c['ref_signal'][m]; raw=c['raw_readout'][m]; a,d=np.linalg.lstsq(np.column_stack((x,np.ones_like(x))),raw,rcond=None)[0]; print('bank',k,len(x),a,d,np.max(np.abs(raw-a*x-d))); y[b==k]=(s['y_obs_raw'][b==k]-d)/a
print('calibrated projected target max error',np.max(np.abs(proj(y)-g['r_y'])));
def axis(k, endp, react):
 kap=s['kappa_'+k]; v=s['vel_'+k]; rob=s['robin_'+k]; nn=len(kap); h=s['L'+k]/(nn-endp); A=np.zeros((nn,nn)); i=np.arange(1,nn-1); face=(kap[1:]+kap[:-1])/(2*h*h); A[i,i-1]=-face[:-1]-np.maximum(v[i],0)/h; A[i,i+1]=-face[1:]-np.maximum(-v[i],0)/h; A[i,i]=face[:-1]+face[1:]+abs(v[i])/h; A[0,0]=kap[0]/h**2+rob/h; A[-1,-1]=kap[-1]/h**2+rob/h; A[0,1]=-kap[0]/h**2; A[-1,-2]=-kap[-1]/h**2; return A+np.diag(react*s['reaction_'+k]);
rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']]); records=[]
for endp,react,vterm in itertools.product([0,1],[0,1],[0,1]):
 A=np.kron(np.eye(ny),axis('x',endp,react))+np.kron(axis('y',endp,react),np.eye(nx))+np.diag(vterm*s['V_reaction_2d'].ravel(order='F')); fields=np.array([solve(A+mu*np.eye(n),rhs) for mu in s['mu_shifts']]); sampled=fields[:,idx,:]; f=(fields[-1]@g['w_opt']).reshape((nx,ny),order='F'); ferr=np.linalg.norm(f-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']);
 for order in [0,1]:
  mat=(sampled if order==0 else sampled.transpose(1,0,2)).reshape(-1,rhs.shape[1]); serr=np.linalg.norm(proj(mat)-g['S_tilde'])/np.linalg.norm(g['S_tilde']); records.append((serr,ferr,endp,react,vterm,order));
print('top conventions (relative sensor error, relative field error, endpoint-grid flag, axis reactions flag, V flag, sensor-major flag):'); print(*sorted(records)[:8],sep='\n');
PY
```
- **Container Exit Code**: `0` (`455 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "beta_m97_c01_s00_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_beta_m97_c01_s00.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_beta_m97_c01_s00(st)
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
sensor ranges [1 1] [16 14]
bank pattern first 48 [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
 1 2 3 0 1 2 3 0 1 2 3]
V versus axis reaction sum 6.661338147750939e-16
bank 0 12 1.0000000022216748 -2.7921655285242685e-09 2.404828555477453e-07
bank 1 12 1.000000040594541 -1.0403355780077292e-08 1.4466506077180722e-07
bank 2 12 1.000000029440011 -3.439106732878748e-08 2.0974693398638758e-07
bank 3 12 1.0000000145823815 -2.1090909588839085e-08 1.2110866866370541e-07
calibrated projected target max error 7.150326286620157e-08
top conventions (relative sensor error, relative field error, endpoint-grid flag, axis reactions flag, V flag, sensor-major flag):
(9.373297749948545e-16, 4.654530432029322e-16, 1, 0, 1, 0)
(1.0941400927839217e-15, 6.430892445361256e-16, 1, 1, 0, 0)
(0.0584879770660635, 0.03312615750599489, 0, 0, 1, 0)
(0.05848797706606364, 0.03312615750599519, 0, 1, 0, 0)
(0.23845139573013144, 0.22234147028709916, 1, 1, 1, 0)
(0.26911055961827807, 0.24149860493127506, 0, 1, 1, 0)
(0.41553495284118264, 0.3387017114767082, 0, 0, 0, 0)
(0.5228820396023003, 0.3998225060226465, 1, 0, 0, 0)
```

### Turn 4 — `execute_in_remote_docker` (`68629 ms` LLM)
- **Rationale**: Implement the verified forward conventions and independently certify the constrained optimum. The public validator will catch integration or certificate sign errors; checking for the archive determines whether its record format can be verified directly.
- **Alternatives Weighed**: `Add the axis reaction arrays to the supplied 2D reaction field` (rejected: This double-counts reaction on the public study and produces a 0.238 relative sensor error.)

```bash
set -e
cd /holdout_task
cat > beta_m97_c01_s00_solver.py <<'PY'
"""Open-Robin reactor tomography and box-simplex KKT inversion."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrated_target(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'])
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    result = np.empty_like(obs)
    archive = None
    for bank in range(4):
        rows = banks == bank
        x, y = ref[rows], raw[rows]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, 1)
            dx = x[j] - x[i]
            distinct = dx != 0.0
            if np.any(distinct):
                slope = np.median((y[j] - y[i])[distinct] / dx[distinct])
                offsets = y - slope * x
                clean = np.abs(offsets - np.median(offsets)) <= 3.0
        prior = None
        if x.size < 2 or not np.all(clean):
            if archive is None:
                archive = json.loads(_CALIBRATION_ARCHIVE.read_text()) if _CALIBRATION_ARCHIVE.is_file() else {}
            record = archive.get(str(bank))
            if record is not None:
                prior = (float(record['gain']), float(record['offset']))
        if np.count_nonzero(clean) >= 4 or (prior is None and np.count_nonzero(clean) >= 2):
            gain, offset = _fit_line(x[clean], y[clean])
        elif prior is not None:
            gain, offset = prior
        else:
            raise ValueError(f'Insufficient calibration data for bank {bank}')
        rows = row_banks == bank
        result[rows] = (obs[rows] - offset) / gain
    return result


def _axis_operator(kappa, velocity, length, robin):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    v = velocity[1:-1]
    lower, upper = -faces.copy(), -faces.copy()
    lower[:-1] -= np.maximum(v, 0) / h
    upper[1:] -= np.maximum(-v, 0) / h
    upper[0], lower[-1] = -kappa[[0, -1]] / h**2
    main = np.empty(n)
    main[1:-1] = faces[:-1] + faces[1:] + np.abs(v) / h
    main[[0, -1]] = kappa[[0, -1]] / h**2 + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis_operator(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis_operator(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    potential = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(potential, format='csc'))
    identity = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.transpose(0, 2, 1).reshape(modes.shape[0], -1).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_index = i + nx * j
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(operator + float(mu) * identity).solve(rhs)
        sampled.append(field[sensor_index])
    return np.vstack(sampled), field


def _box_simplex(design, target, ridge, upper, budget):
    """Feasible active-set QP, releasing bounds with the wrong KKT sign."""
    n = design.shape[1]
    hessian = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    tol = 1e-12
    if budget == 0:
        w = np.zeros(n)
        return w, float(np.min(-linear))
    if budget == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(hessian @ w - linear))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1: lower, +1: upper
    for _ in range(500):
        free = np.flatnonzero(active == 0)
        lo, hi = active < 0, active > 0
        grad = hessian @ w - linear
        if free.size == 0:
            i = np.argmin(np.where(lo, grad, np.inf))
            j = np.argmax(np.where(hi, grad, -np.inf))
            nu = (grad[i] + grad[j]) / 2
            if grad[i] >= grad[j] - tol:
                return w, float(nu)
            active[[i, j]] = 0
            continue

        # Minimize on the current face with its equality constraint intact.
        system = np.ones((free.size + 1, free.size + 1))
        system[:-1, :-1] = hessian[np.ix_(free, free)]
        system[-1, -1] = 0.0
        rhs = np.r_[linear[free] - hessian[free][:, hi] @ w[hi], budget - w[hi].sum()]
        face = np.linalg.solve(system, rhs)
        candidate = face[:-1]
        direction = candidate - w[free]
        ratios = np.full(free.size, np.inf)
        below, above = candidate < -tol, candidate > upper + tol
        ratios[below] = -w[free[below]] / direction[below]
        ratios[above] = (upper - w[free[above]]) / direction[above]
        k = int(np.argmin(ratios))
        if ratios[k] < 1:
            w[free] += max(0.0, ratios[k]) * direction
            active[free[k]] = -1 if below[k] else 1
            w[free[k]] = 0.0 if below[k] else upper
            continue

        w[free] = np.clip(candidate, 0.0, upper)
        nu = -face[-1]
        grad = hessian @ w - linear
        violation = np.where(lo, nu - grad, np.where(hi, grad - nu, 0.0))
        k = int(np.argmax(violation))
        if violation[k] <= tol:
            return w, float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_s00(study: dict) -> dict:
    if (study['recirculation_loop_mode'] != 0
            or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('This solver implements the open-Robin regime')
    target = _calibrated_target(study)
    sensor, field = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    stacked = np.column_stack((target, sensor))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = stacked - z @ np.linalg.solve(gram, z.T @ stacked)
    target, sensor = projected[:, 0], projected[:, 1:]
    ridge, upper, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(sensor, target, ridge, upper, budget)
    residual = sensor @ w - target
    grad = sensor.T @ residual + ridge * w
    objective = 0.5 * (residual @ residual + ridge * (w @ w))
    reduced = grad - nu
    lower_bound = objective - grad @ w + nu * budget + upper * np.minimum(reduced, 0.0).sum()
    cert = {
        'weights': w.tolist(), 'nu_sum': nu,
        'lam_lower': np.maximum(reduced, 0.0).tolist(),
        'lam_upper': np.maximum(-reduced, 0.0).tolist(),
        'primal_objective': float(objective), 'dual_lower_bound': float(lower_bound),
    }
    terminal = (field @ w).reshape((study['nx'], study['ny']), order='F')
    return {'weights': w, 'terminal_field': terminal, 'kkt_certificate': cert}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%f\n' 2>/dev/null || true
```
- **Container Exit Code**: `0` (`424 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75ff32af5ec52f28 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75ff32af (directory).
```

### Turn 5 — `execute_in_remote_docker` (`36019 ms` LLM)
- **Rationale**: The schema validator does not measure terminal-field accuracy or exercise calibration fallback and active bounds. Compare both numerical outputs against the public reference, then test controlled bank drift, missing rows, EMI, and binding budget slices; a failure isolates the branch needing correction.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, tempfile
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/beta_m97_c01_s00_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r=pickle.load(open('public_beta_m97_c01_s00.pkl','rb'))[0]; s,g=r['study'],r['gt']; solve=m.solve_beta_m97_c01_s00
out=solve(s); w=out['weights']; field=out['terminal_field']; cert=out['kkt_certificate']; S=g['S_tilde']; y=g['r_y']; ridge=s['source_ridge']; upper=s['u_box']; budget=s['total_budget']
def gap(S,y,w,nu,ridge,upper,budget):
    d=S.T@(S@w-y)+ridge*w
    return float(d@w-nu*budget-upper*np.minimum(d-nu,0).sum())
we=np.max(np.abs(w-g['w_opt'])); fe=np.linalg.norm(field-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']); pg=gap(S,y,w,cert['nu_sum'],ridge,upper,budget)
print('public weight max error, field relative error, independent KKT gap:',we,fe,pg)
assert we<1e-5 and fe<1e-5 and -1e-10<=pg<1e-6
# Keep the real forward observations fixed while imposing affine bank drift and an uncoupled telemetry change.
base=m._calibrated_target(s); gain=np.array([1.2,.8,1.4,.7]); offset=np.array([.2,-.4,.6,1.]); bank=s['row_bank_id']; cal=s['sensor_calibration_sweeps']; b=cal['bank_id']; x=cal['ref_signal']; raw=gain[b]*x+offset[b]; t=dict(s,y_obs_raw=gain[bank]*base+offset[bank],aux_manifold_telemetry=s['aux_manifold_telemetry']+1000)
with tempfile.TemporaryDirectory() as td:
    m._CALIBRATION_ARCHIVE=Path(td)/'archive_calibration_history.json'
    m._CALIBRATION_ARCHIVE.write_text('not JSON')
    t['sensor_calibration_sweeps']=dict(bank_id=b,ref_signal=x,raw_readout=raw)
    drift=solve(t)
    import json
    m._CALIBRATION_ARCHIVE.write_text(json.dumps({str(i):{'gain':gain[i],'offset':offset[i]} for i in range(4)}))
    keep=(b!=2)|(np.arange(b.size)==np.flatnonzero(b==2)[0]); spiked=raw.copy(); spiked[np.flatnonzero(b==1)[0]]+=8
    t['sensor_calibration_sweeps']=dict(bank_id=b[keep],ref_signal=x[keep],raw_readout=spiked[keep])
    fallback=solve(t)
    errs=[max(np.max(np.abs(res['weights']-w)),np.max(np.abs(res['terminal_field']-field))) for res in (drift,fallback)]
    print('affine drift and injected archive/EMI maximum output deviations:',errs)
    assert max(errs)<1e-10
# Use public design plus perturbed targets to exercise empty/full, vertex, and mixed active sets.
rng=np.random.default_rng(97); worst_gap=worst_feas=0.; slices=set()
for total in (0.,.01,1.,2.6,6.,11.99,12.):
    for _ in range(40):
        obs=y+rng.normal(0,2,size=y.shape)
        v,nu=m._box_simplex(S,obs,ridge,upper,total)
        worst_gap=max(worst_gap,abs(gap(S,obs,v,nu,ridge,upper,total)))
        worst_feas=max(worst_feas,abs(v.sum()-total),max(0,-v.min(),v.max()-upper))
        slices.add((np.any(v==0),np.any(v==upper),np.any((v>0)&(v<upper))))
print('280 QPs: worst absolute KKT gap, feasibility, active-set patterns:',worst_gap,worst_feas,len(slices))
assert worst_gap<1e-9 and worst_feas<1e-10 and len(slices)>=5
PY
```
- **Container Exit Code**: `0` (`533 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public weight max error, field relative error, independent KKT gap: 2.272734997144088e-07 3.076586923770212e-07 8.30519690554754e-08
affine drift and injected archive/EMI maximum output deviations: [3.247402347028583e-15, 2.4702462297909733e-15]
280 QPs: worst absolute KKT gap, feasibility, active-set patterns: 3.552713678800501e-15 8.881784197001252e-16 6

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75ff32af5ec52f28 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75ff32af (directory).
```

### Turn 6 — `submit_completed_task` (`18184 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated schema validation and independent checks of the weights, terminal field, and certificate. Stress tests covered calibration drift, simulated archive fallback, EMI, and six active-set patterns.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:7267'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_m97_c01_s00_solver.py`. The public validator passed. Maximum weight error was 2.27e-7, relative terminal-field error was 3.08e-7, and the independent KKT gap was 8.31e-8. Calibration fallback was tested with a simulated archive; no real archive was available.`

### Turn 7 — `execute_in_remote_docker` (`34248 ms` LLM)
- **Rationale**: The earlier stress test covered one large spike and one missing calibration row pattern. This audit can expose bank-specific projection errors, missed spikes just above the specified threshold, or incorrect KKT multipliers. A failure would identify the branch to fix; passing requires no solver change.
- **Alternatives Weighed**: `Test relay modes, stochastic drift, and within-batch classification` (rejected: This deliverable specifies deterministic open-Robin transport and continuous source inversion; it has no relay, stochastic drift, or classification labels.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
p=Path('/holdout_task'); spec=importlib.util.spec_from_file_location('solver',p/'beta_m97_c01_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.loads((p/'public_beta_m97_c01_s00.pkl').read_bytes())[0]; s,g=item['study'],item['gt']; out=m.solve_beta_m97_c01_s00(s); w=out['weights']; c=out['kkt_certificate']; upper=s['u_box']; ll=np.array(c['lam_lower']); lu=np.array(c['lam_upper']); grad=g['S_tilde'].T@(g['S_tilde']@w-g['r_y'])+s['source_ridge']*w
stationarity=np.max(np.abs(grad-c['nu_sum']-ll+lu)); complementarity=max(np.max(np.abs(ll*w)),np.max(np.abs(lu*(upper-w)))); print('independent stationarity and complementarity:',stationarity,complementarity); assert max(stationarity,complementarity)<1e-6
sensor,_=m._forward(s); z=s['Z_nuisance']; a=np.column_stack((m._calibrated_target(s),sensor)); projected=a-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a); expected=np.column_stack((g['r_y'],g['S_tilde'])); bank=s['row_bank_id']; bank_errors={int(b):float(np.max(np.abs(projected[bank==b]-expected[bank==b]))) for b in np.unique(bank)}; print('projected input max error by bank:',bank_errors); assert max(bank_errors.values())<1e-6
sweeps=s['sensor_calibration_sweeps']; sb=sweeps['bank_id']; ref=sweeps['ref_signal']; raw=sweeps['raw_readout']; base=m._calibrated_target(s); worst=0.
with tempfile.TemporaryDirectory() as tmp:
    m._CALIBRATION_ARCHIVE=Path(tmp)/'calibration.json'; m._CALIBRATION_ARCHIVE.write_text(json.dumps({str(b):{'gain':1.0,'offset':0.0} for b in range(4)}))
    for i in range(raw.size):
        for spike in (-3.01,3.01):
            perturbed=raw.copy(); perturbed[i]+=spike
            t=dict(s,sensor_calibration_sweeps=dict(sweeps,raw_readout=perturbed)); delta=float(np.max(np.abs(m._calibrated_target(t)-base))); worst=max(worst,delta)
    for b in range(4):
        keep=sb!=b
        t=dict(s,sensor_calibration_sweeps={'bank_id':sb[keep],'ref_signal':ref[keep],'raw_readout':raw[keep]}); delta=float(np.max(np.abs(m._calibrated_target(t)-base))); worst=max(worst,delta)
print('worst calibrated-target change across threshold-adjacent EMI and complete single-bank sweep loss:',worst); assert worst<1e-6
PY
```
- **Container Exit Code**: `0` (`645 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
independent stationarity and complementarity: 6.366350567554857e-08 3.867195427925395e-17
projected input max error by bank: {0: 5.88319828409567e-08, 1: 7.15032641984692e-08, 2: 4.652717633391035e-08, 3: 5.371307487678223e-08}
worst calibrated-target change across threshold-adjacent EMI and complete single-bank sweep loss: 1.5812791742675358e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=75ff32af5ec52f28 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_75ff32af (directory).
```

### Turn 8 — `submit_completed_task` (`19198 ms` LLM)
- **Rationale**: The saved artifact passed isolated validation and independent checks of all required outputs. The focused audit found no tolerance failure requiring a solver change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c01_s00_solver.py': 'FILE:7267'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The public validator passed. Maximum weight error was 2.27e-7, relative terminal-field error was 3.08e-7, and the independent KKT gap was 8.31e-8. Calibration fallback passed threshold-adjacent EMI and missing-bank tests using a simulated archive; no real archive was available.`

### Turn 9 — `submit_completed_task` (`23339 ms` LLM)
- **Rationale**: Independent public-reference checks and stress tests covered every required output and the applicable calibration and active-bound slices. The latest audit found no failure that would justify changing the verified artifact.
- **Alternatives Weighed**: `Repeat the focused audit or add relay and classification tests` (rejected: The unchanged artifact already passed the focused audit in an isolated process. Relay modes and classification metrics are outside this open-Robin inversion contract.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c01_s00_solver.py`. The public validator passed: maximum weight error was 2.27e-7, relative terminal-field error was 3.08e-7, and the independent KKT gap was 8.31e-8. Calibration fallback passed EMI and missing-bank tests with a simulated archive; no real archive was available.`

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
