# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Localized Premise Counterexample: Dual-Channel Separable + Non-Separable 2D Vortex Pocket & Certified Box-Simplex KKT` |
| **Wave / Wormhole Family** | `m93_stage_C_counterexample` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `284,342` (`272,942` / `56,192` / `11,400`) |
| **Total Episode Wall Time (s)** | `439.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`19503 ms` LLM)
- **Rationale**: Inspect the validator and public study shapes to catch sensor-order, ridge-scaling, and certificate-schema mismatches before implementation. Check whether a compatible profiling or KKT primitive is mounted.
- **Alternatives Weighed**: `Reuse Channel A's separable solver for Channel B` (rejected: It omits the specified non-additive cross-potential.)

```bash
cd /holdout_task && sed -n '1,260p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_suc02_C_studies.pkl','rb') as f: p=pickle.load(f)
def show(o,d=0):
 if isinstance(o,dict) and d<3:
  for k,v in o.items():
   print(' '*d,repr(k),end=': '); show(v,d+1)
 elif isinstance(o,(list,tuple)) and d<2:
  print(type(o).__name__,len(o));
  if o: show(o[0],d+1)
 elif hasattr(o,'shape'):
  print(type(o).__name__,o.shape,getattr(o,'dtype',None), (o.min(),o.max()) if isinstance(o,np.ndarray) and o.size and o.dtype.kind in 'fiu' else '')
 else: print(type(o).__name__,str(o)[:190])
show(p)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT' 2>&1 || true
```
- **Container Exit Code**: `0` (`657 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`23574 ms` LLM)
- **Rationale**: Recover the exact input keys and certificate conventions from the truncated output. If the ridge helper returns only fitted residuals rather than square-root-profiled residuals, build the profile explicitly and compare it with the public reference before choosing the implementation.

```bash
cd /holdout_task && sed -n '48,180p' validate_public_schema.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_suc02_C_studies.pkl','rb'))
def inspect(x,name='root',d=0):
 if isinstance(x,dict):
  print(name, list(x))
  for k,v in x.items():
   if isinstance(v,(list,tuple,dict)) and d<2: inspect(v,k,d+1)
   elif hasattr(v,'shape'): print(k,v.shape,v.dtype, repr(v.flat[0]) if v.size else '')
   elif np.isscalar(v): print(k,repr(v))
 elif isinstance(x,(list,tuple)):
  print(name,len(x)); inspect(x[0],name+'[0]',d+1)
inspect(p)
PY
```
- **Container Exit Code**: `0` (`643 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
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
root 1
root[0] ['study', 'gt']
study ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'vel_x', 'robin_x', 'kappa_y', 'vel_y', 'robin_y', 'mu_shifts', 'source_modes', 'sensor_ij', 'Z_nuisance', 'y_obs', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget', 'V_pocket_channel_b']
Lx 2.2
Ly 1.9
nx 36
ny 32
kappa_x (36,) float64 0.5986320816657817
vel_x (36,) float64 0.49045868912971896
robin_x 1.35
kappa_y (32,) float64 0.3860441896139749
vel_y (32,) float64 -0.3204815273327803
robin_y 1.15
mu_shifts (8,) float64 0.08
source_modes (12, 36, 32) float64 1.519258418264986e-20
sensor_ij (36, 2) int64 12
Z_nuisance (576, 5) float64 0.5855255825815593
y_obs (576,) float64 2.78151795899176
nuisance_ridge 0.65
source_ridge 0.04
u_box 1.0
total_budget 2.6
V_pocket_channel_b (36, 32) float64 0.5306944064663008
gt ['Ax', 'By', 'S_tilde', 'r_y', 'w_opt', 'cert_opt', 'terminal_field_A', 'terminal_field_B']
Ax (36, 36) float64 187.11445579971053
By (32, 32) float64 136.02765360485841
S_tilde (576, 12) float64 0.0034825893804771
r_y (576,) float64 0.025937869361801535
w_opt (12,) float64 0.19980285615552668
terminal_field_A (36, 32) float64 0.0010157997496101022
terminal_field_B (36, 32) float64 0.000782242008871285
```

### Turn 3 — `execute_in_remote_docker` (`41090 ms` LLM)
- **Rationale**: The study supplies axis coefficients rather than assembled matrices. Compare the public diagonals with candidate spacing, advection, and Robin conventions to reconstruct the stencil; compare both nuisance transforms against r_y to identify the required objective.
- **Alternatives Weighed**: `Use the mounted ridge fit residual directly` (rejected: It does not produce the square-root profile; compare both transformations against the public target before selecting the convention.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_suc02_C_studies.pkl','rb'))[0]; s,g=p['study'],p['gt']
for a,key in [('x','Ax'),('y','By')]:
 A=g[key]; n=len(A); L=s['L'+a]; k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]; print(a,n,L,r)
 for i in [0,1,2,3,n//2,n-2,n-1]:
  js=np.flatnonzero(A[i]); sl=slice(max(0,i-1),i+2); print(i,js,A[i,js], 'rowsum',A[i].sum(),'k',k[sl],'v',v[sl]);
 for h in [L/n,L/(n-1),L/(n+1)]:
  face=(k[1:]+k[:-1])/2/h**2; lo=A.diagonal(-1); up=A.diagonal(1)
  print('h',h,'lower_diff',np.round((lo+face)[:5],5),'upper_diff',np.round((up+face)[:5],5),'v/h',np.round(v[:5]/h,5),'sum_diff',np.round((lo+up+2*face)[:5],5))
Z=s['Z_nuisance']; y=s['y_obs']; lam=s['nuisance_ridge']; U,d,_=np.linalg.svd(Z,full_matrices=False); yz=U.T@y
for name,yp in [('residual',y-U@((d*d/(d*d+lam))*yz)),('sqrt',y-U@((1-np.sqrt(lam/(d*d+lam)))*yz))]: print(name,np.max(abs(yp-g['r_y'])))
V=s['V_pocket_channel_b']; additive=V.mean(0)+V.mean(1)[:,None]-V.mean(); print('nonadditive relative Frobenius',np.linalg.norm(V-additive)/np.linalg.norm(V)); print('shift range',s['mu_shifts'][[0,-1]])
PY
sed -n '36,52p' validate_public_schema.py
```
- **Container Exit Code**: `0` (`635 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
113.72429475  230.04772452 -116.32342977] rowsum 1.4210854715202004e-14 k [0.42104897 0.38263249 0.39711194] v [-0.33180787 -0.34269895 -0.35656067]
16 [15 16 17] [-145.95609388  306.18662361 -160.23052973] rowsum 2.842170943040401e-14 k [0.5103644  0.51880939 0.55016813] v [-0.51951847 -0.51951847 -0.51569403]
30 [29 30 31] [-107.16603651  225.36540432 -118.19936781] rowsum 2.842170943040401e-14 k [0.37941672 0.37620125 0.42122771] v [-0.33180787 -0.32430597 -0.32048153]
31 [30 31] [-112.73737259  136.05125607] rowsum 23.313883486313287 k [0.37620125 0.42122771] v [-0.32430597 -0.32048153]
h 0.059375 lower_diff [0.10453 0.01982 0.26044 0.03813 0.96226] upper_diff [-5.29305 -5.44217 -5.3279  -5.73364 -5.04297] v/h [-5.39758 -5.462   -5.58834 -5.77177 -6.00523] sum_diff [-5.18852 -5.42235 -5.06746 -5.6955  -4.08071]
h 0.06129032258064516 lower_diff [-6.84392 -7.23407 -6.75229 -6.76573 -6.63876] upper_diff [-12.2415  -12.69607 -12.34063 -12.5375  -12.64399] v/h [-5.22891 -5.29131 -5.41371 -5.5914  -5.81757] sum_diff [-19.08542 -19.93014 -19.09292 -19.30323 -19.28276]
h 0.05757575757575757 lower_diff [7.27357 7.504   7.4958  7.05799 8.80459] upper_diff [1.87598 2.042   1.90746 1.28622 2.79936] v/h [-5.56626 -5.63268 -5.76298 -5.95214 -6.1929 ] sum_diff [ 9.14955  9.546    9.40326  8.34422 11.60394]
residual 6.217248937900877e-15
sqrt 0.22190585008006813
nonadditive relative Frobenius 0.5455482466672369
shift range [0.08 3.4 ]
    if float(np.min(ll)) < -feas_tol or float(np.min(lu)) < -feas_tol:
        return {"valid": False, "reason": "negative_bound_multipliers", "gap": 1e9}

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
```

### Turn 4 — `execute_in_remote_docker` (`83001 ms` LLM)
- **Rationale**: Write the solver and compare the stencil, stacked design, fields, and certificate separately. These checks distinguish boundary-stencil errors from sensor stacking, nuisance profiling, or optimization errors before testing active-bound slices.
- **Alternatives Weighed**: `Solve Channel B with an additive axis potential` (rejected: The public potential retains 54.6% of its Frobenius norm after removing its best additive fit.)
- **ATMS NOGOOD DECLARED**: Refuted `The benchmark uses the square-root Schur-complement profile.` | Witness: `Public r_y agrees with ordinary ridge-fit residuals to 6.22e-15 in max absolute error; the square-root profile differs by 0.222.` | Blanket Diagnosis: `The nuisance-residual transform defines the observation-space objective; swapping it changes the QP even with the same forward fields.` | Sublated Pivot: `Keep ridge nuisance elimination but apply the ordinary residual maker to both the stacked design and the observations.`

```bash
cd /holdout_task && cat > coupled_channel_vortex_kkt_solver.py <<'PY'
"""Dual-channel tracer inversion with a box-simplex optimality certificate."""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_stencil(length, diffusivity, velocity, robin):
    """Cell-centered harmonic diffusion, upwind advection, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    upstream = np.maximum(v, 0) / h
    downstream = np.maximum(-v, 0) / h
    lower = -face - upstream[1:]
    upper = -face - downstream[:-1]
    main = upstream + downstream
    main[1:] += face
    main[:-1] += face
    ends = [0, k.size - 1]
    main[ends] += (robin / h) / (1 + robin * h / (2 * k[ends]))
    return lower, main, upper


def _axis_spectrum(stencil):
    lower, main, upper = stencil
    # D^{-1} A D is symmetric. 'left' and 'right' are inverse transforms.
    d = np.r_[1.0, np.cumprod(np.sqrt(lower / upper))]
    eigenvalues, q = eigh_tridiagonal(main, -np.sqrt(lower * upper))
    return eigenvalues, q.T / d, d[:, None] * q


def _responses(study):
    sources = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = sources.shape
    i, j = np.asarray(study['sensor_ij']).T
    sx, sy = (_axis_stencil(study['L' + axis], study['kappa_' + axis],
                           study['vel_' + axis], study['robin_' + axis])
              for axis in ('x', 'y'))
    ex, lx, rx = _axis_spectrum(sx)
    ey, ly, ry = _axis_spectrum(sy)
    spectral_sources = lx @ sources @ ly.T
    eigenvalue_sum = ex[:, None] + ey

    ax = diags(sx, [-1, 0, 1], format='csc')
    by = diags(sy, [-1, 0, 1], format='csc')
    base = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    pocket = np.asarray(study['V_pocket_channel_b'], dtype=float).ravel()
    rhs = sources.reshape(m, nx * ny).T
    a_rows, b_rows = [], []
    for shift in study['mu_shifts']:
        a = rx @ (spectral_sources / (eigenvalue_sum + shift)) @ ry.T
        factor = splu(base + diags(pocket + shift, format='csc'))
        b = factor.solve(rhs).T.reshape(m, nx, ny)
        a_rows.append(a[:, i, j].T)
        b_rows.append(b[:, i, j].T)
    return np.vstack(a_rows + b_rows), a, b


def _profile(design, target, nuisance, ridge):
    # The survey objective uses ordinary ridge-fit residuals, not the
    # square-root Schur-complement transform of a joint ridge objective.
    u, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    if ridge == 0:
        cutoff = np.finfo(float).eps * max(nuisance.shape) * singular.max(initial=0)
        shrink = singular > cutoff
    else:
        shrink = singular**2 / (singular**2 + ridge)
    data = np.column_stack((design, target))
    residual = data - u @ (shrink[:, None] * (u.T @ data))
    return residual[:, :-1], residual[:, -1]


def _box_simplex_qp(design, target, ridge, cap, budget):
    """Optimize on equality-constrained faces, adding/releasing box bounds."""
    n = design.shape[1]
    if budget == 0:
        return np.zeros(n)
    if budget == n * cap:
        return np.full(n, cap)

    h = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1: lower, +1: upper
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if free.size == 0:
            g = h @ w - linear
            upper = np.flatnonzero(active == 1)
            lower = np.flatnonzero(active == -1)
            i, j = upper[np.argmax(g[upper])], lower[np.argmin(g[lower])]
            if g[i] <= g[j] + 1e-12:
                return w
            active[[i, j]] = 0
            continue

        k = free.size
        system = np.empty((k + 1, k + 1))
        system[:k, :k] = h[np.ix_(free, free)]
        system[:k, k] = system[k, :k] = 1.0
        system[k, k] = 0.0
        rhs = np.r_[linear[free] - h[np.ix_(free, fixed)] @ w[fixed],
                    budget - w[fixed].sum()]
        solution = np.linalg.solve(system, rhs)
        trial, nu = solution[:-1], -solution[-1]
        current = w[free]
        below, above = trial < -1e-13, trial > cap + 1e-13
        if np.any(below | above):
            fraction = np.full(k, np.inf)
            fraction[below] = current[below] / (current[below] - trial[below])
            fraction[above] = (cap - current[above]) / (trial[above] - current[above])
            hit = np.argmin(fraction)
            w[free] = np.clip(current + fraction[hit] * (trial - current), 0, cap)
            active[free[hit]] = -1 if below[hit] else 1
            w[free[hit]] = 0.0 if below[hit] else cap
            continue

        w[free] = np.clip(trial, 0, cap)
        if fixed.size == 0:
            return w
        g = h @ w - linear
        wrong_sign = active[fixed] * (g[fixed] - nu)
        release = np.argmax(wrong_sign)
        if wrong_sign[release] <= 1e-12:
            return w
        active[fixed[release]] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, target, ridge, cap, budget, w):
    residual = design @ w - target
    objective = 0.5 * (float(residual @ residual) + ridge * float(w @ w))
    gradient = design.T @ residual + ridge * w
    # The linearized box-simplex problem fills coordinates in gradient order.
    # Its cutoff gradient maximizes the tangential lower bound over nu.
    cutoff = 0 if cap == 0 else int(np.ceil(budget / cap)) - 1
    nu = float(np.sort(gradient)[np.clip(cutoff, 0, w.size - 1)])
    reduced = gradient - nu
    lower = objective - float(gradient @ w) + nu * budget
    lower += cap * float(np.minimum(0.0, reduced).sum())
    return {'weights': w.copy(), 'nu_sum': nu,
            'lam_lower': np.maximum(reduced, 0.0),
            'lam_upper': np.maximum(-reduced, 0.0),
            'primal_objective': objective,
            'dual_lower_bound': min(objective, lower)}


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Return source weights, last-shift fields, and a KKT certificate."""
    design, a, b = _responses(study)
    design, target = _profile(design, np.asarray(study['y_obs']).ravel(),
                              study['Z_nuisance'], study['nuisance_ridge'])
    ridge, cap, budget = (float(study[key]) for key in
                          ('source_ridge', 'u_box', 'total_budget'))
    weights = _box_simplex_qp(design, target, ridge, cap, budget)
    return {'weights': weights,
            'terminal_field_channel_a': np.einsum('m,mij->ij', weights, a),
            'terminal_field_channel_b': np.einsum('m,mij->ij', weights, b),
            'kkt_certificate': _certificate(design, target, ridge, cap, budget, weights)}
PY
python3 -I validate_public_schema.py
python3 - <<'PY'
import pickle
import numpy as np
from coupled_channel_vortex_kkt_solver import _axis_stencil, _responses, _profile, solve_coupled_channel_vortex_inversion
s,g=pickle.load(open('public_suc02_C_studies.pkl','rb'))[0].values()
for axis,key in [('x','Ax'),('y','By')]:
    lo,d,up = _axis_stencil(s['L'+axis],s['kappa_'+axis],s['vel_'+axis],s['robin_'+axis]); mat=np.diag(lo,-1)+np.diag(d)+np.diag(up,1)
    print(key,'max stencil error',np.max(abs(mat-g[key])))
D,a,b=_responses(s); S,y=_profile(D,s['y_obs'],s['Z_nuisance'],s['nuisance_ridge']); print('profile max errors',np.max(abs(S-g['S_tilde'])),np.max(abs(y-g['r_y'])))
out=solve_coupled_channel_vortex_inversion(s)
for k,l in [('weights','w_opt'),('terminal_field_channel_a','terminal_field_A'),('terminal_field_channel_b','terminal_field_B')]:
    print(k,'max error',np.max(abs(out[k]-g[l])), 'relative L2', np.linalg.norm(out[k]-g[l])/np.linalg.norm(g[l]))
c=out['kkt_certificate']; print('certificate gap', c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`1355 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_channel_vortex_kkt_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Ax max stencil error 5.684341886080802e-14
By max stencil error 5.684341886080802e-14
profile max errors 2.5049406993105094e-15 4.884981308350689e-15
weights max error 1.2406742300186124e-14 relative L2 4.017997558232106e-14
terminal_field_channel_a max error 5.776629175002768e-16 relative L2 3.434125930202836e-14
terminal_field_channel_b max error 2.8796409701215e-16 relative L2 1.9019901453004693e-14
certificate gap 8.326672684688674e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8d6a5ce421b464 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c8d6a5c (directory).
```

### Turn 5 — `execute_in_remote_docker` (`43415 ms` LLM)
- **Rationale**: The public optimum has no active bounds. Check endpoint budgets and a forced all-bound vertex against an independent supporting-hyperplane minimizer; a gap failure would isolate the active-set or certificate logic. Compare both forward maps against independent solves at every shift, including a stronger nonseparable potential, to catch indexing or operator errors that the public terminal-field comparison might miss.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/coupled_channel_vortex_kkt_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_suc02_C_studies.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
Ax, By = gt['Ax'], gt['By']
sources = s['source_modes']
m, nx, ny = sources.shape
ij = tuple(s['sensor_ij'].T)
rhs = sources.reshape(m, -1).T
base = kron(csr_matrix(Ax), eye(ny), format='csc') + kron(eye(nx), csr_matrix(By), format='csc')
Z = s['Z_nuisance']
profile = lambda x: x - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ x)
max_field = max_design = max_gap = max_feas = 0.0
for pocket_scale in (1.0, 3.0):
    potential = s['V_pocket_channel_b'] * pocket_scale
    sa, sb = [], []
    for shift in s['mu_shifts']:
        a = np.stack([solve_sylvester(Ax + shift * np.eye(nx), By.T, src) for src in sources])
        b = spsolve(base + diags(potential.ravel() + shift, format='csc'), rhs).T.reshape(m, nx, ny)
        sa.append(a[(slice(None),) + ij].T)
        sb.append(b[(slice(None),) + ij].T)
    raw = np.vstack(sa + sb)
    design = profile(raw)
    model_raw, _, _ = mod._responses(dict(s, V_pocket_channel_b=potential))
    max_design = max(max_design, np.max(np.abs(model_raw - raw)))
    assert np.allclose(model_raw, raw, rtol=1e-9, atol=1e-12)
    contrast = np.linspace(-3, 3, m)
    y_obs = s['y_obs'] + 120 * (design @ contrast)
    target = profile(y_obs)
    cap = 0.4
    for budget in (0.0, 0.09, 2 * cap, 2.6, m * cap - 0.09, m * cap):
        out = mod.solve_coupled_channel_vortex_inversion(dict(s, V_pocket_channel_b=potential, y_obs=y_obs, u_box=cap, total_budget=budget))
        w = out['weights']
        cert = out['kkt_certificate']
        assert np.array_equal(w, cert['weights']) and np.min(w) >= 0 and np.max(w) <= cap
        max_feas = max(max_feas, abs(w.sum() - budget))
        assert abs(w.sum() - budget) < 1e-12
        residual = design @ w - target
        g = design.T @ residual + s['source_ridge'] * w
        j = (residual @ residual + s['source_ridge'] * (w @ w)) / 2
        # Independent linear minimizer for the supporting hyperplane.
        fill = np.clip(budget - cap * np.arange(m), 0, cap)
        gap = max(j - cert['dual_lower_bound'], g @ w - np.sort(g) @ fill)
        assert -1e-10 <= j - cert['dual_lower_bound'] <= 1e-6
        assert gap <= 1e-6 and abs(j - cert['primal_objective']) < 1e-10
        assert np.max(np.abs(g - cert['nu_sum'] - cert['lam_lower'] + cert['lam_upper'])) < 1e-9
        lo, hi = np.count_nonzero(w < 1e-12), np.count_nonzero(w > cap - 1e-12)
        if budget == 2 * cap:
            assert (lo, hi) == (m - 2, 2), (lo, hi)
        for key, fields in [('terminal_field_channel_a', a), ('terminal_field_channel_b', b)]:
            ref = np.tensordot(w, fields, axes=1)
            max_field = max(max_field, np.max(np.abs(out[key] - ref)))
            assert np.allclose(out[key], ref, atol=1e-12, rtol=1e-9)
        max_gap = max(max_gap, gap)
        print(f'pocket={pocket_scale:g} budget={budget:g} active lower/upper={lo}/{hi} gap={gap:.3e}')
print(f'worst: design={max_design:.3e}, terminal fields={max_field:.3e}, budget error={max_feas:.3e}, KKT gap={max_gap:.3e}')
PY
```
- **Container Exit Code**: `1` (`617 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pocket=1 budget=0 active lower/upper=12/0 gap=5.684e-14
pocket=1 budget=0.09 active lower/upper=11/0 gap=0.000e+00

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8d6a5ce421b464 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c8d6a5c (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 58, in <module>
AssertionError: (9, 1)
```

### Turn 6 — `execute_in_remote_docker` (`47427 ms` LLM)
- **Rationale**: Keep the contrast cases and add a separate vertex case whose prescribed gradient proves that its exact optimum has two upper bounds and ten lower bounds. Independent forward solves and KKT checks will distinguish a real solver defect from the invalid active-count assumption.
- **Alternatives Weighed**: `Change the optimizer to force an all-bound solution at an integer-multiple budget` (rejected: That budget permits a vertex but does not require one. The failed assertion checked the number of active bounds, not optimality.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/coupled_channel_vortex_kkt_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_suc02_C_studies.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
Ax, By = gt['Ax'], gt['By']
sources = s['source_modes']
m, nx, ny = sources.shape
ij = tuple(s['sensor_ij'].T)
rhs = sources.reshape(m, -1).T
base = kron(csr_matrix(Ax), eye(ny), format='csc') + kron(eye(nx), csr_matrix(By), format='csc')
Z = s['Z_nuisance']
normal = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
def profile(x):
    return x - Z @ np.linalg.solve(normal, Z.T @ x)

max_field = max_design = max_gap = max_feas = max_stationarity = 0.0
cap, ridge = 0.4, s['source_ridge']
for pocket_scale in (1.0, 3.0):
    potential = s['V_pocket_channel_b'] * pocket_scale
    sa, sb = [], []
    for shift in s['mu_shifts']:
        a = np.stack([solve_sylvester(Ax + shift * np.eye(nx), By.T, src) for src in sources])
        b = spsolve(base + diags(potential.ravel() + shift, format='csc'), rhs).T.reshape(m, nx, ny)
        sa.append(a[(slice(None),) + ij].T)
        sb.append(b[(slice(None),) + ij].T)
    raw = np.vstack(sa + sb)
    design = profile(raw)
    model_raw, _, _ = mod._responses(dict(s, V_pocket_channel_b=potential))
    max_design = max(max_design, np.max(np.abs(model_raw - raw)))
    assert np.allclose(model_raw, raw, rtol=1e-9, atol=1e-12)
    contrast_y = s['y_obs'] + 120 * (design @ np.linspace(-3, 3, m))
    cases = [('contrast', budget, contrast_y) for budget in (0.0, 0.09, 2 * cap, 2.6, m * cap - 0.09, m * cap)]
    vertex = np.zeros(m)
    vertex[[1, 8]] = cap
    desired_gradient = np.where(vertex > 0, -0.05, 0.05)
    coeff = np.linalg.solve(design.T @ design, ridge * vertex - desired_gradient)
    vertex_y = raw @ (vertex + coeff)
    assert np.allclose(design.T @ (design @ vertex - profile(vertex_y)) + ridge * vertex, desired_gradient, atol=1e-10)
    cases.append(('known_vertex', 2 * cap, vertex_y))
    for name, budget, y_obs in cases:
        out = mod.solve_coupled_channel_vortex_inversion(dict(s, V_pocket_channel_b=potential, y_obs=y_obs, u_box=cap, total_budget=budget))
        w = out['weights']
        cert = out['kkt_certificate']
        assert np.array_equal(w, cert['weights']) and np.min(w) >= 0 and np.max(w) <= cap
        err = abs(w.sum() - budget)
        max_feas = max(max_feas, err)
        assert err < 1e-12
        residual = design @ w - profile(y_obs)
        g = design.T @ residual + ridge * w
        j = (residual @ residual + ridge * (w @ w)) / 2
        fill = np.clip(budget - cap * np.arange(m), 0, cap)
        gap = max(j - cert['dual_lower_bound'], g @ w - np.sort(g) @ fill)
        assert -1e-10 <= j - cert['dual_lower_bound'] <= 1e-6
        assert gap <= 1e-6 and abs(j - cert['primal_objective']) < 1e-10
        ll, lu = cert['lam_lower'], cert['lam_upper']
        stationarity = np.max(np.abs(g - cert['nu_sum'] - ll + lu))
        max_stationarity = max(max_stationarity, stationarity)
        assert stationarity < 1e-9 and min(ll.min(), lu.min()) >= 0
        assert max(np.max(np.abs(ll * w)), np.max(np.abs(lu * (cap - w)))) < 1e-9
        if name == 'known_vertex':
            assert np.allclose(w, vertex, atol=1e-12, rtol=0), w
        for key, fields in [('terminal_field_channel_a', a), ('terminal_field_channel_b', b)]:
            ref = np.tensordot(w, fields, axes=1)
            max_field = max(max_field, np.max(np.abs(out[key] - ref)))
            assert np.allclose(out[key], ref, atol=1e-12, rtol=1e-9)
        max_gap = max(max_gap, gap)
        print(f'pocket={pocket_scale:g} {name} budget={budget:g} active lower/upper={np.count_nonzero(w < 1e-12)}/{np.count_nonzero(w > cap-1e-12)} gap={gap:.3e}')
print(f'worst: design={max_design:.3e}, terminal fields={max_field:.3e}, budget error={max_feas:.3e}, KKT gap={max_gap:.3e}, stationarity={max_stationarity:.3e}')
PY
```
- **Container Exit Code**: `0` (`983 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pocket=1 contrast budget=0 active lower/upper=12/0 gap=5.684e-14
pocket=1 contrast budget=0.09 active lower/upper=11/0 gap=0.000e+00
pocket=1 contrast budget=0.8 active lower/upper=9/1 gap=1.137e-13
pocket=1 contrast budget=2.6 active lower/upper=5/6 gap=5.684e-14
pocket=1 contrast budget=4.71 active lower/upper=0/11 gap=1.705e-13
pocket=1 contrast budget=4.8 active lower/upper=0/12 gap=1.137e-13
pocket=1 known_vertex budget=0.8 active lower/upper=10/2 gap=7.105e-15
pocket=3 contrast budget=0 active lower/upper=12/0 gap=0.000e+00
pocket=3 contrast budget=0.09 active lower/upper=11/0 gap=0.000e+00
pocket=3 contrast budget=0.8 active lower/upper=9/1 gap=5.684e-14
pocket=3 contrast budget=2.6 active lower/upper=5/6 gap=0.000e+00
pocket=3 contrast budget=4.71 active lower/upper=0/11 gap=1.137e-13
pocket=3 contrast budget=4.8 active lower/upper=0/12 gap=1.705e-13
pocket=3 known_vertex budget=0.8 active lower/upper=10/2 gap=1.421e-14
worst: design=1.360e-15, terminal fields=6.314e-16, budget error=0.000e+00, KKT gap=1.705e-13, stationarity=6.106e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8d6a5ce421b464 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c8d6a5c (directory).
```

### Turn 7 — `submit_completed_task` (`13408 ms` LLM)
- **Rationale**: The exact submitted artifact passed the public schema validator and isolated-process checks against independent forward solves. Stress cases covered endpoint budgets, mixed active bounds, a known all-bound optimum, and a threefold stronger cross-potential.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:6854'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public validator passed, and all 14 stress cases passed with a worst KKT gap of 1.705e-13 (limit: 1e-6).`

### Turn 8 — `execute_in_remote_docker` (`31889 ms` LLM)
- **Rationale**: Use the public operators to check both terminal fields against their governing equations. Hold the weights fixed when replacing the pocket with its best additive fit so the comparison isolates the cross-potential rather than a change in the optimizer. A failed exact-field residual would warrant revisiting channel assembly; a failed tangent gap would warrant revisiting the profiled objective or optimizer.
- **Alternatives Weighed**: `Run classification, cohort, or relay-mode checks` (rejected: The public study is a continuous inverse problem with no class labels, cohort variable, or relay state.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/coupled_channel_vortex_kkt_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('public_suc02_C_studies.pkl', 'rb') as f:
    studies = pickle.load(f)
for item in studies:
    s, gt = item['study'], item['gt']
    v = s['V_pocket_channel_b']
    additive = v.mean(axis=0) + v.mean(axis=1)[:, None] - v.mean()
    out = solver.solve_coupled_channel_vortex_inversion(s)
    _, _, b_add = solver._responses(dict(s, V_pocket_channel_b=additive))
    w = out['weights']
    a, b = (out['terminal_field_channel_' + key] for key in ('a', 'b'))
    b_surrogate = np.tensordot(w, b_add, axes=1)
    source = np.tensordot(w, s['source_modes'], axes=1)
    shift = s['mu_shifts'][-1]
    def pde_err(field, potential):
        return np.linalg.norm(gt['Ax'] @ field + field @ gt['By'].T + (potential + shift) * field - source) / np.linalg.norm(source)
    err_a, err_b, err_add = pde_err(a, 0), pde_err(b, v), pde_err(b_surrogate, v)
    residual = gt['S_tilde'] @ w - gt['r_y']
    g = gt['S_tilde'].T @ residual + s['source_ridge'] * w
    fill = np.clip(s['total_budget'] - s['u_box'] * np.arange(w.size), 0, s['u_box'])
    public_gap = max(0.0, g @ w - np.sort(g) @ fill)
    print(f'Public PDE relative residual A={err_a:.3e} B={err_b:.3e} B-additive={err_add:.3e}; public tangent gap={public_gap:.3e}')
    print(f'Additive B terminal relative error={np.linalg.norm(b_surrogate-b)/np.linalg.norm(b):.3e}; nonadditive potential fraction={np.linalg.norm(v-additive)/np.linalg.norm(v):.3e}')
    assert max(err_a, err_b) < 1e-10 and public_gap <= 1e-6
PY
```
- **Container Exit Code**: `0` (`743 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public PDE relative residual A=6.017e-14 B=1.453e-14 B-additive=1.695e-01; public tangent gap=1.638e-15
Additive B terminal relative error=6.025e-02; nonadditive potential fraction=5.455e-01

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8d6a5ce421b464 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c8d6a5c (directory).
```

### Turn 9 — `submit_completed_task` (`27878 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process validation against independent forward solves and KKT checks covering endpoint budgets, mixed active bounds, and known vertex optima. The public audit confirmed that retaining the cross-potential is necessary; no further edit is warranted.
- **Alternatives Weighed**: `Replace the Channel B pocket with its best additive fit` (rejected: At fixed weights, the additive fit gave a 6.025% terminal-field error and a 0.1695 relative residual under the full public operator.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:6854'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public validator and all 14 stress cases passed. The worst KKT gap was 1.705e-13, below the 1e-6 limit. On the public study, the exact Channel B solve reduced the relative PDE residual from 0.1695 for the additive approximation to 1.453e-14.`

### Turn 10 — `execute_in_remote_docker` (`29709 ms` LLM)
- **Rationale**: Rotate the public nuisance basis and consistently reorder sensors. Neither operation should change the inverse problem. A rotation failure would implicate ridge profiling; a permutation failure would implicate sensor stacking or row alignment. Check each certificate against the unchanged public objective.
- **Alternatives Weighed**: `Run cohort classification or relay-mode checks` (rejected: The study has continuous observations and no class labels, cohort variable, or relay state.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/coupled_channel_vortex_kkt_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
rng = np.random.default_rng(107)
with open('public_suc02_C_studies.pkl', 'rb') as f:
    cases = pickle.load(f)
worst_weights = worst_fields = worst_gap = 0.0
for case in cases:
    s, gt = case['study'], case['gt']
    z = s['Z_nuisance']
    q, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
    ns = len(s['sensor_ij'])
    p = rng.permutation(ns)
    rows = (ns * np.arange(len(s['y_obs']) // ns)[:, None] + p).ravel()
    ref = solver.solve_coupled_channel_vortex_inversion(s)
    for changed in (dict(s, Z_nuisance=z @ q), dict(s, sensor_ij=s['sensor_ij'][p], Z_nuisance=z[rows], y_obs=s['y_obs'][rows])):
        out = solver.solve_coupled_channel_vortex_inversion(changed)
        w = out['weights']
        weight_error = np.max(np.abs(w - ref['weights']))
        field_error = max(np.max(np.abs(out[k] - ref[k])) for k in ('terminal_field_channel_a', 'terminal_field_channel_b'))
        c = out['kkt_certificate']
        residual = gt['S_tilde'] @ w - gt['r_y']
        g = gt['S_tilde'].T @ residual + s['source_ridge'] * w
        primal = 0.5 * (residual @ residual + s['source_ridge'] * (w @ w))
        fill = np.clip(s['total_budget'] - s['u_box'] * np.arange(w.size), 0, s['u_box'])
        tangent_lb = primal - (g @ w - np.sort(g) @ fill)
        gap = primal - min(c['dual_lower_bound'], tangent_lb)
        assert weight_error < 1e-10 and field_error < 1e-10
        assert abs(c['primal_objective'] - primal) < 1e-10 and -1e-9 <= gap <= 1e-6
        worst_weights = max(worst_weights, weight_error)
        worst_fields = max(worst_fields, field_error)
        worst_gap = max(worst_gap, gap)
print(f'Public nuisance rotation and sensor permutation: worst weight change={worst_weights:.3e}, field change={worst_fields:.3e}, KKT gap={worst_gap:.3e}')
PY
```
- **Container Exit Code**: `0` (`702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public nuisance rotation and sensor permutation: worst weight change=1.832e-15, field change=4.684e-17, KKT gap=1.860e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c8d6a5ce421b464 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c8d6a5c (directory).
```

### Turn 11 — `submit_completed_task` (`30561 ms` LLM)
- **Rationale**: The verified artifact passed independent forward and KKT checks at endpoint budgets, mixed active bounds, and known vertex optima. Rotating the public nuisance basis or consistently permuting sensor rows changed weights by at most 1.832e-15. No further edit is warranted.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public validator and all 14 stress cases passed. The worst KKT gap was 1.705e-13, below the 1e-6 limit. On the public study, the exact Channel B solver reduced the relative PDE residual from 0.1695 for the additive approximation to 1.453e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_channel_vortex_kkt_solver.py": "FILE:6854"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_weights_rel_l2": 0.0,
    "max_field_a_rel_l2": 0.0,
    "max_field_b_rel_l2": 0.0,
    "max_kkt_gap": 0.0,
    "student_elapsed_sec": 0.5852,
    "instance_diagnostics": [
      {
        "seed": 99621,
        "w_rel_l2": 0.0,
        "fa_rel_l2": 0.0,
        "fb_rel_l2": 0.0,
        "kkt_gap": 0.0
      },
      {
        "seed": 99622,
        "w_rel_l2": 0.0,
        "fa_rel_l2": 0.0,
        "fb_rel_l2": 0.0,
        "kkt_gap": 0.0
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.043856,
      "import_cpu_sec": 0.040291,
      "candidate_call_wall_sec": 0.045646,
      "candidate_call_cpu_sec": 0.041894,
      "ipc_and_codec_wall_sec": 0.137851,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.481486,
      "verifier_and_gt_cpu_sec": 0.476968,
      "total_supervisor_wall_sec": 0.708839
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5852, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_6d28f88bbe328ecb3b60171d5f25b22f__:{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5852, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.043856, \"import_cpu_sec\": 0.040291, \"candidate_call_wall_sec\": 0.045646, \"candidate_call_cpu_sec\": 0.041894, \"ipc_and_codec_wall_sec\": 0.137851, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.481486, \"verifier_and_gt_cpu_sec\": 0.476968, \"total_supervisor_wall_sec\": 0.708839}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6890.5,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
