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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `3` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `240,588` (`230,937` / `45,056` / `9,651`) |
| **Total Episode Wall Time (s)** | `351.82s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_11_separable_transport_inversio`**: On the recorded rank-six slice, recomputing an arbitrary minimum-norm face minimizer moved the newly released coordinate to −1.82e−3, producing a zero-length feasible step and a repeating working set. A shortest least-squares correction in an orthonormal sum-zero basis avoids resetting the iterate along flat directions; the solver accepts a feasible solution only when the supporting-hyperplane gap, maximized over the equality multiplier by a gradient order statistic, is at most 1e−9 by default. For the forward model, diagonally symmetrized tridiagonal eigensystems are reused across all modes and shifts, with sparse direct solves used when the log-similarity span exceeds 25 or any spectral relative residual exceeds 1e−12. The reported three-slice public probe reached worst Sylvester residual 2.81e−13 and KKT gap 1.10e−15; these certify the probed solves, not uniqueness of source weights on a rank-deficient slice. [REGIME SPLIT GUARD `tangent_nullity_of_augmented_design_on_current_free_face > 0` — Counter-Regime Warning: This is an identifiability split, not a transport phase transition. Positive tangent nullity allows objective-flat directions on the current face; certify optimality without claiming unique weights. Zero tangent nullity gives uniqueness on that face but does not replace the global box-simplex certificate.]
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `An arbitrary least-squares minimizer on each enlarged face suffices to make progress after releasing a bound.` (witness: `On rank-six QP slice 845, releasing coordinate 7 sends the trial to -1.82e-3 in that coordinate. The feasible step length is zero, and the same working sets repeat.`) -> Pivot `Use an orthonormal sum-zero basis to solve for the smallest correction from the current iterate, and use the tangent gap to decide whether to stop on a rank-deficient face.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`22017 ms` LLM)
- **Rationale**: Inspect the public contract and study structure to prevent axis, stacking, and certificate mismatches; confirm SciPy availability and look up the ridge-profiling primitive before choosing the implementation.
- **Alternatives Weighed**: `Implement directly from the equation summary` (rejected: The validator and public study can resolve sensor indexing, array layout, and certificate requirements before implementation.)

```bash
cd /holdout_task && sed -n '1,240p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np
with open('public_suc02_C_studies.pkl','rb') as f: data=pickle.load(f)
def show(x,name='data',level=0):
 print(name,type(x).__name__,getattr(x,'shape',''),getattr(x,'dtype',''))
 if level>=3: return
 if isinstance(x,dict):
  for k,v in x.items(): show(v,str(k),level+1)
 elif isinstance(x,(tuple,list)):
  print('length:',len(x))
  if x: show(x[0],name+'[0]',level+1)
 elif np.isscalar(x): print('value:',str(x)[:150])
show(data)
import scipy
print('scipy:',scipy.__version__)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
iled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_separable_transport_inversion@02792fbe | recover_transport_mixture(fields: np.ndarray, sensor_indices: np.ndarray, observations: np.ndarray, nuisance: np.ndarray, cap: float, budget: float, *, nuisance_ridge: float=0.0, source_ridge: float=0.0, gap_tol: float=1e-09) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float] | Fit a linear source mixture with 0 <= w <= cap and sum(w) = budget | REQ: {'field': 'axis_length', 'op': '>', 'value': 0}; {'field': 'cell_diffusivity', 'op': '>' | --describe autogenous_separable_transport_inversion.recover_transport_mixture
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
```

### Turn 2 — `execute_in_remote_docker` (`28363 ms` LLM)
- **Rationale**: Recover the input keys and certificate checks omitted from the truncated output, measure operator asymmetry to choose the Channel A decomposition, and inspect the available primitives before selecting a profiling method.
- **Alternatives Weighed**: `Assume the 1D operators are symmetric and the existing profiling helper supplies the required residual transform.` (rejected: Operator asymmetry would rule out a symmetric eigensolver, and a ridge-fit residual is not necessarily the square-root transform of the profiled objective.)

```bash
cd /holdout_task && sed -n '48,180p' validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_suc02_C_studies.pkl','rb') as f: d=pickle.load(f)
def visit(o,n,depth=0):
 if isinstance(o,dict):
  print(n, 'keys:', list(o))
  if depth<3:
   for k,v in o.items():
    if k not in ('gt','cert_opt'): visit(v,str(k),depth+1)
 elif isinstance(o,(list,tuple)):
  print(n,len(o))
  if len(o): visit(o[0],n+'[0]',depth+1)
 elif isinstance(o,np.ndarray):
  print(n,o.shape,o.dtype,'head:',o.ravel()[:8], 'relative asymmetry:',np.linalg.norm(o-o.T)/max(1,np.linalg.norm(o)) if o.ndim==2 and o.shape[0]==o.shape[1] else '-')
 else: print(n,repr(o)[:120])
visit(d,'public')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_separable_transport_inversion.recover_transport_mixture
```
- **Container Exit Code**: `0` (`650 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,
          "value": 0
        },
        {
          "field": "total_budget",
          "op": ">=",
          "value": 0
        },
        {
          "field": "total_capacity_minus_budget",
          "op": ">=",
          "value": 0
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "recover_transport_mixture",
          "signature": "recover_transport_mixture(fields: np.ndarray, sensor_indices: np.ndarray, observations: np.ndarray, nuisance: np.ndarray, cap: float, budget: float, *, nuisance_ridge: float=0.0, source_ridge: float=0.0, gap_tol: float=1e-09) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float]",
          "doc_summary": "Fit a linear source mixture with 0 <= w <= cap and sum(w) = budget."
        }
      ],
      "index_cards": [
        "autogenous_separable_transport_inversion@02792fbe | cell_transport(length: float, diffusivity: np.ndarray, velocity: np.ndarray, robin: float) -> np.ndarray | Cell-centered diffusion, row-wise upwind advection, and Robin loss | REQ: {'field': 'axis_length', 'op': '>', 'value': 0}; {'field': 'cell_diffusivity', 'op': '>' | --describe autogenous_separable_transport_inversion.cell_transport",
        "autogenous_separable_transport_inversion@02792fbe | shifted_transport_fields(ax: np.ndarray, by: np.ndarray, shifts: np.ndarray, modes: np.ndarray, *, residual_tol: float=1e-12, log_span_limit: float=25.0) -> np.ndarray | Solve Ax U + U By | REQ: {'field': 'axis_length', 'op': '>', 'value': 0}; {'field': 'cell_diffusivity', 'op': '>' | --describe autogenous_separable_transport_inversion.shifted_transport_fields",
        "autogenous_separable_transport_inversion@02792fbe | recover_transport_mixture(fields: np.ndarray, sensor_indices: np.ndarray, observations: np.ndarray, nuisance: np.ndarray, cap: float, budget: float, *, nuisance_ridge: float=0.0, source_ridge: float=0.0, gap_tol: float=1e-09) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float] | Fit a linear source mixture with 0 <= w <= cap and sum(w) = budget | REQ: {'field': 'axis_length', 'op': '>', 'value': 0}; {'field': 'cell_diffusivity', 'op': '>' | --describe autogenous_separable_transport_inversion.recover_transport_mixture"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_separable_transport_inversion",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`39858 ms` LLM)
- **Rationale**: Check the library stencil and Channel A reduction against the public operators and an independent sparse solve. Compare terminal fields, sensor stacking, and ridge conventions so any mismatch is fixed before building the QP. Measure the additive pocket approximation on the actual sensor readouts.
- **Alternatives Weighed**: `Use the ridge square-root transform without checking the public reference.` (rejected: The validator uses the supplied transformed arrays; comparing transforms now will reveal which objective the certificate must match.)

```bash
cd /holdout_task && sed -n '33,52p' validate_public_schema.py && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
import pickle, numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from hegel_wormhole_skills.autogenous_separable_transport_inversion import cell_transport, shifted_transport_fields
with open('public_suc02_C_studies.pkl','rb') as f: item=pickle.load(f)[0]
s,t=item['study'],item['gt']
a,b=[cell_transport(s['L'+d],s['kappa_'+d],s['vel_'+d],s['robin_'+d]) for d in 'xy']
for h,d in zip((a,b),('Ax','By')): print(d,'max operator error',np.max(abs(h-t[d])),'asymmetry',np.max(abs(h-h.T)))
A,B=t['Ax'],t['By']; modes=s['source_modes']; m,nx,ny=modes.shape; ij=s['sensor_ij']; mu=s['mu_shifts']; V=s['V_pocket_channel_b']; z=s['Z_nuisance']; y=s['y_obs']; r=s['nuisance_ridge']; print('sensor shape and first entries',ij.shape,ij[:4]); print('nuisance',z.shape,r,'source ridge, cap, budget',s['source_ridge'],s['u_box'],s['total_budget']); print('gt certificate',t['cert_opt']); print('vortex mixed interaction fraction',np.linalg.norm(V-V.mean(0)-V.mean(1,keepdims=True)+V.mean())/np.linalg.norm(V))
base=sp.kron(A,sp.eye(ny),format='csc')+sp.kron(sp.eye(nx),B,format='csc'); F=[]
for pocket in (np.zeros_like(V),V,V.mean(0)+V.mean(1,keepdims=True)-V.mean()):
 F.append(np.array([splu(base+sp.diags((pocket+k).ravel(),format='csc')).solve(modes.reshape(m,-1).T).T.reshape(m,nx,ny) for k in mu]))
q=shifted_transport_fields(a,b,mu,modes); print('primitive field shape',q.shape,'reference shape',F[0].shape)
if q.shape==F[0].shape: print('primitive field max error',np.max(abs(q-F[0])))
if q.shape==F[0].swapaxes(0,1).shape: print('primitive transposed field max error',np.max(abs(q.swapaxes(0,1)-F[0])))
for j,d in enumerate('AB'): print('oracle-weight terminal error',d,np.max(abs(np.einsum('m,mxy->xy',t['w_opt'],F[j][-1])-t['terminal_field_'+d])))
print('additive pocket relative sensor error',np.linalg.norm((F[2]-F[1])[:,:,ij[:,0],ij[:,1]])/np.linalg.norm(F[1][:,:,ij[:,0],ij[:,1]]))
X=np.concatenate([f[:,:,ij[:,0],ij[:,1]].transpose(0,2,1).reshape(-1,m) for f in F[:2]]); D=np.column_stack((X,y)); T=np.column_stack((t['S_tilde'],t['r_y'])); U,d,_=np.linalg.svd(z,full_matrices=False); coef=U.T@D
for name,P in [('raw',D),('ridge residual',D-U@((d*d/(d*d+r))[:,None]*coef)),('ridge square root',D-U@((1-np.sqrt(r/(d*d+r)))[:,None]*coef))]: print(name,'S error',np.max(abs(P[:,:m]-T[:,:m])),'y error',np.max(abs(P[:,m]-T[:,m])),'augmented Gram error',np.max(abs(P.T@P-T.T@T)))
PY
```
- **Container Exit Code**: `0` (`585 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.5 * float(res @ res) + 0.5 * lw * float(w_feas @ w_feas)
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
Ax max operator error 1.1368683772161603e-13 asymmetry 9.81070958898016
By max operator error 5.684341886080802e-14 asymmetry 8.749784802816322
sensor shape and first entries (36, 2) [[12  4]
 [33 23]
 [28 15]
 [ 3 15]]
nuisance (576, 5) 0.65 source ridge, cap, budget 0.04 1.0 2.6
gt certificate {'weights': [0.19980285615552668, 0.2820880689112017, 0.13680709178655026, 0.19883325186053738, 0.2424978383116777, 0.21359584368263293, 0.28191488767485495, 0.16386085611060675, 0.21194801304194347, 0.23309355773769624, 0.1584592920678694, 0.27709844265890243], 'nu_sum': 0.007586173485640911, 'lam_lower': [2.0816681711721685e-17, 1.3877787807814457e-17, 0.0, 6.938893903907228e-18, 1.3877787807814457e-17, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, 0.0, 0.0, 6.938893903907228e-18, 2.0816681711721685e-17], 'lam_upper': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.938893903907228e-18, 6.938893903907228e-18, 0.0, 0.0], 'primal_objective': 0.05621509876361763, 'dual_lower_bound': 0.056215098763617605}
vortex mixed interaction fraction 0.5455482466672369
primitive field shape (8, 12, 36, 32) reference shape (8, 12, 36, 32)
primitive field max error 3.608224830031759e-16
oracle-weight terminal error A 4.458239333260394e-16
oracle-weight terminal error B 3.469446951953614e-17
additive pocket relative sensor error 0.08925286865402977
raw S error 0.0029749255584734977 y error 6.984212871199382 augmented Gram error 2032.1012279398608
ridge residual S error 2.563921297493721e-15 y error 4.884981308350689e-15 augmented Gram error 5.578870698741412e-15
ridge square root S error 9.847542664847718e-05 y error 0.22190585008006547 augmented Gram error 2.188495656487617
```

### Turn 4 — `execute_in_remote_docker` (`26587 ms` LLM)
- **Rationale**: Read the verified stencil to preserve its boundary and upwind conventions in the standalone artifact. Confirm the isolated SciPy Sylvester interface to avoid a transpose or scale error in the Channel A solver.
- **Alternatives Weighed**: `Use the ridge square-root objective transform` (rejected: It misses the public augmented Gram matrix by 2.19, versus 5.58e-15 for the ridge-residual transform.)
- **ATMS NOGOOD DECLARED**: Refuted `An additive axis-only potential can stand in for the full Channel B pocket.` | Witness: `The public pocket has 54.55% relative interaction norm after removal of additive axis terms; the additive replacement changes the stacked Channel B sensor responses by 8.93% relative L2.` | Blanket Diagnosis: `The mixed spatial reaction term couples the two axes in Channel B; it is absent from Channel A.` | Sublated Pivot: `Keep the separable Channel A reduction and use the exact sparse Channel B operator with the full pocket diagonal.`

```bash
cd /holdout_task && PYTHONPATH=/tmp python3 - <<'PY'
import inspect
from hegel_wormhole_skills.autogenous_separable_transport_inversion import cell_transport
print(inspect.getsource(cell_transport))
PY
python3 -I - <<'PY'
from scipy.linalg import lapack
print(lapack.dtrsyl.__doc__)
PY
```
- **Container Exit Code**: `0` (`1246 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
@track_wormhole_skill("autogenous_separable_transport_inversion")
def cell_transport(length: float, diffusivity: np.ndarray, velocity: np.ndarray, robin: float) -> np.ndarray:
    """Cell-centered diffusion, row-wise upwind advection, and Robin loss.

    Inputs are cell arrays with positive diffusivities; Robin impedance is
    nonnegative and advection has zero exterior inflow. No regularization.
    Diffusivity, velocity, and impedance have units L²/T, L/T, and L/T;
    the stencil uses a normalized cell width of 1/n and returns rates in 1/T.
    """
    (k, v) = (np.asarray(diffusivity, dtype=float), np.asarray(velocity, dtype=float))
    if k.ndim != 1 or not k.size or v.shape != k.shape:
        raise ValueError('Expected matching nonempty cell coefficient vectors')
    if length <= 0 or np.any(k <= 0) or robin < 0:
        raise ValueError('Expected positive length and diffusivity, nonnegative impedance')
    h = 1.0 / k.size
    (k, v, r) = (k / length / length, v / length, robin / length)
    face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h ** 2
    diag = np.abs(v) / h
    diag[:-1] += face
    diag[1:] += face
    for i in (0, k.size - 1):
        diag[i] += r * k[i] / (h * (0.5 * r * h + k[i]))
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face - np.maximum(-v[:-1], 0) / h
    return np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)

x,scale,info = dtrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])

Wrapper for ``dtrsyl``.

Parameters
----------
a : input rank-2 array('d') with bounds (m,m)
b : input rank-2 array('d') with bounds (n,n)
c : input rank-2 array('d') with bounds (m,n)

Other Parameters
----------------
trana : input bytes, optional
    Default: 'N'
tranb : input bytes, optional
    Default: 'N'
isgn : input int, optional
    Default: 1
overwrite_c : input int, optional
    Default: 0

Returns
-------
x : rank-2 array('d') with bounds (m,n) and c storage
scale : float
info : int
```

### Turn 5 — `execute_in_remote_docker` (`64326 ms` LLM)
- **Rationale**: Implement the confirmed stencil, full Channel B pocket, and public ridge-residual convention, then run the isolated public validator to catch certificate or schema mismatches before stress-testing active bounds and both terminal fields.
- **Alternatives Weighed**: `Use the same two-dimensional factorization for both channels` (rejected: It would discard Channel A's verified separable reduction; real Schur factors retain it without assuming symmetric advection operators.)

```bash
set -e
cd /holdout_task
cat > coupled_channel_vortex_kkt_solver.py <<'PY'
"""Dual-channel tracer inversion and box-simplex optimality certificate."""

import numpy as np
from scipy import sparse
from scipy.linalg import helmert, schur
from scipy.linalg.lapack import dtrsyl
from scipy.sparse.linalg import splu


def _cell_transport(length, diffusivity, velocity, robin):
    """Harmonic face diffusion, row-wise upwind transport, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float) / length**2
    v = np.asarray(velocity, dtype=float) / length
    r = robin / length
    h = 1.0 / k.size
    face = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    main = np.abs(v) / h
    main[:-1] += face
    main[1:] += face
    for i in (0, k.size - 1):
        main[i] += r * k[i] / (h * (k[i] + 0.5 * r * h))
    return (np.diag(main)
            + np.diag(-face - np.maximum(v[1:], 0) / h, -1)
            + np.diag(-face - np.maximum(-v[:-1], 0) / h, 1))


def _responses(ax, by, shifts, modes, pocket, sensors):
    m, nx, ny = modes.shape
    # Schur bases preserve the separable Sylvester equation even for upwinding.
    tx, qx = schur(ax, output='real')
    ty, qy = schur(by, output='real')
    rhs_a = qx.T @ modes @ qy
    rhs_b = modes.reshape(m, -1).T
    base_b = (sparse.kron(ax, sparse.eye(ny), format='csc')
              + sparse.kron(sparse.eye(nx), by, format='csc')
              + sparse.diags(pocket.ravel(), format='csc'))
    eye_x = np.eye(nx)
    eye_b = sparse.eye(nx * ny, format='csc')
    i, j = sensors.T
    sa = np.empty((len(shifts), len(i), m))
    sb = np.empty_like(sa)
    for k, shift in enumerate(shifts):
        tx_shift = tx + shift * eye_x
        a = np.empty_like(modes)
        for p in range(m):
            field, scale, info = dtrsyl(tx_shift, ty, rhs_a[p], tranb='T')
            if info != 0 or scale == 0:
                raise np.linalg.LinAlgError('Singular or ill-conditioned Channel A Sylvester equation')
            a[p] = qx @ (field / scale) @ qy.T
        b = splu(base_b + shift * eye_b).solve(rhs_b).T.reshape(m, nx, ny)
        sa[k] = a[:, i, j].T
        sb[k] = b[:, i, j].T
    return np.concatenate((sa.reshape(-1, m), sb.reshape(-1, m))), a, b


def _ridge_residuals(design, data, ridge):
    """Apply I - Z (Z.T Z + ridge I)^(-1) Z.T to both S and y."""
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    if ridge == 0:
        if not singular.size:
            return data
        keep = singular > np.finfo(float).eps * max(design.shape) * singular[0]
        u = u[:, keep]
        return data - u @ (u.T @ data)
    shrink = singular**2 / (singular**2 + ridge)
    return data - u @ (shrink[:, None] * (u.T @ data))


def _box_simplex(design, target, cap, budget):
    """Minimize a least-squares objective on a capped simplex by active faces."""
    m = design.shape[1]
    w = np.full(m, budget / m)
    if budget == 0 or budget == m * cap:
        return w
    # Active coordinates are fixed at 0 (-1) or cap (+1).
    active = np.zeros(m, dtype=int)
    for _ in range(50 * m):
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active == -1)
        upper = np.flatnonzero(active == 1)
        residual = design @ w - target
        if free.size > 1:
            basis = helmert(free.size, full=False).T
            delta = basis @ np.linalg.lstsq(design[:, free] @ basis, -residual, rcond=None)[0]
            trial = w[free] + delta
            outside = (trial < 0) | (trial > cap)
            if np.any(outside):
                hit = np.flatnonzero(outside)
                bounds = np.where(trial[hit] < 0, 0.0, cap)
                steps = (bounds - w[free[hit]]) / delta[hit]
                first = np.argmin(steps)
                w[free] = np.clip(w[free] + steps[first] * delta, 0.0, cap)
                idx = free[hit[first]]
                w[idx] = bounds[first]
                active[idx] = -1 if bounds[first] == 0 else 1
                continue
            w[free] = trial
            residual = design @ w - target
        g = design.T @ residual
        lo = lower[np.argmin(g[lower])] if lower.size else None
        hi = upper[np.argmax(g[upper])] if upper.size else None
        # Pick the most violated bound; at a vertex a feasible move requires a pair.
        tol = 1e-12 / max(cap, 1.0)
        if not free.size:
            if g[hi] - g[lo] <= tol:
                return w
            active[[lo, hi]] = 0
        else:
            nu = np.mean(g[free])
            bad_lo = nu - g[lo] if lo is not None else 0.0
            bad_hi = g[hi] - nu if hi is not None else 0.0
            if max(bad_lo, bad_hi) <= tol:
                return w
            active[lo if bad_lo >= bad_hi else hi] = 0
    raise RuntimeError('Box-simplex active-set iteration did not converge')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Fit the shared source weights and return both terminal mixture fields."""
    ax, by = [_cell_transport(study['L' + d], study['kappa_' + d],
                              study['vel_' + d], study['robin_' + d]) for d in 'xy']
    design, a, b = _responses(ax, by, np.asarray(study['mu_shifts']),
                              np.asarray(study['source_modes'], dtype=float),
                              np.asarray(study['V_pocket_channel_b']),
                              np.asarray(study['sensor_ij'], dtype=int))
    data = _ridge_residuals(np.asarray(study['Z_nuisance']),
                            np.column_stack((design, study['y_obs'])), study['nuisance_ridge'])
    design, target = data[:, :-1], data[:, -1]
    cap, budget, ridge = (float(study[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    m = design.shape[1]
    w = _box_simplex(np.vstack((design, np.sqrt(ridge) * np.eye(m))),
                     np.r_[target, np.zeros(m)], cap, budget)
    residual = design @ w - target
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    g = design.T @ residual + ridge * w
    # A budget quantile of g makes the linearized box-simplex bound tightest.
    rank = int(np.clip(np.ceil(budget / cap) - 1, 0, m - 1)) if cap > 0 else 0
    nu = np.partition(g, rank)[rank]
    lower = np.maximum(g - nu, 0.0)
    upper = np.maximum(nu - g, 0.0)
    # This is J - L_KKT, evaluated as nonnegative complementarity terms.
    gap = lower @ w + upper @ (cap - w)
    certificate = dict(weights=w.copy(), nu_sum=float(nu), lam_lower=lower, lam_upper=upper,
                       primal_objective=float(primal),
                       dual_lower_bound=float(np.nextafter(primal - gap, -np.inf)))
    return dict(weights=w, terminal_field_channel_a=np.einsum('m,mij->ij', w, a),
                terminal_field_channel_b=np.einsum('m,mij->ij', w, b),
                kkt_certificate=certificate)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`480 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_channel_vortex_kkt_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=91653a03928203f7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_91653a03 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`45669 ms` LLM)
- **Rationale**: Compare stacked readouts and both terminal fields against independent sparse solves to expose axis or stacking errors, then check certificates where bounds bind and the budget is tight. A forward mismatch would require correcting the operator or layout; a gap failure would require correcting the active-set solve.
- **Alternatives Weighed**: `Submit after the public schema pass` (rejected: The public optimum is interior, and the schema checks terminal field shapes but not their values.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, runpy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
p = 'coupled_channel_vortex_kkt_solver.py'
spec = importlib.util.spec_from_file_location('candidate', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Load the independent verifier without its command-line entry point.
ns = {}
code = open('validate_public_schema.py').read().split('\n_d =')[0]
exec(code, ns)
verify = ns['verify_box_simplex_kkt_certificate']
with open('public_suc02_C_studies.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
modes, shifts, ij = (s[k] for k in ('source_modes','mu_shifts','sensor_ij'))
m,nx,ny = modes.shape
base = sparse.kron(gt['Ax'], sparse.eye(ny), format='csc') + sparse.kron(sparse.eye(nx), gt['By'], format='csc')
rhs = modes.reshape(m,-1).T
refs = []
for v in (np.zeros(nx*ny), s['V_pocket_channel_b'].ravel()):
    refs.append(np.array([splu(base+sparse.diags(v+shift,format='csc')).solve(rhs).T.reshape(m,nx,ny) for shift in shifts]))
X = np.concatenate([f[:,:,ij[:,0],ij[:,1]].transpose(0,2,1).reshape(-1,m) for f in refs])
ax,by = [mod._cell_transport(s['L'+d],s['kappa_'+d],s['vel_'+d],s['robin_'+d]) for d in 'xy']
readouts,a,b = mod._responses(ax,by,shifts,modes,s['V_pocket_channel_b'],ij)
for name,l,r in [('all sensor reads',readouts,X),('terminal A basis',a,refs[0][-1]),('terminal B basis',b,refs[1][-1])]:
    err = np.max(np.abs(l-r)); print(name, 'max error',err); assert err < 1e-10
z = s['Z_nuisance']; rn = s['nuisance_ridge']; zw = z.T@z + rn*np.eye(z.shape[1])
P = lambda d: d-z@np.linalg.solve(zw,z.T@d)
worst_gap = worst_field = worst_budget = worst_random = 0.0
for name,cap,budget,y in [('public',s['u_box'],s['total_budget'],s['y_obs']),('lower/upper vertex',0.65,2.60,1000*X[:,0]),('mixed active',0.26,2.60,100*X[:,5]),('nearly full',0.24,2.83,-100*X[:,4]),('empty',1.0,0.0,s['y_obs']),('full',0.25,3.0,s['y_obs'])]:
    out = mod.solve_coupled_channel_vortex_inversion(dict(s,u_box=cap,total_budget=budget,y_obs=y))
    w=out['weights']; cert=out['kkt_certificate']; assert np.array_equal(w,cert['weights'])
    chk=verify(gt['S_tilde'] if name=='public' else P(X),gt['r_y'] if name=='public' else P(y),s['source_ridge'],cap,budget,cert)
    assert chk['valid'], (name,chk)
    worst_gap=max(worst_gap,chk['gap']); worst_budget=max(worst_budget,abs(w.sum()-budget))
    fields=[np.max(np.abs(out['terminal_field_channel_'+d]-np.einsum('m,mij->ij',w,f[-1]))) for d,f in zip('ab',refs)]
    worst_field=max(worst_field,*fields); assert max(fields)<1e-10
    if name=='public':
        err=np.max(np.abs(w-gt['w_opt'])); print('public weight max error',err); assert err<1e-8
        for d in 'ab': assert np.max(np.abs(out['terminal_field_channel_'+d]-gt['terminal_field_'+d.upper()]))<1e-10
    print(name,'gap',chk['gap'],'lower/upper active',np.count_nonzero(w<1e-10),np.count_nonzero(w>cap-1e-10))
rng=np.random.default_rng(381)
for case in range(200):
    n=12; x=rng.normal(size=(24,n))@np.diag(np.geomspace(1,0.005,n)); r=10.0**rng.uniform(-5,-1)
    x=np.vstack((x,np.sqrt(r)*np.eye(n))); y=np.r_[rng.normal(size=24)*10.0**rng.uniform(-2,2),np.zeros(n)]
    cap=rng.uniform(0.05,1.0); budget=(rng.integers(1,n) if case%2==0 else rng.uniform(0,n))*cap
    w=mod._box_simplex(x,y,cap,budget); g=x.T@(x@w-y)
    remaining=budget; wl=np.zeros(n)
    for i in np.argsort(g):
        wl[i]=min(cap,max(0,remaining)); remaining-=wl[i]
    gap=g@(w-wl); worst_random=max(worst_random,gap)
    assert -1e-10<=gap<1e-8 and abs(w.sum()-budget)<1e-10 and w.min()>=0 and w.max()<=cap, (case,gap,w)
print('worst independent certificate gap:',worst_gap,'terminal field error:',worst_field,'budget error:',worst_budget,'random QP gap:',worst_random)
PY
```
- **Container Exit Code**: `0` (`866 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
all sensor reads max error 5.204170427930421e-16
terminal A basis max error 2.0122792321330962e-16
terminal B basis max error 6.418476861114186e-17
public weight max error 1.3516965324811281e-14
public gap 8.298917109073045e-15 lower/upper active 0 0
lower/upper vertex gap 1.8189894035458565e-12 lower/upper active 8 4
mixed active gap 0.0 lower/upper active 2 10
nearly full gap 5.684341886080802e-14 lower/upper active 0 11
empty gap 1.5265566588595902e-15 lower/upper active 12 0
full gap 1.0547118733938987e-15 lower/upper active 0 12
worst independent certificate gap: 1.8189894035458565e-12 terminal field error: 2.6020852139652106e-16 budget error: 8.881784197001252e-16 random QP gap: 5.976063684005886e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=91653a03928203f7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_91653a03 (directory).
```

### Turn 7 — `submit_completed_task` (`14593 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, independent forward-solve comparisons for both channels, and certificate checks on boundary cases. An additional 200 randomized QPs had a worst independently computed gap of 5.98e-14.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:6750'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public schema validator passed. Across six end-to-end cases, including binding bounds and empty and full budgets, the worst KKT gap was 1.82e-12 against a tolerance of 1e-6. Both terminal fields matched independent sparse solves within 2.61e-16.`

### Turn 8 — `execute_in_remote_docker` (`35852 ms` LLM)
- **Rationale**: Reuse the public surveys to test two distinct failure modes. A certificate failure after a known nuisance perturbation would warrant revisiting the ridge transform; an axis-swap or terminal PDE failure would warrant correcting the operator orientation or sensor indexing.
- **Alternatives Weighed**: `Run relay, drift, or within-batch classification checks` (rejected: This survey has no relay, drift operator, batch labels, or classification outcome. Its relevant checks are mixed spatial coupling and ridge nuisance profiling.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
solve = runpy.run_path('coupled_channel_vortex_kkt_solver.py')['solve_coupled_channel_vortex_inversion']
with open('validate_public_schema.py') as f:
    preamble = f.read().split('\n_d =')[0]
ns = {}
exec(preamble, ns)
verify = ns['verify_box_simplex_kkt_certificate']
with open('public_suc02_C_studies.pkl', 'rb') as f:
    studies = pickle.load(f)
worst_gap = worst_swap = worst_pde = 0.0
for k, item in enumerate(studies):
    s, gt = item['study'], item['gt']
    z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    coef = np.linspace(-1.0, 1.0, z.shape[1])
    # With ridge, adding Zc leaves rho Z (Z.T Z + rho I)^(-1)c, not zero.
    ry = gt['r_y'] + ridge * (z @ np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), coef))
    shifted = dict(s, y_obs=s['y_obs'] + z @ coef)
    transposed = dict(shifted, source_modes=s['source_modes'].transpose(0, 2, 1), sensor_ij=s['sensor_ij'][:, ::-1], V_pocket_channel_b=s['V_pocket_channel_b'].T)
    for key in ('L', 'kappa_', 'vel_', 'robin_'):
        transposed[key+'x'], transposed[key+'y'] = s[key+'y'], s[key+'x']
    out, swapped = solve(shifted), solve(transposed)
    for result in (out, swapped):
        chk = verify(gt['S_tilde'], ry, s['source_ridge'], s['u_box'], s['total_budget'], result['kkt_certificate'])
        assert chk['valid'], (k, chk)
        worst_gap = max(worst_gap, chk['gap'])
    weights = out['weights']
    swap = np.max(np.abs(weights - swapped['weights']))
    rhs = np.einsum('m,mij->ij', weights, s['source_modes'])
    for ch in 'ab':
        name = 'terminal_field_channel_' + ch
        f = out[name]
        swap = max(swap, np.max(np.abs(f - swapped[name].T)))
        diag = s['mu_shifts'][-1] + (s['V_pocket_channel_b'] if ch == 'b' else 0)
        residual = gt['Ax'] @ f + f @ gt['By'].T + diag * f - rhs
        scale = max(np.linalg.norm(rhs), np.finfo(float).eps)
        defect = np.linalg.norm(residual) / scale
        assert defect < 1e-10, (k, ch, defect)
        worst_pde = max(worst_pde, defect)
    assert swap < 1e-8, (k, swap)
    worst_swap = max(worst_swap, swap)
    v = s['V_pocket_channel_b']
    print('study', k, 'max mixed pocket difference', np.max(np.abs(np.diff(np.diff(v, axis=0), axis=1))))
print('studies', len(studies), 'worst verified KKT gap', worst_gap, 'worst axis-swap discrepancy', worst_swap, 'worst relative PDE residual', worst_pde)
PY
```
- **Container Exit Code**: `0` (`442 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study 0 max mixed pocket difference 0.3210987331503401
studies 1 worst verified KKT gap 8.555656183517613e-15 worst axis-swap discrepancy 1.1102230246251565e-16 worst relative PDE residual 8.997731890674134e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=91653a03928203f7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_91653a03 (directory).
```

### Turn 9 — `submit_completed_task` (`14545 ms` LLM)
- **Rationale**: The archived artifact passed isolated validation, independent field comparisons, and budget-boundary checks. The follow-up nuisance perturbation and axis-swap audit also passed, with a worst relative terminal PDE residual of 9.00e-14.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:6750'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public schema passed, and both terminal fields matched independent sparse solves within 2.61e-16. The worst verified KKT gap was 1.82e-12, below the 1e-6 tolerance.`

### Turn 10 — `submit_completed_task` (`15498 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated validation, active-bound and budget-endpoint tests, and the follow-up nuisance-perturbation and axis-swap audit. The latter measured a worst relative terminal PDE residual of 9.00e-14. The survey has no batch labels, relay modes, or classification outcomes to audit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_channel_vortex_kkt_solver.py`. The public schema passed. Both terminal fields matched independent sparse solves within 2.61e-16, and the worst verified KKT gap was 1.82e-12 against the 1e-6 tolerance.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_channel_vortex_kkt_solver.py": "FILE:6750"
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
    "student_elapsed_sec": 0.5653,
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
      "import_wall_sec": 0.039344,
      "import_cpu_sec": 0.036476,
      "candidate_call_wall_sec": 0.048458,
      "candidate_call_cpu_sec": 0.044938,
      "ipc_and_codec_wall_sec": 0.132723,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.46212,
      "verifier_and_gt_cpu_sec": 0.460589,
      "total_supervisor_wall_sec": 0.682644
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5653, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_c6b8c10ae24a84ac4b4d97c6e0c0019d__:{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5653, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.039344, \"import_cpu_sec\": 0.036476, \"candidate_call_wall_sec\": 0.048458, \"candidate_call_cpu_sec\": 0.044938, \"ipc_and_codec_wall_sec\": 0.132723, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.46212, \"verifier_and_gt_cpu_sec\": 0.460589, \"total_supervisor_wall_sec\": 0.682644}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 7027.71,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
