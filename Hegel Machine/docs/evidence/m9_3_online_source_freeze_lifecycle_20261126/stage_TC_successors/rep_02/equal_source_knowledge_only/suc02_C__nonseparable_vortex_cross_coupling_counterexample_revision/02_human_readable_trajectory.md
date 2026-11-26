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
| **Total Tokens (Prompt / Cached / Out)** | `278,465` (`267,942` / `78,464` / `10,523`) |
| **Total Episode Wall Time (s)** | `381.67s` |

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

### Turn 1 — `execute_in_remote_docker` (`24123 ms` LLM)
- **Rationale**: Read the validator and public metadata to establish sensor indexing, objective normalization, and certificate fields before implementing the solver; a mismatch would change the assembly or certificate calculation.

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && python3 - <<'PY'
import pickle
with open('public_suc02_C_studies.pkl','rb') as f:
    p=pickle.load(f)
def show(x,depth=0):
    prefix='  '*depth
    if isinstance(x,dict):
        print(prefix,'dict keys:',list(x)[:30])
        if depth<3:
            for k,v in list(x.items())[:30]:
                print(prefix,k, end=': '); show(v,depth+1)
    elif isinstance(x,(tuple,list)):
        print(prefix,type(x).__name__,len(x))
        if x and depth<3: show(x[0],depth+1)
    elif hasattr(x,'shape'): print(prefix,type(x).__name__,x.shape,getattr(x,'dtype',''))
    else: print(prefix,str(x)[:150])
show(p)
PY
if [ -f /tmp/hegel_knowledge_ref/describe_ref.py ]; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'ridge nuisance profiling'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'box simplex KKT'; fi
```
- **Container Exit Code**: `0` (`652 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
weights (n,), positive cap and gap_ | REQ: Array inputs contain finite real values representable in float64. Shapes, finite values, | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_separable_least_squares.verify_box_simplex_certificate | STATUS: APPLICABLE_SUBROUTINE(domain_matched_online_l1_primitive)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_separable_least_squares@ef9581d1 | solve_shifted_sylvester(a: np.ndarray, b: np.ndarray, shifts: np.ndarray, rhs: np.ndarray, *, residual_tol: float=1e-12, log_span_limit: float=25.0) -> np.ndarray | For finite real (nx,nx), (ny,ny), (s,), and (m,nx,ny) inputs with positive dim | REQ: Array inputs contain finite real values representable in float64. Shapes, finite values, | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_separable_least_squares.solve_shifted_sylvester | STATUS: APPLICABLE_SUBROUTINE(domain_matched_online_l1_primitive)
```

### Turn 2 — `execute_in_remote_docker` (`25830 ms` LLM)
- **Rationale**: The previous output truncated the study keys and certificate checks. Recover those details and inspect operator symmetry, which determines whether Channel A can use an orthogonal eigendecomposition, before writing the solver.

```bash
cd /holdout_task && sed -n '40,180p' validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np, scipy
with open('public_suc02_C_studies.pkl','rb') as f: p=pickle.load(f)
print('SciPy',scipy.__version__)
def show(v,d=0):
    if isinstance(v,dict):
        print('dict',list(v))
        if d<3:
            for k,a in v.items():
                print(' '*d,k,end=': '); show(a,d+1)
    elif isinstance(v,(tuple,list)):
        print(type(v).__name__,len(v))
        if v and d<3: show(v[0],d+1)
    elif hasattr(v,'shape'):
        print(v.shape,v.dtype,end=' ')
        if v.size and np.issubdtype(v.dtype,np.number): print('range',v.min(),v.max(),'symmetric error',np.max(np.abs(v-v.T)) if v.ndim==2 and v.shape[0]==v.shape[1] else '-')
        else: print()
    else: print(repr(v)[:100])
show(p)
PY
r=/tmp/hegel_knowledge_ref/describe_ref.py; if [ -f "$r" ]; then python3 "$r" --search 'nuisance' | head -12; python3 "$r" --describe autogenous_prim_separable_least_squares.solve_box_simplex_least_squares | head -45; python3 "$r" --describe autogenous_prim_separable_least_squares.solve_shifted_sylvester | head -35; find /tmp/hegel_knowledge_ref -iname '*separable*reference.txt' -o -iname '*nuisance*reference.txt'; fi
```
- **Container Exit Code**: `0` (`736 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
upplied iteration limit must be a positive integer.",
        "Tangent rank is determined numerically by least squares. Full tangent rank gives uniqueness on that face only; a passing global gap certificate does not establish uniqueness of the weights."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "solve_shifted_sylvester": "APPLICABLE_SUBROUTINE(domain_matched_online_l1_primitive)",
        "solve_box_simplex_least_squares": "APPLICABLE_SUBROUTINE(domain_matched_online_l1_primitive)",
        "verify_box_simplex_certificate": "APPLICABLE_SUBROUTINE(domain_matched_online_l1_primitive)"
      },
      "functions": [
        {
          "function": "solve_shifted_sylvester",
          "signature": "solve_shifted_sylvester(a: np.ndarray, b: np.ndarray, shifts: np.ndarray, rhs: np.ndarray, *, residual_tol: float=1e-12, log_span_limit: float=25.0) -> np.ndarray",
          "doc_summary": "For finite real (nx,nx), (ny,ny), (s,), and (m,nx,ny) inputs with positive dimensions, nonsingular shifts, 0 < residual_tol < 1 and log_span_limit > 0, return (s,m,nx,ny) solutions of a @ U + U @ b.T + shift * U = rhs that pass the per-RHS residual test."
        }
      ],
      "index_cards": [
        "autogenous_prim_separable_least_squares@ef9581d1 | solve_shifted_sylvester(a: np.ndarray, b: np.ndarray, shifts: np.ndarray, rhs: np.ndarray, *, residual_tol: float=1e-12, log_span_limit: float=25.0) -> np.ndarray | For finite real (nx,nx), (ny,ny), (s,), and (m,nx,ny) inputs with positive dim | REQ: Array inputs contain finite real values representable in float64. Shapes, finite values, | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_separable_least_squares.solve_shifted_sylvester",
        "autogenous_prim_separable_least_squares@ef9581d1 | solve_box_simplex_least_squares(design: np.ndarray, target: np.ndarray, cap: float, budget: float, *, initial: np.ndarray | None=None, gap_tol: float=1e-09, max_iter: int | None=None) -> np.ndarray | For finite real design (r,n), target (r,), optional feasible initial (n,), pos | REQ: Array inputs contain finite real values representable in float64. Shapes, finite values, | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_separable_least_squares.solve_box_simplex_least_squares",
/tmp/hegel_knowledge_ref/autogenous_separable_transport_inversion.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_separable_least_squares.reference.txt
```

### Turn 3 — `execute_in_remote_docker` (`28825 ms` LLM)
- **Rationale**: Recover the study keys and sensor representation from the previously truncated output. Quantify the non-additive potential component and identify the profiling convention before implementing the forward operators and certificate.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_suc02_C_studies.pkl','rb'))[0]
for name,d in p.items():
 print(name)
 for k,v in d.items():
  a=np.asarray(v); print(k, (a.shape,str(a.dtype),str(a.ravel()[:3])) if a.ndim else str(v)[:130])
s=p['study']; v=s['V_pocket_channel_b']; print('potential relative additive-approximation error',np.linalg.norm(v-v.mean(0)-v.mean(1,keepdims=True)+v.mean())/np.linalg.norm(v))
PY
r=/tmp/hegel_knowledge_ref; python3 "$r/describe_ref.py" --search 'nuisance' | cut -c1-160 | head -10; grep -inE 'nuisance|profil' "$r/autogenous_separable_transport_inversion.reference.txt" | cut -c1-225 | head -18; grep -inE 'active.set|tangent|gap' "$r/autogenous_prim_separable_least_squares.reference.txt" | cut -c1-200 | head -12
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
observations: np.ndarray, nuisance
autogenous_prim_exponential_memory@39b21121 | profiled_log_sensitivity(j_log: np.ndarray, j_linear: np.ndarray, total_weight: float, *, rcond: float | None=None
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.n
91:def recover_transport_mixture(fields: np.ndarray, sensor_indices: np.ndarray, observations: np.ndarray, nuisance: np.ndarray, cap: float, budget: float, *, nuisance_ridge: float=0.0, source_ridge: float=0.0, gap_tol: float
94:    Fields are indexed (shift, mode, x, y). Observations and nuisance rows
95:    are shift-major, then sensor-major. An empty nuisance matrix is (rows, 0).
98:    Both ridges default to zero; nuisance rank is then resolved by SVD cutoff.
107:    z = np.asarray(nuisance, dtype=float)
111:        raise ValueError('Observations and nuisance must match shift-sensor rows')
112:    if not 0 <= budget <= n * cap or min(nuisance_ridge, source_ridge) < 0 or gap_tol <= 0:
118:        if nuisance_ridge:
119:            shrink = sigma ** 2 / (sigma ** 2 + nuisance_ridge)
96:    (cap, budget, tol) = (_scalar(v, k) for (v, k) in zip((cap, budget, tol), ('cap', 'budget', 'gap_tol')))
101:        raise ValueError('Expected positive cap and gap tolerance and a feasible budget')
109:def _gap_ok(g: np.ndarray, w: np.ndarray, cap: float, budget: float, tol: float) -> bool:
114:    gap = np.maximum(g - nu, 0) @ w + np.maximum(nu - g, 0) @ (cap - w) + abs(nu) * abs(w.sum() - budget)
115:    return bool(np.isfinite(gap) and gap <= tol)
118:def solve_box_simplex_least_squares(design: np.ndarray, target: np.ndarray, cap: float, budget: float, *, initial: np.ndarray | None=None, gap_tol: float=1e-09, max_iter: int | None=None) -> np.nd
119:    """For finite real design (r,n), target (r,), optional feasible initial (n,), positive cap and gap_tol, feasible budget and positive optional max_iter, return a certified box-simplex least-squ
120:    (a, y, cap, budget, tol) = _box_inputs(design, target, cap, budget, gap_tol)
133:        if _box_feasible(w, cap, budget) and _gap_ok(g, w, cap, budget, tol):
149:            tangent = (active - active[:, :1]) @ basis
150:            trial[free] += basis @ np.linalg.lstsq(tangent, y - a @ trial, rcond=None)[0]
170:def verify_box_simplex_certificate(design: np.ndarray, target: np.ndarray, weights: np.ndarray, cap: float, budget: float, *, gap_tol: float=1e-09) -> bool:
```

### Turn 4 — `execute_in_remote_docker` (`30352 ms` LLM)
- **Rationale**: The study supplies transport coefficients, not assembled operators. Compare stencil entries against the public reference to recover boundary and advection conventions, and isolate nuisance profiling with reference operators so a mismatch identifies which calculation to change.
- **Alternatives Weighed**: `Fit separate one-dimensional corrections to Channel B` (rejected: The least-squares additive projection leaves more than half the potential norm unexplained.)
- **ATMS NOGOOD DECLARED**: Refuted `The Channel B potential can be represented as a sum of x-only and y-only terms.` | Witness: `Removing row and column means and restoring the grand mean leaves relative Frobenius norm 0.5455482467.` | Blanket Diagnosis: `The additive operator omits a measured x–y interaction specific to Channel B.` | Sublated Pivot: `Preserve the tensor spectral solve in A; add the measured 2D potential to the sparse Kronecker sum in B.`

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
from scipy.sparse import csc_matrix,eye,kron,diags
from scipy.sparse.linalg import splu
p=pickle.load(open('public_suc02_C_studies.pkl','rb'))[0]; s,g=p['study'],p['gt']
for a,key in [('x','Ax'),('y','By')]:
 A=g[key]; n=len(A); h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]
 print(a,'h',h,'robin',r,'max offdiagonal asymmetry',abs(A-A.T).max()); print('j k v lower diag upper rowsum')
 for j in [0,1,2,n//2,n-2,n-1]: print(j,k[j],v[j],A[j,j-1] if j else 0,A[j,j],A[j,j+1] if j<n-1 else 0,A[j].sum())
 dx=np.diff(np.arange(n))*h; face=(k[:-1]+k[1:])/2/h**2
 print('adv remainders first 6',np.diag(A,1)[:6]+face[:6],np.diag(A,-1)[:6]+face[:6]); print('row sum boundary / (r/h)',A.sum(1)[[0,-1]]/(r/h))
ax,by=map(csc_matrix,(g['Ax'],g['By'])); f=s['source_modes']; m,nx,ny=f.shape; i,j=s['sensor_ij'].T; op=kron(ax,eye(ny))+kron(eye(nx),by); mats=[]
for v in [np.zeros((nx,ny)),s['V_pocket_channel_b']]:
 for mu in s['mu_shifts']:
  U=splu((op+diags(v.ravel()+mu)).tocsc()).solve(f.reshape(m,-1).T); mats.append(U[i*ny+j])
S=np.vstack(mats); z=s['Z_nuisance']; Y=np.column_stack((S,s['y_obs'])); u,t,_=np.linalg.svd(z,full_matrices=False); d=s['nuisance_ridge']/(t*t+s['nuisance_ridge']); target=np.column_stack((g['S_tilde'],g['r_y'])); print('profile residual maker and square-root comparisons:')
for d0 in [d,np.sqrt(d),np.zeros_like(d)]:
 diff=Y-u@((1-d0)[:,None]*(u.T@Y))-target; print('max design/data error',np.max(abs(diff[:,:-1])),np.max(abs(diff[:,-1])))
PY
sed -n '1,51p;112,132p' /tmp/hegel_knowledge_ref/autogenous_separable_transport_inversion.reference.txt; python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | grep -m 1 'doc_summary'
```
- **Container Exit Code**: `0` (`860 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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


def shifted_transport_fields(ax: np.ndarray, by: np.ndarray, shifts: np.ndarray, modes: np.ndarray, *, residual_tol: float=1e-12, log_span_limit: float=25.0) -> np.ndarray:
    """Solve Ax U + U By.T + mu U = F for nonsingular shifted operators.

    Ax and By are tridiagonal transport generators with strictly negative
    off-diagonals. Modes have shape (mode, x, y); output is (shift, mode, x, y).
    No regularization is applied. Check each nonzero RHS in the infinity norm
    relative to its own amplitude; zero RHSs must remain zero.
    """
    (ax, by) = (np.asarray(ax, dtype=float), np.asarray(by, dtype=float))
    (shifts, modes) = (np.asarray(shifts, dtype=float), np.asarray(modes, dtype=float))
    if modes.ndim != 3 or not modes.size or shifts.ndim != 1 or (not shifts.size):
    if not 0 <= budget <= n * cap or min(nuisance_ridge, source_ridge) < 0 or gap_tol <= 0:
        raise ValueError('Expected a feasible budget, nonnegative ridges, and positive gap tolerance')
    data = np.column_stack((y, s))
    eps = np.finfo(float).eps
    if z.shape[1]:
        (q, sigma, _) = np.linalg.svd(z, full_matrices=False)
        if nuisance_ridge:
            shrink = sigma ** 2 / (sigma ** 2 + nuisance_ridge)
        else:
            shrink = sigma > eps * max(z.shape) * sigma[0]
        data -= q @ (shrink[:, None] * (q.T @ data))
    design = np.vstack((data[:, 1:], np.sqrt(source_ridge) * np.eye(n)))
    target = np.r_[data[:, 0], np.zeros(n)]
    w = np.full(n, budget / n)
    if budget == n * cap:
        w.fill(cap)
    bound = np.zeros(n, dtype=np.int8)
    mass_tol = 64 * eps * max(cap, budget)
    box_tol = 8 * eps * cap
    order = min(n - 1, max(0, int(np.ceil(budget / cap)) - 1)) if cap else 0
    for _ in range(40 * n):
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
```

### Turn 5 — `execute_in_remote_docker` (`71859 ms` LLM)
- **Rationale**: Implement the transport stencil, preserve spectral reduction in A, retain the full measured pocket in B, and stop the active-set solve only after the convex gap check. Compare operators, all returned arrays, and the certificate separately so any mismatch identifies which calculation needs revision.
- **Alternatives Weighed**: `Apply the square root of the ridge residual maker` (rejected: It does not match the benchmark’s profiled arrays: the public target differs by up to 0.222, versus 4.89e-15 with direct residualization.)

```bash
set -e
cd /holdout_task
cat > coupled_channel_vortex_kkt_solver.py <<'PY'
"""Forward transport and certified box-simplex inversion for two tracer channels."""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _transport(length, diffusivity, velocity, robin):
    """Cell-centered harmonic diffusion, upwind advection, and Robin loss."""
    k = np.asarray(diffusivity, dtype=float) / length**2
    v = np.asarray(velocity, dtype=float) / length
    r = robin / length
    h = 1.0 / k.size
    face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) / h**2
    diagonal = abs(v) / h
    diagonal[:-1] += face
    diagonal[1:] += face
    for i in (0, k.size - 1):
        diagonal[i] += r * k[i] / (h * (0.5 * r * h + k[i]))
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face - np.maximum(-v[:-1], 0) / h
    return diags((lower, diagonal, upper), (-1, 0, 1), shape=(k.size, k.size), format='csc')


def _spectral_basis(operator):
    # A = D Q diag(eigenvalues) Q.T D^-1, with positive diagonal D.
    lower = operator.diagonal(-1)
    upper = operator.diagonal(1)
    log_scale = np.r_[0., 0.5 * np.cumsum(np.log(lower / upper))]
    scale = np.exp(log_scale - 0.5 * (log_scale.max() + log_scale.min()))
    values, vectors = eigh_tridiagonal(operator.diagonal(), -np.sqrt(lower * upper))
    return values, scale[:, None] * vectors, vectors.T / scale


def _forward(ax, by, modes, shifts, pocket, sensors):
    m, nx, ny = modes.shape
    i, j = sensors.T
    ex, x_forward, x_inverse = _spectral_basis(ax)
    ey, y_forward, y_inverse = _spectral_basis(by)
    rhs_a = x_inverse @ modes @ y_inverse.T
    spectrum = ex[:, None] + ey[None, :]
    rhs_b = modes.reshape(m, nx * ny).T
    base_b = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    base_b += diags(pocket.ravel(), format='csc')
    identity = eye(nx * ny, format='csc')
    rows_a, rows_b = [], []
    for mu in shifts:
        terminal_a = x_forward @ (rhs_a / (spectrum + mu)) @ y_forward.T
        terminal_b = splu(base_b + mu * identity).solve(rhs_b).T.reshape(m, nx, ny)
        rows_a.append(terminal_a[:, i, j].T)
        rows_b.append(terminal_b[:, i, j].T)
    return np.vstack(rows_a + rows_b), terminal_a, terminal_b


def _profile(design, observations, nuisance, ridge):
    """Apply the same ridge residual maker to the design and observations."""
    data = np.column_stack((design, observations))
    if nuisance.shape[1]:
        q, sigma, _ = np.linalg.svd(nuisance, full_matrices=False)
        if ridge:
            shrink = sigma**2 / (sigma**2 + ridge)
        else:
            shrink = sigma > np.finfo(float).eps * max(nuisance.shape) * sigma[0]
        data -= q @ (shrink[:, None] * (q.T @ data))
    return data[:, :-1], data[:, -1]


def _multipliers(gradient, cap, budget):
    if budget == 0:
        nu = float(gradient.min())
    elif budget == cap * gradient.size:
        nu = float(gradient.max())
    else:
        rank = min(gradient.size - 1, max(0, int(np.ceil(budget / cap)) - 1))
        nu = float(np.partition(gradient, rank)[rank])
    return nu, np.maximum(gradient - nu, 0), np.maximum(nu - gradient, 0)


def _box_simplex(design, target, ridge, cap, budget):
    """Minimize on successive faces, using the global tangent gap to stop."""
    n = design.shape[1]
    a = np.vstack((design, np.sqrt(ridge) * np.eye(n)))
    y = np.r_[target, np.zeros(n)]
    if budget == 0 or budget == n * cap:
        return np.full(n, 0. if budget == 0 else cap)

    w = np.full(n, budget / n)
    bounds = np.zeros(n, dtype=int)  # -1: lower; 0: free; +1: upper
    for _ in range(40 * n):
        residual = a @ w - y
        gradient = a.T @ residual
        _, lower, upper = _multipliers(gradient, cap, budget)
        if lower @ w + upper @ (cap - w) <= 1e-11:
            return w

        free = np.flatnonzero(bounds == 0)
        if free.size >= 2:
            # Use one free coordinate to enforce the budget on this face.
            pivot = free[-1]
            rest = free[:-1]
            dz = np.linalg.lstsq(a[:, rest] - a[:, [pivot]], -residual, rcond=None)[0]
            direction = np.r_[dz, -dz.sum()]
            trial = w[free] + direction
            violated = (trial < 0) | (trial > cap)
            if np.any(violated):
                hits = np.flatnonzero(violated)
                values = np.where(direction[hits] > 0, cap, 0.)
                steps = (values - w[free[hits]]) / direction[hits]
                first = np.argmin(steps)
                w[free] = np.clip(w[free] + steps[first] * direction, 0., cap)
                index = free[hits[first]]
                w[index] = values[first]
                bounds[index] = 1 if values[first] == cap else -1
                continue
            w[free] = trial
            gradient = a.T @ (a @ w - y)
            _, lower, upper = _multipliers(gradient, cap, budget)
            if lower @ w + upper @ (cap - w) <= 1e-11:
                return w

        # Release the active bounds involved in the best feasible transfer.
        receivers = np.flatnonzero(bounds != 1)
        donors = np.flatnonzero(bounds != -1)
        receive = receivers[np.argmin(gradient[receivers])]
        donate = donors[np.argmax(gradient[donors])]
        bounds[[receive, donate]] = 0
    raise RuntimeError('Box-simplex solve did not reach the target certificate gap')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Return the source mixture, terminal tracer fields, and a KKT certificate."""
    ax = _transport(study['Lx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    by = _transport(study['Ly'], study['kappa_y'], study['vel_y'], study['robin_y'])
    raw, terminal_a, terminal_b = _forward(
        ax, by, np.asarray(study['source_modes'], dtype=float), study['mu_shifts'],
        np.asarray(study['V_pocket_channel_b'], dtype=float), np.asarray(study['sensor_ij']),
    )
    design, target = _profile(raw, np.asarray(study['y_obs']).ravel(),
                              np.asarray(study['Z_nuisance']), study['nuisance_ridge'])
    ridge, cap, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _box_simplex(design, target, ridge, cap, budget)
    residual = design @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    gradient = design.T @ residual + ridge * w
    nu, lower, upper = _multipliers(gradient, cap, budget)
    gap = float(lower @ w + upper @ (cap - w) + nu * (w.sum() - budget))
    certificate = {
        'weights': w.tolist(), 'nu_sum': nu,
        'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
        'primal_objective': objective,
        'dual_lower_bound': objective - max(0., gap),
    }
    return {
        'weights': w,
        'terminal_field_channel_a': np.einsum('m,mij->ij', w, terminal_a),
        'terminal_field_channel_b': np.einsum('m,mij->ij', w, terminal_b),
        'kkt_certificate': certificate,
    }
PY
# The public validator checks the certificate against independently supplied design and target.
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import coupled_channel_vortex_kkt_solver as mod
for item in pickle.load(open('public_suc02_C_studies.pkl','rb')):
    s,g=item['study'],item['gt']; out=mod.solve_coupled_channel_vortex_inversion(s)
    for a,k in [('x','Ax'),('y','By')]:
        op=mod._transport(s['L'+a],s['kappa_'+a],s['vel_'+a],s['robin_'+a]); print(k,np.max(abs(op.toarray()-g[k])))
    for key,ref in [('weights','w_opt'),('terminal_field_channel_a','terminal_field_A'),('terminal_field_channel_b','terminal_field_B')]:
        print(key,'max_abs',np.max(abs(out[key]-g[ref])), 'relative_l2',np.linalg.norm(out[key]-g[ref])/np.linalg.norm(g[ref]))
    c=out['kkt_certificate']; print('reported_gap',c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`703 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_channel_vortex_kkt_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Ax 8.526512829121202e-14
By 1.1368683772161603e-13
weights max_abs 1.2351231148954867e-14 relative_l2 3.969493217088096e-14
terminal_field_channel_a max_abs 5.637851296924623e-16 relative_l2 3.3924586935483796e-14
terminal_field_channel_b max_abs 3.209238430557093e-16 relative_l2 1.960579289280232e-14
reported_gap 6.938893903907228e-18

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b3763266e70d068e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b3763266 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37608 ms` LLM)
- **Rationale**: The public optimum does not exercise clipping at either bound. Test active bounds, tight caps, and endpoint budgets against an independently assembled certificate and both reference fields. A gap failure would identify an active-set or multiplier defect; a field mismatch would identify a forward-assembly defect.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from pathlib import Path
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
p = Path('coupled_channel_vortex_kkt_solver.py')
spec = importlib.util.spec_from_file_location('candidate', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
verify = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], verify)
verify = verify['verify_box_simplex_kkt_certificate']
item = pickle.loads(Path('public_suc02_C_studies.pkl').read_bytes())[0]
s, gt = item['study'], item['gt']
m, nx, ny = s['source_modes'].shape
ax, by = csc_matrix(gt['Ax']), csc_matrix(gt['By'])
base = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
f = s['source_modes'].reshape(m, -1).T
i,j = s['sensor_ij'].T
cases = [('public',s['V_pocket_channel_b'],s['y_obs'],1.,2.6), ('active_bounds',1.7*s['V_pocket_channel_b'],12*s['y_obs'],.7,2.1), ('tight_cap',s['V_pocket_channel_b'],s['y_obs'],.23,2.6), ('empty',s['V_pocket_channel_b'],s['y_obs'],1.,0.), ('full',s['V_pocket_channel_b'],s['y_obs'],1.,float(m))]
for label, pocket, y, cap, budget in cases:
    st = dict(s, V_pocket_channel_b=pocket, y_obs=y, u_box=cap, total_budget=budget)
    out = mod.solve_coupled_channel_vortex_inversion(st)
    w = out['weights']
    matrices, terminal = [], []
    for v in [np.zeros_like(pocket), pocket]:
        for mu in s['mu_shifts']:
            u = splu(base + diags(v.ravel()+mu, format='csc')).solve(f)
            matrices.append(u[i*ny+j])
        terminal.append(u)
    z = s['Z_nuisance']
    data = np.column_stack((np.vstack(matrices),y))
    prof = data-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@data)
    cert = out['kkt_certificate']
    chk = verify(prof[:,:-1],prof[:,-1],s['source_ridge'],cap,budget,cert)
    assert chk['valid'], (label, chk)
    np.testing.assert_array_equal(w,cert['weights'])
    errors = []
    for channel,u in zip('ab',terminal):
        predicted = out['terminal_field_channel_'+channel].ravel()
        reference = u@w
        errors.append(np.max(np.abs(predicted-reference)))
        np.testing.assert_allclose(predicted, reference, rtol=1e-10, atol=1e-12)
    at_lower, at_upper = np.count_nonzero(w<1e-10), np.count_nonzero(w>cap-1e-10)
    if label == 'active_bounds':
        assert at_lower and at_upper, (at_lower,at_upper)
    print(label, 'lower',at_lower,'upper',at_upper,'gap',chk['gap'],'field_max_abs_A_B',errors)
PY
```
- **Container Exit Code**: `0` (`938 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public lower 0 upper 0 gap 1.9637069748057456e-15 field_max_abs_A_B [1.5612511283791264e-16, 5.4643789493269423e-17]
active_bounds lower 8 upper 1 gap 1.2434497875801753e-14 field_max_abs_A_B [2.0469737016526324e-16, 3.9898639947466563e-17]
tight_cap lower 0 upper 8 gap 3.885780586188048e-16 field_max_abs_A_B [1.5612511283791264e-16, 5.204170427930421e-17]
empty lower 12 upper 0 gap 6.800116025829084e-16 field_max_abs_A_B [0.0, 0.0]
full lower 0 upper 12 gap 2.220446049250313e-16 field_max_abs_A_B [7.28583859910259e-16, 2.42861286636753e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b3763266e70d068e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b3763266 (directory).
```

### Turn 7 — `submit_completed_task` (`14418 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification. Independent sparse forward solves and certificate checks agreed on the public study, active-bound and tight-cap cases, and both budget endpoints.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:7004'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_channel_vortex_kkt_solver.py` with a separable spectral solve for Channel A, an exact sparse 2D solve for Channel B, and a certified box-simplex inversion. The public schema passed. Across five validation cases, the largest independently verified KKT gap was 1.24e-14, below the 1e-6 limit.`

### Turn 8 — `execute_in_remote_docker` (`40450 ms` LLM)
- **Rationale**: Measure how much the additive approximation changes the terminal field, then remove high- and low-potential sensor groups from the real observations. An independent KKT or PDE residual failure would identify a sensor-stacking, nuisance-profiling, or forward-operator defect to fix; otherwise retain the verified solver.
- **Alternatives Weighed**: `Test batch confounding, relay modes, or Fokker–Planck drift` (rejected: The supplied study contains tracer measurements and a static linear transport model, not batch labels, relay states, or a drift equation.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

solve = load('coupled_channel_vortex_kkt_solver').solve_coupled_channel_vortex_inversion
ns = {}
exec(Path('validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
worst_gap = worst_field = 0.
for item in pickle.loads(Path('public_suc02_C_studies.pkl').read_bytes()):
    s, gt = item['study'], item['gt']
    modes = s['source_modes']
    m, nx, ny = modes.shape
    shifts = s['mu_shifts']
    v = s['V_pocket_channel_b']
    additive = v.mean(axis=0) + v.mean(axis=1, keepdims=True) - v.mean()
    ax, by = gt['Ax'], gt['By']
    base = kron(csc_matrix(ax), eye(ny), format='csc') + kron(eye(nx), csc_matrix(by), format='csc')
    rhs = modes.reshape(m, -1).T
    i, j = s['sensor_ij'].T
    raw = []
    for pot in [np.zeros_like(v), v]:
        for mu in shifts:
            fields = splu(base + diags(pot.ravel() + mu, format='csc')).solve(rhs)
            raw.append(fields[i*ny+j])
    raw = np.vstack(raw)
    out = solve(s)
    w = out['weights']
    surrogate = splu(base + diags(additive.ravel() + shifts[-1], format='csc')).solve(rhs @ w)
    exact = out['terminal_field_channel_b'].ravel()
    print('additive-ablation terminal relative L2 error at public weights', np.linalg.norm(surrogate-exact)/np.linalg.norm(exact))
    order = np.argsort(v[i,j])
    for label, dropped in [('drop_low_pocket_sensors', order[:len(i)//4]), ('drop_high_pocket_sensors', order[-(len(i)//4):])]:
        mask = np.ones(len(i), dtype=bool)
        mask[dropped] = False
        rows = np.tile(mask, 2*len(shifts))
        y, z = s['y_obs'][rows], s['Z_nuisance'][rows]
        st = dict(s, sensor_ij=s['sensor_ij'][mask], y_obs=y, Z_nuisance=z)
        out = solve(st)
        w = out['weights']
        data = np.column_stack((raw[rows], y))
        prof = data - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
        chk = verify(prof[:,:-1], prof[:,-1], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
        assert chk['valid'], (label, chk)
        source = np.einsum('m,mij->ij', w, modes)
        errors = []
        for channel, pot in [('a', 0.), ('b', v)]:
            field = out['terminal_field_channel_'+channel]
            error = np.max(np.abs(ax @ field + field @ by.T + (pot + shifts[-1])*field - source))/np.max(np.abs(source))
            assert error < 1e-10, (label, channel, error)
            errors.append(error)
        worst_gap = max(worst_gap, chk['gap'])
        worst_field = max(worst_field, *errors)
        print(label, 'gap', chk['gap'], 'relative PDE residuals A/B', errors)
print('audit worst gap', worst_gap, 'worst relative PDE residual', worst_field)
PY
```
- **Container Exit Code**: `0` (`587 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
additive-ablation terminal relative L2 error at public weights 0.06024792095726644
drop_low_pocket_sensors gap 7.216449660063518e-16 relative PDE residuals A/B [1.1399634897383202e-13, 3.5235235137366264e-14]
drop_high_pocket_sensors gap 1.124100812432971e-15 relative PDE residuals A/B [1.1031634551448283e-13, 3.0119028363849654e-14]
audit worst gap 1.124100812432971e-15 worst relative PDE residual 1.1399634897383202e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b3763266e70d068e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b3763266 (directory).
```

### Turn 9 — `submit_completed_task` (`15041 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification on the public study, active-bound and tight-cap cases, both budget endpoints, and both sensor-dropout slices. The worst relative terminal PDE residual in the dropout audit was 1.14e-13; no further change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/coupled_channel_vortex_kkt_solver.py': 'FILE:7004'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_channel_vortex_kkt_solver.py` with a separable spectral solve for Channel A and an exact sparse 2D solve for Channel B. The public schema passed. Across seven validation cases, the largest independently verified KKT gap was 1.24e-14, below the 1e-6 limit.`

### Turn 10 — `execute_in_remote_docker` (`33952 ms` LLM)
- **Rationale**: Test two previously unchecked invariances using the public measurements. Failure under an orthogonal nuisance rotation would point to basis-dependent ridge profiling; failure under source relabeling would point to source indexing or the active-set solve. Neither test requires changing the artifact or repeating the settled additive-potential ablation.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/coupled_channel_vortex_kkt_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_suc02_C_studies.pkl', 'rb') as f:
    studies = pickle.load(f)

rng = np.random.default_rng(42)
worst_weight = worst_field = worst_gap = 0.
for item in studies:
    s = item['study']
    base = mod.solve_coupled_channel_vortex_inversion(s)
    z = s['Z_nuisance']
    q = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))[0]
    for label, changes in [('nuisance_rotation', {'Z_nuisance': z @ q}), ('source_relabeling', {})]:
        p = rng.permutation(len(base['weights'])) if label == 'source_relabeling' else np.arange(len(base['weights']))
        out = mod.solve_coupled_channel_vortex_inversion(dict(s, source_modes=s['source_modes'][p], **changes))
        error_w = np.max(np.abs(out['weights'] - base['weights'][p]))
        error_f = max(np.max(np.abs(out[k]-base[k])) for k in ('terminal_field_channel_a', 'terminal_field_channel_b'))
        cert = out['kkt_certificate']
        gap = cert['primal_objective'] - cert['dual_lower_bound']
        primal_change = abs(cert['primal_objective'] - base['kkt_certificate']['primal_objective'])
        print(label, 'weight max abs', error_w, 'field max abs', error_f, 'primal change', primal_change, 'reported gap', gap)
        assert error_w < 1e-9 and error_f < 1e-10 and primal_change < 1e-9 and 0 <= gap <= 1e-6, label
        worst_weight = max(worst_weight, error_w)
        worst_field = max(worst_field, error_f)
        worst_gap = max(worst_gap, gap)
print('worst weight max abs', worst_weight, 'worst field max abs', worst_field, 'worst reported gap', worst_gap)
PY
```
- **Container Exit Code**: `0` (`476 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nuisance_rotation weight max abs 1.942890293094024e-15 field max abs 4.5102810375396984e-17 primal change 6.800116025829084e-16 reported gap 2.0816681711721685e-17
source_relabeling weight max abs 1.3877787807814457e-16 field max abs 5.204170427930421e-18 primal change 0.0 reported gap 1.3877787807814457e-17
worst weight max abs 1.942890293094024e-15 worst field max abs 4.5102810375396984e-17 worst reported gap 2.0816681711721685e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b3763266e70d068e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b3763266 (directory).
```

### Turn 11 — `submit_completed_task` (`15030 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification on the public study, active bounds, tight caps, budget endpoints, and sensor dropouts. The final invariance audit passed: rotating the nuisance basis or relabeling sources changed weights by at most 1.95e-15. No solver change is warranted.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_channel_vortex_kkt_solver.py` with a separable spectral solve for Channel A, an exact sparse 2D solve for Channel B, and a box-simplex KKT certificate. The public schema passed, and the largest independently verified KKT gap was 1.24e-14, below the 1e-6 limit.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_channel_vortex_kkt_solver.py": "FILE:7004"
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
      "import_wall_sec": 0.039776,
      "import_cpu_sec": 0.036729,
      "candidate_call_wall_sec": 0.039877,
      "candidate_call_cpu_sec": 0.036822,
      "ipc_and_codec_wall_sec": 0.135158,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.469685,
      "verifier_and_gt_cpu_sec": 0.466602,
      "total_supervisor_wall_sec": 0.684496
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5653, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_2dd9f00820553cde5c890cfaa0d080be__:{\"strict_pass\": true, \"reward\": 1.0, \"max_weights_rel_l2\": 0.0, \"max_field_a_rel_l2\": 0.0, \"max_field_b_rel_l2\": 0.0, \"max_kkt_gap\": 0.0, \"student_elapsed_sec\": 0.5653, \"instance_diagnostics\": [{\"seed\": 99621, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}, {\"seed\": 99622, \"w_rel_l2\": 0.0, \"fa_rel_l2\": 0.0, \"fb_rel_l2\": 0.0, \"kkt_gap\": 0.0}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.039776, \"import_cpu_sec\": 0.036729, \"candidate_call_wall_sec\": 0.039877, \"candidate_call_cpu_sec\": 0.036822, \"ipc_and_codec_wall_sec\": 0.135158, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.469685, \"verifier_and_gt_cpu_sec\": 0.466602, \"total_supervisor_wall_sec\": 0.684496}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6626.09,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
