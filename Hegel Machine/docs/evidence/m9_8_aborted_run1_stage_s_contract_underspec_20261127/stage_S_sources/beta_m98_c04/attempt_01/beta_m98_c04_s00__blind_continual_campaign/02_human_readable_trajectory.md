# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_c04_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `469,053` (`449,789` / `118,144` / `19,264`) |
| **Total Episode Wall Time (s)** | `564.49s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15756 ms` LLM)
- **Rationale**: Inspect the schema, calibration bank layout, and lineage records to determine which forward-model parameters can be fitted directly and which need to be inferred from prior campaigns.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))
def show(x,k='',d=0):
 if isinstance(x,dict):
  for a,b in x.items(): show(b,k+'/'+str(a),d+1)
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype,x.ravel()[:8])
 elif isinstance(x,(list,tuple)) and len(x) and isinstance(x[0],(dict,np.ndarray)): print(k,'len',len(x));show(x[0],k+'/0',d+1)
 else: print(k,str(x)[:500])
show(s)
PY
printf '\nArchive inventory:\n'; ls -lah /tmp/hegel_continual_archive/ 2>/dev/null || true; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 12000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; printf '\nAvailable numerical primitives:\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex'
```
- **Container Exit Code**: `0` (`422 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19914 ms` LLM)
- **Rationale**: The prior output was truncated before the sensor layout and validator. Reading those directly will prevent row-order errors; sweep samples will distinguish independent nonlinear bank response from cross-channel mixing before fitting the inverse calibration.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb')); print('batch:',type(p), len(p)); s=p[0]['study'] if isinstance(p,list) else p.get('study',p); print('keys:',list(s));
for k in ('sensor_ij','row_bank_id','mu_shifts','vel_x','vel_y','aux_manifold_telemetry'):
 print(k,s.get(k))
for k in ('source_modes','reference_tracer_pulses'):
 print(k, np.shape(s[k]), (np.min(s[k]),np.max(s[k])) if k=='source_modes' else [x['w_pulse'] for x in s[k]])
c=s['sensor_calibration_sweeps']; print('bank sweeps:',[(k,np.shape(v),v[:10]) for k,v in c['single_bank_sweeps'].items()]); print('rosettes:',c['ref_rosette'][:5],c['raw_rosette'][:5]); print('ranges:',[(k,np.min(c[k]),np.max(c[k])) for k in ('ref_rosette','raw_rosette')])
PY
printf '\nLocal reference files:\n'; ls -la /pretrain_task /tmp/hegel_continual_archive 2>/dev/null; printf '\nRidge primitive contract:\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -c 2500
```
- **Container Exit Code**: `0` (`375 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
  "access_mode": "executable",
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
      "mo
```

### Turn 3 — `execute_in_remote_docker` (`37056 ms` LLM)
- **Rationale**: Fit four temporary wall diagonals to isolate interior stencil errors. If no convention fits at the sweep noise scale, inspect the residual before adopting a wall law for pulse-free campaigns.
- **Alternatives Weighed**: `Use the nominal Robin coefficients with a textbook discretization` (rejected: Several grid and flux conventions satisfy the prose but predict different tracer responses; the public pulses can distinguish them.)

```bash
cat >/tmp/transport_probe.py <<'PY'
import itertools,pickle
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def axis(k,v,L,r,grammar):
    spacing,avg,adv,edge=grammar
    n=len(k); h=L/(n+spacing)
    d={'ar':(k[:-1]+k[1:])/2,'ha':2*k[:-1]*k[1:]/(k[:-1]+k[1:]),'ge':np.sqrt(k[:-1]*k[1:])}[avg]/h**2
    vf=(v[:-1]+v[1:])/2/h
    p=np.maximum(vf,0); m=np.maximum(-vf,0)
    lo=-d-p; up=-d-m
    diag=np.zeros(n)
    diag[:-1]+=d; diag[1:]+=d
    if adv=='flux': diag[:-1]+=p; diag[1:]+=m
    elif adv=='rows': diag[:-1]+=m; diag[1:]+=p
    elif adv=='cell':
        lo=-d-np.maximum(v[1:],0)/h; up=-d-np.maximum(-v[:-1],0)/h
        diag+=np.abs(v)/h
    elif adv=='cellclamp':
        lo=-d-np.maximum(v[1:],0)/h; up=-d-np.maximum(-v[:-1],0)/h
        diag[:-1]+=np.maximum(-v[:-1],0)/h; diag[1:]+=np.maximum(v[1:],0)/h
    if edge=='out': diag[[0,-1]]+=np.maximum([-v[0],v[-1]],0)/h
    if edge=='abs': diag[[0,-1]]+=np.abs(v[[0,-1]])/h
    if edge=='in': diag[[0,-1]]+=np.maximum([v[0],-v[-1]],0)/h
    diag[[0,-1]]+=np.broadcast_to(r,(2,))
    return sp.diags((lo,diag,up),(-1,0,1),format='csc')


def predict(s,r,grammar,w=None,last=False):
    nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T
    A=sp.kron(sp.eye(ny),axis(s['kappa_x'],s['vel_x'],s['Lx'],r[:2],grammar))+sp.kron(axis(s['kappa_y'],s['vel_y'],s['Ly'],r[2:],grammar),sp.eye(nx))+sp.diags(s['V_reaction_2d'].ravel(order='F'))
    F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    if w is not None: F=F@w
    I=sp.eye(nx*ny,format='csc'); obs=[]
    for mu in s['mu_shifts']:
        U=splu((A+mu*I).tocsc()).solve(F)
        obs.append(U[i+nx*j])
    return (np.vstack(obs),U) if last else np.vstack(obs)

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; print('calibration identity RMS',np.std(y-x),'linear',np.std(y-np.c_[np.ones(len(x)),x]@np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0])); print('telemetry summary',np.mean(s['aux_manifold_telemetry'],axis=0),np.std(s['aux_manifold_telemetry'],axis=0)); print('kappa endpoints',s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]])
 W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; P=lambda a:a-Q@(Q.T@a)
 scores=[]
 for g in itertools.product((-1,0,1),('ar','ha','ge'),('flux','rows','cell','cellclamp'),('none','out','in','abs')):
    def fun(r): return P(predict(s,r,g,W)-Y).ravel()
    fit=least_squares(fun,[3]*4,bounds=(-3.,70.),max_nfev=25)
    scores.append((np.sqrt(np.mean(fit.fun**2)),g,fit.x,np.max(np.abs(fit.fun))))
 for a in sorted(scores,key=lambda a:a[0])[:18]: print(a)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_probe.py
```
- **Container Exit Code**: `0` (`11383 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3 0.94134901 0.92829305]
kappa endpoints [0.41288044 0.41288044] [0.51912269 0.27912269]
(3.251448036681059e-05, (-1, 'ar', 'cell', 'none'), array([3.65897819, 4.45237195, 6.50993101, 5.36762856]), 0.0001781810669761062)
(3.251448036681059e-05, (-1, 'ar', 'cell', 'out'), array([3.65897819, 4.45237195, 6.50993101, 5.36762856]), 0.0001781810669761062)
(3.251448036681059e-05, (-1, 'ar', 'cellclamp', 'in'), array([3.65897819, 4.45237195, 6.50993101, 5.36762856]), 0.0001781810669761062)
(3.251448036681059e-05, (-1, 'ar', 'cellclamp', 'abs'), array([3.65897819, 4.45237195, 6.50993101, 5.36762856]), 0.0001781810669761062)
(3.2514480366812436e-05, (-1, 'ar', 'cell', 'in'), array([1.76314486, 3.33802909, 6.50993101, 4.12174239]), 0.00017818105889651746)
(3.2514480366812436e-05, (-1, 'ar', 'cell', 'abs'), array([1.76314486, 3.33802909, 6.50993101, 4.12174239]), 0.00017818105889651746)
(3.25144803668154e-05, (-1, 'ar', 'cellclamp', 'none'), array([5.55481152, 5.56671482, 6.50993101, 6.61351473]), 0.00017818106836633104)
(3.25144803668154e-05, (-1, 'ar', 'cellclamp', 'out'), array([5.55481152, 5.56671482, 6.50993101, 6.61351473]), 0.00017818106836633104)
(3.6001255724827676e-05, (-1, 'ge', 'cell', 'in'), array([1.76313516, 3.34358586, 6.51032465, 4.12377858]), 0.00016420357593721188)
(3.6001255724827676e-05, (-1, 'ge', 'cell', 'abs'), array([1.76313516, 3.34358586, 6.51032465, 4.12377858]), 0.00016420357593721188)
(3.60012557248327e-05, (-1, 'ge', 'cell', 'none'), array([3.6589685 , 4.45792872, 6.51032465, 5.36966475]), 0.00016420355752527253)
(3.60012557248327e-05, (-1, 'ge', 'cell', 'out'), array([3.6589685 , 4.45792872, 6.51032465, 5.36966475]), 0.00016420355752527253)
(3.60012557248327e-05, (-1, 'ge', 'cellclamp', 'in'), array([3.6589685 , 4.45792872, 6.51032465, 5.36966475]), 0.00016420355752527253)
(3.60012557248327e-05, (-1, 'ge', 'cellclamp', 'abs'), array([3.6589685 , 4.45792872, 6.51032465, 5.36966475]), 0.00016420355752527253)
(3.600125572483376e-05, (-1, 'ge', 'cellclamp', 'none'), array([5.55480182, 5.57227159, 6.51032465, 6.61555092]), 0.00016420357932205723)
(3.600125572483376e-05, (-1, 'ge', 'cellclamp', 'out'), array([5.55480182, 5.57227159, 6.51032465, 6.61555092]), 0.00016420357932205723)
(4.2916601174496185e-05, (-1, 'ha', 'cell', 'none'), array([3.65895883, 4.46349689, 6.51071956, 5.37170075]), 0.0001502235035759908)
(4.2916601174496185e-05, (-1, 'ha', 'cell', 'out'), array([3.65895883, 4.46349689, 6.51071956, 5.37170075]), 0.0001502235035759908)
```

### Turn 4 — `execute_in_remote_docker` (`33387 ms` LLM)
- **Rationale**: Compare fits with and without nuisance removal to see whether the projection hides a forward-model error. Localize the remaining residual by shift, bank, and position, and compare fitted wall terms against Robin closures that can transfer to pulse-free campaigns.
- **Alternatives Weighed**: `Treat the fitted wall diagonals as a universal closure` (rejected: A single campaign identifies effective wall diagonals, not how they transfer to campaigns with different coefficients or to campaigns without tracer pulses.)

```bash
cat >/tmp/transport_refine.py <<'PY'
import pickle
import numpy as np
from scipy import sparse as sp
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from transport_probe import axis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; V=s['V_reaction_2d'].ravel(order='F'); Z=s['Z_nuisance']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W; I=sp.eye(nx*ny,format='csc'); g=(-1,'ar','cell','none'); bases=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d],0,g) for d in ('x','y')]
def pred(r,axes=bases,extra=None):
 Ax,Ay=[B+sp.diags(np.r_[r[2*i],np.zeros(B.shape[0]-2),r[2*i+1]]) for i,B in enumerate(axes)]; A=sp.kron(sp.eye(ny),Ax)+sp.kron(Ay,sp.eye(nx))+sp.diags(V if extra is None else V+extra); return np.vstack([splu((A+m*I).tocsc()).solve(F)[idx] for m in s['mu_shifts']])
r0=np.array([3.66,4.45,6.51,5.37]); rnom=np.repeat([s['robin_x'],s['robin_y']],2); h=np.repeat([s['Lx']/(nx-1),s['Ly']/(ny-1)],2); k=np.r_[s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]]]; v=np.r_[s['vel_x'][[0,-1]],s['vel_y'][[0,-1]]]; print('wall nominal/r_over_h/series/series_half/k_r_over_h/velocity',rnom,rnom/h,rnom/h/(1+rnom*h/k),rnom/h/(1+.5*rnom*h/k),k*rnom/h,v)
for project in (False,True):
 def fun(r):
  e=pred(r)-Y
  return e-Z@np.linalg.lstsq(Z,e,rcond=None)[0] if project else e
 out=least_squares(lambda r:fun(r).ravel(),r0,gtol=1e-13); e=fun(out.x); print('project',project,'walls',out.x,'RMS',np.sqrt(np.mean(e**2)),'nuisance fit',np.linalg.lstsq(Z,pred(out.x)-Y,rcond=None)[0].T); print('shift RMS',np.sqrt(np.mean(e.reshape(3,32,2)**2,axis=(1,2))),'bank RMS',np.sqrt(np.mean(e.reshape(-1,4,2)**2,axis=(0,2)))); print('top',[(int(i//32),ij[i%32].tolist(),np.round(e[i],7).tolist()) for i in np.argsort(np.max(np.abs(e),axis=1))[-9:]])
print('Y extrema',Y.min(axis=0),Y.max(axis=0),'pulses',W.T); print('source ranges',[(round(a.min(),4),round(a.max(),4)) for a in s['source_modes']]); print('telemetry col correlation',np.round(np.corrcoef(s['aux_manifold_telemetry'].T),2)); print('telemetry blocks means',np.round(s['aux_manifold_telemetry'].reshape(3,-1,4).mean(axis=1),3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_refine.py
```
- **Container Exit Code**: `0` (`472 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
wall nominal/r_over_h/series/series_half/k_r_over_h/velocity [0.97920239 0.97920239 1.19160514 1.19160514] [5.30401297 5.30401297 6.55382824 6.55382824] [3.68887407 3.68887407 4.62400367 3.68979978] [4.35140258 4.35140258 5.42232628 4.7214354 ] [2.18992321 2.18992321 3.40224097 1.82932219] [ 0.35       -0.20572484 -0.         -0.22652476]
project False walls [3.65876488 4.45457865 6.50935251 5.36802743] RMS 3.2988956694806495e-05 nuisance fit [[-1.17144928e-06 -1.80807110e-06 -5.28167669e-06]
 [-1.85667791e-06  1.63042149e-06 -4.52626200e-06]]
shift RMS [3.56096810e-05 2.68470226e-05 3.57211672e-05] bank RMS [2.19452158e-05 5.27730269e-05 2.10019287e-05 2.54051014e-05]
top [(2, [0, 4], [7.03e-05, 6.76e-05]), (0, [0, 0], [-6.06e-05, -7.4e-05]), (2, [2, 6], [8e-05, 4.87e-05]), (2, [2, 8], [8.1e-05, 3.82e-05]), (0, [0, 8], [8.28e-05, -7.48e-05]), (0, [0, 4], [-4.73e-05, -8.73e-05]), (0, [0, 2], [-9.57e-05, -0.0001181]), (1, [0, 8], [0.0001443, 1.73e-05]), (2, [0, 8], [0.0001773, 8.14e-05])]
project True walls [3.65897819 4.45237195 6.50993101 5.36762856] RMS 3.251448036681418e-05 nuisance fit [[-9.60362294e-07 -1.82056133e-06 -5.78555977e-06]
 [-1.66141139e-06  1.60258948e-06 -4.97464752e-06]]
shift RMS [3.51082194e-05 2.58512408e-05 3.56468874e-05] bank RMS [2.06031884e-05 5.23729906e-05 2.14965419e-05 2.44794384e-05]
top [(0, [0, 0], [-5.22e-05, -6.56e-05]), (2, [2, 8], [7.32e-05, 2.92e-05]), (2, [0, 4], [7.57e-05, 6.77e-05]), (2, [2, 6], [8.04e-05, 5.15e-05]), (0, [0, 8], [8.32e-05, -7.23e-05]), (0, [0, 4], [-5.58e-05, -9.44e-05]), (0, [0, 2], [-9.09e-05, -0.0001183]), (1, [0, 8], [0.0001425, 1.56e-05]), (2, [0, 8], [0.0001782, 8.2e-05])]
Y extrema [0.00592495 0.00617789] [0.24932544 0.21492444] pulses [[1.  0.  0.  0.5 0.  0.  0.  0. ]
 [0.  1.  0.  0.  0.5 0.  0.  0. ]]
source ranges [(0.0, 0.9596), (0.0, 0.9261), (0.0, 0.923), (0.0, 0.9689), (0.0, 0.9623), (0.0, 0.9568), (0.0, 0.9296), (0.0, 0.9172)]
telemetry col correlation [[ 1.    0.51 -0.27 -0.07]
 [ 0.51  1.   -0.15 -0.08]
 [-0.27 -0.15  1.    0.07]
 [-0.07 -0.08  0.07  1.  ]]
telemetry blocks means [[-0.281  0.037  0.678  0.289]
 [ 0.26   0.668 -0.794  0.2  ]
 [ 0.388 -0.07   0.076 -0.158]]
```

### Turn 5 — `execute_in_remote_docker` (`28129 ms` LLM)
- **Rationale**: Fit each candidate stencil on two shifts and predict the third. Retain a correction only if it improves the worst held-out shift; otherwise keep the simpler diagonal closure.
- **Alternatives Weighed**: `Attribute the pulse residual to nuisance drift` (rejected: Nuisance removal barely lowers RMS, and the largest errors recur near the same wall.)

```bash
cat >/tmp/wall_grammar.py <<'PY'
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transport_refine import *
from itertools import product
from scipy.optimize import least_squares

def axes_for(p,kind):
 out=[]
 for t,B in enumerate(bases):
  d=('x','y')[t]; a=B.toarray(); n=len(a); k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(n-1)
  if kind in ('diffusion','both'):
   # Interpolate from face diffusion to its adjacent cell's value.
   z=p[-2+t]*(k[:-1]-k[1:])/(2*h*h); i=np.arange(n-1)
   a[i,i+1]-=z; a[i+1,i]+=z; a[i,i]+=z; a[i+1,i+1]-=z
  if kind in ('wall-row','wall-column','both'):
   if kind=='wall-column': a[1,0]+=p[4+2*t]; a[-2,-1]+=p[5+2*t]
   else: a[0,1]+=p[4+2*t]; a[-1,-2]+=p[5+2*t]
  out.append(sp.csc_matrix(a))
 return out

def rms(a): return np.sqrt(np.mean(a*a))
for kind,n in [('diag',4),('diffusion',6),('wall-row',8),('wall-column',8),('both',10)]:
 def residual(p): return pred(p[:4],axes_for(p,kind))-Y
 init=np.r_[r0,np.zeros(n-4)]; fit=least_squares(lambda p:residual(p).ravel(),init,gtol=1e-12)
 oof=np.empty_like(Y)
 for test in range(3):
  keep=np.arange(len(Y))//len(idx)!=test
  q=least_squares(lambda p:residual(p)[keep].ravel(),init,gtol=1e-12)
  oof[~keep]=residual(q.x)[~keep]
 print(kind,'params',np.round(fit.x,6),'fit',rms(fit.fun),'worst held shift',max(map(rms,np.split(oof,3))))
print('observation range',s['y_obs_raw'].min(),s['y_obs_raw'].max())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_grammar.py
```
- **Container Exit Code**: `0` (`2069 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
diag params [3.658765 4.454579 6.509353 5.368027] fit 3.2988956694806495e-05 worst held shift 6.265242196140738e-05
diffusion params [3.651641e+00 4.435682e+00 6.507480e+00 5.360430e+00 5.684000e-03
 1.292000e-03] fit 3.135776656247255e-05 worst held shift 5.962551657866267e-05
wall-row params [ 2.916668  3.698159  6.627348  5.234422  0.491511  0.491511 -0.07352
  0.07352 ] fit 3.305215119080609e-14 worst held shift 7.007589222695822e-14
wall-column params [ 3.657560e+00  3.697878e+00  6.516252e+00  5.214444e+00  1.641000e-03
  5.294070e-01 -7.259000e-03  9.240700e-02] fit 3.137991475857253e-05 worst held shift 5.611176587388187e-05
both params [ 2.916668  3.698159  6.627348  5.234422  0.491511  0.491511 -0.07352
  0.07352  -0.       -0.      ] fit 6.801892627729355e-14 worst held shift 6.846637269560078e-14
observation range -0.2777799507641361 0.4642071878152962
```

### Turn 6 — `execute_in_remote_docker` (`31529 ms` LLM)
- **Rationale**: Check whether the fitted wall terms admit a physical scaling and whether the known-pulse operator also explains the primary observation. If the primary observation noise is small, it can constrain wall terms when no diagnostic pulses or usable lineage are available.
- **Alternatives Weighed**: `Apply the eight public fitted wall terms unchanged to blind campaigns` (rejected: The public pulses identify this campaign's operator but do not establish that its wall terms are invariant across campaigns.)
- **ATMS NOGOOD DECLARED**: Refuted `Changing only the wall diagonals is sufficient to explain the public tracer responses.` | Witness: `Adding only each wall row's inward-neighbor coefficient reduced worst held-out shift RMS from 6.27e-5 to 7.01e-14; wall-column and interior diffusion corrections did not.` | Blanket Diagnosis: `The missing terms are localized to the boundary rows of the transport operator, not to interior diffusion or nuisance drift.` | Sublated Pivot: `Preserve the identified interior stencil and allow boundary diagonal and inward-neighbor terms to vary independently.`

```bash
cat >/tmp/lineage_probe.py <<'PY'
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transport_refine import *
from scipy.optimize import minimize
for a,B in zip('xy',bases):
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1); print(a,'robin',s['robin_'+a],'h',h,'k',np.round(k,6),'v',np.round(v,6),'base wall diagonals',B.diagonal()[[0,-1]],'base wall inward',B[0,1],B[-1,-2]); print('wall coefficients normalized',np.array([.491511,.491511] if a=='x' else [-.07352,.07352])*h*h/k[[0,-1]])
c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Yc=c['raw_rosette']; print('sweep shape',X.shape,'max identity error',abs(Yc-X).max()); print('budget/box/ridges',s['total_budget'],s['u_box'],s['source_ridge'],s['nuisance_ridge']); print('telemetry first four rows',np.round(s['aux_manifold_telemetry'][:4],4)); print('Z stats',np.linalg.norm(Z,axis=0))
# Known-pulse wall law, used only to estimate the observation noise floor.
r=np.array([2.916668,3.698159,6.627348,5.234422]); ds=[.491511,.491511,-.07352,.07352]; axes=[]
for t,B in enumerate(bases):
 a=B.toarray();a[0,1]+=ds[2*t];a[-1,-2]+=ds[2*t+1];axes.append(sp.csc_matrix(a))
from transport_probe import predict
F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
S=pred(r,axes); z=np.c_[S,Z]; yy=s['y_obs_raw']; M=S.shape[1]; w=np.linalg.lstsq(z,yy,rcond=None)[0]; print('unconstrained joint fit',np.round(w,8),'sum',sum(w[:M]),'RMS',np.std(z@w-yy)); fit=minimize(lambda w:0.5*np.sum((z@w-yy)**2),w,jac=lambda w:z.T@(z@w-yy),bounds=list(zip(np.zeros(M),np.broadcast_to(s['u_box'],M)))+[(None,None)]*Z.shape[1],constraints={'type':'eq','fun':lambda w:sum(w[:M])-s['total_budget'],'jac':lambda w:np.r_[np.ones(M),np.zeros(Z.shape[1])]},method='SLSQP',options={'ftol':1e-14}); print('box joint fit',np.round(fit.x,8),'RMS',np.std(z@fit.x-yy))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/lineage_probe.py
if test -d /tmp/hegel_continual_archive; then ls -lah /tmp/hegel_continual_archive; if test -f /tmp/hegel_continual_archive/archive_apparatus_log.json; then head -c 7000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; else echo 'No continual archive mounted for the public campaign'; fi
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x robin 0.9792023947175681 h 0.1846153846153846 k [0.41288  0.446385 0.477942 0.505718 0.528098 0.543783 0.55186  0.55186
 0.543783 0.528098 0.505718 0.477942 0.446385 0.41288 ] v [ 0.35      0.345004  0.330159  0.305889  0.272886  0.232093  0.184674
  0.131983  0.075524  0.01691  -0.042188 -0.100081 -0.155117 -0.205725] base wall diagonals [14.50137132 13.71988086] base wall inward -12.605537989327402 -12.605537989327402
wall coefficients normalized [0.04057371 0.04057371]
y robin 1.1916051351743588 h 0.18181818181818182 k [0.519123 0.514262 0.500073 0.477706 0.448972 0.4162   0.382045 0.349273
 0.320539 0.298172 0.283984 0.279123] v [-0.       -0.055605 -0.108996 -0.158044 -0.200797 -0.235551 -0.260922
 -0.275899 -0.279886 -0.272723 -0.254697 -0.226525] base wall diagonals [15.62994122  9.7628679 ] base wall inward -15.629941217917592 -8.516981723696965
wall coefficients normalized [-0.00468177  0.00870733]
sweep shape (80, 4) max identity error 3.4358700448214145e-05
budget/box/ridges 1.4 0.42 0.04 0.08
telemetry first four rows [[ 5.8530e-01  3.4300e-01  6.3360e-01  8.3100e-01]
 [-8.4110e-01  5.0000e-04  9.6090e-01  1.0658e+00]
 [ 1.4160e+00  1.1057e+00 -2.0420e-01 -1.7339e+00]
 [ 4.4230e-01 -6.0440e-01 -1.9630e-01 -6.8210e-01]]
Z stats [9.51208869 7.97820876 9.564323  ]
unconstrained joint fit [ 0.09172342  0.64587646 -0.14512187  0.1217715  -0.03113055] sum 0.7375998833187224 RMS 0.022471086248459466
box joint fit [ 0.41999848  0.42       -0.14607211  0.12357235 -0.02911143] RMS 0.026502065267790777
No continual archive mounted for the public campaign
```

### Turn 7 — `execute_in_remote_docker` (`24345 ms` LLM)
- **Rationale**: Refit the observation using all source modes. Its residual at the pulse-identified walls will determine whether routine observations can constrain the wall stencil without pulses. Inspect the sweep layout and available lineage assets before implementing the transfer path.
- **Alternatives Weighed**: `Use the reported observation residual to reject blind wall fitting` (rejected: The diagnostic still used the two-pulse design, so its observation residual does not test the eight-mode model.)

```bash
cat >/tmp/observation_probe.py <<'PY'
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import transport_refine as t
from wall_grammar import axes_for
import numpy as np
from scipy.optimize import least_squares
s=t.s; c=s['sensor_calibration_sweeps']; print('study keys',list(s)); print('calibration keys',list(c)); print('metadata',[(k,v) for k,v in s.items() if isinstance(v,(str,int,float))]); print('bank sweep details');
for k,v in c['single_bank_sweeps'].items():
 a=np.asarray(v); print(k,a.shape,a[:3],a[-3:]);
x=c['ref_rosette']; y=c['raw_rosette']; print('sweep ranges',x.min(0),x.max(0)); print('cross coefficients',np.linalg.lstsq(np.c_[np.ones(len(x)),x],y-x,rcond=None)[0]);
init=np.r_[t.r0,np.zeros(4)]; p=least_squares(lambda q:(t.pred(q[:4],axes_for(q,'wall-row'))-t.Y).ravel(),init,gtol=1e-13).x; print('identified walls',p)
F=t.F; t.F=s['source_modes'].transpose(0,2,1).reshape(-1,s['nx']*s['ny']).T
S=t.pred(p[:4],axes_for(p,'wall-row')); y=s['y_obs_raw']; X=np.c_[S,t.Z]; w=np.linalg.lstsq(X,y,rcond=None)[0]; print('all-mode design:',S.shape,'OLS weights + nuisance',w,'weight sum',w[:S.shape[1]].sum(),'residual rms',np.std(X@w-y)); print('shift RMS',np.sqrt(np.mean((X@w-y).reshape(len(s['mu_shifts']),-1)**2,axis=1))); print('design singular values',np.linalg.svd(X,compute_uv=False)); print('terminal candidate magnitude',S.min(),S.max()); print('pulse Gram singular values',np.linalg.svd(F,compute_uv=False));
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/observation_probe.py
printf '\nAvailable assets:\n'; find /pretrain_task /holdout_task /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s\n' 2>/dev/null | head -35
```
- **Container Exit Code**: `0` (`2042 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e+00  6.516252e+00  5.214444e+00  1.641000e-03
  5.294070e-01 -7.259000e-03  9.240700e-02] fit 3.137991475857253e-05 worst held shift 5.611176587388187e-05
both params [ 2.916668  3.698159  6.627348  5.234422  0.491511  0.491511 -0.07352
  0.07352  -0.       -0.      ] fit 6.801892627729355e-14 worst held shift 6.846637269560078e-14
observation range -0.2777799507641361 0.4642071878152962
study keys ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'reference_tracer_pulses', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
calibration keys ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
metadata [('Lx', 2.4), ('Ly', 2.0), ('nx', 14), ('ny', 12), ('robin_x', 0.9792023947175681), ('robin_y', 1.1916051351743588), ('nuisance_ridge', 0.08), ('source_ridge', 0.04), ('u_box', 0.42), ('total_budget', 1.4)]
bank sweep details
bank_id (24,) [0 0 0] [3 3 3]
ref_signal (24,) [-1.1 -0.6 -0.1] [0.4 0.9 1.4]
raw_readout (24,) [-1.0999889  -0.60000728 -0.09999984] [0.40000498 0.9000005  1.40001195]
sweep ranges [-1.36774969 -1.33988098 -1.33524232 -1.39662664] [1.63920147 1.79389004 1.74431893 1.68217558]
cross coefficients [[ 4.63150409e-07 -5.02730986e-07  1.20097850e-06  6.71427024e-07]
 [-3.26608060e-07  7.95382629e-07 -7.19213998e-07 -1.26996434e-06]
 [-6.99532086e-07  4.11510611e-07  1.40399100e-08  7.76241803e-07]
 [ 5.23998042e-07  2.32291965e-06 -4.75847312e-07 -6.15982746e-07]
 [ 5.46286559e-07 -8.36380353e-07 -2.05152508e-06  7.52143180e-08]]
identified walls [ 2.91666847  3.69815893  6.6273485   5.23442182  0.49151116  0.49151116
 -0.07352025  0.07352025]
all-mode design: (96, 8) OLS weights + nuisance [ 0.12917172  0.17836126  0.18680715  0.27976861  0.09600793  0.00790757
  0.26057348  0.25350483 -0.14263682  0.1201659  -0.03036908] weight sum 1.392102548075175 residual rms 0.00679023041474654
shift RMS [0.00706295 0.0061893  0.00708605]
design singular values [10.28440872  8.84573401  7.85247323  1.44337921  0.6651755   0.54406659
  0.37301389  0.16717876  0.1402639   0.10419274  0.02495069]
terminal candidate magnitude 0.0008966250643339984 0.21202994511855605
pulse Gram singular values [4.84201298 1.10244575]

Available assets:
/holdout_task/public_batch.pkl 26378
/holdout_task/public_asset_index.json 3667
/holdout_task/validate_public_schema.py 1064
```

### Turn 8 — `execute_in_remote_docker` (`85663 ms` LLM)
- **Rationale**: Build the solver around the pulse-validated boundary stencil. Compare the diagnostic and pulse-free paths on the same public study before choosing how much to trust blind wall fitting.
- **Alternatives Weighed**: `Use the final source ridge while identifying blind wall parameters` (rejected: Source shrinkage could be absorbed into boundary coefficients. Estimate the walls with profiled source weights, then apply the specified ridge in the final QP.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrate the rosette, identify the wall stencil, and solve the source QP."""
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu


def _features(x, degree, interactions):
    return np.column_stack([np.ones(len(x)), *[x**d for d in range(1, degree+1)],
                            *([x[:, i]*x[:, j] for i in range(4) for j in range(i+1, 4)] if interactions else [])])


def _calibrator(sweeps):
    ref, raw = (np.asarray(sweeps[k], dtype=float) for k in ('ref_rosette', 'raw_rosette'))
    banks = sweeps.get('single_bank_sweeps')
    n = len(ref)
    # Score inverse maps on held-out rosette sweeps. Fit small departures from
    # the identity so the nominal-map comparison includes the measured offset.
    models = [(0, False), (1, False), (2, False), (3, False), (2, True), (3, True)]
    delta = ref - raw
    scores = [np.mean(delta**2)]
    for degree, cross in models[1:]:
        x = _features(raw, degree, cross)
        err = np.empty_like(delta)
        for f in range(5):
            train = np.arange(n) % 5 != f
            coef = np.linalg.lstsq(x[train], delta[train], rcond=None)[0]
            err[~train] = x[~train] @ coef - delta[~train]
        scores.append(np.mean(err**2))
    cutoff = min(scores)*1.06 + 1e-27
    degree, cross = next(m for m, e in zip(models, scores) if e <= cutoff)
    if degree == 0:
        return lambda y: np.asarray(y, dtype=float)
    coefs = []
    for i in range(4):
        x = _features(raw, degree, cross)
        target = delta[:, i]
        if banks is not None:
            j = np.asarray(banks['bank_id']) == i
            sample = np.asarray(banks['raw_readout'])[j]
            truth = np.asarray(banks['ref_signal'])[j]
            # Isolated bank sweeps give the diagonal response with all other
            # reference channels at zero. Infer their raw baselines from the
            # full sweep before using these points in the inverse regression.
            forward = np.linalg.lstsq(_features(ref, degree, cross), raw, rcond=None)[0]
            isolated = np.zeros((len(sample), 4)); isolated[:, i] = truth
            simulated = _features(isolated, degree, cross) @ forward
            simulated[:, i] = sample
            x = np.vstack([x, _features(simulated, degree, cross)])
            target = np.r_[target, truth-sample]
        coefs.append(np.linalg.lstsq(x, target, rcond=None)[0])
    coef = np.column_stack(coefs)
    def apply(y):
        a = np.asarray(y, dtype=float).reshape(-1, 4)
        return (a + _features(a, degree, cross) @ coef).ravel()
    return apply


class _Chamber:
    def __init__(self, study):
        self.nx, self.ny = nx, ny = int(study['nx']), int(study['ny'])
        self.F = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T
        ij = np.asarray(study['sensor_ij'], dtype=int)
        self.sensors = ij[:, 0] + nx*ij[:, 1]
        self.shifts = np.asarray(study['mu_shifts'])
        self.eye = sparse.eye(nx*ny, format='csc')
        axes, nominal, inward, scales = [], [], [], []
        for a in 'xy':
            k, v = (np.asarray(study[key+'_'+a]) for key in ('kappa', 'vel'))
            n = len(k); h = study['L'+a] / (n-1)
            d = (k[:-1]+k[1:])/(2*h*h)
            lo = -d-np.maximum(v[1:], 0)/h
            hi = -d-np.maximum(-v[:-1], 0)/h
            diag = np.abs(v)/h
            diag[:-1] += d; diag[1:] += d
            axes.append(sparse.diags((lo, diag, hi), (-1, 0, 1), format='csc'))
            nominal.extend([study['robin_'+a]/h]*2)
            inward.extend([hi[0], lo[-1]])
            scales.extend([d[0], d[-1]])
        ix, iy = sparse.eye(nx), sparse.eye(ny)
        self.base = (sparse.kron(iy, axes[0])+sparse.kron(axes[1], ix)
                     + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
        self.prior = np.r_[nominal, np.zeros(4)]
        self.inward, self.scales = np.array(inward), np.array(scales)
        self.directions = []
        for coupling in (False, True):
            for axis, n in enumerate((nx, ny)):
                for row, inner in ((0, 1), (n-1, n-2)):
                    edge = sparse.coo_matrix(([1.], ([row], [inner if coupling else row])), shape=(n, n))
                    term = sparse.kron(iy, edge) if axis == 0 else sparse.kron(edge, ix)
                    self.directions.append(term.tocsc())

    def response(self, wall, rhs=None, derivatives=False, shifts=None):
        if rhs is None:
            rhs = self.F
        A = self.base.copy()
        for a, D in zip(wall, self.directions):
            A += a*D
        obs, jac = [], []
        for mu in self.shifts if shifts is None else shifts:
            lu = splu(A + mu*self.eye)
            field = lu.solve(rhs)
            obs.append(field[self.sensors])
            if derivatives:
                du = lu.solve(np.concatenate([-D@field for D in self.directions], axis=1))
                jac.append(du[self.sensors].reshape(len(self.sensors), 8, rhs.shape[1]).transpose(0, 2, 1))
        return np.vstack(obs), field, np.concatenate(jac) if derivatives else None


def _box_qp(H, g, upper, budget):
    """Primal feasible active set for a strictly convex equality-constrained QP."""
    n = len(g)
    if budget <= 0 or budget >= upper.sum():
        w = np.zeros(n) if budget <= 0 else upper.copy()
        grad = H@w+g
        return w, -grad.min() if budget <= 0 else -grad.max()
    w = upper*(budget/upper.sum())
    active = np.zeros(n, dtype=int)
    active[upper == 0] = -1
    for _ in range(20*n+20):
        free = np.flatnonzero(active == 0)
        grad = H@w+g
        if not len(free):
            lo = np.flatnonzero((active < 0) & (upper > 0)); hi = np.flatnonzero(active > 0)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j]-grad[i] <= 2e-12:
                return w, -(grad[i]+grad[j])/2
            active[[i, j]] = 0
            continue
        Hf = H[np.ix_(free, free)]
        q, a = solve(Hf, np.column_stack((grad[free], np.ones(len(free)))), assume_a='pos').T
        nu = -q.sum()/a.sum()
        direction = -q-nu*a
        with np.errstate(divide='ignore', invalid='ignore'):
            steps = np.where(direction > 1e-14, (upper[free]-w[free])/direction,
                             np.where(direction < -1e-14, -w[free]/direction, np.inf))
        j = int(np.argmin(steps)); t = min(1., max(0., steps[j]))
        w[free] += t*direction
        if t < 1.-1e-12:
            i = free[j]
            active[i] = 1 if direction[j] > 0 else -1
            w[i] = upper[i] if active[i] == 1 else 0.
            continue
        reduced = H@w+g+nu
        violation = np.where(upper > 0, active*reduced, 0.)
        i = int(np.argmax(violation))
        if violation[i] <= 2e-12:
            return w, float(nu)
        active[i] = 0
    raise RuntimeError('Source QP active set did not converge')


def _projector(Z, ridge):
    Pz = np.linalg.solve(Z.T@Z + ridge*np.eye(Z.shape[1]), Z.T)
    return lambda a: a-Z@(Pz@a)


def _fit_walls(ch, study, calibrate, y):
    pulses = study.get('reference_tracer_pulses')
    scales = np.r_[ch.prior[:4], ch.scales]
    lo = np.r_[np.full(4, .02), -ch.scales*.65]
    hi = np.r_[np.maximum(20., ch.prior[:4]*4), -ch.inward*.65]
    x0 = ch.prior.copy()
    if pulses:
        W = np.column_stack([p['w_pulse'] for p in pulses])
        target = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
        rhs = ch.F@W
        def fun(w):
            return (ch.response(w, rhs)[0]-target).ravel()
        def jac(w):
            return ch.response(w, rhs, True)[2].reshape(-1, 8)
        return least_squares(fun, x0, jac=jac, x_scale=scales, bounds=(lo, hi),
                             gtol=2e-14, ftol=1e-12, xtol=1e-12).x
    return _blind_walls(ch, study, y, x0, scales, lo, hi)


def _blind_walls(ch, study, y, prior, scales, lo, hi):
    Z = np.asarray(study['Z_nuisance']); Q = np.linalg.qr(Z, mode='reduced')[0]
    project = lambda a: a-Q@(Q.T@a)
    y = project(y)
    M = ch.F.shape[1]; budget = study['total_budget']
    upper = np.broadcast_to(study['u_box'], (M,))
    strength = .025
    def residual(wall):
        S = project(ch.response(wall)[0]); H = S.T@S + 1e-10*np.eye(M)
        w, _ = _box_qp(H, -S.T@y, upper, budget)
        return np.r_[S@w-y, strength*(wall-prior)/scales]
    return least_squares(residual, prior, bounds=(lo, hi), x_scale=scales, max_nfev=30).x


def solve_batch(study: dict) -> dict:
    calibrate = _calibrator(study['sensor_calibration_sweeps'])
    y = calibrate(study['y_obs_raw'])
    chamber = _Chamber(study)
    walls = _fit_walls(chamber, study, calibrate, y)
    S, U, _ = chamber.response(walls)
    P = _projector(np.asarray(study['Z_nuisance']), study['nuisance_ridge'])
    X, b = P(S), P(y)
    ridge = study['source_ridge']; budget = study['total_budget']
    upper = np.broadcast_to(study['u_box'], (S.shape[1],))
    H = X.T@X + ridge*np.eye(S.shape[1]); g = -X.T@b
    w, nu = _box_qp(H, g, upper, budget)
    reduced = H@w+g+nu
    lam_lower = np.where(w <= 1e-10, np.maximum(reduced, 0.), 0.)
    lam_upper = np.where(w >= upper-1e-10, np.maximum(-reduced, 0.), 0.)
    r = X@w-b
    primal = float((r@r+ridge*(w@w))/2)
    stationarity = reduced-lam_lower+lam_upper
    lagrangian = primal + nu*(w.sum()-budget)-lam_lower@w+lam_upper@(w-upper)
    dual = min(primal, float(lagrangian-.5*stationarity@solve(H, stationarity, assume_a='pos')))
    return {'weights': w, 'terminal_field': (U@w).reshape((chamber.nx, chamber.ny), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': float(nu), 'lam_lower': lam_lower,
                                'lam_upper': lam_upper, 'primal_objective': primal, 'dual_lower_bound': dual}}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
import batch_solver as b
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=b._calibrator(s['sensor_calibration_sweeps']); ch=b._Chamber(s); y=c(s['y_obs_raw']); wall=b._fit_walls(ch,s,c,y)
print('calibration correction RMS',np.sqrt(np.mean((y-s['y_obs_raw'])**2)))
print('pulse walls',wall,'nominal',ch.prior)
for pulses in (s['reference_tracer_pulses'],None):
 a=dict(s,reference_tracer_pulses=pulses); o=b.solve_batch(a); print('pulses',bool(pulses),'weights',o['weights'],'certificate',o['kkt_certificate'])
print('banks',np.asarray(s['row_bank_id']).shape,np.asarray(s['row_bank_id']).ravel()[:24])
PY
```
- **Container Exit Code**: `0` (`798 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration correction RMS 0.0
pulse walls [ 2.91666847  3.69815893  6.6273485   5.23442182  0.49151116  0.49151116
 -0.07352025  0.07352025] nominal [5.30401297 5.30401297 6.55382824 6.55382824 0.         0.
 0.         0.        ]
pulses True weights [0.16570015 0.16962509 0.18203617 0.18460553 0.15896711 0.0482149
 0.24510049 0.24575056] certificate {'weights': array([0.16570015, 0.16962509, 0.18203617, 0.18460553, 0.15896711,
       0.0482149 , 0.24510049, 0.24575056]), 'nu_sum': -0.008751476234667557, 'lam_lower': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'lam_upper': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'primal_objective': 0.007832832680075747, 'dual_lower_bound': 0.007832832680075745}
pulses False weights [0.16068526 0.1659691  0.19741239 0.20207719 0.18165494 0.03187481
 0.2081773  0.25214901] certificate {'weights': array([0.16068526, 0.1659691 , 0.19741239, 0.20207719, 0.18165494,
       0.03187481, 0.2081773 , 0.25214901]), 'nu_sum': -0.007980828656535596, 'lam_lower': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'lam_upper': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'primal_objective': 0.007742985558703626, 'dual_lower_bound': 0.007742985558703626}
banks (96,) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81982512e8932545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81982512 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`41834 ms` LLM)
- **Rationale**: Test held shifts using the packaged forward operator, compare the blind path to the measured pulse anchor, and verify the QP against an independent optimizer across interior and active-bound cases.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: The schema does not test wall transfer, nonlinear calibration, or certificate correctness at active bounds.)

```bash
cat >/tmp/verify_batch.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares, minimize
spec=importlib.util.spec_from_file_location('b','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._calibrator(s['sensor_calibration_sweeps']); ch=b._Chamber(s); y=c(s['y_obs_raw']); p=s['reference_tracer_pulses']; yref=np.column_stack([c(a['y_pulse_raw']) for a in p]); rhs=ch.F@np.column_stack([a['w_pulse'] for a in p]); ns=len(ch.sensors); r=np.arange(len(y))//ns
rms=lambda z:np.sqrt(np.mean(z*z))
for j in range(len(ch.shifts)):
 keep=r!=j
 def f(w): return (ch.response(w,rhs)[0]-yref)[keep].ravel()
 def jac(w): return ch.response(w,rhs,True)[2][keep].reshape(-1,8)
 fit=least_squares(f,ch.prior,jac=jac,gtol=2e-14)
 print('held shift',j,'rms',rms((ch.response(fit.x,rhs)[0]-yref)[~keep]))
W=b._fit_walls(ch,s,c,y); S,U,_=ch.response(W); blind=b._fit_walls(ch,dict(s,reference_tracer_pulses=None),c,y); print('blind walls',blind,'wall prediction max-shift RMS',max(rms(v) for v in np.split(ch.response(blind,rhs)[0]-yref,len(ch.shifts))))
wp=b.solve_batch(s); wb=b.solve_batch(dict(s,reference_tracer_pulses=None)); print('public pulse-free relative errors',np.linalg.norm(wp['weights']-wb['weights'])/np.linalg.norm(wp['weights']),np.linalg.norm(wp['terminal_field']-wb['terminal_field'])/np.linalg.norm(wp['terminal_field']))
Z=s['Z_nuisance']; proj=b._projector(Z,s['nuisance_ridge']); X,yy=proj(S),proj(y); n=X.shape[1]; ridge=s['source_ridge']; H=X.T@X+ridge*np.eye(n); g=-X.T@yy; u=np.broadcast_to(s['u_box'],n); q=wp['kkt_certificate']; w=wp['weights']; print('public equality/stationarity/gap/field consistency',abs(w.sum()-s['total_budget']),np.max(np.abs(H@w+g+q['nu_sum']-q['lam_lower']+q['lam_upper'])),q['primal_objective']-q['dual_lower_bound'],np.max(abs(wp['terminal_field'].ravel(order='F')-U@w)))
rng=np.random.default_rng(153); worst=np.zeros(4)
for i in range(150):
 n=int(rng.integers(3,15)); A=rng.normal(size=(n,n)); H=A.T@A+.04*np.eye(n); g=rng.normal(size=n)*8; u=rng.uniform(.04,.9,n); u[rng.random(n)<.15]=0; B=u.sum()*rng.choice([0.,.01,.5,.99,1.]) if i%2==0 else rng.uniform(.02,.98)*u.sum()
 w,nu=b._box_qp(H,g,u,B); d=H@w+g+nu
 q=lambda t:.5*t@H@t+g@t
 res=minimize(q,u*(B/u.sum()),jac=lambda t:H@t+g,bounds=list(zip(np.zeros(n),u)),constraints=[{'type':'eq','fun':lambda t:t.sum()-B,'jac':lambda t:np.ones(n)}],method='SLSQP',options={'ftol':1e-12})
 low=(w<=1e-9)&(u>0); high=(w>=u-1e-9)&(u>0); mid=~(low|high)&(u>0)
 worst=np.maximum(worst,[abs(w.sum()-B),max(np.max(-w),np.max(w-u)),max(np.max(-d[low],initial=0),np.max(d[high],initial=0),np.max(abs(d[mid]),initial=0)),abs(q(w)-q(res.x)) if res.success else 0])
print('stress worst equality/box/stationarity/oracle objective',worst)
assert np.max(worst[:3])<1e-8
# Compare the empirical inverse on held-range observations under known response laws.
cal=s['sensor_calibration_sweeps']; a=cal['ref_rosette']; test=rng.uniform(-.42,.60,(200,4)); mix=np.array([[1.1,.08,0,.03],[-.03,.91,.05,0],[0,-.04,1.03,.02],[.02,0,.04,.96]])
for name,fn in [('affine',lambda x:x@mix+.04),('cubic-cross',lambda x:(x+.08*x**3)@mix+.09*np.roll(x,1,axis=1)*x+.04),('rational',lambda x:(x/(1+.2*x*x))@mix+.04)]:
 cr={'ref_rosette':a,'raw_rosette':fn(a)}; inv=b._calibrator(cr); print('calibration',name,'RMS',rms(inv(fn(test)).reshape(-1,4)-test))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py
if test -f /tmp/hegel_continual_archive/archive_apparatus_log.json; then head -c 4500 /tmp/hegel_continual_archive/archive_apparatus_log.json; else echo 'No public apparatus log'; fi
```
- **Container Exit Code**: `0` (`873 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held shift 0 rms 1.5113389397747338e-16
held shift 1 rms 7.823996521088756e-16
held shift 2 rms 5.85032317617729e-17
blind walls [ 5.33042543  4.88880236  6.4599701   6.72994192 -0.20950303 -2.78255256
  0.276433    0.56855798] wall prediction max-shift RMS 0.006147018098769414
public pulse-free relative errors 0.10093695742407319 0.03310497144853251
public equality/stationarity/gap/field consistency 2.220446049250313e-16 9.71445146547012e-17 1.734723475976807e-18 0.0
stress worst equality/box/stationarity/oracle objective [ 2.66453526e-15 -0.00000000e+00  7.10542736e-15  1.15790044e-10]
calibration affine RMS 5.897460287825592e-17
calibration cubic-cross RMS 0.010204096271002347
calibration rational RMS 0.022284552374751857
No public apparatus log

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81982512e8932545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81982512 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`34832 ms` LLM)
- **Rationale**: The pulse-anchored operator predicts withheld pulses to numerical precision, but removing the pulses materially changes the weights. Compare blind regularization with the nominal closure, and check whether the observation residual has repeatable structure before changing the estimator.
- **Alternatives Weighed**: `Increase blind wall flexibility to lower the in-sample residual` (rejected: More flexibility can absorb observation error into wall coefficients; withheld shifts and pulses test whether that change improves the recovered operator.)

```bash
cat >/tmp/blind_diagnostics.py <<'PY'
import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._calibrator(s['sensor_calibration_sweeps']); ch=b._Chamber(s); y=c(s['y_obs_raw']); wall=b._fit_walls(ch,s,c,y); S=ch.response(wall)[0]; Z=s['Z_nuisance']; T=s['aux_manifold_telemetry']; ij=s['sensor_ij']; ns=len(ij); N=len(y); W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); X=np.c_[S,Z]; e=y-X@np.linalg.lstsq(X,y,rcond=None)[0]; E=e.reshape(-1,ns); print('residual-shift correlation',np.corrcoef(E)); print('residual RMS by bank',np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0))); print('structured telemetry correlations',np.corrcoef(np.c_[e,T.ravel(),np.tile(T,4).ravel()[:N]].T)[0]); print('table sensor x,y, shift residual'); print(np.round(np.c_[ij,E.T],5)); print('item keys',list(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]));
P=b._projector(Z,s['nuisance_ridge']); upper=np.broadcast_to(s['u_box'], S.shape[1]); M=len(upper)
def q(wall):
 m,u,_=ch.response(wall); x=P(m); return b._box_qp(x.T@x+s['source_ridge']*np.eye(M),-x.T@P(y),upper,s['total_budget'])[0],u
wr,ur=q(wall); rms=lambda a:np.sqrt(np.mean(a*a)); shifts=np.arange(N)//ns; Q=np.linalg.qr(Z)[0];
for strength in (0.,.003,.01,.025,.07,.2,1.,np.inf):
 scale=np.r_[ch.prior[:4],ch.scales]; lo=np.r_[np.full(4,.02),-ch.scales*.65]; hi=np.r_[np.maximum(20,ch.prior[:4]*4),-ch.inward*.65]; oof=np.empty(N)
 def fit(keep):
  zz=Z[keep]; proj=b._projector(zz,0.); yy=proj(y[keep]);
  def profile(w):
   x=proj(ch.response(w)[0][keep]); a=b._box_qp(x.T@x+1e-10*np.eye(M),-x.T@yy,upper,s['total_budget'])[0]; return x@a-yy,a
  w=ch.prior if np.isinf(strength) else least_squares(lambda w:np.r_[profile(w)[0],strength*(w-ch.prior)/scale],ch.prior,bounds=(lo,hi),x_scale=scale,max_nfev=42).x
  a=profile(w)[1]; m=ch.response(w)[0]; g=np.linalg.lstsq(zz,y[keep]-m[keep]@a,rcond=None)[0]; return w,m@a+Z@g
 w,_=fit(np.ones(N,dtype=bool)); a,u=q(w)
 for j in range(len(ch.shifts)):
  test=shifts==j; _,pred=fit(~test); oof[test]=pred[test]-y[test]
 print(strength,'held observation',max(rms(r) for r in np.split(oof,len(ch.shifts))),'held pulse',max(rms(r) for r in np.split(ch.response(w,ch.F@W)[0]-Y,len(ch.shifts))),'relative weights and field',np.linalg.norm(a-wr)/np.linalg.norm(wr),np.linalg.norm(u@a-ur@wr)/np.linalg.norm(ur@wr),'walls',np.round(w,3))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/blind_diagnostics.py
```
- **Container Exit Code**: `0` (`3037 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-03]
 [ 2.000e+00  0.000e+00 -6.870e-03 -2.300e-03  6.940e-03]
 [ 2.000e+00  1.000e+01 -1.790e-03  5.430e-03 -3.370e-03]
 [ 8.000e+00  4.000e+00  3.900e-03  1.300e-04 -7.400e-03]
 [ 6.000e+00  2.000e+00  3.360e-03  2.540e-03 -2.620e-03]
 [ 4.000e+00  8.000e+00  6.490e-03  9.450e-03  3.110e-03]
 [ 1.200e+01  8.000e+00  4.950e-03  7.600e-04  1.530e-03]
 [ 2.000e+00  8.000e+00 -1.109e-02  4.660e-03  3.730e-03]
 [ 6.000e+00  1.000e+01 -4.640e-03  2.010e-03  8.720e-03]
 [ 8.000e+00  0.000e+00  3.900e-03  5.100e-04  7.000e-03]
 [ 6.000e+00  4.000e+00  4.170e-03  5.910e-03 -1.142e-02]]
item keys ['study']
0.0 held observation 0.00835069447991882 held pulse 0.006766583023251787 relative weights and field 0.1178893705822148 0.036615338638936276 walls [18.096  0.02   0.02  19.939 -8.194  0.857  4.717 -5.536]
0.003 held observation 0.00839047598860093 held pulse 0.006964244582517107 relative weights and field 0.11895661518936985 0.03781115630978719 walls [ 7.893  3.216  0.02   7.085 -1.832 -1.738  4.72   0.737]
0.01 held observation 0.008427139444470582 held pulse 0.006756472333177071 relative weights and field 0.11300792786515713 0.036940915496890804 walls [ 5.564  4.739  5.818  6.828 -0.393 -2.905  0.709  0.845]
0.025 held observation 0.008534611959325904 held pulse 0.006147008867000467 relative weights and field 0.1009368081381003 0.033104919688342734 walls [ 5.33   4.889  6.46   6.73  -0.209 -2.783  0.276  0.569]
0.07 held observation 0.008810361202569898 held pulse 0.0045181216779031 relative weights and field 0.06205400805634151 0.02072959168929457 walls [ 5.286  5.06   6.55   6.564 -0.222 -1.835  0.087  0.027]
0.2 held observation 0.008893164017269096 held pulse 0.004239473660671358 relative weights and field 0.03800152049804458 0.01836034628443989 walls [ 5.279  5.246  6.527  6.532 -0.231 -0.488 -0.205 -0.069]
1.0 held observation 0.008967568529586507 held pulse 0.005380123224289824 relative weights and field 0.040847583292673566 0.027630951436669455 walls [ 5.301  5.301  6.548  6.551 -0.024 -0.029 -0.045 -0.008]
inf held observation 0.008982915151355889 held pulse 0.005529228238336566 relative weights and field 0.04138389959796935 0.028859439357445157 walls [5.304 5.304 6.554 6.554 0.    0.    0.    0.   ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81982512e8932545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81982512 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`30773 ms` LLM)
- **Rationale**: Test the inferred wall formula without fitting the pulses or the main observation. If it reproduces the tracer responses and both deliverables, use it as the blind baseline rather than fitting around a misspecified stencil.
- **Alternatives Weighed**: `Increase blind regularization around the existing prior` (rejected: It reduces the public blind error but does not explain the fitted coefficients. Their magnitudes and signs match the endpoint-to-face diffusivity differences.)

```bash
cat >/tmp/wall_law.py <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('b','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ch=b._Chamber(s); c=b._calibrator(s['sensor_calibration_sweeps']); y=c(s['y_obs_raw']); fit=b._fit_walls(ch,s,c,y)
wall=np.zeros(8)
for j,a in enumerate('xy'):
 k=np.asarray(s['kappa_'+a]); v=np.asarray(s['vel_'+a]); h=s['L'+a]/(len(k)-1); ix=slice(2*j,2*j+2)
 diffusion=(k[[1,-2]]-k[[0,-1]])/(2*h*h)
 wall[ix]=ch.prior[ix]-diffusion-np.abs(v[[0,-1]])/h
 wall[4+2*j:6+2*j]=-k[[0,-1]]/h**2-ch.inward[ix]
print('inferred wall law coefficients:',wall,'max parameter error',np.max(abs(wall-fit)))
W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([c(p['y_pulse_raw']) for p in s['reference_tracer_pulses']]); e=ch.response(wall,ch.F@W)[0]-Y
print('no-fit shift RMS',np.sqrt(np.mean(e.reshape(len(ch.shifts),-1)**2,axis=1)))
P=b._projector(s['Z_nuisance'],s['nuisance_ridge']); S,U,_=ch.response(wall); X=P(S); u=np.broadcast_to(s['u_box'],S.shape[1]); w,_=b._box_qp(X.T@X+s['source_ridge']*np.eye(len(u)),-X.T@P(y),u,s['total_budget']); o=b.solve_batch(s)
print('no-fit weight and field relative errors',np.linalg.norm(w-o['weights'])/np.linalg.norm(o['weights']),np.linalg.norm(U@w-o['terminal_field'].ravel(order='F'))/np.linalg.norm(o['terminal_field']))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/wall_law.py
```
- **Container Exit Code**: `0` (`479 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
inferred wall law coefficients: [ 2.91666847  3.69815893  6.6273485   5.23442182  0.49151116  0.49151116
 -0.07352025  0.07352025] max parameter error 7.904787935331115e-13
no-fit shift RMS [2.46219234e-16 1.55377331e-16 1.01131656e-16]
no-fit weight and field relative errors 9.624693455644137e-15 2.4204903771195063e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81982512e8932545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81982512 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`29902 ms` LLM)
- **Rationale**: Install the parameter-free wall law for pulse-free runs and retain pulse fitting for measured departures. Recheck the exact artifact for pulse-free agreement, field ordering, and KKT errors so a packaging regression can be distinguished from a model failure.
- **Alternatives Weighed**: `Retune regularization around the old blind stencil` (rejected: The endpoint Robin law reproduces the tracer responses without fitting, whereas profiling the noisy observation shifts the recovered weights.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
s = s.replace('axes, nominal, inward, scales = [], [], [], []', 'axes, nominal, inward, scales, offsets = [], [], [], [], []')
s = s.replace("            nominal.extend([study['robin_'+a]/h]*2)\n            inward.extend([hi[0], lo[-1]])", "            # Boundary rows replace the interior stencil: endpoint diffusion\n            # to the inward neighbor, Robin loss, and no boundary advection.\n            edge = np.array([0, n-1])\n            edge_diffusion = k[edge]/h**2\n            nominal.extend(study['robin_'+a]/h + edge_diffusion - diag[edge])\n            inward.extend([hi[0], lo[-1]])\n            offsets.extend(-edge_diffusion - np.array([hi[0], lo[-1]]))")
s = s.replace('self.prior = np.r_[nominal, np.zeros(4)]', 'self.prior = np.r_[nominal, offsets]')
start = s.index('    scales = np.r_[ch.prior[:4], ch.scales]', s.index('def _fit_walls'))
end = s.index('\ndef solve_batch', start)
s = s[:start] + '''    prior = ch.prior
    if not pulses:
        return prior.copy()
    W = np.column_stack([p['w_pulse'] for p in pulses])
    target = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    rhs = ch.F@W
    def fun(w):
        return (ch.response(w, rhs)[0]-target).ravel()
    def jac(w):
        return ch.response(w, rhs, True)[2].reshape(-1, 8)
    # Exact or nearly exact agreement does not justify fitting extra wall terms.
    if np.sqrt(np.mean(fun(prior)**2)) < 5e-7:
        return prior.copy()
    scales = np.r_[ch.scales, ch.scales]
    lo = prior - .65*scales
    hi = prior + .65*scales
    hi[4:] = np.minimum(hi[4:], prior[4:] - .9*(ch.inward+prior[4:]))
    return least_squares(fun, prior, jac=jac, x_scale=scales, bounds=(lo, hi),
                         gtol=2e-14, ftol=1e-12, xtol=1e-12).x

''' + s[end:]
s = s.replace('def _fit_walls(ch, study, calibrate, y):', 'def _fit_walls(ch, study, calibrate):').replace('_fit_walls(chamber, study, calibrate, y)', '_fit_walls(chamber, study, calibrate)')
p.write_text(s)
p = Path('/tmp/verify_batch.py')
s = p.read_text().replace('_fit_walls(ch,s,c,y)', '_fit_walls(ch,s,c)').replace('_fit_walls(ch,dict(s,reference_tracer_pulses=None),c,y)', '_fit_walls(ch,dict(s,reference_tracer_pulses=None),c)')
p.write_text(s)
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py
```
- **Container Exit Code**: `0` (`757 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held shift 0 rms 2.462192340461645e-16
held shift 1 rms 1.5537733070847887e-16
held shift 2 rms 1.011316564842877e-16
blind walls [ 2.91666847  3.69815893  6.6273485   5.23442182  0.49151116  0.49151116
 -0.07352025  0.07352025] wall prediction max-shift RMS 2.462192340461645e-16
public pulse-free relative errors 0.0 0.0
public equality/stationarity/gap/field consistency 0.0 6.591949208711867e-17 0.0 0.0
stress worst equality/box/stationarity/oracle objective [ 2.66453526e-15 -0.00000000e+00  7.10542736e-15  1.15790044e-10]
calibration affine RMS 5.897460287825592e-17
calibration cubic-cross RMS 0.010204096271002347
calibration rational RMS 0.022284552374751857

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d876e98ed023656c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d876e98e (directory).
```

### Turn 13 — `execute_in_remote_docker` (`17551 ms` LLM)
- **Rationale**: The boundary law and QP already pass their checks. Inspect just the calibration data needed to test a forward-response fit and determine whether the stress observations represent interpolation or extrapolation.
- **Alternatives Weighed**: `Submit the verified boundary and QP implementation unchanged` (rejected: The synthetic nonlinear calibration checks exposed a material error. Sweep ranges and the isolated-bank layout will determine whether a forward fit can fix it without disturbing the public solve.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('items',len(b)); s=b[0]['study']; print('study keys',list(s)); c=s['sensor_calibration_sweeps']; print('calibration keys',list(c));
for k,v in c['single_bank_sweeps'].items():
 a=np.asarray(v); print(k,a.shape,a[:8],a.min(),a.max())
for k in ('ref_rosette','raw_rosette'):
 a=np.asarray(c[k]); print(k,a.shape,a.min(0),a.max(0),a.std(0))
print('observation min/max by bank',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))
PY
sed -n '1,61p' /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`329 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[])])


def _calibrator(sweeps):
    ref, raw = (np.asarray(sweeps[k], dtype=float) for k in ('ref_rosette', 'raw_rosette'))
    banks = sweeps.get('single_bank_sweeps')
    n = len(ref)
    # Score inverse maps on held-out rosette sweeps. Fit small departures from
    # the identity so the nominal-map comparison includes the measured offset.
    models = [(0, False), (1, False), (2, False), (3, False), (2, True), (3, True)]
    delta = ref - raw
    scores = [np.mean(delta**2)]
    for degree, cross in models[1:]:
        x = _features(raw, degree, cross)
        err = np.empty_like(delta)
        for f in range(5):
            train = np.arange(n) % 5 != f
            coef = np.linalg.lstsq(x[train], delta[train], rcond=None)[0]
            err[~train] = x[~train] @ coef - delta[~train]
        scores.append(np.mean(err**2))
    cutoff = min(scores)*1.06 + 1e-27
    degree, cross = next(m for m, e in zip(models, scores) if e <= cutoff)
    if degree == 0:
        return lambda y: np.asarray(y, dtype=float)
    coefs = []
    for i in range(4):
        x = _features(raw, degree, cross)
        target = delta[:, i]
        if banks is not None:
            j = np.asarray(banks['bank_id']) == i
            sample = np.asarray(banks['raw_readout'])[j]
            truth = np.asarray(banks['ref_signal'])[j]
            # Isolated bank sweeps give the diagonal response with all other
            # reference channels at zero. Infer their raw baselines from the
            # full sweep before using these points in the inverse regression.
            forward = np.linalg.lstsq(_features(ref, degree, cross), raw, rcond=None)[0]
            isolated = np.zeros((len(sample), 4)); isolated[:, i] = truth
            simulated = _features(isolated, degree, cross) @ forward
            simulated[:, i] = sample
            x = np.vstack([x, _features(simulated, degree, cross)])
            target = np.r_[target, truth-sample]
        coefs.append(np.linalg.lstsq(x, target, rcond=None)[0])
    coef = np.column_stack(coefs)
    def apply(y):
        a = np.asarray(y, dtype=float).reshape(-1, 4)
        return (a + _features(a, degree, cross) @ coef).ravel()
    return apply


class _Chamber:
    def __init__(self, study):

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d876e98ed023656c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d876e98e (directory).
```

### Turn 14 — `execute_in_remote_docker` (`45271 ms` LLM)
- **Rationale**: Test whether the forward fit reduces nonlinear calibration errors while preserving the measured public pulse and QP results. If it improves the synthetic cases but degrades the public anchor, keep the verified calibration for the public response regime.
- **Alternatives Weighed**: `Increase the degree of the direct inverse polynomial` (rejected: A forward model can represent saturation with fewer terms and can use isolated-bank measurements directly, without imputing unmeasured raw channels.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
a, z = s.index('def _features'), s.index('class _Chamber:')
s = s[:a] + '''def _features(x, degree, cross, saturation=None):
    parts = [np.ones((len(x), 1)), x]
    if saturation is not None:
        # This basis spans both a linear component and x/(1+a*x*x),
        # without becoming degenerate as a approaches zero.
        parts.append(x**3/(1+saturation*x*x))
    else:
        parts.extend(x**d for d in range(2, degree+1))
    for i, j in cross:
        parts.append((x[:, i]*x[:, j])[:, None])
        if degree == 4:
            parts.extend(((x[:, i]**2*x[:, j])[:, None], (x[:, i]*x[:, j]**2)[:, None]))
    if degree == 4:
        parts.extend((x[:, i]*x[:, j]*x[:, k])[:, None]
                     for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4))
    return np.column_stack(parts)


def _calibrator(sweeps):
    ref, raw = (np.asarray(sweeps[k], dtype=float) for k in ('ref_rosette', 'raw_rosette'))
    delta = raw-ref
    banks = sweeps.get('single_bank_sweeps')
    if banks is not None:
        bank = np.asarray(banks['bank_id'], dtype=int)
        signal = np.asarray(banks['ref_signal'])
        isolated = np.zeros((len(bank), 4)); isolated[np.arange(len(bank)), bank] = signal
        isolated_delta = np.asarray(banks['raw_readout'])-signal
    pairs = tuple((i, j) for i in range(4) for j in range(i+1, 4))
    models = [(0, (), False), (1, (), False), (2, (), False), (2, pairs, False),
              (3, (), False), (3, pairs, False), (4, pairs, False),
              (3, (), True), (3, pairs, True)]
    indices = np.arange(len(ref))

    def fit(model, keep):
        degree, cross, rational = model
        def coefficients(saturation):
            X = _features(ref[keep], degree, cross, saturation)
            if banks is None:
                return np.linalg.lstsq(X, delta[keep], rcond=None)[0]
            B = _features(isolated, degree, cross, saturation)
            return np.column_stack([np.linalg.lstsq(np.vstack((X, B[bank == i])),
                                   np.r_[delta[keep, i], isolated_delta[bank == i]],
                                   rcond=None)[0] for i in range(4)])
        saturation = None
        if rational:
            def residual(a):
                coef = coefficients(a)
                e = _features(ref[keep], degree, cross, a) @ coef - delta[keep]
                if banks is None:
                    return e.ravel()
                b = (_features(isolated, degree, cross, a)@coef)[np.arange(len(bank)), bank]
                return np.r_[e.ravel(), b-isolated_delta]
            saturation = least_squares(residual, np.full(4, .15), bounds=(0., 4.),
                                       gtol=1e-13, max_nfev=40).x
        return saturation, coefficients(saturation)

    scores = [np.mean(delta**2)]
    for model in models[1:]:
        degree, cross, rational = model
        err = np.empty_like(delta)
        for f in range(5):
            keep = indices % 5 != f
            sat, coef = fit(model, keep)
            err[~keep] = _features(ref[~keep], degree, cross, sat) @ coef-delta[~keep]
        scores.append(np.mean(err**2))
    cutoff = min(scores)*1.06 + 1e-27
    model = next(m for m, score in zip(models, scores) if score <= cutoff)
    degree, cross, _ = model
    if degree == 0:
        return lambda y: np.asarray(y, dtype=float)
    sat, coef = fit(model, np.ones(len(ref), dtype=bool))
    linear = np.eye(4) + coef[1:5]
    def forward(x):
        return x+_features(x, degree, cross, sat)@coef
    def apply(y):
        raw = np.asarray(y, dtype=float).reshape(-1, 4)
        x = np.linalg.lstsq(linear.T, (raw-coef[0]).T, rcond=None)[0].T
        if degree == 1:
            return x.ravel()
        # Damped Newton iteration starting at the affine inverse. Each
        # accepted update has a smaller raw-space error for that rosette.
        e = forward(x)-raw
        steps = np.eye(4)*1e-5
        for _ in range(20):
            if np.max(np.abs(e), initial=0.) < 1e-12:
                break
            J = np.stack([(forward(x+h)-forward(x-h))/2e-5 for h in steps], axis=2)
            inv = np.linalg.pinv(J, rcond=1e-9)
            d = (inv@e[:, :, None])[:, :, 0]
            d /= np.maximum(1., np.max(np.abs(d), axis=1, keepdims=True))
            old = np.sum(e*e, axis=1)
            pending = np.ones(len(x), dtype=bool)
            for t in (1., .5, .25, .125, .0625, .03125):
                trial = x-t*d
                candidate = forward(trial)-raw
                take = pending & (np.sum(candidate*candidate, axis=1) < old)
                x[take], e[take] = trial[take], candidate[take]
                pending[take] = False
            if pending.all():
                break
        return x.ravel()
    return apply


''' + s[z:]
p.write_text(s)
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py
```
- **Container Exit Code**: `0` (`1645 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held shift 0 rms 2.462192340461645e-16
held shift 1 rms 1.5537733070847887e-16
held shift 2 rms 1.011316564842877e-16
blind walls [ 2.91666847  3.69815893  6.6273485   5.23442182  0.49151116  0.49151116
 -0.07352025  0.07352025] wall prediction max-shift RMS 2.462192340461645e-16
public pulse-free relative errors 0.0 0.0
public equality/stationarity/gap/field consistency 0.0 6.591949208711867e-17 0.0 0.0
stress worst equality/box/stationarity/oracle objective [ 2.66453526e-15 -0.00000000e+00  7.10542736e-15  1.15790044e-10]
calibration affine RMS 1.8580918800462402e-16
calibration cubic-cross RMS 6.86184168381458e-17
calibration rational RMS 3.5113227643284295e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=113a609c42898cbd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_113a609c (directory).
```

### Turn 15 — `submit_completed_task` (`15775 ms` LLM)
- **Rationale**: The final artifact passed isolated-process schema, tracer, field-ordering, calibration, and QP checks. The boundary correction is supported by the public tracer measurements; no claim is made about transfer across unseen apparatus histories.
- **Alternatives Weighed**: `Infer wall corrections from the main observation without pulses` (rejected: On the public campaign this shifted weights by 10.09% relative to the tracer-anchored solve; the endpoint Robin stencil matched that solve without fitting the observation.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` with `solve_batch(study)`. The public schema check passed, and 150 QP stress tests had a worst-case stationarity residual of 7.11e-15. On the public campaign, withholding tracer pulses left the weights and terminal field unchanged. Archive-dependent behavior could not be verified because no apparatus log was mounted.`

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
