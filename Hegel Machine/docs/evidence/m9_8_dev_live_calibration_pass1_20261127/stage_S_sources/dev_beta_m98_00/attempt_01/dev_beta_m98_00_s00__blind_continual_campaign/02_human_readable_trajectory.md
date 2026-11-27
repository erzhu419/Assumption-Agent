# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m98_00_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `340,887` (`326,390` / `122,496` / `14,497`) |
| **Total Episode Wall Time (s)** | `427.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`17330 ms` LLM)
- **Rationale**: Inspect the input contract and bounded archive metadata to identify which calibration and boundary parameters can be inferred before implementing the inverse solver.

```bash
ls -la /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def show(x,d=0):
 if isinstance(x,dict) and d<3:
  for k,v in x.items():
   print(' '*d,k,end=': ');show(v,d+1)
 elif hasattr(x,'shape'): print(type(x).__name__,x.shape,x.dtype, x.ravel()[:6])
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x));
  if len(x) and d<3:show(x[0],d+1)
 else:print(repr(x)[:300])
show(s)
PY
if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 12000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rrs.append(f"missing_kkt:{kk}")
        if not errs:
            kw = np.asarray(kc["weights"], dtype=float); ll = np.asarray(kc["lam_lower"], dtype=float); lu = np.asarray(kc["lam_upper"], dtype=float)
            po = float(kc["primal_objective"]); dl = float(kc["dual_lower_bound"])
            if np.min(kw) < -1e-6 or np.max(kw) > float(st["u_box"]) + 1e-6: errs.append("kkt_box_violation")
            if abs(float(np.sum(kw)) - float(st["total_budget"])) > 1e-5: errs.append("kkt_budget_violation")
            if np.min(ll) < -1e-6 or np.min(lu) < -1e-6: errs.append("kkt_neg_multiplier")
            if dl < -1e-7 or po < dl - 1e-6 or (po - dl) > 2e-4: errs.append(f"kkt_duality_gap_or_sign:{po:.6f}_vs_{dl:.6f}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
list 1
  study:    Lx: 2.4
   Ly: 2.0
   nx: 14
   ny: 12
   kappa_x: ndarray (14,) float64 [0.42130881 0.454813   0.48637006 0.51414598 0.53652655 0.55221109]
   kappa_y: ndarray (12,) float64 [0.51535314 0.5104923  0.49630357 0.47393643 0.44520295 0.41243092]
   vel_x: ndarray (14,) float64 [0.35       0.34500413 0.33015913 0.30588881 0.27288602 0.23209293]
   vel_y: ndarray (12,) float64 [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099]
   robin_x: 0.9023444699977767
   robin_y: 1.1567834842388094
   V_reaction_2d: ndarray (14, 12) float64 [0.25 0.25 0.25 0.25 0.25 0.25]
   mu_shifts: ndarray (3,) float64 [0.15 0.55 1.1 ]
   source_modes: ndarray (8, 14, 12) float64 [0.00097197 0.00719273 0.03367342 0.09973152 0.1868656  0.2215022 ]
   sensor_ij: ndarray (32, 2) int64 [12  2  2  2 10  6]
   row_bank_id: ndarray (96,) int64 [0 1 2 3 0 1]
   sensor_calibration_sweeps: {'ref_rosette': array([[-1.2       ,  0.        ,  0.        ,  0.        ],
       [-0.4       ,  0.        ,  0.        ,  0.        ],
       [ 0.5       ,  0.        ,  0.        ,  0.        ],
       [ 1.3       ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -1.2       ,  0.   
   Z_nuisance: ndarray (96, 3) float64 [-0.39100862 -0.8425231  -0.74584925  0.47635506 -0.94900402 -0.20782659]
   y_obs_raw: ndarray (96,) float64 [ 0.11649357 -0.11772128  0.24816319 -0.22450744  0.24200413  0.03886201]
   reference_tracer_pulses: list 2
   aux_manifold_telemetry: ndarray (24, 4) float64 [-0.67131433 -0.81728537 -1.31265215  2.13519789 -1.26179512 -0.48693358]
   nuisance_ridge: 0.08
   source_ridge: 0.04
   u_box: 0.42
   total_budget: 1.4
```
- **Stderr (tail)**:
```text
ls: cannot access '/tmp/hegel_continual_archive': No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`23405 ms` LLM)
- **Rationale**: Held-out mixed sweeps can distinguish affine response from nonlinear coupling; pulse and bank layouts determine how to test the wall operator independently of the source inversion.

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; print('sweep keys',c.keys()); np.set_printoptions(precision=5,suppress=True,linewidth=130)
for k,v in c.items():
 if k not in ('ref_rosette','raw_rosette'): print(k,repr(v)[:4000])
x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); print('first sweeps',np.c_[x,y][:20]); print('mixed sweeps',np.c_[x,y][20:27]);
for degree in [1,2,3,4]:
 from itertools import combinations_with_replacement
 A=np.column_stack([np.ones(len(x))]+[x[:,ind].prod(1) for i in range(1,degree+1) for ind in combinations_with_replacement(range(4),i)]); b=np.linalg.lstsq(A[:-16],y[:-16],rcond=None)[0]; print(degree,'train and held out RMSE',*[np.mean(z*z)**.5 for z in [A[:-16]@b-y[:-16],A[-16:]@b-y[-16:]]]);
 if degree==1: print('affine',b)
print('sensor_ij',s['sensor_ij'].T); print('telemetry',s['aux_manifold_telemetry'][:8]); print('pulse shapes, weights and raw first',[(p['w_pulse'],p['y_pulse_raw'][:8]) for p in s['reference_tracer_pulses']]); print('bank repetitions',s['row_bank_id'].reshape(-1,4)[:6]);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -20
```
- **Container Exit Code**: `0` (`408 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`34713 ms` LLM)
- **Rationale**: Compare nominal, axis-shared, and side-specific wall models on held-out pulses. If the nominal model matches pulse noise, retain it; otherwise require a correction to transfer across pulses before using it in the inverse solve.
- **Alternatives Weighed**: `Fit an unrestricted correction to the sensor response matrix` (rejected: Two tracer pulses constrain only two source directions; testing shared wall parameters preserves predictions for unseen source modes and the terminal field.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -28
cat > /tmp/wall_probe.py <<'PY'
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

def axis(k,v,h,r, p):
    f=(k[1:]+k[:-1])/2/h**2
    lo=-np.r_[f[:-1]+np.maximum(v[1:-1],0)/h,k[-1]/h**2*p[3]]
    hi=-np.r_[k[0]/h**2*p[2], f[1:]+np.maximum(-v[1:-1],0)/h]
    d=np.r_[-hi[0]+r/h*p[0], f[:-1]+f[1:]+abs(v[1:-1])/h, -lo[-1]+r/h*p[1]]
    return diags([lo,d,hi],[-1,0,1],format='csc')

def response(s,p,modes=None):
    nx,ny=s['nx'],s['ny']; ix,iy=eye(nx),eye(ny)
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:4]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[4:]); L=(kron(iy,ax)+kron(ay,ix)+diags(s['V_reaction_2d'].ravel('F'))).tocsc(); I=eye(nx*ny,format='csc')
    modes=s['source_modes'] if modes is None else modes
    rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T
    obs=[]; ij=np.asarray(s['sensor_ij']); ns=len(s['y_obs_raw'])//len(s['mu_shifts'])
    for a,mu in enumerate(s['mu_shifts']):
        u=splu(L+mu*I).solve(rhs)
        pts=ij if len(ij)==ns else ij[a*ns:(a+1)*ns]
        obs.append(u[pts[:,0]+nx*pts[:,1]])
    return np.vstack(obs),u.T.reshape(-1,ny,nx).transpose(0,2,1)

if __name__=='__main__':
 import pickle
 from scipy.optimize import least_squares
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; print('identity sweep RMSE/max',np.sqrt(np.mean((x-y)**2)),np.max(abs(x-y)))
 print('pulse shapes',[(p['w_pulse'].shape,p['y_pulse_raw'].shape) for p in s['reference_tracer_pulses']]); print('ij bounds',s['sensor_ij'].min(0),s['sensor_ij'].max(0)); print('telemetry mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('item keys',pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0].keys())
 P=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in P]); Y=np.stack([p['y_pulse_raw'] for p in P],axis=1); M=np.einsum('pm,mxy->pxy',W,s['source_modes']); base=np.ones(8); D=response(s,base,M)[0]-Y; print('nominal pulse RMSE/max by pulse',np.sqrt(np.mean(D**2,axis=0)),np.max(abs(D),axis=0)); print('nominal residual by shift',np.sqrt(np.mean(D.reshape(3,-1,2)**2,axis=1)))
 for groups in [[[0,1],[4,5]],[[0],[1],[4],[5]],[[0],[1],[4],[5],[2],[3],[6],[7]]]:
  def params(t):
   p=base.copy()
   for z,g in zip(t,groups): p[g]=z
   return p
  for train in [[0],[1],[0,1]]:
   def fun(t): return (response(s,params(t),M)[0]-Y)[:,train].ravel()
   opt=least_squares(fun,np.ones(len(groups)),bounds=(0,10),gtol=1e-13,ftol=1e-13,xtol=1e-13); D=response(s,params(opt.x),M)[0]-Y; print('groups',groups,'train',train,'p',np.round(params(opt.x),6),'RMSE',np.sqrt(np.mean(D**2,axis=0)),'worst',np.max(abs(D)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
```
- **Container Exit Code**: `0` (`707 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nergy >= 0."
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
identity sweep RMSE/max 9.876394415149639e-06 3.218186882469887e-05
pulse shapes [((8,), (96,)), ((8,), (96,))]
ij bounds [0 0] [12 10]
telemetry mean/std [ 0.08484326 -0.07238645 -0.26906385  0.16277306] [1.18833604 0.86661921 0.87744644 1.04950912]
item keys dict_keys(['study'])
nominal pulse RMSE/max by pulse [1.33352724e-16 1.56317641e-16] [3.33066907e-16 5.82867088e-16]
nominal residual by shift [[1.76389531e-16 2.34086455e-16]
 [1.31644826e-16 1.21346976e-16]
 [7.00372785e-17 6.15146923e-17]]
groups [[0, 1], [4, 5]] train [0] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0, 1], [4, 5]] train [1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0, 1], [4, 5]] train [0, 1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5]] train [0] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5]] train [1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5]] train [0, 1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5], [2], [3], [6], [7]] train [0] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5], [2], [3], [6], [7]] train [1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
groups [[0], [1], [4], [5], [2], [3], [6], [7]] train [0, 1] p [1. 1. 1. 1. 1. 1. 1. 1.] RMSE [1.33352724e-16 1.56317641e-16] worst 5.828670879282072e-16
```

### Turn 4 — `execute_in_remote_docker` (`78476 ms` LLM)
- **Rationale**: Build the full estimator and run the schema gate first to detect output or certificate failures. The public pulses support the baseline; the next stress checks should target changed calibration, altered walls, and active box faces.
- **Alternatives Weighed**: `Force a non-nominal wall model on the measured campaign` (rejected: The nominal operator already reproduces both pulses to floating-point precision.)

```bash
cd /holdout_task
cat > batch_solver.py <<'PY'
"""Finite-volume chamber inversion and box-constrained source estimation."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(k, v, h, robin, wall):
    face = 0.5 * (k[1:] + k[:-1]) / h**2
    left = -np.r_[face[:-1] + np.maximum(v[1:-1], 0) / h, k[-1] * wall[3] / h**2]
    right = -np.r_[k[0] * wall[2] / h**2, face[1:] + np.maximum(-v[1:-1], 0) / h]
    center = np.r_[-right[0] + robin * wall[0] / h,
                   face[:-1] + face[1:] + np.abs(v[1:-1]) / h,
                   -left[-1] + robin * wall[1] / h]
    return diags((left, center, right), (-1, 0, 1), format='csc')


def _response(study, wall, modes):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), study['robin_x'], wall[:4])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), study['robin_y'], wall[4:])
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), -1).T
    ij = np.asarray(study['sensor_ij'])
    shifts = study['mu_shifts']
    count = len(study['y_obs_raw']) // len(shifts)
    blocks = []
    for i, mu in enumerate(shifts):
        field = splu(operator + mu * identity).solve(rhs)
        sensors = ij if len(ij) == count else ij[i * count:(i + 1) * count]
        blocks.append(field[sensors[:, 0] + nx * sensors[:, 1]])
    return np.vstack(blocks), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _powers(degree):
    return np.array([np.bincount(c, minlength=4) for d in range(degree + 1)
                     for c in combinations_with_replacement(range(4), d)])


def _features(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=-1)


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    folds = np.arange(len(x)) % 4
    models = [None] + [_powers(d) for d in range(1, 4)]
    errors = []
    for p in models:
        if p is None:
            err = x - y
        else:
            a = _features(x, p)
            err = np.empty_like(y)
            for k in range(4):
                train = folds != k
                err[~train] = a[~train] @ np.linalg.lstsq(a[train], (y - x)[train], rcond=None)[0] + x[~train] - y[~train]
        errors.append(np.max([np.sqrt(np.mean(err[folds == k]**2)) for k in range(4)]))
    p = models[next(i for i, e in enumerate(errors) if e <= 1.12 * min(errors) + 1e-12)]
    if p is None:
        return lambda raw: np.array(raw, dtype=float, copy=True)
    coeff = np.linalg.lstsq(_features(x, p), y - x, rcond=None)[0]
    matrix = np.eye(4) + coeff[1:5]
    derivative_powers = [np.maximum(0, p - e) for e in np.eye(4, dtype=int)]

    def unmix(raw):
        values = np.asarray(raw, dtype=float).reshape(-1, 4)
        estimate = np.linalg.solve(matrix.T, (values - coeff[0]).T).T
        if len(p) > 5:
            for i, goal in enumerate(values):
                def fun(q):
                    return q + _features(q[None, :], p)[0] @ coeff - goal

                def jac(q):
                    delta = np.array([(_features(q[None, :], dp)[0] * p[:, j]) @ coeff
                                      for j, dp in enumerate(derivative_powers)])
                    return np.eye(4) + delta.T

                estimate[i] = least_squares(fun, estimate[i], jac=jac, gtol=1e-12).x
        return estimate.reshape(np.shape(raw))

    return unmix


def _walls(study, calibrate, source):
    nominal = np.ones(8)
    pulses = study.get('reference_tracer_pulses')
    if not pulses:
        return nominal
    modes = np.array([np.tensordot(p['w_pulse'], source, axes=1) for p in pulses])
    observed = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    scale = max(np.sqrt(np.mean(observed**2)), 0.01)
    if np.sqrt(np.mean((_response(study, nominal, modes)[0] - observed)**2)) < 2e-4 * scale:
        return nominal
    groups = ([0, 1], [4, 5]), ([0], [1], [4], [5]), tuple([i] for i in range(8))
    best = (np.inf, nominal)
    for group in groups:
        def expand(x):
            result = nominal.copy()
            for z, inds in zip(x, group):
                result[inds] = z
            return result

        def fit(columns):
            def residual(x):
                return ((_response(study, expand(x), modes)[0] - observed)[:, columns] / scale).ravel()
            return expand(least_squares(residual, np.ones(len(group)), bounds=(0., np.inf),
                                       max_nfev=60, gtol=1e-10).x)

        wall = fit(slice(None))
        if len(pulses) > 1:
            score = max(np.sqrt(np.mean((_response(study, fit(np.arange(len(pulses)) != i), modes)[0][:, i]
                                         - observed[:, i])**2)) for i in range(len(pulses)))
        else:
            score = np.sqrt(np.mean((_response(study, wall, modes)[0] - observed)**2))
        if score < best[0]:
            best = (score, wall)
        if score < 2e-4 * scale:
            return wall
    baseline = np.max(np.sqrt(np.mean((_response(study, nominal, modes)[0] - observed)**2, axis=0)))
    return best[1] if best[0] < baseline else nominal


def _project(q, cap, total):
    if total <= 0:
        return np.zeros_like(q)
    if total >= len(q) * cap:
        return np.full_like(q, cap)
    lo, hi = np.min(q) - cap, np.max(q)
    for _ in range(53):
        mid = (lo + hi) * .5
        if np.clip(q - mid, 0., cap).sum() > total:
            lo = mid
        else:
            hi = mid
    return np.clip(q - (lo + hi) * .5, 0., cap)


def _dual(g, w, cap):
    free = (w > 1e-10) & (w < cap - 1e-10)
    if free.any():
        return float(np.mean(g[free]))
    upper = w > cap / 2
    if upper.all():
        return float(g.max())
    if not upper.any():
        return float(g.min())
    return float(.5 * (g[upper].max() + g[~upper].min()))


def _weights(s, y, ridge, cap, total):
    h = s.T @ s + ridge * np.eye(s.shape[1])
    b = s.T @ y
    w = np.full(len(b), total / len(b))
    eig = np.linalg.eigvalsh(h)
    lip = eig[-1]
    if lip == 0 or total <= 0 or total >= len(b) * cap:
        return w
    beta = (np.sqrt(lip) - np.sqrt(max(eig[0], 0.))) / (np.sqrt(lip) + np.sqrt(max(eig[0], 0.)))
    q = w.copy()
    for i in range(2000):
        nxt = _project(q - (h @ q - b) / lip, cap, total)
        q = nxt + beta * (nxt - w)
        w = nxt
        g = h @ w - b
        d = g - _dual(g, w, cap)
        gap = np.maximum(d, 0.) @ w + np.maximum(-d, 0.) @ (cap - w)
        if gap < 1e-12:
            break
        if i % 8 == 0:
            # Solve on the current face, accepting only a feasible KKT point.
            free = (w > 1e-10) & (w < cap - 1e-10)
            if free.any():
                trial = np.where(w >= cap - 1e-10, cap, 0.)
                hf = h[np.ix_(free, free)]
                ones = np.ones(free.sum())
                try:
                    sol, direction = np.linalg.solve(hf, np.column_stack((b[free] - h[free] @ trial, ones))).T
                except np.linalg.LinAlgError:
                    continue
                nu = (total - trial.sum() - sol.sum()) / direction.sum()
                trial[free] = sol + nu * direction
                d = h @ trial - b - nu
                if (trial.min() >= -1e-12 and trial.max() <= cap + 1e-12 and
                    np.min(d[trial <= 0], initial=0.) >= -1e-11 and
                    np.max(d[trial >= cap], initial=0.) <= 1e-11):
                    return np.clip(trial, 0., cap)
    return w


def solve_batch(study: dict) -> dict:
    source = np.asarray(study['source_modes'])
    calibrate = _calibration(study['sensor_calibration_sweeps'])
    wall = _walls(study, calibrate, source)
    response, fields = _response(study, wall, source)
    observed = calibrate(study['y_obs_raw'])
    z = np.asarray(study['Z_nuisance'])
    pz = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s = response - z @ (pz @ response)
    y = observed - z @ (pz @ observed)
    ridge, box, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _weights(s, y, ridge, box, budget)
    residual = s @ w - y
    obj = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    nu = _dual(g, w, box)
    lower = np.maximum(0., g - nu)
    upper = np.maximum(0., nu - g)
    dual = obj - float(g @ w) + nu * budget - box * float(upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': obj, 'dual_lower_bound': dual}}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`508 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cd850921c2eebcba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cd850921 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`41515 ms` LLM)
- **Rationale**: Stress the exact artifact in an isolated process on altered sensor and wall models, comparing recovered weights and fields with an independent optimizer. This distinguishes calibration or wall errors from an internally valid but physically mismatched KKT certificate.

```bash
cat > /tmp/stress_batch.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s0 = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
rng = np.random.default_rng(32)
x = s0['sensor_calibration_sweeps']['ref_rosette']
a = np.eye(4) + rng.normal(0, .08, (4, 4))
b = np.array([.023, -.036, .04, .01])
def transform(z, mode):
    x = np.asarray(z).reshape(-1,4)
    if mode == 'identity':
        return np.asarray(z).copy()
    y = x @ a + b
    for j in range(4):
        p,q,r = (j+1)%4,(j+2)%4,(j+3)%4
        if mode == 'cubic':
            y[:,j] += .19*x[:,p]*x[:,q] + .12*x[:,p]*x[:,q]*x[:,r]
        if mode == 'rational':
            y[:,j] += .19*x[:,p]*x[:,q]/(1+.8*(x[:,r]-.4*x[:,j])**2)
    return y.reshape(np.shape(z))
source = s0['source_modes']
w0 = np.array([.42,.24,0,.08,.42,0,.24,0])
W = np.array([p['w_pulse'] for p in s0['reference_tracer_pulses']])
Z = s0['Z_nuisance']; rho = s0['nuisance_ridge']; R = np.eye(len(Z))-Z @ np.linalg.solve(Z.T@Z+rho*np.eye(Z.shape[1]),Z.T)
max_gap = 0
for mode, wall, pulses in [('identity',[1]*8,False),('affine',[1]*8,True),('cubic',[.75,.75,1,1,1.55,1.55,1,1],True),('affine',[.7,1.4,.52,1.25,1.6,.65,1.3,.6],True),('rational',[1]*8,True)]:
    s = dict(s0)
    s['sensor_calibration_sweeps'] = dict(s0['sensor_calibration_sweeps'],raw_rosette=transform(x,mode)+rng.normal(0,1e-5,x.shape))
    S, U = m._response(s, np.asarray(wall), source)
    yc = S @ w0 + Z @ [.02,-.015,.06] + rng.normal(0,.005,len(Z))
    s['y_obs_raw'] = transform(yc,mode)
    s['reference_tracer_pulses'] = [dict(w_pulse=w,y_pulse_raw=transform(S @ w,mode)) for w in W] if pulses else None
    out = m.solve_batch(s)
    S2,y2 = R @ S,R @ yc
    ridge = s['source_ridge']
    H, f = S2.T @ S2 + ridge*np.eye(len(w0)), S2.T @ y2
    opt = minimize(lambda w: .5*w@H@w-f@w+w[0]*0.,np.ones(len(w0))*s['total_budget']/len(w0),jac=lambda w: H@w-f,method='SLSQP',bounds=[(0,s['u_box'])]*len(w0),constraints=[dict(type='eq',fun=lambda w:w.sum()-s['total_budget'],jac=lambda w:np.ones_like(w))],options={'ftol':1e-14,'maxiter':300})
    cert=out['kkt_certificate']; gap=cert['primal_objective']-cert['dual_lower_bound']; max_gap=max(max_gap,gap)
    cal=m._calibration(s['sensor_calibration_sweeps']); fitted_wall = m._walls(s,cal,source)
    g=S2.T@(S2@out['weights']-y2)+ridge*out['weights']
    stat=g-cert['nu_sum']-np.array(cert['lam_lower'])+np.array(cert['lam_upper'])
    print(mode,'pulses',pulses,'cal max',np.max(abs(cal(s['y_obs_raw'])-yc)),'wall max',np.max(abs(fitted_wall-wall)),'weight max',np.max(abs(out['weights']-opt.x)),'field max',np.max(abs(out['terminal_field']-np.tensordot(opt.x,U,axes=1))),'oracle KKT residual',np.max(abs(stat)),'internal gap',gap)
    assert opt.success and abs(sum(out['weights'])-s['total_budget']) < 1e-9 and abs(gap)<1e-8
print('max internal gap',max_gap)
for cap,total in [(.3,0.),(.3,.3),(.3,1.7),(.3,2.4)]:
    X=np.eye(8); y=np.arange(8.)
    w=m._weights(X,y,.04,cap,total)
    g=1.04*w-y; nu=m._dual(g,w,cap); d=g-nu
    gap=np.maximum(d,0)@w+np.maximum(-d,0)@(cap-w)
    print('active-face total',total,'gap',gap,'budget error',abs(w.sum()-total))
    assert abs(w.sum()-total)<1e-9 and gap<1e-8
archive=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
print('apparatus log present:',archive.is_file())
if archive.is_file():
    with archive.open() as f: print(f.read(3500))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_batch.py
```
- **Container Exit Code**: `0` (`7630 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
identity pulses False cal max 0.0 wall max 0.0 weight max 1.809283259601724e-07 field max 2.2442054603555306e-09 oracle KKT residual 2.7755575615628914e-17 internal gap 6.765421556309548e-17
affine pulses True cal max 1.263022702200408e-06 wall max 0.0 weight max 3.329827048759171e-06 field max 3.6951835638565633e-07 oracle KKT residual 1.0647382332109462e-06 internal gap 1.1102230246251565e-16
cubic pulses True cal max 5.055790570203389e-06 wall max 5.009214418105756e-05 weight max 1.2739851583953854e-05 field max 1.6097105826340385e-06 oracle KKT residual 3.0505639466715806e-06 internal gap 6.765421556309548e-17
affine pulses True cal max 2.0004123991307354e-06 wall max 0.0005268904823558795 weight max 5.601016491885158e-06 field max 7.759237088070942e-07 oracle KKT residual 6.725372002289176e-07 internal gap 1.1535911115245767e-16
rational pulses True cal max 0.02273153875764619 wall max 0.49839343493488353 weight max 0.06479437370461547 field max 0.008071094942996043 oracle KKT residual 0.015845117178172774 internal gap 9.367506770274758e-17
max internal gap 1.1535911115245767e-16
active-face total 0.0 gap 0.0 budget error 0.0
active-face total 0.3 gap 3.0553337637684335e-16 budget error 8.881784197001252e-16
active-face total 1.7 gap 0.0 budget error 2.220446049250313e-16
active-face total 2.4 gap 0.0 budget error 0.0
apparatus log present: False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cd850921c2eebcba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cd850921 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`49564 ms` LLM)
- **Rationale**: The stress test separates optimization error from physical mismatch. Add the missing denominator term, retain simpler models when sweeps support them, and rerun the same oracle comparison to check whether the wall bias disappears.
- **Alternatives Weighed**: `Increase the polynomial degree` (rejected: A larger polynomial basis does not encode saturation and can fit sweep noise without improving inversion near operating signals.)
- **ATMS NOGOOD DECLARED**: Refuted `Cubic polynomial calibration is sufficient to separate sensor distortion from wall mismatch in the saturation stress slice.` | Witness: `With nominal walls and rational rosette coupling, the polynomial solver produced 0.02273 maximum calibration error, 0.49839 spurious wall-factor error, and 0.06479 weight error despite an internal KKT gap below 1e-16.` | Blanket Diagnosis: `The wall estimator absorbed structured sensor error left by the polynomial basis.` | Sublated Pivot: `Fit a bounded rational coupling from sweep data alone and select it by held-out sweep error before estimating walls.`

```bash
cd /holdout_task
cat > /tmp/rational_calibration.txt <<'PY'
def _rational_fit(x, target):
    """Profile linear coefficients for a quadratic coupling with saturating gain."""
    affine = np.column_stack((np.ones(len(x)), x))
    powers = _powers(2)[5:]
    quadratic = _features(x, powers)

    def design(q):
        return np.column_stack((affine, quadratic / (1. + (x @ q)**2)[:, None]))

    def profile(q):
        a = design(q)
        c = np.linalg.lstsq(a, target, rcond=1e-10)[0]
        return a @ c - target, c

    def cost(q):
        r, _ = profile(q)
        return r @ r

    eye4 = np.eye(4)
    starts = list(eye4 * .75)
    starts += [.55 * (eye4[i] + sign * eye4[j]) for i in range(4)
               for j in range(i + 1, 4) for sign in (-1, 1)]
    starts.sort(key=cost)
    best = min((least_squares(lambda q: profile(q)[0], start, bounds=(-6., 6.),
                              max_nfev=45, gtol=1e-12, ftol=1e-9, xtol=1e-9)
                for start in starts[:3]), key=lambda opt: opt.cost)
    return best.x, profile(best.x)[1]


def _rational_predict(x, q, c):
    return c[0] + x @ c[1:5] + (_features(x, _powers(2)[5:]) @ c[5:]) / (1. + (x @ q)**2)


PY
python3 - <<'PY'
from pathlib import Path
p=Path('batch_solver.py')
s=p.read_text().replace('def _calibration(sweeps):',Path('/tmp/rational_calibration.txt').read_text()+'def _calibration(sweeps):')
a=s.index('    p = models[next(')
s=s[:a]+'''    selected = next(i for i, e in enumerate(errors) if e <= 1.12 * min(errors) + 1e-12)
    p = models[selected]
    # Near the sweep noise floor, a larger nonlinear family cannot improve inversion.
    rational = [None] * 4
    if errors[selected] > max(5e-5, 3 * errors[0] * 1e-3):
        reference = np.asarray(sweeps.get('single_bank_sweeps', {}).get('ref_signal', []))
        raw_bank = np.asarray(sweeps.get('single_bank_sweeps', {}).get('raw_readout', []))
        noise = np.median(np.abs(raw_bank - reference)) if reference.size else 0.
        if errors[selected] > max(5e-5, min(1e-4, 4 * noise)):
            err = np.empty_like(y)
            poly_err = np.empty_like(y)
            for k in range(4):
                train = folds != k
                if p is None:
                    poly_err[~train] = x[~train] - y[~train]
                else:
                    a = _features(x, p)
                    poly_err[~train] = a[~train] @ np.linalg.lstsq(a[train], (y-x)[train], rcond=None)[0] + x[~train] - y[~train]
                for j in range(4):
                    q, c = _rational_fit(x[train], (y-x)[train, j])
                    err[~train, j] = _rational_predict(x[~train], q, c) + x[~train, j] - y[~train, j]
            for j in range(4):
                score = lambda e: max(np.sqrt(np.mean(e[folds == k, j]**2)) for k in range(4))
                if score(err) < 0.7 * score(poly_err):
                    rational[j] = _rational_fit(x, (y-x)[:, j])
    if any(item is not None for item in rational):
        powers = _powers(1) if p is None else p
        coeff = np.linalg.lstsq(_features(x, powers), y - x, rcond=None)[0]
        affine = coeff[:5].copy()
        for j, item in enumerate(rational):
            if item is not None:
                affine[:, j] = item[1][:5]

        def unmix(raw):
            values = np.asarray(raw, dtype=float).reshape(-1, 4)
            initial = np.linalg.solve((np.eye(4) + affine[1:]).T, (values-affine[0]).T).T

            def predict(z):
                z = z[None, :]
                delta = (_features(z, powers) @ coeff)[0]
                for j, item in enumerate(rational):
                    if item is not None:
                        delta[j] = _rational_predict(z, *item)[0]
                return z[0] + delta

            estimate = [least_squares(lambda z: predict(z) - goal, start, gtol=1e-12).x
                        for goal, start in zip(values, initial)]
            return np.asarray(estimate).reshape(np.shape(raw))

        return unmix
'''+s[s.index('    if p is None:\n        return lambda raw:',a):]
s=s.replace('    if errors[selected] > max(5e-5, 3 * errors[0] * 1e-3):','    if errors[selected] > 5e-5:')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_batch.py
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`3083 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
identity pulses False cal max 0.0 wall max 0.0 weight max 1.809283259601724e-07 field max 2.2442054603555306e-09 oracle KKT residual 2.7755575615628914e-17 internal gap 6.765421556309548e-17
affine pulses True cal max 1.263022702200408e-06 wall max 0.0 weight max 3.329827048759171e-06 field max 3.6951835638565633e-07 oracle KKT residual 1.0647382332109462e-06 internal gap 1.1102230246251565e-16
cubic pulses True cal max 5.055790570203389e-06 wall max 5.009214418105756e-05 weight max 1.2739851583953854e-05 field max 1.6097105826340385e-06 oracle KKT residual 3.0505639466715806e-06 internal gap 6.765421556309548e-17
affine pulses True cal max 2.0004123991307354e-06 wall max 0.0005268904823558795 weight max 5.601016491885158e-06 field max 7.759237088070942e-07 oracle KKT residual 6.725372002289176e-07 internal gap 1.1535911115245767e-16
rational pulses True cal max 4.90902178464786e-06 wall max 0.0 weight max 1.0581951340155049e-05 field max 1.0122132573231468e-06 oracle KKT residual 4.0140605814080466e-06 internal gap 6.505213034913027e-17
max internal gap 1.1535911115245767e-16
active-face total 0.0 gap 0.0 budget error 0.0
active-face total 0.3 gap 3.0553337637684335e-16 budget error 8.881784197001252e-16
active-face total 1.7 gap 0.0 budget error 2.220446049250313e-16
active-face total 2.4 gap 0.0 budget error 0.0
apparatus log present: False
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9858c9cbdcc0379c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9858c9cb (directory).
```

### Turn 7 — `execute_in_remote_docker` (`21278 ms` LLM)
- **Rationale**: The verified candidate resolves the saturation failure. Check whether the mounted context provides a documented wall-state input for pulse-free campaigns; if it does not, keep the nominal fallback rather than inventing an archive format or telemetry mapping.
- **Alternatives Weighed**: `Infer a wall state from unlabeled telemetry` (rejected: No telemetry-to-wall mapping has been established; check the supplied metadata before adding one.)

```bash
printf '%s\n' '--- mounted public context ---'; for d in /pretrain_task /tmp/hegel_continual_archive; do if [ -d "$d" ]; then ls -lah "$d" | head -25; fi; done; printf '%s\n' '--- public study fields ---'; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
with open('/holdout_task/public_batch.pkl','rb') as f:
    for item in pickle.load(f):
        s=item['study']
        for k,v in s.items():
            print(k, getattr(v,'shape',type(v).__name__))
        if s.get('reference_tracer_pulses'):
            print('pulse keys:', list(s['reference_tracer_pulses'][0]))
PY
```
- **Container Exit Code**: `0` (`384 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
--- mounted public context ---
total 8.0K
drwxr-xr-x 2 root root 4.0K Nov 23 09:31 .
drwxr-xr-x 1 root root 4.0K Nov 27 12:46 ..
--- public study fields ---
Lx float
Ly float
nx int
ny int
kappa_x (14,)
kappa_y (12,)
vel_x (14,)
vel_y (12,)
robin_x float
robin_y float
V_reaction_2d (14, 12)
mu_shifts (3,)
source_modes (8, 14, 12)
sensor_ij (32, 2)
row_bank_id (96,)
sensor_calibration_sweeps dict
Z_nuisance (96, 3)
y_obs_raw (96,)
reference_tracer_pulses list
aux_manifold_telemetry (24, 4)
nuisance_ridge float
source_ridge float
u_box float
total_budget float
pulse keys: ['w_pulse', 'y_pulse_raw']

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9858c9cbdcc0379c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9858c9cb (directory).
```

### Turn 8 — `submit_completed_task` (`13923 ms` LLM)
- **Rationale**: The verified artifact passed the public schema and isolated stress tests. Rational calibration reduced the saturation-slice weight error from 0.06479 to 1.06e-5 without degrading the other tested slices.
- **Alternatives Weighed**: `Infer wall state from unlabeled telemetry on pulse-free campaigns` (rejected: The available data establish no telemetry-to-wall mapping, and no apparatus log was present.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:12622'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed, and isolated stress tests showed a maximum weight error of 1.27e-5 against an independent optimizer. No apparatus archive was available; campaigns without tracer pulses use the nominal wall model.`

### Turn 9 — `execute_in_remote_docker` (`39176 ms` LLM)
- **Rationale**: Use real sweeps to test whether rational coupling improves held-out calibration, and real pulses to detect wall mismatch by shift and channel. An independent constrained solve checks whether the public weights, terminal field, and certificate agree with the specified nuisance-projected objective. A mismatch would identify which stage needs revision.
- **Alternatives Weighed**: `Run batch-classification, relay-mode, or HJB checks` (rejected: The public study is a continuous steady-state inverse problem with one campaign, no class labels or relay modes, and a specified discrete transport operator.)

```bash
cat > /tmp/audit_public_batch.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('batch_solver', root / 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_batch.pkl').open('rb') as f:
    batch = pickle.load(f)
for index, item in enumerate(batch):
    s = item['study']
    c = s['sensor_calibration_sweeps']
    x, raw = map(np.asarray, (c['ref_rosette'], c['raw_rosette']))
    folds = np.arange(len(x)) % 4
    corrected = np.empty_like(x)
    rational = np.empty_like(x)
    for fold in range(4):
        t = folds != fold
        fit = m._calibration({'ref_rosette': x[t], 'raw_rosette': raw[t], 'single_bank_sweeps': c.get('single_bank_sweeps', {})})
        corrected[~t] = fit(raw[~t]) - x[~t]
        for j in range(4):
            q, a = m._rational_fit(x[t], (raw-x)[t, j])
            rational[~t, j] = m._rational_predict(x[~t], q, a) + x[~t, j] - raw[~t, j]
    score = lambda e: float(np.max([np.sqrt(np.mean(e[folds == k]**2, axis=0)) for k in range(4)]))
    print('study', index, 'worst fold/channel sweep RMSE, identity / nested selected inverse / forced rational forward:', score(raw-x), score(corrected), score(rational))
    source = np.asarray(s['source_modes'])
    unmix = m._calibration(c)
    response, fields = m._response(s, np.ones(8), source)
    for j, p in enumerate(s.get('reference_tracer_pulses') or []):
        e = (response @ p['w_pulse'] - unmix(p['y_pulse_raw'])).reshape(len(s['mu_shifts']), -1, 4)
        print('pulse', j, 'worst shift/channel RMSE', np.sqrt(np.mean(e**2, axis=1)).max(), 'worst row', np.max(abs(e)))
    Z = s['Z_nuisance']; y = unmix(s['y_obs_raw']); nr = s['nuisance_ridge']
    rr = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']
    P = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + nr * np.eye(Z.shape[1]), Z.T)
    X, Y = P @ response, P @ y
    def objective(w):
        r = X @ w - Y
        return float((r @ r + rr * (w @ w)) / 2)
    def grad(w):
        return X.T @ (X @ w - Y) + rr*w
    out = m.solve_batch(s); w = out['weights']; cert = out['kkt_certificate']
    oracle = minimize(objective, np.full(len(w), budget / len(w)), jac=grad, method='SLSQP', bounds=[(0, cap)] * len(w), constraints=[{'type': 'eq', 'fun': lambda p: p.sum()-budget, 'jac': lambda p: np.ones_like(p)}], options={'ftol': 1e-14, 'maxiter': 300})
    lo, hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
    stationarity = np.max(abs(grad(w) - cert['nu_sum'] - lo + hi))
    gap = cert['primal_objective'] - cert['dual_lower_bound']
    field_error = np.max(abs(out['terminal_field'] - np.tensordot(oracle.x, fields, axes=1)))
    print('public oracle:', {'success': oracle.success, 'budget_error': abs(w.sum()-budget), 'weight_max_error': np.max(abs(w-oracle.x)), 'terminal_field_max_error': field_error, 'certificate_stationarity': stationarity, 'certificate_gap': gap})
    assert oracle.success and abs(w.sum()-budget) < 1e-9 and np.min(w) >= -1e-10 and np.max(w) <= cap + 1e-10
    assert abs(objective(w)-cert['primal_objective']) < 1e-9 and abs(gap) < 1e-9 and stationarity < 1e-9
    assert np.max(abs(w-oracle.x)) < 1e-5 and field_error < 1e-5
    print('nuisance rank:', np.linalg.matrix_rank(Z), 'ridge projection algebra residual:', np.max(abs(P @ Z - nr * Z @ np.linalg.inv(Z.T @ Z + nr*np.eye(Z.shape[1])))))
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_public_batch.py
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1267 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study 0 worst fold/channel sweep RMSE, identity / nested selected inverse / forced rational forward: 1.3790757852587899e-05 1.3790757852587899e-05 5.505757129055499e-05
pulse 0 worst shift/channel RMSE 1.9101266978071048e-16 worst row 3.3306690738754696e-16
pulse 1 worst shift/channel RMSE 2.6495310240288706e-16 worst row 5.551115123125783e-16
public oracle: {'success': True, 'budget_error': 0.0, 'weight_max_error': 3.14553567914988e-07, 'terminal_field_max_error': 1.3318050801069958e-08, 'certificate_stationarity': 5.117434254131581e-17, 'certificate_gap': 9.367506770274758e-17}
nuisance rank: 3 ridge projection algebra residual: 1.4777675610977425e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9858c9cbdcc0379c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9858c9cb (directory).
```

### Turn 10 — `submit_completed_task` (`14662 ms` LLM)
- **Rationale**: The nominal operator reproduced both measured tracer pulses to floating-point precision. The isolated audit verified the public weights, terminal field, and KKT certificate; synthetic stress tests separately verified the rational calibration and altered-wall paths.
- **Alternatives Weighed**: `Force rational calibration on the public campaign` (rejected: It increased the worst held-out sweep RMSE from 1.38e-5 to 5.51e-5.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:12622'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Against an independent optimizer on the public data, the maximum weight error was 3.15e-7 and the maximum terminal-field error was 1.33e-8. No apparatus archive was available; without tracer pulses, the solver uses the nominal wall model.`

### Turn 11 — `execute_in_remote_docker` (`33255 ms` LLM)
- **Rationale**: Fit wall corrections with tracer data withheld and keep each sensor location together across shifts. A correction that improves the source fit but worsens independent tracer predictions would show why source-only wall fitting should not replace the nominal fallback.
- **Alternatives Weighed**: `Repeat the rational calibration audit` (rejected: The prior nested sweep audit settled that question for the public campaign; it did not test whether routine observations could identify the wall state without pulses.)

```bash
cat > /tmp/audit_blind_wall.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    studies = pickle.load(f)
for item in studies:
    s = item['study']
    source = s['source_modes']
    cal = m._calibration(s['sensor_calibration_sweeps'])
    y = cal(s['y_obs_raw'])
    z = s['Z_nuisance']
    nr, rr, cap, budget = (s[k] for k in ('nuisance_ridge','source_ridge','u_box','total_budget'))
    pts = np.asarray(s['sensor_ij'])
    if len(pts) != len(y):
        pts = np.tile(pts, (len(s['mu_shifts']), 1))
    _, loc = np.unique(pts, axis=0, return_inverse=True)
    folds = loc % 4
    pulse = s.get('reference_tracer_pulses') or []
    nominal = m._response(s, np.ones(8), source)[0]
    y_error = {tag: [] for tag in ('nominal','blind')}
    pulse_error = {tag: [] for tag in y_error}
    for k in np.unique(folds):
        train = folds != k
        zt = z[train]
        p = np.linalg.solve(zt.T @ zt + nr*np.eye(z.shape[1]), zt.T)
        yt = y[train]
        yp = yt - zt @ (p @ yt)
        def fit(S):
            St = S[train]
            X = St - zt @ (p @ St)
            w = m._weights(X, yp, rr, cap, budget)
            gamma = p @ (yt - St @ w)
            return w, gamma, X @ w - yp
        def response(t):
            wall = np.ones(8)
            wall[[0,1]], wall[[4,5]] = t
            return m._response(s, wall, source)[0]
        def residual(t):
            w, _, r = fit(response(t))
            return np.r_[r, np.sqrt(rr)*w]
        opt = least_squares(residual, [1.,1.], bounds=(0.25, 4.), max_nfev=30)
        for tag, S in [('nominal',nominal),('blind',response(opt.x))]:
            w, gamma, _ = fit(S)
            e = (S @ w + z @ gamma - y)[~train]
            y_error[tag].append(np.sqrt(np.mean(e**2)))
            if pulse:
                pulse_error[tag].append(max(np.sqrt(np.mean(((S @ q['w_pulse'] - cal(q['y_pulse_raw']))[~train])**2)) for q in pulse))
        print('fold', k, 'blind Robin scale factors', np.round(opt.x, 5), 'held-out measurement RMSE nominal/blind', *(y_error[tag][-1] for tag in y_error))
    print('Worst spatial fold RMSE, measurements:', {tag: max(v) for tag,v in y_error.items()})
    print('Worst held-out spatial fold and independent pulse RMSE:', {tag: max(v) for tag,v in pulse_error.items() if v})
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_blind_wall.py
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1012 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fold 0 blind Robin scale factors [0.93752 1.06744] held-out measurement RMSE nominal/blind 0.009586228000859653 0.009329583769248553
fold 1 blind Robin scale factors [0.83967 1.12655] held-out measurement RMSE nominal/blind 0.009698658668992513 0.009346405158888805
fold 2 blind Robin scale factors [0.74004 1.19432] held-out measurement RMSE nominal/blind 0.009007477632334454 0.009237422469568112
fold 3 blind Robin scale factors [0.82307 1.13206] held-out measurement RMSE nominal/blind 0.0058498105372649795 0.005501343444420692
Worst spatial fold RMSE, measurements: {'nominal': 0.009698658668992513, 'blind': 0.009346405158888805}
Worst held-out spatial fold and independent pulse RMSE: {'nominal': 1.7431345636114038e-16, 'blind': 0.005400642216999298}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9858c9cbdcc0379c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9858c9cb (directory).
```

### Turn 12 — `submit_completed_task` (`19690 ms` LLM)
- **Rationale**: The tracer audit directly tested physical wall recovery and supports the existing fallback. The unchanged artifact passed isolated checks of the weights, terminal field, and KKT certificate, as well as the public schema.
- **Alternatives Weighed**: `Promote blind Robin fitting on the strength of routine cross-validation` (rejected: The small improvement on routine observations came at the cost of a much larger error on independent tracer measurements.)
- **ATMS NOGOOD DECLARED**: Refuted `A wall correction that improves held-out routine observations necessarily recovers the physical wall state.` | Witness: `Fitting Robin factors on routine measurements reduced worst spatial-fold RMSE from 9.70e-3 to 9.35e-3 but increased worst independent tracer RMSE from 1.74e-16 to 5.40e-3.` | Blanket Diagnosis: `Routine measurements jointly depend on source weights, nuisance effects, and wall coefficients. Their predictive error alone did not distinguish a wall change from compensation among these effects.` | Sublated Pivot: `Retain the verified solver's nominal fallback and use independent tracer pulses to support non-nominal wall fits.`
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Against an independent optimizer, the maximum public weight error was 3.15e-7 and the maximum terminal-field error was 1.33e-8. No apparatus log was available, so campaigns without tracer pulses use the nominal wall model.`

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
