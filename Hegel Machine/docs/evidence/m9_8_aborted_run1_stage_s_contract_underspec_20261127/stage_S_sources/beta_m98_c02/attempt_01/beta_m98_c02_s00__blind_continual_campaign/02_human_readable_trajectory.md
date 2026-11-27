# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_c02_s00__blind_continual_campaign`

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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `341,698` (`326,612` / `56,064` / `15,086`) |
| **Total Episode Wall Time (s)** | `439.28s` |

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

### Turn 1 — `execute_in_remote_docker` (`19788 ms` LLM)
- **Rationale**: Read the local schema and summarize the supplied data without copying arrays or archive payloads. This identifies which evidence is available for fitting and for separate checks of calibration, transport, and the QP certificate.
- **Alternatives Weighed**: `Implement directly from the task description` (rejected: The study and archive schemas determine which calibration and boundary parameters can be inferred, especially when tracer pulses are absent.)

```bash
set -e
cd /holdout_task
cat public_asset_index.json validate_public_schema.py
printf '\n--- archive inventory ---\n'
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s bytes\n' | head -60; fi
printf '\n--- study schema ---\n'
python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))
def describe(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d+str(k)+':',end=' '); describe(v,d+2)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,x.ravel()[:8])
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x))
  if x: describe(x[0],d+2)
 else: print(repr(x)[:700])
describe(s)
PY
printf '\n--- skill availability ---\n'
if [ -d /tmp/hegel_wormhole_skills ]; then PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic program'; fi
```
- **Container Exit Code**: `0` (`411 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18994 ms` LLM)
- **Rationale**: The affine residual determines whether nonlinear calibration is needed. Sensor ordering and archive metadata determine how to assemble the forward response and what evidence is available for campaigns without pulses.

```bash
cd /holdout_task; cat validate_public_schema.py; ls -la /pretrain_task /tmp/hegel_continual_archive 2>/dev/null; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 2400 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -35; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'));print(type(p),list(p) if isinstance(p,dict) else len(p)); s=(p[0] if isinstance(p,list) else p.get('cases',[p])[0]);s=s.get('study',s);print(list(s));
for k in ['sensor_ij','row_bank_id','mu_shifts','kappa_x','vel_x','vel_y','source_modes']:
 a=np.asarray(s[k]);print(k,a.shape,a.min(),a.max(),a.ravel()[:26]);
c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];print('cal ranges',x.min(0),x.max(0),y.min(0),y.max(0)); a=np.c_[np.ones(len(x)),x];b=np.linalg.lstsq(a,y,rcond=None)[0];print('affine',b,'rmse',np.sqrt(np.mean((a@b-y)**2,0)));print('bank');
for k,v in c['single_bank_sweeps'].items():print(k,np.shape(v),np.ravel(v)[:28]);
PY
```
- **Container Exit Code**: `0` (`410 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
sensor_ij (32, 2) 0 12 [ 0  2  8  6  4  6 12 10  4  4  2  4  6  8  6 10  8 10  0  8 12  6  8  0
 10  6]
row_bank_id (96,) 0 3 [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1]
mu_shifts (3,) 0.15 1.1 [0.15 0.55 1.1 ]
kappa_x (14,) 0.43316833766040264 0.5721475800341301 [0.43316834 0.46667253 0.49822958 0.52600551 0.54838608 0.56407061
 0.57214758 0.57214758 0.56407061 0.54838608 0.52600551 0.49822958
 0.46667253 0.43316834]
vel_x (14,) -0.20572483830236554 0.35 [ 0.35        0.34500413  0.33015913  0.30588881  0.27288602  0.23209293
  0.18467409  0.13198319  0.07552447  0.01690968 -0.04218784 -0.10008099
 -0.15511704 -0.20572484]
vel_y (12,) -0.279885813991893 -0.0 [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907 -0.27988581 -0.27272334 -0.25469696 -0.22652476]
source_modes (8, 14, 12) 4.1508628561618226e-15 0.9940952851255738 [3.12972983e-10 1.41286748e-08 4.03503938e-07 7.29030828e-06
 8.33289245e-05 6.02556017e-04 2.75645461e-03 7.97730198e-03
 1.46053842e-02 1.69169360e-02 1.23960180e-02 5.74638147e-03
 1.52069043e-09 6.86491863e-08 1.96056724e-06 3.54225529e-05
 4.04883185e-04 2.92773248e-03 1.33932141e-02 3.87605558e-02
 7.09654482e-02 8.21969440e-02 6.02304573e-02 2.79208360e-02
 5.02055545e-09 2.26645107e-07]
cal ranges [-1.34414968 -1.2        -1.22364327 -1.36051815] [1.76780212 1.78199005 1.79914087 1.74606382] [-1.34414227 -1.19999166 -1.22365004 -1.3605231 ] [1.76780873 1.78199838 1.79914535 1.74605545]
affine [[-3.41408442e-07  2.62617178e-06 -9.57448085e-08 -6.30500816e-07]
 [ 1.00000069e+00 -1.38022127e-06 -4.13553364e-07  4.46995532e-07]
 [-9.34872118e-07  9.99997877e-01 -8.78127119e-07 -4.19473113e-07]
 [-1.16279750e-06  7.95473752e-07  1.00000076e+00  1.44827000e-06]
 [ 1.40535177e-06 -1.66884989e-06  1.92411692e-06  1.00000123e+00]] rmse [9.27298292e-06 9.06194575e-06 9.16174376e-06 9.49710953e-06]
bank
bank_id (24,) [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3]
ref_signal (24,) [-1.1 -0.6 -0.1  0.4  0.9  1.4 -1.1 -0.6 -0.1  0.4  0.9  1.4 -1.1 -0.6
 -0.1  0.4  0.9  1.4 -1.1 -0.6 -0.1  0.4  0.9  1.4]
raw_readout (24,) [-1.09999877 -0.59999545 -0.09999465  0.39999793  0.89998528  1.39998488
 -1.09997465 -0.59999628 -0.0999982   0.39999324  0.90001346  1.40000054
 -1.10000136 -0.60000326 -0.09999965  0.39998239  0.89999321  1.3999882
 -1.09998566 -0.59999937 -0.10000915  0.40000697  0.8999938   1.39998853]
```

### Turn 3 — `execute_in_remote_docker` (`47970 ms` LLM)
- **Rationale**: The public calibration sweep supports an affine baseline. Compare explicit grid, face-flux, and Robin conventions using the known pulses, both with and without nuisance removal, to distinguish an operator mismatch from a nuisance offset.
- **Alternatives Weighed**: `Fit unrestricted wall diagonals without comparing finite-volume conventions` (rejected: A free wall fit could hide a stencil error and would not transfer to campaigns without tracers.); `Use the ridge-profiling skill in the solver` (rejected: The specified residualization requires just one small Gram solve with multiple right-hand sides; the retrieved matching primitive is unnecessary.)

```bash
cat > /tmp/transport_probe.py <<'PY'
import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def axis(k, v, h, adv, face):
    f = ((k[:-1] + k[1:])*.5 if face=='arith' else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]))/h**2
    if adv in ('center','negative'):
        a = ((v[:-1]+v[1:])*.5)/h * (1 if adv=='center' else -1)
        p = np.maximum(a,0); m = np.minimum(a,0)
        lower, upper = -f-p, -f+m
        d = np.r_[f+p,0] + np.r_[0,f-m]
    elif adv=='row':
        a=v/h; lower=-f-np.maximum(a[1:],0); upper=-f+np.minimum(a[:-1],0)
        d=np.r_[-upper,0]+np.r_[0,-lower]
    elif adv=='rowface':
        a=(v[:-1]+v[1:])*.5/h;lower=-f-np.maximum(a,0);upper=-f+np.minimum(a,0)
        d=np.r_[-upper,0]+np.r_[0,-lower]
    else:
        lower=upper=-f;d=np.r_[f,0]+np.r_[0,f]
    return diags([lower,d,upper],[-1,0,1],format='csc')


def prepare(s, adv='center', face='arith', grid=0):
    nx,ny=s['nx'],s['ny']; h=np.array([s['Lx']/(nx+grid),s['Ly']/(ny+grid)])
    ax=axis(s['kappa_x'],s['vel_x'],h[0],adv,face);ay=axis(s['kappa_y'],s['vel_y'],h[1],adv,face)
    ident=eye(nx*ny,format='csc')
    base=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    B=np.array([s['source_modes'][i].ravel(order='F') for i in range(len(s['source_modes']))]).T
    i,j=np.asarray(s['sensor_ij']).T; idx=i+nx*j
    wall=np.zeros((nx,ny,4));wall[0,:,0]=1;wall[-1,:,1]=1;wall[:,0,2]=1;wall[:,-1,3]=1
    wall=wall.reshape(-1,4,order='F')
    def predict(boundary,rhs=B):
        A=base+diags(wall@boundary)
        sol=[splu(A+m*ident).solve(rhs) for m in s['mu_shifts']]
        return np.concatenate([u[idx] for u in sol]),sol[-1]
    return predict, B, h


def calibrate(s):
    c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette']; off,*lin=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]; lin=np.array(lin)
    return lambda yy: np.linalg.solve(lin.T,(yy.reshape(-1,4)-off).T).T.ravel()

if __name__=='__main__':
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];cal=calibrate(s)
    ps=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in ps],axis=1); Y=np.stack([cal(p['y_pulse_raw']) for p in ps],axis=1)
    Z=s['Z_nuisance']; q=np.linalg.qr(Z)[0]; nom=np.repeat([s['robin_x'],s['robin_y']],2)
    print('pulse calibrated range',Y.min(0),Y.max(0),'noise behavior',Y[:6].T,'Z first rows',Z[:4]);print('scipy sparse okay')
    results=[]
    for ad in ['center','row','rowface','negative','none']:
     for face in ['arith','harm']:
      for grid in [-1,0,1]:
       f,B,h=prepare(s,ad,face,grid); rhs=B@W; hh=np.repeat(h,2); kap=np.array([s['kappa_x'][0],s['kappa_x'][-1],s['kappa_y'][0],s['kappa_y'][-1]])
       fits=[]
       for mode in range(2):
        def residual(b):
         e=f(b,rhs)[0]-Y
         return ((e-q@(q.T@e)) if mode else e).ravel()
        candidates=[nom,nom/hh,nom*kap/hh,nom/hh/(1+nom*hh/(2*kap)),nom*kap/hh/(1+nom*hh/2),np.zeros(4)]
        norms=[np.linalg.norm(residual(a))/np.sqrt(Y.size) for a in candidates]
        for n in (2,4):
         exp=lambda b: np.repeat(b,2) if n==2 else b
         r=least_squares(lambda b:residual(exp(b)), (nom/hh)[::2] if n==2 else nom/hh,bounds=(0,np.inf),gtol=1e-12)
         fits.append((np.sqrt(np.mean(r.fun**2)),np.round(exp(r.x),5)))
        if mode==0: raw=norms
       results.append((fits[2][0],ad,face,grid,np.round(raw,6),fits))
    for r in sorted(results,key=lambda t:t[0])[:10]:print(r)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_probe.py
ls -ld /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`3143 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
261, 6.6936 ])), (0.0003142596893037657, array([5.89875, 5.89875, 6.57334, 6.57334])), (0.00025374779792324974, array([5.90899, 5.74828, 6.49976, 6.68872]))])
(0.0038699943804808763, 'none', 'harm', 0, array([0.061151, 0.004689, 0.029784, 0.008714, 0.032818, 0.128179]), [(0.003939182791061159, array([7.5023, 7.5023, 6.7092, 6.7092])), (0.0037528035294974935, array([7.45779, 7.7331 , 6.00521, 7.52478])), (0.0038699943804808763, array([7.52012, 7.52012, 6.77747, 6.77747])), (0.0036546864686244794, array([7.47145, 7.69876, 6.02968, 7.69   ]))])
(0.0038797435779059928, 'none', 'arith', 0, array([0.061146, 0.004694, 0.029773, 0.008707, 0.032807, 0.12818 ]), [(0.003949533441356258, array([7.50226, 7.50226, 6.7004 , 6.7004 ])), (0.003766663121264785, array([7.45849, 7.72789, 6.00276, 7.50873])), (0.0038797435779059928, array([7.52003, 7.52003, 6.76891, 6.76891])), (0.003668110237533736, array([7.47221, 7.69312, 6.02735, 7.67429]))])
(0.004580029225365145, 'rowface', 'harm', 0, array([0.054416, 0.00651 , 0.023276, 0.004758, 0.026332, 0.132319]), [(0.004741671039818171, array([5.74218, 5.74218, 5.19499, 5.19499])), (0.00439095630553421, array([5.68939, 6.21688, 6.12371, 4.47582])), (0.004580029225365145, array([5.7444 , 5.7444 , 5.27071, 5.27071])), (0.004263810127428755, array([5.70579, 6.15465, 6.15984, 4.57146]))])
(0.004600728235652958, 'rowface', 'arith', 0, array([0.054419, 0.006533, 0.023273, 0.004778, 0.026329, 0.13232 ]), [(0.004762844740461409, array([5.74392, 5.74392, 5.18935, 5.18935])), (0.004411298859009572, array([5.69112, 6.21712, 6.12172, 4.46861])), (0.004600728235652958, array([5.74613, 5.74613, 5.26522, 5.26522])), (0.004283788761832585, array([5.70757, 6.15453, 6.15789, 4.56446]))])
(0.004668834800289461, 'row', 'harm', 0, array([0.053735, 0.006814, 0.022819, 0.004813, 0.025851, 0.131048]), [(0.00484018563815733, array([5.72814, 5.72814, 5.08035, 5.08035])), (0.004467262518510926, array([5.66111, 6.37627, 5.97756, 4.34284])), (0.004668834800289461, array([5.73113, 5.73113, 5.1564 , 5.1564 ])), (0.004331912272569095, array([5.67876, 6.31037, 6.01592, 4.43776]))])
(0.004689272858528126, 'row', 'arith', 0, array([0.053737, 0.006837, 0.022816, 0.004833, 0.025848, 0.131049]), [(0.004861089494515824, array([5.72995, 5.72995, 5.07482, 5.07482])), (0.004487423518911062, array([5.66292, 6.37629, 5.97558, 4.33585])), (0.004689272858528126, array([5.73292, 5.73292, 5.15102, 5.15102])), (0.004351719877059705, array([5.68062, 6.31003, 6.01398, 4.43098]))])
```

### Turn 4 — `execute_in_remote_docker` (`28661 ms` LLM)
- **Rationale**: Holding out tracer pulses, sensor groups, and shifts tests whether four wall coefficients generalize or merely absorb errors in the stencil. Use shared coefficients if the additional freedom fails these tests.
- **ATMS NOGOOD DECLARED**: Refuted `Using L/n spacing and conservative face upwinding is an adequate discrete reconstruction of the public response.` | Witness: `The best L/n stencil fit retained milliscale tracer residuals, whereas L/(n−1) with arithmetic diffusion and row upwinding reached 3.27e-5 with four wall terms.` | Blanket Diagnosis: `The residual persists after nuisance projection; grid spacing and the placement of the upwind velocity affect the interior response in ways that wall adjustment alone cannot absorb.` | Sublated Pivot: `Use the endpoint-spaced stencil and compare nominal, axis-shared, and side-specific Robin terms on held-out tracer slices.`

```bash
cd /tmp; cat > /tmp/pulse_cv.py <<'PY'
from transport_probe import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];print('public item keys',list(s));s=s['study'];cal=calibrate(s)
ps=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in ps],axis=1);Y=np.stack([cal(p['y_pulse_raw']) for p in ps],axis=1)
for adv in ['row','rowface']:
 for face in ['arith','harm']:
  predict,B,h=prepare(s,adv,face,-1); nom=np.repeat([s['robin_x'],s['robin_y']],2)/np.repeat(h,2)
  n=len(s['sensor_ij']); group=np.tile(np.arange(n)//4,len(s['mu_shifts']));shift=np.repeat(np.arange(len(s['mu_shifts'])),n)
  for mode in ['nominal','axis','wall']:
   p0={'nominal':np.array([]),'axis':nom[::2],'wall':nom}[mode]
   expand=lambda p:nom if mode=='nominal' else np.repeat(p,2) if mode=='axis' else p
   def score(tr,te):
    def resid(p,m):return (predict(expand(p),B@W)[0]-Y)[m].ravel()
    p=least_squares(resid,p0,args=(tr,),bounds=(0,np.inf),gtol=1e-12).x if len(p0) else p0
    return np.sqrt(np.mean(resid(p,te)**2)),expand(p)
   allm=np.ones_like(Y,dtype=bool); v=[score(np.broadcast_to(group[:,None]!=i,Y.shape),np.broadcast_to(group[:,None]==i,Y.shape))[0] for i in np.unique(group)];u=[score(np.broadcast_to(shift[:,None]!=i,Y.shape),np.broadcast_to(shift[:,None]==i,Y.shape))[0] for i in np.unique(shift)];t=[score(allm & (np.arange(Y.shape[1])!=i),allm & (np.arange(Y.shape[1])==i))[0] for i in range(Y.shape[1])];print(adv,face,mode,'all',score(allm,allm),'worst group/shift/pulse',np.round([max(v),max(u),max(t)],8),'individual pulse',np.round(t,8))
print('archives',__import__('os').path.isdir('/tmp/hegel_continual_archive'))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/pulse_cv.py
```
- **Container Exit Code**: `0` (`3082 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public item keys ['study']
row arith nominal all (0.0005815665001985202, array([5.6217187 , 5.6217187 , 6.39461455, 6.39461455])) worst group/shift/pulse [0.00101638 0.00077431 0.00062206] individual pulse [0.00053804 0.00062206]
row arith axis all (8.255437452119171e-05, array([5.87307048, 5.87307048, 6.40930222, 6.40930222])) worst group/shift/pulse [0.0001441  0.00011408 0.00010867] individual pulse [0.00010867 0.00010367]
row arith wall all (3.295508609634023e-05, array([5.87023403, 5.88615271, 6.35971086, 6.44967804])) worst group/shift/pulse [6.840e-05 5.955e-05 4.355e-05] individual pulse [3.609e-05 4.355e-05]
row harm nominal all (0.0005894885073548605, array([5.6217187 , 5.6217187 , 6.39461455, 6.39461455])) worst group/shift/pulse [0.00102507 0.00078692 0.00062888] individual pulse [0.00054727 0.00062888]
row harm axis all (9.277410248120487e-05, array([5.87219131, 5.87219131, 6.41632134, 6.41632134])) worst group/shift/pulse [0.00014652 0.00012514 0.00011913] individual pulse [0.00011913 0.0001136 ]
row harm wall all (4.233121555780027e-05, array([5.86893391, 5.88822055, 6.36197467, 6.4600986 ])) worst group/shift/pulse [6.783e-05 6.659e-05 5.272e-05] individual pulse [4.472e-05 5.272e-05]
rowface arith nominal all (0.0009289894414586963, array([5.6217187 , 5.6217187 , 6.39461455, 6.39461455])) worst group/shift/pulse [0.00129168 0.00123094 0.00101021] individual pulse [0.00083995 0.00101021]
rowface arith axis all (0.00031480200741203993, array([5.9016149 , 5.9016149 , 6.57049983, 6.57049983])) worst group/shift/pulse [0.00043426 0.00038348 0.00042357] individual pulse [0.00024623 0.00042357]
rowface arith wall all (0.000254413635582167, array([5.91297718, 5.74674749, 6.50033627, 6.68269368])) worst group/shift/pulse [0.00036469 0.00031152 0.00046465] individual pulse [0.00046465 0.00038051]
rowface harm nominal all (0.000940979175447749, array([5.6217187 , 5.6217187 , 6.39461455, 6.39461455])) worst group/shift/pulse [0.00130426 0.00124805 0.00102181] individual pulse [0.00085251 0.00102181]
rowface harm axis all (0.0003237711605420738, array([5.90081864, 5.90081864, 6.57770328, 6.57770328])) worst group/shift/pulse [0.0004383  0.00039353 0.00043694] individual pulse [0.00025243 0.00043694]
rowface harm wall all (0.00026211892542008967, array([5.91176935, 5.74855226, 6.50260516, 6.6936036 ])) worst group/shift/pulse [0.00038059 0.00032053 0.00046936] individual pulse [0.00046936 0.00038829]
archives False
```

### Turn 5 — `execute_in_remote_docker` (`35606 ms` LLM)
- **Rationale**: Withhold the public pulses from boundary fitting, then use them as independent validation. Sensor-group holdouts also test whether blind fitting improves prediction rather than just trading wall terms against source weights.
- **Alternatives Weighed**: `Always use nominal Robin terms when pulses are absent` (rejected: Nominal terms show a measurable tracer discrepancy; test whether the blind observations identify a better wall correction before choosing the fallback.)

```bash
cd /tmp; cat > blind_probe.py <<'PY'
from transport_probe import *
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; cal=calibrate(s)
f,B,h=prepare(s,'row','arith',-1); nom=np.repeat([s['robin_x'],s['robin_y']],2)/np.repeat(h,2)
y=cal(s['y_obs_raw']); Z=s['Z_nuisance'];q=np.linalg.qr(Z)[0]; W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]);Y=np.column_stack([cal(p['y_pulse_raw']) for p in s['reference_tracer_pulses']]); C=null_space(np.ones((1,B.shape[1]))); w0=np.full(B.shape[1],s['total_budget']/B.shape[1]);
def fit_blind(mask,pen):
 q=np.linalg.qr(Z[mask])[0]; y0=y[mask];
 def fun(z):
  S=f(nom*np.exp(z))[0][mask]; a=y0-S@w0; X=S@C; a=a-q@(q.T@a);X=X-q@(q.T@X); x=np.linalg.lstsq(X,a,rcond=None)[0]; return a-X@x,w0+C@x
 r=least_squares(lambda z:np.r_[fun(z)[0],pen*z],np.zeros(4),bounds=(-2,2),gtol=1e-13); Sd=f(nom*np.exp(r.x))[0]; err=y-Sd@fun(r.x)[1];g=np.linalg.lstsq(Z[mask],err[mask],rcond=None)[0];return nom*np.exp(r.x),fun(r.x)[1],err-Z@g
r=least_squares(lambda b:(f(b,B@W)[0]-Y).ravel(),nom,bounds=(0,np.inf),gtol=1e-13); print('tracer wall',r.x,'pulse residual by channel',np.sqrt(np.mean(r.fun.reshape(-1,4,2)**2,axis=(0,2))))
for pen in (0,1e-4,1e-3,1e-2):
 b,w,e=fit_blind(np.ones(len(y),bool),pen); group=np.tile(np.arange(len(s['sensor_ij']))//4,len(s['mu_shifts'])); err=[np.sqrt(np.mean(fit_blind(group!=i,pen)[2][group==i]**2)) for i in np.unique(group)];print('penalty',pen,'blind wall',b,'w',np.round(w,5),'in sample',np.sqrt(np.mean(e**2)),'worst held sensor group',max(err),'external tracer',np.sqrt(np.mean((f(b,B@W)[0]-Y)**2)))
print('telemetry correlations with raw',np.round(np.corrcoef(np.c_[s['aux_manifold_telemetry'],s['y_obs_raw'].reshape(-1,4)].T)[:4,4:],3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/blind_probe.py; cat /holdout_task/public_asset_index.json | head -100
```
- **Container Exit Code**: `0` (`2134 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"bytes": 26378,
      "summary": {
        "first": {
          "study": {
            "Lx": 2.4,
            "Ly": 2.0,
            "V_reaction_2d": {
              "dtype": "float64",
              "shape": [
                14,
                12
              ],
              "type": "ndarray"
            },
            "Z_nuisance": {
              "dtype": "float64",
              "shape": [
                96,
                3
              ],
              "type": "ndarray"
            },
            "aux_manifold_telemetry": {
              "dtype": "float64",
              "shape": [
                24,
                4
              ],
              "type": "ndarray"
            },
            "kappa_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
            },
            "kappa_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "mu_shifts": {
              "dtype": "float64",
              "shape": [
                3
              ],
              "type": "ndarray"
            },
            "nuisance_ridge": 0.08,
            "nx": 14,
            "ny": 12,
            "reference_tracer_pulses": {
              "first": {
                "w_pulse": "ndarray",
                "y_pulse_raw": "ndarray"
              },
              "len": 2,
              "type": "list"
            },
            "robin_x": 1.0378557606513326,
            "robin_y": 1.1626571902766067,
            "row_bank_id": {
              "dtype": "int64",
              "shape": [
                96
              ],
              "type": "ndarray"
            },
            "sensor_calibration_sweeps": {
              "raw_rosette": {
                "dtype": "float64",
                "shape": [
                  80,
                  4
                ],
                "type": "ndarray"
              },
              "ref_rosette": {
                "dtype": "float64",
                "shape": [
                  80,
                  4
                ],
                "type": "ndarray"
              },
              "single_bank_sweeps": {
                "bank_id": "ndarray",
                "raw_readout": "ndarray",
                "ref_signal": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
```

### Turn 6 — `execute_in_remote_docker` (`31550 ms` LLM)
- **Rationale**: Compare constrained blind fits with the nominal fallback. If a wall correction improves the observation fit but worsens prediction on untouched tracer pulses, it is not reliable evidence of a boundary change.
- **Alternatives Weighed**: `Reject blind inference based on unconstrained weights` (rejected: The negative and oversized source weights in that fit violate the study's box constraints; it did not test the actual feasible model.)

```bash
cat > /tmp/constrained_probe.py <<'PY'
from transport_probe import *
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; cal=calibrate(s)
f,B,h=prepare(s,'row','arith',-1);nom=np.repeat([s['robin_x'],s['robin_y']],2)/np.repeat(h,2)
y=cal(s['y_obs_raw']);Z=s['Z_nuisance']; u=np.broadcast_to(s['u_box'],B.shape[1]); budget=s['total_budget']; ridge=s['source_ridge'];print('budget box ridge',budget,u,ridge,'obs range',y.min(),y.max())
W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]);Y=np.column_stack([cal(p['y_pulse_raw']) for p in s['reference_tracer_pulses']]); tr=least_squares(lambda b:(f(b,B@W)[0]-Y).ravel(),nom,bounds=(0,np.inf),gtol=1e-13).x
G=np.tile(np.arange(len(s['sensor_ij']))//4,len(s['mu_shifts'])); A=s['aux_manifold_telemetry'];print('aux range',A.min(0),A.max(0));print('raw rows',y[:16].reshape(-1,4));
def fit(mask,wall):
 S=f(wall)[0];z=Z[mask];pz=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); a=y[mask]-z@(pz@y[mask]); X=S[mask]-z@(pz@S[mask]); H=X.T@X+ridge*np.eye(len(u));c=X.T@a
 r=minimize(lambda w:100*(.5*w@H@w-c@w),u*budget/u.sum(),jac=lambda w:100*(H@w-c),bounds=list(zip(np.zeros_like(u),u)),constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones_like(w)}],method='SLSQP',options={'ftol':1e-13}); w=r.x
 err=y-S@w; gamma=pz@err[mask];return err-Z@gamma,w
mask=np.ones(len(y),bool)
for label,wall in [('nominal',nom),('tracer',tr)]:
 e,w=fit(mask,wall);print(label,'w',w,'rmse',np.sqrt(np.mean(e*e)),'channel mean and std',e.reshape(-1,4).mean(0),e.reshape(-1,4).std(0),'aux corr',np.round(np.corrcoef(np.c_[A,e.reshape(-1,4)].T)[:4,4:],2),'worst group',max(np.sqrt(np.mean(fit(G!=g,wall)[0][G==g]**2)) for g in np.unique(G)))
for prior in [.02,.05,.1,.2]:
 def blind(m):
  r=least_squares(lambda z:np.r_[fit(m,nom*np.exp(z))[0][m],prior*z],np.zeros(4),bounds=(-.7,.7),gtol=1e-8,diff_step=1e-4);return nom*np.exp(r.x)
 wall=blind(mask);e,w=fit(mask,wall); cv=[np.sqrt(np.mean(fit(G!=g,blind(G!=g))[0][G==g]**2)) for g in np.unique(G)];print('prior',prior,'wall',wall,'w',w,'obs rmse',np.sqrt(np.mean(e*e)),'worst/group pooled',max(cv),np.linalg.norm(cv)/np.sqrt(len(cv)),'external tracer',np.sqrt(np.mean((f(wall,B@W)[0]-Y)**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/constrained_probe.py
```
- **Container Exit Code**: `0` (`2202 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
budget box ridge 1.4 [0.42 0.42 0.42 0.42 0.42 0.42 0.42 0.42] 0.04 obs range -0.4813562095411732 0.6617572384212785
aux range [-1.18255648 -3.24890432 -3.36794063 -2.73200976] [2.11462431 1.33425128 1.51508101 1.69095615]
raw rows [[ 0.05154787  0.19336471  0.27210357  0.27129541]
 [ 0.03664163  0.45389894  0.36313272 -0.12068094]
 [ 0.15662262 -0.25179023  0.2104939  -0.18646787]
 [-0.08577111  0.33225689  0.11599549 -0.33179047]]
nominal w [0.0982776  0.12488941 0.13826431 0.10417759 0.30179107 0.08059762
 0.3360832  0.2159192 ] rmse 0.009070922372454437 channel mean and std [-0.0007979  -0.00180614 -0.00323138 -0.00127348] [0.00760799 0.00882833 0.00843295 0.01030672] aux corr [[-0.11 -0.15  0.05 -0.47]
 [-0.25 -0.1   0.26  0.13]
 [-0.    0.27 -0.13 -0.19]
 [ 0.24 -0.54  0.14  0.05]] worst group 0.012057943761804434
tracer w [0.09516577 0.12621821 0.13951666 0.10557972 0.2980197  0.08124045
 0.33903005 0.21522945] rmse 0.009040978189301321 channel mean and std [-0.00056466 -0.00142497 -0.00285949 -0.00092698] [0.00759532 0.00889444 0.00853773 0.01028975] aux corr [[-0.1  -0.14  0.06 -0.47]
 [-0.25 -0.08  0.27  0.12]
 [ 0.    0.26 -0.13 -0.19]
 [ 0.23 -0.55  0.13  0.05]] worst group 0.011760205917650684
prior 0.02 wall [ 5.68137557  4.76628441  5.69013194 10.94007491] w [0.1195739  0.12093972 0.13368157 0.09535954 0.26955804 0.09752303
 0.32215878 0.24120542] obs rmse 0.008657176412645375 worst/group pooled 0.011405189727149344 0.010062847388427974 external tracer 0.0039061914614203857
prior 0.05 wall [5.77951993 5.38322324 6.21945524 8.12877412] w [0.10553199 0.12363074 0.13647236 0.10131044 0.2861707  0.08923828
 0.33093487 0.22671062] obs rmse 0.008818513587961032 worst/group pooled 0.011276811145462914 0.01022736095936198 external tracer 0.0017861699119394883
prior 0.1 wall [5.72974439 5.61917239 6.42401338 6.99470941] w [0.09960998 0.12463619 0.13764943 0.10371259 0.29566092 0.08443678
 0.33479195 0.21950216] obs rmse 0.008954909111087893 worst/group pooled 0.0117057991679729 0.010320071054044925 external tracer 0.0006150722294921272
prior 0.2 wall [5.66377223 5.64041321 6.42610654 6.57194225] w [0.09836599 0.12485443 0.13807996 0.10416984 0.29988252 0.08191142
 0.33583249 0.21690335] obs rmse 0.009029731405861368 worst/group pooled 0.011943642268971148 0.010353910499483953 external tracer 0.000394524377731765
```

### Turn 7 — `execute_in_remote_docker` (`79065 ms` LLM)
- **Rationale**: Build the solver around the validated stencil and exact QP. The isolated schema run will expose import, calibration, or output-interface failures before stress-testing the certificate and blind branch.
- **Alternatives Weighed**: `Use weakly regularized blind wall fits` (rejected: They improve observation residuals but raise independent tracer RMSE above nominal, so the apparent improvement does not identify better wall coefficients.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated steady transport inversion on a box simplex."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibrator(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    x = (x - mean) / scale
    powers = [()] + [(i,) for i in range(4)]
    candidates = [powers[:]]
    for degree in (2, 3):
        powers += list(combinations_with_replacement(range(4), degree))
        if len(powers) <= len(x) * 0.7:
            candidates.append(powers[:])

    def design(z, terms):
        return np.column_stack([np.prod(z[:, t], axis=1) for t in terms])

    best = np.inf
    for terms in candidates:
        D = design(x, terms)
        sq = 0.0
        for k in range(5):
            hold = np.arange(len(x)) % 5 == k
            coef = np.linalg.lstsq(D[~hold], raw[~hold], rcond=None)[0]
            sq += np.sum((D[hold] @ coef - raw[hold]) ** 2)
        if sq < 0.85 * best:
            best = sq
            selected = terms
    coef = np.linalg.lstsq(design(x, selected), raw, rcond=None)[0]
    linear = np.linalg.lstsq(design(x, candidates[0]), raw, rcond=None)[0]

    def inverse(y):
        y = np.asarray(y, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(linear[1:].T, (y - linear[0]).T).T
        if len(selected) > 5:
            for i, target in enumerate(y):
                def residual(p):
                    return design(p[None, :], selected)[0] @ coef - target

                def jacobian(p):
                    jac = np.zeros((4, 4))
                    for term, c in zip(selected[1:], coef[1:]):
                        for j, channel in enumerate(term):
                            jac[:, channel] += c * np.prod(p[list(term[:j] + term[j+1:])])
                    return jac

                z[i] = least_squares(residual, z[i], jac=jacobian,
                                     gtol=1e-11, max_nfev=40).x
        return (z * scale + mean).ravel()

    return inverse


def _axis(k, v, h):
    k, v = np.asarray(k), np.asarray(v) / h
    f = (k[:-1] + k[1:]) / (2 * h**2)
    lower = -f - np.maximum(v[1:], 0)
    upper = -f + np.minimum(v[:-1], 0)
    return diags((lower, np.r_[-upper, 0] + np.r_[0, -lower], upper),
                 (-1, 0, 1), format='csc')


def _transport(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx'] / (nx - 1), s['Ly'] / (ny - 1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    identity = eye(nx * ny, format='csc')
    base = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
            diags(np.asarray(s['V_reaction_2d']).ravel(order='F'))).tocsc()
    B = np.column_stack([np.asarray(src).ravel(order='F') for src in s['source_modes']])
    i, j = np.asarray(s['sensor_ij']).T
    ix = i + nx * j
    walls = np.column_stack([np.tile(np.arange(nx) == k, ny) for k in (0, nx-1)] +
                            [np.repeat(np.arange(ny) == k, nx) for k in (0, ny-1)])
    nominal = np.repeat([s['robin_x'] / hx, s['robin_y'] / hy], 2)

    def forward(wall, rhs=B):
        A = base + diags(walls @ wall, format='csc')
        readings = []
        for mu in s['mu_shifts']:
            U = splu(A + mu * identity).solve(rhs)
            readings.append(U[ix])
        return np.concatenate(readings, axis=0), U

    return forward, B, nominal


def _box_simplex(H, c, u, budget):
    """Feasible active-set QP. Status is -1 at lower, +1 at upper, 0 free."""
    n = len(c)
    x = u * (budget / u.sum()) if u.sum() else np.zeros(n)
    status = np.zeros(n, dtype=int)
    status[u == 0] = -1
    scale = max(1.0, np.max(np.abs(c)), np.max(np.abs(H)) * max(1., budget))
    tol = 1e-12 * scale
    for _ in range(20 * (n + 1)**2):
        free = status == 0
        g = H @ x - c
        if not free.any():
            lo = (status == -1) & (u > 0)
            hi = status == 1
            a = np.max(-g[lo], initial=-np.inf)
            b = np.min(-g[hi], initial=np.inf)
            if a <= b + tol:
                nu = np.clip(0., a, b)
                return x, nu
            status[np.flatnonzero(lo)[np.argmax(-g[lo])]] = 0
            status[np.flatnonzero(hi)[np.argmin(-g[hi])]] = 0
            continue
        inds = np.flatnonzero(free)
        k = len(inds)
        F = H[np.ix_(inds, inds)]
        K = np.block([[F, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
        rhs = np.r_[-g[free], budget - x.sum()]
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        p, nu = sol[:-1], sol[-1]
        trial = x[free] + p
        ratio = np.full(k, np.inf)
        low = trial < -1e-13
        high = trial > u[free] + 1e-13
        ratio[low] = -x[free][low] / p[low]
        ratio[high] = (u[free][high] - x[free][high]) / p[high]
        hit = np.argmin(ratio)
        if ratio[hit] < 1:
            x[free] += max(0., ratio[hit]) * p
            i = inds[hit]
            status[i] = 1 if high[hit] else -1
            x[i] = u[i] if high[hit] else 0.
            continue
        x[free] = np.clip(trial, 0., u[free])
        g = H @ x - c + nu
        violation = np.where(u > 0, status * g, 0.)
        worst = np.argmax(violation)
        if violation[worst] <= tol:
            return x, nu
        status[worst] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    calibrate = _calibrator(s['sensor_calibration_sweeps'])
    y = calibrate(s['y_obs_raw'])
    forward, B, nominal = _transport(s)
    m = B.shape[1]
    u = np.broadcast_to(np.asarray(s['u_box'], dtype=float), (m,))
    budget = float(s['total_budget'])
    ridge = float(s['source_ridge'])
    Z = np.asarray(s['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    y_tilde = y - Z @ (Pz @ y)

    def fit(S):
        X = S - Z @ (Pz @ S)
        H = X.T @ X + ridge * np.eye(m)
        c = X.T @ y_tilde
        w, nu = _box_simplex(H, c, u, budget)
        return X, H, c, w, nu

    pulses = s['reference_tracer_pulses']
    if pulses:
        W = np.column_stack([p['w_pulse'] for p in pulses])
        Y = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
        rhs = B @ W
        wall = least_squares(lambda b: (forward(b, rhs)[0] - Y).ravel(),
                             nominal, bounds=(0., np.inf), gtol=1e-12).x
    else:
        # Blind observations confound wall and source effects. Anchor the wall
        # correction to the supplied coefficients on a relative scale.
        S0, _ = forward(nominal)
        anchor = max(1e-8, 0.2 * np.linalg.norm(S0 @ fit(S0)[3]))

        def residual(delta):
            S, _ = forward(nominal * np.exp(delta))
            X, _, _, w, _ = fit(S)
            return np.r_[X @ w - y_tilde, anchor * delta]

        delta = least_squares(residual, np.zeros(4), bounds=(-0.7, 0.7),
                              gtol=1e-9).x
        wall = nominal * np.exp(delta)

    S, U = forward(wall)
    X, H, c, w, nu = fit(S)
    g = H @ w - c + nu
    lower = np.where(w <= 1e-12, np.maximum(g, 0.), 0.)
    upper = np.where(u - w <= 1e-12, np.maximum(-g, 0.), 0.)
    residual = X @ w - y_tilde
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lagrangian = primal + nu * (w.sum() - budget) - lower @ w + upper @ (w - u)
    stationarity = g - lower + upper
    if ridge > 0.:
        dual = lagrangian - 0.5 * stationarity @ np.linalg.solve(H, stationarity)
    else:
        # Supporting hyperplane of the equality Lagrangian on the box.
        dual = (primal + nu * (w.sum() - budget) +
                np.minimum(-g*w, g*(u-w)).sum())
    return {'weights': w, 'terminal_field': (U @ w).reshape((s['nx'], s['ny']), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': float(nu),
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': primal,
                                'dual_lower_bound': float(min(primal, dual))}}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
ls -ld /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7a88ffb4ee12656 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7a88ffb (directory).
```

### Turn 8 — `execute_in_remote_docker` (`38133 ms` LLM)
- **Rationale**: Measure tracer holdouts using the saved implementation, then check that the certificate and terminal field agree with a separately assembled objective. Stress the QP at active bounds and budget endpoints, and exercise the blind branch on the public study with pulses removed.

```bash
cat > /tmp/verify_batch.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares, minimize

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
y = b._calibrator(s['sensor_calibration_sweeps'])
f, B, nom = b._transport(s)
pulses = s['reference_tracer_pulses']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([y(p['y_pulse_raw']) for p in pulses])
R = B @ W

def fit(mask):
    return least_squares(lambda wall: (f(wall, R)[0] - Y)[mask], nom,
                         bounds=(0, np.inf), gtol=1e-12).x

idx = np.arange(Y.shape[0]); n = len(s['sensor_ij'])
groups = [np.broadcast_to(((idx % n // 4) == g)[:, None], Y.shape) for g in range(n // 4)]
shifts = [np.broadcast_to((idx // n == g)[:, None], Y.shape) for g in range(len(s['mu_shifts']))]
pulse_slices = [np.broadcast_to(np.arange(Y.shape[1]) == g, Y.shape) for g in range(Y.shape[1])]
for name, slices in [('group', groups), ('shift', shifts), ('pulse', pulse_slices)]:
    worst_fit = max(np.sqrt(np.mean(((f(fit(~mask), R)[0]-Y)[mask])**2)) for mask in slices)
    worst_nom = max(np.sqrt(np.mean(((f(nom, R)[0]-Y)[mask])**2)) for mask in slices)
    assert worst_fit < worst_nom
    print('tracer', name, 'worst-fit', worst_fit, 'worst-nominal', worst_nom)
wall = fit(np.ones_like(Y, dtype=bool)); S, U = f(wall)
wout = b.solve_batch(s); w = wout['weights']; cert = wout['kkt_certificate']; budget = s['total_budget']; u = np.broadcast_to(s['u_box'], w.shape)
Z = s['Z_nuisance']; P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
X = P @ S; yr = P @ y(s['y_obs_raw']); r = X @ w - yr; ridge = s['source_ridge']
stationarity = np.max(np.abs(X.T @ r + ridge*w + cert['nu_sum'] - cert['lam_lower'] + cert['lam_upper']))
field_error = np.max(np.abs(wout['terminal_field'].ravel(order='F') - U @ w))
gap = cert['primal_objective'] - cert['dual_lower_bound']
assert np.allclose(w, cert['weights']) and abs(w.sum()-budget) < 1e-12 and np.all((w>=0)&(w<=u))
assert abs(cert['primal_objective'] - .5*(r@r+ridge*(w@w))) < 1e-12
assert max(stationarity, field_error, gap) < 1e-10
print('public certificate stationarity, gap, field error', stationarity, gap, field_error)
blind = b.solve_batch(dict(s, reference_tracer_pulses=None)); wb = blind['weights']; cb = blind['kkt_certificate']
assert abs(wb.sum()-budget) < 1e-12 and np.all((wb>=0)&(wb<=u))
assert np.isfinite(blind['terminal_field']).all() and 0 <= cb['primal_objective']-cb['dual_lower_bound'] < 1e-10
print('public with pulses removed: weight and field deviations', np.linalg.norm(wb-w), np.linalg.norm(blind['terminal_field']-wout['terminal_field']))
rng = np.random.default_rng(327); worst = np.zeros(3)
for t in range(500):
    m = int(rng.integers(2, 40)); A = rng.normal(size=(m,m)); H = A.T @ A / m + .001*np.eye(m); c = rng.normal(size=m)
    u = rng.uniform(.01, 1, m); u[t % m] = 0
    budget = [0, u.sum(), u.sum()*1e-8, u.sum()*(1-1e-8), rng.uniform()*u.sum(), min(u[u>0])][t%6]
    x, nu = b._box_simplex(H,c,u,budget); g = H@x-c+nu
    dual_gap = float(-np.minimum(-g*x, g*(u-x)).sum())
    worst = np.maximum(worst, [abs(x.sum()-budget), max(0., np.max(-x), np.max(x-u)), dual_gap])
assert np.max(worst) < 1e-9, worst
print('500 box-simplex cases worst budget residual, box violation, supporting-plane gap', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py
```
- **Container Exit Code**: `0` (`1688 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tracer group worst-fit 6.839849338813825e-05 worst-nominal 0.001016375837967981
tracer shift worst-fit 5.9548783190973805e-05 worst-nominal 0.0007743064157658654
tracer pulse worst-fit 4.3548851286261336e-05 worst-nominal 0.0006220556817058294
public certificate stationarity, gap, field error 1.8041124150158794e-16 0.0 0.0
public with pulses removed: weight and field deviations 0.006100043301755656 0.002439043532383739
500 box-simplex cases worst budget residual, box violation, supporting-plane gap [3.55271368e-15 0.00000000e+00 4.42522649e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7a88ffb4ee12656 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7a88ffb (directory).
```

### Turn 9 — `submit_completed_task` (`14434 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, separate checks of the public objective and terminal field, and 500 box-simplex stress cases. Held-out tracer groups, shifts, and pulses all favored side-specific walls over nominal walls.
- **Alternatives Weighed**: `Use weakly anchored blind wall corrections` (rejected: They improved the fit to routine observations but predicted withheld tracer responses worse than the nominal walls.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:8309'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` with `solve_batch(study)`. The isolated public schema check passed. Worst held-out tracer RMSE was 6.84e-5, and the public KKT stationarity residual was 1.81e-16. The blind branch was tested by withholding public tracer pulses; no apparatus archive was available to validate transfer across campaigns.`

### Turn 10 — `execute_in_remote_docker` (`32004 ms` LLM)
- **Rationale**: Test whether simple rational attenuation along individual or paired channels beats the affine calibration on real held-out sweeps. Promote a richer calibration only if it improves held-out prediction; compare the affine fit separately against the single-bank sweeps and verify the saved entry point in an isolated process.
- **Alternatives Weighed**: `Run relay-mode, drift-operator, or classification checks` (rejected: The stated problem is a steady linear transport inversion, not a relay, stochastic drift, or classification task; inspect the actual study fields for any relevant cohort or apparatus metadata.)

```bash
cat > /tmp/audit_campaign.py <<'PY'
import importlib.util, pickle
from itertools import combinations, combinations_with_replacement
import numpy as np
spec = importlib.util.spec_from_file_location('batch', '/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('study fields', list(s))
c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); bank=c['single_bank_sweeps']; print('sweep count',len(x),'batch, relay and state-like fields',[k for k in s if any(t in k.lower() for t in ('batch','relay','state','apparatus','bank','lineage'))]);
def d(x, degree):
    terms=[()] + [t for i in range(1,degree+1) for t in combinations_with_replacement(range(4),i)]
    return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
q=np.column_stack([x[:,i] for i in range(4)]+[(x[:,i]+sg*x[:,j])/np.sqrt(2) for i,j in combinations(range(4),2) for sg in (-1,1)])
models={'affine':d(x,1),'quadratic':d(x,2),'cubic':d(x,3)}
# Search rational axes and pairwise sums/differences using the inner folds only.
rat=[(f'axis_or_pair={i},a={a}', d(x,1)/(1+a*q[:,i,None]**2)) for i in range(q.shape[1]) for a in (.002,.01,.05,.2)]
def err(D,t,v):
    return D[v]@np.linalg.lstsq(D[t],y[t],rcond=None)[0]-y[v]
rng=np.random.default_rng(311); folds=np.empty(len(x),int); folds[rng.permutation(len(x))]=np.arange(len(x))%5
out={k:[] for k in (*models,'rational_nested')}
for i in range(5):
    t=folds!=i; v=~t
    for k,D in models.items(): out[k].append(err(D,t,v))
    name,D=min(rat,key=lambda item:sum(np.sum(err(item[1],t&(folds!=j),folds==j)**2) for j in range(5) if j!=i))
    out['rational_nested'].append(err(D,t,v))
for k,e in out.items():
    print('calibration',k,'worst fold/channel RMSE',max(np.sqrt(np.mean(a*a,axis=0)).max() for a in e),'pooled',np.sqrt(np.mean(np.concatenate(e)**2)))
D=d(x,1); L=np.linalg.lstsq(D,y,rcond=None)[0]; bi=bank['bank_id']; br=bank['ref_signal']; bankerr=L[0,bi]+br*L[1+bi,bi]-bank['raw_readout']; print('single-bank external worst bank RMSE',max(np.sqrt(np.mean(bankerr[bi==i]**2)) for i in range(4)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_campaign.py
if [ -d /tmp/hegel_continual_archive ]; then ls -l /tmp/hegel_continual_archive; fi
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`710 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study fields ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'reference_tracer_pulses', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
sweep count 80 batch, relay and state-like fields ['row_bank_id']
calibration affine worst fold/channel RMSE 1.2204220694942287e-05 pooled 9.923269528308575e-06
calibration quadratic worst fold/channel RMSE 1.4533112061444217e-05 pooled 1.0923551897633387e-05
calibration cubic worst fold/channel RMSE 2.7582050266745123e-05 pooled 1.6780322655040468e-05
calibration rational_nested worst fold/channel RMSE 0.003138416154838839 pooled 0.0016984764018276605
single-bank external worst bank RMSE 1.042912006811545e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7a88ffb4ee12656 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7a88ffb (directory).
```

### Turn 11 — `execute_in_remote_docker` (`28123 ms` LLM)
- **Rationale**: Hold calibration, wall-fit freedom, and validation slices fixed while changing spacing and upwind placement separately. If a control matches the saved stencil on held-out tracers, narrow the discretization claim rather than attributing the gain to both changes.
- **Alternatives Weighed**: `Rely on schema validation or nominal-wall comparisons` (rejected: Neither isolates whether the stencil itself improves predictions when competing stencils receive the same four-wall fit.)

```bash
cat > /tmp/check_stencil_holdouts.py <<'PY'
import importlib.util
import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/tmp')
from transport_probe import prepare
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    s = pickle.load(stream)[0]['study']
cal = b._calibrator(s['sensor_calibration_sweeps'])
pulses = s['reference_tracer_pulses']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([cal(p['y_pulse_raw']) for p in pulses])
row = np.arange(Y.shape[0]); n = len(s['sensor_ij'])
slices = {'rosette': [np.broadcast_to((row % n // 4 == i)[:,None], Y.shape) for i in range(n//4)],
          'shift': [np.broadcast_to((row // n == i)[:,None], Y.shape) for i in range(len(s['mu_shifts']))],
          'pulse': [np.broadcast_to(np.arange(Y.shape[1]) == i, Y.shape) for i in range(Y.shape[1])]}

def scores(forward, B, nominal):
    R = B @ W
    def held(mask):
        def e(wall): return forward(wall, R)[0] - Y
        wall = least_squares(lambda w: e(w)[~mask], nominal, bounds=(0, np.inf), gtol=1e-12).x
        return np.sqrt(np.mean(e(wall)[mask]**2))
    return {name: max(map(held, masks)) for name, masks in slices.items()}

def trial(adv, grid):
    f, B, h = prepare(s, adv, 'arith', grid)
    return scores(f, B, np.repeat([s['robin_x'], s['robin_y']], 2)/np.repeat(h, 2))

candidate = scores(*b._transport(s))
controls = {'cell_width_face_upwind': trial('center',0), 'cell_width_row_upwind': trial('row',0),
            'endpoint_face_upwind': trial('center',-1)}
print('Saved endpoint row-upwind candidate; worst held-out tracer RMSE:', candidate)
for name, score in controls.items():
    print(name, score, 'control/candidate', {k: score[k]/candidate[k] for k in slices})
assert all(candidate[k] < score[k] for score in controls.values() for k in slices)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_stencil_holdouts.py
```
- **Container Exit Code**: `0` (`2398 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Saved endpoint row-upwind candidate; worst held-out tracer RMSE: {'rosette': 6.839849338813825e-05, 'shift': 5.9548783190973805e-05, 'pulse': 4.3548851286261336e-05}
cell_width_face_upwind {'rosette': 0.006808715057852494, 'shift': 0.005879343202865086, 'pulse': 0.005903603727943339} control/candidate {'rosette': 99.54481042755351, 'shift': 98.73154223840221, 'pulse': 135.56278876650393}
cell_width_row_upwind {'rosette': 0.006200274311153737, 'shift': 0.005504340054757727, 'pulse': 0.004792765302838187} control/candidate {'rosette': 90.6492819362158, 'shift': 92.43413147679658, 'pulse': 110.05491904559591}
endpoint_face_upwind {'rosette': 0.012095949050052195, 'shift': 0.010394247943044664, 'pulse': 0.009336846933618564} control/candidate {'rosette': 176.84525566099515, 'shift': 174.55013160739423, 'pulse': 214.39938500890207}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7a88ffb4ee12656 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7a88ffb (directory).
```

### Turn 12 — `submit_completed_task` (`15628 ms` LLM)
- **Rationale**: Matched stencil ablations favored the saved candidate on held-out rosettes, shifts, and pulses even when each control could fit four wall terms. The same artifact passed isolated schema validation, terminal-field and certificate checks, and 500 box-simplex stress cases.
- **Alternatives Weighed**: `Replace the affine rosette fit with a higher-order or rational fit` (rejected: On the public sweeps, each richer fit increased the worst held-out channel error.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema check passed in an isolated Python process. Worst held-out tracer RMSE was 6.84e-5, and the public KKT stationarity residual was 1.81e-16. The blind branch was tested with public tracer pulses withheld; cross-campaign transfer could not be verified because no apparatus archive was available.`

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
