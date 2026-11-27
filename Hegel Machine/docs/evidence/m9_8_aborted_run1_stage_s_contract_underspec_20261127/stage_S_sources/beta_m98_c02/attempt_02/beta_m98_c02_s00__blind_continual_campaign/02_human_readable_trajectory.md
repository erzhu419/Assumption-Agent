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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `385,995` (`371,128` / `144,768` / `14,867`) |
| **Total Episode Wall Time (s)** | `453.56s` |

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

### Turn 1 — `execute_in_remote_docker` (`16184 ms` LLM)
- **Rationale**: Inspect the actual array ordering, calibration sweep structure, and available lineage evidence before choosing a sensor inverse or boundary model.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle, numpy as np
with open('public_batch.pkl','rb') as f: s=pickle.load(f)
def show(x,p='',d=0):
 if isinstance(x,dict) and d<4:
  for k,v in x.items(): show(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)) and len(x) and isinstance(x[0],dict) and d<4:
  print(p,type(x).__name__,len(x)); show(x[0],p+'/0',d+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,x.ravel()[:min(10,x.size)])
 else: print(p,repr(x)[:350])
show(s)
PY
if [ -d /tmp/hegel_continual_archive ]; then du -ah /tmp/hegel_continual_archive | sort -h | tail -25; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 9000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling calibration' 2>&1 | head -30
```
- **Container Exit Code**: `0` (`370 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
iled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_creep_corrected_retraction_calibration@cab6fbef | calibrate_retraction(times_s: list[np.ndarray], forces_N: list[np.ndarray], raw_displacements_um: list[np.ndarray], frame_compliance_um_per_N: float, *, transition_window_s: float=0.015, curvature_to_noise: float=0.8, slip_threshold_um: float=0.35) -> tuple[float, list[float | None]] | Return median elastic stiffness (N/µm) and first loading slip forces (N) | REQ: {'field': 'specimen_count', 'op': '>=', 'value': 1}; {'field': 'hold_sample_count', 'op' | --describe autogenous_creep_corrected_retraction_calibration.calibrate_retraction
```

### Turn 2 — `execute_in_remote_docker` (`17376 ms` LLM)
- **Rationale**: The first inspection was truncated. A compact inventory will show whether single-bank sweeps identify the calibration structure and whether tracer or archive evidence is available for boundary fitting.

```bash
cd /holdout_task && python3 - <<'PY'
import numpy as np,pickle
b=pickle.load(open('public_batch.pkl','rb')); print('cases',len(b))
for item in b:
 print('item',list(item)); s=item['study']
 for k,v in s.items():
  if isinstance(v,dict):
   print(k,[(j,np.shape(a) if not isinstance(a,(dict,list)) else (type(a).__name__,len(a))) for j,a in v.items()]);
   for j,a in v.items():
    if isinstance(a,np.ndarray): print(j,a[:5]);
    elif isinstance(a,list): print(j,repr(a[:1])[:850]);
    elif isinstance(a,dict): print(j,[(q,np.shape(z)) for q,z in a.items()]);
  elif isinstance(v,list): print(k,len(v),[(j,np.shape(a)) for j,a in v[0].items()] if v and isinstance(v[0],dict) else repr(v)[:400]);
  elif isinstance(v,np.ndarray): print(k,v.shape,np.min(v),np.max(v), v.tolist() if v.size<=40 else '')
  else: print(k,repr(v))
PY
if [ -d /tmp/hegel_continual_archive ]; then ls -la /tmp/hegel_continual_archive; head -c 3500 /tmp/hegel_continual_archive/archive_apparatus_log.json; else echo 'No continual archive'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -25
```
- **Container Exit Code**: `0` (`428 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
76356834465 
mu_shifts (3,) 0.15 1.1 [0.15, 0.55, 1.1]
source_modes (8, 14, 12) 4.1508628561618226e-15 0.9940952851255738 
sensor_ij (32, 2) 0 12 
row_bank_id (96,) 0 3 
sensor_calibration_sweeps [('ref_rosette', (80, 4)), ('raw_rosette', (80, 4)), ('single_bank_sweeps', ('dict', 3))]
ref_rosette [[-1.2  0.   0.   0. ]
 [-0.4  0.   0.   0. ]
 [ 0.5  0.   0.   0. ]
 [ 1.3  0.   0.   0. ]
 [ 0.  -1.2  0.   0. ]]
raw_rosette [[-1.19998264e+00  1.86531004e-06  9.43561705e-07  1.27946941e-05]
 [-3.99998519e-01  3.96163490e-07 -2.04448679e-05  1.14334134e-06]
 [ 4.99995940e-01 -1.58914905e-05 -7.27891336e-06 -1.85020647e-05]
 [ 1.29999313e+00  7.49505950e-06  7.84360136e-06 -6.56276159e-06]
 [-6.13073073e-06 -1.19999166e+00  1.60310971e-05 -5.25927337e-06]]
single_bank_sweeps [('bank_id', (24,)), ('ref_signal', (24,)), ('raw_readout', (24,))]
Z_nuisance (96, 3) -2.976173027132081 2.5972055285562865 
y_obs_raw (96,) -0.48135233705871805 0.6617577306394699 
reference_tracer_pulses 2 [('w_pulse', (8,)), ('y_pulse_raw', (96,))]
aux_manifold_telemetry (24, 4) -3.367940625828947 2.1146243082381235 
nuisance_ridge 0.08
source_ridge 0.04
u_box 0.42
total_budget 1.4
No continual archive
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
```

### Turn 3 — `execute_in_remote_docker` (`17862 ms` LLM)
- **Rationale**: Quantify whether the near-identity appearance holds across the full calibration sweep, and inspect tracer layout and nuisance signatures before comparing finite-volume conventions.

```bash
ls -la /pretrain_task; cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; q=s['sensor_calibration_sweeps']; x=q['ref_rosette']; y=q['raw_rosette']; print('calibration RMS/max by channel',np.sqrt(np.mean((x-y)**2,axis=0)),np.max(abs(x-y),axis=0)); print('single bank',[(k,v[:8].tolist()) for k,v in q['single_bank_sweeps'].items()]); print('sensors',s['sensor_ij'].tolist()); print('banks',s['row_bank_id'][:32]);
for p in s['reference_tracer_pulses']:
 a=p['y_pulse_raw'].reshape(3,-1); print('pulse weights',p['w_pulse'],'range and means',a.min(1),a.max(1),a.mean(1)); print('first four of each shift',a[:,:4]); print('Z projection',np.linalg.lstsq(s['Z_nuisance'],a.ravel(),rcond=None)[0]);
PY
```
- **Container Exit Code**: `0` (`354 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 12:27 ..
calibration RMS/max by channel [9.46087274e-06 9.56862296e-06 9.32215656e-06 9.62108383e-06] [2.80798509e-05 2.01749577e-05 3.19056831e-05 2.36769568e-05]
single bank [('bank_id', [0, 0, 0, 0, 0, 0, 1, 1]), ('ref_signal', [-1.1, -0.6, -0.1, 0.4, 0.9, 1.4, -1.1, -0.6]), ('raw_readout', [-1.099998766282047, -0.5999954535956783, -0.09999464688094413, 0.39999792625379693, 0.8999852834465344, 1.3999848754546098, -1.0999746522031162, -0.5999962759246986])]
sensors [[0, 2], [8, 6], [4, 6], [12, 10], [4, 4], [2, 4], [6, 8], [6, 10], [8, 10], [0, 8], [12, 6], [8, 0], [10, 6], [12, 4], [10, 0], [8, 2], [10, 2], [12, 8], [10, 4], [6, 6], [10, 8], [8, 8], [2, 2], [8, 4], [2, 0], [0, 0], [12, 2], [12, 0], [4, 8], [0, 6], [2, 6], [0, 4]]
banks [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
pulse weights [1.  0.  0.  0.5 0.  0.  0.  0. ] range and means [0.01206802 0.00806345 0.00500907] [0.22337434 0.1893829  0.1591497 ] [0.09078106 0.07168581 0.0553643 ]
first four of each shift [[0.02342531 0.17522757 0.16736846 0.03058704]
 [0.01608683 0.14172437 0.13519512 0.02189933]
 [0.01035415 0.11258668 0.10679731 0.01482738]]
Z projection [ 0.00306673 -0.02506725  0.01743575]
pulse weights [0.  1.  0.  0.  0.5 0.  0.  0. ] range and means [0.01753612 0.01207278 0.00750508] [0.20870199 0.17858828 0.15116992] [0.08832625 0.0702399  0.05468422]
first four of each shift [[0.03422369 0.12309067 0.20870199 0.01809568]
 [0.02515782 0.09466826 0.17559848 0.01207278]
 [0.01758003 0.07061928 0.14555404 0.00750508]]
Z projection [ 0.0018533  -0.0248107   0.01713919]
```

### Turn 4 — `execute_in_remote_docker` (`37618 ms` LLM)
- **Rationale**: Compare grid spacing, face diffusion, and upwind conventions while profiling two or four boundary diagonal terms. Leave-one-pulse-out errors will show whether a convention generalizes to a different source mixture.
- **Alternatives Weighed**: `Fit an unrestricted transport correction` (rejected: Two public pulses can test a small set of physically distinct stencil and boundary conventions, but cannot identify an unrestricted correction.)

```bash
cat >/tmp/probe_transport.py <<'PY'
import numpy as np
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
import pickle,itertools

def axis(k,v,L,grid,mix,adv):
 n=len(k); h=L/(n-grid); f=(k[:-1]+k[1:])/2 if mix==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); a=f/h**2
 vf=(v[:-1]+v[1:])/2; p=np.maximum(vf,0)/h; m=np.maximum(-vf,0)/h
 if adv==0: l=-a-p; u=-a-m; d=np.r_[a+p,0]+np.r_[0,a+m]
 elif adv==1: l=-a-np.maximum(v[1:],0)/h; u=-a-np.maximum(-v[:-1],0)/h; d=np.r_[-u,0]+np.r_[0,-l]
 elif adv==2: l=-a-p; u=-a-m; d=np.r_[-u,0]+np.r_[0,-l]
 elif adv==3: l=-a-np.maximum(v[:-1],0)/h; u=-a-np.maximum(-v[1:],0)/h; d=np.r_[-l,0]+np.r_[0,-u]
 else: l=-a-np.maximum(v[1:],0)/h; u=-a-np.maximum(-v[:-1],0)/h; d=np.r_[a,0]+np.r_[0,a]+abs(v)/h
 return diags([l,d,u],[-1,0,1],format='csc')

def build(s,grid,mix,adv):
 nx,ny=s['nx'],s['ny']; ax=axis(s['kappa_x'],s['vel_x'],s['Lx'],grid,mix,adv); ay=axis(s['kappa_y'],s['vel_y'],s['Ly'],grid,mix,adv)
 return csc_matrix(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')))

def setup(s):
 nx,ny=s['nx'],s['ny']; ii,jj=np.indices((nx,ny)); edge=np.vstack([a.ravel(order='F') for a in (ii==0,ii==nx-1,jj==0,jj==ny-1)]).T.astype(float)
 idx=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; return edge,idx

def pred(A,edge,idx,mu,rhs,b):
 D=diags(edge@b); I=eye(A.shape[0],format='csc'); return np.concatenate([splu(A+D+t*I).solve(rhs)[idx] for t in mu])

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; edge,idx=setup(s); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); B=s['source_modes'].reshape(len(W),-1,order='F').T@W; Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); rows=[]
 for grid,mix,adv in itertools.product(range(2),range(2),range(5)):
  A=build(s,grid,mix,adv)
  for symmetry in [2,4]:
   E=edge@np.array([[1,0],[1,0],[0,1],[0,1]]) if symmetry==2 else edge
   def fit(ids, mask):
    return least_squares(lambda b:(pred(A,E,idx,mu,B[:,ids].copy(),b)-Y[:,ids])[mask].ravel(),np.ones(symmetry)*5,bounds=(-0.4,100),max_nfev=45,gtol=1e-10).x
   b=fit([0,1],slice(None)); er=pred(A,E,idx,mu,B,b)-Y
   rows.append((np.sqrt(np.mean(er**2)),grid,mix,adv,symmetry,b,np.max(abs(er))))
 for r in sorted(rows,key=lambda a:a[0])[:20]: print(r)
 for r in sorted(rows,key=lambda a:a[0])[:8]:
  _,grid,mix,adv,sy,b,_=r; A=build(s,grid,mix,adv); E=edge@np.array([[1,0],[1,0],[0,1],[0,1]]) if sy==2 else edge
  cv=[]
  for p in range(2):
   tr=1-p
   q=least_squares(lambda z:pred(A,E,idx,mu,B[:,tr:tr+1].copy(),z).ravel()-Y[:,tr],b,bounds=(-.4,100),max_nfev=40).x
   e=pred(A,E,idx,mu,B[:,p:p+1].copy(),q).ravel()-Y[:,p]; cv.append((np.sqrt(np.mean(e**2)),np.max(abs(e)),q.round(6)))
  print('LOPO',grid,mix,adv,sy,cv)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_transport.py
```
- **Container Exit Code**: `0` (`2188 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 1, 2, 4, array([5.68911693, 6.21636642, 6.12392663, 4.47589182]), 0.013945134265982512)
(0.0044111916343882234, 0, 0, 2, 4, array([5.69084915, 6.21660789, 6.12193831, 4.46868472]), 0.014016047842262264)
(0.004467162964943015, 0, 1, 4, 4, array([3.61916342, 5.17567969, 5.9777717 , 2.9837686 ]), 0.014225436854015422)
(0.004467162964943326, 0, 1, 1, 4, array([5.66083001, 6.3757394 , 5.97777208, 4.34291767]), 0.014225437544182234)
(0.00448732419073618, 0, 0, 4, 4, array([3.62097784, 5.17570286, 5.97579051, 2.97677919]), 0.014295972140790508)
(0.0044873241907362835, 0, 0, 1, 4, array([5.66264418, 6.37576334, 5.97579034, 4.33592834]), 0.014295971772236293)
(0.0046896515841394325, 0, 0, 0, 4, array([ 5.64847949, 16.65251844, 11.28188967,  6.86000756]), 0.011810559986985839)
(0.004698710282960936, 0, 1, 0, 4, array([ 5.64803052, 16.68477338, 11.28884464,  6.87438334]), 0.011856083052410968)
LOPO 1 0 4 4 [(3.5816959054339014e-05, 0.0001640595957518587, array([3.970308, 4.773769, 6.362692, 5.210659])), (4.3186582862965835e-05, 0.0001629505720396815, array([3.981106, 4.772119, 6.34865 , 5.201099]))]
LOPO 1 0 1 4 [(3.581695627216671e-05, 0.00016405959148740878, array([5.866141, 5.888112, 6.362692, 6.456545])), (4.3186586088447105e-05, 0.00016295058782783006, array([5.87694 , 5.886462, 6.34865 , 6.446985]))]
LOPO 1 1 4 4 [(4.4514392516799765e-05, 0.00014428641151535693, array([3.969652, 4.775344, 6.36562 , 5.2225  ])), (5.2467626078853404e-05, 0.00017634467193387848, array([3.978179, 4.774933, 6.348228, 5.211705]))]
LOPO 1 1 1 4 [(4.4514401338701965e-05, 0.0001442864090526047, array([5.865485, 5.889687, 6.36562 , 6.468386])), (5.246762877376731e-05, 0.00017634466398424448, array([5.874012, 5.889276, 6.348228, 6.457591]))]
LOPO 1 0 1 2 [(0.000108423358379806, 0.00040379302086795965, array([5.874488, 6.388015])), (0.00010347377071649368, 0.00038604154339558217, array([5.87317 , 6.426792]))]
LOPO 1 1 1 2 [(0.00011889048224974272, 0.00038362433155150755, array([5.874622, 6.393342])), (0.00011344123116659823, 0.0004303774074024269, array([5.870592, 6.435649]))]
LOPO 1 0 2 4 [(0.00046471505054080677, 0.0020701344170392655, array([5.908641, 5.680845, 6.468318, 6.90613 ])), (0.00038057926679546737, 0.0009661277604328805, array([5.872139, 5.74224 , 6.613623, 6.639367]))]
LOPO 1 1 2 4 [(0.00046942700508866995, 0.002136377408102491, array([5.908062, 5.682155, 6.471252, 6.919166])), (0.0003883609573520788, 0.0009545108432046506, array([5.869294, 5.744729, 6.613222, 6.650372]))]
```

### Turn 5 — `execute_in_remote_docker` (`35272 ms` LLM)
- **Rationale**: Measure the discrepancy against nominal Robin scaling, then test whether edge slopes, corners, or observation offsets improve held-out pulse predictions before adding terms to the solver.
- **Alternatives Weighed**: `Adopt the best unrestricted wall fit immediately` (rejected: The residual remains several times larger than calibration noise, and a blind campaign needs a rule for choosing boundary terms without tracer pulses.)
- **ATMS NOGOOD DECLARED**: Refuted `A shared Robin coefficient for opposite walls on each axis captures the public tracer responses.` | Witness: `Independent wall corrections cut worst held-out pulse RMSE from 1.08e-4 to 4.32e-5 for the same interior stencil.` | Blanket Diagnosis: `The inferred boundary diagonal has observable wall-dependent structure; the interior stencil is held fixed in this comparison.` | Sublated Pivot: `Fit wall-specific terms and test whether remaining errors track spatial boundary structure or observation-side effects.`

```bash
cat >/tmp/probe_mismatch.py <<'PY'
from probe_transport import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; edge,idx=setup(s); A=build(s,1,0,1); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); B=s['source_modes'].reshape(len(W),-1,order='F').T@W; Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); base=np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2); print('nominal robin / h',base); print('nominal RMSE',np.sqrt(np.mean((pred(A,edge,idx,mu,B,base)-Y)**2,axis=0))); print('solved row layout by shift',len(idx),Y.shape); t=s['sensor_calibration_sweeps']['ref_rosette']; print('reference sweeps groups',[(int(i),int(np.sum(np.count_nonzero(t,axis=1)==i))) for i in range(5)]); print('rosettes 16:36',t[16:36]); print('rosette ranges',t.min(0),t.max(0)); print('telemetry mean/cov',np.mean(s['aux_manifold_telemetry'],axis=0),np.cov(s['aux_manifold_telemetry'].T))
xx,yy=np.indices((nx,ny)); x=(xx.ravel(order='F')/(nx-1)-.5)*2; y=(yy.ravel(order='F')/(ny-1)-.5)*2; Z=s['Z_nuisance']; bank=np.eye(4)[s['row_bank_id']]; rms=lambda a:np.sqrt(np.mean(a*a)); E2=np.column_stack((edge[:,:2].sum(1),edge[:,2:].sum(1))); corner=(E2[:,0]*E2[:,1])[:,None]
for tag,E,N in [('symmetric',E2,None),('walls',edge,None),('walls+slopes',np.column_stack((edge,edge[:,:2]*y[:,None],edge[:,2:]*x[:,None])),None),('walls+corner',np.column_stack((edge,corner)),None),('walls+rosette offset',edge,bank),('walls+nuisance',edge,Z),('walls+curvature',np.column_stack((edge,edge[:,:2]*y[:,None]**2,edge[:,2:]*x[:,None]**2)),None)]:
  E=np.asarray(E); n=E.shape[1]; bb=np.r_[base if n>=4 else base[::2],np.zeros(max(0,n-4))]; fs=lambda b,p: pred(A,E,idx,mu,B[:,p].copy(),b[:n])+(N@b[n:])[:,None] if N is not None else pred(A,E,idx,mu,B[:,p].copy(),b)
  if N is not None: bb=np.r_[bb,np.zeros(N.shape[1])]
  scores=[]
  for ps,pts in [([0,1],[0,1]),([0],[1]),([1],[0])]:
   z=least_squares(lambda b:(fs(b,ps)-Y[:,ps]).ravel(),bb,gtol=1e-11).x; scores.append(rms(fs(z,pts)-Y[:,pts]));
   if len(ps)==2: pars=z; resid=fs(z,ps)-Y[:,ps]
  print(tag,np.round(pars,7),scores,'max',np.max(abs(resid)))
  if tag=='walls':
   print('residual shift/bank rms',np.sqrt((resid.reshape(3,8,4,2)**2).mean((1,3)))); print('largest residuals',[(i//len(idx),s['sensor_ij'][i%len(idx)].tolist(),resid[i].round(7).tolist()) for i in np.argsort(abs(resid).max(1))[-12:]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_mismatch.py
```
- **Container Exit Code**: `0` (`1178 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0304661  0.3026242 ]
 [ 1.76780212  0.03829004 -0.32380646  1.26680418]]
rosette ranges [-1.34414968 -1.2        -1.22364327 -1.36051815] [1.76780212 1.78199005 1.79914087 1.74606382]
telemetry mean/cov [ 0.22475997  0.02821447 -0.10287911 -0.20301069] [[ 0.82385354  0.09841182 -0.16741293  0.23489158]
 [ 0.09841182  1.11553868 -0.01049593  0.03889927]
 [-0.16741293 -0.01049593  1.44118504 -0.40464576]
 [ 0.23489158  0.03889927 -0.40464576  1.69690935]]
symmetric [5.8727657 6.4094085] [8.234623262059451e-05, 0.00010347376793238717, 0.00010842333936901893] max 0.0002729036217975489
walls [5.869951  5.8856356 6.3599141 6.4497619] [3.268327867400244e-05, 4.3186582531437485e-05, 3.5816959877616116e-05] max 0.00015688874293866978
residual shift/bank rms [[3.80150564e-05 3.72942529e-05 2.46406482e-05 4.00334205e-05]
 [2.09778321e-05 4.27871577e-05 1.84712812e-05 2.05949370e-05]
 [1.65673240e-05 5.96666805e-05 2.54209307e-05 1.59861092e-05]]
largest residuals [(2, [2, 6], [4.23e-05, 4.46e-05]), (1, [0, 4], [-1.79e-05, -5.22e-05]), (0, [2, 2], [-3.54e-05, -5.67e-05]), (0, [2, 4], [-3.21e-05, -5.88e-05]), (1, [0, 2], [-2.84e-05, -6.16e-05]), (0, [0, 0], [-4.09e-05, -6.48e-05]), (0, [0, 8], [8.8e-05, 8.9e-06]), (2, [0, 6], [6.94e-05, 9.2e-05]), (0, [0, 2], [-6.77e-05, -0.0001172]), (0, [0, 4], [-6.96e-05, -0.0001266]), (1, [0, 8], [0.0001334, 7.65e-05]), (2, [0, 8], [0.0001569, 0.0001227])]
walls+slopes [ 5.8646438e+00  5.8834484e+00  6.3640632e+00  6.4458335e+00
  5.0937100e-02 -1.4969000e-03  3.4827000e-03 -8.3550000e-04] [2.2305367793801166e-05, 3.163029936912759e-05, 2.6940970998684614e-05] max 0.0001044491621544491
walls+corner [ 5.8716099  5.8879446  6.3625276  6.4506585 -0.0946083] [3.196446096332887e-05, 4.2208707079011464e-05, 3.551038260473506e-05] max 0.00015720754236293877
walls+rosette offset [ 5.8646977e+00  5.8737844e+00  6.3569235e+00  6.4472853e+00
 -9.3000000e-06 -3.2100000e-05 -2.2800000e-05 -1.4300000e-05] [3.070661305812161e-05, 4.207424425960796e-05, 3.290917580483742e-05] max 0.0001427846685123288
walls+nuisance [ 5.8699833e+00  5.8852876e+00  6.3596420e+00  6.4494405e+00
 -3.2000000e-06  1.8000000e-06 -2.6000000e-06] [3.239780879883649e-05, 4.313820711170803e-05, 3.53989555858968e-05] max 0.0001607282416998118
walls+curvature [ 5.8683421e+00  5.8742785e+00  6.3684354e+00  6.4333386e+00
 -1.8005000e-03  4.2896900e-02 -3.8757000e-02  1.0574970e-01] [3.1121856615342884e-05, 4.167724754566699e-05, 3.514513661446064e-05] max 0.0001481995376266601
```

### Turn 6 — `execute_in_remote_docker` (`26324 ms` LLM)
- **Rationale**: The asset index may specify what blind lineage evidence is available. Shift and bank holdouts test whether the apparent left-wall pattern predicts unsampled observations or merely absorbs residuals in the diagnostic pulses.
- **Alternatives Weighed**: `Fit variation on all four walls` (rejected: The largest residuals cluster on one wall. Test whether a localized correction transfers across shifts and banks before fitting more wall coefficients.)

```bash
cat /holdout_task/public_asset_index.json; cat >/tmp/probe_boundary.py <<'PY'
from probe_transport import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; E,idx=setup(s); A=build(s,1,0,1); mu=s['mu_shifts']; P=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in P]); B=s['source_modes'].reshape(len(W),-1,order='F').T@W; Y=np.column_stack([p['y_pulse_raw'] for p in P]); base=np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2); X,J=np.indices((nx,ny)); x=X.ravel(order='F')/(nx-1); y=J.ravel(order='F')/(ny-1); rms=lambda x:np.sqrt(np.mean(x*x)); nr=len(idx)
left=E[:,0]; masks=[np.ones(nr*len(mu),dtype=bool)]+[np.arange(nr*len(mu))//nr!=k for k in range(len(mu))]+[np.arange(nr*len(mu))%nr%4!=k for k in range(4)]
for name,F in [('walls',E),('left_linear',np.c_[E,left*(2*y-1)]),('left_quad',np.c_[E,left*(2*y-1),left*(2*y-1)**2]),('left_cubic',np.c_[E,left*(2*y-1),left*(2*y-1)**2,left*(2*y-1)**3]),('left_fourier',np.c_[E,left*np.sin(2*np.pi*y),left*np.cos(2*np.pi*y)]),('left_nodes',np.c_[E,*[left*(J.ravel(order='F')==j) for j in range(1,ny)]])]:
 b0=np.r_[base,np.zeros(F.shape[1]-4)]; f=lambda b,p:pred(A,F,idx,mu,B[:,p].copy(),b); fit=lambda p,mask:least_squares(lambda b:(f(b,p)-Y[:,p])[mask].ravel(),b0,gtol=1e-10,max_nfev=25).x
 b=fit([0,1],masks[0]); print(name,'fit',rms(f(b,[0,1])-Y),'LOPO',max(rms(f(fit([p],masks[0]),[1-p])-Y[:,[1-p]]) for p in [0,1]),'worst shift/bank',[max(rms((f(fit([0,1],m),[0,1])-Y)[~m]) for m in mm) for mm in [masks[1:4],masks[4:]]], 'coef',np.round(b,5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_boundary.py
```
- **Container Exit Code**: `0` (`6336 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"sensor_ij": {
              "dtype": "int64",
              "shape": [
                32,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                8,
                14,
                12
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 1.4,
            "u_box": 0.42,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                96
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1064
    }
  },
  "schema": "hegel_public_asset_index_v1"
}walls fit 3.268327867400244e-05 LOPO 4.3186582531437485e-05 worst shift/bank [5.9047679223451774e-05, 6.395986528216478e-05] coef [5.86995 5.88564 6.35991 6.44976]
left_linear fit 2.2342729758004297e-05 LOPO 3.076655339935553e-05 worst shift/bank [4.986697991629472e-05, 3.407426102645455e-05] coef [5.86425 5.88383 6.36491 6.44568 0.0525 ]
left_quad fit 2.2336702873889463e-05 LOPO 3.136906518537622e-05 worst shift/bank [4.991133869893733e-05, 3.4036769709460825e-05] coef [5.86363e+00 5.88391e+00 6.36483e+00 6.44557e+00 5.26100e-02 3.54000e-03]
left_cubic fit 2.2220663276056678e-05 LOPO 3.170037305371669e-05 worst shift/bank [4.9637937766808454e-05, 3.381623270007853e-05] coef [ 5.86502  5.8839   6.36476  6.44648  0.07141 -0.00885 -0.0489 ]
left_fourier fit 2.2392984137514376e-05 LOPO 3.2348289302615964e-05 worst shift/bank [4.923372365919524e-05, 3.3991302717399955e-05] coef [ 5.86057e+00  5.88404e+00  6.36447e+00  6.44731e+00 -3.19300e-02
 -5.69000e-03]
left_nodes fit 1.541147627251597e-05 LOPO 4.6834654152159184e-05 worst shift/bank [3.114699851246432e-05, 0.002344707882481545] coef [ 5.896240e+00  5.882530e+00  6.365970e+00  6.456430e+00 -1.853000e-01
  3.552000e-02 -1.846500e-01 -5.059000e-02  3.703000e-02 -4.930000e-02
  2.507000e-02  8.046500e-01 -5.778700e+00  3.351488e+01 -1.632515e+01]
```

### Turn 7 — `execute_in_remote_docker` (`77455 ms` LLM)
- **Rationale**: Build the solver now, then withhold the public tracers to see whether routine observations recover a comparable terminal field and source solution. A large discrepancy would motivate refining the blind boundary fit before certifying the artifact.
- **Alternatives Weighed**: `Fit leakage separately at every left-wall node` (rejected: Worst bank-holdout RMSE rises to 2.34e-3 despite the better in-sample fit.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated chamber inversion and box-simplex optimality certificate."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _features(x, level):
    cols = [np.ones(len(x)), *x.T]
    if level >= 1:
        cols.extend((x*x).T)
        cols.extend((x**3).T)
    if level >= 2:
        cols.extend(x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4))
    if level >= 3:
        cols.extend(x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i != j)
        cols.extend(x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4))
    return np.column_stack(cols)


def _calibration(sweeps):
    x, y = [np.asarray(sweeps[k],float) for k in ('ref_rosette','raw_rosette')]
    # Use the paired rosettes to estimate crosstalk. Stratify the validation
    # folds so that every fit sees single-channel as well as mixed inputs.
    order = np.argsort(np.count_nonzero(x,axis=1),kind='stable')
    folds = np.empty(len(x),int)
    folds[order] = np.arange(len(x)) % 5
    designs = [_features(x,k) for k in range(4)]
    scores = []
    for f in designs:
        scores.append(max(np.sqrt(np.mean((f[folds==j] @ np.linalg.lstsq(f[folds!=j],(y-x)[folds!=j],rcond=None)[0] - (y-x)[folds==j])**2)) for j in range(5)))
    level = next(i for i,v in enumerate(scores) if v <= min(scores)*1.12)
    coef = np.linalg.lstsq(designs[level],y-x,rcond=None)[0]
    affine = (np.eye(4)+coef[1:5]).T

    def invert(raw):
        y = np.asarray(raw).reshape(-1,4)
        x = np.linalg.solve(affine,(y-coef[0]).T).T
        if level:
            for _ in range(12):
                r = x + _features(x,level)@coef - y
                if np.max(np.abs(r)) < 1e-12:
                    break
                jac = np.broadcast_to(affine,(len(x),4,4)).copy()
                for i in range(4):
                    d = np.zeros(4); d[i] = 1e-5
                    jac[:,:,i] = np.eye(4)[:,i] + ((_features(x+d,level)-_features(x-d,level))@coef)/(2e-5)
                x -= np.linalg.solve(jac,r[:,:,None])[:,:,0]
        return x.ravel()
    return invert


def _axis(k,v,h):
    k,v = np.asarray(k),np.asarray(v)
    d = (k[:-1]+k[1:])/(2*h*h)
    lower = -d-np.maximum(v[1:],0)/h
    upper = -d-np.maximum(-v[:-1],0)/h
    return diags([lower,np.r_[-upper,0]+np.r_[0,-lower],upper],[-1,0,1],format='csc')


class _Chamber:
    def __init__(self,s):
        nx,ny = self.shape = s['nx'],s['ny']
        hx,hy = s['Lx']/(nx-1),s['Ly']/(ny-1)
        ax,ay = [_axis(s['kappa_'+k],s['vel_'+k],h) for k,h in zip('xy',(hx,hy))]
        self.A = csc_matrix(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(np.asarray(s['V_reaction_2d']).ravel(order='F')))
        self.I = eye(nx*ny,format='csc')
        x,y = np.indices(self.shape)
        self.edges = np.column_stack([a.ravel(order='F') for a in (x==0,x==nx-1,y==0,y==ny-1)]).astype(float)
        self.slopes = self.edges * np.column_stack([a.ravel(order='F') for a in (2*y/(ny-1)-1,2*y/(ny-1)-1,2*x/(nx-1)-1,2*x/(nx-1)-1)])
        self.nominal = np.repeat([s['robin_x']/hx,s['robin_y']/hy],2)
        ij = np.asarray(s['sensor_ij']); self.sensors = ij[:,0]+nx*ij[:,1]
        self.shifts = np.asarray(s['mu_shifts'])
        self.rhs = np.asarray(s['source_modes']).reshape(-1,nx*ny,order='F').T.copy()

    def response(self,delta,rhs):
        A = self.A + diags(delta,format='csc')
        observations = []
        for shift in self.shifts:
            field = splu(A+shift*self.I).solve(rhs)
            observations.append(field[self.sensors])
        return np.concatenate(observations),field


def _fit_boundary(c,s,y,cal):
    E = c.edges
    pulses = s.get('reference_tracer_pulses')
    if pulses:
        # Diagnostic mixtures provide a direct boundary measurement, with
        # independent walls and the spatial mode indicated by pulse holdouts.
        F = np.column_stack((E,c.slopes[:,0]))
        W = np.column_stack([p['w_pulse'] for p in pulses])
        R = c.rhs@W
        Y = np.column_stack([cal(p['y_pulse_raw']) for p in pulses])
        b = least_squares(lambda b:(c.response(F@b,R)[0]-Y).ravel(),np.r_[c.nominal,0.],gtol=1e-11).x
        return F@b

    # Profile concentrations and nuisance out of the physical fit. The
    # source ridge is used in the final QP, not to infer chamber leakage.
    R = c.rhs
    Z = np.asarray(s['Z_nuisance'])
    m = R.shape[1]
    w0 = np.full(m,s['total_budget']/m)
    Q = np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]
    def residual(b):
        S = c.response(E@b,R)[0]
        D = np.column_stack((S@Q,Z))
        r = y-S@w0
        return D@np.linalg.lstsq(D,r,rcond=None)[0]-r
    b0 = c.nominal
    scale = np.maximum(abs(b0),1)
    b = least_squares(residual,b0,bounds=(b0-0.75*scale,b0+1.25*scale),gtol=1e-11).x
    return E@b


def _quadratic(A,r,ridge,budget,cap):
    m = A.shape[1]
    H = A.T@A+ridge*np.eye(m)
    g = -A.T@r
    cap = np.broadcast_to(np.asarray(cap,float),(m,))
    objective = lambda w: 0.5*np.linalg.norm(A@w-r)**2+0.5*ridge*(w@w)
    def certificate(w,nu):
        q = H@w+g+nu
        lo = np.where(w==0,np.maximum(q,0),0)
        hi = np.where(w==cap,np.maximum(-q,0),0)
        d = q-lo+hi
        p = objective(w)
        lower = p + nu*(w.sum()-budget)-lo@w+hi@(w-cap)-0.5*d@np.linalg.solve(H,d)
        return dict(weights=w,nu_sum=float(nu),lam_lower=lo,lam_upper=hi,primal_objective=float(p),dual_lower_bound=float(lower))

    if budget == 0 or budget == cap.sum():
        w = np.zeros(m) if budget == 0 else cap.copy()
        grad = H@w+g
        return certificate(w,-grad.min() if budget==0 else -grad.max())
    opt = minimize(lambda w:0.5*w@H@w+g@w,cap*(budget/cap.sum()),jac=lambda w:H@w+g,
                   bounds=list(zip(np.zeros(m),cap)),constraints={'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(m)},
                   method='SLSQP',options={'ftol':1e-13,'maxiter':300})
    # Snap approximate active bounds and solve the equality-constrained
    # KKT system on the free coordinates; release an incorrect bound.
    side = np.where(opt.x < 1e-8,-1,np.where(cap-opt.x < 1e-8,1,0))
    for _ in range(4*m+4):
        free = side==0
        w = np.where(side==1,cap,0.)
        if not free.any():
            rest = budget-w.sum(); grad = H@w+g
            down = np.flatnonzero(side==1); up = np.flatnonzero(side==-1)
            i = up[np.argmin(grad[up])] if len(up) else -1
            j = down[np.argmax(grad[down])] if len(down) else -1
            if abs(rest)<1e-12 and i>=0 and j>=0 and grad[i]>=grad[j]-1e-12:
                return certificate(w,-0.5*(grad[i]+grad[j]))
            if rest>=-1e-12 and i>=0: side[i]=0
            if rest<=1e-12 and j>=0: side[j]=0
            continue
        hf = H[np.ix_(free,free)]
        a,b = np.linalg.solve(hf,np.column_stack((-(H@w+g)[free],np.ones(free.sum())))).T
        nu = (a.sum()+w.sum()-budget)/b.sum()
        w[free] = a-nu*b
        invalid = np.where(free,np.maximum(-w,w-cap),-np.inf)
        i = invalid.argmax()
        if invalid[i]>1e-12:
            side[i] = -1 if w[i]<0 else 1
            continue
        q = H@w+g+nu
        violation = np.where(side!=0,side*q,-np.inf)
        i = violation.argmax()
        if violation[i] > 1e-11:
            side[i]=0
            continue
        return certificate(np.clip(w,0,cap),nu)
    raise RuntimeError('Box-simplex KKT refinement did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    c = _Chamber(s)
    calibrate = _calibration(s['sensor_calibration_sweeps'])
    y = calibrate(s['y_obs_raw'])
    diag = _fit_boundary(c,s,y,calibrate)
    S,U = c.response(diag,c.rhs)
    Z = np.asarray(s['Z_nuisance'])
    Pz = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    A,r = S-Z@(Pz@S),y-Z@(Pz@y)
    cert = _quadratic(A,r,s['source_ridge'],s['total_budget'],s['u_box'])
    w = cert['weights']
    return dict(weights=w,terminal_field=(U@w).reshape(c.shape,order='F'),kkt_certificate=cert)
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import batch_solver as b, numpy as np, pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=b._Chamber(s); cal=b._calibration(s['sensor_calibration_sweeps']); y=cal(s['y_obs_raw']); d=b._fit_boundary(c,s,y,cal); a=b.solve_batch(s); s['reference_tracer_pulses']=None; db=b._fit_boundary(c,s,y,cal); bb=b.solve_batch(s); print('diag diagnostic/blind at wall maxima',d.max(),db.max(),'wall deltas',np.linalg.lstsq(np.c_[c.edges,c.slopes[:,0]],d,rcond=None)[0],np.linalg.lstsq(c.edges,db,rcond=None)[0]); print('diagnostic/blind weights',a['weights'],bb['weights']); print('relative field difference',np.linalg.norm(a['terminal_field']-bb['terminal_field'])/np.linalg.norm(a['terminal_field'])); print('KKT certificate',a['kkt_certificate'])
PY
ls -ld /tmp/hegel_continual_archive /tmp/hegel_continual_archive/* 2>/dev/null || true
```
- **Container Exit Code**: `0` (`824 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
diag diagnostic/blind at wall maxima 12.36285666381421 17.748752897935496 wall deltas [5.86450797 5.884342   6.36473318 6.44557229 0.0527764 ] [ 7.58951839  4.80846595  4.69668346 10.15923451]
diagnostic/blind weights [0.09519027 0.12620994 0.13952421 0.10556273 0.29807971 0.08123842
 0.33900416 0.21519055] [0.11515217 0.13201952 0.14740704 0.09617211 0.25630532 0.08902915
 0.32538185 0.23853283]
relative field difference 0.038810506785699944
KKT certificate {'weights': array([0.09519027, 0.12620994, 0.13952421, 0.10556273, 0.29807971,
       0.08123842, 0.33900416, 0.21519055]), 'nu_sum': -0.020694869046285966, 'lam_lower': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'lam_upper': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'primal_objective': 0.010170858460568026, 'dual_lower_bound': 0.010170858460568026}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cc9ff5f86c511bb5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cc9ff5f8 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`32646 ms` LLM)
- **Rationale**: Compare blind wall fits on held-out shifts and against the tracer-based field. Inspect routine residuals and fitted source weights to distinguish weak boundary identification from errors in the observation model.
- **Alternatives Weighed**: `Use nominal boundaries on all blind campaigns` (rejected: Nominal boundaries miss the public tracers; test whether the routine observations can support a smaller, reproducible correction.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,batch_solver as b
from scipy.optimize import least_squares
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=b._Chamber(s); cal=b._calibration(s['sensor_calibration_sweeps']); y=cal(s['y_obs_raw']); E=c.edges; Z=s['Z_nuisance']; m=c.rhs.shape[1]; w0=np.full(m,s['total_budget']/m); Q=np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]; nr=len(c.sensors); rms=lambda x:np.sqrt(np.mean(x*x)); d=b._fit_boundary(c,s,y,cal); target=c.response(d,c.rhs)[0]; zsolve=lambda S,y: b._quadratic(S-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@S),y-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@y),s['source_ridge'],s['total_budget'],s['u_box'])['weights']; wt=zsolve(target,y); Ut=c.response(d,c.rhs)[1]@wt
print('shifts',c.shifts,'budget cap',s['total_budget'],s['u_box']); print('nominal versus diagnostic',c.nominal,np.linalg.lstsq(E,d,rcond=None)[0]);
def fit(S,mask):
 D=np.column_stack((S@Q,Z)); r=y-S@w0; a=np.linalg.lstsq(D[mask],r[mask],rcond=None)[0]; return D@a-r,w0+Q@a[:m-1]
for key,S in [('nominal',c.response(E@c.nominal,c.rhs)[0]),('diagnostic',target)]:
 r,w=fit(S,np.ones(len(y),bool)); print(key,'observation residual RMS and max',rms(r),np.max(abs(r)), 'physical weights',w.round(4)); print('residual shift RMS',np.sqrt((r.reshape(-1,nr)**2).mean(1)), 'largest',np.sort(abs(r))[-10:]); print('largest indices',np.argsort(abs(r))[-10:]);
for case,F in [('axes',E@np.array([[1,0],[1,0],[0,1],[0,1]])),('walls',E)]:
 for penalty in [0,0.00015,0.0005,0.0015,0.005,0.015]:
  def solve(mask):
   def fun(p):
    S=c.response(E@c.nominal+F@p,c.rhs)[0]; return np.r_[fit(S,mask)[0][mask],penalty*p]
   return least_squares(fun,np.zeros(F.shape[1]),bounds=(-4,4),gtol=1e-10).x
  p=solve(np.ones(len(y),bool)); S,U=c.response(E@c.nominal+F@p,c.rhs); w=zsolve(S,y); preds=[]
  for j in range(len(c.shifts)):
   mask=np.arange(len(y))//nr!=j; q=solve(mask); Sq=c.response(E@c.nominal+F@q,c.rhs)[0]; preds.append(rms(fit(Sq,mask)[0][~mask]));
  print(case,penalty,p.round(4),'train/holdout',rms(fit(S,np.ones(len(y),bool))[0]),max(preds),'field vs diag',np.linalg.norm(U@w-Ut)/np.linalg.norm(Ut),'weights',np.linalg.norm(w-wt))
PY
if [ -d /tmp/hegel_continual_archive ]; then ls -lah /tmp/hegel_continual_archive; if [ -f /tmp/hegel_continual_archive/archive_apparatus_log.json ]; then head -c 5000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; else echo 'No continual archive mounted'; fi
```
- **Container Exit Code**: `0` (`1857 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
59735041878 0.024159768864237915 physical weights [-0.0022 -0.8518  1.1815 -0.026   0.3623  0.1826  0.492   0.0616]
residual shift RMS [0.0073129  0.00895474 0.00743999] largest [0.0126735  0.01381481 0.01396387 0.01426828 0.01465826 0.01541532
 0.01617529 0.01678382 0.01965754 0.02415977]
largest indices [71 77  5 48 66 43 83 59 54 27]
axes 0 [ 1.2171 -0.3378] train/holdout 0.007877311883710363 0.010036324192766153 field vs diag 0.01121169901247399 weights 0.020987990483630047
axes 0.00015 [ 1.2165 -0.3373] train/holdout 0.007877311896423772 0.010036230828900388 field vs diag 0.011206105980634746 weights 0.020975261341818253
axes 0.0005 [ 1.2109 -0.3331] train/holdout 0.007877313370025877 0.01003529930151442 field vs diag 0.011152040333140217 weights 0.020851952260594615
axes 0.0015 [ 1.1648 -0.2982] train/holdout 0.007877419885253853 0.010028042600172675 field vs diag 0.010711375191197895 weights 0.019831541146278817
axes 0.005 [ 0.854  -0.0675] train/holdout 0.00788263960132517 0.009996277119429534 field vs diag 0.008116412678783208 weights 0.013006795637426702
axes 0.015 [0.3743 0.15  ] train/holdout 0.007910837530208716 0.010003876487926937 field vs diag 0.0041260626076244766 weights 0.003271478911955571
walls 0 [ 1.9678 -0.8133 -1.698   3.7648] train/holdout 0.007738745104865691 0.010084028775883463 field vs diag 0.03881154632631103 weights 0.055847542343488474
walls 0.00015 [ 1.9684 -0.809  -1.6945  3.7476] train/holdout 0.007738746219388405 0.010083802374789059 field vs diag 0.03868730374972313 weights 0.05567882922356363
walls 0.0005 [ 1.9728 -0.7686 -1.6612  3.5861] train/holdout 0.007738866658323633 0.010081574671047736 field vs diag 0.03750765715509667 weights 0.05407312930418612
walls 0.0015 [ 1.939  -0.5272 -1.4396  2.7094] train/holdout 0.007743759062025015 0.01006542191507025 field vs diag 0.030624100919257626 weights 0.04455638373117723
walls 0.005 [ 1.2346  0.0496 -0.5715  0.9841] train/holdout 0.007804196078771965 0.010010633163841539 field vs diag 0.013418192062812497 weights 0.018873607861118528
walls 0.015 [0.3612 0.1521 0.0103 0.2701] train/holdout 0.007909417072658766 0.010021930327805743 field vs diag 0.0032266406421036526 weights 0.003478475110471839
No continual archive mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cc9ff5f86c511bb5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cc9ff5f8 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`33415 ms` LLM)
- **Rationale**: The public routine residual is roughly two orders of magnitude larger than tracer residuals, so an unconstrained fit can mistake observation noise for leakage. Compare the nominal fit and several regularization strengths on the real pulse-withheld case and on controlled leakage changes using its measured residual pattern.
- **Alternatives Weighed**: `Select the penalty solely by agreement with public tracers` (rejected: That could suppress genuine larger leakage changes on routine campaigns; compare clean and noisy perturbations before choosing an estimator.)

```bash
cat >/tmp/blind_slices.py <<'PY'
import sys, pickle, itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
import batch_solver as b
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._Chamber(s); cal=b._calibration(s['sensor_calibration_sweeps']); y=cal(s['y_obs_raw']); E=c.edges; Z=s['Z_nuisance']; m=c.rhs.shape[1]; w0=np.full(m,s['total_budget']/m); Q=np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]; base=E@c.nominal; diag=b._fit_boundary(c,s,y,cal); S0,U0=c.response(diag,c.rhs); ridge=s['nuisance_ridge']; Pz=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T); fit=lambda S,v: b._quadratic(S-Z@(Pz@S),v-Z@(Pz@v),s['source_ridge'],s['total_budget'],s['u_box'])['weights']; rms=lambda v: np.sqrt(np.mean(v*v));
D0=np.c_[S0@Q,Z]; rr=y-S0@w0; noise=rr-D0@np.linalg.lstsq(D0,rr,rcond=None)[0];
cases=[('public',diag,y)]
patterns=[[0,0,0,0],[.7,.7,-.5,-.5],[.7,-.7,-.5,.5],[1.4,-.8,.4,1.0],[-.3,-.65,.75,-.65]]
for i,p in enumerate(patterns):
 d=base+E@p+c.slopes@(np.array([.20,-.07,.15,0.]) if i%2 else np.zeros(4)); S,_=c.response(d,c.rhs)
 for no in [0,1]:
  w=np.full(m,s['total_budget']/m) if i%2 else np.array([.42,.24,0,.03,.42,.14,.08,.07]); w*=s['total_budget']/w.sum(); cases.append((f'{i}/noise{no}',d,S@w+Z@np.array([.005,-.01,.015])+no*noise))
for label,basis in [('axes',E@np.repeat(np.eye(2),2,axis=0)),('walls',E),('walls+slopes',np.c_[E,c.slopes])]:
 for pen in [.0015,.005,.010,.015,.025,np.inf]:
  vals=[]
  for tag,d,v in cases:
   def fun(p):
    S,_=c.response(base+basis@p,c.rhs); D=np.c_[S@Q,Z]; r=v-S@w0
    return np.r_[r-D@np.linalg.lstsq(D,r,rcond=None)[0],pen*p]
   p=np.zeros(basis.shape[1]) if not np.isfinite(pen) else least_squares(fun,np.zeros(basis.shape[1]),bounds=(-2.5,3),gtol=1e-10).x
   S,U=c.response(base+basis@p,c.rhs); St,Ut=c.response(d,c.rhs); w,wt=fit(S,v),fit(St,v); vals.append((np.linalg.norm(U@w-Ut@wt)/np.linalg.norm(Ut@wt),np.linalg.norm(w-wt),tag))
  print(label,pen,'public %.4g/%.4g'%vals[0][:2], 'worst clean',max(vals[1::2]),'worst noisy',max(vals[2::2]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/blind_slices.py
```
- **Container Exit Code**: `0` (`6087 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t clean (0.02088032718031768, 0.02484964402631591, '3/noise0') worst noisy (0.020965330431647333, 0.025455447011222283, '3/noise1')
walls 0.0015 public 0.03062/0.04456 worst clean (0.0020508950390274896, 0.002947850066264646, '3/noise0') worst noisy (0.03270437165493805, 0.04910079309992953, '4/noise1')
walls 0.005 public 0.01342/0.01887 worst clean (0.00831993217529456, 0.011810306982403517, '3/noise0') worst noisy (0.020109904121893743, 0.030747059605919037, '4/noise1')
walls 0.01 public 0.006124/0.007066 worst clean (0.01282684630843404, 0.01742605399977552, '3/noise0') worst noisy (0.015462430298998152, 0.023678257055868634, '4/noise1')
walls 0.015 public 0.003227/0.003478 worst clean (0.015247953497199762, 0.0199297960338991, '3/noise0') worst noisy (0.014202408362476476, 0.02165051888526841, '4/noise1')
walls 0.025 public 0.00194/0.003466 worst clean (0.017871363432315508, 0.022312545531190785, '3/noise0') worst noisy (0.015936240065656422, 0.020942184250497386, '3/noise1')
walls inf public 0.004487/0.006167 worst clean (0.02088032718031768, 0.02484964402631591, '3/noise0') worst noisy (0.020965330431647333, 0.025455447011222283, '3/noise1')
walls+slopes 0.0015 public 0.03315/0.04574 worst clean (0.0021210608934805145, 0.0031184948411309065, '3/noise0') worst noisy (0.04126051286524674, 0.061651653116802124, '4/noise1')
walls+slopes 0.005 public 0.01533/0.02202 worst clean (0.008401789641460274, 0.012023178904745862, '3/noise0') worst noisy (0.0219948980213789, 0.03392286268127462, '4/noise1')
walls+slopes 0.01 public 0.006541/0.008109 worst clean (0.012836786241344758, 0.017495517088043838, '3/noise0') worst noisy (0.01584087321566893, 0.024988046002752336, '4/noise1')
walls+slopes 0.015 public 0.003368/0.003746 worst clean (0.015246284871492826, 0.01996715202617735, '3/noise0') worst noisy (0.01426951611170043, 0.022244368666514486, '4/noise1')
walls+slopes 0.025 public 0.001999/0.003352 worst clean (0.017864882900274274, 0.02232860670023237, '3/noise0') worst noisy (0.015916099159970797, 0.020891316184117856, '3/noise1')
walls+slopes inf public 0.004487/0.006167 worst clean (0.02088032718031768, 0.02484964402631591, '3/noise0') worst noisy (0.020965330431647333, 0.025455447011222283, '3/noise1')

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cc9ff5f86c511bb5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cc9ff5f8 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`24744 ms` LLM)
- **Rationale**: Compare residual-scaled shrinkage with and without enforcing the source box in the physical fit. Adopt the source constraint only if it improves field recovery without degrading the public pulse-withheld case.
- **Alternatives Weighed**: `Use the same absolute penalty on every campaign` (rejected: The clean and noisy slices favor different penalty strengths; a fixed strong penalty suppresses recoverable boundary changes.)

```bash
cat >/tmp/adaptive_blind.py <<'PY'
exec(open('/tmp/blind_slices.py').read().split('for label,basis')[0])
from scipy.optimize import minimize
for constrained in [False,True]:
 for ratio in [1.,1.8,2.5]:
  vals=[]
  for tag,d,v in cases:
   def profile(S):
    D=np.c_[S@Q,Z]; r=v-S@w0
    a=np.linalg.lstsq(D,r,rcond=None)[0]
    if not constrained: return r-D@a
    P=np.linalg.pinv(Z); A=S-Z@(P@S); t=v-Z@(P@v)
    w=b._quadratic(A,t,1e-9,s['total_budget'],s['u_box'])['weights']
    return t-A@w
   F=np.c_[E,c.slopes]; r0=profile(c.response(base,c.rhs)[0]); scale=max(3e-5,ratio*rms(r0)); penalties=scale*np.r_[np.ones(4),np.full(4,2.5)]
   def fun(p):
    S,_=c.response(base+F@p,c.rhs); return np.r_[profile(S),penalties*p]
   p=least_squares(fun,np.zeros(F.shape[1]),bounds=(-2.5,3),gtol=1e-10).x
   S,U=c.response(base+F@p,c.rhs); St,Ut=c.response(d,c.rhs); w,wt=fit(S,v),fit(St,v)
   vals.append((np.linalg.norm(U@w-Ut@wt)/np.linalg.norm(Ut@wt),np.linalg.norm(w-wt),tag))
  print(constrained,ratio,'public',vals[0], 'worst clean',max(vals[1::2]),'worst noisy',max(vals[2::2]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/adaptive_blind.py
```
- **Container Exit Code**: `0` (`7100 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
False 1.0 public (0.008175488236310003, 0.010405383914617476, 'public') worst clean (0.0022078089462415963, 0.0032444533071867947, '3/noise0') worst noisy (0.016716741149908846, 0.02587772440798038, '4/noise1')
False 1.8 public (0.0034908752200954155, 0.003751631274111417, 'public') worst clean (0.0048040392892337615, 0.006989212806844816, '3/noise0') worst noisy (0.0143600469682662, 0.021989634742730198, '4/noise1')
False 2.5 public (0.0019901725478898755, 0.0029393428198054247, 'public') worst clean (0.0068399310304474974, 0.009825012134345935, '3/noise0') worst noisy (0.014587141711875189, 0.019704235876769602, '3/noise1')
True 1.0 public (0.010977341639844707, 0.012840685297912904, 'public') worst clean (0.0022724923782054137, 0.003339878879645357, '3/noise0') worst noisy (0.016709542201511395, 0.02581424115630064, '4/noise1')
True 1.8 public (0.004895489359586439, 0.0060640201001858594, 'public') worst clean (0.004919366418016584, 0.007152026890034858, '3/noise0') worst noisy (0.014360067798081242, 0.021989916796059227, '4/noise1')
True 2.5 public (0.002967982020571898, 0.0047957981240618775, 'public') worst clean (0.006973605094523482, 0.010008306137779057, '3/noise0') worst noisy (0.014524180694468897, 0.01951599480556795, '3/noise1')

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cc9ff5f86c511bb5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cc9ff5f8 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`27751 ms` LLM)
- **Rationale**: Promote the adaptive estimator, then test the actual artifact rather than the probe implementation. If its dimension-scaled bounds change the stress results materially, adjust them before final verification.
- **Alternatives Weighed**: `Use a fixed penalty of 0.015` (rejected: It reduces public pulse-withheld error to 0.323%, but its worst clean-slice field error is 1.52%, versus 0.684% for the adaptive challenger.); `Enforce the box in the preliminary leakage fit` (rejected: It did not materially improve the noisy stress slices and increased public pulse-withheld error; the final certified QP still enforces the box.)

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/batch_solver.py')
s=p.read_text()
a=s.index('    # Profile concentrations')
z=s.index('\n\ndef _quadratic',a)
s=s[:a]+'''    # Profile the budget equality and nuisance before estimating leakage.
    # Noisy observations cannot support the same correction as clean data.
    F = np.column_stack((E,c.slopes))
    R = c.rhs
    Z = np.asarray(s['Z_nuisance'])
    m = R.shape[1]
    w0 = np.full(m,s['total_budget']/m)
    Q = np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]
    nominal = E@c.nominal

    def profile(delta):
        S = c.response(delta,R)[0]
        D = np.column_stack((S@Q,Z))
        r = y-S@w0
        return r-D@np.linalg.lstsq(D,r,rcond=None)[0]

    scale = max(3e-5,2.5*np.sqrt(np.mean(profile(nominal)**2)))
    penalties = scale*np.repeat([1.,2.5],4)
    limit = np.maximum(abs(c.nominal),1.)
    span = np.r_[limit,limit*0.5]
    opt = least_squares(lambda p:np.r_[profile(nominal+F@p),penalties*p],
                        np.zeros(8),bounds=(-0.45*span,0.6*span),gtol=1e-11)
    return nominal+F@opt.x
'''+s[z:]
p.write_text(s)
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
exec(open('/tmp/blind_slices.py').read().split('for label,basis')[0])
rows=[]
for tag,d,v in cases:
    cal_id=lambda k:k
    fitted=b._fit_boundary(c,dict(s,reference_tracer_pulses=None),v,cal_id)
    S,U=c.response(fitted,c.rhs); St,Ut=c.response(d,c.rhs)
    w,wt=fit(S,v),fit(St,v)
    rows.append((np.linalg.norm(U@w-Ut@wt)/np.linalg.norm(Ut@wt),np.linalg.norm(w-wt),tag))
print('Actual solver: public, worst clean, worst noisy:',rows[0],max(rows[1::2]),max(rows[2::2]))
PY
python3 - <<'PY'
import json
p=json.load(open('/holdout_task/public_asset_index.json'))
print('asset index metadata:', {k:v for k,v in p.items() if k!='files'})
PY
```
- **Container Exit Code**: `0` (`1400 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Actual solver: public, worst clean, worst noisy: (0.0019901135842927214, 0.0029393478637728416, 'public') (0.006839930987069162, 0.009825011415027427, '3/noise0') (0.014587143051740059, 0.01970424331015378, '3/noise1')
asset index metadata: {'schema': 'hegel_public_asset_index_v1'}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=78a1f1956420a545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_78a1f195 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`27103 ms` LLM)
- **Rationale**: Test the exact artifact against its symmetric-wall ablation on unseen tracer mixtures. Independently stress the certificate near budget endpoints and with zero-cap coordinates, where the active-set refinement could otherwise pass the public schema while returning invalid multipliers.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=b._Chamber(s); cal=b._calibration(s['sensor_calibration_sweeps']); y=cal(s['y_obs_raw']); p=s['reference_tracer_pulses']; R=c.rhs@np.column_stack([q['w_pulse'] for q in p]); Y=np.column_stack([cal(q['y_pulse_raw']) for q in p]); E=c.edges@np.repeat(np.eye(2),2,axis=0); errs=[]
for j in range(len(p)):
    keep=np.arange(len(p))!=j
    bs=least_squares(lambda a:(c.response(E@a,R[:,keep])[0]-Y[:,keep]).ravel(),c.nominal[::2],gtol=1e-11).x
    dc=b._fit_boundary(c,dict(s,reference_tracer_pulses=[q for i,q in enumerate(p) if i!=j]),y,cal)
    errs.append([np.sqrt(np.mean((c.response(d,R[:,j])[0]-Y[:,j])**2)) for d in (E@bs,dc)])
print('Leave-one-tracer-out worst RMSE: symmetric ablation, delivered independent-wall fit:',np.max(errs,axis=0)); assert np.max(errs,axis=0)[1]<np.max(errs,axis=0)[0]
rng=np.random.default_rng(549); worst=np.zeros(5)
for k in range(450):
    m=int(rng.integers(2,40)); Ac=rng.normal(size=(int(rng.integers(1,60)),m))*10.**rng.uniform(-2,1); r=rng.normal(size=len(Ac)); cap=rng.uniform(.05,1,m); cap[rng.random(m)<.15]=0; cap[0]=.42
    budget=[0.,cap.sum(),cap.sum()*1e-8,cap.sum()*(1-1e-8),rng.random()*cap.sum(),cap[:m//2].sum()][k%6]; ridge=10.**rng.uniform(-5,0)
    cert=b._quadratic(Ac,r,ridge,budget,cap); w=cert['weights']; lo=cert['lam_lower']; hi=cert['lam_upper']; grad=Ac.T@(Ac@w-r)+ridge*w
    metrics=[abs(w.sum()-budget),max(0.,-w.min(),np.max(w-cap)),np.max(abs(grad+cert['nu_sum']-lo+hi)),max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))),abs(cert['primal_objective']-cert['dual_lower_bound'])]
    worst=np.maximum(worst,metrics); assert np.min(lo)>=0 and np.min(hi)>=0
print('450 QP stress instances: worst budget, bound, stationarity, complementarity, dual gap:',worst); assert np.all(worst<1e-8)
PY
```
- **Container Exit Code**: `0` (`1135 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Leave-one-tracer-out worst RMSE: symmetric ablation, delivered independent-wall fit: [1.08669844e-04 3.10881204e-05]
450 QP stress instances: worst budget, bound, stationarity, complementarity, dual gap: [8.62929007e-15 0.00000000e+00 2.72848411e-12 0.00000000e+00
 5.45696821e-12]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=78a1f1956420a545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_78a1f195 (directory).
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/scipy/optimize/_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  fx = wrapped_fun(x)
/usr/local/lib/python3.12/dist-packages/scipy/optimize/_slsqp_py.py:441: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  g = append(wrapped_grad(x), 0.0)
```

### Turn 13 — `submit_completed_task` (`13904 ms` LLM)
- **Rationale**: The exact archived candidate passed the public schema in an isolated process. Held-out tracer results support independent walls, and stress tests separately verified the blind fallback and QP certificate.
- **Alternatives Weighed**: `Share one Robin coefficient across opposite walls.` (rejected: Worst held-out tracer RMSE was 1.09e-4, versus 3.11e-5 for the delivered model.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed. Across 450 box-simplex stress tests, the worst primal–dual gap was 5.46e-12.`

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
