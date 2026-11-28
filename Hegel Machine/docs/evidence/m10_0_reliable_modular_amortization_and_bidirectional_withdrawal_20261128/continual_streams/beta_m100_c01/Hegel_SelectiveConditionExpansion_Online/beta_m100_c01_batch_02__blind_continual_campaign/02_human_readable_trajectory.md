# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c01_batch_02__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `276,498` (`265,712` / `98,944` / `10,786`) |
| **Total Episode Wall Time (s)** | `344.79s` |

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

### Turn 1 — `execute_in_remote_docker` (`18287 ms` LLM)
- **Rationale**: Inspect the calibrated interface to avoid applying the sensor transform twice, and check telemetry and diagnostic availability before choosing a boundary model.
- **Alternatives Weighed**: `Implement the open-wall solver immediately` (rejected: The archive or calibrated diagnostics may identify a different boundary, and the staged wrapper determines which sensor operations are already performed.)

```bash
cd /holdout_task && ls -l /tmp/hegel_continual_archive /holdout_task && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py public_asset_index.json && for f in /tmp/hegel_continual_archive/*.json; do echo ==== "$f"; wc -c "$f"; head -c 4800 "$f"; echo; done && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('batch length',len(b)); i=b[0]; print('item keys',list(i));
for k,v in i['study'].items():
 print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else (str(v)[:120] if np.size(v)<25 else ''))
d=i.get('campaign_diagnostics'); print('diagnostics', type(d).__name__,len(d) if d is not None else None)
if d: print([(k,np.shape(v)) for k,v in d[0].items()])
PY
cd /tmp; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'box simplex KKT ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`473 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e_batch",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py"
  },
  {
    "episode_short_name": "beta_m100_c01_batch_01",
    "task_id": "beta_m100_c01_batch_01__blind_continual_campaign",
    "entrypoint": "solve_batch",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py"
  }
]
==== /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
127 /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "pre_llm_gate_accepted": false,
  "repair_target": "downstream_module_solve",
  "failure_kind": "scientific_or_numerical"
}
batch length 1
item keys ['study', 'campaign_diagnostics']
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) [0.47478108 0.50828527 0.53984232 0.56761825 0.58999882 0.60568335
 0.61376032 0.61376032 0.60568335 0.58999882 0.567618
kappa_y ndarray (12,) [0.50168525 0.4968244  0.48263567 0.46026854 0.43153505 0.39876303
 0.36460747 0.33183545 0.30310196 0.28073482 0.266546
vel_x ndarray (14,) [ 0.35        0.34500413  0.33015913  0.30588881  0.27288602  0.23209293
  0.18467409  0.13198319  0.07552447  0.0169096
vel_y ndarray (12,) [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907 -0.27988581 -0.2727233
robin_x float None 1.0123957203938798
robin_y float None 1.2636561147288357
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) [0.15 0.55 1.1 ]
source_modes ndarray (8, 14, 12) 
sensor_ij ndarray (32, 2) 
row_bank_id ndarray (96,) 
sensor_calibration_sweeps dict None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3) 
y_obs_raw ndarray (96,) 
aux_manifold_telemetry ndarray (24, 4) 
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
diagnostics list 3
[('w_pulse', (8,)), ('y_pulse_raw', (96,))]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19601 ms` LLM)
- **Rationale**: Read the prior wall implementations and summarize telemetry to define plausible boundary candidates; compare sweep-fit gains and offsets with the staged payload to separate sensor mismatch from boundary mismatch.
- **Alternatives Weighed**: `Assume the latest archived wall operator transfers unchanged` (rejected: Three public tracer pulses can test transfer directly after applying the staged upstream calibration.)

```bash
cd /holdout_task && cat _upstream_param_payload.json validate_public_schema.py && tail -12 _verified_upstream.py && diff -u /tmp/hegel_continual_archive/*s00_solver.py /tmp/hegel_continual_archive/*batch_01_solver.py; cat /tmp/hegel_continual_archive/*batch_01_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log structure',type(log),len(log)); records=log.values() if isinstance(log,dict) else log
for r in records:
 if not isinstance(r,dict): print(type(r),str(r)[:200]); continue
 print({k:(str(v)[:240] if not isinstance(v,list) else (np.shape(v),str(v[:1])[:180])) for k,v in r.items()})
i=pickle.load(open('public_batch.pkl','rb'))[0]; s=i['study']; sensor=s['sensor_calibration_sweeps']; print('sensor',[(k,type(v).__name__,str(v)[:200]) for k,v in sensor.items()]); a=sensor['raw_rosette']; b=sensor['ref_rosette']; m=(abs(a)<=8.5).all(1); print('affine estimates', [np.linalg.lstsq(np.c_[np.ones(m.sum()),b[m,j]],a[m,j],rcond=None)[0].tolist() for j in range(4)]); print('pulses',[p['w_pulse'].tolist() for p in i['campaign_diagnostics']]); print('nuisance range',np.min(s['Z_nuisance'],0),np.max(s['Z_nuisance'],0))
PY
cd /tmp; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'nuisance ridge profiling'
```
- **Container Exit Code**: `0` (`482 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
log structure <class 'list'> 3
{'completed_campaign_index': '0', 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': ((24, 8), '[[-0.80172, 0.58231, 0.74263, -0.26792, -0.26256, 0.85295, -0.2553, -0.64406]]')}
{'completed_campaign_index': '1', 'completed_campaign_short_name': 'beta_m100_c01_s00', 'facility_telemetry_trace': ((24, 8), '[[-0.89062, -0.80884, -0.38836, 2.49024, 0.70915, -0.78874, -0.32619, -0.65685]]')}
{'completed_campaign_index': '2', 'completed_campaign_short_name': 'beta_m100_c01_batch_01', 'facility_telemetry_trace': ((24, 8), '[[-0.84713, 0.78435, -0.29872, -0.45479, 0.79107, -0.86178, 2.62423, -0.1667]]')}
sensor [('ref_rosette', 'ndarray', '[[-1.2         0.          0.          0.        ]\n [-0.4         0.          0.          0.        ]\n [ 0.5         0.          0.          0.        ]\n [ 1.3         0.          0.          0.      '), ('raw_rosette', 'ndarray', '[[-1.45400711 -0.06000077  0.08000868 -0.05002001]\n [-0.43799984 -0.05999255  0.08000587 -0.04999762]\n [ 0.70498827 -0.05999801  0.08002427 -0.05000014]\n [ 1.72099112 -0.06000154  0.07999086 -0.049997'), ('single_bank_sweeps', 'dict', "{'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,\n       3, 3, 3, 3]), 'ref_signal': array([")]
affine estimates [[0.06999962453775567, 1.2700001723248289], [-0.05999963574497941, 0.8100021715192898], [0.08000050330489568, 1.2600002715394283], [-0.05000134496974551, 0.8199989241956948]]
pulses [[1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]]
nuisance range [-1.88254111 -3.27999897 -2.7009554 ] [2.88662981 3.19791976 2.33915181]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`33487 ms` LLM)
- **Rationale**: Compare exact boundary-row candidates and refit each family without one pulse at a time. Report both raw and nuisance-projected residuals so nuisance projection cannot hide a wrong boundary operator.
- **Alternatives Weighed**: `Reuse the archived reflecting-wall solver as-is` (rejected: The failed pre-LLM gate warrants testing the actual wall rows on the three calibrated tracer pulses.)

```bash
cd /holdout_task && sed -n '1,95p' /tmp/hegel_continual_archive/*batch_01_solver.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json,pickle,numpy as np
from _verified_upstream import calibrate_upstream
from scipy.optimize import least_squares
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
i=pickle.load(open('public_batch.pkl','rb'))[0]; raw=i['study']; pay=json.load(open('_upstream_param_payload.json')); s=calibrate_upstream(raw,pay); n,m=s['nx'],s['ny']; ij=s['sensor_ij']; ind=ij[:,0]+n*ij[:,1]; W=np.array([p['w_pulse'] for p in i['campaign_diagnostics']]).T
Y=np.column_stack([calibrate_upstream(dict(raw,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in i['campaign_diagnostics']]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; I=eye(n*m,format='csc'); B=np.stack([a.ravel(order='F') for a in s['source_modes']],axis=1)@W
print('aux summary',np.mean(raw['aux_manifold_telemetry'],axis=0),np.std(raw['aux_manifold_telemetry'],axis=0)); print('pulse RMS',np.sqrt(np.mean(Y**2,axis=0)))
def axis(k,v,h,p):
 f=(k[1:]+k[:-1])/2/h**2; a=v[1:-1]/h; l=np.r_[-f[:-1]-np.maximum(a,0),-p[3]*k[-1]/h**2]; r=np.r_[-p[1]*k[0]/h**2,-f[1:]-np.maximum(-a,0)]; d=np.r_[p[0]/h-r[0],f[:-1]+f[1:]+abs(a),p[2]/h-l[-1]]; return diags([l,d,r],[-1,0,1],format='lil')
def pred(p):
 a=axis(s['kappa_x'],s['vel_x'],s['Lx']/(n-1),p[:4]); b=axis(s['kappa_y'],s['vel_y'],s['Ly']/(m-1),p[4:8]); a[0,-1]=-p[8];a[0,0]+=p[8];a[-1,0]=-p[9];a[-1,-1]+=p[9]; b[0,-1]=-p[10];b[0,0]+=p[10];b[-1,0]=-p[11];b[-1,-1]+=p[11]; L=kron(eye(m),a)+kron(b,eye(n))+diags(s['V_reaction_2d'].ravel(order='F')); return np.vstack([splu((L+mu*I).tocsc()).solve(B)[ind] for mu in s['mu_shifts']])
def report(name,p):
 R=pred(p)-Y; print(name,'parameters',np.round(p,7),'direct RMS',np.round(np.sqrt(np.mean(R**2,axis=0)),9),'projected RMS',np.round(np.sqrt(np.mean((R-Q@(Q.T@R))**2,axis=0)),9))
x,y=s['robin_x'],s['robin_y']; p=np.array([x,1,x,1,y,1,y,1,0,0,0,0.]); r=np.array([0,1,0,1,0,1,0,1,0,0,0,0.]); report('open',p);report('reflect',r)
for npar in [4,8,12]:
 ix=[0,2,4,6] if npar==4 else list(range(npar)); expand=lambda z: np.put(p.copy(),ix,z)
 def unpack(z):
  w=p.copy(); w[ix]=z; return w
 for hold in [-1,0,1,2]:
  keep=[j for j in range(3) if j!=hold]
  def fun(z):
   R=(pred(unpack(z))-Y)[:,keep];return (R-Q@(Q.T@R)).ravel()
  fit=least_squares(fun,p[ix],bounds=(0,np.inf),max_nfev=100,gtol=1e-12); report(f'{npar} parameters hold={hold}',unpack(fit.x))
PY
```
- **Container Exit Code**: `0` (`10554 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
02977391 0.03421149 0.0416517 ]
8 parameters hold=-1 parameters [0.00000000e+00 1.22424900e+00 0.00000000e+00 7.28421900e-01
 0.00000000e+00 1.51463960e+00 2.37635586e+04 2.52049107e+05
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00] direct RMS [0.02982587 0.0366677  0.03030434] projected RMS [0.02981447 0.03626475 0.03002377]
8 parameters hold=0 parameters [0.00000000e+00 8.10131200e-01 0.00000000e+00 1.16334970e+00
 0.00000000e+00 2.34292360e+00 2.70987846e+03 2.33306466e+04
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00] direct RMS [0.03621365 0.03898294 0.02389359] projected RMS [0.03619944 0.03860467 0.0235056 ]
8 parameters hold=1 parameters [0.00000000e+00 2.21528780e+00 0.00000000e+00 6.55752300e-01
 0.00000000e+00 6.10963370e+00 3.65692828e+03 2.37292176e+04
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00] direct RMS [0.0316192  0.04441658 0.02433486] projected RMS [0.03158511 0.04356941 0.02419382]
8 parameters hold=2 parameters [0.00000000e+00 8.02734191e+01 0.00000000e+00 6.35138100e-01
 0.00000000e+00 4.43749600e-01 0.00000000e+00 4.07221884e+06
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00] direct RMS [0.01850772 0.0239665  0.08142942] projected RMS [0.01819603 0.02365447 0.08113898]
12 parameters hold=-1 parameters [3.20000000e-06 1.03525220e+00 0.00000000e+00 1.03538760e+00
 0.00000000e+00 9.94856600e-01 0.00000000e+00 1.00970110e+00
 1.58268508e+01 1.50474703e+01 1.15428353e+01 1.27965851e+01] direct RMS [4.655e-06 4.693e-06 4.675e-06] projected RMS [4.594e-06 4.628e-06 4.612e-06]
12 parameters hold=0 parameters [3.00000000e-07 1.03524110e+00 0.00000000e+00 1.03535920e+00
 0.00000000e+00 9.94896000e-01 0.00000000e+00 1.00970650e+00
 1.58380401e+01 1.50525353e+01 1.15426840e+01 1.27957356e+01] direct RMS [1.0552e-05 4.6520e-06 4.6490e-06] projected RMS [1.0373e-05 4.5900e-06 4.5770e-06]
12 parameters hold=1 parameters [7.00000000e-07 1.03509710e+00 0.00000000e+00 1.03546060e+00
 0.00000000e+00 9.94770400e-01 1.60000000e-06 1.00981490e+00
 1.58248685e+01 1.50481255e+01 1.15422927e+01 1.27977897e+01] direct RMS [4.647e-06 5.163e-06 4.668e-06] projected RMS [4.589e-06 5.111e-06 4.600e-06]
12 parameters hold=2 parameters [4.70000000e-06 1.03529940e+00 0.00000000e+00 1.03537320e+00
 0.00000000e+00 9.94749900e-01 0.00000000e+00 1.00980890e+00
 1.58268701e+01 1.50464904e+01 1.15406251e+01 1.27968642e+01] direct RMS [4.647e-06 4.691e-06 4.775e-06] projected RMS [4.589e-06 4.625e-06 4.725e-06]
```

### Turn 4 — `execute_in_remote_docker` (`31616 ms` LLM)
- **Rationale**: Check whether the periodic stencil predicts all calibrated tracer pulses at the calibration noise floor without fitted coefficients. Inspect the archived QP implementation and skill contract before selecting the constrained solver.
- **Alternatives Weighed**: `Hardcode the fitted endpoint coefficients` (rejected: The fitted scales and wrap rates suggest an exact stencil that also transfers across grid spacings and coefficients.)
- **ATMS NOGOOD DECLARED**: Refuted `The boundary rows couple only to the adjacent interior cell.` | Witness: `Reflecting-wall pulse RMS reached 0.0419; allowing opposite-end coupling gave at most 1.06×10⁻⁵ RMS on held-out pulses.` | Blanket Diagnosis: `Projecting nuisance signals barely changed the reflecting-wall errors; the missing opposite-end flux explains the discrepancy.` | Sublated Pivot: `Test the parameter-free periodic stencil with arithmetic face diffusivity and endpoint upwinding.`

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json,pickle,numpy as np
from _verified_upstream import calibrate_upstream
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
item=pickle.load(open('public_batch.pkl','rb'))[0]; raw=item['study']; pay=json.load(open('_upstream_param_payload.json')); s=calibrate_upstream(raw,pay)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]
def axis(a,n):
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(n-1); face=(k+np.roll(k,-1))/(2*h*h)
 lo=np.roll(face,1)+np.maximum(v,0)/h; hi=face+np.maximum(-v,0)/h
 print(a,'inward scales',hi[0]/(k[0]/h**2),lo[-1]/(k[-1]/h**2),'wrap rates',lo[0],hi[-1], 'constant row sum',np.max(np.abs(lo+hi-lo-hi)))
 A=diags((-lo[1:],lo+hi,-hi[:-1]),(-1,0,1),format='lil'); A[0,-1]=-lo[0]; A[-1,0]=-hi[-1]; return A.tocsc()
L=kron(eye(ny),axis('x',nx),format='csc')+kron(axis('y',ny),eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); S=np.vstack([splu(L+u*I).solve(rhs)[slots] for u in s['mu_shifts']]);
for j,p in enumerate(item['campaign_diagnostics']):
 y=calibrate_upstream(dict(raw,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw']; R=(S@p['w_pulse']-y).reshape(len(s['mu_shifts']),-1); print('pulse',j,'shift RMS',np.sqrt(np.mean(R*R,axis=1)),'max',abs(R).max())
print('observation range',s['y_obs_raw'].min(),s['y_obs_raw'].max())
PY
sed -n '45,120p' /tmp/hegel_continual_archive/*batch_01_solver.py
cd /tmp; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
          "op": ">=",
          "value": 0
        },
        {
          "field": "eps",
          "op": ">",
          "value": 0
        },
        {
          "field": "max_leaves",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "solve_and_certify_l0",
          "signature": "solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict",
          "doc_summary": "Certify boxed sparse ridge regression with at most 16 real predictors."
        }
      ],
      "index_cards": [
        "autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_sparse_ridge_certification",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 5 — `execute_in_remote_docker` (`26214 ms` LLM)
- **Rationale**: Install the empirically supported periodic operator, retain the archived equality-constrained active-set method, and run the public validator to catch interface or output-shape errors before numerical stress testing.
- **Alternatives Weighed**: `Use fitted endpoint coefficients` (rejected: The periodic face stencil matches the calibrated pulses at the calibration noise floor without fitting endpoint constants.); `Apply another calibration downstream` (rejected: The staged wrapper already passes observations in canonical coordinates.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py').read_text()
imports = '''"""Periodic transport and a certified box-simplex fit in calibrated coordinates."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    face = 0.5 * (kappa + np.roll(kappa, -1)) / h**2
    lower = np.roll(face, 1) + np.maximum(velocity, 0.0) / h
    upper = face + np.maximum(-velocity, 0.0) / h
    op = diags((-lower[1:], lower + upper, -upper[:-1]), (-1, 0, 1), format='lil')
    # Add the periodic faces; for a two-node axis these share the interior neighbor.
    op[0, -1] -= lower[0]
    op[-1, 0] -= upper[-1]
    return op.tocsc()


'''
transport = src[src.index('def _transport'):src.index('def _calibrate')]
fit = src[src.index('def _box_simplex'):]
fit = fit.replace('def solve_batch(study: dict) -> dict:\n', 'def solve_downstream(cal_item: dict) -> dict:\n    """Reconstruct sources from the canonical observations supplied upstream."""\n    study = cal_item\n').replace('y = _calibrate(study)', "y = np.asarray(study['y_obs_raw'], dtype=float)")
Path('downstream_solver.py').write_text(imports + transport + fit)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`500 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf205be1a7458aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1cf205be (directory).
```

### Turn 6 — `execute_in_remote_docker` (`41298 ms` LLM)
- **Rationale**: Rebuild the periodic stencil independently and check calibrated pulses, the terminal field, and every certificate component against it. Stress the active set at empty, saturated, near-saturated, and integer-cap budgets to catch failures at degenerate faces.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: The schema check cannot detect a wrong transport index, terminal-field layout, or KKT certificate.)

```bash
set -e
cd /holdout_task
cat batch_solver.py _verified_upstream.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, json, pickle
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _box_simplex
from _verified_upstream import calibrate_upstream
item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']; pay = json.load(open('_upstream_param_payload.json'))
cal = calibrate_upstream(s, pay)
nx, ny = s['nx'], s['ny']; rows, cols, vals = [], [], []
for j in range(ny):
 for i in range(nx):
  row = i + nx*j; main = s['V_reaction_2d'][i,j]
  for axis, a, n in [('x', i, nx), ('y', j, ny)]:
   h = s['L'+axis]/(n-1); k = s['kappa_'+axis]; v = s['vel_'+axis][a]
   for sign in (-1, 1):
    b = (a + sign) % n
    rate = (k[a]+k[b])/(2*h*h) + max(-sign*v, 0)/h
    rows.append(row); cols.append(b + nx*j if axis == 'x' else i + nx*b); vals.append(-rate); main += rate
  rows.append(row); cols.append(row); vals.append(main)
L = coo_matrix((vals, (rows, cols)), shape=(nx*ny, nx*ny)).tocsc()
B = np.column_stack([m.ravel(order='F') for m in s['source_modes']]); I = eye(nx*ny, format='csc')
fields = [spsolve(L + mu*I, B) for mu in s['mu_shifts']]
indices = s['sensor_ij'] @ [1, nx]; S = np.vstack([u[indices] for u in fields]); errors = []
for p in item['campaign_diagnostics']:
 y = calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), pay)['y_obs_raw']
 d = (S@p['w_pulse'] - y).reshape(len(fields), -1)
 errors.extend(np.sqrt(np.mean(d*d, axis=1)))
assert max(errors) < 2e-5, errors
Z = s['Z_nuisance']; project = np.eye(len(S)) - Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
A = project @ S; r = project @ cal['y_obs_raw']; ridge = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']
out = solve_batch(s); direct = solve_downstream(cal); w = out['weights']; cert = out['kkt_certificate']; nu = cert['nu_sum']; lo = np.asarray(cert['lam_lower']); hi = np.asarray(cert['lam_upper'])
assert np.allclose(w,direct['weights'],atol=1e-12,rtol=0)
g = A.T@(A@w-r)+ridge*w; J = float((np.linalg.norm(A@w-r)**2+ridge*(w@w))/2)
D = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
checks = {'mass': abs(w.sum()-budget), 'box': max(0., -w.min(), w.max()-cap), 'stationarity': np.max(abs(g-nu-lo+hi)), 'complementarity': max(np.max(abs(w*lo)), np.max(abs((cap-w)*hi))), 'multiplier_sign': max(0.,-lo.min(),-hi.min()), 'weights_certificate': np.max(abs(w-cert['weights'])), 'primal': abs(J-cert['primal_objective']), 'dual': abs(D-cert['dual_lower_bound']), 'duality_gap': abs(J-D), 'field': np.max(abs(out['terminal_field'] - (fields[-1]@w).reshape((nx,ny),order='F')))}
assert max(checks.values()) < 1e-10, checks
rng = np.random.default_rng(26); worst = 0.
for n in (2, 8, 32):
 for f in (0., 1e-8, 1/n, .43, (n-1)/n, 1.-1e-8, 1.):
  for _ in range(24):
   M = rng.normal(size=(n,n)); H = M.T@M+.04*np.eye(n); c = rng.normal(size=n)*3; q = f*n*cap
   a, b = _box_simplex(H,c,cap,q); grad = H@a+c-b
   err = max(abs(a.sum()-q), max(0.,-a.min(),a.max()-cap), np.max(a*np.maximum(grad,0)), np.max((cap-a)*np.maximum(-grad,0)))
   worst = max(worst, err)
assert worst < 1e-9, worst
print('worst pulse-by-shift RMS:',max(errors)); print('independent public certificate and field errors:',checks); print('504 box-simplex stress cases; max feasibility/complementarity error:',worst)
PY
```
- **Container Exit Code**: `0` (`642 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
from copy import deepcopy

import numpy as np


def calibrate_upstream(study: dict, param_payload=None) -> dict:
    """Return a deep copy with canonical observations and normalized inlier sweeps."""
    cal_study = deepcopy(study)
    sensor = cal_study['sensor_calibration_sweeps']
    raw = np.array(sensor['raw_rosette'], dtype=float, copy=True)
    ref = np.asarray(sensor['ref_rosette'], dtype=float)
    inlier = np.all(np.abs(raw) <= 8.5, axis=1)

    if param_payload is not None and 'gains' in param_payload and 'offsets' in param_payload:
        gains = np.asarray(param_payload['gains'], dtype=float).reshape(4)
        offsets = np.asarray(param_payload['offsets'], dtype=float).reshape(4)
    else:
        ref_fit, raw_fit = ref[inlier], raw[inlier]
        ones = np.ones(len(ref_fit))
        offsets, gains = np.empty(4), np.empty(4)
        for j in range(4):
            design = np.column_stack((ones, ref_fit[:, j]))
            offsets[j], gains[j] = np.linalg.lstsq(design, raw_fit[:, j], rcond=None)[0]

    observations = np.asarray(cal_study['y_obs_raw'], dtype=float).reshape(-1, 4)
    cal_study['y_obs_raw'] = ((observations - offsets) / gains).ravel()
    raw[inlier] = ref[inlier]
    sensor['raw_rosette'] = raw

    if 'single_bank_sweeps' in cal_study:
        bank = cal_study['single_bank_sweeps']
        readout = np.array(bank['raw_readout'], dtype=float, copy=True)
        signal = np.asarray(bank['ref_signal'], dtype=float)
        inlier = np.abs(readout) <= 8.5
        readout[inlier] = signal[inlier]
        bank['raw_readout'] = readout
    return cal_study
worst pulse-by-shift RMS: 5.173479976253373e-06
independent public certificate and field errors: {'mass': 0.0, 'box': 0.0, 'stationarity': 8.264222639553509e-15, 'complementarity': 2.3774586533388096e-16, 'multiplier_sign': 0.0, 'weights_certificate': 0.0, 'primal': 5.724587470723463e-17, 'dual': 1.5073012282762477e-14, 'duality_gap': 1.610864219792063e-14, 'field': 2.498001805406602e-16}
504 box-simplex stress cases; max feasibility/complementarity error: 1.9595436384634013e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf205be1a7458aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1cf205be (directory).
```

### Turn 7 — `submit_completed_task` (`15485 ms` LLM)
- **Rationale**: An independent periodic stencil reproduced the calibrated tracer measurements and matched the final solver’s terminal field and certificate. The exact submitted bundle passed isolated-process and active-set stress checks.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4662'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented the periodic downstream solver and verified it through the staged batch wrapper. The public schema passed. Worst tracer RMS across pulses and shifts was 5.17×10⁻⁶; the public KKT duality gap was 1.61×10⁻¹⁴, and the maximum feasibility or complementarity error across 504 stress cases was 1.96×10⁻¹⁴.`

### Turn 8 — `execute_in_remote_docker` (`38645 ms` LLM)
- **Rationale**: Use the real tracer pulses to test whether each axis independently needs wrap coupling and whether nuisance projection masks either omission. Compare the staged calibration with a fresh sweep fit to determine whether calibration changes the inferred weights or field enough to warrant action.
- **Alternatives Weighed**: `Test classification confounding, relay modes, or HJB drift` (rejected: This contract is a steady continuous inverse problem with no class labels, relay state, or time dynamics. Nuisance confounding is testable here by comparing tracer residuals before and after nuisance projection.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _transport, solve_downstream
from batch_solver import solve_batch
with open('public_batch.pkl','rb') as f: items=pickle.load(f)
with open('_upstream_param_payload.json') as f: pay=json.load(f)
for idx,item in enumerate(items):
 s=item['study']; cal=calibrate_upstream(s,pay); dynamic=calibrate_upstream(s)
 S,U=_transport(cal); Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0]
 nx,ny=int(s['nx']),int(s['ny']); grid=(nx,ny); positions=np.asarray(s['sensor_ij']); obs=positions@[1,nx]
 mats=[]; locals_=[]
 for a,n in zip('xy',grid):
  h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; A=_axis(k,v,h); mats.append(A)
  W=A.toarray(); W[[0,-1],:]=0.; W[0,1]=-k[0]/h**2; W[-1,-2]=-k[-1]/h**2; W[0,0]=-W[0,1]; W[-1,-1]=-W[-1,-2]; locals_.append(W)
  for j in range(n):
   expected=np.zeros(n)
   for side in (-1,1):
    b=(j+side)%n; rate=(k[j]+k[b])/(2*h*h)+max(-side*v[j],0.)/h
    expected[j]+=rate; expected[b]-=rate
   np.testing.assert_allclose(A.getrow(j).toarray().ravel(),expected,rtol=2e-14,atol=2e-14)
 I=eye(nx*ny,format='csc'); R=diags(np.asarray(s['V_reaction_2d']).ravel(order='F'),format='csc'); B=np.column_stack([m.ravel(order='F') for m in s['source_modes']])
 def predict(ax,ay):
  L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+R
  return np.vstack([splu(L+mu*I).solve(B)[obs] for mu in s['mu_shifts']])
 families={'periodic_xy':S,'periodic_x_only':predict(mats[0],locals_[1]),'periodic_y_only':predict(locals_[0],mats[1]),'reflecting_xy':predict(*locals_)}
 for name,P in families.items():
  scores=[]; projected=[]; channel=[]
  for p in item.get('campaign_diagnostics') or []:
   y=calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw']; e=P@p['w_pulse']-y
   scores.extend(np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1)**2,axis=1))); projected.append(np.sqrt(np.mean((e-Q@(Q.T@e))**2))); channel.append(np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)))
  if scores:
   print(idx,name,'worst pulse/shift RMS',max(scores),'worst projected pulse RMS',max(projected),'worst channel RMS',np.max(channel,axis=0));
   if name=='periodic_xy': assert max(scores)<2e-5
 out=solve_batch(s); alt=solve_downstream(dynamic); print(idx,'calibration staged vs sweep-fit max observation, weight, field deltas',*[np.max(np.abs(a-b)) for a,b in [(cal['y_obs_raw'],dynamic['y_obs_raw']),(out['weights'],alt['weights']),(out['terminal_field'],alt['terminal_field'])]])
 d=[]
 for p in item.get('campaign_diagnostics') or []:
  y=calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']))['y_obs_raw']; d.extend(np.sqrt(np.mean((S@p['w_pulse']-y).reshape(len(s['mu_shifts']),-1)**2,axis=1)))
 if d: print(idx,'sweep-fit worst pulse/shift RMS',max(d))
 print(idx,'active bounds',np.sum(out['weights']<1e-12),np.sum(out['weights']>s['u_box']-1e-12),'KKT gap',out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`632 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 periodic_xy worst pulse/shift RMS 5.173479976319813e-06 worst projected pulse RMS 4.667069092976775e-06 worst channel RMS [1.58180154e-06 5.77500824e-06 2.37521775e-06 7.09941859e-06]
0 periodic_x_only worst pulse/shift RMS 0.06030650942092801 worst projected pulse RMS 0.04056418139808627 worst channel RMS [0.0393993  0.04500273 0.04323632 0.03523854]
0 periodic_y_only worst pulse/shift RMS 0.03848834781684256 worst projected pulse RMS 0.027617063480939567 worst channel RMS [0.02777953 0.03120401 0.0275938  0.02406152]
0 reflecting_xy worst pulse/shift RMS 0.061880576728536016 worst projected pulse RMS 0.041651705248046 worst channel RMS [0.04374296 0.04274687 0.04379103 0.03699559]
0 calibration staged vs sweep-fit max observation, weight, field deltas 1.1509186185865161e-05 2.608038170143212e-05 3.11002289869744e-06
0 sweep-fit worst pulse/shift RMS 1.189042800989839e-06
0 active bounds 0 0 KKT gap 1.0928757898653885e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1cf205be1a7458aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1cf205be (directory).
```

### Turn 9 — `execute_in_remote_docker` (`34067 ms` LLM)
- **Rationale**: Use the verified upstream module’s existing sweep-fit branch without changing the independently validated downstream solver. Check that the final wrapper preserves the public certificate and field and produces the same outputs after a consistent change to the four raw sensor gains and offsets.
- **Alternatives Weighed**: `Keep the fixed payload in the batch wrapper` (rejected: It bypasses the required per-study fit, and fresh sweeps reduce the worst tracer residual by more than fourfold.)

```bash
set -e
cd /holdout_task
cat > batch_solver.py <<'PY'
"""Calibrate each study's rosette before solving in canonical coordinates."""

from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _transport
from _verified_upstream import calibrate_upstream
with open('public_batch.pkl','rb') as f:
    item=pickle.load(f)[0]
with open('_upstream_param_payload.json') as f:
    pay=json.load(f)
s=item['study']; cal=calibrate_upstream(s); S,U=_transport(cal)
for name,p in [('staged',pay),('sweep-fit',None)]:
    errs=[]
    for pulse in item['campaign_diagnostics']:
        pred=S@pulse['w_pulse']; y=calibrate_upstream(dict(s,y_obs_raw=pulse['y_pulse_raw']),p)['y_obs_raw']
        errs.extend(np.sqrt(np.mean((pred-y).reshape(len(s['mu_shifts']),-1)**2,axis=1)))
    print(name, 'worst pulse-by-shift RMS', max(errs))
    if p is None: assert max(errs)<2e-6
out=solve_batch(s); w=out['weights']; k=out['kkt_certificate']; cap=s['u_box']; q=s['total_budget']; ridge=s['source_ridge']; Z=s['Z_nuisance']; P=np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
A=P@S; r=P@cal['y_obs_raw']; e=A@w-r; g=A.T@e+ridge*w; J=(e@e+ridge*(w@w))/2; nu=k['nu_sum']; D=J-g@w+nu*q+cap*np.minimum(0,g-nu).sum(); lo=np.asarray(k['lam_lower']); hi=np.asarray(k['lam_upper'])
errors={'mass':abs(w.sum()-q),'box':max(0.,-w.min(),w.max()-cap),'certificate_weights':np.max(abs(w-k['weights'])),'multiplier_sign':max(0.,-lo.min(),-hi.min()),'stationarity':max(abs(g-nu-lo+hi)),'complementarity':max(max(abs(w*lo)),max(abs((cap-w)*hi))),'objective':abs(J-k['primal_objective']),'dual':abs(D-k['dual_lower_bound']),'gap':abs(J-D),'field':np.max(abs(out['terminal_field']-np.tensordot(w,U,axes=(0,0))))}
assert max(errors.values())<1e-10,errors
print('maximum public certificate or field error',max(errors.values()))
a=copy.deepcopy(s); scale=np.array([0.82,1.13,0.88,1.06]); bias=np.array([0.04,-0.03,0.05,-0.02]); raw=a['sensor_calibration_sweeps']['raw_rosette']; mask=(abs(raw)<=8.5).all(axis=1); raw[mask]=raw[mask]*scale+bias; assert np.array_equal(mask,(abs(raw)<=8.5).all(axis=1)); a['y_obs_raw']=(a['y_obs_raw'].reshape(-1,4)*scale+bias).ravel(); z=solve_batch(a)
shift=max(np.max(abs(out[key]-z[key])) for key in ('weights','terminal_field')); assert shift<1e-10,shift
print('maximum output delta after changing all four raw sensor gains and offsets',shift)
PY
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
staged worst pulse-by-shift RMS 5.173479976319813e-06
sweep-fit worst pulse-by-shift RMS 1.189042800989839e-06
maximum public certificate or field error 6.574601973952099e-16
maximum output delta after changing all four raw sensor gains and offsets 6.966649479522857e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0108b80e6927642f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0108b80e (directory).
```

### Turn 10 — `execute_in_remote_docker` (`31653 ms` LLM)
- **Rationale**: The host rejected the combined probe as resolution evidence. Run a standalone, read-only probe on the original tracer measurements: independently assemble the stencil, remove wrap coupling one axis at a time, and test whether nuisance projection could hide either omission. A discrepancy would reopen the wall model; a pass supports submitting the unchanged bundle.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _transport

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['study']
s = calibrate_upstream(raw)
S, _ = _transport(s)
nx, ny = int(s['nx']), int(s['ny'])
obs = s['sensor_ij'] @ [1, nx]
B = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
I = eye(nx * ny, format='csc')
V = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
Z = s['Z_nuisance']
Q = np.linalg.qr(Z, mode='reduced')[0]

def axis(a, n, wrap):
    k, v = s['kappa_'+a], s['vel_'+a]
    h = s['L'+a] / (n-1)
    A = np.zeros((n,n))
    for i in range(n):
        if not wrap and i in (0, n-1):
            A[i,i] = k[i]/h**2
            A[i,1 if i == 0 else n-2] = -A[i,i]
        else:
            for step in (-1,1):
                j = (i+step) % n
                rate = (k[i]+k[j])/(2*h**2) + max(-step*v[i],0)/h
                A[i,i] += rate
                A[i,j] -= rate
    return A

W = np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']])
Y = np.column_stack([calibrate_upstream(dict(raw, y_obs_raw=p['y_pulse_raw']))['y_obs_raw'] for p in item['campaign_diagnostics']])
metrics = {}
for label, wx, wy in [('periodic_xy',True,True),('local_x',False,True),('local_y',True,False)]:
    L = kron(eye(ny),axis('x',nx,wx),format='csc') + kron(axis('y',ny,wy),eye(nx),format='csc') + V
    design = np.vstack([splu(L+mu*I).solve(B)[obs] for mu in s['mu_shifts']])
    if wx and wy:
        np.testing.assert_allclose(design,S,atol=1e-12,rtol=0)
    E = design@W-Y
    P = E-Q@(Q.T@E)
    rms = lambda A: float(np.sqrt(np.mean(A.reshape(len(s['mu_shifts']),len(obs),-1)**2,axis=1)).max())
    metrics[label] = rms(E), rms(P)
    print(label, 'worst pulse-by-shift RMS:', rms(E), 'after nuisance projection:', rms(P))
assert max(metrics['periodic_xy']) < 2e-6, metrics
assert min(*metrics['local_x'], *metrics['local_y']) > 1e-2, metrics
print('PASS ng_t04: independent stencil matches the downstream operator and real pulses require wrap coupling on both axes')
PY
```
- **Container Exit Code**: `0` (`433 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
periodic_xy worst pulse-by-shift RMS: 1.1890428009898391e-06 after nuisance projection: 1.1990682402614762e-06
local_x worst pulse-by-shift RMS: 0.03848868540598885 after nuisance projection: 0.03827527030823876
local_y worst pulse-by-shift RMS: 0.06030700367802398 after nuisance projection: 0.06041171361597839
PASS ng_t04: independent stencil matches the downstream operator and real pulses require wrap coupling on both axes

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0108b80e6927642f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0108b80e (directory).
```

### Turn 11 — `submit_completed_task` (`15508 ms` LLM)
- **Rationale**: The final bundle passed the public schema and isolated-process checks. An independently assembled stencil matched the delivered operator; removing wrap coupling on either axis raised the worst tracer RMS above 0.038, even after nuisance projection.
- **Submitted Final Answer / Summary**: `Verified both solvers in the remote container. Per-study rosette calibration and periodic transport on both axes reduced the worst tracer RMS to 1.19×10⁻⁶. The public schema passed, the maximum independently recomputed certificate or field discrepancy was 6.57×10⁻¹⁶, and the box-simplex solver passed 504 stress cases.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
