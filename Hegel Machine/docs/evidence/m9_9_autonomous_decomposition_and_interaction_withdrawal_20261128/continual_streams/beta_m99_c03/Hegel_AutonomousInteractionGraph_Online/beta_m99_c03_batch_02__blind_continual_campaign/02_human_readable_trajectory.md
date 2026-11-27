# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m99_c03_batch_02__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `390,310` (`378,004` / `178,048` / `12,306`) |
| **Total Episode Wall Time (s)** | `360.79s` |

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

### Turn 1 — `execute_in_remote_docker` (`14114 ms` LLM)
- **Rationale**: Inspect the adapter, validator, public input structure, and apparatus log first to identify the solver interface and active wall setting without pulling raw data out of the container.
- **Alternatives Weighed**: `Implement the baseline immediately` (rejected: The blind campaign may inherit a different wall setting, and inspecting the upstream adapter determines which calibrations must not be applied twice.)

```bash
cd /holdout_task && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && cat /tmp/hegel_continual_archive/archive_apparatus_log.json && python3 - <<'PY'
import pickle
x=pickle.load(open('public_batch.pkl','rb'))
print('items',len(x))
for item in x:
 print('item keys',list(item),'diagnostics',type(item.get('campaign_diagnostics')).__name__)
 for k,v in item['study'].items(): print(k, type(v).__name__, getattr(v,'shape',None), list(v) if isinstance(v,dict) else (v if isinstance(v,(float,int,str)) else ''))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.33502,
        0.51935,
        2.60371,
        0.80967,
        -0.84508
      ],
      [
        -0.44501,
        0.58372,
        -0.45186,
        -0.5744,
        0.78087,
        2.45981,
        0.81694,
        -0.7431
      ],
      [
        -0.33661,
        0.80504,
        0.17214,
        -0.69418,
        0.86553,
        2.62025,
        0.70698,
        -0.40437
      ],
      [
        -0.33522,
        0.837,
        0.75835,
        -0.83301,
        0.63613,
        2.62455,
        0.47886,
        0.02862
      ],
      [
        -0.27426,
        0.63565,
        0.85017,
        -0.83184,
        0.33211,
        2.56179,
        0.28134,
        0.49257
      ],
      [
        -0.22849,
        0.32274,
        0.46096,
        -0.85886,
        -0.14425,
        2.59234,
        0.06056,
        0.71407
      ],
      [
        -0.4223,
        -0.14882,
        -0.14751,
        -0.76186,
        -0.56544,
        2.46866,
        -0.14676,
        0.90611
      ],
      [
        -0.48914,
        -0.58374,
        -0.67011,
        -0.66905,
        -0.74056,
        2.49207,
        -0.37695,
        0.75193
      ],
      [
        -0.43396,
        -0.83681,
        -0.80377,
        -0.48374,
        -0.86967,
        2.57447,
        -0.52445,
        0.48803
      ]
    ]
  }
]items 1
item keys ['study', 'campaign_diagnostics'] diagnostics list
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 1.0240161106792895
robin_y float None 1.1758119992635399
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) 
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
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18651 ms` LLM)
- **Rationale**: Recover the archived boundary formulas and pulse layout without printing raw arrays. The next test will compare their predictions after upstream calibration and retain the specified Robin operator if it fits.
- **Alternatives Weighed**: `Select the boundary from the apparatus log alone` (rejected: This batch has tracer pulses, which can directly distinguish the current boundary operator from archived alternatives.)

```bash
cd /holdout_task && ls -lh /tmp/hegel_continual_archive && grep -nE '^def |robin|boundary|wall|sensor_ij|mu_shifts' /tmp/hegel_continual_archive/*solver.py | head -130 && python3 - <<'PY'
import json,pickle,numpy as np
from pathlib import Path
root=Path('/tmp/hegel_continual_archive')
def brief(x):
 if isinstance(x,dict):return {k:brief(v) for k,v in x.items()}
 if isinstance(x,list):return [brief(v) for v in x] if len(x)<=5 else f'list[{len(x)}]'
 return x
for f in ['archive_manifest.json','archive_apparatus_log.json']:
 if (root/f).exists(): print(f, str(brief(json.loads((root/f).read_text())))[:3800])
it=pickle.load(open('public_batch.pkl','rb'))[0]; s=it['study']; d=it['campaign_diagnostics'];print('pulses',len(d),{k:np.shape(v) for k,v in d[0].items()});print('mu',s['mu_shifts'],'sensor span',s['sensor_ij'].min(0),s['sensor_ij'].max(0));print('banks',s['row_bank_id'][:40]); print('pulse range',[(float(np.linalg.norm(p['w_pulse'])),float(np.linalg.norm(p['y_pulse_raw']))) for p in d]);print('telemetry summary',np.round(s['aux_manifold_telemetry'][:3],4))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_solver.py:49:def _box_simplex(H, c, cap, budget):
/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py:95:def solve_batch(study: dict) -> dict:
archive_manifest.json [{'episode_short_name': 'beta_m99_c03_s00', 'task_id': 'beta_m99_c03_s00__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py'}, {'episode_short_name': 'beta_m99_c03_batch_01', 'task_id': 'beta_m99_c03_batch_01__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['kkt_certificate', 'terminal_field', 'weights'], 'solver_path': '/tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py'}]
archive_apparatus_log.json [{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': 'list[24]'}, {'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m99_c03_s00', 'facility_telemetry_trace': 'list[24]'}, {'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m99_c03_batch_01', 'facility_telemetry_trace': 'list[24]'}]
pulses 3 {'w_pulse': (8,), 'y_pulse_raw': (96,)}
mu [0.15 0.55 1.1 ] sensor span [0 0] [12 10]
banks [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
 1 2 3]
pulse range [(1.118033988749895, 2.319243261117572), (1.118033988749895, 2.3225034901127994), (1.118033988749895, 2.2800447602780936)]
telemetry summary [[-0.3825  0.5493 -1.0498 -0.6973]
 [-0.2421  1.3417 -0.9576 -0.9786]
 [ 1.5195 -0.4722  0.7843  0.082 ]]
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
```

### Turn 3 — `execute_in_remote_docker` (`30402 ms` LLM)
- **Rationale**: Compare the two documented wall settings against a nested boundary model. Raw and nuisance-projected pulse residuals will show whether the boundary mismatch is real or attributable to nuisance offsets.
- **Alternatives Weighed**: `Reuse reflecting walls from the prior batch` (rejected: Reflecting walls are one candidate, but the current calibrated pulses can distinguish them from Robin and mixed wall settings.)

```bash
cd /holdout_task && cat > /tmp/probe_boundary.py <<'PY'
import sys,json,pickle
sys.path.insert(0,'/holdout_task')
import _verified_upstream as up
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
P=item['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in P],axis=1); Y=np.stack([up.calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in P],axis=1)
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); modes=s['source_modes'].reshape(-1,nx*ny,order='F').T; B=modes@W

def axis(k,v,h,rob,loss,link):
 face=(k[:-1]+k[1:])/(2*h*h)
 l=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-link[1]*k[-1]/h**2]
 r=np.r_[-link[0]*k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]
 d=-np.r_[0,l]-np.r_[r,0]; d[[0,-1]]+=np.array(loss)*rob/h
 return diags([l,d,r],[-1,0,1],format='csc')

def pred(p,rhs=B):
 A=[axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a],p[2*j:2*j+2],p[4+2*j:6+2*j]) for j,a in enumerate('xy')]
 L=(kron(eye(ny),A[0])+kron(A[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])

def report(label,p):
 r=pred(p)-Y; rr=r-Q@(Q.T@r); print(label,'parameters',np.round(p,7),'raw pulse RMS',np.sqrt(np.mean(r*r,axis=0)),'projected pulse RMS',np.sqrt(np.mean(rr*rr,axis=0)))

if __name__=='__main__':
 for name,p in [('Robin',np.ones(8)),('reflecting',np.r_[np.zeros(4),np.ones(4)])]:report(name,p)
 for nuisance in (False,True):
  def resid(p):
   r=pred(p)-Y
   return r-Q@(Q.T@r) if nuisance else r
  for n in (4,8):
   fit=least_squares(lambda x:resid(np.r_[x,np.ones(8-n)]).ravel(),np.ones(n),bounds=(0,10),gtol=1e-12)
   report(f'fit {n}, nuisance={nuisance}',np.r_[fit.x,np.ones(8-n)])
   print('Jacobian singular values',np.round(np.linalg.svd(fit.jac,compute_uv=False),6))
 print('pulses',W.T)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_boundary.py && cat validate_public_schema.py && sed -n '57,150p' /tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py
```
- **Container Exit Code**: `0` (`1351 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0, cap)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                    'jac': lambda w: ones}],
                      options={'ftol': 1e-14, 'maxiter': 500})
    w = result.x
    tol = 1e-9 * cap
    lo, hi = w <= tol, w >= cap - tol
    free = ~(lo | hi)
    if np.any(free):
        # Remove the numerical optimization tolerance on the identified face.
        polished = np.where(hi, cap, 0.0)
        count = int(free.sum())
        a = np.ones(count)
        kkt = np.block([[H[np.ix_(free, free)] / scale, -a[:, None]],
                        [a[None, :], np.zeros((1, 1))]])
        rhs = np.r_[-(H @ polished + c)[free] / scale, budget - polished.sum()]
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        polished[free] = solution[:-1]
        nu = solution[-1] * scale
        reduced = H @ polished + c - nu
        if (np.all((polished >= 0) & (polished <= cap))
                and np.all(reduced[lo] >= -1e-11 * scale)
                and np.all(reduced[hi] <= 1e-11 * scale)):
            return polished, float(nu)
    g = H @ w + c
    if np.any(free):
        nu = np.mean(g[free])
    else:
        nu = 0.5 * (g[hi].max() + g[lo].min())
    return w, float(nu)


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrated_observations(study)
    Z = study['Z_nuisance']
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = np.column_stack((S, y))
    projected -= Z @ np.linalg.solve(gram, Z.T @ projected)
    design, target = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(design.T @ target), cap, budget)
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    reduced = g - nu
    lower = np.maximum(reduced, 0)
    upper = np.maximum(-reduced, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * upper.sum())
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
```

### Turn 4 — `execute_in_remote_docker` (`30300 ms` LLM)
- **Rationale**: Periodic closure adds the endpoint coupling missing from the first search. Comparing exact stencil variants and sweep fit errors will determine whether to pursue topology or revisit calibration.
- **Alternatives Weighed**: `Deploy the unrestricted local Robin fit` (rejected: It retains a sizable residual and drives a link multiplier to the search limit, suggesting that the stencil rather than the coefficient is wrong.)
- **ATMS NOGOOD DECLARED**: Refuted `Changing only the documented Robin loss with the baseline boundary links explains the current pulses.` | Witness: `Fitting four separate nonnegative Robin losses left a worst projected pulse RMS of 0.0417, versus 0.0425 for reflecting walls and 0.1252 for open walls.` | Blanket Diagnosis: `Nuisance projection barely changes the errors, and fitting local diffusion links pushes a link multiplier to its search bound.` | Sublated Pivot: `Test endpoint coupling and alternative local boundary stencils against the calibrated pulses.`

```bash
cat > /tmp/test_topology.py <<'PY'
from probe_boundary import *
from itertools import product

def op(a,kind):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]; n=len(k)
 if kind=='open': return axis(k,v,h,r,[1,1],[1,1])
 if kind=='reflect': return axis(k,v,h,r,[0,0],[1,1])
 if kind=='mirror': return axis(k,v,h,r,[0,0],[2,2])
 if kind=='absorb': return axis(k,v,h,r,[k[0]/(h*r),k[-1]/(h*r)],[1,1])
 A=axis(k,v,h,r,[0,0],[1,1]).tolil()
 for j,m,b in [(0,n-1,1),(n-1,0,n-2)]:
  if kind=='periodic_endpoint':
   out=k[j]/h**2; inside=out
  else:
   out=(k[j]+k[m])/(2*h*h); inside=(k[j]+k[b])/(2*h*h)
  if kind=='periodic':
   out+=max(v[j] if j==0 else -v[j],0)/h
   inside+=max(-v[j] if j==0 else v[j],0)/h
  A[j,m]=-out; A[j,b]=-inside; A[j,j]=out+inside
 return A.tocsc()

def test(a,b):
 L=(kron(eye(ny),a)+kron(b,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(B)[rows] for mu in s['mu_shifts']])

kinds=['open','reflect','mirror','absorb','periodic_endpoint','periodic_diffusion','periodic']; ops={a:{m:op(a,m) for m in kinds} for a in 'xy'}
runs=[]
for x,y in product(kinds,repeat=2):
 r=test(ops['x'][x],ops['y'][y])-Y
 e=np.sqrt(np.mean(r*r,axis=0)); runs.append((max(e),x,y,e))
for q in sorted(runs)[:14]: print(q)
print('sweep fit check')
sw=s['sensor_calibration_sweeps']; print({k:np.shape(v) for k,v in sw.items() if k!='single_bank_sweeps'}); print({k:np.shape(v) for k,v in sw['single_bank_sweeps'].items()})
a=sw['raw_rosette']; b=sw['ref_rosette']; keep=np.all(abs(a)<=8.5,axis=1)
for j in range(4):
 c=np.linalg.lstsq(np.c_[np.ones(sum(keep)),b[keep,j]],a[keep,j],rcond=None)[0]; print(j,c,np.std(a[keep,j]-c[0]-c[1]*b[keep,j]))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 t=np.asarray(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'],t.shape,t.mean(axis=0).round(3),t.std(axis=0).round(3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/test_topology.py; cat /holdout_task/public_asset_index.json | head -105
```
- **Container Exit Code**: `0` (`513 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
": {
    "/holdout_task/_upstream_param_payload.json": {
      "bytes": 120,
      "summary": {
        "gains": {
          "first": 1.2699989,
          "len": 4,
          "type": "list"
        },
        "offsets": {
          "first": 0.05999695,
          "len": 4,
          "type": "list"
        }
      }
    },
    "/holdout_task/_verified_upstream.py": {
      "bytes": 1399
    },
    "/holdout_task/batch_solver.py": {
      "bytes": 408
    },
    "/holdout_task/public_batch.pkl": {
      "bytes": 27279,
      "summary": {
        "first": {
          "campaign_diagnostics": {
            "first": {
              "w_pulse": {
                "dtype": "float64",
                "shape": [
                  8
                ],
                "type": "ndarray"
              },
              "y_pulse_raw": {
                "dtype": "float64",
                "shape": [
                  96
                ],
                "type": "ndarray"
              }
            },
            "len": 3,
            "type": "list"
          },
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
            "robin_x": 1.0240161106792895,
            "robin_y": 1.1758119992635399,
            "row_bank_id": {
              "dtype": "int64",
              "shape": [
                96
              ],
```

### Turn 5 — `execute_in_remote_docker` (`43932 ms` LLM)
- **Rationale**: Implement the pulse-selected periodic stencil and reuse the verified box-simplex method. Check both calibration paths by shift, then independently recompute feasibility, stationarity, complementarity, objective, and terminal-field assembly on the public study.
- **Alternatives Weighed**: `Use periodic diffusion without periodic advection` (rejected: The best stencil omitting wraparound advection on an axis has a worst pulse RMS more than 360 times larger.); `Fix the upstream wrapper to the staged gains and offsets` (rejected: The batch interface requires fitting each study's sweeps; the downstream interface accepts the staged adapter's calibrated observations directly.)

```bash
set -e
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Periodic transport and box-simplex source inversion in calibrated coordinates."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_operator(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    minus = (kappa + kappa[left]) / (2 * h**2) + np.maximum(velocity, 0) / h
    plus = (kappa + kappa[right]) / (2 * h**2) + np.maximum(-velocity, 0) / h
    return coo_matrix((np.r_[minus + plus, -minus, -plus],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n)).tocsc()


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis_operator(study['kappa_' + a], study['vel_' + a], study['L' + a])
              for a in 'xy')
    identity = eye(nx * ny, format='csc')
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx))
                + diags(study['V_reaction_2d'].ravel(order='F'))).tocsc()
    modes = study['source_modes']
    rhs = modes.reshape(len(modes), nx * ny, order='F').T
    i, j = study['sensor_ij'].T
    readings = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        readings.append(fields[i + nx * j])
    return np.concatenate(readings), fields.T.reshape(len(modes), nx, ny, order='F')


PY
sed -n '/^def _box_simplex/,/^def solve_batch/{ /^def solve_batch/!p; }' /tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py >> downstream_solver.py
sed -n '/^def solve_batch/,$p' /tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py | sed -e 's/^def solve_batch(study: dict)/def solve_downstream(study: dict)/' -e "s/y = _calibrated_observations(study)/y = np.asarray(study['y_obs_raw'], dtype=float)/" >> downstream_solver.py
cat > batch_solver.py <<'PY'
"""Calibrate the incoming study, then solve its downstream inverse problem."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _forward
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
with open('/holdout_task/_upstream_param_payload.json') as f:
    pay = json.load(f)
s = item['study']
S, U = _forward(s)
m = len(s['sensor_ij'])
for p in (None, pay):
    errors = []
    for pulse in item['campaign_diagnostics']:
        y = calibrate_upstream(dict(s, y_obs_raw=pulse['y_pulse_raw']), p)['y_obs_raw']
        errors.append(np.sqrt(np.mean((S @ pulse['w_pulse'] - y).reshape(-1, m)**2, axis=1)))
    err = np.max(errors)
    print('worst pulse / shift RMS', 'sweep fit' if p is None else 'staged payload', err)
    assert err < 1e-5
for cal, out in [(calibrate_upstream(s), solve_batch(s)), (calibrate_upstream(s, pay), solve_downstream(calibrate_upstream(s, pay)))]:
    w = out['weights']; c = out['kkt_certificate']; z = s['Z_nuisance']
    pz = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    A = S - z @ (pz @ S); y = cal['y_obs_raw']; r = A @ w - (y - z @ (pz @ y))
    grad = A.T @ r + s['source_ridge'] * w
    lo, hi = np.array(c['lam_lower']), np.array(c['lam_upper']); cap = s['u_box']
    obj = (r @ r + s['source_ridge'] * (w @ w)) / 2
    errors = [abs(w.sum()-s['total_budget']), max(0., -w.min(), w.max()-cap), np.max(abs(grad-c['nu_sum']-lo+hi)), np.max(abs(lo*w)), np.max(abs(hi*(cap-w))), abs(obj-c['primal_objective']), abs(obj-c['dual_lower_bound'])]
    print('worst primal/KKT/objective error', max(errors))
    assert max(errors) < 1e-9 and np.all(lo >= 0) and np.all(hi >= 0)
    np.testing.assert_allclose(out['terminal_field'], np.einsum('i,ijk->jk',w,U), atol=1e-12)
    np.testing.assert_array_equal(w, c['weights'])
PY
```
- **Container Exit Code**: `0` (`817 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
worst pulse / shift RMS sweep fit 1.1173682460688908e-06
worst pulse / shift RMS staged payload 5.150570330353098e-06
worst primal/KKT/objective error 6.696032617270475e-16
worst primal/KKT/objective error 2.1996293675385914e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`28529 ms` LLM)
- **Rationale**: Select the stencil using two tracer pulses and test it on the third, repeating for each pulse. If the winner changes or a held-out residual exceeds tolerance, revisit the boundary stencil; otherwise retain the already verified artifacts.
- **Alternatives Weighed**: `Submit using the existing schema and KKT verification` (rejected: The host did not accept the structural probe that ran in the same step as the artifact edits; a read-only pulse comparison separates that evidence from the build.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import sys
sys.path[:0] = ['/tmp', '/holdout_task']
from probe_boundary import s, W, Y, B, pred
from downstream_solver import _forward
import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from itertools import product

# Compare including or omitting wraparound advection on each axis, with
# predictions produced from the calibrated study and no fitted parameters.
nx, ny = s['nx'], s['ny']
i, j = s['sensor_ij'].T
I = eye(nx * ny, format='csc')
def axis(a, advect_at_wrap):
    k, v = s['kappa_'+a], s['vel_'+a]
    n = len(k); h = s['L'+a]/(n-1)
    row = np.arange(n); left = (row-1)%n; right = (row+1)%n
    m = (k+k[left])/(2*h*h); p = (k+k[right])/(2*h*h)
    ml, pr = np.maximum(v,0)/h, np.maximum(-v,0)/h
    if not advect_at_wrap:
        ml[[0,-1]] = 0; pr[[0,-1]] = 0
    m += ml; p += pr
    return coo_matrix((np.r_[m+p,-m,-p],(np.tile(row,3),np.r_[row,left,right])),shape=(n,n)).tocsc()

def pulse_slices(f):
    return np.sqrt(np.mean((f-Y).reshape(len(s['mu_shifts']),len(i),W.shape[1])**2,axis=1))

S, terminal = _forward(s)
errors = {'periodic_upwind': pulse_slices(S@W), 'robin': pulse_slices(pred(np.ones(8))), 'reflecting': pulse_slices(pred(np.r_[np.zeros(4),np.ones(4)]))}
for x,y in [(False,True),(True,False),(False,False)]:
    L = (kron(eye(ny),axis('x',x))+kron(axis('y',y),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    errors[f'wrap_advection_{x}_{y}'] = pulse_slices(np.concatenate([splu(L+mu*I).solve(B)[i+nx*j] for mu in s['mu_shifts']]))
for name,e in errors.items():
    print(name, 'worst_shift_pulse_rms', float(e.max()))
for pulse in range(W.shape[1]):
    train = [k for k in range(W.shape[1]) if k!=pulse]
    winner = min(errors,key=lambda name: np.max(errors[name][:,train]))
    held = float(errors[winner][:,pulse].max())
    print('held_pulse',pulse,'selected',winner,'worst_shift_rms',held)
    assert winner == 'periodic_upwind' and held < 1e-5
PY
```
- **Container Exit Code**: `0` (`450 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
periodic_upwind worst_shift_pulse_rms 5.150570330353098e-06
robin worst_shift_pulse_rms 0.1981403566661936
reflecting worst_shift_pulse_rms 0.0577521839127571
wrap_advection_False_True worst_shift_pulse_rms 0.0031214778286415266
wrap_advection_True_False worst_shift_pulse_rms 0.0027501869562850676
wrap_advection_False_False worst_shift_pulse_rms 0.003016847201760741
held_pulse 0 selected periodic_upwind worst_shift_rms 5.150570330353098e-06
held_pulse 1 selected periodic_upwind worst_shift_rms 5.1409664551803405e-06
held_pulse 2 selected periodic_upwind worst_shift_rms 5.11515780270679e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 7 — `submit_completed_task` (`13683 ms` LLM)
- **Rationale**: Each held-out pulse independently favored the same boundary stencil, and the verified artifact passed the schema, terminal-field assembly, and numerical certificate checks.
- **Alternatives Weighed**: `Remove periodic advection on one axis` (rejected: The best such variant had a worst pulse-by-shift RMS of 2.75e-3, compared with 5.15e-6 for full periodic upwinding.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:271', '/holdout_task/downstream_solver.py': 'FILE:4529'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified both solvers. Periodic upwind boundaries on both axes matched all held-out tracer pulses, with a worst pulse-by-shift RMS of 5.15e-6. The public schema passed, and the largest KKT or objective verification error was 2.20e-15.`

### Turn 8 — `execute_in_remote_docker` (`44463 ms` LLM)
- **Rationale**: An independently assembled dense stencil tests periodic indexing and terminal-field assembly without reusing the solver's forward code. Sensor-held slices test whether model selection depends on a few sensors or on nuisance projection. If either check fails, revisit the stencil or calibration; otherwise retain the verified artifacts.
- **Alternatives Weighed**: `Add nonlinear transport, relay, or classification terms` (rejected: This is a steady linear forward problem with continuous observations; the measured alternatives are boundary stencils and nuisance projection, not relay states or class labels.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -B validate_public_schema.py /holdout_task/batch_solver.py
python3 -I -B - <<'PY'
import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream
from _verified_upstream import calibrate_upstream
with open('public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
with open('_upstream_param_payload.json') as f:
    payload = json.load(f)
s = item['study']; nx, ny = s['nx'], s['ny']; shifts = s['mu_shifts']; ij = s['sensor_ij']
rows = ij[:, 0] + nx * ij[:, 1]; n = nx * ny
I = np.eye(n); modes = np.column_stack([a.flatten(order='F') for a in s['source_modes']])

def axis(a, kind):
    k, v = s['kappa_'+a], s['vel_'+a]; h = s['L'+a]/(len(k)-1); m = len(k)
    A = np.zeros((m,m))
    for i in range(m):
        if i in (0,m-1) and kind in ('robin','reflecting'):
            j = 1 if i == 0 else m-2
            A[i,j] = -k[i]/h**2
            A[i,i] = -A[i,j] + (s['robin_'+a]/h if kind == 'robin' else 0.)
            continue
        for step in (-1,1):
            j = (i+step)%m
            rate = (k[i]+k[j])/(2*h**2)
            if kind != 'diffusion_only_at_wall' or i not in (0,m-1):
                rate += max(0.,-step*v[i])/h
            A[i,j] -= rate
            A[i,i] += rate
    if kind != 'robin':
        np.testing.assert_allclose(A.sum(axis=1),0.,atol=1e-12)
    return A

kinds = [('periodic','periodic'),('robin','robin'),('reflecting','reflecting'),('diffusion_only_at_wall','periodic'),('periodic','diffusion_only_at_wall')]
W = np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']]); z = s['Z_nuisance']
P = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
Y = np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),payload)['y_obs_raw'] for p in item['campaign_diagnostics']])
raw, projected = [], []
for k, (x,y) in enumerate(kinds):
    L = np.kron(np.eye(ny),axis('x',x))+np.kron(axis('y',y),np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
    U = [np.linalg.solve(L+mu*I,modes) for mu in shifts]
    S = np.concatenate([u[rows] for u in U])
    if k == 0:
        reference_S, reference_U = S, U[-1]
        forward_error = max(np.max(abs((L+mu*I)@u-modes)) for mu,u in zip(shifts,U))
        assert forward_error < 1e-10
    r = S@W-Y
    raw.append(r.reshape(len(shifts),len(rows),-1))
    projected.append((r-z@(P@r)).reshape(len(shifts),len(rows),-1))
raw, projected = np.array(raw), np.array(projected)
print('independent dense forward max residual:',forward_error)
for label,errors in [('raw',raw),('ridge-projected',projected)]:
    scores = np.sqrt(np.mean(errors**2,axis=2)).max(axis=(1,2))
    print(label,'worst pulse/shift RMS:',dict(zip(kinds,scores)))
    winners, held = [], []
    for j in range(len(rows)):
        train = np.arange(len(rows))!=j
        winner = int(np.argmin(np.sqrt(np.mean(errors[:,:,train,:]**2,axis=2)).max(axis=(1,2))))
        winners.append(winner); held.append(np.max(abs(raw[winner,:,j,:])))
    print(label,'sensor-held folds selecting full periodic:',winners.count(0),'/',len(rows),'worst held point error:',max(held))
    assert all(w==0 for w in winners) and max(held)<2e-5

for label, cal, out in [('batch',calibrate_upstream(s),solve_batch(s)),('downstream',calibrate_upstream(s,payload),solve_downstream(calibrate_upstream(s,payload)))]:
    y = cal['y_obs_raw']; w = out['weights']; c = out['kkt_certificate']; cap = s['u_box']; budget = s['total_budget']; ridge = s['source_ridge']
    S = reference_S-z@(P@reference_S); r = S@w-(y-z@(P@y)); g = S.T@r+ridge*w
    lo, hi = np.array(c['lam_lower']), np.array(c['lam_upper']); nu = c['nu_sum']
    J = .5*(r@r+ridge*(w@w)); dual = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum(); field = (reference_U@w).reshape(nx,ny,order='F')
    e = max(abs(w.sum()-budget),max(0.,-w.min(),w.max()-cap),np.max(abs(g-nu-lo+hi)),np.max(abs(w*lo)),np.max(abs((cap-w)*hi)),abs(J-c['primal_objective']),abs(dual-c['dual_lower_bound']),abs(J-dual),np.max(abs(field-out['terminal_field'])))
    print(label,'max dense-reference certificate/field error:',e)
    assert e < 1e-9 and np.all(lo>=0) and np.all(hi>=0)
    np.testing.assert_array_equal(c['weights'],w)
PY
```
- **Container Exit Code**: `0` (`777 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
independent dense forward max residual: 7.049916206369744e-15
raw worst pulse/shift RMS: {('periodic', 'periodic'): 5.150570330466934e-06, ('robin', 'robin'): 0.1981403566661936, ('reflecting', 'reflecting'): 0.057752183912757216, ('diffusion_only_at_wall', 'periodic'): 0.003121477828641502, ('periodic', 'diffusion_only_at_wall'): 0.002750186956285194}
raw sensor-held folds selecting full periodic: 32 / 32 worst held point error: 7.944585370955881e-06
ridge-projected worst pulse/shift RMS: {('periodic', 'periodic'): 5.098762243292795e-06, ('robin', 'robin'): 0.19631565627263825, ('reflecting', 'reflecting'): 0.05753791392675607, ('diffusion_only_at_wall', 'periodic'): 0.0030979340421212246, ('periodic', 'diffusion_only_at_wall'): 0.0027233380906424944}
ridge-projected sensor-held folds selecting full periodic: 32 / 32 worst held point error: 7.944585370955881e-06
batch max dense-reference certificate/field error: 2.3401419690927128e-15
downstream max dense-reference certificate/field error: 1.4398204850607499e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`23838 ms` LLM)
- **Rationale**: Hold out a tracer pulse and a shift jointly, fit nuisance terms on training rows only, and score the untouched pulse-by-shift block without nuisance adjustment. If periodic upwind loses a fold or exceeds 1e-5 RMS, revisit the boundary or calibration; otherwise retain the verified artifacts.
- **Alternatives Weighed**: `Repeat the schema and certificate validation` (rejected: Both already passed on the unchanged artifact. Neither tests whether the boundary selection depends on a particular shift or on nuisance estimates fitted outside the fold.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import sys
sys.path[:0] = ['/tmp', '/holdout_task']
import numpy as np
from probe_boundary import s, W, Y, pred
from downstream_solver import _forward
S, _ = _forward(s)
models = {'periodic_upwind': S @ W, 'open_robin': pred(np.ones(8)), 'reflecting': pred(np.r_[np.zeros(4), np.ones(4)])}
Z = s['Z_nuisance']; ridge = s['nuisance_ridge']; m = len(s['sensor_ij'])
worst = 0.0; margin = np.inf; count = 0
for shift in range(len(s['mu_shifts'])):
    for pulse in range(W.shape[1]):
        rows = np.arange(len(Y)) // m != shift
        cols = np.arange(W.shape[1]) != pulse
        zs = Z[rows]
        proj = np.linalg.solve(zs.T @ zs + ridge*np.eye(zs.shape[1]), zs.T)
        scores = []
        for name, prediction in models.items():
            r = (prediction - Y)[rows][:, cols]
            r -= zs @ (proj @ r)
            scores.append((np.sqrt(np.mean(r*r)), name))
        scores.sort()
        winner = scores[0][1]
        held = models[winner][~rows, pulse] - Y[~rows, pulse]
        worst = max(worst, np.sqrt(np.mean(held*held)))
        margin = min(margin, scores[1][0]/scores[0][0])
        count += 1
        assert winner == 'periodic_upwind', scores
print(f'ng_t04: periodic upwind won {count}/{count} joint pulse-and-shift holdouts; worst raw held-out RMS={worst:.8g}; minimum training runner-up/winner RMS={margin:.3g}')
assert worst < 1e-5
PY
```
- **Container Exit Code**: `0` (`461 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ng_t04: periodic upwind won 9/9 joint pulse-and-shift holdouts; worst raw held-out RMS=5.1505703e-06; minimum training runner-up/winner RMS=4.96e+03

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 10 — `submit_completed_task` (`13961 ms` LLM)
- **Rationale**: The unchanged bundle has a bound verification receipt and passed independent forward, certificate, and field checks in an isolated process. The structural probe supports the selected stencil empirically, but the host did not accept it as formal closure of ng_t04.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:271', '/holdout_task/downstream_solver.py': 'FILE:4529'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Both solvers are saved and verified. Periodic upwinding on both axes won all 9 joint pulse-and-shift holdouts, with a worst held-out RMS of 5.15e-6. The public schema passed, and the largest dense-reference certificate or field error was 2.34e-15.`

### Turn 11 — `execute_in_remote_docker` (`21657 ms` LLM)
- **Rationale**: Inspect whether the batch wrapper and staged adapter implement distinct calibration contracts, then measure their tolerance margins by shift and check whether nuisance projection removes source directions. A calibration contract mismatch would warrant a wrapper change; rank loss would warrant checking identifiability before altering the specified regularization.
- **Alternatives Weighed**: `Add nonlinear, relay, or classification terms` (rejected: The supplied measurements are continuous and the specified forward map is linear in source weights; these terms do not correspond to a measured mechanism.)

```bash
set -e
cd /holdout_task
cat _verified_upstream.py batch_solver.py validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import json,pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward
with open('public_batch.pkl','rb') as f: item=pickle.load(f)[0]
with open('_upstream_param_payload.json') as f: pay=json.load(f)
s=item['study']; S,_=_forward(s); m=len(s['sensor_ij']); blocks=[]
for pulse in item['campaign_diagnostics']:
    p=dict(s,y_obs_raw=pulse['y_pulse_raw'])
    prediction=S@pulse['w_pulse']
    blocks.append([np.sqrt(np.mean((prediction-calibrate_upstream(p,params)['y_obs_raw']).reshape(-1,m)**2,axis=1)) for params in (None,pay)])
print('Worst real pulse RMS by calibration path and shift:',np.max(blocks,axis=0))
assert np.max(blocks)<1e-5
z=s['Z_nuisance']; pr=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); A=S-z@(pr@S)
print('Projected inverse design shape and rank:',A.shape,np.linalg.matrix_rank(A))
print('Minimum objective Hessian eigenvalue:',np.linalg.eigvalsh(A.T@A+s['source_ridge']*np.eye(A.shape[1]))[0])
PY
```
- **Container Exit Code**: `0` (`495 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.column_stack((np.ones(x.size), x)), raw[inlier, bank], rcond=None
            )[0]

    calibrated = deepcopy(study)
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    calibrated['y_obs_raw'] = ((observed - offsets) / gains).ravel()
    cal_sweep = calibrated['sensor_calibration_sweeps']
    cal_sweep['raw_rosette'] = np.where(inlier[:, None], ref, raw)
    if 'single_bank_sweeps' in cal_sweep:
        single = cal_sweep['single_bank_sweeps']
        readout = np.asarray(single['raw_readout'])
        single['raw_readout'] = np.where(np.abs(readout) <= 8.5, single['ref_signal'], readout)
    return calibrated
"""Calibrate the incoming study, then solve its downstream inverse problem."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
m_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (m_m,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not isinstance(out.get("kkt_certificate"), dict): errs.append("missing_or_bad:kkt_certificate")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
Worst real pulse RMS by calibration path and shift: [[1.11736825e-06 1.07142002e-06 1.05179241e-06]
 [5.15057033e-06 4.79621335e-06 4.65547389e-06]]
Projected inverse design shape and rank: (96, 8) 8
Minimum objective Hessian eigenvalue: 0.04029452542791693

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`30905 ms` LLM)
- **Rationale**: Remove each directed endpoint link separately while preserving zero transport row sums. If an ablation wins a training fold, reconsider that link; otherwise use the held-out tracer residuals to close the boundary-stencil finding without changing the verified solvers.
- **Alternatives Weighed**: `Repeat schema, calibration, or KKT checks` (rejected: Those checks passed on the unchanged artifacts; they cannot tell whether one of the four directed endpoint links is unnecessary.)

```bash
set -e
cat > /tmp/audit_wrap_links.py <<'PY'
import sys
sys.path[:0] = ['/tmp', '/holdout_task']
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from probe_boundary import s, W, Y
from downstream_solver import _axis_operator, _forward

nx, ny = s['nx'], s['ny']
axes = [_axis_operator(s['kappa_'+a], s['vel_'+a], s['L'+a]) for a in 'xy']
i, j = s['sensor_ij'].T
modes = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
reaction = diags(s['V_reaction_2d'].ravel(order='F'))
identity = eye(nx*ny, format='csc')
S, _ = _forward(s)
models = {'periodic': S @ W}
for axis, label in enumerate('xy'):
    for end in (0, -1):
        ops = axes.copy()
        a = ops[axis].tolil()
        row, opposite = (0, a.shape[0]-1) if end == 0 else (a.shape[0]-1, 0)
        a[row,row] += a[row,opposite]
        a[row,opposite] = 0
        assert np.max(np.abs(np.asarray(a.sum(axis=1)))) < 1e-12
        ops[axis] = a.tocsc()
        L = (kron(eye(ny),ops[0])+kron(ops[1],eye(nx))+reaction).tocsc()
        models[f'no_{label}_wrap_row_{row}'] = np.concatenate([splu(L+mu*identity).solve(modes @ W)[i+nx*j] for mu in s['mu_shifts']])
nshift, m, npulse = len(s['mu_shifts']), len(i), W.shape[1]
errors = {name:(y-Y).reshape(nshift,m,npulse) for name,y in models.items()}
worst, min_margin, wins = 0., np.inf, 0
for shift in range(nshift):
    for pulse in range(npulse):
        train_shift = np.arange(nshift) != shift
        train_pulse = np.arange(npulse) != pulse
        scores = sorted((np.sqrt(np.mean(r[train_shift][:,:,train_pulse]**2)), name) for name,r in errors.items())
        e, name = scores[0]
        wins += name == 'periodic'
        worst = max(worst, float(np.sqrt(np.mean(errors[name][shift,:,pulse]**2))))
        min_margin = min(min_margin, scores[1][0]/e)
print('Single-link ablation worst pulse-by-shift RMS:', {name:float(np.sqrt(np.mean(r*r,axis=1)).max()) for name,r in errors.items()})
print(f'ng_t04: full periodic won {wins}/{nshift*npulse} folds against every individual wrap-link deletion; worst held-out RMS={worst:.8g}; minimum runner-up/winner training RMS={min_margin:.4g}')
assert wins == nshift*npulse and worst < 1e-5
PY
OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/audit_wrap_links.py
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Single-link ablation worst pulse-by-shift RMS: {'periodic': 5.150570330353098e-06, 'no_x_wrap_row_0': 0.1056196505278028, 'no_x_wrap_row_13': 0.0871757356238687, 'no_y_wrap_row_0': 0.03999804645707423, 'no_y_wrap_row_11': 0.06813999528482068}
ng_t04: full periodic won 9/9 folds against every individual wrap-link deletion; worst held-out RMS=5.1505703e-06; minimum runner-up/winner training RMS=2016

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=24e8bd44b99097a5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_24e8bd44 (directory).
```

### Turn 13 — `submit_completed_task` (`15013 ms` LLM)
- **Rationale**: The unchanged bundle has a bound verification receipt and passed clean-process schema and dense-reference checks. The endpoint ablations support retaining all four wraparound links, but the host rejected the probe for formal closure of ng_t04.
- **Alternatives Weighed**: `Delete an individual directed endpoint link` (rejected: Each deletion increased the worst pulse-by-shift RMS to at least 0.0400, versus 5.15e-6 with all four links.)
- **Submitted Final Answer / Summary**: `Both solvers are saved and verified. Periodic upwinding on both axes won all 9 joint pulse-and-shift holdouts, with a worst held-out RMS of 5.15e-6. The public schema passed, and the largest error against the independent dense reference for the KKT certificate or terminal field was 2.34e-15. The host did not formally close the boundary finding.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
