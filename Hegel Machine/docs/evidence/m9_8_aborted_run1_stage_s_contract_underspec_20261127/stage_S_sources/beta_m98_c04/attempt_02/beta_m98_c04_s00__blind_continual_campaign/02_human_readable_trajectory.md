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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `377,099` (`361,631` / `136,192` / `15,468`) |
| **Total Episode Wall Time (s)** | `487.75s` |

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

### Turn 1 — `execute_in_remote_docker` (`16335 ms` LLM)
- **Rationale**: Inspect the public schema, calibration sweeps, and archive inventory to detect missing or differently encoded inputs that would change the solver design.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py; ls -la /tmp/hegel_continual_archive /pretrain_task; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT' 2>&1; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def show(v,level=0):
 if isinstance(v,dict):
  for k,x in v.items():
   print(' '*level,k,end=': '); show(x,level+2)
 elif isinstance(v,np.ndarray): print(v.shape,v.dtype,v.ravel()[:8])
 elif isinstance(v,(list,tuple)) and len(v)>0:
  print(type(v).__name__,len(v)); show(v[0],level+2)
 else: print(repr(v)[:400])
show(s)
PY
```
- **Container Exit Code**: `0` (`372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
edictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
list 1
   study:      Lx: 2.4
     Ly: 2.0
     nx: 14
     ny: 12
     kappa_x: (14,) float64 [0.41288044 0.44638463 0.47794168 0.50571761 0.52809818 0.54378271
 0.55185968 0.55185968]
     kappa_y: (12,) float64 [0.51912269 0.51426185 0.50007312 0.47770598 0.44897249 0.41620047
 0.38204491 0.34927289]
     vel_x: (14,) float64 [0.35       0.34500413 0.33015913 0.30588881 0.27288602 0.23209293
 0.18467409 0.13198319]
     vel_y: (12,) float64 [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907]
     robin_x: 0.9792023947175681
     robin_y: 1.1916051351743588
     V_reaction_2d: (14, 12) float64 [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25]
     mu_shifts: (3,) float64 [0.15 0.55 1.1 ]
     source_modes: (8, 14, 12) float64 [1.23101758e-06 1.79793522e-05 1.66125228e-04 9.71066356e-04
 3.59098833e-03 8.40099733e-03 1.24336738e-02 1.16417926e-02]
     sensor_ij: (32, 2) int64 [ 8  2  0  8 12  0 10  0]
     row_bank_id: (96,) int64 [0 1 2 3 0 1 2 3]
     sensor_calibration_sweeps:        ref_rosette: (80, 4) float64 [-1.2  0.   0.   0.  -0.4  0.   0.   0. ]
       raw_rosette: (80, 4) float64 [-1.20000235e+00  1.01314946e-05 -4.73745907e-06  1.63428058e-06
 -4.00003186e-01 -9.63808683e-06  1.10314040e-06  3.49588899e-06]
       single_bank_sweeps:          bank_id: (24,) int64 [0 0 0 0 0 0 1 1]
         ref_signal: (24,) float64 [-1.1 -0.6 -0.1  0.4  0.9  1.4 -1.1 -0.6]
         raw_readout: (24,) float64 [-1.0999889  -0.60000728 -0.09999984  0.40001076  0.90000378  1.40001666
 -1.10000517 -0.60000864]
     Z_nuisance: (96, 3) float64 [-1.66197023  0.42191104 -1.20390816 -0.11152306 -0.8442129   0.51979822
  0.10626283 -0.37960863]
     y_obs_raw: (96,) float64 [ 0.43789904 -0.06801007 -0.01291247  0.22783454  0.35114672 -0.08670025
  0.1930493   0.1836566 ]
     reference_tracer_pulses: list 2
         w_pulse: (8,) float64 [1.  0.  0.  0.5 0.  0.  0.  0. ]
         y_pulse_raw: (96,) float64 [0.07472316 0.06955358 0.01579633 0.02661024 0.17108078 0.0537509
 0.04186169 0.04904068]
     aux_manifold_telemetry: (24, 4) float64 [ 5.85296184e-01  3.43035241e-01  6.33648731e-01  8.30998239e-01
 -8.41083179e-01  5.10690736e-04  9.60906240e-01  1.06583510e+00]
     nuisance_ridge: 0.08
     source_ridge: 0.04
     u_box: 0.42
     total_budget: 1.4
```
- **Stderr (tail)**:
```text
ls: cannot access '/tmp/hegel_continual_archive': No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`34935 ms` LLM)
- **Rationale**: Measure the affine calibration residual and compare plausible finite-volume conventions against known tracer responses. A residual above the calibration noise after boundary fitting would identify a missing transport term rather than justify retuning source weights.
- **Alternatives Weighed**: `Fit weights and boundary physics jointly from the blind observation` (rejected: Known pulse weights allow transport errors to be distinguished from source-weight errors.)

```bash
cd /holdout_task; cat validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np,os
p=pickle.load(open('public_batch.pkl','rb')); print('wrapper:',[(list(a), {k:type(v).__name__ for k,v in a.items()}) for a in p]); s=p[0]['study']; print('pulse weights', [a['w_pulse'] for a in s['reference_tracer_pulses']]); print('sensor indices',s['sensor_ij'].tolist()); print('telemetry first six',s['aux_manifold_telemetry'][:6]); print('source sums',s['source_modes'].sum((1,2))); print('files',os.listdir('/pretrain_task')); print(open('public_asset_index.json').read()[-1300:]);
PY
cat > /tmp/transport_probe.py <<'PY'
import numpy as np,pickle,itertools
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; y=cal['raw_rosette']; cf=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; print('affine fit rmse',np.mean((np.c_[x,np.ones(len(x))]@cf-y)**2)**.5, 'map',cf.tolist())
def inv(y):return np.linalg.solve(cf[:4].T,(y.reshape(-1,4)-cf[4]).T).T.ravel()
nx=s['nx']; ny=s['ny']; obs=np.array([inv(a['y_pulse_raw']) for a in s['reference_tracer_pulses']]).T; w=np.array([a['w_pulse'] for a in s['reference_tracer_pulses']]); rhs=np.column_stack([a.ravel(order='F') for a in np.tensordot(w,s['source_modes'],axes=1)]); idx=s['sensor_ij']@np.array([1,nx]); Z=s['Z_nuisance']
def axis(ax,sp,avg,adv,robs):
 k=s['kappa_'+ax]; v=s['vel_'+ax]; n=len(k); h=s['L'+ax]/(n-sp); a,b=k[:-1],k[1:]; d={'arith':(a+b)/2,'harm':2*a*b/(a+b),'geom':np.sqrt(a*b),'left':a,'right':b}[avg]/h**2
 l=-d.copy(); u=-d.copy(); main=np.r_[d,0]+np.r_[0,d]; vf=(v[1:]+v[:-1])/2/h; vp=np.maximum(vf,0); vm=np.minimum(vf,0)
 if adv.startswith('flux'):
  l-=vp; u+=vm; main+=np.r_[vp,0]-np.r_[0,vm];
  if adv=='fluxout':main[0]+=max(-v[0],0)/h; main[-1]+=max(v[-1],0)/h
  if adv=='fluxfull':main[0]-=v[0]/h; main[-1]+=v[-1]/h
 elif adv.startswith('local'):
  v=(v if adv=='local' else np.r_[vf[0]*h,(vf[:-1]+vf[1:])*h/2,vf[-1]*h])/h; l-=np.maximum(v[1:],0); u+=np.minimum(v[:-1],0); main+=abs(v)
 elif adv=='noncon':
  l-=vp; u+=vm; main+=np.r_[0,vp]-np.r_[vm,0]
 main[[0,-1]]+=robs
 return diags([l,main,u],[-1,0,1],format='csc')
def pred(sp,avg,adv,r):
 a=kron(eye(ny),axis('x',sp,avg,adv,r[:2]))+kron(axis('y',sp,avg,adv,r[2:]),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); return np.concatenate([spsolve((a+mu*I).tocsc(),rhs)[idx] for mu in s['mu_shifts']]);
if __name__=='__main__':
 print('pulse RMS',np.mean(obs**2)**.5)
 results=[]
 for sp,avg,adv in itertools.product([0,1],['arith','harm','geom','left','right'],['flux','fluxout','fluxfull','local','localface','noncon','none']):
  f=lambda r:(pred(sp,avg,adv,r)-obs).ravel(); nom=np.repeat([s['robin_x'],s['robin_y']],2); fit=least_squares(f,nom,bounds=(0,100),gtol=2e-10); results.append((np.linalg.norm(fit.fun)/np.sqrt(obs.size),sp,avg,adv,fit.x.tolist(),np.linalg.norm(f(nom))/np.sqrt(obs.size)))
 for row in sorted(results)[:20]:print(row)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/transport_probe.py
```
- **Container Exit Code**: `0` (`8093 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
28543188001], 0.028427058325887256)
(5.858620757512267e-05, 1, 'arith', 'localface', [3.69086589896157, 4.598552245939622, 6.573819566362091, 5.342958953415031], 0.02854439088867116)
(6.60354572540863e-05, 1, 'geom', 'localface', [3.690760315868437, 4.604102130321366, 6.574045400402113, 5.345359087922957], 0.028544960161951037)
(7.492687021503815e-05, 1, 'harm', 'localface', [3.6906547550756104, 4.609663418657778, 6.574272448782696, 5.3477592770211055], 0.028545535045896138)
(0.00046510406699017544, 1, 'arith', 'noncon', [5.581580808406508, 5.72293540560386, 6.926529164989795, 6.8663472573321], 0.0575138036672152)
(0.00046878091157087127, 1, 'left', 'localface', [3.4317878032699203, 4.357116980610141, 6.791436214422827, 5.496579725919626], 0.02845826959445795)
(0.0004716366252607704, 1, 'geom', 'noncon', [5.58151146832943, 5.728576184455049, 6.926868722154969, 6.86880587774076], 0.05751092237347566)
(0.0004778041668545219, 1, 'left', 'local', [3.400384354053721, 4.215020534548086, 6.725114895684538, 5.52086197185913], 0.02833765895317542)
(0.0004781434076188633, 1, 'left', 'noncon', [5.318646320669879, 5.485580951677402, 7.1467770534400294, 7.025775394969305], 0.057174962110728464)
(0.00047842412208561167, 1, 'harm', 'noncon', [5.581442155111038, 5.734228753526806, 6.9272095798997855, 6.8712645494080045], 0.05750804141948578)
(0.0005092482547203002, 1, 'right', 'local', [3.920720549080465, 4.71627014783995, 6.2941600022289155, 5.215137207126922], 0.02853520605054837)
(0.0005239643693008181, 1, 'right', 'localface', [3.9535765282475035, 4.861723427270635, 6.356295340501703, 5.189086813719953], 0.028651779927105117)
(0.0008353762545239014, 1, 'right', 'noncon', [5.848463043477573, 5.981275251292015, 6.70620902841746, 6.706418609000269], 0.05785366944135706)
(0.004452498728174212, 0, 'right', 'none', [7.853211975538659, 6.306297927499897, 5.27834528530342, 10.983252176679558], 0.06603518689710788)
(0.004530475299285694, 0, 'right', 'noncon', [5.198272879583429, 5.760436613165075, 6.889355054502964, 5.118444543553693], 0.057080045744615046)
(0.004543751303220725, 0, 'harm', 'none', [7.569102533315271, 5.906942496010922, 5.517269740527946, 11.129781915731591], 0.06576684371169657)
(0.0045473079873246496, 0, 'geom', 'none', [7.567879900141021, 5.899565382940354, 5.517390236980254, 11.121466230683804], 0.06576302853619823)
(0.00455090835065202, 0, 'arith', 'none', [7.566658691332969, 5.892207254715951, 5.517512748768119, 11.113157395895612], 0.06575921829584831)
```

### Turn 3 — `execute_in_remote_docker` (`34622 ms` LLM)
- **Rationale**: Compare dimensionally plausible Robin conventions for the blind fallback and hold out tracer pulses, shifts, banks, and sensors. Unstable endpoint estimates would favor a shared boundary model rather than four independently fitted losses.
- **Alternatives Weighed**: `Use a more flexible interior correction` (rejected: The best discrete stencil already fits the tracer data near the measurement scale; held-out errors will determine whether any further correction is warranted.)
- **ATMS NOGOOD DECLARED**: Refuted `The two nominal Robin scalars can be inserted directly as identical endpoint diagonal terms on each axis.` | Witness: `Under the best tested interior stencil, freeing the four endpoint diagonals reduced public tracer RMSE from 2.84e-2 to 3.31e-5.` | Blanket Diagnosis: `The nominal Robin inputs do not directly encode the effective endpoint losses; the calibration sweep's 9.82e-6 RMSE rules out sensor error of comparable magnitude.` | Sublated Pivot: `Retain the arithmetic-diffusion, node-upwind interior stencil and distinguish a spacing-aware nominal prior from pulse-identified endpoint losses.`

```bash
cat > /tmp/boundary_probe.py <<'PY'
from transport_probe import *
np.set_printoptions(precision=7,suppress=True)
print('endpoint data [spacing, robin, k_left, k_right, v_left, v_right]')
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); print(ax,h,s['robin_'+ax],s['kappa_'+ax][[0,-1]],s['vel_'+ax][[0,-1]])
print('telemetry column mean/std/range/correlation', np.mean(s['aux_manifold_telemetry'],0), np.std(s['aux_manifold_telemetry'],0), np.ptp(s['aux_manifold_telemetry'],0),np.corrcoef(s['aux_manifold_telemetry'].T))
N=len(idx); fun=lambda r:pred(1,'arith','local',r)-obs
fit=lambda mask,init:least_squares(lambda r:fun(r)[mask].ravel(),init,bounds=(0,np.inf),gtol=1e-12,ftol=1e-12,xtol=1e-12).x
nom=[]; series=[]; kapp=[]; adv=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); k=s['kappa_'+ax][[0,-1]]; v=s['vel_'+ax][[0,-1]]; r=s['robin_'+ax]; nom.extend([r/h]*2); series.extend(r/h/(1+r*h/(2*k))); kapp.extend(k*r/h); adv.extend(np.maximum(v*[1,-1],0)/h)
r=fit(np.ones(obs.shape,dtype=bool),nom); print('fit',r,'shift/pulse RMSE',np.sqrt(np.mean(fun(r).reshape(3,N,2)**2,axis=1)))
for name,b in [('nom/h',nom),('nom/h - inflow advection',np.array(nom)-adv),('series',series),('series - inflow advection',np.array(series)-adv),('k*nom/h',kapp)]:
 print(name,np.array(b),np.sqrt(np.mean(fun(b)**2)))
for label, groups in [('pulse',np.broadcast_to(np.arange(2),obs.shape)),('shift',np.broadcast_to(np.repeat(np.arange(3),N)[:,None],obs.shape)),('bank',np.broadcast_to(s['row_bank_id'][:,None],obs.shape)),('sensor',np.broadcast_to(np.tile(np.arange(N),3)[:,None],obs.shape))]:
 errors=[]; coefs=[]
 for g in np.unique(groups):
  test=groups==g; rb=fit(~test,nom); errors.append(np.sqrt(np.mean(fun(rb)[test]**2))); coefs.append(rb)
 print('held out',label,'worst RMSE',max(errors),'coefficient ranges',np.ptp(coefs,axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/boundary_probe.py
```
- **Container Exit Code**: `0` (`3539 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine fit rmse 9.819398601636478e-06 map [[0.9999996733919401, 7.953826292032246e-07, -7.192139983775168e-07, -1.269964336635818e-06], [-6.995320862928982e-07, 1.0000004115106125, 1.4039909790319882e-08, 7.762418029466644e-07], [5.23998041620725e-07, 2.322919654850532e-06, 0.9999995241526886, -6.159827457815387e-07], [5.462865582978264e-07, -8.363803528866776e-07, -2.0515250773561252e-06, 1.0000000752143174], [4.631504093274041e-07, -5.027309864390662e-07, 1.2009784958492569e-06, 6.71427024194533e-07]]
endpoint data [spacing, robin, k_left, k_right, v_left, v_right]
x 0.1846153846153846 0.9792023947175681 [0.4128804 0.4128804] [ 0.35      -0.2057248]
y 0.18181818181818182 1.1916051351743588 [0.5191227 0.2791227] [-0.        -0.2265248]
telemetry column mean/std/range/correlation [ 0.1223978  0.2117698 -0.0134144  0.1103221] [1.1302839 1.0424817 0.941349  0.9282931] [3.6911461 4.5887673 3.9696059 3.4462604] [[ 1.         0.5079531 -0.266135  -0.0719559]
 [ 0.5079531  1.        -0.1500155 -0.0838539]
 [-0.266135  -0.1500155  1.         0.0710725]
 [-0.0719559 -0.0838539  0.0710725  1.       ]]
fit [3.6587525 4.454899  6.509586  5.3681338] shift/pulse RMSE [[0.0000289 0.0000413]
 [0.0000354 0.0000142]
 [0.000044  0.0000253]]
nom/h [5.304013  5.304013  6.5538282 6.5538282] 0.004057121992099224
nom/h - inflow advection [3.4081796 4.1896701 6.5538282 5.3079421] 0.000626303952258766
series [4.3514026 4.3514026 5.4223263 4.7214354] 0.0016330269149477882
series - inflow advection [2.4555692 3.2370597 5.4223263 3.4755492] 0.006644208926351628
k*nom/h [2.1899232 2.1899232 3.402241  1.8293222] 0.014573845646611844
held out pulse worst RMSE 4.259136890606676e-05 coefficient ranges [0.0125155 0.0021774 0.0047581 0.0043494]
held out shift worst RMSE 6.291043071552219e-05 coefficient ranges [0.0138004 0.0193349 0.0032136 0.0015135]
held out bank worst RMSE 5.383186022803386e-05 coefficient ranges [0.0012913 0.002448  0.0050897 0.0011373]
held out sensor worst RMSE 0.00012019253444145129 coefficient ranges [0.0058103 0.0017088 0.0042679 0.0024251]
```

### Turn 4 — `execute_in_remote_docker` (`39059 ms` LLM)
- **Rationale**: The pulse fits establish a stable transport stencil. Compare blind fits with the nominal and pulse-inferred boundaries to see whether observation-only corrections recover the physical boundary or merely absorb nuisance residuals. Check calibration sweep coverage before choosing the dynamic inversion basis.
- **Alternatives Weighed**: `Fit four boundary losses from the blind source observation` (rejected: Source weights and boundary losses can compensate for each other; first test whether a smaller two-axis correction improves held-out observations and recovers the pulse-inferred losses.)

```bash
cat > /tmp/blind_probe.py <<'PY'
from transport_probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
P=Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
R=np.eye(len(Z))-P
y=inv(s['y_obs_raw']); yp=R@y
rp=[]; boundary_h=[]; ad=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); v=s['vel_'+ax][[0,-1]]
 rp+= [s['robin_'+ax]]*2; boundary_h += [h]*2; ad.extend(np.maximum(v*[1,-1],0)/h)
rp=np.array(rp); boundary_h=np.array(boundary_h); ad=np.array(ad)
from transport_probe import __dict__ as probe
probe['rhs']=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); nom=rp/boundary_h-ad
pulsefit=least_squares(lambda b:(pred(1,'arith','local',b)@w.T-obs).ravel(),nom,gtol=1e-13).x
print('effective Robin',((pulsefit+ad)*boundary_h).tolist(), 'nominal',rp.tolist())
print('nuisance-free pulse residuals by pulse, mean/std',np.mean(pred(1,'arith','local',pulsefit)@w.T-obs,axis=0),np.std(pred(1,'arith','local',pulsefit)@w.T-obs,axis=0))
for b in [nom,pulsefit]:
 X=R@pred(1,'arith','local',b); print('S projected sv',np.linalg.svd(X,compute_uv=False));
 for ridge in [0,s['source_ridge']]:
  H=X.T@X+ridge*np.eye(len(w[0])); c=X.T@yp
  res=minimize(lambda q:(q@H@q/2-q@c,H@q-c),np.full(len(c),s['total_budget']/len(c)),jac=True,method='SLSQP',bounds=[(0,s['u_box'])]*len(c),constraints=[{'type':'eq','fun':lambda q:sum(q)-s['total_budget'],'jac':lambda q:np.ones(len(c))}],options={'ftol':1e-13}); print('nom/pulse',b,'ridge',ridge,'weights',res.x.round(6),'rmse',np.mean((X@res.x-yp)**2)**.5)
for reg in [0,0.01,0.05,0.2]:
 # Hold out complete sensor groups across shifts to diagnose blind correction.
 X0=pred(1,'arith','local',nom); out=[]
 for test in [np.zeros(len(Z),bool)]+[np.tile(np.arange(len(idx))%4==i,3) for i in range(4)]:
  tr=~test; zz=Z[tr]; Zinv=np.linalg.solve(zz.T@zz+0.000001*np.eye(zz.shape[1]),zz.T); test=test if np.any(test) else tr
  def fun(q):
   bb=nom+np.repeat(q[-2:],2)/boundary_h; residual=(pred(1,'arith','local',bb)@q[:-2]-y); g=Zinv@residual[tr]; a=residual[tr]-zz@g; return (a@a+reg*(q[-2:]@q[-2:]))/2
  b0=np.r_[np.full(w.shape[1],s['total_budget']/w.shape[1]),0,0]; eq={'type':'eq','fun':lambda q:sum(q[:-2])-s['total_budget']}; r=minimize(fun,b0,method='SLSQP',constraints=[eq],bounds=[(0,s['u_box'])]*w.shape[1]+[(-.45,.6)]*2,options={'ftol':1e-12,'maxiter':120}); residual=pred(1,'arith','local',nom+np.repeat(r.x[-2:],2)/boundary_h)@r.x[:-2]-y; out.append((r.x[-2:].round(5).tolist(),float(np.mean((residual[test]-Z[test]@(Zinv@residual[tr]))**2)**.5)))
 print('blind correction regularizer',reg,'full and held-out:',out)
print('calibration row sparsity',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('max affine residual',np.max(abs(np.c_[x,np.ones(len(x))]@cf-y),axis=0)); print('paired sweep span',np.ptp(x,axis=0)); print('bank sweep residual versus affine',np.std(cal['single_bank_sweeps']['raw_readout']-cal['single_bank_sweeps']['ref_signal']))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/blind_probe.py
```
- **Container Exit Code**: `1` (`26053 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.0515250773561252e-06, 1.0000000752143174], [4.631504093274041e-07, -5.027309864390662e-07, 1.2009784958492569e-06, 6.71427024194533e-07]]
effective Robin [1.025462001749499, 1.0281677305633903, 1.1835610897465518, 1.2025490787352546] nominal [0.9792023947175681, 0.9792023947175681, 1.1916051351743588, 1.1916051351743588]
nuisance-free pulse residuals by pulse, mean/std [ 1.41093587e-05 -2.82919854e-06] [3.37768785e-05 2.90123149e-05]
S projected sv [1.45114455 0.66701836 0.54695696 0.37422225 0.16773934 0.14062577
 0.104518   0.02499028]
nom/pulse [3.40817964 4.1896701  6.55382824 5.30794207] ridge 0 weights [0.113744 0.175034 0.197238 0.289138 0.095216 0.008899 0.26726  0.253471] rmse 0.0068430616625622205
nom/pulse [3.40817964 4.1896701  6.55382824 5.30794207] ridge 0.04 weights [0.163958 0.16817  0.182327 0.185843 0.157735 0.047981 0.246745 0.24724 ] rmse 0.0071323311643628675
S projected sv [1.44352977 0.66518311 0.54368342 0.37306234 0.16690183 0.14014184
 0.10417814 0.02493562]
nom/pulse [3.65875251 4.454899   6.50958599 5.36813376] ridge 0 weights [0.12544  0.17174  0.195522 0.281256 0.098995 0.009754 0.265869 0.251424] rmse 0.006801559830712815
nom/pulse [3.65875251 4.454899   6.50958599 5.36813376] ridge 0.04 weights [0.165731 0.169655 0.181958 0.184555 0.158965 0.048405 0.245125 0.245606] rmse 0.007074832446227272
blind correction regularizer 0 full and held-out: [([0.06029, 0.05174], 0.006768779548719281), ([0.07537, -0.0154], 0.008697837055946544), ([-0.13598, 0.15747], 0.006344720058780718), ([0.08141, 0.18754], 0.008311207148521751), ([0.11915, -0.00637], 0.008038797882890943)]
blind correction regularizer 0.01 full and held-out: [([0.03061, 0.03534], 0.006780029817217694), ([0.01849, 0.00889], 0.008759911366367437), ([0.00205, 0.03717], 0.005696088885038801), ([0.05169, 0.07117], 0.007380296366961749), ([0.03357, 0.01913], 0.007971650510724295)]
blind correction regularizer 0.05 full and held-out: [([0.01183, 0.01432], 0.006810044694935875), ([0.00576, 0.00402], 0.008812268422254585), ([0.00277, 0.01216], 0.005726623704438207), ([0.01797, 0.02348], 0.0069781586434680305), ([0.01057, 0.00768], 0.007992239677084454)]
blind correction regularizer 0.2 full and held-out: [([0.00364, 0.00446], 0.0068303838894965005), ([0.00165, 0.00124], 0.008834491163278434), ([0.00094, 0.00352], 0.00575011677416213), ([0.00526, 0.00679], 0.006857406246683195), ([0.00302, 0.00231], 0.008006989629364485)]
calibration row sparsity (array([1, 4]), array([16, 64]))
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/blind_probe.py", line 31, in <module>
    print('calibration row sparsity',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('max affine residual',np.max(abs(np.c_[x,np.ones(len(x))]@cf-y),axis=0)); print('paired sweep span',np.ptp(x,axis=0)); print('bank sweep residual versus affine',np.std(cal['single_bank_sweeps']['raw_readout']-cal['single_bank_sweeps']['ref_signal']))
                                                                                                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~^~
ValueError: operands could not be broadcast together with shapes (80,4) (96,)
```

### Turn 5 — `execute_in_remote_docker` (`63859 ms` LLM)
- **Rationale**: Implement the pulse-validated stencil, nominal blind fallback, dynamic calibration, and exact nuisance-projected QP. The schema run detects interface or shape failures before numerical stress testing; the archive listing determines whether archive-based inference can be tested.
- **Alternatives Weighed**: `Infer unconstrained boundary corrections from the blind observation` (rejected: The fitted x-axis correction changes sign across sensor folds, so the small pooled residual improvement does not establish a transferable correction.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated steady transport and a certified box-simplex source fit."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibration(study):
    sweep = study['sensor_calibration_sweeps']
    ref = np.asarray(sweep['ref_rosette'])
    raw = np.asarray(sweep['raw_rosette'])
    coef = np.linalg.lstsq(np.column_stack((ref, np.ones(len(ref)))), raw, rcond=None)[0]

    def decode(values):
        groups = np.asarray(values).reshape(-1, 4)
        return np.linalg.solve(coef[:4].T, (groups - coef[4]).T).T.ravel()

    return decode


class _Transport:
    def __init__(self, study):
        self.nx, self.ny = int(study['nx']), int(study['ny'])
        self.shifts = study['mu_shifts']
        sensors = np.asarray(study['sensor_ij'])
        self.sensors = sensors[:, 0] + self.nx * sensors[:, 1]
        operators, nominal, faces = [], [], []
        for ax in 'xy':
            k = np.asarray(study['kappa_' + ax])
            h = study['L' + ax] / (len(k) - 1)
            v = np.asarray(study['vel_' + ax]) / h
            d = (k[:-1] + k[1:]) / (2 * h**2)
            lo = -d - np.maximum(v[1:], 0)
            hi = -d + np.minimum(v[:-1], 0)
            center = np.r_[d, 0.] + np.r_[0., d] + np.abs(v)
            center[[0, -1]] -= np.maximum(v[[0, -1]] * [1, -1], 0)
            operators.append(diags((lo, center, hi), (-1, 0, 1)))
            nominal.extend([study['robin_' + ax] / h] * 2)
            endpoints = np.zeros((len(k), 2))
            endpoints[0, 0] = endpoints[-1, 1] = 1
            faces.append(endpoints)
        self.nominal = np.array(nominal)
        self.faces = np.column_stack((np.tile(faces[0], (self.ny, 1)),
                                      np.repeat(faces[1], self.nx, axis=0)))
        self.base = (kron(eye(self.ny), operators[0]) + kron(operators[1], eye(self.nx))
                     + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
        self.sources = np.column_stack([f.ravel(order='F') for f in study['source_modes']])

    def response(self, losses, rhs, with_jac=False):
        base = self.base + diags(self.faces @ losses)
        readings, deriv = [], []
        for shift in self.shifts:
            lu = splu((base + diags(np.full(base.shape[0], shift))).tocsc())
            field = lu.solve(rhs)
            readings.append(field[self.sensors])
            if with_jac:
                drhs = (-field[:, :, None] * self.faces[:, None, :]).reshape(len(field), -1)
                deriv.append(lu.solve(drhs)[self.sensors].reshape(-1, 4))
        readings = np.concatenate(readings, axis=0)
        return readings, field, np.concatenate(deriv, axis=0) if with_jac else None


def _pulse_losses(transport, pulses, decode):
    rhs = transport.sources @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([decode(p['y_pulse_raw']) for p in pulses])
    last, cached = None, None

    def evaluate(loss):
        nonlocal last, cached
        if last is None or not np.array_equal(loss, last):
            signals, _, jac = transport.response(loss, rhs, with_jac=True)
            cached = (signals - observed).ravel(), jac
            last = loss.copy()
        return cached

    fit = least_squares(lambda r: evaluate(r)[0], np.maximum(transport.nominal, 0),
                        jac=lambda r: evaluate(r)[1], bounds=(0, np.inf),
                        ftol=1e-11, xtol=1e-11, gtol=1e-12)
    return fit.x


def _box_simplex(H, c, cap, budget):
    """Feasible active-set Newton, including the all-at-bounds case."""
    n = len(c)
    w = cap * (budget / cap.sum()) if cap.sum() else np.zeros(n)
    if budget <= 0:
        return np.zeros(n), float(np.max(c))
    if budget >= cap.sum():
        return cap.copy(), -float(np.max(H @ cap - c))
    active = np.zeros(n, dtype=np.int8)
    active[cap == 0] = -1
    tol = 1e-12 * max(1., float(np.max(np.abs(c))), float(np.max(np.abs(H @ cap))))
    for _ in range(50*n + 50):
        free = np.flatnonzero(active == 0)
        grad = H @ w - c
        if free.size == 0:
            lo = np.flatnonzero((active == -1) & (cap > 0))
            hi = np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j] - grad[i] <= tol:
                return w, -float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue
        w_star, inv_one = np.linalg.solve(H[np.ix_(free, free)], np.column_stack((
            c[free] - H[np.ix_(free, active != 0)] @ w[active != 0], np.ones(free.size)))).T
        nu = (w_star.sum() - (budget - w[active != 0].sum())) / inv_one.sum()
        w_star -= nu * inv_one
        old = w[free]
        step = w_star - old
        alpha = np.ones(free.size)
        down = (w_star < -1e-13) & (step < 0)
        up = (w_star > cap[free] + 1e-13) & (step > 0)
        alpha[down] = -old[down] / step[down]
        alpha[up] = (cap[free][up] - old[up]) / step[up]
        k = np.argmin(alpha)
        if alpha[k] < 1:
            w[free] = old + alpha[k] * step
            i = free[k]
            active[i] = 1 if up[k] else -1
            w[i] = cap[i] if up[k] else 0.
            continue
        w[free] = np.clip(w_star, 0., cap[free])
        q = H @ w - c + nu
        viol = np.where(cap > 0, q * active, 0.)
        i = np.argmax(viol)
        if viol[i] <= tol:
            return w, float(nu)
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _source_fit(study, S, y):
    Z = np.asarray(study['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    X = S - Z @ (Pz @ S)
    target = y - Z @ (Pz @ y)
    n = X.shape[1]
    ridge = study['source_ridge']
    H = X.T @ X + ridge * np.eye(n)
    c = X.T @ target
    cap = np.broadcast_to(np.asarray(study['u_box'], dtype=float), (n,))
    budget = float(study['total_budget'])
    w, nu = _box_simplex(H, c, cap, budget)
    q = H @ w - c + nu
    lower = np.where(w <= 0., np.maximum(q, 0.), 0.)
    upper = np.where(w >= cap, np.maximum(-q, 0.), 0.)
    stationarity = q - lower + upper
    residual = X @ w - target
    primal = (residual @ residual + ridge * (w @ w)) / 2
    dual = (primal + nu * (w.sum() - budget) - lower @ w + upper @ (w - cap)
            - stationarity @ np.linalg.solve(H, stationarity) / 2)
    return w, {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
               'primal_objective': float(primal), 'dual_lower_bound': float(min(primal, dual))}


def solve_batch(study: dict) -> dict:
    decode = _calibration(study)
    transport = _Transport(study)
    pulses = study.get('reference_tracer_pulses')
    loss = _pulse_losses(transport, pulses, decode) if pulses else transport.nominal
    signals, terminal, _ = transport.response(loss, transport.sources)
    w, certificate = _source_fit(study, signals, decode(study['y_obs_raw']))
    return {'weights': w, 'terminal_field': (terminal @ w).reshape((transport.nx, transport.ny), order='F'),
            'kkt_certificate': certificate}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
ls -la /tmp/hegel_continual_archive 2>/dev/null || true
```
- **Container Exit Code**: `0` (`447 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`35074 ms` LLM)
- **Rationale**: Test the saved solver for failures the schema cannot detect: transport mismatch on tracer slices, inconsistent terminal-field assembly, and invalid certificates at active bounds or near extreme budgets. Inspect the single-bank sweep layout to determine whether it can independently validate calibration.

```bash
cat > /tmp/verify_batch.py <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import minimize
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s)
p=s['reference_tracer_pulses']; b=m._pulse_losses(t,p,decode)
W=np.stack([i['w_pulse'] for i in p],axis=1); y=np.stack([decode(i['y_pulse_raw']) for i in p],axis=1)
print('tracer RMSE by shift and pulse, nominal then fitted')
for loss in (t.nominal,b):
    r=t.response(loss,t.sources@W)[0]-y
    print(np.sqrt(np.mean(r.reshape(len(t.shifts),len(t.sensors),len(p))**2,axis=1)))
for pulsed in (True,False):
    study=dict(s,reference_tracer_pulses=p if pulsed else None)
    out=m.solve_batch(study); w=out['weights']; cert=out['kkt_certificate'];lo=cert['lam_lower']; hi=cert['lam_upper']; u=s['u_box']
    S,U,_=t.response(b if pulsed else t.nominal,t.sources)
    Y=decode(s['y_obs_raw']); Z=s['Z_nuisance']; R=np.eye(len(Y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    X=R@S; Y=R@Y; ridge=s['source_ridge']; grad=X.T@(X@w-Y)+ridge*w
    kkt=np.max(np.abs(grad+cert['nu_sum']-lo+hi)); gap=cert['primal_objective']-cert['dual_lower_bound']
    field_error=np.max(abs(out['terminal_field']-(U@w).reshape(t.nx,t.ny,order='F')))
    assert np.all(lo>=0) and np.all(hi>=0) and np.all(w>=0) and np.all(w<=u)
    assert np.array_equal(cert['weights'],w) and abs(w.sum()-s['total_budget'])<1e-12
    assert kkt<1e-11 and field_error<1e-12 and 0<=gap<1e-12 and max(abs(lo@w),abs(hi@(u-w)))<1e-12
    assert abs(cert['primal_objective']-0.5*(np.linalg.norm(X@w-Y)**2+ridge*(w@w)))<1e-12
    print('pulsed',pulsed,'KKT',kkt,'duality gap',gap,'terminal field max error',field_error)
rng=np.random.default_rng(1482); worst=np.zeros(4); counts=np.zeros(3,dtype=int)
for i in range(1800):
    n=int(rng.integers(2,34)); a=rng.normal(size=(n,n)); H=a.T@a/n+10**rng.uniform(-5,0)*np.eye(n); c=rng.normal(size=n)*10**rng.uniform(-1,2); cap=rng.choice([0.,.1,.3,.7,1.],n); budget=cap.sum()*rng.choice([0.,1.,1e-9,1-1e-9,rng.uniform()]) if i%3 else cap[rng.random(n)<.5].sum()
    w,nu=m._box_simplex(H,c,cap,budget); q=H@w-c+nu
    L=(w<=1e-13)&(cap>0); U=(w>=cap-1e-13)&(cap>0); F=~(L|U)&(cap>0)
    counts += [L.any(),U.any(),F.any()]; errors=[abs(w.sum()-budget),max(0.,np.max(-q[L],initial=0)),max(0.,np.max(q[U],initial=0)),np.max(abs(q[F]),initial=0)]
    worst=np.maximum(worst,errors)
    assert min(w)>=-1e-13 and max(w-cap)<=1e-13
assert max(worst)<1e-8
print('QP stress worst: budget, lower, upper, free',worst,'slice counts',counts)
cal=s['sensor_calibration_sweeps']; print('sweep fields',list(cal)); print('single bank fields and dimensions',[(k,np.shape(v),np.asarray(v).ravel()[:8].tolist()) for k,v in cal['single_bank_sweeps'].items()]); print('study fields',list(s)); print('banks and counts',np.unique(s['row_bank_id'],return_counts=True))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1767 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tracer RMSE by shift and pulse, nominal then fitted
[[0.00080835 0.00086226]
 [0.00054649 0.00060024]
 [0.00036039 0.00040962]]
[[2.89303300e-05 4.13458170e-05]
 [3.53522384e-05 1.41816178e-05]
 [4.39671447e-05 2.52697741e-05]]
pulsed True KKT 1.9949319973733282e-16 duality gap 0.0 terminal field max error 0.0
pulsed False KKT 1.0755285551056204e-16 duality gap 0.0 terminal field max error 0.0
QP stress worst: budget, lower, upper, free [3.21964677e-14 8.88178420e-16 3.55271368e-15 2.84217094e-14] slice counts [1209 1260 1244]
sweep fields ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
single bank fields and dimensions [('bank_id', (24,), [0, 0, 0, 0, 0, 0, 1, 1]), ('ref_signal', (24,), [-1.1, -0.6, -0.1, 0.4, 0.9, 1.4, -1.1, -0.6]), ('raw_readout', (24,), [-1.0999888975238197, -0.6000072805944341, -0.09999984021354792, 0.4000107609960614, 0.9000037794139513, 1.4000166613613358, -1.100005169075844, -0.6000086406454542])]
study fields ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'reference_tracer_pulses', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
banks and counts (array([0, 1, 2, 3]), array([24, 24, 24, 24]))
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`26463 ms` LLM)
- **Rationale**: Compare held-out mixed-channel and single-channel sweeps, including the separate bank measurements. A reproducible nonlinear gain would change the calibration model; otherwise retain the verified affine solver.
- **Alternatives Weighed**: `Always fit a nonlinear rosette model` (rejected: Additional terms can fit calibration noise and worsen inversion on unseen rosettes.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; single=np.count_nonzero(x,axis=1)==1; folds=np.arange(len(x))%5
terms=[t for d in range(4) for t in combinations_with_replacement(range(4),d)]
A=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); banks=c['single_bank_sweeps']; xr=np.eye(4)[banks['bank_id']]*banks['ref_signal'][:,None]; Ar=np.column_stack([np.prod(xr[:,t],axis=1) for t in terms]); bank=banks['bank_id']; raw=banks['raw_readout']
for name,mask in [('affine',[len(t)<=1 for t in terms]),('quadratic',[len(t)<=2 for t in terms]),('quadratic + univariate cubes',[len(t)<=2 or len(set(t))==1 for t in terms]),('cubic',[True]*len(terms))]:
 a=A[:,mask]; e=np.empty_like(y)
 for i in range(5):
  tr=folds!=i; e[~tr]=a[~tr]@np.linalg.lstsq(a[tr],y[tr],rcond=None)[0]-y[~tr]
 b=np.linalg.lstsq(a,y,rcond=None)[0]; er=(Ar[:,mask]@b)[np.arange(len(bank)),bank]-raw
 print(name,'OOF RMS [joint,single]',*[float(np.sqrt(np.mean(e[g]**2))) for g in [~single,single]],'worst bank RMS',max(np.sqrt(np.mean(er[bank==i]**2)) for i in np.unique(bank)),'condition',np.linalg.cond(a))
PY
```
- **Container Exit Code**: `0` (`308 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine OOF RMS [joint,single] 1.0722343292228115e-05 9.813647885067929e-06 worst bank RMS 1.0991370866448358e-05 condition 1.5366020570685035
quadratic OOF RMS [joint,single] 1.2205970762774676e-05 1.0113440999701547e-05 worst bank RMS 1.0428962693562423e-05 condition 5.318448837622633
quadratic + univariate cubes OOF RMS [joint,single] 1.3193136565501043e-05 1.0062726799561683e-05 worst bank RMS 1.023303377537434e-05 condition 12.33022634215986
cubic OOF RMS [joint,single] 2.1378561398671773e-05 1.2750235331983343e-05 worst bank RMS 1.017452232118469e-05 condition 21.142756402527187

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`26901 ms` LLM)
- **Rationale**: Check whether an apparatus log exposes a usable blind-campaign prior, and whether nuisance contamination of diagnostic pulses materially biases inferred losses. The held-out pulse test determines whether profiling improves prediction rather than just reducing the fitted residual.
- **Alternatives Weighed**: `Replace the affine calibration with an unconditional nonlinear fit` (rejected: Full cubic calibration doubled the held-out mixed-rosette RMSE on the public sweeps.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import json,os,pickle
p=json.load(open('public_asset_index.json')); print('Index keys:',list(p)); print('Non-file metadata:',{k:v for k,v in p.items() if k!='files'}); print('Indexed files:',list(p.get('files',{}))); a='/tmp/hegel_continual_archive'; print('Archive entries:',os.listdir(a) if os.path.isdir(a) else 'not mounted'); log=a+'/archive_apparatus_log.json';
if os.path.isfile(log):
 with open(log) as f: print('Apparatus log sample:',f.read(3500))
p=pickle.load(open('public_batch.pkl','rb')); print('Public cases:',len(p)); print('Wrapper metadata:',[{k:v for k,v in c.items() if k!='study'} for c in p]);
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); d=m._calibration(s); ps=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in ps],axis=1); rhs=t.sources@W; y=np.stack([d(p['y_pulse_raw']) for p in ps],axis=1); full=m._pulse_losses(t,ps,d); Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0]; project=lambda v:v-Q@(Q.T@v)
fit=lambda k:least_squares(lambda r:project(t.response(r,rhs)[0]-y)[:,k].ravel(),t.nominal,bounds=(0,np.inf),gtol=1e-12).x
for i in range(len(ps)):
 loss=fit([j for j in range(len(ps)) if j!=i]); r=t.response(loss,rhs)[0]-y; print('Nuisance-profiled held-out pulse',i,'raw RMS',np.sqrt(np.mean(r[:,i]**2)),'profiled RMS',np.sqrt(np.mean(project(r)[:,i]**2)))
for label,loss in [('nominal',t.nominal),('pulse unchanged',full),('pulse nuisance-profiled',fit(list(range(len(ps)))) )]:
 S,U,_=t.response(loss,t.sources); w,_=m._source_fit(s,S,d(s['y_obs_raw'])); print(label,'endpoint loss',loss.round(6),'weights',w.round(6),'terminal norm',np.linalg.norm(U@w))
PY
```
- **Container Exit Code**: `0` (`586 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Index keys: ['files', 'schema']
Non-file metadata: {'schema': 'hegel_public_asset_index_v1'}
Indexed files: ['/holdout_task/public_batch.pkl', '/holdout_task/validate_public_schema.py']
Archive entries: not mounted
Public cases: 1
Wrapper metadata: [{}]
Nuisance-profiled held-out pulse 0 raw RMS 4.271744009985129e-05 profiled RMS 4.2238816757633204e-05
Nuisance-profiled held-out pulse 1 raw RMS 4.0599000451360136e-05 profiled RMS 4.0092931458565066e-05
nominal endpoint loss [5.304013 5.304013 6.553828 6.553828] weights [0.163958 0.16817  0.182327 0.185843 0.157735 0.047982 0.246744 0.24724 ] terminal norm 0.7225247651734732
pulse unchanged endpoint loss [5.554586 5.569242 6.509586 6.61402 ] weights [0.165727 0.16966  0.181956 0.184555 0.158965 0.048405 0.245125 0.245606] terminal norm 0.720919248076378
pulse nuisance-profiled endpoint loss [5.554801 5.567022 6.510168 6.613619] weights [0.165723 0.169656 0.18197  0.184556 0.15897  0.048392 0.245119 0.245614] terminal norm 0.720924535558778

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 9 — `submit_completed_task` (`17422 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process schema validation and numerical checks for both pulse and no-pulse calls. The tracer data support fitted endpoint losses, but do not establish a transferable correction for blind campaigns.
- **Alternatives Weighed**: `Use nonlinear rosette calibration or profile nuisance terms in tracer pulses` (rejected: Nonlinear calibration worsened held-out mixed-rosette error; pulse nuisance profiling did not materially improve held-out prediction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:7176'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed, and the worst KKT residual across 1,800 stress cases was 3.22e-14. Without tracer pulses, the solver uses the nominal Robin boundary losses; no apparatus archive was available to validate a better fallback.`

### Turn 10 — `execute_in_remote_docker` (`30122 ms` LLM)
- **Rationale**: Compare a rational Robin resistance, shared axis terms, and endpoint pair differences on held-out public tracer slices. A physically restrained model with lower worst-slice error would replace the independent endpoint fit; bank residuals can reveal whether nuisance projection hides a failing bank.
- **Alternatives Weighed**: `Run relay-mode, HJB, or balanced-accuracy tests` (rejected: This study is a steady linear transport inverse problem with neither a relay state nor classification labels; bank-stratified residuals are the relevant nuisance check.)

```bash
cat > /tmp/gate_audit.py <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s); ps=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in ps]); rhs=t.sources@W; y=np.column_stack([decode(p['y_pulse_raw']) for p in ps]); full=m._pulse_losses(t,ps,decode)
params=[]; names=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); params.append((h,np.asarray(s['kappa_'+ax])[[0,-1]],s['robin_'+ax])); names.extend([ax+'-',ax+'+'])
# Robin resistance in series with a signed effective exchange length; also check shared axis corrections.
def resistance(q):
 return np.concatenate([r/h/(1+q[i]*r*h/(2*k)) for i,(h,k,r) in enumerate(params)])
def plain(q): return t.nominal+np.repeat(q,2)
def antisym(q): return t.nominal+np.array([q[0]+q[2],q[0]-q[2],q[1]+q[3],q[1]-q[3]])
banks=s['row_bank_id']; Z=s['Z_nuisance']; shifts=np.repeat(np.arange(len(t.shifts)),len(t.sensors)); groups=[banks,shifts,np.tile(np.arange(len(t.sensors)),len(t.shifts))]; print('endpoint correction',dict(zip(names,(full-t.nominal).tolist())))
for label,model,num in [('nominal',lambda _:t.nominal,0),('series resistance',resistance,2),('axis shared',plain,2),('axis plus pair differences',antisym,4)]:
 def fit(tr,cols):
  if not num:return model([])
  return model(least_squares(lambda q:(t.response(model(q),rhs)[0]-y)[tr][:,cols].ravel(),np.zeros(num),bounds=(-1.,1.) if label=='series resistance' else (-3.,3.),gtol=1e-12).x)
 worst=[]
 for group in groups:
  vals=[]
  for g in np.unique(group):
   te=group==g; loss=fit(~te,slice(None)); vals.append(np.sqrt(np.mean((t.response(loss,rhs)[0]-y)[te]**2)))
  worst.append(max(vals))
 worst.append(max(np.sqrt(np.mean((t.response(fit(np.ones(len(y),bool),[1-i]),rhs)[0]-y)[:,i]**2)) for i in range(2)))
 print(label,'worst held-out bank/shift/sensor/pulse',np.array(worst))
raw=decode(s['y_obs_raw']); S,U,_=t.response(full,t.sources); w,_=m._source_fit(s,S,raw); before=raw-S@w; after=before-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@before)
for g in np.unique(banks):
 a=banks==g; print('bank',g,'residual RMS before/after nuisance',np.sqrt(np.mean(before[a]**2)),np.sqrt(np.mean(after[a]**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/gate_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`4038 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
endpoint correction {'x-': 0.25057287202187606, 'x+': 0.26522890492576856, 'y-': -0.044242250001683026, 'y+': 0.06019168813132403}
nominal worst held-out bank/shift/sensor/pulse [0.00087713 0.00083574 0.00167383 0.00065104]
series resistance worst held-out bank/shift/sensor/pulse [8.70043314e-05 1.15236952e-04 1.86762144e-04 8.43479587e-05]
axis shared worst held-out bank/shift/sensor/pulse [0.00011293 0.00012954 0.00020595 0.00010093]
axis plus pair differences worst held-out bank/shift/sensor/pulse [5.38318604e-05 6.29104534e-05 1.20192542e-04 4.25913695e-05]
bank 0 residual RMS before/after nuisance 0.2030787951739667 0.007803199671691561
bank 1 residual RMS before/after nuisance 0.1502275073443795 0.005766925981978262
bank 2 residual RMS before/after nuisance 0.20108117977687096 0.0064333803266591215
bank 3 residual RMS before/after nuisance 0.1593830080997298 0.008042217046091276
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`34302 ms` LLM)
- **Rationale**: The remaining risk is blind-campaign boundary bias. Use the measured pulsed boundary as a reference and preserve measured observation residuals under signed transport perturbations. Promote an observation-only correction only if it reduces field and weight errors without worsening the worst held-out sensor fold.
- **Alternatives Weighed**: `Replace the four-endpoint pulse fit with shared Robin resistance` (rejected: The shared resistance model had higher RMSE in every held-out tracer grouping.)

```bash
cat > /tmp/blind_margin.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s); b=m._pulse_losses(t,s['reference_tracer_pulses'],decode); Z=s['Z_nuisance']; y=decode(s['y_obs_raw']); S0,U0,_=t.response(b,t.sources)
cap=s['u_box']; budget=s['total_budget']; w0,_=m._source_fit(s,S0,y); resid=y-S0@w0; G=np.repeat(np.eye(2),2,axis=0); banks=np.asarray(s['row_bank_id']); groups=np.tile(np.arange(len(t.sensors))%4,len(t.shifts)); print('Measured endpoint losses minus nominal:',b-t.nominal)
configs=[(0.0,0.0),(0.002,0.002),(0.01,0.01),(0.04,0.04),(0.0,0.01)]
errors={k:[] for k in ['nominal',*configs]}; sensors={k:[] for k in errors}; real=[]
for scale in [None,-2,-1,0,1,2]:
    truth=b if scale is None else t.nominal+scale*(b-t.nominal); S,U,_=t.response(truth,t.sources); obs=y if scale is None else S@w0+resid
    target_w,_=m._source_fit(s,S,obs); target=U@target_w
    for kind in ['nominal',*configs]:
        out=[]
        for test in [np.zeros(len(y),bool),*[groups==g for g in range(4)]]:
            train=~test; zz=Z[train]; P=np.linalg.pinv(zz); project=lambda v:v[train]-zz@(P@v[train]); yp=project(obs)
            def profile(q,return_state=False):
                boundary=t.nominal+G@q; X,V,_=t.response(boundary,t.sources); Xp=project(X)
                rig,pen=(0.002,0.) if kind=='nominal' else kind; H=Xp.T@Xp+(rig+1e-9)*np.eye(X.shape[1]); w,_=m._box_simplex(H,Xp.T@yp,np.full(X.shape[1],cap),budget)
                if return_state:return X,V,w,boundary
                return np.r_[Xp@w-yp,np.sqrt(rig)*(w-budget/len(w)),np.sqrt(pen)*q]
            q=np.zeros(2) if kind=='nominal' else least_squares(profile,np.zeros(2),bounds=(-2,2),gtol=1e-9).x
            X,V,w,loss=profile(q,True); r=obs-X@w; gamma=P@r[train]
            if np.any(test):out.append(np.sqrt(np.mean((r[test]-Z[test]@gamma)**2)))
            else:
                w,_=m._source_fit(s,X,obs); error=(np.max(abs(loss-truth)),np.linalg.norm(V@w-target)/np.linalg.norm(target),np.max(abs(w-target_w))); errors[kind].append(error); raw=(q,error)
        sensors[kind].append(max(out))
        if scale is None: print('real',kind,'correction',raw[0].round(4),'loss/field/weight error',np.round(raw[1],6),'worst held-out RMSE',max(out))
for k in errors:print('perturbed real residual',k,'worst loss/field/weight error',np.max(errors[k][1:],axis=0),'worst held-out RMSE',max(sensors[k][1:]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/blind_margin.py
```
- **Container Exit Code**: `0` (`3462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Measured endpoint losses minus nominal: [ 0.25057287  0.2652289  -0.04424225  0.06019169]
real nominal correction [0. 0.] loss/field/weight error [0.265229 0.004914 0.001769] worst held-out RMSE 0.008730900769418964
real (0.0, 0.0) correction [0.3266 0.2846] loss/field/weight error [0.328799 0.006637 0.003378] worst held-out RMSE 0.008697834553042544
real (0.002, 0.002) correction [0.0634 0.0638] loss/field/weight error [0.201846 0.003103 0.002069] worst held-out RMSE 0.008703004048153861
real (0.01, 0.01) correction [0.0182 0.0134] loss/field/weight error [0.247008 0.004401 0.001702] worst held-out RMSE 0.008583975836795486
real (0.04, 0.04) correction [0.0059 0.0025] loss/field/weight error [0.259369 0.00477  0.001718] worst held-out RMSE 0.008880404665112494
real (0.0, 0.01) correction [0.0138 0.0167] loss/field/weight error [0.251414 0.004449 0.001762] worst held-out RMSE 0.008837474785913154
perturbed real residual nominal worst loss/field/weight error [0.53045781 0.01046561 0.00386334] worst held-out RMSE 0.008874971262251724
perturbed real residual (0.0, 0.0) worst loss/field/weight error [0.46948253 0.00698829 0.00351238] worst held-out RMSE 0.0086993234325077
perturbed real residual (0.002, 0.002) worst loss/field/weight error [0.50816804 0.01003685 0.00373278] worst held-out RMSE 0.008818482101490532
perturbed real residual (0.01, 0.01) worst loss/field/weight error [0.52779438 0.01038473 0.00383611] worst held-out RMSE 0.008694679763646702
perturbed real residual (0.04, 0.04) worst loss/field/weight error [0.53064715 0.01044996 0.00386021] worst held-out RMSE 0.009064432840808618
perturbed real residual (0.0, 0.01) worst loss/field/weight error [0.52455944 0.01035626 0.00381953] worst held-out RMSE 0.008966520669603403

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`24847 ms` LLM)
- **Rationale**: Check whether candidate corrections generalize across held-out shifts and banks, rather than only the sensor partitions already examined. If a restrained correction fails those independent slices, retain the nominal fallback rather than selecting on the perturbation result alone.
- **Alternatives Weighed**: `Adopt the unregularized blind correction` (rejected: It reduced worst perturbed error but worsened both field and weight recovery on the unmodified observation with pulse-derived boundaries.)

```bash
python3 - <<'PY'
p='/tmp/blind_margin.py'
s=open(p).read().replace("groups=np.tile(np.arange(len(t.sensors))%4,len(t.shifts));", "groups=np.tile(np.arange(len(t.sensors))%4,len(t.shifts)); shifts=np.repeat(np.arange(len(t.shifts)),len(t.sensors));")
s=s.replace("configs=[(0.0,0.0),(0.002,0.002),(0.01,0.01),(0.04,0.04),(0.0,0.01)]", "configs=[(0.0,0.0),(0.0,0.001),(0.002,0.002),(0.01,0.01)]")
s=s.replace("*[groups==g for g in range(4)]", "*[a==g for a in (groups,shifts,banks) for g in np.unique(a)]")
s=s.replace("if np.any(test):out.append(np.sqrt(np.mean((r[test]-Z[test]@gamma)**2)))", "if np.any(test):out.append(np.sqrt(np.mean((r[test]-Z[test]@gamma)**2)))")
s=s.replace("'worst held-out RMSE',max(out)", "'worst sensor/shift/bank RMSE', [max(out[:4]),max(out[4:7]),max(out[7:])]")
s=s.replace("S@w0+resid", "S@w0+resid").replace("for scale in [None,-2,-1,0,1,2]:", "for scale in [None,-2,0,2]:")
open('/tmp/blind_crosscheck.py','w').write(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/blind_crosscheck.py
```
- **Container Exit Code**: `0` (`5001 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Measured endpoint losses minus nominal: [ 0.25057287  0.2652289  -0.04424225  0.06019169]
real nominal correction [0. 0.] loss/field/weight error [0.265229 0.004914 0.001769] worst sensor/shift/bank RMSE [0.008730900769418964, 0.009075204343549899, 0.008730900769418964]
real (0.0, 0.0) correction [0.3266 0.2846] loss/field/weight error [0.328799 0.006637 0.003378] worst sensor/shift/bank RMSE [0.008697834553042544, 0.009115820082999495, 0.008697834553042544]
real (0.0, 0.001) correction [0.0937 0.1109] loss/field/weight error [0.171512 0.002557 0.002468] worst sensor/shift/bank RMSE [0.008797652756981963, 0.009114090709523697, 0.008797652756981963]
real (0.002, 0.002) correction [0.0634 0.0638] loss/field/weight error [0.201846 0.003103 0.002069] worst sensor/shift/bank RMSE [0.008703004048153861, 0.009067364640433357, 0.008703004048153861]
real (0.01, 0.01) correction [0.0182 0.0134] loss/field/weight error [0.247008 0.004401 0.001702] worst sensor/shift/bank RMSE [0.008583975836795486, 0.008961522042153237, 0.008583975836795486]
perturbed real residual nominal worst loss/field/weight error [0.53045781 0.01046561 0.00386334] worst held-out RMSE 0.009199264041410775
perturbed real residual (0.0, 0.0) worst loss/field/weight error [0.46948253 0.00698829 0.00351238] worst held-out RMSE 0.009119110292105
perturbed real residual (0.0, 0.001) worst loss/field/weight error [0.48049525 0.00962693 0.00417559] worst held-out RMSE 0.00920672411441987
perturbed real residual (0.002, 0.002) worst loss/field/weight error [0.50816804 0.01003685 0.00373278] worst held-out RMSE 0.009176605747259339
perturbed real residual (0.01, 0.01) worst loss/field/weight error [0.52779438 0.01038473 0.00383611] worst held-out RMSE 0.009097110604161913

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d4de75e268156b75 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d4de75e2 (directory).
```

### Turn 13 — `submit_completed_task` (`14913 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated-process schema and numerical verification. Separate endpoint losses outperformed shared and rational Robin models on held-out tracer slices, and nuisance projection reduced residuals in every sensor bank.
- **Alternatives Weighed**: `Fit boundary corrections from blind observations` (rejected: The unregularized fit improved the signed perturbation stress cases but worsened field and weight recovery on the unmodified public observation. Regularization reduced that trade-off but did not establish a transferable correction.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst KKT residual across 1,800 stress cases was 3.22e-14. The solver fits boundary losses when tracer pulses are available and uses spacing-scaled nominal Robin losses otherwise. No apparatus archive was available to validate a campaign-specific fallback.`

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
