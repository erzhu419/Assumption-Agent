# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a01_batch_08__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `291,983` (`280,320` / `102,784` / `11,663`) |
| **Total Episode Wall Time (s)** | `448.89s` |

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

### Turn 1 — `execute_in_remote_docker` (`16973 ms` LLM)
- **Rationale**: Inspect the frozen interface, scoring protocol, and array shapes before fitting. These checks determine which inputs the sink can use and whether the public CV separates calibration from held-out responses.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_batch.pkl','rb') as f: x=pickle.load(f)
def show(x,p='root', depth=0):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'.'+str(k),depth+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if x: show(x[0],p+'[0]',depth+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,'sample',x.reshape(-1)[:10]);
  if x.size and x.dtype.kind in 'uif': print('  min/max/std',np.nanmin(x),np.nanmax(x),np.nanstd(x))
 else: print(p,type(x).__name__,str(x)[:120])
show(x)
PY
```
- **Container Exit Code**: `0` (`383 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
std -7.655944584670875 6.923609913249616 1.7415136486450644
root[0].study.panel_b.u (24, 4) float64 sample [ 1.02050173 -0.57032133  0.31951828 -0.75334028 -1.0590791   1.13644624
 -0.62551637 -0.17026907  0.52840037  0.8457305 ]
  min/max/std -1.2083718223881248 1.2275320934448923 0.7864568554958693
root[0].study.panel_b.v (24, 4) float64 sample [-0.97959048  0.6892284  -0.2752258  -0.57937069  1.16822355 -0.12173256
  0.75459362 -0.10064068  0.05843126 -0.6524635 ]
  min/max/std -1.5561648476014998 1.1682235529584668 0.6947670871757526
root[0].study.panel_c.u (24, 4) float64 sample [-0.25580564 -0.07845265  1.18937717  0.54304068  0.06188184 -1.17680494
  0.24334397 -0.31108812 -1.15540966 -0.25506043]
  min/max/std -1.2443029633134872 1.2152198161740508 0.6908062347557175
root[0].study.panel_c.v1 (24, 4) float64 sample [-0.04809841  1.14821575 -0.14569272 -0.35681708  0.22126849  0.3850186
  1.15967221 -0.17069023  0.78654191  0.88651042]
  min/max/std -1.5062793412284066 1.5329870861407233 0.6515322187436658
root[0].study.panel_c.v2 (24, 4) float64 sample [ 0.07860216 -0.53933786 -0.69319813 -0.23281473  0.15591632  0.18454225
 -0.26100595  0.11662847  0.28197304 -0.11362042]
  min/max/std -0.9962161205277777 0.7272388994660736 0.3407700900986726
root[0].study.train.x (72, 4) float64 sample [-0.80795288  0.41609593 -0.94981075  0.73844298  0.6454008  -0.66625005
 -0.71688417 -0.93382027  0.16678133 -0.9843887 ]
  min/max/std -1.1413798319225994 1.147116005103889 0.6695026210010858
root[0].study.train.y (72,) float64 sample [ 0.9181182   0.84985712  0.42576376  0.39475948  0.09257838 -0.46081517
 -0.16322769  0.13048245  0.55742078  0.78302665]
  min/max/std -1.9052432297757005 1.0572540474428613 0.6203957932523012
root[0].study.train.traj (72, 8) float64 sample [0.06608055 0.14911574 0.23009876 0.316052   0.37544544 0.45745588
 0.51071924 0.56723491 0.25509363 0.56132438]
  min/max/std -1.0574232583982741 1.5903391613106592 0.41365130184091675
root[0].study.train.summary (4,) float64 sample [ 0.11519913  0.62063534 -0.02290514  0.34602494]
  min/max/std -0.02290514489079197 0.6206353382383104 0.24411622994754767
root[0].study.test.x (36, 4) float64 sample [-0.89594003  0.51098357 -0.01877662 -0.19859755 -0.4105117  -1.09645177
 -0.46243137 -0.92876726  0.73511919  0.97562803]
  min/max/std -1.1481973199208668 1.1118163154413057 0.6735492408421424
root[0].study.telemetry_vec (4,) float64 sample [1.7 0.2 0.1 0.2]
  min/max/std 0.1 1.7 0.6652067347825036
```

### Turn 2 — `execute_in_remote_docker` (`23467 ms` LLM)
- **Rationale**: Recover the exact validation split and test whether sparse nonlinear C terms improve leave-one-row-out calibration. Inspect trajectory rank to decide whether to model shared temporal modes or individual times.

```bash
cd /holdout_task && sed -n '45,160p' test_dag_candidate.py && cat validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np,itertools as it
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; n,d=u.shape
A=np.c_[np.ones(n),u]; fs={}
for deg in [2,3]:
 for js in it.combinations_with_replacement(range(d),deg): fs[str(js)]=np.prod(u[:,js],axis=1)
for j in range(d):
 for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('abs',np.abs),('exp',np.exp)]: fs[f'{name}{j}']=f(u[:,j])
 for k in range(j+1,d):
  for sgn in [-1,1]:
   z=u[:,j]+sgn*u[:,k]
   for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:fs[f'{name}({j},{sgn},{k})']=f(z)
keys=list(fs); F=np.column_stack(list(fs.values()))
def score(X,y):
 Q=np.linalg.qr(X)[0]; r=y-Q@(Q.T@y); return np.mean(np.abs(r/(1-(Q*Q).sum(1))))
for p in ['v1','v2']:
 for j,y in enumerate(s['panel_c'][p].T):
  ids=[]; print(p,j,'linear',score(A,y))
  for step in range(4):
   opts=sorted((score(np.c_[A,F[:,ids+[k]]],y),k) for k in range(len(keys)) if k not in ids)
   print([(round(e,5),keys[k]) for e,k in opts[:4]]);ids.append(opts[0][1])
T=s['train']['traj']; print('traj sv',np.linalg.svd(T,compute_uv=False));print('traj samples',T[:7]);print('input keys',s.keys())
PY
```
- **Container Exit Code**: `0` (`497 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,3)'), (0.02431, 'tanh(0,-1,3)'), (0.02739, 'cos(0,-1,1)'), (0.02885, 'abs1')]
v2 0 linear 0.08493946588349414
[(0.05993, 'cos(0,1,3)'), (0.06408, 'abs1'), (0.06725, 'cos1'), (0.06736, 'exp1')]
[(0.04149, 'abs1'), (0.04252, 'cos1'), (0.04279, '(1, 1)'), (0.04385, 'exp1')]
[(0.03122, '(0, 3)'), (0.03184, 'cos(0,-1,3)'), (0.03206, 'abs0'), (0.03251, 'exp0')]
[(0.0181, '(2, 2)'), (0.01815, 'cos2'), (0.01871, 'tanh(2,-1,3)'), (0.01908, 'abs2')]
v2 1 linear 0.10389770661975717
[(0.06781, 'abs3'), (0.07039, '(0, 2, 2)'), (0.07054, 'cos3'), (0.0719, '(3, 3)')]
[(0.03428, 'abs0'), (0.04215, 'cos0'), (0.04331, '(0, 0)'), (0.04333, 'cos(0,1,3)')]
[(0.01809, 'abs2'), (0.01832, '(2, 2)'), (0.01852, 'cos2'), (0.02004, 'exp2')]
[(0.0129, 'abs1'), (0.01479, 'cos1'), (0.01502, '(1, 1)'), (0.01583, '(2, 3, 3)')]
v2 2 linear 0.10988914322604214
[(0.06612, 'abs2'), (0.06719, 'cos2'), (0.06784, '(2, 2)'), (0.0694, 'exp2')]
[(0.03596, '(1, 1)'), (0.03681, 'cos1'), (0.03886, 'exp1'), (0.04538, 'abs1')]
[(0.0184, 'exp0'), (0.02089, '(0, 0)'), (0.02113, 'cos0'), (0.02202, 'abs0')]
[(0.01011, 'cos3'), (0.01015, '(3, 3)'), (0.01024, 'abs3'), (0.01134, 'exp3')]
v2 3 linear 0.07430002595419809
[(0.04815, 'cos(1,-1,2)'), (0.05137, 'tanh(0,1,2)'), (0.05193, 'sin(0,1,2)'), (0.05517, 'cos(2,1,3)')]
[(0.03871, '(1, 2, 3)'), (0.0411, 'cos(1,1,2)'), (0.04199, 'cos(2,1,3)'), (0.0433, '(1, 2)')]
[(0.03452, '(1, 1, 2)'), (0.03604, 'tanh(1,-1,3)'), (0.0371, 'tanh1'), (0.03737, '(3, 3, 3)')]
[(0.02983, 'tanh(1,-1,3)'), (0.03036, '(1, 3, 3)'), (0.03102, '(0, 3)'), (0.03179, 'sin(1,-1,3)')]
traj sv [9.68947149 2.27233978 0.05472247 0.04862014 0.0445937  0.03721879
 0.03351494 0.03158632]
traj samples [[ 0.06608055  0.14911574  0.23009876  0.316052    0.37544544  0.45745588
   0.51071924  0.56723491]
 [ 0.25509363  0.56132438  0.7178805   0.74880551  0.65110489  0.44520218
   0.20982681 -0.01820094]
 [ 0.00935027  0.02137033 -0.00245652 -0.0427793  -0.11272185 -0.19360543
  -0.27683912 -0.35385535]
 [-0.03681284 -0.09328116 -0.11901436 -0.11633688 -0.10242649 -0.08963996
  -0.06400889 -0.02098424]
 [-0.0700448  -0.14438547 -0.18835406 -0.19133874 -0.15371248 -0.09765065
  -0.0213239   0.0489456 ]
 [-0.09895994 -0.20581562 -0.2715133  -0.30552513 -0.30300222 -0.27158353
  -0.22001115 -0.18208433]
 [-0.00710596 -0.01500633 -0.01766684  0.02176293  0.07250426  0.12946126
   0.19596749  0.25384843]]
input keys dict_keys(['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec'])
```

### Turn 3 — `execute_in_remote_docker` (`37254 ms` LLM)
- **Rationale**: Compare compact panel bases and separately measure response and trajectory error on the official split and five additional splits. This distinguishes a readout-basis problem from a trajectory regression problem.
- **Alternatives Weighed**: `Fit separate unrestricted cubic readouts to panel C` (rejected: A full cubic has 35 coefficients for 24 calibration rows, so it cannot identify a panel map.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine panel-C v2 basis is sufficient.` | Witness: `For the third v2 channel, sparse nonlinear terms reduce panel-row leave-one-out MAE from 0.10989 to 0.01011 during screening.` | Blanket Diagnosis: `The affine readout omits curved dependence on the calibrated coordinates; screening does not yet identify the exact basis.` | Sublated Pivot: `Preserve the frozen A/B path and compare shared nonlinear C bases before fitting the sink response.`

```bash
cd /holdout_task && cat > /tmp/probe05.py <<'PY'
import numpy as np, pickle, itertools as it, sys, importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
spec=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=m.run_node({'__input__':s})['q_tr'];x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];u=s['panel_c']['u'];v1=s['panel_c']['v1'];v2=s['panel_c']['v2']
def poly(z,k=2):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,js],axis=1) for i in range(1,k+1) for js in it.combinations_with_replacement(range(z.shape[1]),i)])
def report(A,Y,cv=False):
 if cv:
  splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(72) for i in range(5)]]
  es=[np.abs(A[j]@np.linalg.lstsq(A[i],Y[i],rcond=None)[0]-Y[j]).mean() for i,j in splits];return np.round([es[0],np.mean(es),max(es)],5)
 Q=np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)];r=Y-Q@(Q.T@Y);return np.round([np.abs(r).mean(),np.abs(r/(1-(Q*Q).sum(1)[:,None])).mean(),*np.sqrt(np.mean(r*r,axis=0))],6)
for lbl,A in [('lin',poly(u,1)),('quad',poly(u)),('lin+tanh',np.c_[poly(u,1),np.tanh(u)]),('lin+sin',np.c_[poly(u,1),np.sin(u)]),('lin+abs',np.c_[poly(u,1),np.abs(u)]),('quad+tanh',np.c_[poly(u),np.tanh(u)]),('square',np.c_[poly(u,1),u*u]),('no-linear tanh',np.c_[np.ones(len(u)),np.tanh(u)])]:print('panel',lbl,'v1',report(A,v1),'v2',report(A,v2))
for name,z in [('x',x),('q',q)]:
 for lbl,A in [('lin',poly(z,1)),('quad',poly(z)),('cub',poly(z,3)),('sin',np.c_[poly(z,2),np.sin(z),np.cos(z)]),('tanh',np.c_[poly(z,2),np.tanh(z)]),('tanh-only',poly(np.tanh(z),1))]: print(name,lbl,'y',report(A,y,True),'T',report(A,T,True))
for P in [v1,v2]:
 for func in [np.tanh,np.sin,np.cos,np.exp]:
  def design(z,w):return np.c_[poly(z,1),func(z*w[:4]+w[4:])]
  def err(w):
   A=design(u,w); return (A@np.linalg.lstsq(A,P,rcond=None)[0]-P).ravel()
  fit=least_squares(err,np.r_[np.ones(4),np.zeros(4)],max_nfev=300); print('cal-opt',func.__name__,np.round(fit.x,4),report(design(u,fit.x),P))
print('correlations q, y,T modes',np.round(np.corrcoef(np.c_[q,y,np.linalg.svd(T,full_matrices=False)[0][:,:2]].T),3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe05.py
```
- **Container Exit Code**: `0` (`633 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.101916] v2 [0.003234 0.005421 0.004523 0.002873 0.004796 0.003839]
panel no-linear tanh v1 [0.046361 0.058609 0.046474 0.056306 0.062403 0.057742] v2 [0.078284 0.100531 0.088263 0.098619 0.114037 0.07257 ]
x lin y [0.23663 0.25546 0.29888] T [0.08466 0.09538 0.10827]
x quad y [0.06077 0.07788 0.10274] T [0.01337 0.01572 0.01772]
x cub y [0.0982  0.0998  0.13338] T [0.01345 0.01713 0.02252]
x sin y [0.06156 0.09283 0.1357 ] T [0.01478 0.01614 0.02066]
x tanh y [0.07626 0.08605 0.11986] T [0.01467 0.0158  0.01709]
x tanh-only y [0.23285 0.2571  0.2901 ] T [0.07887 0.09647 0.11193]
q lin y [0.23663 0.25546 0.29888] T [0.08466 0.09538 0.10827]
q quad y [0.06077 0.07788 0.10274] T [0.01337 0.01572 0.01772]
q cub y [0.0982  0.0998  0.13338] T [0.01345 0.01713 0.02252]
q sin y [0.0705  0.08045 0.09683] T [0.01084 0.0126  0.01469]
q tanh y [0.07485 0.08313 0.09836] T [0.01083 0.01221 0.01323]
q tanh-only y [0.25325 0.25722 0.29304] T [0.0845  0.09551 0.1124 ]
cal-opt tanh [ 2.2132e+00  2.2386e+00  2.3140e+00  2.5402e+00  5.5000e-03 -1.0000e-04
  4.3000e-03 -1.8000e-03] [0.003637 0.005858 0.004739 0.004973 0.00415  0.00414 ]
cal-opt sin [ 3.3169  3.3704  3.4864  3.6933  0.0213 -0.019   0.017  -0.0179] [0.00809  0.013034 0.012625 0.009011 0.00913  0.011236]
cal-opt cos [ 3.3169  3.3704  3.4864  3.6933  1.5921  1.5518  7.871  -1.5887] [0.00809  0.013034 0.012625 0.009011 0.00913  0.011236]
cal-opt exp [-3.4352  4.545   5.2314  4.0105 -3.3628 -2.1437 -1.7618 -5.3337] [0.070577 0.118582 0.068108 0.099213 0.085721 0.095226]
cal-opt tanh [ 0.734   0.6918  0.681  -0.6719 -0.755   0.7524 -0.7439  0.7211] [0.000664 0.001131 0.000738 0.000814 0.000675 0.000949]
cal-opt sin [ 1.2865  1.2696  1.291   1.3292 -1.5626  1.5744 -1.5718  1.571 ] [0.000272 0.000467 0.000334 0.000256 0.000489 0.000231]
cal-opt cos [ 1.2865e+00  1.2696e+00  1.2910e+00  1.3292e+00  8.2000e-03  3.6000e-03
 -1.0000e-03  2.0000e-04] [0.000272 0.000467 0.000334 0.000256 0.000489 0.000231]
cal-opt exp [ 0.1625 -0.0736  0.0155 -0.0615  3.2479 -0.0192 -4.5761  2.8822] [0.003103 0.005259 0.00405  0.002814 0.004496 0.003654]
correlations q, y,T modes [[ 1.    -0.236  0.043 -0.059  0.183 -0.174 -0.741]
 [-0.236  1.    -0.008  0.026  0.231  0.256  0.57 ]
 [ 0.043 -0.008  1.    -0.132  0.486 -0.809 -0.029]
 [-0.059  0.026 -0.132  1.     0.572  0.55  -0.413]
 [ 0.183  0.231  0.486  0.572  1.    -0.192 -0.338]
 [-0.174  0.256 -0.809  0.55  -0.192  1.    -0.   ]
 [-0.741  0.57  -0.029 -0.413 -0.338 -0.     1.   ]]
```

### Turn 4 — `execute_in_remote_docker` (`29937 ms` LLM)
- **Rationale**: Refit the shared calibration frequency within each held-out panel-row fold, then compare readout features and low-order interactions on response and trajectory separately. This tests whether the sharper calibration actually helps predict the response.
- **Alternatives Weighed**: `Use a full cubic sink on the frozen coordinates` (rejected: It raises official-split response MAE from 0.06077 to 0.09820.)

```bash
cat > /tmp/discover05.py <<'PY'
exec(open('/tmp/probe05.py').read().split("for lbl,A in")[0])
from scipy.optimize import minimize_scalar

def basis(z,mode,w,linear):
 f={'tanh':np.tanh,'cos':np.cos,'sin':np.sin}[mode]
 return np.c_[poly(z,1) if linear else np.ones(len(z)),f(z*w)]
def calib(U,V,mode,linear):
 def obj(w):
  A=basis(U,mode,w,linear); R=A@np.linalg.lstsq(A,V,rcond=None)[0]-V;return np.mean(R**2)
 w=minimize_scalar(obj,bounds=(.6,3.3),method='bounded').x
 A=basis(U,mode,w,linear); c=np.linalg.lstsq(A,V,rcond=None)[0]
 return w,lambda Z:basis(Z,mode,w,linear)@c
P=[]
for V,mode in [(v1,'tanh'),(v2,'cos')]:
 for lin in [False,True]:
  w,fn=calib(u,V,mode,lin)
  E=[]
  for i in range(len(u)):
   k=np.arange(len(u))!=i; _,f=calib(u[k],V[k],mode,lin);E.append(np.abs(f(u[~k])-V[~k]).mean())
  print('cal',mode,lin,round(w,5),np.abs(fn(u)-V).mean(),np.mean(E),max(E));P.append(fn(q))
print('q range',q.min(0),q.max(0),'u',u.min(0),u.max(0))
a,b,c,d=P
for tag,z in [('q',q),('a',a),('b',b),('c',c),('d',d),('a,c',np.c_[a,c]),('b,d',np.c_[b,d]),('q,a',np.c_[q,a]),('q,c',np.c_[q,c]),('q,b,d',np.c_[q,b,d]),('tanh_cos',np.c_[np.tanh(2.3*q),np.cos(1.3*q)]),('all',np.c_[q,np.tanh(2.3*q),np.cos(1.3*q)])]:
 for style in ['linear','square','quad','quadblock','limited']:
  if style=='linear': A=poly(z,1)
  if style=='square': A=np.c_[poly(z,1),z*z]
  if style=='quad':
   if z.shape[1]>8:continue
   A=poly(z)
  if style=='quadblock': A=np.column_stack([poly(z[:,i:i+4]) for i in range(0,z.shape[1],4)])
  if style=='limited': A=np.c_[poly(q),z,z*z]
  print('sink',tag,style,report(A,y,True),report(A,T,True))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/discover05.py
```
- **Container Exit Code**: `0` (`658 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
559]
sink d quad [0.13136 0.1402  0.16476] [0.06388 0.06317 0.07206]
sink d quadblock [0.13136 0.1402  0.16476] [0.06388 0.06317 0.07206]
sink d limited [0.06996 0.07678 0.10858] [0.01388 0.01501 0.0164 ]
sink a,c linear [0.28587 0.28109 0.32287] [0.08622 0.09788 0.11339]
sink a,c square [0.17816 0.20895 0.27861] [0.07819 0.08821 0.09777]
sink a,c quad [0.17163 0.14279 0.19739] [0.06128 0.05288 0.06128]
sink a,c quadblock [0.13546 0.12336 0.13997] [0.0602  0.06812 0.0756 ]
sink a,c limited [0.08466 0.08341 0.10297] [0.01617 0.01479 0.01733]
sink b,d linear [0.29952 0.27605 0.29952] [0.11195 0.10375 0.12112]
sink b,d square [0.1965  0.23587 0.27932] [0.05398 0.0665  0.08418]
sink b,d quad [0.23768 0.268   0.37506] [0.07331 0.10948 0.20208]
sink b,d quadblock [0.16412 0.14634 0.17408] [0.04173 0.04608 0.05933]
sink b,d limited [0.06407 0.0846  0.09584] [0.01419 0.01581 0.01843]
sink q,a linear [0.26072 0.27178 0.31921] [0.092   0.10387 0.12581]
sink q,a square [0.18775 0.24522 0.29694] [0.07187 0.07724 0.08743]
sink q,a quad [0.17456 0.13575 0.17456] [0.03101 0.03286 0.04252]
sink q,a quadblock [0.06034 0.07526 0.10268] [0.01079 0.01507 0.01966]
sink q,a limited [0.08183 0.08153 0.09995] [0.01157 0.01393 0.01826]
sink q,c linear [0.24968 0.26272 0.29242] [0.08742 0.09145 0.10462]
sink q,c square [0.28259 0.27279 0.30948] [0.09397 0.09156 0.10424]
sink q,c quad [0.10754 0.19041 0.35744] [0.0345  0.03346 0.04075]
sink q,c quadblock [0.07061 0.09649 0.12845] [0.01645 0.01924 0.02286]
sink q,c limited [0.06275 0.08803 0.10853] [0.01559 0.01763 0.02021]
sink q,b,d linear [0.28237 0.27402 0.29836] [0.0821  0.08801 0.10095]
sink q,b,d square [0.19148 0.24401 0.31318] [0.04549 0.05694 0.07036]
sink q,b,d quadblock [0.1118  0.14344 0.24079] [0.02682 0.02848 0.03946]
sink q,b,d limited [0.06407 0.0846  0.09584] [0.01419 0.01581 0.01843]
sink tanh_cos linear [0.3033  0.2812  0.32341] [0.09143 0.10112 0.11855]
sink tanh_cos square [0.32186 0.28669 0.34537] [0.12385 0.11274 0.12385]
sink tanh_cos quad [0.25969 0.23994 0.30526] [0.06148 0.05696 0.07849]
sink tanh_cos quadblock [0.1835  0.17265 0.18479] [0.09062 0.10843 0.14048]
sink tanh_cos limited [0.12817 0.09096 0.12817] [0.02002 0.01871 0.0224 ]
sink all linear [0.28232 0.27401 0.29819] [0.08206 0.08803 0.10101]
sink all square [0.37903 0.34275 0.38527] [0.13847 0.11709 0.16063]
sink all quadblock [0.12691 0.13719 0.17376] [0.02736 0.02658 0.03545]
sink all limited [0.12817 0.09096 0.12817] [0.02002 0.01871 0.0224 ]
```

### Turn 5 — `execute_in_remote_docker` (`38002 ms` LLM)
- **Rationale**: Test whether the missing signal is an interaction between the frozen coordinates and the calibrated readout. Compare with polynomial and smooth-kernel baselines to distinguish a specific algebraic interaction from generic curvature.
- **Alternatives Weighed**: `Use additive C v2 features as the response correction` (rejected: Accurate panel calibration alone did not reduce official response MAE below 0.038.)

```bash
cat > /tmp/interact05.py <<'PY'
exec(open('/tmp/discover05.py').read().split('P=[]')[0])
from itertools import combinations_with_replacement, product
from scipy.optimize import minimize_scalar
P=[]
for V,mode in [(v1,'tanh'),(v2,'cos')]:
 w,f=calib(u,V,mode,True); A=basis(u,mode,w,True); c=np.linalg.lstsq(A,V,rcond=None)[0];print(mode,w,np.round(c,3)); P.append(f(q))
a,b=P
splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(72) for i in range(12)]]
def cv(A,name,Y=y, al=(0,.0001,.001,.01,.05,.2,1.)):
 A=A[:,1:] if np.std(A[:,0])<1e-10 else A
 L=[]
 for i,j in splits:
  mean=A[i].mean(0);sd=A[i].std(0);sd[sd<1e-10]=1
  X=(A[i]-mean)/sd;Z=(A[j]-mean)/sd
  U,S,V=np.linalg.svd(X,full_matrices=False); M=(U.T@(Y[i]-Y[i].mean(0))).reshape(len(S),-1);Yj=Y[j].reshape(len(j),-1)
  L.append([np.abs((Z@V.T)@((np.where(S>1e-8,S/(S*S+max(l,1e-16)),0))[:,None]*M)+Y[i].mean(0)-Yj).mean() for l in al])
 L=np.array(L);k=L.mean(0).argmin();print(name,A.shape[1],al[k],np.round([L[0,k],L[:,k].mean(),L[:,k].max()],5))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
Q=poly(q);h=np.cos(q*1.3)-1;g=np.tanh(q*2.3);D=np.c_[q,h,g,a,b]
for nm,z in [('cos',h),('tanh',g),('a',a),('b',b),('ab',np.c_[a,b]),('trig',np.c_[h,g])]:
 for nn,A in [('main',np.c_[Q,z]),('crossq',np.c_[Q,z,cross(q,z)]),('crossab',np.c_[Q,z,cross(a,z)]),('crossb',np.c_[Q,z,cross(b,z)]),('crossall',np.c_[Q,z,cross(np.c_[q,a,b],z)])]:cv(A,nm+nn)
for k in [3,4,5]:cv(poly(q,k),'poly'+str(k))
for z,nm in [(q,'q'),(np.c_[q,b],'qb'),(np.c_[q,a,b],'qab')]:
 for l in [0.5,1,2,4]:
  K=np.exp(-((z[:,None]-z[None,:])**2).sum(2)/(2*l*l));L=[]
  for i,j in splits:
   Ki=K[np.ix_(i,i)];Kj=K[np.ix_(j,i)];L.append([np.mean(np.abs(Kj@np.linalg.solve(Ki+v*np.eye(len(i)),y[i]-y[i].mean())+y[i].mean()-y[j])) for v in [0.0001,.001,.01,.1]])
  L=np.array(L);k=L.mean(0).argmin();print('rbf',nm,l,k,np.round([L[0,k],L[:,k].mean(),L[:,k].max()],5))
for nm,Z in [('q',q),('a',a),('b',b)]:
 A=poly(Z);c=np.linalg.lstsq(A,np.c_[y,T[:,2],T[:,-1]],rcond=None)[0];print(nm,'quad coeff',np.round(c,3).T)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/interact05.py
```
- **Container Exit Code**: `0` (`775 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9475]
tanhcrossall 66 0.2 [0.07914 0.0671  0.07978]
amain 18 1.0 [0.06182 0.07497 0.09844]
acrossq 34 1.0 [0.05972 0.06547 0.0805 ]
acrossab 34 1.0 [0.06661 0.07222 0.09063]
acrossb 34 1.0 [0.05818 0.07207 0.09754]
acrossall 66 1.0 [0.05019 0.06453 0.07789]
bmain 18 0.01 [0.05322 0.07554 0.09917]
bcrossq 34 1.0 [0.06792 0.07929 0.10476]
bcrossab 34 1.0 [0.05674 0.07175 0.09789]
bcrossb 34 1.0 [0.06752 0.08142 0.10018]
bcrossall 66 1.0 [0.05886 0.07516 0.10002]
abmain 22 0.001 [0.06647 0.07222 0.09923]
abcrossq 54 1.0 [0.0535  0.06575 0.09068]
abcrossab 54 1.0 [0.06408 0.06851 0.0842 ]
abcrossb 54 1.0 [0.06224 0.07458 0.10095]
abcrossall 118 1.0 [0.05979 0.06716 0.08533]
trigmain 22 0.01 [0.06704 0.07248 0.09945]
trigcrossq 54 0.2 [0.04988 0.07612 0.101  ]
trigcrossab 54 1.0 [0.07898 0.09158 0.12407]
trigcrossb 54 1.0 [0.05989 0.08182 0.10329]
trigcrossall 118 1.0 [0.06593 0.08215 0.1114 ]
poly3 34 0.2 [0.09335 0.0901  0.12373]
poly4 69 0.2 [0.14491 0.15027 0.19116]
poly5 125 0.2 [0.16951 0.17162 0.25374]
rbf q 0.5 0 [0.09608 0.14516 0.18313]
rbf q 1 0 [0.02834 0.04136 0.05377]
rbf q 2 0 [0.04089 0.04448 0.07045]
rbf q 4 0 [0.04813 0.06383 0.08603]
rbf qb 0.5 0 [0.11674 0.16267 0.19247]
rbf qb 1 0 [0.03572 0.05228 0.07269]
rbf qb 2 0 [0.04169 0.04245 0.06448]
rbf qb 4 0 [0.04518 0.06313 0.08705]
rbf qab 0.5 0 [0.24739 0.25625 0.31363]
rbf qab 1 0 [0.10846 0.10837 0.15605]
rbf qab 2 0 [0.05013 0.04966 0.06361]
rbf qab 4 0 [0.05059 0.0593  0.07719]
q quad coeff [[ 0.173  0.117  0.196  0.588  0.292  0.076 -0.442 -0.42   0.317  0.017
  -0.116 -0.755 -0.084 -0.673  0.032]
 [ 0.049 -0.137 -0.125  0.725 -0.545  0.034 -0.146 -0.175  0.102 -0.002
   0.234 -0.293  0.098 -0.249  0.056]
 [-0.012  0.23  -0.323  0.416 -0.176  0.034 -0.097 -0.128  0.047  0.048
  -0.067 -0.178 -0.041 -0.23   0.04 ]]
a quad coeff [[ 0.191 -0.295  0.344 -0.187 -0.339  0.406  0.205 -0.415 -0.582 -0.05
   0.25   0.303 -0.196 -0.404 -0.238]
 [-0.084  0.506  0.396  0.364 -0.386  0.227  0.089 -0.03  -0.21  -0.056
  -0.108  0.057 -0.068 -0.051 -0.066]
 [-0.122 -0.02   0.211  0.394 -0.293  0.172  0.062 -0.039 -0.1   -0.02
   0.081  0.232  0.04  -0.122 -0.085]]
b quad coeff [[ 0.253 -0.68  -1.41  -1.144 -1.424 -1.116 -2.85  -0.259 -3.517 -2.283
  -2.216 -1.483 -1.233 -2.391  0.591]
 [ 0.259  1.363  0.264 -1.098 -1.748 -0.033  0.115  0.709 -1.402 -0.328
  -0.858 -0.761 -1.425 -1.445  0.703]
 [-0.007  0.437  0.511 -1.393 -0.582  0.121  0.524 -0.503 -0.554 -0.453
  -1.295 -0.868 -0.757 -2.022  0.167]]
```

### Turn 6 — `execute_in_remote_docker` (`30379 ms` LLM)
- **Rationale**: Install and test a baseline that uses the independently validated cosine calibration without letting it dominate the response geometry. The official validator will detect interface, diagnostics, or accuracy failures before further response-model refinement.
- **Alternatives Weighed**: `Give the C v2 channels the same kernel weight as the frozen coordinates` (rejected: Equal weights raised worst-split response MAE from 0.05377 to 0.07269 at unit bandwidth.)

```bash
set -e
cd /holdout_task
cat _frozen_nodes/node_00.py
cat > node_01.py <<'PY'
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _quadratic(z):
    return np.column_stack((_affine(z), *[z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]))


def _readout(u, v, z):
    def design(a, w):
        return np.column_stack((_affine(a), np.cos(w * a)))

    def fit(w):
        a = design(u, w)
        return a, np.linalg.lstsq(a, v, rcond=None)[0]

    def loss(w):
        a, coef = fit(w)
        return np.mean((a @ coef - v) ** 2)

    w = minimize_scalar(loss, bounds=(0.6, 2.2), method='bounded').x
    _, coef = fit(w)
    return design(z, w) @ coef


def _kernel(a, b):
    d = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * d)


def run_node(inputs: dict) -> dict:
    study, calibrated = inputs['__input__'], inputs['node_00']
    q, qt = calibrated['q_tr'], calibrated['q_te']
    train, panel = study['train'], study['panel_c']
    z = np.vstack((q, qt))
    z = np.column_stack((z, 0.2 * _readout(panel['u'], panel['v2'], z)))
    n = len(q)
    k = _kernel(z, z[:n])
    mu = np.mean(train['y'])
    y_hat = mu + k[n:] @ np.linalg.solve(k[:n] + 1e-4 * np.eye(n), train['y'] - mu)
    state_pred = _quadratic(qt) @ np.linalg.lstsq(_quadratic(q), train['traj'], rcond=None)[0]
    end = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`541 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
import numpy as np


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x, xt = study['train']['x'], study['test']['x']
    a = study['panel_a']
    scale, bias = [], []
    for u, v in zip(a['u'].T, a['v'].T):
        i, j = np.triu_indices(len(u), 1)
        dx = u[i] - u[j]
        keep = np.abs(dx) > 1e-9
        slope = np.median((v[i] - v[j])[keep] / dx[keep])
        scale.append(slope)
        bias.append(np.median(v - slope * u))
    b = study['panel_b']
    coef = np.linalg.lstsq(_linear(b['u']), b['v'], rcond=None)[0]
    q = _linear((x - bias) / scale) @ coef
    qt = _linear((xt - bias) / scale) @ coef
    return {'q_tr': q, 'q_te': qt}
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.028473, "public_cv_traj_mae": 0.013369}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81a20009aa6c0aab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81a20009 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`40716 ms` LLM)
- **Rationale**: Stress-test the exact candidate on random and coordinate-tail slices. Compare with a kernel challenger that selects its settings inside each outer fold; promote it only if it improves the response bottleneck without sacrificing official acceptance or trajectory accuracy.
- **Alternatives Weighed**: `Submit based on the official split alone` (rejected: The earlier multi-split results were for the q-only kernel, not the actual artifact; they do not establish how the calibrated candidate behaves at the edges of the observed coordinate range.)

```bash
set -e
cd /holdout_task
sed -n '70,150p' test_dag_candidate.py
cat > /tmp/stress05.py <<'PY'
import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

root = '/holdout_task/'
a = load('frozen', root + '_frozen_nodes/node_00.py')
b = load('candidate', root + 'node_01.py')
for batch in pickle.load(open(root + 'public_batch.pkl', 'rb')):
    s = batch['study']
    tr = s['train']; y = tr['y'].ravel(); T = tr['traj']; n = len(y)
    pred = a.run_node({'__input__': s}); q = pred['q_tr']
    p = s['panel_c']; c = b._readout(p['u'], p['v2'], q)
    loo = []
    for j in range(len(p['u'])):
        keep = np.arange(len(p['u'])) != j
        cp = b._readout(p['u'][keep], p['v2'][keep], p['u'][~keep]); loo.append(np.mean(abs(cp-p['v2'][~keep])))
    print('C calibration LOO mean/worst', np.mean(loo), np.max(loo))
    nq = 20; idx = np.arange(n); splits = [('first', idx[:-nq], idx[-nq:])]
    for seed in range(24):
        r = np.random.default_rng(seed).permutation(n)
        splits.append((str(seed),r[:-nq],r[-nq:]))
    for k in range(q.shape[1]):
        r = np.argsort(q[:,k]); splits.extend([(f'q{k} low',r[nq:],r[:nq]),(f'q{k} high',r[:-nq],r[-nq:])])
    d0 = ((q[:,None]-q[None,:])**2).sum(2); dc = ((c[:,None]-c[None,:])**2).sum(2)
    pairs = b._quadratic(q); Kpoly = pairs @ pairs.T
    family = []
    for ell in (.65,.85,1.,1.2,1.45,1.75,2.1):
        for eps in (1e-5,3e-4,3e-3):
            for trend in (0,.1,1.):
                K = np.exp(-.5*(d0+.04*dc)/ell**2) + trend * Kpoly + 10
                family.append(((ell,eps,trend),K))
    errs = []; trajerrs = []; shapeerrs = []
    for name,i,j in splits:
        sub = dict(s, train=dict(tr,x=tr['x'][i],y=tr['y'][i],traj=T[i]),test=dict(s['test'],x=tr['x'][j]))
        result = b.run_node({'__input__': sub, 'node_00': a.run_node({'__input__':sub})})
        assert result['y_hat'].shape == tr['y'][j].shape and result['state_pred'].shape == T[j].shape
        assert result['diagnostics_vec'].shape == (4,) and all(np.isfinite(v).all() for v in result.values())
        base = np.abs(result['y_hat'].ravel()-y[j]); traj = np.abs(result['state_pred']-T[j]); trajerrs.append(traj.mean(0)); shapeerrs.append(base.max())
        options=[]
        for config,K in family:
            eps=config[1]; inv=np.linalg.inv(K[np.ix_(i,i)]+eps*np.eye(len(i))); w=inv @ y[i]
            cv=np.mean(np.abs(w / inv.diagonal())); options.append((cv,config,K[np.ix_(j,i)]@w))
        cv,config,new=min(options,key=lambda x:x[0]); errs.append((name,base.mean(),np.mean(abs(new-y[j])),traj.mean(),config))
    for title, rows in [('first',errs[:1]),('random',errs[1:25]),('tails',errs[25:])]:
        print(title,'mean base/nested/traj',np.mean([r[1:4] for r in rows],axis=0), 'worst',np.max([r[1:4] for r in rows],axis=0)); print('worst cases',max(rows,key=lambda r:r[1]),max(rows,key=lambda r:r[2]))
    print('trajectory worst per horizon', np.max(trajerrs,axis=0)); print('worst single response error',max(shapeerrs))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress05.py
```
- **Container Exit Code**: `0` (`758 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
C calibration LOO mean/worst 0.0005177464657397 0.0015252239386435349
first mean base/nested/traj [0.02847252 0.0290223  0.01336903] worst [0.02847252 0.0290223  0.01336903]
worst cases ('first', 0.028472522341034933, 0.029022299885129627, 0.013369028914323238, (1.2, 0.0003, 0)) ('first', 0.028472522341034933, 0.029022299885129627, 0.013369028914323238, (1.2, 0.0003, 0))
random mean base/nested/traj [0.04201473 0.04091759 0.01552134] worst [0.08394329 0.08992192 0.02050397]
worst cases ('18', 0.08394328726281647, 0.08992192391750746, 0.012863878017368746, (1.0, 0.003, 0.1)) ('18', 0.08394328726281647, 0.08992192391750746, 0.012863878017368746, (1.0, 0.003, 0.1))
tails mean base/nested/traj [0.11722559 0.09777699 0.03596562] worst [0.17237237 0.15221421 0.06991714]
worst cases ('q3 low', 0.17237237045713888, 0.13406452142638786, 0.0371212446438249, (1.45, 0.0003, 0)) ('q1 high', 0.15728084577692072, 0.15221420822199577, 0.029263929878848998, (1.0, 0.0003, 0))
trajectory worst per horizon [0.01011284 0.02283916 0.04432802 0.05229345 0.07563888 0.0988635
 0.11426179 0.14815443]
worst single response error 0.6282951276111992

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81a20009aa6c0aab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81a20009 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`32445 ms` LLM)
- **Rationale**: Keep the accepted artifact intact while testing whether response curvature is concentrated in a low-dimensional index. Fit the index only on each split's training rows; promote it only if it improves extrapolation and retains official acceptance.
- **Alternatives Weighed**: `Promote the nested kernel immediately` (rejected: It improves the worst tail slice but worsens the worst random slice, from 0.08394 to 0.08992.)

```bash
cat > /tmp/index05.py <<'PY'
exec(open('/tmp/probe05.py').read().split('for lbl,A in')[0])
from scipy.optimize import least_squares
n=len(q); Q=poly(q); H=Q[:,5:]; splits=[('official',np.arange(n-20),np.arange(n-20,n))]
for k in range(24):
 r=np.random.default_rng(k).permutation(n);splits.append((str(k),r[:-20],r[-20:]))
for k in range(4):
 r=q[:,k].argsort();splits.extend([(f'{k}low',r[20:],r[:20]),(f'{k}high',r[:-20],r[-20:])])
def fit(i,j,kind,structure,Y=y):
 Z={'linear':Q[:,1:5],'quadratic':Q[:,1:],'pairs':np.c_[q,H[:,[1,2,3,5,6,8]]]}[structure]; core=Q[:,1:5] if structure=='quadratic' else Q[:,1:]; W=np.c_[np.ones(n),core]; blocks=np.linalg.qr(W[i])[0]; yi=Y[i]; yi=yi-blocks@(blocks.T@yi)
 def calc(p,k):
  t=p[0]+Z[k]@p[1:];return {'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'soft':lambda x:x/np.sqrt(1+x*x)}[kind](t)
 def loss(p):
  v=calc(p,i);v=v-blocks@(blocks.T@v); coef=(v@yi)/(v@v+1e-9);return np.r_[coef*v-yi,.002*p]
 w=np.linalg.lstsq(Q[i],Y[i],rcond=None)[0]; d=Z.shape[1]; rng=np.random.default_rng(15); guesses=[np.r_[0.,w[1:1+d]] if structure!='pairs' else np.r_[0.,w[1:5],w[[6,7,8,10,11,13]]]]
 guesses.extend(rng.normal(0,.8,(7,d+1))); cand=min((least_squares(loss,p,max_nfev=105) for p in guesses), key=lambda a:a.cost); p=cand.x
 A=np.c_[W[i],calc(p,i)];c=np.linalg.lstsq(A,Y[i],rcond=None)[0];return np.c_[W[j],calc(p,j)]@c,p,c
for struct in ['linear','pairs','quadratic']:
 for kind in ['sin','tanh','cos','soft']:
  es=[]
  for name,i,j in splits:
   yh,p,c=fit(i,j,kind,struct); es.append(np.mean(abs(yh-y[j])))
  _,p,c=fit(np.arange(n),np.arange(n),kind,struct); print(struct,kind,np.round([es[0],np.mean(es[1:25]),max(es[1:25]),np.mean(es[25:]),max(es[25:])],5),'p',np.round(p,2),'amp',round(c[-1],3),flush=True)
print('trajectory singular values',np.linalg.svd(T,compute_uv=False))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/index05.py
```
- **Container Exit Code**: `0` (`62293 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
linear sin [0.05055 0.05604 0.09739 0.09129 0.20501] p [-1.43  0.78 -0.63 -0.58  1.61] amp 0.578
linear tanh [0.05421 0.05334 0.07737 0.09118 0.20002] p [ 0.85 -0.56  0.46  0.43 -1.1 ] amp -1.317
linear cos [0.05056 0.05262 0.0945  0.09129 0.20501] p [ 0.14  0.78 -0.63 -0.58  1.61] amp -0.578
linear soft [0.05532 0.0651  0.23013 0.09455 0.19944] p [-0.69  0.45 -0.37 -0.34  0.88] amp 1.861
pairs sin [0.06942 0.05794 0.10888 0.12821 0.25385] p [-0.46  0.33 -0.54 -0.28  1.08 -0.29 -0.53  0.24 -0.11 -0.9  -0.87] amp 0.359
pairs tanh [0.07014 0.07053 0.30877 0.16805 0.5671 ] p [-0.21  0.15 -0.24 -0.13  0.49 -0.14 -0.23  0.1  -0.04 -0.41 -0.41] amp 2.11
pairs cos [0.06942 0.05729 0.10792 0.11747 0.27876] p [-0.96 -0.07  0.18  0.21  0.14 -0.03 -0.3   0.39  0.06 -1.07 -0.78] amp 1.109
pairs soft [0.07024 0.05701 0.09399 0.16594 0.56403] p [-0.16  0.12 -0.19 -0.1   0.38 -0.11 -0.17  0.07 -0.03 -0.32 -0.32] amp 3.115
quadratic sin [0.03432 0.04438 0.0667  0.07758 0.17692] p [-0.36 -0.03 -0.22 -0.11 -0.11 -0.12  0.49  0.4  -0.3  -0.08  0.03  0.91
  0.04  0.74 -0.17] amp -1.222
quadratic tanh [0.02922 0.03822 0.05508 0.05096 0.07392] p [ 0.29 -0.04  0.12  0.05  0.12  0.13 -0.51 -0.41  0.31  0.1  -0.01 -0.94
 -0.03 -0.76  0.22] amp 1.284
quadratic cos [0.03432 0.04306 0.06672 0.07758 0.17692] p [ 1.21 -0.03 -0.22 -0.11 -0.11 -0.12  0.49  0.4  -0.3  -0.08  0.03  0.91
  0.04  0.74 -0.17] amp 1.222
quadratic soft [0.03058 0.04008 0.05481 0.05381 0.0739 ] p [-0.24  0.05 -0.09 -0.04 -0.12 -0.12  0.46  0.37 -0.29 -0.09  0.01  0.86
  0.02  0.69 -0.2 ] amp -1.427
trajectory singular values [9.68947149 2.27233978 0.05472247 0.04862014 0.0445937  0.03721879
 0.03351494 0.03158632]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81a20009aa6c0aab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81a20009 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`38552 ms` LLM)
- **Rationale**: Compare residual correction and a small calibrated-kernel blend against the plain index with all response and trajectory fits restricted to training rows. Keep the quadratic trajectory unless a nonlinear alternative improves its tail errors without losing official accuracy.
- **Alternatives Weighed**: `Replace the calibrated kernel with the plain tanh index` (rejected: It improves extrapolation but would remove panel C from the response path.)

```bash
cat > /tmp/joint05.py <<'PY'
exec(open('/tmp/index05.py').read().split("for struct in ['linear'")[0])
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/node_01.py'); b=importlib.util.module_from_spec(sp); sp.loader.exec_module(b)
panel=s['panel_c']; c=b._readout(panel['u'],panel['v2'],q)
d=((q[:,None]-q[None,:])**2).sum(2)+.04*((c[:,None]-c[None,:])**2).sum(2)
K=np.exp(-.5*d); Z=Q[:,1:]; W=Q[:,:5]

def joint(i,j,Y,kind,starts=6):
    w=W[i]; u=np.linalg.qr(w)[0]; yi=Y[i]; yr=yi-u@(u.T@yi)
    f=np.tanh if kind=='tanh' else lambda z:z/np.sqrt(1+z*z)
    def loss(p):
        h=f(p[0]+Z[i]@p[1:]); h-=u@(u.T@h)
        return np.r_[(h[:,None]*(h@yr)/(h@h+1e-10)-yr).ravel(),.002*p]
    cov=np.linalg.lstsq(Q[i],yi,rcond=None)[0]; direction=cov[:,0]
    rng=np.random.default_rng(15)
    init=[direction]+list(rng.normal(0,.8,(starts-1,len(direction))))
    p=min((least_squares(loss,g,max_nfev=90) for g in init),key=lambda a:a.cost).x
    A=np.c_[W,f(p[0]+Z@p[1:])]; coef=np.linalg.lstsq(A[i],yi,rcond=None)[0]
    return A@coef,p

records={}; params=[]
for name,i,j in splits:
    ki=K[np.ix_(i,i)]+.003*np.eye(len(i)); kj=K[np.ix_(j,i)]; ntr=len(i)
    def record(key,pred,target): records.setdefault(key,[]).append(np.mean(abs(pred-target)))
    X=W[i]; tcoef=np.linalg.lstsq(X,T[i],rcond=None)[0]; tr=T-W@tcoef
    U,S,V=np.linalg.svd(tr[i],full_matrices=False); time=V[:2].T
    joint_y=np.column_stack((y, tr@time/np.maximum(np.std(U[:,:2]*S[:2],axis=0),1e-6)*.2))
    for kind in ('tanh','soft'):
        pred,p=joint(i,j,y[:,None],kind,8); base=pred[:,0]; params.append(p)
        correction=kj@np.linalg.solve(ki,y[i]-base[i]); kr=y[i].mean()+K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+.0001*np.eye(ntr),y[i]-y[i].mean())
        record(kind,base[j],y[j]); record(kind+' residual kernel',base[j]+correction,y[j]); record(kind+' blend',.8*base[j]+.2*kr,y[j]); record(kind+' shared',joint(i,j,joint_y,kind)[0][j,0],y[j]);
        tf,_=joint(i,j,T@time,kind); tp=W@tcoef+tf@time.T-(W@tcoef@time)@time.T
        record('traj '+kind,tp[j],T[j]); record('traj '+kind+' blend',.5*tp[j]+.5*Q[j]@np.linalg.lstsq(Q[i],T[i],rcond=None)[0],T[j])
    record('traj quadratic',Q[j]@np.linalg.lstsq(Q[i],T[i],rcond=None)[0],T[j])
for k,v in records.items():
    v=np.array(v); print(k,np.round([v[0],v[1:25].mean(),v[1:25].max(),v[25:].mean(),v[25:].max()],6), 'worst_tail',splits[25+v[25:].argmax()][0])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/joint05.py
```
- **Container Exit Code**: `0` (`18824 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tanh [0.029222 0.037943 0.055078 0.050956 0.073924] worst_tail 1low
tanh residual kernel [0.021215 0.031177 0.045822 0.047806 0.066732] worst_tail 1low
tanh blend [0.024405 0.032683 0.049076 0.049893 0.081734] worst_tail 1low
tanh shared [0.021644 0.050552 0.065096 0.071457 0.11383 ] worst_tail 2high
traj tanh [0.024132 0.023552 0.031524 0.041532 0.061752] worst_tail 3high
traj tanh blend [0.016274 0.016282 0.020699 0.035341 0.063873] worst_tail 3high
soft [0.030576 0.040079 0.054812 0.053805 0.073904] worst_tail 1low
soft residual kernel [0.021964 0.031629 0.045581 0.047569 0.064481] worst_tail 1low
soft blend [0.02557  0.0342   0.048863 0.051089 0.08091 ] worst_tail 1low
soft shared [0.021168 0.050804 0.065593 0.072987 0.115214] worst_tail 2high
traj soft [0.024142 0.023844 0.03433  0.041402 0.061874] worst_tail 3high
traj soft blend [0.016278 0.0165   0.023044 0.035372 0.064354] worst_tail 3high
traj quadratic [0.013369 0.015521 0.020504 0.035966 0.069917] worst_tail 3high

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81a20009aa6c0aab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_81a20009 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`31671 ms` LLM)
- **Rationale**: Promote the response challenger because it improves official, random, and coordinate-tail MAE while retaining panel C and leaving the trajectory model unchanged. Run the official validator and slice tests on the integrated file to catch any discrepancy between the probe and the actual node.
- **Alternatives Weighed**: `Change the trajectory fit` (rejected: Nonlinear trajectory variants modestly improved the worst tail but worsened official and random-split MAE; the quadratic fit already passes the public contract.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('node_01.py')
s = p.read_text().replace('from scipy.optimize import minimize_scalar', 'from scipy.optimize import least_squares, minimize_scalar')
start = s.index('def run_node')
s = s[:start] + '''def _response_trend(train_basis, test_basis, y):
    # Profile out the affine trend and the amplitude of the saturating term.
    affine = train_basis[:, :5]
    terms = train_basis[:, 1:]
    orth = np.linalg.qr(affine)[0]
    target = y - orth @ (orth.T @ y)

    def loss(p):
        h = np.tanh(p[0] + terms @ p[1:])
        h -= orth @ (orth.T @ h)
        amplitude = (h @ target) / (h @ h + 1e-10)
        return np.r_[amplitude * h - target, 0.002 * p]

    direction = np.linalg.lstsq(train_basis, y, rcond=None)[0]
    rng = np.random.default_rng(15)
    starts = [direction, *rng.normal(0, 0.8, (7, len(direction)))]
    p = min((least_squares(loss, g, max_nfev=90) for g in starts),
            key=lambda fit: fit.cost).x

    def design(basis):
        return np.column_stack((basis[:, :5], np.tanh(p[0] + basis[:, 1:] @ p[1:])))

    a = design(train_basis)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return y - a @ coef, design(test_basis) @ coef


def run_node(inputs: dict) -> dict:
    study, calibrated = inputs['__input__'], inputs['node_00']
    q, qt = calibrated['q_tr'], calibrated['q_te']
    train, panel = study['train'], study['panel_c']
    basis, test_basis = _quadratic(q), _quadratic(qt)
    residual, trend = _response_trend(basis, test_basis, train['y'])
    z = np.vstack((q, qt))
    z = np.column_stack((z, 0.2 * _readout(panel['u'], panel['v2'], z)))
    n = len(q)
    k = _kernel(z, z[:n])
    y_hat = trend + k[n:] @ np.linalg.solve(k[:n] + 0.003 * np.eye(n), residual)
    state_pred = test_basis @ np.linalg.lstsq(basis, train['traj'], rcond=None)[0]
    end = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
'''
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I /tmp/stress05.py
```
- **Container Exit Code**: `0` (`3467 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.021215, "public_cv_traj_mae": 0.013369}, "exec_counts": {"node_00": 1, "node_01": 1}}
C calibration LOO mean/worst 0.0005177464657397 0.0015252239386435349
first mean base/nested/traj [0.02121471 0.0290223  0.01336903] worst [0.02121471 0.0290223  0.01336903]
worst cases ('first', 0.02121470710407914, 0.029022299885129627, 0.013369028914323238, (1.2, 0.0003, 0)) ('first', 0.02121470710407914, 0.029022299885129627, 0.013369028914323238, (1.2, 0.0003, 0))
random mean base/nested/traj [0.031177   0.04091759 0.01552134] worst [0.04582157 0.08992192 0.02050397]
worst cases ('14', 0.04582157237428104, 0.03836909941388269, 0.01673054107152506, (1.0, 1e-05, 0.1)) ('18', 0.03255006106322584, 0.08992192391750746, 0.012863878017368746, (1.0, 0.003, 0.1))
tails mean base/nested/traj [0.04780602 0.09777699 0.03596562] worst [0.06673157 0.15221421 0.06991714]
worst cases ('q1 low', 0.06673156645340952, 0.08322477279272841, 0.019884782638470322, (1.45, 1e-05, 0)) ('q1 high', 0.06593694439517703, 0.15221420822199577, 0.029263929878848998, (1.0, 0.0003, 0))
trajectory worst per horizon [0.01011284 0.02283916 0.04432802 0.05229345 0.07563888 0.0988635
 0.11426179 0.14815443]
worst single response error 0.4007005265047596

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c3f1e89412cd0ec7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c3f1e894 (directory).
```

### Turn 11 — `submit_completed_task` (`15307 ms` LLM)
- **Rationale**: The exact final artifact passed isolated public validation and improved response MAE on the official split and the worst random and coordinate-tail splits, without changing the trajectory fit. The cosine calibration also achieved 0.00153 worst leave-one-panel-row-out MAE.
- **Alternatives Weighed**: `Retain the calibrated kernel without the saturating trend` (rejected: Its worst coordinate-tail response MAE was 0.17237, versus 0.06673 for the integrated model.)
- **Submitted Final Answer / Summary**: `Updated `/holdout_task/node_01.py`; the frozen A/B node is unchanged. The public validator accepted the DAG: response MAE 0.021215, trajectory MAE 0.013369, and one execution per node. On coordinate-tail stress tests, worst-slice MAE was 0.06673 for response and 0.06992 for trajectory.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel_M10_1_UnknownCut_DAG_Discovery",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
