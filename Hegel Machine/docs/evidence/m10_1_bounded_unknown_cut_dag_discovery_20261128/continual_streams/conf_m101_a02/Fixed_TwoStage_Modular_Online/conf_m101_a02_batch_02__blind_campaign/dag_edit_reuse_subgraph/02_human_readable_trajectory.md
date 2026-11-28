# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a02_batch_02__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `8 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `202,466` (`193,106` / `61,568` / `9,360`) |
| **Total Episode Wall Time (s)** | `320.03s` |

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

### Turn 1 — `execute_in_remote_docker` (`14533 ms` LLM)
- **Rationale**: Inspect the frozen output keys and evaluation contract first: these determine what the sink can use, how to preserve the trajectory path, and which held-out errors must improve.
- **Alternatives Weighed**: `Edit the upstream calibration` (rejected: The upstream node is frozen and the supplied diagnosis localizes the failure to label prediction.)

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_00.py && cat test_dag_candidate.py && cat validate_public_schema.py && cat public_asset_index.json && ls -la /pretrain_task /tmp
```
- **Container Exit Code**: `0` (`284 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"type": "ndarray"
              }
            },
            "panel_c": {
              "u": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v1": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v2": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              }
            },
            "telemetry_vec": {
              "dtype": "float64",
              "shape": [
                4
              ],
              "type": "ndarray"
            },
            "test": {
              "x": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "summary": {
                "dtype": "float64",
                "shape": [
                  4
                ],
                "type": "ndarray"
              },
              "traj": {
                "dtype": "float64",
                "shape": [
                  72,
                  8
                ],
                "type": "ndarray"
              },
              "x": {
                "dtype": "float64",
                "shape": [
                  72,
                  4
                ],
                "type": "ndarray"
              },
              "y": {
                "dtype": "float64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 5206
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2573
    }
  },
  "schema": "hegel_public_asset_index_v1"
}/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 28 09:23 ..

/tmp:
total 16
drwxrwxrwt 1 root   root   4096 Nov 28 09:23 .
drwxr-xr-x 1 root   root   4096 Nov 28 09:23 ..
drwxr-xr-x 2 root   root   4096 Nov 28 09:23 hegel_continual_archive
drwxr-xr-x 2 ubuntu ubuntu 4096 Nov 28 09:23 hegel_wormhole_skills
```

### Turn 2 — `execute_in_remote_docker` (`26647 ms` LLM)
- **Rationale**: Read the evaluator’s exact contract and compare out-of-fold errors across affine, pairwise, and cubic bases. This distinguishes missing label terms from a failure of the retained trajectory model before editing the sink.
- **Alternatives Weighed**: `Use panel C directly in the sink` (rejected: Its connection to the study outputs is not established; first measure which relationships the frozen coordinates already explain.)

```bash
cd /holdout_task && sed -n '34,180p' test_dag_candidate.py && ls -la /tmp/hegel_continual_archive && cat > /tmp/probe_dag.py <<'PY'
import pickle, numpy as np, itertools, importlib.util
from pathlib import Path
p=Path('/holdout_task'); s=importlib.util.spec_from_file_location('frozen',p/'_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
raw=pickle.load(open(p/'public_batch.pkl','rb')); print('batch', type(raw), len(raw)); study=raw[0].get('study',raw[0]); z=m.run_node({'__input__':study}); x=z['z_tr']; xt=z['z_te']; y=study['train']['y']; t=study['train']['traj']; n=len(y)
print('keys',raw[0].keys(),study.keys()); print('times',study['eval_times'],'summary',study['train']['summary'],'telemetry',study['telemetry_vec']); print('z train ranges',x.min(0),x.max(0),'test',xt.min(0),xt.max(0)); print('traj singular',np.linalg.svd(t,compute_uv=False)); print('sample z,y,traj'); print(np.column_stack((x,y,t))[:6]);
def poly(x,d):
    powers=[ix for k in range(1,d+1) for ix in itertools.combinations_with_replacement(range(x.shape[1]),k)]
    return np.column_stack([np.ones(len(x))]+[x[:,ix].prod(1) for ix in powers])
def loo(X,Y):
    B=np.linalg.lstsq(X,Y,rcond=None)[0]; e=Y-X@B; h=np.sum(X*np.linalg.pinv(X).T,axis=1); E=e/(1-h).reshape((-1,)+(1,)*(e.ndim-1)); return np.mean(abs(e),0), np.mean(abs(E),0), np.max(abs(E),0)
for d in (1,2,3):
    print('degree',d,'label',loo(poly(x,d),y),'traj',loo(poly(x,d),t))
for k in ('panel_b','panel_c'):
    a=study[k]; print(k,'u range',a['u'].min(0),a['u'].max(0));
    for v in a.keys()-{'u'}:
        print(v,'range',a[v].min(0),a[v].max(0),'linear/quad',loo(poly(a['u'],1),a[v])[:2],loo(poly(a['u'],2),a[v])[:2]); print('first rows', np.column_stack((a['u'],a[v]))[:3]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_dag.py
```
- **Container Exit Code**: `0` (`437 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
295, 0.00237182]), array([0.00657292, 0.00534213, 0.0057016 , 0.00680568, 0.00627773,
       0.00506709, 0.00624876, 0.00483196]), array([0.02941563, 0.01699399, 0.01716204, 0.03580066, 0.040441  ,
       0.01818908, 0.02057139, 0.01634466]))
panel_b u range [-1.04508272 -1.02238986 -1.18199977 -1.12628292] [1.23128892 1.18195261 1.23229707 1.14900911]
v range [-1.16975385 -1.25303071 -1.07201229 -1.06809099] [1.18871688 1.78761188 0.91646338 1.00842662] linear/quad (array([6.80789051e-05, 8.06393856e-05, 9.18600397e-05, 7.27111128e-05]), array([8.71818326e-05, 1.04249197e-04, 1.14348097e-04, 9.31209101e-05])) (array([4.05878979e-05, 5.81733200e-05, 8.09309251e-05, 5.15617216e-05]), array([0.00012615, 0.00018087, 0.00024461, 0.00015263]))
first rows [[-0.8886569   0.08486801  0.4439247  -0.39729658  0.4942171   0.15774997
   0.4480176   0.63405127]
 [ 0.44815835 -0.28131962  0.30551068  1.08960052 -0.17387978 -0.20622748
  -0.93955564 -0.20864277]
 [ 0.44731265  0.39763599 -0.05513119 -1.12360799  0.24027497  0.62527143
   0.85055604 -0.63662792]]
panel_c u range [-1.23681942 -1.13527865 -1.14306375 -1.11136772] [1.211465   1.24932725 1.19150202 0.9680721 ]
v1 range [-1.05778764 -0.89201492 -1.18045107 -1.23064479] [1.22664358 1.38068577 1.24453065 1.50642281] linear/quad (array([0.08138143, 0.09346831, 0.08592852, 0.08461374]), array([0.10604797, 0.1179476 , 0.1091212 , 0.1068903 ])) (array([0.06208797, 0.07123201, 0.05477548, 0.06330309]), array([0.22491263, 0.20581934, 0.1598926 , 0.21359741]))
first rows [[ 1.211465   -0.46880658 -0.84406145  0.59544331 -0.35148651  0.70685835
   1.22933707 -0.76095721]
 [-0.77077439  0.05456112 -0.8297546   0.18805964  0.78741082  0.88216493
   0.11170619  0.24655558]
 [-0.93503044  0.90176408 -0.43460867 -0.44195015  1.22664358  0.06931813
  -0.59890475  0.07295691]]
v2 range [-0.7860936  -0.60211458 -0.32178264 -1.00534833] [0.45111401 0.73667136 1.05437424 0.54545382] linear/quad (array([0.06423592, 0.08243041, 0.07694418, 0.07280741]), array([0.07983919, 0.10554516, 0.09921477, 0.09445746])) (array([0.00266789, 0.00269091, 0.00262025, 0.00265816]), array([0.00858217, 0.00824199, 0.00860413, 0.00919079]))
first rows [[ 1.211465   -0.46880658 -0.84406145  0.59544331  0.39767311  0.36079806
   0.72840042  0.54545382]
 [-0.77077439  0.05456112 -0.8297546   0.18805964  0.38458906 -0.08327309
   0.04330826 -0.30045275]
 [-0.93503044  0.90176408 -0.43460867 -0.44195015  0.10814816  0.17513294
   0.20133945 -0.81342951]]
```

### Turn 3 — `execute_in_remote_docker` (`39560 ms` LLM)
- **Rationale**: Compare sparse polynomial and bounded nonlinear terms in both coordinate systems to identify a label basis that improves held-out errors without sacrificing the accurate affine trajectory fit.
- **Alternatives Weighed**: `Use an unrestricted cubic label fit` (rejected: Its leave-one-out MAE already exceeds the public cutoff, so compare smaller nonlinear bases on random and coordinate-tail splits first.)

```bash
cd /holdout_task && tail -25 test_dag_candidate.py && cat > /tmp/label_basis.py <<'PY'
import numpy as np, pickle, itertools, runpy
P='/holdout_task/'
st=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; f=runpy.run_path(P+'_frozen_nodes/node_00.py'); enc=f['run_node']({'__input__':st}); y=st['train']['y']; z=enc['z_tr']; a,b=f['_calibration'](st['panel_a']); x=(st['train']['x']-a)/b
print('z range',z.min(0),z.max(0),'test range',enc['z_te'].min(0),enc['z_te'].max(0),'y range',y.min(),y.max()); print('mixer', f['_fit'](f['_affine'](st['panel_b']['u']),st['panel_b']['v']))
rng=np.random.default_rng(19); groups=[np.arange(54,72)]+[rng.permutation(len(y))[:18] for _ in range(16)]; groups += [np.argsort(abs(z[:,j]))[-12:] for j in range(4)];
def fit_eval(F,tr,te,alpha=0):
    A=F[tr]; scale=np.sqrt((A*A).mean(0)); scale[scale<1e-12]=1; A=A/scale
    if alpha:
        pen=np.eye(A.shape[1])*alpha; pen[0,0]=0; w=np.linalg.solve(A.T@A+pen,A.T@y[tr]);
    else: w=np.linalg.lstsq(A,y[tr],rcond=None)[0]
    return np.mean(abs(F[te]/scale@w-y[te]))
def score(F,alpha=0):
    e=[fit_eval(F,np.setdiff1d(np.arange(len(y)),g),g,alpha) for g in groups]; return np.round([e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],6)
def poly(p,d):
    ii=[i for k in range(1,d+1) for i in itertools.combinations_with_replacement(range(4),k)]; return np.column_stack([np.ones(len(p))]+[p[:,i].prod(1) for i in ii]),['1']+[''.join(map(str,i)) for i in ii]
def greedy(F,tr,k):
    A=F[tr]; resid=y[tr].copy(); sel=[0];
    while len(sel)<k:
        q=np.linalg.qr(A[:,sel])[0]; G=A-q@(q.T@A); norm=np.linalg.norm(G,axis=0); obj=abs(G.T@resid)/np.maximum(norm,1e-9); obj[sel]=-1; sel.append(np.argmax(obj)); resid=y[tr]-A[:,sel]@np.linalg.lstsq(A[:,sel],y[tr],rcond=None)[0]
    return sel
for label,p in [('z',z),('x',x)]:
    for d in [2,3,4,5]:
        F,terms=poly(p,d); print(label,d,[(r,score(F,r).tolist()) for r in (0,.001,.01,.1,1)])
        if d in (3,5):
            for k in (7,10,14,20):
                sel=greedy(F,np.arange(len(y)),k); e=[fit_eval(F[:,greedy(F,np.setdiff1d(np.arange(len(y)),g),k)],np.setdiff1d(np.arange(len(y)),g),g) for g in groups]; print('sparse',k,[terms[j] for j in sel], np.round([e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],6))
    Q,_=poly(p,2)
    for func in (np.sin,np.tanh,np.arctan):
        for freq in (.7,1,1.5,2,2.5,3):
            P=p*freq; S=func(P); pairs=list(itertools.combinations(range(4),2)); I=np.column_stack([func(P[:,i]+s*P[:,j]) for i,j in pairs for s in (1,-1)]); T=np.column_stack([S[:,i]*p[:,j] for i,j in itertools.permutations(range(4),2)]); print(label,func.__name__,freq,score(np.column_stack([Q,S])),score(np.column_stack([Q,S,I]),.03),score(np.column_stack([Q,S,T]),.03))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/label_basis.py
```
- **Container Exit Code**: `0` (`1209 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
22', '23', '13', '1111', '122', '22'] [0.091942 0.102272 0.165181 0.191522]
sparse 20 ['1', '0', '2', '1', '02222', '3', '1222', '03', '01222', '23', '13', '1111', '122', '22', '113', '00001', '0233', '002', '00122', '012'] [0.080187 0.102776 0.196947 0.133938]
x sin 0.7 [0.097995 0.083821 0.118977 0.142424] [0.05027  0.06239  0.087231 0.146255] [0.100374 0.0867   0.116696 0.160201]
x sin 1 [0.097902 0.083811 0.118961 0.142353] [0.048885 0.069048 0.095443 0.180574] [0.110189 0.094885 0.127434 0.166486]
x sin 1.5 [0.097652 0.083783 0.118913 0.142167] [0.054585 0.07648  0.123233 0.203656] [0.125506 0.105218 0.143906 0.17123 ]
x sin 2 [0.097241 0.083768 0.118819 0.141872] [0.062626 0.08631  0.160835 0.243415] [0.130847 0.109695 0.149922 0.17594 ]
x sin 2.5 [0.096593 0.083792 0.118638 0.141422] [0.072845 0.09798  0.194761 0.250302] [0.131874 0.110962 0.151323 0.177181]
x sin 3 [0.095578 0.083808 0.118286 0.140743] [0.083319 0.099872 0.137703 0.179644] [0.13087  0.111322 0.152504 0.181469]
x tanh 0.7 [0.09742  0.08378  0.118911 0.142048] [0.053689 0.068872 0.096544 0.154994] [0.107245 0.092516 0.124625 0.165591]
x tanh 1 [0.096772 0.083791 0.118863 0.141695] [0.058584 0.077232 0.117003 0.192261] [0.116238 0.100071 0.137156 0.168732]
x tanh 1.5 [0.095468 0.083863 0.118834 0.141103] [0.066882 0.084982 0.136244 0.215766] [0.121787 0.106339 0.147542 0.173439]
x tanh 2 [0.094452 0.083945 0.1189   0.140647] [0.072452 0.088099 0.12998  0.220508] [0.119518 0.108466 0.151445 0.182604]
x tanh 2.5 [0.093742 0.084098 0.119057 0.1404  ] [0.076411 0.08978  0.121023 0.216784] [0.115884 0.108959 0.153902 0.18927 ]
x tanh 3 [0.093456 0.08433  0.119276 0.140361] [0.079301 0.09126  0.119675 0.212197] [0.113264 0.109233 0.156087 0.195708]
x arctan 0.7 [0.09718  0.083777 0.118914 0.141931] [0.054939 0.068012 0.095608 0.147647] [0.105831 0.091452 0.123334 0.164929]
x arctan 1 [0.09645  0.083823 0.118915 0.141581] [0.058971 0.07562  0.109938 0.180796] [0.113206 0.098205 0.134455 0.168392]
x arctan 1.5 [0.09528  0.083898 0.119011 0.141107] [0.065699 0.082595 0.126611 0.202674] [0.117948 0.104076 0.144815 0.171062]
x arctan 2 [0.094539 0.083999 0.119189 0.140814] [0.070295 0.086003 0.126879 0.211948] [0.117046 0.106604 0.149556 0.179131]
x arctan 2.5 [0.093997 0.084128 0.119409 0.140683] [0.07342  0.087942 0.123191 0.214551] [0.115487 0.107568 0.152554 0.184889]
x arctan 3 [0.093736 0.084307 0.119647 0.140677] [0.075946 0.089416 0.119577 0.214654] [0.113844 0.107919 0.1549   0.188788]
```

### Turn 4 — `execute_in_remote_docker` (`45462 ms` LLM)
- **Rationale**: Fit explicit pairwise, three-way, and rational terms inside each training split. Compare them with a fitted single-index nonlinearity to see which explains the label residual without changing the accurate affine trajectory fit.
- **Alternatives Weighed**: `Increase the global polynomial degree` (rejected: The cubic already has large tail errors, and higher-degree fits have performed worse on the measured splits.)

```bash
cat > /tmp/structured_dag.py <<'PY'
import numpy as np, pickle, runpy, itertools
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; m=runpy.run_path(P+'_frozen_nodes/node_00.py'); z=m['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; n=len(y)
rng=np.random.default_rng(11)
groups=[np.arange(52,72)]+[rng.permutation(n)[:20] for _ in range(8)]+[np.argsort(abs(a))[-12:] for a in z.T]; allidx=np.arange(n)
def folds(g): return np.setdiff1d(allidx,g),g
def design(terms): return np.column_stack([np.ones(n)]+[z[:,k].prod(1) for k in terms])
terms=[k for d in (1,2,3) for k in itertools.combinations_with_replacement(range(4),d)]
A=design(terms); coef=np.linalg.lstsq(A,y,rcond=None)[0]; print('cubic coefficients',list(zip(terms,np.round(coef[1:],4))))
Q=A[:,:15]; L=A[:,:5]
def fit(F,tr,te):
 c=np.linalg.lstsq(F[tr],y[tr],rcond=None)[0]; return np.mean(abs(F[te]@c-y[te]))
def cv(F):
 e=[fit(F,*folds(g)) for g in groups]; return np.round([e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5)
for name,F in [('cubic',A),('mixed cubic',A[:,[0]+[i+1 for i,k in enumerate(terms) if len(k)<3 or len(set(k))>1]]),('pure cubic',np.column_stack([Q,z**3])),('triple',np.column_stack([Q,*[z[:,k].prod(1) for k in itertools.combinations(range(4),3)]]))]: print(name,cv(F))
dirs=[(str(i),z[:,i]) for i in range(4)]+[(f'{i}{sgn:+d}{j}',z[:,i]+sgn*z[:,j]) for i,j in itertools.combinations(range(4),2) for sgn in (-1,1)]
library=[]; names=[]
for name,u in dirs:
 for f in (.5,1.,1.5,2.,2.5,3.):
  for kind,v in [('sin',np.sin(f*u)-f*u),('cos',np.cos(f*u)-1+(f*u)**2/2),('tanh',np.tanh(f*u)-f*u),('rat',u/(1+f*u*u)-u),('gauss',np.exp(-f*u*u)-1+f*u*u)]: names.append((name,kind,f)); library.append(v)
 for j,w in enumerate(z.T):
  for f in (.5,1.,2.):
   for kind,v in [('crossrat',w/(1+f*u*u)-w),('crosssin',w*(np.sin(f*u)-f*u)),('crosscos',w*(np.cos(f*u)-1))]: names.append((name,kind,j,f)); library.append(v)
for k in terms[14:]: names.append(k); library.append(z[:,k].prod(1))
D=np.column_stack(library)
def greedy(B,tr,te,k):
 X=B.copy(); sel=[]
 for _ in range(k):
  U=np.linalg.qr(X[tr],mode='reduced')[0]; G=D[tr]-U@(U.T@D[tr]); norm=np.linalg.norm(G,axis=0); scores=abs(G.T@(y[tr]-U@(U.T@y[tr])))/np.maximum(norm,1e-9); scores[sel]=-1; j=scores.argmax(); sel.append(j); X=np.column_stack((X,D[:,j]))
 return fit(X,tr,te),sel
for B in (L,Q):
 for k in (1,2,3,4,6,8):
  e=[greedy(B,*folds(g),k)[0] for g in groups]; a,sel=greedy(B,allidx,allidx,k); print('greedy',B.shape[1],k, np.round([a,e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5),[names[j] for j in sel])
def projected(kind,B,tr,te,k=1):
 def f(t):
  u=z@t.reshape(k,5)[:,:4].T+t.reshape(k,5)[:,4];
  if kind=='sin': H=np.sin(u)
  if kind=='tanh': H=np.tanh(u)
  if kind=='atan': H=np.arctan(u)
  if kind=='rat': H=u/(1+u*u)
  if kind=='gauss': H=np.exp(-u*u)
  return np.column_stack((B,H))
 def res(t):
  X=f(t)[tr]; return X@np.linalg.lstsq(X,y[tr],rcond=None)[0]-y[tr]
 gen=np.random.default_rng(62); best=(np.inf,None)
 for i in range(10):
  t=gen.normal(0,.9,size=5*k)
  r=least_squares(res,t,bounds=(-4,4),max_nfev=65)
  if r.cost<best[0]: best=(r.cost,r.x)
 return fit(f(best[1]),tr,te),best[1]
for kind in ('sin','tanh','atan','rat','gauss'):
 for B in (L,Q):
  ae,p=projected(kind,B,allidx,allidx); e=[projected(kind,B,*folds(g))[0] for g in groups]; print('ridge term',kind,B.shape[1],np.round([ae,e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5),p.round(3))
PY
OPENBLAS_NUM_THREADS=1 python3 -u /tmp/structured_dag.py
```
- **Container Exit Code**: `0` (`26932 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'crosscos', 1, 0.5)]
greedy 5 6 [0.0386  0.10063 0.08066 0.11931 0.24108] [('0', 'crosssin', 2, 2.0), ('1', 'crosssin', 3, 2.0), ('0+11', 'crossrat', 3, 1.0), ('0-12', 'crosscos', 1, 0.5), ('2-13', 'sin', 3.0), ('1-13', 'gauss', 3.0)]
greedy 5 8 [0.03039 0.10582 0.07115 0.10746 0.22884] [('0', 'crosssin', 2, 2.0), ('1', 'crosssin', 3, 2.0), ('0+11', 'crossrat', 3, 1.0), ('0-12', 'crosscos', 1, 0.5), ('2-13', 'sin', 3.0), ('1-13', 'gauss', 3.0), (1, 2, 3), ('0', 'crosssin', 2, 1.0)]
greedy 15 1 [0.04549 0.08732 0.08297 0.11399 0.12369] [('0-12', 'crosscos', 3, 0.5)]
greedy 15 2 [0.03582 0.08431 0.07183 0.09695 0.11331] [('0-12', 'crosscos', 3, 0.5), ('2+13', 'crossrat', 1, 0.5)]
greedy 15 3 [0.02837 0.07382 0.06342 0.08962 0.08502] [('0-12', 'crosscos', 3, 0.5), ('2+13', 'crossrat', 1, 0.5), ('0+11', 'crossrat', 3, 0.5)]
greedy 15 4 [0.02311 0.06838 0.05501 0.07694 0.09983] [('0-12', 'crosscos', 3, 0.5), ('2+13', 'crossrat', 1, 0.5), ('0+11', 'crossrat', 3, 0.5), ('0-12', 'crosscos', 1, 1.0)]
greedy 15 6 [0.01691 0.05427 0.05048 0.06924 0.15564] [('0-12', 'crosscos', 3, 0.5), ('2+13', 'crossrat', 1, 0.5), ('0+11', 'crossrat', 3, 0.5), ('0-12', 'crosscos', 1, 1.0), ('0+13', 'crosscos', 0, 2.0), ('0+11', 'crossrat', 0, 1.0)]
greedy 15 8 [0.01417 0.04935 0.05642 0.08148 0.25492] [('0-12', 'crosscos', 3, 0.5), ('2+13', 'crossrat', 1, 0.5), ('0+11', 'crossrat', 3, 0.5), ('0-12', 'crosscos', 1, 1.0), ('0+13', 'crosscos', 0, 2.0), ('0+11', 'crossrat', 0, 1.0), ('2', 'crosscos', 1, 1.0), ('0+13', 'sin', 3.0)]
ridge term sin 5 [0.06645 0.0873  0.08443 0.12865 0.13312] [-1.209 -1.109  1.299  1.221  0.439]
ridge term sin 15 [0.01155 0.02087 0.02106 0.03603 0.04386] [-1.452 -1.42   1.593  1.485  0.122]
ridge term tanh 5 [0.06666 0.07915 0.08097 0.10945 0.13161] [ 0.694  0.618 -0.745 -0.684 -0.244]
ridge term tanh 15 [0.00832 0.01233 0.01426 0.01903 0.02829] [-1.001 -0.969  1.124  1.026  0.122]
ridge term atan 5 [0.06693 0.08053 0.08086 0.10634 0.13186] [ 0.67   0.596 -0.72  -0.661 -0.233]
ridge term atan 15 [0.00819 0.0125  0.01434 0.01928 0.02977] [-1.047 -1.012  1.173  1.072  0.138]
ridge term rat 5 [0.06662 0.07881 0.08109 0.11059 0.13157] [ 0.431  0.384 -0.462 -0.424 -0.152]
ridge term rat 15 [0.00841 0.01269 0.0143  0.01925 0.02783] [ 0.607  0.588 -0.682 -0.623 -0.072]
ridge term gauss 5 [0.06953 0.08115 0.09041 0.14568 0.13797] [-0.759 -0.62   0.808  0.794 -0.666]
ridge term gauss 15 [0.00847 0.01276 0.01434 0.02252 0.02628] [-0.782 -0.776  0.892  0.812  0.811]
```

### Turn 5 — `execute_in_remote_docker` (`50071 ms` LLM)
- **Rationale**: Save the first model with margin against the public threshold and verify the node contract. Compare simpler interaction sets and saturation functions on held-out rows to see whether a more exact label law improves the worst slices.
- **Alternatives Weighed**: `Add higher-order polynomial terms` (rejected: They worsened held-out errors; a saturating term captured the dominant residual with fewer parameters.)
- **ATMS NOGOOD DECLARED**: Refuted `An unrestricted cubic in the frozen coordinates is an adequate label basis.` | Witness: `Cubic label MAE was 0.04681 on the public split and 0.09081 on the worst coordinate-tail split; quadratic plus tanh reduced these to 0.01233 and 0.02829.` | Blanket Diagnosis: `The failure is confined to the label basis: an affine trajectory fit already gives leave-one-out horizon MAEs below 0.00484.` | Sublated Pivot: `Keep the frozen coordinates and affine trajectory fit; use a bounded single-index label term and test whether a smaller interaction basis explains the remaining residual.`

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'variable projection nonlinear least squares' || true
cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _quadratic(z):
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((_affine(z), z[:, i] * z[:, j]))


def _label(z, z_test, y):
    q, qt = _quadratic(z), _quadratic(z_test)
    linear = np.linalg.lstsq(_affine(z), y, rcond=None)[0][1:]

    def design(base, x, p):
        return np.column_stack((base, np.tanh(x @ p[:-1] + p[-1])))

    def residual(p):
        d = design(q, z, p)
        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y

    fits = [least_squares(residual, np.r_[a * linear, 0.0], max_nfev=100)
            for a in (0.6, 1.0, 1.6)]
    p = min(fits, key=lambda r: r.cost).x
    coef = np.linalg.lstsq(design(q, z, p), y, rcond=None)[0]
    return design(qt, z_test, p) @ coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z, z_test = (inputs['node_00'][k] for k in ('z_tr', 'z_te'))
    train = study['train']
    y_hat = _label(z, z_test, train['y'])
    coef = np.linalg.lstsq(_affine(z), train['traj'], rcond=None)[0]
    state_pred = _affine(z_test) @ coef
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat > /tmp/probe_exact.py <<'PY'
import numpy as np, pickle, runpy
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; frozen=runpy.run_path(P+'_frozen_nodes/node_00.py')['run_node']; z=frozen({'__input__':s})['z_tr']; c=runpy.run_path(P+'node_01.py'); L=c['_affine'](z); Q=c['_quadratic'](z); C=Q[:,[5+2,5+6]]; Zs=np.column_stack((z,C)); ls=np.linalg.lstsq(L,y,rcond=None)[0][1:]
rng=np.random.default_rng(133); ix=np.arange(len(y)); groups=[ix[-20:]]+[rng.permutation(ix)[:20] for _ in range(16)]+[np.argsort(abs(v))[-12:] for v in z.T]
funcs={'tanh':np.tanh,'atan':np.arctan,'rat':lambda u:u/(1+u*u),'sqrt':lambda u:u/np.sqrt(1+u*u),'sin':np.sin,'asinh':np.arcsinh}
def experiment(base, X, kind, tr, te):
    proj=funcs[kind]; U=base[tr]; Xt=X[tr]; Y=y[tr]; O=np.linalg.qr(U,mode='reduced')[0]; Y0=Y-O@(O.T@Y)
    def res(p):
        v=proj(Xt@p[:-1]+p[-1]); v=v-O@(O.T@v); return v*(v@Y0/max(v@v,1e-25))-Y0
    starts=[np.r_[a*ls,np.zeros(X.shape[1]-4),0.] for a in (.5,1.,1.6)]
    r=min((least_squares(res,p,max_nfev=90) for p in starts),key=lambda r:r.cost)
    H=np.column_stack((base,proj(X@r.x[:-1]+r.x[-1]))); b=np.linalg.lstsq(H[tr],Y,rcond=None)[0]
    return abs(H[te]@b-y[te]).mean(),r.x,b
for bn,B in [('full',Q),('cross',np.column_stack((L,C))),('pairs',Q[:,[0,1,2,3,4,6,7,8,10,11,13]])]:
 for xn,X in [('z',z), ('zC',Zs)]:
  for k in funcs:
   train,p,b=experiment(B,X,k,ix,ix); e=[experiment(B,X,k,np.setdiff1d(ix,g),g)[0] for g in groups]
   print(bn,xn,k,np.round([train,e[0],np.mean(e[1:17]),max(e[1:17]),max(e[17:])],5),'p',p.round(3),'b',b.round(3),flush=True)
print('candidate split scores',np.round([np.mean(abs(c['_label'](z[tr],z[g],y[tr])-y[g])) for g in groups for tr in [np.setdiff1d(ix,g)]],5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_exact.py
```
- **Container Exit Code**: `0` (`5733 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
1] p [-1.231 -1.22   1.415  1.317  0.095] b [-0.008 -0.619 -0.621  0.64   0.644  0.022  0.243 -0.025 -0.031  0.246
  0.027  0.383]
pairs z rat [0.00971 0.01618 0.01592 0.02379 0.02132] p [-0.676 -0.67   0.779  0.72   0.045] b [ 0.    -0.718 -0.719  0.752  0.749  0.02   0.246 -0.022 -0.028  0.248
  0.023  0.527]
pairs z sqrt [0.01001 0.0176  0.01633 0.02515 0.02214] p [-0.968 -0.96   1.114  1.034  0.071] b [-0.006 -0.642 -0.644  0.666  0.669  0.021  0.244 -0.024 -0.03   0.247
  0.026  0.457]
pairs z sin [0.01181 0.02065 0.01864 0.02432 0.03314] p [-1.501 -1.489  1.707  1.6    0.061] b [ 0.005 -0.828 -0.832  0.881  0.865  0.019  0.251 -0.014 -0.015  0.255
  0.008  0.137]
pairs z asinh [0.01049 0.01955 0.01714 0.02723 0.0233 ] p [-1.841 -1.826  2.112  1.975  0.153] b [-0.016 -0.54  -0.542  0.549  0.558  0.023  0.242 -0.027 -0.033  0.245
  0.029  0.305]
pairs zC tanh [0.00371 0.00681 0.00584 0.00861 0.00735] p [-0.953 -0.957  1.023  1.002  0.264  0.262  0.014] b [-0.003 -0.611 -0.614  0.654  0.638 -0.001  0.178 -0.003 -0.003  0.178
  0.004  0.476]
pairs zC atan [0.00375 0.00662 0.00589 0.00887 0.00649] p [-0.965 -0.97   1.033  1.011  0.272  0.264  0.013] b [-0.003 -0.536 -0.539  0.576  0.562 -0.     0.155 -0.003 -0.003  0.158
  0.003  0.554]
pairs zC rat [0.00374 0.00693 0.0059  0.00855 0.00765] p [-0.585 -0.587  0.628  0.615  0.162  0.161  0.009] b [-0.002 -0.674 -0.678  0.722  0.704 -0.001  0.196 -0.003 -0.003  0.195
  0.004  0.666]
pairs zC sqrt [0.00371 0.00662 0.00579 0.00873 0.00678] p [-0.786 -0.79   0.842  0.825  0.22   0.215  0.011] b [-0.003 -0.573 -0.577  0.615  0.6   -0.     0.167 -0.003 -0.003  0.168
  0.003  0.63 ]
pairs zC sin [0.00542 0.01025 0.00878 0.01186 0.01736] p [-1.52  -1.517  1.626  1.611  0.455  0.43   0.04 ] b [-0.003 -0.829 -0.833  0.887  0.863 -0.003  0.238 -0.003  0.002  0.234
  0.002  0.141]
pairs zC asinh [0.00387 0.00707 0.00617 0.00923 0.00685] p [-1.327 -1.336  1.416  1.389  0.381  0.362  0.018] b [-0.005 -0.416 -0.418  0.449  0.437  0.     0.119 -0.003 -0.003  0.126
  0.003  0.497]
candidate split scores [0.01233 0.01337 0.01851 0.0103  0.01794 0.01485 0.01528 0.01268 0.01638
 0.0105  0.01453 0.01473 0.01204 0.01721 0.01849 0.0151  0.01628 0.01654
 0.01742 0.02829 0.01781]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5eacee961a4c91a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5eacee9 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37777 ms` LLM)
- **Rationale**: Fit each challenger using only its fold's training rows and compare on fresh random splits and signed coordinate tails. If tying the linear and nonlinear paths or sharing the two cross-product coefficients worsens worst-slice error, keep those parameters independent. Measure trajectory error separately by horizon.
- **Alternatives Weighed**: `Keep the accepted full-quadratic candidate` (rejected: The cross-product challenger substantially reduced held-out errors; test whether its near-equal coefficients support a simpler shared-index form before replacing the verified file.)

```bash
cat > /tmp/compact_label.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _features(z, shared):
    cross = z[:, :2] * z[:, 2:]
    if shared:
        cross = cross.sum(axis=1, keepdims=True)
    return np.column_stack((z, cross))


def _label(z, z_test, y, shared=True, tied=True):
    x, xt = _features(z, shared), _features(z_test, shared)
    lin = _affine(x)
    start = np.linalg.lstsq(lin, y, rcond=None)[0][1:]

    def design(x, p):
        index = x @ p[:-1] + p[-1]
        base = _affine(x) if not tied else np.column_stack((np.ones(len(x)), index))
        return np.column_stack((base, np.tanh(index)))

    def residual(p):
        d = design(x, p)
        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y

    fits = [least_squares(residual, np.r_[scale * start, 0.0], max_nfev=100)
            for scale in (0.6, 1.0, 1.6)]
    p = min(fits, key=lambda r: r.cost).x
    b = np.linalg.lstsq(design(x, p), y, rcond=None)[0]
    return design(xt, p) @ b
PY
cat > /tmp/check_compact_label.py <<'PY'
import pickle, runpy, numpy as np
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; t=s['train']['traj']
z=runpy.run_path(P+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr']; old=runpy.run_path(P+'node_01.py'); new=runpy.run_path('/tmp/compact_label.py')
rng=np.random.default_rng(9061); ix=np.arange(len(z)); a=np.column_stack((np.ones(len(z)),z)); groups=[ix[-20:]]+[rng.permutation(ix)[:20] for _ in range(24)] + [np.argsort(abs(v))[-12:] for v in z.T] + [np.argsort(v)[-12:] for v in (*z.T, z[:,2]+z[:,3]-z[:,0]-z[:,1])] + [np.argsort(v)[:12] for v in (*z.T, z[:,2]+z[:,3]-z[:,0]-z[:,1])]; folds=[(np.setdiff1d(ix,g),g) for g in groups]
for name, fn in [('baseline',old['_label'])]+[(str((sh,ti)),lambda z,zt,y,sh=sh,ti=ti:new['_label'](z,zt,y,sh,ti)) for sh in (True,False) for ti in (True,False)]:
    mae=np.array([np.mean(abs(fn(z[tr],z[te],y[tr])-y[te])) for tr,te in folds]); print(name,'train/public/random_mean/random_max/tail_max:',np.round([np.mean(abs(fn(z,z,y)-y)),mae[0],mae[1:25].mean(),mae[1:25].max(),mae[25:].max()],6))
horizons=np.array([np.mean(abs(a[te]@np.linalg.lstsq(a[tr],t[tr],rcond=None)[0]-t[te]),axis=0) for tr,te in folds]); print('worst trajectory horizon:',horizons.max())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_compact_label.py
```
- **Container Exit Code**: `0` (`2273 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline train/public/random_mean/random_max/tail_max: [0.008321 0.012325 0.015431 0.028605 0.155216]
(True, True) train/public/random_mean/random_max/tail_max: [0.003868 0.005023 0.00447  0.006852 0.009049]
(True, False) train/public/random_mean/random_max/tail_max: [0.003881 0.005974 0.00484  0.006937 0.008448]
(False, True) train/public/random_mean/random_max/tail_max: [0.003865 0.005016 0.004527 0.00679  0.009002]
(False, False) train/public/random_mean/random_max/tail_max: [0.003883 0.00642  0.005109 0.007356 0.008977]
worst trajectory horizon: 0.0073104954114596955

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5eacee961a4c91a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c5eacee9 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`37855 ms` LLM)
- **Rationale**: Promote the shared-index model because it cuts the worst tail error while preserving the affine trajectory fit. Verify the saved node in an isolated process against the official contract and the 39 held-out slices; a regression would block submission.
- **Alternatives Weighed**: `Fit separate readouts or separate cross-product coefficients` (rejected: The worst-slice gains were below 0.0007, while the shared-index model uses fewer coefficients and fits the public split better than the untied model.)

```bash
set -e
cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _label_features(z):
    cross = (z[:, :2] * z[:, 2:]).sum(axis=1)
    return np.column_stack((z, cross))


def _label(z, z_test, y):
    x, x_test = _label_features(z), _label_features(z_test)
    initial = np.linalg.lstsq(_affine(x), y, rcond=None)[0][1:]

    def design(features, p):
        index = features @ p[:-1] + p[-1]
        return np.column_stack((np.ones(len(index)), index, np.tanh(index)))

    def residual(p):
        basis = design(x, p)
        return basis @ np.linalg.lstsq(basis, y, rcond=None)[0] - y

    fits = (least_squares(residual, np.r_[scale * initial, 0.0], max_nfev=100)
            for scale in (0.6, 1.0, 1.6))
    p = min(fits, key=lambda fit: fit.cost).x
    coef = np.linalg.lstsq(design(x, p), y, rcond=None)[0]
    return design(x_test, p) @ coef


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    z, z_test = (inputs['node_00'][k] for k in ('z_tr', 'z_te'))
    y_hat = _label(z, z_test, train['y'])
    coef = np.linalg.lstsq(_affine(z), train['traj'], rcond=None)[0]
    state_pred = _affine(z_test) @ coef
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import pickle, runpy, numpy as np
p = '/holdout_task/'
with open(p + 'public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
z = runpy.run_path(p + '_frozen_nodes/node_00.py')['run_node']({'__input__': study})['z_tr']
run = runpy.run_path(p + 'node_01.py')['run_node']
y, state = (study['train'][k] for k in ('y', 'traj'))
ix = np.arange(len(z)); rng = np.random.default_rng(9061)
dirs = (*z.T, z[:, 2] + z[:, 3] - z[:, 0] - z[:, 1])
slices = [ix[-20:]] + [rng.permutation(ix)[:20] for _ in range(24)]
slices += [np.argsort(abs(v))[-12:] for v in z.T]
slices += [g for v in dirs for g in (np.argsort(v)[:12], np.argsort(v)[-12:])]
label_errors, horizon_errors = [], []
for te in slices:
    tr = np.setdiff1d(ix, te)
    result = run({'__input__': {'train': {'y': y[tr], 'traj': state[tr]}},
                  'node_00': {'z_tr': z[tr], 'z_te': z[te]}})
    label_errors.append(np.mean(abs(result['y_hat'] - y[te])))
    horizon_errors.append(np.mean(abs(result['state_pred'] - state[te]), axis=0).max())
print(f'{len(slices)} slices | label random max={max(label_errors[1:25]):.6f} | label tail max={max(label_errors[25:]):.6f} | trajectory worst horizon={max(horizon_errors):.6f}')
assert max(label_errors) < 0.010 and max(horizon_errors) < 0.008
PY
```
- **Container Exit Code**: `0` (`920 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.005023, "public_cv_traj_mae": 0.00421}, "exec_counts": {"node_00": 1, "node_01": 1}}
39 slices | label random max=0.006852 | label tail max=0.009049 | trajectory worst horizon=0.007310

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=53eb41cfe1050ad8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_53eb41cf (directory).
```

### Turn 8 — `submit_completed_task` (`13832 ms` LLM)
- **Rationale**: The saved candidate passed the isolated official evaluator and the 39-slice stress test without changing the frozen upstream node or the affine trajectory fit.
- **Alternatives Weighed**: `Retain the full quadratic label readout` (rejected: Its worst held-out tail MAE was 0.155216, compared with 0.009049 for the shared-index readout.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py`. The official evaluator accepted it: label MAE 0.005023 and trajectory MAE 0.004210. Across 39 held-out slices, the worst label MAE was 0.009049 and the worst trajectory-horizon MAE was 0.007310.`

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
