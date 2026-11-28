# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m101_c04_batch_01__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `163,707` (`156,853` / `65,408` / `6,854`) |
| **Total Episode Wall Time (s)** | `227.79s` |

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

### Turn 1 — `execute_in_remote_docker` (`16865 ms` LLM)
- **Rationale**: Read the frozen interface, public fold protocol, and compact numerical summaries to identify which signals the sink can use and avoid changing the verified calibration.

```bash
cd /holdout_task && ls -la _frozen_nodes && cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py validate_public_schema.py && sed -n '1,210p' public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def show(o,k='',depth=0):
 if isinstance(o,dict):
  print(k,'dict',list(o));
  for a,b in o.items(): show(b,k+'/'+str(a),depth+1)
 else:
  a=np.asarray(o); print(k,type(o).__name__,a.shape,a.dtype,repr(a.ravel()[:12]))
show(d)
PY
```
- **Container Exit Code**: `0` (`468 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,  0.05636105, -0.52564845],
              [ 0.66231941,  0.38794543,  0.32323772,  0.49997825],
              [ 1.02987982, -1.28613444, -0.03061603,  0.0993336 ],
              [ 0.39873086, -0.95499511,  0.43039624,  0.53771093],
              [-0.82095006, -0.44006295, -0.06011516, -0.67510228],
              [ 0.42208247, -0.00496055,  0.57090536, -0.10811543],
              [ 0.31815735,  0.48029954, -0.37019793, -0.56594003],
              [-1.0093969 , -0.28937527, -0.67916536, -0.23088613],
              [ 0.4324289 , -0.60033231,  0.73315613,  0.55411928],
              [-0.35427016, -1.13800701, -0.29953182,  0.21680536],
              [ 0.76049568, -0.49546535, -0.90804485, -0.48897092],
              [ 0.21894148,  0.16964903,  0.726702  , -0.53094896],
              [ 0.03894171,  0.21874968,  0.59688416,  0.83869864],
              [-0.55519899, -1.14826697,  0.81314014, -0.02815346],
              [-0.39050065,  0.26915728,  0.65444142,  0.62152493],
              [-0.12939773, -0.63945811,  0.78288995, -0.36533788],
              [ 0.3565416 ,  0.57370131,  0.71270875,  0.86284101],
              [ 0.09779093,  0.97191418, -0.53191349, -0.96354638],
              [-1.23811736, -0.99021288, -0.78697351,  0.43177067],
              [ 0.8139315 ,  0.32523344,  0.8940169 ,  0.40922864],
              [ 1.25002075, -0.89423   , -0.09099454, -1.01758202],
              [ 0.77386266, -0.61918568,  0.38405347,  0.04364685],
              [ 0.91934188, -1.03667762,  0.54066236, -0.24954597],
              [ 1.08024921, -0.49699722,  0.22483369,  0.63928139],
              [ 0.19484566, -0.30969701, -0.73096749, -0.72252173],
              [ 0.35896017, -0.52056634,  0.45198238, -1.03536646],
              [ 0.18377018,  0.5113964 ,  0.66986376, -0.8159275 ],
              [-1.22056159, -0.28216087, -0.66604999,  0.64815282],
              [ 1.00614019,  0.04827686,  1.00812559, -0.82050727],
              [-0.57711154,  0.83930148,  0.72064993, -0.92680437],
              [-0.83715863,  0.07774929, -0.23756588,  0.30650292],
              [-1.2513234 , -0.39033625,  0.57269304, -0.15629798],
              [ 1.29384607,  0.45868436, -0.59665966,  0.8174032 ],
              [-0.01000897,  0.00145921,  0.93106715,  0.84128614],
              [-1.10583618,  0.17394952, -0.90333611, -0.97242792]])}, 'telemetry_vec': array([0.2, 0.2, 0.1, 1.7])}}                                                                                       ],
      dtype=object)
```

### Turn 2 — `execute_in_remote_docker` (`14906 ms` LLM)
- **Rationale**: The previous dump truncated the fold and schema details. Read those directly, summarize array shapes without dumping rows, and locate any prior trajectory implementation so the scalar revision does not alter it.

```bash
cd /holdout_task && sed -n '39,180p' test_dag_candidate.py && cat validate_public_schema.py && ls -la /pretrain_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,key='',indent=0):
 if isinstance(x,dict):
  print(key, list(x));
  for k,v in x.items(): show(v,key+'/'+k,indent+1)
 elif isinstance(x,(list,tuple)) or isinstance(x,np.ndarray) and x.dtype==object:
  print(key,type(x).__name__,len(x));
  if len(x):show(x[0],key+'/0',indent+1)
 else:
  a=np.asarray(x);print(key,a.shape,a.dtype,a.ravel()[:4])
show(x)
PY
```
- **Container Exit Code**: `0` (`433 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t))
n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        yh = np.asarray(out["y_hat"], dtype=float)
        sp = np.asarray(out["state_pred"], dtype=float)
        dv = np.asarray(out["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 28 07:59 ..
 list 1
/0 ['study']
/0/study ['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec']
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143]
/0/study/panel_a ['u', 'v']
/0/study/panel_a/u (48, 4) float64 [0.21338677 0.4127539  0.0470332  0.63110039]
/0/study/panel_a/v (48, 4) float64 [0.27487231 0.28462855 0.10831655 0.4435507 ]
/0/study/panel_b ['u', 'v']
/0/study/panel_b/u (24, 4) float64 [ 0.54933267  0.00504249  0.77528023 -0.8557417 ]
/0/study/panel_b/v (24, 4) float64 [-0.51933791  0.38391884 -1.0799201  -0.39766445]
/0/study/panel_c ['u', 'v1', 'v2']
/0/study/panel_c/u (24, 4) float64 [-0.65600093  0.28696564 -0.74496768  0.28405017]
/0/study/panel_c/v1 (24, 4) float64 [0.60530898 0.4496051  1.00251073 0.27392424]
/0/study/panel_c/v2 (24, 4) float64 [ 0.00295914 -0.00128734 -0.16678417  0.27375622]
/0/study/train ['x', 'y', 'traj', 'summary']
/0/study/train/x (72, 4) float64 [-0.99896171 -0.70824226 -0.42086318  0.12810881]
/0/study/train/y (72,) float64 [-0.54339848 -1.53368953  0.66371054  0.06494693]
/0/study/train/traj (72, 8) float64 [-0.1873431  -0.38875629 -0.54894137 -0.67432593]
/0/study/train/summary (4,) float64 [-0.33753949  1.04642573  0.22311366  1.07139275]
/0/study/test ['x']
/0/study/test/x (36, 4) float64 [-0.91604395 -1.33204866  0.21503057  0.62711514]
/0/study/telemetry_vec (4,) float64 [0.2 0.2 0.1 1.7]
```

### Turn 3 — `execute_in_remote_docker` (`34979 ms` LLM)
- **Rationale**: Compare polynomial and sparse nonlinear terms on the official row split, and use the trajectory singular values and exponential profiles to determine whether a shared response can preserve trajectory accuracy.
- **Alternatives Weighed**: `Fit panel C calibration first` (rejected: The frozen node already supplies calibrated coordinates; test the direct relationship before adding a dependency on the shifted panel.)

```bash
cat > /tmp/probe_node.py <<'PY'
import numpy as np, pickle, runpy, itertools
from scipy.optimize import least_squares
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
z=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; r=s['train']['traj']; t=s['eval_times']; a,b,c,d=z.T
print('times',t,'z ranges',z.min(0),z.max(0),'trajectory svd',np.linalg.svd(r,compute_uv=False))
cols={'1':np.ones(len(z))}
for deg in (1,2,3):
 for inds in itertools.combinations_with_replacement(range(4),deg): cols[''.join('abcd'[i] for i in inds)]=np.prod(z[:,inds],axis=1)
for i,j in itertools.combinations(range(4),2):
 for k in (-2,-1,-.5,.5,1,2): cols[f'{i}+{k}*{j}']=z[:,i]+k*z[:,j]
for name,x in list(cols.items()):
 if name=='1':continue
 for f in (np.sin,np.cos,np.tanh,np.abs):cols[f'{f.__name__}({name})']=f(x)
 for k in (.2,.4,.5,1,2):
  for j in ('1','a','b','c','d'):
   if len(name)<=2:cols[f'{j}/(1+{k}*{name}^2)']=cols[j]/(1+k*x*x)
def fit(X,Y,itr,ite):
 w=np.linalg.lstsq(X[itr],Y[itr],rcond=None)[0]; return w,np.mean(np.abs(X[ite]@w-Y[ite]),axis=0)
tr=np.arange(52);te=np.arange(52,72)
for deg in (1,2,3):
 names=[n for n in cols if n=='1' or (len(n)<=deg and all(c in 'abcd' for c in n))];X=np.array([cols[n] for n in names]).T
 for label,Y in [('scalar',y),('traj-end',r[:,-1]),('traj',r)]:
  w,err=fit(X,Y,tr,te);print(deg,label,err,[(n,round(v,4)) for n,v in zip(names,w)] if Y.ndim==1 else '')
for form in ('exp','exp-shift','exp-offset'):
 def curve(p):
  if form=='exp':return -np.expm1(-p[0]*t)
  if form=='exp-shift':return -np.expm1(-p[0]*(t+p[1]))
  return p[1]-np.exp(-p[0]*t)
 def res(p):
  g=curve(p); amp=r[tr]@g/(g@g);return (amp[:,None]*g-r[tr]).ravel()
 v=least_squares(res,[.6] if form=='exp' else [.6,.1 if form=='exp-shift' else 1.]).x
 g=curve(v); amp=r@g/(g@g);print(form,v,g,'trajectory profile MAE',np.mean(np.abs(amp[:,None]*g-r),axis=0));print('amp linear',fit(np.c_[np.ones(len(z)),z],amp,tr,te))
X=np.column_stack(list(cols.values()));names=np.array(list(cols));
for Y in (y,r[:,-1]):
 selected=[0]; unused=np.arange(1,X.shape[1]); print('greedy')
 for _ in range(12):
  G=X[tr][:,selected];q=np.linalg.qr(G)[0]; E=X[tr][:,unused]; E=E-q@(q.T@E);res=Y[tr]-q@(q.T@Y[tr]); j=np.argmax(np.abs(E.T@res)/np.maximum(np.linalg.norm(E,axis=0),1e-9)); selected.append(unused[j]); unused=np.delete(unused,j); w,err=fit(X[:,selected],Y,tr,te); print(names[selected[-1]],round(float(err),5),np.round(w,3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_node.py
cat /holdout_task/public_asset_index.json | head -100
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2]
aaa 0.00803 [ 0.107  0.805 -0.367 -0.014 -0.004 -0.016 -0.1   -0.063 -0.012 -0.007
 -0.004 -0.003]
cos(2+2*3) 0.00838 [ 0.128  0.805 -0.364 -0.011 -0.003 -0.018 -0.117 -0.076 -0.012 -0.012
 -0.005 -0.005  0.003]
{
  "files": {
    "/holdout_task/_dag_spec.json": {
      "bytes": 476,
      "summary": {
        "dag_id": "dag_02",
        "edges": {
          "first": {
            "first": "__input__",
            "len": 2,
            "type": "list"
          },
          "len": 3,
          "type": "list"
        },
        "editable_node_ids": {
          "first": "node_01",
          "len": 1,
          "type": "list"
        },
        "incoming": {
          "node_00": {
            "first": "__input__",
            "len": 1,
            "type": "list"
          },
          "node_01": {
            "first": "__input__",
            "len": 2,
            "type": "list"
          }
        },
        "nodes": {
          "first": "node_00",
          "len": 2,
          "type": "list"
        },
        "sink_node_id": "node_01",
        "topo_order": {
          "first": "node_00",
          "len": 2,
          "type": "list"
        }
      }
    },
    "/holdout_task/public_batch.pkl": {
      "bytes": 16432,
      "summary": {
        "first": {
          "study": {
            "eval_times": {
              "dtype": "float64",
              "shape": [
                8
              ],
              "type": "ndarray"
            },
            "panel_a": {
              "u": {
                "dtype": "float64",
                "shape": [
                  48,
                  4
                ],
                "type": "ndarray"
              },
              "v": {
                "dtype": "float64",
                "shape": [
                  48,
                  4
                ],
                "type": "ndarray"
              }
            },
            "panel_b": {
              "u": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
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
```

### Turn 4 — `execute_in_remote_docker` (`43922 ms` LLM)
- **Rationale**: The scan will distinguish a missing compact interaction or saturation term from general polynomial curvature. Rank terms by training residual reduction and report their held-out errors separately.
- **Alternatives Weighed**: `Increase polynomial degree` (rejected: A full quartic has 70 terms for 52 training rows, and the cubic barely improves held-out error.)
- **ATMS NOGOOD DECLARED**: Refuted `An unconstrained low-degree polynomial in calibrated coordinates suffices for scalar prediction.` | Witness: `Quadratic and cubic public-split scalar MAEs are 0.07147 and 0.07103, both above 0.035; the affine trajectory fit passes at every horizon.` | Blanket Diagnosis: `The observed miss is in the scalar functional basis, not the shared coordinate handoff or trajectory branch.` | Sublated Pivot: `Keep the calibration and affine trajectory path; test compact interaction, quotient, and nonlinear projection terms in the scalar path.`

```bash
cd /holdout_task && sed -n '84,96p' test_dag_candidate.py && OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_node.py | grep -A13 '^greedy' | head -20
cat > /tmp/scalar_scan.py <<'PY'
import pickle,runpy,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr'];y=s['train']['y'];a,b,c,d=z.T
L=np.c_[np.ones(len(z)),z]; Q=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)])
tr=np.arange(52);te=np.arange(52,72)
def score(A):
 w=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];e=A@w-y;return np.sqrt(np.mean(e[tr]**2)),np.mean(abs(e[te])),w
print('linear',score(L)[:2]);print('structured interactions',score(np.c_[L,a*c,b*d])[:2]);print('quadratic',score(np.c_[L,Q])[:2]);
F={};
for x,v in zip('abcd',z.T): F[x]=v
for i,j in itertools.combinations(range(4),2):
 for k in (-2,-1,-.5,.5,1,2):F[f'{i}:{k}:{j}']=z[:,i]+k*z[:,j]
for i,j in itertools.combinations_with_replacement(range(4),2): F['abcd'[i]+'abcd'[j]]=z[:,i]*z[:,j]
for i,j,k in itertools.combinations_with_replacement(range(4),3):F['abcd'[i]+'abcd'[j]+'abcd'[k]]=z[:,i]*z[:,j]*z[:,k]
F['ac+bd']=a*c+b*d;F['ac-bd']=a*c-b*d
S={};
for n,v in F.items():
 S[n]=v
 for q in (.5,1,1.5,2,3):
  for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('softsign',lambda x:x/(1+abs(x)))]:S[f'{fn}({q}*{n})']=f(q*v)
  if len(n)<3 or ':' in n:
   for nn,vv in [('1',1),*zip('abcd',z.T),('ac',a*c),('bd',b*d),('ac+bd',a*c+b*d),('ac-bd',a*c-b*d)]:S[f'{nn}/(1+{q}*{n}^2)']=vv/(1+q*v*v)
X=np.column_stack(list(S.values()));names=np.array(list(S));
for base in [L,np.c_[L,a*c,b*d],np.c_[L,Q]]:
 A=base.copy(); selected=[]
 for step in range(5):
  U=np.linalg.svd(A[tr],full_matrices=False)[0][:,:np.linalg.matrix_rank(A[tr])];V=X[tr]-U@(U.T@X[tr]);norm=np.linalg.norm(V,axis=0);gain=np.abs(V.T@(y[tr]-U@(U.T@y[tr])))/np.maximum(norm,1e-8);gain[norm<1e-7]=0
  ids=np.argsort(-gain)[:4];print('base',base.shape[1],'step',step,[(names[j],*np.round(score(np.c_[A,X[:,j]])[:2],5)) for j in ids]);j=ids[0];A=np.c_[A,X[:,j]];selected.append(names[j])
# Fit one compact nonlinear ridge component using variable projection.
for fname,f in [('sin',np.sin),('tanh',np.tanh),('softsign',lambda v:v/(1+abs(v))),('rational',lambda v:1/(1+v*v))]:
 hits=[];rng=np.random.default_rng(1)
 for base in [L,np.c_[L,a*c,b*d]]:
  def mat(p):return np.c_[base,f(L@p)]
  def res(p):
   A=mat(p)[tr];return A@np.linalg.lstsq(A,y[tr],rcond=None)[0]-y[tr]
  for p in rng.normal(0,.8,(24,5)):
   opt=least_squares(res,p,bounds=(-7,7),max_nfev=160);rmse,err,w=score(mat(opt.x));hits.append((rmse,err,len(base.T),np.round(opt.x,3),np.round(w,3)))
 print(fname,sorted(hits,key=lambda h:h[0])[:1])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/scalar_scan.py
```
- **Container Exit Code**: `0` (`6502 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
03438, 0.05708)]
base 7 step 0 [('c/(1+0.5*0:1:3^2)', 0.06896, 0.08343), ('c/(1+1*0:1:3^2)', 0.06965, 0.0811), ('c/(1+1.5*0:1:3^2)', 0.07024, 0.08009), ('c/(1+0.5*0:2:3^2)', 0.0705, 0.09183)]
base 7 step 1 [('b/(1+0.5*0:0.5:3^2)', 0.0549, 0.07015), ('b/(1+1*0:0.5:3^2)', 0.05493, 0.07), ('b/(1+1.5*0:0.5:3^2)', 0.05513, 0.07017), ('b/(1+2*0:0.5:3^2)', 0.05539, 0.07036)]
base 7 step 2 [('cos(1*1:-1:2)', 0.04195, 0.06223), ('1/(1+0.5*1:-1:2^2)', 0.04206, 0.06579), ('cos(0.5*1:-1:2)', 0.04208, 0.06053), ('cos(1.5*1:-1:2)', 0.04247, 0.0658)]
base 7 step 3 [('ac+bd/(1+3*cd^2)', 0.03194, 0.07215), ('ac+bd/(1+2*cd^2)', 0.03197, 0.0715), ('ac+bd/(1+1.5*cd^2)', 0.032, 0.07104), ('ac+bd/(1+1*cd^2)', 0.03207, 0.07043)]
base 7 step 4 [('bd/(1+3*0:-1:3^2)', 0.02857, 0.08678), ('a/(1+0.5*1:2:3^2)', 0.02857, 0.06435), ('a/(1+1*1:2:3^2)', 0.02872, 0.0655), ('a/(1+1.5*1:2:3^2)', 0.02888, 0.06647)]
base 15 step 0 [('d/(1+1.5*0:0.5:1^2)', 0.04475, 0.11177), ('d/(1+2*0:0.5:1^2)', 0.04489, 0.11129), ('d/(1+1*0:0.5:1^2)', 0.04499, 0.11142), ('d/(1+3*0:0.5:1^2)', 0.04568, 0.1092)]
base 15 step 1 [('c/(1+3*0:1:3^2)', 0.03359, 0.08877), ('c/(1+2*0:1:3^2)', 0.03378, 0.08641), ('c/(1+1.5*0:1:3^2)', 0.03389, 0.0845), ('c/(1+1*0:1:3^2)', 0.03403, 0.08264)]
base 15 step 2 [('ac-bd/(1+3*0:-2:2^2)', 0.02788, 0.0965), ('bd/(1+0.5*2:1:3^2)', 0.02802, 0.09448), ('ac-bd/(1+2*0:-2:2^2)', 0.02809, 0.09581), ('ac+bd/(1+0.5*2:1:3^2)', 0.02818, 0.09203)]
base 15 step 3 [('1/(1+3*1:-1:2^2)', 0.02188, 0.09468), ('1/(1+2*1:-1:2^2)', 0.02221, 0.09555), ('1/(1+1.5*1:-1:2^2)', 0.0225, 0.09623), ('1/(1+1*1:-1:2^2)', 0.02294, 0.09718)]
base 15 step 4 [('d/(1+0.5*0:0.5:3^2)', 0.01867, 0.08215), ('d/(1+1*0:0.5:3^2)', 0.01873, 0.08268), ('sin(3*bbb)', 0.01877, 0.10411), ('d/(1+1.5*0:0.5:3^2)', 0.01879, 0.08312)]
sin [(0.02020762030562371, 0.029085510092280963, 7, array([-0.043, -1.793, -1.482,  1.242, -1.743]), array([ 0.007, -1.037, -0.889,  0.64 , -1.146,  0.275,  0.246,  0.152]))]
tanh [(0.01545449636548797, 0.026737896308804708, 7, array([ 0.02 ,  1.325,  1.123, -0.915,  1.358]), array([ 0.008, -0.821, -0.7  ,  0.491, -0.913,  0.271,  0.254, -0.408]))]
softsign [(0.015316785087215805, 0.02645694602027967, 7, array([-0.001, -0.535, -0.455,  0.363, -0.533]), array([ 0.003, -0.593, -0.506,  0.341, -0.695,  0.267,  0.254,  1.65 ]))]
rational [(0.01929108581608655, 0.04779562562959101, 7, array([-0.709, -0.437, -0.375,  0.334, -0.408]), array([-2.216, -0.386, -0.32 ,  0.126, -0.564,  0.285,  0.277,  3.314]))]
```

### Turn 5 — `execute_in_remote_docker` (`37684 ms` LLM)
- **Rationale**: The fitted nonlinear projection nearly aligns with the affine scalar coefficients. Test whether they share one index, whether paired interactions belong inside or outside the link, and whether larger bases offer a reproducible improvement.
- **Alternatives Weighed**: `Deploy the first passing softsign fit` (rejected: Several link functions pass one split, so compare coefficient structure and held-out performance across additional splits before choosing the scalar law.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'variable projection separable nonlinear least squares ridge profiling' 2>&1 | head -25
cat > /tmp/link_scan.py <<'PY'
import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr'];y=s['train']['y'];a,b,c,d=z.T
L=np.c_[np.ones(len(z)),z];I=np.c_[a*c,b*d];pairs=np.c_[a*a,a*b,a*c,a*d,b*b,b*c,b*d,c*c,c*d,d*d]; tri=np.c_[a*b*c,a*b*d,a*c*d,b*c*d]
links={'tanh':np.tanh,'atan':np.arctan,'erf':__import__('scipy.special',fromlist=['erf']).erf,'softsign':lambda x:x/(1+abs(x)),'root':lambda x:x/np.sqrt(1+x*x),'sin':np.sin,'rational-odd':lambda x:x/(1+x*x),'asinh':np.arcsinh,'cubic-saturation':lambda x:x/np.cbrt(1+abs(x)**3),'exp-saturation':lambda x:np.sign(x)*(-np.expm1(-abs(x)))}
splits=[(np.r_[0:52],np.r_[52:72])];rng=np.random.default_rng(24)
for _ in range(4):
 ix=rng.permutation(72);splits.append((ix[20:],ix[:20]))
for name,f in links.items():
 for mode in ('independent','linked','interaction-in-index','quadratic','triple'):
  scores=[]
  for tr,te in splits:
   q0=np.linalg.lstsq(np.c_[L,I][tr],y[tr],rcond=None)[0]
   def predict(p):
    q=L@p[:5]
    if mode=='interaction-in-index':q=q+I@p[6:8];return q+p[5]*f(p[8]*q)
    return q+p[5]*f(p[6]*q)+I@p[7:9]
   if mode in ('linked','interaction-in-index'):
    starts=[np.r_[q0[:5]*k,m,1.7,q0[5:]] if mode=='linked' else np.r_[q0[:5]*k,m,q0[5:],1.7] for k,m in [(1,.3),(1,-.3),(.7,.6)]]
    opts=[least_squares(lambda p:(predict(p)-y)[tr],p,max_nfev=120) for p in starts];opt=min(opts,key=lambda x:x.cost);pred=predict(opt.x); pars=opt.x
   else:
    base=np.c_[L,I] if mode=='independent' else np.c_[L,pairs] if mode=='quadratic' else np.c_[L,I,tri]
    def solve(p):
     A=np.c_[base,f(L@p)]; w=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];return A@w,w
    opts=[least_squares(lambda p:(solve(p)[0]-y)[tr],q0[:5]*k,bounds=(-6,6),max_nfev=100) for k in (.6,1.4,2.)];opt=min(opts,key=lambda x:x.cost);pred,w=solve(opt.x);pars=np.r_[opt.x,w]
   e=pred-y;scores.append([np.sqrt(np.mean(e[tr]**2)),np.mean(abs(e[te])),np.max(abs(e[te]))]);
   if len(scores)==1: pfirst=pars
  scores=np.array(scores);print(name,mode,'train/public/worst/row',np.round([scores[0,0],scores[0,1],max(scores[:,1]),max(scores[:,2])],5),'params',np.round(pfirst,3),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/link_scan.py > /tmp/link_scan.log
awk '{print $5, $0}' /tmp/link_scan.log | sort -g | cut -d' ' -f2- | head -22
```
- **Container Exit Code**: `0` (`3272 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
 -0.414 -0.867  0.893 -0.722 -0.434  0.624 -0.479 -0.213  0.52  -0.3
  0.258  0.238 -0.021  0.005 -0.085 -0.009  0.143]
  0.256  0.252 -0.033  0.016 -0.052  0.007  0.632]
  0.261  0.252 -0.018  0.011 -0.051  0.006  0.361]
  0.261  0.252 -0.02   0.01  -0.051  0.006  0.574]
  0.261  0.253 -0.021  0.011 -0.05   0.007  0.4  ]
  0.26   0.253 -0.023  0.012 -0.048  0.008  0.505]
  0.26   0.253 -0.025  0.012 -0.048  0.008  0.427]
  0.263  0.253 -0.016  0.012 -0.048  0.007  0.388]
  0.259  0.253 -0.028  0.013 -0.047  0.008  0.346]
 -0.006 -0.056  0.295 -0.038 -0.03   0.028  0.205  0.01   0.018 -0.017
  0.002 -0.042  0.279 -0.025 -0.022  0.013  0.217  0.017  0.006 -0.013
  0.002 -0.04   0.279 -0.026 -0.02   0.013  0.218  0.017  0.006 -0.015
  0.004 -0.036  0.277 -0.024 -0.018  0.01   0.221  0.018  0.005 -0.015
  0.006 -0.031  0.275 -0.022 -0.015  0.007  0.224  0.019  0.004 -0.016
  0.007 -0.029  0.273 -0.021 -0.014  0.005  0.225  0.019  0.004 -0.016
  0.007 -0.028  0.27  -0.02  -0.012  0.002  0.227  0.021  0.003 -0.019
  0.008 -0.026  0.272 -0.02  -0.012  0.003  0.227  0.02   0.003 -0.017
  3.000e-03 -2.000e-02  2.374e+00]
tanh interaction-in-index train/public/worst/row [0.00442 0.0034  0.00475 0.01048] params [-0.    -0.786 -0.666  0.499 -0.86   0.458  0.18   0.179  1.589]
rational-odd interaction-in-index train/public/worst/row [0.00453 0.00345 0.00481 0.01078] params [-0.    -0.862 -0.73   0.547 -0.942  0.653  0.198  0.195  0.879]
erf interaction-in-index train/public/worst/row [0.00489 0.00359 0.00494 0.01242] params [-0.    -0.806 -0.683  0.512 -0.881  0.411  0.185  0.183  1.454]
```

### Turn 6 — `execute_in_remote_docker` (`38890 ms` LLM)
- **Rationale**: Implement the compact scalar law while keeping the affine trajectory fit and frozen calibration. The official test checks the DAG contract; blocked, random, and coordinate-tail slices test whether the symmetry constraint or extrapolation hurts either output.
- **Alternatives Weighed**: `Fit a separate coefficient for each paired product` (rejected: The exploratory tanh fit gives nearly equal coefficients (0.180 and 0.179); test their shared coefficient before adding a parameter.)

```bash
cat /holdout_task/_frozen_nodes/node_00.py
awk '/interaction-in-index/ {print}' /tmp/link_scan.log
cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _scalar_design(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), z, a*c + b*d))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z_train = np.asarray(inputs['node_00']['z_tr'], dtype=float)
    z_test = np.asarray(inputs['node_00']['z_te'], dtype=float)
    train = study['train']
    x = _scalar_design(z_train)
    xt = _scalar_design(z_test)
    y = np.asarray(train['y'], dtype=float)
    initial = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(design, p):
        q = design @ p[:-2]
        return q + p[-2] * np.tanh(p[-1] * q)

    p = least_squares(
        lambda p: predict(x, p) - y,
        np.r_[initial, 0.4, 1.7],
        bounds=(np.r_[np.full(x.shape[1], -np.inf), 0., 0.], np.inf),
    ).x
    y_hat = predict(xt, p)

    weights = np.linalg.lstsq(x[:, :5], train['traj'], rcond=None)[0]
    state_pred = xt[:, :5] @ weights
    terminal = state_pred[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state_pred,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()]),
    }
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, numpy as np
p = '/holdout_task/'
s = pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
cal = runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']
node = runpy.run_path(p+'node_01.py')['run_node']
z = cal({'__input__': s})['z_tr']
rows = np.arange(len(z)); rng = np.random.default_rng(24)
slices = {f'block{i}': rows[i:i+18] for i in range(0,len(z),18)}
slices.update({f'random{i}': rng.permutation(rows)[:20] for i in range(5)})
for j in range(z.shape[1]):
    rank = np.argsort(z[:,j]); slices[f'axis{j}-low'] = rank[:12]; slices[f'axis{j}-high'] = rank[-12:]
results = []
for label, te in slices.items():
    tr = np.setdiff1d(rows, te)
    fold = dict(s, train={k: s['train'][k][tr] for k in ('x','y','traj')}, test={'x':s['train']['x'][te]})
    out = node({'__input__':fold, 'node_00':cal({'__input__':fold})})
    y, r = out['y_hat'], out['state_pred']
    np.testing.assert_allclose(out['diagnostics_vec'], [y.mean(),y.std(),r[:,-1].mean(),r[:,-1].std()])
    results.append((label, np.mean(abs(y-s['train']['y'][te])), np.mean(abs(r-s['train']['traj'][te]),axis=0).max()))
scalar = max(results,key=lambda v:v[1]); traj = max(results,key=lambda v:v[2])
print(f'{len(results)} held-out slices: worst scalar={scalar[1]:.6f} ({scalar[0]}); worst horizon={traj[2]:.6f} ({traj[0]})')
assert scalar[1] <= .035 and traj[2] <= .035
PY
```
- **Container Exit Code**: `0` (`809 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
'u']), b['v'])
    return offset, gain, mixing


def _coordinates(x, calibration):
    offset, gain, mixing = calibration
    return _affine((x - offset) / gain) @ mixing


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibration = _calibration(study)
    return {'z_tr': _coordinates(study['train']['x'], calibration),
            'z_te': _coordinates(study['test']['x'], calibration)}
tanh interaction-in-index train/public/worst/row [0.00442 0.0034  0.00475 0.01048] params [-0.    -0.786 -0.666  0.499 -0.86   0.458  0.18   0.179  1.589]
atan interaction-in-index train/public/worst/row [0.0045  0.00397 0.00503 0.02397] params [-0.    -0.712 -0.603  0.453 -0.779  0.497  0.164  0.162  1.859]
erf interaction-in-index train/public/worst/row [0.00489 0.00359 0.00494 0.01242] params [-0.    -0.806 -0.683  0.512 -0.881  0.411  0.185  0.183  1.454]
softsign interaction-in-index train/public/worst/row [0.00635 0.00758 0.00761 0.06689] params [-0.    -0.548 -0.464  0.348 -0.6    1.889  0.127  0.126  0.891]
root interaction-in-index train/public/worst/row [0.00438 0.00359 0.00476 0.0159 ] params [-0.    -0.749 -0.634  0.476 -0.819  0.584  0.172  0.17   1.41 ]
sin interaction-in-index train/public/worst/row [0.0115  0.00887 0.01482 0.17943] params [ 1.000e-03 -1.030e+00 -8.760e-01  6.540e-01 -1.129e+00  1.580e-01
rational-odd interaction-in-index train/public/worst/row [0.00453 0.00345 0.00481 0.01078] params [-0.    -0.862 -0.73   0.547 -0.942  0.653  0.198  0.195  0.879]
asinh interaction-in-index train/public/worst/row [0.00496 0.00475 0.00573 0.03783] params [-0.    -0.59  -0.5    0.375 -0.646  0.412  0.136  0.135  3.253]
cubic-saturation interaction-in-index train/public/worst/row [0.00562 0.00389 0.00542 0.01254] params [-0.    -0.796 -0.674  0.505 -0.87   0.443  0.182  0.18   1.518]
exp-saturation interaction-in-index train/public/worst/row [0.00578 0.00684 0.00702 0.05808] params [-0.    -0.692 -0.586  0.44  -0.757  0.735  0.161  0.159  1.51 ]
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.003294, "public_cv_traj_mae": 0.004452}, "exec_counts": {"node_00": 1, "node_01": 1}}
17 held-out slices: worst scalar=0.005414 (axis0-low); worst horizon=0.007251 (axis2-low)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=66a4fb8f506ca260 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_66a4fb8f (directory).
```

### Turn 7 — `submit_completed_task` (`13452 ms` LLM)
- **Rationale**: The exact saved artifact passed the isolated official test, including one execution per node, and both outputs stayed below tolerance on blocked, random, and coordinate-tail slices. The verification receipt is bound to the saved candidate.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py` without changing the frozen calibration node. The public DAG test passed: scalar MAE 0.003294 and trajectory MAE 0.004452, both below 0.035. Across 17 additional held-out slices, worst scalar MAE was 0.005414 and worst single-horizon trajectory MAE was 0.007251.`

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
