# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a04_batch_01__blind_campaign__reuse_subgraph`

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
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `158,481` (`151,828` / `24,192` / `6,653`) |
| **Total Episode Wall Time (s)** | `223.63s` |

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

### Turn 1 — `execute_in_remote_docker` (`17620 ms` LLM)
- **Rationale**: Inspect the frozen code to detect key or shape mismatches before implementation, and inspect the public measurements and scoring harness to identify the available branch evidence and relevant validation slices.
- **Alternatives Weighed**: `Fit a full cubic in each branch` (rejected: With 24 calibration rows, a dense cubic could interpolate without identifying the readout.)

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py _frozen_nodes/node_03.py test_dag_candidate.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def scan(x,p='',depth=0):
 if isinstance(x,dict):
  for k,v in x.items(): scan(v,p+'/'+str(k),depth+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if x and depth<3: scan(x[0],p+'/0',depth+1)
 elif isinstance(x,np.ndarray): print(p,x.shape,x.dtype,x.reshape(-1)[:6])
 else: print(p,type(x).__name__,str(x)[:120])
scan(d)
PY
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sp_idx:].copy()
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
 list 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571]
/0/study/panel_a/u (48, 4) float64 [-0.94712242  1.00314251  0.7893907   0.32597144  0.40406923 -0.03654589]
/0/study/panel_a/v (48, 4) float64 [-0.94188284  1.01177655  0.93976023  0.28437768  0.41483347 -0.18654373]
/0/study/panel_b/u (24, 4) float64 [ 0.60039812  1.21836718 -1.12303636 -0.88707392 -0.27315206  0.13105484]
/0/study/panel_b/v (24, 4) float64 [ 0.22274097 -0.85334574  0.24447134  1.38551605  0.73421778  0.66490696]
/0/study/panel_c/u (24, 4) float64 [-0.51084439 -1.18511473  0.78264237 -0.51119677  0.02877639 -0.55054836]
/0/study/panel_c/v1 (24, 4) float64 [-0.33429432  0.95193006  0.54208955 -0.87022811 -0.65087144  0.04152681]
/0/study/panel_c/v2 (24, 4) float64 [ 0.45541295 -0.23663186 -0.18087072 -0.10392037  0.35680101 -0.11935412]
/0/study/train/x (72, 4) float64 [ 0.09378272  1.12436393  1.0426425  -0.42531394  0.51420915 -1.19348863]
/0/study/train/y (72,) float64 [ 0.08311726 -0.11791709 -0.66567901  0.44622124  2.14265601  0.24759716]
/0/study/train/traj (72, 8) float64 [-0.05776713 -0.08534237 -0.09759123 -0.078167   -0.05499967 -0.04018006]
/0/study/train/summary (4,) float64 [ 0.20100494  0.62394635 -0.14838362  0.85338517]
/0/study/test/x (36, 4) float64 [ 0.51248838 -0.62829931  0.67340692 -0.61592451  0.89821037  0.87276927]
/0/study/telemetry_vec (4,) float64 [0.1 0.2 0.2 0.2]
```

### Turn 2 — `execute_in_remote_docker` (`31744 ms` LLM)
- **Rationale**: Measure which candidate terms reduce panel prediction error and inspect the public harness for handoff constraints. The probe screens hypotheses; its leave-one-out scores are not independent validation after term selection.
- **Alternatives Weighed**: `Start with an unconstrained cubic fit` (rejected: The 24 panel rows cannot identify a full four-variable cubic; compare compact candidate terms first.)

```bash
cd /holdout_task && sed -n '1,85p' test_dag_candidate.py && cat >/tmp/panel_probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; x=c['u']; t=np.c_[c['v1'],c['v2']]
D={}
for i in range(4): D[str(i)]=x[:,i]
for k in (2,3):
 for p in combinations_with_replacement(range(4),k): D['*'.join(map(str,p))]=np.prod(x[:,p],axis=1)
for i in range(4):
 for j in range(i+1,4):
  for a in [-2,-1,-.5,.5,1,2]: D[f'{i}+{a:g}*{j}']=x[:,i]+a*x[:,j]
A=np.c_[np.ones(len(x)),x]; B=np.c_[np.ones(len(x)),*[D[k] for k in list(D)[:14]]]
def score(M,y):
 q=np.linalg.pinv(M); w=q@y; r=y-M@w; h=np.sum(M*q.T,axis=1); return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(1-h,1e-12))**2)),w
print('ranges x',np.round(np.stack([x.min(0),x.max(0)]),3)); print('ranges targets',np.round(np.stack([t.min(0),t.max(0)]),3))
F={k:v for k,v in D.items() if '+' not in k}
for k,z in D.items():
 for a in [.5,1,1.5,2,3]:
  for f,g in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda z:z/(1+z*z)),('abs',np.abs)]: F[f'{f}({a:g}*{k})']=g(a*z)
for k,z in list(D.items())[:14]:
 for j,w in D.items():
  if '*' in j and '+' not in j: continue
  for a in [.3,1.]: F[f'{k}/(1+{a}*({j})^2)']=z/(1+a*w*w)
for j,y in enumerate(t.T):
 print('target',j,'lin',score(A,y)[:2], 'quad',score(B,y)[:2], 'lincoef',np.round(score(A,y)[2],3))
 for name,base in [('constant',A[:,:1]),('linear',A)]:
  out=sorted((score(np.c_[base,v],y)[1],k) for k,v in F.items()); print(name,[(round(a,4),k) for a,k in out[:6]])
 m=A.copy(); sel=[]
 for _ in range(4):
  _,k=min((score(np.c_[m,v],y)[1],k) for k,v in F.items() if k not in sel); sel.append(k); m=np.c_[m,F[k]]
 print('greedy',sel,np.round(score(m,y)[0:2],4),np.round(score(m,y)[2],3))
 for f in [np.arctanh,np.arcsinh]:
  if f==np.arctanh and np.max(abs(y))>=1: continue
  print('inverse',f.__name__,*[np.round(score(M,f(y))[0:2],4) for M in (A,B)])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel_probe.py
```
- **Container Exit Code**: `0` (`7676 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.0943 0.1237] [0.0049 0.0194]
target 5 lin (0.07555649249619198, 0.09458074730871874) quad (0.0029923063112370007, 0.008455826106839133) lincoef [-0.033  0.071 -0.072 -0.36  -0.193]
constant [(0.1019, 'sin(1*2+0.5*3)'), (0.1029, 'rat(0.5*2+0.5*3)'), (0.1033, 'tanh(1*2+0.5*3)'), (0.108, 'tanh(1.5*2+0.5*3)'), (0.108, 'tanh(0.5*2+0.5*3)'), (0.1102, 'sin(1.5*2+0.5*3)')]
linear [(0.0516, 'abs(1.5*2*2*3)'), (0.0516, 'abs(3*2*2*3)'), (0.0516, 'abs(1*2*2*3)'), (0.0516, 'abs(2*2*2*3)'), (0.0516, 'abs(0.5*2*2*3)'), (0.0533, '2*2/(1+1.0*(0+-0.5*1)^2)')]
greedy ['abs(1.5*2*2*3)', '0*0/(1+0.3*(1+1*3)^2)', '0/(1+0.3*(0+1*1)^2)', 'sin(3*2+-2*3)'] [0.0138 0.0218] [-0.029 -0.324 -0.031 -0.384 -0.216 -0.157  0.096  0.486  0.023]
inverse arctanh [0.0821 0.1026] [0.0029 0.0122]
inverse arcsinh [0.0733 0.0919] [0.0039 0.0119]
target 6 lin (0.07944473534191926, 0.108529622045062) quad (0.0033057588918556672, 0.013015630383786165) lincoef [ 0.163 -0.249  0.367 -0.134  0.059]
constant [(0.1328, 'tanh(0.5*0+-2*1)'), (0.1342, 'sin(0.5*0+-2*1)'), (0.136, '1/(1+0.3*(0+2*1)^2)'), (0.1462, 'tanh(1*0+-2*1)'), (0.1474, '1/(1+1.0*(1+-0.5*3)^2)'), (0.1492, '1/(1+1.0*(0+2*1)^2)')]
linear [(0.0308, '1*1/(1+1.0*(0+-0.5*2)^2)'), (0.0323, '1*1/(1+0.3*(0+-1*2)^2)'), (0.0323, '1*1/(1+0.3*(0+-0.5*2)^2)'), (0.0369, '1*1/(1+0.3*(1+1*3)^2)'), (0.0377, '1*1/(1+0.3*(1+0.5*3)^2)'), (0.0378, '1*1/(1+0.3*(0)^2)')]
greedy ['1*1/(1+1.0*(0+-0.5*2)^2)', '0*3/(1+1.0*(0+1*2)^2)', 'cos(3*0+-1*1)', 'cos(1*0*2)'] [0.0078 0.0119] [-0.062 -0.218  0.372 -0.153  0.018  0.266  0.073  0.03   0.135]
inverse arctanh [0.1137 0.1523] [0.0203 0.0892]
inverse arcsinh [0.0747 0.1026] [0.0038 0.0129]
target 7 lin (0.09280254546358498, 0.1262573541409385) quad (0.0023836107537773855, 0.007867705620388122) lincoef [-0.239 -0.286 -0.259 -0.089  0.246]
constant [(0.2083, 'sin(0.5*0+1*1)'), (0.2102, 'tanh(0.5*0+1*1)'), (0.2152, '3/(1+1.0*(0+1*3)^2)'), (0.2198, 'rat(0.5*0+1*1)'), (0.2206, 'tanh(1*0+1*1)'), (0.2215, '1/(1+1.0*(0+-1*1)^2)')]
linear [(0.0757, '3*3/(1+1.0*(1+0.5*2)^2)'), (0.0762, '1*1/(1+0.3*(1+-2*3)^2)'), (0.0767, '3*3/(1+1.0*(1)^2)'), (0.0768, '3*3/(1+0.3*(0+-2*1)^2)'), (0.0777, '1*1/(1+0.3*(2+-2*3)^2)'), (0.078, '3*3/(1+0.3*(1+0.5*2)^2)')]
greedy ['3*3/(1+1.0*(1+0.5*2)^2)', '1*1/(1+0.3*(1+-2*3)^2)', '0*0/(1+0.3*(2+-2*3)^2)', 'cos(1.5*1+-2*3)'] [0.0122 0.0167] [-0.194 -0.284 -0.263 -0.093  0.261  0.137 -0.139 -0.095  0.02 ]
inverse arctanh [0.1503 0.2122] [0.0279 0.1323]
inverse arcsinh [0.0856 0.1158] [0.0033 0.0134]
```

### Turn 3 — `execute_in_remote_docker` (`28955 ms` LLM)
- **Rationale**: Establish an executable baseline and measure both sink errors. The coefficient and residual probe will guide whether to search for a sparse exact grammar or retain a well-validated approximation.
- **Alternatives Weighed**: `Deploy the greedily selected rational features immediately` (rejected: They were selected on the same 24 rows used to score them, and the fixed quadratic v2 basis scores better.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine basis suffices for the shifted panel v2 readout.` | Witness: `Adding quadratic products lowers leave-one-out RMSE from 0.09458–0.13267 to 0.00787–0.01462 across its four channels.` | Blanket Diagnosis: `The missed structure is in the branch mapping from calibrated inputs to the v2 readout.` | Sublated Pivot: `Use a quadratic basis for v2 and test whether the inverse-sinh quadratic candidate for v1 transfers through the unchanged sink.`

```bash
cd /holdout_task && cat >node_01.py <<'PY'
import numpy as np


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i,x.shape[1]))))


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    coef = np.linalg.lstsq(_basis(panel['u']), np.arcsinh(panel['v1']), rcond=None)[0]
    return {'b': np.sinh(_basis(panel['x']) @ coef)}
PY
cat >node_02.py <<'PY'
import numpy as np


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i,x.shape[1]))))


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    coef = np.linalg.lstsq(_basis(panel['u']), panel['v2'], rcond=None)[0]
    return {'d': _basis(panel['x']) @ coef}
PY
export OPENBLAS_NUM_THREADS=1
python3 -I test_dag_candidate.py
python3 - <<'PY'
import pickle,numpy as np
import node_02
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; x=c['u']; M=node_02._basis(x); pinv=np.linalg.pinv(M); h=(M*pinv.T).sum(1)
for k in ('v1','v2'):
 y=c[k]; print(k,'range',np.max(abs(y),axis=0).round(3));
 for name,z in [('raw',y),('asinh',np.arcsinh(y))]:
  w=pinv@z; r=z-M@w; print(name,'rms',np.sqrt(np.mean(r*r,0)).round(5),'loo',np.sqrt(np.mean((r/(1-h[:,None]))**2,0)).round(5)); print(w.round(3).T)
PY
```
- **Container Exit Code**: `0` (`466 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py', '/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
"public_cv_traj_mae": 0.064458}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
v1 range [1.164 1.621 0.856 1.374]
raw rms [5.e-05 5.e-05 8.e-05 3.e-05] loo [0.00013 0.00021 0.00022 0.0001 ]
[[ 0.123  0.     0.     0.    -0.     0.    -0.292  0.     0.571 -0.
   0.017 -0.    -0.     1.035 -0.   ]
 [ 0.037  0.    -0.     0.     0.     0.     1.088  0.     0.53  -0.
  -0.134 -0.     0.     0.016 -0.   ]
 [-0.12   0.     0.     0.    -0.     0.    -0.455  0.     0.816 -0.
  -0.535 -0.     0.    -0.569  0.   ]
 [-0.049 -0.     0.     0.     0.     0.    -0.085 -0.     0.459 -0.
   1.086 -0.     0.    -0.295 -0.   ]]
asinh rms [0.01084 0.03027 0.00943 0.01837] loo [0.04392 0.08859 0.02579 0.06062]
[[ 0.125 -0.003 -0.01   0.003 -0.001  0.003 -0.279  0.003  0.504 -0.01
   0.015 -0.039 -0.008  0.958  0.001]
 [-0.01  -0.006 -0.006  0.019  0.023  0.021  0.959 -0.008  0.456  0.046
  -0.103  0.038  0.018  0.085  0.04 ]
 [-0.131 -0.002  0.002  0.002  0.008  0.017 -0.428 -0.002  0.767 -0.
  -0.51   0.001  0.014 -0.542  0.019]
 [-0.094  0.004 -0.001 -0.014  0.003  0.027 -0.094 -0.036  0.439  0.003
   0.928 -0.043  0.015 -0.262  0.039]]
v2 range [0.795 0.539 0.852 0.898]
raw rms [0.00359 0.00299 0.00331 0.00238] loo [0.01462 0.00846 0.01302 0.00787]
[[ 0.    -0.294 -0.092  0.15  -0.344 -0.129  0.003  0.01  -0.003 -0.04
   0.008  0.003  0.074  0.003 -0.156]
 [ 0.084  0.078 -0.115 -0.412 -0.206  0.038  0.007 -0.001  0.004 -0.052
   0.007  0.004 -0.184  0.002 -0.092]
 [ 0.144 -0.243  0.379 -0.172  0.032 -0.112 -0.008  0.001 -0.003  0.167
  -0.002 -0.005 -0.074 -0.001  0.015]
 [-0.144 -0.283 -0.26  -0.113  0.265 -0.129  0.001  0.004  0.001 -0.115
   0.002  0.004 -0.047  0.     0.118]]
asinh rms [0.00488 0.00388 0.00381 0.0033 ] loo [0.01936 0.01187 0.01291 0.01338]
[[ 0.001 -0.283 -0.089  0.15  -0.324 -0.122  0.003  0.009  0.008 -0.041
   0.008  0.004  0.071  0.008 -0.142]
 [ 0.087  0.078 -0.115 -0.408 -0.203  0.033  0.007  0.007  0.008 -0.052
   0.002 -0.001 -0.184 -0.    -0.094]
 [ 0.133 -0.228  0.353 -0.169  0.021 -0.1    0.006  0.01   0.004  0.151
  -0.002 -0.02  -0.069  0.     0.01 ]
 [-0.14  -0.259 -0.24  -0.105  0.249 -0.111  0.021  0.013  0.005 -0.101
   0.009  0.004 -0.043 -0.001  0.102]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=506e8e6eda914fd2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_506e8e6e (directory).
```

### Turn 4 — `execute_in_remote_docker` (`35811 ms` LLM)
- **Rationale**: Test the v1 correction through the unchanged DAG, then refit each v2 candidate inside leave-one-out folds to distinguish a transferable additive law from a polynomial approximation.
- **Alternatives Weighed**: `Add mixed cubic terms to v2` (rejected: The fitted quadratic signal is concentrated on coordinate squares, so test additive nonlinear terms before adding interactions.)
- **ATMS NOGOOD DECLARED**: Refuted `The v1 readout needs an inverse-sinh link around a quadratic.` | Witness: `Raw quadratic v1 has worst-channel leave-one-out RMSE 0.00022 versus 0.08859 after the inverse-sinh transform; its fitted linear and squared terms are negligible.` | Blanket Diagnosis: `The inverse-sinh link distorts the v1 panel response before it reaches the frozen sink.` | Sublated Pivot: `Replace the link with a constant and pairwise products, then isolate the remaining v2 approximation error.`

```bash
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
cat >node_01.py <<'PY'
import numpy as np


def _basis(x):
    return np.column_stack((np.ones(len(x)), *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i + 1, x.shape[1]))))


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    coef = np.linalg.lstsq(_basis(panel['u']), panel['v1'], rcond=None)[0]
    return {'b': _basis(panel['x']) @ coef}
PY
python3 -I test_dag_candidate.py
cat >/tmp/v2_probe.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; y=s['panel_c']['v2']
print('panel bounds',x.min(0),x.max(0)); print('train bounds',s['train']['x'].min(0),s['train']['x'].max(0)); print('test bounds',s['test']['x'].min(0),s['test']['x'].max(0))
funcs={'cos':lambda x,a:(1-np.cos(x*a))/(a*a), 'cosh':lambda x,a:(np.cosh(x*a)-1)/(a*a), 'rational':lambda x,a:x*x/(1+a*x*x), 'log':lambda x,a:np.log1p(a*x*x)/a, 'exp':lambda x,a:-np.expm1(-a*x*x)/a, 'sqrt':lambda x,a:x*x/(1+np.sqrt(1+a*x*x)), 'exp_asym':lambda x,a:(np.expm1(x*a)-x*a)/(a*a), 'sin':lambda x,a:(x*a-np.sin(x*a))/(a**3)}
def estimate(f,x,y,hi,shared):
 def basis(x,a): return np.c_[np.ones(len(x)),x,f(x,a)]
 def residual(a):
  b=basis(x,a); return (b@np.linalg.lstsq(b,y,rcond=None)[0]-y).ravel()
 a=least_squares(residual,np.full(1 if shared else 4,min(.6,hi/2)),bounds=(.001,hi),ftol=1e-11,gtol=1e-11).x
 return a,basis,np.linalg.lstsq(basis(x,a),y,rcond=None)[0]
for deg in [2,3,4,5]:
 m=np.c_[np.ones(len(x)),*[x**i for i in range(1,deg+1)]]; p=np.linalg.pinv(m); e=y-m@(p@y); h=(m*p.T).sum(1); print('additive polynomial',deg,'rms',np.sqrt(np.mean(e**2)),'loo',np.sqrt(np.mean((e/(1-h[:,None]))**2)))
 if deg==4: print('quadratic to quartic coefficients', (p@y)[5:].round(4))
for name,f in funcs.items():
 for shared in [True,False]:
  hi=4 if name in ('cos','cosh','sin') else 3
  a,b,w=estimate(f,x,y,hi,shared); e=y-b(x,a)@w; loo=[]
  for i in range(len(x)):
   m=np.arange(len(x))!=i; aa,bb,ww=estimate(f,x[m],y[m],hi,shared); loo.append(y[i]-bb(x[i:i+1],aa)[0]@ww)
  print(name,shared,'param',a.round(4),'rms',np.sqrt(np.mean(e**2)).round(7),'worst loo',np.sqrt(np.mean(np.array(loo)**2,0)).max().round(7))
PY
python3 /tmp/v2_probe.py
```
- **Container Exit Code**: `0` (`1865 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
rain bounds [-1.12777588 -1.46963156 -1.13094141 -0.96501658] [1.11385047 1.12436393 1.32056567 0.96686309]
test bounds [-1.00885822 -1.41960603 -1.08451966 -0.95282813] [1.09067675 1.17745484 1.24299908 0.93715082]
additive polynomial 2 rms 0.004257482571960744 loo 0.00750753077775667
additive polynomial 3 rms 0.0035261967763040923 loo 0.01015644595722631
additive polynomial 4 rms 0.0001938947409060522 loo 0.001006488697208597
quadratic to quartic coefficients [[-1.605e-01  4.110e-02 -1.287e-01 -1.538e-01]
 [-4.710e-02 -6.420e-02  2.011e-01 -1.389e-01]
 [ 7.450e-02 -2.216e-01 -9.250e-02 -6.090e-02]
 [-1.823e-01 -1.105e-01  1.880e-02  1.431e-01]
 [-7.000e-04  7.000e-04 -6.000e-04  1.400e-03]
 [ 3.000e-04  1.000e-04  1.200e-03  1.000e-04]
 [-1.300e-03  2.000e-04 -8.000e-04  1.200e-03]
 [-0.000e+00  3.000e-04  8.000e-04 -4.000e-04]
 [ 2.340e-02 -4.700e-03  1.650e-02  1.990e-02]
 [ 5.100e-03  8.000e-03 -2.480e-02  1.720e-02]
 [-9.200e-03  2.940e-02  1.150e-02  8.200e-03]
 [ 2.220e-02  1.370e-02 -2.600e-03 -1.890e-02]]
additive polynomial 5 rms 9.53657007240846e-05 loo 0.0021641529078931434
cos True param [1.2914] rms 0.0002659 worst loo 0.0005788
cos False param [1.3185 1.2748 1.3054 1.2918] rms 0.0002393 worst loo 0.0005412
cosh True param [0.001] rms 0.0042575 worst loo 0.0085335
cosh False param [0.0043 0.0047 0.0043 0.0045] rms 0.0042575 worst loo 0.0085335
rational True param [0.1655] rms 0.0001522 worst loo 0.0003428
rational False param [0.1701 0.1627 0.1676 0.1653] rms 0.0001383 worst loo 0.0003178
log True param [0.3666] rms 0.0001038 worst loo 0.0002336
log False param [0.3734 0.3631 0.3692 0.3653] rms 9.73e-05 worst loo 0.0002171
exp True param [0.3003] rms 0.0002116 worst loo 0.0004673
exp False param [0.3112 0.2937 0.3056 0.3003] rms 0.0001907 worst loo 0.0004359
sqrt True param [0.9165] rms 0.0001152 worst loo 0.0002062
sqrt False param [0.9119 0.927  0.9122 0.9062] rms 0.0001121 worst loo 0.0002206
exp_asym True param [0.0709] rms 0.004132 worst loo 0.009298
exp_asym False param [0.0139 0.0473 0.3048 0.0137] rms 0.0038591 worst loo 0.0106252
sin True param [2.9345] rms 0.0715975 worst loo 0.1561957
sin False param [3.5061e+00 4.0000e+00 1.0000e-03 1.0000e-03] rms 0.0699716 worst loo 0.1555774

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=50bd7f625013b898 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_50bd7f62 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32457 ms` LLM)
- **Rationale**: Keep the accepted files intact while comparing compact even links. Refit each link parameter within each held-out panel split and compare residuals with the v1 noise scale to identify a justified challenger for end-to-end stress testing.
- **Alternatives Weighed**: `Keep the accepted dense quadratic v2 without testing its extrapolation` (rejected: Its worst panel-channel leave-one-out RMSE is 0.01462, versus 0.000206 for a shared square-root link, and train and test contain coordinates outside the calibration panel range.)

```bash
export OPENBLAS_NUM_THREADS=1
cat >/tmp/v2_link_probe.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x=s['panel_c']['u']; y=s['panel_c']['v2']; u=np.arange(len(x)); v=s['panel_c']['v1']
p=np.column_stack([np.ones(len(x))]+[x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)])
print('v1 fit noise',np.sqrt(np.mean((p@np.linalg.lstsq(p,v,rcond=None)[0]-v)**2)))
links={
 'log1p':lambda x,a:np.log1p(a*x*x)/a,
 'sqrt':lambda x,a:x*x/(1+np.sqrt(1+a*x*x)),
 'logcosh':lambda x,a:np.log(np.cosh(a*x))/a**2,
 'xtanh':lambda x,a:x*np.tanh(a*x)/a,
 'tanh2':lambda x,a:(np.tanh(a*x)/a)**2,
 'atan2':lambda x,a:(np.arctan(a*x)/a)**2,
 'asin_h2':lambda x,a:(np.arcsinh(a*x)/a)**2,
 'atan_square':lambda x,a:np.arctan(a*x*x)/a,
 'xatan':lambda x,a:x*np.arctan(a*x)/a,
 'sinc':lambda x,a:(1-np.sinc(a*x/np.pi))/a**2,
 'cos':lambda x,a:(1-np.cos(a*x))/a**2,
 'rational':lambda x,a:x*x/(1+a*x*x),
 'pseudo_huber':lambda x,a:x*x/np.sqrt(1+a*x*x),
 'sech':lambda x,a:(1-1/np.cosh(a*x))/a**2,
 'soft_exp':lambda x,a:-np.expm1(-a*x*x)/a,
}
result=[]
for name,f in links.items():
 def basis(x,a): return np.c_[np.ones(len(x)),x,f(x,a)]
 def fit(mask):
  xx=x[mask]; yy=y[mask]
  def loss(a,ret=False):
   m=basis(xx,a); w=np.linalg.lstsq(m,yy,rcond=None)[0]
   return w if ret else np.mean((m@w-yy)**2)
  opt=minimize_scalar(loss,bounds=(.01,3),method='bounded',options={'xatol':1e-10})
  return opt.x,loss(opt.x,True),np.sqrt(opt.fun)
 a,w,r=fit(u)
 e=[]; params=[]
 for i in u:
  b,ww,_=fit(u[u!=i]); e.append(basis(x[i:i+1],b)[0]@ww-y[i]); params.append(b)
 result.append((np.sqrt(np.mean(np.array(e)**2,axis=0)).max(),name,a,r,min(params),max(params)))
for z in sorted(result): print('%0.8f %-14s a=%0.5f rms=%0.8f LOO a-range=[%0.5f,%0.5f]'%z)
PY
python3 -I /tmp/v2_link_probe.py
```
- **Container Exit Code**: `0` (`750 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
v1 fit noise 8.443513523092796e-05
0.00017192 logcosh        a=1.10073 rms=0.00008626 LOO a-range=[1.09942,1.10186]
0.00017721 xatan          a=0.80401 rms=0.00009678 LOO a-range=[0.80303,0.80505]
0.00017984 asin_h2        a=0.77283 rms=0.00008649 LOO a-range=[0.77197,0.77363]
0.00018337 pseudo_huber   a=0.39065 rms=0.00008708 LOO a-range=[0.38975,0.39146]
0.00020624 sqrt           a=0.91646 rms=0.00011524 LOO a-range=[0.91263,0.91971]
0.00023365 log1p          a=0.36664 rms=0.00010383 LOO a-range=[0.36554,0.36789]
0.00025929 xtanh          a=0.72933 rms=0.00011459 LOO a-range=[0.72813,0.73087]
0.00029144 atan2          a=0.50956 rms=0.00012861 LOO a-range=[0.50865,0.51088]
0.00034280 rational       a=0.16549 rms=0.00015219 LOO a-range=[0.16483,0.16656]
0.00035278 sech           a=0.62777 rms=0.00015686 LOO a-range=[0.62650,0.62986]
0.00039855 tanh2          a=0.48722 rms=0.00017853 LOO a-range=[0.48616,0.48909]
0.00046728 soft_exp       a=0.30033 rms=0.00021158 LOO a-range=[0.29892,0.30305]
0.00054628 sinc           a=1.68560 rms=0.00024999 LOO a-range=[1.68135,1.69440]
0.00057877 cos            a=1.29141 rms=0.00026588 LOO a-range=[1.28808,1.29849]
0.00160384 atan_square    a=0.52432 rms=0.00078101 LOO a-range=[0.51877,0.53541]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=50bd7f625013b898 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_50bd7f62 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34791 ms` LLM)
- **Rationale**: Test the exact candidate in a clean process. The coordinate-tail splits check extrapolation that ordinary row deletion may miss; if the new branch fails either the public contract or the tail check, inspect the failure before promoting it.
- **Alternatives Weighed**: `Use a fitted additive quartic` (rejected: It has higher held-out panel error and polynomial growth beyond the calibration range.); `Hard-code the observed log-cosh scale` (rejected: Profiling one shared scale lets each supplied calibration panel determine it without using sink labels.)
- **ATMS NOGOOD DECLARED**: Refuted `The dense quadratic captures v2 to the precision of the panel measurements.` | Witness: `Refitting a shared log-cosh scale in each leave-one-out fold cuts worst-channel RMSE from 0.01462 to 0.000172; its in-sample residual scale is comparable to v1.` | Blanket Diagnosis: `The coordinate-square terms approximate an even nonlinear response, leaving systematic error despite additional mixed terms.` | Sublated Pivot: `Fit affine and shared log-cosh terms to panel v2, then test both branches on omitted coordinate tails and rerun the frozen DAG.`

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
cat >node_02.py <<'PY'
import numpy as np


def _basis(x, scale):
    even = np.logaddexp(scale * x, -scale * x) - np.log(2.0)
    return np.column_stack((np.ones(len(x)), x, even))


def _fit(u, v):
    # Profile out the linear coefficients; all channels share one scale.
    def fit_at(scale):
        basis = _basis(u, scale)
        coef = np.linalg.lstsq(basis, v, rcond=None)[0]
        return np.sum((basis @ coef - v)**2), coef

    grid = np.geomspace(0.1, 4.0, 17)
    i = min(range(len(grid)), key=lambda j: fit_at(grid[j])[0])
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    for _ in range(28):
        a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
        if fit_at(a)[0] <= fit_at(b)[0]:
            hi = b
        else:
            lo = a
    scale = (lo + hi) / 2
    return scale, fit_at(scale)[1]


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    scale, coef = _fit(panel['u'], panel['v2'])
    return {'d': _basis(panel['x'], scale) @ coef}
PY
python3 -I test_dag_candidate.py
python3 -I - <<'PY'
import pickle, runpy
import numpy as np
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
branches = [runpy.run_path('/holdout_task/node_0%d.py' % i)['run_node'] for i in (1, 2)]
for item in batch:
    c = item['study']['panel_c']; u = c['u']
    splits = {'leave-one-out': [[i] for i in range(len(u))], 'coordinate tails': [np.argsort(sign*u[:, j])[-3:] for j in range(u.shape[1]) for sign in (-1,1)]}
    for branch, target, key in zip(branches, ('v1','v2'), ('b','d')):
        for name, tests in splits.items():
            scores = []
            for rows in tests:
                keep = np.ones(len(u), dtype=bool); keep[rows] = False
                pred = branch({'node_00': {'u': u[keep], target: c[target][keep], 'x': u[rows]}})[key]
                assert pred.shape == c[target][rows].shape and np.isfinite(pred).all()
                scores.append(pred-c[target][rows])
            print(target, name, 'worst channel RMSE', np.sqrt(np.mean(np.concatenate(scores)**2, axis=0)).max(), 'max absolute error', np.max(np.abs(scores)))
PY
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.009885, "public_cv_traj_mae": 0.016725}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
v1 leave-one-out worst channel RMSE 0.0001392152745669075 max absolute error 0.00032014820454751014
v1 coordinate tails worst channel RMSE 0.00015005986147925834 max absolute error 0.0003552683923145894
v2 leave-one-out worst channel RMSE 0.00017192633727140282 max absolute error 0.0003499193725260419
v2 coordinate tails worst channel RMSE 0.000196454405880084 max absolute error 0.0005094649955590391

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fa436dc64421c8cb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_fa436dc6 (directory).
```

### Turn 7 — `submit_completed_task` (`14924 ms` LLM)
- **Rationale**: The exact saved bundle passed the isolated Python DAG test, and both branches passed held-out row and coordinate-tail checks. No further edits or verification are needed.
- **Alternatives Weighed**: `Revert v2 to the accepted dense quadratic` (rejected: The shared log-cosh model improved both public DAG errors and reduced worst-channel panel leave-one-out RMSE from 0.01462 to 0.000172.)
- **Submitted Final Answer / Summary**: `Implemented and verified `node_01.py` and `node_02.py`. The public DAG test accepted both: target MAE was 0.009885 and trajectory MAE was 0.016725, below the 0.038 limit. Each node ran once, and the frozen nodes were unchanged.`

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
