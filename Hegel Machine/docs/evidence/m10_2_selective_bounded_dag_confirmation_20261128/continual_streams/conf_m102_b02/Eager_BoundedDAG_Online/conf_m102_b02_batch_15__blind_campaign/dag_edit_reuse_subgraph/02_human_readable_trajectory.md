# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b02_batch_15__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_2_selective_bounded_dag_discovery_h16` / `m102_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `365,158` (`349,745` / `87,040` / `15,413`) |
| **Total Episode Wall Time (s)** | `504.8s` |

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

### Turn 1 — `execute_in_remote_docker` (`14923 ms` LLM)
- **Rationale**: Inspect the frozen interface, public validation protocol, measured arrays, and archived implementation before selecting a feature transformation.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_01.py _frozen_nodes/node_02.py test_dag_candidate.py && ls -la /tmp/hegel_continual_archive && find /tmp/hegel_continual_archive -maxdepth 3 -type f | head -100 && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,p='',d=0):
 if isinstance(x,dict):
  for k,v in x.items():show(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if x and d<4:show(x[0],p+'/0',d+1)
 elif isinstance(x,np.ndarray):print(p,x.shape,x.dtype,x.reshape(-1)[:8]);
 else:print(p,type(x).__name__,str(x)[:180])
show(x)
PY
```
- **Container Exit Code**: `0` (`375 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ot 2692 Nov 28 17:06 archive_manifest.json
-rw-r--r-- 1 root root 1803 Nov 28 17:06 conf_m102_b02_batch_00_solver.py
-rw-r--r-- 1 root root 2111 Nov 28 17:06 conf_m102_b02_batch_01_solver.py
-rw-r--r-- 1 root root 2651 Nov 28 17:06 conf_m102_b02_batch_05_solver.py
-rw-r--r-- 1 root root 2357 Nov 28 17:06 conf_m102_b02_batch_11_solver.py
-rw-r--r-- 1 root root 1882 Nov 28 17:06 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py
/tmp/hegel_continual_archive/conf_m102_b02_batch_11_solver.py
/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py
 list 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
/0/study/panel_a/u (48, 4) float64 [ 0.49744411 -0.61150627 -0.19449032 -1.28182484 -0.69719608  0.80421556
 -0.05836287  0.48826406]
/0/study/panel_a/v (48, 4) float64 [ 0.85697539 -0.87280927 -0.45885936 -0.77942045 -0.6230805   0.59170071
  0.31603087  0.37678342]
/0/study/panel_b/u (24, 4) float64 [ 0.14413493 -0.29402077 -0.8014562  -0.15324534  0.38744535  0.8441116
 -0.69211191 -0.52597976]
/0/study/panel_b/v (24, 4) float64 [ 0.40841857  0.10732401 -0.70918207  0.83485516  0.1297776   0.48860062
  0.40774943  0.18070683]
/0/study/panel_c/u (24, 4) float64 [ 0.41474094  0.73284424 -1.22440503  1.04585734 -1.02310357 -0.77407634
 -0.97754817  0.26356452]
/0/study/panel_c/v1 (24, 4) float64 [ 0.45602532  0.39579942  1.30877225  1.56509542 -1.28601043  0.39968105
  0.41904177 -0.12946417]
/0/study/panel_c/v2 (24, 4) float64 [-0.41607414 -0.21990105 -0.73781166  0.30193003  0.33313988 -0.18112236
 -0.01033079  0.49165327]
/0/study/train/x (72, 4) float64 [ 1.23761379  0.56748235  0.07813859 -0.27549949 -0.23207887 -1.4280667
 -0.0661218   0.884053  ]
/0/study/train/y (72,) float64 [ 0.44839213  0.64180119  0.80241578 -0.71006414  1.32656707  0.29753955
  0.06335055  0.80527903]
/0/study/train/traj (72, 8) float64 [0.45911624 0.86770879 1.09668099 1.24155323 1.32966398 1.38229299
 1.42353969 1.45997408]
/0/study/train/summary (4,) float64 [-0.00715112  0.81396893 -0.0808493   1.38505377]
/0/study/test/x (36, 4) float64 [ 0.531864   -0.32967186  0.24875211  0.05329849  0.55655964 -0.49719266
 -0.86356122  0.43623934]
/0/study/telemetry_vec (4,) float64 [1.25 2.15 0.22 2.15]
```

### Turn 2 — `execute_in_remote_docker` (`10857 ms` LLM)
- **Rationale**: Read the archived feature equations and public split before transferring calibration or proposing new terms.

```bash
cd /holdout_task && sed -n '100,220p' test_dag_candidate.py; for p in /tmp/hegel_continual_archive/*; do echo ====; cat "$p"; done
```
- **Container Exit Code**: `0` (`268 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e),
             (erf(1.4*b), .003, False, True))
    y = np.zeros(len(xt))
    predicted_modes = np.zeros((len(xt), len(basis)))
    eye = np.eye(n)
    for h, ridge, scalar, dynamic in specs:
        f = _mixed_design(b, h)
        p = (f[n:] @ f[:n].T) @ np.linalg.solve(f[:n] @ f[:n].T + ridge*eye, targets)
        if scalar:
            y += p[:, 0] / 3
        if dynamic:
            predicted_modes += p[:, 1:] / 4
    state = predicted_modes @ basis
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
====
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:0.771458",
    "public_cv_traj_mae_exceeded:1.324520"
  ],
  "public_input_sha256": "73adc95a7be34c9cae47bbe3901695dd83dc5bc62d7966ebbf84ce04e13a8c02",
  "candidate_raw_sha256": "6e3f174081f1e338179abaa31464da93f2271be17a4db1c562a0e2bb5cea59e7",
  "setup_source_sha256": "94eb20b6813d49a072d1b701bcd44436d67b799bc3532756f7ed1fe7bd7b48db",
  "cv_metrics": {
    "public_cv_y_mae": 0.771458,
    "public_cv_traj_mae": 1.32452
  },
  "dag_runtime_trace": {
    "active_node_count": 4,
    "all_declared_edges_consumed": true,
    "all_exec_count_eq_1": true,
    "dag_id": "dag_09",
    "depth": 3,
    "edge_consumption_checks": {
      "node_00": [
        "__input__"
      ],
      "node_01": [
        "__input__"
      ],
      "node_02": [
        "node_00",
        "node_01"
      ],
      "node_03": [
        "node_00",
        "node_02"
      ]
    },
    "exec_counts": {
      "node_00": 1,
      "node_01": 1,
      "node_02": 1,
      "node_03": 1
    },
    "max_fanout": 2,
    "node_code_shas": {
      "node_00": "3f495f8fc0db85ef622874e3135becb281a158f8fb03ecc1ae8a16064391d206",
      "node_01": "9f2130796f6dec25710f814b513d6ac2bbf79d108106009f418093e2f33a36f9",
      "node_02": "8191b2fbe0ef047dd00aa1118b60f793228a9027208fb499ffeeeecc165374fb",
      "node_03": "05c4381ac51f9832b4875fb486f4d8570b53b68e5f7da93cf87dc06c559d4b49"
    },
    "non_identity_checks": {
      "node_00": true,
      "node_01": true,
      "node_02": true
    },
    "shared_fanout_nodes": [
      "node_00"
    ],
    "topo_order": [
      "node_00",
      "node_01",
      "node_02",
      "node_03"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2743.39
}
```

### Turn 3 — `execute_in_remote_docker` (`31433 ms` LLM)
- **Rationale**: Recover the archived calibration equations and public split, which were truncated in the previous output, before testing feature maps through the frozen kernel and sink.
- **Alternatives Weighed**: `Copy the prior standalone solver` (rejected: It returns predictions, not the feature matrices the frozen kernel consumes.)

```bash
cd /holdout_task; cat public_asset_index.json; cat /tmp/hegel_continual_archive/archive_manifest.json; for f in /tmp/hegel_continual_archive/*solver.py; do echo "$f"; sed -n '1,44p' "$f"; done; grep -n 'sp_idx\|vx =' test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment calibration'
```
- **Container Exit Code**: `0` (`292 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
decay):
    dot=x@x.T/x.shape[1]
    dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0)
    return (1+dot)**degree+np.exp(-decay*dist)

def predict(study):
    z,b=parts(study)
    f=times(study)
    tr=study['train']; n=len(tr['x'])
/tmp/hegel_continual_archive/conf_m102_b02_batch_11_solver.py
"""Calibrate the input, apply the measured mixing operator, and predict two shared temporal modes."""
import numpy as np
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    panel = study['panel_a']
    a, v = _design(panel['u']), np.asarray(panel['v'])
    keep = np.ones(len(a), dtype=bool)
    for _ in range(10):
        coef = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
        e = np.linalg.norm(a @ coef - v, axis=1)
        new = e < max(.001, 3 * np.median(e))
        if np.array_equal(new, keep) or new.sum() < a.shape[1]:
            break
        keep = new
    z = np.linalg.solve(coef[1:].T, (x - coef[0]).T).T
    panel = study['panel_b']
    mix = np.linalg.lstsq(_design(panel['u']), panel['v'], rcond=None)[0]
    return _design(z) @ mix


def _mixed_design(b, h):
    # Include every bilinear interaction, but only diagonal pure quadratics.
    return _design(np.column_stack((b, h, b*b, h*h,
                                   (b[:, :, None] * h[:, None, :]).reshape(len(b), 16))))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = np.asarray(train['x']), np.asarray(study['test']['x'])
    n = len(x)
    b = _coordinates(study, np.concatenate((x, xt)))
    traj = np.asarray(train['traj'])
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    modes = traj @ basis.T
    targets = np.column_stack((train['y'], modes))
    soft = b / np.sqrt(1 + (1.4*b)**2)
    specs = ((soft, .0003, True, True),
             (np.tanh(1.75*b), .0003, True, False),
152:    sp_idx = max(40, len(st["train"]["x"]) - 20)
153:    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
155:        "x": st["train"]["x"][:sp_idx].copy(),
156:        "y": st["train"]["y"][:sp_idx].copy(),
157:        "traj": st["train"]["traj"][:sp_idx].copy(),
158:        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
No matching skill cards found.
```

### Turn 4 — `execute_in_remote_docker` (`18558 ms` LLM)
- **Rationale**: Measure which archived transforms explain each reference panel before using the training responses to select predictive features; inspect the trajectory spectrum separately to establish the rank-two approximation error.

```bash
for f in /tmp/hegel_continual_archive/*batch_0[015]*py; do echo ====; cat "$f"; done
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(x):return np.c_[np.ones(len(x)),x]
for k in ('panel_a','panel_b','panel_c'):
 p=s[k];u=p['u']; cyc=u*np.roll(u,-1,axis=1); print(k, 'u range',u.min(),u.max())
 for l,v in p.items():
  if l=='u':continue
  for name,f in [('aff',D(u)),('cyc+cubic',D(np.c_[u,cyc,u**3])),('deadzone',D(np.sign(u)*np.maximum(abs(u)-.35,0)**1.5)),('tanh',D(np.tanh(1.25*u)))]:
   w=np.linalg.lstsq(f,v,rcond=None)[0]; e=np.linalg.norm(f@w-v,axis=1); print(l,name,np.round(np.quantile(e,[0,.25,.5,.75,.9,1]),5))
   if name=='aff':print('coef',np.round(w,3))
y=s['train']['y'];t=s['train']['traj']; print('traj sv',np.linalg.svd(t,compute_uv=False)); print('corr',np.corrcoef(np.c_[y,t].T)[0]); print('telemetry',s['telemetry_vec'])
PY
```
- **Container Exit Code**: `0` (`359 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nd(a,b,w,target):
        pred=head(a,target)
        return pred if w==1 else w*pred+(1-w)*head(b,target)
    y=blend(*((0, 2, 1.4, 1e-06), (2, 3, 0.15, 1e-06), 0.5),tr['y'])
    state=blend(*((1, 3, 0.15, 1e-06), (1, 3, 1.4, 1e-06), 0.25),modes)@f.T
    return y,state

def solve_batch(study:dict)->dict:
    y,state=predict(study); end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
panel_a u range -1.2886115296840148 1.2877000869877093
v aff [ 0.11748  0.55986  0.70932  1.05081  1.47204 11.99227]
coef [[ 0.326 -0.073 -0.018  0.049]
 [ 1.53  -0.389  0.069 -0.352]
 [ 0.253  1.348 -0.182 -0.564]
 [-0.233 -0.478  1.483  0.068]
 [-0.183 -0.626  0.337  1.202]]
v cyc+cubic [ 0.17254  0.61185  1.0011   1.60821  2.39675 10.85679]
v deadzone [ 0.23225  0.59691  0.8496   1.26038  1.74149 12.35632]
v tanh [ 0.16259  0.63995  0.76326  0.99631  1.49966 11.79291]
panel_b u range -1.2362636301822425 1.1827453379279982
v aff [0.49261 0.90143 1.01486 1.08577 1.29134 1.49059]
coef [[ 0.069 -0.068 -0.018 -0.155]
 [-0.183  0.25   0.223 -0.013]
 [ 0.241  0.298  0.088 -0.06 ]
 [-0.18   0.096  0.324 -0.244]
 [-0.597 -0.688 -0.093  0.096]]
v cyc+cubic [0.24993 0.51689 0.63993 1.0142  1.12432 1.34567]
v deadzone [0.52428 0.86929 0.98138 1.07083 1.14741 1.67261]
v tanh [0.49987 0.9045  1.03659 1.12879 1.26988 1.46074]
panel_c u range -1.2244050340401715 1.244976225249716
v1 aff [0.19688 1.01878 1.22071 1.53179 1.88978 1.99891]
coef [[-0.076  0.675 -0.012  0.028]
 [-0.263 -0.053  0.063  0.299]
 [ 0.018 -0.286  0.197 -0.071]
 [-0.283 -0.012 -0.06  -0.183]
 [ 0.05  -0.307  0.351  0.38 ]]
v1 cyc+cubic [3.0e-05 9.0e-05 1.3e-04 1.5e-04 1.8e-04 2.2e-04]
v1 deadzone [0.43504 0.97408 1.24865 1.57747 1.75685 2.04915]
v1 tanh [0.1647  1.03969 1.19686 1.51737 1.90024 2.08981]
v2 aff [0.06893 0.13862 0.17103 0.20014 0.22202 0.25724]
coef [[-0.067  0.007 -0.07  -0.044]
 [-0.474  0.178  0.161  0.045]
 [-0.151 -0.338 -0.274 -0.302]
 [ 0.071  0.264 -0.005 -0.35 ]
 [-0.038  0.164 -0.402  0.188]]
v2 cyc+cubic [0.05919 0.10507 0.12254 0.14649 0.17951 0.20912]
v2 deadzone [0.08542 0.20501 0.2648  0.33108 0.37493 0.40872]
v2 tanh [0.08653 0.1624  0.21458 0.23238 0.26296 0.30882]
traj sv [25.21328704  2.27274012  0.04773718  0.04343753  0.04004401  0.03807459
  0.03559885  0.03314832]
corr [ 1.         -0.10675044 -0.15842943 -0.20434216 -0.24136413 -0.27125884
 -0.29514003 -0.31311336 -0.32761563]
telemetry [1.25 2.15 0.22 2.15]
```

### Turn 5 — `execute_in_remote_docker` (`30147 ms` LLM)
- **Rationale**: Use robust residuals for the outlier-prone panel and leave-one-out residuals for the smaller panels to distinguish genuine structure from interpolation.
- **ATMS NOGOOD DECLARED**: Refuted `An affine calibration basis can describe panel C v1.` | Witness: `Median row residual falls from 1.22071 to 0.00013 with cyclic products and coordinate cubes.` | Blanket Diagnosis: `The reference panel directly isolates the nonlinear mapping from downstream response fitting.` | Sublated Pivot: `Retain cyclic products and cubes for panel C v1, and select the other panel bases independently.`

```bash
cat >/tmp/probe21.py <<'PY'
import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(*x):return np.column_stack((np.ones(len(x[0])),*x))
def pair(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)])
def cyc(x):return x*np.roll(x,-1,axis=1)
def reg(f,v):return np.linalg.lstsq(f,v,rcond=1e-10)[0]
def cand(u):
 yield 'aff',D(u);yield 'cyc3',D(u,cyc(u),u**3);yield 'Q',D(u,u*u,pair(u));yield 'Q3',D(u,u*u,pair(u),u**3)
 for a in [.22,.5,1,1.25,1.4,2.15,3,3.5,4]:
  for name,h in [('tanh',np.tanh(a*u)),('asinh',np.arcsinh(a*u)),('sin',np.sin(a*u)),('atan',np.arctan(a*u)),('erf',erf(a*u)),('soft',u/np.sqrt(1+(a*u)**2)),('rat',u/(1+a*u*u)),('dead',np.sign(u)*np.maximum(abs(u)-a,0)**1.5)]:
   yield (name,a,'h'),D(h);yield (name,a,'hc'),D(h,cyc(h));yield (name,a,'hq'),D(h,h*h,pair(h));yield (name,a,'linhc'),D(u,h,cyc(u));yield (name,a,'hcu'),D(h,cyc(u))
  yield ('sin_cos',a),D(np.sin(a*u),np.cos(a*u))
def panelfit():
 for k,l in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=s[k];u,v=p['u'],p[l];res=[]
  for name,f in cand(u):
   if k=='panel_a':
    keep=np.ones(len(u),bool)
    for _ in range(9):
     w=reg(f[keep],v[keep]);e=np.linalg.norm(f@w-v,axis=1);good=e<max(.001,2.8*np.median(e));
     if good.sum()<f.shape[1]+2 or np.array_equal(good,keep):break
     keep=good
   else:
    w=reg(f,v);hh=(np.linalg.svd(f,full_matrices=False)[0]**2).sum(1);e=np.linalg.norm((f@w-v)/np.maximum(1-hh[:,None],.000001),axis=1)
   res.append((np.median(e),np.quantile(e,.75),name))
  print(k,l,sorted(res,key=lambda x:x[0])[:17]);print('sample',np.round(D(u,v)[:6,1:],4))
if __name__=='__main__':panelfit()
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe21.py; sed -n '136,164p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`758 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
hq')), (0.022098280426169417, 0.03220316951403966, ('sin', 0.5, 'hq')), (0.026349729074176063, 0.04313839668228826, 'Q3'), (0.03830656024172256, 0.05367552827161361, ('atan', 0.5, 'hq')), (0.04083765173400257, 0.05661738761724136, ('tanh', 0.5, 'hq')), (0.04234475085059049, 0.058221286217016235, ('erf', 0.5, 'hq')), (0.054856933983289086, 0.07558552699291089, ('soft', 0.5, 'hq')), (0.05591791163901699, 0.08014281041380347, ('asinh', 1, 'hq'))]
sample [[ 0.4147  0.7328 -1.2244  1.0459 -0.4161 -0.2199 -0.7378  0.3019]
 [-1.0231 -0.7741 -0.9775  0.2636  0.3331 -0.1811 -0.0103  0.4917]
 [ 0.6019  1.0529 -1.0235 -0.2977 -0.554  -0.6494 -0.0906 -0.1712]
 [ 0.864  -0.2182  0.8143  0.2899 -0.3322  0.541   0.0873 -0.1566]
 [-1.1804 -1.2226  1.1198  0.283   0.5763  0.5461 -0.0334 -0.2543]
 [ 0.533   1.2332  0.9    -1.0685 -0.3664 -0.2336 -0.0797 -0.9582]]
errs = []
for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
    if k not in out_full:
        errs.append(f"missing:{k}"); continue
    a = np.asarray(out_full[k], dtype=float)
    if a.shape != shp or not np.all(np.isfinite(a)):
        errs.append(f"bad:{k}:{a.shape}")
if not errs:
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")

cv_metrics = {}
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
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
```

### Turn 6 — `execute_in_remote_docker` (`27709 ms` LLM)
- **Rationale**: Fit nonlinear frequencies and inspect coefficient sparsity on the reference panels, while recovering the exact frozen kernel contract needed for predictive validation.
- **Alternatives Weighed**: `Use the approximate panel B fit unchanged` (rejected: Its median leave-one-out residual of 0.0363 suggests missing structure at the scale of the downstream error target.)

```bash
cat /holdout_task/_frozen_nodes/node_0{1,2}.py; ls -1 /tmp/hegel_continual_archive; cat >/tmp/fit21.py <<'PY'
from probe21 import *
from scipy.optimize import minimize_scalar
A=s['panel_a'];B=s['panel_b'];C=s['panel_c'];
def rms(e):return np.sqrt(np.mean(e*e))
def robust(f,v):
 k=np.ones(len(f),bool)
 for _ in range(16):
  w=reg(f[k],v[k]); e=np.linalg.norm(f@w-v,axis=1); nk=e<max(.001,3*np.median(e));
  if nk.sum()<f.shape[1]+2 or np.array_equal(k,nk):break
  k=nk
 return w,k
forms={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'erf':erf,'sinh':np.sinh}
for name,h in forms.items():
 def er(a):
  u,v=A['u'],A['v']; f=D(u,h(a*u),cyc(u));w,k=robust(f,v);return rms((f@w-v)[k])
 o=minimize_scalar(er,bounds=(.15,3.2),method='bounded');print('A',name,o.x,o.fun)
 if o.fun<.004:
  f=D(A['u'],h(o.x*A['u']),cyc(A['u']));print(np.round(robust(f,A['v'])[0],4))
def bfeat(u,a,mode):
 h=np.sin(a*u); c=np.cos(a*u)
 if mode=='quad':return D(h,h*h,pair(h))
 if mode=='cosquad':return D(h,c,pair(h))
 if mode=='cyc':return D(h,c,cyc(h))
 if mode=='cyclin':return D(h,c,cyc(u))
 if mode=='quadlin':return D(h,c,pair(u))
 if mode=='phase':return D(h,c,cyc(h),cyc(c))
 if mode=='sumdiff':return D(h,c,np.cos(a*(u+np.roll(u,-1,axis=1))),np.sin(a*(u+np.roll(u,-1,axis=1))))
for mode in ('quad','cosquad','cyc','cyclin','quadlin','phase','sumdiff'):
 def er(a):
  f=bfeat(B['u'],a,mode); return rms(f@reg(f,B['v'])-B['v'])
 o=minimize_scalar(er,bounds=(.4,2.5),method='bounded');print('B',mode,o.x,o.fun,er(1.25),er(2.15));
 if o.fun<.025:print(np.round(reg(bfeat(B['u'],o.x,mode),B['v']),4))
for label,g,v in [('C1',lambda u:D(u,cyc(u),u**3),C['v1']),('C2',lambda u:D(np.sin(.22*u),np.cos(.22*u)),C['v2'])]:
 f=g(C['u']); w=reg(f,v);print(label,rms(f@w-v),np.round(w,4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/fit21.py
```
- **Container Exit Code**: `0` (`481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-5.563e-01]
 [ 7.116e-01  2.758e-01 -4.735e-01 -7.900e-02]
 [-1.100e-03  1.900e-03  3.600e-03 -1.450e-02]
 [-1.120e-02  6.000e-03 -4.500e-03 -5.200e-03]
 [-6.900e-03  1.500e-02  1.140e-02  4.000e-03]
 [ 7.300e-03 -1.000e-03  1.100e-03 -8.900e-03]
 [ 1.100e-02 -6.300e-03  7.100e-03 -3.760e-02]
 [ 6.800e-03 -7.000e-04  6.400e-03 -2.030e-02]
 [-1.190e-02 -6.000e-03  1.210e-02  7.000e-04]
 [ 6.400e-03  2.000e-04  3.300e-03  1.330e-02]]
B sumdiff 2.4999956339833576 0.008246693234868234 0.0753266614797211 0.032668087801617104
[[-9.560e-02 -1.051e-01  1.865e-01 -1.487e-01]
 [ 1.090e-02 -9.600e-03 -7.500e-03 -1.150e-02]
 [-0.000e+00  1.110e-02 -8.500e-03 -1.130e-02]
 [-8.900e-03 -8.300e-03 -1.690e-02 -6.100e-03]
 [ 4.200e-03  1.670e-02 -9.400e-03  4.000e-03]
 [-2.417e-01  5.562e-01 -1.535e-01  6.443e-01]
 [-1.978e-01 -5.429e-01 -6.555e-01  2.473e-01]
 [-4.231e-01  3.766e-01 -3.973e-01 -5.606e-01]
 [ 7.174e-01  2.813e-01 -4.810e-01 -7.430e-02]
 [ 9.800e-03 -3.300e-03  1.200e-03 -5.800e-03]
 [ 5.000e-03 -1.370e-02  8.300e-03  8.500e-03]
 [-9.900e-03 -4.800e-03  1.010e-02 -6.100e-03]
 [-3.000e-04 -1.300e-03  6.300e-03  1.570e-02]
 [-3.000e-04  6.700e-03  1.900e-03  6.900e-03]
 [-1.690e-02  1.290e-02  9.200e-03 -8.900e-03]
 [-3.500e-03  1.900e-03 -6.100e-03 -6.100e-03]
 [-2.280e-02 -1.330e-02  9.400e-03  6.000e-04]]
C1 6.633967542681748e-05 [[ 3.2400e-02  1.1600e-01  6.2200e-02  7.1200e-02]
 [ 3.0000e-04  1.0000e-04  1.0000e-04 -0.0000e+00]
 [-1.0000e-04  1.0000e-04  0.0000e+00 -0.0000e+00]
 [-3.0000e-04  0.0000e+00 -2.0000e-04  0.0000e+00]
 [ 0.0000e+00  0.0000e+00 -1.0000e-04 -1.0000e-04]
 [-8.3550e-01  7.3300e-01 -1.0290e-01  4.8750e-01]
 [-8.8000e-01 -6.1220e-01  5.8000e-02 -5.7530e-01]
 [ 5.7400e-02  1.3250e-01 -1.1589e+00 -3.4560e-01]
 [-8.8700e-02 -7.4430e-01 -3.5550e-01  8.9180e-01]
 [-2.0000e-04 -0.0000e+00 -0.0000e+00  1.0000e-04]
 [ 2.0000e-04 -0.0000e+00 -1.0000e-04  0.0000e+00]
 [ 2.0000e-04 -0.0000e+00  1.0000e-04 -1.0000e-04]
 [ 0.0000e+00  0.0000e+00  2.0000e-04  2.0000e-04]]
C2 0.004402680703745106 [[-9.5011e+00  6.7746e+00 -9.4213e+00 -8.1768e+00]
 [-2.0201e+00  6.8030e-01  5.2110e-01  3.0700e-02]
 [-7.3570e-01 -1.2612e+00 -1.0278e+00 -1.2969e+00]
 [ 4.4760e-01  1.4690e+00 -4.7000e-02 -1.5973e+00]
 [-1.9120e-01  8.3070e-01 -1.8988e+00  7.7500e-01]
 [ 8.0830e+00 -2.8221e+00 -2.2571e+00 -6.5200e-02]
 [ 2.7186e+00  5.1886e+00  4.1428e+00  5.3244e+00]
 [-1.8468e+00 -5.8880e+00  4.2000e-03  6.1765e+00]
 [ 6.1730e-01 -3.3658e+00  7.5765e+00 -3.1489e+00]]
```

### Turn 7 — `execute_in_remote_docker` (`70147 ms` LLM)
- **Rationale**: Exploit coefficient sparsity to invert panel A, then quantify which panel-derived coordinates actually improve both outputs through the frozen kernel on disjoint validation slices.

```bash
cat >/tmp/lab21.py <<'PY'
from probe21 import *
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
A=s['panel_a'];B=s['panel_b'];C=s['panel_c'];tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y)
def rms(a):return np.sqrt(np.mean(a*a))
def fit_a():
 u,v=A['u'],A['v']; good=np.linalg.norm(v,axis=1)<4; print('A excluded',np.where(~good)[0]);
 def fw(a,j,x):return D(x[:,j],np.sin(a*x[:,j]),np.sin(a*x[:,(j+2)%4]))
 def err(a):return np.column_stack([f@reg(f[good],v[good,j])-v[:,j] for j in range(4) for f in [fw(a,j,u)]])[good].ravel()
 o=minimize_scalar(lambda a:rms(err(a)),bounds=(1.5,3.3),method='bounded'); print('A fit',o.x,o.fun,rms(err(2.4))); a=o.x; w=np.column_stack([reg(fw(a,j,u)[good],v[good,j]) for j in range(4)]); print(w.round(4)); return a,w

def invert(x,a,w):
 z=(x-w[0])/w[1];
 for _ in range(25):
  pred=(z*w[1]+np.sin(a*z)*w[2]+np.sin(a*np.roll(z,2,axis=1))*w[3]+w[0]); J=np.zeros((len(z),4,4)); j=np.arange(4); J[:,j,j]=w[1]+a*np.cos(a*z)*w[2]; J[:,j,(j+2)%4]=a*np.cos(a*np.roll(z,2,axis=1))*w[3]; z-=np.clip(np.linalg.solve(J,(pred-x)[...,None])[...,0],-.35,.35)
 return z

def fit_b():
 u,v=B['u'],B['v']; score=[]
 funcs=[('cos',lambda x,a:np.cos(a*x)),('sincos',lambda x,a:np.c_[np.cos(a*x),np.sin(a*x)]),('coscyc',lambda x,a:np.c_[np.cos(a*x),cyc(x)]),('cosu',lambda x,a:np.c_[np.cos(a*x),x]),('phase',lambda x,a:np.c_[np.cos(a*x),np.sin(a*x),cyc(np.cos(a*x)),cyc(np.sin(a*x))])];
 for name,f in funcs:
  def e(a):
   h=D(f(u,a));return rms(h@reg(h,v)-v)
  o=minimize_scalar(e,bounds=(1.8,3.4),method='bounded'); score.append((o.fun,name,f,o.x));print('B',name,o.x,o.fun, e(2.55));
 _,name,f,a=min(score,key=lambda p:p[0]); return f,a,reg(D(f(u,a)),v)

def fit_c():
 u,v=C['u'],C['v2']; funcs=[('trig',lambda x,a:np.c_[np.sin(a*x),np.cos(a*x)]),('quad',lambda x,a:np.c_[x,x*x,a*x**3]),('rational',lambda x,a:np.c_[x/(1+a*x*x),x*x/(1+a*x*x)]),('tanh',lambda x,a:np.c_[np.tanh(a*x),np.tanh(a*x)**2]),('squash',lambda x,a:np.c_[x/(1+a*x*x)**.5,x*x/(1+a*x*x)**.5]),('exp',lambda x,a:np.c_[x,np.exp(a*x)]),('ratmix',lambda x,a:np.c_[x/(1+a*x*x),x*x])];L=[]
 for name,f in funcs:
  def e(a):
   h=D(f(u,a)); q=np.linalg.qr(h)[0];return rms((h@reg(h,v)-v)/(1-(q*q).sum(1))[:,None]);
  o=minimize_scalar(e,bounds=(.001,3.4),method='bounded');print('C2',name,o.x,o.fun);L.append((o.fun,f,o.x));
 _,f,a=min(L,key=lambda p:p[0]);return f,a,reg(D(f(u,a)),v)

def get():
 a,w=fit_a();x=np.r_[tr['x'],s['test']['x']];z=invert(x,a,w); print('range Z',z.min(),z.max());f,a,w=fit_b(); b=D(f(z,a))@w; f,a,w=fit_c(); c=D(f(z,a))@w; p=cyc(z);v=D(p)@reg(D(cyc(C['u'])),C['v1']);return x,z,b,c,p,v

def polys(x,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(x[:,ix],1) for j in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])

def scores(K,regu):
 vals=[]
 for i,j in [(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(52),j),j) for j in np.array_split(np.random.default_rng(2).permutation(52),4)]:
  p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+regu*np.eye(len(i)),Y[i]);err=np.abs(p-Y[j]);vals.append([err[:,0].mean(),err[:,1:].mean()]);
 return np.array(vals)
if __name__=='__main__':
 import sys
 sys.path.insert(0,'/holdout_task/_frozen_nodes');from node_01 import _kernel
 x,z,b,c,p,v=get();np.savez('/tmp/data21.npz',x=x,z=z,b=b,c=c,p=p,v=v,Y=Y)
 combos={'x':x,'z':z,'b':b,'c':c,'v':v,'zb':np.c_[z,b],'bc':np.c_[b,c],'zbcv':np.c_[z,b,c,v],'zbc':np.c_[z,b,c],'bv':np.c_[b,v],'cv':np.c_[c,v],'zcv':np.c_[z,c,v],'cosz':np.cos(2.55*z),'cosz+p+c':np.c_[np.cos(2.55*z),p,c]}; best=[]
 for name,f in combos.items():
  f=(f-f[:52].mean(0))/np.maximum(f[:52].std(0),1e-5)
  for scale in [.35,.65,1,1.7,3]:
   G=_kernel(scale*f,scale*f);q=scores(G,.004);best.append((max(q[0]),q[0],q[1:].mean(0),name,scale,'frozen'))
  g=f@f.T/f.shape[1]; dd=cdist(f,f,'sqeuclidean')/f.shape[1];
  for deg in [1,2,3]:
   for l in [0,.1,.5,2]:
    G=(1+g)**deg+(np.exp(-l*dd) if l else 0)
    for ridge in [.0001,.005,.05]:
     q=scores(G,ridge);best.append((max(q[0]),q[0],q[1:].mean(0),name,deg,(l,ridge)))
 for z0 in sorted(best,key=lambda k:k[0])[:12]:print('joint',z0)
 for ix in [0,1]:
  for z0 in sorted(best,key=lambda k:k[1][ix])[:6]:print('head',ix,z0)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/lab21.py
for f in /tmp/hegel_continual_archive/*batch_0[15]*py; do grep -En 'sin|cos|mix|telemetry|panel|basis|design' "$f"; done
```
- **Container Exit Code**: `0` (`868 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
int (0.20541133859929336, array([0.20541134, 0.15430582]), array([0.29877298, 0.30251929]), 'zb', 3, (0.5, 0.05))
joint (0.2086448123664954, array([0.20864481, 0.15912613]), array([0.29808361, 0.30268164]), 'zb', 3, (0.1, 0.05))
joint (0.20963302124949265, array([0.20963302, 0.16028259]), array([0.29719833, 0.30342127]), 'zb', 3, (0, 0.05))
joint (0.21039859213434658, array([0.21039859, 0.15312434]), array([0.29449668, 0.26422568]), 'bc', 1, (0.1, 0.0001))
head 0 (0.19403502226644642, array([0.19403502, 0.12890789]), array([0.32967716, 0.30123991]), 'zb', 1, (0.5, 0.0001))
head 0 (0.1986406115762091, array([0.19864061, 0.14100107]), array([0.29734656, 0.29037384]), 'zb', 3, (0.5, 0.005))
head 0 (0.19898291411663843, array([0.19898291, 0.13859908]), array([0.29661757, 0.28826163]), 'zb', 3, (0.5, 0.0001))
head 0 (0.20121935487721437, array([0.20121935, 0.14247978]), array([0.29483242, 0.28641921]), 'zb', 3, (0.1, 0.0001))
head 0 (0.20129201503482333, array([0.20129202, 0.15722312]), array([0.32977615, 0.30797219]), 'zb', 1, (0.5, 0.005))
head 0 (0.2013461614191971, array([0.20134616, 0.14523679]), array([0.29536936, 0.28896816]), 'zb', 3, (0.1, 0.005))
head 1 (0.19403502226644642, array([0.19403502, 0.12890789]), array([0.32967716, 0.30123991]), 'zb', 1, (0.5, 0.0001))
head 1 (0.19898291411663843, array([0.19898291, 0.13859908]), array([0.29661757, 0.28826163]), 'zb', 3, (0.5, 0.0001))
head 1 (0.21351750682507026, array([0.21351751, 0.14089463]), array([0.28376641, 0.30091877]), 'bc', 3, (0.5, 0.005))
head 1 (0.1986406115762091, array([0.19864061, 0.14100107]), array([0.29734656, 0.29037384]), 'zb', 3, (0.5, 0.005))
head 1 (0.21385980184226433, array([0.2138598 , 0.14113642]), array([0.29308341, 0.26371115]), 'zb', 2, (0.5, 0.0001))
head 1 (0.21220761037712652, array([0.21220761, 0.14155086]), array([0.28337444, 0.29882803]), 'bc', 3, (0.5, 0.0001))
7:def _design(x):
16:    a = study['panel_a']
17:    A, v = _design(a['u']), a['v']
27:    p = study['panel_b']
30:        return _design(erf(q[0]*u) + q[1]*u)
35:    q = min(sols, key=lambda o:o.cost).x
7: p=s['panel_a']; F=D(p['u']); y=p['v']; keep=np.ones(len(F),bool)
13: p=s['panel_a']; u,v=p['u'],p['v']; F=AF(u); _,_,e=cal(s,x); w=fit(F[e<2],v[e<2]); z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
21:    tr=study['train']; z=coords(study,np.r_[tr['x'],study['test']['x']]); p=study['panel_b']; u=p['u'];v=p['v']
48:    designs=(np.arcsinh(sym),np.arcsinh(sep),sym,sep)
51:        k=kernel(designs[form],degree,decay)
```

### Turn 8 — `execute_in_remote_docker` (`39896 ms` LLM)
- **Rationale**: Panel B supports a cosine map at frequency 2.6 with residual RMS below 0.00009, but calibration alone has not delivered predictive accuracy. Check inversion directly and test sparse algebraic and harmonic bases for each response before constructing the editable node.
- **Alternatives Weighed**: `Tune the frozen kernel's input scale alone` (rejected: Calibrated feature groups still miss the public threshold by severalfold, so the next test targets missing response structure.)

```bash
cat /tmp/hegel_continual_archive/*batch_0[15]*py | head -150; cat > /tmp/struct21.py <<'PY'
from lab21 import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
x=np.r_[tr['x'],s['test']['x']]; a,w=fit_a(); z=invert(x,a,w)
print('inversion residual',np.max(abs(z*w[1]+np.sin(a*z)*w[2]+np.sin(a*np.roll(z,2,axis=1))*w[3]+w[0]-x),axis=0)); print('panel lengths',[len(p['u']) for p in [A,B,C]])
# Separate even and odd terms with two independently profiled frequencies.
u,v=C['u'],C['v2']; opts={}
for n1,h in [('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x)),('asinh',np.arcsinh)]:
 for n2,g in [('cos',np.cos),('tanh2',lambda x:np.tanh(x)**2),('rat2',lambda x:x*x/(1+x*x)),('linear2',lambda x:x*x),('tanh_sq',lambda x:np.tanh(x*x))]:
  def fit(u,q):return D(h(q[0]*u),g(q[1]*u))
  def err(q):
   f=fit(u,q);return (f@reg(f,v)-v).ravel()
  o=least_squares(err,[.4,.4],bounds=(.015,4)); opts[n1,n2]=(rms(o.fun),o.x)
print('C2 forms', sorted(opts.items(),key=lambda q:q[1][0])[:8]);
f=D(z,z*z); print('Y ranges',Y.min(0),Y.max(0)); print('X ranges',x.min(0),x.max(0)); print('Y first',np.round(np.c_[z,Y[:,[0,-1]]][:12],4));
T=Y[:,1:]; tm=s['eval_times']; print('times',tm); E=lambda p:-np.expm1(-tm[:,None]*p); o=least_squares(lambda p:(E(p)@reg(E(p),T.T)-T.T).ravel(),[2.15,.22],bounds=([1,.05],[6,1])); print('rates',o.x,rms(o.fun)); W=np.c_[Y[:,0],reg(E(o.x),T.T).T];
# Test families of sparse supervised bases, sharing functions but not coefficients.
cos=np.cos(2.6*z); co=D(np.cos(2.6*B['u'])); b=D(cos)@reg(co,B['v']); p=cyc(z); v1=D(p)@reg(D(cyc(C['u'])),C['v1']); v2=D(np.tanh(.21647*z),np.tanh(.21647*z)**2)@reg(D(np.tanh(.21647*u),np.tanh(.21647*u)**2),v);
np.savez('/tmp/struct21.npz',x=x,z=z,b=b,p=p,v1=v1,v2=v2,Y=Y,W=W);
def basis(h):
 return [('raw',h),('poly',polys(h,3)),('harmonics',np.c_[h,h*h,*[fun(a*h) for a in [.22,.5,1.,1.25,2.15,2.6,3.5,4.3,5.2,6.5,8.6,10.4] for fun in [np.sin,np.cos,np.tanh]]]),('poly2',polys(h,2))]
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet
res=[]
for name,h in [('z',z),('x',x),('b',b),('cos',cos),('v1',v1),('v2',v2),('zb',np.c_[z,b]),('mix+',z+b),('mix-',z-b),('all',np.c_[z,b,v1,v2]),('mixa',np.c_[z+b,z-b]),('h',np.c_[z,b,p,v1,v2,z+b,z-b,b+v1,b-v1])]:
 for nn,f in basis(h):
  if f.shape[1]>1500:continue
  f=f[:72];mu=f[:52].mean(0);sd=f[:52].std(0); f=(f-mu)/np.maximum(sd,1e-8)
  for k in [4,8,12,20,32]:
   if k>f.shape[1]:continue
   model=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(f[:52],W[:52]); e=np.abs(model.predict(f[52:])-W[52:]).mean(0); res.append((tuple(e),name,nn,k));
for j in range(3):print('sparse head',j,sorted(res,key=lambda p:p[0][j])[:8]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/struct21.py 2>/tmp/struct21.err; tail -6 /tmp/struct21.err
```
- **Container Exit Code**: `0` (`566 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
linalg.solve(k[:n,:n]+ridge*np.eye(n),target)
    def blend(a,b,w,target):
        pred=head(a,target)
        return pred if w==1 else w*pred+(1-w)*head(b,target)
    y=blend(*((0, 2, 1.4, 1e-06), (2, 3, 0.15, 1e-06), 0.5),tr['y'])
    state=blend(*((1, 3, 0.15, 1e-06), (1, 3, 1.4, 1e-06), 0.25),modes)@f.T
    return y,state

def solve_batch(study:dict)->dict:
    y,state=predict(study); end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
A excluded [ 5 17 31]
A fit 2.399332363623964 0.003880879268389074 0.0038822149972917155
[[ 0.1255 -0.1135  0.0674  0.1428]
 [ 1.0488  1.171   0.9919  0.9599]
 [ 0.0667  0.0662  0.0662  0.0666]
 [-0.3204 -0.3201 -0.3199 -0.3203]]
inversion residual [4.44089210e-16 3.33066907e-16 2.22044605e-16 2.22044605e-16]
panel lengths [48, 24, 24]
C2 forms [(('asinh', 'rat2'), (0.00018323177209548277, array([0.04141579, 0.4036778 ]))), (('tanh', 'rat2'), (0.00018323288962513473, array([0.02938482, 0.40367673]))), (('sin', 'rat2'), (0.00018323298611583498, array([0.04150247, 0.40367711]))), (('rat', 'rat2'), (0.0001832428366189383, array([0.0173185 , 0.40367004]))), (('asinh', 'tanh2'), (0.00022344830357100945, array([0.04604488, 0.48235411]))), (('tanh', 'tanh2'), (0.00022344919279944952, array([0.03257939, 0.48235368]))), (('sin', 'tanh2'), (0.0002234496751750023, array([0.04605188, 0.48235378]))), (('rat', 'tanh2'), (0.0002234499644888927, array([0.01889374, 0.48235175])))]
Y ranges [-1.82729443 -0.6530542  -1.33479653 -1.77016446 -2.07385153 -2.29142704
 -2.47404574 -2.60773742 -2.72039819] [1.77249527 1.01020935 2.02625133 2.63641605 3.02823722 3.30155802
 3.50108677 3.65471639 3.77918121]
X ranges [-1.36312635 -1.75216307 -1.3545189  -1.1963621 ] [1.55173676 1.53307075 1.52690865 1.56152311]
    f=D(z,z*z); print('Y ranges',Y.min(0),Y.max(0)); print('X ranges',x.min(0),x.max(0)); print('Y first',np.round(np.c_[z,Y[:,[0,-1]]][:12],4));
                                                                                                                   ~~~~~^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/index_tricks.py", line 418, in __getitem__
    res = self.concatenate(tuple(objs), axis=axis)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 108 and the array at index 1 has size 72
```

### Turn 9 — `execute_in_remote_docker` (`29396 ms` LLM)
- **Rationale**: Save a contract-compatible calibrated node, then rerun the interrupted sparse-basis and relaxation diagnostics to identify the predictive bottleneck.

```bash
sed -i 's/np.c_\[z,Y\[:,\[0,-1\]\]\]/np.c_[z[:72],Y[:,[0,-1]]]/' /tmp/struct21.py
cat > /holdout_task/node_00.py <<'PY'
"""Panel-calibrated coordinates for the frozen prediction nodes."""
import numpy as np
from scipy.optimize import minimize_scalar


def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _coordinates(s, x):
    p = s['panel_a']
    u, v = p['u'], p['v']
    w = []
    for j in range(u.shape[1]):
        f = _design(u[:, j], np.sin(2.4*u[:, j]), np.sin(2.4*u[:, (j+2)%4]))
        wc = _fit(f,v[:,j]); keep = abs(f@wc-v[:,j]) < 1.5
        w.append(_fit(f[keep],v[keep,j]))
    b,d,h,c = np.array(w).T
    z = (x-b)/d
    for _ in range(35):
        zp = np.roll(z,2,axis=1)
        r = b+d*z+h*np.sin(2.4*z)+c*np.sin(2.4*zp)-x
        a = d+2.4*h*np.cos(2.4*z)
        q = 2.4*c*np.cos(2.4*zp)
        ar,qr,rr = [np.roll(t,2,axis=1) for t in (a,q,r)]
        z -= np.clip((ar*r-q*rr)/(a*ar-q*qr),-.3,.3)
    p = s['panel_b']; f = lambda x: _design(np.cos(2.6*x))
    b = f(z)@_fit(f(p['u']),p['v'])
    p = s['panel_c']; f = lambda x: _design(x*np.roll(x,-1,axis=1))
    c = f(z)@_fit(f(p['u']),p['v1'])
    def r(x,a): return _design(x,x*x/(1+a*x*x))
    u,v = p['u'],p['v2']
    a = minimize_scalar(lambda t: np.linalg.norm(r(u,t)@_fit(r(u,t),v)-v),bounds=(.001,1),method='bounded').x
    d = r(z,a)@_fit(r(u,a),v)
    return z,b,c,d


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']; n = len(s['train']['x'])
    z,b,c,d = _coordinates(s,np.vstack((s['train']['x'],s['test']['x'])))
    f = np.c_[z,b,c,d]
    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-6)
    return {'train_features':f[:n].copy(),'test_features':f[n:].copy()}
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/struct21.py 2>/tmp/struct21.err | tail -18; tail -8 /tmp/struct21.err
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py | tail -15
```
- **Container Exit Code**: `0` (`1309 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
zb', 'poly', 12), ((0.3900551487708427, 0.1641033912060277, 0.01834348789487973), 'zb', 'poly', 8), ((0.7553896581485231, 0.16655424012515813, 1.708130793196218), 'b', 'poly2', 4), ((0.23407853279255791, 0.17438147511458563, 0.04510937220776172), 'zb', 'poly', 20), ((0.7926440941448962, 0.17556096479220493, 1.7222012889013016), 'b', 'poly2', 8), ((0.6803437874026688, 0.17937557046072125, 0.13127152038429796), 'h', 'poly2', 4), ((0.22769954803130035, 0.18020196045419076, 0.0630000336875634), 'zb', 'poly', 32)]
sparse head 2 [((0.32557091465923005, 0.16036306585653304, 0.0164258738841723), 'zb', 'poly', 12), ((0.3900551487708427, 0.1641033912060277, 0.01834348789487973), 'zb', 'poly', 8), ((0.23407853279255791, 0.17438147511458563, 0.04510937220776172), 'zb', 'poly', 20), ((0.22769954803130035, 0.18020196045419076, 0.0630000336875634), 'zb', 'poly', 32), ((0.4012033630280376, 0.5410372601617243, 0.10371603164601048), 'z', 'poly', 8), ((0.2485322642009829, 0.19081725416410394, 0.10427589474312246), 'zb', 'poly2', 12), ((0.26003615219434895, 0.20687249032139063, 0.10483353753935024), 'zb', 'poly2', 8), ((0.4358436870515162, 0.24328077166840628, 0.1151881666211877), 'mixa', 'poly2', 12)]
/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py:186: RuntimeWarning: Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary. The requested precision might not have been met.
  return func(*args, **kwargs)
/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py:186: RuntimeWarning: Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary. The requested precision might not have been met.
  return func(*args, **kwargs)
/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py:186: RuntimeWarning: Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary. The requested precision might not have been met.
  return func(*args, **kwargs)
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.313690", "public_cv_traj_mae_exceeded:0.212792"], "cv_metrics": {"public_cv_y_mae": 0.31369, "public_cv_traj_mae": 0.212792}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f37edebdd3d8b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f37ed (directory).
```

### Turn 10 — `execute_in_remote_docker` (`47364 ms` LLM)
- **Rationale**: The calibrated coordinates alone miss both thresholds. Search sparse pairwise, harmonic, rational, and nonlinear-index features on the training prefix to identify compact predictors before testing them through the frozen nodes.

```bash
cat /tmp/hegel_continual_archive/*batch_11_solver.py; cat >/tmp/search21.py <<'PY'
import sys
sys.path.insert(0,'/holdout_task')
from node_00 import _coordinates
from probe21 import s,np,D,reg,least_squares,combinations
from itertools import combinations_with_replacement
from sklearn.linear_model import Lars, Ridge
from scipy.special import erf
from scipy.optimize import minimize_scalar
tr=s['train']; n=len(tr['x']); X=tr['x']; z,b,c,d=_coordinates(s,X); h=np.cos(2.6*z); t=s['eval_times']; E=lambda p:-np.expm1(-t[:,None]*np.array(p)); T=tr['traj']; W=np.c_[tr['y'],reg(E([1.8,.42]),T.T).T]; ix=np.arange(52); jx=np.arange(52,n)
base={'z':z,'x':X,'b':b,'c':c,'d':d,'h':h,'p':z*np.roll(z,-1,axis=1)}
def lib(base):
 F={}
 def add(k,x):
  if x.ndim==1:F[k]=x
  else:
   for j,v in enumerate(x.T):F[f'{k}{j}']=v
 for k,x in base.items():
  add(k,x)
  for a in [.5,1.25,2.15,2.6,4.3,5.2]:
   for fun in [np.sin,np.cos,np.tanh]:add(f'{fun.__name__}{a}({k})',fun(a*x))
 for k,l in combinations_with_replacement(base,2):
  for j,i in ((j,i) for j in range(4) for i in range(j if k==l else 0,4)):
   f=base[k][:,j];g=base[l][:,i];add(f'{k}{j}*{l}{i}',f*g)
   if k in ('z','b','h') and l in ('z','b','h') and (k,j)!=(l,i):
    for r in [-2,-1,-.5,.5,1,2]:
     for a in [1.25,2.15,2.6]:
      for fun in [np.sin,np.cos,np.tanh]:add(f'{fun.__name__}{a}({k}{j}+{r}{l}{i})',fun(a*(f+r*g)))
 for k,x in base.items():
  for ix in combinations_with_replacement(range(4),3):add(f'{k}^{ix}',x[:,ix].prod(1))
 for k in ('b','c','d','z'):
  for l in ('b','c','d','z'):
   for a in [.22,.5,1.25,2.15]:
    for i in range(4):
     for j in range(4):add(f'{k}{i}/(1+{a}{l}{j}^2)',base[k][:,i]/(1+a*base[l][:,j]**2))
 return F
F=lib(base); keys=np.array(list(F)); G=np.column_stack(list(F.values())); G=(G-G[ix].mean(0))/np.maximum(G[ix].std(0),1e-8)
for H in range(3):
 y=W[:,H]; y0=y-y[ix].mean(); corr=G[ix].T@y0[ix]; top=np.argsort(-abs(corr))[:8]; print('TOP',H,list(zip(keys[top],np.round(corr[top],2))))
 paths=[]
 for gs in [np.arange(G.shape[1]),np.where(np.array([not any(w in k for w in ('x','d','c')) for k in keys]))[0]]:
  path=Lars(n_nonzero_coefs=32,fit_intercept=False).fit(G[ix][:,gs],y0[ix]);
  for j,coef in enumerate(path.coef_path_.T):
   sel=gs[np.flatnonzero(coef)]; f=D(G[:,sel]); w=reg(f[ix],y[ix]); p=f@w; err=(abs(p[ix]-y[ix]).mean(),abs(p[jx]-y[jx]).mean());paths.append((err,keys[sel].tolist(),w))
 for q in sorted(paths,key=lambda q:q[0][1])[:3]:print('LARS',H,q[0],q[1],np.round(q[2],3))
 for k,g in base.items():
  f=D(g);w=reg(f[ix],y[ix]);print('LIN',H,k,np.round(w,3),np.round([abs((f@w-y)[j]).mean() for j in [ix,jx]],4))
# Profile small multi-index models with calibrated coordinates and nonlinear readout.
Z=np.c_[z,b,c,d]; Z=(Z-Z[ix].mean(0))/np.maximum(Z[ix].std(0),1e-10); rng=np.random.default_rng(441)
for H in range(3):
 y=W[:,H]; opts=[]
 for kk in [4,8,12,16]:
  for fn in [np.sin,np.tanh,np.cos,lambda t:t*t,lambda t:t/(1+t*t)]:
   for typ in [0,1,2]:
    P=Z[:,:kk]; f0=D(P[:,:min(kk,8)] if typ else P[:,:0]); m=kk+1
    def f(q):return np.c_[f0,fn(P@q[:kk]+q[kk]), fn(P@q[m:m+kk]+q[m+kk])] if typ==2 else np.c_[f0,fn(P@q[:kk]+q[kk])]
    def e(q):
     g=f(q)[ix]; return g@reg(g,y[ix])-y[ix]
    for a in range(3):
     sol=least_squares(e,rng.normal(0,.5,m*(2 if typ==2 else 1)),max_nfev=100); g=f(sol.x); w=reg(g[ix],y[ix]); rr=abs(g@w-y); opts.append((rr[ix].mean(),rr[jx].mean(),kk,getattr(fn,'__name__','fn'),typ,sol.x,w))
 for row in sorted(opts,key=lambda q:q[1])[:3]:print('INDEX',H,row[:5],np.round(row[5],3),np.round(row[6],3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/search21.py >/tmp/search21.out 2>/tmp/search21.err; grep -E 'TOP|LARS|LIN|INDEX' /tmp/search21.out; tail -5 /tmp/search21.err
```
- **Container Exit Code**: `0` (`29745 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.058 -0.142
LIN 2 z [ 0.007  0.782 -0.73  -1.219 -1.458] [0.0834 0.1337]
LIN 2 x [ 0.176  0.377 -0.966 -1.065 -1.686] [0.2204 0.2203]
LIN 2 b [ 0.063  0.421 -0.065  0.332 -0.811] [1.0821 1.4436]
LIN 2 c [ 0.335 -0.147  0.108  0.367 -0.85 ] [1.0675 1.1888]
LIN 2 d [ 0.1   -0.98  -1.08   3.395  1.214] [0.3402 0.3275]
LIN 2 h [ 0.179 -0.701 -0.445  0.115  0.181] [1.0821 1.4436]
LIN 2 p [ 0.305 -0.25   0.573 -0.125 -0.956] [1.0675 1.1888]
INDEX 0 (0.21509933535555564, 0.23805627003889565, 8, '<lambda>', 1) [  0.083   0.728  -0.122   0.239   1.249  -2.358   2.342   2.526 -24.884] [-12.046  -0.185   0.343   0.194   0.702   1.194  -2.153   1.942   2.336
INDEX 0 (0.21808591774904787, 0.24522090608841424, 8, 'cos', 1) [ 0.148  0.44  -0.144 -0.027  0.062 -0.922  0.867  0.817 -0.883] [-0.006 -0.295 -0.283  0.282  0.486  0.052 -0.123 -0.083  0.131 -0.509]
INDEX 0 (0.1292849944263899, 0.26179386920668973, 8, 'tanh', 2) [-0.075 -0.005 -0.224 -1.307 -0.576  1.143 -0.024  0.361  0.54   0.007
INDEX 1 (0.1182293595966751, 0.23052212751035167, 16, '<lambda>', 1) [-1.3062e+01 -1.1800e+00  1.1309e+01 -7.9880e+00  9.0660e+00 -6.6990e+00
INDEX 1 (0.1479780073460868, 0.3176729192863237, 16, 'tanh', 1) [-2.950e-01 -3.120e-01  1.379e+00 -6.210e-01 -7.400e-02  1.980e-01
INDEX 1 (0.31327675818585426, 0.3220624733923635, 12, 'cos', 0) [ 0.101 -0.004  0.209  0.207  0.609 -0.741 -0.604 -0.508 -0.039  0.023
INDEX 2 (0.04942573328861191, 0.10938693732139444, 8, '<lambda>', 1) [ 0.019 -0.292 -0.207 -0.103  0.412 -0.376  0.382  0.173 -0.103] [ 0.163  0.544 -0.549 -0.792 -1.046  0.08  -0.089  0.102  0.056 -0.506]
INDEX 2 (0.04859101884982368, 0.11940680624870303, 12, 'cos', 1) [-0.828  0.439 -0.473 -0.121 -0.82  -0.15  -0.176 -0.368 -0.028 -0.627
INDEX 2 (0.03989595868837628, 0.12037772202121175, 12, '<lambda>', 1) [ 0.95  -3.423  0.45   0.326 -2.981 -1.079 -1.795  0.92   1.467  0.015
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_least_angle.py:725: ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 18 iterations, i.e. alpha=2.292e-01, with an active set of 14 regressors, and the smallest cholesky pivot element being 2.220e-16. Reduce max_iter or increase eps parameters.
  warnings.warn(

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f37edebdd3d8b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f37ed (directory).
```

### Turn 11 — `execute_in_remote_docker` (`86369 ms` LLM)
- **Rationale**: Fit feature-group weights separately using only the supplied training data, then verify the unchanged DAG on the official split and four disjoint outer folds.
- **Alternatives Weighed**: `Use a single sparse model for all outputs` (rejected: The sparse probes predicted the slow trajectory mode substantially better than the fast mode or scalar, indicating different feature requirements.)

```bash
set -e
cp /holdout_task/node_00.py /tmp/node21_baseline.py
python3 - <<'PY'
p='/holdout_task/node_00.py'
s=open(p).read().split('def run_node')[0].replace('wc = _fit(f,v[:,j]); keep = abs(f@wc-v[:,j]) < 1.5','wc = _fit(f,v[:,j]); keep = np.linalg.norm(v,axis=1) < 4\n        if keep.sum() < f.shape[1] + 2:\n            keep = abs(f@wc-v[:,j]) < 1.5')
open(p,'w').write(s)
PY
cat >> /holdout_task/node_00.py <<'PY'
from itertools import combinations, combinations_with_replacement
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _library(s, x):
    z,b,c,d = _coordinates(s,x)
    h = np.cos(2.6*z); p = z*np.roll(z,-1,axis=1)
    groups = []
    def add(*f):
        g = np.column_stack(f)
        groups.append(g)
    def pairs(u,v):
        return (u[:,:,None]*v[:,None,:]).reshape(len(u),-1)
    def poly(u,j):
        return np.column_stack([u[:,k].prod(1) for k in combinations_with_replacement(range(u.shape[1]),j)])
    def norm(u):
        return np.maximum(u.std(0),1e-6)
    add(z); add(z**2); add(h); add(b,np.tanh(1.25*b),np.sin(2.15*b)); add(p); add(d); add(x)
    add(z**3,z**5,np.tanh(1.25*z),np.sin(2.15*z),np.sin(2.6*z))
    add(poly(z,2)); add(poly(z,3)); add(poly(z,4)); add(poly(h,2)); add(poly(b,2)); add(poly(c,2)); add(poly(d,2))
    add(pairs(z,h)); add(pairs(z,b)); add(pairs(z,c)); add(pairs(z,d)); add(pairs(b,c)); add(pairs(b,d)); add(pairs(c,d))
    add(pairs(z,h*h),pairs(z*z,h)); add(pairs(z,np.sin(2.6*z))); add(pairs(np.sin(2.6*z),h)); add(pairs(x,b))
    for v in (z,b,c,d):
        add(v/(1+.22*v*v),v/(1+1.25*v*v),np.arcsinh(v),np.tanh(2.15*v),np.sin(1.25*v),np.cos(2.15*v))
    for u,v in ((z,z),(z,b),(b,b),(b,c),(b,d)):
        if u is v:
            ij=list(combinations(range(4),2)); aa=u[:,[i for i,j in ij]]; bb=u[:,[j for i,j in ij]]
        else:
            aa=np.repeat(u,4,axis=1); bb=np.tile(v,(1,4))
        add(*[fun(a*(aa+sg*bb)) for a in (1.25,2.15,2.6) for sg in (-1,1) for fun in (np.sin,np.cos,np.tanh)])
    for u,v in ((z,b),(b,z),(z,c),(b,c),(z,d),(b,d),(c,d)):
        a = np.repeat(u,4,axis=1); bb=np.tile(v,(1,4))
        add(a/(1+.22*bb*bb),a/(1+1.25*bb*bb),a/(1+2.15*bb*bb))
    for j in range(4):
        add(z[:,j]); add(h[:,j]); add(p[:,j]); add(z[:,j]**3,np.sin(2.6*z[:,j]),np.tanh(2.15*z[:,j])); add(z[:,j,None]*h)
    return groups, (z,h,b,c,d,x)


def _kernels(groups, coords, n):
    a=[np.ones((len(coords[0]),len(coords[0])))]
    for f in groups:
        f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7)
        a.append(f@f.T/f.shape[1])
    for u in (coords[0],coords[1],coords[2],np.c_[coords[0],coords[1]],np.c_[coords[0],*coords[2:5]],coords[-1]):
        u=(u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-7)
        r2=cdist(u,u,'sqeuclidean')/u.shape[1]
        for scale in (.3,1.,2.5):
            r=np.sqrt(scale*r2); a.append((1+r+r*r/3)*np.exp(-r))
    return np.asarray(a)


def _head(ks,n,y):
    mean=y.mean(); sd=max(y.std(),1e-8); y=(y-mean)/sd
    g=np.ascontiguousarray(ks[:,:n,:n]); ff=g.reshape(len(g),-1); eye=np.eye(n)
    base=np.full(len(g)+1,.015); base[0]=.4; base[-1]=.025
    opts=[]
    for penalty in (.035,.25,1.2):
        logbase=np.log(base)
        def score(q):
            w=np.exp(q); k=(w[:-1]@ff).reshape(n,n)+w[-1]*eye
            chol=cho_factor(k,check_finite=False); alpha=cho_solve(chol,y,check_finite=False)
            ki=cho_solve(chol,eye,check_finite=False); dif=q-logbase
            v=.5*y@alpha+np.log(np.diag(chol[0])).sum()+.5*penalty*(dif@dif)
            d=.5*(ki-np.outer(alpha,alpha)); grad=w*np.r_[ff@d.ravel(),np.trace(d)]+penalty*dif
            return v,grad
        sol=minimize(score,logbase,method='L-BFGS-B',jac=True,bounds=[(-13,4)]*(len(base)-1)+[(-11,1)],options={'maxiter':140,'ftol':2e-8})
        w=np.exp(sol.x); k=(w[:-1]@ff).reshape(n,n)+w[-1]*eye
        chol=cho_factor(k,check_finite=False); alpha=cho_solve(chol,y,check_finite=False)
        h=cho_solve(chol,eye,check_finite=False).diagonal(); mse=np.mean((alpha/h)**2)
        pred=np.einsum('k,kij,j->i',w[:-1],ks[:,:,:n],alpha,optimize=True)
        opts.append((mse,pred))
    err=np.asarray([e for e,p in opts]); wt=(min(err)/np.maximum(err,1e-12))**4
    return mean+sd*np.average(np.array([p for e,p in opts]),axis=0,weights=wt)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; tr=s['train']; n=len(tr['x']); x=np.vstack((tr['x'],s['test']['x']))
    g,z=_library(s,x); k=_kernels(g,z,n)
    times=np.asarray(s['eval_times']); modes=-np.expm1(-times[:,None]*[1.8,.42])
    train=np.column_stack((tr['y'],_fit(modes,np.asarray(tr['traj']).T).T))
    f=np.column_stack([_head(k,n,y) for y in train.T]); f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-8)
    f*=.52
    return {'train_features':f[:n].copy(),'test_features':f[n:].copy()}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from test_dag_candidate import run_dag
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; all_ix=np.arange(len(t['x'])); results=[]
for j in np.array_split(np.random.default_rng(23).permutation(all_ix),4):
 i=np.setdiff1d(all_ix,j); y=t['y'][i]; e=t['traj'][i,-1]; b=dict(s,train=dict(x=t['x'][i],y=y,traj=t['traj'][i],summary=np.array([y.mean(),y.std(),e.mean(),e.std()])),test=dict(x=t['x'][j])); r,_=run_dag(b)
 results.append([np.mean(abs(r['y_hat']-t['y'][j])),np.mean(abs(r['state_pred']-t['traj'][j]))])
print('Random four-fold MAEs:',results,'worst:',np.max(results,axis=0).tolist())
PY
```
- **Container Exit Code**: `1` (`659 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.224479", "public_cv_traj_mae_exceeded:0.161672"], "cv_metrics": {"public_cv_y_mae": 0.224479, "public_cv_traj_mae": 0.161672}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e55a70c3464195a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9e55a70c (directory).
```

### Turn 12 — `execute_in_remote_docker` (`43202 ms` LLM)
- **Rationale**: Test whether explicit interaction and harmonic groups improve the actual frozen DAG on four disjoint folds, audit trajectory horizons separately, and retain an edit only if it improves the worst-fold bottleneck without materially worsening either output. Rerun the official validator on the saved artifact.
- **Alternatives Weighed**: `Submit the current artifact` (rejected: The official clean-process test rejected both error metrics.)

```bash
cd /holdout_task; cp node_00.py /tmp/node21_incumbent.py; cat >/tmp/check21.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import test_dag_candidate as test
import node_00 as node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; ids=np.arange(len(t['x'])); kernel=node._kernels

def augmented(groups,coords,n):
    z,h,b,c,d,x=coords
    extra=[]
    for u in (z,b,c,d):
        for j in range(u.shape[1]):
            a=u[:,j,None]; q=a/np.maximum(a[:n].std(),1e-6)
            for lo,hi in ((.5,2.5),(2.5,5.5),(5.5,10.5)):
                r=q*np.linspace(lo,hi,5); extra.append(np.c_[np.sin(r),np.cos(r)])
            for k in range(j):
                v=u[:,k,None]; v=v/np.maximum(v[:n].std(),1e-6)
                extra.append(np.c_[q*v,q*q*v,q*v*v,q*q*v*v])
                for w in (q+v,q-v):
                    extra.append(np.c_[np.sin(w),np.cos(w),np.sin(2*w),np.cos(2*w)])
    for u,v in ((z,b),(z,c),(z,d),(b,c),(b,d),(c,d)):
        for j in range(4):
            a=u[:,j,None]; q=v[:,j,None]; extra.append(np.c_[a*q,a*q*q,a*a*q,a/(1+.22*q*q),q/(1+.22*a*a)])
    # Preserve the panel C cubic terms in the profiled feature bank.
    p=s['panel_c']; u=p['u']; design=lambda a: node._design(a,a*np.roll(a,-1,axis=1),a**3)
    w=node._fit(design(u),p['v1']); cv=design(z)@w
    extra += [cv,np.tanh(cv),np.sin(1.25*cv),np.sin(2.15*cv),cv*z,cv*b,cv*d]
    return kernel(groups+extra,coords,n)

splits=np.array_split(np.random.default_rng(23).permutation(ids),4)
scores=[]
for mode in (kernel,augmented):
    node._kernels=mode
    fold=[]; horizons=[]
    for j in splits:
        i=np.setdiff1d(ids,j); y=t['y'][i]; e=t['traj'][i,-1]
        b=dict(s,train=dict(x=t['x'][i],y=y,traj=t['traj'][i],summary=np.array([y.mean(),y.std(),e.mean(),e.std()])),test=dict(x=t['x'][j])); r,_=test.run_dag(b)
        h=np.abs(r['state_pred']-t['traj'][j]); fold.append([np.abs(r['y_hat']-t['y'][j]).mean(),h.mean()]); horizons.append(h.mean(0))
    f=np.array(fold); print(mode.__name__, 'fold MAEs',f.tolist(),'worst',f.max(0).tolist(),'worst horizon',np.max(horizons,axis=0).tolist(),flush=True)
    scores.append(f)
a,b=scores
if np.all(b.max(0)<=a.max(0)+.003) and np.max(b.max(0))<np.max(a.max(0))-.003:
    import inspect
    text=inspect.getsource(augmented).replace('def augmented(groups,coords,n):','def _kernels(groups,coords,n,panel_c):').replace('node._design','_design').replace('node._fit','_fit').replace("p=s['panel_c'];",'p=panel_c;').replace('return kernel(groups+extra,coords,n)','return _base_kernels(groups+extra,coords,n)')
    path='/holdout_task/node_00.py'
    src=open(path).read().replace('def _kernels(groups, coords, n):','def _base_kernels(groups, coords, n):').replace('k=_kernels(g,z,n)',"k=_kernels(g,z,n,s['panel_c'])")
    open(path,'w').write(src+'\n\n'+text)
    print('Promoted the richer feature bank.',flush=True)
else:
    print('Retained the incumbent under the worst-fold rule.',flush=True)
PY
OPENBLAS_NUM_THREADS=1 timeout 145s python3 -I /tmp/check21.py; OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `1` (`1094 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.224479", "public_cv_traj_mae_exceeded:0.161672"], "cv_metrics": {"public_cv_y_mae": 0.224479, "public_cv_traj_mae": 0.161672}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.224479", "public_cv_traj_mae_exceeded:0.161672"], "cv_metrics": {"public_cv_y_mae": 0.224479, "public_cv_traj_mae": 0.161672}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
```

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
