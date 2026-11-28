# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_a03_batch_01__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `8 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `198,827` (`192,712` / `74,368` / `6,115`) |
| **Total Episode Wall Time (s)** | `214.93s` |

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

### Turn 1 — `execute_in_remote_docker` (`13003 ms` LLM)
- **Rationale**: Inspect the frozen feature contract, public fold evaluator, batch arrays, and archived evidence before choosing or editing the encoder.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_01.py && cat public_asset_index.json && sed -n '1,260p' test_dag_candidate.py && ls -la /tmp/hegel_continual_archive && find /tmp/hegel_continual_archive -maxdepth 4 -type f | head -100 && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
def describe(obj,path='root',depth=0):
 if isinstance(obj,dict):
  for k,v in obj.items(): describe(v,path+'.'+str(k),depth+1)
 elif isinstance(obj,(list,tuple)):
  print(path,type(obj).__name__,len(obj));
  if obj and depth<4: describe(obj[0],path+'[0]',depth+1)
 elif isinstance(obj,np.ndarray):
  print(path,obj.shape,obj.dtype,repr(obj.reshape(-1)[:24]))
 else: print(path,type(obj).__name__,repr(obj)[:300])
describe(p)
PY
```
- **Container Exit Code**: `0` (`481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9])
root[0].study.panel_c.v1 (24, 4) float64 array([-1.11552741,  0.4125683 ,  0.41593965,  0.92707897, -0.10311949,
        1.08891641, -0.3864823 , -0.5490804 , -0.93484649,  0.06857027,
        0.71265283, -0.34317075, -0.356826  , -0.10011617, -0.82531571,
       -1.15411766, -0.46024931, -0.33865998, -0.55779218,  0.87107855,
       -0.40267382,  0.18862527,  0.75830773,  1.04442911])
root[0].study.panel_c.v2 (24, 4) float64 array([-0.49326479,  0.80828145, -0.39543017,  0.5825854 ,  0.49537159,
        0.40984668,  0.43429311,  0.10206573, -0.06221838,  0.19547311,
       -0.39000075, -0.14033587,  0.27445792, -0.16507451, -0.03507655,
        0.15776733, -0.51074918,  0.4549021 , -0.17729727,  0.82968084,
       -0.57334274,  0.64748258, -0.26138141,  0.33397917])
root[0].study.train.x (72, 4) float64 array([ 1.04302487,  0.71417094, -0.03796375, -0.65937003, -0.62825751,
        0.28472858,  0.06104589, -0.33383217,  0.03159855, -0.38472226,
        0.91577697, -0.04721433, -1.2937986 ,  1.30737005,  0.56338778,
       -0.56962481,  1.0103185 ,  1.16846643,  0.37794759, -2.0117979 ,
       -1.45272074,  0.43929097,  0.78655313,  0.81761596])
root[0].study.train.y (72,) float64 array([-0.06143887, -0.94814589, -0.3253584 , -0.98983421, -0.99305655,
        0.62526038, -0.93107819,  0.80096802, -1.382945  , -0.0140798 ,
       -0.92476706, -0.88393534, -1.35622538, -0.48879378, -1.28844572,
        0.80839494, -1.57080021,  0.68037074, -0.01192785, -0.44131289,
       -0.60134618,  0.99477938, -0.37933021, -0.32714634])
root[0].study.train.traj (72, 8) float64 array([ 0.41509061,  0.8826455 ,  1.216516  ,  1.46304689,  1.64199708,
        1.80746882,  1.92335595,  2.0240635 , -0.26627704, -0.51932554,
       -0.63831863, -0.70275577, -0.74139086, -0.7515297 , -0.77381246,
       -0.78676194,  0.16248979,  0.30850903,  0.3679554 ,  0.39202248,
        0.39799993,  0.40803935,  0.40410062,  0.40430906])
root[0].study.train.summary (4,) float64 array([-0.18426735,  0.72869025,  0.15430069,  1.24041064])
root[0].study.test.x (36, 4) float64 array([ 0.07323944,  0.59873932,  0.39983986, -0.61566172,  0.19109093,
       -0.93137617,  1.03206617, -0.60278902, -1.28341961,  1.03321509,
        1.33487006,  0.22715204,  0.6301638 , -1.31728513,  0.49062884,
       -1.33583255,  0.7360475 , -0.20483395, -1.05326791,  0.95810745,
        0.15503971,  0.87914362, -1.49223546,  0.93640167])
root[0].study.telemetry_vec (4,) float64 array([0.22, 0.22, 1.25, 0.22])
```

### Turn 2 — `execute_in_remote_docker` (`11830 ms` LLM)
- **Rationale**: The initial output truncated the evaluator and archive inventory; isolate both and summarize the panel arrays before fitting an encoder.

```bash
cd /holdout_task && cat test_dag_candidate.py && echo ARCHIVE && du -ah /tmp/hegel_continual_archive | sort -h | tail -65 && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb')); print(type(p),len(p))
for e in p:
 s=e['study']; print('outer',e.keys(),'study',s.keys());
 for k,v in s.items():
  if isinstance(v,dict): print(k,[(a,np.shape(b),np.round(np.asarray(b).ravel()[:8],4)) for a,b in v.items()])
  else: print(k,repr(v)[:400])
PY
```
- **Container Exit Code**: `0` (`367 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
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
ARCHIVE
4.0K	/tmp/hegel_continual_archive/archive_manifest.json
4.0K	/tmp/hegel_continual_archive/conf_m102_a03_batch_00_solver.py
4.0K	/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
16K	/tmp/hegel_continual_archive
<class 'list'> 1
outer dict_keys(['study']) study dict_keys(['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec'])
eval_times array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ])
panel_a [('u', (48, 4), array([-0.7609, -0.7816, -0.7784,  0.2744,  0.531 ,  0.1156, -0.4496,
       -0.0601])), ('v', (48, 4), array([-0.3408, -0.2662, -0.8562,  0.4523,  0.4263,  0.3907, -0.3148,
       -0.2371]))]
panel_b [('u', (24, 4), array([-0.5447, -0.4587,  0.181 ,  0.4827, -1.0835, -1.1942,  0.0221,
       -1.1333])), ('v', (24, 4), array([-0.2272,  0.1134, -0.2079,  0.601 ,  0.1725,  1.1682, -1.3036,
        0.1404]))]
panel_c [('u', (24, 4), array([ 1.2448, -0.8744, -0.5136, -1.0178, -0.0484, -0.5451, -0.3918,
        1.1197])), ('v1', (24, 4), array([-1.1155,  0.4126,  0.4159,  0.9271, -0.1031,  1.0889, -0.3865,
       -0.5491])), ('v2', (24, 4), array([-0.4933,  0.8083, -0.3954,  0.5826,  0.4954,  0.4098,  0.4343,
        0.1021]))]
train [('x', (72, 4), array([ 1.043 ,  0.7142, -0.038 , -0.6594, -0.6283,  0.2847,  0.061 ,
       -0.3338])), ('y', (72,), array([-0.0614, -0.9481, -0.3254, -0.9898, -0.9931,  0.6253, -0.9311,
        0.801 ])), ('traj', (72, 8), array([0.4151, 0.8826, 1.2165, 1.463 , 1.642 , 1.8075, 1.9234, 2.0241])), ('summary', (4,), array([-0.1843,  0.7287,  0.1543,  1.2404]))]
test [('x', (36, 4), array([ 0.0732,  0.5987,  0.3998, -0.6157,  0.1911, -0.9314,  1.0321,
       -0.6028]))]
telemetry_vec array([0.22, 0.22, 1.25, 0.22])
```

### Turn 3 — `execute_in_remote_docker` (`31386 ms` LLM)
- **Rationale**: Read the small archived solver and diagnostics in full to recover the prior operator and design targeted numerical comparisons against the new panels.
- **Alternatives Weighed**: `Fit a generic supervised encoder immediately` (rejected: The archived solver and paired panels may identify the transformation without using outcome labels to estimate calibration.)

```bash
cat /tmp/hegel_continual_archive/*.json /tmp/hegel_continual_archive/*solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment calibration orthogonal Procrustes' || true
```
- **Container Exit Code**: `0` (`320 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
as": {
      "node_00": "1cbb261bff4025eba9291fa441be191940cf9aff1fb6560a8524c6db774d70ee"
    },
    "non_identity_checks": {},
    "shared_fanout_nodes": [],
    "topo_order": [
      "node_00"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 3154.3
}"""Calibrated bilinear readout and reduced-rank trajectory prediction."""
import numpy as np


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = _affine(u)
    coef = _fit(a, v)
    for _ in range(24):
        residual = np.linalg.norm(a @ coef - v, axis=1)
        scale = max(float(np.median(residual)), 1e-12)
        weight = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(residual, 1e-12)))
        coef = _fit(a * weight[:, None], v * weight[:, None])
    residual = np.linalg.norm(a @ coef - v, axis=1)
    good = residual <= 5 * max(float(np.median(residual)), 1e-12)
    if good.sum() >= a.shape[1] and np.linalg.matrix_rank(a[good]) == a.shape[1]:
        coef = _fit(a[good], v[good])
        mat = coef[1:]
        off = mat - np.diag(np.diag(mat))
        if np.linalg.norm(off) <= 5 * max(float(np.median(residual)), 1e-12):
            for j in range(u.shape[1]):
                coef[0, j], coef[j + 1, j] = _fit(a[good][:, [0, j + 1]], v[good, j])
            coef[1:] = np.diag(np.diag(coef[1:]))
    return coef


def _readout(z):
    return np.column_stack((z, z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    target = np.asarray(study['test']['x'], dtype=float)
    a = _calibration(study['panel_a'])
    b = _fit(_affine(study['panel_b']['u']), study['panel_b']['v'])

    def encode(rows):
        calibrated = np.linalg.solve(a[1:].T, (rows - a[0]).T).T
        return _affine(calibrated) @ b

    z, zt = encode(x), encode(target)
    yh = _readout(zt) @ _fit(_readout(z), train['y'])
    q, r = np.linalg.qr(z, mode='reduced')
    u, sv, vh = np.linalg.svd(q.T @ train['traj'], full_matrices=False)
    rank = min(2, len(sv))
    coef = _fit(r, (u[:, :rank] * sv[:rank]) @ vh[:rank])
    sp = zt @ coef
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
No matching skill cards found.
```

### Turn 4 — `execute_in_remote_docker` (`23995 ms` LLM)
- **Rationale**: Measure archived-encoder transfer on the current batch and use leave-one-row-out panel residuals to distinguish nonlinear calibration from isolated corruptions or cross-coordinate mixing.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import glob,runpy,pickle,numpy as np,itertools,copy
np.set_printoptions(precision=4,suppress=True,linewidth=135)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; old=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); cv=copy.deepcopy(s); cv['test']={'x':s['train']['x'][52:]}; cv['train']={k:v[:52] for k,v in s['train'].items()}; out=old['solve_batch'](cv); print('archive transfer',*[np.mean(np.abs(out[k]-s['train'][t][52:])) for k,t in [('y_hat','y'),('state_pred','traj')]])
def analyze(u,v):
 a=np.column_stack([np.ones(len(u)),u]); c=np.linalg.lstsq(a,v,rcond=None)[0]; print('affine',c); print('res norm sorted',np.sort(np.linalg.norm(a@c-v,axis=1))); print('ranges u/v',u.min(0),u.max(0),v.min(0),v.max(0));
 for rev in [False,True]:
  x,y=(v,u) if rev else (u,v)
  for deg in [1,2,3,4,5]:
   err=[]
   for j in range(4):
    d=np.polynomial.polynomial.polyvander(x[:,j],deg); pin=np.linalg.pinv(d); yh=d@(pin@y[:,j]); err.append(np.median(np.abs((y[:,j]-yh)/np.maximum(1-(d*pin.T).sum(1),1e-12))))
   print('mono LOOCV',rev,deg,np.array(err))
  for deg in [1,2,3]:
   terms=[t for n in range(deg+1) for t in itertools.combinations_with_replacement(range(4),n)]; d=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]);
   if d.shape[1]>=len(x):continue
   pin=np.linalg.pinv(d); er=np.linalg.norm((y-d@(pin@y))/np.maximum(1-(d*pin.T).sum(1),1e-12)[:,None],axis=1); print('full LOOCV',rev,deg,np.quantile(er,[0,.25,.5,.75,.9,1]))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; print('\n',k)
 for l in p:
  if l!='u': print(l); analyze(p['u'],p[l])
p=s['panel_a']; c=old['_calibration'](p); print('robust A',c); print('panel A u v and residual'); print(np.column_stack([p['u'],p['v'],p['v']-old['_affine'](p['u'])@c]))
PY
```
- **Container Exit Code**: `0` (`355 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.6233  0.1987  0.0412]
 [-0.9714 -0.9081 -0.9986 -0.9844 -0.4688 -0.2699 -0.3529 -0.5033  0.5445  0.7151  0.4537  0.5145]
 [-0.9198  0.8756  0.7563 -1.2725 -1.5122  1.1     0.9277 -0.9123 -0.5523 -0.0822  0.0724  0.4159]
 [ 0.481  -1.0577 -0.6994 -0.4066  0.5696 -0.6384 -0.3872 -0.6254  0.0835  0.5286  0.136  -0.2299]
 [-0.5945 -0.6819  1.1157  0.8361 -0.2507 -1.1849  1.3363  0.9504  0.3734 -0.4747  0.1407  0.0075]
 [-0.8644 -0.5826  0.9116  0.4689 -0.557  -0.928   1.0071  0.6275  0.3458 -0.3384  0.0047  0.0802]
 [-0.0645  0.4214 -0.7399 -0.9096 -0.2449  0.7179 -0.1484 -0.9749 -0.1679  0.0876  0.4132 -0.0377]
 [ 0.5803  0.9375  0.1551  0.5949  0.3684  1.2134  0.0568  0.5464 -0.2202 -0.044  -0.2292 -0.1367]
 [ 1.2745 -1.0059  0.7169  1.082   1.3133 -1.4806  0.6846  1.304   0.0081 -0.3767 -0.1335  0.0962]
 [-0.2113  0.218   0.1189 -0.2719  5.3841  6.4374 -6.1388 -5.9829  5.6127  6.0542 -6.3905 -5.7325]
 [ 1.0983  0.3582 -0.6585 -1.1298  1.1393  0.6483  0.0375 -2.0902  0.016   0.0947  0.522  -0.9157]
 [ 0.067   0.6098 -0.72    0.2095 -0.1682  0.8833 -0.7472  0.193  -0.2269  0.024  -0.2044 -0.0751]
 [-0.6706 -0.9025  0.4077  1.2968 -0.1974 -1.1202  0.2097  1.5647  0.5054 -0.1419 -0.3155  0.1256]
 [-1.0806  0.3835  0.1041 -0.1945 -1.4477  0.5194  0.227   0.2225 -0.3216 -0.0649 -0.0107  0.3896]
 [-1.2165  0.7325  0.0293  0.9752 -1.9167  0.94   -0.1993  1.1207 -0.6505 -0.0683 -0.3662  0.028 ]
 [-1.2237 -0.1699  0.4507  0.1309 -1.2877 -0.1915  0.4752  0.4715 -0.0141 -0.1034 -0.0907  0.2881]
 [-0.4593 -0.78   -1.2391 -0.9811 -0.098  -0.002  -0.6416 -0.8242  0.3866  0.8274  0.3927  0.1901]
 [ 1.0783 -0.6366 -0.797  -1.0151  1.0831 -0.1123 -0.1386 -1.8605 -0.0195  0.5429  0.477  -0.8096]
 [-0.9245  0.9932 -1.2913 -0.631  -1.5919  1.3    -0.9691 -0.177  -0.627  -0.0251  0.1147  0.4603]
 [ 0.1755  0.5831  0.5073 -0.5973 -0.0444  0.6946  0.6496 -0.7101 -0.2151 -0.1323  0.0301 -0.1092]
 [ 0.7684 -0.0529 -0.8714  0.6447  0.7041  0.3885 -1.2005  0.5925 -0.0787  0.3345 -0.5144 -0.1443]
 [ 0.3667 -0.9023  0.1858  0.8642  0.4667 -0.9848  0.0205  0.9406  0.0986 -0.0067 -0.2946 -0.0326]
 [-0.8053  0.79   -0.1404  1.2693 -1.2894  1.0257 -0.5129  1.51   -0.4476 -0.0525 -0.5191  0.1005]
 [ 0.0889 -0.7453  1.0212 -0.3447  0.2504 -1.2338  1.1657 -0.3823  0.1691 -0.4466  0.0595 -0.0534]
 [-1.2677 -0.4564 -0.3299 -0.8385 -1.1488 -0.1698  0.1309 -0.1724  0.1703  0.2665  0.3041  0.6883]
 [ 0.328  -1.2915  1.2153 -0.29    0.5255 -2.3789  1.4403 -0.4232  0.1974 -0.928   0.1503 -0.1532]]
```

### Turn 5 — `execute_in_remote_docker` (`31583 ms` LLM)
- **Rationale**: Compare calibration families on held-out panel rows after isolating gross corruptions. The coefficient and residual patterns can identify an invertible forward map without using task outcomes.
- **Alternatives Weighed**: `Treat the Panel A discrepancy as outliers alone` (rejected: Many ordinary rows retain sizable residuals after robust affine fitting.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived affine calibration grammar suffices for both prediction heads on the new batch.` | Witness: `Archived transfer exceeds 0.038 on both heads; Panel A also has substantial residuals beyond its three gross response outliers, while Panel B remains affine to about 0.0002.` | Blanket Diagnosis: `Panel A is the leading source of shared encoder error; the frozen readout and Panel B need not be replaced.` | Sublated Pivot: `Retain paired-panel calibration but test sparse nonlinear and cross-coordinate terms in the Panel A forward operator before choosing an inverse.`

```bash
cat > /tmp/probe_cal.py <<'PY'
import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=120)
def poly(x,d):
 t=[t for k in range(d+1) for t in itertools.combinations_with_replacement(range(x.shape[1]),k)]
 return np.column_stack([np.prod(x[:,q],axis=1) for q in t]),t
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def stats(a,y):
 p=np.linalg.pinv(a); c=p@y; e=y-a@c; loo=e/(1-(a*p.T).sum(1))[:,None]; return c,np.sqrt(np.mean(e**2)),np.sqrt(np.mean(loo**2)),np.max(np.abs(loo))
for key in ('panel_a','panel_b','panel_c'):
 p=s[key]; x=p['u'];
 for col in set(p)-{'u'}:
  y=p[col]; good=np.max(abs(y),axis=1)<3; x1,y1=x[good],y[good]; print(key,col,len(x1))
  for name,u,v in [('forward',x1,y1),('inverse',y1,x1)]:
   for d in range(1,4):
    a,t=poly(u,d)
    if a.shape[1]>=len(u): continue
    c,*r=stats(a,v); print(name,d,r)
    if key=='panel_a' and name=='forward':
     for j in range(4): print(j,[(q,round(c[i,j],5)) for i,q in enumerate(t) if abs(c[i,j])>=.012])
  if key=='panel_a':
   a,t=poly(x1,2); c=fit(a,y1); print('quadratic residual correlations with trig, rational and high degrees')
   e=y1-a@c
   for j in range(4):
    bank=[]
    for i in range(4):
     for k in range(4):
      for sg in [-1,0,1]:
       z=x1[:,i]+sg*x1[:,k]
       for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('cube',lambda t:t**3),('rat',lambda t:t/(1+t*t)),('abs',np.abs),('rat2',lambda t:1/(1+t*t))]:
        for w in ([.5,1.,1.25,2.] if fn in ('sin','cos','rat','rat2','tanh') else [1.]):
         b=f(w*z)
         for mult in [-1,j]:
          v=b if mult==-1 else b*x1[:,mult]; v=v-a@fit(a,v); norm=np.linalg.norm(v)*np.linalg.norm(e[:,j]);
          if norm>1e-8:bank.append((abs(v@e[:,j])/norm,(i,k,sg,fn,w,mult)))
    print(j,sorted(bank,reverse=True)[:12])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_cal.py
```
- **Container Exit Code**: `0` (`875 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8, (1, 0, 0, 'cube', 1.0, -1)), (0.999980272288035, (1, 3, 0, 'sin', 0.5, -1)), (0.999980272288035, (1, 2, 0, 'sin', 0.5, -1)), (0.999980272288035, (1, 1, 0, 'sin', 0.5, -1)), (0.999980272288035, (1, 0, 0, 'sin', 0.5, -1)), (0.9999473130123403, (1, 3, 0, 'cos', 0.5, 1)), (0.9999473130123403, (1, 2, 0, 'cos', 0.5, 1)), (0.9999473130123403, (1, 1, 0, 'cos', 0.5, 1))]
2 [(0.9999976050797866, (2, 3, 0, 'cube', 1.0, -1)), (0.9999976050797866, (2, 2, 1, 'cube', 1.0, -1)), (0.9999976050797866, (2, 2, 0, 'cube', 1.0, -1)), (0.9999976050797866, (2, 1, 0, 'cube', 1.0, -1)), (0.9999976050797866, (2, 0, 0, 'cube', 1.0, -1)), (0.999982306147128, (2, 3, 0, 'sin', 0.5, -1)), (0.999982306147128, (2, 2, 0, 'sin', 0.5, -1)), (0.999982306147128, (2, 1, 0, 'sin', 0.5, -1)), (0.999982306147128, (2, 0, 0, 'sin', 0.5, -1)), (0.9999569524906823, (2, 3, 0, 'cos', 0.5, 2)), (0.9999569524906823, (2, 2, 0, 'cos', 0.5, 2)), (0.9999569524906823, (2, 1, 0, 'cos', 0.5, 2))]
3 [(0.9999980682534991, (3, 3, 1, 'cube', 1.0, -1)), (0.9999980682534991, (3, 3, 0, 'cube', 1.0, -1)), (0.9999980682534991, (3, 2, 0, 'cube', 1.0, -1)), (0.9999980682534991, (3, 1, 0, 'cube', 1.0, -1)), (0.9999980682534991, (3, 0, 0, 'cube', 1.0, -1)), (0.9999873512833339, (3, 3, 0, 'sin', 0.5, -1)), (0.9999873512833339, (3, 2, 0, 'sin', 0.5, -1)), (0.9999873512833339, (3, 1, 0, 'sin', 0.5, -1)), (0.9999873512833339, (3, 0, 0, 'sin', 0.5, -1)), (0.9999663079446404, (3, 3, 0, 'cos', 0.5, 3)), (0.9999663079446404, (3, 2, 0, 'cos', 0.5, 3)), (0.9999663079446404, (3, 1, 0, 'cos', 0.5, 3))]
panel_b v 24
forward 1 [8.173982174082755e-05, 0.00010388056021336145, 0.00022861380864558828]
forward 2 [5.830107989560476e-05, 0.00018777918180520084, 0.0007057061943419865]
inverse 1 [9.28888890060559e-05, 0.00011804864240791005, 0.00031981244906443765]
inverse 2 [6.625311160038229e-05, 0.00021338885967275858, 0.0006397173271418869]
panel_c v2 24
forward 1 [0.09230759153590513, 0.12334276483263643, 0.5128179820731564]
forward 2 [0.0032036069856967637, 0.012413516320696549, 0.04679373040028014]
inverse 1 [0.17982514463235197, 0.24429741546000897, 0.9203173665058821]
inverse 2 [0.05917512292071766, 0.413630094223658, 2.2703288603085405]
panel_c v1 24
forward 1 [0.11242619831431142, 0.14720498570489213, 0.39375215008683523]
forward 2 [0.07646351273760947, 0.3158758671072787, 1.035990740287379]
inverse 1 [0.11989869885748855, 0.15485439580129548, 0.3741744758892213]
inverse 2 [0.07827419541655545, 0.3361911443139293, 1.8154026194423614]
```

### Turn 6 — `execute_in_remote_docker` (`35308 ms` LLM)
- **Rationale**: Fit the shared cyclic terms from paired calibration data only, downweight and trim gross panel outliers, then invert by damped Newton iteration. Preserve the frozen heads and test the exact node in an isolated Python process.
- **Alternatives Weighed**: `Use a dense polynomial inverse` (rejected: Its cubic leave-one-out RMSE is 0.172; the sparse forward fit captures the observed cyclic coupling and has much smaller calibration residuals.)

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Invert the cyclic cubic sensor calibration, then align to readout coordinates."""

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_sensor(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    d = u.shape[1]
    nxt = np.roll(u, -1, axis=1)
    # Coordinate-specific offsets and gains; shared cyclic and cubic terms.
    eyes = np.broadcast_to(np.eye(d), (len(u), d, d))
    design = np.concatenate((eyes, u[..., None] * eyes,
                             np.stack((nxt, u * nxt, u ** 3), axis=-1)), axis=-1)
    a = design.reshape(-1, 2 * d + 3)
    b = v.ravel()
    w = np.ones(len(u))
    for _ in range(24):
        rw = np.repeat(w, d)
        coef = np.linalg.lstsq(a * rw[:, None], b * rw, rcond=None)[0]
        err = np.linalg.norm(design @ coef - v, axis=1)
        scale = max(np.median(err), 1e-12)
        w = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(err, 1e-12)))

    good = err <= 5. * scale
    if np.count_nonzero(good) * d >= a.shape[1]:
        refit, _, rank, _ = np.linalg.lstsq(design[good].reshape(-1, a.shape[1]),
                                           v[good].ravel(), rcond=None)
        if rank == a.shape[1]:
            coef = refit
    return coef[:d], coef[d:2*d], coef[2*d:]


def _sensor(u, offset, gain, coupling, interaction, cubic):
    nxt = np.roll(u, -1, axis=1)
    return offset + gain * u + (coupling + interaction * u) * nxt + cubic * u**3


def _inverse(x, calibration):
    offset, gain, (coupling, interaction, cubic) = calibration
    j = np.arange(x.shape[1])
    nxt = (j + 1) % len(j)
    linear = np.diag(gain)
    linear[j, nxt] += coupling
    u = np.linalg.solve(linear, (x - offset).T).T
    for _ in range(40):
        res = _sensor(u, offset, gain, coupling, interaction, cubic) - x
        error = np.linalg.norm(res, axis=1)
        if not len(error) or np.max(error) <= 1e-11:
            break
        jac = np.broadcast_to(linear, (len(u), *linear.shape)).copy()
        jac[:, j, j] += 3. * cubic * u**2 + interaction * u[:, nxt]
        jac[:, j, nxt] += interaction * u
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        rate = np.ones((len(u), 1))
        for _ in range(18):
            trial = u - rate * step
            trial_err = np.linalg.norm(_sensor(trial, offset, gain, coupling, interaction, cubic) - x, axis=1)
            if np.all((trial_err < error) | (error <= 1e-11)):
                break
            rate[trial_err >= error] *= 0.5
        u = np.where((trial_err < error)[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibration = _fit_sensor(study['panel_a'])
    panel = study['panel_b']
    align = np.linalg.lstsq(_affine(np.asarray(panel['u'], dtype=float)),
                            np.asarray(panel['v'], dtype=float), rcond=None)[0]

    def encode(rows):
        return _affine(_inverse(np.asarray(rows, dtype=float), calibration)) @ align

    return {'z': encode(study['train']['x']), 'zt': encode(study['test']['x'])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.003415, "public_cv_traj_mae": 0.00381}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8edb46aba47302eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8edb46ab (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32658 ms` LLM)
- **Rationale**: Stress-test the verified artifact without editing it. Check both frozen heads on random and coordinate-tail holdouts, then test calibration-row sensitivity and forward–inverse consistency independently.
- **Alternatives Weighed**: `Submit on the public gate alone` (rejected: The gate passes, but coordinate-tail and calibration-row holdouts directly test whether the inverse remains reliable away from the nominal split.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, numpy as np
root = '/holdout_task/'
enc = runpy.run_path(root + 'node_00.py')
head = runpy.run_path(root + '_frozen_nodes/node_01.py')['run_node']
batch = pickle.load(open(root + 'public_batch.pkl', 'rb'))
rng = np.random.default_rng(41)
worst_cv = np.zeros(2)
worst_panel = np.zeros(2)
worst_roundtrip = 0.
worst_panel_resid = 0.
keys = ('y_hat', 'state_pred')
def evaluate(s, tr, va):
    fold = dict(s, train=dict(s['train']), test={'x': s['train']['x'][va]})
    for k in ('x', 'y', 'traj'):
        fold['train'][k] = s['train'][k][tr]
    z = enc['run_node']({'__input__': fold})
    assert set(z) == {'z', 'zt'} and z['z'].shape == (len(tr), 4) and z['zt'].shape == (len(va), 4)
    p = head({'__input__': fold, 'node_00': z})
    return np.array([np.mean(np.abs(p[k] - s['train'][t][va])) for k, t in zip(keys, ('y', 'traj'))])
for item in batch:
    s = item['study']
    n = len(s['train']['x'])
    m = n - 52 if n > 52 else max(1, n // 4)
    x = s['train']['x']
    a = enc['_fit_sensor'](s['panel_a'])
    rows = np.concatenate((x, s['test']['x']))
    z = enc['_inverse'](rows, a)
    worst_roundtrip = max(worst_roundtrip, np.max(np.abs(enc['_sensor'](z, a[0], a[1], *a[2]) - rows)))
    ids = np.arange(n)
    slices = [ids[-m:]] + [rng.permutation(n)[:m] for _ in range(24)]
    slices += [np.argsort(v)[-m:] for v in [np.linalg.norm(x, axis=1), *x.T, *(-x.T)]]
    for va in slices:
        worst_cv = np.maximum(worst_cv, evaluate(s, np.setdiff1d(ids, va), va))
    panel = s['panel_a']
    full_resid = np.linalg.norm(enc['_sensor'](panel['u'], a[0], a[1], *a[2]) - panel['v'], axis=1)
    inliers = full_resid < 0.01
    for i in range(len(panel['u'])):
        keep = np.arange(len(panel['u'])) != i
        sub = dict(s, panel_a={k: v[keep] for k, v in panel.items()})
        sub_a = enc['_fit_sensor'](sub['panel_a'])
        if inliers[i]:
            residual = enc['_sensor'](panel['u'][i:i+1], sub_a[0], sub_a[1], *sub_a[2]) - panel['v'][i:i+1]
            worst_panel_resid = max(worst_panel_resid, np.linalg.norm(residual))
        worst_panel = np.maximum(worst_panel, evaluate(sub, ids[:-m], ids[-m:]))
print('Worst random/tail holdout scalar and trajectory MAE:', worst_cv)
print('Worst leave-one-calibration-row-out scalar and trajectory MAE:', worst_panel)
print('Max inverse roundtrip error:', worst_roundtrip, 'max held-out inlier panel L2 error:', worst_panel_resid)
assert np.all(worst_cv <= .038) and np.all(worst_panel <= .038) and worst_roundtrip < 1e-8 and worst_panel_resid < .001
PY
```
- **Container Exit Code**: `0` (`912 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst random/tail holdout scalar and trajectory MAE: [0.00458364 0.00447728]
Worst leave-one-calibration-row-out scalar and trajectory MAE: [0.00341633 0.00381072]
Max inverse roundtrip error: 4.440892098500626e-16 max held-out inlier panel L2 error: 0.00046855403491489155

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8edb46aba47302eb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8edb46ab (directory).
```

### Turn 8 — `submit_completed_task` (`15222 ms` LLM)
- **Rationale**: The hash-bound candidate passed the isolated public evaluator and independent random, coordinate-tail, and leave-one-calibration-row-out checks. Both heads stayed below the required MAE threshold on every tested slice.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py`, preserving the `z`/`zt` contract and leaving `node_01` unchanged. The public evaluator accepted the candidate: scalar MAE 0.003415 and trajectory MAE 0.003810, both below 0.038. Each node executed once.`

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
