# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b02_batch_06__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `398,714` (`383,291` / `109,056` / `15,423`) |
| **Total Episode Wall Time (s)** | `536.27s` |

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

### Turn 1 — `execute_in_remote_docker` (`11940 ms` LLM)
- **Rationale**: Inspect public inputs, frozen consumers, and archive contents before selecting a calibration or prediction model.

```bash
ls -la /holdout_task /holdout_task/_frozen_nodes /tmp/hegel_continual_archive; cat /holdout_task/_dag_spec.json; cat /holdout_task/_frozen_nodes/*.py; cat /holdout_task/test_dag_candidate.py; cat /holdout_task/public_asset_index.json; du -ah /tmp/hegel_continual_archive | sort -h | tail -60; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def f(p,d=0):
 if isinstance(p,dict):
  for k,v in p.items(): print(' '*d,k); f(v,d+1)
 elif isinstance(p,np.ndarray): print(' '*d,p.shape,p.dtype,p.ravel()[:15])
 else: print(' '*d,type(p),str(p)[:300])
f(p)
PY
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pe": "float64",
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
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}4.0K	/tmp/hegel_continual_archive/archive_manifest.json
4.0K	/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py
4.0K	/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py
4.0K	/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py
4.0K	/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
24K	/tmp/hegel_continual_archive
 <class 'list'> [{'study': {'eval_times': array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ]), 'panel_a': {'u': array([[ 0.41959874,  0.77161493,  1.25571668, -0.74124415],
       [ 0.42985905, -0.07963645,  0.72131337,  0.88892917],
       [-0.67099802, -0
```

### Turn 2 — `execute_in_remote_docker` (`17494 ms` LLM)
- **Rationale**: The prior inspection was truncated; isolate the archive, frozen sink contract, CV logic, and numerical panel diagnostics needed to design both editable nodes.

```bash
cat /tmp/hegel_continual_archive/*; cat /holdout_task/_frozen_nodes/node_03.py; sed -n '75,240p' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment ridge nuisance viscoelastic relaxation' 2>&1 | head -30; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
for key,z in s.items():
 for k,x in (z.items() if isinstance(z,dict) else [(key,z)]): print(key,k,x.shape,x[:3], 'mean',x.mean(axis=0),'sd',x.std(axis=0))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; A=np.c_[np.ones(len(u)),u];
 for v in list(p)[1:]:
  co=np.linalg.lstsq(A,p[v],rcond=None)[0]; e=p[v]-A@co; print(k,v,co,'resid',np.std(e,axis=0),'sorted row errors',np.sort(np.linalg.norm(e,axis=1)))
PY
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-0.1283 -0.2614 -0.3698 -0.4506 -0.5101 -0.5681 -0.6202 -0.6509]
 [-0.0156 -0.1474 -0.2654 -0.3437 -0.403  -0.4393 -0.4723 -0.4937]] mean [-0.0654 -0.1335 -0.1844 -0.2239 -0.2583 -0.2853 -0.3115 -0.3322] sd [0.1552 0.3419 0.4856 0.5954 0.6813 0.7495 0.803  0.8452]
train summary (4,) [-0.0885  0.9477 -0.3323] mean 0.34298525638419886 sd 0.5612753273812433
test x (36, 4) [[-0.5991  0.9753 -0.6915  0.4717]
 [ 0.0328 -0.184   0.0228  0.5249]
 [-0.1926  0.9271 -0.0883  0.1357]] mean [ 0.2663  0.1816 -0.2223  0.2889] sd [0.6404 0.6858 0.7507 0.6865]
telemetry_vec telemetry_vec (4,) [1.25 1.25 0.22] mean 0.9925 sd 0.4460030829489859
panel_a v [[ 0.3028 -0.0125 -0.045   0.0008]
 [ 1.1312 -0.0694  0.0953 -0.372 ]
 [-0.3188  1.1976 -0.0055 -0.1419]
 [-0.0793 -0.2314  0.9327  0.0797]
 [ 0.0095 -0.1797 -0.0758  0.8712]] resid [1.5238 1.4934 1.5486 1.5234] sorted row errors [ 0.0818  0.1088  0.2029  0.2297  0.2452  0.263   0.2767  0.281   0.291
  0.3012  0.3053  0.3158  0.3561  0.3565  0.3566  0.3698  0.373   0.3735
  0.3846  0.4097  0.4167  0.4227  0.4355  0.4563  0.4835  0.4924  0.509
  0.5132  0.5193  0.5261  0.5263  0.5273  0.5304  0.5483  0.5726  0.5897
  0.6181  0.6213  0.6223  0.6234  0.6351  0.6533  0.6583  0.7377  0.7714
 11.6373 11.7533 12.7087]
panel_b v [[-0.1274 -0.0932  0.1517 -0.1562]
 [-0.1728  0.3049 -0.1109  0.355 ]
 [-0.0396 -0.2778 -0.3797  0.1824]
 [-0.2421  0.1607 -0.2533 -0.2298]
 [ 0.4804  0.2608 -0.309  -0.0794]] resid [0.1252 0.1144 0.1652 0.1292] sorted row errors [0.1396 0.1487 0.157  0.1673 0.17   0.1814 0.1994 0.227  0.2321 0.2334
 0.2494 0.2531 0.2575 0.2632 0.2642 0.2707 0.2766 0.2771 0.3032 0.307
 0.3101 0.4058 0.4158 0.4606]
panel_c v1 [[-0.0188 -0.0203  0.1316  0.1805]
 [ 0.1323 -0.1122 -0.0798  0.3394]
 [-0.0501  0.1097  0.042  -0.1103]
 [ 0.2832 -0.0723 -0.0064  0.0609]
 [ 0.1334 -0.0884 -0.0718 -0.1128]] resid [0.6095 0.5766 0.5721 0.5661] sorted row errors [0.2495 0.3659 0.4941 0.5548 0.6048 0.6281 0.6719 0.8138 0.9028 0.9111
 0.9193 0.9317 0.976  1.0218 1.1694 1.2014 1.2984 1.327  1.5019 1.5649
 1.5812 1.6496 1.8417 2.211 ]
panel_c v2 [[-0.0452 -0.0357 -0.0785  0.0194]
 [-0.489   0.1613  0.127   0.0454]
 [-0.1407 -0.2769 -0.2541 -0.2341]
 [ 0.1188  0.2801 -0.0274 -0.3333]
 [-0.0848  0.1378 -0.4382  0.1324]] resid [0.0748 0.086  0.1026 0.0921] sorted row errors [0.0972 0.111  0.1211 0.1358 0.1438 0.1471 0.148  0.1521 0.1534 0.1604
 0.1643 0.169  0.1733 0.1775 0.1827 0.1894 0.2012 0.2038 0.2119 0.214
 0.2153 0.222  0.2456 0.2485]
```

### Turn 3 — `execute_in_remote_docker` (`11044 ms` LLM)
- **Rationale**: Read the nearest archived solver and exact frozen-sink and fold contracts without truncation. Panel A supports testing robust calibration, but the residual gap alone does not explain both prediction failures.

```bash
cat /tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py; cat /holdout_task/_frozen_nodes/node_03.py; sed -n '75,220p' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment'
```
- **Container Exit Code**: `0` (`369 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
r s in srcs) or not out_num or not (out_num - in_num):
                raise RuntimeError(f"DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check")
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
    return outputs[spec_obj["sink_node_id"]], ec

st = pickle.load(open("/holdout_task/public_batch.pkl", "rb"))[0]["study"]
out_full, ec = run_dag(st)
n_te, k_t = len(st["test"]["x"]), len(st["eval_times"])
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
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
No matching skill cards found.
```

### Turn 4 — `execute_in_remote_docker` (`29565 ms` LLM)
- **Rationale**: Compare the archive against low-order feature models on the official split and four additional splits. Use leave-one-out panel residuals to distinguish actual calibration gains from polynomial overfit.

```bash
cat /holdout_task/_frozen_nodes/node_03.py; cat /tmp/hegel_continual_archive/conf_m102_b02_batch_0{0,1}_solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat >/tmp/diagnose.py <<'PY'
import sys,pickle,numpy as np, importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
z,b=old.parts(s); f=old.times(s); t=s['train']; n=len(t['y']); T=np.c_[t['y'],old.fit(f,t['traj'].T).T]; sl=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(41).permutation(n),4)]
def D(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,cs],axis=1) for k in range(1,d+1) for cs in combinations_with_replacement(range(x.shape[1]),k)])
def assess(a):
 out=[]
 for i,j in sl:
  e=a[j]@old.fit(a[i],T[i])-T[j];out.append([np.abs(e[:,0]).mean(),np.abs(e[:,1:]@f.T).mean()]);
 return np.round(out[0],5),np.round(np.mean(out[1:],axis=0),5)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u']; print(k)
 for key in set(p)-{'u'}:
  v=p[key]; keep=np.ones(len(u),bool)
  if k=='panel_a':keep=old.cal(s,u)[2]<2
  for nam,A in [('aff',D(u)),('AF',old.AF(u)),('pow2',poly(u,2)),('sinh',D(np.sinh(u))),('asinh',D(np.arcsinh(u))),('tanh',D(np.tanh(u)))]:
   A=A[keep];y=v[keep];w=old.fit(A,y);e=y-A@w; lev=np.sum(A*np.linalg.pinv(A).T,axis=1);print(key,nam,'rmse',np.std(e),'LOO',np.std(e/np.maximum(1-lev,.0001)[:,None]))
  if k!='panel_a':
   H=lambda q: D(np.sign(u)*np.maximum(abs(u)-q[0],0)**q[1]); sol=least_squares(lambda q:(H(q)@old.fit(H(q),v)-v).ravel(),[.3,1.5],bounds=([0,.2],[.8,5]));print('threshold',sol.x,np.std(sol.fun));print('range',u.min(0),u.max(0))
print('T basis errors',np.abs(T[:,1:]@f.T-t['traj']).mean(),T[:8]); print('z',z.min(0),z.max(0))
for name,a in [('raw',np.r_[t['x'],s['test']['x']]),('z',z),('b',b),('zb',np.c_[z,b]),('asinhz',np.arcsinh(z)),('asinhzb',np.arcsinh(np.c_[z,b]))]:
 for d in range(1,4 if a.shape[1]<5 else 3): print(name,d,assess(poly(a,d)))
for i,j in sl:
 r=dict(s,train=dict(x=t['x'][i],y=t['y'][i],traj=t['traj'][i]),test=dict(x=t['x'][j])); yp,tp=old.predict(r); print('archive',np.abs(yp-t['y'][j]).mean(),np.abs(tp-t['traj'][j]).mean())
print('corr',np.corrcoef(np.c_[z[:n],b[:n],T].T)[-3:])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/diagnose.py
```
- **Container Exit Code**: `0` (`516 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
804889912
v2 sinh rmse 0.08925666659051976 LOO 0.11873025469165754
v2 asinh rmse 0.09163465349907109 LOO 0.11943843793392303
v2 tanh rmse 0.09655012916900786 LOO 0.12437659753833913
threshold [0.     1.0577] 0.0893033391631162
range [-0.9992 -1.2393 -1.1483 -1.1658] [1.1932 1.1579 1.2381 1.1471]
v1 aff rmse 0.5813147045535819 LOO 0.8112581194152404
v1 AF rmse 7.792419500156482e-05 LOO 0.00018293515498389663
v1 pow2 rmse 5.881040699939683e-05 LOO 0.00018090355625105368
v1 sinh rmse 0.5795723429510369 LOO 0.8162906618867589
v1 asinh rmse 0.5824827331094854 LOO 0.8072102682008695
v1 tanh rmse 0.5837907379170743 LOO 0.8013507733811777
threshold [0.8 5. ] 0.5406315954661992
range [-0.9992 -1.2393 -1.1483 -1.1658] [1.1932 1.1579 1.2381 1.1471]
T basis errors 0.008181377121257133 [[ 0.3407 -0.0434 -1.2045]
 [-1.1191 -0.0354 -0.7369]
 [ 0.5599  0.0885 -0.7256]
 [ 1.5698 -0.0509 -1.2606]
 [ 1.0227 -0.0493 -1.2786]
 [-0.4752 -0.0319 -0.8524]
 [-1.2919  0.0559  1.2083]
 [-1.2154  0.03   -0.2585]]
z [-1.1225 -1.1398 -1.1244 -1.1309] [1.1101 1.1447 1.1269 1.1353]
raw 1 (array([0.2947, 0.1986]), array([0.3449, 0.1928]))
raw 2 (array([0.1448, 0.1019]), array([0.2459, 0.1782]))
raw 3 (array([0.1525, 0.1101]), array([0.1691, 0.1627]))
z 1 (array([0.1696, 0.1905]), array([0.2268, 0.1994]))
z 2 (array([0.1181, 0.0999]), array([0.2138, 0.1702]))
z 3 (array([0.1383, 0.1666]), array([0.1454, 0.1698]))
b 1 (array([0.3169, 0.2313]), array([0.3888, 0.2415]))
b 2 (array([0.297 , 0.1814]), array([0.3942, 0.1882]))
b 3 (array([0.6498, 0.2117]), array([0.5598, 0.3668]))
zb 1 (array([0.1712, 0.1859]), array([0.2177, 0.2121]))
zb 2 (array([0.2806, 0.179 ]), array([0.3277, 0.1848]))
asinhz 1 (array([0.1653, 0.1935]), array([0.2233, 0.2033]))
asinhz 2 (array([0.114 , 0.1045]), array([0.2074, 0.1821]))
asinhz 3 (array([0.1467, 0.175 ]), array([0.1578, 0.1861]))
asinhzb 1 (array([0.1735, 0.1857]), array([0.2185, 0.2118]))
asinhzb 2 (array([0.2742, 0.1757]), array([0.3003, 0.164 ]))
archive 0.1142692841761884 0.07480713777954118
archive 0.09905871543769357 0.03599620532488127
archive 0.09536074051912526 0.03688895222459436
archive 0.06110219223673273 0.041917207664459784
archive 0.15364577404476473 0.0628166341537022
corr [[-0.4124 -0.4548  0.4369  0.6798  0.4342  0.466  -0.1229 -0.6225  1.     -0.3967 -0.526 ]
 [ 0.7087  0.048  -0.5931 -0.1675  0.0689  0.1417  0.2311  0.8033 -0.3967  1.      0.7374]
 [ 0.6417 -0.0934 -0.485  -0.5087 -0.2407  0.1156  0.4125  0.7069 -0.526   0.7374  1.    ]]
```

### Turn 5 — `execute_in_remote_docker` (`30886 ms` LLM)
- **Rationale**: Use calibration residuals to identify how panel C maps the latent coordinates, then compare its contribution to each head over the official and auxiliary splits without changing the frozen basis.
- **Alternatives Weighed**: `Tune only the archived A/B kernel` (rejected: Both heads fail the official split despite near-noise-floor calibration on panels A and B; panel C has independently measurable structure worth testing first.)

```bash
cat >/tmp/panelc.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
p=s['panel_c']; u=p['u']; v1=p['v1']; v2=p['v2']; w1=old.fit(old.AF(u),v1); c1=old.AF(z)@w1
print('C1 AF',w1);print('C2 quadratic',old.fit(poly(u,2),v2));print('C2 quad using v1',old.fit(poly(v1,2),v2));print('C2 resid using v1',np.std(poly(v1,2)@old.fit(poly(v1,2),v2)-v2))
c2=poly(z,2)@old.fit(poly(u,2),v2)
from scipy.linalg import solve
from scipy.spatial.distance import cdist
P={'z':z,'b':b,'c1':c1,'c2':c2,'z+b':np.c_[z,b],'z+c1':np.c_[z,c1],'z+c2':np.c_[z,c2],'b+c2':np.c_[b,c2],'z+b+c1':np.c_[z,b,c1],'z+b+c2':np.c_[z,b,c2],'z+b+c1+c2':np.c_[z,b,c1,c2]}
for key,A in P.items():
 print(key,'linear',assess(D(A)),'quadratic',assess(poly(A,2)) if A.shape[1]<=8 else '')
 if A.shape[1]==4: print(' cubic',assess(poly(A,3)))
 res=[]
 for trf in [lambda a:a,lambda a:np.arcsinh(3*a)]:
  for standard in [False,True]:
   for deg in [1,2,3]:
    for decay in [.05,.3,1.5]:
     for ridge in [1e-6,.002,.02]:
      scores=[]
      for i,j in sl:
       x=trf(A);x=(x-x[i].mean(0))/(x[i].std(0) if standard else .5);d=x.shape[1]; k=(1+x@x.T/d)**deg+np.exp(-decay*cdist(x,x,'sqeuclidean')/d)
       e=k[np.ix_(j,i)]@solve(k[np.ix_(i,i)]+ridge*np.eye(len(i)),T[i],assume_a='pos')-T[j];scores.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()])
      scores=np.array(scores);res.append((scores.mean(0),scores[0],scores.max(0),(trf.__code__.co_firstlineno,standard,deg,decay,ridge)))
 for h in [0,1]:
  for r in sorted(res,key=lambda r:r[0][h])[:2]: print(' head',h,r)
q=least_squares(lambda q:(-np.expm1(-s['eval_times'][:,None]*q)@old.fit(-np.expm1(-s['eval_times'][:,None]*q),t['traj'].T)-t['traj'].T).ravel(),[1.8,.42],bounds=([1,.15],[3,.85]));print('rates',q.x,'svd',np.linalg.svd(t['traj'],compute_uv=False)); print('corr',np.corrcoef(np.c_[z[:n],b[:n],c1[:n],c2[:n],T].T)[-3:])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panelc.py
```
- **Container Exit Code**: `0` (`1875 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
False, 3, 0.05, 0.002))
 head 0 (array([0.1039, 0.0527]), array([0.0955, 0.0794]), array([0.1481, 0.0794]), (12, False, 3, 0.3, 0.002))
 head 1 (array([0.1027, 0.0522]), array([0.0954, 0.0782]), array([0.1478, 0.0782]), (12, False, 3, 0.05, 0.002))
 head 1 (array([0.1039, 0.0527]), array([0.0955, 0.0794]), array([0.1481, 0.0794]), (12, False, 3, 0.3, 0.002))
z+b+c1 linear (array([0.1592, 0.1158]), array([0.1794, 0.1945])) quadratic 
 head 0 (array([0.1186, 0.1133]), array([0.1052, 0.0982]), array([0.1477, 0.1381]), (12, False, 3, 0.3, 0.002))
 head 0 (array([0.1186, 0.1139]), array([0.1065, 0.0995]), array([0.1481, 0.1385]), (12, False, 3, 0.3, 1e-06))
 head 1 (array([0.1293, 0.0965]), array([0.1102, 0.0837]), array([0.1456, 0.1388]), (12, True, 3, 0.05, 0.02))
 head 1 (array([0.122 , 0.0969]), array([0.1126, 0.0864]), array([0.1436, 0.1374]), (12, True, 3, 0.05, 0.002))
z+b+c2 linear (array([0.1885, 0.1859]), array([0.2262, 0.1879])) quadratic 
 head 0 (array([0.0938, 0.0626]), array([0.0863, 0.0803]), array([0.1616, 0.0865]), (12, False, 3, 0.3, 0.002))
 head 0 (array([0.0941, 0.0614]), array([0.0865, 0.0775]), array([0.1639, 0.0834]), (12, False, 3, 0.05, 0.002))
 head 1 (array([0.0972, 0.0511]), array([0.0876, 0.0737]), array([0.163 , 0.0737]), (12, True, 3, 0.05, 1e-06))
 head 1 (array([0.0951, 0.0512]), array([0.0866, 0.0735]), array([0.1591, 0.0735]), (12, True, 3, 0.05, 0.002))
z+b+c1+c2 linear (array([0.1733, 0.123 ]), array([0.2005, 0.17  ])) quadratic 
 head 0 (array([0.1192, 0.1121]), array([0.0961, 0.1011]), array([0.1479, 0.1421]), (12, False, 3, 0.3, 0.02))
 head 0 (array([0.1199, 0.1139]), array([0.1002, 0.1061]), array([0.1578, 0.1421]), (12, False, 3, 0.05, 0.002))
 head 1 (array([0.1223, 0.1023]), array([0.0938, 0.0997]), array([0.145 , 0.1456]), (12, True, 3, 0.05, 0.02))
 head 1 (array([0.1217, 0.103 ]), array([0.0959, 0.1023]), array([0.1502, 0.1467]), (12, True, 3, 0.05, 0.002))
rates [3.     0.5876] svd [16.0474  0.2722  0.1842  0.0498  0.0453  0.0416  0.0367  0.0309]
corr [[-0.4124 -0.4548  0.4369  0.6798  0.4342  0.466  -0.1229 -0.6225  0.0626 -0.1301 -0.084  -0.2001  0.4794  0.7618 -0.4989  0.1725  1.
  -0.3967 -0.526 ]
 [ 0.7087  0.048  -0.5931 -0.1675  0.0689  0.1417  0.2311  0.8033 -0.0206 -0.0211  0.0428 -0.1699 -0.6651 -0.3504  0.3107  0.3761 -0.3967
   1.      0.7374]
 [ 0.6417 -0.0934 -0.485  -0.5087 -0.2407  0.1156  0.4125  0.7069  0.2375  0.1124  0.1648 -0.1781 -0.5171 -0.2773  0.6558  0.208  -0.526
   0.7374  1.    ]]
```

### Turn 6 — `execute_in_remote_docker` (`18375 ms` LLM)
- **Rationale**: Profile the scalar C2 coordinate function against the independent calibration panel and inspect the frozen basis and prior solver for candidate head compositions.
- **Alternatives Weighed**: `Concatenate every calibrated panel into a kernel` (rejected: Adding all panels did not produce either required accuracy margin; the nearly proportional C2 linear and quadratic coefficients suggest a more compact shared transform.)

```bash
cat /holdout_task/_frozen_nodes/node_01.py; cat /tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat >/tmp/grammar.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
p=s['panel_c'];u=p['u']; v=p['v2']
for name,h in {'quad':lambda u:u+.45*u*u,'exp':lambda u:np.expm1(.9*u)/.9,'cubic':lambda u:np.c_[u,u*u,u**3],'quartic':lambda u:np.c_[u,u*u,u**3,u**4],'sinquad':lambda u:np.c_[u,u*u,np.sin(u)],'tanhquad':lambda u:np.c_[np.tanh(u),u*u,u]}.items():
 A=D(h(u));w=old.fit(A,v);print(name,np.std(A@w-v),w)
A=lambda p:D(sum(q*u**(j+1) for j,q in enumerate(np.r_[1.,p])))
for d in (1,2,3,4,5):
 q=least_squares(lambda p:(A(p)@old.fit(A(p),v)-v).ravel(),np.zeros(d));print('C2 shared series',q.x,np.std(q.fun))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/grammar.py
```
- **Container Exit Code**: `0` (`521 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.2407 -0.1972 -0.253 ]
 [ 0.0896  0.2862 -0.0126 -0.2966]
 [-0.0371  0.1614 -0.3652  0.1459]]
cubic 0.003985131318559187 [[ 0.075  -0.1149  0.0526  0.1185]
 [-0.451   0.1556  0.1111  0.0105]
 [-0.1657 -0.2791 -0.2221 -0.298 ]
 [ 0.0933  0.325  -0.014  -0.367 ]
 [-0.0389  0.1897 -0.4157  0.1679]
 [-0.2025  0.0719  0.0511  0.0101]
 [-0.067  -0.1234 -0.0968 -0.1263]
 [ 0.0477  0.1386 -0.0063 -0.1582]
 [-0.0213  0.0812 -0.1863  0.0748]
 [ 0.0114 -0.0053 -0.001  -0.0114]
 [ 0.0098 -0.0004 -0.0019  0.0123]
 [ 0.0001 -0.0073 -0.0015  0.0102]
 [-0.0023 -0.0069  0.0025 -0.0004]]
quartic 0.00025690914648918606 [[ 0.0824 -0.1193  0.0594  0.1209]
 [-0.4439  0.1497  0.1122  0.0069]
 [-0.1552 -0.2781 -0.2261 -0.2864]
 [ 0.0986  0.3179 -0.0148 -0.3512]
 [-0.0409  0.1821 -0.4124  0.1692]
 [-0.238   0.0817  0.0601  0.0053]
 [-0.0827 -0.1476 -0.1173 -0.1504]
 [ 0.053   0.1668 -0.0065 -0.1835]
 [-0.0189  0.0971 -0.2183  0.0907]
 [-0.0006  0.0029  0.0001  0.    ]
 [-0.0009  0.0005  0.0016  0.001 ]
 [ 0.0001 -0.001   0.0003 -0.0002]
 [ 0.0009 -0.0004 -0.0001 -0.0004]
 [ 0.0302 -0.0118 -0.0076 -0.0014]
 [ 0.0102  0.0184  0.0132  0.0173]
 [-0.0068 -0.0194  0.      0.0207]
 [ 0.0009 -0.0123  0.0266 -0.0119]]
sinquad 0.003993816351762007 [[ 0.075  -0.1149  0.0526  0.1186]
 [-0.3776  0.1212  0.1049 -0.0613]
 [-0.1003 -0.2816 -0.234  -0.2193]
 [ 0.0947  0.278  -0.0226 -0.3016]
 [-0.0535  0.1446 -0.3997  0.1661]
 [-0.2023  0.0719  0.0511  0.0098]
 [-0.0671 -0.1233 -0.0968 -0.1262]
 [ 0.0477  0.1385 -0.0063 -0.1582]
 [-0.0214  0.0812 -0.1863  0.0748]
 [-0.0737  0.0344  0.0061  0.072 ]
 [-0.066   0.0025  0.0118 -0.0788]
 [-0.0015  0.0472  0.0083 -0.0652]
 [ 0.0146  0.0454 -0.016   0.0017]]
tanhquad 0.004045910808077554 [[ 0.0752 -0.1148  0.0527  0.1187]
 [-0.0555  0.0259  0.0034  0.0473]
 [-0.0595  0.0014  0.0073 -0.055 ]
 [-0.0032  0.0351  0.002  -0.0435]
 [ 0.0093  0.0362 -0.0116  0.001 ]
 [-0.2016  0.0716  0.0509  0.0085]
 [-0.0672 -0.1232 -0.0966 -0.1261]
 [ 0.0476  0.1383 -0.0063 -0.1582]
 [-0.0218  0.0813 -0.1864  0.0747]
 [-0.3974  0.1303  0.1073 -0.0358]
 [-0.1099 -0.2806 -0.2296 -0.2435]
 [ 0.0957  0.2906 -0.0173 -0.3225]
 [-0.0483  0.1551 -0.4043  0.1667]]
C2 shared series [0.4437] 0.0052687850648155
C2 shared series [ 0.4396 -0.0108] 0.005099178063242445
C2 shared series [ 0.5291 -0.0005 -0.0636] 0.00048611577368056223
C2 shared series [ 0.5291 -0.001  -0.0636  0.0003] 0.00048592822714742847
C2 shared series [ 0.5466 -0.0002 -0.097   0.0001  0.0156] 9.740077756301531e-05
```

### Turn 7 — `execute_in_remote_docker` (`34144 ms` LLM)
- **Rationale**: Use the panel to recover a compact coordinate transform, then compare head mechanisms by held-out error rather than calibration fit alone.
- **ATMS NOGOOD DECLARED**: Refuted `Panel C2 is adequately described by a shared linear-plus-quadratic coordinate transform at the calibration noise floor.` | Witness: `The shared quadratic fit has residual SD 0.005269; adding even fourth- and sixth-order terms lowers it to 0.0000974.` | Blanket Diagnosis: `The omitted even saturation terms enter upstream of the predictive heads, so regression error alone does not identify the head mechanism.` | Sublated Pivot: `Profile compact even saturation functions on the independent panel and fit nonlinear head projections on training rows only.`

```bash
cat >/tmp/heads.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.special import erf
from scipy.linalg import solve
p=s['panel_c'];u=p['u']; v=p['v2'];
forms=[lambda x,q:x+q[0]*x*x/(1+q[1]*x*x),lambda x,q:x+q[0]*np.log1p(q[1]*x*x),lambda x,q:x+q[0]*np.tanh(q[1]*x*x),lambda x,q:x+q[0]*(np.sqrt(1+q[1]*x*x)-1),lambda x,q:x+q[0]*(1-np.exp(-q[1]*x*x))]
rr=[]
for fn in forms:
 def design(x,q):return D(fn(x,q))
 q=least_squares(lambda q:(design(u,q)@old.fit(design(u,q),v)-v).ravel(),[.55,.2],bounds=([0,.001],[10,5]),max_nfev=400); print('C2',q.x,np.std(q.fun));rr.append((q.cost,fn,q.x))
_,fn,q=min(rr,key=lambda r:r[0]); c= D(fn(z,q))@old.fit(D(fn(u,q)),v)
c1=D(z*np.roll(z,-1,axis=1))@old.fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
print('y quantiles',np.quantile(T[:,0],np.linspace(0,1,11))); print('trajectory endpoint', np.quantile(t['traj'][:,-1],np.linspace(0,1,11)))
np.savez('/tmp/feat.npz',z=z,b=b,c=c,c1=c)
# Profile one nonlinear index plus a low-dimensional linear background. Optimize on training rows only.
inputs={'z':z,'b':b,'c':c,'c1':c1,'zb':np.c_[z,b],'bc':np.c_[b,c],'zc':np.c_[z,c],'zbc':np.c_[z,b,c],'zbc1':np.c_[z,b,c1],'bc1':np.c_[b,c1],'all':np.c_[z,b,c,c1]}
Y=np.c_[T[:,0],t['traj'][:,-1]]
acts={'tanh':np.tanh,'sin':np.sin,'erf':erf,'rational':lambda x:x/np.sqrt(1+x*x),'quadratic':lambda x:x*x,'cubic':lambda x:x**3,'radial':lambda x:np.exp(-x*x)}
res=[]
for name,X in inputs.items():
 X=D((X-X[:n].mean(0))/X[:n].std(0)); d=X.shape[1];
 for typ,g in acts.items():
  for back in ['0','z','self','bc']:
   B=np.ones((len(X),1)) if back=='0' else D(inputs[name if back=='self' else back]); scores=[]; rms=[]
   for i,j in sl[:2]:
    vals=[]; rng=np.random.default_rng(23)
    for h in range(2):
     w0=old.fit(X[i],Y[i,h]); w0[0]=0;w0/=max(.01,np.linalg.norm(w0));
     def f(p,ix):return np.c_[B[ix],g(X[ix]@p)]
     def fun(p):
      A=f(p,i); return A@old.fit(A,Y[i,h])-Y[i,h]
     fits=[least_squares(fun,a,max_nfev=65) for a in [w0,2*w0,rng.normal(0,.4,d)]]; fit=min(fits,key=lambda r:r.cost); vals.append(np.abs(f(fit.x,j)@old.fit(f(fit.x,i),Y[i,h])-Y[j,h]).mean()); rms.append(np.std(fit.fun))
    scores.append(vals)
   res.append((np.array(scores),np.array(rms).reshape(-1,2),name,typ,back))
for h in range(2):
 for r in sorted(res,key=lambda r:r[0][:,h].mean())[:16]: print(h,r)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/heads.py
```
- **Container Exit Code**: `0` (`93907 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 0.1511]]), 'zbc', 'tanh', 'z')
0 (array([[0.1886, 0.1729],
       [0.1677, 0.2622]]), array([[0.1778, 0.1671],
       [0.1464, 0.1447]]), 'zbc', 'sin', 'z')
0 (array([[0.134 , 0.2733],
       [0.2245, 0.2625]]), array([[0.168 , 0.1597],
       [0.1543, 0.186 ]]), 'zbc1', 'quadratic', 'z')
0 (array([[0.1705, 0.2443],
       [0.1908, 0.2784]]), array([[0.1724, 0.2031],
       [0.1581, 0.1724]]), 'all', 'quadratic', '0')
1 (array([[0.2565, 0.1601],
       [0.2022, 0.2118]]), array([[0.1254, 0.133 ],
       [0.0974, 0.1095]]), 'zbc', 'quadratic', 'self')
1 (array([[0.282 , 0.1876],
       [0.2574, 0.2012]]), array([[0.0775, 0.0812],
       [0.0678, 0.0965]]), 'all', 'sin', 'self')
1 (array([[0.2571, 0.1733],
       [0.2122, 0.2452]]), array([[0.1341, 0.1592],
       [0.1197, 0.1232]]), 'zbc1', 'erf', 'bc')
1 (array([[0.2174, 0.1749],
       [0.2229, 0.2453]]), array([[0.1381, 0.1586],
       [0.1175, 0.1233]]), 'zbc1', 'sin', 'bc')
1 (array([[0.2658, 0.1742],
       [0.2132, 0.2461]]), array([[0.1334, 0.1585],
       [0.1204, 0.1231]]), 'zbc1', 'tanh', 'bc')
1 (array([[0.2932, 0.1754],
       [0.2134, 0.2469]]), array([[0.1321, 0.1588],
       [0.1217, 0.1231]]), 'zbc1', 'rational', 'bc')
1 (array([[0.2163, 0.1729],
       [0.2051, 0.2562]]), array([[0.1907, 0.1671],
       [0.1111, 0.1539]]), 'zbc', 'quadratic', 'z')
1 (array([[0.1868, 0.2187],
       [0.1668, 0.2108]]), array([[0.178 , 0.159 ],
       [0.1459, 0.1514]]), 'zbc', 'rational', 'z')
1 (array([[0.1706, 0.181 ],
       [0.259 , 0.2492]]), array([[0.1776, 0.1632],
       [0.1549, 0.1351]]), 'zb', 'sin', 'bc')
1 (array([[0.1738, 0.1811],
       [0.2362, 0.2495]]), array([[0.177 , 0.1634],
       [0.1537, 0.1351]]), 'zb', 'rational', 'bc')
1 (array([[0.5127, 0.1968],
       [0.2317, 0.2379]]), array([[0.149 , 0.1464],
       [0.1273, 0.1361]]), 'zbc1', 'cubic', 'bc')
1 (array([[0.1886, 0.1729],
       [0.1677, 0.2622]]), array([[0.1778, 0.1671],
       [0.1464, 0.1447]]), 'zbc', 'sin', 'z')
1 (array([[0.1348, 0.1816],
       [0.1984, 0.2553]]), array([[0.1432, 0.1687],
       [0.1161, 0.15  ]]), 'zc', 'quadratic', 'self')
1 (array([[0.1883, 0.1803],
       [0.258 , 0.2572]]), array([[0.2398, 0.2276],
       [0.212 , 0.2055]]), 'zbc', 'rational', '0')
1 (array([[0.1881, 0.1838],
       [0.2571, 0.2542]]), array([[0.24  , 0.2243],
       [0.212 , 0.203 ]]), 'zbc', 'tanh', '0')
1 (array([[0.1877, 0.1869],
       [0.2569, 0.252 ]]), array([[0.2401, 0.2218],
       [0.212 , 0.2008]]), 'zbc', 'erf', '0')
```

### Turn 8 — `execute_in_remote_docker` (`32457 ms` LLM)
- **Rationale**: Probe cross-panel sums, products, and saturations for stable sparse terms, and retain the exact frozen-node contracts.

```bash
cat /tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py; cat /holdout_task/public_asset_index.json | tail -80; cat >/tmp/dict.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.special import erf
from scipy.linalg import solve
from sklearn.linear_model import ARDRegression, ElasticNet, OrthogonalMatchingPursuit
p=s['panel_c'];u=p['u'];v=p['v2']; forms={'logcosh':lambda x,q:x+q[0]*np.log(np.cosh(q[1]*x)),'xtanh':lambda x,q:x+q[0]*x*np.tanh(q[1]*x),'abs':lambda x,q:x+q[0]*(np.hypot(1,q[1]*x)-1),'cos':lambda x,q:x+q[0]*(1-np.cos(q[1]*x)),'rat':lambda x,q:x+q[0]*x*x/(1+q[1]*x*x)}; rs=[]
for name,fn in forms.items():
 h=lambda a,q:D(fn(a,q)); o=least_squares(lambda q:(h(u,q)@old.fit(h(u,q),v)-v).ravel(),[1,1],bounds=([.01,.01],[20,5]));print(name,o.x,np.std(o.fun));rs.append((o.cost,fn,o.x))
_,fn,q=min(rs,key=lambda r:r[0]); c=D(fn(z,q))@old.fit(D(fn(u,q)),v); c1=D(z*np.roll(z,-1,axis=1))@old.fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
np.savez('/tmp/feat2.npz',z=z,b=b,c=c,c1=c1)
Y=np.c_[t['y'],t['traj'][:,-1]]; dic={}; groups=[z,b,c1,c]; labels=['z','b','p','c'];
def add(name,a):
 for j,col in enumerate(a.T):dic[name+str(j)]=col
for label,g in zip(labels,groups):add(label,g)
for i,g in enumerate(groups):
 for j,h in enumerate(groups[:i+1]):
  label=labels[i]+labels[j]
  add(label+'*',(g[:,:,None]*h[:,None,:]).reshape(len(z),-1))
  for k in [.5,1,2]:
   add(label+'+'+str(k),(g[:,:,None]+k*h[:,None,:]).reshape(len(z),-1));add(label+'-'+str(k),(g[:,:,None]-k*h[:,None,:]).reshape(len(z),-1))
for k,x in list(dic.items()):
 std=x.std();
 if std<1e-8:continue
 for l,fn in [('sin',lambda a:np.sin(1.5*a)),('tanh',np.tanh),('asin',np.arcsinh),('rat',lambda a:a/(1+.5*a*a)),('sq',np.square),('abs',abs),('cube',lambda a:a**3),('soft',lambda a:np.maximum(abs(a)-.45,0)**1.5*np.sign(a))]:dic[l+k]=fn(x/std*.65)
X=np.column_stack(list(dic.values())); keep=X.std(0)>1e-10; X=X[:,keep]; keys=np.array(list(dic))[keep];
# Rank and prune duplicate columns before sparse feature selection.
X=(X-X[:n].mean(0))/X[:n].std(0); _,uq=np.unique(np.round(X[:n]*np.sign(X[0]),5).T,axis=0,return_index=True);X=X[:,uq];keys=keys[uq]; print('dict',X.shape)
for head in range(2):
 for i,j in sl[:2]:
  y=Y[i,head]; order=np.argsort(abs(X[i].T@(y-y.mean()))/(X[i]**2).sum(0)**.5)[::-1]; print('top',head,[(keys[k],round(np.corrcoef(X[i,k],y)[0,1],3)) for k in order[:15]])
  for a in [.002,.005,.01,.02,.04,.07]:
   m=ElasticNet(alpha=a,l1_ratio=.96,max_iter=5000,tol=1e-5).fit(X[i],y); inds=np.argsort(abs(m.coef_))[-12:][::-1]; print(head,a,np.mean(abs(m.predict(X[j])-Y[j,head])),np.mean(abs(m.predict(X[i])-y)),[(keys[k],round(m.coef_[k],3)) for k in inds if m.coef_[k]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/dict.py
```
- **Container Exit Code**: `0` (`10061 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7), ('softbb*1', -0.037)]
1 0.07 0.23982073297107825 0.12677819502798515 [('asincb+111', 0.157), ('sinbz+18', 0.139), ('bb*1', -0.134), ('sincp-0.511', 0.096), ('cz-0.510', 0.077), ('cb+111', 0.058), ('tanhcb+111', 0.048), ('sinbz-0.56', 0.046), ('sinbb*11', 0.042), ('sinbz*8', 0.04), ('asincz-0.510', 0.036), ('softzz-0.58', -0.032)]
top 1 [('asincb+111', 0.895), ('tanhcb+111', 0.895), ('cb+111', 0.89), ('ratcb+111', 0.887), ('asincb+211', 0.878), ('tanhcb+211', 0.874), ('cb+211', 0.874), ('sincb+111', 0.871), ('bz+18', 0.864), ('ratcb+211', 0.863), ('asinbz+18', 0.863), ('tanhbz+18', 0.856), ('cc-12', -0.849), ('ratbz+18', 0.846), ('tanhcb+0.511', 0.844)]
1 0.002 0.14030630583496126 0.010905464927579597 [('ratbz+18', 0.147), ('cb+111', 0.12), ('sqbb-11', 0.099), ('cz-0.510', 0.079), ('sincz+0.58', 0.077), ('sqpz-115', -0.066), ('cubebb*1', -0.065), ('ratcb+0.58', 0.062), ('absbb-111', -0.057), ('tanhcb+111', 0.051), ('softpz+20', 0.05), ('sinbz-0.511', 0.044)]
1 0.005 0.16596414948108945 0.023774963189417447 [('ratbz+18', 0.176), ('asincb+111', 0.121), ('cb+111', 0.106), ('cubebb*1', -0.086), ('cubepz-212', -0.084), ('sinpz-0.52', 0.075), ('bz*8', 0.065), ('tanhcb+111', 0.061), ('ratbz*10', -0.061), ('ratcb+0.58', 0.059), ('cz-0.510', 0.054), ('absbb-111', -0.052)]
1 0.01 0.17578064319969347 0.036711620278274064 [('ratbz+18', 0.194), ('asincb+111', 0.171), ('cb+111', 0.117), ('cubepz-212', -0.093), ('cubebb*1', -0.093), ('sinpz-0.52', 0.077), ('cz-0.510', 0.076), ('bz*8', 0.063), ('sincb*15', 0.051), ('sqbb-11', 0.051), ('ratcb+0.58', 0.049), ('sinpb*6', 0.034)]
1 0.02 0.1922844639172897 0.05881299300468982 [('asincb+111', 0.19), ('ratbz+18', 0.186), ('cb+111', 0.139), ('cubepz-212', -0.095), ('sinpz-0.52', 0.091), ('cubebb*1', -0.073), ('sinbz*10', -0.05), ('cz-0.510', 0.048), ('bz*8', 0.044), ('sincb*15', 0.044), ('sinpb*6', 0.04), ('ratcb+0.58', 0.039)]
1 0.04 0.22243562808760936 0.09590899140707435 [('asincb+111', 0.187), ('ratbz+18', 0.179), ('cb+111', 0.123), ('sinpz-0.52', 0.12), ('cubepz-212', -0.093), ('asinbz*8', 0.042), ('ratcb+0.58', 0.041), ('sinbz*10', -0.04), ('cz-0.510', 0.038), ('sinpb*6', 0.037), ('cubebb*1', -0.03), ('softbb*1', -0.023)]
1 0.07 0.2569511053493694 0.13840207458130982 [('asincb+111', 0.195), ('ratbz+18', 0.134), ('sinpz-0.52', 0.134), ('cb+111', 0.102), ('cubepz-212', -0.08), ('tanhcb+111', 0.071), ('asinbz*8', 0.066), ('bb*1', -0.04), ('ratcb+0.58', 0.031), ('tanhbz+18', 0.031), ('sinbz*10', -0.023), ('sincz+10', 0.02)]
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_coordinate_descent.py:697: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.041e-04, tolerance: 5.291e-04
  model = cd_fast.enet_coordinate_descent(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_coordinate_descent.py:697: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 6.533e-04, tolerance: 5.291e-04
  model = cd_fast.enet_coordinate_descent(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_coordinate_descent.py:697: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.711e-04, tolerance: 4.622e-04
  model = cd_fast.enet_coordinate_descent(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_coordinate_descent.py:697: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 6.273e-04, tolerance: 3.701e-04
  model = cd_fast.enet_coordinate_descent(
```

### Turn 9 — `execute_in_remote_docker` (`48831 ms` LLM)
- **Rationale**: Save runnable nodes now, then check the calibration inverse and compare head-specific kernel geometry across folds. Fit kernel weights only on each fold’s training rows.
- **Alternatives Weighed**: `Deploy the sparse interaction dictionary` (rejected: Its training error dropped substantially without corresponding improvement on held-out trajectory endpoints.)

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py').read_text()
base=p[:p.index('def times')]
Path('/holdout_task/node_00.py').write_text(base+'''
def features(study):
    z,b=parts(study)
    p=study['panel_c'];u=p['u'];v=p['v2']
    def H(x,q):
        return D(x+q[0]*(np.logaddexp(q[1]*x,-q[1]*x)-np.log(2.)))
    def err(q):
        f=H(u,q);return (f@fit(f,v)-v).ravel()
    q=least_squares(err,[1.,1.],bounds=([.01,.01],[20.,5.])).x
    c=H(z,q)@fit(H(u,q),v)
    cz=D(z*np.roll(z,-1,axis=1))@fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
    return z,b,c,cz

def kernel(x):
    dot=x@x.T/x.shape[1]
    dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0.)
    return (1+dot)**3+np.exp(-.3*dist)

def run_node(inputs:dict)->dict:
    s=inputs['__input__']; z,b,c,p=features(s); y=s['train']['y']; n=len(y)
    f=np.arcsinh(2*np.c_[z,b,c]); k=kernel(f)
    yhat=k[n:,:n]@np.linalg.solve(k[:n,:n]+.002*np.eye(n),y)
    return {'y_hat':yhat,'z':z,'b':b,'c':c,'p':p}
''')
PY
cat >/holdout_task/node_02.py <<'PY'
import numpy as np


def run_node(inputs:dict)->dict:
    a=inputs['node_00']; t=inputs['node_01']; modes=t['modes']; n=len(modes)
    x=np.arcsinh(2*np.c_[a['b'],a['c']]); dot=x@x.T/x.shape[1]
    d=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0.)
    k=(1+dot)**3+np.exp(-.05*d)
    return {'state_pred':(k[n:,:n]@np.linalg.solve(k[:n,:n]+.002*np.eye(n),modes))@t['basis'].T}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/test_dag_candidate.py || true
cat >/tmp/gpdiag.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task');import node_00 as m
z,b,c,p=m.features(s)
a=s['panel_a']; af=m.AF(a['u']); keep=m.cal(s,a['u'])[2]<2; aw=m.fit(af[keep],a['v'][keep]); print('A coef',aw,'noise',np.std(af[keep]@aw-a['v'][keep]),'ranges',[(g.min(0),g.max(0)) for g in (z,b,c,p)], flush=True)
print('inverse residual',np.max(abs(m.AF(z)@aw-np.r_[t['x'],s['test']['x']])),flush=True)
from scipy.spatial.distance import cdist
U=np.linalg.svd(t['traj'],full_matrices=False)[2]; YY=np.c_[T[:,0],t['traj']@U[:2].T]

def gp(f,Y,i,j,kind):
 f=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-9); f=f[:n];Y=(Y-Y[i].mean(0))/Y[i].std(0)
 I=np.eye(len(i)); d=(f[:,None,:]-f[None,:,:])**2; dt=d[j][:,i]; d=d[i][:,i]; aa=f[i]; zz=f[j]; N=aa.shape[1]
 def calc(q,dm,L,R):
  w=np.exp(q[:N]); r2=dm@w; scale=np.exp(q[N]); dot=(L*w)@R.T
  if kind==0: k=scale*np.exp(-.5*r2); dk=-.5*k[:,:,None]*dm*w
  else:
   r=np.sqrt(5*r2); er=np.exp(-r); k=scale*(1+r+r*r/3)*er;dk=-5/6*scale*((1+r)*er)[:,:,None]*dm*w
  poly=np.exp(q[N+1])*(1+dot)**2; lin=np.exp(q[N+2])*np.ones_like(k)
  dk+=2*np.exp(q[N+1])*(1+dot)[:,:,None]*(L[:,None,:]*R[None,:,:])*w
  return k+poly+lin,(dk,k,poly,lin)
 def fun(q):
  k,(dk,k0,p,l)=calc(q,d,aa,aa); noise=np.exp(q[-1]); cf=cho_factor(k+noise*I); Yi=Y[i]; v=cho_solve(cf,Yi); h=Y.shape[1]; g=.5*(h*cho_solve(cf,I)-v@v.T)
  return .5*np.sum(Yi*v)+h*np.log(np.diag(cf[0])).sum(),np.r_[np.einsum('ij,ijk->k',g,dk),np.sum(g*k0),np.sum(g*p),np.sum(g*l),noise*np.trace(g)]
 q=np.r_[np.full(N,-2.3),0.,-2.,0.,-6.]; rr=minimize(fun,q,jac=True,method='L-BFGS-B',bounds=[(-13,4)]*N+[(-12,8),(-15,6),(-12,7),(-14,-1)],options={'maxiter':130}); q=rr.x; k=calc(q,d,aa,aa)[0]; w=cho_solve(cho_factor(k+np.exp(q[-1])*I),Y[i]); pred=calc(q,dt,zz,aa)[0]@w; return pred,np.exp(q[:N]),rr.fun

Fs={'z':z,'zb':np.c_[z,b],'zc':np.c_[z,c],'bc':np.c_[b,c],'zbc':np.c_[z,b,c],'zbcp':np.c_[z,b,c,p],'z0b0c0':np.c_[z,np.sign(z)*np.maximum(abs(z)-.35,0)**1.5,z*z]}
for name,F in Fs.items():
 for kind in [0,1]:
  vals=[]
  for i,j in sl:
   a,w,v=gp(F,YY[:,:1],i,j,kind); o,w2,v2=gp(F,YY[:,1:],i,j,kind); e=a[:,0]*YY[i,0].std()+YY[i,0].mean()-YY[j,0]; E=(o*YY[i,1:].std(0)+YY[i,1:].mean(0))@U[:2]-t['traj'][j]; vals.append([abs(e).mean(),abs(E).mean()]);
   if len(vals)==1: print('ARD',name,kind,w.round(3),w2.round(3),flush=True)
  print('GP',name,kind,np.round(vals,5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/gpdiag.py
```
- **Container Exit Code**: `0` (`5879 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py', '/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
07]]
ARD zc 1 [0.699 0.116 0.474 1.029 2.324 0.467 0.014 0.174] [0.227 0.01  0.137 0.157 0.074 0.069 0.    0.   ]
GP zc 1 [[0.1294 0.1084]
 [0.1654 0.164 ]
 [0.1223 0.1648]
 [0.1105 0.1246]
 [0.128  0.1267]]
ARD bc 0 [0.098 0.027 0.    0.074 0.577 0.52  0.207 0.051] [0.341 0.627 0.221 0.504 0.055 0.    0.049 0.   ]
GP bc 0 [[0.1143 0.1921]
 [0.1505 0.1153]
 [0.1648 0.1224]
 [0.0923 0.1358]
 [0.1548 0.0956]]
ARD bc 1 [0.035 0.008 0.    0.028 0.15  0.14  0.07  0.023] [0.081 0.106 0.069 0.16  0.01  0.002 0.015 0.006]
GP bc 1 [[0.0944 0.1468]
 [0.1596 0.1211]
 [0.1726 0.1113]
 [0.1074 0.1364]
 [0.1783 0.1072]]
ARD zbc 0 [0.284 0.02  0.076 0.11  0.022 0.    0.    0.    0.    0.038 0.    0.006] [0.273 0.    0.081 0.    0.2   0.163 0.123 0.146 0.    0.    0.118 0.   ]
GP zbc 0 [[0.1106 0.1063]
 [0.153  0.1208]
 [0.1369 0.1653]
 [0.0763 0.1333]
 [0.1111 0.0893]]
ARD zbc 1 [0.101 0.009 0.03  0.045 0.011 0.    0.    0.001 0.023 0.027 0.    0.   ] [0.107 0.    0.029 0.    0.075 0.071 0.042 0.065 0.    0.    0.047 0.   ]
GP zbc 1 [[0.0859 0.1068]
 [0.1493 0.1182]
 [0.1288 0.1572]
 [0.1094 0.1313]
 [0.1661 0.0869]]
ARD zbcp 0 [0.    0.03  0.105 0.089 0.032 0.019 0.    0.    0.67  0.433 0.    0.    0.    0.    0.029 0.073] [0.531 0.    0.412 0.    0.109 0.061 0.108 0.    0.    0.    0.203 0.    0.    0.    0.    0.202]
GP zbcp 0 [[0.0826 0.0648]
 [0.1239 0.14  ]
 [0.1374 0.1622]
 [0.1218 0.1211]
 [0.1631 0.1176]]
ARD zbcp 1 [0.    0.014 0.053 0.041 0.012 0.012 0.    0.    0.237 0.135 0.    0.    0.002 0.    0.006 0.035] [0.266 0.    0.204 0.    0.048 0.028 0.031 0.    0.    0.    0.089 0.    0.    0.    0.    0.133]
GP zbcp 1 [[0.0731 0.0694]
 [0.136  0.14  ]
 [0.1238 0.1575]
 [0.0954 0.1231]
 [0.1807 0.1184]]
ARD z0b0c0 0 [0.091 0.041 0.057 0.064 0.018 0.008 0.    0.02  0.013 0.001 0.024 0.03 ] [0.385 0.024 0.152 0.2   0.155 0.101 0.    0.188 0.096 0.012 0.116 0.022]
GP z0b0c0 0 [[0.1109 0.0813]
 [0.1001 0.0847]
 [0.1018 0.0728]
 [0.105  0.0805]
 [0.1746 0.1151]]
ARD z0b0c0 1 [0.091 0.041 0.057 0.064 0.018 0.008 0.    0.02  0.013 0.001 0.024 0.03 ] [0.455 0.027 0.186 0.229 0.174 0.102 0.    0.21  0.107 0.014 0.128 0.027]
GP z0b0c0 1 [[0.1109 0.083 ]
 [0.1351 0.0837]
 [0.0997 0.0706]
 [0.1123 0.0821]
 [0.1746 0.115 ]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9a83d1e3739331b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9a83d1e3 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`34200 ms` LLM)
- **Rationale**: The panel A inverse reconstructs the measured inputs to numerical precision. Focus the remaining comparison on head geometry, measuring both official and auxiliary errors before changing the saved nodes.
- **Alternatives Weighed**: `Deploy the fitted ARD kernels` (rejected: The best official-split gains did not carry over to the auxiliary splits.)

```bash
cat >/tmp/ratchet.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
sys.path.insert(0,'/holdout_task');import node_00 as m
z,b,c,p=m.features(s)
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from itertools import product
from scipy.special import erf
from pathlib import Path
print('basis residual',abs(t['traj']-T[:,1:]@f.T).mean(), 'ranks',np.linalg.svd(t['traj'],compute_uv=False),flush=True)
# Coordinates before and after the panel mixing, and candidate paired sums/differences.
forms={'z':z,'b':b,'sym':np.c_[.3*z+b,.3*z-b],'sep':np.c_[.3*z,b],'zbc':np.c_[z,b,c],'zbcp':np.c_[z,b,c,p],'can':np.c_[z,np.sign(z)*np.maximum(abs(z)-.35,0)**1.5,z*z],'cz':np.c_[z,c],'zbp':np.c_[z,b,p]}
def grids(forms):
 for nm,F in forms.items():
  for tf in [0,1,3]:
   X=(np.arcsinh(tf*F)/tf if tf else F); x=X/F.std(); dot=x@x.T/x.shape[1]; d=cdist(x,x,'sqeuclidean')/x.shape[1]; u=abs(x[:,None,:]-x[None,:,:]);
   for decay in [.025,.15,.6,1.8,5.]:
    G=np.exp(-decay*d); g=np.exp(-decay*u*u); a=g.mean(2); pair=(a*a-(g*g).mean(2)/x.shape[1])*x.shape[1]/(x.shape[1]-1)
    for family,H in enumerate([G,(1+dot)**2+G,(1+dot)**3+G,(1+dot)**4+G,1+dot+a+.75*pair,(1+dot)**2+a+.75*pair]):
     yield (nm,tf,decay,family),H
Y=T; folds=sl; rankings=[]
for params,k in grids(forms):
 for ridge in [1e-7,1e-5,.0003,.003,.025]:
  scores=[]; preds=[]
  for i,j in folds:
   pred=k[np.ix_(j,i)]@cho_solve(cho_factor(k[np.ix_(i,i)]+ridge*np.eye(len(i))),Y[i]);e=pred-Y[j]; scores.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()]);preds.append(pred)
  rankings.append((params+(ridge,),np.array(scores),preds))
for h in (0,1):
 for weight in (0.,.25,.6):
  def score(e):return (1-weight)*(.65*e[:,h].mean()+.35*e[:,h].max())+weight*e[0,h]
  best=sorted(rankings,key=lambda r:score(r[1])); print('TOP',h,weight,[(r[0],r[1][:,h].round(5).tolist()) for r in best[:4]],flush=True)
  for number in (4,12,35,100):
   picks=[]; seen=set()
   for r in best:
    if r[0][:4] not in seen: picks.append(r);seen.add(r[0][:4]);
    if len(picks)==number:break
   E=[np.mean([r[2][k] for r in picks],axis=0)-Y[j] for k,(i,j) in enumerate(folds)]; scores=np.array([[abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()] for e in E]);print('BLEND',h,weight,number,scores[:,h].round(6),picks[0][0],flush=True)
  if weight==.25:
   Path('/tmp/params'+str(h)).write_text(repr([r[0] for r in best[:24]]))
# Assess whether the head is an additive sum of known transformed coordinate functions.
for nm,A in forms.items():
 X=D(np.c_[A,np.arcsinh(3*A),np.tanh(2*A),A*A]); print('additive',nm,assess(X),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/ratchet.py
```
- **Container Exit Code**: `0` (`2117 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8]), (('sym', 0, 0.15, 2, 1e-05), [0.07074, 0.03, 0.03561, 0.02716, 0.0573])]
BLEND 1 0.0 4 [0.0672 0.0258 0.0339 0.0278 0.0547] ('sym', 0, 0.025, 4, 1e-05)
BLEND 1 0.0 12 [0.0672 0.0255 0.0306 0.0307 0.0577] ('sym', 0, 0.025, 4, 1e-05)
BLEND 1 0.0 35 [0.0666 0.0341 0.0267 0.035  0.0623] ('sym', 0, 0.025, 4, 1e-05)
BLEND 1 0.0 100 [0.0708 0.0404 0.0354 0.036  0.0761] ('sym', 0, 0.025, 4, 1e-05)
TOP 1 0.25 [(('sym', 0, 0.025, 4, 1e-07), [0.05584, 0.03945, 0.04527, 0.03395, 0.06132]), (('sym', 0, 0.025, 4, 1e-05), [0.05845, 0.04683, 0.03813, 0.03701, 0.0576]), (('sym', 0, 0.15, 5, 1e-07), [0.0449, 0.06167, 0.0572, 0.047, 0.06469]), (('sym', 0, 0.15, 5, 1e-05), [0.04495, 0.06171, 0.05723, 0.04697, 0.06477])]
BLEND 1 0.25 4 [0.0575 0.0297 0.0262 0.0294 0.0523] ('sym', 0, 0.025, 4, 1e-07)
BLEND 1 0.25 12 [0.0628 0.0278 0.0268 0.0324 0.0574] ('sym', 0, 0.025, 4, 1e-07)
BLEND 1 0.25 35 [0.0631 0.0386 0.0266 0.0338 0.0624] ('sym', 0, 0.025, 4, 1e-07)
BLEND 1 0.25 100 [0.0689 0.0424 0.0362 0.0359 0.0763] ('sym', 0, 0.025, 4, 1e-07)
TOP 1 0.6 [(('sym', 0, 0.15, 5, 1e-07), [0.0449, 0.06167, 0.0572, 0.047, 0.06469]), (('sym', 0, 0.15, 5, 1e-05), [0.04495, 0.06171, 0.05723, 0.04697, 0.06477]), (('sym', 0, 0.15, 5, 0.0003), [0.04611, 0.06301, 0.05827, 0.04681, 0.06721]), (('sym', 0, 0.025, 4, 1e-07), [0.05584, 0.03945, 0.04527, 0.03395, 0.06132])]
BLEND 1 0.6 4 [0.0474 0.0535 0.0411 0.0381 0.0617] ('sym', 0, 0.15, 5, 1e-07)
BLEND 1 0.6 12 [0.0553 0.0383 0.026  0.0304 0.0612] ('sym', 0, 0.15, 5, 1e-07)
BLEND 1 0.6 35 [0.0611 0.0423 0.0278 0.0314 0.065 ] ('sym', 0, 0.15, 5, 1e-07)
BLEND 1 0.6 100 [0.0654 0.0453 0.0374 0.0367 0.0789] ('sym', 0, 0.15, 5, 1e-07)
additive z (array([0.2224, 0.2007]), array([0.2345, 0.1937]))
additive b (array([0.3159, 0.2341]), array([0.391 , 0.2838]))
additive sym (array([0.1517, 0.2813]), array([0.1669, 0.257 ]))
additive sep (array([0.3846, 0.3104]), array([0.3341, 0.3026]))
additive zbc (array([2.585 , 1.7591]), array([0.5472, 0.5269]))
additive zbcp (array([0.4958, 0.3491]), array([0.5184, 0.4101]))
additive can (array([0.6612, 0.5232]), array([0.6784, 0.5694]))
additive cz (array([0.2388, 0.2275]), array([0.2599, 0.2222]))
additive zbp (array([0.2728, 0.4959]), array([0.3885, 0.3527]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9a83d1e3739331b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9a83d1e3 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`62658 ms` LLM)
- **Rationale**: Test second- through fourth-order interactions in paired calibrated coordinates and score both heads over the official and eight auxiliary splits. These splits screen candidates rather than provide unbiased final error estimates. Validate the resulting files in a clean process and report the actual acceptance result.
- **Alternatives Weighed**: `Keep the existing C-panel kernel heads` (rejected: Both official MAEs exceed 0.093, while the paired-coordinate experiments improved both heads without changing the frozen nodes.)

```bash
cat >/tmp/finalfit.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from pathlib import Path
import contextlib, io
root=Path('/holdout_task')
common='''
def components(z, b, c, p, spec):
    a, family, warp, length, order, mix = spec
    if family == 0: X=np.c_[a*z+b,a*z-b]
    elif family == 1: X=np.c_[a*z,b]
    elif family == 2: X=a*z+b
    elif family == 3: X=a*z-b
    elif family == 4: X=np.c_[a*z+b,a*z-b,.24*c,.12*p]
    else: X=np.c_[a*z+b,a*z-b,a*z]
    X=X/max(np.sqrt(np.mean(X[:1]*0)+np.var(X)),1e-8)
    if warp: X=np.arcsinh(warp*X)/warp
    dot=X@X.T/X.shape[1]
    df=(X[:,None,:]-X[None,:,:])**2
    G=np.exp(-length*df)
    d=X.shape[1]
    M=G.mean(2)
    e2=(d*M*M-(G*G).mean(2))/(d-1)
    e3=((d*M)**3-3*d*d*M*(G*G).mean(2)+2*d*(G**3).mean(2))/(d*(d-1)*(d-2))
    e4=((d*M)**4-6*(d*M)**2*d*(G*G).mean(2)+3*(d*(G*G).mean(2))**2+8*d*d*M*(G**3).mean(2)-6*d*(G**4).mean(2))/(d*(d-1)*(d-2)*(d-3))
    if order == 0: return 1+dot+np.exp(-length*df.mean(2))
    if order == 1: return 1+dot+M+mix*e2
    if order == 2: return 1+dot+M+mix*e2+mix**2*e3
    if order == 3: return 1+dot+M+mix*e2+mix**2*e3+mix**3*e4
    if order == 4: return (1+dot)**2+M+mix*e2
    if order == 5: return (1+dot)**3+np.exp(-length*df.mean(2))
    if order == 6: return (1+dot)**2+np.exp(-length*df.mean(2))
    vals=X[:,:,None]*X[:,None,:]; vals=vals[:,np.triu_indices(d)[0],np.triu_indices(d)[1]]
    pw=vals@vals.T/vals.shape[1]
    if order==7: return 1+dot+pw+mix*((1+dot)**3+M+e2)
    gg=np.exp(-length*(vals[:,None,:]-vals[None,:,:])**2).mean(2)
    return 1+dot+M+mix*e2+gg

def predict_kernel(z,b,c,p,target,models):
    from scipy.linalg import cho_factor, cho_solve
    n=len(target); ntest=len(z)-n
    if not ntest: return np.empty((0,)+target.shape[1:])
    out=0.
    for spec,ridge,weight in models:
        K=components(z,b,c,p,spec)
        out=out+weight*(K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+ridge*np.eye(n)),target))
    return out
'''
# Train-only scaling, so predictions are independent of test cohort composition.
common=common.replace('p, spec):','p, spec, n):').replace('np.mean(X[:1]*0)+np.var(X)','np.var(X[:n])').replace('p,spec)\n','p,spec,n)\n')
exec(common)
sys.path.insert(0,str(root)); import node_00 as m
z,b,c,p=m.features(s); Y=T
from scipy.linalg import cho_factor,cho_solve
from itertools import product
sl=sl+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(129).permutation(n),4)]
Xs=[tuple(a[np.r_[i,j]] for a in (z,b,c,p)) for i,j in sl]
I=[np.eye(len(i)) for i,j in sl]
records=[]
def ev(spec,rdgs):
    ks=[components(*a,spec,len(i)) for a,(i,j) in zip(Xs,sl)]
    for r in rdgs:
        P=[]; e=[]
        for k,(i,j),eye in zip(ks,sl,I):
            l=len(i); q=k[l:,:l]@cho_solve(cho_factor(k[:l,:l]+r*eye,check_finite=False),Y[i],check_finite=False); P.append(q); diff=q-Y[j]; e.append([abs(diff[:,0]).mean(),abs(diff[:,1:]@f.T).mean()])
        records.append((spec,r,np.array(e),P))
for a,warp,l,o in product([.08,.17,.27,.37,.52,.75,1.15],[0.,.65],[.012,.045,.16,.55,1.8],range(9)):
    ev((a,0,warp,l,o,.75),[1e-7,1e-5,.0004,.005])
for fam,a,l,o in product(range(1,6),[.22,.45,.85],[.035,.18,.8],[1,2,4,5,6,8]):
    ev((a,fam,0.,l,o,.75),[1e-7,.0001,.004])
for h in range(2):
    rank=sorted(records,key=lambda r: .35*r[2][0,h]+.45*r[2][1:,h].mean()+.2*r[2][1:,h].max()); print('top',h,[(r[0],r[1],r[2][:,h].round(5)) for r in rank[:4]],flush=True)
    candidates=[]
    def score(pred):
        es=[q-Y[j] for q,(_,j) in zip(pred,sl)]
        E=np.array([abs(e[:,0]).mean() if h==0 else abs(e[:,1:]@f.T).mean() for e in es]); return .35*E[0]+.45*E[1:].mean()+.2*E[1:].max(),E
    for k in [1,3,6,12,25,60]:
        chosen=[]; seen=set()
        for rec in rank:
            if rec[0] in seen: continue
            chosen.append(rec);seen.add(rec[0]);
            if len(chosen)==k:break
        P=[np.mean([r[3][i] for r in chosen],axis=0) for i in range(len(sl))]
        val,e=score(P); candidates.append((val,e,[(r[0],r[1],1/k) for r in chosen]))
    for start in range(4):
        chosen=[rank[start]]; P=rank[start][3]; best=score(P)[0]; candidates.append((best,score(P)[1],[(chosen[0][0],chosen[0][1],1.)]))
        for step in range(7):
            w=1/(step+2); trial=min(rank[:100],key=lambda r:score([(1-w)*x+w*y for x,y in zip(P,r[3])])[0]); PP=[(1-w)*x+w*y for x,y in zip(P,trial[3])]; val,e=score(PP)
            if val>=best: break
            P=PP; best=val;chosen.append(trial); candidates.append((val,e,[(r[0],r[1],w) for r in chosen]))
    v,e,models=min(candidates,key=lambda r:r[0]); print('chosen',h,e.round(6),models,flush=True)
    Path('/tmp/finalmodels'+str(h)).write_text(repr(models))
base=(root/'node_00.py').read_text().split('\ndef kernel')[0]
mods=[Path('/tmp/finalmodels'+str(h)).read_text() for h in range(2)]
(root/'node_00.py').write_text(base+common+'\nMODELS='+mods[0]+'''

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    z,b,c,p=features(study)
    yhat=predict_kernel(z,b,c,p,np.asarray(study['train']['y']),MODELS)
    return {'y_hat':yhat,'z':z,'b':b,'c':c,'p':p}
''')
(root/'node_02.py').write_text('import numpy as np\n'+common+'\nMODELS='+mods[1]+'''

def run_node(inputs: dict) -> dict:
    a=inputs['node_00']; t=inputs['node_01']
    weights=predict_kernel(a['z'],a['b'],a['c'],a['p'],t['modes'],MODELS)
    return {'state_pred':weights@t['basis'].T}
''')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/finalfit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`15946 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py', '/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
top 0 [((0.37, 0, 0.0, 0.16, 1, 0.75), 0.005, array([0.0791, 0.1032, 0.1093, 0.0765, 0.1146, 0.0706, 0.0803, 0.1159, 0.093 ])), ((0.37, 0, 0.0, 0.16, 2, 0.75), 0.005, array([0.0785, 0.107 , 0.1164, 0.083 , 0.1182, 0.0642, 0.0823, 0.1175, 0.1027])), ((0.37, 0, 0.0, 0.16, 1, 0.75), 0.0004, array([0.0846, 0.1189, 0.11  , 0.0765, 0.1187, 0.0483, 0.0742, 0.1087, 0.0968])), ((0.08, 0, 0.0, 0.16, 0, 0.75), 0.0004, array([0.0897, 0.079 , 0.0548, 0.0806, 0.1323, 0.0562, 0.0847, 0.1051, 0.0868]))]
chosen 0 [0.0732 0.086  0.0654 0.0552 0.1111 0.0358 0.0675 0.107  0.0842] [((0.08, 0, 0.0, 0.16, 0, 0.75), 0.0004, 0.25), ((0.37, 0, 0.0, 0.045, 1, 0.75), 1e-05, 0.25), ((0.22, 4, 0.0, 0.035, 5, 0.75), 0.004, 0.25), ((0.37, 0, 0.0, 0.16, 1, 0.75), 0.005, 0.25)]
top 1 [((0.52, 0, 0.0, 0.012, 5, 0.75), 1e-07, array([0.0656, 0.0344, 0.0295, 0.029 , 0.0627, 0.0319, 0.0348, 0.0658, 0.0393])), ((0.52, 0, 0.0, 0.012, 5, 0.75), 1e-05, array([0.0655, 0.0344, 0.0295, 0.029 , 0.0627, 0.0319, 0.0348, 0.0658, 0.0394])), ((0.52, 0, 0.0, 0.045, 5, 0.75), 1e-07, array([0.0657, 0.0343, 0.0297, 0.0289, 0.0628, 0.032 , 0.0349, 0.0657, 0.0393])), ((0.52, 0, 0.0, 0.045, 5, 0.75), 1e-05, array([0.0657, 0.0343, 0.0297, 0.0289, 0.0628, 0.0321, 0.0349, 0.0657, 0.0393]))]
chosen 1 [0.0585 0.0382 0.0338 0.0262 0.0558 0.0276 0.0382 0.0661 0.0369] [((0.52, 0, 0.0, 0.012, 5, 0.75), 1e-05, 0.3333333333333333), ((0.37, 0, 0.0, 0.012, 1, 0.75), 1e-07, 0.3333333333333333), ((0.75, 0, 0.0, 0.012, 5, 0.75), 0.0004, 0.3333333333333333)]
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.073181", "public_cv_traj_mae_exceeded:0.059718"], "cv_metrics": {"public_cv_y_mae": 0.073181, "public_cv_traj_mae": 0.059718}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4e48a04027b0f49 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a4e48a04 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 8, in <module>
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 991, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1128, in get_code
  File "<frozen importlib._bootstrap_external>", line 1186, in get_data
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/batch_solver.py'
```

### Turn 12 — `execute_in_remote_docker` (`49563 ms` LLM)
- **Rationale**: Fit the frozen temporal basis separately within each split. Change a head only if the alternative improves the measured split score without worsening the official split or the worst auxiliary split, then rerun the DAG verifier on the saved files.
- **Alternatives Weighed**: `Submit the saved candidate as complete` (rejected: The isolated DAG test failed both numerical acceptance thresholds, although all nodes executed once.); `Run the single-file schema validator again` (rejected: It requires batch_solver.py, which is not a deliverable for this DAG task.)

```bash
cat >/tmp/last_compare.py <<'PY'
exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from pathlib import Path
from scipy.linalg import cho_factor,cho_solve
from itertools import product
sys.path.insert(0,'/holdout_task');import node_00 as a,node_02 as b
z,v,c,p=a.features(s); folds=sl+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(129).permutation(n),4)]; inputs=[tuple(x[np.r_[i,j]] for x in (z,v,c,p)) for i,j in folds]
code='''
def latent_kernel(z,b,c,p,n,spec):
    a, form, warp, deg, gamma, mix = spec
    u=a*z+b; v=a*z-b
    if form==0: x=np.c_[u,v]
    elif form==1: x=np.c_[u,v,z*np.roll(z,-1,axis=1)*.20]
    elif form==2: x=np.c_[u,v,c*.25]
    elif form==3: x=np.c_[u,v,z*.25]
    else: x=np.c_[u,v,(u*v)*.5]
    x=x/max(1e-9,x[:n].std())
    if warp==1: x=np.arcsinh(x)
    if warp==2: x=x/(1+.2*x*x)**.5
    dot=x@x.T/x.shape[1]; d=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0.)
    u=np.exp(-gamma*(x[:,None,:]-x[None,:,:])**2); m=u.mean(2); l=x.shape[1]
    pair=(l*m*m-(u*u).mean(2))/(l-1)
    return (1+dot)**deg+mix*(m+pair)+(1-mix)*np.exp(-gamma*d)

def latent_predict(z,b,c,p,target,specs):
    from scipy.linalg import cho_factor,cho_solve
    n=len(target); ans=0.
    for spec,ridge,w in specs:
        k=latent_kernel(z,b,c,p,n,spec)
        ans=ans+w*(k[n:,:n]@cho_solve(cho_factor(k[:n,:n]+ridge*np.eye(n)),target))
    return ans
'''
exec(code)
# Fit the temporal basis separately in every fold, exactly as the DAG does.
FT=[]; Y=[]
for i,j in folds:
 ss=dict(s,train={k:x[i] for k,x in t.items()}); ff=old.times(ss); FT.append(ff);Y.append(np.c_[t['y'][i],old.fit(ff,t['traj'][i].T).T])
def measure(h,ps):
 return np.array([np.abs(q-(t['y'][j] if h==0 else t['traj'][j])).mean() for q,(_,j) in zip(ps,folds)])
def loss(es): return .4*es[0]+.4*es[1:].mean()+.2*es[1:].max()
P0=[[],[]]
for xx,ff,yy,(i,j) in zip(inputs,FT,Y,folds):
 P0[0].append(a.predict_kernel(*xx,yy[:,0],a.MODELS));P0[1].append(b.predict_kernel(*xx,yy[:,1:],b.MODELS)@ff.T)
base=[measure(h,P0[h]) for h in range(2)]; print('verified baseline split MAEs',*[e.round(6).tolist() for e in base],flush=True)
records=[[],[]]
for spec in product([.10,.28,.48,.72],range(5),range(3),[1,2,3],[.03,.18,.7],[.0,.85]):
 ks=[latent_kernel(*xx,len(i),spec) for xx,(i,j) in zip(inputs,folds)]
 for ridge in [1e-6,.0002,.003]:
  pp=[[],[]]
  for k,ff,yy,(i,j) in zip(ks,FT,Y,folds):
   m=len(i); q=k[m:,:m]@cho_solve(cho_factor(k[:m,:m]+ridge*np.eye(m),check_finite=False),yy,check_finite=False); pp[0].append(q[:,0]);pp[1].append(q[:,1:]@ff.T)
  for h in range(2):
   e=measure(h,pp[h]);records[h].append((loss(e),spec,ridge,pp[h]))
for h in range(2):
 ranked=sorted(records[h],key=lambda r:r[0]); options=[]; es=base[h]
 for L in [1,3,6,12,24]:
  chosen=[]; seen=set()
  for _,spec,r,ps in ranked:
   if spec in seen:continue
   chosen.append((spec,r,ps));seen.add(spec)
   if len(chosen)==L:break
  mean=[np.mean(q,axis=0) for q in zip(*(p for _,_,p in chosen))]
  for w in [.25,.5,.75,1.]:
   ee=measure(h,[(1-w)*u+w*v for u,v in zip(P0[h],mean)])
   if ee[0]<=es[0] and ee[1:].max()<=es[1:].max() and loss(ee)<loss(es)-.001:options.append((loss(ee),w,ee,[(sp,r,1/L) for sp,r,_ in chosen]))
 if not options:continue
 _,w,ee,models=min(options,key=lambda x:x[0]);print('promote head',h,ee.round(6),flush=True)
 path=Path('/holdout_task')/('node_00.py' if h==0 else 'node_02.py');text=path.read_text(); oldline="yhat=predict_kernel(z,b,c,p,np.asarray(study['train']['y']),MODELS)" if h==0 else "weights=predict_kernel(a['z'],a['b'],a['c'],a['p'],t['modes'],MODELS)"; lhs,call=oldline.split('=',1); newcall=call.replace('predict_kernel','latent_predict').replace('MODELS','ALT_MODELS');text=text.replace(oldline,f'{lhs}={1-w!r}*{call}+{w!r}*{newcall}');path.write_text(text+'\n'+code+'\nALT_MODELS='+repr(models)+'\n')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/last_compare.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `1` (`765 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.073181", "public_cv_traj_mae_exceeded:0.059718"], "cv_metrics": {"public_cv_y_mae": 0.073181, "public_cv_traj_mae": 0.059718}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/last_compare.py", line 36, in <module>
    ss=dict(s,train={k:x[i] for k,x in t.items()}); ff=old.times(ss); FT.append(ff);Y.append(np.c_[t['y'][i],old.fit(ff,t['traj'][i].T).T])
                       ~^^^
IndexError: index 4 is out of bounds for axis 0 with size 4
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
