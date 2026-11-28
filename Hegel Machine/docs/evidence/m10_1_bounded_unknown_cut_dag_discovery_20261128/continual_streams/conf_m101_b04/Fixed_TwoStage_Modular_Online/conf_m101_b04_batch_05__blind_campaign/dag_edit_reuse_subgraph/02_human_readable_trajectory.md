# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b04_batch_05__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `138,529` (`133,601` / `61,568` / `4,928`) |
| **Total Episode Wall Time (s)** | `177.35s` |

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

### Turn 1 — `execute_in_remote_docker` (`11890 ms` LLM)
- **Rationale**: Inspect the sink and public evaluator to identify the exact intermediate contract and available calibration evidence before fitting.
- **Alternatives Weighed**: `Rewrite the sink` (rejected: The sink is frozen and the reported shift is upstream.)

```bash
cd /holdout_task && ls -la && cat _dag_spec.json _frozen_nodes/node_01.py test_dag_candidate.py validate_public_schema.py public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust Cauchy IRLS calibration reference panel'
```
- **Container Exit Code**: `0` (`349 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
}autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`17610 ms` LLM)
- **Rationale**: Inspect the exact evaluator and look for the named precedent. Panel dimensions, linear residual spread, and coordinate ranges will determine which nonlinear bases and inversion checks are warranted.
- **Alternatives Weighed**: `Use an unrelated robust solver from the skill index` (rejected: The returned solvers target network assimilation and boxed ridge certification, not calibration inversion.)

```bash
cd /holdout_task && cat _frozen_nodes/node_01.py test_dag_candidate.py; ls -la /pretrain_task; find /pretrain_task -maxdepth 3 -type f | head -45; python3 - <<'PY'
import numpy as np,pickle
p=pickle.load(open('public_batch.pkl','rb'))[0]
for k,v in p.items():
 print(k, [(a,np.shape(b)) for a,b in v.items()] if isinstance(v,dict) else np.shape(v))
a=p['panel_a']; u,v=a['u'],a['v']; X=np.c_[np.ones(len(u)),u]; r=v-X@np.linalg.lstsq(X,v,rcond=None)[0]; print('u limits',u.min(0),u.max(0),'v limits',v.min(0),v.max(0)); print('linear residual norms', np.round(np.sort(np.linalg.norm(r,axis=1)),5)); print('first data',np.c_[u,v][:5]); print('x limits',p['train']['x'].min(0),p['train']['x'].max(0))
PY
```
- **Container Exit Code**: `1` (`441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Error(f"Node {nid} failed anti-identity-shell check")
        outputs[nid] = out_v
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
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 28 09:38 ..
study [('eval_times', (8,)), ('panel_a', ()), ('panel_b', ()), ('panel_c', ()), ('train', ()), ('test', ()), ('telemetry_vec', (4,))]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
KeyError: 'panel_a'
```

### Turn 3 — `execute_in_remote_docker` (`26569 ms` LLM)
- **Rationale**: Read the study at its actual nested key. The residuals will distinguish an affine fit distorted by three outliers from a genuinely nonlinear calibration, and the coordinate ranges will show whether inversion must extrapolate.
- **Alternatives Weighed**: `Fit the inverse directly to Panel A pairs` (rejected: The three corrupted observations would enter as predictors; a robust forward fit keeps the reference coordinates clean and makes the corruption visible in the residuals.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
from scipy.special import erf
from itertools import combinations_with_replacement
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); n,d=u.shape
np.set_printoptions(precision=5,suppress=True,linewidth=130)
print('shapes',u.shape,v.shape,'ranges',u.min(0),u.max(0),v.min(0),v.max(0));print('study x ranges',s['train']['x'].min(0),s['train']['x'].max(0),s['test']['x'].min(0),s['test']['x'].max(0));print('first pairs',np.c_[u,v][:8]);
base=np.c_[np.ones(n),u]; groups={'linear':base}
for j in (2,3,4): groups['poly'+str(j)] = np.column_stack([np.prod(u[:,ix],axis=1) for ix in combinations_with_replacement(range(d),j)])
for name,parts in [('linear',['linear']),('quadratic',['linear','poly2']),('cubic',['linear','poly2','poly3']),('odd cubic',['linear','poly3']),('additive powers',[]),('additive smooth',[])]:
 X=np.column_stack([groups[k] for k in parts]) if parts else (np.c_[base,u**2,u**3,u**4,u**5] if name=='additive powers' else np.c_[base,np.tanh(u),np.sin(u),erf(u),np.cos(u)])
 if X.shape[1]>=n-3: continue
 w=np.ones(n)
 for it in range(25):
  a=np.linalg.lstsq(X*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(X@a-v,axis=1); w=1/np.sqrt(1+(r/max(np.median(r)*.7,1e-8))**2)
 keep=np.argsort(r)[:-3]; a=np.linalg.lstsq(X[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(X@a-v,axis=1); print(name,X.shape[1], 'median/inlier max/outlier norms',np.median(r),r[keep].max(),r[np.argsort(r)[-3:]], 'outlier rows',np.argsort(r)[-3:]);
 if name in ('linear','quadratic'): print('coefficients',a)
PY
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
29462 1.29106] [-4.74717 -5.63258 -7.5402  -5.39043] [7.17332 5.91203 5.22132 6.51851]
study x ranges [-1.83649 -1.71595 -1.71118 -1.78963] [1.48217 1.06243 1.19848 1.2859 ] [-1.70874 -1.98221 -1.76957 -1.67673] [1.53468 1.04697 1.16886 1.2102 ]
first pairs [[-0.99142 -0.73113 -0.34432 -0.13309 -0.54804 -0.68093 -0.39917  0.2931 ]
 [-1.18672 -0.65705  1.14091  1.12695 -0.86032 -1.39138  1.29762  1.19216]
 [-0.67118 -0.38797 -0.47598 -0.1304  -0.37997 -0.29081 -0.53518  0.17884]
 [-1.22354 -0.39724  1.1659  -0.63821 -1.09951 -1.03267  1.25326  0.10211]
 [ 0.03996  0.40334  0.09695 -1.03779  0.04554  0.22859  0.28345 -1.04435]
 [ 0.17665  0.19787 -0.46372  0.84433  6.75441 -5.63258  5.22132 -5.14274]
 [ 0.42542  0.30725  0.94321 -0.02769  0.50956 -0.05632  0.94955 -0.11581]
 [ 0.7242  -1.17927  0.68053  1.07019  1.01644 -1.95938  0.49807  1.17534]]
linear 5 median/inlier max/outlier norms 0.31684639055201425 0.7273843455084618 [11.85928 12.25276 12.68792] outlier rows [31  5 17]
coefficients [[ 0.09802 -0.18164 -0.10312  0.04797]
 [ 1.10332 -0.01904 -0.03015 -0.25462]
 [-0.31791  1.05133 -0.03218 -0.04008]
 [ 0.00168 -0.3407   1.19443 -0.02797]
 [-0.06977 -0.03705 -0.27537  0.99194]]
quadratic 15 median/inlier max/outlier norms 0.07072887155568755 0.12493356108332879 [12.13662 12.29664 12.30251] outlier rows [31 17  5]
coefficients [[ 0.14046 -0.12594 -0.10323  0.02895]
 [ 1.15373 -0.00137 -0.01224 -0.30746]
 [-0.30913  1.08461 -0.01396  0.00898]
 [-0.00146 -0.30974  1.15994 -0.00175]
 [-0.00571 -0.00755 -0.29845  0.99627]
 [-0.02213  0.00572 -0.01789  0.01849]
 [ 0.30215  0.01416  0.02434  0.03028]
 [ 0.00145 -0.02534  0.01028  0.01626]
 [ 0.00103  0.01066  0.00162  0.34432]
 [-0.00687 -0.01902 -0.02048 -0.00384]
 [ 0.00386  0.32708  0.00068 -0.00535]
 [ 0.03089  0.0144   0.019    0.02182]
 [ 0.0111   0.00196 -0.00513  0.02464]
 [ 0.01425  0.00493  0.33067  0.01121]
 [-0.03299 -0.01034  0.01202  0.02139]]
cubic 35 median/inlier max/outlier norms 6.517109399868133e-05 0.00020952980744281413 [12.01232 12.2609  12.35592] outlier rows [31  5 17]
odd cubic 25 median/inlier max/outlier norms 0.18466510142300538 0.45835294846914376 [11.32099 12.11351 12.55677] outlier rows [31  5 17]
additive powers 21 median/inlier max/outlier norms 0.2181654960307554 0.4990410504916921 [11.65143 12.26282 12.64488] outlier rows [31  5 17]
additive smooth 21 median/inlier max/outlier norms 0.22992836190122623 0.5077190928617452 [11.60976 12.10955 12.72879] outlier rows [31  5 17]
```

### Turn 4 — `execute_in_remote_docker` (`25977 ms` LLM)
- **Rationale**: Coefficient support and leave-one-out residuals will distinguish a reproducible cyclic formula from a near-interpolating cubic fit before implementing its inverse.
- **Alternatives Weighed**: `Deploy the full 35-term cubic` (rejected: Its small training residual establishes the missing degree, but the remaining 45 clean rows may not support stable extrapolation without reducing the basis.)
- **ATMS NOGOOD DECLARED**: Refuted `A quadratic basis captures the Panel A forward calibration once sparse outliers are removed.` | Witness: `Both robust fits identify rows 5, 17, and 31 as outliers, but adding cubic terms decreases the maximum remaining row residual from 0.12493 to 0.00021.` | Blanket Diagnosis: `The residual persists after downweighting the same corrupted rows and depends on missing higher-order structure in the reference coordinates.` | Sublated Pivot: `Extract the cyclic cubic terms and test whether their reduced support predicts left-out calibration pairs.`

```bash
cd /holdout_task && cat > /tmp/panel_a_basis.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v'))
terms=[()]+[t for k in (1,2,3) for t in combinations_with_replacement(range(4),k)]
X=np.column_stack([np.prod(u[:,t],axis=1) for t in terms]); n=len(u)
A=X[:,:5]; w=np.ones(n)
for _ in range(20):
 c=np.linalg.lstsq(A*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@c-v,axis=1); w=1/np.hypot(1,r/(.7*np.median(r)))
good=np.sort(np.argsort(r)[:-3]); C=np.linalg.lstsq(X[good],v[good],rcond=None)[0]
print('condition of clean full basis',np.linalg.cond(X[good])); np.set_printoptions(precision=8)
for j in range(4):
 print('OUTPUT',j,[(terms[i],round(C[i,j],7)) for i in np.argsort(abs(C[:,j]))[-14:][::-1]])
 keep=np.flatnonzero(abs(C[:,j])>.005); a=X[:,keep]; c=np.linalg.lstsq(a[good],v[good,j],rcond=None)[0]; q=a[good]@c-v[good,j]; h=np.sum(np.linalg.qr(a[good],mode='reduced')[0]**2,axis=1); print('reduced',list(zip([terms[i] for i in keep],c)), 'max',np.max(abs(q)),'LOO max',np.max(abs(q/(1-h))))
print('y / trajectory scales',np.std(s['train']['y']),np.std(s['train']['traj'],axis=0))
PY
python3 /tmp/panel_a_basis.py
```
- **Container Exit Code**: `0` (`382 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
condition of clean full basis 71.70406742601264
OUTPUT 0 [((0,), 0.996238), ((0, 1), 0.3199487), ((1,), -0.3196297), ((0, 0, 0), 0.1600814), ((), 0.129898), ((3, 3, 3), 0.0002483), ((1, 2, 3), 0.0002454), ((0, 0, 1), -0.0002139), ((1, 1, 1), -0.0001941), ((2, 3, 3), -0.0001864), ((1, 2, 2), -0.0001767), ((1, 3, 3), -0.000149), ((0, 0, 3), -0.0001267), ((1, 3), 0.0001255)]
reduced [((), 0.12980066066736073), ((0,), 0.9964222914491812), ((1,), -0.31999699438743384), ((0, 1), 0.31994667619253014), ((0, 0, 0), 0.15998062558254525)] max 0.00025934914613512383 LOO max 0.0003015179198639445
OUTPUT 1 [((1,), 0.9219222), ((2,), -0.3201189), ((1, 2), 0.3200423), ((1, 1, 1), 0.15992), ((), -0.1351704), ((0, 0, 0), -0.0001123), ((0,), 0.0001026), ((0, 3, 3), 9.83e-05), ((0, 3), -9.79e-05), ((1, 1, 3), 8.17e-05), ((1, 3), 7.88e-05), ((2, 2), 7.61e-05), ((2, 2, 2), 7.51e-05), ((3, 3), -6.62e-05)]
reduced [((), -0.13515305853441156), ((1,), 0.9218610306094994), ((2,), -0.32001351859227634), ((1, 2), 0.3200058079882234), ((1, 1, 1), 0.16001718994107925)] max 0.00020950887294679532 LOO max 0.0002216496927010408
OUTPUT 2 [((2,), 0.9824866), ((3,), -0.3202774), ((2, 3), 0.3199516), ((2, 2, 2), 0.1599293), ((), -0.1119004), ((3, 3, 3), 0.0001923), ((1, 2, 2), 0.0001534), ((1,), -0.0001473), ((1, 3), -0.0001393), ((0, 0, 1), 0.00013), ((0, 0, 3), 0.000101), ((1, 3, 3), -9.7e-05), ((0, 3), -9.26e-05), ((0, 1, 2), -8.61e-05)]
reduced [((), -0.11176150546664627), ((2,), 0.9824730080061387), ((3,), -0.3199758219908242), ((2, 3), 0.32003113567552194), ((2, 2, 2), 0.1599176663625369)] max 0.0001865516416651447 LOO max 0.0002212920205962803
OUTPUT 3 [((3,), 0.8548697), ((0,), -0.3200597), ((0, 3), 0.3198552), ((3, 3, 3), 0.1601274), ((), 0.047801), ((0, 3, 3), 0.00022), ((1,), 0.0002175), ((1, 1, 1), -0.0002042), ((0, 1, 2), -0.0001136), ((0, 2, 2), -0.000107), ((2, 2, 3), -9.85e-05), ((2, 3, 3), 8.03e-05), ((0, 2), -7.53e-05), ((0, 2, 3), 6.82e-05)]
reduced [((), 0.04785679901740905), ((0,), -0.3199914098550918), ((3,), 0.8548783747277744), ((0, 3), 0.3199745038715082), ((3, 3, 3), 0.16006090396691341)] max 0.00023605522974384785 LOO max 0.00024687585461309895
y / trajectory scales 0.46251130904103904 [0.22141306 0.43861939 0.57008194 0.65693683 0.71833697 0.76611361
 0.80458502 0.83896381]
```

### Turn 5 — `execute_in_remote_docker` (`30862 ms` LLM)
- **Rationale**: Implement the measured cyclic forward operator, remove the three corrupted rows after robust fitting, and pass the inverted coordinates to the unchanged sink. The public evaluator will show whether the recalibration resolves both output errors.
- **Alternatives Weighed**: `Fit a separate unrestricted cubic for each coordinate` (rejected: All four outputs show the same cyclic support and shared nonlinear coefficients; the unrestricted fit adds terms unsupported by the measurements.)

```bash
cat > /holdout_task/node_00.py <<'PY'
import numpy as np


def _fit_panel(panel):
    """Fit the forward cyclic calibration on the clean reference coordinates."""
    u, v = panel['u'], panel['v']
    neighbor = np.roll(u, -1, axis=1)
    n, d = u.shape
    eye = np.eye(d)
    offsets = np.broadcast_to(eye, (n, d, d))
    design = np.concatenate((offsets, u[..., None] * eye,
                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)

    def fit(weights):
        return np.linalg.lstsq((design * weights[:, None, None]).reshape(-1, 2*d+3),
                               (v * weights[:, None]).ravel(), rcond=None)[0]

    weights = np.ones(n)
    for _ in range(24):
        residual = np.linalg.norm(design @ fit(weights) - v, axis=1)
        scale = max(0.7 * np.median(residual), 1e-12)
        weights = 1 / np.hypot(1, residual / scale)

    # Refine without shrinkage after separating the three corrupted panel rows.
    clean = np.ones(n)
    clean[np.argsort(residual)[-3:]] = 0
    return fit(clean)


def _invert(x, coefficients):
    d = x.shape[1]
    bias, gain = coefficients[:d], coefficients[d:2*d]
    cross, interaction, cubic = coefficients[2*d:]
    i = np.arange(d)
    j = (i + 1) % d
    linear = np.diag(gain)
    linear[i, j] = cross
    z = np.linalg.solve(linear, (x - bias).T).T
    jac = np.zeros((len(x), d, d))
    for _ in range(25):
        neighbor = z[:, j]
        residual = bias + gain*z + cross*neighbor + interaction*z*neighbor + cubic*z**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, i, i] = gain + interaction*neighbor + 3*cubic*z*z
        jac[:, i, j] = cross + interaction*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= step
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    coefficients = _fit_panel(study['panel_a'])
    calibrated = _invert(np.concatenate((train['x'], test['x']), axis=0), coefficients)
    n = len(train['x'])
    return dict(z_tr=calibrated[:n], z_te=calibrated[n:], panel_b=study['panel_b'],
                eval_times=study['eval_times'], y=train['y'], traj=train['traj'])
PY
python3 /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.010454, "public_cv_traj_mae": 0.008622}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=65882a4158cf2692 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_65882a41 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33178 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated Python process. The calibration jackknife would expose unstable coefficients or inversion, and the additional readout folds would expose a pass dependent on the official split.
- **Alternatives Weighed**: `Submit on the official split alone` (rejected: That split does not test sensitivity to calibration-row leverage or to a different readout holdout.)

```bash
set -e
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

node = load('candidate', root / 'node_00.py')
sink = load('frozen_sink', root / '_frozen_nodes/node_01.py')
with (root / 'public_batch.pkl').open('rb') as f:
    study = pickle.load(f)[0]['study']
a = study['panel_a']
u, v = a['u'], a['v']
x = np.concatenate((study['train']['x'], study['test']['x']))
def forward(z, c):
    d = z.shape[1]
    neighbor = np.roll(z, -1, axis=1)
    return c[:d] + c[d:2*d]*z + c[-3]*neighbor + c[-2]*z*neighbor + c[-1]*z**3

c = node._fit_panel(a)
residual = np.linalg.norm(forward(u, c) - v, axis=1)
clean = np.argsort(residual)[:-3]
worst_loo = 0.
worst_roundtrip = 0.
for i in clean:
    keep = np.arange(len(u)) != i
    ci = node._fit_panel({'u': u[keep], 'v': v[keep]})
    worst_loo = max(worst_loo, np.max(np.abs(forward(u[i:i+1], ci) - v[i])))
    z = node._invert(x, ci)
    assert np.all(np.isfinite(z))
    worst_roundtrip = max(worst_roundtrip, np.max(np.abs(forward(z, ci) - x)))
print(f'Clean-row calibration jackknife: worst coordinate residual={worst_loo:.6g}; worst inverse round-trip={worst_roundtrip:.6g}')
assert worst_loo < 0.001 and worst_roundtrip < 1e-9

train = study['train']
n = len(train['x'])
worst = np.zeros(2)
for idx in np.array_split(np.random.default_rng(42).permutation(n), 5):
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    sub = dict(study, train={k: train[k][keep] for k in ('x', 'y', 'traj')}, test={'x': train['x'][idx]})
    pred = sink.run_node({'node_00': node.run_node({'__input__': sub})})
    errors = [np.mean(np.abs(pred['y_hat'] - train['y'][idx])), np.mean(np.abs(pred['state_pred'] - train['traj'][idx]))]
    assert np.all(np.isfinite(errors))
    worst = np.maximum(worst, errors)
print(f'Five-fold worst MAE: y={worst[0]:.6f}; trajectory={worst[1]:.6f}')
assert np.all(worst <= 0.038)
PY
```
- **Container Exit Code**: `0` (`895 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.010454, "public_cv_traj_mae": 0.008622}, "exec_counts": {"node_00": 1, "node_01": 1}}
Clean-row calibration jackknife: worst coordinate residual=0.000273888; worst inverse round-trip=6.66134e-16
Five-fold worst MAE: y=0.013691; trajectory=0.012316

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=65882a4158cf2692 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_65882a41 (directory).
```

### Turn 7 — `submit_completed_task` (`13174 ms` LLM)
- **Rationale**: The isolated-process evaluator accepted the saved artifact, and each node ran exactly once. The calibration jackknife had a worst held-out coordinate residual of 0.000274, and both worst-fold prediction errors remained below the acceptance threshold.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py`. The frozen sink was unchanged. Public CV passed: scalar MAE 0.010454 and trajectory MAE 0.008622, both below 0.038.`

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
