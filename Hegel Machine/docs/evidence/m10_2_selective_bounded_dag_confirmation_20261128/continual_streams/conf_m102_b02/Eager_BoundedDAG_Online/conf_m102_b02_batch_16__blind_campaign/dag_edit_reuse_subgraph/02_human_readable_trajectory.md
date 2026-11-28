# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b02_batch_16__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `341,764` (`326,294` / `108,544` / `15,470`) |
| **Total Episode Wall Time (s)** | `556.81s` |

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

### Turn 1 — `execute_in_remote_docker` (`11863 ms` LLM)
- **Rationale**: Inspect the frozen consumers, public folds, measurement schema, and archived candidates before choosing a transform.

```bash
cd /holdout_task; cat _dag_spec.json; cat _frozen_nodes/*.py; cat test_dag_candidate.py; cat public_asset_index.json; du -ah /tmp/hegel_continual_archive | sort -hr | head -85; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,d=0):
 print(' '*d,type(x),getattr(x,'shape',''),getattr(x,'dtype',''),len(x) if hasattr(x,'__len__') else '')
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=' ');show(v,d+1)
 elif isinstance(x,(tuple,list)) and x: show(x[0],d+1)
 elif isinstance(x,np.ndarray): print(' '*d,x.ravel()[:16])
show(x)
PY
```
- **Container Exit Code**: `0` (`332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1.2081967   0.75656156 -0.55566797 -0.65713462
  0.59565407  0.1139063  -1.2090315   0.92096841]
    v      <class 'numpy.ndarray'> (24, 4) float64 24
     [-0.23454482  0.23741542  0.03537348  0.23944029 -0.25905577 -0.34424403
  0.55744134 -0.47963924 -0.01404531 -0.94855749  0.20229853 -0.65614503
  0.74527899 -0.22737704  0.23373038  0.52346518]
   panel_c     <class 'dict'>   3
    u      <class 'numpy.ndarray'> (24, 4) float64 24
     [-0.66812123  1.21599825 -1.02833065  1.16041079  1.01059511 -0.06407089
  0.64471547 -0.26711388 -0.10064115 -0.30530087  0.49221325 -1.1829938
 -0.93834505  0.17641373  0.69502478 -1.09285549]
    v1      <class 'numpy.ndarray'> (24, 4) float64 24
     [ 1.81193306  0.70475349  1.7317909   0.1152826   0.13702731  0.27178619
  0.36176301 -0.11784595  0.09514534  0.06480916  0.6828291   0.48001893
 -0.07184592 -0.94414374  0.60220392  1.09697571]
    v2      <class 'numpy.ndarray'> (24, 4) float64 24
     [-0.13466318 -0.55454573 -1.12514428  0.07362558 -0.47336621  0.35226557
  0.3239573  -0.18824276  0.24898586  0.03060391  0.33128308 -0.11860353
  0.39964348 -0.0548616   0.17256137 -0.35916517]
   train     <class 'dict'>   4
    x      <class 'numpy.ndarray'> (72, 4) float64 72
     [-0.47880009 -0.14237036 -0.25556901 -0.08287091 -0.30786279 -0.74291812
 -0.24122696  0.63371188  0.76272972 -1.3782      0.77711207  1.41487452
  0.41248073  0.54202938 -0.69030705  0.88079567]
    y      <class 'numpy.ndarray'> (72,) float64 72
     [-0.09873482  0.2382896   1.23095534 -0.16343076 -0.77350561  0.51528453
  0.68194124  0.65126955  0.47946206  1.31873562 -0.21916804  0.53051897
  0.14508567 -0.28014166 -0.35442961  0.45835733]
    traj      <class 'numpy.ndarray'> (72, 8) float64 72
     [-1.93691472e-02 -4.00565779e-05  1.97375169e-02  7.01045437e-02
  1.17101615e-01  1.57552302e-01  1.96904275e-01  2.29235961e-01
 -7.12423581e-02 -1.21324814e-01 -1.37004487e-01 -1.46274784e-01
 -1.44080598e-01 -1.34888639e-01 -1.37281562e-01 -1.32510589e-01]
    summary      <class 'numpy.ndarray'> (4,) float64 4
     [ 0.09475091  0.58601786 -0.36451825  1.14689131]
   test     <class 'dict'>   1
    x      <class 'numpy.ndarray'> (36, 4) float64 36
     [-0.07299867 -0.53536802 -0.48073498 -0.05800215  0.8617715  -0.51095134
  0.54995959  0.11627493 -0.30926378  0.52117286 -0.12785706 -0.62063514
 -0.78213046 -0.49703973 -0.19742317 -0.59036867]
   telemetry_vec     <class 'numpy.ndarray'> (4,) float64 4
    [1.25 2.15 0.22 1.25]
```

### Turn 2 — `execute_in_remote_docker` (`9258 ms` LLM)
- **Rationale**: Recover the archive paths and validation split definition omitted from the truncated inspection, then compare calibration candidates against the measurements.

```bash
find /tmp/hegel_continual_archive -type f | head -100; ls -R /pretrain_task | head -80; sed -n '85,240p' /holdout_task/test_dag_candidate.py; cat /holdout_task/public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment nonlinear calibration'
```
- **Container Exit Code**: `0` (`291 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}No matching skill cards found.
```

### Turn 3 — `execute_in_remote_docker` (`28933 ms` LLM)
- **Rationale**: Index the archived solver functions and recover the exact fold protocol without repeating the truncated schema dump.

```bash
for f in /tmp/hegel_continual_archive/*solver.py; do echo ===$f; wc -l "$f"; grep -nE '^def |^class |^ *return ' "$f"; done; sed -n '145,250p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`357 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rn z
20:def parts(study):
27:    return z,b
29:def times(study):
34:    return F(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
36:def kernel(x, degree, decay):
39:    return (1+dot)**degree+np.exp(-decay*dist)
41:def predict(study):
52:        return k[n:,:n]@np.linalg.solve(k[:n,:n]+ridge*np.eye(n),target)
55:        return pred if w==1 else w*pred+(1-w)*head(b,target)
58:    return y,state
60:def solve_batch(study:dict)->dict:
62:    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
===/tmp/hegel_continual_archive/conf_m102_b02_batch_11_solver.py
62 /tmp/hegel_continual_archive/conf_m102_b02_batch_11_solver.py
6:def _design(x):
7:    return np.column_stack((np.ones(len(x)), x))
10:def _coordinates(study, x):
24:    return _design(z) @ mix
27:def _mixed_design(b, h):
29:    return _design(np.column_stack((b, h, b*b, h*h,
33:def solve_batch(study: dict) -> dict:
61:    return {'y_hat': y, 'state_pred': state,
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
```

### Turn 4 — `execute_in_remote_docker` (`10710 ms` LLM)
- **Rationale**: Recover the archived coordinate transforms and prediction bases so they can be compared against the current measurements without modifying frozen nodes.

```bash
for f in /tmp/hegel_continual_archive/*solver.py; do echo ====; cat "$f"; done
```
- **Container Exit Code**: `0` (`283 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ict(study); end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
====
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
             (np.arctan(1.75*b), .0003, True, False),
             (b / (1 + b*b), .003, False, True),
             (np.tanh(1.4*b), .003, False, True),
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
```

### Turn 5 — `execute_in_remote_docker` (`28797 ms` LLM)
- **Rationale**: Compare archived solvers on the public split and use the panel residuals to distinguish affine, polynomial, threshold-power, and saturating maps before fitting the editable node.

```bash
cat /tmp/hegel_continual_archive/*batch_05_solver.py; sed -n '14,39p' /tmp/hegel_continual_archive/*batch_01_solver.py; cat > /tmp/probe23.py <<'PY'
import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(u): return np.c_[np.ones(len(u)),u]
def fit(u,v): return np.linalg.lstsq(u,v,rcond=1e-10)[0]
def err(F,v): return np.linalg.norm(F@fit(F,v)-v,axis=1)
if __name__=='__main__':
 for name in ('panel_a','panel_b','panel_c'):
  p=S[name]; u=p['u']; pair=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)])
  for key in p.keys()-{'u'}:
   v=p[key]; print(name,key,'std',v.std(0).round(4))
   for nm,F in [('linear',D(u)),('diagonal2',D(np.c_[u,u*u])),('pair+3',D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])),('poly2',D(np.c_[u,u*u,pair])),('erf',D(erf(1.25*u))),('tanh',D(np.tanh(1.25*u))),('threshold',D(np.sign(u)*np.maximum(abs(u)-.22,0)**1.25))]:
    e=err(F,v); k=e<3*np.median(e); print(nm, np.round(np.quantile(e,[0,.5,.9,1]),5),'trim',k.sum(),np.sqrt(np.mean((F[k]@fit(F[k],v[k])-v[k])**2)).round(5))
   print('linear coeff',fit(D(u),v).round(4))
   for nm,tf,p0,lo,hi in [('thresh',lambda u,q: np.sign(u)*np.maximum(abs(u)-q[0],0)**q[1],[.22,1.25],[0,.1],[1,4]),('erf+u',lambda u,q: erf(q[0]*u)+q[1]*u,[1.25,.1],[.05,-2],[8,2]),('tanh+u',lambda u,q: np.tanh(q[0]*u)+q[1]*u,[1.25,.1],[.05,-2],[8,2]),('rational+u',lambda u,q: u/(1+q[0]*u*u)**q[1],[1.25,.5],[.01,-2],[20,3])]:
    def res(q):
     F=D(tf(u,q));return (F@fit(F,v)-v).ravel()
    r=least_squares(res,p0,bounds=(lo,hi));print(nm,r.x.round(5),np.sqrt(np.mean(r.fun**2)).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe23.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,runpy,glob,numpy as np
sys.path.insert(0,'/tmp');from probe23 import S
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 fun=runpy.run_path(f)['solve_batch']; a=S['train']; n=52; s={**S,'train':{k:v[:n] for k,v in a.items()},'test':{'x':a['x'][n:]}}; p=fun(s);print(f.split('/')[-1],np.mean(abs(p['y_hat']-a['y'][n:])),np.mean(abs(p['state_pred']-a['traj'][n:])))
PY
```
- **Container Exit Code**: `0` (`766 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5913 0.41599] trim 24 0.13139
diagonal2 [0.11203 0.22561 0.28574 0.33096] trim 24 0.1147
pair+3 [0.00775 0.0323  0.05101 0.07228] trim 24 0.01883
poly2 [0.02369 0.15743 0.2197  0.26488] trim 24 0.08193
erf [0.05127 0.35963 0.46995 0.55501] trim 24 0.18627
tanh [0.05048 0.34573 0.45702 0.53679] trim 24 0.17976
threshold [0.01653 0.11239 0.1724  0.21545] trim 24 0.06115
linear coeff [[-0.1004 -0.0428  0.0618 -0.1126]
 [-0.1079  0.3765 -0.1799  0.459 ]
 [-0.156  -0.4164 -0.3564  0.1842]
 [-0.2595  0.2664 -0.3144 -0.2905]
 [ 0.3727  0.1705 -0.231  -0.062 ]]
thresh [0.35008 1.49981] 8.6e-05
erf+u [0.05 2.  ] 0.131395
tanh+u [0.05 2.  ] 0.131394
rational+u [20.      -0.80837] 0.021342
panel_c v1 std [0.672  0.4564 0.7709 0.6055]
linear [0.22739 0.99832 1.37486 1.9905 ] trim 24 0.53105
diagonal2 [0.30236 0.90973 1.38937 1.74938] trim 24 0.4902
pair+3 [3.0e-05 1.1e-04 1.7e-04 2.0e-04] trim 24 6e-05
poly2 [2.0e-05 9.0e-05 1.5e-04 2.1e-04] trim 24 5e-05
erf [0.17638 0.991   1.50191 2.06577] trim 24 0.55178
tanh [0.18422 0.98996 1.4675  2.06144] trim 24 0.54928
threshold [0.13533 0.99023 1.36415 1.78201] trim 24 0.50884
linear coeff [[-0.1042  0.0664  0.3014  0.126 ]
 [ 0.0532  0.0459 -0.1708 -0.4493]
 [ 0.4515  0.1075  0.4362  0.0566]
 [-0.0223 -0.0284  0.0828  0.1413]
 [ 0.2259  0.1974  0.4248 -0.1244]]
thresh [0.88771 1.0422 ] 0.434323
erf+u [0.05 2.  ] 0.531053
tanh+u [0.05 2.  ] 0.531053
rational+u [20. -2.] 0.455149
panel_c v2 std [0.2728 0.2335 0.4025 0.3872]
linear [0.06191 0.16615 0.22843 0.2631 ] trim 24 0.08454
diagonal2 [0.00235 0.0091  0.01199 0.01695] trim 24 0.00466
pair+3 [0.04713 0.10233 0.18383 0.23669] trim 24 0.06475
poly2 [0.0028  0.00602 0.00965 0.01499] trim 24 0.00356
erf [0.05487 0.15705 0.26612 0.27607] trim 24 0.08501
tanh [0.0626  0.15199 0.25445 0.26949] trim 24 0.08362
threshold [0.04511 0.21744 0.32459 0.37062] trim 24 0.11629
linear coeff [[-0.0094  0.0025 -0.0928  0.0197]
 [-0.3622  0.1236  0.2019 -0.013 ]
 [-0.1404 -0.2004 -0.2349 -0.2873]
 [ 0.0939  0.2652  0.0416 -0.3027]
 [-0.0301  0.1329 -0.4155  0.1574]]
thresh [0.13287 0.62831] 0.082275
erf+u [ 0.05    -0.05614] 0.080745
tanh+u [ 0.05    -0.04976] 0.080746
rational+u [0.0669 3.    ] 0.080944
conf_m102_b02_batch_00_solver.py 0.2501593895747285 0.43787395013523034
conf_m102_b02_batch_01_solver.py 0.16164793529337323 0.18266123779501128
conf_m102_b02_batch_05_solver.py 0.10038911986257675 0.13155319597118015
conf_m102_b02_batch_11_solver.py 0.13133980429926834 0.22648811164240357
```

### Turn 6 — `execute_in_remote_docker` (`9725 ms` LLM)
- **Rationale**: Panel B is explained to 0.000086 RMSE by the fitted threshold-power map, but archived heads do not meet either target. Recover the exact frozen trajectory-head interface and its panel-C grammar before designing additional features.

```bash
cat /holdout_task/_frozen_nodes/node_01.py /holdout_task/_frozen_nodes/node_02.py /holdout_task/_frozen_nodes/node_03.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json | head -c 3500; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`299 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s['node_01']
    modes = temporal['modes']
    n = len(modes)

    def head(decay):
        k = kernel(design, 3, decay)
        return k[n:, :n] @ np.linalg.solve(k[:n, :n] + 1e-06 * np.eye(n), modes)

    state = (0.25 * head(0.15) + 0.75 * head(1.4)) @ temporal['basis'].T
    return {'state_pred': state}
import numpy as np


def run_node(inputs: dict) -> dict:
    y = inputs['node_00']['y_hat']
    state = inputs['node_02']['state_pred']
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:0.100389",
    "public_cv_traj_mae_exceeded:0.131553"
  ],
  "public_input_sha256": "0c01b39b6083bb867e2ca0df887384f09b6f8baeee1fa7c40d8ab692ba6849f4",
  "candidate_raw_sha256": "6e3f174081f1e338179abaa31464da93f2271be17a4db1c562a0e2bb5cea59e7",
  "setup_source_sha256": "71546bb4bd8a148220345c229cb5b11ce83111110a36cc49ca8b68b476e68775",
  "cv_metrics": {
    "public_cv_y_mae": 0.100389,
    "public_cv_traj_mae": 0.131553
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
  "wall_ms": 2114.79
}No matching skill cards found.
```

### Turn 7 — `execute_in_remote_docker` (`37657 ms` LLM)
- **Rationale**: Quantify inverse consistency and screen explicit panel-C and symmetric features before fitting the scalar head and constructing the trajectory representation.

```bash
cat > /tmp/model23.py <<'PY'
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def D(x):return np.c_[np.ones(len(x)),x]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def poly(x):return D(np.c_[x,x*np.roll(x,-1,axis=1),x**3])
def get(s):
 tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; u,v=(s['panel_a'][k] for k in ('u','v')); A=poly(u); keep=np.ones(len(u),bool)
 for _ in range(12):
  w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e));
  if new.sum()<18 or np.array_equal(new,keep):break
  keep=new
 z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for i in range(20):
  K=w[5:9]; J=w[1:5]+np.roll(z,-1,axis=1)[:,:,None]*K+np.roll(z[:,:,None]*K,1,axis=1)+3*z[:,:,None]**2*w[9:13]; z-=np.clip(np.linalg.solve(J.transpose(0,2,1),(poly(z)@w-x)[...,None])[...,0],-.4,.4)
 u,v=(s['panel_b'][k] for k in ('u','v'))
 def H(x,q):return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1])
 def r(q):
  F=H(u,q);return (F@fit(F,v)-v).ravel()
 q=least_squares(r,[.35,1.5],bounds=([.1,.5],[.7,3])).x; b=H(z,q)@fit(H(u,q),v)
 p=s['panel_c']; u=p['u']; K=fit(poly(u),p['v1']); c=lambda x:poly(x)@K
 t=s['eval_times']; T=tr['traj'].T
 def F(q):return -np.expm1(-t[:,None]*q)
 def e(q):
  f=F(q);return (f@fit(f,T)-T).ravel()
 q=least_squares(e,[1.8,.42],bounds=([1.,.15],[3.,.85])).x; f=F(q); targets=np.c_[tr['y'],fit(f,T).T]
 return z,b,c,f,targets, (w,keep,x-poly(z)@w,u,p['v1'],p['v2'],q,e(q))

if __name__=='__main__':
 from probe23 import S
 z,b,c,f,Y,aux=get(S); w,keep,ie,u,v,h,q,te=aux
 print('A',keep.sum(),w.round(3),'inverse max',abs(ie).max(),'z range',z.min(),z.max(),'B',b.min(),b.max(),'C',fit(poly(u),v).round(3),'t',q,'temporal rmse',np.mean(te**2)**.5,'target std',Y.std(0));print('C ranges',u.min(),u.max(),v.min(),v.max(),h.min(),h.max()); n=52; N=len(Y)
 def forms(x):
  yield 'linear',x
  yield 'diag2',np.c_[x,x*x]
  yield 'cube',np.c_[x,x*x,x**3]
  for k in (.5,1,1.25,1.5,2.15,3):
   for nm,g in [('tanh',np.tanh),('erf',erf),('sin',np.sin),('soft',lambda v:v/np.sqrt(1+v*v)),('rational',lambda v:v/(1+v*v)),('asinh',np.arcsinh)]:
    a=g(k*x); yield f'{nm}{k}',np.c_[x,a,a*a]
 def pol2(x):return np.column_stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
 def test(items,tar,n):
  rows=[]
  for nm,x in items:
   for nn,xx in forms(x):
    for pp,X in [('main',xx),('all2',np.c_[xx,pol2(x)])]:
     F=D(X); p=F[n:len(tar)]@fit(F[:n],tar[:n]); er=abs(p-tar[n:]); rows.append((np.mean(er[:,0]),np.mean(abs(er[:,1:]@f.T)) if tar.shape[1]==3 else np.mean(er),nm,nn,pp))
  for idx in [0,1]:print(sorted(rows,key=lambda a:a[idx])[:7]);
 print('C2 regression');test([('u',u),('v1',v),('u-v',u-v),('u+v',u+v),('u,v',np.c_[u,v])],h,19)
 print('Targets');test([('z',z),('b',b),('cz',c(z)),('cb',c(b)),('z,b',np.c_[z,b]),('z,cz',np.c_[z,c(z)]),('b,cz',np.c_[b,c(z)]),('b,cb',np.c_[b,c(b)]),('b+z',b+z),('b-z',b-z),('cz+b',c(z)+b),('cz-b',c(z)-b)],Y,n)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/model23.py
```
- **Container Exit Code**: `0` (`722 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.     0.    -0.   ]
 [-0.     0.     0.    -0.   ]
 [-0.835  0.733 -0.103  0.487]
 [-0.88  -0.612  0.058 -0.575]
 [ 0.057  0.133 -1.159 -0.346]
 [-0.089 -0.744 -0.355  0.892]
 [ 0.    -0.    -0.    -0.   ]
 [ 0.    -0.     0.    -0.   ]
 [-0.    -0.    -0.     0.   ]
 [ 0.    -0.    -0.     0.   ]] t [1.7698855  0.41101944] temporal rmse 0.00449167711245413 target std [0.58579359 0.26117676 1.43750067]
C ranges -1.2476408773255074 1.2239082602152336 -1.6648058732965012 1.811933061486131 -1.1251442782414522 0.6930580548967048
C2 regression
[(0.0006730366345079258, 0.0006690640879899242, 'u', 'tanh0.5', 'main'), (0.0013034624914565987, 0.0012003304550483418, 'u', 'erf0.5', 'main'), (0.0058459788770908305, 0.00450800365339078, 'u', 'sin0.5', 'main'), (0.005931930243645829, 0.004792525904933134, 'u', 'soft0.5', 'main'), (0.006021317283035709, 0.004661938206961319, 'u', 'asinh1', 'main'), (0.007039880360613916, 0.005551390886841399, 'u', 'asinh0.5', 'main'), (0.008369148888523527, 0.007236905119840027, 'u', 'diag2', 'main')]
[(0.0006730366345079258, 0.0006690640879899242, 'u', 'tanh0.5', 'main'), (0.0013034624914565987, 0.0012003304550483418, 'u', 'erf0.5', 'main'), (0.0058459788770908305, 0.00450800365339078, 'u', 'sin0.5', 'main'), (0.006021317283035709, 0.004661938206961319, 'u', 'asinh1', 'main'), (0.005931930243645829, 0.004792525904933134, 'u', 'soft0.5', 'main'), (0.007039880360613916, 0.005551390886841399, 'u', 'asinh0.5', 'main'), (0.008369148888523527, 0.007236905119840027, 'u', 'diag2', 'main')]
Targets
[(0.11271572372171659, 0.24810340596787844, 'b-z', 'asinh3', 'all2'), (0.11352744723229877, 0.26272804569012626, 'b-z', 'soft1.25', 'all2'), (0.11604609395606665, 0.2653077346430188, 'b-z', 'tanh1.5', 'all2'), (0.11646862053145335, 0.2553041162863374, 'b-z', 'soft1.5', 'all2'), (0.11946121942996127, 0.2495873746125883, 'b-z', 'soft2.15', 'all2'), (0.11968446996821328, 0.266789128416162, 'b-z', 'erf1.5', 'all2'), (0.1197521081911348, 0.261584375123892, 'b-z', 'asinh2.15', 'all2')]
[(0.17634887480632214, 0.17142501679644898, 'z', 'diag2', 'all2'), (0.17634887480632247, 0.17142501679645045, 'z', 'linear', 'all2'), (0.1369631973571402, 0.18011203584576646, 'b-z', 'tanh0.5', 'main'), (0.143489406123235, 0.18262451875118346, 'b-z', 'soft0.5', 'main'), (0.134206560148812, 0.18417521787961097, 'b-z', 'erf0.5', 'main'), (0.15081793376332622, 0.18745558994784342, 'b-z', 'asinh1', 'main'), (0.14086661155191366, 0.1880936807973546, 'b-z', 'asinh0.5', 'main')]
```

### Turn 8 — `execute_in_remote_docker` (`69113 ms` LLM)
- **Rationale**: Identify calibration maps from panel measurements, then rank prediction features on shuffled folds separately from the official public split.
- **Alternatives Weighed**: `Retain the archived kernel and cubic inverse` (rejected: It missed both public MAE thresholds, and improving the head alone would not address the inverse residual.)
- **ATMS NOGOOD DECLARED**: Refuted `The cyclic quadratic plus diagonal cubic panel-A map is adequate for inversion.` | Witness: `Its trimmed calibration RMSE is 0.01857, and its numerical inverse leaves a forward residual of up to 1.112 on observed inputs.` | Blanket Diagnosis: `A truncated cubic can introduce turning points and corrupt latent coordinates before either prediction head.` | Sublated Pivot: `Retain robust trimming and the exact threshold-power panel B; profile saturation families for panel A and test panel-C compositions.`

```bash
cat > /tmp/exact23.py <<'PY'
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from model23 import D,fit,get
from probe23 import S

def sat(x,a,k):
 if k=='tanh': return np.tanh(a*x)
 if k=='erf': return erf(a*x)
 if k=='sin': return np.sin(a*x)
 if k=='rat': return x/(1+(a*x)**2)
 if k=='soft': return x/np.sqrt(1+(a*x)**2)
 if k=='atan': return np.arctan(a*x)

def models(s,verbose=False):
 z0,b0,c,f,Y,aux=get(s); keep=aux[1]; u,v=[s['panel_a'][k][keep] for k in ['u','v']]; x=np.r_[s['train']['x'],s['test']['x']]
 cand=[]
 for k in ['tanh','erf','sin','rat','soft','atan']:
  for typ in [0,1,2]:
   def F(z,a,k=k,typ=typ):
    h=sat(z,a,k); return D(np.c_[z,h] if typ==0 else np.c_[z,h,z*z] if typ==1 else np.c_[z,h,z*np.roll(z,-1,axis=1)])
   def e(a):
    A=F(u,a);return (A@fit(A,v)-v).ravel()
   r=least_squares(e,1.25,bounds=(.1,5));cand.append((np.mean(r.fun**2)**.5,k,typ,r.x,F))
 cand.sort(key=lambda t:t[0]); ee,k,typ,a,F=cand[0]; w=fit(F(u,a),v); z=np.array([least_squares(lambda zz:(F(zz[None,:],a)@w-xx).ravel(),zz,bounds=(-3,3)).x for xx,zz in zip(x,z0)])
 if verbose:print('A',[(round(e,7),k,t,a.round(5)) for e,k,t,a,_ in cand[:10]],'z',z.min(),z.max(),'inverse',np.max(abs(F(z,a)@w-x)), 'w',w.round(4),flush=True)
 u,v=[s['panel_b'][k] for k in ['u','v']]; H=lambda z:D(np.sign(z)*np.maximum(abs(z)-.35,0)**1.5); B=lambda z:H(z)@fit(H(u),v)
 u,h=[s['panel_c'][k] for k in ['u','v2']]; cand=[]
 for k in ['tanh','erf','sin','rat','soft','atan']:
  for typ in range(4):
   def F(z,a,k=k,typ=typ):
    h=sat(z,a,k);return D([h,h*h][0] if typ==0 else np.c_[z,h] if typ==1 else np.c_[h,h*h] if typ==2 else np.c_[z,h,h*h])
   def e(a):
    A=F(u,a);return (A@fit(A,h)-h).ravel()
   r=least_squares(e,1.25,bounds=(.1,5));cand.append((np.mean(r.fun**2)**.5,k,typ,r.x,F))
 cand.sort(key=lambda t:t[0]); ee,k,typ,a,F=cand[0];w=fit(F(u,a),h); E=lambda z:F(z,a)@w
 if verbose:print('C2',[(round(e,7),k,t,a.round(5)) for e,k,t,a,_ in cand[:10]],w.round(4),flush=True)
 return z,B,c,E,f,Y

if __name__=='__main__':
 z,B,C,E,f,Y=models(S,True); b=B(z); n=len(Y); rng=np.random.default_rng(171); splits=[]
 for rep in range(2):
  ii=rng.permutation(n)
  for ts in np.array_split(ii,4):splits.append((np.setdiff1d(np.arange(n),ts),ts))
 splits.append((np.arange(52),np.arange(52,n)))
 def score(K,ridge):
  a=[]
  for tr,ts in splits:
   p=K[np.ix_(ts,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+ridge*np.eye(len(tr)),Y[tr]); e=p-Y[ts]; a.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()]);
  a=np.array(a);return a[:-1].mean(0),a[-1]
 base={'z':z,'b':b,'cz':C(z),'cb':C(b),'ez':E(z),'eb':E(b)}
 for a in [-2,-1,-.5,.5,1,2]:
  for name,g in [('i',lambda x:x),('c',C),('e',E)]:base[f'{name}(b{a:+}z)']=g(b+a*z)
 sets=dict(base)
 for a,xa in base.items():
  for b,xb in base.items():
   if a<b and (a in ['z','b','cz','cb','ez','eb'] or b in ['z','b','cz','cb','ez','eb']): sets[a+' '+b]=np.c_[xa,xb]
 sets['all']=np.c_[tuple(base[k] for k in ['z','b','cz','cb','ez','eb'])]
 rows=[]
 for nm,xx in sets.items():
  for gf,x in [('raw',xx),('asinh',np.arcsinh(xx)),('tanh',np.tanh(1.25*xx)),('rational',xx/(1+(1.25*xx)**2))]:
   x=x[:n]; dot=x@x.T/x.shape[1]; dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0)
   for deg,dec in [(1,0),(2,0),(3,0),(2,.5),(3,1.4)]:
    K=(1+dot)**deg+(np.exp(-dec*dist) if dec else 0)
    for r in [1e-6,.001,.02]:
     m,p=score(K,r);rows.append((*m,*p,nm,gf,deg,dec,r))
 for i in range(2):
  print('CV',i,sorted(rows,key=lambda t:t[i])[:12]); print('public',i,sorted(rows,key=lambda t:t[i+2])[:6]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/exact23.py
```
- **Container Exit Code**: `0` (`5099 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
09277214138665091, 0.06528964331168427, 0.10049675479524593, 'e(b+2z) z', 'asinh', 2, 0.5, 1e-06), (0.09027016283671696, 0.08563093594334809, 0.06584545565930996, 0.08468533368218674, 'e(b-0.5z) z', 'raw', 3, 1.4, 1e-06), (0.08724461581921913, 0.09558436019777873, 0.06610993648898222, 0.09539224969560746, 'ez z', 'raw', 3, 1.4, 1e-06)]
CV 1 [(0.08123853451220636, 0.0656411391149319, 0.10667813221688356, 0.09624649974686934, 'cb i(b-1z)', 'raw', 2, 0.5, 1e-06), (0.07845999334852713, 0.06689450623327273, 0.0894700953779493, 0.08066038177108714, 'b i(b-1z)', 'raw', 3, 0, 0.001), (0.06596760895592375, 0.06696152051383324, 0.08662469325047355, 0.08977070726471002, 'eb i(b-0.5z)', 'raw', 2, 0.5, 1e-06), (0.07204752570725, 0.0671486804167195, 0.09309023447330941, 0.10262178164708077, 'b i(b+0.5z)', 'raw', 2, 0.5, 1e-06), (0.0793988769218667, 0.06743344748360684, 0.09043380518465752, 0.07741638151083031, 'b i(b-1z)', 'raw', 3, 1.4, 0.001), (0.08254527194954296, 0.06782677251907311, 0.1061661902853969, 0.08955566211877425, 'b z', 'raw', 3, 0, 0.001), (0.08388072492086825, 0.06816854762421394, 0.10206460112828117, 0.08860839879396112, 'b z', 'raw', 3, 1.4, 0.001), (0.08341538964897077, 0.0689193200385359, 0.11023347467367321, 0.11166099339134535, 'cb z', 'raw', 2, 0.5, 1e-06), (0.06789284918652062, 0.06903411591568494, 0.08714434303793625, 0.08990776282549544, 'b i(b-0.5z)', 'raw', 2, 0.5, 1e-06), (0.06814217778533642, 0.06932135570161789, 0.0828859238421259, 0.09127730100053694, 'eb i(b-0.5z)', 'asinh', 2, 0.5, 1e-06), (0.08262944750096388, 0.0694614173970791, 0.1010134093289746, 0.09055574878989123, 'i(b-0.5z) z', 'raw', 3, 0, 0.001), (0.08058015834034664, 0.07049873395016293, 0.09376269245132618, 0.08464700640110012, 'e(b-0.5z) z', 'raw', 3, 0, 1e-06)]
public 1 [(0.08868382895271386, 0.08625949492425392, 0.10588619818223213, 0.06473830317111891, 'all', 'tanh', 1, 0, 1e-06), (0.07321503044625895, 0.07224157362871324, 0.07400455014325176, 0.06556092841169318, 'e(b-0.5z) z', 'raw', 2, 0, 1e-06), (0.09068088433022553, 0.08804851004523, 0.09589107923931187, 0.06613977373120358, 'i(b-1z)', 'raw', 3, 1.4, 0.001), (0.09384896413170135, 0.08666736553656523, 0.05532124458304273, 0.06753973274153686, 'e(b+1z) z', 'raw', 3, 0, 1e-06), (0.09909741031641361, 0.09599985993600633, 0.07867260905480108, 0.06772074012365284, 'e(b-1z) ez', 'raw', 3, 1.4, 1e-06), (0.10130645314266473, 0.09203634773662087, 0.14232086020125204, 0.07063204647656442, 'cb e(b+0.5z)', 'raw', 3, 0, 0.001)]
```

### Turn 9 — `execute_in_remote_docker` (`65279 ms` LLM)
- **Rationale**: Save a runnable candidate now and test whether sparse selection among explicit panel and symmetry features improves both heads without changing the frozen DAG.
- **Alternatives Weighed**: `Use the best generic kernel from the coordinate sweep` (rejected: Neither head reached the required 0.038 MAE.)

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Fit reference-panel coordinates and learn a compact spatial representation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.linalg import solve

def D(x): return np.c_[np.ones(len(x)),x]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def cyc(z): return z*np.roll(z,-1,axis=1)

def coordinates(s):
    p=s['panel_a']; u,v=p['u'],p['v']; x=np.r_[s['train']['x'],s['test']['x']]
    A=D(np.c_[u,cyc(u),u**3]); keep=np.ones(len(u),bool)
    for _ in range(12):
        w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e))
        if new.sum()<18 or np.array_equal(new,keep): break
        keep=new
    u,v=u[keep],v[keep]
    def form(z,q): return D(np.c_[z,np.sin(q[0]*z),z*z])
    def err(q):
        a=form(u,q); return (a@fit(a,v)-v).ravel()
    q=least_squares(err,[2.15],bounds=(1,4)).x; w=fit(form(u,q),v)
    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for _ in range(14):
        j=w[1:5]+(q[0]*np.cos(q[0]*z))[:,:,None]*w[5:9]+2*z[:,:,None]*w[9:13]
        z-=np.clip(np.linalg.solve(j.transpose(0,2,1),(form(z,q)@w-x)[...,None])[...,0],-.3,.3)
    p=s['panel_b']; H=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); wb=fit(H(p['u']),p['v']); B=lambda a:H(a)@wb
    p=s['panel_c']; u=p['u']; wc=fit(D(cyc(u)),p['v1']); C=lambda a:D(cyc(a))@wc
    def g(a,q): return D(a+q[1]*a*a/(1+q[0]*a*a))
    def e(q):
        a=g(u,q); return (a@fit(a,p['v2'])-p['v2']).ravel()
    q=least_squares(e,[.25,.45],bounds=([0,-3],[8,3])).x; we=fit(g(u,q),p['v2']); E=lambda a:g(a,q)@we
    return z,B,C,E

def features(z,B,C,E):
    b=B(z); c=C(z); e=E(z)
    basic=[z,b,c,e,B(c),B(e),C(b),E(b),C(e),E(c)]
    f=basic.copy()
    for x in [z,b,c,e,z+b,z-b,b-.5*z,c+e,c-e,b+c,b-c,b+e,b-e,z+c,z-c,z+e,z-e]:
        f.extend([x*x,cyc(x),x*np.roll(x,2,axis=1)])
        for a in [.65,1.25,2.15]:
            f.extend([np.sin(a*x)-a*x,np.tanh(a*x)-a*x,x/(1+(a*x)**2)-x,np.arctan(a*x)**2])
    for i,a in enumerate(basic[:4]):
        for b1 in basic[i:4]:
            f.append((a[:,:,None]*b1[:,None,:]).reshape(len(z),-1))
    return np.column_stack(f)

def head(F,Y):
    n=len(Y); A=F[:n]; mean=A.mean(0); scale=np.maximum(A.std(0),.015); X=(F-mean)/scale; A=X[:n]; m=Y.mean(0); y=Y-m
    # Sparse Bayesian regression, with shared scale evidence within each response.
    result=[]
    for k in range(y.shape[1]):
        t=y[:,k]; means=[]
        for power in [0.,.5,1.]:
            a=scale**power; a/=np.median(a); Q=A*a; beta=np.ones(Q.shape[1])/Q.shape[1]; noise=max(.0001*np.var(t),1e-9)
            for _ in range(60):
                W=Q*beta; inv=solve(W@Q.T+noise*np.eye(n),np.c_[t,Q],assume_a='pos'); w=beta*(Q.T@inv[:,0]); lev=np.sum(Q*inv[:,1:],axis=0); proposed=abs(w)/np.sqrt(np.maximum(lev,1e-10)); beta=np.maximum(1e-11,.35*beta+.65*proposed)
            means.append(X@(a*w)+m[k])
        result.append(np.mean(means,axis=0))
    return np.column_stack(result)

def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; z,B,C,E=coordinates(s); tr=s['train']; t=s['eval_times']; T=tr['traj'].T
    def basis(q): return -np.expm1(-t[:,None]*q)
    def err(q):
        a=basis(q); return (a@fit(a,T)-T).ravel()
    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
    modes=fit(a,T).T; n=len(modes); F=features(z,B,C,E)
    p=head(F,np.c_[tr['y'],modes]); scale=np.maximum(modes.std(0),1e-5)
    return {'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/scale,.035*z,.035*B(z)]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys
sys.path[:0]=['/tmp','/holdout_task'];from probe23 import S;from node_00 import *
z,B,C,E=coordinates(S);u=S['panel_c']['u'];print('C2 RMSE',np.mean((E(u)-S['panel_c']['v2'])**2)**.5)
PY
```
- **Container Exit Code**: `0` (`2282 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.103720", "public_cv_traj_mae_exceeded:0.058292"], "cv_metrics": {"public_cv_y_mae": 0.10372, "public_cv_traj_mae": 0.058292}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
C2 RMSE 0.0002517676141295486

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d86d9d46018a167a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d86d9d46 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`51211 ms` LLM)
- **Rationale**: Fit covariance weights on training data only to identify which panel compositions explain each output, then check both the official split and a separate interleaved split.
- **Alternatives Weighed**: `Select the best single kernel on the public split` (rejected: The best screened single kernels missed both thresholds, and selecting on that split would overstate generalization.)

```bash
cp /holdout_task/node_00.py /tmp/base23.py; for f in /tmp/hegel_continual_archive/*solver.py; do echo ====; tail -31 "$f"; done; sed -n '156,200p' /holdout_task/test_dag_candidate.py; cat > /tmp/search23.py <<'PY'
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve, solve
from scipy.special import erf
from base23 import coordinates, D, cyc, head

def banks(z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    q={'z':z,'b':b,'c':c,'e':e}
    for nm,x in list(q.items()):
        if nm!='z':
            for name,func in [('b',B),('c',C),('e',E)]:q[name+nm]=func(x)
    for nm,x in [('z',z),('b',b),('c',c),('e',e)]:
        for nn,y in [('z',z),('b',b),('c',c),('e',e)]:
            if nm<nn:
                for a in [-1.,1.]:
                    x1=x+a*y; k=f'{nm}{a:+}{nn}';q[k]=x1
                    for name,func in [('b',B),('c',C),('e',E)]:q[name+k]=func(x1)
    g={}
    for nm,x in q.items():
        i,j=np.triu_indices(x.shape[1],1)
        g[nm+'lin']=x
        g[nm+'quad']=np.c_[x*x,x[:,i]*x[:,j]]
        g[nm+'cyc']=cyc(x)
        for a in [.65,1.25,2.15]:
            for k,h in [('tanh',np.tanh(a*x)),('erf',erf(a*x)),('sin',np.sin(a*x)),('rat',x/(1+(a*x)**2)),('th',np.sign(x)*np.maximum(abs(x)-.22,0)**1.25)]:
                g[nm+k+str(a)]=h
                if a==1.25:g[nm+k+'2']=h*h
    for i,(nm,x) in enumerate(list(q.items())[:13]):
        for nn,y in list(q.items())[i+1:13]:
            g[nm+'*'+nn]=(x[:,:,None]*y[:,None,:]).reshape(len(z),-1)
    return g

def covs(g,n):
    groups=[];names=[]
    for name,F in g.items():
        F=F-F[:n].mean(0); scale=np.maximum(F[:n].std(0),.008); F=F/np.sqrt(scale); K=F@F.T
        K/=max(np.mean(K.diagonal()[:n]),1e-10);groups.append(K);names.append(name)
    # Broad smooth components as a fallback.
    for name in ['z','b','c','e','bz','cb','eb','ez','cz','bz-1.0e']:
        if name+'lin' not in g:continue
        x=g[name+'lin']; x=(x-x[:n].mean(0))/np.maximum(x[:n].std(0),.01)
        dot=x@x.T/x.shape[1]; dist=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0)
        for a in [.08,.3,1.2]:groups.append(np.exp(-a*dist));names.append(name+'rbf'+str(a))
    return names,np.array(groups)

def regress(K,Y,verbose=False,names=None):
    n=len(Y); avg=Y.mean(0); sd=np.maximum(Y.std(0),1e-9); Y=(Y-avg)/sd
    flat=np.ascontiguousarray(K[:,:n,:n].reshape(len(K),-1)); cross=np.ascontiguousarray(K[:,n:,:n].reshape(len(K),-1)); eye=np.eye(n); ps=[]
    for t in Y.T:
        def obj(w):
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye
            c=cho_factor(V,check_finite=False); a=cho_solve(c,t,check_finite=False); W=cho_solve(c,eye,check_finite=False)-a[:,None]*a[None,:]
            return np.log(c[0].diagonal()).sum()+.5*(t@a), np.r_[.5*(flat@W.ravel()),.5*W.trace()]
        pp=[]
        for penalty in [0.,2.,8.]:
            def fn(w):
                l,d=obj(w);return l+penalty*np.sum(w[:-1]),d+np.r_[np.full(len(K),penalty),0.]
            init=np.r_[np.full(len(K),1./len(K)),.015]; w=minimize(fn,init,jac=True,method='L-BFGS-B',bounds=[(0,15.)]*len(K)+[(.000025,1.)],options={'maxiter':180,'ftol':4e-8}).x
            if verbose and penalty==2.:print('groups',[(names[i],round(w[i],3)) for i in np.argsort(w[:-1])[-12:][::-1] if w[i]>.0001], 'noise',w[-1],flush=True)
            V=(w[:-1]@flat).reshape(n,n)+(w[-1]+1e-7)*eye; cross1=(w[:-1]@cross).reshape(-1,n); a=solve(V,t,assume_a='pos'); pp.append(np.r_[t-w[-1]*a,cross1@a]);
        ps.append(np.mean(pp,axis=0))
    return np.column_stack(ps)*sd+avg

if __name__=='__main__':
    from probe23 import S
    from model23 import get
    z,B,C,E=coordinates(S); _,_,_,f,Y,_=get(S); u=S['panel_a']['u']; v=S['panel_a']['v']; print('panels',S.keys(),[(k,{j:(a.shape,np.round(np.array([a.min(),a.max()]),4)) for j,a in S[k].items()}) for k in ['panel_a','panel_b','panel_c']]);g=banks(z,B,C,E)
    for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(72)[np.arange(72)%4!=0],np.arange(0,72,4))]:
        n=len(tr); ids=np.r_[tr,va]; names,K=covs({name:a[ids] for name,a in g.items()},n); P=regress(K,np.c_[Y[:,0],Y[:,1:]@np.linalg.svd(f,full_matrices=False)[2].T][tr],True)[n:]; err=Y[va,1:]@np.linalg.svd(f,full_matrices=False)[2].T-P[:,1:]; print('MAE',np.mean(abs(P[:,0]-Y[va,0])),np.mean(abs(err@np.linalg.svd(f,full_matrices=False)[2]@f.T)),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/search23.py
```
- **Container Exit Code**: `1` (`2011 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, True),
             (np.tanh(1.75*b), .0003, True, False),
             (np.arctan(1.75*b), .0003, True, False),
             (b / (1 + b*b), .003, False, True),
             (np.tanh(1.4*b), .003, False, True),
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
panels dict_keys(['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec']) [('panel_a', {'u': ((48, 4), array([-1.298 ,  1.2867])), 'v': ((48, 4), array([-6.7744,  7.7657]))}), ('panel_b', {'u': ((24, 4), array([-1.2324,  1.2406])), 'v': ((24, 4), array([-1.0185,  1.202 ]))}), ('panel_c', {'u': ((24, 4), array([-1.2476,  1.2239])), 'v1': ((24, 4), array([-1.6648,  1.8119])), 'v2': ((24, 4), array([-1.1251,  0.6931]))})]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d86d9d46018a167a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d86d9d46 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/search23.py", line 70, in <module>
    n=len(tr); ids=np.r_[tr,va]; names,K=covs({name:a[ids] for name,a in g.items()},n); P=regress(K,np.c_[Y[:,0],Y[:,1:]@np.linalg.svd(f,full_matrices=False)[2].T][tr],True)[n:]; err=Y[va,1:]@np.linalg.svd(f,full_matrices=False)[2].T-P[:,1:]; print('MAE',np.mean(abs(P[:,0]-Y[va,0])),np.mean(abs(err@np.linalg.svd(f,full_matrices=False)[2]@f.T)),flush=True)
                                                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/search23.py", line 60, in regress
    if verbose and penalty==2.:print('groups',[(names[i],round(w[i],3)) for i in np.argsort(w[:-1])[-12:][::-1] if w[i]>.0001], 'noise',w[-1],flush=True)
                                                ~~~~~^^^
TypeError: 'NoneType' object is not subscriptable
```

### Turn 11 — `execute_in_remote_docker` (`57716 ms` LLM)
- **Rationale**: Select the scalar blend and trajectory design separately using held-out fold errors, then run the official validator in a fresh isolated Python process against the exact saved artifact.
- **Alternatives Weighed**: `Promote a model based solely on the official split` (rejected: A single split does not establish whether either head improves on other held-out samples.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed -i 's/\[tr\],True)\[n:\]/[tr],True,names)[n:]/' /tmp/search23.py
cat /holdout_task/_frozen_nodes/node_02.py
grep -n sp_idx /holdout_task/test_dag_candidate.py
cat > /tmp/finalize23.py <<'PY'
from pathlib import Path
base=Path('/tmp/base23.py').read_text().split('\ndef run_node')[0]
search=Path('/tmp/search23.py').read_text().split('\ndef banks')[1].split("\nif __name__")[0]
search='\ndef banks'+search
search=search.replace('for penalty in [0.,2.,8.]:','for penalty in [0.,4.]:').replace("'maxiter':180", "'maxiter':145")
extra='''
def extra_groups(g,z,B,C,E):
    b,c,e=B(z),C(z),E(z)
    vals=[z,b,c,e,b-.5*z,b-z,b+z,b+c,b-c,b+e,b-e,e+c,e-c]
    for i,x in enumerate(vals):
        opp=np.roll(x,2,axis=1); nxt=np.roll(x,-1,axis=1)
        for j,v in enumerate([opp,nxt]):
            for s in [-1,1]:
                u=x+s*v
                for k,a in enumerate([.65,1.25,2.15]):
                    h=np.tanh(a*u); r=u/(1+(a*u)**2)
                    g[f'p{i},{j},{s},{k}']=np.c_[h,r,h*h,r*r]
            g[f'triple{i},{j}']=x*v*np.roll(x,j+1,axis=1)
        for k,a in enumerate([.65,1.25,2.15]):
            h=np.arcsinh(a*x); r=x/(1+(a*x)**2)
            g[f'quot{i},{k}']=np.c_[h,h*h,x*h,r,r*r]
    for i,x in enumerate(vals[:4]):
        for j,v in enumerate(vals[:4]):
            if i!=j:
                g[f'mix{i},{j}']=np.c_[x/(1+1.25*v*v),x/(1+2.15*v*v),x*np.tanh(1.25*v),x*x*v]
    return g

def predict_local(s,variant):
    z,B,C,E=coordinates(s); tr=s['train']; t=s['eval_times']; T=tr['traj'].T
    def basis(q): return -np.expm1(-t[:,None]*q)
    def err(q):
        a=basis(q); return (a@fit(a,T)-T).ravel()
    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
    modes=fit(a,T).T; n=len(modes); rotation=np.linalg.svd(a,full_matrices=False)[2].T
    target=np.c_[tr['y'],modes@rotation]
    g=banks(z,B,C,E)
    if variant==2:g=extra_groups(g,z,B,C,E)
    names,K=covs(g,n)
    p=regress(K,target)
    p[:,1:]=p[:,1:]@rotation.T
    return p,modes,z,B(z)

def run_node(inputs: dict) -> dict:
    p,modes,z,b=predict_local(inputs['__input__'],VARIANT)
    n=len(modes)
    scale=np.maximum(modes.std(0),1e-5)
    return {'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/scale,.035*z,.035*b]}
'''
Path('/tmp/challenger23.py').write_text(base+'\nfrom scipy.optimize import minimize\nfrom scipy.linalg import cho_factor,cho_solve\n'+search+extra+'\nVARIANT=1\n')
PY
python3 /tmp/finalize23.py
cat > /tmp/compare23.py <<'PY'
import sys,pickle,re,runpy,numpy as np
sys.path.insert(0,'/tmp')
import challenger23 as m
import base23 as old
from pathlib import Path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; N=len(a['y']); i=np.arange(N)
text=Path('/holdout_task/test_dag_candidate.py').read_text(); print([l for l in text.splitlines() if 'sp_idx =' in l],flush=True)
match=re.search(r'sp_idx\s*=\s*int\([^\n]*?\*\s*(0\.\d+)',text); n=int(N*float(match[1])) if match else 52
nodes=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%j)['run_node'] for j in [1,2,3]]
def check(tr,va):
    sub={**s,'train':{'x':a['x'][tr],'y':a['y'][tr],'traj':a['traj'][tr]},'test':{'x':a['x'][va]}}
    sub['train']['summary']=np.array([a['y'][tr].mean(),a['y'][tr].std(),a['traj'][tr,-1].mean(),a['traj'][tr,-1].std()]); args={'__input__':sub}; q=nodes[0](args); outs=[]
    for v in [0,1,2]:
        m.VARIANT=v; r=(old if v==0 else m).run_node(args)
        o=nodes[2]({'node_00':r,'node_02':nodes[1]({'node_00':r,'node_01':q})}); outs.append((o['y_hat'],o['state_pred']))
    return outs
splits=[]
for off in range(4):
    va=i[i%4==off]; tr=i[i%4!=off]; outs=check(tr,va); splits.append((va,outs)); print('independent',off,[(float(abs(y-a['y'][va]).mean()),float(abs(t-a['traj'][va]).mean())) for y,t in outs],flush=True)
def mix(items, weights, j):return sum(w*r[j] for w,r in zip(weights,items))
opts=[np.eye(3)[k] for k in range(3)]+[np.array(t) for t in [(0,.5,.5),(.2,.4,.4),(.5,.25,.25),(0,.25,.75),(0,.75,.25)]]
w=[]
for j,label in enumerate(['y','traj']):
    ranked=[]
    for weights in opts:
        errs=np.array([abs(mix(o,weights,j)-a[label][va]).mean() for va,o in splits]); ranked.append((.65*errs.mean()+.35*errs.max(),weights,errs))
    _,b,e=min(ranked,key=lambda k:k[0]); w.append(b); print(label,'weights',b,'mean/worst',e.mean(),e.max(),flush=True)
# The frozen trajectory kernel is nonlinear in the design: promote a single trajectory model.
j=1
idx=min(range(3),key=lambda k:.65*np.mean([abs(o[k][j]-a['traj'][va]).mean() for va,o in splits])+.35*max(abs(o[k][j]-a['traj'][va]).mean() for va,o in splits))
print('selected trajectory',idx,flush=True)
code=Path('/tmp/challenger23.py').read_text().split('\ndef run_node')[0]; code+='\n'+Path('/tmp/base23.py').read_text().split('\ndef run_node')[1].join(['def _baseline','']) if False else ''
base=Path('/tmp/base23.py').read_text().split('\ndef run_node')[1]
code+='\ndef _baseline'+base
code+='\n_Y_WEIGHTS='+repr(w[0].tolist())+'\n_TRAJECTORY='+str(idx)+'\n'
code+='''
def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; pred=None; design=None
    for k,weight in enumerate(_Y_WEIGHTS):
        if not weight and k!=_TRAJECTORY:
            continue
        if k==0:
            out=_baseline(inputs)
        else:
            p,m,z,b=predict_local(s,k); n=len(m)
            out={'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/np.maximum(m.std(0),1e-5),.035*z,.035*b]}
        if weight:
            pred=weight*out['y_hat'] if pred is None else pred+weight*out['y_hat']
        if k==_TRAJECTORY: design=out['trajectory_design']
    return {'y_hat':pred,'trajectory_design':design}
'''
fp=Path('/holdout_task/node_00.py'); fp.write_text(code)
PY
python3 /tmp/compare23.py
python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `1` (`48823 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
rnel(x, degree, decay):
    dot = x @ x.T / x.shape[1]
    dist = np.maximum(np.diag(dot)[:, None] + np.diag(dot) - 2 * dot, 0)
    return (1 + dot)**degree + np.exp(-decay * dist)


def run_node(inputs: dict) -> dict:
    design = inputs['node_00']['trajectory_design']
    temporal = inputs['node_01']
    modes = temporal['modes']
    n = len(modes)

    def head(decay):
        k = kernel(design, 3, decay)
        return k[n:, :n] @ np.linalg.solve(k[:n, :n] + 1e-06 * np.eye(n), modes)

    state = (0.25 * head(0.15) + 0.75 * head(1.4)) @ temporal['basis'].T
    return {'state_pred': state}
152:    sp_idx = max(40, len(st["train"]["x"]) - 20)
153:    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
155:        "x": st["train"]["x"][:sp_idx].copy(),
156:        "y": st["train"]["y"][:sp_idx].copy(),
157:        "traj": st["train"]["traj"][:sp_idx].copy(),
158:        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
['    sp_idx = max(40, len(st["train"]["x"]) - 20)']
independent 0 [(0.042616747589414763, 0.041182476044745515), (0.046667012796323, 0.035114470853011996), (0.043296169831209035, 0.025605299305253584)]
independent 1 [(0.0507767011508596, 0.05453306107406216), (0.04179091744494856, 0.03211726887682125), (0.0511775982153411, 0.02889288654947548)]
independent 2 [(0.0830561301282145, 0.07716450523566884), (0.07507382945946701, 0.038476273723294596), (0.11368919827926804, 0.06243902281853327)]
independent 3 [(0.09302628078454192, 0.06951789705806401), (0.04022824688592298, 0.028830195604699602), (0.04751123597726543, 0.05484346885208104)]
y weights [0. 1. 0.] mean/worst 0.05094000164666539 0.07507382945946701
traj weights [0. 1. 0.] mean/worst 0.033634552264456864 0.038476273723294596
selected trajectory 1
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.083388", "public_cv_traj_mae_exceeded:0.050877"], "cv_metrics": {"public_cv_y_mae": 0.083388, "public_cv_traj_mae": 0.050877}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d09cff22350bd4b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d09cff22 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`60241 ms` LLM)
- **Rationale**: Test whether training-only kernel stacking and a regularized trajectory embedding improve held-out errors. Keep the existing artifact if neither head improves the independent-fold metric without regressing on the public split, then rerun the official validator.
- **Alternatives Weighed**: `Submit the current artifact as accepted` (rejected: The validator rejected it on both public MAE thresholds.)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; cp /holdout_task/node_00.py /tmp/v02_final23.py; cat > /tmp/augment23.py <<'PY'

# Penalized kernel heads with training-only leave-one-out stacking.
def _stack23(z,B,C,E,Y):
    from scipy.optimize import minimize
    n=len(Y); N=len(z); I=np.eye(n); b,c,e=B(z),C(z),E(z)
    mu=Y.mean(0); sd=np.maximum(Y.std(0),1e-8); T=(Y-mu)/sd
    coords=[z,b,c,e,np.c_[z,e],np.c_[z,b],np.c_[b,E(b-.5*z)],np.c_[C(b),E(b-z)],np.c_[z,E(b+z)],np.c_[z,b,c,e],np.c_[b,C(b),E(b)],np.c_[z,C(b),E(b)],np.c_[z, B(z+c), C(b+z), E(b+z)]]
    gram=[]
    for x in coords:
        for scheme in (0,1):
            X=x-x[:n].mean(0)
            if scheme: X=X/np.sqrt(np.maximum(X[:n].std(0),.04))
            G=X@X.T; G/=max(G.diagonal()[:n].mean(),1e-8)
            d=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)
            gram.extend([(1+.35*G)**2,(1+.35*G)**3,(1+.35*G)**4,(1+.35*G)**2+.3*np.exp(-.3*d)])
    loo=[]; val=[]
    for G in gram:
        for ridge in (1e-6,3e-5,.001,.025):
            inv=solve(G[:n,:n]+ridge*I,I,assume_a='pos'); a=inv@T
            loo.append(T-a/inv.diagonal()[:,None]); val.append(G[:,:n]@a)
    L=np.array(loo); V=np.array(val); out=[]
    for j in range(T.shape[1]):
        U=L[:,:,j].T; t=T[:,j]; errs=np.mean(np.abs(U-t[:,None]),axis=0); ix=np.argsort(errs)[:16]; U=U[:,ix]; m=len(ix)
        def objective(w):
            err=U@w-t; q=np.sqrt(err*err+.04**2)
            return q.mean()+.02*np.dot(w,w), U.T@(err/q)/n+.04*w
        res=minimize(objective,np.ones(m)/m,method='SLSQP',jac=True,bounds=[(0,1)]*m,constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)}],options={'ftol':1e-9,'maxiter':80})
        out.append(res.x@V[ix,:,j])
    return np.column_stack(out)*sd+mu


def _predict23(s):
    z,B,C,E=coordinates(s); tr=s['train']; times=s['eval_times']; T=tr['traj'].T
    def basis(q):return -np.expm1(-times[:,None]*q)
    def err(q):
        f=basis(q);return (f@fit(f,T)-T).ravel()
    f=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x); modes=fit(f,T).T
    U,sv,R=np.linalg.svd(f,full_matrices=False); rot=R.T; target=np.c_[tr['y'],modes@rot]
    _,K=covs(banks(z,B,C,E),len(T.T)); p=regress(K,target); h=_stack23(z,B,C,E,target)
    b=B(z); x=np.c_[z,b]; n=len(modes); d=((x[:,None,:]-x[None,:n,:])**2).sum(2); near=np.exp(-d/.20)
    return p,h,rot,modes,z,b,near

def _emit23(cache, yw,tw,rw):
    p,h,rot,m,z,b,near=cache; n=len(m); M=((1-tw)*p[:,1:]+tw*h[:,1:])@rot.T
    if rw:
        # Rescale for the frozen kernel's dimension-normalized dot product.
        q=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]
        design=np.c_[q*np.sqrt((q.shape[1]+near.shape[1])/q.shape[1]),rw*np.sqrt(q.shape[1]+near.shape[1])*near]
    else: design=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]
    return {'y_hat':(1-yw)*p[n:,0]+yw*h[n:,0],'trajectory_design':design}
PY
cat > /tmp/ratchet23.py <<'PY'
import numpy as np, pickle, runpy, sys
from pathlib import Path
base=Path('/tmp/v02_final23.py').read_text().split('\ndef run_node')[0]; extra=Path('/tmp/augment23.py').read_text(); ns={};exec(base+extra,ns)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; i=np.arange(len(a['y'])); f=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%k)['run_node'] for k in (1,2)]
ys=[0.,.25,.5,.75,1.]; ts=[(t,r) for t in (0.,.25,.5,1.) for r in (0.,.02,.06,.15)]
def eval_split(tr,va):
    ss={**s,'train':{k:a[k][tr] for k in ('x','y','traj')},'test':{'x':a['x'][va]}}; d=ns['_predict23'](ss); q=f[0]({'__input__':ss})
    ey=np.array([abs(ns['_emit23'](d,y,0.,0.)['y_hat']-a['y'][va]).mean() for y in ys]); et=[]
    for t,r in ts:
        o=ns['_emit23'](d,0.,t,r); et.append(abs(f[1]({'node_00':o,'node_01':q})['state_pred']-a['traj'][va]).mean())
    return ey,np.array(et)
R=[]
for off in range(4):
    R.append(eval_split(i[i%4!=off],i[i%4==off])); print('stress',off,R[-1][0],R[-1][1],flush=True)
P=eval_split(i[:-20],i[-20:]); print('public diagnostics',P,flush=True)
opts=[]
for j in range(2):
    A=np.array([r[j] for r in R]); metrics=.65*A.mean(0)+.35*A.max(0)
    # Only consider gains on the independent folds; allow some fold variability.
    allowed=(metrics<=metrics[0]*1.015)&(P[j]<=P[j][0]+1e-7)
    order=np.where(allowed,metrics,np.inf); k=int(np.argmin(order)); opts.append(k); print('promotion',j,k,'mean',A[:,k].mean(),'worst',A[:,k].max(),'public',P[j][k],flush=True)
y=ys[opts[0]]; t,r=ts[opts[1]]
if any(opts):
    code=base+extra+'\n\ndef run_node(inputs: dict) -> dict:\n    return _emit23(_predict23(inputs["__input__"]), '+repr(y)+', '+repr(t)+', '+repr(r)+')\n'; Path('/holdout_task/node_00.py').write_text(code)
PY
python3 /tmp/ratchet23.py; python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`33500 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
stress 0 [0.04666701 0.03768807 0.02885829 0.02070926 0.01513431] [0.03511447 0.03539618 0.03539877 0.03659236 0.02823151 0.02878963
 0.02887371 0.0301901  0.02157351 0.02245094 0.02249556 0.02383558
 0.01135924 0.0115572  0.01112317 0.01306611]
stress 1 [0.04179092 0.03552885 0.02926678 0.02319618 0.01778317] [0.03211727 0.03578589 0.03736167 0.04159008 0.02487188 0.02897278
 0.03063689 0.0349997  0.01921371 0.02316557 0.0247329  0.02912119
 0.01641088 0.01877872 0.02029242 0.0242317 ]
stress 2 [0.07507383 0.0585768  0.04219928 0.02586383 0.01626556] [0.03847627 0.03849742 0.03761813 0.03837389 0.03259505 0.03263009
 0.0325769  0.03448977 0.02824903 0.02816615 0.02910331 0.03185906
 0.03008648 0.02773042 0.02921312 0.03298728]
stress 3 [0.04022825 0.03071623 0.02244046 0.01554524 0.01242084] [0.0288302  0.02935841 0.03005244 0.03371228 0.02215836 0.02265186
 0.0233755  0.0272305  0.01709257 0.01719356 0.01759016 0.0215162
 0.01835474 0.01809488 0.01828309 0.02003321]
public diagnostics (array([0.08338763, 0.06667603, 0.04996443, 0.03434917, 0.01999708]), array([0.05087727, 0.05041511, 0.0510365 , 0.0529664 , 0.04293999,
       0.04271694, 0.04332535, 0.04534956, 0.03623224, 0.03555991,
       0.03613073, 0.03802179, 0.0282497 , 0.02783862, 0.02800335,
       0.02792281]))
promotion 0 4 mean 0.015400970041994023 worst 0.017783173435919484 public 0.019997079053450585
promotion 1 13 mean 0.019040305459223524 worst 0.027730418615010594 public 0.027838616951898893
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.019997, "public_cv_traj_mae": 0.027839}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=282b1eb1395cb7ec (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_282b1eb1 (directory).
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
