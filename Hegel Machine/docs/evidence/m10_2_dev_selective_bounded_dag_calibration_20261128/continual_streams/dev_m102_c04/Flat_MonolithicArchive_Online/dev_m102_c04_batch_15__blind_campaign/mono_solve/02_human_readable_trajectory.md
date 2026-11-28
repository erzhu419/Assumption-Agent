# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c04_batch_15__blind_campaign`

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
| **Turns Used / Max Turns** | `9 / 14` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `245,363` (`237,783` / `101,376` / `7,580`) |
| **Total Episode Wall Time (s)** | `246.87s` |

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

### Turn 1 — `execute_in_remote_docker` (`9822 ms` LLM)
- **Rationale**: Inspect the public schema and prior public validation before selecting or refitting a graph.

```bash
ls -lah /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json; cat /holdout_task/validate_public_schema.py; head -c 12000 /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; echo; head -c 18000 /tmp/hegel_continual_archive/archive_manifest.json; echo; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment robust regression'
```
- **Container Exit Code**: `0` (`338 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"stat_14": 1.25,
      "stat_15": 0.22,
      "stat_16": 1.25,
      "stat_17": 2.15
    }
  },
  {
    "short_name": "dev_m102_c04_batch_13",
    "task_id": "dev_m102_c04_batch_13__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/dev_m102_c04_batch_13_solver.py",
    "stat_summary": {
      "stat_00": 1.0,
      "stat_01": 3.6e-05,
      "stat_02": 0.0,
      "stat_03": 0.811715,
      "stat_04": 1.0,
      "stat_05": 0.0,
      "stat_06": 0.0,
      "stat_07": 0.632795,
      "stat_08": 0.967499,
      "stat_09": 0.032313,
      "stat_10": 0.882038,
      "stat_11": 0.069062,
      "stat_12": 1.042092,
      "stat_13": 0.32959,
      "stat_14": 0.22,
      "stat_15": 0.22,
      "stat_16": 2.15,
      "stat_17": 0.22
    }
  },
  {
    "short_name": "dev_m102_c04_batch_14",
    "task_id": "dev_m102_c04_batch_14__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/dev_m102_c04_batch_14_solver.py",
    "stat_summary": {
      "stat_00": 0.967972,
      "stat_01": 0.22723,
      "stat_02": 0.032024,
      "stat_03": 0.930897,
      "stat_04": 1.0,
      "stat_05": 0.0,
      "stat_06": 0.0,
      "stat_07": 0.67327,
      "stat_08": 0.947731,
      "stat_09": 0.052228,
      "stat_10": 0.886613,
      "stat_11": 0.099108,
      "stat_12": 1.306979,
      "stat_13": 0.480946,
      "stat_14": 2.15,
      "stat_15": 0.22,
      "stat_16": 2.15,
      "stat_17": 2.15
    }
  }
]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`16027 ms` LLM)
- **Rationale**: Extract operator signatures and archive model structure before running train-only model comparisons.

```bash
cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; for f in /tmp/hegel_continual_archive/*solver.py; do echo ===$f; grep -E '^def |panel_|return |tanh|exp\(|sin\(' "$f"; done; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('times',s['eval_times'],'telemetry',s['telemetry_vec'],'summary',s['train']['summary']);print('rows',np.c_[s['train']['x'],s['train']['y'],s['train']['traj']][:8]);
for k in ('panel_a','panel_b','panel_c'):
 d=s[k]; u=d['u']; X=np.c_[np.ones(len(u)),u]; print(k, 'u std',u.std(0));
 for j,v in d.items():
  if j=='u':continue
  r=v-X@np.linalg.lstsq(X,v,rcond=None)[0];print(j,'range',v.min(),v.max(),'linear rmse',np.mean(r*r)**.5, 'row errors',np.sort(np.linalg.norm(r,axis=1)))
PY
```
- **Container Exit Code**: `0` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-02 -7.40345276e-03
  -5.47262638e-02]
 [ 2.96840307e-01  5.48062814e-01 -4.30968805e-01  1.14998739e+00
  -1.74329694e+00  4.27660797e-01  7.93252533e-01  9.30401726e-01
   9.75428374e-01  9.85986938e-01  9.60717896e-01  9.46303606e-01
   9.23840418e-01]
 [-4.90704181e-01  4.53745406e-01 -8.22405067e-01 -2.19389893e-01
  -4.72688183e-01  1.57265430e-04  7.47060233e-02  1.82126366e-01
   2.90524533e-01  3.96075536e-01  4.90396753e-01  5.77439979e-01
   6.47425761e-01]]
panel_a u std [0.74682702 0.73621958 0.78422746 0.79397612]
v range -5.9896335461890695 7.427689401108176 linear rmse 1.4927508766160884 row errors [ 0.12472809  0.33032988  0.34022521  0.3897665   0.39508473  0.39915643
  0.40847127  0.42522698  0.46664662  0.46860298  0.47741008  0.47953134
  0.50835435  0.52376949  0.52436616  0.5373207   0.57474138  0.58111268
  0.5818501   0.59268284  0.61492026  0.63025868  0.66513549  0.70920742
  0.75467376  0.7831704   0.7996236   0.80443618  0.84849364  0.88664847
  0.90677201  0.93540552  0.95697464  1.01773409  1.02865113  1.04755816
  1.05227855  1.17430429  1.21673244  1.26308074  1.28737088  1.3291453
  1.56739312  1.68739838  1.87116447 11.20990569 11.36732112 11.80084045]
panel_b u std [0.67647039 0.74582969 0.70228864 0.71198867]
v range -1.484920952317507 1.5024978847270158 linear rmse 7.688253854623239e-05 row errors [7.60820473e-05 8.28593199e-05 8.87975290e-05 9.94686503e-05
 1.08298906e-04 1.16257902e-04 1.17734139e-04 1.21225330e-04
 1.22949106e-04 1.22987319e-04 1.29009375e-04 1.32925446e-04
 1.33079200e-04 1.35738649e-04 1.38511367e-04 1.56005695e-04
 1.56678154e-04 1.72210056e-04 1.92550743e-04 1.95355576e-04
 2.00534368e-04 2.02922737e-04 2.37909834e-04 2.67514995e-04]
panel_c u std [0.76477927 0.73516498 0.7754119  0.8833795 ]
v1 range -1.4363367320963987 1.4184833311043321 linear rmse 0.10640707448239224 row errors [0.07663267 0.1348908  0.14585696 0.15250283 0.15445526 0.17002331
 0.17252748 0.17780377 0.18114087 0.18277364 0.19013269 0.1901695
 0.2084224  0.21057971 0.22005794 0.2299928  0.23022987 0.238227
 0.2415545  0.24887098 0.25188801 0.30229433 0.30924578 0.30994895]
v2 range -1.682322606259053 1.1929521520682171 linear rmse 0.17426551584163563 row errors [0.11993028 0.18795198 0.19298384 0.21666433 0.22220357 0.23122065
 0.25174229 0.25522334 0.28764839 0.29669954 0.31315794 0.31729829
 0.32744169 0.33591696 0.34121182 0.38590872 0.38920276 0.39161297
 0.40640812 0.44815874 0.46570194 0.49923516 0.51854689 0.53790455]
```

### Turn 3 — `execute_in_remote_docker` (`10460 ms` LLM)
- **Rationale**: Read the complete calibration, panel, and trajectory formulas from the closest archived mechanisms to build explicit alternatives for held-out comparison.
- **Alternatives Weighed**: `Reuse the pre-LLM candidate unchanged` (rejected: Both public validation errors exceed 0.038.)

```bash
for i in 03 13 14; do cat /tmp/hegel_continual_archive/*batch_${i}_solver.py; done
```
- **Container Exit Code**: `0` (`295 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
h = np.tanh(k[0]*z)
        err = offset+gain*z+a*h+b*np.sin(k[1]*r)-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z),4,4))
        jac[:,j,j] = gain+a*k[0]*(1-h*h)
        jac[:,j,(j+2)%4] = b*k[1]*np.cos(k[1]*r)
        z -= np.linalg.solve(jac, err[...,None])[...,0]
    return z


_RING = ((0,1),(0,3),(1,2),(2,3))


def _panels(panel, z):
    from scipy.optimize import minimize_scalar
    u, vp, vq = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))

    def latent(x, k):
        return np.column_stack((x / np.sqrt(1 + k*x*x),
                                *(x[:, i]*x[:, j] for i,j in _RING)))

    def profile(target, design, bounds):
        def residual(k):
            a = design(u, k)
            return np.linalg.norm(a @ _fit(a, target) - target)
        k = minimize_scalar(residual, bounds=bounds, method='bounded').x
        return k, _fit(design(u, k), target)

    def q_design(x, k):
        return _design(latent(x, k))

    kq, cq = profile(vq, q_design, (.2, 1.5))

    def p_design(x, k):
        return _design(x, x/(1+k*x*x), latent(x, kq))

    kp, cp = profile(vp, p_design, (.2, 2.5))
    return p_design(z, kp) @ cp, q_design(z, kq) @ cq


def _time_basis(times, rates):
    return -np.expm1(-np.outer(times, rates))


def _temporal_fit(times, traj):
    def residual(rates):
        a = _time_basis(times, rates)
        return (a @ _fit(a, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [1.8, .42],
                         bounds=([.75, .03], [6., .74])).x
    t = _time_basis(times, rates)
    return t, _fit(t, traj.T).T


def solve_batch(study: dict) -> dict:
    """Fit calibration and two response modes on the supplied panels and training rows."""
    train = study['train']
    y = np.asarray(train['y'], dtype=float)
    n = len(y)
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _calibrate(study['panel_a'], x)
    p, q = _panels(study['panel_c'], z)
    interaction = (p[:, :, None] * q[:, None, :]).reshape(len(x), 16)
    a = _design(p, q, interaction)
    t, amplitude = _temporal_fit(study['eval_times'], train['traj'])
    prediction = a[n:] @ _fit(a[:n], np.column_stack((y, amplitude)))
    y_hat = prediction[:, 0]
    state = prediction[:, 1:] @ t.T
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
```

### Turn 4 — `execute_in_remote_docker` (`14741 ms` LLM)
- **Rationale**: Benchmark the transferable solvers on the prescribed split and use the trajectory singular spectrum to constrain the temporal model.

```bash
for f in /tmp/hegel_continual_archive/*solver.py; do echo ==== $f; grep -nE 'def |least_squares|poly|expm1|tanh|sin\(|cos\(|sqrt|log1p|z\*|interaction|panel_b' "$f"; done; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,glob,importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];print('TIMES',s['eval_times'],'TELEM',s['telemetry_vec'],'SUMMARY',t['summary']); print('SVD',np.linalg.svd(t['traj'],compute_uv=False));n=52;s['test']={'x':t['x'][n:]};s['train']={k:v[:n] if k!='summary' else v for k,v in t.items()};
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 try:
  q=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(q);q.loader.exec_module(m);p=m.solve_batch(s); print(f[-12:],np.mean(abs(p['y_hat']-t['y'][n:])),np.mean(abs(p['state_pred']-t['traj'][n:])))
 except Exception as e: print(f[-12:],e)
PY
```
- **Container Exit Code**: `0` (`705 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_calibrate(panel, observed):
23:def _operators(panel, z):
25:    def first(x, k):
27:    def second(x, k):
32:        def score(k):
40:def _predict_state(times, traj, common, r, cross):
49:        def design(rates, rows):
50:            basis = -np.expm1(-times[:, None] * rates)
55:        def residual(rates):
59:        fit = least_squares(residual, [1.8, .43], bounds=([.8, .03], [8., .79]))
67:def solve_batch(study: dict) -> dict:
==== /tmp/hegel_continual_archive/dev_m102_c04_batch_14_solver.py
3:from scipy.optimize import least_squares
6:def _design(*x):
10:def _fit(x, y):
14:def _calibrate(panel, observed):
22:    def profile(rate):
23:        a = np.concatenate((*base, np.tanh(rate[0]*u)[:, :, None],
24:                            np.sin(rate[1]*np.roll(u, 2, axis=1))[:, :, None]), axis=2).reshape(-1, 10)
28:    k = least_squares(lambda p: profile(p)[0], [1.8, 2.4], bounds=([.5, 1.5], [4., 3.3])).x
34:        update = (observed-offset-a*np.tanh(k[0]*z)-b*np.sin(k[1]*np.roll(z, 2, axis=1)))/gain
41:        h = np.tanh(k[0]*z)
42:        err = offset+gain*z+a*h+b*np.sin(k[1]*r)-observed
47:        jac[:,j,(j+2)%4] = b*k[1]*np.cos(k[1]*r)
55:def _panels(panel, z):
59:    def latent(x, k):
60:        return np.column_stack((x / np.sqrt(1 + k*x*x),
63:    def profile(target, design, bounds):
64:        def residual(k):
70:    def q_design(x, k):
75:    def p_design(x, k):
82:def _time_basis(times, rates):
83:    return -np.expm1(-np.outer(times, rates))
86:def _temporal_fit(times, traj):
87:    def residual(rates):
91:    rates = least_squares(residual, [1.8, .42],
97:def solve_batch(study: dict) -> dict:
105:    interaction = (p[:, :, None] * q[:, None, :]).reshape(len(x), 16)
106:    a = _design(p, q, interaction)
TIMES [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ] TELEM [0.22 0.22 2.15 1.25] SUMMARY [-0.05977745  1.0875869   0.09697071  0.9110814 ]
SVD [17.91743363  2.92796072  0.0490423   0.0449515   0.04210343  0.03914935
  0.0339926   0.03178362]
11_solver.py 0.29010300147236023 0.09971218382392098
03_solver.py 0.10604955134238636 0.10943026921142544
12_solver.py 0.882759926357432 0.6665741610083492
08_solver.py 1.082439535218234 0.4860951270209254
00_solver.py 0.14918205613579305 0.13014052947256266
14_solver.py 0.14835530516845646 0.09977216426443562
13_solver.py 0.20216073231598858 0.11678942889982183
05_solver.py 0.9663102025496224 0.7418983817115181
04_solver.py 0.3072492161424467 0.1537948803798909
```

### Turn 5 — `execute_in_remote_docker` (`29719 ms` LLM)
- **Rationale**: Independent panel fits can identify operator formulas without using held-out response rows and narrow the readout search.
- **ATMS NOGOOD DECLARED**: Refuted `An unchanged archived solver is sufficient for this campaign.` | Witness: `All nine archived solvers miss the 0.038 threshold on the prescribed split; the trajectory singular values still support two dominant temporal modes.` | Blanket Diagnosis: `The transfer failure could be in the calibration, auxiliary operators, or readout interactions; it does not by itself refute the two-mode temporal model.` | Sublated Pivot: `Compare explicit calibration couplings and nonlinear panel bases against their independent reference measurements before testing readout graphs.`

```bash
sed -n '1,49p' /tmp/hegel_continual_archive/*batch_12_solver.py; sed -n '1,56p' /tmp/hegel_continual_archive/*batch_13_solver.py; cat > /tmp/explore15.py <<'PY'
import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; aff=lambda *x: np.column_stack((np.ones(len(x[0])),*x)); rm=lambda a:np.mean(a*a)**.5
u,v=s['panel_a'].values(); good=np.max(abs(u-v),axis=1)<3;u,v=u[good],v[good]; print('clean',len(u));eye=np.broadcast_to(np.eye(4),(len(u),4,4));
def cal_d(u,j,kind,k):
 r=np.roll(u,j,axis=1)
 if kind=='poly': return u*r,u**3
 if kind=='trig': return np.tanh(k[0]*u),np.sin(k[1]*r)
 if kind=='trig2': return np.sin(k[0]*u),np.tanh(k[1]*r)
 if kind=='trig3': return np.tanh(k[0]*u),np.cos(k[1]*r)
for j in (1,2,3):
 for kind in ('poly','trig','trig2','trig3'):
  for free in (False,True):
   def fit(k):
    h,g=cal_d(u,j,kind,k);d=np.concatenate((eye,eye*u[:,:,None],*((eye*h[:,:,None],eye*g[:,:,None]) if free else (h[:,:,None],g[:,:,None]))),axis=2).reshape(len(u)*4,-1);w=ls(d,v.ravel());return d@w-v.ravel(),w
   k=least_squares(lambda k:fit(k)[0],[1.8,2.4],bounds=(.15,6)).x if kind!='poly' else []
   e,w=fit(k);print('a',j,kind,free,rm(e),np.round(k,4),np.round(w,4) if rm(e)<.002 else '')
U,V,W=s['panel_c'].values(); pairs=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in it.combinations(range(4),2)]); funcs={'sat':lambda x,k:x/(1+k*abs(x)), 'rat':lambda x,k:x/(1+k*x*x),'root':lambda x,k:x/np.sqrt(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'exp':lambda x,k:np.exp(-k*x*x),'abs':lambda x,k:abs(x),'sq':lambda x,k:x*x}
for v in (V,W):
 out=[]
 for name,f in funcs.items():
  for typ in range(5):
   def d(k):
    h=f(U,k); return aff(h,*[(),(U,),(pairs(U),),(U,pairs(U)),(pairs(h),)][typ]);
   def obj(k):
    a=d(k);return rm(a@ls(a,v)-v)
   o=minimize_scalar(obj,bounds=(.01,5),method='bounded'); out.append((o.fun,name,typ,o.x))
 print('c best',sorted(out)[:12]);
 for mode in (U,pairs(U),aff(U,pairs(U)),aff(U,U**2,pairs(U))): print('base',mode.shape,rm(mode@ls(mode,v)-v))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/explore15.py
```
- **Container Exit Code**: `0` (`676 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
83450080133746 [2.9931 3.1921] 
a 1 trig3 False 0.3164965609042359 [2.0834 4.2854] 
a 1 trig3 True 0.3094683187437864 [2.1757 4.0401] 
a 2 poly False 0.31608051529783326 [] 
a 2 poly True 0.31162444151654545 [] 
a 2 trig False 0.3170225043126896 [2.1212 2.4929] 
a 2 trig True 0.30505913192713763 [1.4727 5.5131] 
a 2 trig2 False 0.3167910187426196 [3.687 6.   ] 
a 2 trig2 True 0.3145090838144613 [3.0692 6.    ] 
a 2 trig3 False 0.31418208552460264 [2.0189 3.7666] 
a 2 trig3 True 0.30602915935249086 [0.7339 3.2166] 
a 3 poly False 0.24284442521872843 [] 
a 3 poly True 0.2400097606382326 [] 
a 3 trig False 0.18238299233994318 [1.3559 0.4655] 
a 3 trig True 0.17709223888511333 [1.0338 0.4154] 
a 3 trig2 False 0.1822908120638088 [3.0039 0.3128] 
a 3 trig2 True 0.17704996962515296 [2.5035 0.2762] 
a 3 trig3 False 0.31675354573026365 [2.1072 0.15  ] 
a 3 trig3 True 0.31432191298649903 [1.5555 0.15  ] 
c best [(0.0004997620626868605, 'sat', 3, 0.8405949757805024), (0.0006139185599355109, 'sat', 1, 0.8431425967448059), (0.003033870789119806, 'root', 3, 3.4183791401387777), (0.0037195659599214897, 'root', 1, 3.403310436428916), (0.0037351712044248107, 'tanh', 3, 2.169947952350224), (0.0040109274912525595, 'rat', 3, 1.728332711499163), (0.004586970382400322, 'tanh', 1, 2.162722231870778), (0.004927332061781825, 'rat', 1, 1.7149702504843707), (0.005475919943531283, 'sat', 4, 1.4372234145251226), (0.005708900915275907, 'root', 4, 2.1822604835756243), (0.005809759116150876, 'sat', 2, 1.427297091425177), (0.006055944743316521, 'root', 2, 2.1971587592554838)]
base (24, 4) 0.147843872738701
base (24, 6) 0.5726456067609669
base (24, 11) 0.0888877179678641
base (24, 15) 0.07901517389636718
c best [(5.1811552056736e-05, 'root', 3, 0.8471506048936636), (7.237161940730664e-05, 'root', 2, 0.8499484373647866), (0.00011526156461359708, 'tanh', 3, 1.1885510648728792), (0.0001334726340489025, 'rat', 3, 0.5540062547502674), (0.0006524650745946079, 'sin', 3, 2.540973452762669), (0.0013619624771353596, 'tanh', 2, 1.0410369638284394), (0.0014195944403450705, 'sat', 3, 0.010003688288540835), (0.0026478654411253625, 'rat', 2, 0.31746901666749916), (0.006041685963690813, 'sin', 2, 1.159432804171605), (0.006721178468700296, 'sat', 2, 0.6386983940086456), (0.02116040157399902, 'root', 4, 0.7603962486481872), (0.021181208023793276, 'sat', 4, 0.584944229337012)]
base (24, 4) 0.2044905795964236
base (24, 6) 0.4877719347582865
base (24, 11) 0.04695921691677908
base (24, 15) 0.042083030495443374
```

### Turn 6 — `execute_in_remote_docker` (`23334 ms` LLM)
- **Rationale**: Test the missing affine couplings and nonlinear functions of coordinate sums and differences on panel A before fitting downstream readouts.
- **Alternatives Weighed**: `Continue with diagonal gain calibration` (rejected: Its nonlinear variants leave panel-A RMSE above 0.17, and the residual strongly depends on the neighboring coordinate.)

```bash
cat > /tmp/cal15.py <<'PY'
exec(open('/tmp/explore15.py').read().split('for j in (1,2,3):')[0])
print('affine',ls(aff(u),v),rm(aff(u)@ls(aff(u),v)-v)); print(np.round(np.c_[u,v][:12],3))
fn={'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'root':lambda x,k:x/np.sqrt(1+k*x*x),'sat':lambda x,k:x/(1+k*abs(x)), 'rat':lambda x,k:x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'pow':lambda x,k:x*abs(x)**k,'sinh':lambda x,k:np.sinh(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'exp':lambda x,k:np.exp(-k*x*x)}
R=np.roll(u,-1,axis=1); B=aff(u); pair=np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]); dr=[('u',u),('u+r',u+R),('u-r',u-R),('u*r',u*R)]; out=[]
for nm,g in dr:
 for name,f in fn.items():
  def fit(k):
   d=aff(u,f(g,k[0]));w=ls(d,v);return (d@w-v).ravel(),w
  z=min((least_squares(lambda k:fit(k)[0],[k],bounds=(.06,8)) for k in (1.25,2.15,4.0)),key=lambda p:p.cost);out.append((rm(z.fun),nm,name,z.x[0],fit(z.x)[1]))
print('a best');
for er,nm,name,k,w in sorted(out,key=lambda x:x[0])[:12]: print(er,nm,name,k, np.round(w,3) if er<.03 else '')
print('A poly');
for j in (1,2,3):
 r=np.roll(u,j,axis=1)
 for abc in [(u*r,u**3),(u*r,u**3,r**3),(u**2,r**2,u*r),(u**2,r**2,u*r,u**3,r**3,u*u*r,u*r*r)]:
  d=aff(u,*abc); print(j,d.shape,rm(d@ls(d,v)-v))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal15.py; cat /holdout_task/public_asset_index.json
```
- **Container Exit Code**: `0` (`894 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
type": "ndarray"
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
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
```

### Turn 7 — `execute_in_remote_docker` (`39709 ms` LLM)
- **Rationale**: Resolve the calibration on independent reference data, then compare bilinear readouts on the prescribed split with the temporal rates fit only on training rows.
- **ATMS NOGOOD DECLARED**: Refuted `Diagonal calibration with coordinatewise nonlinear corrections adequately explains panel A.` | Witness: `Diagonal variants leave RMSE above 0.17, whereas cyclic bilinear and cubic terms with affine mixing reduce it to 0.000079.` | Blanket Diagnosis: `The residual was dominated by missing cross-coordinate calibration terms, not by measurement noise.` | Sublated Pivot: `Recover the sparse cyclic map, invert it, and compare candidate auxiliary readout interactions.`

```bash
cat > /tmp/read15.py <<'PY'
exec(open('/tmp/explore15.py').read().split('for j in (1,2,3):')[0])
from scipy.spatial.distance import cdist
P=list(it.combinations(range(4),2)); pairs=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in P]); roll=lambda x:np.roll(x,-1,axis=1)
d=lambda x:aff(x,x*roll(x),x**3)
w=ls(d(u),v);print('A coef',np.round(w,6)); print('A residual selection',np.sort(np.max(abs(d(s['panel_a']['u'])@w-s['panel_a']['v']),axis=1))[-8:]);w[abs(w)<.001]=0
x=tr['x'];z=x.copy();j=np.arange(4)
for _ in range(45):
 J=np.broadcast_to(w[1:5].T,(len(z),4,4)).copy();J[:,j,j]+=w[5:9].diagonal()*roll(z)+3*w[9:].diagonal()*z*z;J[:,j,(j+1)%4]+=w[5:9].diagonal()*z; z-=np.clip(np.linalg.solve(J,(d(z)@w-x)[:,:,None])[:,:,0],-.4,.4)
print('inv check',abs(d(z)@w-x).max(),z.min(0),z.max(0));
U,V,W=s['panel_c'].values();sat=lambda z,k:z/(1+k*abs(z));root=lambda z,k:z/np.sqrt(1+k*z*z)
M={}
for key,F,T in [('p',sat,V),('q',root,W)]:
 def d0(x,k):return aff(x,F(x,k),pairs(x))
 o=minimize_scalar(lambda k:rm(d0(U,k)@ls(d0(U,k),T)-T),bounds=(.1,2),method='bounded'); ww=ls(d0(U,o.x),T); print(key,o.x,o.fun,np.round(ww,4));M[key]=d0(z,o.x)@ww
M['z']=z
B,T=s['panel_b'].values(); fs={'sat':sat,'root':root,'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'log':lambda x,k:np.log1p(k*x*x),'rat':lambda x,k:x/(1+k*x*x),'sq':lambda x,k:x*x,'cube':lambda x,k:x**3,'exp':lambda x,k:np.exp(-k*x*x)}
out=[]
for nm,F in fs.items():
 for mode in range(4):
  def bf(x,k):return aff(*[(F(x,k),),(x,F(x,k)),(F(x,k),pairs(x)),(x,F(x,k),pairs(x))][mode]);
  o=minimize_scalar(lambda k:rm(bf(B,k)@ls(bf(B,k),T)-T),bounds=(.03,5),method='bounded');out.append((o.fun,nm,mode,o.x))
print('B best',sorted(out)[:12]);
for bi,(_,nm,mode,k) in enumerate(sorted(out)[:3]):
 F=fs[nm]; bf=lambda x:aff(*[(F(x,k),),(x,F(x,k)),(F(x,k),pairs(x)),(x,F(x,k),pairs(x))][mode]);M['b'+str(bi)]=bf(z)@ls(bf(B),T)
t=s['eval_times']; Y=tr['traj']; bas=lambda k:-np.expm1(-np.outer(t,k)); opt=least_squares(lambda k:(bas(k)@ls(bas(k),Y[:52].T)-Y[:52].T).ravel(),[1.8,.43],bounds=([.8,.03],[8,.79])); Utime=bas(opt.x);print('rates',opt.x,rm(opt.fun));A=ls(Utime,Y.T).T;R=np.c_[tr['y'],A];
def report(f):
 pred=f@ls(f[:52],R[:52]);return np.r_[np.mean(abs(pred[52:]-R[52:]),axis=0),np.mean(abs((pred[52:,1:]@Utime.T)-Y[52:])),np.mean(abs(pred[:52]-R[:52]),axis=0)]
features=[]
for n,a in M.items():
 for m,b in M.items():
  if m<n:continue
  cross=(a[:,:,None]*b[:,None,:]).reshape(len(a),16) if m!=n else np.c_[a*a,pairs(a)];
  for mode,c in [('lin',aff(a,b)),('dot',aff(a,b,np.sum(a*b,axis=1))),('all',aff(a,b,cross)),('sq',aff(a,b,cross,a*a,b*b))]:
   e=report(c);features.append((max(e[0],e[3]),n,m,mode,np.round(e,5)))
  common=aff(a,b,np.sum(a*b,axis=1)); scores=[]
  for i in range(cross.shape[1]):
   c=np.c_[common,cross[:,i]]; rr=R[:52]-c[:52]@ls(c[:52],R[:52]);scores.append(rm(rr/R[:52].std(0)))
  j=np.argmin(scores);e=report(np.c_[common,cross[:,j]]);features.append((max(e[0],e[3]),n,m,'cross'+str(j),np.round(e,5)))
print('readout top');[print(row) for row in sorted(features,key=lambda q:q[0])[:22]]
np.savez('/tmp/dat15.npz',**M,R=R,time=Utime,Y=Y)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/read15.py
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1848917 0.40117813] 0.0042203923370098696
readout top
(0.005117437104391416, 'p', 'q', 'all', array([0.00512, 0.01009, 0.01874, 0.00456, 0.00324, 0.0057 , 0.01015]))
(0.005126494445162789, 'p', 'q', 'cross1', array([0.00513, 0.0103 , 0.01745, 0.00434, 0.00389, 0.00651, 0.01152]))
(0.0067806085292258, 'p', 'q', 'sq', array([0.00678, 0.0183 , 0.03353, 0.00568, 0.00254, 0.00432, 0.00797]))
(0.05416012877040227, 'p', 'z', 'all', array([0.05416, 0.03828, 0.04748, 0.03765, 0.02828, 0.02294, 0.03182]))
(0.054164533915561974, 'b2', 'p', 'all', array([0.05416, 0.03829, 0.04746, 0.03765, 0.02828, 0.02294, 0.03182]))
(0.05416454776824007, 'b1', 'p', 'all', array([0.05416, 0.03829, 0.04746, 0.03765, 0.02828, 0.02294, 0.03182]))
(0.05416505038311924, 'b0', 'p', 'all', array([0.05417, 0.03829, 0.04746, 0.03765, 0.02828, 0.02294, 0.03182]))
(0.05469841568930067, 'b2', 'p', 'sq', array([0.0547 , 0.04239, 0.05263, 0.05043, 0.02463, 0.02016, 0.01856]))
(0.05469842917436503, 'b1', 'p', 'sq', array([0.0547 , 0.04239, 0.05263, 0.05043, 0.02463, 0.02016, 0.01856]))
(0.05469896587057208, 'b0', 'p', 'sq', array([0.0547 , 0.04239, 0.05264, 0.05043, 0.02463, 0.02016, 0.01856]))
(0.05500141982793825, 'p', 'z', 'sq', array([0.05345, 0.04404, 0.06974, 0.055  , 0.02405, 0.02002, 0.02176]))
(0.0644269032013094, 'p', 'q', 'dot', array([0.00494, 0.01922, 0.11203, 0.06443, 0.00393, 0.0111 , 0.07269]))
(0.06858141184096422, 'p', 'p', 'all', array([0.06858, 0.05141, 0.08479, 0.06272, 0.03983, 0.03094, 0.07379]))
(0.06858141184096551, 'p', 'p', 'sq', array([0.06858, 0.05141, 0.08479, 0.06272, 0.03983, 0.03094, 0.07379]))
(0.07303968733442182, 'b0', 'q', 'sq', array([0.05582, 0.09247, 0.04143, 0.07304, 0.01802, 0.02212, 0.01031]))
(0.07304275123300648, 'b1', 'q', 'sq', array([0.05582, 0.09248, 0.04143, 0.07304, 0.01802, 0.02212, 0.01031]))
(0.07304286034134773, 'b2', 'q', 'sq', array([0.05582, 0.09248, 0.04143, 0.07304, 0.01802, 0.02212, 0.01031]))
(0.07998758177415148, 'q', 'z', 'sq', array([0.07999, 0.07041, 0.03321, 0.06265, 0.01718, 0.02194, 0.01178]))
(0.08107682713551054, 'b1', 'z', 'sq', array([0.08108, 0.06181, 0.0715 , 0.06016, 0.02395, 0.01852, 0.00989]))
(0.08111078428872003, 'b2', 'z', 'sq', array([0.08111, 0.06189, 0.07142, 0.06034, 0.02388, 0.01846, 0.00989]))
(0.08383381574050312, 'b0', 'z', 'sq', array([0.08383, 0.05971, 0.0726 , 0.05441, 0.02561, 0.0197 , 0.00991]))
(0.08623750688489473, 'b0', 'q', 'all', array([0.08624, 0.07675, 0.02437, 0.06874, 0.03761, 0.02571, 0.01483]))
```

### Turn 8 — `execute_in_remote_docker` (`68797 ms` LLM)
- **Rationale**: Write the compact solver with interaction selection confined to training rows, then verify the exact artifact on the prescribed split, randomized folds, coordinate-tail splits, and each temporal horizon.
- **Alternatives Weighed**: `Use all 16 bilinear products` (rejected: The 52/20 scalar MAE is nearly identical, but the compact readout has lower trajectory MAE and needs fewer fitted coefficients.); `Use panel B in place of the second panel-C operator` (rejected: The best such exploratory fit has scalar MAE above 0.054, against 0.00513 for the selected graph.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrated dual-operator predictor with a two-rate state response."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(*cols):
    return np.column_stack((np.ones(len(cols[0])), *cols))


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.max(np.abs(v-u), axis=1) < 3
    u, v = u[keep], v[keep]
    r = np.roll(u, -1, axis=1)
    c = np.column_stack([_ls(_aff(u[:, j], r[:, j], u[:, j]*r[:, j], u[:, j]**3), v[:, j])
                         for j in range(u.shape[1])])
    offset, gain, coupling, product, cubic = c
    z = (observed-offset)/gain
    j = np.arange(z.shape[1]); neighbor = (j+1) % len(j)
    for _ in range(45):
        r = z[:, neighbor]
        err = offset+gain*z+coupling*r+product*z*r+cubic*z**3-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = gain+product*r+3*cubic*z*z
        jac[:, j, neighbor] = coupling+product*z
        z -= np.clip(np.linalg.solve(jac, err[..., None])[..., 0], -.4, .4)
    return z


def _operators(panel, z):
    u = np.asarray(panel['u'], dtype=float)
    i, j = np.triu_indices(u.shape[1], 1)
    def design(x, k, mode):
        h = x/(1+k*np.abs(x)) if mode == 1 else x/np.sqrt(1+k*x*x)
        return _aff(x, h, x[:, i]*x[:, j])
    prediction = []
    for mode in (1, 2):
        target = np.asarray(panel[f'v{mode}'], dtype=float)
        def fit(k):
            a = design(u, k, mode)
            w = _ls(a, target)
            return np.mean((a@w-target)**2), w
        k = minimize_scalar(lambda k: fit(k)[0], bounds=(.1, 2.), method='bounded').x
        prediction.append(design(z, k, mode)@fit(k)[1])
    return prediction


def _basis(times, rates):
    return -np.expm1(-np.outer(times, rates))


def _rates(times, traj):
    def residual(rates):
        b = _basis(times, rates)
        return (b@_ls(b, traj.T)-traj.T).ravel()
    return least_squares(residual, [1.8, .43], bounds=([.8, .03], [8., .79])).x


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    p, q = _operators(study['panel_c'], _calibrate(study['panel_a'], x))
    pq = (p[:, :, None]*q[:, None, :]).reshape(len(x), -1)
    base = _aff(p, q, np.sum(p*q, axis=1))
    traj = np.asarray(train['traj'], dtype=float)
    times = np.asarray(study['eval_times'], dtype=float)
    b = _basis(times, _rates(times, traj))
    target = np.column_stack((train['y'], _ls(b, traj.T).T))
    scale = np.maximum(target.std(axis=0), 1e-12)
    def fit_candidate(j):
        a = np.column_stack((base, pq[:, j]))
        w = _ls(a[:n], target)
        return np.mean(((a[:n]@w-target)/scale)**2), a[n:]@w
    _, prediction = min((fit_candidate(j) for j in range(pq.shape[1])), key=lambda x: x[0])
    y_hat = prediction[:, 0]
    state = prediction[:, 1:]@b.T
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])}
PY
cat > /tmp/validate15.py <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
tr=s['train']; result=mod.solve_batch(s); y,st=result['y_hat'],result['state_pred']; np.testing.assert_array_equal(result['diagnostics_vec'], [y.mean(), y.std(), st[:,-1].mean(), st[:,-1].std()]); print('test',y.shape,st.shape,result['diagnostics_vec']); assert np.isfinite(y).all() and np.isfinite(st).all()
z=mod._calibrate(s['panel_a'],tr['x']); ztest=mod._calibrate(s['panel_a'],s['test']['x']);print('test calibrated range',ztest.min(),ztest.max());
splits=[('required',np.arange(52,72))]+[(f'random{i}',np.random.default_rng(i).permutation(72)[52:]) for i in range(12)]+[(f'extreme{j}{side}',np.argsort(side*z[:,j])[-20:]) for j in range(4) for side in (-1,1)]; worst=np.zeros(3)
for name, val in splits:
    train=np.setdiff1d(np.arange(72),val); ss=dict(s,train={k:(v[train] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][val]}); p=mod.solve_batch(ss); e=np.abs(p['state_pred']-tr['traj'][val]); m=np.array([np.mean(np.abs(p['y_hat']-tr['y'][val])),e.mean(),e.mean(0).max()]);worst=np.maximum(worst,m); print(name, np.round(m,6)); assert max(m)<.038
print('worst y / trajectory / single horizon', worst)
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/validate15.py
```
- **Container Exit Code**: `0` (`765 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
test (36,) (36, 8) [0.19855792 1.26418415 0.33271239 0.83084919]
test calibrated range -1.1262378604460377 1.1451665590764688
required [0.005131 0.004344 0.00527 ]
random0 [0.003419 0.003942 0.004719]
random1 [0.005758 0.004849 0.006175]
random2 [0.003764 0.004278 0.005159]
random3 [0.006202 0.004189 0.004968]
random4 [0.005453 0.003807 0.004971]
random5 [0.00593  0.004411 0.005187]
random6 [0.004974 0.003789 0.004839]
random7 [0.004067 0.004322 0.005878]
random8 [0.004078 0.00429  0.005645]
random9 [0.005082 0.004491 0.005824]
random10 [0.005367 0.004139 0.004792]
random11 [0.004505 0.003825 0.004804]
extreme0-1 [0.006097 0.004304 0.005505]
extreme01 [0.005548 0.003734 0.00435 ]
extreme1-1 [0.005211 0.004325 0.005451]
extreme11 [0.005509 0.00441  0.005392]
extreme2-1 [0.00662  0.004067 0.004835]
extreme21 [0.008968 0.004234 0.005321]
extreme3-1 [0.006145 0.003809 0.005012]
extreme31 [0.008257 0.004169 0.004936]
worst y / trajectory / single horizon [0.00896831 0.00484923 0.0061752 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6130c0138ee7181c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6130c013 (directory).
```

### Turn 9 — `submit_completed_task` (`12346 ms` LLM)
- **Rationale**: The exact archived candidate passed the public schema check and isolated Python verification, including diagnostics, randomized splits, coordinate-tail splits, and each evaluation horizon. All measured errors were below tolerance.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed. On the required 52/20 split, MAE was 0.00513 for scalar predictions and 0.00434 for trajectories, both below 0.038.`

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
