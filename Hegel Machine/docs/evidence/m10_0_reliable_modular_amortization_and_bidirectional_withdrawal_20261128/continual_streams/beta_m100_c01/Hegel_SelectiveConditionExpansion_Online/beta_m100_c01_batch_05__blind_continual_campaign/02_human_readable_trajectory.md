# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c01_batch_05__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `324,333` (`311,995` / `114,688` / `12,338`) |
| **Total Episode Wall Time (s)** | `391.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`16628 ms` LLM)
- **Rationale**: Inspect downstream units, array layouts and calibration coverage to determine which fields to transform and how to test inversion at the operating boundaries.

```bash
cd /holdout_task && cat batch_solver.py validate_public_schema.py _verified_downstream.py && cat public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))
def show(v,name,depth=0):
 if isinstance(v,dict):
  for k,x in v.items(): show(x,name+'.'+str(k),depth+1)
 elif isinstance(v,(list,tuple)):
  print(name,type(v).__name__,len(v));
  if v: show(v[0],name+'[0]',depth+1)
 elif isinstance(v,np.ndarray):
  print(name,v.shape,v.dtype, np.nanquantile(v,[0,.01,.5,.99,1],axis=0).round(5).tolist() if v.ndim==2 and v.shape[1]<=4 and v.size else '')
 else: print(name,type(v).__name__,repr(v)[:150])
show(s,'study')
PY
ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS nonlinear multichannel calibration'
```
- **Container Exit Code**: `0` (`517 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.22979, 12.48, 1.29055], [13.57001, 1.32835, 12.48001, 1.31406]]
study[0].study.sensor_calibration_sweeps.single_bank_sweeps.bank_id (48,) int64 
study[0].study.sensor_calibration_sweeps.single_bank_sweeps.ref_signal (48,) float64 
study[0].study.sensor_calibration_sweeps.single_bank_sweeps.raw_readout (48,) float64 
study[0].study.Z_nuisance (96, 3) float64 [[-2.48204, -2.82219, -2.52135], [-2.32356, -2.70725, -1.90975], [0.01568, 0.00427, 0.08783], [2.44239, 2.08227, 2.55101], [2.70504, 2.36279, 2.81084]]
study[0].study.y_obs_raw (96,) float64 
study[0].study.aux_manifold_telemetry (24, 4) float64 [[-1.30939, -2.3483, -3.4694, -1.70844], [-1.27104, -1.93022, -3.07752, -1.67059], [-0.11889, 0.40917, -0.5593, 0.09725], [1.54843, 1.86446, 1.98902, 2.45426], [1.60593, 1.90631, 2.01549, 2.6291]]
study[0].study.nuisance_ridge float 0.08
study[0].study.source_ridge float 0.04
study[0].study.u_box float 0.42
study[0].study.total_budget float 1.4
study[0].campaign_diagnostics NoneType None
total 56
-r--r--r-- 1 root root 23261 Nov 28 03:49 archive_apparatus_log.json
-rw-r--r-- 1 root root  1024 Nov 28 03:49 archive_manifest.json
-rw-r--r-- 1 root root  5023 Nov 28 03:49 beta_m100_c01_batch_01_solver.py
-rw-r--r-- 1 root root  4662 Nov 28 03:49 beta_m100_c01_batch_02_solver.py
-rw-r--r-- 1 root root  5004 Nov 28 03:49 beta_m100_c01_s00_solver.py
-rw-r--r-- 1 root root   125 Nov 28 03:49 pre_llm_gate_diagnostic.json
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18779 ms` LLM)
- **Rationale**: Inspect paired sweep rows to distinguish saturation from EMI spikes, and confirm which array shapes and units the downstream solver consumes before fitting or editing.

```bash
cd /holdout_task && cat public_asset_index.json && sed -n '36,130p' _verified_downstream.py; cd /tmp/hegel_continual_archive && cat pre_llm_gate_diagnostic.json archive_manifest.json && grep -inE 'rosette|tanh|saturat|calibrat|apparatus' *_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
l=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'));print('log layout',list(l) if isinstance(l,dict) else len(l));print(str(l)[-4400:]);b=pickle.load(open('/holdout_task/public_batch.pkl','rb'));print('batch size',len(b));
for item in b:
 s=item['study']; x=s['sensor_calibration_sweeps']['ref_rosette']; y=s['sensor_calibration_sweeps']['raw_rosette'];print('study keys',list(s));print('observations range',np.min(s['y_obs_raw']),np.max(s['y_obs_raw']));print('pulses',[(list(p),None if p.get('y_pulse_raw') is None else np.shape(p['y_pulse_raw'])) for p in (s.get('reference_tracer_pulses') or [])]);print('design counts',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('paired sweep data:'); print(np.c_[x,y].round(6))
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-8.2173600e-01  1.0998350e+00 -7.0800000e-04 -7.6324600e-01
  -1.2548710e+00  7.3195400e-01 -2.1998400e-01 -7.5806400e-01]
 [-1.5375100e-01 -1.1792930e+00  1.5802470e+00  6.2716000e-02
   2.4746000e-02 -1.0438240e+00  1.7989540e+00 -2.5371200e-01]
 [-4.9173400e-01  5.9119100e-01  2.5560800e-01 -4.7627900e-01
  -7.9525300e-01  5.1904600e-01  4.2610800e-01 -5.5967900e-01]
 [-2.9734600e-01  1.1814400e+00 -1.4238200e-01 -9.7169200e-01
  -5.6354200e-01  7.0323200e-01 -3.0082800e-01 -8.6155800e-01]
 [-6.0329000e-01  1.6152280e+00 -7.0125100e-01  6.4268800e-01
  -1.2635200e+00  1.1645560e+00 -1.3216570e+00  8.6180800e-01]
 [ 1.3007010e+00  1.6214390e+00  1.3499300e-01  1.1047820e+00
   1.7202220e+00  1.2035940e+00  5.5811400e-01  1.1129530e+00]
 [ 9.1930700e-01 -3.3045700e-01  8.9012300e-01  4.6469200e-01
   1.6200000e+00 -4.1703200e-01  1.5990770e+00  4.5185700e-01]
 [ 1.2484130e+00  3.0035500e-01 -2.1100000e-02 -4.3225000e-01
   1.6563220e+00  2.3085500e-01  2.3111400e-01 -5.4221000e-01]
 [-6.4354500e-01 -4.2442100e-01  1.6389050e+00  1.1419600e-01
  -7.7917500e-01 -5.6238200e-01  1.6630440e+00  1.1623000e-02]
 [ 1.4578200e-01  1.2731280e+00 -1.3092580e+00 -4.1581200e-01
  -1.2308900e-01  8.7628000e-01 -1.4929600e+00 -4.3843100e-01]
 [ 7.8792200e-01  3.3882900e-01  2.8543900e-01 -7.6196000e-02
   1.4272590e+00  3.5266400e-01  7.7833200e-01 -1.0382800e-01]
 [ 1.3396890e+00  1.4516170e+00  7.7931800e-01  1.6656100e-01
   1.8385250e+00  1.0466280e+00  1.5659880e+00  3.2245800e-01]
 [-5.9637500e-01  5.9248700e-01  1.7830600e+00 -4.4042900e-01
  -6.7397700e-01  5.2901600e-01  1.8163160e+00 -5.2261500e-01]
 [-2.0954900e-01 -9.3637000e-02 -1.1494500e+00 -6.5506500e-01
  -7.8841900e-01 -3.3032100e-01 -1.5137020e+00 -8.0867700e-01]
 [ 8.0520900e-01  7.1616500e-01  1.3148330e+00  1.4226300e+00
   1.5876000e+00  8.6463500e-01  1.8254960e+00  1.1148480e+00]
 [ 4.4864000e-02  1.7021400e+00 -6.1871800e-01 -1.2189820e+00
  -5.3780000e-02  8.4399400e-01 -1.0029080e+00 -8.7242700e-01]
 [ 6.3510200e-01 -1.1579710e+00 -4.5720800e-01 -6.8241800e-01
   1.0424110e+00 -1.2021480e+00 -6.6786400e-01 -1.0946800e+00]
 [ 3.9048600e-01 -4.8311700e-01 -1.3442490e+00  1.0812210e+00
   3.2570700e-01 -5.3242100e-01 -1.4639140e+00  8.1923200e-01]
 [ 1.4988000e+00  7.3197900e-01  1.2948600e-01 -1.8318000e-01
   1.8247950e+00  6.9425900e-01  5.8951900e-01 -2.0874200e-01]
 [ 1.0905760e+00 -9.5958000e-02  1.1788400e-01 -1.1382190e+00
   1.6124070e+00 -4.5957400e-01  4.8965100e-01 -1.0605630e+00]]
```

### Turn 3 — `execute_in_remote_docker` (`23962 ms` LLM)
- **Rationale**: The earlier output was truncated. Isolate the apparatus metadata, repeated levels, and axis sweeps to distinguish clipping from EMI and determine whether a censored observation needs special treatment.
- **Alternatives Weighed**: `Treat every extreme reading as an independent EMI spike` (rejected: Repeated values may be hardware rails; check their levels and whether the actual observations reach them before choosing an inverse.)

```bash
cd /holdout_task && python3 - <<'PY'
import json,pickle,numpy as n
p='/tmp/hegel_continual_archive/'
for f in ['archive_manifest.json','pre_llm_gate_diagnostic.json','archive_apparatus_log.json']:
 d=json.load(open(p+f));print(f, list(d) if isinstance(d,dict) else len(d));
 if f!='archive_apparatus_log.json': print(d)
 else:
  if isinstance(d,dict):
   for k,v in d.items(): print(k,type(v).__name__,str(v[-1] if isinstance(v,list) and v else v)[:2400])
  else: print(str(d[-1])[:3000])
b=pickle.load(open('public_batch.pkl','rb'));print('studies',len(b))
for it in b:
 s=it['study']; c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];z=s['y_obs_raw'].reshape(-1,4); a=c['single_bank_sweeps'];print('keys',list(s));print('groups',n.unique(n.count_nonzero(x,axis=1),return_counts=True),'obs range',z.min(0),z.max(0));print('pulses',[(list(t),n.shape(t.get('y_pulse_raw'))) for t in s.get('reference_tracer_pulses') or []]);print('cal rails',[(lambda v:(v.tolist()))(n.unique(y[:,j].round(4),return_counts=True)[0][n.unique(y[:,j].round(4),return_counts=True)[1]>1]) for j in range(4)]); print('axes rows\n',n.c_[x,y][n.count_nonzero(x,axis=1)<2].round(5)); print('single bank\n',n.c_[a['bank_id'],a['ref_signal'],a['raw_readout']].round(5));print('gross rows\n',n.c_[x,y][(abs(y)>3).any(1)].round(5))
PY
sed -n '36,125p' /holdout_task/_verified_downstream.py
```
- **Container Exit Code**: `0` (`464 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nzero(state == 0)
        g = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            state[[i, j]] = 0
            continue
        F = H[np.ix_(free, free)]
        rhs = -c[free] - H[free] @ np.where(state == 0, 0.0, w)
        ones = np.ones(len(free))
        a, b = np.linalg.solve(F, np.column_stack((rhs, ones))).T
        nu = (budget - np.sum(w[state != 0]) - a.sum()) / b.sum()
        wf = a + nu * b
        step = wf - w[free]
        hits = np.full(len(free), np.inf)
        lower, upper = wf < -tol, wf > cap + tol
        hits[lower] = -w[free][lower] / step[lower]
        hits[upper] = (cap - w[free][upper]) / step[upper]
        k = int(np.argmin(hits))
        if hits[k] <= 1.0:
            w[free] += np.clip(hits[k], 0.0, 1.0) * step
            state[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0.0 if lower[k] else cap
            continue
        w[free] = np.clip(wf, 0.0, cap)
        g = H @ w + c
        violation = state * (g - nu)
        i = int(np.argmax(violation))
        if violation[i] <= tol:
            return w, float(nu)
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_study: dict) -> dict:
    S, U_last = _transport(cal_study)
    y_cal = cal_study['y_obs_raw']
    Z = np.asarray(cal_study['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + cal_study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y_cal - Z @ (Pz @ y_cal)
    ridge = cal_study['source_ridge']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    cap, budget = float(cal_study['u_box']), float(cal_study['total_budget'])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
```

### Turn 4 — `execute_in_remote_docker` (`18018 ms` LLM)
- **Rationale**: The previous output omitted the axis sweeps and observation range. Those determine whether to fit cross-channel nonlinearities and whether the inverse must handle clipped operating readings rather than just ignore clipped calibration samples.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as n
n.set_printoptions(precision=6,suppress=True,linewidth=130)
b=pickle.load(open('public_batch.pkl','rb')); print('studies',len(b))
for item in b:
 s=item['study']; c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];z=s['y_obs_raw'].reshape(-1,4); print('counts',n.unique(n.count_nonzero(x,axis=1),return_counts=True),'obs min/max',z.min(0),z.max(0));print('pulses',[(list(t),n.shape(t.get('y_pulse_raw'))) for t in s.get('reference_tracer_pulses') or []]);print('axis designs and responses'); print(n.c_[x,y][n.count_nonzero(x,axis=1)<2]); print('repeated levels', [n.unique(y[abs(y[:,j])<3,j].round(4),return_counts=True)[0][n.unique(y[abs(y[:,j])<3,j].round(4),return_counts=True)[1]>1] for j in range(4)]);print('obs rows');print(z)
PY
```
- **Container Exit Code**: `0` (`380 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.05    ]
 [ -0.4        0.         0.         0.        -0.705173  -0.059998  -0.048756  -0.049999]
 [  0.5        0.         0.         0.         1.006307  -0.060009   0.177535  -0.050011]
 [  1.3        0.         0.         0.         1.686653  -0.060017   0.283608  -0.050002]
 [  0.        -1.2        0.         0.         0.069983  -1.060567   0.079998  -0.342113]
 [  0.        -0.4        0.         0.         0.069995  -0.554398   0.080004  -0.133798]
 [  0.         0.5        0.         0.         0.069982   0.537185   0.079994   0.013477]
 [  0.         1.3        0.         0.         0.07001    0.971078   0.07998    0.082486]
 [  0.         0.        -1.2        0.        -0.382426  -0.059998  -1.476445  -0.050011]
 [  0.         0.        -0.4        0.        -0.05978   -0.060008  -0.689064  -0.050021]
 [  0.         0.         0.5        0.         0.168326  -0.060004   1.008928  -0.050011]
 [  0.         0.         1.3        0.         0.275204  -0.059991   1.683926  -0.049991]
 [  0.         0.         0.        -1.2        0.070011  -0.348575   0.079998  -1.06294 ]
 [  0.         0.         0.        -0.4       13.569996 -12.942777  12.479989 -13.750496]
 [  0.         0.         0.         0.5       13.569993 -12.797284  12.480009 -12.64546 ]
 [  0.         0.         0.         1.3       13.570008 -12.729118  12.479995 -12.206177]]
repeated levels [array([0.07]), array([-0.06]), array([0.08]), array([-0.05])]
obs rows
[[ 0.660911  0.13231   0.224853  0.31916 ]
 [ 0.344799 -0.045443  0.146158 -0.150199]
 [ 0.174147  0.040914  0.103483  0.192072]
 [ 0.273418  0.093066  0.620516  0.064958]
 [ 0.511035 -0.11658   0.171124 -0.028258]
 [ 0.366935  0.266144  0.21378  -0.062684]
 [ 0.372658  0.057673  0.797136 -0.190545]
 [ 0.095636 -0.01303   0.157754  0.318835]
 [ 0.163272 -0.114816 -0.054229  0.050902]
 [ 0.245354 -0.13492  -0.028879 -0.039643]
 [ 0.712458 -0.188442  0.384985 -0.066468]
 [ 0.286334 -0.268966  0.543294 -0.067144]
 [-0.223896  0.035173  0.18649  -0.019926]
 [ 0.68745   0.052847 -0.058443  0.006807]
 [ 0.122187 -0.081625  0.538439 -0.18365 ]
 [ 0.190391 -0.052924  0.493755  0.158328]
 [ 0.486094 -0.054946 -0.139477  0.008885]
 [ 0.385401  0.097818  0.05172  -0.15557 ]
 [ 0.403468 -0.151359  0.509465 -0.10625 ]
 [ 0.056544 -0.029308  0.266591  0.056094]
 [ 0.295633  0.032554  0.365449  0.014748]
 [ 0.360897  0.093485 -0.057921  0.244515]
 [ 0.550527  0.014222  0.454702  0.139839]
 [ 0.264311  0.032662  0.449754  0.022533]]
```

### Turn 5 — `execute_in_remote_docker` (`41280 ms` LLM)
- **Rationale**: Compare affine, additive, sum/difference, and saturating forward maps on held-out axis and mixed sweeps. This distinguishes missing response terms from outlier contamination; the three gross rows are well separated from the remaining readings.
- **Alternatives Weighed**: `Fit an unconstrained four-channel inverse polynomial` (rejected: Axis sweeps reveal two channel pairs; forward fits can test that structure before adding unsupported cross-pair terms.)

```bash
cd /holdout_task && cat > /tmp/rosette_models.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; good=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5
X=X[good];Y=Y[good]; inds=np.arange(len(X)); axis=np.count_nonzero(X,axis=1)==1
print('single-bank range',[(j, c['single_bank_sweeps']['raw_readout'][c['single_bank_sweeps']['bank_id']==j].min(),c['single_bank_sweeps']['raw_readout'][c['single_bank_sweeps']['bank_id']==j].max()) for j in range(4)])

def basis(x,k):
    return np.column_stack([np.prod(x**p,axis=1) for p in product(range(k+1),repeat=2) if sum(p)<=k])

def f(p,x,kind):
    u,v=x.T; a=u+v; b=u-v
    if kind.startswith('poly'): return basis(x,int(kind[4:]))@p
    if kind=='add': return p[0]+p[1]*np.tanh(p[2]*u)+p[3]*np.tanh(p[4]*v)
    if kind=='mix': return p[0]+p[1]*np.tanh(p[2]*a)+p[3]*np.tanh(p[4]*b)
    if kind=='mixlin': return f(p,x,'mix')+p[5]*a+p[6]*b
    if kind=='mixshift': return p[0]+p[1]*np.tanh(p[2]*a+p[5])+p[3]*np.tanh(p[4]*b+p[6])
    if kind=='mixquad': return f(p,x,'mixshift')+p[7]*a+p[8]*b+p[9]*a*a+p[10]*b*b+p[11]*a*b
    if kind=='mixrot': return p[0]+p[1]*np.tanh(p[2]*u+p[4]*v+p[3])+p[5]*np.tanh(p[6]*u+p[7]*v+p[8])
    if kind=='addrot': return f(p,x,'mixrot')
    if kind=='soft': return p[0]+p[1]*a/np.sqrt(1+(p[2]*a)**2)+p[3]*b/np.sqrt(1+(p[4]*b)**2)
    if kind=='rat': return p[0]+p[1]*a/(1+(p[2]*a)**2)+p[3]*b/(1+(p[4]*b)**2)

def fit(x,y,kind):
    if kind.startswith('poly'): return np.linalg.lstsq(basis(x,int(kind[4:])),y,rcond=None)[0]
    p={'add':[0,1.5,1,.3,1],'mix':[0,1,1,.6,1],'mixlin':[0,1,1,.6,1,0,0],'mixshift':[0,1,1,.6,1,0,0],'mixquad':[0,1,1,.6,1,0,0,0,0,0,0,0],'mixrot':[0,1,1,0,1,.6,1,-1,0],'addrot':[0,1.5,1,0,0,.3,0,1,0],'soft':[0,1,1,.6,1],'rat':[0,1,.25,.6,.25]}[kind]
    return least_squares(lambda p:f(p,x,kind)-y,p,max_nfev=300).x

if __name__=='__main__':
 rng=np.random.default_rng(2026); fold=np.empty(len(X),int)
 for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
 for kind in ['poly1','poly3','poly5','add','mix','mixlin','mixshift','mixquad','mixrot','addrot','soft','rat']:
    pred=np.empty_like(Y); params=[]
    for j in range(4):
       x=X[:,[j,j^2]]; y=Y[:,j]; p=fit(x,y,kind); params.append(p)
       for k in range(5):
          tr=fold!=k; pred[~tr,j]=f(fit(x[tr],y[tr],kind),x[~tr],kind)
    e=pred-Y; print(kind,'rms',np.sqrt((e**2).mean(0)).round(6),'axis',round(np.sqrt(np.mean(e[axis]**2)),6),'mixed',round(np.sqrt(np.mean(e[~axis]**2)),6),'max',round(abs(e).max(),6))
    if kind in ['mixshift','mixquad','mixrot','addrot']: print(' parameters', np.array(params).round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_models.py
```
- **Container Exit Code**: `0` (`7633 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
442400e+00  3.095200e-01
   2.498600e-01  4.350600e-01]
 [ 1.187780e+01  3.787118e+01  2.346000e-02  1.756040e+00  3.858900e-01
  -3.384100e-01  3.316000e-01]
 [-1.383430e+00  3.560670e+00  1.634800e-01  1.180930e+00  3.721400e-01
   2.799400e-01  3.301300e-01]]
mixquad rms [0.202245 0.131592 0.221516 0.132323] axis 0.13358 mixed 0.184154 max 0.915259
 parameters [[ 9.745950e+00  8.794667e+01  9.739000e-02  4.530000e-02  1.145553e+01
  -1.100800e-01 -3.916300e-01 -7.546010e+00  5.348400e-01 -1.308600e-01
  -6.769000e-02 -9.640000e-03]
 [ 6.245000e-02  6.373000e-02  7.025800e+00  1.179300e-01  1.196410e+00
  -3.353000e-02 -1.129900e+00  4.842800e-01  3.012200e-01 -2.756000e-02
  -5.535000e-02 -2.581000e-02]
 [-8.691000e+00 -1.742600e-01  3.443531e+01  2.061137e+01  2.146600e-01
   2.934942e+01  4.648600e-01  8.880700e-01 -2.942930e+00 -3.139000e-02
   2.356100e-01 -2.317000e-02]
 [-1.077055e+01  6.688000e-02  6.274920e+01  2.046940e+01  1.958600e-01
  -6.305716e+01  5.864300e-01  4.958000e-01 -2.506770e+00 -3.419000e-02
   2.448000e-01 -2.090000e-02]]
mixrot rms [0.028475 0.037782 0.056265 0.036418] axis 0.025559 mixed 0.043493 max 0.257829
 parameters [[-4.311700e-01  1.835460e+00  1.101950e+00 -9.440000e-03 -8.520000e-03
   8.675500e-01 -1.970000e-03  4.993600e-01  6.612200e-01]
 [-9.394000e-01  1.154840e+00  1.273000e-02  1.006500e+00  3.502900e-01
   1.167490e+00  1.102810e+00  4.620000e-03 -1.037000e-02]
 [-6.111065e+01  6.158329e+01  1.559000e-02  2.890220e+00  3.115100e-01
   1.779940e+00  1.141850e+00  8.220000e-03 -1.773000e-02]
 [-5.727700e+00  5.977150e+00 -2.960000e-03  1.854740e+00  3.096200e-01
   1.194860e+00  1.098960e+00 -9.070000e-03 -1.916000e-02]]
addrot rms [0.028475 0.02693  0.031401 0.025596] axis 0.014272 mixed 0.030237 max 0.178645
 parameters [[-4.311800e-01  1.835460e+00  1.101950e+00 -9.440000e-03 -8.520000e-03
   8.675600e-01 -1.970000e-03  4.993600e-01  6.612300e-01]
 [-9.394200e-01  1.167490e+00  1.102810e+00 -1.037000e-02  4.620000e-03
   1.154860e+00  1.273000e-02  3.502900e-01  1.006510e+00]
 [-7.589696e+01  1.779940e+00  1.141850e+00 -1.772000e-02  8.220000e-03
   7.636983e+01  1.557000e-02  3.111000e-01  2.997820e+00]
 [-5.667880e+00  1.194850e+00  1.098970e+00 -1.916000e-02 -9.070000e-03
   5.917250e+00 -2.960000e-03  3.097900e-01  1.849710e+00]]
soft rms [0.206656 0.139324 0.194767 0.132425] axis 0.113052 mixed 0.181031 max 0.452754
rat rms [0.206314 0.138838 0.19485  0.132467] axis 0.113355 mixed 0.180799 max 0.453081
```

### Turn 6 — `execute_in_remote_docker` (`40807 ms` LLM)
- **Rationale**: Fit candidate saturation families with linear variable projection, report held-out axis and mixed errors, and identify repeated worst residual rows that could instead reflect clipping or remaining corruption.
- **Alternatives Weighed**: `Adopt the degree-five polynomial now` (rejected: Its mixed-sweep OOF RMSE is 0.0225 with a 0.267 maximum error, leaving room to identify a better-conditioned forward map.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine map or a sum of two zero-centered tanh responses captures the paired sensor response.` | Witness: `After gross EMI removal, affine mixed-sweep OOF RMSE is 0.18558 and additive tanh mixed-sweep RMSE is 0.049506, far above the approximately 0.00001 variation at repeated axis baselines.` | Blanket Diagnosis: `Axis sweeps identify channel pairs; asymmetric residuals remain on mixed sweeps after gross EMI exclusion.` | Sublated Pivot: `Keep paired coupling and test rational or saturating components with asymmetric additive and interaction corrections.`

```bash
cat > /tmp/rosette_discover.py <<'PY'
from rosette_models import X,Y,axis,basis
import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(2026); fold=np.empty(len(X),int)
for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
# Saturation with additive corrections, then test whether off-axis terms are needed.
def sat(z,k,kind):
 if kind=='t':return np.tanh(k*z)
 if kind=='r':return z/(1+(k*z)**2)
 if kind=='s':return z/np.sqrt(1+(k*z)**2)
 if kind=='a':return np.arctan(k*z)
 if kind=='e':return z*np.exp(-(k*z)**2)
def design(x,p,spec):
 ku,kv,nu,nv,cross=spec;u,v=x.T
 cols=[np.ones(len(u))]+[u**i for i in range(1,nu+1)]+[v**i for i in range(1,nv+1)]
 if ku:cols.append(sat(u,p[0],ku))
 if kv:cols.append(sat(v,p[1],kv))
 if cross:cols += [u*v,u*u*v,u*v*v]
 return np.column_stack(cols)
def fit(x,y,spec):
 def coef(p):
  D=design(x,p,spec);return np.linalg.lstsq(D,y,rcond=1e-10)[0]
 def err(p):return design(x,p,spec)@coef(p)-y
 p=least_squares(err,[1.1,.8],bounds=(.1,4),max_nfev=70).x
 return p,coef(p)
specs=[(a,b,nu,nv,c) for a in ('t','r','s','a','e') for b in ('t','r','s','a','e') for nu,nv in [(0,2),(1,2),(2,2),(3,3)] for c in (False,True)]+[(None,None,6,6,False),(None,None,9,9,False)]
for j in range(4):
 x=X[:,[j,j^2]]; y=Y[:,j]; results=[]
 for sp in specs:
  p,b=fit(x,y,sp); e=design(x,p,sp)@b-y
  results.append((np.median(abs(e)),np.sqrt(np.mean(e**2)),sp,p,b,e))
 print('CHANNEL',j)
 for med,rms,sp,p,b,e in sorted(results,key=lambda r:r[1])[:8]:
  print(sp,'med/rms/max',np.round([med,rms,max(abs(e))],7),'largest rows',np.argsort(abs(e))[-4:].tolist(),'p',p.round(5),'b',b.round(5))
  pred=np.empty_like(y)
  for k in range(5):
   tr=fold!=k; pp,bb=fit(x[tr],y[tr],sp);pred[~tr]=design(x[~tr],pp,sp)@bb
  print('OOF rms axis/mixed',*[round(np.sqrt(np.mean((pred[m]-y[m])**2)),7) for m in (axis,~axis)])
 print('extreme exact-ish raw values',[(int(i),round(y[i],6),x[i].round(4).tolist()) for i in np.where(abs(y*100-np.round(y*100))<.002)[0] if not axis[i]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_discover.py; grep -inE 'tanh|rosette|saturat' /tmp/hegel_continual_archive/*_solver.py | head -25
```
- **Container Exit Code**: `0` (`4051 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/mixed 0.0014579 0.0070602
('e', 'e', 3, 3, True) med/rms/max [0.0043478 0.0053514 0.0150229] largest rows [53, 52, 31, 44] p [0.67095 0.22514] b [-5.53100e-02 -2.25520e-01 -6.10000e-04  1.77490e-01 -7.75970e+00
 -5.63100e-02  3.87770e-01  1.55417e+00  7.93492e+00  1.05000e-03
  2.10000e-04  2.20000e-04]
OOF rms axis/mixed 0.0047319 0.0072103
('e', 'r', 3, 3, True) med/rms/max [0.0043471 0.0053515 0.0150109] largest rows [53, 52, 31, 44] p [0.67079 0.12447] b [-5.531000e-02 -2.265400e-01 -6.100000e-04  1.776600e-01 -4.195692e+01
 -5.630000e-02  6.383900e-01  1.555180e+00  4.213211e+01  1.050000e-03
  2.100000e-04  2.200000e-04]
OOF rms axis/mixed 0.004731 0.0071957
('e', 't', 3, 3, True) med/rms/max [0.004347  0.0053515 0.0150097] largest rows [53, 52, 31, 44] p [0.67077 0.19451] b [-5.531000e-02 -2.266400e-01 -6.100000e-04  1.776700e-01 -5.276551e+01
 -5.630000e-02  6.533100e-01  1.555280e+00  2.721787e+02  1.050000e-03
  2.100000e-04  2.200000e-04]
OOF rms axis/mixed 0.0047308 0.0072
('e', 's', 3, 3, True) med/rms/max [0.0043469 0.0053515 0.0150075] largest rows [53, 52, 31, 44] p [0.67074 0.13436] b [-5.531000e-02 -2.268200e-01 -6.100000e-04  1.777000e-01 -8.238484e+01
 -5.630000e-02  7.308600e-01  1.555460e+00  8.256002e+01  1.050000e-03
  2.000000e-04  2.200000e-04]
OOF rms axis/mixed 0.0047308 0.0071953
('e', 'a', 3, 3, True) med/rms/max [0.0043468 0.0053515 0.0150056] largest rows [53, 52, 31, 44] p [0.67072 0.14395] b [-5.5310000e-02 -2.2698000e-01 -6.1000000e-04  1.7773000e-01
 -1.1714467e+02 -5.6300000e-02  7.9607000e-01  1.5556100e+00
  8.1500503e+02  1.0500000e-03  2.0000000e-04  2.2000000e-04]
OOF rms axis/mixed 0.0047306 0.0071951
('r', 'e', 3, 3, True) med/rms/max [0.0042473 0.005354  0.0151225] largest rows [53, 52, 31, 44] p [0.43689 0.22424] b [-5.53500e-02 -3.30362e+00 -5.20000e-04  3.55690e-01 -7.92305e+00
 -5.63200e-02  3.92600e-01  4.63327e+00  8.09838e+00  1.04000e-03
  2.00000e-04  2.60000e-04]
OOF rms axis/mixed 0.004767 0.0071766
extreme exact-ish raw values []
/tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py:40:    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
/tmp/hegel_continual_archive/beta_m100_c01_batch_01_solver.py:41:    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py:39:    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py:40:    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
```

### Turn 7 — `execute_in_remote_docker` (`36361 ms` LLM)
- **Rationale**: The fitted partner's even-power coefficients match a squared sine, and the self-response is predominantly odd. Test those structures with shared nonlinear scales and held-out axis and mixed sweeps; retain the additive polynomial if no compact fit improves them.
- **Alternatives Weighed**: `Use the degree-nine polynomial directly` (rejected: A compact saturating law could fit at the measurement noise floor while avoiding polynomial extrapolation and spurious inverse roots.)

```bash
cat > /tmp/rosette_exact.py <<'PY'
from rosette_models import X,Y,axis
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
rng=np.random.default_rng(812); fold=np.empty(len(X),int)
for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
fs={'tanh':np.tanh,'erf':erf,'sin':np.sin,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'exp':lambda x:x*np.exp(-x*x)}
def d(x,k,sp):
 u=x;v=x[:,[2,3,0,1]]
 cols=[np.ones_like(u),v,np.sin(k[-1]*v)**2];
 for i,m in enumerate(sp):cols.append(u if m=='lin' else fs[m](k[i]*u))
 return np.stack(cols,axis=-1)
def fit(x,y,sp):
 def cf(k):
  a=d(x,k,sp); return np.stack([np.linalg.lstsq(a[:,j],y[:,j],rcond=1e-10)[0] for j in range(4)])
 def err(k):return (np.einsum('njk,jk->nj',d(x,k,sp),cf(k))-y).ravel()
 opts=[least_squares(err,p,bounds=(.08,3.5),max_nfev=120,gtol=1e-11) for p in starts[len(sp)]]
 o=min(opts,key=lambda r:r.cost);return np.std(o.fun),o.x,cf(o.x)
starts={1:[[1,1],[.6,1]],2:[[1,1,1],[.5,1.5,1],[1.5,.5,1],[.7,.85,1]]}
if __name__=='__main__':
 results=[]
 for sp in [(f,) for f in fs]+list(combinations_with_replacement([*fs,'lin'],2)):
  rms,k,c=fit(X,Y,sp);results.append((rms,sp,k,c))
 for rms,sp,k,c in sorted(results,key=lambda r:r[0])[:10]:
  pred=np.empty_like(Y)
  for i in range(5):
   m=fold!=i; _,kk,cc=fit(X[m],Y[m],sp);pred[~m]=np.einsum('njk,jk->nj',d(X[~m],kk,sp),cc)
  e=pred-Y;print(sp,'train',rms,'OOF max/rms',np.abs(e).max(),np.sqrt(np.mean(e**2,axis=0)), 'axis/mixed',*[np.sqrt(np.mean(e[m]**2)) for m in (axis,~axis)]);print('scales',k,'coefficients',c.round(7))
 for j in range(4):
  u=X[:,j];v=X[:,j^2]; a=np.column_stack([np.ones(len(u)), *[u**i for i in range(1,12,2)],v,np.sin(v)**2]); c=np.linalg.lstsq(a,Y[:,j],rcond=None)[0];print('odd+sin2',j,c,'rms',np.std(a@c-Y[:,j]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_exact.py
```
- **Container Exit Code**: `0` (`4034 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-8.37333556e+01
   1.08944994e+02]
 [-5.00047000e-02  1.72199000e-01 -9.84018000e-02 -5.44547602e+01
   7.08525653e+01]]
('rat', 'exp') train 0.00027262964780013307 OOF max/rms 0.002939102506460234 [0.00032295 0.00025703 0.00045929 0.00020001] axis/mixed 0.00012584431515397983 0.0003514363928553974
scales [0.39095444 0.33887856 0.99988392] coefficients [[  0.0699975   0.2667131  -0.1523039  40.5316223 -40.6638757]
 [ -0.0599811   0.1701171  -0.0972188  25.8588511 -25.9443195]
 [  0.0800528   0.2646275  -0.1513165  40.2348233 -40.3691157]
 [ -0.0500011   0.1721989  -0.0984004  26.1687214 -26.2541074]]
('tanh', 'soft') train 0.0002834754917984059 OOF max/rms 0.003131660872654596 [0.00033409 0.00027584 0.00049357 0.00020875] axis/mixed 0.00013046605459434554 0.000373336737242258
scales [0.4381435  0.37648978 0.99984077] coefficients [[ 6.99929000e-02  2.66712300e-01 -1.52310300e-01 -2.29903215e+02
   2.73039232e+02]
 [-5.99804000e-02  1.70119200e-01 -9.72192000e-02 -1.46689009e+02
   1.74210671e+02]
 [ 8.00591000e-02  2.64626500e-01 -1.51329500e-01 -2.28260001e+02
   2.71084469e+02]
 [-5.00051000e-02  1.72199200e-01 -9.84018000e-02 -1.48440239e+02
   1.76291501e+02]]
('tanh', 'atan') train 0.0003236788212095829 OOF max/rms 0.0035706364229644016 [0.00038041 0.00031602 0.00056419 0.00023987] axis/mixed 0.00014842097112742337 0.0004267755537546991
scales [0.40202643 0.43960869 0.99981234] coefficients [[ 6.99918000e-02  2.66713400e-01 -1.52298000e-01 -1.67188798e+02
   1.57595871e+02]
 [-5.99762000e-02  1.70122600e-01 -9.72223000e-02 -1.06679663e+02
   1.00557529e+02]
 [ 8.00670000e-02  2.64627900e-01 -1.51346200e-01 -1.66009551e+02
   1.56481179e+02]
 [-5.00058000e-02  1.72198100e-01 -9.84018000e-02 -1.07948357e+02
   1.01754389e+02]]
odd+sin2 0 [ 6.99992991e-02  2.06372649e+00 -8.26661859e-01  2.57984200e-01
 -3.80168748e-02  3.10373489e-03 -1.26073590e-04  2.66700394e-01
 -1.52401018e-01] rms 8.070758990684052e-06
odd+sin2 1 [-6.00006039e-02  1.31622814e+00 -5.27137829e-01  1.64333052e-01
 -2.40910134e-02  1.92922654e-03 -7.45189609e-05  1.70098696e-01
 -9.71999403e-02] rms 9.158658126716365e-06
odd+sin2 2 [ 8.00011468e-02  2.04748246e+00 -8.20190397e-01  2.56059885e-01
 -3.78119094e-02  3.11117489e-03 -1.28675191e-04  2.64598973e-01
 -1.51201289e-01] rms 9.985258820526446e-06
odd+sin2 3 [-5.00002668e-02  1.33249658e+00 -5.33772783e-01  1.66593823e-01
 -2.45527218e-02  2.00228693e-03 -8.06293004e-05  1.72199786e-01
 -9.83992460e-02] rms 9.156825198860575e-06
```

### Turn 8 — `execute_in_remote_docker` (`47862 ms` LLM)
- **Rationale**: Implement the verified forward structure with row-level EMI rejection and Jacobian-based inversion, then run the public validator and check the delivered inverse on the actual observation range.
- **Alternatives Weighed**: `Deploy the additive degree-nine polynomial` (rejected: The trigonometric model reaches the measurement noise floor and avoids high-degree polynomial behavior beyond the sweeps.)

```bash
cd /holdout_task && cat > upstream_transducer.py <<'PY'
"""Calibrate the paired rosette readouts before the canonical PDE solve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_PARTNER = np.array([2, 3, 0, 1])


def _design(ref):
    partner = ref[:, _PARTNER]
    return np.stack((np.ones_like(ref), ref, np.sin(2.5 * ref),
                     partner, np.sin(partner)**2), axis=-1)


def _fit(ref, raw):
    # EMI corrupts all four readings of a sweep; reject at the row level.
    center = np.median(raw, axis=0)
    spread = np.median(np.abs(raw - center), axis=0)
    clean = np.all(np.abs(raw - center) <= 10 * np.maximum(spread, 1e-12), axis=1)
    design = _design(ref)

    def fit(mask):
        return np.stack([np.linalg.lstsq(design[mask, j], raw[mask, j], rcond=None)[0]
                         for j in range(4)])

    coef = fit(clean)
    for _ in range(3):
        residual = np.abs(np.einsum('njk,jk->nj', design, coef) - raw)
        scale = np.median(residual[clean], axis=0)
        refined = clean & np.all(residual <= 10 * np.maximum(scale, 1e-12), axis=1)
        if refined.sum() < 5 or np.array_equal(refined, clean):
            break
        clean = refined
        coef = fit(clean)
    return coef


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    offset, slope, sine, cross, even = coef.T
    diag = np.arange(4)

    def forward(x):
        partner = x[..., _PARTNER]
        return offset + slope*x + sine*np.sin(2.5*x) + cross*partner + even*np.sin(partner)**2

    def jacobian(x):
        jac = np.zeros(x.shape[:-1] + (4, 4))
        jac[..., diag, diag] = slope + 2.5*sine*np.cos(2.5*x)
        jac[..., diag, _PARTNER] = cross + even*np.sin(2*x[..., _PARTNER])
        return jac

    target = raw.reshape(-1, 4)
    start = np.linalg.solve(jacobian(np.zeros(4)), (target - offset).T).T
    estimate = start.copy()
    # Vectorized Newton on the central branch, with a trust-region solve for
    # rows for which a full Newton step does not improve the residual.
    active = np.arange(len(target))
    for _ in range(12):
        x = estimate[active]
        residual = forward(x) - target[active]
        pending = np.max(np.abs(residual), axis=1) > 1e-12
        active, x, residual = active[pending], x[pending], residual[pending]
        if not len(active):
            break
        jac = jacobian(x)
        regular = np.abs(np.linalg.det(jac)) > 1e-10
        active, x, residual, jac = active[regular], x[regular], residual[regular], jac[regular]
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        proposal = x - step
        improves = (np.linalg.norm(forward(proposal) - target[active], axis=1)
                    < np.linalg.norm(residual, axis=1))
        active = active[improves]
        estimate[active] = proposal[improves]
    for i in np.flatnonzero(np.max(np.abs(forward(estimate) - target), axis=1) > 1e-10):
        estimate[i] = least_squares(lambda x: forward(x) - target[i], start[i],
                                    jac=jacobian, gtol=1e-13, xtol=1e-13, ftol=1e-13).x
    return estimate.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    result = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    if study.get('reference_tracer_pulses') is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            if pulse.get('y_pulse_raw') is not None else dict(pulse)
            for pulse in study['reference_tracer_pulses']
        ]
    banks = sweeps['single_bank_sweeps']
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=np.asarray(banks['ref_signal']).copy()),
    )
    return result
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
python3 - <<'PY'
import pickle,numpy as np
import upstream_transducer as u
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];b=u._fit(x,y);print('coefficients',b.round(7));out=u.calibrate_upstream(s); z=out['y_obs_raw'].reshape(-1,4);print('calibrated obs min/max',z.min(0),z.max(0),'round trip max',abs(np.einsum('njk,jk->nj',u._design(z),b)-s['y_obs_raw'].reshape(-1,4)).max());print('pulse collection',type(s.get('reference_tracer_pulses')).__name__)
PY
```
- **Container Exit Code**: `0` (`724 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
coefficients [[ 0.0699995  1.2699992  0.317498   0.2667    -0.1524009]
 [-0.0599998  0.8099965  0.2025048  0.1700991 -0.0972004]
 [ 0.0800009  1.2600027  0.3149996  0.2645998 -0.151202 ]
 [-0.0500001  0.8200012  0.205001   0.1721997 -0.0983993]]
calibrated obs min/max [-0.15294979 -0.16171667 -0.1335084  -0.11905426] [0.32675031 0.25970093 0.35397342 0.28526387] round trip max 7.161493620344572e-13
pulse collection NoneType

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba10a815b633d4a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba10a81 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`37446 ms` LLM)
- **Rationale**: Test the exact deliverable in an isolated process. Held-out sweep errors can reveal calibration bias; a pulse grid spanning and extending beyond the public operating range can reveal inverse branch or shape failures; the composed solve checks that normalization preserves feasibility and the certificate.
- **Alternatives Weighed**: `Submit after the public schema check alone` (rejected: The public study has no tracer pulses, and an observation round trip cannot detect a biased fit or a wrong inverse branch.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import copy, pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
from batch_solver import solve_batch
s = pickle.load(open('public_batch.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
is_axis = np.count_nonzero(x, axis=1) == 1
clean = np.max(np.abs(y - np.median(y, axis=0)), axis=1) < 5
rng = np.random.default_rng(318)
fold = np.empty(len(x), int)
for mask in (is_axis, ~is_axis):
    fold[mask] = rng.permutation(mask.sum()) % 5
pred = np.empty_like(y)
inv_err = []
for k in range(5):
    tr = fold != k; te = ~tr
    b = u._fit(x[tr], y[tr])
    pred[te] = np.einsum('njk,jk->nj', u._design(x[te]), b)
    inner = te & clean & (np.max(np.abs(x), axis=1) <= .6)
    if np.any(inner): inv_err.extend(u._inverse(y[inner], b) - x[inner])
metrics = {name: float(np.sqrt(np.mean((pred[mask & clean]-y[mask & clean])**2, axis=0)).max()) for name,mask in [('axis',is_axis),('mixed',~is_axis)]}
assert max(metrics.values()) < 3e-5, metrics
assert len(inv_err) and np.max(np.abs(inv_err)) < 8e-5
print('Exact artifact OOF worst-channel RMSE by sweep slice:', metrics, '| direct inverse central-sweep max:', np.max(np.abs(inv_err)), 'rows:', len(inv_err))
# Check pulse shape, optional entries, and data ownership on the central operating branch.
from itertools import product
ref = np.array(list(product((-.50,-.25,0,.25,.50), repeat=4)))
p = ref[:, [2,3,0,1]]
raw = np.array([.07,-.06,.08,-.05]) + np.array([1.27,.81,1.26,.82])*(ref+.25*np.sin(2.5*ref)+.21*p-.12*np.sin(p)**2)
probe = copy.deepcopy(s)
probe['reference_tracer_pulses'] = [{'y_pulse_raw': raw.reshape(5,125,4)}, {'y_pulse_raw': None}]
old_raw = probe['y_obs_raw'].copy(); old_cal = probe['sensor_calibration_sweeps']['raw_rosette'].copy()
out = u.calibrate_upstream(probe)
pulse = out['reference_tracer_pulses'][0]['y_pulse_raw']
e = np.max(np.abs(pulse.reshape(-1,4)-ref))
assert pulse.shape == (5,125,4) and e < 3e-5, e
assert out['reference_tracer_pulses'][1]['y_pulse_raw'] is None
assert out['y_obs_raw'].shape == old_raw.shape
assert np.array_equal(probe['y_obs_raw'], old_raw) and np.array_equal(probe['sensor_calibration_sweeps']['raw_rosette'],old_cal)
a = out['sensor_calibration_sweeps']; bank = a['single_bank_sweeps']
assert np.array_equal(a['raw_rosette'], a['ref_rosette']) and not np.shares_memory(a['raw_rosette'], a['ref_rosette'])
assert np.array_equal(bank['raw_readout'],bank['ref_signal']) and not np.shares_memory(bank['raw_readout'],bank['ref_signal'])
print('625-point pulse grid max inverse error:', e)
r = solve_batch(s); w = np.asarray(r['weights']); cert = r['kkt_certificate']; gap = cert['primal_objective'] - cert['dual_lower_bound']
assert np.isfinite(r['terminal_field']).all() and w.min() >= -1e-9 and w.max() <= s['u_box'] + 1e-9 and abs(w.sum()-s['total_budget']) < 1e-8 and abs(gap) < 1e-8, (w, gap)
print('Composed solve budget error:', abs(w.sum()-s['total_budget']), 'duality gap:', gap)
sys.argv = ['validate_public_schema.py', '/holdout_task/batch_solver.py']; runpy.run_path('/holdout_task/validate_public_schema.py', run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`444 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Exact artifact OOF worst-channel RMSE by sweep slice: {'axis': 1.0476926015908765e-05, 'mixed': 1.1145585964020103e-05} | direct inverse central-sweep max: 1.5051988130454497e-05 rows: 7
625-point pulse grid max inverse error: 3.997805677113941e-06
Composed solve budget error: 2.220446049250313e-16 duality gap: 5.204170427930421e-17
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba10a815b633d4a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba10a81 (directory).
```

### Turn 10 — `submit_completed_task` (`14967 ms` LLM)
- **Rationale**: The verified artifacts passed the public schema, held-out axis and mixed sweep checks, and synthetic pulse inversion and normalization checks in an isolated Python process. The composed solve preserved the budget to 2.22e-16 with a duality gap of 5.20e-17.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:204', '/holdout_task/upstream_transducer.py': 'FILE:3971'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust rosette calibration in `/holdout_task/upstream_transducer.py` and verified the composed `/holdout_task/batch_solver.py` pipeline. The public schema passed. Worst held-out channel RMSE was 1.11×10⁻⁵, and the 625-point synthetic pulse grid had a maximum inverse error of 4.00×10⁻⁶.`

### Turn 11 — `execute_in_remote_docker` (`35552 ms` LLM)
- **Rationale**: Compare additive and pairwise sum/difference rational alternatives on real held-out sweeps, measure inverse error through the calibration edges, and check a global Jacobian bound. A failed inverse or materially better alternative would change the implementation. Inspect the study and telemetry fields to determine whether a batch-stratified comparison is applicable.
- **Alternatives Weighed**: `Repeat the central-range round-trip check` (rejected: It cannot expose inverse error on real calibration-edge measurements or establish that the fitted map has a unique inverse.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import json, pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
batch = pickle.load(open('public_batch.pkl', 'rb'))
telemetry = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
print('Telemetry fields:', list(telemetry) if isinstance(telemetry, dict) else sorted({k for row in telemetry if isinstance(row, dict) for k in row}))
for item in batch:
    s = item['study']; c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
    print('Study fields:', sorted(s)); print('Sweep fields:', sorted(c))
    axis = np.count_nonzero(x, axis=1) == 1
    clean = np.max(np.abs(y - np.median(y, axis=0)), axis=1) < 5
    edge = np.max(np.abs(x), axis=1) >= 1.2
    rng = np.random.default_rng(901); fold = np.empty(len(x), int)
    for a in (axis, ~axis): fold[a] = rng.permutation(a.sum()) % 5
    pred = {name: np.zeros_like(y) for name in ('trig','rational_additive','rational_sum_diff')}
    err = np.zeros_like(x); margins = []
    def design(xj, p, paired):
        a, b = xj.T
        q, r = (a+p[2]*b, a-p[2]*b) if paired else (a,b)
        return np.column_stack((np.ones(len(a)), a, b, np.sin(b)**2, q/(1+(p[0]*q)**2), r/(1+(p[1]*r)**2)))
    for k in range(5):
        tr, te = fold != k, fold == k
        b = u._fit(x[tr], y[tr]); offset, slope, sine, cross, even = b.T
        margins.append(np.min(slope-2.5*abs(sine)-abs(cross)-abs(even)))
        pred['trig'][te] = np.einsum('njk,jk->nj', u._design(x[te]), b)
        good = te & clean
        err[good] = u._inverse(y[good], b) - x[good]
        for name, paired in [('rational_additive',False),('rational_sum_diff',True)]:
            for j in range(4):
                z = x[:,[j,j^2]]; yy = y[tr & clean,j]; zz = z[tr & clean]
                def fit(p): return np.linalg.lstsq(design(zz,p,paired), yy, rcond=None)[0]
                def res(p): return design(zz,p,paired) @ fit(p) - yy
                p = min((least_squares(res, [f,f,1], bounds=([.03,.03,.05],[8,8,3]), max_nfev=150) for f in (.25,1.25)), key=lambda r:r.cost).x
                pred[name][te,j] = design(z[te],p,paired) @ fit(p)
    for name, sl in [('axis',axis),('mixed',~axis),('edge',edge)]:
        m = clean & sl
        print(name, 'rows', m.sum(), 'OOF worst-channel forward RMSE', {key:float(np.sqrt(np.mean((value[m]-y[m])**2,axis=0)).max()) for key,value in pred.items()}, 'inverse max', np.max(abs(err[m])))
    print('Minimum global Jacobian dominance margin across folds:', min(margins), 'worst held-out inverse max:', abs(err[clean]).max())
    assert min(margins) > 0
    assert abs(err[clean]).max() < 2e-4
PY
```
- **Container Exit Code**: `0` (`1384 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Telemetry fields: ['completed_campaign_index', 'completed_campaign_short_name', 'facility_telemetry_trace']
Study fields: ['Lx', 'Ly', 'V_reaction_2d', 'Z_nuisance', 'aux_manifold_telemetry', 'kappa_x', 'kappa_y', 'mu_shifts', 'nuisance_ridge', 'nx', 'ny', 'robin_x', 'robin_y', 'row_bank_id', 'sensor_calibration_sweeps', 'sensor_ij', 'source_modes', 'source_ridge', 'total_budget', 'u_box', 'vel_x', 'vel_y', 'y_obs_raw']
Sweep fields: ['raw_rosette', 'ref_rosette', 'single_bank_sweeps']
axis rows 13 OOF worst-channel forward RMSE {'trig': 1.0542305407772384e-05, 'rational_additive': 0.008114820043159977, 'rational_sum_diff': 0.008337640079755364} inverse max 2.393306300429998e-05
mixed rows 64 OOF worst-channel forward RMSE {'trig': 1.085711086519479e-05, 'rational_additive': 0.029168697425543455, 'rational_sum_diff': 0.02973293508276132} inverse max 0.0001039023648354842
edge rows 50 OOF worst-channel forward RMSE {'trig': 1.0682105313558452e-05, 'rational_additive': 0.032576999745531174, 'rational_sum_diff': 0.03330512992307834} inverse max 0.0001039023648354842
Minimum global Jacobian dominance margin across folds: 0.036431018713126206 worst held-out inverse max: 0.0001039023648354842

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8ba10a815b633d4a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8ba10a81 (directory).
```

### Turn 12 — `submit_completed_task` (`16367 ms` LLM)
- **Rationale**: The verified bundle passed isolated execution, public schema validation, and pulse shape and inversion checks. The real-data edge audit stayed below the 2e-4 inverse error threshold and showed a positive global Jacobian dominance margin in every fold. The public data contains no labeled multi-batch cohort or relay modes to stratify.
- **Alternatives Weighed**: `Replace the trigonometric response with an additive or sum/difference rational model` (rejected: Both rational alternatives exceeded 0.032 worst-channel RMSE on held-out edge sweeps, versus 1.07e-5 for the delivered model.)
- **Submitted Final Answer / Summary**: `Implemented robust rosette calibration in `/holdout_task/upstream_transducer.py` and verified the composed `batch_solver.py` pipeline. The public schema passed. Worst held-out channel RMSE was 1.11×10⁻⁵, and the maximum inverse error at the measured calibration edges was 1.04×10⁻⁴.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
