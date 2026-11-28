# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c03_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `10 / 16` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `246,000` (`236,025` / `98,048` / `9,975`) |
| **Total Episode Wall Time (s)** | `308.34s` |

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

### Turn 1 — `execute_in_remote_docker` (`16761 ms` LLM)
- **Rationale**: Inspect bounded log excerpts and study shapes first to determine whether to change the wall operator and which calibration model the sweeps support.
- **Alternatives Weighed**: `Implement the Robin baseline immediately.` (rejected: The public task allows campaign-specific walls, and the apparatus log may identify them without fitting to the inverse target.)

```bash
cd /holdout_task && ls -l && ls -l /tmp/hegel_continual_archive 2>/dev/null; cat /holdout_task/validate_public_schema.py /holdout_task/public_asset_index.json; python3 - <<'PY'
import json,pickle,numpy as np,pathlib
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
print('item keys:', list(item)); print('diagnostics:',type(item.get('campaign_diagnostics')))
for k,v in item['study'].items():
 if isinstance(v,dict):
  print(k,[(a, getattr(b,'shape',None),type(b).__name__) for a,b in v.items()]);continue
 try:
  x=np.asarray(v); print(k,x.shape,x.dtype,repr(x.ravel()[:8]))
 except Exception: print(k,type(v),str(v)[:150])
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*.json'):
 print('\n',p.name,p.stat().st_size)
 with p.open() as f: print(f.read(5500))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT'
```
- **Container Exit Code**: `0` (`462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
391,
        -0.66639
      ],
      [
        -0.36037,
        -0.58261,
        -0.18592,
        0.38593,
        0.0661,
        -0.23094,
        -0.56803,
        -0.78948
      ],
      [
        -0.37873,
        -0.21796,
        -0.64322,
        0.58961,
        -0.35811,
        -0.45668,
        -0.76926,
        -0.79578
      ],
      [
        -0.41838,
        0.23952,
        -0.81329,
        0.66588,
        -0.73108,
        -0.47781,
        -0.79925,
        -0.4716
      ],
      [
        -0.27218,
        0.58781,
        -0.47702,
        0.75411,
        -0.83514,
        -0.39778,
        -0.77459,
        -0.13785
      ],
      [
        -0.40052,
        0.82079,
        0.14315,
        0.85983,
        -0.74633,
        -0.38009,
        -0.81185,
        0.26792
      ],
      [
        -0.23097,
        0.81481,
        0.73517,
        0.83742,
        -0.47372,
        -0.28592,
        -0.72087,
        0.6537
      ],
      [
        -0.37678,
        0.591,
        0.87824,
        0.71212,
        -0.00016,
        -0.42562,
        -0.60492,
        0.85834
      ],
      [
        -0.41297,
        0.20595,
        0.45747,
        0.61825,
        0.35618,
        -0.43497,
        -0.44091,
        0.79652
      ],
      [
        -0.38749,
        -0.19535,
        -0.18814,
        0.48978,
        0.72091,
        -0.29585,
        -0.26227,
        0.51191
      ],
      [
        -0.353,
        -0.61118,
        -0.67725,
        0.30693,
        0.84643,
        -0.38894,
        0.00074,
        0.12444
      ],
      [
        -0.41226,
        -0.82258,
        -0.85468,
        0.04472,
        0.79354,
        -0.40012,
        0.19278,
        -0.21526
      ],
      [
        -0.35211,
        -0.8154,
        -0.49839,
        -0.21195,
        0.45912,
        -0.30026,
        0.45263,
        -0.63283
      ],
      [
        -0.31547,
        -0.58245,
        0.16678,
        -0.44764,
        0.08213,
        -0.42106,
        0.60586,
        -0.74874
      ]
    ]
  }
]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19734 ms` LLM)
- **Rationale**: The first inspection was truncated. Summarize the apparatus log and measure the affine sweep residual to determine whether the wall operator or calibration basis needs to change.

```bash
python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p)); print('APPARATUS TYPE',type(a).__name__)
for x in (a if isinstance(a,list) else [a]):
 for k,v in x.items():
  if isinstance(v,list): print(k,'len',len(v),'first',str(v[:2])[:900],'last',str(v[-1:])[:400])
  else: print(k,str(v)[:900])
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];print('STUDY')
for k,v in s.items():
 if isinstance(v,dict): print(k,[(a,np.shape(b) if not isinstance(b,dict) else list(b)) for a,b in v.items()]);continue
 x=np.asarray(v); print(k,x.shape,x.ravel()[:5])
c=s['sensor_calibration_sweeps']; x=c['ref_rosette'];y=c['raw_rosette'];print('CAL FIRST',x[:4],y[:4]);print('BANK SAMPLE',str(c.get('single_bank_sweeps'))[:1600]);print('affine resid',np.std(y-np.c_[np.ones(len(x)),x]@np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0],axis=0))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`399 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([-1.3       , -1.04545455, -0.79090909, -0.53636364, -0.28181818,
       -0.02727273,  0.22727273,  0.48181818,  0.73636364,  0.99090909,
        1.24545455,  1.5       , -1.3       , -1.04545455, -0.79090909,
       -0.53636364, -0.28181818, -0.02727273,  0.22727273,  0.48181818,
        0.73636364,  0.99090909,  1.24545455,  1.5       , -1.3       ,
       -1.04545455, -0.79090909, -0.53636364, -0.28181818, -0.02727273,
        0.22727273,  0.48181818,  0.73636364,  0.99090909,  1.24545455,
        1.5       , -1.3       , -1.04545455, -0.79090909, -0.53636364,
       -0.28181818, -0.02727273,  0.22727273,  0.48181818,  0.73636364,
        0.99090909,  1.24545455,  1.5       ]), 'raw_readout': array([-1.29999404, -1.04546984, -0.79091283, -0.53636053, -0.28182917,
       -0.02727698,  0.22726819,  0.48181223,  0.73634523,  0.99089334,
        1.24545408,  1.49999502, -1.30000103, -1.04544824, -0.79090427,
       -0.53638088, -0.28182874, -0.02727136,  0.22728436,  0.48181375,
        0.73636968,  0.99089675,  1.24545961,  1.50000874, -1.29999844,
       -1.04545149, -0.79092238, -0.53638518, -0.28182596, -0.02728828,
        0.22727317,  0.48182828,  0.73635957,  0.99091765,  1.24544355,
        1.50000314, -1.2999978 , -1.04545695, -0.79090901, -0.53637611,
       -0.28181305, -0.02726365,  0.22727631,  0.4818095 ,  0.73636083,
        0.99091731,  1.24544433,  1.50000227])}
affine resid [9.64771391e-06 1.02426209e-05 7.50776449e-06 9.72316815e-06]
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
```

### Turn 3 — `execute_in_remote_docker` (`59108 ms` LLM)
- **Rationale**: Implement the documented forward model and exact box-simplex KKT solve first. Shift-wise residuals can flag boundary mismatch that schema validation and a small duality gap cannot detect.
- **Alternatives Weighed**: `Infer wall coefficients directly from the facility trace.` (rejected: The trace has no channel-to-coefficient mapping and no prior solver establishes one.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kap, vel, h, rob):
    kh = (kap[:-1] + kap[1:]) * (0.5 / h**2)
    down = np.r_[-kh[:-1] - np.maximum(vel[1:-1], 0) / h, -kap[-1] / h**2]
    up = np.r_[-kap[0] / h**2, -kh[1:] - np.maximum(-vel[1:-1], 0) / h]
    main = np.r_[kap[0] / h**2 + rob / h, -(down[:-1] + up[1:]), kap[-1] / h**2 + rob / h]
    return diags((down, main, up), (-1, 0, 1), format='csc')


def _calibrate(study, raw):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    observed = np.asarray(sweeps['raw_rosette'])
    offset, *mix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), observed, rcond=None)[0]
    return np.linalg.solve(np.asarray(mix).T, (np.asarray(raw).reshape(-1, 4) - offset).T).T.ravel()


def _profile(study, values):
    z = np.asarray(study['Z_nuisance'])
    return values - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ values)


def _responses(study, axes=None):
    nx, ny = int(study['nx']), int(study['ny'])
    if axes is None:
        axes = tuple(_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]), study['L' + d] / (n - 1), study['robin_' + d]) for d, n in zip('xy', (nx, ny)))
    ax, ay = axes
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T.copy()
    sensors = np.asarray(study['sensor_ij'])
    indices = sensors[..., 0] + nx * sensors[..., 1]
    shifts = study['mu_shifts']
    if indices.ndim == 1 and len(indices) == len(study['y_obs_raw']) and len(shifts) > 1:
        indices = indices.reshape(len(shifts), -1)
    sampled = []
    identity = eye(nx * ny, format='csc')
    for k, shift in enumerate(shifts):
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[indices if indices.ndim == 1 else indices[k]])
    return np.vstack(sampled), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_qp(hessian, linear, cap, budget):
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, 0.0 if budget == 0 else cap)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    eps = 64 * np.finfo(float).eps
    grad_tol = eps * max(np.abs(linear).max(), cap * np.linalg.norm(hessian, ord=np.inf), 1e-100)
    for _ in range(100 + 10 * n):
        free = active == 0
        g = hessian @ w + linear
        if not free.any():
            lower = np.where(active < 0)[0]
            upper = np.where(active > 0)[0]
            lo, hi = lower[np.argmin(g[lower])], upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + grad_tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue
        gf = g[free]
        hf = hessian[np.ix_(free, free)]
        q, v = np.linalg.solve(hf, np.column_stack((gf, np.ones(len(gf))))).T
        nu = (q.sum() + budget - w.sum()) / v.sum()
        step = nu * v - q
        wf = w[free]
        low = step < 0
        bound = np.where(low, 0., cap)
        travel = np.full(len(step), np.inf)
        np.divide(bound - wf, step, out=travel, where=np.abs(step) > eps * cap)
        hit = travel.argmin()
        if travel[hit] < 1 - eps:
            w[free] += max(0., travel[hit]) * step
            i = np.flatnonzero(free)[hit]
            w[i] = bound[hit]
            active[i] = -1 if low[hit] else 1
            continue
        w[free] = np.clip(wf + step, 0., cap)
        g = hessian @ w + linear
        nu = float(g[free].mean())
        violation = active * (g - nu)
        hit = violation.argmax()
        if violation[hit] <= grad_tol:
            return w, nu
        active[hit] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s, terminal = _responses(study)
    y = _calibrate(study, study['y_obs_raw'])
    projected = _profile(study, np.column_stack((s, y)))
    s, y = projected[:, :-1], projected[:, -1]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_qp(s.T @ s + ridge * np.eye(s.shape[1]), -(s.T @ y), cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    obj = float(0.5 * (residual @ residual + ridge * (w @ w)))
    delta = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., delta).tolist(),
                                'lam_upper': np.maximum(0., -delta).tolist(),
                                'primal_objective': obj,
                                'dual_lower_bound': float(obj - g @ w + nu * budget + cap * np.minimum(0., delta).sum())}}
PY
python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys, pickle, json, numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as b
items=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(items))
a=np.asarray(json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))[-1]['facility_telemetry_trace']); print('telemetry mean/std',a.mean(0),a.std(0))
for it in items:
 s=it['study']; out=b.solve_batch(s); c=out['kkt_certificate']; w=out['weights']; S,U=b._responses(s); y=b._calibrate(s,s['y_obs_raw']); r=b._profile(s,y-S@w); print('w',w,'gap',c['primal_objective']-c['dual_lower_bound'],'robin',s['robin_x'],s['robin_y'],'residual RMS by shift',np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1)**2,axis=1))); print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('sensor range',s['sensor_ij'].min(0),s['sensor_ij'].max(0))
PY
```
- **Container Exit Code**: `0` (`565 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
items 1
telemetry mean/std [-3.55525417e-01  4.84166667e-03  2.58583333e-03 -1.55958333e-03
  2.65833333e-04 -3.89467500e-01 -4.99208333e-03  1.36566667e-02] [0.06632241 0.60569699 0.59779727 0.59488489 0.59788013 0.07122488
 0.59146729 0.58740939]
w [0.09927976 0.39264459 0.06833916 0.23814779 0.32090477 0.03939993
 0.1609986  0.0802854 ] gap 1.1622647289044608e-16 robin 1.1134569414466113 1.1384086450100832 residual RMS by shift [0.01133138 0.00735419 0.00960234]
aux mean/std [-0.00333952 -0.1966902   0.11527029  0.13295292] [0.86725983 1.22114801 1.04777781 1.06830623]
sensor range [0 0] [12 10]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0dfa2e8fc1a19824 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0dfa2e8f (directory).
```

### Turn 4 — `execute_in_remote_docker` (`31507 ms` LLM)
- **Rationale**: The KKT gap verifies optimization, not wall physics. Hold out shifts and sensor banks to check whether a plausible wall change improves prediction, and use leave-one-out sweep error to check whether nonlinear calibration terms are warranted.
- **Alternatives Weighed**: `Fit free wall parameters to all observations.` (rejected: Without tracer pulses or a documented telemetry mapping, an in-sample gain could come from fitting noise or trading off wall losses against source weights.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=b._calibrate(s,s['y_obs_raw']);z=s['Z_nuisance'];ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; n=len(y); r=np.arange(n); shifts=len(s['mu_shifts']); banks=s['row_bank_id']; m=len(s['source_modes']); groups={'shift':r//(n//shifts), 'bank':banks};print('rosettes consistent',np.all(banks.reshape(-1,4)==np.arange(4)))
def forward(rx,ry,kind):
 axes=[]
 for d,f in zip('xy',(rx,ry)):
  k=s['kappa_'+d]; h=s['L'+d]/(len(k)-1); a=b._axis(k,s['vel_'+d],h,f*s['robin_'+d]);
  if kind=='wrap':
   a=a.tolil(); c=.5*(k[0]+k[-1])/h**2; a[0,-1]=-c; a[-1,0]=-c; a[0,0]+=c; a[-1,-1]+=c; a=a.tocsc()
  axes.append(a)
 return b._responses(s,axes)[0]
def evaluate(S,mask):
 A=S[mask]; Y=y[mask]; Z=z[mask]; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); T=A-Z@(P@A); R=Y-Z@(P@Y); w,_=b._box_qp(T.T@T+ridge*np.eye(m),-T.T@R,cap,budget); return y-S@w-z@(P@(Y-A@w)),w
for label,rx,ry,kind in [('Robin',1,1,''),('sealed',0,0,''),('half Robin',.5,.5,''),('double Robin',2,2,''),('sealed x',0,1,''),('sealed y',1,0,''),('periodic',0,0,'wrap')]:
 S=forward(rx,ry,kind); e,w=evaluate(S,np.ones(n,bool)); print(label,'full %.6f'%np.std(e), 'slices', {k:[round(np.sqrt(np.mean(evaluate(S,g!=j)[0][g==j]**2)),6) for j in np.unique(g)] for k,g in groups.items()})
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; ycal=c['raw_rosette']; i,j=np.triu_indices(4); A=np.column_stack((np.ones(len(x)),x)); B=np.column_stack((A,x[:,i]*x[:,j],x**3));
for a in (A,B):
 q=np.linalg.qr(a,mode='reduced')[0]; err=(ycal-q@(q.T@ycal))/(1-(q*q).sum(1))[:,None];print('calibration basis',a.shape[1],'LOO per-channel RMS',np.sqrt((err**2).mean(0)))
PY
```
- **Container Exit Code**: `0` (`482 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rosettes consistent True
Robin full 0.009537 slices {'shift': [0.013593, 0.008369, 0.010239], 'bank': [0.010588, 0.009969, 0.010377, 0.010271]}
sealed full 0.059375 slices {'shift': [0.147398, 0.059826, 0.05179], 'bank': [0.090252, 0.092551, 0.086951, 0.100678]}
half Robin full 0.012434 slices {'shift': [0.026201, 0.015497, 0.012462], 'bank': [0.017129, 0.020878, 0.017347, 0.021375]}
double Robin full 0.011678 slices {'shift': [0.01938, 0.012225, 0.014863], 'bank': [0.019629, 0.013673, 0.014611, 0.014361]}
sealed x full 0.015573 slices {'shift': [0.03291, 0.020146, 0.014174], 'bank': [0.020666, 0.029295, 0.021164, 0.023808]}
sealed y full 0.019337 slices {'shift': [0.042942, 0.022246, 0.01801], 'bank': [0.030829, 0.026602, 0.027793, 0.035777]}
periodic full 0.062685 slices {'shift': [0.162879, 0.066438, 0.048309], 'bank': [0.096723, 0.105208, 0.106012, 0.105606]}
calibration basis 5 LOO per-channel RMS [1.03007852e-05 1.08120138e-05 8.04958706e-06 1.04405295e-05]
calibration basis 19 LOO per-channel RMS [1.22535726e-05 1.24979781e-05 9.49056642e-06 1.20698990e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0dfa2e8fc1a19824 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0dfa2e8f (directory).
```

### Turn 5 — `execute_in_remote_docker` (`39618 ms` LLM)
- **Rationale**: Verify the unchanged candidate against an independent dense operator and certificate reconstruction. These checks can expose indexing or multiplier errors missed by schema validation; bound-heavy budgets test active-set behavior not exercised by the public solution.
- **Alternatives Weighed**: `Change the wall model or expand the calibration basis.` (rejected: Every tested wall alternative worsened held-out prediction, and polynomial calibration worsened leave-one-out sweep error.)

```bash
set -e
python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
nx,ny=s['nx'],s['ny']; m=len(s['source_modes']); cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']
def axis(d,n):
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(n-1); a=np.zeros((n,n))
    for i in range(1,n-1):
        km=(k[i-1]+k[i])/2; kp=(k[i+1]+k[i])/2
        a[i,i-1]=-km/h**2-max(v[i],0)/h; a[i,i+1]=-kp/h**2-max(-v[i],0)/h
        a[i,i]=(km+kp)/h**2+abs(v[i])/h
    for i,j in ((0,1),(n-1,n-2)):
        a[i,i]=k[i]/h**2+s['robin_'+d]/h; a[i,j]=-k[i]/h**2
    return a
L=np.kron(np.eye(ny),axis('x',nx))+np.kron(axis('y',ny),np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
rhs=np.column_stack([u.flatten(order='F') for u in s['source_modes']]); ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; blocks=[]
for shift in s['mu_shifts']:
    sol=np.linalg.solve(L+shift*np.eye(nx*ny),rhs); blocks.append(sol[rows])
S=np.concatenate(blocks); U=np.array([x.reshape(nx,ny,order='F') for x in sol.T]); Sa,Ua=b._responses(s)
forward=max(np.max(abs(S-Sa)),np.max(abs(U-Ua))); assert forward<1e-12
cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; raw=cal['raw_rosette']; xc=x-x.mean(0); M=np.linalg.solve(xc.T@xc,xc.T@(raw-raw.mean(0))); y=(((s['y_obs_raw'].reshape(-1,4)-raw.mean(0))@np.linalg.inv(M))+x.mean(0)).ravel()
z=s['Z_nuisance']; P=np.eye(len(y))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); A=P@S; r=P@y
out=b.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; low=np.array(c['lam_lower']); up=np.array(c['lam_upper']); g=A.T@(A@w-r)+ridge*w; J=(np.linalg.norm(A@w-r)**2+ridge*(w@w))/2; D=J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
field=np.max(abs(out['terminal_field']-np.einsum('i,ijk->jk',w,U))); assert field<1e-12
assert np.array_equal(w,c['weights']); assert min(w)>=0 and max(w)<=cap and abs(sum(w)-budget)<1e-12
assert min(low)>=0 and min(up)>=0; assert max(np.max(abs(g-nu-low+up)),np.max(abs(low*w)),np.max(abs(up*(cap-w))),abs(J-c['primal_objective']),abs(D-c['dual_lower_bound']),abs(J-D))<1e-11
rng=np.random.default_rng(62); worst=0.; feasible=0.; count=0
for n in (2,8,20):
    for k in range(40):
        x=rng.normal(size=(n,n)); H=x.T@x+.04*np.eye(n); f=6*rng.normal(size=n)
        for B in (0.,cap,cap*(n-1),cap*n,rng.uniform(0,cap*n)):
            q,t=b._box_qp(H,f,cap,B); gg=H@q+f-t; gap=gg@q-cap*np.minimum(gg,0).sum()+t*(q.sum()-B)
            feasible=max(feasible,abs(q.sum()-B),max(q)-cap,-min(q)); worst=max(worst,abs(gap)); count+=1
assert feasible<1e-11 and worst<1e-9,(feasible,worst)
print(f'dense forward error={forward:.3g}, terminal field error={field:.3g}, public independent KKT gap={abs(J-D):.3g}; {count} active-set stress cases: worst gap={worst:.3g}, worst feasibility error={feasible:.3g}')
PY
```
- **Container Exit Code**: `0` (`696 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
dense forward error=2.22e-16, terminal field error=1.67e-16, public independent KKT gap=6.33e-16; 600 active-set stress cases: worst gap=5.68e-14, worst feasibility error=7.16e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0dfa2e8fc1a19824 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0dfa2e8f (directory).
```

### Turn 6 — `submit_completed_task` (`12976 ms` LLM)
- **Rationale**: The Robin model outperformed every tested wall alternative on held-out slices, and affine calibration outperformed the polynomial basis in leave-one-out validation. The unchanged artifact passed independent clean-process verification of the forward model, terminal field, and KKT certificate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5061'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, the terminal field matched an independent dense solve within 1.67e-16, and the worst KKT duality gap across 600 stress cases was 5.68e-14.`

### Turn 7 — `execute_in_remote_docker` (`43637 ms` LLM)
- **Rationale**: The earlier audit tested large wall changes and polynomial mixing but could miss a nearby boundary stencil or a rational calibration correction. Compare these on real held-out measurements with model selection confined to training folds; change the solver only if the improvement holds out of sample.
- **Alternatives Weighed**: `Test relay initialization, HJB drift, and balanced accuracy.` (rejected: This task has a steady linear forward operator and a continuous inverse problem, with no relay state, stochastic-control drift, or class labels. The applicable nuisance check is held-out prediction across shifts and banks with nuisance coefficients fit on training rows only.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
it=pickle.load(open(p+'public_batch.pkl','rb'))[0]; s=it['study']; c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); print('study keys',list(s)); print('sweep count',len(x),'bank sweep structure',str(type(c.get('single_bank_sweeps'))),str(c.get('single_bank_sweeps'))[:300]);print('diagnostics type',type(it.get('campaign_diagnostics')).__name__)
A=np.column_stack((np.ones(len(x)),x)); terms=np.column_stack([x[:,i]+sign*x[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)]); i,j=np.triu_indices(4); mult=x[:,i]*x[:,j]; triple=np.column_stack([x[:,[i,j,k]].prod(axis=1) for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]); dictionary=[('affine',A),('products',np.column_stack((A,mult,triple)))];
for name,q in [('pair sums',terms),('single channels',x)]:
 for a in (.01,.1,1.):
  for t in range(q.shape[1]): dictionary.append((f'{name} {t} {a}',np.column_stack((A,A[:,1:]/(1+a*q[:,t:t+1]**2)))))
def loo(F,Y):
 Q,S,_=np.linalg.svd(F,full_matrices=False);Q=Q[:,S>S[0]*1e-12];return (Y-Q@(Q.T@Y))/(1-np.sum(Q*Q,axis=1))[:,None]
err=np.empty_like(y); choices={}
for i in range(len(x)):
 keep=np.arange(len(x))!=i; idx=min(range(len(dictionary)),key=lambda j:np.mean(loo(dictionary[j][1][keep],y[keep])**2)); name,F=dictionary[idx]; choices[name]=choices.get(name,0)+1;err[i]=y[i]-F[i]@np.linalg.lstsq(F[keep],y[keep],rcond=1e-12)[0]
print('affine LOO channel RMS',np.sqrt(np.mean(loo(A,y)**2,axis=0)));print('nested dictionary channel RMS',np.sqrt(np.mean(err**2,axis=0)),'choices',choices)
# Test nearby discrete boundary stencils with training-only wall selection.
from scipy.sparse import csc_matrix
Y=b._calibrate(s,s['y_obs_raw']); z=s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']; r=s['source_ridge']; m=len(s['source_modes']); n=len(Y); group=np.repeat(np.arange(len(s['mu_shifts'])),n//len(s['mu_shifts'])); bank=np.asarray(s['row_bank_id']);
D={}
for mode in ('baseline','face-average','half-cell','wall-upwind','advection-flux'):
 for fac in ((1.,) if mode!='baseline' else (1.,.8,1.2)):
  aa=[]
  for d in 'xy':
   k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1); rob=s['robin_'+d]*fac; ax=b._axis(k,v,h,rob).toarray()
   for i,j,sign in ((0,1,-1),(-1,-2,1)):
    diff=(.5*(k[i]+k[j]) if mode=='face-average' else k[i])/h**2; leak=rob/h
    if mode=='half-cell': diff*=2;leak*=2
    if mode=='wall-upwind': diff+=max(sign*v[i],0)/h
    if mode=='advection-flux': leak+=max(sign*v[i],0)/h
    ax[i,i]=diff+leak; ax[i,j]=-diff
   aa.append(csc_matrix(ax))
  D[mode,fac]=b._responses(s,aa)[0]
def fit(S,mask):
 Z=z[mask]; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); A=S[mask]; y=Y[mask]; T=A-Z@(P@A); t=y-Z@(P@y); w,_=b._box_qp(T.T@T+r*np.eye(m),-T.T@t,cap,budget); return Y-S@w-z@(P@(y-A@w))
for name,G,inner in [('shift',group,bank),('bank',bank,group)]:
 scores={k:[] for k in D}; nested=[]; picks=[]
 for g in np.unique(G):
  keep=G!=g; pred={k:np.sqrt(np.mean(fit(S,keep)[~keep]**2)) for k,S in D.items()};
  for k,v in pred.items(): scores[k].append(v)
  def select(S):
   return np.mean([np.mean(fit(S,keep&(inner!=j))[keep&(inner==j)]**2) for j in np.unique(inner[keep])]);
  k=min(D,key=lambda k:select(D[k])); nested.append(pred[k]); picks.append(k)
 print(name,'worst RMS', {str(k):round(max(v),7) for k,v in scores.items()},'nested worst',round(max(nested),7),'picks',picks)
PY
```
- **Container Exit Code**: `0` (`632 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study keys ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
sweep count 80 bank sweep structure <class 'dict'> {'bank_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3]), 'ref_signal': array([-1.3       , -1.04545455, -0.79090909, -0.53636364, -0.28181818,
       -0.02727273,  0.22727273,  0
diagnostics type NoneType
affine LOO channel RMS [1.03007852e-05 1.08120138e-05 8.04958706e-06 1.04405295e-05]
nested dictionary channel RMS [1.04162670e-05 1.10641613e-05 8.23682674e-06 1.07542115e-05] choices {'affine': 78, 'pair sums 0 1.0': 1, 'pair sums 0 0.1': 1}
shift worst RMS {"('baseline', 1.0)": 0.013593, "('baseline', 0.8)": 0.0153331, "('baseline', 1.2)": 0.0140037, "('face-average', 1.0)": 0.0136144, "('half-cell', 1.0)": 0.0138813, "('wall-upwind', 1.0)": 0.013593, "('advection-flux', 1.0)": 0.013593} nested worst 0.013593 picks [('baseline', 1.0), ('face-average', 1.0), ('baseline', 1.0)]
bank worst RMS {"('baseline', 1.0)": 0.0105877, "('baseline', 0.8)": 0.0122922, "('baseline', 1.2)": 0.0124684, "('face-average', 1.0)": 0.0104716, "('half-cell', 1.0)": 0.0106222, "('wall-upwind', 1.0)": 0.0105877, "('advection-flux', 1.0)": 0.0105877} nested worst 0.0105877 picks [('baseline', 1.0), ('face-average', 1.0), ('baseline', 1.0), ('baseline', 1.0)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0dfa2e8fc1a19824 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0dfa2e8f (directory).
```

### Turn 8 — `submit_completed_task` (`15219 ms` LLM)
- **Rationale**: The focused audit did not identify a supported upgrade. The unchanged artifact already passed isolated-process verification of the forward operator, terminal field, and KKT certificate.
- **Alternatives Weighed**: `Use the face-average wall stencil.` (rejected: It slightly improved the worst bank slice but worsened the worst shift slice; nested selection did not improve either worst-slice metric.); `Add rational or cross-channel calibration terms.` (rejected: Nested leave-one-out selection increased calibration RMS on all four channels.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5061'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, the terminal field matched an independent dense solve within 1.67e-16, and the worst KKT duality gap across 600 stress cases was 5.68e-14.`

### Turn 9 — `execute_in_remote_docker` (`34461 ms` LLM)
- **Rationale**: A bank holdout can leave measurements from the same spatial location in training. Holding out each location across all shifts tests whether nuisance profiling transfers without that overlap. The separate single-bank sweeps test whether the affine calibration transfers to isolated channel inputs.
- **Alternatives Weighed**: `Repeat the nonlinear calibration and boundary stencil searches.` (rejected: Those searches already used nested validation; single-bank sweeps and whole-location holdouts test different failure modes.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
s = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = map(np.asarray, (c['ref_rosette'], c['raw_rosette']))
F = np.c_[np.ones(len(x)), x]
coef = np.linalg.lstsq(F, raw, rcond=None)[0]
bank = c['single_bank_sweeps']
print('single-bank sweep columns:', {key: np.shape(value) for key, value in bank.items()})
ids = np.asarray(bank['bank_id'])
ref = np.asarray(bank['ref_signal'])
for key, value in bank.items():
    if not key.startswith('raw'):
        continue
    observed = np.asarray(value)
    expected = coef[0] + ref[:, None] * coef[1:][ids]
    if observed.shape == expected.shape:
        err = observed - expected
    elif observed.shape == ref.shape:
        err = observed - expected[np.arange(len(ids)), ids]
    else:
        continue
    print('single-bank transfer RMS by driven bank:', [float(np.sqrt(np.mean(err[ids == j] ** 2))) for j in np.unique(ids)])
S, U = b._responses(s)
y = b._calibrate(s, s['y_obs_raw'])
z = np.asarray(s['Z_nuisance'])
cap, budget, ridge = (s[k] for k in ('u_box', 'total_budget', 'source_ridge'))
ij = np.asarray(s['sensor_ij']).reshape(-1, 2)
if len(ij) != len(y):
    ij = np.tile(ij, (len(s['mu_shifts']), 1))
groups = ij[:, 0] + s['nx'] * ij[:, 1]
assert len(groups) == len(y)
errors = {False: [], True: []}
for g in np.unique(groups):
    train = groups != g
    A, t = S[train], y[train]
    Z = z[train]
    P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    for profile in errors:
        C, d = (A - Z @ (P @ A), t - Z @ (P @ t)) if profile else (A, t)
        w, _ = b._box_qp(C.T @ C + ridge * np.eye(C.shape[1]), -(C.T @ d), cap, budget)
        pred = S[~train] @ w
        if profile:
            pred += z[~train] @ (P @ (t - A @ w))
        errors[profile].append(np.sqrt(np.mean((y[~train] - pred) ** 2)))
print('whole-location holdouts:', len(np.unique(groups)), 'worst RMS without / with training-only nuisance profiling:', max(errors[False]), max(errors[True]))
out = b.solve_batch(s)
cert = out['kkt_certificate']
w = out['weights']
assert np.isfinite(out['terminal_field']).all() and out['terminal_field'].shape == (s['nx'], s['ny'])
assert min(w) >= 0 and max(w) <= cap and abs(sum(w) - budget) < 1e-12
assert abs(cert['primal_objective'] - cert['dual_lower_bound']) < 1e-11
PY
```
- **Container Exit Code**: `0` (`441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
single-bank sweep columns: {'bank_id': (48,), 'ref_signal': (48,), 'raw_readout': (48,)}
single-bank transfer RMS by driven bank: [9.51102942240325e-06, 8.793479666274174e-06, 1.0974843333376083e-05, 6.789941756732138e-06]
whole-location holdouts: 32 worst RMS without / with training-only nuisance profiling: 0.49503659791123217 0.019336330105589085

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0dfa2e8fc1a19824 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0dfa2e8f (directory).
```

### Turn 10 — `submit_completed_task` (`14868 ms` LLM)
- **Rationale**: The independent single-bank sweeps support affine calibration, and training-only nuisance profiling improves prediction at unseen locations. The unchanged artifact passed isolated-process checks of the forward model, terminal field, and KKT certificate.
- **Alternatives Weighed**: `Omit nuisance profiling.` (rejected: Holding out all shifts at each sensor location increased worst-location RMS from 0.01934 to 0.4950.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, the terminal field matched an independent dense solve within 1.67e-16, and the worst KKT duality gap across 600 stress cases was 5.68e-14. Nuisance profiling reduced worst held-out sensor-location RMS from 0.4950 to 0.01934.`

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
