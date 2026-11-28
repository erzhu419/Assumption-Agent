# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c04_batch_04__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `287,764` (`276,745` / `102,784` / `11,019`) |
| **Total Episode Wall Time (s)** | `343.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`19748 ms` LLM)
- **Rationale**: Inspect the staged interface to avoid applying sensor calibration twice, and summarize diagnostics and campaign history to determine which wall models can be tested.
- **Alternatives Weighed**: `Implement the baseline operator immediately` (rejected: The calibrated interface and available wall-regime evidence determine whether the baseline is appropriate.)

```bash
cd /holdout_task && ls -la && cat batch_solver.py validate_public_schema.py _verified_upstream.py _upstream_param_payload.json && ls -lah /tmp/hegel_continual_archive && python3 - <<'PY'
import json,pickle
from pathlib import Path
from _verified_upstream import calibrate_upstream
b=pickle.load(open('public_batch.pkl','rb')); print('batch count',len(b)); i=b[0]; print('item keys',list(i)); s=i['study']; c=calibrate_upstream(s,json.load(open('_upstream_param_payload.json'))); print('raw, calibrated keys',list(s),list(c));
for k,v in c.items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else repr(v)[:140] if not hasattr(v,'shape') or v.ndim==0 else '')
d=i.get('campaign_diagnostics'); print('diagnostics', None if d is None else (len(d), [(k,getattr(v,'shape',None)) for k,v in d[0].items()] if d else []))
for p in [Path('public_asset_index.json'),*Path('/tmp/hegel_continual_archive').glob('*.json')]:
 a=json.loads(p.read_text()); print('\n',p, list(a) if isinstance(a,dict) else type(a),str(a)[:3200])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rs, coef[9:]))
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = z - step
        best = error[active]
        for _ in range(18):
            worse = np.max(np.abs(_basis(trial) @ coef - y[active]), axis=1) >= best
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return the study in canonical readout coordinates without mutating it."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    normalized = dict(sweeps, raw_rosette=ref.copy())
    if 'single_bank_sweeps' in sweeps:
        single = sweeps['single_bank_sweeps']
        normalized['single_bank_sweeps'] = dict(single, raw_readout=np.copy(single['ref_signal']))
    result = dict(study, sensor_calibration_sweeps=normalized,
                  y_obs_raw=_invert(study['y_obs_raw'], coef))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
{"gains": [0.80999431, 1.28000134, 1.29999971, 0.78999157], "offsets": [-0.04999795, 0.08000369, 0.0699984, -0.06000104]}total 52K
drwxr-xr-x 2 root root 4.0K Nov 28 03:44 .
drwxrwxrwt 1 root root 4.0K Nov 28 03:44 ..
-r--r--r-- 1 root root  19K Nov 28 03:44 archive_apparatus_log.json
-rw-r--r-- 1 root root  675 Nov 28 03:44 archive_manifest.json
-rw-r--r-- 1 root root 5.3K Nov 28 03:44 beta_m100_c04_batch_01_solver.py
-rw-r--r-- 1 root root 5.1K Nov 28 03:44 beta_m100_c04_s00_solver.py
-rw-r--r-- 1 root root  127 Nov 28 03:44 pre_llm_gate_diagnostic.json
batch count 1
item keys ['study', 'campaign_diagnostics']
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
TypeError: calibrate_upstream() takes 1 positional argument but 2 were given
```

### Turn 2 — `execute_in_remote_docker` (`17312 ms` LLM)
- **Rationale**: Use the upstream function’s actual one-argument signature, then inspect available pulse evidence and the prior operator to determine which boundary laws can be tested.
- **Alternatives Weighed**: `Assume the previous campaign closure still applies` (rejected: Current diagnostics or chronological telemetry may distinguish a changed wall law.)

```bash
cd /holdout_task && python3 - <<'PY'
import json,pickle
from pathlib import Path
from _verified_upstream import calibrate_upstream
b=pickle.load(open('public_batch.pkl','rb')); i=b[0]; s=i['study']; c=calibrate_upstream(s)
print('batch count',len(b))
for k,v in c.items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else repr(v)[:250] if not hasattr(v,'shape') or v.ndim==0 else '')
d=i.get('campaign_diagnostics'); print('diagnostics',None if d is None else (len(d),list(d[0]) if d else []))
for p in [Path('public_asset_index.json'),*Path('/tmp/hegel_continual_archive').glob('*.json')]:
 a=json.loads(p.read_text()); print('\n',p,list(a) if isinstance(a,dict) else type(a),str(a)[:5500]);
 if isinstance(a,dict) and 'apparatus' in p.name:
  for k,v in a.items():
   if isinstance(v,list): print(k,'len',len(v),'last',str(v[-2:])[:4300])
PY
cat /tmp/hegel_continual_archive/beta_m100_c04_batch_01_solver.py
```
- **Container Exit Code**: `0` (`393 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(1.0, cap)
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        fixed = np.flatnonzero(state != 0)
        if len(free):
            A = H[np.ix_(free, free)]
            rhs = -f[free] - H[np.ix_(free, fixed)] @ w[fixed]
            cf = np.linalg.solve(A, np.column_stack((rhs, np.ones(len(free)))))
            nu = (budget - w[fixed].sum() - cf[:, 0].sum()) / cf[:, 1].sum()
            dest = cf[:, 0] + nu * cf[:, 1]
            delta = dest - w[free]
            hits = np.full(len(free), np.inf)
            down = dest < -tol
            up = dest > cap + tol
            hits[down] = -w[free[down]] / delta[down]
            hits[up] = (cap - w[free[up]]) / delta[up]
            j = int(np.argmin(hits))
            if hits[j] < 1.0:
                w[free] += max(0.0, hits[j]) * delta
                state[free[j]] = -1 if down[j] else 1
                w[free[j]] = 0.0 if down[j] else cap
                continue
            w[free] = dest
        grad = H @ w + f
        if not len(free):
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            a, b = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[a] >= grad[b] - tol:
                return w, float((grad[a] + grad[b]) / 2)
            state[[a, b]] = 0
            continue
        violation = state * (grad - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol * max(1.0, np.max(np.abs(grad))):
            return np.clip(w, 0.0, cap), float(nu)
        state[j] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _forward(study)
    S, y = _project(study, S, _calibrate(study))
    cap, budget, ridge = (float(study[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, cap, budget)
    res = S @ w - y
    g = S.T @ res + ridge * w
    obj = float((res @ res + ridge * (w @ w)) / 2)
    d = g - nu
    dual = obj - g @ w + nu * budget + cap * np.minimum(0., d).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., d).tolist(),
                                'lam_upper': np.maximum(0., -d).tolist(),
                                'primal_objective': obj, 'dual_lower_bound': float(dual)}}
```

### Turn 3 — `execute_in_remote_docker` (`23286 ms` LLM)
- **Rationale**: Read the archived boundary formulas and apparatus metadata separately so output truncation does not hide candidate wall mechanisms. Calibrate the pulses through the staged upstream pulse interface and inspect their excitation and boundary coverage before designing a held-out comparison.
- **Alternatives Weighed**: `Infer the wall law from the study's main readout` (rejected: The main readout confounds wall effects with unknown source weights and nuisance terms; the diagnostic pulses have known weights.)

```bash
cd /holdout_task && cat /tmp/hegel_continual_archive/{archive_manifest,pre_llm_gate_diagnostic}.json && for p in /tmp/hegel_continual_archive/*solver.py; do echo "$p"; sed -n '1,75p' "$p"; done && python3 - <<'PY'
import json,pickle,numpy as np
from _verified_upstream import calibrate_upstream
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
print('log type',type(a).__name__, 'keys',list(a) if isinstance(a,dict) else len(a))
for k,v in (a.items() if isinstance(a,dict) else [('rows',a)]):
 if isinstance(v,list):
  print('log',k,len(v));
  for r in v: print({kk:(str(vv)[:320] if isinstance(vv,(dict,list)) else vv) for kk,vv in r.items()} if isinstance(r,dict) else str(r)[:320]);
 else: print(k,str(v)[:500])
i=pickle.load(open('public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(i['study'],reference_tracer_pulses=i['campaign_diagnostics'])); print('pulse weights',np.array([r['w_pulse'] for r in s['reference_tracer_pulses']])); print('pulse readout norm',np.linalg.norm([r['y_pulse_raw'] for r in s['reference_tracer_pulses']],axis=1)); print('sensor boundary counts',[(j,int(np.sum(s['sensor_ij'][:,j]==k))) for j,n in enumerate([s['nx'],s['ny']]) for k in (0,n-1)])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.13143, 0.69198], [-0.47782'}
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c04_batch_01', 'facility_telemetry_trace': '[[0.69968, 0.34991, -0.36914, -0.18624, -0.35398, -0.61244, -0.74933, -0.77555], [0.8298, -0.03374, -0.46082, 0.01717, -0.32098, -0.81981, -0.69833, -0.8633], [0.81627, -0.51882, -0.56561, 0.1995, -0.40248, -0.64401, -0.54937, -0.75375], [0.82249, -0.69073, -0.41849, 0.41254, -0.35489, -0.01036, -0.27575, -0.36844], [0'}
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c04_batch_02', 'facility_telemetry_trace': '[[-0.52905, 0.3378, -0.5087, 0.43981, 2.61833, -0.4656, 0.71112, 0.29513], [-0.66655, -0.08686, -0.37917, 0.28092, 2.40855, -0.80304, 0.58804, -0.18712], [-0.78803, -0.49257, -0.39191, 0.07085, 2.50982, -0.73145, 0.42162, -0.55407], [-0.83606, -0.73512, -0.38856, -0.20795, 2.42663, -0.19574, 0.2236, -0.812], [-0.83852,'}
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c04_batch_03', 'facility_telemetry_trace': '[[-0.22962, 0.10719, 2.64951, -0.66575, -0.29263, 0.40309, -0.19864, -0.70584], [-0.02714, -0.38235, 2.52797, -0.83542, -0.22378, 0.78042, 0.10354, -0.93853], [0.22646, -0.68273, 2.50594, -0.87243, -0.28964, 0.79855, 0.29171, -0.79466], [0.36455, -0.88172, 2.52261, -0.82955, -0.43045, 0.19829, 0.48034, -0.42248], [0.57'}
pulse weights [[1.  0.  0.  0.5 0.  0.  0.  0. ]
 [0.  1.  0.  0.  0.5 0.  0.  0. ]
 [0.  0.  1.  0.  0.  0.5 0.  0. ]]
pulse readout norm [1.90666582 1.90945844 1.68508648]
sensor boundary counts [(0, 4), (0, 0), (1, 6), (1, 0)]
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

### Turn 4 — `execute_in_remote_docker` (`36924 ms` LLM)
- **Rationale**: Boundary coupling and row-sum loss span the local two-point wall stencil. Compare canonical closures and fit this stencil inside each pulse fold; residuals and Jacobian singular values will show whether to simplify the stencil or test a broader boundary law.
- **Alternatives Weighed**: `Fit the source weights and wall law jointly from the main readout` (rejected: The known-weight pulses isolate wall response from source estimation and permit an independent held-out test.)

```bash
cat > /tmp/wall_probe.py <<'PY'
import sys,pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,p):
    # p = (left coupling factor, right coupling factor, left Robin factor, right Robin factor)
    q=(k[1:]+k[:-1])*(.5/h**2)
    a=-q.copy(); b=-q.copy()
    a[:-1]-=np.maximum(v[1:-1],0)/h
    b[1:]-=np.maximum(-v[1:-1],0)/h
    b[0]=-p[0]*k[0]/h**2; a[-1]=-p[1]*k[-1]/h**2
    d=np.r_[-b[0]+p[2]*r/h,-a[:-1]-b[1:],-a[-1]+p[3]*r/h]
    return sparse.diags((a,d,b),(-1,0,1),format='csc')

def matrices(s,p):
    nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
    ax,ay=[axis(s['kappa_'+j],s['vel_'+j],s['L'+j]/(n-1),s['robin_'+j],q) for j,n,q in zip('xy',(nx,ny),(p[:4],p[4:]))]
    L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    eye=sparse.eye(nx*ny,format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    obs=[]
    for shift in s['mu_shifts']:
        field=splu(L+shift*eye).solve(rhs)
        obs.append(field[rows])
    return np.vstack(obs),field.T.reshape(-1,ny,nx).transpose(0,2,1)

def load():
    i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s=calibrate_upstream(dict(i['study'],reference_tracer_pulses=i['campaign_diagnostics']))
    w,y=(np.array([row[k] for row in s['reference_tracer_pulses']]).T for k in ('w_pulse','y_pulse_raw'))
    return s,w,y

if __name__=='__main__':
    s,w,y=load(); np.set_printoptions(precision=7,suppress=True)
    for name,p in [('open',[1]*8),('sealed',[1,1,0,0]*2),('double+open',[2,2,1,1]*2),('double+sealed',[2,2,0,0]*2)]:
        err=matrices(s,p)[0]@w-y
        print(name,'rel by pulse',np.linalg.norm(err,axis=0)/np.linalg.norm(y,axis=0))
    for held in [-1,0,1,2]:
        tr=[i for i in range(3) if i!=held]
        fit=least_squares(lambda p:(matrices(s,p)[0]@w[:,tr]-y[:,tr]).ravel(),[1]*8,bounds=(0,30),gtol=1e-12)
        err=matrices(s,fit.x)[0]@w-y
        print('held',held,'p',fit.x,'rel',np.linalg.norm(err,axis=0)/np.linalg.norm(y,axis=0),'jac singulars',np.linalg.svd(fit.jac,compute_uv=False))
    print('aux mean and std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_probe.py
sed -n '/def _axis/,/def _forward/p' /tmp/hegel_continual_archive/beta_m100_c04_s00_solver.py
```
- **Container Exit Code**: `0` (`2033 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
open rel by pulse [0.6169554 0.5807093 0.6102424]
sealed rel by pulse [0.1438138 0.3031958 0.1444946]
double+open rel by pulse [0.5210713 0.4694871 0.4842276]
double+sealed rel by pulse [0.1467194 0.4250641 0.2451486]
held -1 p [ 1.3854053 30.         0.2219498  0.         0.9500608  2.1447546
  0.         0.       ] rel [0.2164838 0.1730461 0.0620045] jac singulars [3.7532083 1.6224252 0.9681662 0.1254359 0.0582534 0.0456839 0.0162877
 0.000112 ]
held 0 p [ 0.5949202 30.         0.1467856  0.         1.7177508  8.4211
  0.         0.       ] rel [0.2593364 0.1423085 0.074959 ] jac singulars [3.3627046 1.0989361 0.2696731 0.1521091 0.0357822 0.0244669 0.0010837
 0.0000462]
held 1 p [0.8750026 5.5604943 0.        0.5354885 5.2236605 4.0407926 0.3122978
 0.       ] rel [0.1392353 0.358382  0.1290817] jac singulars [3.0292145 0.5779196 0.3839764 0.1550389 0.0925845 0.0038555 0.0036207
 0.0019194]
held 2 p [ 2.8587011 12.0485516  0.4890072  0.004994   1.0157341  2.001041
  0.         0.       ] rel [0.2073335 0.1820283 0.0789124] jac singulars [2.5110382 1.2436427 0.5428365 0.1036853 0.0684026 0.0176814 0.0119662
 0.0005594]
aux mean and std [ 0.1606844 -0.0818361  0.259182  -0.0751485] [0.8521072 0.6557414 1.0271099 0.7663685]
def _axis(kappa, vel, h, robin):
    n = len(kappa)
    face = (kappa[1:] + kappa[:-1]) * (0.5 / h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(vel[1:-1], 0) / h
    upper[1:] -= np.maximum(-vel[1:-1], 0) / h
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    diag = np.empty(n)
    diag[1:-1] = -lower[:-1] - upper[1:]
    diag[0] = -upper[0] + robin / h
    diag[-1] = -lower[-1] + robin / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
```

### Turn 5 — `execute_in_remote_docker` (`28145 ms` LLM)
- **Rationale**: Compare staged and payload calibration on the same pulses, then localize residuals by bank and shift. This distinguishes a readout mismatch from a wall-law omission before changing the transport operator.
- **Alternatives Weighed**: `Promote the eight-factor wall fit` (rejected: It worsens the worst held-out pulse error and pushes a weakly observed wall factor to its bound.)

```bash
cat > /tmp/readout_probe.py <<'PY'
from wall_probe import *
from _verified_upstream import _fit, _basis
import json
s,w,y=load(); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['study']; sw=raw['sensor_calibration_sweeps']; x=sw['ref_rosette']; z=sw['raw_rosette']; p=json.load(open('/holdout_task/_upstream_param_payload.json')); np.set_printoptions(precision=5,suppress=True)
c=_fit(x,z); print('coef',c); print('sweep fit error percentiles',np.percentile(np.linalg.norm(_basis(x)@c-z,axis=1),[0,25,50,75,90,95,100])); print('single bank fields',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in sw['single_bank_sweeps'].items()]); print('vel endpoints',[raw['vel_'+q][[0,-1]] for q in 'xy']); print('shifts',s['mu_shifts']); print('ij',s['sensor_ij'].tolist()); print('bank',s['row_bank_id'][:32]);
r=np.vstack([d['y_pulse_raw'] for d in item['campaign_diagnostics']]).T; a=((r.reshape(-1,4,3)-np.array(p['offsets'])[None,:,None])/np.array(p['gains'])[None,:,None]).reshape(r.shape); print('raw vs fitted vs payload norms',*[np.linalg.norm(t,axis=0) for t in (r,y,a)]); print('calibrated readout minmax by pulse',y.min(0),y.max(0))
for name,q in [('open',[1]*8),('sealed',[1,1,0,0]*2)]:
 pred=matrices(s,q)[0]@w
 for tag,t in [('raw',r),('staged',y),('payload',a)]:
  e=t-pred; print(name,tag,'rel',np.linalg.norm(e,axis=0)/np.linalg.norm(t,axis=0),'Z residual',np.linalg.norm(e-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],e,rcond=None)[0],axis=0),'channel RMSE',np.mean(e.reshape(-1,4,3)**2,axis=0)**.5)
 print('pred vs calibrated by shift first pulse',np.stack([pred[:,0],y[:,0]],axis=1).reshape(3,32,2)[:,:8,:])
for tag,t in [('staged',y),('payload',a)]:
 for bounds in [([.01,.01,-2,-2]*2,[10,10,5,5]*2)]:
  fit=least_squares(lambda q:(matrices(s,q)[0]@w-t).ravel(),[1,1,.02,.02]*2,bounds=bounds,max_nfev=160)
  print('signed stencil',tag,fit.x,np.linalg.norm(fit.fun.reshape(-1,3),axis=0)/np.linalg.norm(t,axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/readout_probe.py
```
- **Container Exit Code**: `0` (`1020 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
raw rel [0.69487 0.68022 0.71595] Z residual [1.37518 1.41616 1.43949] channel RMSE [[0.04713 0.05262 0.04761]
 [0.20633 0.20495 0.19053]
 [0.1798  0.19205 0.21528]
 [0.04999 0.05005 0.04539]]
open staged rel [0.61696 0.58071 0.61024] Z residual [1.15183 1.0863  1.01021] channel RMSE [[0.13139 0.11331 0.10253]
 [0.1236  0.12053 0.11709]
 [0.1088  0.10639 0.1108 ]
 [0.11524 0.11201 0.08694]]
open payload rel [0.53609 0.50487 0.54989] Z residual [0.83574 0.79822 0.78225] channel RMSE [[0.10017 0.08073 0.07143]
 [0.09164 0.08986 0.08159]
 [0.07261 0.08093 0.09893]
 [0.08089 0.07988 0.06672]]
pred vs calibrated by shift first pulse [[[0.12931 0.28649]
  [0.0296  0.21422]
  [0.14034 0.30843]
  [0.0486  0.22209]
  [0.05017 0.30157]
  [0.10356 0.30123]
  [0.17463 0.32512]
  [0.09546 0.26259]]

 [[0.10154 0.15812]
  [0.0221  0.09424]
  [0.11554 0.17894]
  [0.03659 0.10093]
  [0.03998 0.16857]
  [0.08531 0.17036]
  [0.14241 0.19348]
  [0.0761  0.1375 ]]

 [[0.07768 0.09983]
  [0.01594 0.04749]
  [0.09398 0.12086]
  [0.02647 0.05283]
  [0.03124 0.10836]
  [0.06945 0.11184]
  [0.1139  0.13252]
  [0.05958 0.08473]]]
sealed raw rel [0.52771 0.58708 0.49313] Z residual [1.03828 1.21876 0.98714] channel RMSE [[0.10172 0.17193 0.12303]
 [0.10616 0.08672 0.07275]
 [0.09147 0.07605 0.10264]
 [0.12623 0.1408  0.10148]]
sealed staged rel [0.14381 0.3032  0.14449] Z residual [0.26982 0.56959 0.23968] channel RMSE [[0.03749 0.07731 0.02973]
 [0.03023 0.06313 0.03413]
 [0.01647 0.03429 0.01719]
 [0.02329 0.05316 0.01122]]
sealed payload rel [0.21711 0.50858 0.33553] Z residual [0.33179 0.80849 0.47092] channel RMSE [[0.04365 0.1102  0.06156]
 [0.02618 0.08933 0.06644]
 [0.02691 0.05627 0.0259 ]
 [0.04058 0.06811 0.02836]]
pred vs calibrated by shift first pulse [[[0.2691  0.28649]
  [0.18079 0.21422]
  [0.2759  0.30843]
  [0.22786 0.22209]
  [0.2186  0.30157]
  [0.24873 0.30123]
  [0.32308 0.32512]
  [0.23447 0.26259]]

 [[0.15035 0.15812]
  [0.07709 0.09424]
  [0.16134 0.17894]
  [0.107   0.10093]
  [0.11015 0.16857]
  [0.13803 0.17036]
  [0.19532 0.19348]
  [0.12322 0.1375 ]]

 [[0.09607 0.09983]
  [0.03839 0.04749]
  [0.11091 0.12086]
  [0.05682 0.05283]
  [0.06576 0.10836]
  [0.09124 0.11184]
  [0.13444 0.13252]
  [0.07709 0.08473]]]
signed stencil staged [ 1.73005 10.       0.39771  0.60407  0.69645  0.39747 -0.05744 -0.08653] [0.20381 0.14483 0.0789 ]
signed stencil payload [ 0.697   10.       0.23826  1.42229  0.55719  0.14637 -0.04814 -0.08519] [0.22786 0.18213 0.14397]
```

### Turn 6 — `execute_in_remote_docker` (`28464 ms` LLM)
- **Rationale**: Test whether the calibrated pulses require coupling between opposite endpoints, and separate that effect from endpoint advection and face averaging. A sharp error reduction would determine which operator to implement.
- **Alternatives Weighed**: `Expand the flexible local wall fit` (rejected: Its weakly observed parameters already worsened the worst held-out pulse; test physically distinct stencil terms first.)

```bash
cat /holdout_task/_verified_upstream.py; cat /holdout_task/batch_solver.py; cat > /tmp/topology_probe.py <<'PY'
from wall_probe import load, axis
import itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
s,w,y=load()
nx,ny=s['nx'],s['ny']; rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; E=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@w

def make(j, kind):
 k,v=s['kappa_'+j],s['vel_'+j]; n=len(k); h=s['L'+j]/(n-1); r=s['robin_'+j]; a=axis(k,v,h,r,[1,1,0,0]).tolil()
 if kind=='open': return axis(k,v,h,r,[1,1,1,1])
 if kind=='double': return axis(k,v,h,r,[2,2,0,0])
 if kind=='sealed': return a.tocsc()
 for i,inside,out,sign in [(0,1,n-1,1),(n-1,n-2,0,-1)]:
  fi=.5*(k[i]+k[inside])/h**2; fo=.5*(k[i]+k[out])/h**2
  if kind=='periodic': a[i,inside]=-fi-max(-sign*v[i],0)/h; a[i,out]=-fo-max(sign*v[i],0)/h
  if kind=='periodic_noadv': a[i,inside]=-fi; a[i,out]=-fo
  if kind=='periodic_local': a[i,inside]=-k[i]/h**2-max(-sign*v[i],0)/h; a[i,out]=-k[i]/h**2-max(sign*v[i],0)/h
  if kind=='periodic_local_noadv': a[i,inside]=-k[i]/h**2; a[i,out]=-k[i]/h**2
  if kind=='sealed_face': a[i,inside]=-fi
  if kind=='sealed_upwind': a[i,inside]=-fi-max(-sign*v[i],0)/h
  a[i,i]=-a[i,inside]-a[i,out]
 return a.tocsc()

def calc(ax,ay):
 L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+V
 return np.vstack([splu(L+t*E).solve(rhs)[rows] for t in s['mu_shifts']])

kinds='open double sealed periodic periodic_noadv periodic_local periodic_local_noadv sealed_face sealed_upwind'.split(); axes=[{k:make(j,k) for k in kinds} for j in 'xy']; scores=[]
for x,yy in itertools.product(kinds,repeat=2):
 e=np.linalg.norm(calc(axes[0][x],axes[1][yy])-y,axis=0)/np.linalg.norm(y,axis=0); scores.append((max(e),x,yy,e))
for row in sorted(scores)[:15]:print(row)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/topology_probe.py
```
- **Container Exit Code**: `0` (`652 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sweeps['single_bank_sweeps']
        normalized['single_bank_sweeps'] = dict(single, raw_readout=np.copy(single['ref_signal']))
    result = dict(study, sensor_calibration_sweeps=normalized,
                  y_obs_raw=_invert(study['y_obs_raw'], coef))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return result
from __future__ import annotations
import inspect, json
import _verified_upstream as _up
import downstream_solver as _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
(1.142286613065573e-05, 'periodic', 'sealed', array([1.09633958e-05, 1.10477261e-05, 1.14228661e-05]))
(0.0005991061417781085, 'periodic', 'sealed_face', array([0.000596  , 0.00059911, 0.00042068]))
(0.0005991061417781085, 'periodic', 'sealed_upwind', array([0.000596  , 0.00059911, 0.00042068]))
(0.004930193394990747, 'periodic_local', 'sealed_face', array([0.00312465, 0.00399306, 0.00493019]))
(0.004930193394990747, 'periodic_local', 'sealed_upwind', array([0.00312465, 0.00399306, 0.00493019]))
(0.005048615125688756, 'periodic_local', 'sealed', array([0.00342098, 0.00423996, 0.00504862]))
(0.01137968629245534, 'periodic_local_noadv', 'sealed', array([0.01040602, 0.01137969, 0.0023557 ]))
(0.01168273597654758, 'periodic_local_noadv', 'sealed_face', array([0.01011139, 0.01168274, 0.00239224]))
(0.01168273597654758, 'periodic_local_noadv', 'sealed_upwind', array([0.01011139, 0.01168274, 0.00239224]))
(0.015682515570309048, 'periodic_noadv', 'sealed', array([0.00719177, 0.01568252, 0.00716277]))
(0.015982030346162707, 'periodic_noadv', 'sealed_face', array([0.00692453, 0.01598203, 0.00727081]))
(0.015982030346162707, 'periodic_noadv', 'sealed_upwind', array([0.00692453, 0.01598203, 0.00727081]))
(0.07836628547644946, 'periodic_local', 'double', array([0.07771969, 0.07836629, 0.04976364]))
(0.08228709162414885, 'periodic', 'double', array([0.08111446, 0.08228709, 0.054256  ]))
(0.09356905327561177, 'periodic_local_noadv', 'double', array([0.07108004, 0.09356905, 0.0556118 ]))
```

### Turn 7 — `execute_in_remote_docker` (`64982 ms` LLM)
- **Rationale**: Implement the identified stencil and the specified nuisance-profiled quadratic program. Select the stencil using only the training pulses within each fold, then check the implemented operator by shift and pulse for orientation or stacking mistakes.
- **Alternatives Weighed**: `Use face-averaged coupling at the sealed y walls` (rejected: It increased worst pulse error from 1.1423e-5 to 5.9911e-4.); `Replace the staged batch adapter` (rejected: It already composes dynamic upstream calibration with the required downstream entrypoint.)
- **ATMS NOGOOD DECLARED**: Refuted `Nearest-interior-neighbor boundary rows suffice on both axes.` | Witness: `Allowing periodic endpoint coupling in x and retaining sealed y reduced worst calibrated pulse error from 30.32% to 0.001142%; a flexible local stencil reached 35.84% worst held-out error.` | Blanket Diagnosis: `The local stencil excluded the opposite endpoint from the x-wall row, so local coupling and Robin adjustments could not represent wraparound transport.` | Sublated Pivot: `Add the conservative endpoint-to-endpoint upwind FVM terms in x and retain the sealed local y stencil.`

```bash
cat > /holdout_task/downstream_solver.py <<'PY'
"""Transport and source reconstruction in calibrated chamber coordinates."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, spacing, periodic):
    """Row-local upwind transport with periodic or sealed boundary rows."""
    n = len(kappa)
    faces = (kappa[:-1] + kappa[1:]) / (2 * spacing**2)
    lower = -faces.copy()
    upper = -faces.copy()
    lower[:-1] -= np.maximum(velocity[1:-1], 0) / spacing
    upper[1:] -= np.maximum(-velocity[1:-1], 0) / spacing
    if periodic:
        upper[0] -= max(-velocity[0], 0) / spacing
        lower[-1] -= max(velocity[-1], 0) / spacing
        wrap = (kappa[0] + kappa[-1]) / (2 * spacing**2)
        left_wrap = wrap + max(velocity[0], 0) / spacing
        right_wrap = wrap + max(-velocity[-1], 0) / spacing
    else:
        upper[0] = -kappa[0] / spacing**2
        lower[-1] = -kappa[-1] / spacing**2
        left_wrap = right_wrap = 0.0
    diagonal = np.r_[-upper[0] + left_wrap,
                     -lower[:-1] - upper[1:],
                     -lower[-1] + right_wrap]
    matrix = sparse.diags((lower, diagonal, upper), (-1, 0, 1), format='csc')
    if periodic:
        wrap_entries = sparse.csc_matrix(([-left_wrap, -right_wrap],
                                          ([0, n-1], [n-1, 0])), shape=(n, n))
        matrix = matrix + wrap_entries
    return matrix


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), True)
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), False)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = study['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = study['sensor_ij']
    sensor_rows = ij[:, 0] + nx * ij[:, 1]
    sampled = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(rhs)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(hessian, linear, cap, budget):
    """Feasible active-set QP with the equality multiplier signed as g - nu."""
    n = len(linear)
    if budget == 0:
        return np.zeros(n), float(np.min(linear))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(hessian @ w + linear))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1: lower; +1: upper
    tol = 1e-12 * max(1, np.linalg.norm(linear, np.inf), cap * np.linalg.norm(hessian, np.inf))
    for _ in range(100 * n):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        if free.size == 0:
            gradient = hessian @ w + linear
            low = np.where(active == -1, gradient, np.inf).argmin()
            high = np.where(active == 1, gradient, -np.inf).argmax()
            if gradient[high] <= gradient[low] + tol:
                return w, float((gradient[high] + gradient[low]) / 2)
            active[[low, high]] = 0
            continue
        block = hessian[np.ix_(free, free)]
        rhs = -linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed]
        ones = np.ones(free.size)
        base, response = np.linalg.solve(block, np.column_stack((rhs, ones))).T
        nu = (budget - w[fixed].sum() - base.sum()) / response.sum()
        target = base + nu * response
        direction = target - w[free]
        fractions = np.full(free.size, np.inf)
        below = (target < -1e-14) & (direction < 0)
        above = (target > cap + 1e-14) & (direction > 0)
        fractions[below] = -w[free[below]] / direction[below]
        fractions[above] = (cap - w[free[above]]) / direction[above]
        hit = fractions.argmin()
        if fractions[hit] < 1.0:
            w[free] += max(0.0, fractions[hit]) * direction
            active[free[hit]] = 1 if above[hit] else -1
            w[free[hit]] = cap if above[hit] else 0.0
            continue
        w[free] = np.clip(target, 0.0, cap)
        violation = active * (hessian @ w + linear - nu)
        release = violation.argmax()
        if violation[release] <= tol:
            return w, float(nu)
        active[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Reconstruct sources and terminal field and certify the profiled objective."""
    study = cal_item
    sensors, terminal_modes = _forward(study)
    z = study['Z_nuisance']
    readings = study['y_obs_raw']
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    combined = np.column_stack((sensors, readings))
    profiled = combined - z @ np.linalg.solve(gram, z.T @ combined)
    s, y = profiled[:, :-1], profiled[:, -1]
    cap, budget, ridge = (study[key] for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    g = s.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced).tolist(),
            'lam_upper': np.maximum(0.0, -reduced).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(dual),
        },
    }
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys
sys.path.insert(0, '/tmp')
import numpy as np
import topology_probe as p
from downstream_solver import _forward
candidates={(x,y):p.calc(p.axes[0][x],p.axes[1][y]) for x,y in p.itertools.product(p.kinds,repeat=2)}
for hold in range(p.w.shape[1]):
    train=np.arange(p.w.shape[1])!=hold
    key=min(candidates,key=lambda k: np.linalg.norm((candidates[k]-p.y)[:,train]))
    print('held pulse',hold,'selected',key,'relative error',np.linalg.norm((candidates[key]-p.y)[:,hold])/np.linalg.norm(p.y[:,hold]))
pred=_forward(p.s)[0]@p.w
print('implemented per shift/pulse relative error',np.linalg.norm((pred-p.y).reshape(3,32,3),axis=1)/np.linalg.norm(p.y.reshape(3,32,3),axis=1))
PY
```
- **Container Exit Code**: `0` (`923 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
(1.142286613065573e-05, 'periodic', 'sealed', array([1.09633958e-05, 1.10477261e-05, 1.14228661e-05]))
(0.0005991061417781085, 'periodic', 'sealed_face', array([0.000596  , 0.00059911, 0.00042068]))
(0.0005991061417781085, 'periodic', 'sealed_upwind', array([0.000596  , 0.00059911, 0.00042068]))
(0.004930193394990747, 'periodic_local', 'sealed_face', array([0.00312465, 0.00399306, 0.00493019]))
(0.004930193394990747, 'periodic_local', 'sealed_upwind', array([0.00312465, 0.00399306, 0.00493019]))
(0.005048615125688756, 'periodic_local', 'sealed', array([0.00342098, 0.00423996, 0.00504862]))
(0.01137968629245534, 'periodic_local_noadv', 'sealed', array([0.01040602, 0.01137969, 0.0023557 ]))
(0.01168273597654758, 'periodic_local_noadv', 'sealed_face', array([0.01011139, 0.01168274, 0.00239224]))
(0.01168273597654758, 'periodic_local_noadv', 'sealed_upwind', array([0.01011139, 0.01168274, 0.00239224]))
(0.015682515570309048, 'periodic_noadv', 'sealed', array([0.00719177, 0.01568252, 0.00716277]))
(0.015982030346162707, 'periodic_noadv', 'sealed_face', array([0.00692453, 0.01598203, 0.00727081]))
(0.015982030346162707, 'periodic_noadv', 'sealed_upwind', array([0.00692453, 0.01598203, 0.00727081]))
(0.07836628547644946, 'periodic_local', 'double', array([0.07771969, 0.07836629, 0.04976364]))
(0.08228709162414885, 'periodic', 'double', array([0.08111446, 0.08228709, 0.054256  ]))
(0.09356905327561177, 'periodic_local_noadv', 'double', array([0.07108004, 0.09356905, 0.0556118 ]))
held pulse 0 selected ('periodic', 'sealed') relative error 1.0963395793046542e-05
held pulse 1 selected ('periodic', 'sealed') relative error 1.1047726098219795e-05
held pulse 2 selected ('periodic', 'sealed') relative error 1.1422866130655732e-05
implemented per shift/pulse relative error [[9.68794545e-06 9.73388434e-06 1.00537091e-05]
 [1.24207115e-05 1.24864408e-05 1.26867549e-05]
 [1.57381730e-05 1.56511165e-05 1.60221125e-05]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fb68b774b01fd210 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fb68b774 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`39328 ms` LLM)
- **Rationale**: Use an independent dense assembly in an isolated Python process to detect field-ordering and certificate discrepancies. Stress exact and near-endpoint budgets to expose active-set failures; any failed assertion identifies which part of the solver needs correction.
- **Alternatives Weighed**: `Submit after schema and pulse validation` (rejected: Those checks do not independently verify terminal-field ordering or the KKT certificate at active bounds.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _box_simplex
from _verified_upstream import calibrate_upstream

raw = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
out = solve_batch(raw)
s = calibrate_upstream(raw)

def dense_axis(k, v, h, periodic):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        if not periodic and i in (0,n-1):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
        else:
            for step in (-1,1):
                j = (i+step)%n
                a[i,j] = -(k[i]+k[j])/(2*h**2) - max(0,-step*v[i])/h
        a[i,i] = -a[i].sum()
    return a

nx, ny = s['nx'], s['ny']
ax, ay = [dense_axis(s['kappa_'+j],s['vel_'+j],s['L'+j]/(n-1),j=='x') for j,n in zip('xy',(nx,ny))]
operator = np.kron(np.eye(ny),ax) + np.kron(ay,np.eye(nx)) + np.diag(s['V_reaction_2d'].ravel(order='F'))
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
rows = [i+nx*j for i,j in s['sensor_ij']]
fields = [np.linalg.solve(operator+t*np.eye(nx*ny),rhs) for t in s['mu_shifts']]
design = np.vstack([f[rows] for f in fields])
z = s['Z_nuisance']
profile = np.eye(len(z)) - z @ np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
a, b = profile@design, profile@s['y_obs_raw']
w, c = out['weights'], out['kkt_certificate']
cap, budget, ridge = [s[k] for k in ('u_box','total_budget','source_ridge')]
res = a@w-b
g = a.T@res+ridge*w
objective = (res@res+ridge*(w@w))/2
nu = c['nu_sum']
lo, hi = np.asarray(c['lam_lower']), np.asarray(c['lam_upper'])
field_error = np.max(np.abs(out['terminal_field']-(fields[-1]@w).reshape((nx,ny),order='F')))
stationarity = np.max(np.abs(g-nu-lo+hi))
complementarity = max(np.max(np.abs(lo*w)),np.max(np.abs(hi*(cap-w))))
dual = objective-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
assert np.array_equal(w,c['weights']) and w.min()>=0 and w.max()<=cap and abs(w.sum()-budget)<1e-10
assert lo.min()>=0 and hi.min()>=0
assert max(field_error,stationarity,complementarity,abs(c['primal_objective']-objective),abs(c['dual_lower_bound']-dual),abs(objective-dual))<1e-10
print('public field max error, stationarity, complementarity, duality gap:',field_error,stationarity,complementarity,objective-dual)

rng = np.random.default_rng(11)
worst = np.zeros(3)
for _ in range(80):
    n = int(rng.integers(2,19))
    q = rng.normal(size=(max(2,n//2),n))
    h = q.T@q + .04*np.eye(n)
    linear = rng.normal(size=n)*rng.choice([.02,1,8])
    cap = 0.42
    for budget in (0,1e-7,cap,cap*(n//2),cap*(n-.2),cap*n-1e-7,cap*n):
        w, nu = _box_simplex(h,linear,cap,budget)
        assert w.min()>=0 and w.max()<=cap
        d = h@w+linear-nu
        lower, upper = np.maximum(d,0), np.maximum(-d,0)
        worst = np.maximum(worst,[abs(w.sum()-budget),max(np.max(lower*w),np.max(upper*(cap-w))),abs(d@w-cap*np.minimum(d,0).sum())])
assert worst.max()<1e-9, worst
print('560 QP stress cases, worst budget residual, complementarity, gap:',worst)
PY
```
- **Container Exit Code**: `0` (`702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public field max error, stationarity, complementarity, duality gap: 6.938893903907228e-17 1.5057399771478686e-15 2.0543209589926326e-16 3.200564813177209e-15
560 QP stress cases, worst budget residual, complementarity, gap: [3.01425551e-14 9.54208070e-15 4.26325641e-14]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fb68b774b01fd210 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fb68b774 (directory).
```

### Turn 9 — `submit_completed_task` (`13855 ms` LLM)
- **Rationale**: All three leave-one-pulse-out folds selected the implemented wall stencil. The verified bundle passed public schema validation, independent dense field and KKT checks in isolated Python, and 560 active-set stress cases.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:5813'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/downstream_solver.py` with the existing `/holdout_task/batch_solver.py` adapter. Public schema validation passed. Worst held-out pulse relative error was 1.14×10⁻⁵, and the maximum duality gap across 560 stress cases was 4.27×10⁻¹⁴.`

### Turn 10 — `execute_in_remote_docker` (`29504 ms` LLM)
- **Rationale**: Test whether nuisance projection changes wall-law selection and whether aggregate pulse error hides larger wall-sensor errors. A ranking reversal would warrant revisiting the stencil; a localized error would direct inspection to that boundary stencil or sensor indexing.
- **Alternatives Weighed**: `Run relay, classification, or nonlinear drift checks` (rejected: This study has no relay states, class labels, multi-batch cohorts, or time-dependent drift operator. The relevant confounder is the supplied nuisance design.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, io, sys
import numpy as np
sys.path[:0] = ['/tmp', '/holdout_task']
with contextlib.redirect_stdout(io.StringIO()):
    import topology_probe as p
from downstream_solver import _forward

s = p.s
z = s['Z_nuisance']
profile = np.eye(len(z)) - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
chosen = ('periodic', 'sealed')
residuals = {(a,b): p.calc(p.axes[0][a], p.axes[1][b]) - p.y
             for a,b in p.itertools.product(p.kinds, repeat=2)}
implemented = _forward(s)[0] @ p.w - p.y
np.testing.assert_allclose(implemented, residuals[chosen], atol=1e-12, rtol=0)
for label, projection in [('canonical', np.eye(len(z))), ('nuisance-profiled', profile)]:
    scores = {key: projection @ residual for key, residual in residuals.items()}
    for held in range(p.w.shape[1]):
        train = np.arange(p.w.shape[1]) != held
        ranks = sorted((np.linalg.norm(residual[:,train]),key) for key,residual in scores.items())
        rel = np.linalg.norm(scores[chosen][:,held]) / np.linalg.norm((projection @ p.y)[:,held])
        print(label, 'held pulse', held, 'winner', ranks[0][1], 'runner-up factor', ranks[1][0]/ranks[0][0], 'held relative error', rel)
        assert ranks[0][1] == chosen
ij = s['sensor_ij']
on_x_wall = (ij[:,0] == 0) | (ij[:,0] == s['nx']-1)
on_y_wall = (ij[:,1] == 0) | (ij[:,1] == s['ny']-1)
for name, mask in [('x endpoints',on_x_wall),('y endpoints',on_y_wall),('interior',~(on_x_wall | on_y_wall))]:
    if not mask.any():
        print(name, 'no sampled sensors')
        continue
    errors = np.linalg.norm(implemented.reshape(len(s['mu_shifts']),len(ij),-1)[:,mask,:],axis=1)
    norms = np.linalg.norm(p.y.reshape(len(s['mu_shifts']),len(ij),-1)[:,mask,:],axis=1)
    print(name, 'sensor count', mask.sum(), 'worst shift-by-pulse relative error', (errors/norms).max())
PY
```
- **Container Exit Code**: `0` (`759 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
canonical held pulse 0 winner ('periodic', 'sealed') runner-up factor 47.12645374082329 held relative error 1.0963395793046542e-05
canonical held pulse 1 winner ('periodic', 'sealed') runner-up factor 47.13385993134983 held relative error 1.1047726098219795e-05
canonical held pulse 2 winner ('periodic', 'sealed') runner-up factor 54.29533376894113 held relative error 1.1422866130655732e-05
nuisance-profiled held pulse 0 winner ('periodic', 'sealed') runner-up factor 46.900615301579045 held relative error 1.1083555052510927e-05
nuisance-profiled held pulse 1 winner ('periodic', 'sealed') runner-up factor 46.76047496649418 held relative error 1.1080573025059673e-05
nuisance-profiled held pulse 2 winner ('periodic', 'sealed') runner-up factor 54.105667768454815 held relative error 1.1490323140874458e-05
x endpoints sensor count 4 worst shift-by-pulse relative error 2.0277216799818267e-05
y endpoints sensor count 6 worst shift-by-pulse relative error 2.5011363988984242e-05
interior sensor count 23 worst shift-by-pulse relative error 1.7706632329068036e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fb68b774b01fd210 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fb68b774 (directory).
```

### Turn 11 — `submit_completed_task` (`14153 ms` LLM)
- **Rationale**: All three held-out folds selected the implemented stencil with and without nuisance projection. The runner-up had at least 46.76× the training residual in every comparison. Independent dense field and KKT checks passed in isolated Python on the unchanged artifact bundle.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/downstream_solver.py` and verified it through the existing `/holdout_task/batch_solver.py` adapter. Public schema validation passed. Worst held-out pulse relative error was 1.15×10⁻⁵ after nuisance projection, and the largest duality gap across 560 stress cases was 4.26×10⁻¹⁴.`

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
