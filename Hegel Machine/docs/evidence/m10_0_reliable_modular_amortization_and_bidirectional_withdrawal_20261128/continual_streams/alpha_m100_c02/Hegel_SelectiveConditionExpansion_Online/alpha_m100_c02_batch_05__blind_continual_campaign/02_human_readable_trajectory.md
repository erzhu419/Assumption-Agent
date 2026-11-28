# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m100_c02_batch_05__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_viscoelastic_chamber_hereditary_mechanics` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_alpha` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `287,214` (`274,440` / `102,784` / `12,774`) |
| **Total Episode Wall Time (s)** | `415.01s` |

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

### Turn 1 — `execute_in_remote_docker` (`17362 ms` LLM)
- **Rationale**: Inspect the composition and array ranges first: a schema mismatch changes which fields are transformed, and the available calibration coverage determines how to test nonlinear terms without trusting extrapolation.

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_downstream.py validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def show(x,p='',level=0):
 if isinstance(x,dict) and level<3:
  for k,v in x.items(): show(v,p+'/'+str(k),level+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype, (np.nanmin(x),np.nanmax(x)) if x.size and x.dtype.kind in 'fiu' else '')
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x))
  if x and level<3: show(x[0],p+'/0',level+1)
 else: print(p,repr(x)[:220])
show(d)
PY
for f in /tmp/hegel_continual_archive/{archive_manifest.json,archive_apparatus_log.json,pre_llm_gate_diagnostic.json}; do if test -f "$f"; then echo "--- $f"; tail -c 4800 "$f"; echo; fi; done
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS multivariate calibration nonlinear inversion' 2>&1 | head -30
```
- **Container Exit Code**: `0` (`411 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
30383,
        0.0633,
        -0.88284,
        -0.64501,
        -0.36785,
        -0.64525,
        -0.5183,
        0.79839
      ],
      [
        -0.31852,
        0.48413,
        -0.60542,
        -0.52607,
        -0.30374,
        -0.11154,
        -0.66265,
        0.83176
      ],
      [
        -0.52291,
        0.8154,
        -0.0339,
        -0.31281,
        -0.35333,
        0.51664,
        -0.69581,
        0.66412
      ],
      [
        -0.51669,
        0.79579,
        0.65159,
        -0.14317,
        -0.2813,
        0.83413,
        -0.80823,
        0.32115
      ],
      [
        -0.40794,
        0.6788,
        0.84419,
        0.11819,
        -0.30671,
        0.65574,
        -0.82734,
        -0.11998
      ],
      [
        -0.56205,
        0.30995,
        0.60421,
        0.28911,
        -0.2261,
        0.11658,
        -0.79932,
        -0.52724
      ],
      [
        -0.50749,
        -0.13318,
        -0.07663,
        0.52191,
        -0.32719,
        -0.52723,
        -0.6298,
        -0.81004
      ],
      [
        -0.42544,
        -0.55316,
        -0.61021,
        0.76204,
        -0.44697,
        -0.85474,
        -0.51769,
        -0.82616
      ]
    ]
  }
]
--- /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "pre_llm_gate_accepted": false,
  "repair_target": "upstream_module_solve",
  "failure_kind": "scientific_or_numerical"
}
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`20894 ms` LLM)
- **Rationale**: Recover the field layout and bridge ranges that were obscured by truncated output. History loads outside calibration support would require extrapolation tests before selecting a nonlinear model.

```bash
cd /holdout_task; cat validate_public_schema.py; tail -28 _verified_downstream.py; python3 - <<'PY'
import json,pickle,numpy as np
from pathlib import Path
D=pickle.load(open('public_batch.pkl','rb'))
for k,v in D.items():
 print(k,({kk:(type(vv).__name__,getattr(vv,'shape',None),list(vv) if isinstance(vv,dict) else None) for kk,vv in v.items()} if isinstance(v,dict) else (type(v).__name__,getattr(v,'shape',None))))
c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw'];print('loads range',x.min(0),x.max(0),'raw range',y.min(0),y.max(0));print('sweep',[(k,np.shape(v)) for k,v in c['single_channel_sweeps'].items()]);print('first cal rows',x[:3],y[:3]);
for k in ['train','test','chamber_verification_trials']:
 v=D.get(k)
 if isinstance(v,dict):
  for f in ['past_channel_raw','hold_channel_raw']:
   a=v.get(f)
   if a is not None: print(k,f,np.min(a,axis=tuple(range(a.ndim-1))),np.max(a,axis=tuple(range(a.ndim-1))))
P=Path('/tmp/hegel_continual_archive')
for p in P.glob('*.json'):
 j=json.loads(p.read_text());print(p.name,type(j).__name__,len(j), list(j)[:10] if isinstance(j,dict) else (list(j[-1]) if j and isinstance(j[-1],dict) else None));
 if p.name=='archive_manifest.json':print(str(j)[-2100:])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -18
```
- **Container Exit Code**: `0` (`488 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
d:{k}:{a.shape}")
    if not errs:
        try:
            coupon = verify_alpha_pooled_ref_coupon_invariance(mod.solve_batch, ds, base_out=out)
            if not coupon["valid"]:
                errs.append("pooled_reference_coupon_invariance_failed")
        except (KeyError, TypeError, ValueError, IndexError):
            errs.append("invalid_output_on_reference_exchange")
print(json.dumps({"schema_valid": not errs, "errors": errs, "pooled_reference_check": coupon}, allow_nan=False))
sys.exit(1 if errs else 0)
    p = min(candidates, key=lambda candidate: candidate.cost).x
    for _ in range(2):
        a, _ = arrays(p)
        err = a @ p[2:] - target
        scales = []
        for group in (err[:n_curve], err[n_curve:]):
            scales.append(max(1e-7, 1.4826 * np.median(np.abs(group - np.median(group)))))
        p = fit(p, 1.5 * np.repeat(scales, [n_curve, len(r)])).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_downstream(cal_dataset: dict) -> dict:
    train, test = cal_dataset['train'], cal_dataset['test']
    dt, times = (np.asarray(cal_dataset[k]) for k in ('past_dt', 'eval_times'))
    past = np.sum(train['past_channel_raw'], axis=-1)
    hold = np.sum(train['hold_channel_raw'], axis=-1)
    x = _features(train, test)
    n = len(hold)
    tau, theta = _fit(past, hold, x[:n], dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])
    past_test = np.sum(test['past_channel_raw'], axis=-1)
    hold_test = np.sum(test['hold_channel_raw'], axis=-1)
    curve, rate, _, _ = _design(np.log(tau), past_test, hold_test, x[n:], dt, times)
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
{
  "ok": true,
  "target": "autogenous_robust_network_assimilation.assimilate_diffusive_network",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_robust_network_assimilation",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "2262dea7fb83d907b5e4ba036cab3459cc080447cbb668b784227f746f8023d6",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "dt",
          "op": ">",
          "value": 0
        },
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 3 — `execute_in_remote_docker` (`20808 ms` LLM)
- **Rationale**: Use the validator’s batch nesting to inspect calibration coverage and trial layout. If operational histories extend beyond calibration support, include a boundary holdout before choosing the bridge model.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,json
from pathlib import Path
batch=pickle.load(open('public_batch.pkl','rb')); print('items:',len(batch)); d=batch[0]['dataset']; print('dataset keys:',list(d))
c=d['calibration_load_cells']; print('calibration fields:',[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in c.items()]); x=c['true_channel_loads']; y=c['bridge_raw']; print('calibrated ranges:',x.min(0),x.max(0),'raw ranges:',y.min(0),y.max(0)); print('first 5 calibration pairs:',np.column_stack([x,y])[:5]); s=c['single_channel_sweeps']; print('sweep shapes:',[(k,np.shape(v)) for k,v in s.items()]); print('sweep ranges:',[(k,np.min(v),np.max(v)) for k,v in s.items() if np.asarray(v).dtype.kind in 'fiu'])
for k in ('train','test','chamber_verification_trials'):
 v=d.get(k); print(k,type(v).__name__,list(v) if isinstance(v,dict) else None)
 if isinstance(v,dict):
  for f in ('past_channel_raw','hold_channel_raw'):
   if f in v:
    a=np.asarray(v[f]); z=a.reshape(-1,4); print(f,a.shape,z.min(0),z.max(0))
p=Path('/tmp/hegel_continual_archive/archive_manifest.json'); j=json.loads(p.read_text()); print('manifest sample:',str(j)[-2700:])
p=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json'); j=json.loads(p.read_text()); print('apparatus entries:',len(j)); print('last entry summary:',[(k,(type(v).__name__,len(v)) if isinstance(v,(dict,list)) else v) for k,v in j[-1].items()])
PY
```
- **Container Exit Code**: `0` (`453 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.64600804  0.13996205
  -0.18775162  0.12006564]
 [ 1.7         0.          0.          0.          1.07572784  0.14007467
  -0.30112238  0.12013995]
 [ 0.         -1.6         0.          0.         -0.12011243 -1.69406303
  -0.11009158  0.69901393]]
sweep shapes: [('lab_id', (48,)), ('cmd_stress', (48,)), ('true_stress', (48,))]
sweep ranges: [('lab_id', 0, 3), ('cmd_stress', -1.693967912043454, 15.190071138500624), ('true_stress', -1.6, 1.6)]
train dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'strain_curves_obs', 'instant_rate_obs', 'aux_chamber_telemetry']
past_channel_raw (72, 6, 4) [-0.9583383  -1.14958309 -0.91780035 -1.13111537] [0.96432269 1.90487184 0.9970118  1.85300546]
hold_channel_raw (72, 4) [0.275274   0.67167889 0.230445   0.64075494] [0.86110953 1.70662706 0.83760287 1.63893514]
test dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'aux_chamber_telemetry']
past_channel_raw (36, 6, 4) [-0.92737416 -1.15861289 -0.92346821 -1.15642322] [1.00196968 1.86971582 0.97094618 1.83796967]
hold_channel_raw (36, 4) [0.21913425 0.63727746 0.21857352 0.73091223] [0.86011117 1.64686017 0.85499985 1.66545378]
chamber_verification_trials NoneType None
manifest sample: [{'episode_short_name': 'alpha_m100_c02_s00', 'task_id': 'alpha_m100_c02_s00__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['estimated_taus', 'hereditary_weights', 'initial_rate_pred', 'predicted_strain'], 'solver_path': '/tmp/hegel_continual_archive/alpha_m100_c02_s00_solver.py'}, {'episode_short_name': 'alpha_m100_c02_batch_01', 'task_id': 'alpha_m100_c02_batch_01__blind_continual_campaign', 'entrypoint': 'solve_batch', 'output_schema_keys': ['estimated_taus', 'hereditary_weights', 'initial_rate_pred', 'predicted_strain'], 'solver_path': '/tmp/hegel_continual_archive/alpha_m100_c02_batch_01_solver.py'}, {'episode_short_name': 'alpha_m100_c02_batch_04', 'task_id': 'alpha_m100_c02_batch_04__blind_continual_campaign', 'entrypoint': 'solve_downstream', 'output_schema_keys': ['estimated_taus', 'hereditary_weights', 'initial_rate_pred', 'predicted_strain'], 'solver_path': '/tmp/hegel_continual_archive/alpha_m100_c02_batch_04_solver.py'}]
apparatus entries: 6
last entry summary: [('completed_campaign_index', 5), ('completed_campaign_short_name', 'alpha_m100_c02_batch_04'), ('facility_telemetry_trace', ('list', 24))]
```

### Turn 4 — `execute_in_remote_docker` (`30637 ms` LLM)
- **Rationale**: Compare affine, pairwise, and higher-order terms on held-out calibration rows and at the load boundary. If polynomial error grows at the boundary, test a saturating response before implementing the inverse.

```bash
cd /holdout_task && grep -nEi 'calibrat|bridge|tanh|sine|irls' /tmp/hegel_continual_archive/*solver.py | head -45; cat > /tmp/bridge_probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']; N=len(X)

def basis(x,kind):
 cols=[np.ones(len(x)),*x.T]
 if kind!='affine': cols += [x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]
 if kind=='cubic': cols += [x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations_with_replacement(range(4),3)]
 if kind=='axis3': cols += list((x**3).T)
 if kind=='axis5': cols += list((x**3).T)+list((x**5).T)
 return np.column_stack(cols)

def fit(a,b):
 keep=np.ones(len(a),bool)
 for i in range(10):
  coeff=lstsq(a[keep],b[keep])[0]; r=np.linalg.norm(a@coeff-b,axis=1); k=r<max(0.015,4*np.median(r))
  if np.array_equal(k,keep): break
  keep=k
 return coeff

if __name__=='__main__':
 print('design groups',np.unique(np.count_nonzero(X,axis=1),return_counts=True));print('cal row 8..19\n',np.round(np.c_[X,Y][8:20],5)); print('cal last 8\n',np.round(np.c_[X,Y][-8:],5))
 folds=np.random.default_rng(8).permutation(N)%5
 edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 for kind in ['affine','quadratic','axis3','axis5','cubic']:
  a=basis(X,kind); full=fit(a,Y); out=np.linalg.norm(a@full-Y,axis=1)>2; pred=np.empty_like(Y)
  for f in range(5):
   m=folds!=f;pred[~m]=a[~m]@fit(a[m],Y[m])
  err=np.linalg.norm(pred-Y,axis=1); er=np.linalg.norm(a[edge]@fit(a[~edge],Y[~edge])-Y[edge],axis=1)
  print(kind,'spikes',np.flatnonzero(out),'OOF median/p90/max',np.quantile(err[~out],[.5,.9,1]),'edge clean p90/max',np.quantile(er[~out[edge]],[.9,1]));
  if kind=='axis5':print('coeff row0/linear/axis3/axis5\n',np.round(full[np.r_[0:5,15:23]],5))
 print('nearest rows', np.quantile(np.partition(cdist(X,X),1,axis=1)[:,1],[0,.5,1]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_probe.py
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.210518e+01]
 [-3.793700e-01  4.200300e-01  7.058900e-01 -5.145900e-01 -6.708400e-01
   1.146160e+00  7.280200e-01 -9.250600e-01]
 [-5.282400e-01 -8.680300e-01 -5.169500e-01  2.436100e-01 -6.322100e-01
  -1.314280e+00 -6.026500e-01  9.309700e-01]
 [-1.491740e+00  3.790000e-03 -7.938100e-01  8.840000e-02 -1.031380e+00
   1.249900e-01 -5.820700e-01  3.009500e-01]
 [-9.910400e-01  5.665300e-01 -1.008430e+00 -3.958000e-02 -8.064900e-01
   1.208750e+00 -7.994900e-01 -6.869000e-02]]
cal last 8
 [[ 1.11646  0.45724 -0.3974  -0.75171  0.93588  1.30806 -0.70544 -1.22938]
 [ 0.67303 -0.34459 -0.68816 -1.22818  0.78273 -0.08086 -0.93398 -1.35318]
 [-1.07296  1.28723 -0.64602  0.78562 -0.93358  1.65209 -0.57223  1.20817]
 [ 1.56542  1.1984   0.13802 -0.98751  0.98501  2.12324 -0.099   -1.51949]
 [ 1.31085 -0.04393  0.99934 -0.3048   0.81332  0.14454  0.67877 -0.47428]
 [ 1.36677 -0.15226  0.77905  0.65054  0.84751 -0.29913  0.56172  1.30736]
 [-0.09527 -1.54544  0.60793 -1.4569  -0.31662 -1.10265  0.59296 -1.01339]
 [-1.51925  1.19211 -0.39231  0.2997  -1.14611  1.67372 -0.23911  0.53058]]
affine spikes [13 14 15] OOF median/p90/max [0.32883393 0.45779035 0.66148157] edge clean p90/max [0.55049331 0.70370905]
quadratic spikes [13 14 15] OOF median/p90/max [0.38412462 0.49776708 0.6862707 ] edge clean p90/max [0.65913311 0.69010162]
axis3 spikes [13 14 15] OOF median/p90/max [0.12045041 0.18533057 0.28384105] edge clean p90/max [0.33444257 0.38036261]
axis5 spikes [13 14 15] OOF median/p90/max [0.03331077 0.07478316 0.16627878] edge clean p90/max [0.12676283 0.14639252]
coeff row0/linear/axis3/axis5
 [[-1.13660e-01  1.52040e-01 -1.05640e-01  1.26720e-01]
 [ 1.29177e+00  1.37000e-03 -1.71840e-01 -1.34500e-02]
 [ 3.86000e-03  2.04917e+00  4.12000e-03 -2.70360e-01]
 [-1.68160e-01 -9.60000e-04  1.25970e+00 -1.72200e-02]
 [-2.06000e-03 -2.66000e-01  2.29000e-03  2.03633e+00]
 [-4.37070e-01 -3.06000e-03  1.02900e-02  6.23000e-03]
 [-8.24000e-03 -6.59700e-01 -6.88000e-03  1.47700e-02]
 [ 2.00000e-03  1.51700e-02 -4.12610e-01  2.86300e-02]
 [ 7.28000e-03 -1.12400e-02 -1.46000e-03 -7.04100e-01]
 [ 8.21400e-02  1.22000e-03 -4.16000e-03 -5.00000e-04]
 [ 2.04000e-03  1.20460e-01  2.11000e-03 -6.53000e-03]
 [-2.79000e-03 -6.38000e-03  7.59600e-02 -9.05000e-03]
 [-2.75000e-03  2.47000e-03 -1.90000e-04  1.38040e-01]]
cubic spikes [13 14 15] OOF median/p90/max [0.13626999 0.32217685 1.25178679] edge clean p90/max [0.39187454 0.46096714]
nearest rows [0.37532437 0.69662683 1.33406368]
```

### Turn 5 — `execute_in_remote_docker` (`38589 ms` LLM)
- **Rationale**: Compare saturating and rational families on clean calibration folds and a high-load holdout. The residual pattern will show whether a paired nonlinear term suffices or the nonlinearity must act on a mixed input before inversion.
- **Alternatives Weighed**: `Use the quintic as the final forward map` (rejected: It still has substantial held-out error and its derivative can behave poorly at the calibration boundary.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine mixing matrix, even with quadratic interactions, adequately represents the clean bridge response.` | Witness: `Clean boundary maximum forward error is 0.704 for the affine model and 0.690 with quadratic interactions; adding axis cubic and quintic terms reduces it to 0.146. Single-axis sweeps show strong compression.` | Blanket Diagnosis: `The missing structure is predominantly an odd nonlinear response on each driven channel; the three gross spikes do not explain it.` | Sublated Pivot: `Retain the joint four-channel calibration and explicit spike separation, but compare smooth diagonal and paired coupling laws by held-out error.`

```bash
cat > /tmp/bridge_shape.py <<'PY'
from bridge_probe import X,Y,N,basis
from itertools import product
import numpy as np
from scipy.optimize import minimize

# The three gross bridge spikes are separated from all other measurements by an order of magnitude.
clean=np.max(np.abs(Y-np.median(Y,axis=0)),axis=1)<8
pair=[2,3,0,1]
forms={'tanh':np.tanh,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'root':lambda z:z/np.sqrt(1+z*z),'sin':np.sin,'erf':__import__('scipy').special.erf}

def fit(m,j,diag,cross=None,degree=2,shift=False):
 def des(x,p):
  u=x[:,j]; v=x[:,pair[j]]
  cols=[basis(x,'quadratic' if degree==2 else 'affine'),forms[diag](p[0]*u+(p[-1] if shift else 0.))]
  if cross=='cubic': cols.append(v**3)
  elif cross: cols.append(forms[cross](p[1]*v))
  return np.column_stack(cols)
 def loss(p):
  a=des(X[m],p); b=np.linalg.lstsq(a,Y[m,j],rcond=None)[0];return np.sum((a@b-Y[m,j])**2)
 bounds=[(.15,5)]*(1+int(cross in forms))+([(-1,1)] if shift else [])
 p=min((minimize(loss,[start]*(len(bounds)-int(shift))+([0] if shift else []),bounds=bounds,method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-10}) for start in (.65,1.6)),key=lambda r:r.fun).x
 b=np.linalg.lstsq(des(X[m],p),Y[m,j],rcond=None)[0]
 return des(X,p)@b,p,b

if __name__=='__main__':
 folds=np.random.default_rng(8).permutation(N)%5
 edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 rows=[]
 for diag,cross,degree,shift in [(d,c,2,s) for d,c,s in product(forms,(None,'cubic','tanh','atan','rat'),(False,))]+[(d,c,2,True) for d,c in product(('tanh','atan','rat'),(None,'tanh'))]:
  res=[fit(clean,j,diag,cross,degree,shift) for j in range(4)]
  rm=np.sqrt(np.mean((np.column_stack([r[0] for r in res])[clean]-Y[clean])**2))
  rows.append((rm,diag,cross,degree,shift,res))
 for rm,d,c,g,s,res in sorted(rows,key=lambda row:row[0])[:12]:
  pred=np.zeros_like(Y)
  for f in range(5):
   for j in range(4): pred[folds==f,j]=fit(clean&(folds!=f),j,d,c,g,s)[0][folds==f]
  ep=np.column_stack([fit(clean&~edge,j,d,c,g,s)[0] for j in range(4)])
  print(d,c,s,'rms',round(rm,7),'OOF max',round(np.max(np.linalg.norm((pred-Y)[clean],axis=1)),6),'edge max',round(np.max(np.linalg.norm((ep-Y)[clean&edge],axis=1)),6),'p',np.round([r[1] for r in res],4).tolist())
 rm,d,c,g,s,res=min(rows,key=lambda row:row[0]);print('best coeffs:\n',np.round(np.column_stack([r[2] for r in res]),6)); print('best residual quantiles',np.quantile(abs((np.column_stack([r[0] for r in res])-Y)[clean]),[.5,.9,1],axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_shape.py
```
- **Container Exit Code**: `0` (`8666 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
alse rms 0.0130483 OOF max 0.087202 edge max 0.084071 p [[2.5116], [2.4754], [2.4794], [2.5141]]
sin rat False rms 0.01305 OOF max 0.086597 edge max 0.087245 p [[2.5116, 0.15], [2.4752, 0.6067], [2.4792, 0.15], [2.5139, 0.15]]
sin None False rms 0.0133614 OOF max 0.082314 edge max 0.089126 p [[2.5181], [2.4697], [2.4687], [2.5047]]
rat tanh True rms 0.0139552 OOF max 0.081776 edge max 0.105622 p [[0.9235, 0.15, -0.0146], [0.8992, 0.6767, -0.0137], [0.8643, 4.0568, -0.019], [0.8846, 0.15, -0.025]]
tanh tanh True rms 0.0141651 OOF max 0.083665 edge max 0.10896 p [[1.497, 0.15, -0.0232], [1.4557, 0.6743, -0.0221], [1.3921, 4.0079, -0.0304], [1.4289, 0.15, -0.0408]]
erf tanh False rms 0.0141781 OOF max 0.087922 edge max 0.101543 p [[1.4249, 0.15], [1.4018, 0.9367], [1.3615, 0.15], [1.3782, 0.15]]
erf cubic False rms 0.0141791 OOF max 0.088381 edge max 0.100639 p [[1.4249], [1.4019], [1.3616], [1.3783]]
erf atan False rms 0.0141795 OOF max 0.087766 edge max 0.101448 p [[1.4249, 0.15], [1.4018, 0.7716], [1.3614, 0.15], [1.3782, 0.15]]
erf rat False rms 0.0141812 OOF max 0.088381 edge max 0.099701 p [[1.425, 0.15], [1.4018, 0.5953], [1.3613, 0.15], [1.3781, 0.15]]
rat None True rms 0.0142371 OOF max 0.071384 edge max 0.109209 p [[0.9292, -0.0141], [0.8964, -0.013], [0.865, -0.0183], [0.8732, -0.0234]]
best coeffs:
 [[-1.096320e-01  1.504140e-01 -1.033760e-01  1.273110e-01]
 [ 8.110460e-01  1.144000e-03 -6.511660e-01 -3.221000e-03]
 [-2.305000e-03  1.297761e+00  6.600000e-05 -1.138708e+00]
 [-8.544690e-01  4.175000e-03  7.948420e-01 -7.100000e-04]
 [ 7.590000e-04 -2.975980e-01  1.004000e-03  1.280506e+00]
 [ 3.077000e-03  2.852000e-03  4.452800e-02  9.550000e-04]
 [-3.270000e-04  3.330000e-03  2.455000e-03  7.807000e-03]
 [ 6.210000e-04 -1.074000e-03 -5.720000e-04  8.730000e-04]
 [ 1.288000e-03  1.140000e-03 -1.216000e-03 -2.551000e-03]
 [-1.167000e-03  5.630000e-04  6.510000e-04  6.792300e-02]
 [-1.053000e-03  6.602000e-03  2.240000e-04 -1.038700e-02]
 [ 2.336000e-03  1.478000e-03  1.372000e-03 -1.285000e-03]
 [ 4.415000e-02  3.328000e-03  3.068000e-03  5.285000e-03]
 [ 9.200000e-05 -2.506000e-03 -7.360000e-04  7.061000e-03]
 [ 4.870000e-04  6.958800e-02  4.330000e-04  6.459000e-03]
 [ 2.049350e-01  3.230610e-01  2.004440e-01  3.136710e-01]
 [ 4.592669e+00  3.335000e-02  3.252349e+00  5.867069e+00]]
best residual quantiles [[0.00698555 0.01057418 0.00838588 0.01048051]
 [0.01181487 0.02268893 0.01522328 0.02541352]
 [0.03961878 0.04164785 0.03174293 0.04244329]]
```

### Turn 6 — `execute_in_remote_docker` (`41407 ms` LLM)
- **Rationale**: Test whether the unexplained error comes from the diagonal response, paired coupling, or both. The sweep and high-load holdouts determine whether to pool sweep data into the forward fit and which model to invert.
- **Alternatives Weighed**: `Apply the best in-sample sinusoidal fit immediately` (rejected: Its held-out residual remains far above the apparent sweep noise; comparing diagonal sweeps and mixed-load rows can distinguish the missing terms.)

```bash
cat > /tmp/bridge_structure.py <<'PY'
from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
S=D['calibration_load_cells']['single_channel_sweeps']; pair=[2,3,0,1]
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8
for j in range(4):
 m=S['lab_id']==j; print('sweep',j,np.round(np.c_[S['true_stress'][m],S['cmd_stress'][m]],6).tolist()); p=pair[j]; m=(np.count_nonzero(X,axis=1)==1)&(X[:,p]!=0)&clean;print('cross axis',j,np.round(np.c_[X[m,p],Y[m,j]],6).tolist())
forms={'sin':np.sin,'tanh':np.tanh,'rat':lambda q:q/(1+q*q),'atan':np.arctan,'zero':lambda q:q*0}

def fit(j,m,f,g,mode,sw=False):
 p=pair[j]; x=X[m]; a=x[:,j]; b=x[:,p]; t=Y[m,j]
 if sw:
  sm=(S['lab_id']==j)&(abs(S['cmd_stress']-np.median(S['cmd_stress']))<8); ss=S['true_stress'][sm]; a=np.r_[a,ss]; b=np.r_[b,np.zeros_like(ss)]; t=np.r_[t,S['cmd_stress'][sm]]
 def design(u,v,z):
  k,l,h=z; q=u if mode not in ('mixed','both') else u+h*v
  cols=[np.ones_like(u),u,v,v*v,forms[f](k*q),forms[g](l*v)]
  if mode in ('gate','both'): cols.extend([u*v, v*np.cos(k*q),v*v*np.cos(k*q)])
  if mode=='diag2': cols.extend([u*u,u**3,np.sin(l*u)])
  return np.array(cols).T
 def res(z):
  A=design(a,b,z); return A@np.linalg.lstsq(A,t,rcond=1e-10)[0]-t
 p0={'sin':2.5,'tanh':1.5,'rat':.9,'atan':2.,'zero':1.}; bounds=([.2,.2,-.8],[5,6,.8]); sol=min((least_squares(res,[p0[f],l,h],bounds=bounds,max_nfev=100,ftol=1e-10) for l,h in [(p0[g],0.),(2.8,-.15)]),key=lambda r:r.cost); z=sol.x; C=np.linalg.lstsq(design(a,b,z),t,rcond=1e-10)[0]
 return design(X[:,j],X[:,pair[j]],z)@C,z,C

specs=[(f,g,mode) for f in ['sin','tanh','rat'] for g in ['zero','sin','tanh','rat'] for mode in ['add','mixed','gate','both']]+[(f,'zero','diag2') for f in ['sin','tanh','rat']]
folds=np.random.default_rng(8).permutation(N)%5; edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
for sw in [False,True]:
 rows=[]
 for f,g,mode in specs:
  r=[fit(j,clean,f,g,mode,sw) for j in range(4)]; e=np.column_stack([v[0] for v in r])-Y; rows.append((np.sqrt(np.mean(e[clean]**2)),f,g,mode,r))
 for rms,f,g,mode,r in sorted(rows,key=lambda x:x[0])[:6]:
  pred=np.zeros_like(Y)
  for fold in range(5):
   h=folds==fold
   for j in range(4): pred[h,j]=fit(j,clean&~h,f,g,mode,sw)[0][h]
  ep=np.column_stack([fit(j,clean&~edge,f,g,mode,sw)[0] for j in range(4)])
  print(sw,f,g,mode,'rms/oof p90/max/edge max',rms,*np.quantile(np.linalg.norm((pred-Y)[clean],axis=1),[.9,1]),max(np.linalg.norm((ep-Y)[edge&clean],axis=1)), 'z',np.round([v[1] for v in r],4).tolist())
 best=min(rows,key=lambda v:v[0]); print('best coeff',np.round(np.array([v[2] for v in best[-1]]),6).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_structure.py
```
- **Container Exit Code**: `0` (`12474 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
]
False sin tanh gate rms/oof p90/max/edge max 0.013226285981045494 0.047088541281092855 0.15409860987311175 0.11846080327318091 z [[2.5225, 0.2, -0.15], [2.477, 0.2, 0.0], [2.4685, 0.2, 0.0], [2.5485, 0.2, 0.0]]
False sin rat gate rms/oof p90/max/edge max 0.013234570632008454 0.04679652030746194 0.15357277232237865 0.11839598666549887 z [[2.5226, 0.2, -0.15], [2.4769, 0.2, -0.15], [2.4684, 0.2, 0.0], [2.5485, 0.2, 0.0]]
best coeff [[-0.110002, 0.812337, -1.267703, 0.047199, 0.203405, 5.532689, 0.000238, 0.00965, -0.008148], [0.154218, 1.297367, -0.289729, 0.073752, 0.323236, 0.016362, -0.001855, 0.001044, -0.002753], [-0.100597, 0.794082, -0.743481, 0.04505, 0.198235, 2.899474, -0.000756, 0.030161, -0.007132], [0.134429, 1.287299, -1.406457, 0.072213, 0.30726, 5.746615, -0.004803, -0.03233, -0.009224]]
True sin sin both rms/oof p90/max/edge max 0.0133514946547176 0.05328272886499696 0.1785968992824345 0.14869210545595615 z [[2.518, 0.2, -0.026], [2.484, 1.3527, 0.0053], [2.475, 0.2, -0.0477], [2.5367, 0.2, 0.0352]]
True sin tanh both rms/oof p90/max/edge max 0.0133586783842057 0.05339565164525366 0.17818792872418268 0.15055151287688293 z [[2.5181, 0.2, -0.0262], [2.4838, 0.7416, 0.0051], [2.4749, 0.2, -0.0476], [2.5367, 0.2, 0.0353]]
True sin rat both rms/oof p90/max/edge max 0.01337494087543317 0.05345664962768113 0.17743558583408423 0.15049436148793988 z [[2.5181, 0.2, -0.0265], [2.4832, 0.7738, 0.006], [2.4749, 0.2, -0.0476], [2.5368, 0.2, 0.0354]]
True sin sin gate rms/oof p90/max/edge max 0.013402256686582298 0.04887588509657482 0.16634512463549614 0.1296234074605835 z [[2.5182, 0.2, -0.15], [2.4842, 0.2, 0.0], [2.473, 0.2, -0.15], [2.5356, 0.2, 0.0]]
True sin tanh gate rms/oof p90/max/edge max 0.013407427900175115 0.048686626243238815 0.16607608578511743 0.1315811072877151 z [[2.5182, 0.2, -0.15], [2.4841, 0.2, -0.15], [2.473, 0.2, 0.0], [2.5356, 0.2, -0.15]]
True sin rat gate rms/oof p90/max/edge max 0.01341585242751805 0.04836641310808821 0.1655759385347511 0.1315240433156547 z [[2.5183, 0.2, 0.0], [2.4841, 0.2, -0.15], [2.4729, 0.2, -0.15], [2.5357, 0.2, 0.0]]
best coeff [[-0.112521, 0.811772, -1.311979, 0.049099, 0.202608, 5.754081, 0.000393, 0.018842, -0.008706], [0.150985, 1.298639, -0.291457, 0.076148, 0.323405, 0.018507, -0.002053, 0.000911, -0.003059], [-0.102978, 0.795336, -0.716907, 0.046575, 0.198185, 2.764445, -0.001117, 0.025783, -0.007346], [0.131006, 1.28198, -1.423581, 0.074151, 0.310656, 5.836281, -0.004005, -0.027169, -0.009187]]
```

### Turn 7 — `execute_in_remote_docker` (`40764 ms` LLM)
- **Rationale**: The independent sweeps constrain the diagonal transfer curve. Subtract it from mixed-load observations to test whether the remaining response is additive in the paired load, then use boundary validation to choose the forward model.
- **Alternatives Weighed**: `Fit a fully mixed nonlinear response` (rejected: The mixed-input variants tested so far worsened the boundary holdout; first isolate the diagonal response using the measured sweeps.)

```bash
cat > /tmp/bridge_additive.py <<'PY'
from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
P=np.array([2,3,0,1]); S=D['calibration_load_cells']['single_channel_sweeps']; SX=np.eye(4)[S['lab_id']]*S['true_stress'][:,None]; SY=S['cmd_stress']; clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8; sc=abs(SY-np.median(SY))<8
forms=[np.sin,np.tanh,np.arctan,lambda q:q/(1+q*q),lambda q:q/np.sqrt(1+q*q)]
def fit(j,m,typ,deg,interaction=False,sm=sc):
 p=P[j]; x=np.r_[X[m],SX[sm&(S['lab_id']==j)]]; y=np.r_[Y[m,j],SY[sm&(S['lab_id']==j)]]
 def design(x,z):
  u=x[:,j]; v=x[:,p]; cols=[u*0+1,u,v,v*v];
  if typ=='poly': cols.extend(u**i for i in range(3,deg+1,2))
  else:
   cols.extend(forms[k](z[a]*u) for a,k in enumerate(typ));
   if deg: cols.append(u**3)
  cols.extend((v**3,v**4,v**5))
  if interaction: cols.extend((u*v,u*u*v,u*v*v))
  return np.array(cols).T
 def obj(z):
  a=design(x,z); return a@np.linalg.lstsq(a,y,rcond=1e-9)[0]-y
 if typ=='poly': z=[]
 else:
  z=min((least_squares(obj,q,bounds=(.25,10.),max_nfev=160,gtol=1e-11) for q in ([[2.5],[1.1]] if len(typ)==1 else [[2.5,1.3],[2.5,4.6],[1.1,2.5]])),key=lambda r:r.cost).x
 b=np.linalg.lstsq(design(x,z),y,rcond=1e-9)[0]; return design(X,z)@b, design(SX,z)@b,z,b
spec=[('poly',i,g) for i in [5,7,9,11] for g in [False,True]]+[(t,d,g) for t in [(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(1,1),(1,3),(3,3)] for d,g in [(0,False),(1,False),(0,True)]]
edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8); folds=np.random.default_rng(8).permutation(N)%5
rows=[]
for s in spec:
 r=[fit(j,clean,*s) for j in range(4)]; ef=np.linalg.norm((np.array([v[0] for v in r]).T-Y),axis=1); rows.append((np.sqrt(np.mean(ef[clean]**2)),s,r))
for rms,s,r in sorted(rows,key=lambda r:r[0])[:12]:
 pred=np.empty_like(Y)
 for f in range(5):
  h=folds==f
  for j in range(4): pred[h,j]=fit(j,clean&~h,*s)[0][h]
 e=np.linalg.norm(pred-Y,axis=1); ep=np.array([fit(j,clean&~edge,*s)[0] for j in range(4)]).T
 print(s, 'rms/p90/max/edge',*[round(v,7) for v in (rms,np.quantile(e[clean],.9),max(e[clean]),max(np.linalg.norm((ep-Y)[edge&clean],axis=1)))], 'z', [np.round(v[2],4).tolist() for v in r]);
 if rms<.002: print('coeffs',np.round([v[3] for v in r],5))
# Test whether subtracting the independently interpolated diagonal leaves a univariate paired curve.
from scipy.interpolate import CubicSpline
for j in range(4):
 sm=sc&(S['lab_id']==j); f=CubicSpline(S['true_stress'][sm],SY[sm]); v=X[clean,P[j]]; u=X[clean,j]; e=Y[clean,j]-f(u); A=np.c_[v,v*v,v**3,v**4,v**5]; b=np.linalg.lstsq(A,e,rcond=None)[0]; print('paired decomposition',j,np.round(b,5),np.quantile(abs(e-A@b),[.5,.9,1]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_additive.py
```
- **Container Exit Code**: `0` (`6416 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
((0, 0), 0, True) rms/p90/max/edge 0.0028439 0.005044 0.0206705 0.0205999 z [[2.4971, 7.4905], [2.4998, 7.5105], [0.25, 2.5229], [2.5018, 6.8032]]
((0, 3), 0, True) rms/p90/max/edge 0.0028463 0.0051141 0.020616 0.0201132 z [[2.5012, 4.1137], [2.503, 6.3474], [2.5258, 0.25], [2.5057, 8.698]]
((1, 3), 0, True) rms/p90/max/edge 0.0028497 0.0049789 0.0204675 0.0205469 z [[1.0649, 0.6694], [1.0827, 0.6802], [1.0282, 0.6474], [1.0422, 0.6557]]
((0, 1), 0, True) rms/p90/max/edge 0.0028503 0.0051019 0.0206196 0.0201562 z [[2.501, 6.4322], [2.503, 10.0], [2.5241, 0.25], [2.5063, 10.0]]
('poly', 11, True) rms/p90/max/edge 0.0028519 0.0049569 0.0206626 0.0219601 z [[], [], [], []]
((0, 2), 0, True) rms/p90/max/edge 0.0028541 0.0050991 0.0206438 0.0202238 z [[2.5009, 7.5594], [2.5031, 10.0], [2.5246, 0.25], [2.5064, 10.0]]
('poly', 9, True) rms/p90/max/edge 0.0028612 0.0051399 0.0207964 0.0206134 z [[], [], [], []]
((0,), 0, True) rms/p90/max/edge 0.0028915 0.0052032 0.0210326 0.0210023 z [[2.499], [2.5008], [2.5003], [2.5034]]
((0, 3), 1, False) rms/p90/max/edge 0.0029135 0.0049563 0.0204397 0.0192418 z [[2.5436, 2.4143], [2.5461, 3.7608], [2.1797, 0.25], [2.5073, 8.0824]]
((0, 1), 1, False) rms/p90/max/edge 0.0029152 0.0049538 0.0204394 0.0192354 z [[2.5442, 3.7857], [2.5478, 5.9282], [2.1777, 0.25], [2.517, 10.0]]
('poly', 7, True) rms/p90/max/edge 0.0029174 0.0051372 0.0198884 0.0216372 z [[], [], [], []]
((0, 2), 1, False) rms/p90/max/edge 0.0029175 0.0049584 0.0204376 0.0192367 z [[2.5472, 3.3931], [2.5505, 6.1341], [2.1721, 0.25], [2.5218, 10.0]]
paired decomposition 0 [-0.16922  0.08945 -0.00264 -0.02058  0.00106] [0.00095592 0.001837   0.00477915]
paired decomposition 1 [-2.7201e-01  1.4315e-01 -9.6000e-04 -3.2910e-02  2.5000e-04] [0.00170523 0.00282541 0.00508106]
paired decomposition 2 [-0.16774  0.08742 -0.00104 -0.0198   0.0005 ] [0.00095408 0.00192381 0.00485842]
paired decomposition 3 [-0.26417  0.13828 -0.00462 -0.03121  0.00167] [0.00141927 0.00339631 0.00558018]
```

### Turn 8 — `execute_in_remote_docker` (`42102 ms` LLM)
- **Rationale**: Test compact even coupling laws under robust fitting. If the worst residuals persist on the same rows, inspect those measurements rather than adding terms to explain them.
- **Alternatives Weighed**: `Invert the paired quintic fit` (rejected: The paired coefficients suggest an even response plus a linear term; a compact fit would retain that structure without quintic extrapolation.)

```bash
cat > /tmp/bridge_compact.py <<'PY'
from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
S=D['calibration_load_cells']['single_channel_sweeps']; lab=S['lab_id']; sx=S['true_stress']; sy=S['cmd_stress']; P=[2,3,0,1]
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8; sc=abs(sy-np.median(sy))<8
forms={'cos':lambda v,k:1-np.cos(k*v),'rat':lambda v,k:v*v/(1+k*v*v),'exp':lambda v,k:-np.expm1(-k*v*v),'poly':lambda v,k:v**4}
def design(u,v,z,g,h):
 a=[np.ones(len(u)),u,np.sin(z[0]*u),v,forms[g](v,z[1])]
 if g=='poly': a.append(v*v)
 if h=='cubic': a.append(u**3)
 if h=='sin3': a.append(np.sin(3*z[0]*u))
 if h=='inter': a.extend([u*v,u*u*v,u*v*v])
 return np.array(a).T
def model(m,g,h,robust=False,sm=sc):
 pred=[]; co=[]; swept=[]
 for j,p in enumerate(P):
  s=sm&(lab==j); u=np.r_[X[m,j],sx[s]]; v=np.r_[X[m,p],np.zeros(s.sum())]; y=np.r_[Y[m,j],sy[s]]; w=np.ones(len(y))
  def coeff(z):
   a=design(u,v,z,g,h); b=np.linalg.lstsq(a*w[:,None],y*w,rcond=None)[0];return a@b-y,b
  start=[2.5,1.6 if g=='cos' else .3]
  for _ in range(6 if robust else 1):
   z=least_squares(lambda z:coeff(z)[0]*w,start,bounds=([1,.05],[4,3.5]),max_nfev=75).x; e,b=coeff(z); start=z; w=(1+(e/max(.00025,2.5*np.median(abs(e))))**2)**(-.5)
  pred.append(design(X[:,j],X[:,p],z,g,h)@b); swept.extend(zip(np.where(s)[0],e[m.sum():])); co.append((z,b))
 return np.array(pred).T,co,swept
if __name__=='__main__':
 folds=np.random.default_rng(8).permutation(N)%5; edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 for g in forms:
  for h in ['none','cubic','sin3','inter']:
   pred,co,se=model(clean,g,h,True); e=np.linalg.norm(pred-Y,axis=1); err=np.zeros(N)
   for f in range(5):
    q=folds==f; pp,_,_=model(clean&~q,g,h,True); err[q]=np.linalg.norm((pp-Y)[q],axis=1)
   pp,_,_=model(clean&~edge,g,h,True); ee=np.linalg.norm(pp-Y,axis=1);print(g,h,'fit/OOF/edge p90,max',*[np.round(np.quantile(q,[.9,1]),7).tolist() for q in [e[clean],err[clean],ee[clean&edge]]]);
   if h=='none':
    print('params',[(np.round(z,5).tolist(),np.round(b,6).tolist()) for z,b in co]); idx=np.where(clean)[0][np.argsort(e[clean])[-4:]]; print('worst rows',idx.tolist(),np.round((pred-Y)[idx],6).tolist(),'worst sweep',sorted(se,key=lambda q:abs(q[1]))[-2:]);
 for split in ['train','test']:
  for k in ['past_channel_raw','hold_channel_raw']:
   a=np.asarray(D[split][k]).reshape(-1,4); print(split,k,len(a),a.min(0),a.max(0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_compact.py
```
- **Container Exit Code**: `0` (`4353 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5727, 0.0231399] [0.0231333, 0.0260399]
exp none fit/OOF/edge p90,max [0.0079332, 0.0172894] [0.0081937, 0.0163513] [0.0145119, 0.0197766]
params [([2.50168, 0.90004], [-0.120203, 0.810235, 0.202352, -0.170364, 0.114699]), ([2.49842, 0.89007], [0.139802, 1.299677, 0.325167, -0.273668, 0.185468]), ([2.49924, 0.92557], [-0.110241, 0.799872, 0.20006, -0.167996, 0.112241]), ([2.49867, 0.90559], [0.119754, 1.269775, 0.317559, -0.266473, 0.180193])]
worst rows [3, 31, 12, 7] [[0.000109, -0.000273, 0.009793, -0.000386], [-0.0001, 0.002767, 0.001056, 0.009953], [-9.4e-05, 0.011448, -0.000184, -0.000419], [-0.000332, -0.000353, -0.000252, 0.017281]] worst sweep [(13, -0.00046233599678635606), (36, -0.00047041518711443686)]
exp cubic fit/OOF/edge p90,max [0.0079076, 0.0170086] [0.0080469, 0.0159868] [0.0145393, 0.0197736]
exp sin3 fit/OOF/edge p90,max [0.0077073, 0.0171121] [0.0079201, 0.0160187] [0.0144488, 0.0197629]
exp inter fit/OOF/edge p90,max [0.0065069, 0.0146502] [0.0077604, 0.0150821] [0.0145715, 0.0177425]
poly none fit/OOF/edge p90,max [0.0048595, 0.0141915] [0.0063893, 0.0177213] [0.0190804, 0.0293609]
params [([2.49866, 0.3], [-0.119659, 0.809732, 0.202637, -0.170125, -0.021691, 0.090927]), ([2.50039, 0.3], [0.140529, 1.300243, 0.325287, -0.272621, -0.033718, 0.143956]), ([2.50052, 0.3], [-0.109678, 0.800092, 0.199764, -0.168036, -0.02049, 0.088067]), ([2.50279, 0.3], [0.120597, 1.270731, 0.317208, -0.267, -0.031694, 0.138531])]
worst rows [12, 3, 11, 7] [[0.000451, -0.008415, 0.000379, -4e-06], [5e-06, 0.000455, -0.010837, 0.000457], [-0.013874, 0.000575, 0.0006, 0.000515], [0.000212, 0.000741, 0.000311, -0.014167]] worst sweep [(47, 0.0008905500828508828), (19, 0.0010856696897290075)]
poly cubic fit/OOF/edge p90,max [0.0049339, 0.0142946] [0.0063262, 0.0177129] [0.0191621, 0.0293171]
poly sin3 fit/OOF/edge p90,max [0.0049903, 0.0140174] [0.006187, 0.0173405] [0.0183852, 0.0286678]
poly inter fit/OOF/edge p90,max [0.0051441, 0.0129858] [0.006251, 0.0158207] [0.0192322, 0.0261082]
train past_channel_raw 432 [-0.9583383  -1.14958309 -0.91780035 -1.13111537] [0.96432269 1.90487184 0.9970118  1.85300546]
train hold_channel_raw 72 [0.275274   0.67167889 0.230445   0.64075494] [0.86110953 1.70662706 0.83760287 1.63893514]
test past_channel_raw 216 [-0.92737416 -1.15861289 -0.92346821 -1.15642322] [1.00196968 1.86971582 0.97094618 1.83796967]
test hold_channel_raw 36 [0.21913425 0.63727746 0.21857352 0.73091223] [0.86011117 1.64686017 0.85499985 1.66545378]
```

### Turn 9 — `execute_in_remote_docker` (`51601 ms` LLM)
- **Rationale**: Implement the compact fit with joint bridge-row downweighting and a damped paired inverse. The public validator checks that the unchanged downstream wrapper can consume the calibrated dataset; separate load-space and worst-slice tests are still needed.
- **Alternatives Weighed**: `Add cubic, harmonic, or interaction terms` (rejected: Their changes to boundary error are small relative to measurement noise, while they make extrapolation and inversion less predictable.)

```bash
cat > /holdout_task/upstream_transducer.py <<'PY'
"""Calibrate the paired four-channel load-cell bridge."""
from __future__ import annotations

import numpy as np

_PAIR = np.array([2, 3, 0, 1])
# Shared frequencies identified from the bridge rows and single-axis sweeps.
_OWN_FREQ = 2.5
_PAIR_FREQ = 2.0
_HISTORY_FIELDS = ('past_channel_raw', 'hold_channel_raw')


def _design(u, v):
    return np.stack((np.ones_like(u), u, np.sin(_OWN_FREQ * u),
                     v, 1 - np.cos(_PAIR_FREQ * v)), axis=-1)


def _fit(calibration):
    x = np.asarray(calibration['true_channel_loads'], dtype=float)
    y = np.asarray(calibration['bridge_raw'], dtype=float)
    sweep = calibration['single_channel_sweeps']
    lab = np.asarray(sweep['lab_id'])
    sx = np.asarray(sweep['true_stress'], dtype=float)
    sy = np.asarray(sweep['cmd_stress'], dtype=float)
    bridge_design = _design(x, x[:, _PAIR])
    sweep_design = _design(sx, np.zeros_like(sx))
    row_weight, sweep_weight = np.ones(len(x)), np.ones(len(sx))
    coef = np.empty((4, 5))
    for _ in range(8):
        for j in range(4):
            s = lab == j
            a = np.concatenate((bridge_design[:, j], sweep_design[s]))
            b = np.concatenate((y[:, j], sy[s]))
            w = np.sqrt(np.concatenate((row_weight, sweep_weight[s])))
            coef[j] = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)[0]
        # A gross spike compromises the calibration row, not four independent fits.
        row_residual = np.linalg.norm(np.einsum('njk,jk->nj', bridge_design, coef) - y, axis=1)
        sweep_residual = np.abs(np.einsum('nk,nk->n', sweep_design, coef[lab]) - sy)
        row_weight = _robust_weight(row_residual, row_residual)
        sweep_weight = _robust_weight(sweep_residual, np.r_[sweep_residual, row_residual / 2])
    return coef


def _robust_weight(residual, reference):
    scale = max(1e-8, 4 * np.median(reference))
    return 1 / (1 + (residual / scale) ** 2)


def _forward(x, coef):
    b, a, s, c, q = coef.T
    v = x[:, _PAIR]
    return b + a*x + s*np.sin(_OWN_FREQ*x) + c*v + q*(1 - np.cos(_PAIR_FREQ*v))


def _inverse(raw, coef):
    shape = np.shape(raw)
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    b, a, s, c, q = coef.T
    x = (y - b) / a
    for _ in range(32):
        residual = _forward(x, coef) - y
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-11
        if not np.any(active):
            return x.reshape(shape)
        u, target, r = x[active], y[active], residual[active]
        diag = a + (_OWN_FREQ*s)*np.cos(_OWN_FREQ*u)
        off = c + (_PAIR_FREQ*q)*np.sin(_PAIR_FREQ*u[:, _PAIR])
        det = diag*diag[:, _PAIR] - off*off[:, _PAIR]
        step = (diag[:, _PAIR]*r - off*r[:, _PAIR]) / det
        candidate = u - step
        norm = np.sum(r*r, axis=1)
        for _ in range(24):
            trial = _forward(candidate, coef) - target
            bad = np.sum(trial*trial, axis=1) >= norm
            if not np.any(bad):
                break
            step[bad] *= 0.5
            candidate[bad] = u[bad] - step[bad]
        x[active] = candidate
    raise RuntimeError('Load-cell bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return calibrated histories and identity calibration without mutating the input."""
    calibration = dataset['calibration_load_cells']
    coef = _fit(calibration)
    result = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(key)
        if split is not None:
            result[key] = {**split, **{field: _inverse(split[field], coef)
                                       for field in _HISTORY_FIELDS if field in split}}
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = {
        **calibration,
        'bridge_raw': calibration['true_channel_loads'].copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
    }
    return result
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`1011 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 0.0, "ref_swap_rate_max_diff": 0.0, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b058bfb3998008c4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b058bfb3 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`37717 ms` LLM)
- **Rationale**: Test the exact candidate on held-out measured loads, check its invertibility margin and joint-load round trips, and verify each history field and optional trial mode in an isolated process. A measured error failure would reopen calibration; a closure or field failure would isolate the inverse or schema handling.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
from batch_solver import solve_batch
with open('public_batch.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']; s = c['single_channel_sweeps']
# Score measured calibration rows, with gross spikes scored separately from inversion accuracy.
spike = np.max(np.abs(y-np.median(y, axis=0)), axis=1)>8
fold = np.random.default_rng(8).permutation(len(x))%5
edge = np.max(abs(x),axis=1)>=np.quantile(np.max(abs(x),axis=1),.8)
for name, masks in [('5-fold', [fold==k for k in range(5)]), ('high-load', [edge])]:
    errors = []; forward_errors = []; spike_residuals = []
    for h in masks:
        p = u._fit({**c, 'true_channel_loads':x[~h], 'bridge_raw':y[~h]})
        clean = h & ~spike
        residual = np.linalg.norm(u._forward(x[h],p)-y[h],axis=1)
        forward_errors.extend(residual[~spike[h]])
        spike_residuals.extend(residual[spike[h]])
        errors.extend(np.max(abs(u._inverse(y[clean],p)-x[clean]),axis=1))
    print(name, 'max forward L2', max(forward_errors), 'max load error',max(errors), 'minimum gross-spike residual',min(spike_residuals,default=None))
    assert max(forward_errors)<0.001 and max(errors)<0.002 and min(spike_residuals,default=np.inf)>5
p = u._fit(c)
a, b, e, q = p[:,1:].T
margin = a - u._OWN_FREQ*abs(b) - abs(e) - u._PAIR_FREQ*abs(q)
print('global Jacobian diagonal-dominance margins',margin)
assert np.min(margin)>0
rng = np.random.default_rng(581)
grid = np.stack(np.meshgrid(*([np.linspace(-1.7,1.7,9)]*4),indexing='ij'),axis=-1).reshape(-1,4)
loads = np.r_[grid, rng.uniform(-3.6,3.6,(2000,4))]
err = np.max(abs(u._inverse(u._forward(loads,p),p)-loads))
print('joint load grid and extrapolation max round-trip error',err)
assert err<1e-8
for extra in ('absent','none','present'):
    inp = dict(d)
    if extra=='none': inp['chamber_verification_trials']=None
    if extra=='present': inp['chamber_verification_trials']=copy.deepcopy(d['test'])
    before = pickle.dumps(inp)
    out = u.calibrate_upstream(inp)
    assert before==pickle.dumps(inp)
    if extra=='absent': assert 'chamber_verification_trials' not in out
    if extra=='none': assert out['chamber_verification_trials'] is None
    for key in ('train','test','chamber_verification_trials'):
        if inp.get(key) is None: continue
        for field in ('past_channel_raw','hold_channel_raw'):
            old = inp[key][field]; new = out[key][field]
            assert old.shape==new.shape and np.isfinite(new).all()
            reencoded = u._forward(new.reshape(-1,4),p).reshape(old.shape)
            worst = np.max(abs(reencoded-old))
            assert worst<1e-9
            if extra=='present': print(key,field,'forward closure',worst)
    nc = out['calibration_load_cells']; ns = nc['single_channel_sweeps']
    assert np.array_equal(nc['bridge_raw'],x) and not np.shares_memory(nc['bridge_raw'],x)
    assert np.array_equal(ns['cmd_stress'],s['true_stress']) and not np.shares_memory(ns['cmd_stress'],s['true_stress'])
res = solve_batch(d)
assert all(np.isfinite(v).all() for v in res.values())
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`1482 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5-fold max forward L2 0.0004007907482210337 max load error 0.000987170834157114 minimum gross-spike residual 27.333897642695142
high-load max forward L2 0.0003984049526212669 max load error 0.000537078510137956 minimum gross-spike residual 27.33396477045418
global Jacobian diagonal-dominance margins [0.03644168 0.05850282 0.03600555 0.05717622]
joint load grid and extrapolation max round-trip error 4.578404322330698e-11
train past_channel_raw forward closure 8.939293749676835e-12
train hold_channel_raw forward closure 9.750755758375362e-12
test past_channel_raw forward closure 8.280265362259343e-12
test hold_channel_raw forward closure 8.708589405159728e-13
chamber_verification_trials past_channel_raw forward closure 8.280265362259343e-12
chamber_verification_trials hold_channel_raw forward closure 8.708589405159728e-13
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 0.0, "ref_swap_rate_max_diff": 0.0, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b058bfb3998008c4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b058bfb3 (directory).
```

### Turn 11 — `submit_completed_task` (`15788 ms` LLM)
- **Rationale**: The exact submitted bundle passed isolated-process validation for measured holdouts, numerical inversion, identity normalization, and all three optional trial modes. The compact nonlinear model reduced the high-load forward error from 0.704 for the affine baseline to 0.0003984 raw units.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel calibration in `/holdout_task/upstream_transducer.py` and verified it through the existing `batch_solver.py`. The public schema and reference-swap checks passed. Maximum absolute load error was 0.00099 across five held-out folds and 0.00054 on the high-load holdout.`

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
