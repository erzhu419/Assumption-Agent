# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m100_02_batch_05__blind_continual_campaign__upstream_transducer`

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
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `356,122` (`341,406` / `102,784` / `14,716`) |
| **Total Episode Wall Time (s)** | `489.9s` |

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

### Turn 1 — `execute_in_remote_docker` (`18371 ms` LLM)
- **Rationale**: Inspect the pipeline contract and calibration coverage before choosing a model; this will detect shape or coordinate mismatches and identify which validation splits are possible.

```bash
cd /holdout_task && cat batch_solver.py validate_public_schema.py _verified_downstream.py public_asset_index.json && python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
def view(x,k='',d=0):
 if isinstance(x,dict) and d<3:
  for a,b in x.items(): view(b,k+'/'+str(a),d+1)
 elif isinstance(x,(list,tuple)):
  print(k,type(x).__name__,len(x));
  if len(x) and isinstance(x[0],dict) and d<3: view(x[0],k+'/0',d+1)
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype, (float(np.nanmin(x)),float(np.nanmax(x))) if x.size and x.dtype.kind in 'fiu' else '')
 else: print(k,type(x).__name__,str(x)[:120])
view(p)
for f in ('pre_llm_gate_diagnostic.json','archive_manifest.json','archive_apparatus_log.json'):
 try:
  x=json.load(open('/tmp/hegel_continual_archive/'+f)); print(f, str(x)[:2600])
 except FileNotFoundError: pass
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS coupled nonlinear calibration' 2>&1 | head -45
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, -0.29681, 0.81742, 0.55584, -0.25546], [-0.38921, -0.77471, -0.81742, 0.59164, -0.37534, 0.73162, 0.36875, -0.73163], [-0.44044, -0.84489, -0.73917, 0.71909, -0.29125, 0.27737, 0.1285, -0.79095], [-0.46519, -0.7087, -0.23181, 0.83501, -0.30812, -0.41391, -0.04601, -0.78171], [-0.35889, -0.37734, 0.40354, 0.86849, -0.37711, -0.78913, -0.25604, -0.60752], [-0.49628, 0.03296, 0.80193, 0.84213, -0.25193, -0.79742, -0.45591, -0.13622], [-0.45877, 0.4656, 0.73836, 0.74647, -0.40647, -0.14777, -0.65302, 0.26499], [-0.31721, 0.80399, 0.17632, 0.5826, -0.30172, 0.46696, -0.74322, 0.64963], [-0.39859, 0.78518, -0.41754, 0.42144, -0.41803, 0.8065, -0.90931, 0.82152], [-0.3426, 0.70018, -0.79803, 0.20574, -0.19862, 0.74085, -0.83363, 0.81238], [-0.37025, 0.30023, -0.76623, -0.07311, -0.42926, 0.24157, -0.82757, 0.53444]]}, {'completed_campaign_index': 1, 'completed_campaign_short_name': 'dev_beta_m100_02_s00', 'facility_telemetry_trace': [[-0.37312, 0.45014, 0.68274, 0.34434, 2.37925, -0.82381, 0.84434, 0.29405], [-0.38532, 0.06778, 0.165, 0.17204, 2.44168, -0.55374, 0.83593, -0.1528], [-0.33254, -0.38014, -0.50454, -0.09782, 2.35828, 0.0285, 0.70397, -0.55299], [-0.50022, -0.70162, -0.81517, -0.24336, 2.37335, 0.63356, 0.60451, -0.76644], [-0.29246, -0.87836, -0.69122, -0.44137, 2.52818, 0.89436, 0.43326, -0.80052], [-0.41231, -0.72745, -0.14737, -0.65404, 2.58522, 0.54469, 0.22941, -0.6442], [-0.44642, -0.51924, 0.
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`20102 ms` LLM)
- **Rationale**: The first report was truncated. This targeted inspection checks whether operational inputs leave calibration support and whether pulse or bank layouts need different handling; either finding changes the implementation or validation splits.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,json
p=pickle.load(open('public_batch.pkl','rb')); print('batch',len(p))
for it in p:
 s=it['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; print('entry',list(it), list(s),list(c)); print('cal',x.shape,np.round(x.min(0),5),np.round(x.max(0),5),np.round(y.min(0),5),np.round(y.max(0),5)); z=s['y_obs_raw'].reshape(-1,4); print('obs',z.shape,np.round(z.min(0),5),np.round(z.max(0),5)); print('pulses',[(list(a),None if a['y_pulse_raw'] is None else np.shape(a['y_pulse_raw'])) for a in s['reference_tracer_pulses']]); print('first cal pairs',np.round(np.c_[x,y][:8],5)); print('first obs',np.round(z[:5],5)); print('bank',type(c['single_bank_sweeps']),list(c['single_bank_sweeps'])[:8]);
for fn in ['public_asset_index.json','/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json','/tmp/hegel_continual_archive/archive_manifest.json','/tmp/hegel_continual_archive/archive_apparatus_log.json']:
 a=json.load(open(fn)); print(fn, 'keys',list(a) if isinstance(a,dict) else len(a)); print(str(a)[:1150] if 'log' not in fn else {k:(len(v) if isinstance(v,list) else str(v)[:140]) for k,v in a.items()} if isinstance(a,dict) else list(a[0]))
PY
sed -n '32,150p' _verified_downstream.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network 2>&1 | head -15
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
d] > 0, upper, 0.) - w[free[bad]]) / delta[bad]
                pos = np.argmin(ratios)
                j = bad[pos]
                w[free] = np.clip(w[free] + ratios[pos] * delta, 0., upper)
                active[free[j]] = crosses[j]
                w[free[j]] = upper if crosses[j] > 0 else 0.
                continue
            w[free] = candidate
        g = H @ w + c
        if not len(free):
            lo = np.max(g[active > 0], initial=-np.inf)
            hi = np.min(g[active < 0], initial=np.inf)
            nu = (lo + hi) / 2
        violation = active * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_study: dict) -> dict:
    S, U_last = _responses(cal_study, _operator(cal_study))
    y_cal = cal_study['y_obs_raw']
    Z = np.asarray(cal_study['Z_nuisance'])
    data = np.column_stack((S, y_cal))
    data -= Z @ np.linalg.solve(Z.T @ Z + cal_study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = cal_study['source_ridge']
    upper, total = cal_study['u_box'], cal_study['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, upper, total)
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    diff = g - nu
    dual = float(objective - g @ w + nu * total + upper * np.minimum(0., diff).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., diff).tolist(),
                                'lam_upper': np.maximum(0., -diff).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
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
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
KeyError: 'reference_tracer_pulses'
```

### Turn 3 — `execute_in_remote_docker` (`36211 ms` LLM)
- **Rationale**: Use held-out sweeps to distinguish real nonlinear terms from EMI spikes and polynomial overfit. The public study has no pulse key, so pulse handling will also need a separate contract test.

```bash
cd /holdout_task && cat >/tmp/cal_probe.py <<'PY'
import itertools,json,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; z=s['y_obs_raw'].reshape(-1,4)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
print('shape',x.shape,z.shape,s['y_obs_raw'].shape,'pulses',len(s.get('reference_tracer_pulses',[])))
print('cal first/mid/last',np.c_[x,y][np.r_[0:9,37:42,74:80]])
print('x singular',np.linalg.svd(x-x.mean(0),compute_uv=False))
print('bank type/shape',type(c['single_bank_sweeps']),[(k,np.shape(v)) for k,v in c['single_bank_sweeps'].items()] if isinstance(c['single_bank_sweeps'],dict) else [(type(a),[(k,np.shape(v)) for k,v in a.items()]) for a in c['single_bank_sweeps'][:2]])
print('observations first',z[:8]); print('aux',type(s['aux_manifold_telemetry']),np.shape(s['aux_manifold_telemetry']))
for fn in ['public_asset_index.json','/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json','/tmp/hegel_continual_archive/archive_manifest.json']:
 a=json.load(open(fn)); print(fn,str(a)[:1200])

def basis(x,m):
 a=[np.ones(len(x)),*x.T]
 if m=='full2': a += [x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]
 if m=='full3': a += [np.prod(x[:,idx],axis=1) for q in (2,3) for idx in itertools.combinations_with_replacement(range(4),q)]
 if m=='qcross_cubic': a += [x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]+list((x**3).T)
 if m=='cubic_cross': a += [x[:,i]*x[:,j] for i,j in itertools.combinations(range(4),2)]+list((x**3).T)
 if m=='cubic': a += list((x**3).T)
 if m=='quad_cubic': a += list((x**2).T)+list((x**3).T)
 if m=='cubic5': a += list((x**3).T)+list((x**5).T)
 if m.startswith('tanh'): a += list(np.tanh(float(m[4:])*x).T)
 return np.column_stack(a)

def fit(a,y):
 w=np.ones(len(a))
 for _ in range(25):
  p=np.linalg.lstsq(a*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=1e-10)[0]; r=np.linalg.norm(y-a@p,axis=1); w=np.minimum(1, max(np.median(r)*1.5,1e-7)/np.maximum(r,1e-12))
 keep=r<max(8*np.median(r),.03)
 return np.linalg.lstsq(a[keep],y[keep],rcond=1e-10)[0]

for m in ['affine','full2','cubic','quad_cubic','cubic_cross','qcross_cubic','cubic5','tanh0.4','tanh0.8','tanh1.2','full3']:
 a=basis(x,m); p=fit(a,y); r=np.linalg.norm(a@p-y,axis=1); print(m,'condition',round(np.linalg.cond(a),1),'fit p50/p90',np.quantile(r,[.5,.9]),'spikes',np.flatnonzero(r>2))
 for order in (np.arange(len(x)),np.random.default_rng(11).permutation(len(x)),np.argsort(np.linalg.norm(x,axis=1))):
  scores=[]
  for test in np.array_split(order,5):
   train=np.setdiff1d(np.arange(len(x)),test); rt=np.linalg.norm(a[test]@fit(a[train],y[train])-y[test],axis=1); rt=rt[r[test]<2]; scores.append(np.sqrt(np.mean(rt**2)))
  print(' worst',round(max(scores),6),end='')
 print()
 if m in ['affine','cubic_cross','qcross_cubic','tanh0.8']: print(' coeff',p.T)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_probe.py
```
- **Container Exit Code**: `0` (`805 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s [13 14 15]
 worst 0.211166 worst 0.176484 worst 0.270928
cubic_cross condition 9.2 fit p50/p90 [0.14136 0.21919] spikes [13 14 15]
 worst 0.255387 worst 0.222565 worst 0.399814
 coeff [[-0.01159  1.12881 -0.00659 -0.19002 -0.00444  0.02301 -0.00447  0.00665 -0.00136  0.00159 -0.00127 -0.1784   0.00245  0.01681  0.01298]
 [ 0.15373  0.04669  1.72285  0.04989 -0.26656  0.01047  0.01416 -0.00265  0.0047   0.01451  0.00502 -0.02684 -0.23965 -0.03011  0.01942]
 [ 0.12794 -0.32812 -0.0106   1.76037  0.      -0.0176   0.00963  0.01153 -0.01945  0.01719 -0.01994  0.04848 -0.00341 -0.23344  0.00578]
 [-0.0038  -0.00501 -0.17413 -0.00839  1.0926  -0.00537  0.00147  0.01092  0.00146  0.00783  0.0042  -0.00299  0.01299  0.00291 -0.16374]]
qcross_cubic condition 11.8 fit p50/p90 [0.10304 0.15969] spikes [13 14 15]
 worst 0.215474 worst 0.187317 worst 0.330341
 coeff [[-0.06047  1.13823 -0.01217 -0.15266 -0.00349  0.0281   0.01431 -0.00716  0.0057  -0.00344 -0.0026   0.0048   0.05275  0.00908  0.00041 -0.19002
   0.00636 -0.01965  0.00244]
 [ 0.08842  0.03028  1.77488  0.00896 -0.22374 -0.02367  0.0142  -0.00979 -0.00237  0.0414   0.01444  0.00782 -0.03244 -0.00265  0.10409 -0.00324
  -0.26969  0.0101  -0.0211 ]
 [ 0.06469 -0.27689 -0.03095  1.78595  0.00265  0.07623 -0.02141  0.01038  0.01581 -0.0063  -0.0162   0.02484  0.04148 -0.01604 -0.01045  0.00198
   0.00955 -0.26397 -0.00213]
 [-0.06005 -0.01333 -0.13506  0.00364  1.09365  0.00214 -0.00633  0.00494  0.00681  0.04639  0.00401  0.00962  0.02179  0.00602  0.01054  0.00459
  -0.01802 -0.00752 -0.17409]]
cubic5 condition 53.9 fit p50/p90 [0.08276 0.11446] spikes [13 14 15]
 worst 0.166063 worst 0.106027 worst 0.202602
tanh0.4 condition 133.5 fit p50/p90 [0.13945 0.20341] spikes [13 14 15]
 worst 0.222901 worst 0.192254 worst 0.288753
tanh0.8 condition 30.3 fit p50/p90 [0.11768 0.16729] spikes [13 14 15]
 worst 0.171657 worst 0.153961 worst 0.225258
 coeff [[-0.01618 -0.37332  0.02968 -0.05749  0.08337  1.98381 -0.05367 -0.17384 -0.11578]
 [ 0.14677 -0.03897 -0.50713 -0.09908 -0.15455  0.06065  3.0005   0.16344 -0.14847]
 [ 0.12934 -0.01749  0.00112 -0.36638  0.07373 -0.38237 -0.01275  2.84359 -0.10552]
 [-0.00495 -0.02176 -0.07602  0.0129  -0.35753  0.02114 -0.13059 -0.02662  1.93614]]
tanh1.2 condition 16.4 fit p50/p90 [0.09787 0.13978] spikes [13 14 15]
 worst 0.133431 worst 0.120418 worst 0.172446
full3 condition 23.5 fit p50/p90 [0.07392 0.13736] spikes [13 14 15]
 worst 0.379111 worst 0.274612 worst 0.519066
```

### Turn 4 — `execute_in_remote_docker` (`40431 ms` LLM)
- **Rationale**: Fit targeted saturation families and compare held-out amplitudes and sweep blocks; systematic residuals will determine whether the next model needs additive interactions or saturation applied after mixing.
- **Alternatives Weighed**: `Full cubic polynomial` (rejected: Its worst held-out row RMSE was 0.5191 despite lower training error.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine coupled map is sufficient for the calibration sweeps.` | Witness: `Excluding three gross spike rows, worst held-out row RMSE was 0.4740 for affine versus 0.1724 with tanh features; axis sweeps visibly bend as amplitude increases.` | Blanket Diagnosis: `The affine basis models channel mixing but omits sensor saturation and amplitude-dependent coupling.` | Sublated Pivot: `Retain coupling and compare bounded saturation laws with small interaction bases.`

```bash
cd /holdout_task && grep -inE 'rosette|tanh|calibrat|pulse|saturat' /tmp/hegel_continual_archive/*solver.py | cut -c1-190 | head -50; cat >/tmp/cal_struct.py <<'PY'
import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; z=s['y_obs_raw'].reshape(-1,4)
keep=(abs(y)<5).all(1); X,Y=x[keep],y[keep]; print('axis pairs',np.round(np.c_[x,y][:16],6));
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def design(x,k,w,j,ext):
 t=x[:,j]; g=w[0]; f={'tanh':lambda:np.tanh(g*t),'rat':lambda:t/(1+g*t*t),'root':lambda:t/np.sqrt(1+g*t*t),'atan':lambda:np.arctan(g*t),'erf':lambda:__import__('scipy').special.erf(g*t)}[k]()
 a=[np.ones(len(x)),f,*x[:,np.arange(4)!=j].T]
 if 'l' in ext:a.append(t)
 if 'q' in ext:a.extend((x*x).T)
 if 'c' in ext:a.extend((x**3).T[np.arange(4)!=j])
 if 'r' in ext:a.extend([x[:,i]/(1+w[1]*x[:,i]**2) for i in range(4) if i!=j])
 if 'p' in ext:a.extend([x[:,i]*x[:,l] for i,l in itertools.combinations(range(4),2)])
 if 'd' in ext:a.extend([x[:,i]*t*t for i in range(4) if i!=j]+[x[:,i]**2*t for i in range(4) if i!=j])
 return np.array(a).T
def solve(x,y,k,ext):
 out=[]
 for j in range(4):
  def profile(w, ret=False):
   a=design(x,k,w,j,ext); p=np.linalg.lstsq(a,y[:,j],rcond=1e-10)[0]; return p if ret else a@p-y[:,j]
  w=least_squares(profile,[.8,.6] if 'r' in ext else [.8],bounds=(.005,8),ftol=1e-10).x; out.append((w,profile(w,True)))
 return out
def pred(x,k,e,f): return np.column_stack([design(x,k,w,j,e)@b for j,(w,b) in enumerate(f)])
for k in ['tanh','rat','root','atan','erf']:
 for e in ['', 'q','qc','qr','qcp','qcd','lq','lqr']:
  f=solve(X,Y,k,e); r=pred(X,k,e,f)-Y; rms=lambda r:np.sqrt(np.mean(r*r)); sc=[]
  for order in [np.arange(len(X)),np.argsort(np.linalg.norm(X,axis=1)),np.random.default_rng(11).permutation(len(X))]:
   for test in np.array_split(order,5):
    tr=np.setdiff1d(np.arange(len(X)),test); sc.append(rms(pred(X[test],k,e,solve(X[tr],Y[tr],k,e))-Y[test]))
  print(k,e,'fit',round(rms(r),6),'cv',round(max(sc),6),'scales',[np.round(w,4).tolist() for w,b in f]);
  if k=='tanh' and e in ('q','qr','qcd') or rms(r)<.003: print(' coefs',np.array([b for w,b in f])); print(' resid axis',r[:13])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_struct.py
```
- **Container Exit Code**: `0` (`7919 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.3356, 0.005], [0.3254, 0.005], [0.3167, 8.0], [0.3391, 0.005]]
rat qcp fit 0.028414 cv 0.081815 scales [[0.3479], [0.3254], [0.33], [0.3363]]
rat qcd fit 0.026742 cv 0.11129 scales [[0.3396], [0.3411], [0.3275], [0.3478]]
rat lq fit 0.020113 cv 0.042775 scales [[0.7907], [0.8221], [0.8934], [0.8688]]
rat lqr fit 0.017705 cv 0.041956 scales [[0.8012, 0.005], [0.7929, 0.005], [0.9648, 0.005], [0.8117, 0.005]]
root  fit 0.052101 cv 0.082032 scales [[1.0269], [1.0735], [0.863], [1.0305]]
root q fit 0.023667 cv 0.048566 scales [[1.0626], [0.9915], [0.9952], [1.0222]]
root qc fit 0.02175 cv 0.045733 scales [[1.0403], [1.0207], [0.9969], [1.0409]]
root qr fit 0.021762 cv 0.047623 scales [[1.0405, 0.005], [1.0207, 0.005], [0.9968, 0.005], [1.0408, 0.005]]
root qcp fit 0.019613 cv 0.056181 scales [[1.0635], [1.0251], [1.0223], [1.0327]]
root qcd fit 0.018415 cv 0.063363 scales [[1.0444], [1.0604], [1.0067], [1.0499]]
root lq fit 0.02133 cv 0.045524 scales [[1.3601], [1.4203], [1.6292], [1.5434]]
root lqr fit 0.019114 cv 0.04509 scales [[1.3856, 0.005], [1.3607, 0.005], [1.785, 0.005], [1.4224, 0.005]]
atan  fit 0.050659 cv 0.076578 scales [[1.3964], [1.4322], [1.2732], [1.3921]]
atan q fit 0.022425 cv 0.0442 scales [[1.4083], [1.3591], [1.3554], [1.3784]]
atan qc fit 0.02049 cv 0.041876 scales [[1.3906], [1.3777], [1.358], [1.3885]]
atan qr fit 0.020505 cv 0.042922 scales [[1.3907, 0.005], [1.3776, 0.005], [1.3579, 0.005], [1.3883, 0.005]]
atan qcp fit 0.018593 cv 0.051726 scales [[1.4025], [1.3838], [1.3738], [1.3831]]
atan qcd fit 0.017629 cv 0.051568 scales [[1.3934], [1.4015], [1.3616], [1.3909]]
atan lq fit 0.021867 cv 0.046788 scales [[1.4174], [1.4513], [1.581], [1.5226]]
atan lqr fit 0.019727 cv 0.046519 scales [[1.4331, 0.005], [1.4177, 0.005], [1.6618, 0.005], [1.4572, 0.005]]
erf  fit 0.058037 cv 0.099305 scales [[0.9858], [1.0019], [0.9108], [0.9986]]
erf q fit 0.030407 cv 0.063738 scales [[1.0194], [0.9848], [0.9968], [1.005]]
erf qc fit 0.02861 cv 0.061109 scales [[1.0124], [1.0006], [0.9966], [1.0181]]
erf qr fit 0.028397 cv 0.062958 scales [[1.0124, 0.005], [1.0006, 0.005], [0.9911, 8.0], [1.0181, 0.005]]
erf qcp fit 0.025936 cv 0.074053 scales [[1.0293], [1.0004], [1.0108], [1.0139]]
erf qcd fit 0.0244 cv 0.100683 scales [[1.0153], [1.0242], [1.0055], [1.0279]]
erf lq fit 0.019357 cv 0.040982 scales [[1.3923], [1.4131], [1.4536], [1.4502]]
erf lqr fit 0.016826 cv 0.039781 scales [[1.3993, 0.005], [1.3911, 0.005], [1.5036, 0.005], [1.405, 0.005]]
```

### Turn 5 — `execute_in_remote_docker` (`41210 ms` LLM)
- **Rationale**: The isolated-axis readings show that channels couple in pairs and that the cross response is not odd around zero. Test a compact even-plus-odd cross response on held-out sweeps; remaining structured residuals would motivate interactions between the paired inputs.
- **Alternatives Weighed**: `Add unrestricted mixed polynomial terms` (rejected: The isolated-axis sweeps identify a specific cross-channel pairing and asymmetry; unrestricted terms previously worsened holdout error.)

```bash
cat >/tmp/cal_sep.py <<'PY'
import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=(abs(y)<5).all(1); X,Y=x[good],y[good]
np.set_printoptions(precision=6,suppress=True,linewidth=130)
print('axis output with nominal zero offset removed:');print(np.c_[x,y-[-.06,.07,.06,-.05]][:16])
def odd(t,k,m):
 if m=='T':return np.tanh(k*t)
 if m=='R':return t/(1+k*t*t)
 if m=='A':return np.arctan(k*t)
 if m=='S':return t/np.sqrt(1+k*t*t)
 if m=='L':return t

def basis(x,j,k,m,extras):
 t=x[:,j]; u=x[:,(j+2)%4]; a,b=k; p,q=m[:2]; f,g=odd(t,a,p),odd(u,b,q)
 A=[np.ones(len(t)),f,g]
 for c in m[2:]:
  A.append({'Q':u*u,'D':t*t,'U':u*u/(1+b*u*u),'V':u*u/(1+b*b*u*u),'W':g*g,'I':t*u,'H':t,'J':u,'F':f*f,'G':t*t/(1+a*t*t)}[c])
 if extras: A.extend(x[:,[i for i in range(4) if i not in (j,(j+2)%4)]].T)
 return np.column_stack(A)
def fit(x,y,j,m,extras=False):
 def res(k,coef=False):
  A=basis(x,j,k,m,extras); p=np.linalg.lstsq(A,y[:,j],rcond=1e-10)[0]; return p if coef else A@p-y[:,j]
 k=least_squares(res,[1.,.8],bounds=(.005,10),max_nfev=100,ftol=1e-10,gtol=1e-11).x; return k,res(k,True)
rms=lambda z:np.sqrt(np.mean(z*z)); out=[]
for a,b in itertools.product('TRAS','LTRAS'):
 for ext in ('Q','U','V','W','QI','UI','VI','WI','DQ','DU','DV','DW','HQU','HQ','HU','HW','HQI','HUI','HWI','QJ','UJ','WJ','QG','UG','WG'):
  m=a+b+ext; f=[fit(X,Y,j,m) for j in range(4)]; r=np.column_stack([basis(X,j,k,m,False)@p-Y[:,j] for j,(k,p) in enumerate(f)]); out.append((rms(r),m,f,r))
for v,m,f,r in sorted(out,key=lambda a:a[0])[:14]:
 scores=[]
 for order in (np.arange(len(X)),np.argsort(np.linalg.norm(X,axis=1)),np.random.default_rng(12).permutation(len(X))):
  for te in np.array_split(order,5):
   tr=np.setdiff1d(np.arange(len(X)),te)
   rr=[]
   for j in range(4):
    k,p=fit(X[tr],Y[tr],j,m); rr.append(basis(X[te],j,k,m,False)@p-Y[te,j])
   scores.append(rms(np.array(rr)))
 print(m,'train/heldout',v,max(scores));
 if m in [a[1] for a in sorted(out,key=lambda a:a[0])[:3]]: print(f); print('axis and mixed residual',r[:17]);print('residue correlations',np.corrcoef(np.c_[X,r].T)[:4,4:])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_sep.py
```
- **Container Exit Code**: `0` (`11929 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.000543 -0.003979 -0.003834]
 [ 0.00914  -0.003646 -0.020882  0.015711]
 [-0.00485  -0.01403   0.009982  0.006363]
 [-0.001016  0.000766 -0.002347  0.000855]
 [-0.003744 -0.01039  -0.001939  0.009305]]
residue correlations [[ 0.        0.024496 -0.00008  -0.109919]
 [-0.136396 -0.        0.016868  0.00028 ]
 [ 0.00006  -0.15125   0.        0.045118]
 [-0.059654  0.000706 -0.305414  0.      ]]
RAHQU train/heldout 0.012849461969386416 0.032409257510152535
[(array([0.836762, 0.135794]), array([-0.058136,  0.935484, -1.237418,  0.405808, -0.135733,  0.231303])), (array([0.904513, 0.229125]), array([ 0.068129,  1.461681, -1.185279,  0.692684, -0.147603,  0.32593 ])), (array([0.997744, 0.101507]), array([ 0.056027,  1.456198, -2.67645 ,  0.748711, -0.313498,  0.472833])), (array([0.896612, 0.279692]), array([-0.04769 ,  0.912415, -0.61311 ,  0.420103, -0.053443,  0.148254]))]
axis and mixed residual [[-0.005983 -0.001877 -0.000055  0.002321]
 [-0.002163 -0.001878 -0.003925  0.002318]
 [ 0.001799 -0.001863 -0.002141  0.002324]
 [ 0.014802 -0.001861  0.007195  0.002287]
 [ 0.00186  -0.013732 -0.003969 -0.005351]
 [ 0.001849 -0.008427 -0.003964  0.004008]
 [ 0.001868 -0.003128 -0.003983 -0.000433]
 [ 0.001872  0.020722 -0.003988 -0.004047]
 [-0.000112 -0.001865 -0.013685  0.002297]
 [ 0.001764 -0.001876 -0.012271  0.002315]
 [ 0.001923 -0.001873 -0.005304  0.002308]
 [ 0.003487 -0.001876  0.019251  0.002315]
 [ 0.001871  0.000522 -0.003977 -0.003835]
 [ 0.00914  -0.003653 -0.020878  0.015704]
 [-0.004848 -0.014046  0.009982  0.006369]
 [-0.001018  0.000784 -0.002345  0.000868]
 [-0.003745 -0.010392 -0.001939  0.009319]]
residue correlations [[-0.        0.024662 -0.000079 -0.109843]
 [-0.136379  0.        0.0168    0.000275]
 [ 0.00006  -0.151197  0.        0.045316]
 [-0.059591  0.000711 -0.305428  0.      ]]
RLHUI train/heldout 0.012978952822107712 0.0368663161535855
RSHQU train/heldout 0.013044396111967687 0.034932321404588615
RRHQU train/heldout 0.01308812593091182 0.03481007580551659
RLHU train/heldout 0.013318102512083768 0.0307651944903613
TLHQU train/heldout 0.013508578342645216 0.03291904643074768
TTHQU train/heldout 0.013576672285359056 0.03396490726831287
TAHQU train/heldout 0.013576901685037737 0.03396572412481082
TLHUI train/heldout 0.013647479002272277 0.03853651723226944
TSHQU train/heldout 0.013770263704298293 0.036484625362251
TRHQU train/heldout 0.013813503686168115 0.0363670538711737
TLHU train/heldout 0.014009896734485185 0.032187967042646516
```

### Turn 6 — `execute_in_remote_docker` (`40380 ms` LLM)
- **Rationale**: The separated model leaves residuals well above the isolated-axis noise. Compare pre-saturation mixing and compact paired interactions on held-out slices; keep the simpler model if mixing does not improve those slices.

```bash
cd /holdout_task; python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p)); print('log structure',[(k,type(v).__name__,len(v) if hasattr(v,'__len__') else '') for k,v in a.items()] if isinstance(a,dict) else len(a)); print(str(a)[-3600:])
PY
cat >/tmp/cal_pair.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; X,Y=X[(abs(Y)<5).all(1)],Y[(abs(Y)<5).all(1)]

def sat(x,k,kind):
 if kind=='T': return np.tanh(k*x)
 if kind=='A': return np.arctan(k*x)
 if kind=='S': return x/np.sqrt(1+k*x*x)
 if kind=='R': return x/(1+k*x*x)
 if kind=='E': return erf(k*x)
 return x/(1+k*abs(x))

def design(X,j,k,m):
 t,u=X[:,j],X[:,(j+2)%4]; v=t+k[2]*u; h=k[1]; f=sat(v,k[0],m[0]);
 A=[np.ones(len(t)),f,u,t,u*u,t*t]
 extra={'a':[abs(u),abs(t)],'r':[u/(1+h*u*u),u*u/(1+h*u*u)],'t':[np.tanh(h*u),np.tanh(h*u)**2],'p':[v**3/(1+h*v*v),u**3/(1+h*u*u)],'q':[u*u*t,u*t*t],'s':[sat(t,h,m[0]),sat(u,h,m[0])],'m':[t*u,t*t*t,u*u*u],'b':[np.sin(h*u),np.cos(h*u)],'d':[np.sin(h*u),np.sin(h*t)],'h':[v**2/(1+h*v*v),u*u/(1+h*u*u)],'w':[t/(1+h*t*t),u/(1+h*u*u)]}
 for key in m[1:]: A+=extra[key]
 return np.column_stack(A)

def fit(X,Y,j,m):
 def resid(k,coeff=False):
  A=design(X,j,k,m); p=np.linalg.lstsq(A,Y[:,j],rcond=1e-9)[0]
  return p if coeff else A@p-Y[:,j]
 start=[[1,1,c] for c in (0,-.18,.18)]; best=min((least_squares(resid,k,bounds=([.03,.03,-.8],[8,10,.8]),max_nfev=100,gtol=1e-10) for k in start),key=lambda r:r.cost)
 return best.x,resid(best.x,True)

def pred(X,m,p):return np.column_stack([design(X,j,k,m)@b for j,(k,b) in enumerate(p)])
def train(X,Y,m):return [fit(X,Y,j,m) for j in range(4)]
def rms(z):return np.sqrt(np.mean(z*z))
if __name__=='__main__':
 results=[]
 for a in 'TASREB':
  for b in ('','a','r','t','p','q','s','m','b','d','h','w','rq','rh','bh','dh'):
   m=a+b; p=train(X,Y,m); results.append((rms(pred(X,m,p)-Y),m,p))
 for e,m,p in sorted(results)[:12]:
  errs=[]
  for order in (np.arange(len(X)),np.argsort(np.max(abs(X),axis=1)),np.random.default_rng(45).permutation(len(X))):
   for te in np.array_split(order,5):
    tr=np.setdiff1d(np.arange(len(X)),te); errs.append(rms(pred(X[te],m,train(X[tr],Y[tr],m))-Y[te]))
  print(m,e,max(errs),flush=True)
  if e<.004: print([(np.round(k,5),np.round(b,5)) for k,b in p],flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_pair.py
```
- **Container Exit Code**: `0` (`31079 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
000e-01])), (array([7.5633e-01, 2.1318e-01, 5.0000e-05]), array([-5.0130000e-02,  5.6370300e+00, -1.5167000e-01,  3.4654230e+01,
       -7.2660000e-02, -7.4000000e-04, -6.8680000e-02, -1.7652581e+02,
        1.0200000e-03,  1.6955000e-01]))]
Sdh 0.0005850787309944414 0.0071248394608588415
[(array([ 2.9890e-01,  2.0755e-01, -9.0000e-05]), array([-5.9930000e-02,  5.7839000e+00, -1.9394000e-01,  4.1069750e+01,
       -7.4910000e-02,  1.2300000e-03,  1.2863000e-01, -2.1946857e+02,
       -1.5200000e-03,  1.7236000e-01])), (array([3.0801e-01, 1.8783e-01, 3.0000e-05]), array([ 7.0070000e-02,  8.8522200e+00, -2.5136000e-01,  7.8315860e+01,
       -1.3605000e-01,  3.8000000e-03, -9.3930000e-02, -4.5298378e+02,
       -4.4800000e-03,  2.9054000e-01])), (array([3.1634e-01, 1.8923e-01, 7.0000e-05]), array([ 6.0010000e-02,  8.6840300e+00, -2.2120000e-01,  7.5951860e+01,
       -1.3927000e-01,  6.2600000e-03, -2.7873000e-01, -4.3606395e+02,
       -7.7700000e-03,  2.9682000e-01])), (array([2.9434e-01, 2.1452e-01, 6.0000e-05]), array([-5.0130000e-02,  5.8391500e+00, -1.5037000e-01,  3.8441020e+01,
       -7.2160000e-02, -8.3000000e-04, -7.4840000e-02, -2.0042332e+02,
        1.1400000e-03,  1.6913000e-01]))]
Adh 0.0006216645906100542 0.007537033292361399
[(array([ 6.0431e-01,  2.0808e-01, -9.0000e-05]), array([-5.9920000e-02,  1.2214430e+01, -1.9515000e-01,  4.3776690e+01,
       -7.4640000e-02,  1.2500000e-03,  1.3490000e-01, -2.3960196e+02,
       -1.5400000e-03,  1.7210000e-01])), (array([6.1443e-01, 1.8747e-01, 3.0000e-05]), array([ 7.0080000e-02,  1.8297510e+01, -2.5197000e-01,  8.4615960e+01,
       -1.3628000e-01,  3.9600000e-03, -9.1110000e-02, -5.0020604e+02,
       -4.6500000e-03,  2.9071000e-01])), (array([6.2344e-01, 1.8900e-01, 7.0000e-05]), array([ 6.0010000e-02,  1.7634410e+01, -2.1763000e-01,  8.1890170e+01,
       -1.3956000e-01,  6.5300000e-03, -2.9908000e-01, -4.8024356e+02,
       -8.1000000e-03,  2.9711000e-01])), (array([5.9929e-01, 2.1551e-01, 6.0000e-05]), array([-5.0140000e-02,  1.2457860e+01, -1.4953000e-01,  4.0702780e+01,
       -7.1790000e-02, -8.9000000e-04, -7.8990000e-02, -2.1754322e+02,
        1.2300000e-03,  1.6883000e-01]))]
Bdh 0.0056321433310645 0.018927105109859097
Ebh 0.007189390661884459 0.039918548784693496
Erh 0.007458827307886859 0.040662090271146506
Eh 0.007495659191456243 0.04135932608101599
Rbh 0.008156837737608362 0.044494556547560474
Rrh 0.008426797749005644 0.04493441388028641
Rh 0.008466556216679974 0.04609408266570742
```

### Turn 7 — `execute_in_remote_docker` (`44767 ms` LLM)
- **Rationale**: Isolated-axis responses have nearly constant ratios across channels, suggesting a shared shape with channel-specific gains. Compare that restriction against flexible paired polynomials on sweep-block and amplitude holdouts; use the results to choose a stable forward map for inversion.
- **Alternatives Weighed**: `Keep the per-channel mixing fit as-is` (rejected: Its fitted mixing is effectively zero and its large cancelling coefficients may destabilize inversion outside the central sweep range.)

```bash
cat >/tmp/cal_shared.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=np.max(abs(y),axis=1)<5; X,Y=x[good],y[good]

def poly(x,n):
 u=x[:,[2,3,0,1]]
 return np.stack([np.ones_like(x)]+[x**k for k in range(1,n+1,2)]+[u**k for k in range(1,n+1)],axis=-1)
def lin(a,y): return np.array([np.linalg.lstsq(a[:,j],y[:,j],rcond=1e-10)[0] for j in range(4)])
def pred(a,b): return np.einsum('nij,ij->ni',a,b)
def rms(r):return np.sqrt(np.mean(r*r))
folds=[(np.setdiff1d(np.arange(len(X)),t),t) for order in (np.arange(len(X)),np.argsort(np.max(abs(X),axis=1)),np.random.default_rng(11).permutation(len(X))) for t in np.array_split(order,5)]
for n in (5,7,9,11,13):
 a=poly(X,n); b=lin(a,Y); print('poly',n,rms(pred(a,b)-Y),max(rms(pred(a[te],lin(a[tr],Y[tr]))-Y[te]) for tr,te in folds)); print(np.round(b,7))

def fn(t,k,m):
 if m=='T': return np.tanh(k*t)
 if m=='E': return erf(k*t)
 if m=='A': return np.arctan(k*t)
 if m=='R': return t/(1+k*t*t)
 if m=='S': return t/np.sqrt(1+k*t*t)
 if m=='N': return np.sin(k*t)

def design(x,k,m):
 u=x[:,[2,3,0,1]]; a,b,c=k
 own={'1':lambda:[fn(x,a,m[0])],'L':lambda:[fn(x,a,m[0]),x],'C':lambda:[fn(x,a,m[0]),x,x**3],'N':lambda:[fn(x,a,m[0]),np.sin(b*x),x],'D':lambda:[fn(x,a,m[0]),fn(x,b,m[0])],'P':lambda:[fn(x,a,m[0]),x/(1+b*x*x),x],'G':lambda:[x/(1+a*x*x+b*x**4),x]}[m[1]]()
 cross={'Q':lambda:[u,u*u],'R':lambda:[u,u*u/(1+c*u*u)],'L':lambda:[u,u*u,u*u/(1+c*u*u)],'N':lambda:[u,np.sin(c*u),1-np.cos(c*u)],'K':lambda:[u,1-np.cos(c*u)],'T':lambda:[u,np.tanh(c*u),np.tanh(c*u)**2],'S':lambda:[u,u/(1+c*u*u),u*u/(1+c*u*u)],'P':lambda:[u,u**3,u**2,u**4],'G':lambda:[u,u*np.exp(-c*u*u),1-np.exp(-c*u*u)]}[m[2]]()
 return np.stack([np.ones_like(x),*own,*cross],axis=-1)
def fit(x,y,m):
 def f(k,ret=False):
  a=design(x,k,m); b=lin(a,y); return b if ret else (pred(a,b)-y).ravel()
 z=min((least_squares(f,k,bounds=([.005,.005,.005],[6,6,6]),max_nfev=90,ftol=1e-10,gtol=1e-11) for k in ([.8,.6,.5],[1.5,1.2,1.2])),key=lambda z:z.cost)
 return z.x,f(z.x,True),rms(z.fun)
rs=[]
for a in 'TEARSN':
 for b in '1LCNDPG' if a=='R' else '1LCNDP':
  for c in 'QRLNKTSPG':
   m=a+b+c; k,p,e=fit(X,Y,m); rs.append((e,m,k,p))
for e,m,k,p in sorted(rs)[:12]:
 vals=[]
 for tr,te in folds:
  a,b,_=fit(X[tr],Y[tr],m); vals.append(rms(pred(design(X[te],a,m),b)-Y[te]))
 print('shared',m,e,max(vals),'scales',k,flush=True)
 if m in [r[1] for r in sorted(rs)[:4]]: print(p,flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_shared.py
```
- **Container Exit Code**: `0` (`15405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
6  4.73977402e-02]]
shared RNN 9.356000064298296e-06 2.4496134201904903e-05 scales [0.75375971 2.50000699 2.00004689]
[[-6.00022363e-02 -2.23730234e-05  2.00005740e-01  8.00011468e-01
  -1.68000618e-01 -3.77546019e-07  4.80007942e-02]
 [ 6.99991869e-02  2.22728112e-04  3.19953531e-01  1.27989969e+00
  -2.68801254e-01  2.01970369e-06  7.68012699e-02]
 [ 5.99968117e-02 -2.73511457e-05  3.25004217e-01  1.30001303e+00
  -2.72999934e-01 -3.43483377e-07  7.79987570e-02]
 [-4.99986203e-02 -7.19787320e-05  1.97513976e-01  7.90031960e-01
  -1.65897406e-01 -5.44610579e-06  4.73977401e-02]]
shared NCN 9.366451323214993e-06 1.4294650155089892e-05 scales [2.50006967 1.2        2.00004526]
[[-6.00021357e-02  1.99983188e-01  8.00035049e-01 -1.31732902e-05
  -1.68000681e-01 -2.70249965e-07  4.80008195e-02]
 [ 6.99992090e-02  3.19965266e-01  1.28006418e+00 -2.73645475e-05
  -2.68801457e-01  2.21615086e-06  7.68012583e-02]
 [ 5.99963622e-02  3.24973803e-01  1.30004758e+00 -1.70537084e-05
  -2.73000219e-01 -9.11668932e-09  7.79990618e-02]
 [-4.99982626e-02  1.97481923e-01  7.90031583e-01 -1.22070691e-05
  -1.65897190e-01 -5.67546747e-06  4.73973422e-02]]
shared NNN 9.366451357335577e-06 1.4291912398789526e-05 scales [2.50006967 0.0078549  2.00004526]
[[-6.00021357e-02  1.99983187e-01  1.63100124e+02 -4.81100128e-01
  -1.68000681e-01 -2.70252411e-07  4.80008194e-02]
 [ 6.99992090e-02  3.19965265e-01  3.38798993e+02 -1.38116807e+00
  -2.68801457e-01  2.21614465e-06  7.68012583e-02]
 [ 5.99963621e-02  3.24973801e-01  2.11146093e+02 -3.58483887e-01
  -2.73000219e-01 -9.11578038e-09  7.79990618e-02]
 [-4.99982625e-02  1.97481922e-01  1.51137526e+02 -3.97138585e-01
  -1.65897190e-01 -5.67547737e-06  4.73973422e-02]]
shared RNK 9.430445233560684e-06 2.1346209843808893e-05 scales [0.84420012 2.50000448 2.00003393]
shared NPK 9.430445247516412e-06 2.1345798079507123e-05 scales [2.50000448 0.84408865 2.00003393]
shared NDN 9.432651045752337e-06 1.2070548868777381e-05 scales [2.50006177 0.00931676 2.00004434]
shared NLN 9.447185144303754e-06 1.20758039997383e-05 scales [2.50000267 0.96323645 2.00004184]
shared NCK 9.452173850739887e-06 1.47937433079891e-05 scales [2.5000613  0.60559872 2.00003385]
shared NNK 9.452173998501044e-06 1.4792418985645019e-05 scales [2.5000613  0.02143316 2.00003385]
shared NDK 9.50893781817392e-06 1.2078773485001065e-05 scales [2.50005313 0.00860166 2.00002954]
shared NLK 9.519476159386974e-06 1.2067833170445885e-05 scales [2.50000278 1.19998964 2.00002817]
```

### Turn 8 — `execute_in_remote_docker` (`52750 ms` LLM)
- **Rationale**: Implement the compact map and run the public schema gate. The fitted coefficients and operational inversion residual will show whether the next step needs a schema correction or a more stable inverse.
- **Alternatives Weighed**: `Higher-degree paired polynomial` (rejected: It fits calibration noise in sample but extrapolates worse on amplitude holdouts and complicates inversion.)
- **ATMS NOGOOD DECLARED**: Refuted `Conventional saturation or rational terms are needed to explain the observed bending.` | Witness: `The compact paired sinusoidal basis reduced worst-slice forward RMSE to 1.21e-5, near sweep noise, versus 0.00654 for the prior Tdh model.` | Blanket Diagnosis: `The prior basis approximated the curvature with large cancelling terms while omitting a shared odd self response and even cross response.` | Sublated Pivot: `Fit shared frequencies and channel-specific coefficients, reject gross EMI by row residual, then invert the paired systems.`

```bash
set -e
cd /holdout_task
cat > upstream_transducer.py <<'PY'
"""Fit the paired rosette response and convert readings to reference coordinates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_PAIR = [2, 3, 0, 1]


def _features(x, freq):
    u = x[:, _PAIR]
    return np.stack((np.ones_like(x), x, np.sin(freq[0] * x),
                     u, 1 - np.cos(freq[1] * u)), axis=-1)


def _predict(x, freq, coef):
    return np.einsum('nck,ck->nc', _features(x, freq), coef)


def _fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    freq = np.array([2.5, 2.0])
    weights = np.ones(len(x))

    def profile(f, weights):
        a = _features(x, f)
        w = np.sqrt(weights)
        return np.array([np.linalg.lstsq(a[:, j] * w[:, None], y[:, j] * w,
                                        rcond=None)[0] for j in range(4)])

    # Row weights prevent an EMI spike from contaminating all four gains.
    for _ in range(20):
        coef = profile(freq, weights)
        err = np.linalg.norm(_predict(x, freq, coef) - y, axis=1)
        cutoff = max(2 * np.median(err), 1e-8)
        weights = np.minimum(1, cutoff / np.maximum(err, 1e-15))
    clean = err <= 8 * cutoff
    x, y = x[clean], y[clean]
    weights = np.ones(len(x))
    for _ in range(2):
        def residual(f):
            return ((_predict(x, f, profile(f, weights)) - y)
                    * np.sqrt(weights[:, None])).ravel()

        freq = least_squares(residual, freq, bounds=([1.8, 1.2], [3.2, 2.8]),
                             max_nfev=60).x
        coef = profile(freq, weights)
        err = np.linalg.norm(_predict(x, freq, coef) - y, axis=1)
        weights = np.minimum(1, max(2 * np.median(err), 1e-8)
                             / np.maximum(err, 1e-15))
    return freq, coef


def _invert(raw, freq, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    bias, own, sine, cross, cosine = coef.T
    x = (y - bias) / (own + freq[0] * sine)
    for _ in range(60):
        r = _predict(x, freq, coef) - y
        err = np.max(np.abs(r), axis=1)
        active = err > 2e-12
        if not np.any(active):
            break
        t, rr = x[active], r[active]
        a = own + freq[0] * sine * np.cos(freq[0] * t)
        b = cross + freq[1] * cosine * np.sin(freq[1] * t[:, _PAIR])
        # Each opposite-channel pair forms a 2-by-2 Newton system.
        step = (a[:, _PAIR] * rr - b * rr[:, _PAIR]) / (a * a[:, _PAIR] - b * b[:, _PAIR])
        scale = np.ones(len(t))
        for _ in range(18):
            cand = t - scale[:, None] * step
            bad = np.max(np.abs(_predict(cand, freq, coef) - y[active]), axis=1) >= err[active]
            if not np.any(bad):
                break
            scale[bad] *= 0.5
        x[active] = cand
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study without mutating the input or its sweeps."""
    out = dict(study)
    sweeps = dict(study['sensor_calibration_sweeps'])
    freq, coef = _fit(sweeps['ref_rosette'], sweeps['raw_rosette'])
    out['y_obs_raw'] = _invert(study['y_obs_raw'], freq, coef)
    if study.get('reference_tracer_pulses') is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], freq, coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in study['reference_tracer_pulses']
        ]
    sweeps['raw_rosette'] = np.asarray(sweeps['ref_rosette']).copy()
    banks = sweeps['single_bank_sweeps']

    def identity(bank):
        return dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy())

    if isinstance(banks, dict):
        sweeps['single_bank_sweeps'] = (identity(banks) if 'ref_signal' in banks
                                        else {k: identity(v) for k, v in banks.items()})
    else:
        sweeps['single_bank_sweeps'] = type(banks)(identity(bank) for bank in banks)
    out['sensor_calibration_sweeps'] = sweeps
    return out
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; b=s['sensor_calibration_sweeps']['single_bank_sweeps']; print('bank layout:', type(b), list(b)[:4] if isinstance(b,dict) else (len(b),list(b[0]))); from upstream_transducer import _fit,_predict,_invert
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; f,p=_fit(x,y); print('fit',f,p); print('obs inverse residual',np.max(abs(_predict(_invert(s['y_obs_raw'],f,p).reshape(-1,4),f,p)-s['y_obs_raw'].reshape(-1,4))))
PY
```
- **Container Exit Code**: `0` (`747 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
bank layout: <class 'dict'> ['bank_id', 'ref_signal', 'raw_readout']
fit [2.5000018  2.00003105] [[-0.06000205  0.80000074  0.20000138 -0.16800093  0.04800109]
 [ 0.06999926  1.27999748  0.32000123 -0.26880059  0.07680107]
 [ 0.05999721  1.29999983  0.32499868 -0.27300004  0.07799924]
 [-0.04999858  0.78999883  0.1975004  -0.16590021  0.04739859]]
obs inverse residual 1.8240964294591322e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8057392a2875e135 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8057392a (directory).
```

### Turn 9 — `execute_in_remote_docker` (`31599 ms` LLM)
- **Rationale**: The public schema passes, but it has no pulse and checks no held-out reference reconstructions. Test both in an isolated process and stress wide amplitudes; any Newton stalls would justify a monotone inverse fallback.

```bash
set -e
cd /holdout_task
sed -n '55,150p' _verified_downstream.py
cat >/tmp/verify_rosette.py <<'PY'
import copy, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _predict, _invert, calibrate_upstream
from batch_solver import solve_batch
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x,y = c['ref_rosette'],c['raw_rosette']; spike = np.max(abs(y),axis=1)>5
worst = np.zeros(3)
for order in (np.arange(len(x)), np.argsort(np.max(abs(x),axis=1)), np.random.default_rng(11).permutation(len(x))):
    for te in np.array_split(order,5):
        tr = np.setdiff1d(np.arange(len(x)),te); te = te[~spike[te]]
        f,p = _fit(x[tr],y[tr]); e = _invert(y[te],f,p)-x[te]
        worst = np.maximum(worst,[np.sqrt(np.mean((_predict(x[te],f,p)-y[te])**2)),np.sqrt(np.mean(e**2)),np.max(abs(e))])
print('public worst fold [forward RMSE, inverse RMSE, inverse max]',worst)
assert np.all(worst<[3e-5,6e-5,2e-4])
f,p = _fit(x,y)
rng=np.random.default_rng(82)
for amp in (.8,1.8,4.,12.):
    t=rng.uniform(-amp,amp,(20000,4)); raw=_predict(t,f,p); out=_invert(raw,f,p)
    print('inverse roundtrip',amp,np.max(abs(out-t)), np.sqrt(np.mean((out-t)**2)))
    assert np.max(abs(out-t))<1e-8
s=copy.deepcopy(s); obs=s['y_obs_raw'].copy(); raw_sweep=s['sensor_calibration_sweeps']['raw_rosette'].copy(); t=rng.uniform(-1.6,1.6,(18,4)); pulse=_predict(t,f,p).reshape(2,3,12); s['reference_tracer_pulses']=[{'y_pulse_raw':pulse,'label':'example'},{'y_pulse_raw':None}]
o=calibrate_upstream(s); oc=o['sensor_calibration_sweeps']; b=oc['single_bank_sweeps']
np.testing.assert_allclose(o['reference_tracer_pulses'][0]['y_pulse_raw'],t.reshape(pulse.shape),atol=1e-9,rtol=0)
assert o['reference_tracer_pulses'][0]['label']=='example' and o['reference_tracer_pulses'][1]['y_pulse_raw'] is None
for lhs,rhs in ((oc['raw_rosette'],oc['ref_rosette']),(b['raw_readout'],b['ref_signal'])):
    np.testing.assert_array_equal(lhs,rhs); assert not np.shares_memory(lhs,rhs)
np.testing.assert_array_equal(s['y_obs_raw'],obs); np.testing.assert_array_equal(s['sensor_calibration_sweeps']['raw_rosette'],raw_sweep)
assert o['y_obs_raw'].shape==obs.shape
out=solve_batch(s); print('pipeline fields',list(out)); print('certificate summary',[(k,list(v)) for k,v in out.items() if isinstance(v,dict)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_rosette.py
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`1342 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ue
            w[free] = candidate
        g = H @ w + c
        if not len(free):
            lo = np.max(g[active > 0], initial=-np.inf)
            hi = np.min(g[active < 0], initial=np.inf)
            nu = (lo + hi) / 2
        violation = active * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_study: dict) -> dict:
    S, U_last = _responses(cal_study, _operator(cal_study))
    y_cal = cal_study['y_obs_raw']
    Z = np.asarray(cal_study['Z_nuisance'])
    data = np.column_stack((S, y_cal))
    data -= Z @ np.linalg.solve(Z.T @ Z + cal_study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = cal_study['source_ridge']
    upper, total = cal_study['u_box'], cal_study['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, upper, total)
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    diff = g - nu
    dual = float(objective - g @ w + nu * total + upper * np.minimum(0., diff).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., diff).tolist(),
                                'lam_upper': np.maximum(0., -diff).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
public worst fold [forward RMSE, inverse RMSE, inverse max] [1.20379545e-05 2.73989992e-05 9.92840338e-05]
inverse roundtrip 0.8 3.590017172427906e-12 1.9522448587670446e-13
inverse roundtrip 1.8 1.1751266626447432e-11 5.199647773446361e-13
inverse roundtrip 4.0 1.4912515666765103e-11 4.1276240847574843e-13
inverse roundtrip 12.0 1.1388223697395006e-11 3.8321755558574015e-13
pipeline fields ['weights', 'terminal_field', 'kkt_certificate']
certificate summary [('kkt_certificate', ['weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'])]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8057392a2875e135 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8057392a (directory).
```

### Turn 10 — `submit_completed_task` (`14488 ms` LLM)
- **Rationale**: The final bundle passed isolated-process verification on chronological, amplitude, and shuffled holdouts, plus pulse and identity-sweep contract tests. The unchanged downstream composition passed the public schema gate.
- **Alternatives Weighed**: `Add an inverse fallback` (rejected: Wide-amplitude round-trip tests found no convergence failures; maximum absolute error was 1.50e-11.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:204', '/holdout_task/upstream_transducer.py': 'FILE:4028'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration in `/holdout_task/upstream_transducer.py` and verified its integration with `/holdout_task/batch_solver.py`. The public schema passed, worst held-out inverse RMSE was 2.74e-5, and a synthetic pulse test verified shape preservation and handling of missing readings.`

### Turn 11 — `execute_in_remote_docker` (`40126 ms` LLM)
- **Rationale**: Fit residual challengers only on training sweeps and measure held-out errors and signed bias separately for axis and mixed sweeps. Propagate each fold's calibration through the actual composed solver to detect changes in weights, terminal fields, or KKT feasibility that the schema test would miss.
- **Alternatives Weighed**: `Run relay, classification, or drift-operator benchmarks` (rejected: The public study is a continuous calibration and PDE inversion problem; it supplies no relay states or class labels, and the verified downstream operator is outside the calibration task.)

```bash
set -e
cat >/tmp/audit_rosette.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _predict, _invert, calibrate_upstream
from batch_solver import solve_batch
p = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
s = p[0]['study']; c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
clean = np.max(abs(y), axis=1) < 5
axis = np.count_nonzero(x, axis=1) == 1
pairing = [2, 3, 0, 1]
def extra(a, kind, j):
    t, u = a[:, j], a[:, pairing[j]]
    if kind == 'other_channels':
        v, w = a[:, (j+1)%4], a[:, (j+3)%4]
        return np.column_stack((v,w,v*v,w*w,v*w))
    if kind == 'pair_difference':
        return np.column_stack((t*u,t*t*u,t*u*u,np.sin(2.5*(t-.21*u))-np.sin(2.5*t),np.sin(2.5*(t+.21*u))-np.sin(2.5*t)))
    return np.column_stack([z/(1+.6*z*z) for z in (t,u,t-.21*u,t+.21*u)] + [u*u/(1+.6*u*u)])
models = ('baseline','other_channels','pair_difference','rational')
errs = {k: [] for k in models}
base = solve_batch(s); w0 = np.asarray(base['weights']); field0 = base['terminal_field']
max_changes = np.zeros(3); gaps = []
orders = (np.arange(len(x)), np.argsort(np.max(abs(x),axis=1)))
for order in orders:
    for te in np.array_split(order, 5):
        tr = np.setdiff1d(np.arange(len(x)),te)
        f,b = _fit(x[tr], y[tr]); goodtr = tr[clean[tr]]; goodte = te[clean[te]]
        r = y[goodtr] - _predict(x[goodtr],f,b)
        pred = _predict(x[goodte],f,b)
        for m in models:
            z = pred.copy()
            if m != 'baseline':
                for j in range(4):
                    beta = np.linalg.lstsq(extra(x[goodtr],m,j),r[:,j],rcond=1e-8)[0]
                    z[:,j] += extra(x[goodte],m,j) @ beta
            e = z-y[goodte]
            errs[m].append([np.sqrt(np.mean(e**2)), *[np.max(np.abs(e[a].mean(0))) if np.any(a) else 0. for a in (axis[goodte],~axis[goodte])]])
        cc = dict(c, ref_rosette=x[tr], raw_rosette=y[tr]); out = solve_batch(dict(s,sensor_calibration_sweeps=cc)); w = np.asarray(out['weights']); cert = out['kkt_certificate']
        max_changes = np.maximum(max_changes,[np.max(abs(w-w0)),np.max(abs(out['terminal_field']-field0)),abs(w.sum()-s['total_budget'])])
        assert np.isfinite(out['terminal_field']).all() and w.min() >= -1e-10 and w.max() <= s['u_box']+1e-10
        gaps.append(abs(cert['primal_objective']-cert['dual_lower_bound']))
for m in models: print(m, 'worst [forward RMSE, axis mean bias, mixed mean bias]', np.max(errs[m],axis=0))
print('max [weight change, terminal field change, budget error]',max_changes,'max certificate gap',max(gaps))
print('public studies',len(p),'axis clean rows',sum(axis&clean),'mixed clean rows',sum(~axis&clean),'EMI rows',np.flatnonzero(~clean).tolist(),'cohort / mode fields',[k for k in s if any(z in k for z in ('batch','class','relay','meter','cohort'))])
assert max_changes[2] < 1e-10 and max(gaps) < 1e-9
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_rosette.py
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline worst [forward RMSE, axis mean bias, mixed mean bias] [1.20379545e-05 1.16989497e-05 7.68583938e-06]
other_channels worst [forward RMSE, axis mean bias, mixed mean bias] [1.21186546e-05 1.23767185e-05 7.57263553e-06]
pair_difference worst [forward RMSE, axis mean bias, mixed mean bias] [1.21096881e-05 9.94647149e-06 7.52199679e-06]
rational worst [forward RMSE, axis mean bias, mixed mean bias] [1.23086940e-05 1.14661651e-05 7.19839775e-06]
max [weight change, terminal field change, budget error] [7.22424314e-06 6.64884568e-07 2.22044605e-16] max certificate gap 1.8041124150158794e-16
public studies 1 axis clean rows 13 mixed clean rows 64 EMI rows [13, 14, 15] cohort / mode fields []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8057392a2875e135 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8057392a (directory).
```

### Turn 12 — `submit_completed_task` (`16585 ms` LLM)
- **Rationale**: The unchanged bundle passed isolated-process verification. Refitting on held-out sweep splits changed downstream weights by at most 7.23e-6 and terminal fields by at most 6.65e-7, with budget error and certificate gap below 3e-16. The public study has no relay modes or classification cohorts to audit.
- **Alternatives Weighed**: `Add rational, pair-difference, or other-channel terms` (rejected: Some slice biases fell slightly, but no challenger improved worst-fold forward RMSE or demonstrated a calibration bottleneck that justified the extra terms.)
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration in `/holdout_task/upstream_transducer.py` and verified it through `/holdout_task/batch_solver.py`. The public schema passed, and the worst held-out inverse RMSE was 2.74e-5. A synthetic pulse test confirmed shape preservation and correct handling of missing readings.`

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
