# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m98_00_batch_04__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `248,165` (`239,358` / `65,920` / `8,807`) |
| **Total Episode Wall Time (s)** | `310.44s` |

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

### Turn 1 — `execute_in_remote_docker` (`25236 ms` LLM)
- **Rationale**: Inspect the composition contract and calibration ranges before selecting a model. Unexpected shapes would change how observations are transformed, and extreme calibration rows would motivate robust fitting rather than ordinary least squares.

```bash
cd /holdout_task && ls -l && cat batch_solver.py validate_public_schema.py _verified_downstream.py && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
print('batch',type(x),len(x))
if isinstance(x,dict): print('keys',list(x)); studies=[x] if 'sensor_calibration_sweeps' in x else next((v for v in x.values() if isinstance(v,list) and v and isinstance(v[0],dict)),[])
else: studies=x
for s in studies[:8]:
 print('study',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in s.items()]); c=s.get('sensor_calibration_sweeps',{}); print('cal',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in c.items()]);
 for k in ('ref_rosette','raw_rosette'):
  if k in c: print(k,np.quantile(c[k],[0,.01,.5,.99,1],axis=0))
 p=s.get('reference_tracer_pulses'); print('pulses',type(p).__name__,None if p is None else [(list(t),np.shape(t.get('y_pulse_raw'))) for t in p][:4])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS multivariate nonlinear calibration'; ls -l /tmp/hegel_continual_archive
```
- **Container Exit Code**: `0` (`467 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: NOT_NEEDED_FOR_STATIC_REGRESSION(1d_monotone_time_relaxation_only)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: NOT_NEEDED_FOR_STATIC_REGRESSION(1d_step_split_only)
total 52
-r--r--r-- 1 root root 10549 Nov 27 13:51 archive_apparatus_log.json
-rw-r--r-- 1 root root  1387 Nov 27 13:51 archive_manifest.json
-rw-r--r-- 1 root root  4955 Nov 27 13:51 dev_beta_m98_00_batch_01_solver.py
-rw-r--r-- 1 root root  4799 Nov 27 13:51 dev_beta_m98_00_batch_02_solver.py
-rw-r--r-- 1 root root  4809 Nov 27 13:51 dev_beta_m98_00_batch_03_solver.py
-rw-r--r-- 1 root root  4799 Nov 27 13:51 dev_beta_m98_00_s00_solver.py
-rw-r--r-- 1 root root   227 Nov 27 13:51 pre_llm_gate_diagnostic.json
```

### Turn 2 — `execute_in_remote_docker` (`20033 ms` LLM)
- **Rationale**: Inspect the nested study and compare sweep and observation ranges. The field locations determine what must be transformed, and any observations outside sweep support will call for an extrapolation check before model selection.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb'));print('items',len(b))
for it in b:
 s=it['study']; c=s['sensor_calibration_sweeps'];p=s['reference_tracer_pulses']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); z=np.asarray(s['y_obs_raw']);print('item',list(it),'study',list(s),'cal',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in c.items()]);print('sweep quantiles',np.quantile(x,[0,.5,1],axis=0),np.quantile(y,[0,.5,1],axis=0));print('obs',z.shape,np.quantile(z.reshape(-1,4),[0,.5,1],axis=0)); print('pulses',None if p is None else [(list(q),np.shape(q['y_pulse_raw'])) for q in p]);print('first refs/raw',x[:5],y[:5]);print('single-bank locations',[(k,list(v) if isinstance(v,dict) else type(v).__name__) for d in (s,c) for k,v in d.items() if 'bank' in k]);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -45
rg -n 'rosette|single_bank|y_obs|pulse' /holdout_task/_verified_downstream.py /tmp/hegel_continual_archive/*solver.py | head -45
```
- **Container Exit Code**: `0` (`416 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
loss in {'huber', 'cauchy'}.",
        "weights >= 0 when provided; sensitivity must have nonzero weighted norm above gain_floor for identifiable slope."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "propagate_linear_basis": "APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)",
        "profile_weighted_affine": "APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)",
        "robust_residual_weights": "APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)"
      },
      "functions": [
        {
          "function": "robust_residual_weights",
          "signature": "robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray",
          "doc_summary": "For residuals of shape S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_affine_projection",
      "reference_source_txt": null
    }
  ]
}
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
KeyError: 'reference_tracer_pulses'
bash: line 8: rg: command not found
```

### Turn 3 — `execute_in_remote_docker` (`19822 ms` LLM)
- **Rationale**: The previous inspection stopped at a missing optional pulse key. Read the schema without requiring that key, and compare sweep coverage with the downstream input range to determine whether held-out edge rows should drive model selection.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
for it in pickle.load(open('public_batch.pkl','rb')):
 s=it['study']; c=s['sensor_calibration_sweeps']; p=s.get('reference_tracer_pulses'); x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); z=np.asarray(s['y_obs_raw']);print('item',list(it),'study',list(s),'cal',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in c.items()]); print('sweep',x.shape,np.quantile(x,[0,.5,1],axis=0),np.quantile(y,[0,.5,1],axis=0));print('obs',z.shape,np.quantile(z.reshape(-1,4),[0,.5,1],axis=0)); print('pulses',None if p is None else [(list(q),np.shape(q['y_pulse_raw'])) for q in p]);print('first refs/raw',x[:5],y[:5]);print('single-bank locations',[(k,list(v) if isinstance(v,dict) else type(v).__name__) for d in (s,c) for k,v in d.items() if 'bank' in k]);
PY
sed -n '1,140p' _verified_downstream.py
```
- **Container Exit Code**: `0` (`381 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tatus)
        if len(free):
            nf = len(free)
            kkt = np.zeros((nf+1, nf+1))
            kkt[:nf, :nf] = hessian[np.ix_(free, free)]
            kkt[:nf, nf] = -1
            kkt[nf, :nf] = 1
            rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed],
                        budget - w[fixed].sum()]
            answer = np.linalg.solve(kkt, rhs)
            target, nu = answer[:-1], float(answer[-1])
            violation = np.flatnonzero((target < -1e-13) | (target > cap+1e-13))
            if len(violation):
                delta = target - w[free]
                bounds = np.where(target[violation] < 0, 0., cap)
                alpha = (bounds - w[free[violation]]) / delta[violation]
                i = int(np.argmin(alpha))
                w[free] += max(0., alpha[i]) * delta
                j = free[violation[i]]
                w[j] = bounds[i]
                status[j] = -1 if bounds[i] == 0 else 1
                continue
            w[free] = np.clip(target, 0., cap)
        else:
            g = hessian @ w + linear
            nu = float(g[status == 1].max())
        g = hessian @ w + linear
        violation = status * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= 1e-12:
            return w, nu
        status[j] = 0
    raise RuntimeError('Boxed source fit did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _design(study)
    y = _calibrate(study)
    z = study['Z_nuisance']
    projection = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    design -= z @ (projection @ design)
    y -= z @ (projection @ y)
    cap, budget, ridge = study['u_box'], study['total_budget'], study['source_ridge']
    w, nu = _box_qp(design.T @ design + ridge * np.eye(design.shape[1]), -design.T @ y, cap, budget)
    residual = design @ w - y
    g = design.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = float(objective - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., g - nu).tolist(),
                                'lam_upper': np.maximum(0., nu - g).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': lower}}
```

### Turn 4 — `execute_in_remote_docker` (`26497 ms` LLM)
- **Rationale**: Compare robust held-out residuals and fitted coefficient patterns. This will show whether the observed curvature is captured by coordinate powers and pairwise coupling or requires a different saturation law.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import itertools,pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study'];c=s['sensor_calibration_sweeps'];x,y=map(np.asarray,(c['ref_rosette'],c['raw_rosette']))
def des(x,terms):
 return np.stack([np.prod(x[:,t],axis=1) for t in terms],axis=1)
def fit(a,y):
 w=np.ones(len(y))
 for _ in range(25):
  b=np.linalg.lstsq(a*w[:,None],y*w[:,None],rcond=None)[0];r=np.linalg.norm(a@b-y,axis=1);h=max(np.median(r),1e-5);w=1/np.sqrt(1+(r/(2*h))**2)
 return b
allterms=lambda d:[t for k in range(d+1) for t in itertools.combinations_with_replacement(range(4),k)]
models={'affine':allterms(1),'quadratic+coordinate cubic':allterms(2)+[(i,)*3 for i in range(4)],'full cubic':allterms(3),'cubic+coordinate fourth/fifth':allterms(3)+[(i,)*j for j in (4,5) for i in range(4)]}
for name,terms in models.items():
 a=des(x,terms);b=fit(a,y);r=np.linalg.norm(a@b-y,axis=1);ro=np.zeros(len(y));
 for k in range(5):
  q=np.arange(len(y))%5==k;ro[q]=np.linalg.norm(a[q]@fit(a[~q],y[~q])-y[q],axis=1)
 clean=r<1; print(name,'train/cv',np.quantile(r,[.5,.8,.9,1]),np.quantile(ro[clean],[.5,.8,.95,1]));
 if name!='affine':
  for i in range(4): print(i,[(t,round(v,6)) for t,v in zip(terms,b[:,i]) if abs(v)>0.0015]);
print('axial',np.round(np.c_[x[:16],y[:16]],6));print('bank:',{k:(np.shape(v),np.asarray(v).ravel()[:8]) for k,v in c['single_bank_sweeps'].items()})
PY
```
- **Container Exit Code**: `0` (`464 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
89997), ((0,), 0.2656), ((2,), -0.132796), ((3,), 0.83), ((0, 3), 0.182602), ((3, 3, 3), 0.149401)]
cubic+coordinate fourth/fifth train/cv [9.62181062e-06 1.76852224e-05 2.47197691e-05 2.59632718e+01] [3.34292167e-05 5.45181378e-05 1.02399873e-04 1.53647089e-04]
0 [((), 0.08), ((0,), 1.290001), ((1,), 0.412793), ((3,), -0.206405), ((0, 1), 0.2838), ((0, 0, 0), 0.232191)]
1 [((), -0.069996), ((0,), -0.124812), ((1,), 0.780001), ((2,), 0.249604), ((1, 2), 0.171598), ((1, 1, 1), 0.140393)]
2 [((), 0.060002), ((1,), -0.198408), ((2,), 1.239986), ((3,), 0.396797), ((2, 3), 0.272805), ((2, 2, 2), 0.223211)]
3 [((), -0.09), ((0,), 0.265611), ((2,), -0.1328), ((3,), 0.830007), ((0, 3), 0.182601), ((3, 3, 3), 0.149389)]
axial [[ -1.2        0.         0.         0.        -1.869251   0.079764
    0.060013  -0.408713]
 [ -0.4        0.         0.         0.        -0.450849  -0.020069
    0.059988  -0.196245]
 [  0.5        0.         0.         0.         0.754016  -0.132405
    0.060007   0.042813]
 [  1.3        0.         0.         0.         2.267133  -0.232231
    0.060003   0.255275]
 [  0.        -1.2        0.         0.        -0.415353  -1.248612
    0.298072  -0.090012]
 [  0.        -0.4        0.         0.        -0.085127  -0.390973
    0.139368  -0.090016]
 [  0.         0.5        0.         0.         0.286393   0.337545
   -0.039212  -0.089995]
 [  0.         1.3        0.         0.         0.616651   1.252456
   -0.197927  -0.089987]
 [  0.         0.        -1.2        0.         0.080005  -0.369511
   -1.813671   0.069352]
 [  0.         0.        -0.4        0.         0.08      -0.169844
   -0.450283  -0.036886]
 [  0.         0.         0.5        0.         0.080001   0.054817
    0.707909  -0.156381]
 [  0.         0.         1.3        0.         0.079999   0.254453
    2.162365  -0.26263 ]
 [  0.         0.         0.        -1.2        0.327685  -0.070012
   -0.416154  -1.34417 ]
 [  0.         0.         0.        -0.4       13.662575 -12.869994
   12.301293 -13.631555]
 [  0.         0.         0.         0.5       13.476801 -12.870018
   12.658412 -12.856341]
 [  0.         0.         0.         1.3       13.311694 -12.869997
   12.975838 -11.882765]]
bank: {'bank_id': ((24,), array([0, 0, 0, 0, 0, 0, 1, 1])), 'ref_signal': ((24,), array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6])), 'raw_readout': ((24,), array([-1.64806558, -0.74417415, -0.04923069,  0.61086143,  1.41027582,
       17.0231714 , -1.11485233, -0.56831766]))}
```

### Turn 5 — `execute_in_remote_docker` (`42500 ms` LLM)
- **Rationale**: Implement the better held-out forward model and invert it jointly. The public schema check will expose integration or certificate failures before testing inverse accuracy on held-out sweeps and the optional pulse path.
- **Alternatives Weighed**: `Full cubic or higher-degree polynomial` (rejected: Both increased worst held-out error without revealing additional coefficients above the noise scale.); `Direct inverse regression on raw sweep rows` (rejected: EMI would contaminate the predictors, and the inverse of the observed forward polynomial is not itself a low-degree polynomial.)
- **ATMS NOGOOD DECLARED**: Refuted `Cross-channel affine calibration is sufficient.` | Witness: `Worst held-out residual on clean sweep rows fell from 0.974 to 4.74e-5 after adding quadratic and coordinate-cubic terms.` | Blanket Diagnosis: `An affine fit conflates systematic gain curvature and pairwise coupling with noise.` | Sublated Pivot: `Retain the full linear coupling and fit quadratic and coordinate-cubic terms with rowwise outlier rejection.`

```bash
cd /holdout_task && cat > upstream_transducer.py <<'PY'
"""Robust forward calibration and inversion of the coupled four-channel rosette."""

import numpy as np


_I, _J = np.triu_indices(4)


def _features(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit(ref, raw):
    a = _features(ref)
    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-12)
    a = a / scale
    weight = np.ones(len(a))
    for _ in range(25):
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(a @ coef - raw, axis=1)
        noise = max(np.median(error), 1e-10)
        weight = 1 / np.hypot(1, error / (2 * noise))

    # Reject EMI by row, then remove only nonlinear terms below the noise scale.
    clean = error <= 6 * noise
    if np.count_nonzero(clean) >= a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
        for j in range(4):
            keep = np.abs(coef[:, j]) > 4 * noise
            keep[:5] = True
            coef[~keep, j] = 0
            coef[keep, j] = np.linalg.lstsq(a[clean][:, keep], raw[clean, j], rcond=None)[0]
    return coef / scale[:, None]


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    linear = coef[1:5].T
    x = np.linalg.solve(linear, (target - coef[0]).T).T.copy()
    quad, cubic = coef[5:15], coef[15:19]
    for _ in range(24):
        residual = _features(x) @ coef - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-12 * (1 + np.linalg.norm(target, axis=1))
        if not np.any(active):
            break
        xa, ra = x[active], residual[active]
        jac = np.broadcast_to(linear, (len(xa), 4, 4)).copy()
        for i, j, q in zip(_I, _J, quad):
            jac[:, :, i] += xa[:, j, None] * q
            jac[:, :, j] += xa[:, i, None] * q
        jac += cubic.T[None, :, :] * (3 * xa[:, None, :]**2)
        step = np.linalg.solve(jac, ra[..., None])[..., 0]
        alpha = np.ones((len(xa), 1))
        for _ in range(14):
            trial = xa - alpha * step
            worse = np.linalg.norm(_features(trial) @ coef - target[active], axis=1) >= error[active]
            if not np.any(worse):
                break
            alpha[worse] *= 0.5
        trial[worse] = xa[worse]
        x[active] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return an identity-calibrated study without modifying the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    out = dict(study)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in pulses
        ]
    canonical = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    if 'single_bank_sweeps' in sweeps:
        bank = sweeps['single_bank_sweeps']
        canonical['single_bank_sweeps'] = dict(bank, raw_readout=bank['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = canonical
    return out
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`435 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc980690c4a3aad5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc980690 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`38894 ms` LLM)
- **Rationale**: The schema gate does not measure inverse accuracy or exercise pulses. Hold out real sweep rows, including whole single-axis groups, to detect calibration that depends on a particular sweep. Check multidimensional pulse inversion, input preservation, and the composed certificate in an isolated Python process; a failure would identify whether to revise the fit, inverse, or field handling.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch
s = pickle.load(open('public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x = c['ref_rosette']; y = c['raw_rosette']
# Gross spikes are visibly separated in the public raw readings (>12 vs normal sweep range).
clean = np.max(np.abs(y), axis=1) < 8
axis = np.count_nonzero(x, axis=1) == 1
print('rows: clean', clean.sum(), 'EMI', (~clean).sum())
for name, labels, groups in [('interleaved', np.arange(len(x)) % 5, range(5)), ('axis-group', np.where(axis, np.argmax(np.abs(x), axis=1), -1), range(4))]:
    worst_fwd = worst_inv = 0.
    for g in groups:
        hold = labels == g; val = hold & clean
        coef = up._fit(x[~hold], y[~hold]); pred = up._features(x[val]) @ coef
        worst_fwd = max(worst_fwd, np.linalg.norm(pred - y[val], axis=1).max())
        worst_inv = max(worst_inv, np.linalg.norm(up._invert(y[val], coef) - x[val], axis=1).max())
    print(name, 'worst clean forward/inverse L2', worst_fwd, worst_inv)
    assert worst_inv < 2e-4
coef = up._fit(x, y)
out = up.calibrate_upstream(s)
obs = out['y_obs_raw'].reshape(-1, 4)
obs_error = np.max(np.abs(up._features(obs) @ coef - s['y_obs_raw'].reshape(-1, 4)))
print('public obs forward replay max', obs_error)
assert obs_error < 1e-10 and out['y_obs_raw'].shape == s['y_obs_raw'].shape
assert np.array_equal(out['sensor_calibration_sweeps']['raw_rosette'], x)
assert not np.shares_memory(out['sensor_calibration_sweeps']['raw_rosette'], x)
assert not np.array_equal(c['raw_rosette'], x)
b = c['single_bank_sweeps']; b_new = out['sensor_calibration_sweeps']['single_bank_sweeps']
assert np.array_equal(b_new['raw_readout'], b['ref_signal']) and not np.shares_memory(b_new['raw_readout'], b['ref_signal'])
assert 'reference_tracer_pulses' not in out
assert up.calibrate_upstream(dict(s, reference_tracer_pulses=None))['reference_tracer_pulses'] is None
rng = np.random.default_rng(1234)
truth = rng.uniform(-1.4, 1.7, (6, 16, 4))
raw = (up._features(truth.reshape(-1, 4)) @ coef).reshape(truth.shape)
pulse_out = up.calibrate_upstream(dict(s, reference_tracer_pulses=[{'y_pulse_raw': raw, 'tag': 7}, {'y_pulse_raw': None}]))['reference_tracer_pulses']
pulse_error = np.max(np.abs(pulse_out[0]['y_pulse_raw'] - truth))
print('multidimensional pulse round trip max', pulse_error)
assert pulse_error < 1e-9 and pulse_out[0]['tag'] == 7 and pulse_out[1]['y_pulse_raw'] is None
r = solve_batch(s); w = r['weights']; cert = r['kkt_certificate']
gap = cert['primal_objective'] - cert['dual_lower_bound']
print('weights box/budget', w.min(), w.max(), abs(w.sum() - s['total_budget']), 'duality gap', gap)
assert abs(gap) < 1e-9 and abs(w.sum() - s['total_budget']) < 1e-10 and w.min() >= -1e-10 and w.max() <= s['u_box'] + 1e-10
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`493 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rows: clean 77 EMI 3
interleaved worst clean forward/inverse L2 3.399884713478804e-05 3.2647681932435665e-05
axis-group worst clean forward/inverse L2 3.442069618893598e-05 3.31673447121978e-05
public obs forward replay max 7.532585666325531e-13
multidimensional pulse round trip max 6.59694521232268e-13
weights box/budget 0.07961429547548571 0.42 2.220446049250313e-16 duality gap 1.700029006457271e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc980690c4a3aad5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc980690 (directory).
```

### Turn 7 — `submit_completed_task` (`17931 ms` LLM)
- **Rationale**: The exact submitted bundle passed isolated Python verification. Real sweep holdouts confirmed inverse accuracy, and the pulse, identity-normalization, and downstream certificate checks passed.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:199', '/holdout_task/upstream_transducer.py': 'FILE:3340'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust rosette calibration and verified the composed batch solver. Public schema validation passed, and the worst held-out inverse error was 3.32×10⁻⁵ (L2). Optional pulse inversion and the downstream box and budget constraints also passed.`

### Turn 8 — `execute_in_remote_docker` (`34880 ms` LLM)
- **Rationale**: Test whether rational saturation, pair-difference terms, or three-way interactions reduce errors on real sweep extremes. A consistent gain would justify revising the model; a small Jacobian singular value or a stalled replay on actual observations would instead call for revising the inverse.
- **Alternatives Weighed**: `Apply relay, drift-operator, or classification audits` (rejected: The deliverable fits a static continuous sensor transfer map, not a relay, a drift operator, or a classifier.)

```bash
cd /holdout_task && cat > /tmp/rosette_margin_audit.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
clean = np.max(np.abs(y), axis=1) < 8  # Public EMI rows are visibly separated.
axial = np.count_nonzero(x, axis=1) == 1
edge = np.max(np.abs(x), axis=1) >= np.quantile(np.max(np.abs(x), axis=1), .8)
print('study regime fields:', [k for k in s if any(t in k for t in ('batch', 'relay', 'meter', 'cohort', 'bank'))])
print('diagnostics present:', item['campaign_diagnostics'] is not None)

def rms(a):
    return np.sqrt(np.mean(a**2))

# Compare prespecified nonlinear extensions on the same held-out rows. The
# baseline support and training outlier mask are selected inside each fold.
for name, hold in [('all axial', axial), ('outer radius', edge), ('positive/negative difference tail', np.abs(x[:, 0]-x[:, 1]) >= np.quantile(np.abs(x[:, 0]-x[:, 1]), .8))]:
    train, val = ~hold, hold & clean
    coef = up._fit(x[train], y[train])
    a = up._features(x)
    err = np.linalg.norm(a @ coef - y, axis=1)
    use = train & (err <= 6 * np.median(err[train]))
    inv = np.linalg.norm(up._invert(y[val], coef) - x[val], axis=1).max()
    scores = {}
    for kind in ('coordinate rational', 'pair difference rational', 'three-way'):
        pred = np.empty_like(y[val])
        for j in range(4):
            base = a[:, coef[:, j] != 0]
            q = x[:, j] - x[:, (j+1)%4]
            if kind == 'coordinate rational':
                z = x[:, j]**3 / (1 + .5*x[:, j]**2)
            elif kind == 'pair difference rational':
                z = q / (1 + .5*q**2)
            else:
                z = x[:, j] * x[:, (j+1)%4] * x[:, (j+2)%4]
            aug = np.column_stack((base, z))
            pred[:, j] = aug[val] @ np.linalg.lstsq(aug[use], y[use, j], rcond=None)[0]
        scores[kind] = rms(pred - y[val])
    print(name, 'n:', val.sum(), 'baseline forward RMS:', rms((a @ coef - y)[val]), 'inverse worst L2:', inv, 'extension forward RMS:', scores)
    assert inv < 2e-4

# Detect actual observation rows with a poorly conditioned or stalled inverse.
coef = up._fit(x, y)
raw = np.asarray(s['y_obs_raw']).reshape(-1, 4)
z = up._invert(raw, coef)
h = 1e-5
jac = np.stack([(up._features(z+d) - up._features(z-d)) @ coef / (2*h) for d in h*np.eye(4)], axis=-1)
min_sigma = np.linalg.svd(jac, compute_uv=False)[:, -1].min()
replay = np.abs(up._features(z) @ coef - raw).max()
print('observation inverse: min Jacobian singular value:', min_sigma, 'max replay residual:', replay)
assert min_sigma > .1 and replay < 1e-10
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rosette_margin_audit.py && OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`507 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study regime fields: ['row_bank_id']
diagnostics present: False
all axial n: 13 baseline forward RMS: 1.023412254628798e-05 inverse worst L2: 3.3354495389695526e-05 extension forward RMS: {'coordinate rational': 1.0617419146433402e-05, 'pair difference rational': 1.0267806403525854e-05, 'three-way': 1.0282899293430326e-05}
outer radius n: 16 baseline forward RMS: 1.1364978314177128e-05 inverse worst L2: 2.2607953711284685e-05 extension forward RMS: {'coordinate rational': 1.2201335004971044e-05, 'pair difference rational': 1.1101900936866205e-05, 'three-way': 1.1678701252840627e-05}
positive/negative difference tail n: 16 baseline forward RMS: 8.72047342875428e-06 inverse worst L2: 1.93731713361347e-05 extension forward RMS: {'coordinate rational': 8.646114507770794e-06, 'pair difference rational': 8.64747047956843e-06, 'three-way': 1.0242752981831329e-05}
observation inverse: min Jacobian singular value: 0.7219802108927899 max replay residual: 7.532585666325531e-13
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc980690c4a3aad5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc980690 (directory).
```

### Turn 9 — `submit_completed_task` (`16950 ms` LLM)
- **Rationale**: The unchanged bundle passed isolated Python verification, including pulse inversion, identity normalization, and the downstream certificate. The additional real-data audit found a minimum observation Jacobian singular value of 0.722 and a maximum forward replay residual of 7.53e-13, so neither the model nor the inverse needs revision.
- **Alternatives Weighed**: `Add rational saturation, pair-difference, or three-way terms` (rejected: None consistently improved held-out forward error across the axial, outer-radius, and pair-difference slices.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:199', '/holdout_task/upstream_transducer.py': 'FILE:3340'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration while preserving the verified downstream solver. Public schema validation passed. The worst held-out inverse error was 3.34×10⁻⁵ (L2), and rational and three-way extensions did not consistently improve accuracy.`

### Turn 10 — `execute_in_remote_docker` (`26222 ms` LLM)
- **Rationale**: Contiguous holdouts test a failure that interleaved and geometric splits could miss: dependence on a local sweep segment or on how clustered EMI rows are distributed between training and validation. If a block exposes a material error, the paired rational comparisons help distinguish missing response terms from a fragile robust fit.
- **Alternatives Weighed**: `Test batch classification, relay modes, or stochastic-control drift` (rejected: The upstream task is continuous calibration on paired reference and raw measurements. Row bank IDs identify sensor channels, not classification cohorts, and the upstream interface has no relay state or drift operator.)

```bash
cd /holdout_task && cat > /tmp/rosette_block_audit.py <<'PY'
import pickle
import runpy
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
a = up._features(x)
clean = np.max(np.abs(y), axis=1) < 8  # Gross spikes are separated in the public measurements.
print('EMI row indices:', np.flatnonzero(~clean).tolist())
errors = []
for block in np.array_split(np.arange(len(x)), 5):
    train = np.ones(len(x), dtype=bool)
    train[block] = False
    val = block[clean[block]]
    coef = up._fit(x[train], y[train])
    pred = a @ coef
    base = np.linalg.norm(pred[val]-y[val], axis=1)
    inv = np.linalg.norm(up._invert(y[val], coef)-x[val], axis=1)
    errors.extend(inv)
    fit_error = np.linalg.norm(pred-y, axis=1)
    use = train & (fit_error <= 6*np.median(fit_error[train]))
    variants = []
    for sign in (1, -1):
        for alpha in (.1, .5, 2.):
            p = np.empty_like(y[val])
            for j in range(4):
                v = x[:, j] + sign*x[:, (j+1)%4]
                d = np.column_stack((a[:, coef[:, j] != 0], v**3/(1+alpha*v**2)))
                p[:, j] = d[val] @ np.linalg.lstsq(d[use], y[use, j], rcond=None)[0]
            variants.append(np.linalg.norm(p-y[val], axis=1).max())
    print('held block', int(block[0]), int(block[-1]), 'clean/EMI', len(val), len(block)-len(val), 'inverse max', inv.max(), 'forward max', base.max(), 'best augmented forward max', min(variants))
print('blocked worst inverse L2:', max(errors))
assert max(errors) < 2e-4
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rosette_block_audit.py
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
EMI row indices: [13, 14, 15]
held block 0 15 clean/EMI 13 3 inverse max 3.3354495389695526e-05 forward max 3.4610255861657066e-05 best augmented forward max 3.28592224937023e-05
held block 16 31 clean/EMI 16 0 inverse max 2.368950572115571e-05 forward max 2.6559427482316565e-05 best augmented forward max 2.587517695145286e-05
held block 32 47 clean/EMI 16 0 inverse max 2.2247294748075626e-05 forward max 3.377054430075438e-05 best augmented forward max 3.400395840917735e-05
held block 48 63 clean/EMI 16 0 inverse max 2.9794055373045175e-05 forward max 3.192139928870245e-05 best augmented forward max 3.0940370234619904e-05
held block 64 79 clean/EMI 16 0 inverse max 2.4472239104091692e-05 forward max 3.363514201764174e-05 best augmented forward max 3.254884429796505e-05
blocked worst inverse L2: 3.3354495389695526e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bc980690c4a3aad5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bc980690 (directory).
```

### Turn 11 — `submit_completed_task` (`15926 ms` LLM)
- **Rationale**: The current bundle passed isolated Python verification. Contiguous sweep holdouts, including the block containing all three EMI spikes, did not worsen the previous worst-slice result. Actual observation inversion remained well-conditioned, with a minimum Jacobian singular value of 0.722.
- **Alternatives Weighed**: `Add rational saturation or higher-order cross-channel terms` (rejected: The extensions did not consistently improve held-out errors on real sweep blocks or at the edges of the calibration range.)
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration and verified the composed batch solver. Public schema validation passed. The worst held-out inverse error was 3.34×10⁻⁵ (L2). Optional pulse inversion and the downstream constraint checks also passed.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
