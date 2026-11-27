# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `368,709` (`355,874` / `83,328` / `12,835`) |
| **Total Episode Wall Time (s)** | `417.56s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16933 ms` LLM)
- **Rationale**: Inspect the public schema and available references before fitting. This determines which fields supply training targets and panel statistics, and whether the input layout requires adapting the solver.

```bash
cd /holdout_task && python3 validate_public_schema.py && cat validate_public_schema.py && cat public_asset_index.json && ls -lh /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_03.pkl','rb') as f:
    d=pickle.load(f)
for k,v in d.items():
    if isinstance(v,dict):
        print(k, {a:(type(b).__name__,getattr(b,'shape',None)) for a,b in v.items()})
    elif isinstance(v,np.ndarray):
        print(k,v.shape,v.dtype,'sample',v.ravel()[:5])
    elif isinstance(v,(list,tuple)):
        print(k,len(v),repr(v[:1])[:700])
    else:
        print(k,repr(v)[:300])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation' 2>&1
```
- **Container Exit Code**: `0` (`683 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 6, in <module>
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 991, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1128, in get_code
  File "<frozen importlib._bootstrap_external>", line 1186, in get_data
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/alpha_m97_c02_batch_03_solver.py'
```

### Turn 2 — `execute_in_remote_docker` (`19860 ms` LLM)
- **Rationale**: Inspect field shapes and prior preprocessing to detect mismatched splits or panel conventions before writing the solver. Read the validator now and run it after the artifact exists.
- **Alternatives Weighed**: `Run the public validator before inspecting inputs` (rejected: It imports the solver, which does not exist yet, so it cannot establish the input schema.)

```bash
cd /holdout_task
cat validate_public_schema.py public_asset_index.json
ls -lh /tmp/hegel_continual_archive
sed -n '1,250p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py
python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c02_batch_03.pkl','rb') as f: d=pickle.load(f)
def describe(x, level=0):
    if level>3: return type(x).__name__
    if isinstance(x,dict): return {k:describe(v,level+1) for k,v in x.items()}
    if isinstance(x,np.ndarray): return (x.shape,str(x.dtype),x.ravel()[:5].tolist())
    if isinstance(x,(list,tuple)): return (len(x),[describe(v,level+1) for v in x[:2]])
    return x
for k,v in d.items(): print(k,repr(describe(v))[:2000])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable projection' 2>&1 | head -15
```
- **Container Exit Code**: `0` (`590 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s):
        matrix, diff = design(params[:2], derivatives=True)
        return np.column_stack((diff * params[2:4], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0),
                 (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        weights = np.linalg.lstsq(design(log_taus), target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', f_scale=0.025,
                            x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x.copy()
    if params[0] > params[1]:
        params[[0, 1, 2, 3]] = params[[1, 0, 3, 2]]
    return params[:2], params[2:]


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_taus, weights = _fit(past[:n], hold[:n], features[:n], age, dt, times, curves, rates)
    curve, rate = _basis(log_taus, past[n:], hold[n:], features[n:], age, dt, times)
    return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 3 — `execute_in_remote_docker` (`20770 ms` LLM)
- **Rationale**: Unwrap the public list and inspect measurement coverage, calibration residuals, and the prior panel code. Coverage and residuals determine when to use the archive; panel and target shapes determine the regression design.

```bash
cd /holdout_task && python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))
print('public cases:',len(p),'wrapper keys:',list(p[0]))
d=p[0]['dataset']
for k,v in d.items():
 print(k,{a:(np.shape(b),str(np.asarray(b).dtype)) for a,b in v.items()} if isinstance(v,dict) else (np.shape(v),repr(v)[:280]))
c=d['calibration_load_cells']
for lab in np.unique(c['lab_id']):
 use=c['lab_id']==lab;x=c['cmd_stress'][use];y=c['true_stress'][use];a,b=np.polyfit(x,y,1);print('calibration',lab,len(x),a,b,np.round(y-a*x-b,5))
for name in ('train','test'):
 for k,v in d[name].items():
  a=np.asarray(v)
  if a.dtype.kind=='f': print(name,k,'nonfinite',np.count_nonzero(~np.isfinite(a)),'quantiles',np.round(np.nanpercentile(a,[0,25,50,75,100]),5))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'));print('archive layout:',repr(h)[:2300])
PY
sed -n '1,105p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel 2>&1 | head -22
```
- **Container Exit Code**: `0` (`670 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
for channel in np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_taus)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # A history step contributes its uptake times its decay over subsequent steps.
    memory = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap/tau, features))
    if not derivatives:
        return curve, rate
    dz = past @ (a*memory - q*ea*eq)
{
  "ok": true,
  "target": "autogenous_control_panel_deconfounding.align_control_panel",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_control_panel_deconfounding",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "aa502f6d83a30b6babf8dcdff2012cefc7cf8ce998e199d66c4221b48051b07f",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "observations_per_contributing_control",
          "op": ">=",
          "value": 2
        },
        {
          "field": "usable_control_features_per_center",
          "op": ">=",
          "value": 1
```

### Turn 4 — `execute_in_remote_docker` (`49102 ms` LLM)
- **Rationale**: Build the exact jammed-slip basis and its zero-time derivative, reuse the prior alignment, and fit both targets robustly because the observed strain contains extreme values. The isolated schema run will catch load or output failures before predictive stress testing.
- **Alternatives Weighed**: `Sort fitted relaxation times and exchange their Prony weights` (rejected: The cross term uses branch zero's exponential, so swapping branches does not preserve the model.); `Replace the prior panel preprocessing with the discovered control-panel primitive` (rejected: The task requires both median reference scale and cell-balanced imputation; the archived preprocessing implements those operations directly.)

```bash
set -e
cd /holdout_task
sed '/^def _basis(/,$d' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py > alpha_m97_c02_batch_03_solver.py
cat >> alpha_m97_c02_batch_03_solver.py <<'PY'
def _basis(log_tau, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_tau)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    uptake = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ uptake
    step = -np.expm1(-r)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    phi = times/(1 + 0.85*times)
    delta = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            slip[:, None, None] * phi[None, :, None],
                            delta[:, None, None] * step[None, :, :1]), axis=2)
    rate = np.column_stack((gap/tau, features, slip, delta/tau[0]))
    if not derivatives:
        return curve, rate
    dg = -(past @ (a*uptake - q*ea*eq))
    ds = -r*er
    dc = dg[:, None, :] * step + gap[:, None, :] * ds
    dr = (dg - gap)/tau
    xc = np.stack((dg[:, :1]*step[:, 0] + delta[:, None]*ds[:, 0],
                   -dg[:, 1:]*step[:, 0]), axis=-1)
    xr = np.column_stack((dg[:, 0]-delta, -dg[:, 1]))/tau[0]
    return curve, rate, np.vstack((dc.reshape(-1, 2), dr)), np.vstack((xc.reshape(-1, 2), xr))


def _fit(past, hold, features, age, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]
    observed = np.isfinite(target)
    target = target[observed]
    args = past, hold, features, age, dt, times

    def design(log_tau, derivatives=False):
        curve, rate, *jac = _basis(log_tau, *args, derivatives=derivatives)
        mat = np.vstack((curve.reshape(-1, 9), rate))[observed]
        if derivatives:
            return mat, jac[0][observed], jac[1][observed]
        return mat

    def residual(params):
        return np.einsum('ij,j->i', design(params[:2]), params[2:]) - target

    def jacobian(params):
        mat, branch, cross = design(params[:2], derivatives=True)
        return np.column_stack((branch*params[2:4] + cross*params[-1], mat))

    bounds = (np.r_[np.log([0.01, 0.01]), np.full(9, -np.inf)],
              np.r_[np.log([100., 100.]), np.full(9, np.inf)])
    best = None
    # The cross term acts on branch zero, so exchanging branches is not a symmetry.
    for short, long in ((0.45, 2.4), (0.18, 1.2), (0.20, 6.0), (1., 6.), (0.45, 15.)):
        for pair in ((short, long), (long, short)):
            log_tau = np.log(pair)
            matrix = design(log_tau)
            weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
            for _ in range(3):
                e = np.einsum('ij,j->i', matrix, weights) - target
                w = 1/np.sqrt(1 + (e/0.025)**2)
                weights = np.linalg.lstsq(matrix*w[:, None], target*w, rcond=None)[0]
            fit = least_squares(residual, np.r_[log_tau, weights], jac=jacobian,
                                bounds=bounds, loss='cauchy', f_scale=0.025,
                                x_scale='jac', max_nfev=130, tr_solver='lsmr' if False else 'exact')
            if best is None or fit.cost < best.cost:
                best = fit
    return best.x[:2], best.x[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], age, dt, times, curves, rates)
    curve, rate = _basis(log_tau, past[n:], hold[n:], features[n:], age, dt, times)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
sed -i "s/, tr_solver='lsmr' if False else 'exact'//" alpha_m97_c02_batch_03_solver.py
sed -n '1,43p' alpha_m97_c02_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,time,numpy as np
from alpha_m97_c02_batch_03_solver import solve_alpha_m97_c02_batch_03
item=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]
t=time.perf_counter();out=solve_alpha_m97_c02_batch_03(item['dataset']);print('seconds',time.perf_counter()-t)
for k,v in out.items(): print(k,v if v.size<12 else (v.min(),v.max()))
g=item['gt']; print('public verification fields:',{k:np.shape(v) for k,v in g.items()} if isinstance(g,dict) else type(g))
PY
```
- **Container Exit Code**: `0` (`1055 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
p model for an amorphous melt."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for table in reversed(history):
        take = np.asarray(table['lab_id']) == channel
        x = np.asarray(table['cmd_stress'])[take]
        if np.unique(x).size >= 2:
            return np.polyfit(x, np.asarray(table['true_stress'])[take], 1)
    return None


def _calibrate(table):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        i, j = np.triu_indices(x.size, k=1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        spike = False
        if i.size:
            # Choose the line with the most near-collinear points before
            # testing for EMI. A spike can skew the median slope of four rows.
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            offsets = y[i] - slopes*x[i]
            residuals = np.abs(y - slopes[:, None]*x - offsets[:, None])
            near = residuals <= 0.1
            support = near.sum(axis=1)
            candidates = np.flatnonzero(support == support.max())
            best = candidates[np.argmin(np.sum(np.where(near[candidates], residuals[candidates], 0.), axis=1))]
{"schema_valid": true, "errors": []}
seconds 0.1346176080405712
estimated_taus [0.57528636 4.24430338]
hereditary_weights [ 0.67855331  0.55437143  0.13967072  0.15648323 -0.12776716  0.11855566
 -0.08850026]
regime_mode_weights [ 0.30921424 -0.21829674]
predicted_strain (-3.184154394631143, 3.09185019248195)
initial_rate_pred (-3.8794390457529313, 3.423108775442667)
public verification fields: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 5 — `execute_in_remote_docker` (`36177 ms` LLM)
- **Rationale**: Reconstruct clean public targets with the reference parameters to distinguish a preprocessing or equation mismatch from fit error. Leave-one-lab-out and clean slice errors test whether calibration transfers across channels, while the independent recurrence and finite-difference checks catch timing or initial-rate mistakes.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c02_batch_03(d)
p,h,x,e=m._prepare(d); tr,te=d['train'],d['test']; n=len(tr['lab_id']); dt,t=d['past_dt'],d['eval_times']; age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(a,'reference',g[b],'error',o[a]-g[b])
print('max stress errors',np.max(abs(p[n:]-g['te_hist_true'])),np.max(abs(h[n:]-g['te_hold_true'])))
curve,rate=m._basis(np.log(g['taus']),g['te_hist_true'],g['te_hold_true'],x[n:],age,dt,t); w=np.r_[g['theta'],g['vp_theta']]; print('reference-parameter reconstruction max errors',np.max(abs(e[n:,None]+curve@w-g['te_clean'])),np.max(abs(rate@w-g['te_true_rate0'])))
def summary(err): return tuple(np.round([np.sqrt(np.mean(err**2)),np.median(abs(err)),np.max(abs(err))],6))
ce=o['predicted_strain']-g['te_clean']; re=o['initial_rate_pred']-g['te_true_rate0']; print('clean pooled RMSE/median AE/max AE',summary(ce),summary(re)); print('clean timewise RMSE',np.round(np.sqrt(np.mean(ce**2,axis=0)),6))
missing=np.any(~np.isfinite(te['coupon_features_raw']),axis=1)
for lab in np.unique(te['lab_id']):
 mask=te['lab_id']==lab; print('clean lab',lab,summary(ce[mask]),summary(re[mask]))
for name,mask in [('missing coupons',missing),('complete coupons',~missing),('negative hold',h[n:]<0),('nonnegative hold',h[n:]>=0)]:
 if mask.any(): print('clean',name,mask.sum(),summary(ce[mask]),summary(re[mask]))
y=tr['strain_curves_obs']-e[:n,None]; r=tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
 a=tr['lab_id']!=lab; b=~a; lt,w=m._fit(p[:n][a],h[:n][a],x[:n][a],age,dt,t,y[a],r[a]); c,v=m._basis(lt,p[:n][b],h[:n][b],x[:n][b],age,dt,t); print('leave-lab-out',lab,'tau',np.round(np.exp(lt),5),'curve/rate median AE and P90 AE',np.round(np.percentile(abs(c@w-y[b]),[50,90]),6),np.round(np.percentile(abs(v@w-r[b]),[50,90]),6))
# The recurrence and derivative checks target time indexing and the branch-zero cross term.
z=np.zeros((len(h),2)); tau=o['estimated_taus']; w=np.r_[o['hereditary_weights'],o['regime_mode_weights']]
for j,step in enumerate(dt):
 k=np.exp(-step/tau); z=k*z+(1-k)*p[:,j,None]
a,b=m._basis(np.log(tau),p,h,x,age,dt,np.array([1e-7])); gap=h[:,None]-z
independent=gap@(w[:2]/tau)+x@w[2:7]+w[7]*np.sign(h)*np.sqrt(abs(h))+w[8]*(gap[:,0]-gap[:,1])/tau[0]
assert np.allclose(b@w,independent,rtol=0,atol=1e-11)
assert np.allclose((a@w)[:,0]/1e-7,independent,rtol=1e-6,atol=1e-7)
PY
```
- **Container Exit Code**: `0` (`1274 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus reference [0.576 4.235] error [-0.00071364  0.00930338]
hereditary_weights reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] error [-0.00144669  0.00437143 -0.00032928 -0.00351677  0.00223284 -0.00144434
  0.00149974]
regime_mode_weights reference [ 0.31 -0.22] error [-0.00078576  0.00170326]
max stress errors 0.00010163077591229808 8.012874887941557e-05
reference-parameter reconstruction max errors 0.13732734558146542 0.14313734866375816
clean pooled RMSE/median AE/max AE (0.016931, 0.003112, 0.131572) (0.02188, 0.00466, 0.13409)
clean timewise RMSE [0.005233 0.010763 0.013935 0.015985 0.01742  0.018482 0.019302 0.019955
 0.020487 0.020931]
clean lab 0 (0.008709, 0.003694, 0.024344) (0.011486, 0.006794, 0.027997)
clean lab 1 (0.00297, 0.002891, 0.005031) (0.0043, 0.004517, 0.007131)
clean lab 2 (0.017749, 0.001542, 0.089648) (0.023107, 0.002306, 0.094566)
clean lab 3 (0.025155, 0.007914, 0.131572) (0.03231, 0.011291, 0.13409)
clean missing coupons 3 (0.074387, 0.067348, 0.131572) (0.09513, 0.094566, 0.13409)
clean complete coupons 69 (0.007651, 0.002927, 0.035618) (0.010299, 0.004517, 0.039582)
clean negative hold 39 (0.014596, 0.003072, 0.089648) (0.019137, 0.004727, 0.094566)
clean nonnegative hold 33 (0.01933, 0.003136, 0.131572) (0.024732, 0.004594, 0.13409)
leave-lab-out 0 tau [0.57449 4.23228] curve/rate median AE and P90 AE [0.013077 0.083021] [0.017754 0.073886]
leave-lab-out 1 tau [0.57417 4.25885] curve/rate median AE and P90 AE [0.013852 0.047859] [0.011533 0.042156]
leave-lab-out 2 tau [0.57451 4.25503] curve/rate median AE and P90 AE [0.01524  0.101265] [0.018232 0.043162]
leave-lab-out 3 tau [0.57779 4.2272 ] curve/rate median AE and P90 AE [0.01605  0.119609] [0.021336 0.053788]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 6 — `execute_in_remote_docker` (`35677 ms` LLM)
- **Rationale**: Use public clean targets only to diagnose whether reconstruction errors follow the terminal drift term, then distinguish reference scaling from missing-value fill. The single-spike sweep tests whether the calibration branch preserves a clean in-batch fit; a failure would require correcting spike rejection before submission.
- **Alternatives Weighed**: `Retune the relaxation fit immediately` (rejected: The largest mismatch persists with the reference parameters and physical stresses, so preprocessing should be isolated first.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); p,h,x,e=m._prepare(d)
lab,group,raw,ref=[np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')]; dt,t=d['past_dt'],d['eval_times']; age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; tau=np.log(g['taus']); w=np.r_[g['theta'],g['vp_theta']]
C,R=m._basis(tau,g['te_hist_true'],g['te_hold_true'],x[n:],age,dt,t); phi=t/(1+.85*t); dr=R@w-g['te_true_rate0']; dc=e[n:,None]+C@w-g['te_clean']; print('reference mismatch explained by drift, max error',np.max(abs(dc-dr[:,None]*phi)))
complete=np.all(np.isfinite(raw),axis=1); actual_x=np.zeros_like(raw)
for l in np.unique(lab):
 a=lab==l; panel=ref[a]; center=np.median(np.nanmean(panel,axis=0)); scale=np.median(np.nanstd(panel,axis=0)); actual_x[a]=(raw[a]-center)/scale
 b=a[n:]&complete[n:]; fitted=np.linalg.lstsq(np.c_[np.ones(b.sum()),raw[n:][b]@w[3:7]],(x[n:]@w[2:7]-dr)[b],rcond=None)[0]; print('lab',l,'complete drift RMSE',np.sqrt(np.mean(dr[b]**2)),'current center/scale',center,scale,'reference-implied', (w[2]-fitted[0])/(fitted[1]*w[3:7].sum()),1/fitted[1])
cells={(l,k):np.nanmean(actual_x[(lab==l)&(group==k)],axis=0) for l,k in np.unique(np.c_[lab,group],axis=0)}; global_fill=np.nanmean(list(cells.values()),axis=0)
for j in np.flatnonzero(~complete[n:]):
 row=j+n; miss=~np.isfinite(raw[row]); k=(lab[row],group[row]); implied=(x[row,1:]@w[3:7])-dr[j]-np.nansum(actual_x[row]*w[3:7]); print('missing lab/stratum/features',k,np.flatnonzero(miss),'implied contribution',implied,'global/cell/stratum/lab',*[v[miss]@w[3:7][miss] for v in (global_fill,cells[k],np.nanmean([v for a,v in cells.items() if a[1]==k[1]],axis=0),np.nanmean([v for a,v in cells.items() if a[0]==k[0]],axis=0))])
# Check that a single EMI spike cannot pull a fit away from four or more clean in-batch points.
cal=d['calibration_load_cells']; base=m._calibrate(cal); peak=0.
for j in range(len(cal['lab_id'])):
 for sign in (-1,1):
  alt=dict(cal,true_stress=cal['true_stress'].copy()); alt['true_stress'][j]+=sign*3.2; peak=max(peak,float(np.max(abs(m._calibrate(alt)-base))))
print('worst single-spike coefficient change',peak)
PY
```
- **Container Exit Code**: `0` (`628 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
reference mismatch explained by drift, max error 1.502270530195915e-15
lab 0 complete drift RMSE 0.0151465343180477 current center/scale -0.6788826761826414 0.7738673465280527 reference-implied -0.6500000000000006 0.8200000000000005
lab 1 complete drift RMSE 0.007112809969175167 current center/scale 0.550467304807277 1.251270210325619 reference-implied 0.4800000000000004 1.28
lab 2 complete drift RMSE 0.003410506323114642 current center/scale -0.2823519924499841 0.9312784898813439 reference-implied -0.25000000000000133 0.9400000000000004
lab 3 complete drift RMSE 0.00944264935113217 current center/scale 0.8042729667810693 1.468763776524297 reference-implied 0.8000000000000013 1.42
missing lab/stratum/features (2, 2) [1] implied contribution -0.08422229258881543 global/cell/stratum/lab 0.006545746775803121 -0.05761382282168436 -0.05292301881832608 -0.00216363028129669
missing lab/stratum/features (0, 2) [1 3] implied contribution -0.02455849507943897 global/cell/stratum/lab 0.0041955856298422686 -0.09607450309788508 -0.0818374304901922 -0.004444485998838295
missing lab/stratum/features (3, 0) [0] implied contribution -0.14982102359640131 global/cell/stratum/lab -0.006683674932643196 -0.06335982114813471 -0.06573108045796214 -0.017540633260203994
worst single-spike coefficient change 4.20978614275036e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32377 ms` LLM)
- **Rationale**: Inspect the panel layout to distinguish ordinary sampling error from a recoverable reference structure. The result determines whether to retain the prescribed estimator or test a supported preprocessing correction rather than refit the physical mechanism.
- **Alternatives Weighed**: `Hard-code the normalization inferred from clean public targets` (rejected: Those targets are verification data, not solver inputs; the inferred constants need not transfer.)
- **ATMS NOGOOD DECLARED**: Refuted `The observed reference panel and pooled cell-balanced fill exactly recover the coupon drift used to generate clean targets.` | Witness: `With reference weights and true physical stress, clean reconstruction differs by up to 0.13733 in strain and 0.14314 in initial rate; multiplying the rate residual by the saturation basis reproduces the strain residual to 1.6e-15. The discrepancy persists in complete rows.` | Blanket Diagnosis: `The residual is in the aligned coupon drift, not the hereditary or slip terms. Panel location and scale differ from the implied latent normalization, and missing features add a separate error.` | Sublated Pivot: `Inspect whether the supplied panel supports better estimation while retaining the specified alignment and exact jammed-slip equations.`

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; lab,group,raw,ref=[np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')]
for l in np.unique(lab):
 a=lab==l; p=ref[a]; print('lab',l,'panel shape',p.shape,'column means',np.round(np.nanmean(p,axis=0),3),'column std',np.round(np.nanstd(p,axis=0),3)); print('panel row mean/std percentiles',np.round(np.percentile(np.nanmean(p,axis=1),[0,25,50,75,100]),3),np.round(np.percentile(np.nanstd(p,axis=1),[0,25,50,75,100]),3)); print('panel rows 0-1',p[:2]); print('ref-raw pairwise correlations',np.round(np.corrcoef(np.c_[np.nan_to_num(raw[a]-np.nanmean(raw[a],axis=0)),np.nanmean(p,axis=1),np.nanstd(p,axis=1)].T)[:4],2))
for g in np.unique(group):
 a=group==g; values=raw[a].copy()
 for l in np.unique(lab):
  p=ref[lab==l]; b=lab[a]==l; values[b]=(values[b]-np.median(np.nanmean(p,axis=0)))/np.median(np.nanstd(p,axis=0))
 print('stratum',g,'aligned mean/std',np.round(np.nanmean(values,axis=0),3),np.round(np.nanstd(values,axis=0),3),'correlation',np.round(np.corrcoef(values[np.isfinite(values).all(axis=1)].T),2))
index=json.load(open('public_asset_index.json')); print('asset metadata keys',list(index)); print('relevant schema excerpts')
for line in open('public_asset_index.json'):
 if any(word in line.lower() for word in ('coupon','reference','stratum','noise','panel')): print(line.rstrip()[:300])
PY
```
- **Container Exit Code**: `0` (`523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.677 0.919 1.111 0.857 1.123
 0.67  0.655 0.749 0.997 1.149 0.652 0.876 1.05 ] column std [1.56  1.593 1.486 1.317 1.626 1.524 1.46  1.274 1.532 1.512 1.248 1.51
 1.62  1.305 1.329 1.501 1.466 1.528 1.571 1.224 1.471 1.478 1.54  1.433
 1.236 1.467 1.244 1.375 1.44  1.138 1.193 1.522]
panel row mean/std percentiles [0.15  0.574 0.779 0.948 1.559] [1.053 1.257 1.412 1.527 1.994]
panel rows 0-1 [[ 1.75005106  1.39508744  0.7137822  -0.48801431  1.29340818  3.27552272
          nan -1.02975826  1.00513834  1.10813613  0.08655577 -0.10265182
   0.70970196  2.06665424 -2.51302819         nan -2.09704238 -0.49710855
   2.85273673  2.48531121         nan -1.25496763  0.61094007  2.37309779
  -0.88025248  0.61637322 -0.12826053  1.48143578  2.42440637  1.65141776
   0.11770471 -0.50300346]
 [ 1.64400422 -0.15723211  1.57487778  1.12173593  0.38518474  0.92617441
          nan -0.38436811  0.58318957  2.80657664 -0.79452302         nan
   0.29933436  2.50488189  2.01521222  0.61839345 -0.77310784  2.89940108
   2.04143939  1.35057021  1.51943052  1.99459631  1.31724083  0.4490643
   1.65352628  0.61773697  1.7069073   1.67852898  2.17088812  2.0907334
   1.56719314  3.67155313]]
ref-raw pairwise correlations [[ 1.    0.12 -0.03 -0.16 -0.22  0.04]
 [ 0.12  1.   -0.07  0.16  0.12 -0.1 ]
 [-0.03 -0.07  1.   -0.11  0.07  0.07]
 [-0.16  0.16 -0.11  1.    0.1   0.15]]
stratum 0 aligned mean/std [-0.411 -0.497 -0.281 -0.165] [0.969 0.791 0.979 1.158] correlation [[ 1.    0.04  0.18  0.13]
 [ 0.04  1.   -0.1   0.24]
 [ 0.18 -0.1   1.    0.13]
 [ 0.13  0.24  0.13  1.  ]]
stratum 1 aligned mean/std [-0.022 -0.089  0.02   0.007] [1.069 0.935 1.013 1.064] correlation [[ 1.   -0.25 -0.21  0.02]
 [-0.25  1.   -0.1  -0.07]
 [-0.21 -0.1   1.   -0.03]
 [ 0.02 -0.07 -0.03  1.  ]]
stratum 2 aligned mean/std [0.305 0.396 0.474 0.335] [1.257 0.9   0.94  0.96 ] correlation [[ 1.    0.1  -0.04 -0.23]
 [ 0.1   1.    0.01 -0.08]
 [-0.04  0.01  1.   -0.04]
 [-0.23 -0.08 -0.04  1.  ]]
asset metadata keys ['files', 'schema']
relevant schema excerpts
              "coupon_features_raw": {
              "ref_coupons": {
              "stratum": {
              "coupon_features_raw": {
              "ref_coupons": {
              "stratum": {

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 8 — `execute_in_remote_docker` (`26227 ms` LLM)
- **Rationale**: Read the inherited imputation rule and mask observed training features to test whether pooling erases stratum shifts. This checks the weakest slice without choosing a donor rule from just three naturally incomplete public test rows.
- **Alternatives Weighed**: `Replace the prescribed median panel statistics` (rejected: The panel inspection did not reveal a recoverable calibration structure, and the alignment statistic is explicit.)

```bash
cd /holdout_task && sed -n '1,122p' alpha_m97_c02_batch_03_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,numpy as np
D=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; tr,te=D['train'],D['test']; n=len(tr['lab_id']); lab,group,raw,ref=[np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')]; x=raw.copy(); keys=np.unique(np.c_[lab,group],axis=0)
for l in np.unique(lab):
 a=lab==l; p=ref[a]; x[a]=(x[a]-np.nanmedian(np.nanmean(p,axis=0)))/np.nanmedian(np.nanstd(p,axis=0))
print('lab-stratum cell counts',[(tuple(k),int(np.sum((lab==k[0])&(group==k[1])))) for k in keys]); print('natural missing by split/column',(~np.isfinite(x[:n])).sum(0),(~np.isfinite(x[n:])).sum(0))
# Mask one known train feature at a time; keep that value out of every donor mean.
errors={k:np.full(x[:n].shape,np.nan) for k in ('pooled_cells','matching_cell','matching_stratum_cells')}
for i,j in np.argwhere(np.isfinite(x[:n])):
 v=x[:,j].copy(); v[i]=np.nan
 cells=np.array([np.nanmean(v[(lab==l)&(group==s)]) for l,s in keys]); options=(np.nanmean(cells),cells[(keys[:,0]==lab[i])&(keys[:,1]==group[i])][0],np.nanmean(cells[keys[:,1]==group[i]]))
 for k,p in zip(errors,options): errors[k][i,j]=p-x[i,j]
for k,e in errors.items():
 f=lambda a: np.round(np.sqrt(np.nanmean(a*a)),5)
 print(k,'pooled',f(e),'by lab',[f(e[lab[:n]==l]) for l in np.unique(lab)],'by stratum',[f(e[group[:n]==s]) for s in np.unique(group)])
PY
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_tau, past, hold, features, age, dt, times, derivatives=False):
    tau = np.exp(log_tau)
    a, q, r = age[:, None]/tau, dt[:, None]/tau, times[:, None]/tau
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    uptake = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ uptake
    step = -np.expm1(-r)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    phi = times/(1 + 0.85*times)
    delta = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            slip[:, None, None] * phi[None, :, None],
                            delta[:, None, None] * step[None, :, :1]), axis=2)
    rate = np.column_stack((gap/tau, features, slip, delta/tau[0]))
    if not derivatives:
        return curve, rate
    dg = -(past @ (a*uptake - q*ea*eq))
    ds = -r*er
    dc = dg[:, None, :] * step + gap[:, None, :] * ds
    dr = (dg - gap)/tau
    xc = np.stack((dg[:, :1]*step[:, 0] + delta[:, None]*ds[:, 0],
                   -dg[:, 1:]*step[:, 0]), axis=-1)
    xr = np.column_stack((dg[:, 0]-delta, -dg[:, 1]))/tau[0]
    return curve, rate, np.vstack((dc.reshape(-1, 2), dr)), np.vstack((xc.reshape(-1, 2), xr))


def _fit(past, hold, features, age, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]
    observed = np.isfinite(target)
    target = target[observed]
lab-stratum cell counts [((0, 0), 18), ((0, 1), 20), ((0, 2), 16), ((1, 0), 16), ((1, 1), 36), ((1, 2), 22), ((2, 0), 17), ((2, 1), 10), ((2, 2), 33), ((3, 0), 15), ((3, 1), 21), ((3, 2), 28)]
natural missing by split/column [6 8 4 1] [1 2 0 1]
pooled_cells pooled 1.05772 by lab [1.09361, 1.08545, 1.06981, 0.96951] by stratum [1.02727, 1.0119, 1.11349]
matching_cell pooled 1.04525 by lab [1.09952, 1.06617, 1.05007, 0.95697] by stratum [0.99437, 1.04588, 1.07692]
matching_stratum_cells pooled 1.01832 by lab [1.0605, 1.0465, 1.02129, 0.93318] by stratum [0.96403, 1.01771, 1.05302]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 9 — `execute_in_remote_docker` (`38052 ms` LLM)
- **Rationale**: Lower coupon imputation error may not translate into better strain and rate predictions. Compare both fills on masked leave-lab-out responses and the naturally incomplete clean test slice before changing the solver.
- **Alternatives Weighed**: `Fill using only the matching lab–stratum cell` (rejected: Its worst-lab masked-feature RMSE was 1.09952, worse than both pooled and stratum-balanced fills.)

```bash
set -e
cd /holdout_task
cat > /tmp/compare_rheology_fills.py <<'PY'
import importlib.util
from pathlib import Path
import pickle
import numpy as np

path = Path('/holdout_task/alpha_m97_c02_batch_03_solver.py')
spec = importlib.util.spec_from_file_location('baseline', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
source = path.read_text()
old = '    fill = np.nanmean(means, axis=0)\n    x = np.where(np.isfinite(x), x, fill)'
new = '''    pooled = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        mask = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=pooled.copy(), where=count > 0)
        x[mask] = np.where(np.isfinite(x[mask]), x[mask], fill)'''
assert source.count(old) == 1
candidate = source.replace(old, new)
Path('/tmp/rheology_stratum_candidate.py').write_text(candidate)
alt = {}
exec(compile(candidate, '/tmp/rheology_stratum_candidate.py', 'exec'), alt)
with open('/holdout_task/public_alpha_m97_c02_batch_03.pkl', 'rb') as file:
    item = pickle.load(file)[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
dt, times = d['past_dt'], d['eval_times']
age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
# Hold out entire training labs and mask a known coupon feature per complete held-out row.
# Both methods see the same masks, and no held-out response enters fitting.
raw = tr['coupon_features_raw']
observed = np.isfinite(raw).all(axis=1)
for name, ns in [('pooled', vars(base)), ('stratum-balanced', alt)]:
    print(name)
    out = ns['solve_alpha_m97_c02_batch_03'](d)
    ec = out['predicted_strain'] - gt['te_clean']
    er = out['initial_rate_pred'] - gt['te_true_rate0']
    rmse = lambda a: float(np.sqrt(np.mean(a*a)))
    for label, sel in [('all', np.ones(len(er), bool)), ('natural missing', missing), ('complete', ~missing)]:
        print(label, 'clean strain/rate RMSE', rmse(ec[sel]), rmse(er[sel]))
    print('tau/weight max abs errors', *(float(np.max(np.abs(out[k] - gt[g]))) for k, g in [('estimated_taus', 'taus'), ('hereditary_weights', 'theta'), ('regime_mode_weights', 'vp_theta')]))
    for lab in np.unique(tr['lab_id']):
        holdout = tr['lab_id'] == lab
        masked = holdout & observed
        idx = np.flatnonzero(masked)
        changed = raw.copy()
        changed[idx, np.arange(len(idx)) % raw.shape[1]] = np.nan
        dd = dict(d, train=dict(tr, coupon_features_raw=changed))
        p,h,x,e = ns['_prepare'](dd)
        fit = np.flatnonzero(~holdout)
        y = tr['strain_curves_obs'] - e[:n, None]
        r = tr['instant_rate_obs']
        lt, w = ns['_fit'](p[fit], h[fit], x[fit], age, dt, times, y[fit], r[fit])
        c,v = ns['_basis'](lt, p[idx], h[idx], x[idx], age, dt, times)
        print('masked holdout lab', lab, 'strain/rate median, P90 AE', np.round(np.nanpercentile(np.abs(c@w-y[idx]), [50,90]), 6), np.round(np.nanpercentile(np.abs(v@w-r[idx]), [50,90]), 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/compare_rheology_fills.py
```
- **Container Exit Code**: `0` (`1893 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pooled
all clean strain/rate RMSE 0.01693105773921384 0.021879808743886633
natural missing clean strain/rate RMSE 0.07438702311994465 0.09512957988842594
complete clean strain/rate RMSE 0.007651166377121426 0.010299404301703846
tau/weight max abs errors 0.009303377079067587 0.004371426534100165 0.0017032631381773966
masked holdout lab 0 strain/rate median, P90 AE [0.059243 0.22027 ] [0.077537 0.224372]
masked holdout lab 1 strain/rate median, P90 AE [0.056621 0.199903] [0.071541 0.211773]
masked holdout lab 2 strain/rate median, P90 AE [0.07561  0.230591] [0.099345 0.241526]
masked holdout lab 3 strain/rate median, P90 AE [0.086842 0.202526] [0.119005 0.223359]
stratum-balanced
all clean strain/rate RMSE 0.012643263602019837 0.016772180883703382
natural missing clean strain/rate RMSE 0.049576362882122846 0.06385175375020864
complete clean strain/rate RMSE 0.0077421348354142555 0.010783035753317986
tau/weight max abs errors 0.007425286461888447 0.003777917585771562 0.0022826546538080306
masked holdout lab 0 strain/rate median, P90 AE [0.06094  0.208843] [0.089334 0.220097]
masked holdout lab 1 strain/rate median, P90 AE [0.059102 0.223721] [0.09088  0.224222]
masked holdout lab 2 strain/rate median, P90 AE [0.071089 0.224991] [0.089242 0.235203]
masked holdout lab 3 strain/rate median, P90 AE [0.073793 0.22247 ] [0.09325 0.22682]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2253955e7ac11c81 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2253955e (directory).
```

### Turn 10 — `execute_in_remote_docker` (`29286 ms` LLM)
- **Rationale**: Promote the tested challenger without changing the physical model or reference alignment. Verify the installed artifact in an isolated process to catch copy or schema regressions, and check the single-point archive path that the complete public calibration does not exercise.
- **Alternatives Weighed**: `Retain the pooled fill` (rejected: It is slightly better on complete rows but materially worse on naturally missing rows and on the worst masked leave-lab-out slice.)

```bash
set -e
cd /holdout_task
cp /tmp/rheology_stratum_candidate.py alpha_m97_c02_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_03.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
o = m.solve_alpha_m97_c02_batch_03(d)
te = d['test']
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
e = o['predicted_strain'] - g['te_clean']
r = o['initial_rate_pred'] - g['te_true_rate0']
for name, a in [('all', np.ones(len(r), bool)), ('missing', missing), ('complete', ~missing)] + [(f'lab {lab}', te['lab_id'] == lab) for lab in np.unique(te['lab_id'])]:
    print(name, 'strain/rate RMSE', np.sqrt(np.mean(e[a]**2)), np.sqrt(np.mean(r[a]**2)))
assert np.sqrt(np.mean(e[missing]**2)) < 0.050
assert np.sqrt(np.mean(r[missing]**2)) < 0.064
for key, ref, limit in [('estimated_taus', 'taus', 0.010), ('hereditary_weights', 'theta', 0.005), ('regime_mode_weights', 'vp_theta', 0.003)]:
    err = np.max(np.abs(o[key] - g[ref]))
    print(key, 'max absolute error', err)
    assert err < limit
cal = d['calibration_load_cells']
full = m._calibrate(cal)
for channel in range(4):
    archive = m._archived_calibration(channel)
    assert archive is not None and np.isfinite(archive).all()
    ids = np.flatnonzero(cal['lab_id'] == channel)
    keep = (cal['lab_id'] != channel)
    keep[ids[0]] = True
    sparse = {k: v[keep] for k, v in cal.items()}
    got = m._calibrate(sparse)
    np.testing.assert_allclose(got[channel], archive)
    np.testing.assert_allclose(got[np.arange(4) != channel], full[np.arange(4) != channel])
print('Single-point archive fallback verified on all four channels')
PY
```
- **Container Exit Code**: `0` (`1155 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
all strain/rate RMSE 0.012643263602019837 0.016772180883703382
missing strain/rate RMSE 0.049576362882122846 0.06385175375020864
complete strain/rate RMSE 0.0077421348354142555 0.010783035753317986
lab 0 strain/rate RMSE 0.01563468075681694 0.020897916203447726
lab 1 strain/rate RMSE 0.003410562575965652 0.0055278707576911015
lab 2 strain/rate RMSE 0.006967891202917614 0.009462968913427367
lab 3 strain/rate RMSE 0.017287859381061088 0.02258789094389537
estimated_taus max absolute error 0.007425286461888447
hereditary_weights max absolute error 0.003777917585771562
regime_mode_weights max absolute error 0.0022826546538080306
Single-point archive fallback verified on all four channels

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bb07cd7f730589a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bb07cd7f (directory).
```

### Turn 11 — `submit_completed_task` (`16229 ms` LLM)
- **Rationale**: The installed artifact passed isolated schema, parameter, worst-slice, and calibration fallback checks. Stratum-balanced imputation materially improved the missing-coupon slice without a comparable loss on complete rows; it does not eliminate finite-panel uncertainty.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:8086'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The isolated public schema check passed, and single-point calibration fallback was verified on all four channels. Public strain and initial-rate RMSE were 0.01264 and 0.01677 overall, and 0.04958 and 0.06385 on rows with missing coupon features.`

### Turn 12 — `execute_in_remote_docker` (`41085 ms` LLM)
- **Rationale**: Independently reconstruct the exact equations from measured histories to catch a missing or mis-signed term. Inspect within-cell residuals for lab-linked bias hidden by pooled errors, and test whether a spike can displace a valid fit at the four-clean-point boundary. A mismatch would identify whether to revise the basis, investigate alignment, or correct calibration selection.
- **Alternatives Weighed**: `Run relay-mode, classification, or Fokker–Planck checks` (rejected: The supplied model is a continuous strain and rate regression with no relay state, class label, or stochastic drift operator. The applicable checks are the rational saturation, branch difference, and signed residuals within lab and stratum.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_03.pkl', 'rb') as f:
    i = pickle.load(f)[0]
d, g = i['dataset'], i['gt']
o = m.solve_alpha_m97_c02_batch_03(d)
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
p, h, x, e = m._prepare(d)
tau, hw, rw = (o[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights'))
z = np.zeros((len(h), 2))
for j, dt in enumerate(d['past_dt']):
    a = np.exp(-dt/tau)
    z = a*z + (1-a)*p[:, j, None]
gap = h[:, None]-z
slip = np.sign(h)*np.sqrt(np.abs(h))
cross = rw[1]*(gap[:, 0]-gap[:, 1])
drift = x@hw[2:]+rw[0]*slip
t = d['eval_times']
step = 1-np.exp(-t[:, None]/tau)
curve = e[:, None]+(gap*hw[:2])@step.T+drift[:, None]*(t/(1+0.85*t))+cross[:, None]*step[:, 0]
rate = gap@(hw[:2]/tau)+drift+cross/tau[0]
print('independent equation max strain/rate deviations', np.max(abs(curve[n:]-o['predicted_strain'])), np.max(abs(rate[n:]-o['initial_rate_pred'])))
np.testing.assert_allclose(curve[n:],o['predicted_strain'],atol=1e-12,rtol=0)
np.testing.assert_allclose(rate[n:],o['initial_rate_pred'],atol=1e-12,rtol=0)
ce, re = curve[n:]-g['te_clean'], rate[n:]-g['te_true_rate0']
complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
for lab in np.unique(te['lab_id']):
    for group in np.unique(te['stratum']):
        take = (te['lab_id']==lab)&(te['stratum']==group)&complete
        if take.any():
            print('complete lab/stratum n, strain/rate RMSE and rate signed mean',lab,group,take.sum(),round(np.sqrt(np.mean(ce[take]**2)),6),round(np.sqrt(np.mean(re[take]**2)),6),round(np.mean(re[take]),6))
for k, r, lim in [('estimated_taus','taus',.010),('hereditary_weights','theta',.005),('regime_mode_weights','vp_theta',.003)]:
    err = np.max(abs(o[k]-g[r])); print(k,'public regression margin',lim-err); assert err<lim
cal = d['calibration_load_cells']
max_delta = 0.
for lab in range(4):
    ids = np.flatnonzero(cal['lab_id']==lab)
    others = np.flatnonzero(cal['lab_id']!=lab)
    for j in ids:
        clean = ids[ids!=j][:4]
        expected = np.polyfit(cal['cmd_stress'][clean],cal['true_stress'][clean],1)
        ix = np.r_[others,clean,j]
        for sign in (-1,1):
            sparse = {k:v[ix].copy() for k,v in cal.items()}
            sparse['true_stress'][-1] += sign*3.2
            got = m._calibrate(sparse)[lab]
            max_delta = max(max_delta, float(np.max(abs(got-expected))))
print('four-clean-plus-one-spike maximum coefficient deviation', max_delta)
assert max_delta<1e-10
PY
```
- **Container Exit Code**: `0` (`1281 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
independent equation max strain/rate deviations 8.881784197001252e-16 8.881784197001252e-16
complete lab/stratum n, strain/rate RMSE and rate signed mean 0 0 3 0.005542 0.008056 -0.004973
complete lab/stratum n, strain/rate RMSE and rate signed mean 0 1 7 0.011323 0.015064 0.004208
complete lab/stratum n, strain/rate RMSE and rate signed mean 0 2 5 0.003514 0.00531 -0.001144
complete lab/stratum n, strain/rate RMSE and rate signed mean 1 0 4 0.003924 0.006228 -0.005541
complete lab/stratum n, strain/rate RMSE and rate signed mean 1 1 10 0.003261 0.005533 -0.00389
complete lab/stratum n, strain/rate RMSE and rate signed mean 1 2 3 0.003152 0.004403 -0.003665
complete lab/stratum n, strain/rate RMSE and rate signed mean 2 0 5 0.00325 0.004721 0.002129
complete lab/stratum n, strain/rate RMSE and rate signed mean 2 1 2 0.001618 0.001889 0.001507
complete lab/stratum n, strain/rate RMSE and rate signed mean 2 2 9 0.001636 0.003728 0.000556
complete lab/stratum n, strain/rate RMSE and rate signed mean 3 0 6 0.009665 0.013855 0.001802
complete lab/stratum n, strain/rate RMSE and rate signed mean 3 1 8 0.009465 0.013591 -0.004535
complete lab/stratum n, strain/rate RMSE and rate signed mean 3 2 7 0.014772 0.019662 0.004686
estimated_taus public regression margin 0.002574713538111553
hereditary_weights public regression margin 0.0012220824142284382
regime_mode_weights public regression margin 0.0007173453461919695
four-clean-plus-one-spike maximum coefficient deviation 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bb07cd7f730589a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bb07cd7f (directory).
```

### Turn 13 — `submit_completed_task` (`16027 ms` LLM)
- **Rationale**: The installed artifact passed isolated schema and parameter checks, independent equation reconstruction, and calibration stress tests. All four channels retained the clean in-batch fit with four clean points plus one EMI spike. Stratum-balanced imputation reduced the largest observed error, but finite-panel and missing-feature uncertainty remain.
- **Alternatives Weighed**: `Replace the prescribed reference-panel normalization` (rejected: The measured panel did not support a replacement that recovers latent normalization; the validated imputation change improves the worst slice without changing the specified normalization.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The isolated public schema check passed, and the strain and initial-rate predictions matched an independent implementation of the jammed-slip equations to machine precision. Public RMSE was 0.01264 for strain and 0.01677 for initial rate overall, rising to 0.04958 and 0.06385 on rows with missing coupon features.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
