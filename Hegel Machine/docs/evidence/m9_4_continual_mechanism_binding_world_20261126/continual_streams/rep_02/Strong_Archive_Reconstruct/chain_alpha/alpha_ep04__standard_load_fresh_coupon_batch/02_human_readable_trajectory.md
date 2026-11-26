# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_ep04__standard_load_fresh_coupon_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M94-Continual-Binding-World` |
| **Domain** | `Multi-Lab Polymer Coupon Hereditary Creep & Load-Cell Calibration Stream` |
| **Wave / Wormhole Family** | `m94_stream_step_4` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `239,755` (`229,529` / `71,680` / `10,226`) |
| **Total Episode Wall Time (s)** | `444.26s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_12_split_calibrated_hereditary_creep`**: [BIFURCATED SUB-REGIME `fraction_of_coupon_features_observed_bef < 1.0`]: This episode recorded no new nogoods; the parent study showed why pooled features and cell-mean imputation are insufficient for specimen-specific drift: on three incomplete rows, feature-based initial-rate RMSE was 0.1705 versus 0.0056 for the measured rates. Calibrate both history and hold stress with the same facility-specific affine law, estimate feature offsets and scales from reference controls rather than signal marginals, and give each observed facility–stratum cell one vote per feature during imputation. Propagate two exponential memory branches exactly under piecewise-constant stress, use L2 only for initialization, then fit complete rows with separate strain and rate MAD scales, soft-L1 at 1.5, and Cauchy at 2.0. For a prediction row with a finite measured onset rate, apply its innovation only to linear drift, with gain 1 for any missing feature and 0.5 for complete features as in this episode's winning source. The refactor requires a converged fit with distinct relaxation times and a robustly weighted, column-normalized joint Jacobian singular-value ratio of at least 1e-8; this is a local identifiability check, not a guarantee of transfer to an unseen facility. (Counter-regime `fraction_of_coupon_features_observed_bef` warning: Incomplete features exclude a row from estimation of the shared law and select full drift correction on prediction rows with a finite measured rate. Complete prediction rows use half correction; without a finite measured rate, either branch keeps its feature-based drift. This is a prediction regime split, not a change in the memory law.)
- **`root_A02_nuisance_subspace_and_deconfounding`**: AXIOM A02 (NUISANCE SUBSPACE PROJECTION & IDENTIFIABILITY): When unknown multiplicative gains, batch shifts, random carrier phases, or background geometry changes confound raw observations, never subtract unaligned raw signals or fit pooled confounded shortcuts; separate invariant physical coordinates and project out nuisance subspaces with well-conditioned regularization so local anomalies are not absorbed into nuisance parameters.
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`28490 ms` LLM)
- **Rationale**: Inspect the local schema, stress inputs, and archived implementations before deciding which model terms carry over. A mismatch in state, history, or calibration conventions would change the implementation.

```bash
cd /holdout_task && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && grep -nE '^def |yield|plastic|hereditary|return \{' /tmp/hegel_continual_archive/alpha*solver.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_ep04.pkl','rb') as f: d=pickle.load(f)
def show(d, pre=''):
 for k,v in d.items():
  if isinstance(v,dict): print(pre+k); show(v,pre+'  ')
  else:
   a=np.asarray(v); print(pre+k,a.shape,a.dtype, (float(np.nanmin(a)),float(np.nanmax(a))) if a.size and a.dtype.kind in 'fiu' else str(v)[:130])
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference coupon alignment imputation'
```
- **Container Exit Code**: `0` (`523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rols: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_preloaded_hereditary_transfer@5fcfe742 | align_reference_features(values: np.ndarray, references: np.ndarray, facility: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align features to facility references and return the original completeness mas | REQ: {'field': 'distinct_calibration_commands_per_facility', 'op': '>=', 'value': 2}; {'field | --describe autogenous_preloaded_hereditary_transfer.align_reference_features
autogenous_preloaded_hereditary_transfer@5fcfe742 | fit_hereditary_hold(history_stress: np.ndarray, hold_stress: np.ndarray, history_dt: np.ndarray, hold_times: np.ndarray, initial_memory: np.ndarray, features: np.ndarray, complete: np.ndarray, fit_rows: np.ndarray, initial_strain: np.ndarray, observed_strain: np.ndarray, observed_rate: np.ndarray, critical_strain: np.ndarray, tau_bounds: tuple[float, float], *, stress_floor: float, rank_rtol: float=1e-08) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Return sorted relaxation times, strain, secant compliance, and first crossing | REQ: {'field': 'distinct_calibration_commands_per_facility', 'op': '>=', 'value': 2}; {'field | --describe autogenous_preloaded_hereditary_transfer.fit_hereditary_hold
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`24872 ms` LLM)
- **Rationale**: Use the validator's list-of-cases indexing to inspect the actual panels, then compare their state variables and time conventions against the latest archived solver before reusing its basis.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_ep04.pkl','rb'))
print('cases',len(p),[list(q) for q in p]); d=p[0]['dataset']
for k,v in d.items():
 for n,w in (v.items() if isinstance(v,dict) else [(k,v)]):
  a=np.asarray(w); print(k,n,a.shape,a.dtype, (a.item() if a.ndim==0 else (np.nanmin(a),np.nanmax(a),int(np.isnan(a).sum())) if a.dtype.kind in 'fiu' else ''))
PY
cat /tmp/hegel_continual_archive/alpha_ep03_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_calibrated_hereditary_creep.align_coupon_features
```
- **Container Exit Code**: `0` (`525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"op": ">=",
          "value": 0
        },
        {
          "field": "minimum_hold_time",
          "op": ">=",
          "value": 0
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_coupon_features",
          "signature": "align_coupon_features(raw_features: np.ndarray, reference_coupons: np.ndarray, facility: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray]",
          "doc_summary": "Align rows using facility reference panels, then impute missing features."
        }
      ],
      "index_cards": [
        "autogenous_calibrated_hereditary_creep@1b21d6d8 | align_coupon_features(raw_features: np.ndarray, reference_coupons: np.ndarray, facility: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align rows using facility reference panels, then impute missing features | REQ: {'field': 'relaxation_time_lower_bound', 'op': '>', 'value': 0}; {'field': 'relaxation_t | --describe autogenous_calibrated_hereditary_creep.align_coupon_features",
        "autogenous_calibrated_hereditary_creep@1b21d6d8 | calibrate_stress_history(past_command: np.ndarray, hold_command: np.ndarray, facility: np.ndarray, calibration_facility: np.ndarray, calibration_command: np.ndarray, calibration_stress: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Fit an affine load-cell law per facility and convert past and hold loads | REQ: {'field': 'relaxation_time_lower_bound', 'op': '>', 'value': 0}; {'field': 'relaxation_t | --describe autogenous_calibrated_hereditary_creep.calibrate_stress_history",
        "autogenous_calibrated_hereditary_creep@1b21d6d8 | fit_predict_creep(past_stress: np.ndarray, hold_stress: np.ndarray, history_dt: np.ndarray, hold_times: np.ndarray, features: np.ndarray, complete_features: np.ndarray, fit_rows: np.ndarray, initial_strain: np.ndarray, observed_strain: np.ndarray, observed_rate: np.ndarray, tau_bounds: tuple[float, float], *, rank_rtol: float=1e-08) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Fit two exponential memory branches with zero initial memory | REQ: {'field': 'relaxation_time_lower_bound', 'op': '>', 'value': 0}; {'field': 'relaxation_t | --describe autogenous_calibrated_hereditary_creep.fit_predict_creep"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_calibrated_hereditary_creep",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`32355 ms` LLM)
- **Rationale**: Zero residual plastic state does not rule out yielding during the recorded loading. Measure exceedances after calibration: if neither history nor hold reaches yield, omit the archived plastic terms; otherwise validate their contribution separately.

```bash
sed -n '20,145p' /tmp/hegel_continual_archive/alpha_ep03_solver.py; cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_ep04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; lim=d['yield_stress']; laws=[]
for i in range(4):
 m=c['lab_id']==i; x,y=c['cmd_stress'][m],c['true_stress'][m]; g,b=np.polyfit(x,y,1); laws.append((g,b)); print('calibration',i,'gain',round(g,6),'bias',round(b,6),'RMSE',round(np.sqrt(np.mean((g*x+b-y)**2)),6))
for name in ('train','test'):
 p=d[name]; g,b=np.asarray(laws)[p['lab_id']].T
 for key in ('past_stress_cmd','stress_hold_cmd'):
  x=p[key]; s=g[:,None]*x+b[:,None] if x.ndim==2 else g*x+b; z=np.abs(s)>lim; print(name,key,'range',np.round([s.min(),s.max()],6),'exceedance values',np.count_nonzero(z),'exceedance rows',np.count_nonzero(np.any(z,axis=1) if z.ndim==2 else z))
print('times',d['eval_times'])
PY
```
- **Container Exit Code**: `0` (`532 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4]
            if plastic:
                da[:, :, 0] += dp[rows] * q[-1]
            da[:, :, 0] += (1 - q[1]) * da[:, :, 1]
            da[:, :, 1] *= high - q[0] - separation
            return (np.concatenate([da, a[rows, :, :cols]], axis=2) / scale[None, :, None]).reshape(-1, cols+2)

        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss, f_scale=cutoff,
                             max_nfev=150, x_scale='jac')

    scale = _noise(starts[0][2])
    pilots = [optimize(s[1], scale, 'soft_l1', 1.5) for s in starts[:3]]
    p = min(pilots, key=lambda f: f.cost).x
    scale = _noise(design.matrix(decode(p))[rows, :, :cols] @ p[2:] - y)
    fit = min((optimize(f.x, scale, 'cauchy', 2.0) for f in pilots), key=lambda f: f.cost)
    return np.exp(decode(fit.x)), fit.x[2:]


def _prepare(dataset):
    x, complete = _aligned_panels(dataset)
    train, test = dataset['train'], dataset['test']
    args = dataset['past_dt'], dataset['eval_times']
    n = len(train['lab_id'])
    dtrain = _Design(*_physical_stresses(dataset, train), *args, x[:n], dataset['yield_stress'])
    dtest = _Design(*_physical_stresses(dataset, test), *args, x[n:], dataset['yield_stress'])
    y = np.column_stack([train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']])
    return dtrain, dtest, y, complete


def solve_alpha_ep03(dataset: dict) -> dict:
    train, test, y, complete = _prepare(dataset)
    tau, theta = _fit(train, y, complete[:len(y)])
    pred = test.matrix(np.log(tau)) @ theta
    rate = dataset['test']['instant_rate_obs']
    innovation = np.where(np.isfinite(rate), rate-pred[:, -1], 0.0)
    # With measured features, shrink the noisy rate innovation toward the model.
    # With imputed features, use the measured rate to recover the missing drift.
    drift = np.where(complete[len(y):], 0.25, 1.0) * innovation
calibration 0 gain 1.279979 bias 0.13997 RMSE 5.5e-05
calibration 1 gain 0.81999 bias -0.119959 RMSE 0.000113
calibration 2 gain 1.179967 bias 0.089923 RMSE 5.7e-05
calibration 3 gain 0.879991 bias -0.110021 RMSE 7.2e-05
train past_stress_cmd range [-2.451005  2.45101 ] exceedance values 0 exceedance rows 0
train stress_hold_cmd range [-2.11064   2.349093] exceedance values 0 exceedance rows 0
test past_stress_cmd range [-2.451005  2.45101 ] exceedance values 0 exceedance rows 0
test stress_hold_cmd range [-2.133181  2.318307] exceedance values 0 exceedance rows 0
times [0.25 0.8  1.35 1.9  2.45 3.   3.55 4.1  4.65 5.2 ]
```

### Turn 4 — `execute_in_remote_docker` (`83989 ms` LLM)
- **Rationale**: Fit the identifiable seven coefficients and compare measured-rate correction strengths on held-out labs, separating incomplete coupons so their larger uncertainty does not determine the correction for complete coupons.
- **Alternatives Weighed**: `Fit the archived plastic coefficients` (rejected: Both above-yield design columns are identically zero on the supplied episode, so their coefficients are not identifiable.)

```bash
cat > /holdout_task/alpha_ep04_solver.py <<'PY'
"""Calibrated two-branch hereditary creep for fresh, sub-yield coupons."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for lab in np.unique(labs):
        rows = labs == lab
        r = refs[rows]
        offset = np.nanmedian(np.nanmean(r, axis=0))
        gain = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[rows] = (x[rows] - offset) / gain
    cells = np.unique(np.column_stack([labs, strata]), axis=0)
    fill = np.nanmean([np.nanmean(x[(labs == l) & (strata == s)], axis=0)
                       for l, s in cells], axis=0)
    observed = np.isfinite(x)
    return np.where(observed, x, fill), observed


def _calibrate(cal):
    laws = np.empty((4, 2))
    for lab in range(4):
        rows = cal['lab_id'] == lab
        cmd = cal['cmd_stress'][rows]
        laws[lab] = np.linalg.lstsq(np.column_stack([cmd, np.ones(len(cmd))]),
                                   cal['true_stress'][rows], rcond=None)[0]
    return laws


class _Design:
    def __init__(self, panel, laws, dt, times, x):
        gain, bias = laws[panel['lab_id']].T
        self.past = panel['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = panel['stress_hold_cmd'] * gain + bias
        self.dt = dt
        self.tail = np.cumsum(dt[::-1])[::-1] - dt
        self.times = times
        features = np.column_stack([np.ones(len(x)), x])
        self.linear = features[:, None, :] * np.r_[times, 1.0][None, :, None]

    def matrix(self, log_tau, derivative=False):
        tau = np.exp(log_tau)
        u = self.dt[:, None] / tau
        v = self.tail[:, None] / tau
        h = self.times[:, None] / tau
        decay = np.exp(-v)
        kernel = decay * -np.expm1(-u)
        gap = self.hold[:, None] - self.past @ kernel
        response = np.vstack([-np.expm1(-h), 1.0 / tau])
        a = np.concatenate([gap[:, None, :] * response, self.linear], axis=2)
        if not derivative:
            return a
        dg = -self.past @ (v * kernel - u * decay * np.exp(-u))
        dr = np.vstack([-h * np.exp(-h), -1.0 / tau])
        return a, dg[:, None, :] * response + gap[:, None, :] * dr


def _scale(r):
    def mad(v):
        return max(1e-8, 1.4826 * np.median(np.abs(v - np.median(v))))
    return np.r_[np.full(r.shape[1] - 1, mad(r[:, :-1])), mad(r[:, -1])]


def _fit(design, y, rows):
    y = y[rows]
    low, high = np.log([0.051, 24.9])
    sep = 0.01

    def decode(p):
        return np.array([p[0], p[0] + sep + p[1] * (high - p[0] - sep)])

    grid = np.geomspace(0.085, 19.0, 11)
    starts = []
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            l1, l2 = np.log([t1, t2])
            a = design.matrix([l1, l2])[rows]
            w = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
            r = a @ w - y
            starts.append((np.sum(r*r), np.r_[l1, (l2-l1-sep)/(high-l1-sep), w], r))
    starts.sort(key=lambda item: item[0])
    bounds = (np.r_[low, 0., np.full(7, -np.inf)],
              np.r_[high-2*sep, 1., np.full(7, np.inf)])

    def optimize(p, scale, loss, cutoff):
        def fun(q):
            return ((design.matrix(decode(q))[rows] @ q[2:] - y) / scale).ravel()

        def jac(q):
            a, d = design.matrix(decode(q), derivative=True)
            d = d[rows] * q[2:4]
            d[:, :, 0] += (1 - q[1]) * d[:, :, 1]
            d[:, :, 1] *= high - q[0] - sep
            return (np.concatenate([d, a[rows]], axis=2) / scale[None, :, None]).reshape(-1, 9)

        return least_squares(fun, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=cutoff, max_nfev=150, x_scale='jac')

    scale = _scale(starts[0][2])
    pilots = [optimize(p, scale, 'soft_l1', 1.5) for _, p, _ in starts[:3]]
    p = min(pilots, key=lambda f: f.cost).x
    scale = _scale(design.matrix(decode(p))[rows] @ p[2:] - y)
    fit = min((optimize(f.x, scale, 'cauchy', 2.0) for f in pilots), key=lambda f: f.cost)
    return np.exp(decode(fit.x)), fit.x[2:]


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x, seen = _features(train, test)
    laws = _calibrate(dataset['calibration_load_cells'])
    args = laws, dataset['past_dt'], dataset['eval_times']
    tr = _Design(train, *args, x[:n])
    te = _Design(test, *args, x[n:])
    y = np.column_stack([train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']])
    return tr, te, y, seen


def solve_alpha_ep04(dataset: dict) -> dict:
    tr, te, y, seen = _prepare(dataset)
    complete = seen.all(axis=1)
    tau, w = _fit(tr, y, complete[:len(y)])
    pred = te.matrix(np.log(tau)) @ w
    measured = dataset['test']['instant_rate_obs']
    innovation = np.where(np.isfinite(measured), measured - pred[:, -1], 0.0)
    correction = np.where(complete[len(y):], 0.25, 1.0) * innovation
    return {'estimated_taus': tau, 'hereditary_weights': w,
            'predicted_strain': dataset['test']['strain_0'][:, None] + pred[:, :-1]
                                + correction[:, None] * te.times,
            'initial_rate_pred': pred[:, -1] + correction}
PY
cat > /tmp/ep04_cv.py <<'PY'
import pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_ep04_solver as s
d=pickle.load(open('/holdout_task/public_alpha_ep04.pkl','rb'))[0]['dataset']
tr,te,y,seen=s._prepare(d); seen=seen[:len(y)].all(axis=1)
lab=d['train']['lab_id']; stratum=d['train']['stratum']; p=np.empty_like(y)
for l in np.unique(lab):
    held=lab==l
    tau,w=s._fit(tr,y,~held & seen)
    p[held]=tr.matrix(np.log(tau))[held] @ w
    print('LOLO',l,'taus',np.round(tau,5),'w',np.round(w,5))
innovation=y[:,-1]-p[:,-1]; base=y[:,:-1]-p[:,:-1]
slope=(base@tr.times)/(tr.times@tr.times)
for flag in [True,False]:
    m=seen==flag
    print('complete',flag,'n',m.sum(),'innovation abs q50/90/max',np.round(np.quantile(abs(innovation[m]),[.5,.9,1]),6),'curve slope gap q50/90',np.round(np.quantile(abs(slope[m]),[.5,.9]),6))
    for a in (0,.1,.25,.5,.75,1):
        e=abs(base-a*innovation[:,None]*tr.times)
        slices=[e[m&(lab==l)&(stratum==st)] for l in range(4) for st in range(3) if np.sum(m&(lab==l)&(stratum==st))>=3]
        print(a,'MAE',round(e[m].mean(),6),'median',round(np.median(e[m]),6),'worst cell MAE',round(max(v.mean() for v in slices),6),'worst lab MAE',round(max(e[m&(lab==l)].mean() for l in range(4)),6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/ep04_cv.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`867 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_ep04_solver.py']`
- **Stdout (tail)**:
```text
LOLO 0 taus [0.49956 3.5499 ] w [ 0.70894  0.53276  0.07939 -0.14995  0.12002  0.09981 -0.08985]
LOLO 1 taus [0.49933 3.54763] w [ 0.70881  0.53174  0.0793  -0.14943  0.11988  0.09961 -0.08957]
LOLO 2 taus [0.49921 3.53647] w [ 0.70819  0.53136  0.07939 -0.15021  0.12019  0.10005 -0.08996]
LOLO 3 taus [0.49948 3.52148] w [ 0.70848  0.53141  0.07935 -0.15026  0.12037  0.10019 -0.08995]
complete True n 177 innovation abs q50/90/max [0.007388 0.016388 0.035537] curve slope gap q50/90 [0.007373 0.044079]
0 MAE 0.079555 median 0.008376 worst cell MAE 0.096671 worst lab MAE 0.088592
0.1 MAE 0.079648 median 0.008518 worst cell MAE 0.096831 worst lab MAE 0.088557
0.25 MAE 0.080883 median 0.009474 worst cell MAE 0.098362 worst lab MAE 0.08947
0.5 MAE 0.084482 median 0.012182 worst cell MAE 0.102051 worst lab MAE 0.092693
0.75 MAE 0.08893 median 0.015463 worst cell MAE 0.106165 worst lab MAE 0.0967
1 MAE 0.093781 median 0.019408 worst cell MAE 0.110729 worst lab MAE 0.101107
complete False n 23 innovation abs q50/90/max [0.085589 0.218495 0.304836] curve slope gap q50/90 [0.083926 0.187177]
0 MAE 0.349497 median 0.219353 worst cell MAE 0.541407 worst lab MAE 0.381287
0.1 MAE 0.324116 median 0.191103 worst cell MAE 0.503284 worst lab MAE 0.356019
0.25 MAE 0.287044 median 0.155987 worst cell MAE 0.446099 worst lab MAE 0.318116
0.5 MAE 0.229024 median 0.109047 worst cell MAE 0.367934 worst lab MAE 0.261416
0.75 MAE 0.171573 median 0.055659 worst cell MAE 0.290623 worst lab MAE 0.205598
1 MAE 0.13112 median 0.016738 worst cell MAE 0.225587 worst lab MAE 0.172027
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1b804b12ee29e00c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1b804b12 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`50520 ms` LLM)
- **Rationale**: Keep the stable seven-coefficient fit and change only the measured-rate correction. Rerun the exact solver in isolated Python on held-out labs to check that the complete-coupon improvement survives end-to-end execution and that incomplete-coupon drift recovery still improves independently observed strain curves.
- **Alternatives Weighed**: `Keep the 0.25 measured-rate correction for complete coupons` (rejected: It increased leave-one-lab-out curve MAE from 0.079555 to 0.080883 and worst eligible cell MAE from 0.096671 to 0.098362.)

```bash
set -e
sed -i 's/correction = np.where(complete\[len(y):\], 0.25, 1.0) \* innovation/# Use the measured rate to recover drift only when a feature was imputed.\n    correction = np.where(complete[len(y):], 0.0, innovation)/' /holdout_task/alpha_ep04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_ep04_solver.py'
spec=importlib.util.spec_from_file_location('solver',p)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
o=s.solve_alpha_ep04(d)
print('full fit taus',o['estimated_taus'],'weights',o['hereditary_weights'])
train=d['train']; labs=train['lab_id']; strata=train['stratum']; times=d['eval_times']
complete=np.isfinite(train['coupon_features_raw']).all(axis=1)
curve=np.empty_like(train['strain_curves_obs']); rate=np.empty_like(train['instant_rate_obs']); prev=curve.copy()
for l in np.unique(labs):
    m=labs==l
    tr={k:v[~m] for k,v in train.items()}
    te={k:v[m] for k,v in train.items() if k!='strain_curves_obs'}
    te['residual_plastic_state']=np.zeros(m.sum())
    q=s.solve_alpha_ep04({**d,'train':tr,'test':te})
    curve[m]=q['predicted_strain']; rate[m]=q['initial_rate_pred']
    prev[m]=curve[m]+np.where(complete[m],0.25*(train['instant_rate_obs'][m]-rate[m]),0.)[:,None]*times
for flag in (True,False):
    m=complete==flag
    groups=[m&(labs==l)&(strata==z) for l in np.unique(labs) for z in np.unique(strata)]
    for name,pred in [('previous',prev),('final',curve)]:
        e=np.abs(pred-train['strain_curves_obs'])
        print('curve',name,'complete',flag,'n',m.sum(),'MAE',round(e[m].mean(),6),'worst lab',round(max(e[m&(labs==l)].mean() for l in np.unique(labs)),6),'worst cell n>=3',round(max(e[g].mean() for g in groups if g.sum()>=3),6),'early/late median',np.round(np.median(e[m][:,[0,-1]],axis=0),6))
    if flag:
        er=np.abs(rate-train['instant_rate_obs'])
        print('independent measured-rate proxy MAE',round(er[m].mean(),6),'worst lab',round(max(er[m&(labs==l)].mean() for l in np.unique(labs)),6))
    else:
        np.testing.assert_allclose(rate[m],train['instant_rate_obs'][m],atol=1e-12)
        print('incomplete-coupon rate matches the observation used for drift recovery; held-out strain curves test the correction independently')
PY
```
- **Container Exit Code**: `0` (`906 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_ep04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
full fit taus [0.49943933 3.54005131] weights [ 0.70869537  0.53177234  0.07934033 -0.14991436  0.12011446  0.09987199
 -0.08980444]
curve previous complete True n 177 MAE 0.082136 worst lab 0.092579 worst cell n>=3 0.102568 early/late median [0.006768 0.016384]
curve final complete True n 177 MAE 0.081614 worst lab 0.092258 worst cell n>=3 0.102462 early/late median [0.006844 0.014698]
independent measured-rate proxy MAE 0.008341 worst lab 0.009673
curve previous complete False n 23 MAE 0.131248 worst lab 0.172444 worst cell n>=3 0.225195 early/late median [0.007288 0.032113]
curve final complete False n 23 MAE 0.131248 worst lab 0.172444 worst cell n>=3 0.225195 early/late median [0.007288 0.032113]
incomplete-coupon rate matches the observation used for drift recovery; held-out strain curves test the correction independently

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8759d826b65816ff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8759d826 (directory).
```

### Turn 6 — `submit_completed_task` (`25612 ms` LLM)
- **Rationale**: The final artifact passed the isolated schema check and end-to-end held-out-lab validation. Removing the rate correction for complete coupons improved their held-out strain MAE without changing predictions for incomplete coupons.
- **Alternatives Weighed**: `Retain plastic terms` (rejected: The largest calibrated absolute stress was 2.45101, below the 2.85 yield threshold, so those terms cannot be estimated from this episode.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/alpha_ep04_solver.py': 'FILE:5500'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_ep04_solver.py`. The public schema validator passed in isolated Python. The fitted relaxation times are 0.4994 and 3.5401.`

### Turn 7 — `execute_in_remote_docker` (`52087 ms` LLM)
- **Rationale**: Use real held-out lab measurements to detect omitted stress drift, feature interactions, stress saturation, and residual lab offsets. Screen the added terms at relaxation times fitted within each fold; only a material improvement in held-out strain or rate would justify a joint refit before submission.
- **Alternatives Weighed**: `Test classification balance, relay modes, or Fokker–Planck/HJB dynamics` (rejected: The episode has continuous strain and rate observations, with no class labels, relay state, or stochastic control process. Residual lab drift and nonlinear stress response are the relevant failure modes.)

```bash
set -e
cat > /tmp/ep04_structure.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

path = '/holdout_task/alpha_ep04_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
lab = d['train']['lab_id']
strata = d['train']['stratum']
complete = np.isfinite(d['train']['coupon_features_raw']).all(axis=1)
labels = ('base', 'stress_drift', 'pairwise_drift', 'rational_memory', 'lab_level')
pred = {k: np.empty((len(lab), len(d['eval_times']) + 1)) for k in labels}
for l in np.unique(lab):
    m = lab == l
    train = {k:v[~m] for k,v in d['train'].items()}
    test = {k:v[m] for k,v in d['train'].items() if k != 'strain_curves_obs'}
    test['residual_plastic_state'] = np.zeros(m.sum())
    q = {**d, 'train':train, 'test':test}
    tr, te, y, seen = s._prepare(q)
    eligible = seen[:len(y)].all(axis=1)
    tau, w = s._fit(tr, y, eligible)
    A, B = (z.matrix(np.log(tau)) for z in (tr,te))
    residual = y[eligible] - A[eligible] @ w
    scale = s._scale(residual)
    def extra(z, p, a):
        t = np.r_[z.times, 1.]
        x = z.linear[:, -1, 1:]
        ij = list(itertools.combinations_with_replacement(range(x.shape[1]), 2))
        pairwise = np.column_stack([x[:,i]*x[:,j] for i,j in ij])
        return dict(stress_drift=z.hold[:,None,None]*t[None,:,None],
                    pairwise_drift=pairwise[:,None,:]*t[None,:,None],
                    rational_memory=a[:,:,:2]/(1+0.25*z.hold[:,None,None]**2),
                    lab_level=(p['lab_id'][:,None,None]==np.unique(train['lab_id'])[None,None,1:])*t[None,:,None])
    ea, eb = extra(tr,train,A), extra(te,test,B)
    for name in labels:
        if name == 'base':
            p = B @ w
        else:
            a = np.concatenate([A,ea[name]],axis=2)[eligible]
            b = np.concatenate([B,eb[name]],axis=2)
            aw = (a/scale[None,:,None]).reshape(-1,a.shape[-1]); yw=(y[eligible]/scale).ravel()
            fit = least_squares(lambda v: aw@v-yw, np.r_[w,np.zeros(a.shape[-1]-len(w))], jac=lambda v: aw.copy(), loss='cauchy', f_scale=2.)
            p = b @ fit.x
        rate = test['instant_rate_obs']
        adj = np.where(~complete[m] & np.isfinite(rate), rate-p[:,-1], 0.)
        pred[name][m] = p + adj[:,None]*np.r_[te.times,1.]
obs = d['train']; curves = obs['strain_curves_obs']-obs['strain_0'][:,None]
for name,p in pred.items():
    err = p[:,:-1]-curves
    print(name)
    for cm in (complete, ~complete):
        cells = [cm&(lab==l)&(strata==k) for l in np.unique(lab) for k in np.unique(strata)]
        print('  n',cm.sum(),'curve MAE / worst cell n>=3 / median:',*[round(v,6) for v in (np.mean(abs(err[cm])),max(np.mean(abs(err[c])) for c in cells if c.sum()>=3),np.median(abs(err[cm])))], 'lab signed medians:',np.round([np.median(err[cm&(lab==l)]) for l in np.unique(lab)],6))
    print('  complete rate MAE / worst lab:',*[round(v,6) for v in (np.mean(abs(p[complete,-1]-obs['instant_rate_obs'][complete])), max(np.mean(abs(p[complete&(lab==l),-1]-obs['instant_rate_obs'][complete&(lab==l)])) for l in np.unique(lab)))])
print('test finite rates by completeness',[(bool(v), int(np.sum(m)),int(np.sum(m & np.isfinite(d['test']['instant_rate_obs'])))) for v in (True,False) for m in (np.isfinite(d['test']['coupon_features_raw']).all(axis=1)==v,)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/ep04_structure.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`880 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
base
  n 177 curve MAE / worst cell n>=3 / median: 0.081614 0.102462 0.009758 lab signed medians: [ 0.001335  0.000929 -0.0007    0.000708]
  n 23 curve MAE / worst cell n>=3 / median: 0.131248 0.225195 0.016037 lab signed medians: [-0.003754 -0.017166 -0.004779 -0.001612]
  complete rate MAE / worst lab: 0.008341 0.009673
stress_drift
  n 177 curve MAE / worst cell n>=3 / median: 0.081632 0.102472 0.009821 lab signed medians: [ 0.001702  0.000729 -0.000742  0.000644]
  n 23 curve MAE / worst cell n>=3 / median: 0.131144 0.225223 0.015827 lab signed medians: [-0.00376  -0.016908 -0.004805 -0.001483]
  complete rate MAE / worst lab: 0.008352 0.009665
pairwise_drift
  n 177 curve MAE / worst cell n>=3 / median: 0.081704 0.102309 0.009863 lab signed medians: [ 0.0014    0.001616 -0.000943  0.000482]
  n 23 curve MAE / worst cell n>=3 / median: 0.131396 0.225125 0.015901 lab signed medians: [-0.003752 -0.019251 -0.005076 -0.002028]
  complete rate MAE / worst lab: 0.008344 0.009627
rational_memory
  n 177 curve MAE / worst cell n>=3 / median: 0.081617 0.102372 0.009729 lab signed medians: [ 0.001481  0.001378 -0.000822  0.000606]
  n 23 curve MAE / worst cell n>=3 / median: 0.131707 0.22576 0.016826 lab signed medians: [-0.004017 -0.02251  -0.004427 -0.00117 ]
  complete rate MAE / worst lab: 0.008348 0.009659
lab_level
  n 177 curve MAE / worst cell n>=3 / median: 0.081648 0.102913 0.009638 lab signed medians: [ 5.890e-04  5.000e-05 -1.466e-03  6.190e-04]
  n 23 curve MAE / worst cell n>=3 / median: 0.131309 0.225149 0.01591 lab signed medians: [-0.003753 -0.017738 -0.004695 -0.001887]
  complete rate MAE / worst lab: 0.008334 0.009655
test finite rates by completeness [(True, 74, 74), (False, 6, 6)]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8759d826b65816ff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8759d826 (directory).
```

### Turn 8 — `submit_completed_task` (`30636 ms` LLM)
- **Rationale**: The verified artifact passes the public schema and held-out-lab checks. For imputed coupons, measured initial rate anchors the drift correction; independently held-out strain curves validate that correction.
- **Alternatives Weighed**: `Add stress drift, pairwise feature interactions, rational memory, or lab offsets` (rejected: None materially improved held-out strain or initial-rate error.); `Fit plastic terms` (rejected: Peak calibrated absolute stress was 2.45101, below the 2.85 yield threshold.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_ep04_solver.py': 'FILE:5500'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_ep04_solver.py`. The public schema check passed in isolated Python. The fitted relaxation times were 0.4994 and 3.5401.`

### Turn 9 — `execute_in_remote_docker` (`57701 ms` LLM)
- **Rationale**: An independent recurrence can detect an off-by-one history or rate-kernel error. Comparing calibrated and command stresses in held-out labs measures the practical effect of calibration, while a fold-local cubic challenger tests an omitted drift term without changing the verified artifact.
- **Alternatives Weighed**: `Repeat the rational-memory and lab-offset screens` (rejected: Those terms already failed to improve held-out transfer materially; the remaining checks target calibration, history indexing, and distinct three-feature products.)

```bash
set -e
cat >/tmp/ep04_calibration_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
train, test = d['train'], d['test']
labs = train['lab_id']; strata = train['stratum']; n = len(labs)
x, observed = s._features(train, test)
complete = observed[:n].all(axis=1)
laws = s._calibrate(d['calibration_load_cells'])
dt, times = d['past_dt'], d['eval_times']
cal = s._Design(train, laws, dt, times, x[:n])
raw = s._Design(train, np.tile([1., 0.], (4, 1)), dt, times, x[:n])
public = s.solve_alpha_ep04(d)
tau = public['estimated_taus']
assert np.all(np.isfinite(tau)) and 0 < tau[0] < tau[1]
for name, design in [('train', cal), ('test', s._Design(test, laws, dt, times, x[n:]))]:
    state = np.zeros((len(design.hold), 2))
    for step, h in enumerate(dt):
        q = np.exp(-h/tau)
        state = q*state + (1-q)*design.past[:, step, None]
    gap = design.hold[:, None] - state
    R = design.matrix(np.log(tau))
    np.testing.assert_allclose(R[:, :-1, :2], gap[:, None, :]*(1-np.exp(-times[:, None]/tau)), atol=1e-12)
    np.testing.assert_allclose(R[:, -1, :2], gap/tau, atol=1e-12)
    peak = max(np.abs(design.past).max(), np.abs(design.hold).max())
    print(name, 'physical peak', round(peak, 6), 'yield margin', round(d['yield_stress']-peak, 6))
    assert peak < d['yield_stress']
y = np.column_stack([train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']])
pred = {k: np.empty_like(y) for k in ['calibrated', 'command-as-physical', 'cubic-drift']}
t = np.r_[times, 1.]
# Pairwise linear sums/differences already lie in the original drift span.
# Distinct three-feature products do not, so test their transfer separately.
xtr = x[:n]
cubic = np.column_stack([np.prod(xtr[:, [i,j,k]], axis=1) for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
for lab in np.unique(labs):
    held = labs == lab
    fit = ~held & complete
    for key, z in [('calibrated', cal), ('command-as-physical', raw)]:
        ft, w = s._fit(z, y, fit)
        A = z.matrix(np.log(ft))
        p = A[held] @ w
        p += np.where(~complete[held], y[held, -1]-p[:, -1], 0.)[:, None]*t
        pred[key][held] = p
        if key != 'calibrated':
            continue
        B = np.concatenate([A, cubic[:, None, :]*t[None, :, None]], axis=2)
        scale = s._scale(A[fit] @ w-y[fit])
        M = (B[fit]/scale[:, None]).reshape(-1, B.shape[-1])
        target = (y[fit]/scale).ravel()
        q = least_squares(lambda v: M@v-target, np.r_[w,np.zeros(4)], jac=lambda v: M.copy(), loss='cauchy', f_scale=2.)
        p = B[held] @ q.x
        p += np.where(~complete[held], y[held, -1]-p[:, -1], 0.)[:, None]*t
        pred['cubic-drift'][held] = p
for key, p in pred.items():
    e = np.abs(p-y)
    for mask in (complete, ~complete):
        cells = [mask & (labs == l) & (strata == st) for l in np.unique(labs) for st in np.unique(strata)]
        worst_curve = max(e[m, :-1].mean() for m in cells if m.sum() >= 3)
        print(key, 'n', int(mask.sum()), 'curve MAE', round(e[mask, :-1].mean(), 6), 'worst-cell', round(worst_curve, 6), 'median late', round(np.median(e[mask,-2]),6))
    print(key, 'complete worst-lab rate MAE', round(max(e[complete & (labs == l), -1].mean() for l in np.unique(labs)), 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/ep04_calibration_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1151 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
train physical peak 2.45101 yield margin 0.39899
test physical peak 2.45101 yield margin 0.39899
calibrated n 177 curve MAE 0.079555 worst-cell 0.096671 median late 0.010635
calibrated n 23 curve MAE 0.13112 worst-cell 0.225587 median late 0.03233
calibrated complete worst-lab rate MAE 0.009524
command-as-physical n 177 curve MAE 0.4011 worst-cell 0.57939 median late 0.436821
command-as-physical n 23 curve MAE 1.802138 worst-cell 3.504579 median late 3.302679
command-as-physical complete worst-lab rate MAE 0.869968
cubic-drift n 177 curve MAE 0.079793 worst-cell 0.097298 median late 0.011036
cubic-drift n 23 curve MAE 0.131124 worst-cell 0.225601 median late 0.032866
cubic-drift complete worst-lab rate MAE 0.00953
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8759d826b65816ff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8759d826 (directory).
```

### Turn 10 — `submit_completed_task` (`30637 ms` LLM)
- **Rationale**: The independent recurrence matched the implemented strain and rate kernels, calibration materially improved held-out transfer, and the unchanged artifact passed the isolated public schema check.
- **Alternatives Weighed**: `Use commanded stress without calibration` (rejected: In the calibration audit, this raised worst-cell strain MAE from 0.096671 to 0.579390 for complete coupons and from 0.225587 to 3.504579 for imputed coupons.); `Add cubic drift terms` (rejected: They did not improve worst-cell strain MAE or worst-lab initial-rate MAE.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_ep04_solver.py` and verified it in isolated Python. The fitted relaxation times are 0.4994 and 3.5401. The peak calibrated stress was 2.4510, below the 2.85 yield threshold.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M94-Continual-Binding-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
