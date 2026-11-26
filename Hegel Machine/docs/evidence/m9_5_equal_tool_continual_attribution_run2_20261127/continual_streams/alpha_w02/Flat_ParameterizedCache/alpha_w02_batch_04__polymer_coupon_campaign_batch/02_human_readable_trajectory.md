# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w02_batch_04__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w02_step_04` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `464,167` (`447,616` / `136,576` / `16,551`) |
| **Total Episode Wall Time (s)** | `501.83s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_10_split_preloaded_creep_transfer`**: [BIFURCATED SUB-REGIME `missing_feature_count_with_finite_initia > 0`]: Cell-balanced imputation recovers a population feature mean, not a coupon's latent drift: on held-out labs, omitting the initial-rate correction increased the worst-lab median endpoint error from 0.098 to 1.072. Use separately calibrated stresses and reference-panel alignment by group, then fit the two preloaded exponential memories jointly to strain increments and initial rates using Cauchy loss with separate MAD scales rather than relying on pooled coordinates or a least-squares pilot for the final estimate. Preserve the fast/slow labels of the supplied memory states during optimization and reject fits whose robust, column-normalized Jacobian has a smallest-to-largest singular-value ratio below 1e-6. For a coupon with at least one missing feature and a finite measured initial rate, replace only its fitted drift with the measured rate minus the two modeled memory rates; otherwise retain the feature-based drift. The reported probe supports the in-scope strain, spectrum, secant, and crossing calculations, not cross-epoch parameter persistence or extrapolation to unseen regimes. (Counter-regime `missing_feature_count_with_finite_initia` warning: The diagnostic is the original missing-feature count when the initial-rate observation is finite, and zero otherwise. Retain feature-based drift for complete rows and for incomplete rows without a usable rate; the latter have no coupon-specific drift correction.)
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Cell-balanced feature imputation alone recovers the coupon-specific creep drift when a feature is missing.` (witness: `On held-out labs, removing the initial-rate innovation from the missing-feature drift leaves a worst-lab median endpoint error of 1.072 versus 0.098 with the correction.`) -> Pivot `Keep the required aligned features and shared memory model; update only missing-feature drift with the measured initial rate when it is available.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14797 ms` LLM)
- **Rationale**: Inspect the schema and prior solver interfaces to recover the exact constitutive basis and identify which labs need historical calibration support before writing the solver.
- **Alternatives Weighed**: `Reuse the cached load-cell coefficients unchanged` (rejected: The shared public gate already rejected that payload; inspect current and historical calibration evidence before fitting.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json; ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment robust calibration'; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_alpha_w02_batch_04.pkl','rb') as f:d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None),list(v)[:20] if isinstance(v,dict) else (repr(v)[:190] if not hasattr(v,'shape') else str(v.dtype)))
 if isinstance(v,dict):
  for a,b in v.items():print(' ',a,type(b).__name__,getattr(b,'shape',None),repr(b)[:100])
PY
for f in /tmp/hegel_continual_archive/*solver.py; do wc -l "$f"; grep '^def ' "$f"; done
```
- **Container Exit Code**: `0` (`546 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_hereditary_creep_transfer@c0d59a47 | align_reference_features(features: np.ndarray, reference: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Align features using reference coupons measured in the same group | REQ: {'field': 'max_absolute_calibrated_stress_over_yield_stress', 'op': '<', 'value': 1}; {' | --describe autogenous_hereditary_creep_transfer.align_reference_features
28 /tmp/hegel_continual_archive/alpha_w02_batch_01_solver.py
def solve_alpha_w02_batch_01(dataset: dict) -> dict:
195 /tmp/hegel_continual_archive/alpha_w02_batch_02_solver.py
def _line(x, y, tol):
def _calibration(checks, yield_stress):
def _inputs(dataset):
def _components(inputs, dt, tau, variant=0):
def _design(inputs, dt, times, tau, variant):
def _fit(inputs, dt, times, obs, variant=0):
def _hazard(initial, threshold, times, tau, amplitudes, drift):
def solve_alpha_w02_batch_02(dataset: dict) -> dict:
28 /tmp/hegel_continual_archive/alpha_w02_batch_03_solver.py
def solve_alpha_w02_batch_03(dataset: dict) -> dict:
238 /tmp/hegel_continual_archive/alpha_w02_s00_solver.py
def _checks(checks, lab):
def _consensus(x, y):
def _archive_checks(root):
def _with_prior(x, y, fits, prior):
def _calibrate(checks, root=Path('/tmp/hegel_continual_archive')):
def _features(train, test):
def _design(calibrated, feat, dt, t, logtaus, derivative=False):
def _fit(calibrated, feat, dt, t, y):
def solve_alpha_w02_s00(dataset: dict) -> dict:
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18613 ms` LLM)
- **Rationale**: Read the dataset from its actual case-list wrapper, compare each lab’s calibration rows with the archive, and recover the exact prior basis. These checks determine where archive support is warranted and which expressions can be reused.
- **Alternatives Weighed**: `Port the previous solver unchanged` (rejected: The current contract requires two additional above-yield terms; recover the prior seven-term basis before extending it.)

```bash
cd /holdout_task; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('public_alpha_w02_batch_04.pkl','rb'));print('cases',len(p))
for case in p:
 d=case['dataset']; print('case metadata',[(k,str(v)[:100]) for k,v in case.items() if k!='dataset']);
 for k,v in d.items():
  print(k,{a:(np.shape(b),str(np.asarray(b).dtype)) for a,b in v.items()} if isinstance(v,dict) else (np.shape(v),str(v)[:250] if np.size(v)<25 else ''))
 c=d['calibration_load_cells'];
 for i in range(4):
  q=c['lab_id']==i;print('checks',i,np.round(np.column_stack([c['cmd_stress'][q],c['true_stress'][q]]),6).tolist())
for name in ('archive_manifest','archive_calibration_history'):
 path='/tmp/hegel_continual_archive/'+name+'.json'; x=json.load(open(path)); print(name,type(x),str(x)[:1800]);
PY
sed -n '1,260p' /tmp/hegel_continual_archive/alpha_w02_s00_solver.py; cat /tmp/hegel_continual_archive/alpha_w02_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_hereditary_creep_transfer.align_reference_features
```
- **Container Exit Code**: `0` (`554 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
        {
          "field": "calibration_stress_scale",
          "op": ">",
          "value": 0
        },
        {
          "field": "minimum_finite_reference_rows_per_usable_column",
          "op": ">=",
          "value": 2
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_reference_features",
          "signature": "align_reference_features(features: np.ndarray, reference: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> np.ndarray",
          "doc_summary": "Align features using reference coupons measured in the same group."
        }
      ],
      "index_cards": [
        "autogenous_hereditary_creep_transfer@c0d59a47 | calibrate_affine_stress(command: np.ndarray, measured: np.ndarray, stress_scale: float, *, archive_command: np.ndarray | None=None, archive_measured: np.ndarray | None=None, inlier_fraction: float=0.001, min_span_fraction: float=0.1) -> tuple[float, float, bool] | Return (gain, physical bias, used_archive) for one load-cell group | REQ: {'field': 'max_absolute_calibrated_stress_over_yield_stress', 'op': '<', 'value': 1}; {' | --describe autogenous_hereditary_creep_transfer.calibrate_affine_stress",
        "autogenous_hereditary_creep_transfer@c0d59a47 | align_reference_features(features: np.ndarray, reference: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Align features using reference coupons measured in the same group | REQ: {'field': 'max_absolute_calibrated_stress_over_yield_stress', 'op': '<', 'value': 1}; {' | --describe autogenous_hereditary_creep_transfer.align_reference_features",
        "autogenous_hereditary_creep_transfer@c0d59a47 | fit_subyield_creep(past_stress: np.ndarray, hold_stress: np.ndarray, past_duration: np.ndarray, times: np.ndarray, features: np.ndarray, strain0: np.ndarray, strain_obs: np.ndarray, rate_obs: np.ndarray, yield_stress: float, tau_bounds: tuple[float, float], *, rank_rtol: float=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Fit two exponential memories plus affine feature-dependent creep drift | REQ: {'field': 'max_absolute_calibrated_stress_over_yield_stress', 'op': '<', 'value': 1}; {' | --describe autogenous_hereditary_creep_transfer.fit_subyield_creep"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_hereditary_creep_transfer",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`16605 ms` LLM)
- **Rationale**: Recover the exact hereditary kernels and inspect observed cell coverage and archive layout. This determines how to implement the derivative, balance missing feature values, and pool sparse calibration checks without using public ground truth for fitting.
- **Alternatives Weighed**: `Use the archived subyield skill directly` (rejected: Its applicability excludes above-yield inputs; the prior solver’s hereditary basis can be extended with the two specified plastic terms.)

```bash
a=/tmp/hegel_continual_archive; sed -n '/^def _features/,/^def solve/p' "$a/alpha_w02_s00_solver.py"; sed -n '/^def _components/,/^def _hazard/p' "$a/alpha_w02_batch_02_solver.py"; python3 - <<'PY'
import json,numpy as np,pickle
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; a=json.load(open(p)); print('history',type(a),str(a)[:650]); print('manifest episodes',[(r.get('episode_short_name'),r.get('solver_path')) for r in json.load(open('/tmp/hegel_continual_archive/archive_manifest.json'))]); d=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0]['dataset'];
for s in ['train','test']:
 r=d[s]; print(s,'feature/ref nan',np.isnan(r['coupon_features_raw']).sum(),np.isnan(r['ref_coupons']).sum(),'cells',np.unique(np.stack([r['lab_id'],r['stratum']],1),axis=0,return_counts=True),'target ranges',[(k,np.nanmin(v),np.nanmax(v)) for k,v in r.items() if k in ['instant_rate_obs','strain_curves_obs','residual_plastic_state']]);
PY
```
- **Container Exit Code**: `0` (`578 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
f.cost)
    r = np.full(obs.shape, np.nan)
    r[valid] = fit_at(best.x, False)[1]/w
    mad = lambda a: 1.4826*np.nanmedian(abs(a-np.nanmedian(a)))
    curve_scale = max(mad(r[:, :-1]), 1e-8)
    rate_scale = max(mad(r[:, -1]), curve_scale/3)
    scale = np.broadcast_to(np.r_[np.full(len(times), curve_scale), rate_scale], obs.shape)
    w = (1/scale)[valid]
    opt = least_squares(lambda lt: fit_at(lt, True)[1], best.x, bounds=limits, loss='cauchy', f_scale=2.5)
    tau = np.exp(opt.x)
    coef = fit_at(opt.x, True)[0]
    order = np.argsort(tau)
    return tau[order], np.r_[coef[:2][order], coef[2:]]


def _hazard(initial, threshold, times, tau, amplitudes, drift):
history <class 'list'> [{'episode_short_name': 'alpha_w02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.279777, 1.251072, -0.29186, 1.446319, 0.352542, 0.351338, -1.086733, 2.028137, 1.367536, 1.715104, 0.683447, -1.593154, -0.298087, 0.957208, 1.021057, 0.916675, -0.456617, -1.663793, 1.130581, -2.062901, 0.648259, -0.861173, -0.010596, 1.961708, 1.884995, -0.790091, 1.806036, 1.498177, -0.11913, -1.191317, -1.574003, 1.188772, 1.145896, -1.262392, 0.176833, -0.909595, -1.112766, -0.725082, -0.659505, -0.093528], 'true_stress': [0.279846, 1.251147, -0.29172
manifest episodes [('alpha_w02_s00', '/tmp/hegel_continual_archive/alpha_w02_s00_solver.py'), ('alpha_w02_batch_01', '/tmp/hegel_continual_archive/alpha_w02_batch_01_solver.py'), ('alpha_w02_batch_02', '/tmp/hegel_continual_archive/alpha_w02_batch_02_solver.py'), ('alpha_w02_batch_03', '/tmp/hegel_continual_archive/alpha_w02_batch_03_solver.py')]
train feature/ref nan 21 242 cells (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([18, 13, 14,  8, 19, 20, 16, 11, 19, 21, 15, 26])) target ranges [('instant_rate_obs', -12.446464653808329, 7.0430487368226755), ('strain_curves_obs', -9.180622386926307, 7.494665073865294)]
test feature/ref nan 8 101 cells (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([ 7,  7,  3,  5, 15,  4,  1, 10,  9,  5,  9,  5])) target ranges [('instant_rate_obs', -5.43350370879176, 5.209473168597526), ('residual_plastic_state', 0.0, 0.0)]
```

### Turn 4 — `execute_in_remote_docker` (`69981 ms` LLM)
- **Rationale**: Build the specified solver, verify the output contract in a clean process, and compare calibration consensus and inferred parameters against public evidence. The diagnostics will show whether the next step needs calibration refinement or only held-out stress tests.
- **Alternatives Weighed**: `Pool every calibration sweep` (rejected: Historical sweeps may reflect changing lab gains and biases. Use them as fallback candidates, not as interchangeable current measurements.)

```bash
cat > /holdout_task/alpha_w02_batch_04_solver.py <<'PY'
"""Profile the linear constitutive coefficients at two relaxation times."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _lab_checks(checks, lab):
    x, y = (np.asarray(checks[k], float) for k in ('cmd_stress', 'true_stress'))
    take = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _line(x, y, tol, prior=None):
    i, j = np.triu_indices(x.size, 1)
    usable = abs(x[j] - x[i]) >= 4 * tol
    i, j = i[usable], j[usable]
    gains = (y[j] - y[i]) / (x[j] - x[i])
    lines = np.c_[gains, y[i] - gains * x[i]]
    if prior is not None:
        lines = np.vstack((lines, prior))
    lines = lines[lines[:, 0] > 0]
    if not lines.size:
        return None, np.zeros(x.size, bool)
    r = abs(y - lines[:, :1] * x - lines[:, 1:])
    cost = np.minimum((r / tol)**2, 1).sum(axis=1)
    if prior is not None:
        cost += .025 * np.minimum(((lines - prior)**2).sum(axis=1) / tol**2, 1)
    mask = r[np.argmin(cost)] < tol
    if mask.sum() >= 2 and np.ptp(x[mask]) >= 4 * tol:
        fit = np.linalg.lstsq(np.c_[x[mask], np.ones(mask.sum())], y[mask], rcond=None)[0]
        return fit, mask
    return None, mask


def _calibrate(checks, sy):
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
    fits = []
    tol = .045 * sy
    for lab in range(4):
        x, y = _lab_checks(checks, lab)
        fit, mask = _line(x, y, tol)
        if fit is not None and mask.sum() >= 3 and np.ptp(x[mask]) >= .55 * sy:
            fits.append(fit)
            continue
        priors = []
        for episode in history:
            xx, yy = _lab_checks(episode, lab)
            p, m = _line(xx, yy, tol)
            if p is not None and m.sum() >= 3 and np.ptp(xx[m]) >= .55 * sy:
                priors.append(p)
        if priors:
            priors = np.asarray(priors)
            if x.size:
                r = abs(y - priors[:, :1] * x - priors[:, 1:])
                loss = np.minimum((r / tol)**2, 4).mean(axis=1)
                prior = priors[np.argmin(loss + .03 * np.arange(len(priors))[::-1])]
            else:
                prior = priors[-1]
            if fit is None or mask.sum() <= 2:
                if x.size:
                    delta = y - prior[0] * x - prior[1]
                    consistent = abs(delta) < 2 * tol
                    if consistent.any():
                        prior = prior + [0, np.median(delta[consistent])]
                fit = prior
        fits.append(fit if fit is not None else np.array([1., np.median(y - x) if x.size else 0.]))
    return np.asarray(fits)


def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    feat = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        m = labs == lab
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[m] = (feat[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    _, cell = np.unique(np.c_[labs, strata], axis=0, return_inverse=True)
    cell = cell.ravel()
    finite = np.isfinite(feat)
    means = []
    for k in np.unique(cell):
        rows = cell == k
        count = finite[rows].sum(axis=0)
        total = np.where(finite[rows], feat[rows], 0.).sum(axis=0)
        means.append(np.divide(total, count, out=np.full(feat.shape[1], np.nan), where=count > 0))
    means = np.array(means)
    count = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), count, out=np.zeros(feat.shape[1]), where=count > 0)
    return np.c_[np.ones(len(feat)), np.where(finite, feat, fill)]


def _inputs(split, feat, cal, dt, sy):
    gain, bias = cal[np.asarray(split['lab_id'], int)].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    ov = np.maximum(abs(hold) - sy, 0)**2 * np.sign(hold)
    work = np.sum(np.maximum(abs(past) - sy, 0)**2 * dt, axis=1) * hold
    return past, hold, np.c_[feat, ov], work


def _design(inputs, dt, t, tau):
    past, hold, drift, work = inputs
    end = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    mem = np.exp(-end[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ mem
    response = -np.expm1(-t[:, None] / tau)
    curve = np.concatenate((delta[:, None, :] * response,
                            drift[:, None, :] * t[None, :, None],
                            (work[:, None] * response[:, 0])[:, :, None]), axis=2)
    rate = np.c_[delta / tau, drift, work / tau[0]]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _mad(x):
    return 1.4826 * np.nanmedian(abs(x - np.nanmedian(x)))


def _fit(inputs, dt, t, y):
    valid = np.isfinite(y)
    raw = y[valid]
    active = np.ones(9, bool)
    active[7] = np.any(inputs[2][:, -1])
    active[8] = np.any(inputs[3])
    weights = np.broadcast_to(np.r_[np.ones(len(t)), 1 / 1.5], y.shape)[valid]
    b = raw * weights

    def profile(z):
        tau = np.sort(np.exp(z))
        a = _design(inputs, dt, t, tau)[valid][:, active] * weights[:, None]
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        return coef, (a * coef).sum(axis=1) - b

    limits = np.log(([.051, .051], [24.99, 24.99]))
    starts = ((.26, 1.6), (.64, 3.8), (1.5, 9.))
    pilot = min((least_squares(lambda z: profile(z)[1], np.log(start), bounds=limits)
                 for start in starts), key=lambda fit: fit.cost)
    r = np.full(y.shape, np.nan)
    r[valid] = profile(pilot.x)[1] / weights
    curve_scale = max(_mad(r[:, :-1]), 1e-9)
    rate_scale = _mad(r[:, -1]) if valid[:, -1].any() else curve_scale
    scale = np.r_[np.full(len(t), curve_scale), max(rate_scale, curve_scale / 4)]
    base_weights = np.broadcast_to(1 / scale, y.shape)[valid]
    z = pilot.x
    for _ in range(5):
        resid = profile(z)[1] / weights
        weights = base_weights / np.sqrt(1 + (base_weights * resid / 3)**2)
        b = raw * weights
        z = least_squares(lambda v: profile(v)[1], z, bounds=limits).x
    coef = np.zeros(9)
    coef[active] = profile(z)[0]
    return np.sort(np.exp(z)), coef


def solve_alpha_w02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    sy = float(dataset['yield_stress'])
    cal = _calibrate(dataset['calibration_load_cells'], sy)
    feat = _features(train, test)
    n = len(train['lab_id'])
    y = np.c_[train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']]
    tau, coef = _fit(_inputs(train, feat[:n], cal, dt, sy), dt, t, y)
    pred = _design(_inputs(test, feat[n:], cal, dt, sy), dt, t, tau) @ coef
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], viscoplastic_weights=coef[7:],
                predicted_strain=pred[:, :-1] + np.asarray(test['strain_0'])[:, None], initial_rate_pred=pred[:, -1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,json,numpy as np
sys.path.insert(0,'/holdout_task');import alpha_w02_batch_04_solver as s
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0];d=p['dataset'];sy=d['yield_stress'];o=s.solve_alpha_w02_batch_04(d)
for name,c in [(h['episode_short_name'],h) for h in json.loads(s._ARCHIVE.read_text())]+[('current',d['calibration_load_cells'])]:
 print(name);print([(i,len(x),np.round(f,6).tolist() if f is not None else None,int(m.sum())) for i in range(4) for x,y in [s._lab_checks(c,i)] for f,m in [s._line(x,y,.045*sy)]])
print('outputs',[(k,v) for k,v in o.items() if v.ndim==1 and len(v)<10]);print('truth schema', {k:np.shape(v) for k,v in p['gt'].items()});print('truth small', {k:v for k,v in p['gt'].items() if 0<np.size(v)<12})
PY
```
- **Container Exit Code**: `0` (`720 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
alpha_w02_s00
[(0, 10, [0.999991, 6e-05], 10), (1, 10, [0.999932, -7e-06], 10), (2, 10, [1.000028, 1.4e-05], 10), (3, 10, [0.999934, 5.4e-05], 10)]
alpha_w02_batch_01
[(0, 10, [1.000024, 1.9e-05], 10), (1, 10, [0.999991, 1.1e-05], 10), (2, 10, [1.000038, 8e-06], 10), (3, 10, [1.000015, 6.4e-05], 10)]
alpha_w02_batch_02
[(0, 10, [0.999973, 4e-05], 10), (1, 10, [0.999988, -2.1e-05], 10), (2, 10, [0.999995, -1.4e-05], 10), (3, 10, [1.000003, -3.5e-05], 10)]
alpha_w02_batch_03
[(0, 10, [1.283546, 0.117218], 10), (1, 10, [0.792701, -0.142684], 10), (2, 10, [1.236294, 0.132197], 10), (3, 10, [0.763082, -0.144431], 10)]
current
[(0, 10, [1.283477, 0.117185], 10), (1, 10, [0.792721, -0.142687], 10), (2, 10, [1.23634, 0.132202], 10), (3, 10, [0.763059, -0.144368], 10)]
outputs [('estimated_taus', array([0.57932799, 3.57235394])), ('hereditary_weights', array([ 0.70573033,  0.5694825 ,  0.07981253,  0.14955235, -0.12999609,
        0.10976266, -0.09047238])), ('viscoplastic_weights', array([0.24986694, 0.16885329]))]
truth schema {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (80, 10), 'te_true_rate0': (80,), 'te_cross': (80,), 'te_secant': (80,), 'te_hold_true': (80,), 'te_hist_true': (80, 18)}
truth small {'taus': array([0.582, 3.505]), 'theta': array([ 0.708,  0.561,  0.08 ,  0.15 , -0.13 ,  0.11 , -0.09 ]), 'vp_theta': array([0.25, 0.17]), 'stress_gains': array([1.2835, 0.7927, 1.2363, 0.7631]), 'stress_biases': array([ 0.1172, -0.1427,  0.1322, -0.1444])}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d149b3216e5f3c99 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d149b321 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`36710 ms` LLM)
- **Rationale**: Measure strain and rate errors separately by lab, stress regime, and horizon. Degrade only calibration rows to isolate fallback failures; if prediction error rises materially, revise the calibration decision rather than changing the constitutive law.
- **Alternatives Weighed**: `Submit after the public schema check` (rejected: The schema does not detect a wrong onset derivative or unstable calibration when a wing has only narrow, sparse, or contaminated spot checks.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
path='/holdout_task/alpha_w02_batch_04_solver.py'
spec=importlib.util.spec_from_file_location('solver',path); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; gt=p['gt']; sy=d['yield_stress']; dt=d['past_dt']; t=d['eval_times']; train=d['train']; test=d['test']; checks=d['calibration_load_cells']; n=len(train['lab_id']); cal=s._calibrate(checks,sy)
rmse=lambda x: np.sqrt(np.mean(x*x))
cross=lambda split: (abs(split['stress_hold_cmd']*cal[split['lab_id'],0]+cal[split['lab_id'],1])>sy) | np.any(abs(split['past_stress_cmd']*cal[split['lab_id'],0,None]+cal[split['lab_id'],1,None])>sy,axis=1)
def report(out, tag):
 e=out['predicted_strain']-gt['te_clean']; r=out['initial_rate_pred']-gt['te_true_rate0']; masks={'all':np.ones(len(r),bool),'above':cross(test),'below':~cross(test)}; masks.update({f'lab{i}':test['lab_id']==i for i in range(4)}); print(tag,{k:(int(m.sum()),round(rmse(e[m]),6),round(rmse(r[m]),6)) for k,m in masks.items() if m.any()},'horizon_max',round(max(rmse(e[:,j]) for j in range(e.shape[1])),6))
report(s.solve_alpha_w02_batch_04(d),'public')
feat=s._features(train,test)[:n]; y=np.c_[train['strain_curves_obs']-train['strain_0'][:,None],train['instant_rate_obs']]
for lab in range(4):
 m=train['lab_id']==lab; inputs=s._inputs(train,feat,cal,dt,sy); a=s._fit(tuple(v[~m] for v in inputs),dt,t,y[~m]); resid=(s._design(tuple(v[m] for v in inputs),dt,t,a[0])@a[1])-y[m]; print('leave_lab_out',lab,'tau',np.round(a[0],5),'curve_rmse',round(rmse(resid[:,:-1]),6),'rate_rmse',round(rmse(resid[:,-1]),6))
for wing in ((0,1),(2,3)):
 for scenario in ('spot_only','spot_plus_emi','three_narrow','no_checks'):
  c={k:v.copy() for k,v in checks.items()}; keep=~np.isin(c['lab_id'],wing)
  for lab in wing:
   ii=np.flatnonzero(c['lab_id']==lab); ii=ii[np.argsort(abs(c['cmd_stress'][ii]))];
   if scenario=='spot_only': keep[ii[:1]]=True
   if scenario=='spot_plus_emi': keep[ii[:2]]=True; c['true_stress'][ii[1]]+=.45
   if scenario=='three_narrow':
    idx=np.argsort(c['cmd_stress'][ii]); xs=c['cmd_stress'][ii[idx]]; j=np.argmin(xs[2:]-xs[:-2]); keep[ii[idx[j:j+3]]]=True
  c={k:v[keep] for k,v in c.items()}; altered=dict(d,calibration_load_cells=c); cc=s._calibrate(c,sy); print('calibration_delta',wing,scenario,np.round((cc-cal)[list(wing)],5)); report(s.solve_alpha_w02_batch_04(altered),str((wing,scenario)))
# Isolated genuinely subyield measurements: inactive columns must stay exactly zero.
inputs=s._inputs(train,feat,cal,dt,sy); below=~cross(train); tb,cb=s._fit(tuple(v[below] for v in inputs),dt,t,y[below]); assert np.array_equal(cb[7:],np.zeros(2)); print('subyield_rows',below.sum(),'taus',tb,'vp',cb[7:])
# Check the stated onset derivative independently against the time-domain expression.
inputs=s._inputs(test,s._features(train,test)[n:],cal,dt,sy); out=s.solve_alpha_w02_batch_04(d); w=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]; eps=1e-7; A=s._design(inputs,dt,np.array([0.,eps]),out['estimated_taus']); err=np.max(np.abs(((A[:,1]-A[:,0])/eps-A[:,2])@w)); assert err < 5e-6; print('onset_derivative_max_abs',err)
PY
```
- **Container Exit Code**: `0` (`1344 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1) three_narrow [[-3.0e-04 -2.7e-04]
 [-2.0e-05 -7.0e-05]]
((0, 1), 'three_narrow') {'all': (80, 0.109579, 0.034794), 'above': (40, 0.129775, 0.041291), 'below': (40, 0.084699, 0.026765), 'lab0': (17, 0.113223, 0.036468), 'lab1': (24, 0.119148, 0.037994), 'lab2': (20, 0.1416, 0.044422), 'lab3': (19, 0.007108, 0.002628)} horizon_max 0.181019
calibration_delta (0, 1) no_checks [[ 7.e-05  3.e-05]
 [-2.e-05  0.e+00]]
((0, 1), 'no_checks') {'all': (80, 0.109504, 0.034668), 'above': (40, 0.129643, 0.041051), 'below': (40, 0.084704, 0.026805), 'lab0': (17, 0.11299, 0.035983), 'lab1': (24, 0.119263, 0.038111), 'lab2': (20, 0.141414, 0.044227), 'lab3': (19, 0.006969, 0.002881)} horizon_max 0.180916
calibration_delta (2, 3) spot_only [[-5.e-05  4.e-05]
 [ 2.e-05 -4.e-05]]
((2, 3), 'spot_only') {'all': (80, 0.109481, 0.034633), 'above': (40, 0.1296, 0.040991), 'below': (40, 0.08471, 0.026807), 'lab0': (17, 0.113043, 0.036072), 'lab1': (24, 0.119226, 0.038044), 'lab2': (20, 0.141344, 0.044138), 'lab3': (19, 0.006949, 0.002687)} horizon_max 0.180886
calibration_delta (2, 3) spot_plus_emi [[-5.e-05  4.e-05]
 [ 2.e-05 -4.e-05]]
((2, 3), 'spot_plus_emi') {'all': (80, 0.109481, 0.034633), 'above': (40, 0.1296, 0.040991), 'below': (40, 0.08471, 0.026807), 'lab0': (17, 0.113043, 0.036072), 'lab1': (24, 0.119226, 0.038044), 'lab2': (20, 0.141344, 0.044138), 'lab3': (19, 0.006949, 0.002687)} horizon_max 0.180886
calibration_delta (2, 3) three_narrow [[ 2.3e-04  6.0e-05]
 [ 4.0e-05 -3.0e-05]]
((2, 3), 'three_narrow') {'all': (80, 0.109648, 0.034867), 'above': (40, 0.129875, 0.041404), 'below': (40, 0.084721, 0.026777), 'lab0': (17, 0.113148, 0.03625), 'lab1': (24, 0.119275, 0.038147), 'lab2': (20, 0.141736, 0.044623), 'lab3': (19, 0.007074, 0.002952)} horizon_max 0.181112
calibration_delta (2, 3) no_checks [[-5.e-05 -0.e+00]
 [ 2.e-05 -6.e-05]]
((2, 3), 'no_checks') {'all': (80, 0.109481, 0.034637), 'above': (40, 0.129606, 0.040998), 'below': (40, 0.084702, 0.026805), 'lab0': (17, 0.113043, 0.036071), 'lab1': (24, 0.119218, 0.038041), 'lab2': (20, 0.141354, 0.044152), 'lab3': (19, 0.006939, 0.002689)} horizon_max 0.180885
subyield_rows 100 taus [0.58114657 3.49517235] vp [0. 0.]
onset_derivative_max_abs 4.634424685887638e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d149b3216e5f3c99 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d149b321 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`23297 ms` LLM)
- **Rationale**: Separate complete from imputed coupons and test whether the residual follows the model’s linear-in-time drift. The supplied test onset observations are evaluated as a diagnostic, not substituted into the solver; this distinguishes an imputation limit from a wrong constitutive term.
- **Alternatives Weighed**: `Retune load-cell calibration` (rejected: Sparse and EMI stress tests barely changed prediction errors, so calibration is unlikely to explain the remaining lab gap.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('s','/holdout_task/alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; o=s.solve_alpha_w02_batch_04(d); te=d['test']; t=d['eval_times']; mask=~np.isfinite(te['coupon_features_raw']).all(axis=1)
E=o['predicted_strain']-g['te_clean']; R=o['initial_rate_pred']-g['te_true_rate0']; innov=te['instant_rate_obs']-o['initial_rate_pred']; obs_error=te['instant_rate_obs']-g['te_true_rate0']; rmse=lambda a: np.sqrt(np.mean(a*a))
for label,m in [('complete',~mask),('missing',mask)]:
 print(label,'n',m.sum(),'curve',rmse(E[m]),'rate',rmse(R[m]),'observed_rate',rmse(obs_error[m]),'innovation_max',max(abs(innov[m])),'curve_after_exact_drift_correction',rmse(E[m]-R[m,None]*t),'curve_after_observed_drift_correction',rmse(E[m]+innov[m,None]*t))
print('missing rows: lab, missing columns, predicted rate error, observed rate error, final strain error')
for i in np.flatnonzero(mask): print(te['lab_id'][i],np.flatnonzero(~np.isfinite(te['coupon_features_raw'][i])).tolist(),np.round([R[i],obs_error[i],E[i,-1]],6).tolist())
print('test rate missing count',np.sum(~np.isfinite(te['instant_rate_obs'])))
PY
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete n 72 curve 0.009758597694380134 rate 0.004003554600777712 observed_rate 0.011031245438880647 innovation_max 0.03242563733672843 curve_after_exact_drift_correction 0.008439377294556086 curve_after_observed_drift_correction 0.034789610386321494
missing n 8 curve 0.34505780567047994 rate 0.10898216820329068 observed_rate 0.009919134479186026 innovation_max 0.16466075497752597 curve_after_exact_drift_correction 0.006284392357231148 curve_after_observed_drift_correction 0.03393635734886713
missing rows: lab, missing columns, predicted rate error, observed rate error, final strain error
0 [0] [0.124369, -0.000264, 0.643519]
2 [3] [0.003796, 0.000393, 0.034224]
1 [0] [-0.111447, -0.012097, -0.568061]
2 [3] [0.149847, 0.009, 0.785936]
3 [3] [0.000168, 0.004598, 0.016241]
2 [0] [-0.127531, -0.016867, -0.685355]
1 [0] [-0.148814, 0.015847, -0.777585]
0 [0] [-0.079059, -0.001667, -0.414971]
test rate missing count 0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d149b3216e5f3c99 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d149b321 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`17822 ms` LLM)
- **Rationale**: Check the published asset descriptions and prior inference interface before choosing a refinement. If they do not establish test onset observations as valid inputs, retain feature-only prediction and test whether complete-coupon fitting improves parameter recovery without changing the required imputation.
- **Alternatives Weighed**: `Use the test onset observation to correct creep drift` (rejected: Its presence in the public pickle does not establish that it is an authorized inference input, especially when onset rate is itself a prediction target.)

```bash
cd /holdout_task; python3 -I - <<'PY'
import json,pickle,numpy as np
x=json.load(open('public_asset_index.json'))
print(json.dumps(x,indent=2)[:8500]); d=pickle.load(open('public_alpha_w02_batch_04.pkl','rb'))[0]['dataset'];
for k in ('train','test'): print(k, [(a,np.shape(b)) for a,b in d[k].items()]);
PY
sed -n '/^def solve_alpha_w02_s00/,$p' /tmp/hegel_continual_archive/alpha_w02_s00_solver.py
```
- **Container Exit Code**: `0` (`584 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ype": "float64",
              "shape": [
                80
              ],
              "type": "ndarray"
            },
            "te_true_rate0": {
              "dtype": "float64",
              "shape": [
                80
              ],
              "type": "ndarray"
            },
            "theta": {
              "dtype": "float64",
              "shape": [
                7
              ],
              "type": "ndarray"
            },
            "vp_theta": {
              "dtype": "float64",
              "shape": [
                2
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1292
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
train [('lab_id', (200,)), ('stratum', (200,)), ('coupon_features_raw', (200, 4)), ('ref_coupons', (200, 32)), ('past_stress_cmd', (200, 18)), ('strain_0', (200,)), ('stress_hold_cmd', (200,)), ('instant_rate_obs', (200,)), ('strain_curves_obs', (200, 10))]
test [('lab_id', (80,)), ('stratum', (80,)), ('coupon_features_raw', (80, 4)), ('ref_coupons', (80, 32)), ('past_stress_cmd', (80, 18)), ('strain_0', (80,)), ('stress_hold_cmd', (80,)), ('instant_rate_obs', (80,)), ('residual_plastic_state', (80,))]
def solve_alpha_w02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    calib = _calibrate(dataset['calibration_load_cells'])
    X = _features(train, test)
    n = len(train['lab_id'])

    def stress(split):
        gain, bias = calib[split['lab_id']].T
        return split['past_stress_cmd']*gain[:, None]+bias[:, None], split['stress_hold_cmd']*gain+bias

    y = np.c_[train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']]
    taus, coef = _fit(stress(train), X[:n], dt, time, y)
    pred = _design(stress(test), X[n:], dt, time, np.log(taus)) @ coef
    return {'estimated_taus': taus, 'hereditary_weights': coef, 'viscoplastic_weights': np.zeros(2),
            'predicted_strain': pred[:, :-1]+test['strain_0'][:, None], 'initial_rate_pred': pred[:, -1]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d149b3216e5f3c99 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d149b321 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`34897 ms` LLM)
- **Rationale**: Compare complete-coupon and all-coupon fits in identical leave-one-lab-out folds. Separately test three-check sweeps with one EMI glitch: if the fit accepts a spurious line, prefer archive agreement on sparse sweeps. Neither test changes the solver yet.
- **Alternatives Weighed**: `Correct the predictions directly with test onset measurements` (rejected: The asset index confirms the field exists but does not authorize using a noisy observation of the prediction target as an inference input.)

```bash
cat > /tmp/check_alpha04_fit.py <<'PY'
import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('s','/holdout_task/alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress']; c=d['calibration_load_cells']; cal=s._calibrate(c,sy)
n=len(tr['lab_id']); feat=s._features(tr,te); xx=s._inputs(tr,feat[:n],cal,dt,sy); xt=s._inputs(te,feat[n:],cal,dt,sy)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; g=p['gt']; yt=np.c_[g['te_clean']-te['strain_0'][:,None],g['te_true_rate0']]
complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); complete_te=np.isfinite(te['coupon_features_raw']).all(axis=1)
print('training coverage',[(lab,int((tr['lab_id']==lab).sum()),int(((tr['lab_id']==lab)&complete).sum())) for lab in range(4)])
def rmse(e):
 return np.sqrt(np.nanmean(e**2))
def metrics(e):
 return np.round([rmse(e[:,:-1]),rmse(e[:,-1])],6).tolist()
def subset(x,m):
 return tuple(a[m] for a in x)
for name, eligible in [('all',np.ones(n,bool)),('complete',complete)]:
 pred=np.empty_like(y)
 for lab in range(4):
  m=tr['lab_id']==lab; fit=~m&eligible
  tau,w=s._fit(subset(xx,fit),dt,t,y[fit]); pred[m]=s._design(subset(xx,m),dt,t,tau)@w
 err=pred-y
 print(name,'LOLO by lab / complete',[(i,metrics(err[tr['lab_id']==i]),metrics(err[(tr['lab_id']==i)&complete])) for i in range(4)])
 tau,w=s._fit(subset(xx,eligible),dt,t,y[eligible]); err=s._design(xt,dt,t,tau)@w-yt
 print(name,'tau',tau,'coef',w,'public complete/missing',metrics(err[complete_te]),metrics(err[~complete_te]),'public by lab',[metrics(err[te['lab_id']==i]) for i in range(4)])
# Three spot checks can include one glitch; test whether historical agreement rejects spurious exact pairs.
for lab in range(4):
 ii=np.flatnonzero(c['lab_id']==lab); ii=ii[np.argsort(c['cmd_stress'][ii])]
 scenarios={}
 for inds in (ii[[0,len(ii)//2,-1]],ii[[3,4,5]]):
  for bad in inds:
   for glitch in (-.22,.22):
    keep=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),inds)
    cc={k:v.copy() for k,v in c.items()}; cc['true_stress'][bad]+=glitch; cc={k:v[keep] for k,v in cc.items()}
    got=s._calibrate(cc,sy)
    x=np.r_[tr['past_stress_cmd'][tr['lab_id']==lab].ravel(), tr['stress_hold_cmd'][tr['lab_id']==lab]]
    delta=(got-cal)[lab]; err=np.max(abs(x*delta[0]+delta[1])); scenarios[(tuple(map(int,inds)),int(bad),glitch)]=err
 key=max(scenarios,key=scenarios.get)
 print('three-spot EMI',lab,'worst physical stress delta',round(scenarios[key],6),'case',key)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_alpha04_fit.py
```
- **Container Exit Code**: `0` (`899 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
training coverage [(0, 45, 41), (1, 47, 41), (2, 46, 41), (3, 62, 56)]
all LOLO by lab / complete [(0, [0.337003, 0.025153], [0.326602, 0.01347]), (1, [0.311848, 0.044797], [0.258806, 0.012795]), (2, [0.30226, 0.048231], [0.277739, 0.012229]), (3, [0.294027, 0.04643], [0.26922, 0.010083])]
all tau [0.57932799 3.57235394] coef [ 0.70573033  0.5694825   0.07981253  0.14955235 -0.12999609  0.10976266
 -0.09047238  0.24986694  0.16885329] public complete/missing [0.009759, 0.004004] [0.345058, 0.108982] public by lab [[0.113046, 0.036079], [0.119223, 0.038042], [0.141431, 0.044247], [0.006999, 0.002847]]
complete LOLO by lab / complete [(0, [0.337017, 0.025344], [0.326619, 0.013105]), (1, [0.311515, 0.044044], [0.258615, 0.012128]), (2, [0.302529, 0.048485], [0.278077, 0.01139]), (3, [0.293878, 0.046668], [0.269059, 0.009904])]
complete tau [0.58116555 3.54626982] coef [ 0.70779334  0.56206604  0.0796391   0.1495509  -0.12999588  0.10962579
 -0.09007202  0.24986103  0.16975416] public complete/missing [0.00917, 0.002799] [0.346213, 0.109886] public by lab [[0.113703, 0.0361], [0.119455, 0.037871], [0.141788, 0.045016], [0.005786, 0.001751]]
three-spot EMI 0 worst physical stress delta 0.677576 case ((9, 2, 4), 9, -0.22)
three-spot EMI 1 worst physical stress delta 2.453539 case ((11, 16, 12), 12, -0.22)
three-spot EMI 2 worst physical stress delta 1.423466 case ((20, 24, 25), 25, -0.22)
three-spot EMI 3 worst physical stress delta 0.00021 case ((33, 35, 38), 35, -0.22)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d149b3216e5f3c99 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d149b321 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`34444 ms` LLM)
- **Rationale**: Fix the demonstrated calibration failure and rerun the same EMI cases. Use complete training coupons when they cover the active terms: leave-one-lab-out rate RMSE improved on complete coupons in all four labs, while the required inference imputation stays unchanged.
- **Alternatives Weighed**: `Keep the three-point rule and narrow the residual threshold` (rejected: An erroneous pair can fit exactly; a smaller residual threshold alone does not identify the physical line.); `Use observed test onset to correct imputed drift` (rejected: The task does not establish that an observation of the prediction target is an authorized inference input.)
- **ATMS NOGOOD DECLARED**: Refuted `Three nominally consistent readings can establish the current gain and bias without checking an available archive.` | Witness: `One −0.22 EMI perturbation on a three-check lab 1 subset caused a maximum physical stress error of 2.453539 under the baseline.` | Blanket Diagnosis: `At short command spans, a contaminated reading can rotate the affine fit while keeping check residuals small; extrapolation then magnifies the gain error.` | Sublated Pivot: `Use archive agreement to resolve sparse checks, while allowing broad current sweeps to establish a changed instrument regime.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_w02_batch_04_solver.py')
s = p.read_text()
a = s.index('        if fit is not None and mask.sum() >= 3')
b = s.index('    return np.asarray(fits)', a)
s = s[:a] + '''        broad = fit is not None and mask.sum() >= 3 and np.ptp(x[mask]) >= .55 * sy
        # Four inliers across a useful span establish a current sweep without a prior.
        if broad and mask.sum() >= 4:
            fits.append(fit)
            continue
        priors = []
        for episode in history:
            xx, yy = _lab_checks(episode, lab)
            p, m = _line(xx, yy, tol)
            if p is not None and m.sum() >= 3 and np.ptp(xx[m]) >= .55 * sy:
                priors.append(p)
        if priors:
            priors = np.asarray(priors)
            if x.size:
                delta = y - priors[:, :1] * x - priors[:, 1:]
                # Archived sweeps in the same instrument regime agree much more
                # closely than the broad tolerance used to reject EMI.
                tight = .004 * sy
                count = (abs(delta) < tight).sum(axis=1)
                cost = np.minimum((delta / tight)**2, 1).sum(axis=1)
                best = np.lexsort((-np.arange(len(priors)), cost, -count))[0]
                prior = priors[best]
                good = abs(delta[best]) < tight
                if good.any() and (not broad or good.sum() >= min(2, x.size)):
                    fit = prior + [0, np.median(delta[best, good])]
                elif not broad:
                    # Without a wide current sweep, do not extrapolate a gain
                    # estimated from close command values or shift on an EMI point.
                    fit = priors[-1]
            else:
                fit = priors[-1]
        fits.append(fit if fit is not None else np.array([1., np.median(y - x) if x.size else 0.]))
''' + s[b:]
a = s.index('    tau, coef = _fit(_inputs(train,')
b = s.index('    pred = ',a)
s = s[:a] + '''    inputs = _inputs(train, feat[:n], cal, dt, sy)
    # Mean imputation is for prediction, not a measured material covariate.
    # Keep the full data if complete coupons do not cover an active plastic term.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1) & np.isfinite(y).any(axis=1)
    if (complete.sum() >= 9 and
            all(not np.any(v) or np.any(v[complete]) for v in (inputs[2][:, -1], inputs[3]))):
        inputs, y = tuple(v[complete] for v in inputs), y[complete]
    tau, coef = _fit(inputs, dt, t, y)
''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_alpha04_fit.py
```
- **Container Exit Code**: `0` (`1150 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
training coverage [(0, 45, 41), (1, 47, 41), (2, 46, 41), (3, 62, 56)]
all LOLO by lab / complete [(0, [0.337003, 0.025153], [0.326602, 0.01347]), (1, [0.311848, 0.044797], [0.258806, 0.012795]), (2, [0.30226, 0.048231], [0.277739, 0.012229]), (3, [0.294027, 0.04643], [0.26922, 0.010083])]
all tau [0.57932799 3.57235394] coef [ 0.70573033  0.5694825   0.07981253  0.14955235 -0.12999609  0.10976266
 -0.09047238  0.24986694  0.16885329] public complete/missing [0.009759, 0.004004] [0.345058, 0.108982] public by lab [[0.113046, 0.036079], [0.119223, 0.038042], [0.141431, 0.044247], [0.006999, 0.002847]]
complete LOLO by lab / complete [(0, [0.337017, 0.025344], [0.326619, 0.013105]), (1, [0.311515, 0.044044], [0.258615, 0.012128]), (2, [0.302529, 0.048485], [0.278077, 0.01139]), (3, [0.293878, 0.046668], [0.269059, 0.009904])]
complete tau [0.58116555 3.54626982] coef [ 0.70779334  0.56206604  0.0796391   0.1495509  -0.12999588  0.10962579
 -0.09007202  0.24986103  0.16975416] public complete/missing [0.00917, 0.002799] [0.346213, 0.109886] public by lab [[0.113703, 0.0361], [0.119455, 0.037871], [0.141788, 0.045016], [0.005786, 0.001751]]
three-spot EMI 0 worst physical stress delta 0.000371 case ((3, 4, 6), 3, -0.22)
three-spot EMI 1 worst physical stress delta 0.000222 case ((11, 16, 12), 16, -0.22)
three-spot EMI 2 worst physical stress delta 0.000271 case ((20, 24, 25), 20, -0.22)
three-spot EMI 3 worst physical stress delta 0.00021 case ((33, 35, 38), 35, -0.22)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4c896d4921026545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4c896d49 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`35333 ms` LLM)
- **Rationale**: Run the exact candidate in isolation and measure strain and rate separately under degraded calibration on each wing. If the four-reading slice still amplifies EMI, revise the calibration cutoff; if it holds and subyield plastic weights stay zero, submit.
- **Alternatives Weighed**: `Submit on the calibration-only regression` (rejected: The revised candidate still needs an end-to-end check with each wing degraded; the new four-inlier cutoff also needs a direct EMI check.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver', root+'alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open(root+'public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; gt=p['gt']; c=d['calibration_load_cells']; tr,te=d['train'],d['test']; dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress']; cal=s._calibrate(c,sy)
def metrics(o):
    e=o['predicted_strain']-gt['te_clean']; r=o['initial_rate_pred']-gt['te_true_rate0']
    masks=[te['lab_id']==i for i in range(4)] + [gt['te_cross'].astype(bool), ~gt['te_cross'].astype(bool)]
    return np.array([(np.sqrt(np.mean(e[m]**2)),np.sqrt(np.mean(r[m]**2))) for m in masks])
base=s.solve_alpha_w02_batch_04(d); print('public lab and regime RMSE (strain, rate)',metrics(base).round(6).tolist()); print('maximum parameter errors',*[float(np.max(abs(base[k]-gt[j]))) for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]])
assert np.all(np.diff(base['estimated_taus'])>0)
for wing in ((0,1),(2,3)):
    worst=np.zeros(2)
    for scenario in ('none','one','three_emi'):
        keep=~np.isin(c['lab_id'],wing); cc={k:v.copy() for k,v in c.items()}
        for lab in wing:
            ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]
            if scenario=='one': keep[ix[len(ix)//2]]=True
            if scenario=='three_emi':
                use=ix[3:6]; keep[use]=True; cc['true_stress'][use[-1]]-=.22
        o=s.solve_alpha_w02_batch_04(dict(d,calibration_load_cells={k:v[keep] for k,v in cc.items()})); worst=np.maximum(worst,metrics(o).max(axis=0))
    print('wing',wing,'worst strain/rate RMSE',worst.round(6)); assert np.all(worst < metrics(base).max(axis=0)+.002)
x=s._inputs(tr,s._features(tr,te)[:len(tr['lab_id'])],cal,dt,sy); below=(x[2][:,-1]==0)&(x[3]==0); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; _,w=s._fit(tuple(v[below] for v in x),dt,t,y[below]); assert np.all(w[7:]==0); print('subyield plastic weights',w[7:])
# Check whether a four-reading sweep can still accept a single EMI point as an inlier.
for lab in range(4):
    ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]; cmd=np.r_[tr['past_stress_cmd'][tr['lab_id']==lab].ravel(),tr['stress_hold_cmd'][tr['lab_id']==lab]]; worst=0.
    for use in (ix[2:6],ix[[0,3,6,-1]]):
        keep=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),use)
        for bad in use:
            for emi in (-.22,.22):
                cc={k:v.copy() for k,v in c.items()}; cc['true_stress'][bad]+=emi; delta=s._calibrate({k:v[keep] for k,v in cc.items()},sy)[lab]-cal[lab]; worst=max(worst,np.max(abs(cmd*delta[0]+delta[1])))
    print('four-check EMI lab',lab,'worst physical stress delta',round(worst,6))
PY
```
- **Container Exit Code**: `1` (`605 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public lab and regime RMSE (strain, rate) [[0.113703, 0.0361], [0.119455, 0.037871], [0.141788, 0.045016], [0.005786, 0.001751], [0.109827, 0.03485], [nan, nan]]
maximum parameter errors 0.04126982098111531 0.0010660436975162835 0.0002458449800892548
wing (0, 1) worst strain/rate RMSE [nan nan]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4c896d4921026545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4c896d49 (directory).
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.12/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "<stdin>", line 22, in <module>
AssertionError
```

### Turn 11 — `execute_in_remote_docker` (`34842 ms` LLM)
- **Rationale**: Repair the validation mask, not the solver. Rerun the interrupted wing checks and test whether the four-reading cutoff accepts EMI; revise calibration only if that slice shows material error.
- **Alternatives Weighed**: `Use the public crossing field as a Boolean yield mask` (rejected: Casting it to Boolean produced an empty subyield slice. The calibrated hold and history stresses define the yield split directly.)

```bash
cat > /tmp/verify_alpha04.py <<'PY'
import importlib.util, pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open(root+'public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; c=d['calibration_load_cells']; tr,te=d['train'],d['test']; dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress']; cal=s._calibrate(c,sy)
above=(abs(g['te_hold_true'])>sy)|np.any(abs(g['te_hist_true'])>sy,axis=1)
masks={**{f'lab{i}':te['lab_id']==i for i in range(4)},'above':above,'below':~above}
assert above.any() and (~above).any()
def metrics(o):
    e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']
    return np.array([(np.sqrt(np.mean(e[m]**2)),np.sqrt(np.mean(r[m]**2))) for m in masks.values()])
base=s.solve_alpha_w02_batch_04(d); bm=metrics(base); print('public strain/rate RMSE',dict(zip(masks,bm.round(6).tolist()))); assert np.all(np.diff(base['estimated_taus'])>0)
for wing in ((0,1),(2,3)):
    worst=np.zeros(2)
    for scenario in ('none','one','three_emi'):
        keep=~np.isin(c['lab_id'],wing); cc={k:v.copy() for k,v in c.items()}
        for lab in wing:
            ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]
            if scenario=='one': keep[ix[len(ix)//2]]=True
            if scenario=='three_emi':
                use=ix[3:6]; keep[use]=True; cc['true_stress'][use[-1]]-=.22
        o=s.solve_alpha_w02_batch_04(dict(d,calibration_load_cells={k:v[keep] for k,v in cc.items()})); worst=np.maximum(worst,metrics(o).max(axis=0))
    print('degraded wing',wing,'worst strain/rate RMSE',worst.round(6)); assert np.all(worst < bm.max(axis=0)+.002)
x=s._inputs(tr,s._features(tr,te)[:len(tr['lab_id'])],cal,dt,sy); below=(x[2][:,-1]==0)&(x[3]==0); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; _,w=s._fit(tuple(v[below] for v in x),dt,t,y[below]); assert np.array_equal(w[7:],np.zeros(2)); print('subyield plastic weights',w[7:])
xt=s._inputs(te,s._features(tr,te)[len(tr['lab_id']):],cal,dt,sy); eps=1e-7; a=s._design(xt,dt,np.array([0.,eps]),base['estimated_taus']); w=np.r_[base['hereditary_weights'],base['viscoplastic_weights']]; err=np.max(abs(((a[:,1]-a[:,0])/eps-a[:,2])@w)); assert err<5e-6; print('onset derivative max abs',err)
for lab in range(4):
    ix=np.flatnonzero(c['lab_id']==lab); ix=ix[np.argsort(c['cmd_stress'][ix])]; cmd=np.r_[tr['past_stress_cmd'][tr['lab_id']==lab].ravel(),tr['stress_hold_cmd'][tr['lab_id']==lab]]; worst=0.
    for use in (ix[2:6],ix[[0,3,6,-1]]):
        keep=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),use)
        for bad in use:
            for emi in (-.22,.22):
                cc={k:v.copy() for k,v in c.items()}; cc['true_stress'][bad]+=emi; delta=s._calibrate({k:v[keep] for k,v in cc.items()},sy)[lab]-cal[lab]; worst=max(worst,np.max(abs(cmd*delta[0]+delta[1])))
    print('four-check EMI lab',lab,'max physical stress delta',round(worst,6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_alpha04.py
```
- **Container Exit Code**: `0` (`1274 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public strain/rate RMSE {'lab0': [0.113703, 0.0361], 'lab1': [0.119455, 0.037871], 'lab2': [0.141788, 0.045016], 'lab3': [0.005786, 0.001751], 'above': [0.130096, 0.041306], 'below': [0.084848, 0.026887]}
degraded wing (0, 1) worst strain/rate RMSE [0.141785 0.045002]
degraded wing (2, 3) worst strain/rate RMSE [0.141706 0.044911]
subyield plastic weights [0. 0.]
onset derivative max abs 4.62991451088496e-07
four-check EMI lab 0 max physical stress delta 0.000266
four-check EMI lab 1 max physical stress delta 0.00023
four-check EMI lab 2 max physical stress delta 0.000262
four-check EMI lab 3 max physical stress delta 0.000203

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4c896d4921026545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4c896d49 (directory).
```

### Turn 12 — `submit_completed_task` (`12731 ms` LLM)
- **Rationale**: The saved candidate passed clean-process schema validation, degraded-calibration tests on both wings, exact subyield zero checks, and an independent onset-derivative check with maximum absolute error of 4.63e-7.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_w02_batch_04_solver.py': 'FILE:8200'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w02_batch_04_solver.py`. The public schema passed, subyield plastic weights were exactly zero, and sparse or EMI-contaminated checks did not worsen the reported worst-lab RMSE: 0.1418 for strain and 0.0450 for initial rate. Missing coupon features remain the main source of prediction error.`

### Turn 13 — `execute_in_remote_docker` (`37992 ms` LLM)
- **Rationale**: Ablate each plastic term inside leave-one-lab-out folds and report signed residuals to detect a remaining lab offset. Test all three- and four-check subsets of the measured sweeps with one EMI perturbation; revise calibration only if a witness materially changes stress and predictions.
- **Alternatives Weighed**: `Add unrelated rational, relay, or Fokker–Planck terms` (rejected: The task specifies continuous creep and its onset derivative; the relevant algebraic audit is whether both specified plastic terms earn their place on held-out measurements.); `Center outcomes by lab to improve pooled fit` (rejected: There is no classification target, and an offset fit using held-out outcomes would leak information. Inspect held-out residual medians after physical alignment instead.)

```bash
cat > /tmp/audit_alpha04.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_w02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(root + 'public_alpha_w02_batch_04.pkl', 'rb'))[0]['dataset']
tr, te, c = d['train'], d['test'], d['calibration_load_cells']
dt, t, sy = d['past_dt'], d['eval_times'], d['yield_stress']
cal = s._calibrate(c, sy)
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
x = s._inputs(tr, s._features(tr, te)[:len(lab)], cal, dt, sy)
y = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
rmse = lambda a: np.sqrt(np.nanmean(a*a))
for name, cols in [('full', ()), ('no_hold_overstress', (7,)), ('no_history_work', (8,))]:
    scores = []
    for i in range(4):
        train = complete & (lab != i)
        test = complete & (lab == i)
        u = tuple(a[train].copy() for a in x)
        if 7 in cols: u[2][:, -1] = 0
        if 8 in cols: u[3][:] = 0
        tau, w = s._fit(u, dt, t, y[train])
        e = s._design(tuple(a[test] for a in x), dt, t, tau) @ w - y[test]
        active = (x[2][test, -1] != 0) | (x[3][test] != 0)
        # Median per-lab residuals expose a systematic instrument offset without fitting one to a held-out lab.
        scores.append([rmse(e[active, :-1]), rmse(e[active, -1]), np.nanmedian(e[:, -1]), np.nanmedian(e[:, -2])])
    print(name, 'LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual]', np.round(scores, 6).tolist())
for i in range(4):
    ix = np.flatnonzero(c['lab_id'] == i)
    rows = lab == i
    cmd = np.r_[tr['past_stress_cmd'][rows].ravel(), tr['stress_hold_cmd'][rows]]
    lo, hi = cmd.min(), cmd.max()
    worst = (0., None)
    for size in (3, 4):
        for use in itertools.combinations(ix, size):
            keep = (c['lab_id'] != i) | np.isin(np.arange(len(c['lab_id'])), use)
            for bad in use:
                for emi in (-.22, .22):
                    cc = {k: v.copy() for k, v in c.items()}
                    cc['true_stress'][bad] += emi
                    gain, bias = s._calibrate({k: v[keep] for k, v in cc.items()}, sy)[i] - cal[i]
                    err = max(abs(gain*lo+bias), abs(gain*hi+bias))
                    if err > worst[0]: worst = (err, (use, bad, emi))
    print('lab', i, 'worst sparse-sweep physical stress delta and witness', worst)
    if worst[0] > .01:
        use, bad, emi = worst[1]
        keep = (c['lab_id'] != i) | np.isin(np.arange(len(c['lab_id'])), use)
        cc = {k: v.copy() for k, v in c.items()}
        cc['true_stress'][bad] += emi
        orig, changed = s.solve_alpha_w02_batch_04(d), s.solve_alpha_w02_batch_04(dict(d, calibration_load_cells={k:v[keep] for k,v in cc.items()}))
        m = te['lab_id'] == i
        print('affected lab strain/rate prediction RMSE shift', [rmse((changed[k]-orig[k])[m]) for k in ('predicted_strain','initial_rate_pred')])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_alpha04.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`11306 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.346385, 0.010773, -0.006574, -0.016296], [0.291462, 0.011661, -0.000522, -0.015128], [0.23591, 0.010383, 0.003258, 0.026581], [0.283379, 0.010891, -0.001158, 0.011001]]
no_hold_overstress LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.454486, 0.113103, -0.013278, 0.023097], [0.461653, 0.127399, -0.012676, -0.002689], [0.417057, 0.122024, 0.012408, 0.030907], [0.401512, 0.095218, -0.010439, -0.045311]]
no_history_work LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.825216, 1.965355, -0.095262, 0.323608], [0.868233, 2.230177, -0.06996, 0.051814], [0.862028, 2.429321, -0.110107, 0.012296], [0.603639, 1.804545, 0.031177, 0.119334]]
lab 0 worst sparse-sweep physical stress delta and witness (0.8566901658545183, ((1, 4, 5, 6), 4, 0.22))
affected lab strain/rate prediction RMSE shift [0.7569010512273802, 1.1112926575706112]
lab 1 worst sparse-sweep physical stress delta and witness (1.5085466506609377, ((10, 12, 14, 17), 10, 0.22))
affected lab strain/rate prediction RMSE shift [0.6351979748976626, 0.5810568069822198]
lab 2 worst sparse-sweep physical stress delta and witness (0.8850620140540564, ((20, 22, 24, 27), 22, -0.22))
affected lab strain/rate prediction RMSE shift [0.822191313461204, 1.3536803818234466]
lab 3 worst sparse-sweep physical stress delta and witness (1.076285155235492, ((31, 33, 36, 37), 31, 0.22))
affected lab strain/rate prediction RMSE shift [0.6812917709495784, 1.2522210843598782]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4c896d4921026545 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4c896d49 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`31874 ms` LLM)
- **Rationale**: The audit exposed a specific failure missed by representative subsets. Tighten only the current-sweep consensus threshold, rerun every three- and four-check subset, and verify that a genuine calibration shift still passes despite one EMI reading.
- **Alternatives Weighed**: `Always use archived gains on four-check sweeps` (rejected: A precise, well-spread current sweep can establish a real instrument change and should not be overridden by stale gains.); `Change the creep basis` (rejected: Removing either plastic term worsened held-out above-yield strain and rate RMSE in every lab; the new failure is in calibration.)
- **ATMS NOGOOD DECLARED**: Refuted `Four readings coherent within the broad EMI rejection window are enough to bypass historical calibration.` | Witness: `Lab 1 checks (10, 12, 14, 17) with a +0.22 perturbation at check 10 shifted calibrated physical stress by up to 1.508547; the analogous lab 2 witness shifted strain and onset-rate predictions by RMSE 0.822191 and 1.353680.` | Blanket Diagnosis: `The broad window lets one EMI reading tilt the fitted gain while all four residuals appear acceptable; that gain error is amplified over the coupon stress range.` | Sublated Pivot: `Require current sweeps to agree at the precision of verified sweeps before bypassing the archive; preserve direct calibration for coherent, well-spread sweeps in a changed regime.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_w02_batch_04_solver.py')
s = p.read_text()
s = s.replace('    tol = .045 * sy\n    for lab', '    tol = .045 * sy\n    tight = .004 * sy\n    for lab')
s = s.replace('        fit, mask = _line(x, y, tol)\n        broad', '        # Verified sweeps are precise: a broad EMI rejection window is not\n        # evidence that a short sweep identifies its slope.\n        fit, mask = _line(x, y, tight)\n        broad')
s = s.replace('                tight = .004 * sy\n                count', '                count')
p.write_text(s)
p = Path('/tmp/audit_alpha04.py')
s = p.read_text()
s = s.replace('    if worst[0] > .01:', '    assert worst[0] < .01, (i, worst)\n    if worst[0] > .01:')
s += '''\n# A new, well-measured instrument regime must not be pulled back to the archive.\nfor i in range(4):\n    cc = {k: v.copy() for k, v in c.items()}\n    m = cc['lab_id'] == i\n    cc['true_stress'][m] += .15 * cc['cmd_stress'][m] + .09\n    cc['true_stress'][np.flatnonzero(m)[0]] += .22\n    error = np.max(abs(s._calibrate(cc, sy)[i] - cal[i] - [.15, .09]))\n    assert error < .001, (i, error)\n    print('shifted current sweep with EMI', i, 'max coefficient error', round(error, 6))\n'''
Path('/tmp/audit_alpha04_final.py').write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py && python3 -I /tmp/audit_alpha04_final.py && python3 -I /tmp/verify_alpha04.py
```
- **Container Exit Code**: `0` (`11525 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
full LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.346385, 0.010773, -0.006574, -0.016296], [0.291462, 0.011661, -0.000522, -0.015128], [0.23591, 0.010383, 0.003258, 0.026581], [0.283379, 0.010891, -0.001158, 0.011001]]
no_hold_overstress LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.454486, 0.113103, -0.013278, 0.023097], [0.461653, 0.127399, -0.012676, -0.002689], [0.417057, 0.122024, 0.012408, 0.030907], [0.401512, 0.095218, -0.010439, -0.045311]]
no_history_work LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual] [[0.825216, 1.965355, -0.095262, 0.323608], [0.868233, 2.230177, -0.06996, 0.051814], [0.862028, 2.429321, -0.110107, 0.012296], [0.603639, 1.804545, 0.031177, 0.119334]]
lab 0 worst sparse-sweep physical stress delta and witness (0.00039665795002223993, ((0, 1, 6), 0, -0.22))
lab 1 worst sparse-sweep physical stress delta and witness (0.00024448988243349495, ((10, 11, 14), 10, -0.22))
lab 2 worst sparse-sweep physical stress delta and witness (0.00030964440211953096, ((20, 23, 24), 20, -0.22))
lab 3 worst sparse-sweep physical stress delta and witness (0.00027058012569949674, ((30, 34, 36), 30, -0.22))
shifted current sweep with EMI 0 max coefficient error 5e-06
shifted current sweep with EMI 1 max coefficient error 1.8e-05
shifted current sweep with EMI 2 max coefficient error 6e-06
shifted current sweep with EMI 3 max coefficient error 2e-06
public strain/rate RMSE {'lab0': [0.113703, 0.0361], 'lab1': [0.119455, 0.037871], 'lab2': [0.141788, 0.045016], 'lab3': [0.005786, 0.001751], 'above': [0.130096, 0.041306], 'below': [0.084848, 0.026887]}
degraded wing (0, 1) worst strain/rate RMSE [0.141785 0.045002]
degraded wing (2, 3) worst strain/rate RMSE [0.141706 0.044911]
subyield plastic weights [0. 0.]
onset derivative max abs 4.62991451088496e-07
four-check EMI lab 0 max physical stress delta 0.000266
four-check EMI lab 1 max physical stress delta 0.00023
four-check EMI lab 2 max physical stress delta 0.000262
four-check EMI lab 3 max physical stress delta 0.000203

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5135a4a0a7241dc9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_5135a4a0 (directory).
```

### Turn 15 — `submit_completed_task` (`15621 ms` LLM)
- **Rationale**: The exact saved candidate passed isolated schema and stress checks. Exhaustive three- and four-reading EMI tests covered the prior failure, and shifted current sweeps with EMI still recovered their calibration coefficients within 0.000018. Both plastic terms improved held-out above-yield predictions in every lab, and the onset derivative agreed with the independent finite difference within 4.63e-7.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w02_batch_04_solver.py`. The public schema passed, and subyield plastic weights were exactly zero. The calibration fix reduced the worst stress error across the tested sparse, EMI-contaminated sweeps from 1.508547 to 0.000397. Worst public lab RMSE was 0.1418 for strain and 0.0450 for onset rate; missing coupon features remained the main source of prediction error.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
