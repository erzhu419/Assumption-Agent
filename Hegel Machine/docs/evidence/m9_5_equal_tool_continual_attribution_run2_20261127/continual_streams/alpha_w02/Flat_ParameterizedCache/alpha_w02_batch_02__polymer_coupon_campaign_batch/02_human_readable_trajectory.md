# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w02_batch_02__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w02_step_02` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `341,949` (`329,306` / `58,624` / `12,643`) |
| **Total Episode Wall Time (s)** | `386.45s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center. | [Consolidated `m95_src_alpha_w02`]: No empirical nogoods were supplied; the winning code addresses the failure modes of pooled calibration, slopes inferred from narrow spot checks, and ordinary least squares under large strain glitches. Fit load cells separately by lab, accepting a local slope only when at least four checks produce a unique consensus of at least three spanning 0.10 of the stress scale; otherwise transfer a same-lab archive slope and update its bias only when the current offsets agree. Align coupon features using each lab’s reference location and spread, then fill missing entries with equally weighted lab–stratum cell means rather than a row-weighted pooled mean. Exact exponential memory propagation gives a design linear in the hereditary weights at fixed relaxation times; fit curves jointly with initial rates using a Cauchy loss and separate MAD noise scales, and reject a fit whose robust, column-scaled Jacobian has a smallest-to-largest singular-value ratio below 1e-6.
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

### Turn 1 — `execute_in_remote_docker` (`15877 ms` LLM)
- **Rationale**: Inspect the schema and prior solver interfaces to identify which calibration and constitutive assumptions transfer and which outputs need new tests.

```bash
cd /holdout_task && ls -l && ls -lh /tmp/hegel_continual_archive && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np,json
x=pickle.load(open('public_alpha_w02_batch_02.pkl','rb'))
def show(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,type(v).__name__,getattr(v,'shape',''));show(v,d+2)
 elif isinstance(x,(list,tuple)):
  print(' '*d,'length',len(x),'example',str(x[:1])[:700])
 elif isinstance(x,np.ndarray):
  print(' '*d,'dtype',x.dtype,'example',str(x[:2])[:350] if x.ndim else x)
 else: print(' '*d,str(x)[:500])
show(x)
PY
for f in /tmp/hegel_continual_archive/*solver.py; do echo "=== $f"; wc -l "$f"; grep -nE '^(def|class|import|from) ' "$f"; done
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation'
```
- **Container Exit Code**: `0` (`586 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
togenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_hereditary_creep_transfer@c0d59a47 | align_reference_features(features: np.ndarray, reference: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Align features using reference coupons measured in the same group | REQ: {'field': 'max_absolute_calibrated_stress_over_yield_stress', 'op': '<', 'value': 1}; {' | --describe autogenous_hereditary_creep_transfer.align_reference_features
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
```

### Turn 2 — `execute_in_remote_docker` (`15017 ms` LLM)
- **Rationale**: Check which prior constitutive terms and hazard conventions transfer, and whether historical calibration can support sparse current checks without overriding reliable measurements.

```bash
cd /holdout_task
python3 - <<'PY'
import pickle
x=pickle.load(open('public_alpha_w02_batch_02.pkl','rb'))[0]
print('entry keys',list(x));d=x['dataset']
for k,v in d.items():
 print(k,[(a,getattr(b,'shape',None)) for a,b in v.items()] if isinstance(v,dict) else (getattr(v,'shape',None),str(v)[:250]))
PY
cat /tmp/hegel_continual_archive/archive_calibration_history.json
sed -n '1,340p' /tmp/hegel_continual_archive/alpha_w02_batch_01_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_hereditary_creep_transfer.align_reference_features
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`27320 ms` LLM)
- **Rationale**: Inspect prior equations and compare current calibration residuals with archive fits. The observed stress range will determine whether to test a yield-driven term, and the calibration residuals will determine whether archive fallback is justified.
- **Alternatives Weighed**: `Reuse the previous constitutive fit unchanged` (rejected: It may omit preloaded memory or apply subyield assumptions outside their measured range.)

```bash
sed -n '117,238p' /tmp/hegel_continual_archive/alpha_w02_s00_solver.py; grep -nE '^(def|class)|plastic|crossing|expm1|initial_memory' /tmp/hegel_continual_archive/alpha_w02_batch_01_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,json
D=pickle.load(open('public_alpha_w02_batch_02.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; H=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for c in range(4):
 for row in [*H,C]:
  i=np.asarray(row['lab_id'])==c;x=np.asarray(row['cmd_stress'])[i];y=np.asarray(row['true_stress'])[i];p=np.polyfit(x,y,1);print(c,row.get('episode_short_name','current'),p,np.round(np.sort(abs(y-np.polyval(p,x))),6))
for key in ('train','test'):
 p=D[key];print(key)
 for k in ('past_stress_cmd','stress_hold_cmd','initial_memory_states','instant_rate_obs','residual_plastic_state','critical_strain','coupon_features_raw','ref_coupons'):
  if k in p:
   a=p[k]; print(k,np.nanpercentile(a,[0,10,50,90,100]),np.isnan(a).sum())
 print(np.unique(np.stack([p['lab_id'],p['stratum']],1),axis=0,return_counts=True))
PY
```
- **Container Exit Code**: `0` (`561 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.90e-05
 1.01e-04 1.07e-04]
2 alpha_w02_s00 [1.00002786e+00 1.37513325e-05] [2.00e-06 9.00e-06 1.00e-05 1.70e-05 4.70e-05 5.20e-05 5.30e-05 5.90e-05
 1.15e-04 1.17e-04]
2 alpha_w02_batch_01 [1.00003769e+00 7.76536687e-06] [3.00e-06 4.00e-06 3.60e-05 4.30e-05 1.14e-04 1.49e-04 1.66e-04 1.67e-04
 1.69e-04 1.73e-04]
2 current [ 9.99994653e-01 -1.38283468e-05] [7.00e-06 9.00e-06 3.30e-05 4.20e-05 4.60e-05 4.70e-05 5.20e-05 7.10e-05
 1.51e-04 1.75e-04]
3 alpha_w02_s00 [9.99933557e-01 5.37831969e-05] [1.00e-05 1.40e-05 1.70e-05 2.70e-05 4.90e-05 6.90e-05 7.30e-05 8.70e-05
 9.90e-05 1.26e-04]
3 alpha_w02_batch_01 [1.00001515e+00 6.39376903e-05] [2.00e-06 4.00e-06 1.20e-05 1.50e-05 2.10e-05 2.30e-05 5.00e-05 6.20e-05
 1.07e-04 1.13e-04]
3 current [ 1.00000304e+00 -3.48202302e-05] [3.0e-06 1.8e-05 1.8e-05 2.5e-05 3.2e-05 3.5e-05 3.6e-05 8.5e-05 9.2e-05
 9.8e-05]
train
past_stress_cmd [-2.47800000e+00 -1.84348773e+00 -2.40561246e-04  1.89812560e+00
  2.47800000e+00] 0
stress_hold_cmd [-2.04929515e+00 -1.70559426e+00 -1.68666027e-03  1.65083842e+00
  2.10297789e+00] 0
initial_memory_states [-5.53364766 -2.73020278 -0.41115418  2.61963583  6.06925845] 0
instant_rate_obs [-5.25395852 -2.59295044  0.21318772  3.17996021  5.38877816] 0
coupon_features_raw [-3.74678515 -1.43543613 -0.02968109  1.84575997  5.28328265] 21
ref_coupons [-3.81723565 -1.44873402 -0.05145691  1.70059271  5.75467844] 271
(array([[0, 0],
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
       [3, 2]]), array([21, 15, 11, 14, 23,  8, 22, 16, 22, 15, 14, 19]))
test
past_stress_cmd [-2.478      -1.7         0.02811021  1.74289866  2.478     ] 0
stress_hold_cmd [-1.87520575 -1.30049938  0.71270814  2.124       2.124     ] 0
initial_memory_states [-4.50946824 -2.29904908 -0.09832115  2.34157625  6.79625956] 0
instant_rate_obs [-4.20525487 -1.82486047  1.00864071  4.36191214  4.7597409 ] 0
residual_plastic_state [0. 0. 0. 0. 0.] 0
critical_strain [-0.5180722  -0.04278374  0.4274118   0.78011936  0.98958552] 0
coupon_features_raw [-2.26387648 -1.32903001 -0.04814681  1.71220193  3.45521429] 7
ref_coupons [-4.35431842 -1.48813304 -0.12767938  1.61235754  5.0464873 ] 117
(array([[0, 0],
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
       [3, 2]]), array([10,  4,  5,  3,  8,  9,  9,  5, 11,  7,  3,  6]))
```

### Turn 4 — `execute_in_remote_docker` (`80594 ms` LLM)
- **Rationale**: Implement the measured-memory recurrence, then compare drift bases on held-out labs. Check the public hazard reference only to resolve the output convention, which the schema validator cannot distinguish.
- **Alternatives Weighed**: `Always pool archived and current load-cell checks` (rejected: All four current lab sweeps have sufficient span and residuals below 0.00025; pooling could bias a future batch whose calibration has changed.); `Add a yield-driven plastic term` (rejected: Current calibrated stresses are subyield and the reported residual plastic state is zero.)

```bash
cat > /holdout_task/alpha_w02_batch_02_solver.py <<'PY'
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y, tol):
    if len(x) < 2 or np.ptp(x) == 0:
        return None, np.zeros(len(x), dtype=bool)
    best = None
    for i, j in zip(*np.triu_indices(len(x), 1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        bias = np.median(y - gain*x)
        r = abs(y - gain*x - bias)
        good = r <= tol
        score = (good.sum(), -np.minimum(r, tol).sum())
        if best is None or score > best[0]:
            best = (score, good)
    good = best[1]
    if good.sum() < 2 or np.ptp(x[good]) == 0:
        return None, good
    return np.linalg.lstsq(np.c_[x[good], np.ones(good.sum())], y[good], rcond=None)[0], good


def _calibration(checks, yield_stress):
    tol = 0.001*yield_stress
    history = None
    result = []
    for lab in range(4):
        k = np.asarray(checks['lab_id']) == lab
        x, y = (np.asarray(checks[key])[k] for key in ('cmd_stress', 'true_stress'))
        estimate, good = _line(x, y, tol)
        if estimate is not None and good.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[good]) >= 0.1*yield_stress:
            result.append(estimate)
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
        prior = []
        for sweep in history:
            m = np.asarray(sweep['lab_id']) == lab
            a, b = (np.asarray(sweep[key])[m] for key in ('cmd_stress', 'true_stress'))
            fit, inliers = _line(a, b, tol)
            if fit is not None and inliers.sum() >= max(3, len(a)//2 + 1) and np.ptp(a[inliers]) >= 0.1*yield_stress:
                prior.append(fit)
        if prior:
            gain, bias = np.median(prior, axis=0)
            offsets = y-gain*x
            agree = abs(offsets-bias) <= tol
            if agree.any():
                bias = np.median(np.r_[offsets[agree], bias])
            result.append((gain, bias))
        elif estimate is not None:
            result.append(estimate)
        else:
            result.append((1., np.median(y-x) if len(x) else 0.))
    return np.asarray(result)


def _inputs(dataset):
    train, test = dataset['train'], dataset['test']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    table = _calibration(dataset['calibration_load_cells'], dataset['yield_stress'])
    gain, bias = table[lab].T
    past = np.vstack([train['past_stress_cmd'], test['past_stress_cmd']])*gain[:, None]+bias[:, None]
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]*gain+bias
    mem = np.vstack([train['initial_memory_states'], test['initial_memory_states']])
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    feat = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    for c in np.unique(lab):
        rows = lab == c
        r = ref[rows]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[rows] = (feat[rows]-np.median(np.nanmean(r, axis=0)))/np.median(np.nanstd(r, axis=0, ddof=0))
    cells = np.unique(np.c_[lab, stratum], axis=0)
    means = []
    for c, s in cells:
        a = feat[(lab == c) & (stratum == s)]
        count = np.isfinite(a).sum(axis=0)
        means.append(np.divide(np.nansum(a, axis=0), count, out=np.full(feat.shape[1], np.nan), where=count > 0))
    feat = np.where(np.isfinite(feat), feat, np.nanmean(means, axis=0))
    return past, hold, mem, feat


def _components(inputs, dt, tau, variant=0):
    past, hold, mem, feat = inputs
    remaining = np.cumsum(dt[::-1])[::-1]
    end = np.r_[remaining[1:], 0.]
    weight = np.exp(-end[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
    delta = hold[:, None] - past@weight - mem*np.exp(-remaining[0]/tau)
    z = np.c_[np.ones(len(hold)), feat]
    if variant in (1, 3):
        z = np.c_[z, hold[:, None]*z]
    if variant in (2, 3):
        i, j = np.triu_indices(feat.shape[1]); z = np.c_[z, feat[:, i]*feat[:, j]]
    return delta, z


def _design(inputs, dt, times, tau, variant):
    delta, z = _components(inputs, dt, tau, variant)
    curve = np.concatenate((delta[:, None, :]*(-np.expm1(-times[:, None]/tau)), z[:, None, :]*times[:, None]), axis=2)
    rate = np.c_[delta/tau, z]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _fit(inputs, dt, times, obs, variant=0):
    valid = np.isfinite(obs)
    y = obs[valid]
    scale = np.ones_like(obs)
    scale[:, -1] = 1.5
    w = (1/scale)[valid]

    def fit_at(logtau, robust):
        design = _design(inputs, dt, times, np.exp(logtau), variant)[valid]
        a = design*w[:, None]
        b = y*w
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        if robust:
            for _ in range(6):
                r = a@coef-b
                u = (1+(r/2.5)**2)**(-0.25)
                coef = np.linalg.lstsq(a*u[:, None], b*u, rcond=None)[0]
        return coef, a@coef-b

    limits = np.log(([0.055, 0.055], [24.9, 24.9]))
    starts = [(0.25, 1.8), (0.65, 5.), (1.7, 10.)]
    fits = [least_squares(lambda lt: fit_at(lt, False)[1], np.log(start), bounds=limits) for start in starts]
    best = min(fits, key=lambda f: f.cost)
    r = np.full(obs.shape, np.nan)
    r[valid] = fit_at(best.x, False)[1]/w
    mad = lambda a: 1.4826*np.nanmedian(abs(a-np.nanmedian(a)))
    curve_scale = max(mad(r[:, :-1]), 1e-8)
    rate_scale = max(mad(r[:, -1]), curve_scale/3)
    scale = np.broadcast_to(np.r_[np.full(len(times), curve_scale), rate_scale], obs.shape)
    w = (1/scale)[valid]
    opt = least_squares(lambda lt: fit_at(lt, True)[1], best.x, bounds=limits, loss='soft_l1', f_scale=2.5)
    tau = np.exp(opt.x)
    coef = fit_at(opt.x, True)[0]
    order = np.argsort(tau)
    return tau[order], np.r_[coef[:2][order], coef[2:]]


def _hazard(initial, threshold, times, tau, amplitudes, drift):
    crossing = np.full(len(initial), times[-1])
    for i, (start, critical, amp, slope) in enumerate(zip(initial, threshold, amplitudes, drift)):
        def margin(t):
            return start + amp@(-np.expm1(-t/tau)) + slope*t - critical
        def rate(t):
            return (amp/tau)@np.exp(-t/tau)+slope
        if margin(0.) >= 0:
            crossing[i] = 0.
            continue
        # The second derivative is a sum of two exponentials and has at most one zero.
        knots = [0., float(times[-1])]
        if amp[0]*amp[1] < 0 and tau[0] != tau[1]:
            inflection = np.log(abs(amp[0]/amp[1])*(tau[1]/tau[0])**2)/(1/tau[0]-1/tau[1])
            if 0 < inflection < knots[-1]:
                knots.insert(1, inflection)
        extrema = [brentq(rate, a, b) for a, b in zip(knots[:-1], knots[1:]) if rate(a)*rate(b) < 0]
        a = 0.
        for b in [*extrema, knots[-1]]:
            if margin(b) >= 0:
                crossing[i] = brentq(margin, a, b)
                break
            a = b
    return crossing


def solve_alpha_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    inputs = _inputs(dataset)
    obs = np.c_[train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']]
    tau, coef = _fit(tuple(a[:n] for a in inputs), dt, times, obs)
    delta, z = _components(tuple(a[n:] for a in inputs), dt, tau)
    amp, drift = delta*coef[:2], z@coef[2:]
    initial = test['strain_0']
    increment = amp@(-np.expm1(-times[:, None]/tau)).T + drift[:, None]*times
    pred = initial[:, None]+increment
    return {'estimated_taus': tau, 'predicted_strain': pred,
            'secant_compliance': (pred[:, -1]-initial)/np.maximum(abs(inputs[1][n:]), 0.5),
            'hazard_crossing_times': _hazard(initial, test['critical_strain'], times, tau, amp, drift)}
PY
cat > /tmp/polymer_loso.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_w02_batch_02_solver as m
entry=pickle.load(open('/holdout_task/public_alpha_w02_batch_02.pkl','rb'))[0]; d=entry['dataset']; t=d['train']; dt=d['past_dt']; times=d['eval_times']
print('public reference keys',list(entry['gt']))
for k,v in entry['gt'].items():
 if 'hazard' in k or 'cross' in k:
  a=np.asarray(v); print(k,'range',a.min(),a.max(),'outside hold',sum(a>times[-1]),'at zero',sum(a==0),'on grid',sum(np.isin(a,times)))
a=tuple(v[:len(t['lab_id'])] for v in m._inputs(d)); y=np.c_[t['strain_curves_obs']-t['strain_0'][:,None], t['instant_rate_obs']]
for variant in range(4):
 metrics=[]
 for lab in range(4):
  val=t['lab_id']==lab; tau,c=m._fit(tuple(v[~val] for v in a),dt,times,y[~val],variant); r=m._design(tuple(v[val] for v in a),dt,times,tau,variant)@c-y[val]; metrics.append(np.sqrt(np.mean(r**2,axis=0))); print(variant,lab,'tau',tau.round(4),'curve',np.sqrt(np.mean(r[:,:-1]**2)),'endpoint',metrics[-1][-2],'rate',metrics[-1][-1],flush=True)
 print('max per horizon',np.max(metrics,axis=0).round(5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/polymer_loso.py
```
- **Container Exit Code**: `0` (`2682 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_batch_02_solver.py']`
- **Stdout (tail)**:
```text
d_true', 'te_hist_true']
te_cross range 0.040636698966890046 6.4 outside hold 0 at zero 0 on grid 24
0 0 tau [0.567 3.488] curve 0.3112142909376917 endpoint 0.3172933010849384 rate 0.02896794168854265
0 1 tau [0.5634 3.4642] curve 0.2510438645675152 endpoint 0.31140858333576416 rate 0.02873118174765133
0 2 tau [0.5674 3.4833] curve 0.3298597532639626 endpoint 0.4370176328702271 rate 0.06609003000526523
0 3 tau [0.5646 3.4441] curve 0.2972697504809552 endpoint 0.34871650178875535 rate 0.029268956981262667
max per horizon [0.35278 0.30763 0.31422 0.44968 0.39811 0.37938 0.43158 0.44475 0.35497
 0.37473 0.40058 0.43702 0.06609]
1 0 tau [0.567  3.4647] curve 0.3112935597405683 endpoint 0.3171004418338007 rate 0.028900365270978348
1 1 tau [0.5644 3.4203] curve 0.251074184657651 endpoint 0.3116313505927346 rate 0.027568743152439835
1 2 tau [0.5675 3.4699] curve 0.3298634587336358 endpoint 0.4365994441318615 rate 0.06610762673155977
1 3 tau [0.5647 3.4184] curve 0.29721133837155056 endpoint 0.34832317883098757 rate 0.029226932236175014
max per horizon [0.35276 0.30766 0.31432 0.44965 0.39806 0.37955 0.43143 0.44466 0.35507
 0.37456 0.4007  0.4366  0.06611]
2 0 tau [0.5664 3.4946] curve 0.31135872747736215 endpoint 0.31947957780513436 rate 0.029277651386322875
2 1 tau [0.5644 3.489 ] curve 0.25117828775338047 endpoint 0.31026109995591117 rate 0.028425231077773255
2 2 tau [0.5653 3.4601] curve 0.3297618924074215 endpoint 0.4360474711140396 rate 0.06598442012561097
2 3 tau [0.5634 3.4504] curve 0.2973000598480527 endpoint 0.34835048981429245 rate 0.029757574426210673
max per horizon [0.35288 0.30754 0.3141  0.44976 0.398   0.37958 0.43098 0.44396 0.35438
 0.37474 0.40109 0.43605 0.06598]
3 0 tau [0.5668 3.4509] curve 0.31134054614998063 endpoint 0.3186722852611395 rate 0.028940990744718488
3 1 tau [0.5653 3.442 ] curve 0.2511514432152422 endpoint 0.3105856778672334 rate 0.027844818992806872
3 2 tau [0.5658 3.4541] curve 0.32983972139447887 endpoint 0.43605698922413316 rate 0.0660607125852652
3 3 tau [0.5635 3.4078] curve 0.2971920033439704 endpoint 0.34815156941433123 rate 0.029580023002211717
max per horizon [0.35287 0.30761 0.3142  0.44968 0.39798 0.37967 0.43058 0.44392 0.35426
 0.37484 0.40125 0.43606 0.06606]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cfbcfd8dad4b8ea4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cfbcfd8d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29225 ms` LLM)
- **Rationale**: Distinguish scattered curve corruption from missing constitutive structure before changing the fit. Audit each public output separately, including whether continuous hazard crossings agree with the reference.
- **Alternatives Weighed**: `Promote stress interactions or quadratic drift` (rejected: Neither materially improved worst-lab or worst-horizon error.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_w02_batch_02_solver as m
e=pickle.load(open('public_alpha_w02_batch_02.pkl','rb'))[0]; d=e['dataset']; g=e['gt']; t=d['train']; s=d['test']; n=len(t['lab_id']); tm=d['eval_times']; a=m._inputs(d); tr=tuple(x[:n] for x in a); te=tuple(x[n:] for x in a)
y=np.c_[t['strain_curves_obs']-t['strain_0'][:,None], t['instant_rate_obs']]; tau,c=m._fit(tr,d['past_dt'],tm,y); pred=m._design(tr,d['past_dt'],tm,tau,0)@c; r=pred-y
print('fitted taus',tau,'coefficients',c)
print('abs residual quantiles by horizon/rate',np.quantile(abs(r),[.25,.5,.75,.9,.95,.99,1],axis=0).round(4)); print('row residual lag correlation',np.corrcoef(r[:,:-1].T)[0].round(3))
for k in range(4):
 rows=t['lab_id']==k; print('lab',k,'curve median',np.median(abs(r[rows,:-1])),'rate quantile',np.quantile(abs(r[rows,-1]),[.5,.9,.99]),'outlier fraction by horizon',(abs(r[rows,:-1])>.1).mean(axis=0).round(2))
for count in np.unique(np.isnan(t['coupon_features_raw']).sum(axis=1)):
 rows=np.isnan(t['coupon_features_raw']).sum(axis=1)==count; print('missing count',count,'n',rows.sum(),'median curve and rate',np.median(abs(r[rows,:-1])),np.median(abs(r[rows,-1])))
o=m.solve_alpha_w02_batch_02(d)
for k,j in [('estimated_taus','taus'),('predicted_strain','te_clean'),('secant_compliance','te_secant'),('hazard_crossing_times','te_cross')]:
 v=np.asarray(o[k])-g[j]; print(k,'rmse',np.sqrt(np.mean(v*v)),'abs quantiles',np.quantile(abs(v),[.5,.9,.95,.99,1]),'worst row',np.argmax(np.max(abs(v),axis=1) if v.ndim==2 else abs(v)))
print('test rate obs against clean',np.quantile(abs(s['instant_rate_obs']-g['te_true_rate0']),[.5,.9,.99,1])); print('model rate against clean',np.quantile(abs((m._design(te,d['past_dt'],tm,tau,0)@c)[:,-1]-g['te_true_rate0']),[.5,.9,.99,1])); print('initial already above',sum(s['strain_0']>=s['critical_strain']))
z=np.where(abs(o['hazard_crossing_times']-g['te_cross'])>.05)[0]; print('hazard disagreements >.05 [row,lab,pred,ref]',np.c_[z,s['lab_id'][z],o['hazard_crossing_times'][z],g['te_cross'][z]].round(4)); print('public calibration max hold error',abs(a[1][n:]-g['te_hold_true']).max())
PY
```
- **Container Exit Code**: `0` (`592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
26
  0.3871 0.5757 0.0257]
 [0.9803 0.101  0.7883 0.9899 0.8891 0.9423 0.9422 1.0107 0.9382 0.8734
  0.9145 1.0047 0.0672]
 [1.1539 1.134  1.154  1.1545 1.0987 1.187  1.1674 1.1315 1.0731 1.1898
  1.0919 1.302  0.2067]
 [1.1796 1.1716 1.2223 1.1854 1.1496 1.1983 1.6777 1.1973 1.2042 1.7099
  1.7113 2.3795 0.3671]]
row residual lag correlation [ 1.     0.015  0.006 -0.007  0.059 -0.04   0.061  0.017 -0.015  0.05
  0.074 -0.018]
lab 0 curve median 0.009039357914313595 rate quantile [0.00805801 0.02605336 0.11449608] outlier fraction by horizon [0.09 0.06 0.04 0.13 0.06 0.17 0.23 0.15 0.19 0.19 0.13 0.17]
lab 1 curve median 0.0073293493797997045 rate quantile [0.00764608 0.01821491 0.1123319 ] outlier fraction by horizon [0.09 0.   0.02 0.02 0.04 0.07 0.11 0.22 0.09 0.13 0.09 0.11]
lab 2 curve median 0.011230537677088703 rate quantile [0.00962622 0.03279594 0.27511053] outlier fraction by horizon [0.12 0.05 0.1  0.03 0.18 0.12 0.17 0.13 0.2  0.15 0.17 0.13]
lab 3 curve median 0.01084336365565719 rate quantile [0.00937918 0.02501599 0.11445015] outlier fraction by horizon [0.15 0.08 0.06 0.19 0.04 0.08 0.12 0.1  0.12 0.17 0.12 0.19]
missing count 0 n 182 median curve and rate 0.00862896115038286 0.007804479932613517
missing count 1 n 15 median curve and rate 0.09256721886418529 0.06635180760100967
missing count 2 n 3 median curve and rate 0.16601776337024043 0.12026055163470861
estimated_taus rmse 0.06390825107854796 abs quantiles [0.04755616 0.08171041 0.08597969 0.08939512 0.09024897] worst row 1
predicted_strain rmse 0.10445244656785388 abs quantiles [0.00219706 0.02439619 0.06168477 0.5463751  1.36998161] worst row 23
secant_compliance rmse 0.389605625655837 abs quantiles [0.01309042 0.06200736 0.50752576 2.13943402 2.23597267] worst row 57
hazard_crossing_times rmse 0.016260508798928312 abs quantiles [0.00206836 0.01608075 0.03983632 0.06533628 0.09291085] worst row 53
test rate obs against clean [0.00629522 0.01604501 0.02273459 0.02360535]
model rate against clean [0.00303671 0.01068707 0.1844129  0.21321069]
initial already above 0
hazard disagreements >.05 [row,lab,pred,ref] [[42.      3.      3.7261  3.7842]
 [53.      3.      3.4367  3.5296]]
public calibration max hold error 8.268913011866275e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cfbcfd8dad4b8ea4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cfbcfd8d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34076 ms` LLM)
- **Rationale**: Missing-feature training rows have roughly tenfold larger median strain residuals, while the public test rate observations are substantially better than the model on its worst rate predictions. Compare outlier-resistant fitting and rate correction separately across held-out labs, then check whether either improves secant compliance without degrading hazard crossings.
- **Alternatives Weighed**: `Correct every coupon using its observed initial rate` (rejected: For complete-feature test rows the model is typically more accurate than the rate observation. Test the correction by missingness before applying it.)

```bash
cat > /tmp/polymer_compare.py <<'PY'
import importlib.util, pickle, types
from pathlib import Path
import numpy as np
p=Path('/holdout_task/alpha_w02_batch_02_solver.py')
spec=importlib.util.spec_from_file_location('baseline',p); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=p.read_text().replace('range(6):','range(9):').replace('(1+(r/2.5)**2)**(-0.25)','(1+(r/2.5)**2)**(-0.5)').replace("loss='soft_l1'", "loss='cauchy'")
c=types.ModuleType('cauchy'); exec(compile(s,str(p),'exec'),c.__dict__)
e=pickle.load(open('/holdout_task/public_alpha_w02_batch_02.pkl','rb'))[0]; d=e['dataset']; g=e['gt']; t=d['train']; u=d['test']; n=len(t['lab_id']); tm=d['eval_times']; dt=d['past_dt']; a=b._inputs(d); tr=tuple(x[:n] for x in a); te=tuple(x[n:] for x in a)
y=np.c_[t['strain_curves_obs']-t['strain_0'][:,None],t['instant_rate_obs']]; missing=np.isnan(t['coupon_features_raw']).any(axis=1); missing_te=np.isnan(u['coupon_features_raw']).any(axis=1)
for model,complete in [(b,False),(c,False),(c,True)]:
 print('\nMODEL',model.__name__,complete)
 residual=np.empty_like(y)
 for lab in range(4):
  val=t['lab_id']==lab; fit=~val & (~missing if complete else True); tau,coef=model._fit(tuple(v[fit] for v in tr),dt,tm,y[fit]); residual[val]=model._design(tuple(v[val] for v in tr),dt,tm,tau,0)@coef-y[val]
 for group in (False,True):
  for alpha in (0.,.25,.5,.75,1.):
   rates=residual[:,-1]; r=residual[:,:-1]-alpha*rates[:,None]*tm
   print('OOF missing',group,'alpha',alpha,'worst lab median/90p curve',np.max([np.quantile(abs(r[(missing==group)&(t['lab_id']==lab)]),[.5,.9]) for lab in range(4)],axis=0).round(5),'worst endpoint median',max(np.median(abs(r[(missing==group)&(t['lab_id']==lab),-1])) for lab in range(4)).round(5))
 fit=~missing if complete else np.ones(n,dtype=bool); tau,coef=model._fit(tuple(v[fit] for v in tr),dt,tm,y[fit]); delta,z=model._components(te,dt,tau); amp=delta*coef[:2]; drift=z@coef[2:]; prior_rate=amp@(1/tau)+drift; dr=u['instant_rate_obs']-prior_rate
 print('taus',tau,'coefficients',coef,'test missing rows [index,rate innovation]',np.c_[np.where(missing_te)[0],dr[missing_te]].round(5))
 for alpha in (0.,.5,.75,1.):
  slope=drift+alpha*missing_te*dr
  pred=u['strain_0'][:,None]+amp@(-np.expm1(-tm[:,None]/tau)).T+slope[:,None]*tm
  sec=(pred[:,-1]-u['strain_0'])/np.maximum(abs(te[1]),.5); hz=model._hazard(u['strain_0'],u['critical_strain'],tm,tau,amp,slope)
  errors=[pred-g['te_clean'],sec-g['te_secant'],hz-g['te_cross']]
  print('public alpha',alpha,'RMSE',np.array([np.sqrt(np.mean(q*q)) for q in errors]).round(6),'max',np.array([np.max(abs(q)) for q in errors]).round(6),'hazard censor disagreements',np.sum((hz==tm[-1])!=(g['te_cross']==tm[-1])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/polymer_compare.py
```
- **Container Exit Code**: `0` (`1368 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
664] hazard censor disagreements 0
public alpha 0.5 RMSE [0.054466 0.203841 0.016153] max [0.697328 1.143257 0.082664] hazard censor disagreements 0
public alpha 0.75 RMSE [0.030312 0.112525 0.016118] max [0.368026 0.653361 0.082664] hazard censor disagreements 0
public alpha 1.0 RMSE [0.010813 0.032547 0.01611 ] max [0.088571 0.164368 0.082664] hazard censor disagreements 0

MODEL cauchy True
OOF missing False alpha 0.0 worst lab median/90p curve [0.01207 0.08683] worst endpoint median 0.04735
OOF missing False alpha 0.25 worst lab median/90p curve [0.01065 0.07051] worst endpoint median 0.03662
OOF missing False alpha 0.5 worst lab median/90p curve [0.01049 0.08284] worst endpoint median 0.03283
OOF missing False alpha 0.75 worst lab median/90p curve [0.01142 0.1284 ] worst endpoint median 0.04431
OOF missing False alpha 1.0 worst lab median/90p curve [0.01255 0.17436] worst endpoint median 0.05387
OOF missing True alpha 0.0 worst lab median/90p curve [0.20204 1.2181 ] worst endpoint median 1.07146
OOF missing True alpha 0.25 worst lab median/90p curve [0.14012 1.1081 ] worst endpoint median 0.79642
OOF missing True alpha 0.5 worst lab median/90p curve [0.08866 1.0431 ] worst endpoint median 0.52138
OOF missing True alpha 0.75 worst lab median/90p curve [0.04623 0.99779] worst endpoint median 0.26188
OOF missing True alpha 1.0 worst lab median/90p curve [0.01642 0.97019] worst endpoint median 0.10106
taus [0.57040206 3.57249496] coefficients [ 0.65688581  0.53612336  0.07928536  0.14919362 -0.12924414  0.10942854
 -0.08950403] test missing rows [index,rate innovation] [[ 1.0000e+01 -1.6526e-01]
 [ 2.1000e+01 -1.8890e-02]
 [ 2.3000e+01  2.0544e-01]
 [ 2.9000e+01  6.0210e-02]
 [ 5.7000e+01 -1.6959e-01]
 [ 5.8000e+01  9.4660e-02]
 [ 6.1000e+01  1.0680e-02]]
public alpha 0.0 RMSE [0.103784 0.38861  0.016458] max [1.35308  2.230054 0.091341] hazard censor disagreements 0
public alpha 0.5 RMSE [0.054383 0.203727 0.016274] max [0.695658 1.144688 0.091341] hazard censor disagreements 0
public alpha 0.75 RMSE [0.030247 0.112427 0.016239] max [0.366947 0.649665 0.091341] hazard censor disagreements 0
public alpha 1.0 RMSE [0.010727 0.032186 0.016232] max [0.08574  0.159116 0.091341] hazard censor disagreements 0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cfbcfd8dad4b8ea4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_cfbcfd8d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38103 ms` LLM)
- **Rationale**: Promote the candidate that materially improves the missing-feature bottleneck without changing public hazard censoring. Verify the actual artifact in an isolated process and check sparse calibration with a contaminated sweep; a failure would change the fallback or robust line fit.
- **Alternatives Weighed**: `Exclude incomplete-feature training coupons` (rejected: This did not improve the worst held-out-lab median endpoint error after rate correction.); `Replace predicted drift with observed-rate drift on every coupon` (rejected: Full correction worsened the worst held-out-lab complete-feature residuals by propagating measurement noise.)
- **ATMS NOGOOD DECLARED**: Refuted `Cell-balanced feature imputation alone recovers the coupon-specific creep drift when a feature is missing.` | Witness: `On held-out labs, removing the initial-rate innovation from the missing-feature drift leaves a worst-lab median endpoint error of 1.072 versus 0.098 with the correction.` | Blanket Diagnosis: `Imputation estimates the population feature mean, not the latent value of the coupon. After subtracting the modeled memory contribution, the measured initial rate directly informs coupon-specific drift.` | Sublated Pivot: `Keep the required aligned features and shared memory model; update only missing-feature drift with the measured initial rate when it is available.`

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_w02_batch_02_solver.py')
s=p.read_text().replace('range(6):','range(9):').replace('(1+(r/2.5)**2)**(-0.25)','(1+(r/2.5)**2)**(-0.5)').replace("loss='soft_l1'", "loss='cauchy'")
s=s.replace('    initial = test[\'strain_0\']\n','''    # A missing feature leaves the coupon-specific drift uncertain. The measured
    # initial rate identifies that drift once the two memory rates are removed.
    if 'instant_rate_obs' in test:
        measured = np.asarray(test['instant_rate_obs'])
        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(measured)
        drift[use] = measured[use] - amp[use]@(1/tau)
    initial = test['strain_0']
''')
p.write_text(s)
PY
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np,time
sys.path.insert(0,'/holdout_task')
import alpha_w02_batch_02_solver as m
entries=pickle.load(open('public_alpha_w02_batch_02.pkl','rb'))
print('public entries',len(entries))
for e in entries:
 d,g=e['dataset'],e['gt']; start=time.perf_counter(); o=m.solve_alpha_w02_batch_02(d); print('seconds',round(time.perf_counter()-start,3)); print('tau relative error',abs(o['estimated_taus']/g['taus']-1))
 s=d['test']; hold=m._inputs(d)[1][-len(s['lab_id']):]; tm=d['eval_times']; miss=np.isnan(s['coupon_features_raw']).any(axis=1)
 assert np.allclose(o['secant_compliance'],(o['predicted_strain'][:,-1]-s['strain_0'])/np.maximum(abs(hold),.5))
 for k,j in [('predicted_strain','te_clean'),('secant_compliance','te_secant'),('hazard_crossing_times','te_cross')]:
  r=o[k]-g[j]; rms=lambda a: float(np.sqrt(np.mean(a*a)))
  print(k,'RMSE',round(rms(r),6),'worst lab/stratum RMSE',round(max(rms(r[(s['lab_id']==lab)&(s['stratum']==st)]) for lab,st in np.unique(np.c_[s['lab_id'],s['stratum']],axis=0)),6),'missing/complete',np.round([rms(r[m]) for m in (miss,~miss) if m.any()],6),'max',round(np.max(abs(r)),6))
 assert np.all((o['hazard_crossing_times']>=0)&(o['hazard_crossing_times']<=tm[-1])); print('censor disagreements',np.sum((o['hazard_crossing_times']==tm[-1])!=(g['te_cross']==tm[-1])))
 ch=d['calibration_load_cells']; lab=ch['lab_id']; x=ch['cmd_stress']; y=ch['true_stress'].copy(); y[np.flatnonzero(lab==2)[1]]+=1.2
 keep=(lab!=3); keep[np.flatnonzero(lab==3)[0]]=True
 est=m._calibration(dict(lab_id=lab[keep],cmd_stress=x[keep],true_stress=y[keep]),d['yield_stress']); err=max(np.max(abs(est[:,0]-g['stress_gains'])),np.max(abs(est[:,1]-g['stress_biases'])))
 print('sparse spot check + EMI calibration max coefficient error',err); assert err<.001
PY
```
- **Container Exit Code**: `0` (`911 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_batch_02_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public entries 1
seconds 0.087
tau relative error [0.00193841 0.00442526]
predicted_strain RMSE 0.010813 worst lab/stratum RMSE 0.018237 missing/complete [0.020009 0.009473] max 0.088571
secant_compliance RMSE 0.032547 worst lab/stratum RMSE 0.076671 missing/complete [0.071255 0.025962] max 0.164368
hazard_crossing_times RMSE 0.01611 worst lab/stratum RMSE 0.046787 missing/complete [0.001998 0.016854] max 0.082664
censor disagreements 0
sparse spot check + EMI calibration max coefficient error 3.956191122956894e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d712bdde27e25474 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d712bdde (directory).
```

### Turn 8 — `submit_completed_task` (`12756 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated schema verification and sparse-plus-EMI calibration stress testing. Worst public lab/stratum RMSE was 0.0182 for strain, 0.0767 for secant compliance, and 0.0468 for hazard crossing time.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_w02_batch_02_solver.py': 'FILE:8384'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w02_batch_02_solver.py`. The public schema passed, including all four required outputs. Public RMSE was 0.0108 for strain, 0.0325 for secant compliance, and 0.0161 for hazard crossing time, with no censoring disagreements.`

### Turn 9 — `execute_in_remote_docker` (`38803 ms` LLM)
- **Rationale**: Check whether nonlinear drift terms improve held-out labs, whether signed residuals show a coherent lab offset within strata, and whether the hazard routine misses crossings before a trajectory turns. A clear improvement or a crossing discrepancy would require changing the corresponding component.
- **Alternatives Weighed**: `Add Fokker–Planck/HJB dynamics, relay modes, or classification metrics` (rejected: The measured system has deterministic coupon memory and continuous outcomes, with no relay state or labeled classification task. Lab-by-stratum signed residuals are relevant to checking nuisance confounding.)

```bash
cat > /tmp/polymer_gate_audit.py <<'PY'
import pickle, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '/holdout_task')
import alpha_w02_batch_02_solver as m
with open('/holdout_task/public_alpha_w02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
t = d['train']; n = len(t['lab_id']); labs = t['lab_id']; st = t['stratum']
a = m._inputs(d); tr = tuple(v[:n] for v in a); dt, ts = d['past_dt'], d['eval_times']
y = np.c_[t['strain_curves_obs']-t['strain_0'][:,None], t['instant_rate_obs']]
missing = np.isnan(t['coupon_features_raw']).any(axis=1)
original = m._components
# Pairwise linear combinations are already in the affine span; check nonlinear
# stress and feature couplings, and a fixed saturating stress denominator.
def expanded(inputs, dt, tau, variant):
    delta, z = original(inputs, dt, tau)
    past, hold, mem, feat = inputs
    if variant == 1:
        i,j = np.triu_indices(feat.shape[1]); z = np.c_[z, feat[:,i]*feat[:,j], hold[:,None]*z]
    if variant == 2:
        q = hold/d['yield_stress']; z = np.c_[z, z*(q*q/(1+q*q))[:,None]]
    return delta,z
m._components = expanded
for v in range(3):
    r = np.empty_like(y)
    for lab in np.unique(labs):
        test = labs==lab; train = ~test
        tau,c = m._fit(tuple(w[train] for w in tr),dt,ts,y[train],v)
        r[test] = m._design(tuple(w[test] for w in tr),dt,ts,tau,v)@c-y[test]
    r[missing,:-1] -= r[missing,-1,None]*ts
    print('basis',v,'worst lab complete/missing median curve error',*[round(max(np.median(abs(r[(labs==lab)&(missing==flag),:-1])) for lab in np.unique(labs)),6) for flag in (False,True)],'worst complete lab endpoint median AE', round(max(np.median(abs(r[(labs==lab)&~missing,-2])) for lab in np.unique(labs)),6))
    if v==0:
        print('complete-case rate residual signed median by lab x stratum',np.array([[np.median(r[(labs==lab)&(st==s)&~missing,-1]) for s in np.unique(st)] for lab in np.unique(labs)]).round(5))
m._components = original
u=d['test']; tau,c=m._fit(tr,dt,ts,y); test=tuple(w[n:] for w in a)
delta,z=m._components(test,dt,tau); amp=delta*c[:2]; drift=z@c[2:]; miss=np.isnan(u['coupon_features_raw']).any(axis=1); drift[miss]=u['instant_rate_obs'][miss]-amp[miss]@(1/tau)
# Check first crossing against a fine bracketing grid on fitted real coupons,
# including thresholds just below an interior maximum when one is present.
grid=np.linspace(0,ts[-1],20001); initial=u['strain_0']; threshold=u['critical_strain'].copy(); errors=[]; turning=0; pulses=0
for pass_id in range(2):
    pred=m._hazard(initial,threshold,ts,tau,amp,drift)
    for i in range(len(initial)):
        def value(t): return initial[i]+(-np.expm1(-np.asarray(t)[...,None]/tau))@amp[i]+drift[i]*t
        wave=value(grid); j=np.argmax(wave)
        hits=np.flatnonzero(wave>=threshold[i]); ref=ts[-1] if not len(hits) else (0. if hits[0]==0 else brentq(lambda t:value(t)-threshold[i],grid[hits[0]-1],grid[hits[0]]))
        errors.append(abs(ref-pred[i]))
        if pass_id==0:
            turning+=int(np.any(np.diff(wave)[:-1]*np.diff(wave)[1:]<0))
            if 0<j<len(grid)-1 and wave[j]>max(wave[0],wave[-1]):
                threshold[i]=wave[j]-.01*(wave[j]-max(wave[0],wave[-1])); pulses+=1
print('real fitted turning trajectories',turning,'interior-only crossing stress cases',pulses,'max first-crossing discrepancy',max(errors)); assert max(errors)<1e-8
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/polymer_gate_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2342 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
basis 0 worst lab complete/missing median curve error 0.011625 0.013001 worst complete lab endpoint median AE 0.041995
complete-case rate residual signed median by lab x stratum [[-8.00e-05 -9.70e-03 -5.20e-04]
 [-2.74e-03  3.64e-03  8.88e-03]
 [ 3.46e-03  3.04e-03  2.16e-03]
 [-6.50e-04  4.96e-03  2.90e-04]]
basis 1 worst lab complete/missing median curve error 0.01248 0.013275 worst complete lab endpoint median AE 0.042704
basis 2 worst lab complete/missing median curve error 0.011266 0.012978 worst complete lab endpoint median AE 0.040561
real fitted turning trajectories 22 interior-only crossing stress cases 7 max first-crossing discrepancy 4.78617145915905e-13
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d712bdde27e25474 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d712bdde (directory).
```

### Turn 10 — `submit_completed_task` (`14872 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated schema verification and the focused stress audit. Hazard crossings matched an independent bracketing calculation to within 4.8e-13, and held-out signed rate residuals showed no consistent lab offset across strata.
- **Alternatives Weighed**: `Add quadratic, stress-interaction, or saturating drift terms` (rejected: Quadratic and interaction terms worsened held-out errors. The saturating basis reduced worst-lab complete-case median endpoint error only from 0.0420 to 0.0406 and did not address the incomplete-feature bottleneck.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_w02_batch_02_solver.py': 'FILE:8384'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w02_batch_02_solver.py`. The public schema passed. Public RMSE was 0.0108 for strain, 0.0325 for secant compliance, and 0.0161 for hazard crossing time, with no censoring disagreements. The solver also passed sparse and noisy calibration checks and correctly found all seven stress-tested crossings that occurred before an interior peak.`

### Turn 11 — `execute_in_remote_docker` (`24644 ms` LLM)
- **Rationale**: The previous calibration stress test used a single spot-check and EMI placement. Sweep both wings and every observed placement to detect whether an unfavorable spot makes the archive fallback miscalibrate physical stress; if it does, revise that fallback before submission.
- **Alternatives Weighed**: `Repeat the nonlinear drift and hazard audits` (rejected: Those checks already established a marginal saturating-basis gain, no consistent within-stratum lab offset, and agreement on all seven interior-only crossings.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_w02_batch_02_solver as m
with open('public_alpha_w02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
current = d['calibration_load_cells']
history = json.loads(m._HISTORY.read_text())
yield_stress = d['yield_stress']
reference = m._calibration(current, yield_stress)
inputs = m._inputs(d)
print('archive sweeps', len(history))
for j, sweep in enumerate(history):
    table = m._calibration(sweep, yield_stress)
    print('archive sweep', j, 'gain/bias differences from current by lab', np.round(table-reference, 6).tolist())
labs = current['lab_id']
worst = np.zeros(4)
for wing in (0, 1):
    pair = (2*wing, 2*wing+1)
    for sparse in pair:
        noisy = pair[1] if sparse == pair[0] else pair[0]
        for spot in np.flatnonzero(labs == sparse):
            for emi in np.flatnonzero(labs == noisy):
                keep = labs != sparse
                keep[spot] = True
                checks = {key: np.asarray(val)[keep].copy() for key, val in current.items()}
                checks['true_stress'][np.count_nonzero(keep[:emi])] += 1.2
                table = m._calibration(checks, yield_stress)
                for split in ('train', 'test'):
                    data = d[split]
                    for lab in pair:
                        rows = data['lab_id'] == lab
                        cmds = np.r_[data['stress_hold_cmd'][rows], data['past_stress_cmd'][rows].ravel()]
                        gain, bias = table[lab]-reference[lab]
                        worst[lab] = max(worst[lab], np.max(abs(gain*cmds+bias)))
print('worst physical stress deviation by lab over every single-spot / single-EMI placement', worst.tolist())
print('fit stress consensus scale', 0.001*yield_stress)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`2669 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
archive sweeps 2
archive sweep 0 gain/bias differences from current by lab [[1.8e-05, 2e-05], [-5.5e-05, 1.4e-05], [3.3e-05, 2.8e-05], [-6.9e-05, 8.9e-05]]
archive sweep 1 gain/bias differences from current by lab [[5.1e-05, -2e-05], [3e-06, 3.2e-05], [4.3e-05, 2.2e-05], [1.2e-05, 9.9e-05]]
worst physical stress deviation by lab over every single-spot / single-EMI placement [0.0001878700569908026, 0.00012091485142965583, 0.0001914234131937624, 0.0001808766674315306]
fit stress consensus scale 0.0029500000000000004
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d712bdde27e25474 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d712bdde (directory).
```

### Turn 12 — `submit_completed_task` (`13751 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema verification after the two-wing calibration audit. Independent bracketing verified the earliest crossing in all seven interior-only cases, and the held-out drift comparison did not justify adding nonlinear terms.
- **Alternatives Weighed**: `Revise the archive fallback after the two-wing calibration audit` (rejected: Every tested placement of a sparse check and EMI outlier kept the physical stress deviation below 0.000192, versus a calibration consensus scale of 0.00295.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w02_batch_02_solver.py`. All four outputs passed the public schema validator. Public RMSE was 0.0108 for strain, 0.0325 for secant compliance, and 0.0161 for hazard crossing time, with no censoring disagreements. Across both wings, the sparse and noisy calibration stress tests changed calibrated stress by at most 0.000192.`

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
