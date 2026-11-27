# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `16 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `470,196` (`456,392` / `125,952` / `13,804`) |
| **Total Episode Wall Time (s)** | `465.51s` |

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

### Turn 1 — `execute_in_remote_docker` (`12196 ms` LLM)
- **Rationale**: Inspect the public schema and available prior implementations before fitting so the solver uses the actual input fields and separates reusable calibration from the new regime.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /holdout_task/public_asset_index.json; wc -l /tmp/hegel_continual_archive/*.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`287 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
      "bytes": 1111
    }
  },
  "schema": "hegel_public_asset_index_v1"
}  204 /tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py
  180 /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
  211 /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py
  169 /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py
  764 total
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`15664 ms` LLM)
- **Rationale**: Identify which channels need historical calibration and inspect the previous alignment and fit conventions. A mismatch in panel fields, time grids, or calibration coverage would change preprocessing before any fit.

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']
def walk(o,p=''):
 for k,v in o.items():
  if isinstance(v,dict): print(p+k); walk(v,p+'  ')
  else:
   a=np.asarray(v); print(p+k,a.shape,a.dtype,a if a.size<=24 else (np.unique(a,return_counts=True) if k in ['lab_id','stratum'] else ''))
walk(x)
PY
cat /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; head -85 /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`559 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[
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
        },
        {
          "field": "estimated_shared_scanner_gain",
          "op": ">",
          "value": 0
        },
        {
          "field": "observations_per_training_center_class_cell",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_control_panel",
          "signature": "align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
        "autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_control_panel_deconfounding",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`21723 ms` LLM)
- **Rationale**: Inspect only the prior preprocessing code and compact summaries. Disagreement between historical sweeps would change the channel fallback; the prior panel reshape and observed missingness determine the alignment and imputation implementation.
- **Alternatives Weighed**: `Use the scanner-control alignment primitive` (rejected: Its shared scanner gain does not match the required per-lab, per-feature coupon calibration.)

```bash
sed -n '1,145p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
z=json.load(open(p)); print('archive structure',type(z).__name__, len(z))
for r in (z if isinstance(z,list) else z.values()):
 if isinstance(r,dict):
  print('sweep keys',list(r));
  if 'lab_id' in r:
   l,x,y=map(lambda k:np.array(r[k]),('lab_id','cmd_stress','true_stress'))
   for c in np.unique(l):
    m=l==c; b=np.polyfit(x[m],y[m],1); print(c,m.sum(),b,np.max(abs(y[m]-np.polyval(b,x[m]))))
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']
for s in ('train','test'):
 q=d[s]; print(s,'missing', {k:int(np.isnan(v).sum()) for k,v in q.items()},'panel first row',q['ref_coupons'][0,:8]);
 if s=='train':print('observations range',np.nanmin(q['strain_curves_obs']),np.nanmax(q['strain_curves_obs']),np.nanmin(q['instant_rate_obs']),np.nanmax(q['instant_rate_obs']))
PY
```
- **Container Exit Code**: `0` (`512 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
:] * flow[None, :, None]
    a[:, -1, 2:7] = xd
    a[:, :-1, 7] = slip[:, None] * flow
    a[:, -1, 7] = slip
    a[:, :-1, 8] = diff[:, None] * relaxation[:, 0]
    a[:, -1, 8] = diff / tau[0]
    return a


def _fit(inputs, dt, times, y):
    lo = np.log(max(min(np.min(dt), np.min(times[times > 0])) / 20, 1e-5))
    hi = np.log(max(np.sum(dt), np.max(times)) * 20)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)],
              np.r_[hi, hi, np.full(9, np.inf)])
    valid = np.isfinite(y)
    yobs = y[valid]

    # Interpolation in the terminal-flow coordinate removes coupon drift.
archive structure list 4
sweep keys ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']
0 10 [9.99996399e-01 1.32406004e-05] 0.00017965296462654767
1 10 [ 9.99987895e-01 -2.46087894e-05] 0.0002542608844398053
2 10 [1.00004377e+00 1.86490255e-05] 0.00018935354483656797
3 10 [ 1.00004220e+00 -8.39804234e-05] 0.00012129383951794276
sweep keys ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']
0 10 [1.28998901 0.12000156] 0.00021700626342929796
1 10 [ 0.82003898 -0.15002602] 0.00014183185140925758
2 10 [1.16998089 0.14000927] 0.0001620203768579742
3 10 [ 0.76000152 -0.10998804] 0.0001719282826188693
sweep keys ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']
0 10 [1.28996334 0.11998329] 0.0001703216589850065
1 10 [ 0.82003341 -0.15000563] 0.00013444553809754956
2 10 [1.16998653 0.13997717] 0.0001720728489744694
3 10 [ 0.75996642 -0.11003498] 0.00016497030799489387
sweep keys ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']
0 10 [1.28996282 0.1199782 ] 0.0001717193952429641
1 10 [ 0.81997725 -0.14996309] 0.00024491625334988054
2 10 [1.16999803 0.14000889] 0.00017227354629523184
3 10 [ 0.76000326 -0.11003605] 0.0001500469101773927
train missing {'lab_id': 0, 'stratum': 0, 'coupon_features_raw': 25, 'ref_coupons': 223, 'past_stress_cmd': 0, 'stress_hold_cmd': 0, 'strain_0': 0, 'aux_chamber_telemetry': 0, 'strain_curves_obs': 0, 'instant_rate_obs': 0} panel first row [-1.91861115 -0.74924596 -0.50082072 -0.14166332  0.23250785 -1.75118945
 -1.2213071  -0.51945229]
observations range -38.49181180281658 25.68671670652653 -5.229231681391637 4.732936595212264
test missing {'lab_id': 0, 'stratum': 0, 'coupon_features_raw': 6, 'ref_coupons': 85, 'past_stress_cmd': 0, 'stress_hold_cmd': 0, 'strain_0': 0, 'aux_chamber_telemetry': 0} panel first row [-1.52833751 -0.66558964  0.22465327 -1.23528731 -1.40969444 -0.48761121
 -1.54429229 -0.73829733]
```

### Turn 4 — `execute_in_remote_docker` (`40243 ms` LLM)
- **Rationale**: Reuse archived calibration and coupon preprocessing, then implement the requested curve and initial-rate equations in one design. The public schema check will expose integration errors before held-out validation.
- **Alternatives Weighed**: `Fit relaxation times and linear weights jointly` (rejected: Profiling solves the linear coefficients exactly for each pair of relaxation times and reduces the nonlinear search to two dimensions.)

```bash
set -e
p=/holdout_task/alpha_m97_c03_batch_04_solver.py
sed '/^def _design/,$d' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py | sed '1c\"""Calibrated two-branch Prony model for strain-crystallized flow."""' > "$p"
sed -n '56,127p' "$p"
cat >> "$p" <<'PY'
def _design(logtau, inputs, dt, times, crit, crystallizes):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = np.exp(-tail[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    z = stress @ kernel
    gap = hold[:, None] - z
    damage = np.maximum(np.abs(stress) - crit, 0.0)**2 @ dt
    act = np.tanh(damage) if crystallizes else np.zeros_like(damage)
    flow = np.where(act[:, None] > 0, np.log1p(times), times)
    decay = -np.expm1(-times[:, None] / tau)
    a = np.empty((len(hold), len(times) + 1, 9))
    a[:, :-1, :2] = gap[:, None, :] * decay
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:7] = xd[:, None, :] * flow[:, :, None]
    a[:, -1, 2:7] = xd
    a[:, :-1, 7] = (act * hold)[:, None] * decay[:, 0]
    a[:, -1, 7] = act * hold / tau[0]
    a[:, :-1, 8] = (act * z[:, 0])[:, None] * flow
    a[:, -1, 8] = act * z[:, 0]
    return a


def _fit(inputs, dt, times, crit, crystallizes, y):
    # Profile all nine linear coefficients at each pair of relaxation times.
    valid = np.isfinite(y)
    target = y[valid]
    lo = min(np.min(dt), np.min(times[times > 0])) / 10
    hi = max(np.sum(dt), np.max(times)) * 10

    def profile(logtau, return_weights=False):
        a = _design(logtau, inputs, dt, times, crit, crystallizes)[valid]
        weights = np.linalg.lstsq(a, target, rcond=1e-11)[0]
        return weights if return_weights else a @ weights - target

    starts = ((0.45, 2.5), (1.2, 6.0), (0.22, 1.3), (3.0, 0.65))
    fits = [least_squares(profile, np.log(np.clip(seed, lo * 1.01, hi / 1.01)),
                          bounds=(np.log(lo), np.log(hi)), max_nfev=100,
                          ftol=1e-10, gtol=1e-9, xtol=1e-9) for seed in starts]
    best = min(fits, key=lambda fit: fit.cost)
    logtau = best.x
    if not crystallizes:
        logtau = np.sort(logtau)
    return logtau, profile(logtau, return_weights=True)


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    lines = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    train_inputs = _inputs(train, x_train, lines)
    test_inputs = _inputs(test, x_test, lines)
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    crit = float(dataset['crit_stress'])
    crystallizes = dataset['shear_banding_index'] <= 0.50
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    logtau, weights = _fit(train_inputs, dt, times, crit, crystallizes, y)
    pred = _design(logtau, test_inputs, dt, times, crit, crystallizes) @ weights
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
PY
grep '^def' "$p"
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`591 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
else:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            line = _historical_line(history, c)
            if line is None:
                raise ValueError(f'No usable calibration for channel {c}')
            lines[c] = line
    return lines


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    return np.where(np.isfinite(aligned), aligned, fills[ids])


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _historical_line(records, channel):
def _clean_sweep(x, y):
def _line(x, y):
def _calibrate(cal):
def _align(train, test):
def _inputs(part, aligned, lines):
def _design(logtau, inputs, dt, times, crit, crystallizes):
def _fit(inputs, dt, times, crit, crystallizes, y):
def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=16b28f9a1820ce9e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_16b28f9a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 9, in <module>
    out = mod.solve_alpha_m97_c03_batch_04(ds)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/holdout_task/alpha_m97_c03_batch_04_solver.py", line 162, in solve_alpha_m97_c03_batch_04
    x_train, x_test = _align(train, test)
    ^^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)
```

### Turn 5 — `execute_in_remote_docker` (`30307 ms` LLM)
- **Rationale**: Restore the train/test split after pooled reference alignment, then check the schema and refit without each held-out lab’s responses. Separating active and inactive rows and auditing initial rate independently will show whether any remaining error is concentrated in a specific regime or output.

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_04_solver.py')
s = p.read_text()
old = '    return np.where(np.isfinite(aligned), aligned, fills[ids])'
assert s.count(old) == 1
p.write_text(s.replace(old, "    completed = np.where(np.isfinite(aligned), aligned, fills[ids])\n    n_train = len(train['lab_id'])\n    return completed[:n_train], completed[n_train:]"))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/rheology04_lolo.py <<'PY'
import importlib.util
import pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b = d['train'], d['test']
solve = m.solve_alpha_m97_c03_batch_04
xc, _ = m._align(a,b)
stress, _, _ = m._inputs(a,xc,m._calibrate(d['calibration_load_cells']))
active = (np.maximum(abs(stress)-d['crit_stress'],0)**2 @ d['past_dt']) > 0
missing = np.isnan(a['coupon_features_raw']).any(axis=1)
curve = np.empty_like(a['strain_curves_obs']); rate = np.empty_like(a['instant_rate_obs'])
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    fold = dict(d, train={k: v[~val] for k,v in a.items()}, test={k: np.concatenate((a[k][val],v)) for k,v in b.items()})
    o = solve(fold)
    curve[val] = o['predicted_strain'][:val.sum()]; rate[val] = o['initial_rate_pred'][:val.sum()]
    print('fold taus', lab, o['estimated_taus'])
ce = curve-a['strain_curves_obs']; re = rate-a['instant_rate_obs']
def row(name, use):
    if use.any():
        print(name, int(use.sum()), 'curve_rmse', round(np.sqrt(np.mean(ce[use]**2)),6), 'rate_rmse', round(np.sqrt(np.mean(re[use]**2)),6))
for lab in np.unique(a['lab_id']):
    val=a['lab_id']==lab
    row(('lab',lab),val)
    row(('lab inactive',lab),val & ~active)
    row(('lab active',lab),val & active)
row('missing',missing); row('complete',~missing)
print('time RMSE',np.sqrt(np.mean(ce**2,axis=0)))
o=solve(d)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,o[k])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_lolo.py
```
- **Container Exit Code**: `0` (`1163 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
fold taus 0 [0.57173571 5.29867323]
fold taus 1 [0.58903865 5.56032371]
fold taus 2 [0.5243308 2.4078591]
fold taus 3 [0.57075108 7.89825421]
('lab', 0) 48 curve_rmse 1.89187 rate_rmse 0.148021
('lab inactive', 0) 13 curve_rmse 0.249487 rate_rmse 0.131227
('lab active', 0) 35 curve_rmse 2.210307 rate_rmse 0.153792
('lab', 1) 43 curve_rmse 0.463059 rate_rmse 0.161822
('lab inactive', 1) 14 curve_rmse 0.278382 rate_rmse 0.096347
('lab active', 1) 29 curve_rmse 0.529648 rate_rmse 0.185329
('lab', 2) 42 curve_rmse 1.150227 rate_rmse 0.125464
('lab inactive', 2) 18 curve_rmse 0.353651 rate_rmse 0.124858
('lab active', 2) 24 curve_rmse 1.490465 rate_rmse 0.125916
('lab', 3) 47 curve_rmse 0.746456 rate_rmse 0.135979
('lab inactive', 3) 15 curve_rmse 0.984648 rate_rmse 0.144916
('lab active', 3) 32 curve_rmse 0.603253 rate_rmse 0.13158
missing 24 curve_rmse 0.801254 rate_rmse 0.163096
complete 156 curve_rmse 1.259241 rate_rmse 0.140271
time RMSE [0.7189166  0.2288354  0.18192314 0.4749008  0.24971226 2.98973623
 0.76651744 0.30818952 0.95659202 1.78110385]
estimated_taus [0.57719716 4.88152286]
hereditary_weights [ 0.74030002  0.5388985   0.06651541  0.1076182  -0.08129006  0.06685005
 -0.04877011]
regime_mode_weights [0.29842179 0.23473283]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=062a6384caf1c57f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_062a6384 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`21828 ms` LLM)
- **Rationale**: Distinguish isolated corrupt cells from errors that persist across a specimen’s trajectory. Isolated extremes call for cellwise robust fitting; coherent errors would instead warrant inspecting alignment, gating, or the response basis.
- **Alternatives Weighed**: `Add physical terms to the response model` (rejected: The elevated error at isolated horizons could come from bad observations; check residual patterns before changing the prescribed physical basis.)

```bash
sed '/^xc, _/,$d' /tmp/rheology04_lolo.py > /tmp/rheology04_residual.py
cat >> /tmp/rheology04_residual.py <<'PY'
xc, _ = m._align(a,b)
z = m._inputs(a,xc,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
y=np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
lt,w=m._fit(z,dt,t,crit,True,y)
r=y-m._design(lt,z,dt,t,crit,True)@w
print('absolute residual quantiles by horizon and rate (last):')
print(np.round(np.quantile(abs(r),[.25,.5,.75,.9,.95,.98,.995,1],axis=0),5))
print('largest residuals',[(int(i),int(j),round(r[i,j],3)) for i,j in zip(*np.where(abs(r)>2))]); print('per-row counts',np.unique((abs(r)>.5).sum(axis=1),return_counts=True))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_residual.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS nonlinear variable projection' | head -25
```
- **Container Exit Code**: `0` (`690 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.588000e-01]
 [7.450000e-02 1.584800e-01 1.810400e-01 2.404100e-01 2.517600e-01
  2.989700e-01 3.389500e-01 3.802200e-01 4.764500e-01 5.413800e-01
  2.216600e-01]
 [1.238600e-01 2.260300e-01 2.295900e-01 3.815300e-01 3.244000e-01
  3.677500e-01 4.712800e-01 4.925200e-01 5.998700e-01 6.562200e-01
  2.554100e-01]
 [8.859100e-01 5.626000e-01 2.834000e-01 1.217960e+00 5.221600e-01
  5.561300e-01 1.548510e+00 6.393100e-01 1.156180e+00 1.822860e+00
  2.979900e-01]
 [6.422320e+00 1.615250e+00 8.073400e-01 3.517080e+00 1.264980e+00
  1.051347e+01 4.909250e+00 1.357220e+00 3.194660e+00 6.502400e+00
  3.428100e-01]
 [6.476090e+00 2.017500e+00 1.474860e+00 3.536410e+00 1.517610e+00
  3.904139e+01 6.618790e+00 1.482460e+00 1.150447e+01 2.223453e+01
  3.567200e-01]]
largest residuals [(13, 8, 2.22), (15, 3, 3.515), (16, 0, -6.416), (37, 6, 6.619), (39, 1, -2.017), (39, 5, 7.167), (51, 9, 2.257), (54, 5, -39.041), (54, 6, -4.709), (75, 0, 6.476), (89, 6, 4.163), (89, 9, 4.279), (92, 3, -2.413), (96, 9, 4.657), (118, 3, 3.536), (118, 9, 22.235), (149, 6, -2.592), (166, 0, 2.7), (168, 8, -11.504)]
per-row counts (array([0, 1, 2, 3, 4]), array([119,  44,  13,   3,   1]))
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=062a6384caf1c57f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_062a6384 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`34647 ms` LLM)
- **Rationale**: The largest curve residual is 39.0, while most specimens show no residual above 0.5. Preserve the physical basis and estimate robust weights inside each training fold. Compare both candidates on the same held-out labs and report tail and median errors alongside raw RMSE, which can be dominated by a single bad reading.
- **Alternatives Weighed**: `Drop entire specimens with a large residual` (rejected: Most extreme residuals are isolated curve measurements; specimen-level rejection would discard useful neighboring measurements.)

```bash
set -e
p=/holdout_task/alpha_m97_c03_batch_04_solver.py
cp "$p" /tmp/rheology04_ols.py
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_04_solver.py')
s = p.read_text()
a = s.index('    def profile(', s.index('def _fit('))
b = s.index('\n\ndef solve_', a)
s = s[:a] + '''    sqrt_weight = np.ones(target.shape)

    def profile(logtau, return_weights=False):
        a = _design(logtau, inputs, dt, times, crit, crystallizes)[valid]
        aw = a * sqrt_weight[:, None]
        weights = np.linalg.lstsq(aw, target * sqrt_weight, rcond=1e-11)[0]
        return weights if return_weights else (a @ weights - target) * sqrt_weight

    def optimize(start):
        return least_squares(profile, start, bounds=(np.log(lo), np.log(hi)),
                             max_nfev=100, ftol=1e-10, gtol=1e-9, xtol=1e-9)

    starts = ((0.45, 2.5), (1.2, 6.0), (0.22, 1.3), (3.0, 0.65))
    fits = [optimize(np.log(np.clip(seed, lo * 1.01, hi / 1.01))) for seed in starts]
    logtau = min(fits, key=lambda fit: fit.cost).x
    # Isolated readout spikes affect individual cells, not entire specimens.
    # Estimate a separate residual scale for each curve horizon and the rate.
    for _ in range(12):
        resid = _design(logtau, inputs, dt, times, crit, crystallizes) @ profile(logtau, True) - y
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)
        scale = np.maximum(2.385 * 1.4826 * mad, 1e-6)
        sqrt_weight = (1 / np.hypot(1, resid / scale))[valid]
        logtau = optimize(logtau).x
    if not crystallizes:
        logtau = np.sort(logtau)
    return logtau, profile(logtau, return_weights=True)
''' + s[b:]
p.write_text(s)
PY
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/rheology04_lolo.py')
s=p.read_text().replace("root / 'alpha_m97_c03_batch_04_solver.py'", "Path(__import__('sys').argv[1])")
s=s.replace("round(np.sqrt(np.mean(re[use]**2)),6))", "round(np.sqrt(np.mean(re[use]**2)),6), 'curve_medianAE/p90AE', np.round(np.quantile(abs(ce[use]), [.5,.9]), 6), 'rate_medianAE/p90AE', np.round(np.quantile(abs(re[use]), [.5,.9]), 6))")
Path('/tmp/rheology04_compare.py').write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
for f in /tmp/rheology04_ols.py "$p"; do echo "=== $f"; python3 -I /tmp/rheology04_compare.py "$f"; done
cd /holdout_task; python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`2216 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
e_rmse 0.140978 curve_medianAE/p90AE [0.10888  0.393128] rate_medianAE/p90AE [0.09208  0.235197]
('lab inactive', 0) 13 curve_rmse 0.218168 rate_rmse 0.097538 curve_medianAE/p90AE [0.067938 0.333442] rate_medianAE/p90AE [0.051411 0.157766]
('lab active', 0) 35 curve_rmse 2.197247 rate_rmse 0.154023 curve_medianAE/p90AE [0.131929 0.412662] rate_medianAE/p90AE [0.106484 0.249525]
('lab', 1) 43 curve_rmse 0.433177 rate_rmse 0.123513 curve_medianAE/p90AE [0.107226 0.303246] rate_medianAE/p90AE [0.09573  0.181037]
('lab inactive', 1) 14 curve_rmse 0.257329 rate_rmse 0.087825 curve_medianAE/p90AE [0.04838  0.223229] rate_medianAE/p90AE [0.082062 0.123557]
('lab active', 1) 29 curve_rmse 0.496246 rate_rmse 0.137465 curve_medianAE/p90AE [0.137616 0.329472] rate_medianAE/p90AE [0.118933 0.185538]
('lab', 2) 42 curve_rmse 1.143048 rate_rmse 0.124861 curve_medianAE/p90AE [0.075999 0.317993] rate_medianAE/p90AE [0.076972 0.200432]
('lab inactive', 2) 18 curve_rmse 0.35162 rate_rmse 0.121148 curve_medianAE/p90AE [0.041047 0.325978] rate_medianAE/p90AE [0.067228 0.183275]
('lab active', 2) 24 curve_rmse 1.481131 rate_rmse 0.127575 curve_medianAE/p90AE [0.102357 0.305949] rate_medianAE/p90AE [0.081726 0.203485]
('lab', 3) 47 curve_rmse 0.734361 rate_rmse 0.131194 curve_medianAE/p90AE [0.082125 0.356223] rate_medianAE/p90AE [0.090712 0.206266]
('lab inactive', 3) 15 curve_rmse 0.968011 rate_rmse 0.145177 curve_medianAE/p90AE [0.076709 0.400741] rate_medianAE/p90AE [0.136337 0.195588]
('lab active', 3) 32 curve_rmse 0.594 rate_rmse 0.124098 curve_medianAE/p90AE [0.083688 0.337384] rate_medianAE/p90AE [0.071895 0.208878]
missing 24 curve_rmse 0.785795 rate_rmse 0.155363 curve_medianAE/p90AE [0.143424 0.414756] rate_medianAE/p90AE [0.124377 0.225413]
complete 156 curve_rmse 1.248245 rate_rmse 0.126451 curve_medianAE/p90AE [0.087516 0.331784] rate_medianAE/p90AE [0.082594 0.198559]
time RMSE [0.71878603 0.22925467 0.1650441  0.46703869 0.22317976 2.97309656
 0.73644937 0.28530998 0.93965892 1.76538069]
estimated_taus [0.51706821 2.73027944]
hereditary_weights [ 0.62576608  0.55116883  0.07179932  0.09454776 -0.07412902  0.06459433
 -0.05651983]
regime_mode_weights [0.25812326 0.1959419 ]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`33585 ms` LLM)
- **Rationale**: The robust fit passes the public schema and improves both worst-slice errors. Compare its vectorized design against the explicit memory recurrence and rate equation: unlike cross-validation, this can detect indexing or gating errors that the fitted weights could partly mask.

```bash
set -e
cd /holdout_task
sed -n '1,80p' alpha_m97_c03_batch_04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
from pathlib import Path
import pickle
import numpy as np
p = Path('alpha_m97_c03_batch_04_solver.py')
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c03_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
lines = m._calibrate(d['calibration_load_cells'])
print('calibration gain/bias by lab:', lines)
a, b = d['train'], d['test']; xa, xb = m._align(a, b)
assert np.isfinite(xa).all() and np.isfinite(xb).all()
stress, hold, xd = m._inputs(a, xa, lines)
dt, t, crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
out = m.solve_alpha_m97_c03_batch_04(d)
assert all(np.isfinite(v).all() for v in out.values())
tau, hw, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
z = np.zeros((len(hold), 2))
for j, step in enumerate(dt):
    e = np.exp(-step/tau)
    z = z * e + stress[:, j, None] * (1-e)
gap = hold[:, None] - z
damage = (np.maximum(abs(stress)-crit,0)**2 * dt).sum(axis=1)
for gate in (False, True):
    active = gate & (damage > 0)
    act = np.where(active, np.tanh(damage), 0)
    drift = xd @ hw[2:] + rw[1]*act*z[:,0]
    decay = 1 - np.exp(-t[:,None]/tau)
    expected = (gap * hw[:2]) @ decay.T + rw[0]*(act * hold)[:,None]*decay[:,0] + drift[:,None] * np.where(active[:,None],np.log1p(t),t)
    expected_rate = (gap * hw[:2]/tau).sum(axis=1) + rw[0]*act*hold/tau[0] + drift
    pred = m._design(np.log(tau), (stress,hold,xd), dt,t,crit,gate) @ np.r_[hw,rw]
    err = np.max(abs(pred[:,:-1]-expected)); rate_err = np.max(abs(pred[:,-1]-expected_rate))
    print('gate',gate,'active',active.sum(),'max forward error',err,'max rate error',rate_err)
    np.testing.assert_allclose(pred[:,:-1], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pred[:,-1], expected_rate, atol=1e-12, rtol=1e-12)
print('all five outputs finite')
PY
```
- **Container Exit Code**: `0` (`678 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nstead.
    if x.size >= 3:
        candidates = []
        for omit in range(x.size):
            keep = np.arange(x.size) != omit
            if np.unique(x[keep]).size < 2:
                continue
            gain, bias = _line(x[keep], y[keep])
            residual = y - (gain * x + bias)
            # Clean public and prefix sweeps agree with an affine line well
            # within 0.01 stress units; require that agreement before removal.
            if np.max(np.abs(residual[keep])) < 0.01 and abs(residual[omit]) > 3.0:
                candidates.append((np.sum(residual[keep]**2), keep))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
    i, j = np.triu_indices(x.size, 1)
    distinct = x[j] != x[i]
    i, j = i[distinct], j[distinct]
    slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
    return np.abs(y - slope * x - np.median(y - slope * x)) <= 3.0


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(cal):
    lines = np.empty((4, 2))
    history = None
    lab, cmd, true = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    for c in range(4):
        take = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        clean = _clean_sweep(x, y)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or clean.all()):
            lines[c] = _line(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            line = _historical_line(history, c)
            if line is None:
                raise ValueError(f'No usable calibration for channel {c}')
            lines[c] = line
    return lines


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
calibration gain/bias by lab: [[ 1.2899706   0.11999929]
 [ 0.81997725 -0.14996309]
 [ 1.16999803  0.14000889]
 [ 0.76000326 -0.11003605]]
gate False active 0 max forward error 1.3322676295501878e-15 max rate error 1.3322676295501878e-15
gate True active 120 max forward error 1.3322676295501878e-15 max rate error 1.3322676295501878e-15
all five outputs finite

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 9 — `submit_completed_task` (`14403 ms` LLM)
- **Rationale**: The robust fit improved held-out performance without changing the prescribed physical equations. The final artifact passed schema validation and independently matched the explicit recurrence to within 1.4e-15 in an isolated Python process.
- **Alternatives Weighed**: `Keep the equal-weight fit` (rejected: It had higher worst-slice RMSE for both prediction outputs.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_04_solver.py': 'FILE:8301'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema passed, and the strain and initial-rate calculations matched the explicit recurrence in both gate states.`

### Turn 10 — `execute_in_remote_docker` (`37390 ms` LLM)
- **Rationale**: Audit held-out residuals within each lab and across damage levels. A consistent within-lab pattern would warrant investigating the physical basis or panel alignment; a pooled-only pattern would not justify adding a term. Quantify the prediction sensitivity to the published calibration discrepancy before considering a calibration change.
- **Alternatives Weighed**: `Run relay, classification, or Fokker–Planck checks` (rejected: The public inputs specify continuous strain and rate, with a scalar damage gate; they contain no relay state, classification label, or diffusion operator. Both applicable gate states already matched the explicit recurrence.)

```bash
set -e
sed '/^xc, _/,$d' /tmp/rheology04_lolo.py > /tmp/rheology04_gate_audit.py
cat >> /tmp/rheology04_gate_audit.py <<'PY'
from scipy.stats import spearmanr
lines = m._calibrate(d['calibration_load_cells'])
xc, _ = m._align(a,b)
stress,hold,_ = m._inputs(a,xc,lines)
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
damage = np.maximum(abs(stress)-crit,0)**2 @ dt
act = np.tanh(damage)
y = np.column_stack((a['strain_curves_obs'],a['instant_rate_obs']))
pred = np.empty_like(y)
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    fold = dict(d, train={k:v[~val] for k,v in a.items()}, test={k:np.concatenate((a[k][val],v)) for k,v in b.items()})
    o = solve(fold)
    pred[val] = np.column_stack((o['predicted_strain'][:val.sum()],o['initial_rate_pred'][:val.sum()]))
r = pred - y
for lab in np.unique(a['lab_id']):
    for name,slice_mask in [('inactive',act==0),('weak',(act>0)&(act<=0.2)),('strong',act>0.2)]:
        use = (a['lab_id']==lab) & slice_mask
        if use.any():
            print('lab/damage',lab,name,use.sum(),'curve/rate signed medians',np.round([np.median(r[use,:-1]),np.median(r[use,-1])],4),'curve/rate P90 abs',np.round([np.quantile(abs(r[use,:-1]),.9),np.quantile(abs(r[use,-1]),.9)],4))
# Probe coherent residual patterns, not isolated corrupted time cells.
z = np.zeros((len(hold),2))
o = solve(d)
for h,step in enumerate(dt):
    e = np.exp(-step/o['estimated_taus']); z = z*e + stress[:,h,None]*(1-e)
bases = {'hold': hold,'gap difference': z[:,1]-z[:,0], 'damage':act, 'activated memory': act*z[:,0], 'saturating memory':act*z[:,0]/(1+z[:,0]**2)}
for name,v in bases.items():
    rr = np.column_stack((np.median(r[:,:-1],axis=1),r[:,-1]))
    by_lab = [np.round([spearmanr(v[a['lab_id']==lab],q[a['lab_id']==lab]).statistic for q in rr.T],3).tolist() for lab in np.unique(a['lab_id'])]
    print('residual rank correlation',name,'pooled',np.round([spearmanr(v,q).statistic for q in rr.T],3),'by lab',by_lab)
# Measure the physical sensitivity to the published calibration discrepancy.
reference = np.array([[1.28994604,.12001392],[.81974036,-.14964349],[1.17001889,.1400143],[.7600149,-.11004836]])
_,xb = m._align(a,b)
u,v = (m._inputs(b,xb,c) for c in (lines,reference))
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
args = (dt,t,crit,d['shear_banding_index']<=.5)
delta = (m._design(np.log(o['estimated_taus']),u,*args)-m._design(np.log(o['estimated_taus']),v,*args)) @ w
print('max calibration delta in physical stress',np.max(abs(u[0]-v[0])),'in strain',np.max(abs(delta[:,:-1])),'in rate',np.max(abs(delta[:,-1])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_gate_audit.py
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1564 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab/damage 0 inactive 13 curve/rate signed medians [ 0.0078 -0.0487] curve/rate P90 abs [0.3334 0.1578]
lab/damage 0 strong 35 curve/rate signed medians [-0.0882 -0.1042] curve/rate P90 abs [0.4127 0.2495]
lab/damage 1 inactive 14 curve/rate signed medians [-0.0197 -0.0813] curve/rate P90 abs [0.2232 0.1236]
lab/damage 1 strong 29 curve/rate signed medians [-0.0471 -0.0517] curve/rate P90 abs [0.3295 0.1855]
lab/damage 2 inactive 18 curve/rate signed medians [ 0.005  -0.0672] curve/rate P90 abs [0.326  0.1833]
lab/damage 2 strong 24 curve/rate signed medians [-0.0584 -0.0568] curve/rate P90 abs [0.3059 0.2035]
lab/damage 3 inactive 15 curve/rate signed medians [ 0.0445 -0.1363] curve/rate P90 abs [0.4007 0.1956]
lab/damage 3 strong 32 curve/rate signed medians [-0.0507 -0.0534] curve/rate P90 abs [0.3374 0.2089]
residual rank correlation hold pooled [0.081 0.022] by lab [[0.267, 0.043], [-0.013, 0.237], [-0.069, -0.343], [0.08, 0.1]]
residual rank correlation gap difference pooled [-0.022 -0.125] by lab [[0.133, 0.099], [-0.141, -0.354], [-0.047, -0.044], [-0.045, -0.168]]
residual rank correlation damage pooled [-0.332  0.033] by lab [[-0.417, -0.296], [-0.078, 0.048], [-0.276, 0.176], [-0.517, 0.187]]
residual rank correlation activated memory pooled [-0.022  0.067] by lab [[-0.1, 0.026], [0.072, 0.181], [-0.054, -0.112], [-0.029, 0.208]]
residual rank correlation saturating memory pooled [-0.025  0.062] by lab [[-0.122, 0.004], [0.085, 0.186], [-0.058, -0.115], [-0.01, 0.227]]
max calibration delta in physical stress 0.0014063026885522412 in strain 0.0007870111033564934 in rate 0.0010631999397998282
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 11 — `submit_completed_task` (`17537 ms` LLM)
- **Rationale**: The verified artifact improves both worst-slice errors over equal weighting. Independent recurrence checks agreed within 1.4e-15, and the calibration discrepancy changed test predictions by at most 0.00079 in strain and 0.00107 in initial rate.
- **Alternatives Weighed**: `Add a saturating memory term` (rejected: Held-out residual correlations did not show a consistent within-lab relationship with that term, and it is outside the prescribed equations.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_04_solver.py': 'FILE:8301'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema passed, and both strain and initial-rate predictions matched the explicit recurrence in both gate states.`

### Turn 12 — `execute_in_remote_docker` (`24597 ms` LLM)
- **Rationale**: The previous audit found negative median prediction residuals for initial rate in all eight lab × regime slices. Compare those residuals with the full held-out time profile. A rate-only discrepancy would motivate testing a training-only readout correction; a coherent curve trend would instead point toward calibration, alignment, or the fitted physical coefficients.
- **Alternatives Weighed**: `Add a rate offset to the returned predictions` (rejected: That would break the required equality between initial rate and the derivative of the predicted strain curve.)

```bash
set -e
sed '/^xc, _/,$d' /tmp/rheology04_lolo.py > /tmp/rheology04_bias_audit.py
cat >> /tmp/rheology04_bias_audit.py <<'PY'
xa, _ = m._align(a,b)
inputs = m._inputs(a,xa,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
active = (np.maximum(abs(inputs[0])-crit,0)**2 @ dt) > 0
y = np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    tr = tuple(v[~val] for v in inputs)
    te = tuple(v[val] for v in inputs)
    lt,w = m._fit(tr,dt,t,crit,True,y[~val])
    fit = m._design(lt,tr,dt,t,crit,True) @ w
    pred = m._design(lt,te,dt,t,crit,True) @ w
    print('lab',lab,'taus',np.round(np.exp(lt),4),'train rate quartiles',np.round(np.quantile(y[~val,-1]-fit[:,-1],[.25,.5,.75]),4))
    for on in (False,True):
        use = active[val] == on
        e = y[val][use] - pred[use]
        print('  active',on,'signed medians by horizon then rate',np.round(np.median(e,axis=0),4),'rate sign fraction',round(np.mean(e[:,-1]>0),3))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_bias_audit.py
```
- **Container Exit Code**: `0` (`972 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 taus [0.501  2.3613] train rate quartiles [-0.0064  0.0672  0.1494]
  active False signed medians by horizon then rate [ 0.0118  0.0219  0.0187 -0.0149 -0.0203 -0.0307 -0.0724 -0.0946 -0.1234
 -0.1375  0.0487] rate sign fraction 0.615
  active True signed medians by horizon then rate [0.0228 0.044  0.0797 0.0998 0.1106 0.1512 0.1467 0.1538 0.1399 0.1914
 0.1042] rate sign fraction 0.886
lab 1 taus [0.5179 2.7691] train rate quartiles [-0.0003  0.0738  0.1729]
  active False signed medians by horizon then rate [ 0.0199  0.021   0.0179  0.0274  0.0281  0.0169 -0.008  -0.0393 -0.0122
 -0.0353  0.0813] rate sign fraction 0.714
  active True signed medians by horizon then rate [0.0144 0.0148 0.0669 0.0524 0.0655 0.0995 0.0837 0.0844 0.0963 0.0971
 0.0517] rate sign fraction 0.655
lab 2 taus [0.5169 2.8486] train rate quartiles [-0.0162  0.0786  0.1353]
  active False signed medians by horizon then rate [ 0.0225  0.0269  0.0179 -0.0019 -0.0175 -0.0481 -0.0787 -0.0954 -0.1332
 -0.1889  0.0672] rate sign fraction 0.833
  active True signed medians by horizon then rate [0.0055 0.0309 0.0542 0.0612 0.0759 0.0906 0.0994 0.0702 0.0704 0.1125
 0.0568] rate sign fraction 0.625
lab 3 taus [0.53   3.0138] train rate quartiles [-0.0087  0.0612  0.1163]
  active False signed medians by horizon then rate [ 0.0267  0.025   0.017  -0.0125 -0.0673 -0.1493 -0.1889 -0.2637 -0.3339
 -0.4003  0.1363] rate sign fraction 0.867
  active True signed medians by horizon then rate [0.0124 0.0188 0.0508 0.0121 0.0333 0.0553 0.0799 0.0976 0.0907 0.1682
 0.0534] rate sign fraction 0.625

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`33380 ms` LLM)
- **Rationale**: The signed residuals vary by damage regime and horizon, so a rate-only correction is not justified. Test whether the robust weighting itself drives the trade-off, estimating scales within each training fold and leaving the verified artifact unchanged unless a challenger improves the bottleneck.
- **Alternatives Weighed**: `Fit an independent prediction offset for initial rate` (rejected: It would break the derivative constraint, and the held-out bias also appears along the strain curve.)

```bash
set -e
sed '/^xa, _/,$d' /tmp/rheology04_bias_audit.py > /tmp/rheology04_weight_audit.py
cat >> /tmp/rheology04_weight_audit.py <<'PY'
xa, _ = m._align(a,b)
inputs = m._inputs(a,xa,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
active = (np.maximum(abs(inputs[0])-crit,0)**2 @ dt) > 0
y = np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
src = (root / 'alpha_m97_c03_batch_04_solver.py').read_text()
anchor = '        sqrt_weight = (1 / np.hypot(1, resid / scale))[valid]'
assert src.count(anchor)==1
# Both alternatives keep the same physical response and analytic rate.
variants = {'current': '', 'common_curve_scale': '        scale[:-1] = np.median(scale[:-1])\n', 'common_curve_scale_rate_balance': '        scale[:-1] = np.median(scale[:-1])\n'}
for name, change in variants.items():
    code = src.replace(anchor, change + anchor)
    if name.endswith('rate_balance'):
        code = code.replace(anchor, anchor + '\n        sqrt_weight *= np.where(np.broadcast_to(np.arange(y.shape[1]) == y.shape[1]-1, y.shape)[valid], np.sqrt(3), 1)')
    ns = {}; exec(compile(code, str(root / 'alpha_m97_c03_batch_04_solver.py'), 'exec'),ns)
    errs=[]; taus=[]
    for lab in np.unique(a['lab_id']):
        val = a['lab_id'] == lab
        tr, te = (tuple(v[mask] for v in inputs) for mask in (~val,val))
        lt,w = ns['_fit'](tr,dt,t,crit,True,y[~val]); taus.append(np.exp(lt))
        r = ns['_design'](lt,te,dt,t,crit,True) @ w - y[val]
        for on in (False,True):
            e=r[active[val]==on]; c=e[:,:-1]; q=e[:,-1]
            errs.append([np.sqrt(np.mean(c*c)),np.sqrt(np.mean(q*q)),np.quantile(abs(c),.9),np.quantile(abs(q),.9),abs(np.median(c)),abs(np.median(q))])
    np.set_printoptions(precision=5)
    print(name,'worst curve/rate pairs: RMSE, P90AE, abs median bias',np.max(errs,axis=0),'tau min/max',np.min(taus,axis=0),np.max(taus,axis=0))
    print('per lab x regime',np.asarray(errs)[:,2:])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_weight_audit.py
sed -n '79,118p' /holdout_task/alpha_m97_c03_batch_04_solver.py
```
- **Container Exit Code**: `0` (`2060 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
57343 0.20658 0.11427 0.11081] tau min/max [0.51643 2.52315] [0.53188 3.07855]
per lab x regime [[0.40504 0.12415 0.02725 0.03725]
 [0.37851 0.20658 0.07043 0.08816]
 [0.31033 0.10379 0.01436 0.07581]
 [0.27733 0.15557 0.03894 0.04132]
 [0.47204 0.15046 0.02174 0.05086]
 [0.24405 0.16648 0.04605 0.04527]
 [0.57343 0.14992 0.11427 0.11081]
 [0.29473 0.17739 0.03817 0.03733]]
def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    completed = np.where(np.isfinite(aligned), aligned, fills[ids])
    n_train = len(train['lab_id'])
    return completed[:n_train], completed[n_train:]


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times, crit, crystallizes):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`31645 ms` LLM)
- **Rationale**: The task specifies the active-row equation, but the linear fallback for zero damage was inferred. Its late strain bias has the same sign across all four held-out labs and is opposite to the active-row bias. Test that specific fallback in isolation without changing the required active equation, the analytic rate, or the verified artifact.
- **Alternatives Weighed**: `Promote either observation-weighting challenger` (rejected: Both improve initial rate but worsen worst-slice strain P90 by at least 0.0949, and neither explains the opposite late-horizon biases on active and inactive rows.)

```bash
set -e
sed '/^xa, _/,$d' /tmp/rheology04_weight_audit.py > /tmp/rheology04_inactive_audit.py
cat >> /tmp/rheology04_inactive_audit.py <<'PY'
xa, _ = m._align(a,b)
inputs = m._inputs(a,xa,m._calibrate(d['calibration_load_cells']))
dt,t,crit = (d[k] for k in ('past_dt','eval_times','crit_stress'))
active = (np.maximum(abs(inputs[0])-crit,0)**2 @ dt) > 0
y = np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
src = (root / 'alpha_m97_c03_batch_04_solver.py').read_text()
anchor = '    flow = np.where(act[:, None] > 0, np.log1p(times), times)'
assert src.count(anchor)==1
for name,expr in [('current',anchor),('log_base_in_low_banding','    flow = np.broadcast_to(np.log1p(times) if crystallizes else times, (len(hold), len(times)))')]:
    ns={}; exec(compile(src.replace(anchor,expr),'<audit>','exec'),ns)
    scores=[]
    for lab in np.unique(a['lab_id']):
        val=a['lab_id']==lab
        tr,te=(tuple(v[mask] for v in inputs) for mask in (~val,val))
        lt,w=ns['_fit'](tr,dt,t,crit,True,y[~val])
        err=ns['_design'](lt,te,dt,t,crit,True)@w-y[val]
        for on in (False,True):
            e=err[active[val]==on]; c=e[:,:-1]; q=e[:,-1]
            scores.append([np.sqrt(np.mean(c*c)),np.sqrt(np.mean(q*q)),np.quantile(abs(c),.9),np.quantile(abs(q),.9),np.median(c[:,-1]),np.median(q)])
    s=np.asarray(scores)
    print(name,'worst RMSE and P90 curve/rate',np.round(s[:,:4].max(axis=0),5),'late curve/rate medians by lab x gate',np.round(s[:,4:],4).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology04_inactive_audit.py
```
- **Container Exit Code**: `0` (`1373 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
current worst RMSE and P90 curve/rate [2.19725 0.15402 0.41266 0.24952] late curve/rate medians by lab x gate [[0.1375, -0.0487], [-0.1914, -0.1042], [0.0353, -0.0813], [-0.0971, -0.0517], [0.1889, -0.0672], [-0.1125, -0.0568], [0.4003, -0.1363], [-0.1682, -0.0534]]
log_base_in_low_banding worst RMSE and P90 curve/rate [2.19381 0.08032 0.28682 0.17998] late curve/rate medians by lab x gate [[-0.0084, 0.0023], [0.0009, 0.0006], [-0.0086, -0.0021], [0.0021, 0.0065], [0.0061, 0.0054], [0.0073, 0.0031], [0.0017, -0.0117], [-0.0048, 0.0091]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=86be9c33ac717357 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_86be9c33 (directory).
```

### Turn 15 — `execute_in_remote_docker` (`39732 ms` LLM)
- **Rationale**: Promote the change because it improves both prediction outputs while preserving the specified active-row equations. Recheck held-out errors on the saved solver and compare both outputs against an explicit recurrence at and above the banding threshold to catch implementation or boundary errors.
- **Alternatives Weighed**: `Keep linear drift at zero damage and increase rate weight` (rejected: Preceding weighting comparisons improved rate at the expense of strain P90 and did not resolve the systematic late-strain bias.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero past-work damage restores linear terminal drift in the low-banding campaign.` | Witness: `With linear drift, zero-damage late-strain median residuals were positive in all four held-out labs (0.0353 to 0.4003). Logarithmic base drift reduced their maximum absolute median to 0.0086 and cut worst initial-rate RMSE from 0.15402 to 0.08032.` | Blanket Diagnosis: `The inferred fallback used the damage gate to switch both the base drift law and the two activation corrections, producing opposite late-strain biases across damage strata.` | Sublated Pivot: `Use logarithmic base drift throughout the low-banding regime while retaining damage activation for the two correction terms and the analytic initial rate.`

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_04_solver.py')
s = p.read_text()
old = '    flow = np.where(act[:, None] > 0, np.log1p(times), times)'
new = '''    # Low banding sets the base drift law; damage gates the two corrections.
    flow = np.broadcast_to(np.log1p(times) if crystallizes else times,
                           (len(hold), len(times)))'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -I validate_public_schema.py
python3 -I /tmp/rheology04_compare.py /holdout_task/alpha_m97_c03_batch_04_solver.py
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
p = '/holdout_task/alpha_m97_c03_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c03_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
train, test = d['train'], d['test']
_, xb = m._align(train, test)
stress, hold, xd = m._inputs(test, xb, m._calibrate(d['calibration_load_cells']))
dt, t, crit = (d[k] for k in ('past_dt', 'eval_times', 'crit_stress'))
damage = (np.maximum(np.abs(stress) - crit, 0)**2 * dt).sum(axis=1)
assert np.any(damage == 0) and np.any(damage > 0)
for banding in (0.50, 0.51):
    out = m.solve_alpha_m97_c03_batch_04(dict(d, shear_banding_index=banding))
    assert all(np.isfinite(v).all() for v in out.values())
    tau, hw, rw = (out[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights'))
    z = np.zeros((len(hold), 2))
    for step, load in zip(dt, stress.T):
        e = np.exp(-step / tau)
        z = e*z + (1-e)*load[:, None]
    act = np.tanh(damage) if banding <= .50 else np.zeros_like(damage)
    flow = np.log1p(t) if banding <= .50 else t
    amp = (hold[:, None] - z)*hw[:2]
    amp[:, 0] += rw[0]*act*hold
    drift = xd @ hw[2:] + rw[1]*act*z[:, 0]
    expected_curve = test['strain_0'][:, None] + amp @ (1-np.exp(-t / tau[:, None])) + drift[:, None]*flow
    expected_rate = (amp / tau).sum(axis=1) + drift
    np.testing.assert_allclose(out['predicted_strain'], expected_curve, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out['initial_rate_pred'], expected_rate, atol=1e-12, rtol=1e-12)
    print('banding', banding, 'zero-damage rows', int((damage == 0).sum()), 'max strain/rate error', np.max(abs(out['predicted_strain']-expected_curve)), np.max(abs(out['initial_rate_pred']-expected_rate)))
PY
```
- **Container Exit Code**: `0` (`1815 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
ve_medianAE/p90AE [0.011583 0.053539] rate_medianAE/p90AE [0.019369 0.035914]
('lab active', 0) 35 curve_rmse 2.193815 rate_rmse 0.08032 curve_medianAE/p90AE [0.015268 0.286823] rate_medianAE/p90AE [0.022642 0.17998 ]
('lab', 1) 43 curve_rmse 0.39645 rate_rmse 0.043222 curve_medianAE/p90AE [0.015245 0.093211] rate_medianAE/p90AE [0.019179 0.044016]
('lab inactive', 1) 14 curve_rmse 0.237185 rate_rmse 0.028636 curve_medianAE/p90AE [0.013559 0.081992] rate_medianAE/p90AE [0.018905 0.040627]
('lab active', 1) 29 curve_rmse 0.453752 rate_rmse 0.048725 curve_medianAE/p90AE [0.016219 0.14504 ] rate_medianAE/p90AE [0.019373 0.045052]
('lab', 2) 42 curve_rmse 1.108052 rate_rmse 0.017752 curve_medianAE/p90AE [0.012877 0.037582] rate_medianAE/p90AE [0.010626 0.025055]
('lab inactive', 2) 18 curve_rmse 0.310401 rate_rmse 0.02146 curve_medianAE/p90AE [0.014008 0.053382] rate_medianAE/p90AE [0.014133 0.030823]
('lab active', 2) 24 curve_rmse 1.440955 rate_rmse 0.014357 curve_medianAE/p90AE [0.012414 0.031796] rate_medianAE/p90AE [0.008416 0.023353]
('lab', 3) 47 curve_rmse 0.706188 rate_rmse 0.055686 curve_medianAE/p90AE [0.013001 0.113853] rate_medianAE/p90AE [0.015352 0.055438]
('lab inactive', 3) 15 curve_rmse 0.921678 rate_rmse 0.018324 curve_medianAE/p90AE [0.011192 0.027725] rate_medianAE/p90AE [0.013048 0.027679]
('lab active', 3) 32 curve_rmse 0.57816 rate_rmse 0.066311 curve_medianAE/p90AE [0.014067 0.16772 ] rate_medianAE/p90AE [0.015992 0.088111]
missing 24 curve_rmse 0.780125 rate_rmse 0.135227 curve_medianAE/p90AE [0.099235 0.365351] rate_medianAE/p90AE [0.094967 0.223108]
complete 156 curve_rmse 1.229558 rate_rmse 0.020757 curve_medianAE/p90AE [0.011966 0.034876] rate_medianAE/p90AE [0.013834 0.035576]
time RMSE [0.71757032 0.22276534 0.13974777 0.44371148 0.17758013 2.97538735
 0.70818162 0.2011824  0.88037686 1.71212802]
estimated_taus [0.53682125 3.61474036]
hereditary_weights [ 0.6815807   0.54845505  0.13999326  0.16026966 -0.12973273  0.11960833
 -0.08973413]
regime_mode_weights [0.26099287 0.21798927]
banding 0.5 zero-damage rows 24 max strain/rate error 1.3322676295501878e-15 1.3322676295501878e-15
banding 0.51 zero-damage rows 24 max strain/rate error 8.881784197001252e-16 8.881784197001252e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=228a2a36d3ad02c6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_228a2a36 (directory).
```

### Turn 16 — `submit_completed_task` (`13873 ms` LLM)
- **Rationale**: The saved, verification-bound solver reduced the largest absolute zero-damage late-strain median bias from 0.4003 to 0.0086 across held-out labs. Isolated-process checks confirmed all five outputs were finite and both prediction equations matched the explicit recurrence within 1.4e-15.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema passed, and both prediction outputs matched the explicit recurrence at and above the banding threshold. Worst held-out lab × damage-regime P90 absolute error fell from 0.413 to 0.287 for strain and from 0.250 to 0.180 for initial rate.`

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
