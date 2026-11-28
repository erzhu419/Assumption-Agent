# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m100_02_batch_04__blind_continual_campaign__downstream_module`

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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `311,508` (`299,677` / `89,088` / `11,831`) |
| **Total Episode Wall Time (s)** | `375.89s` |

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

### Turn 1 — `execute_in_remote_docker` (`24369 ms` LLM)
- **Rationale**: Inspect the staged interface, public array shapes, and compact archive evidence first. A contract mismatch changes the downstream input mapping; evidence for a non-Robin boundary changes the assembled operator.
- **Alternatives Weighed**: `Implement the Robin baseline immediately` (rejected: Blind-campaign telemetry may identify a different boundary operator, and the staged calibrator defines the actual downstream input contract.)

```bash
cd /holdout_task && ls -l && cat batch_solver.py validate_public_schema.py _upstream_param_payload.json && cat _verified_upstream.py && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import json,pickle,numpy as np,pathlib
for p in sorted(pathlib.Path('/tmp/hegel_continual_archive').glob('*.json')):
 print('\nARCHIVE',p.name,p.stat().st_size)
 a=json.loads(p.read_text()); print(repr(a)[-6500:] if p.stat().st_size>6500 else repr(a))
b=pickle.load(open('public_batch.pkl','rb'));print('\nSIZE',len(b))
for item in b[:3]:
 print('ITEM',item.keys()); print('DIAG', None if item.get('campaign_diagnostics') is None else [(k,np.shape(v)) for d in item['campaign_diagnostics'][:1] for k,v in d.items()]);
 for k,v in item['study'].items():
  if isinstance(v,dict): print(k,[(kk,type(vv).__name__,np.shape(vv) if not isinstance(vv,dict) else list(vv)) for kk,vv in v.items()]); continue
  print(k,type(v).__name__,np.shape(v),repr(v)[:160])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT' 2>&1 | head -70
```
- **Container Exit Code**: `0` (`490 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
) array([ 0.35      ,  0.34500413,  0.33015913,  0.30588881,  0.27288602,
        0.23209293,  0.18467409,  0.13198319,  0.07552447,  0.01690968,
       -0.042187
vel_y ndarray (12,) array([-0.        , -0.05560533, -0.10899563, -0.1580441 , -0.20079691,
       -0.23555099, -0.26092193, -0.27589907, -0.27988581, -0.27272334,
       -0.254696
robin_x float () 0.8720099638043022
robin_y float () 1.1009862433777702
V_reaction_2d ndarray (14, 12) array([[0.25      , 0.25      , 0.25      , 0.25      , 0.25      ,
        0.25      , 0.25      , 0.25      , 0.25      , 0.25      ,
        0.25      , 0.25
mu_shifts ndarray (3,) array([0.15, 0.55, 1.1 ])
source_modes ndarray (8, 14, 12) array([[[1.18170043e-05, 1.96309904e-04, 2.06314045e-03, ...,
         7.90468152e-02, 2.13149885e-02, 3.63611487e-03],
        [2.61310773e-05, 4.34102344e-04,
sensor_ij ndarray (32, 2) array([[12,  6],
       [10,  8],
       [ 4, 10],
       [ 8,  6],
       [ 0, 10],
       [10,  4],
       [12,  2],
       [ 6,  2],
       [ 8,  8],
       
row_bank_id ndarray (96,) array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
       2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
       0, 1, 2
sensor_calibration_sweeps [('ref_rosette', 'ndarray', (80, 4)), ('raw_rosette', 'ndarray', (80, 4)), ('single_bank_sweeps', 'dict', ['bank_id', 'ref_signal', 'raw_readout'])]
Z_nuisance ndarray (96, 3) array([[ 1.66645269e+00, -1.73832986e-01, -8.91494031e-01],
       [-1.32868813e-01,  1.47255614e+00,  1.88817397e+00],
       [-8.24790297e-01, -3.94523632e-01
y_obs_raw ndarray (96,) array([-0.04017915,  0.54169685,  0.52740448,  0.08549289,  0.19785044,
        0.24614117,  0.64101093,  0.26582031, -0.04423433,  0.34461484,
        0.222273
aux_manifold_telemetry ndarray (24, 4) array([[-2.01533477, -1.60801624, -2.35110528, -0.33662084],
       [-1.13170468,  0.40300365,  1.01729488, -0.77116257],
       [-1.15360011,  0.47781182, -0.3
nuisance_ridge float () 0.08
source_ridge float () 0.04
u_box float () 0.42
total_budget float () 1.4
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`21268 ms` LLM)
- **Rationale**: Read the exact calibration return contract and identify archived boundary candidates before comparing their calibrated pulse residuals. A calibration-key mismatch changes the solver interface; a systematic pulse mismatch changes the operator grammar.
- **Alternatives Weighed**: `Reuse a prior campaign's boundary operator unchanged` (rejected: Campaign boundaries can change, and current tracer pulses can distinguish prior operators from the active one.)

```bash
cd /holdout_task; tail -65 _verified_upstream.py; ls -lh /tmp/hegel_continual_archive; for f in /tmp/hegel_continual_archive/*solver.py; do echo "--- $f"; grep -nE 'def |boundary|robin|wall|leak|periodic' "$f" | head -100; done; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'; a=json.load(open(p)); print('APPARATUS',str(a)[:8500]); print('ASSET INDEX',open('public_asset_index.json').read()[:4400]); b=pickle.load(open('public_batch.pkl','rb'))[0]; print('PULSE COUNT',len(b['campaign_diagnostics'])); print('PULSES',[(d['w_pulse'].tolist(),float(np.std(d['y_pulse_raw']))) for d in b['campaign_diagnostics']]); print('SENSORS',b['study']['sensor_ij'].tolist())
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 2>&1 | head -24
```
- **Container Exit Code**: `0` (`410 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"raw_readout": "ndarray",
                "ref_signal": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
                32,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                8,
                14,
                12
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 1.4,
            "u_box": 0.42,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                96
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
      "bytes": 1091
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
PULSE COUNT 3
PULSES [([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 0.1298527721491056), ([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 0.13567377683219867), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0], 0.12759584955141248)]
SENSORS [[12, 6], [10, 8], [4, 10], [8, 6], [0, 10], [10, 4], [12, 2], [6, 2], [8, 8], [0, 4], [0, 0], [6, 8], [2, 2], [10, 6], [12, 4], [4, 8], [8, 2], [6, 4], [6, 10], [10, 0], [2, 0], [2, 4], [10, 10], [0, 6], [0, 2], [4, 2], [2, 8], [0, 8], [2, 6], [4, 4], [8, 10], [4, 0]]
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
```

### Turn 3 — `execute_in_remote_docker` (`15774 ms` LLM)
- **Rationale**: Retrieve just the archive lines and telemetry omitted by truncation. If they encode additional boundary couplings, include those before using pulse residuals to decide whether a free fit is needed.
- **Alternatives Weighed**: `Fit unrestricted boundary rows now` (rejected: The previous response truncated the archived stencil definitions; those definitions can provide identifiable candidates for the three pulses.)

```bash
cd /holdout_task; ls -lh /tmp/hegel_continual_archive; for f in /tmp/hegel_continual_archive/*solver.py; do echo "--- $(basename "$f")"; grep -nEi 'def |boundary|robin|wall|leak|periodic|telemetry' "$f" | head -55; done; python3 - <<'PY'
import json
l=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'));print(json.dumps(l,separators=(',',':'))[:4100])
PY
 tail -15 _verified_upstream.py
```
- **Container Exit Code**: `0` (`331 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.43326,-0.80052],[-0.41231,-0.72745,-0.14737,-0.65404,2.58522,0.54469,0.22941,-0.6442],[-0.44642,-0.51924,0.48698,-0.80286,2.50352,-0.06249,0.00954,-0.28125],[-0.40481,-0.10363,0.82823,-0.82597,2.39728,-0.59987,-0.26112,0.17134],[-0.4654,0.38806,0.70091,-0.88346,2.38532,-0.85275,-0.40928,0.56039],[-0.37915,0.73297,0.21217,-0.77209,2.48946,-0.57508,-0.63743,0.76422],[-0.3966,0.85186,-0.50816,-0.7269,2.46434,0.05822,-0.73265,0.79912],[-0.33806,0.79118,-0.84534,-0.60247,2.52192,0.63516,-0.86881,0.63628],[-0.32406,0.41436,-0.70688,-0.40414,2.43572,0.8561,-0.87177,0.25677],[-0.36874,0.03816,-0.11983,-0.16475,2.50647,0.56926,-0.82349,-0.11594],[-0.53126,-0.3851,0.44244,0.06504,2.40444,-0.02512,-0.69401,-0.5621],[-0.37078,-0.71606,0.8462,0.29701,2.43719,-0.62084,-0.56816,-0.82871],[-0.45458,-0.85346,0.65102,0.49268,2.50403,-0.84583,-0.43798,-0.80622],[-0.45909,-0.7575,0.20484,0.67247,2.45718,-0.59422,-0.21455,-0.64848],[-0.51088,-0.49474,-0.48161,0.76085,2.37692,0.09042,-0.00098,-0.29921],[-0.51458,-0.05824,-0.80736,0.8769,2.51825,0.62144,0.24311,0.18839],[-0.58368,0.33635,-0.72499,0.84668,2.50616,0.90029,0.44273,0.52664],[-0.39159,0.73157,-0.15717,0.75687,2.5359,0.58135,0.58326,0.7853],[-0.45722,0.83443,0.45188,0.66119,2.56082,-0.00862,0.71272,0.84416],[-0.48863,0.75405,0.79468,0.56234,2.47846,-0.61713,0.8146,0.65316]]},{"completed_campaign_index":2,"completed_campaign_short_name":"dev_beta_m100_02_batch_01","facility_telemetry_trace":[[-0.33276,0.36355,-0.21619,-0.67043,-0.48689,-0.45833,0.76948,-0.64946],[-0.41364,0.67532,0.37049,-0.55595,-0.26086,0.25019,0.58381,-0.83683],[-0.39993,0.82875,0.79752,-0.38423,-0.3676,0.72473,0.33391,-0.83397],[-0.48166,0.77381,0.76086,-0.10885,-0.20957,0.8274,0.21565,-0.60482],[-0.42768,0.49195,0.28818,0.11749,-0.45349,0.47633,-0.02583,-0.23685],[-0.39583,0.00418,-0.36989,0.28181,-0.26245,-0.18693,-0.
    coef = _fit(ref, raw, bank)
    out = dict(study)
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            if pulse['y_pulse_raw'] is not None else dict(pulse)
            for pulse in pulses
        ]
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=np.asarray(sweeps['ref_rosette']).copy(),
        single_bank_sweeps=dict(bank, raw_readout=np.asarray(bank['ref_signal']).copy()),
    )
    return out
```

### Turn 4 — `execute_in_remote_docker` (`39418 ms` LLM)
- **Rationale**: Compare the discrete endpoint stencils first, then use a row-sum-aware fit to detect missing couplings. If those coefficients vary across held-out pulses, do not promote the unrestricted fit.
- **Alternatives Weighed**: `Select a boundary from telemetry alone` (rejected: Current calibrated tracer measurements can test the actual operator directly.)

```bash
cat > /tmp/chamber_probe.py <<'PY'
import sys,pickle,itertools,json
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,p):
    q=(k[:-1]+k[1:])/(2*h*h)
    lo=np.r_[-q[:-1]-np.maximum(v[1:-1],0)/h,0.]
    hi=np.r_[0.,-q[1:]-np.maximum(-v[1:-1],0)/h]
    d=np.r_[0.,-lo[:-1]-hi[1:],0.]
    e=[k[0]/h**2,k[-1]/h**2]
    a=diags([lo,d,hi],[-1,0,1],format='lil')
    for j,i,t,u in [(0,0,1,-1),(1,len(k)-1,len(k)-2,0)]:
        b,c,l=p[j*3:j*3+3]; a[i,i]=(b+c)*e[j]+l*r/h; a[i,t]=-b*e[j]; a[i,u]=-c*e[j]
    return a.tocsc()

def obs(s, p, modes):
    nx,ny=s['nx'],s['ny']; ix=eye(nx,format='csc'); iy=eye(ny,format='csc')
    ax,ay=[axis(s['kappa_'+z],s['vel_'+z],s['L'+z]/(s['n'+z]-1),s['robin_'+z],p[i*6:i*6+6]) for i,z in enumerate('xy')]
    a=kron(iy,ax,format='csc')+kron(ay,ix,format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; rhs=np.stack([m.ravel(order='F') for m in modes],axis=1)
    ident=eye(nx*ny,format='csc'); out=[]
    for shift in s['mu_shifts']:
        u=splu(a+shift*ident).solve(rhs); out.append(u[idx])
    return np.concatenate(out)

if __name__=='__main__':
    b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(b['study'],reference_tracer_pulses=b['campaign_diagnostics'])); pulses=s['reference_tracer_pulses']; y=np.array([d['y_pulse_raw'] for d in pulses]).T; modes=np.tensordot([d['w_pulse'] for d in pulses],s['source_modes'],axes=1)
    grammar={'R':(1,0,1),'N':(1,0,0),'P':(1,1,0),'H':(2,0,0),'B':(2,0,2)}
    results=[]
    rms=lambda a: np.sqrt(np.mean(a*a,axis=0))
    for ks in itertools.product(grammar,repeat=4):
        p=np.array([grammar[k] for k in ks]).ravel(); err=obs(s,p,modes)-y; results.append((np.linalg.norm(err),''.join(ks),p,rms(err)))
    for n,k,p,e in sorted(results)[:8]: print(k,e)
    for k in ['RRRR','NNNN','PPPP']: print('baseline',k,next(e for n,j,p,e in results if j==k))
    p0=sorted(results)[0][2]; np.set_printoptions(precision=7,suppress=True)
    for hold in [-1,0,1,2]:
        sel=np.arange(len(pulses))!=hold
        f=least_squares(lambda p:(obs(s,p,modes[sel])-y[:,sel]).ravel(),p0,bounds=(-0.15,8),max_nfev=70)
        print('fit',hold,'params',f.x.reshape(-1,3),'rmse',rms(obs(s,f.x,modes)-y),'singular values',np.linalg.svd(f.jac,compute_uv=False))
    for x in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
        a=np.array(x['facility_telemetry_trace']);print('telemetry',x['completed_campaign_short_name'],a.shape,np.round(np.median(a,axis=0),2),np.round(a[-4:].mean(axis=0),2))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/chamber_probe.py
```
- **Container Exit Code**: `0` (`2637 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.00218174 0.0029027  0.00036738]
PPHN [0.00914924 0.00569116 0.00733169]
PPPP [0.00712355 0.01346627 0.00194426]
PPNH [0.01436671 0.00663914 0.01045734]
PPNP [0.0109714  0.02198935 0.00220716]
PPHH [0.02002365 0.01217324 0.01621298]
PPPN [0.01453478 0.02628126 0.00345012]
PPHP [0.00702649 0.03384963 0.01382837]
baseline RRRR [0.11915141 0.11492462 0.11547469]
baseline NNNN [0.06131093 0.04735472 0.01080657]
baseline PPPP [0.00712355 0.01346627 0.00194426]
fit -1 params [[ 1.0368955  1.142183  -0.0000019]
 [ 1.0368917  1.083589   0.0000069]
 [ 0.999993   0.0000004 -0.0000006]
 [ 0.9999595  0.0000001 -0.0000016]] rmse [0.0000006 0.0000006 0.0000006] singular values [6.0271426 2.2135742 1.7119095 0.8103154 0.6870706 0.3343227 0.1628278
 0.1015295 0.061033  0.0543346 0.0278235 0.0246428]
fit 0 params [[ 1.0369818  1.1422766  0.0000075]
 [ 1.0367679  1.0834617 -0.000004 ]
 [ 1.0000073  0.0000019  0.       ]
 [ 0.9999637 -0.0000003 -0.0000012]] rmse [0.0000015 0.0000006 0.0000006] singular values [4.7482259 1.8109243 1.0478295 0.5746704 0.4603354 0.1821977 0.1293923
 0.0790447 0.0467128 0.0344597 0.0197948 0.0049073]
fit 1 params [[ 1.0368712  1.1421442 -0.0000011]
 [ 1.0369386  1.0836097  0.0000061]
 [ 0.9999877  0.0000014 -0.000001 ]
 [ 0.9999581 -0.0000014 -0.0000016]] rmse [0.0000006 0.0000007 0.0000006] singular values [5.2474017 1.7097981 0.7439674 0.6089494 0.32081   0.1791057 0.1375162
 0.0633487 0.0471002 0.0374468 0.0177454 0.0046835]
fit 2 params [[ 1.0369862  1.1422705  0.0000039]
 [ 1.0368138  1.0835162 -0.0000003]
 [ 0.9999964  0.0000003 -0.0000002]
 [ 0.999965   0.0000001 -0.0000011]] rmse [0.0000007 0.0000006 0.0000008] singular values [4.9279954 2.0203114 1.5356862 0.7977127 0.5666942 0.3175075 0.1334044
 0.0872025 0.0523192 0.0493249 0.0231649 0.0049141]
telemetry facility_init (24, 8) [-0.39  0.01 -0.01 -0.03 -0.32  0.03  0.01 -0.01] [-0.36  0.65 -0.45  0.28 -0.34  0.56 -0.83  0.7 ]
telemetry dev_beta_m100_02_s00 (24, 8) [-0.41 -0.01  0.02 -0.02  2.47  0.01  0.    0.03] [-0.48  0.66  0.09  0.71  2.52  0.21  0.64  0.7 ]
telemetry dev_beta_m100_02_batch_01 (24, 8) [-0.37 -0.04  0.02  0.04 -0.34  0.    0.02  0.01] [-0.34 -0.53 -0.43 -0.83 -0.39 -0.31  0.81  0.35]
telemetry dev_beta_m100_02_batch_02 (24, 8) [-0.39 -0.01  0.01  0.01  2.44 -0.02 -0.   -0.  ] [-0.46 -0.65  0.08 -0.21  2.39  0.05  0.72  0.14]
telemetry dev_beta_m100_02_batch_03 (24, 8) [ 2.55 -0.02  0.03  0.03 -0.34 -0.04  0.    0.03] [ 2.49  0.22  0.47  0.7  -0.34 -0.54 -0.11 -0.26]
```

### Turn 5 — `execute_in_remote_docker` (`30411 ms` LLM)
- **Rationale**: Fix the adjacent x-wall rates to face-averaged diffusion and refit only the two wrap rates inside each pulse fold. If held-out errors remain near calibration precision, the additional endpoint parameters are unnecessary.
- **Alternatives Weighed**: `Ship the unconstrained 12-coefficient endpoint fit` (rejected: The fitted leakage terms and y-axis deviations are near zero, and the x-axis adjacent coefficients appear to match the documented face averaging. Test the reduced stencil before retaining fitted coefficients.)
- **ATMS NOGOOD DECLARED**: Refuted `The chamber can be represented by Robin leakage or reflecting endpoint rows without cross-wall coupling.` | Witness: `Worst calibrated pulse RMSE was 0.1192 for all-Robin walls and 0.0613 for all-reflecting walls. Adding x-axis cross-wall terms reduced worst held-out pulse RMSE to about 1.5e-6.` | Blanket Diagnosis: `A missing x-endpoint coupling changes the transport response before nuisance projection or weight estimation.` | Sublated Pivot: `Keep the documented interior and reflecting y walls; test whether a shared face-diffusion law and a compact wrap correction explain both x endpoints.`

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/tmp')
from chamber_probe import obs,calibrate_upstream,least_squares
raw=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(raw['study'],reference_tracer_pulses=raw['campaign_diagnostics']))
pulses=s['reference_tracer_pulses']; y=np.array([d['y_pulse_raw'] for d in pulses]).T; modes=np.tensordot([d['w_pulse'] for d in pulses],s['source_modes'],axes=1)
k=s['kappa_x']; v=s['vel_x']; h=s['Lx']/(s['nx']-1); ends=k[[0,-1]]; near=k[[1,-2]]; b=(ends+near)/(2*ends); q=ends/h**2
print('end kappa',ends,'near',near,'vel',v[[0,1,-2,-1]],'h',h,'b',b,'wrap rates from previous fit',q*[1.142183,1.083589],'wrap additions over endpoint diffusion',q*([1.142183,1.083589]-np.ones(2)),'wrap diff in velocity units',h*np.diff(q*[1.142183,1.083589])); print('other scalars', [(key,val) for key,val in s.items() if np.isscalar(val)])
print('archive optimizer:')
for hld in [-1,0,1,2]:
 sel=np.arange(3)!=hld
 def params(c):return [b[0],c[0],0,b[1],c[1],0,1,0,0,1,0,0]
 fit=least_squares(lambda c:(obs(s,params(c),modes[sel])-y[:,sel]).ravel(),[1,1],gtol=3e-13)
 e=obs(s,params(fit.x),modes)-y
 print('hold',hld,'c',fit.x,'wrap velocity increment',((fit.x-1)*q*h),'max and rmse',abs(e).max(axis=0),np.sqrt(np.mean(e**2,axis=0)))
PY
sed -n '71,150p' /tmp/hegel_continual_archive/dev_beta_m100_02_batch_01_solver.py
```
- **Container Exit Code**: `0` (`852 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
:
            rhs = -c[free] - H[np.ix_(free, fixed)] @ w[fixed]
            reduced = H[np.ix_(free, free)]
            x, v = np.linalg.solve(reduced, np.column_stack((rhs, np.ones(len(free))))).T
            nu = (total - w[fixed].sum() - x.sum()) / v.sum()
            candidate = x + nu * v
            delta = candidate - w[free]
            crosses = np.where((candidate < 0) & (delta < 0), -1,
                               np.where((candidate > upper) & (delta > 0), 1, 0))
            bad = np.flatnonzero(crosses)
            if len(bad):
                ratios = (np.where(crosses[bad] > 0, upper, 0.) - w[free[bad]]) / delta[bad]
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


def solve_batch(study: dict) -> dict:
    S, U_last = _responses(study, _operator(study))
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((S, y))
    data -= Z @ np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = study['source_ridge']
    upper, total = study['u_box'], study['total_budget']
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
```

### Turn 6 — `execute_in_remote_docker` (`47184 ms` LLM)
- **Rationale**: The fitted x-wrap increments match the supplied velocities with the correct upwind signs. Implement that stencil, compose it with the staged one-argument calibrator, and check the fixed stencil against every public pulse. A pulse mismatch would reopen the boundary law; a schema or certificate failure would isolate the assembly or optimization layer.
- **Alternatives Weighed**: `Hardcode the fitted endpoint rates` (rejected: The inferred wrap increments in velocity units match the supplied endpoint velocities; the upwind stencil computes those rates from each study.)

```bash
set -eu
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Periodic-x, reflecting-y transport and certified source inversion."""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kap, vel, h, periodic):
    faces = (kap[:-1] + kap[1:]) / (2 * h**2)
    lower = -faces - np.maximum(vel[1:], 0) / h
    upper = -faces - np.maximum(-vel[:-1], 0) / h
    wrap_lower = wrap_upper = 0.
    if periodic:
        face = (kap[0] + kap[-1]) / (2 * h**2)
        wrap_lower = -face - max(vel[0], 0) / h
        wrap_upper = -face - max(-vel[-1], 0) / h
    else:
        upper[0] = -kap[0] / h**2
        lower[-1] = -kap[-1] / h**2
    main = -np.r_[wrap_lower, lower] - np.r_[upper, wrap_upper]
    mat = diags((lower, main, upper), (-1, 0, 1), format='lil')
    if periodic:
        mat[0, -1] += wrap_lower
        mat[-1, 0] += wrap_upper
    return mat.tocsc()


def _responses(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = (_axis(study['kappa_' + a], study['vel_' + a],
                    study['L' + a] / (study['n' + a] - 1), a == 'x')
              for a in ('x', 'y'))
    operator = (kron(eye(ny, format='csc'), ax, format='csc')
                + kron(ay, eye(nx, format='csc'), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    sensor_i, sensor_j = study['sensor_ij'].T
    rows = sensor_i + nx * sensor_j
    ident = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(rhs)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


PY
sed -n '/^def _box_simplex(/,/^def solve_batch(/{ /^def solve_batch(/!p; }' /tmp/hegel_continual_archive/dev_beta_m100_02_batch_01_solver.py >> downstream_solver.py
cat >> downstream_solver.py <<'PY'
def solve_downstream(cal_item: dict) -> dict:
    """Fit weights from observations already in canonical rosette coordinates."""
    S, U_last = _responses(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    data = np.column_stack((S, cal_item['y_obs_raw']))
    data -= Z @ np.linalg.solve(Z.T @ Z + cal_item['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ data)
    S, y = data[:, :-1], data[:, -1]
    ridge = cal_item['source_ridge']
    upper, total = cal_item['u_box'], cal_item['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(len(U_last)), -S.T @ y, upper, total)
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    dual = float(objective - g @ w + nu * total + upper * np.minimum(0., reduced).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., reduced).tolist(),
                                'lam_upper': np.maximum(0., -reduced).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
PY
cat > batch_solver.py <<'PY'
"""Compose the verified rosette calibration with the chamber solver."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle
sys.path.insert(0, '/holdout_task')
import numpy as np
from batch_solver import solve_batch, calibrate_upstream
from downstream_solver import _responses
with open('/holdout_task/public_batch.pkl','rb') as f:
    b = pickle.load(f)[0]
s = b['study']
cal = calibrate_upstream(dict(s, reference_tracer_pulses=b['campaign_diagnostics']))
S, U = _responses(cal)
for pulse in cal['reference_tracer_pulses']:
    rmse = np.sqrt(np.mean((S @ pulse['w_pulse'] - pulse['y_pulse_raw'])**2))
    print('Fixed physical stencil pulse RMSE:', rmse)
    assert rmse < 3e-6
out = solve_batch(s)
w = out['weights']; k = out['kkt_certificate']
gap = k['primal_objective'] - k['dual_lower_bound']
print('budget error:', abs(w.sum()-s['total_budget']), 'gap:', gap, 'field shape:', out['terminal_field'].shape, 'weights:', w)
assert abs(w.sum()-s['total_budget']) < 1e-12 and np.all((w >= 0) & (w <= s['u_box']))
assert abs(gap) < 1e-11 and out['terminal_field'].shape == (s['nx'],s['ny'])
PY
```
- **Container Exit Code**: `0` (`647 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Fixed physical stencil pulse RMSE: 7.370547967188695e-07
Fixed physical stencil pulse RMSE: 7.073384855316755e-07
Fixed physical stencil pulse RMSE: 7.181646602677928e-07
budget error: 0.0 gap: 6.74807432154978e-16 field shape: (14, 12) weights: [0.39082971 0.09175587 0.20990702 0.28629191 0.038622   0.08222581
 0.16826713 0.13210055]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=706b322805bba01b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_706b3228 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`45918 ms` LLM)
- **Rationale**: Run a read-only check on the exact candidate, independent of the schema validator. A pulse failure would reopen the boundary law; a payload mismatch would require restoring explicit payload forwarding; an active-bound failure would isolate the quadratic solver.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import inspect, json, pickle, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _axis, _box_simplex, _responses, solve_downstream
from _verified_upstream import calibrate_upstream
print('Calibration signature:', inspect.signature(calibrate_upstream))
with open('/holdout_task/_upstream_param_payload.json') as f:
    payload = json.load(f)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
worst_pulse = 0.
worst_gap = 0.
for item in items:
    s = item['study']
    marked = dict(s, reference_tracer_pulses=item.get('campaign_diagnostics'))
    cal = calibrate_upstream(marked, payload)
    S, U = _responses(cal)
    # Check calibration composition against the staged payload-aware interface.
    result = solve_batch(s)
    direct = solve_downstream(cal)
    np.testing.assert_allclose(result['weights'], direct['weights'], atol=1e-12, rtol=0)
    for pulse in cal.get('reference_tracer_pulses') or []:
        error = S @ pulse['w_pulse'] - pulse['y_pulse_raw']
        worst_pulse = max(worst_pulse, float(np.sqrt(np.mean(error**2))))
    w = result['weights']; cert = result['kkt_certificate']; Z = cal['Z_nuisance']
    P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = P @ S; d = A @ w - P @ cal['y_obs_raw']; g = A.T @ d + s['source_ridge'] * w
    lower = np.array(cert['lam_lower']); upper = np.array(cert['lam_upper'])
    np.testing.assert_array_equal(cert['weights'], w)
    np.testing.assert_allclose(g - cert['nu_sum'] - lower + upper, 0, atol=1e-11)
    np.testing.assert_allclose(lower*w + upper*(s['u_box']-w), 0, atol=1e-11)
    np.testing.assert_allclose(result['terminal_field'], np.einsum('i,ijk->jk', w, U), atol=1e-12)
    np.testing.assert_allclose(cert['primal_objective'], (d@d + s['source_ridge']*(w@w))/2, atol=1e-12)
    assert np.isclose(w.sum(), s['total_budget'], atol=1e-12, rtol=0) and np.all((w>=0)&(w<=s['u_box']))
    worst_gap = max(worst_gap, abs(cert['primal_objective'] - cert['dual_lower_bound']))
assert 0 < worst_pulse < 3e-6 and worst_gap < 1e-11
print('Public items:', len(items), 'worst calibrated pulse RMSE:', worst_pulse, 'worst certified gap:', worst_gap)

# Exercise both signs at both x endpoints with unequal endpoint diffusivities.
kap = np.array([.4, .7, .8, 1.1]); h = .3
for left in [-.9, .9]:
    for right in [-.8, .8]:
        v = np.array([left, -.3, .2, right])
        for periodic in [False, True]:
            expected = np.zeros((4,4))
            for i in range(4):
                if not periodic and i in [0,3]:
                    expected[i,1 if i == 0 else 2] = -kap[i]/h**2
                else:
                    for step in [-1,1]:
                        j = (i+step)%4
                        expected[i,j] = -(kap[i]+kap[j])/(2*h**2)-max(-step*v[i],0)/h
                expected[i,i] = -expected[i].sum()
            np.testing.assert_allclose(_axis(kap,v,h,periodic).toarray(),expected,atol=1e-13,rtol=0)

# Stress lower/upper-active, all-bound integer budgets and endpoint budgets.
rng = np.random.default_rng(421)
max_comp = max_objective_regret = 0.
for t in range(240):
    n = 8; u = .42
    B = rng.normal(size=(12,n)); H = B.T @ B / 12 + .04*np.eye(n)
    c = rng.normal(size=n)*[.2,1.,10.][t%3]
    total = u*[0,.0001,1,2.5,3,6.5,7.9999,8][t%8]
    w, nu = _box_simplex(H,c,u,total)
    d = H@w+c-nu
    comp = np.maximum(d,0)*w + np.maximum(-d,0)*(u-w)
    max_comp = max(max_comp, float(np.max(comp)))
    assert np.min(w) >= -1e-13 and np.max(w) <= u+1e-13 and abs(w.sum()-total) < 1e-11
    if t < 64 and 0 < total < n*u:
        f = lambda x: float(.5*x@H@x+c@x)
        ref = minimize(f, np.full(n,total/n), jac=lambda x:H@x+c, bounds=[(0,u)]*n, constraints=[{'type':'eq','fun':lambda x:x.sum()-total,'jac':lambda x:np.ones(n)}], method='SLSQP', options={'ftol':1e-12})
        if ref.success:
            max_objective_regret = max(max_objective_regret, f(w)-ref.fun)
assert max_comp < 1e-10 and max_objective_regret < 1e-9
print('240 QP slices: worst complementarity:', max_comp, 'worst SLSQP objective regret:', max_objective_regret)
PY
```
- **Container Exit Code**: `1` (`428 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Calibration signature: (study: 'dict') -> 'dict'

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=706b322805bba01b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_706b3228 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 18, in <module>
TypeError: calibrate_upstream() takes 1 positional argument but 2 were given
```

### Turn 8 — `execute_in_remote_docker` (`42055 ms` LLM)
- **Rationale**: Run the interrupted verification with the actual staged signature and leave both artifacts unchanged. The endpoint sign sweep tests stencil assembly, while active-bound cases test optimizer branches that the public solution does not exercise.
- **Alternatives Weighed**: `Change the production wrapper to pass the payload explicitly` (rejected: The staged calibrator accepts one argument, and the production wrapper already uses that signature; only the test passed a second argument.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _axis, _box_simplex, _responses, solve_downstream
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
worst_pulse = worst_gap = worst_stationarity = worst_comp = 0.
for item in items:
    s = item['study']
    cal = calibrate_upstream(dict(s, reference_tracer_pulses=item.get('campaign_diagnostics')))
    S, U = _responses(cal)
    result = solve_batch(s)
    direct = solve_downstream(cal)
    np.testing.assert_allclose(result['weights'], direct['weights'], atol=1e-12, rtol=0)
    for pulse in cal.get('reference_tracer_pulses') or []:
        error = S @ pulse['w_pulse'] - pulse['y_pulse_raw']
        worst_pulse = max(worst_pulse, float(np.sqrt(np.mean(error**2))))
    w = result['weights']; cert = result['kkt_certificate']; Z = cal['Z_nuisance']
    P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = P @ S; d = A @ w - P @ cal['y_obs_raw']; g = A.T @ d + s['source_ridge'] * w
    lower = np.array(cert['lam_lower']); upper = np.array(cert['lam_upper'])
    np.testing.assert_array_equal(cert['weights'], w)
    assert np.all(lower >= 0) and np.all(upper >= 0)
    worst_stationarity = max(worst_stationarity, np.max(np.abs(g - cert['nu_sum'] - lower + upper)))
    worst_comp = max(worst_comp, np.max(lower*w + upper*(s['u_box']-w)))
    np.testing.assert_allclose(result['terminal_field'], np.einsum('i,ijk->jk', w, U), atol=1e-12)
    np.testing.assert_allclose(cert['primal_objective'], (d@d + s['source_ridge']*(w@w))/2, atol=1e-12)
    assert abs(w.sum()-s['total_budget']) < 1e-12 and np.all((w>=0)&(w<=s['u_box']))
    worst_gap = max(worst_gap, abs(cert['primal_objective'] - cert['dual_lower_bound']))
assert 0 < worst_pulse < 3e-6 and max(worst_gap, worst_stationarity, worst_comp) < 1e-11
print('Public items:', len(items), 'worst pulse RMSE:', worst_pulse, 'worst KKT gap:', worst_gap, 'stationarity:', worst_stationarity, 'complementarity:', worst_comp)
kap = np.array([.4, .7, .8, 1.1]); h = .3
for left in [-.9, .9]:
    for right in [-.8, .8]:
        v = np.array([left, -.3, .2, right])
        for periodic in [False, True]:
            expected = np.zeros((4,4))
            for i in range(4):
                if not periodic and i in [0,3]:
                    expected[i,1 if i == 0 else 2] = -kap[i]/h**2
                else:
                    for step in [-1,1]:
                        j = (i+step)%4
                        expected[i,j] = -(kap[i]+kap[j])/(2*h**2)-max(-step*v[i],0)/h
                expected[i,i] = -expected[i].sum()
            np.testing.assert_allclose(_axis(kap,v,h,periodic).toarray(),expected,atol=1e-13,rtol=0)
rng = np.random.default_rng(421)
max_comp = max_regret = 0.
compared = 0
for t in range(240):
    n = 8; u = .42
    B = rng.normal(size=(12,n)); H = B.T @ B / 12 + .04*np.eye(n)
    c = rng.normal(size=n)*[.2,1.,10.][t%3]
    total = u*[0,.0001,1,2.5,3,6.5,7.9999,8][t%8]
    w, nu = _box_simplex(H,c,u,total)
    d = H@w+c-nu
    max_comp = max(max_comp, float(np.max(np.maximum(d,0)*w + np.maximum(-d,0)*(u-w))))
    assert np.min(w) >= -1e-13 and np.max(w) <= u+1e-13 and abs(w.sum()-total) < 1e-11
    if t < 64 and 0 < total < n*u:
        f = lambda x: float(.5*x@H@x+c@x)
        ref = minimize(f, np.full(n,total/n), jac=lambda x:H@x+c, bounds=[(0,u)]*n, constraints=[{'type':'eq','fun':lambda x:x.sum()-total,'jac':lambda x:np.ones(n)}], method='SLSQP', options={'ftol':1e-12})
        if ref.success:
            compared += 1
            max_regret = max(max_regret, f(w)-ref.fun)
assert compared >= 30 and max_comp < 1e-10 and max_regret < 1e-9
print('240 QP slices, independent objective comparisons:', compared, 'worst complementarity:', max_comp, 'worst objective regret:', max_regret)
PY
```
- **Container Exit Code**: `0` (`653 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public items: 1 worst pulse RMSE: 7.370547967188695e-07 worst KKT gap: 6.74807432154978e-16 stationarity: 1.6653345369377348e-16 complementarity: 1.3885022923444405e-16
240 QP slices, independent objective comparisons: 48 worst complementarity: 1.4921397450962101e-15 worst objective regret: 8.22844015146984e-11

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=706b322805bba01b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_706b3228 (directory).
```

### Turn 9 — `submit_completed_task` (`13855 ms` LLM)
- **Rationale**: The unchanged candidate passed the public schema check and isolated-process tests for tracer prediction, field layout, and KKT conditions, including active bounds and endpoint budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:264', '/holdout_task/downstream_solver.py': 'FILE:4974'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema check passed. The periodic-x, reflecting-y stencil achieved a worst tracer RMSE of 7.37e-7, and the public KKT gap was 6.75e-16. The optimizer passed 240 stress cases.`

### Turn 10 — `execute_in_remote_docker` (`37086 ms` LLM)
- **Rationale**: A wrap-only ablation isolates the missing boundary coupling without changing local transport. Per-shift residuals can expose a failure hidden by pooled RMSE; projected curvature and the KKT gap test the weight fit after nuisance removal. The audit runs in an isolated process and leaves both deliverables, the public fixture, and the validator unchanged.
- **Alternatives Weighed**: `Test unrelated classifier, relay, or drift models` (rejected: This task has continuous tracer readings and a steady linear transport operator, not cohort labels, relay states, or a time-dependent drift model.)

```bash
set -eu
cat > /tmp/chamber_margin_audit.py <<'PY'
import pickle
import sys

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _axis, _responses
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    raw = item['study']
    s = calibrate_upstream(dict(raw, reference_tracer_pulses=item.get('campaign_diagnostics')))
    pulses = s['reference_tracer_pulses']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    S, U = _responses(s)
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx']/(nx-1), s['Ly']/(ny-1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx, True)
    ay = _axis(s['kappa_y'], s['vel_y'], hy, False)
    ablated = ax.tolil()
    for i, j in ((0, nx-1), (nx-1, 0)):
        ablated[i, i] += float(ax[i, j])
        ablated[i, j] = 0.
    I = eye(nx*ny, format='csc')
    rest = kron(ay, eye(nx, format='csc'), format='csc') + diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
    L0 = kron(eye(ny, format='csc'), ablated.tocsc(), format='csc') + rest
    ix, iy = s['sensor_ij'].T
    rows = ix + nx*iy
    rhs = np.column_stack([a.ravel(order='F') for a in s['source_modes']]) @ W
    no_wrap = np.vstack([splu(L0 + mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
    err = (S @ W - Y).reshape(len(s['mu_shifts']), len(rows), -1)
    ablated_err = (no_wrap - Y).reshape(err.shape)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    no_wrap_rmse = np.sqrt(np.mean(ablated_err**2, axis=1))
    print('Periodic-x RMSE by shift x pulse:', rmse.tolist())
    print('Wrap-only ablation RMSE by shift x pulse:', no_wrap_rmse.tolist())
    print('Largest periodic-x point residual:', float(np.max(np.abs(err))))
    assert rmse.max() < 3e-6 and np.max(np.abs(err)) < 5e-6
    assert np.all(no_wrap_rmse > 10*rmse)
    out = solve_batch(raw)
    w = out['weights']
    cert = out['kkt_certificate']
    Z = s['Z_nuisance']
    P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    Q = P @ S
    tangent = np.eye(len(w))[:, :-1] - np.eye(len(w))[:, [-1]]
    hv = np.linalg.eigvalsh(tangent.T @ (Q.T @ Q + s['source_ridge']*np.eye(len(w))) @ tangent)
    gap = cert['primal_objective'] - cert['dual_lower_bound']
    estimate = np.sqrt(2*max(gap, 0.)/s['source_ridge'])
    print('Nuisance-projected budget tangent curvature min/max:', float(hv[0]), float(hv[-1]))
    print('KKT gap and gap-implied weight error bound (up to floating-point precision):', gap, estimate)
    assert hv[0] > 0 and abs(gap) < 1e-11
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/chamber_margin_audit.py
```
- **Container Exit Code**: `0` (`410 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Periodic-x RMSE by shift x pulse: [[9.093053241618872e-07, 8.393264639767022e-07, 8.709763316962091e-07], [6.641457790646486e-07, 6.506263626466778e-07, 6.522572208810787e-07], [6.015176902638344e-07, 6.10900664953192e-07, 6.02695765047482e-07]]
Wrap-only ablation RMSE by shift x pulse: [[0.08529190500039054, 0.06489916668961447, 0.015569439373271151], [0.05422356726541168, 0.04147431430841198, 0.010005667112545509], [0.03561408699412582, 0.026876959514064463, 0.006489477191633733]]
Largest periodic-x point residual: 1.652418137043643e-06
Nuisance-projected budget tangent curvature min/max: 0.040148715652285054 1.3737695814932707
KKT gap and gap-implied weight error bound (up to floating-point precision): 6.74807432154978e-16 1.8368552367497254e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=706b322805bba01b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_706b3228 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`15247 ms` LLM)
- **Rationale**: Separate audit execution from script creation. Run the staged validator and the existing ablation in a turn that writes no files, so the host can assess the boundary evidence against the unchanged candidate. If either command fails, use the failing check to distinguish a contract problem from a stencil problem.
- **Alternatives Weighed**: `Change the solver in response to the probe rejection` (rejected: The rejection concerns the verification record, not a measured failure of the solver. The existing audit passed every pulse-by-shift slice.)

```bash
set -eu
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -B validate_public_schema.py /holdout_task/batch_solver.py
python3 -I -B /tmp/chamber_margin_audit.py
```
- **Container Exit Code**: `0` (`575 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Periodic-x RMSE by shift x pulse: [[9.093053241618872e-07, 8.393264639767022e-07, 8.709763316962091e-07], [6.641457790646486e-07, 6.506263626466778e-07, 6.522572208810787e-07], [6.015176902638344e-07, 6.10900664953192e-07, 6.02695765047482e-07]]
Wrap-only ablation RMSE by shift x pulse: [[0.08529190500039054, 0.06489916668961447, 0.015569439373271151], [0.05422356726541168, 0.04147431430841198, 0.010005667112545509], [0.03561408699412582, 0.026876959514064463, 0.006489477191633733]]
Largest periodic-x point residual: 1.652418137043643e-06
Nuisance-projected budget tangent curvature min/max: 0.040148715652285054 1.3737695814932707
KKT gap and gap-implied weight error bound (up to floating-point precision): 6.74807432154978e-16 1.8368552367497254e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=706b322805bba01b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_706b3228 (directory).
```

### Turn 12 — `submit_completed_task` (`14567 ms` LLM)
- **Rationale**: The saved candidate passed the public schema check, isolated-process transport and KKT checks, and active-bound stress tests. Removing only the x-wrap terms worsened every measured pulse-by-shift slice, supporting the implemented boundary law without further fitting.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema check passed. The periodic-x, reflecting-y solver achieved a worst pulse-by-shift RMSE of 9.09e-7 and a public KKT gap of 6.75e-16. The optimizer passed 240 stress cases.`

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
