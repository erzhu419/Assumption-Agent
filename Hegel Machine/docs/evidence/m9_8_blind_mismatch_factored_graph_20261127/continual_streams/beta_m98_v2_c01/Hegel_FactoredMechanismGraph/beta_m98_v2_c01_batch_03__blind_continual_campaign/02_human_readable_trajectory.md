# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c01_batch_03__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `379,431` (`367,296` / `134,400` / `12,135`) |
| **Total Episode Wall Time (s)** | `420.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`13781 ms` LLM)
- **Rationale**: The manifest, log, and public shapes can distinguish an incorrect boundary-regime reuse from a calibration schema mismatch before implementation.

```bash
ls -l /tmp/hegel_continual_archive /holdout_task; cat /holdout_task/validate_public_schema.py /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/archive_apparatus_log.json; python3 - <<'PY'
import pickle
x=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(x))
for i in x:
 print('diagnostics',type(i.get('campaign_diagnostics')).__name__)
 for k,v in i['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else v if isinstance(v,(str,int,float)) else '')
PY
```
- **Container Exit Code**: `0` (`417 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.28515,
        0.12909,
        1.09713,
        2.71799,
        0.09754,
        1.08564
      ],
      [
        -0.23314,
        1.21233,
        -0.27968,
        -0.05852,
        1.21998,
        2.55496,
        -0.36732,
        1.22386
      ],
      [
        -0.45857,
        0.99083,
        -0.43672,
        -0.70679,
        1.07775,
        2.59524,
        -0.65563,
        1.38948
      ],
      [
        -0.65489,
        1.06516,
        -0.32548,
        -0.87708,
        1.12483,
        2.58568,
        -0.85432,
        1.22486
      ],
      [
        -1.00637,
        0.87384,
        -0.45553,
        -0.90234,
        1.20669,
        2.55501,
        -0.8766,
        1.34879
      ],
      [
        -0.92426,
        1.14329,
        -0.44234,
        -0.81439,
        1.28398,
        2.55889,
        -0.68179,
        1.11613
      ],
      [
        -0.47014,
        1.17571,
        -0.35374,
        -0.66804,
        1.13916,
        2.59557,
        -0.66059,
        1.16252
      ],
      [
        -0.29631,
        1.17067,
        -0.24734,
        -0.27254,
        1.07232,
        2.56605,
        -0.01926,
        1.11328
      ],
      [
        0.02234,
        1.18129,
        -0.42924,
        0.08537,
        1.1392,
        2.41132,
        0.13291,
        0.87864
      ],
      [
        0.59452,
        0.9318,
        -0.40543,
        0.44961,
        1.01302,
        2.72977,
        0.46337,
        0.95858
      ],
      [
        0.76037,
        1.13478,
        -0.30704,
        0.67493,
        1.1064,
        2.51805,
        0.90756,
        1.21356
      ],
      [
        0.69133,
        1.25522,
        -0.30106,
        0.85114,
        1.2213,
        2.54548,
        1.13399,
        1.32773
      ]
    ]
  }
]items 1
diagnostics list
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 0.8942552148014796
robin_y float None 1.163864943166582
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) 
source_modes ndarray (8, 14, 12) 
sensor_ij ndarray (32, 2) 
row_bank_id ndarray (96,) 
sensor_calibration_sweeps dict None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3) 
y_obs_raw ndarray (96,) 
aux_manifold_telemetry ndarray (24, 4) 
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
```

### Turn 2 — `execute_in_remote_docker` (`20161 ms` LLM)
- **Rationale**: The archive comparison identifies which mechanisms can be reused. Pulse shapes determine how to score boundary candidates before changing the forward model.
- **Alternatives Weighed**: `Copy the latest archived solver unchanged` (rejected: The pre-gate flagged a downstream constitutive mismatch, and public tracer pulses can directly test whether the archived boundary still applies.)

```bash
cat /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py; diff -u /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_01_solver.py; diff -u /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py /tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box equality constrained quadratic'; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; print('pulse count',len(x['campaign_diagnostics']));
for p in x['campaign_diagnostics']:
 print({k:(np.shape(v), np.round(v,5).tolist() if np.size(v)<=10 else [float(np.min(v)),float(np.max(v))]) for k,v in p.items()})
for k,v in x['study']['sensor_calibration_sweeps'].items():
 print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else '')
PY
```
- **Container Exit Code**: `0` (`374 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ne=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
pulse count 2
{'w_pulse': ((8,), [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]), 'y_pulse_raw': ((96,), [-0.025926340516802066, 0.5400298633602025])}
{'w_pulse': ((8,), [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]), 'y_pulse_raw': ((96,), [-0.03478404930391608, 0.5137180130826561])}
ref_rosette ndarray (80, 4) 
raw_rosette ndarray (80, 4) 
single_bank_sweeps dict None ['bank_id', 'ref_signal', 'raw_readout']
```

### Turn 3 — `execute_in_remote_docker` (`22039 ms` LLM)
- **Rationale**: Sweep residuals can detect sensor misspecification; pulse error by shift and before and after nuisance removal shows whether the wall model needs to change.

```bash
for f in /tmp/hegel_continual_archive/*solver.py; do echo ====; head -34 "$f"; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -35; cat /holdout_task/public_asset_index.json; cat > /tmp/p98.py <<'PY'
import pickle,numpy as np,glob,importlib.util
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; sw=st['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; X=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(X,y,rcond=None)[0]; print('affine',B,'sweep rms',np.mean((X@B-y)**2,axis=0)**.5); print('banks',st['row_bank_id'][:24]); print('single',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in sw['single_bank_sweeps'].items()]); print('aux',st['aux_manifold_telemetry'][:4]);
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 sp=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); S,U=m._forward(st); print(f)
 for p in item['campaign_diagnostics']:
  r=m._calibrate(dict(st,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse']; Z=st['Z_nuisance']; q=r-Z@np.linalg.lstsq(Z,r,rcond=None)[0]; print('rms',np.linalg.norm(r)/len(r)**.5,np.linalg.norm(q)/len(r)**.5,'by shift',np.mean(r.reshape(3,-1)**2,axis=1)**.5)
PY
python3 /tmp/p98.py
```
- **Container Exit Code**: `0` (`476 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array"
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
      "bytes": 1970
    }
  },
  "schema": "hegel_public_asset_index_v1"
}affine [[ 7.99998267e-02 -7.00019918e-02  6.00011898e-02 -5.00009491e-02]
 [ 1.29000060e+00 -2.44595072e-07 -2.68503376e-07  2.71816701e-06]
 [-2.12352592e-07  8.00002771e-01 -3.32064249e-07  1.78836886e-06]
 [-8.56350385e-07  2.29021367e-07  1.24999780e+00 -1.08135761e-06]
 [ 8.65396371e-07 -1.91178964e-07  1.09392060e-07  8.30001139e-01]] sweep rms [9.80879566e-06 1.05245360e-05 8.98263119e-06 7.90435584e-06]
banks [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
single [('bank_id', (24,), array([0, 0, 0, 0, 0, 0, 1, 1])), ('ref_signal', (24,), array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6])), ('raw_readout', (24,), array([-1.3389874 , -0.69400756, -0.04899154,  0.59600999,  1.24098865,
        1.8859931 , -0.94999503, -0.54998207]))]
aux [[-0.44048338 -1.10945876 -0.88555499 -0.66280842]
 [-0.52877406  0.78730224  0.10524895 -0.97804126]
 [ 0.05482929  0.10561124 -1.44001917  0.45796639]
 [-0.6858121   0.65359442  0.21944881  0.821292  ]]
/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_01_solver.py
rms 0.016089379439839124 0.015855986959091767 by shift [0.02149855 0.01431681 0.01046164]
rms 0.016769063348767415 0.016297232855133056 by shift [0.0241959  0.01368692 0.00841613]
/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py
rms 0.11817780011815184 0.11633685304742658 by shift [0.18727608 0.07519757 0.03421947]
rms 0.10749028946567739 0.10592520447747933 by shift [0.17186628 0.06585941 0.02805367]
/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py
rms 0.11817780011815184 0.11633685304742658 by shift [0.18727608 0.07519757 0.03421947]
rms 0.10749028946567739 0.10592520447747933 by shift [0.17186628 0.06585941 0.02805367]
```

### Turn 4 — `execute_in_remote_docker` (`36391 ms` LLM)
- **Rationale**: A held-out pulse distinguishes reusable wall corrections from parameters that merely absorb one tracer response. The simpler tied models test whether a discrete symmetry explains the mismatch before fitting separate wall coefficients.
- **Alternatives Weighed**: `Adopt the archived reflecting stencil unchanged` (rejected: It improves substantially on Robin but leaves a repeatable discrepancy across both tracer pulses.)
- **ATMS NOGOOD DECLARED**: Refuted `The two archived boundary stencils suffice for the current chamber.` | Witness: `Even the closer reflecting stencil leaves nuisance-projected pulse RMSE of 0.01586 and 0.01630, while the calibration sweeps fit to roughly 1e-5 raw RMSE.` | Blanket Diagnosis: `The mismatch survives nuisance projection and is largest at the first shift, so the next test should isolate wall leakage and inward coupling while holding sensor calibration and interior transport fixed.` | Sublated Pivot: `Fit wall-local corrections on one pulse at a time and score the other pulse before choosing a discrete wall stencil.`

```bash
cat > /tmp/wall98.py <<'PY'
import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; sw=st['sensor_calibration_sweeps']; ref=sw['ref_rosette']; raw=sw['raw_rosette']; B=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]
Y=np.stack([np.linalg.solve(B[1:].T,(p['y_pulse_raw'].reshape(-1,4)-B[0]).T).T.ravel() for p in item['campaign_diagnostics']],axis=-1)
W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1); Z=st['Z_nuisance']; nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; T=ij[:,0]+nx*ij[:,1]; V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); rhs=np.column_stack([u.ravel(order='F') for u in st['source_modes']])@W
def axis(d,loss,mult):
 k=st['kappa_'+d]; u=st['vel_'+d]; h=st['L'+d]/(len(k)-1); f=(k[:-1]+k[1:])*.5/h**2; l=f[:-1]+np.maximum(u[1:-1],0)/h; r=f[1:]+np.maximum(-u[1:-1],0)/h; e=k[[0,-1]]/h**2*mult; b=e+loss*st['robin_'+d]/h
 return diags((-np.r_[l,e[1]],np.r_[b[0],l+r,b[1]],-np.r_[e[0],r]),(-1,0,1),format='csc')
def predict(q):
 ax=axis('x',q[:2],q[4:6]); ay=axis('y',q[2:4],q[6:]); L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[T] for mu in st['mu_shifts']])
one=np.ones(4); zero=np.zeros(4)
models={'common robin':(lambda a:np.r_[np.repeat(a[0],4),one],[.1]),'common edge':(lambda a:np.r_[zero,np.repeat(a[0],4)],[1.]),'axis robin':(lambda a:np.r_[np.repeat(a,2),one],[.1,.1]),'wall robin':(lambda a:np.r_[a,one],[.1]*4),'axis edge':(lambda a:np.r_[zero,np.repeat(a,2)],[1.,1.]),'wall edge':(lambda a:np.r_[zero,a],[1.]*4),'joint':(lambda a:a,[.1]*4+[1.]*4)}
for name,(pack,a0) in models.items():
 for mask in [None,0,1]:
  def f(a):
   r=predict(pack(a))-Y
   return (r if mask is None else r[:,[mask]]).ravel()
  opt=least_squares(f,a0,bounds=(0,5),gtol=1e-14,ftol=1e-12,xtol=1e-12); q=pack(opt.x); r=predict(q)-Y
  print(name,mask,np.round(q,6), 'pulse rms',np.sqrt(np.mean(r*r,axis=0)), 'worst shift',np.max(np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1))))
r=predict(np.r_[zero,one])-Y;print('reflect residual signed',r.min(axis=0),r.max(axis=0),'means by shift',r.reshape(3,32,2).mean(axis=1)); print('axes inputs',[(d,st['robin_'+d],st['kappa_'+d][[0,-1]],st['vel_'+d][[0,-1]]) for d in 'xy'])
PY
python3 /tmp/wall98.py
```
- **Container Exit Code**: `0` (`2578 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
962 1.       1.       1.       1.      ] pulse rms [0.01506624 0.01456788] worst shift 0.020071493365696188
axis robin 1 [0.037137 0.037137 0.       0.       1.       1.       1.       1.      ] pulse rms [0.01621279 0.01316355] worst shift 0.02176805768338507
wall robin None [0.032789 0.       0.       0.012343 1.       1.       1.       1.      ] pulse rms [0.01475701 0.01135505] worst shift 0.019165722593882606
wall robin 0 [0.       0.       0.       0.027031 1.       1.       1.       1.      ] pulse rms [0.01402448 0.01347859] worst shift 0.018150257982203908
wall robin 1 [0.064839 0.       0.       0.       1.       1.       1.       1.      ] pulse rms [0.01621858 0.01048905] worst shift 0.022110858520879267
axis edge None [0.       0.       0.       0.       0.667464 0.667464 1.124742 1.124742] pulse rms [0.01534411 0.01286336] worst shift 0.01955105263919543
axis edge 0 [0.       0.       0.       0.       0.838475 0.838475 0.958506 0.958506] pulse rms [0.0151182  0.01375305] worst shift 0.019403793188467796
axis edge 1 [0.       0.       0.       0.       0.575787 0.575787 1.285424 1.285424] pulse rms [0.01575294 0.01267792] worst shift 0.020053105463553274
wall edge None [0.       0.       0.       0.       0.586084 2.840776 1.992028 0.599712] pulse rms [0.01191958 0.00496669] worst shift 0.01525658407911925
wall edge 0 [0.       0.       0.       0.       0.740341 1.938729 2.25978  0.512073] pulse rms [0.01119198 0.00865574] worst shift 0.013779138622014533
wall edge 1 [0.       0.       0.       0.       0.482222 5.       1.696393 0.767126] pulse rms [0.01435348 0.002034  ] worst shift 0.01906012390730065
joint None [0.       0.       0.       0.       0.586084 2.840772 1.992029 0.599712] pulse rms [0.01191957 0.00496669] worst shift 0.015256582722731126
joint 0 [0.       0.       0.       0.       0.740341 1.938729 2.25978  0.512073] pulse rms [0.01119198 0.00865574] worst shift 0.013779138717801884
joint 1 [7.702000e-03 0.000000e+00 0.000000e+00 5.710000e-04 5.203510e-01
 5.000000e+00 1.707224e+00 7.811530e-01] pulse rms [0.01443901 0.00192325] worst shift 0.01924964535009682
reflect residual signed [-0.0304285  -0.01880135] [0.05444391 0.04854928] means by shift [[ 0.00810884  0.01353416]
 [ 0.00122971  0.0049531 ]
 [-0.00028258  0.00222395]]
axes inputs [('x', 0.8942552148014796, array([0.4503445, 0.4503445]), array([ 0.35      , -0.20572484])), ('y', 1.163864943166582, array([0.57243569, 0.33243569]), array([-0.        , -0.22652476]))]
```

### Turn 5 — `execute_in_remote_docker` (`30602 ms` LLM)
- **Rationale**: Local leakage and inward-coupling fits do not explain both tracers. Testing discrete seam terms before flexible fitting can identify a missing boundary connection and show whether it transfers across pulses.
- **Alternatives Weighed**: `Use the fitted wall-local multipliers` (rejected: The second-pulse fit drives an inward multiplier to its cap and leaves 0.01435 RMSE on the first pulse.)

```bash
sed '/one=np.ones(4)/,$d' /tmp/wall98.py > /tmp/wrap98.py
cat >> /tmp/wrap98.py <<'PY'
from itertools import product
from scipy.sparse import csc_matrix
def wrap_axis(d,loss,inward,wrap,averaged=False,adv=False):
 k=st['kappa_'+d]; v=st['vel_'+d]; h=st['L'+d]/(len(k)-1); e=k[[0,-1]]/h**2
 A=axis(d,loss,inward).toarray(); ew=np.repeat(e.mean(),2) if averaged else e
 for side,i,j,near in [(0,0,len(k)-1,1),(1,len(k)-1,0,len(k)-2)]:
  z=wrap[side]*ew[side]; A[i,j]-=z; A[i,i]+=z
  if adv:
   z=abs(v[i])/h; col=j if ((side==0 and v[i]>0) or (side==1 and v[i]<0)) else near
   A[i,col]-=z; A[i,i]+=z
 return csc_matrix(A)
def pred(q,avg=False,adv=False):
 ax=wrap_axis('x',*q[:,:2],avg,adv); ay=wrap_axis('y',*q[:,2:],avg,adv)
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[T] for mu in st['mu_shifts']])
def metrics(r):
 return np.sqrt(np.mean(r*r)),np.max(np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)))
stencils={'reflect':[0,1,0],'double-reflect':[0,2,0],'robin':[1,1,0],'periodic':[0,1,1],'seam-only':[0,0,1],'half-periodic':[0,.5,.5]}
res=[]
for names in product(stencils,repeat=4):
 q=np.array([stencils[n] for n in names],dtype=float).T
 for avg,adv in product([False,True],repeat=2):
  r=pred(q,avg,adv)-Y; res.append((*metrics(r),names,avg,adv))
print('Best discrete stencils'); print(*sorted(res)[:14],sep='\n')
for signed in [False,True]:
 for train in [[0,1],[0],[1]]:
  def fun(q): return (pred(q.reshape(3,4))-Y)[:,train].ravel()
  sol=least_squares(fun,np.array([[.005]*4,[1.]*4,[.1]*4]).ravel(),bounds=(np.repeat([-1. if signed else 0.,0.,0.],4),5),gtol=1e-13,ftol=1e-12,xtol=1e-12)
  r=pred(sol.x.reshape(3,4))-Y; print('wrap-fit',signed,train,np.round(sol.x.reshape(3,4),5),np.sqrt(np.mean(r*r,axis=0)),metrics(r))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wrap98.py
```
- **Container Exit Code**: `0` (`18301 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lect', 'periodic', 'periodic'), True, True)
(0.00592160081138186, 0.011132320094479644, ('reflect', 'double-reflect', 'half-periodic', 'periodic'), False, True)
(0.006083270841242551, 0.009098321140532455, ('reflect', 'reflect', 'periodic', 'periodic'), False, True)
(0.006130940855249929, 0.010784496756692968, ('periodic', 'periodic', 'double-reflect', 'periodic'), False, False)
(0.0063009106526833395, 0.01191438857603852, ('periodic', 'periodic', 'double-reflect', 'periodic'), False, True)
(0.006588469941625816, 0.012219207240940975, ('periodic', 'periodic', 'double-reflect', 'periodic'), True, False)
(0.006602153616638171, 0.01178976985320533, ('periodic', 'periodic', 'half-periodic', 'periodic'), False, False)
(0.0066759870170462, 0.012173904965310978, ('half-periodic', 'periodic', 'periodic', 'periodic'), False, False)
(0.006771022478113354, 0.013207088107069118, ('periodic', 'periodic', 'double-reflect', 'periodic'), True, True)
wrap-fit False [0, 1] [[0.00000e+00 0.00000e+00 0.00000e+00 1.00000e-05]
 [1.03742e+00 1.03703e+00 9.95870e-01 1.00723e+00]
 [1.14400e+00 1.08443e+00 7.90410e-01 1.48457e+00]] [8.98231671e-07 9.65396014e-07] (9.324187894693999e-07, 1.08712736888225e-06)
wrap-fit False [0] [[0.00000e+00 0.00000e+00 0.00000e+00 1.00000e-05]
 [1.03732e+00 1.03722e+00 9.96080e-01 1.00682e+00]
 [1.14381e+00 1.08428e+00 7.90460e-01 1.48363e+00]] [8.33353591e-07 4.14243417e-06] (2.9878285660480468e-06, 6.2978092125823996e-06)
wrap-fit False [1] [[0.00000e+00 0.00000e+00 0.00000e+00 1.00000e-05]
 [1.03767e+00 1.03939e+00 9.96350e-01 1.00655e+00]
 [1.14456e+00 1.08435e+00 7.89860e-01 1.48149e+00]] [2.83966451e-05 8.93258727e-07] (2.0089392268021963e-05, 4.36608828351918e-05)
wrap-fit True [0, 1] [[ 2.00000e-05 -2.00000e-05 -1.00000e-05  2.00000e-05]
 [ 1.03755e+00  1.03693e+00  9.95820e-01  1.00733e+00]
 [ 1.14408e+00  1.08419e+00  7.90360e-01  1.48469e+00]] [8.73943577e-07 9.58782744e-07] (9.173444624394552e-07, 1.0795765479448262e-06)
wrap-fit True [0] [[ 1.00000e-05 -1.00000e-05 -1.00000e-05  2.00000e-05]
 [ 1.03742e+00  1.03713e+00  9.95990e-01  1.00696e+00]
 [ 1.14390e+00  1.08417e+00  7.90400e-01  1.48383e+00]] [8.26691748e-07 3.63036154e-06] (2.63276890097125e-06, 5.520176937420857e-06)
wrap-fit True [1] [[ 0.00000e+00 -3.00000e-05 -2.00000e-05  5.00000e-05]
 [ 1.03822e+00  1.04033e+00  9.96410e-01  1.00650e+00]
 [ 1.14572e+00  1.08405e+00  7.89380e-01  1.47996e+00]] [4.56246090e-05 8.43095097e-07] (3.226697813347694e-05, 7.026554866930432e-05)
```

### Turn 6 — `execute_in_remote_docker` (`27940 ms` LLM)
- **Rationale**: The fitted inward multipliers match arithmetic face averages, and the seam multipliers match an averaged diffusivity plus endpoint upwinding. Test that parameter-free stencil on each held-out pulse before reusing the archived quadratic solver.
- **Alternatives Weighed**: `Hardcode fitted boundary multipliers` (rejected: They vary across pulse fits and would not transfer to new diffusivity, velocity, or grid spacing.)

```bash
sed '/stencils=/,$d' /tmp/wrap98.py > /tmp/exact98.py
cat >> /tmp/exact98.py <<'PY'
q=np.zeros((3,4))
for a,d in enumerate('xy'):
 k=st['kappa_'+d]; v=st['vel_'+d]; h=st['L'+d]/(len(k)-1); e=k[[0,-1]]
 q[1,2*a:2*a+2]=0.5*(e+k[[1,-2]])/e
 q[2,2*a:2*a+2]=e.mean()/e
print('face-average ratios',q)
r=pred(q,False,True)-Y
print('exact stencil per shift and pulse',np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)), 'max abs',np.max(np.abs(r)))
# Compare the best candidate selected independently on each pulse.
from itertools import product
candidates=[]
for leftavg,seamavg,adv in product([False,True],repeat=3):
 t=q.copy()
 if not leftavg:t[1]=1
 if not seamavg:t[2]=1
 r=pred(t,False,adv)-Y
 candidates.append((np.mean(r*r,axis=0),np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)),(leftavg,seamavg,adv)))
for j in (0,1):
 c=min(candidates,key=lambda a:a[0][j]); print('trained on',j,'stencil',c[2],'heldout worst shift',np.max(c[1][:,1-j]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/exact98.py; python3 - <<'PY'
import json
x=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus log',type(x).__name__)
for z in x:
 if isinstance(z,dict): print({k:v if not isinstance(v,list) else ('list',len(v)) for k,v in z.items()})
 else: print(z)
PY
sed -n '37,180p' /tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`503 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
+ ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }
import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
M, nx, ny = np.asarray(st["source_modes"]).shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "kkt_certificate" not in out or not isinstance(out["kkt_certificate"], dict): errs.append("missing:kkt_certificate")
    else:
        kc = out["kkt_certificate"]
        for kk in ("weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"):
            if kk not in kc: errs.append(f"missing_kkt:{kk}")
        if not errs:
            kw = np.asarray(kc["weights"], dtype=float); ll = np.asarray(kc["lam_lower"], dtype=float); lu = np.asarray(kc["lam_upper"], dtype=float)
            po = float(kc["primal_objective"]); dl = float(kc["dual_lower_bound"])
            if np.min(kw) < -1e-6 or np.max(kw) > float(st["u_box"]) + 1e-6: errs.append("kkt_box_violation")
            if abs(float(np.sum(kw)) - float(st["total_budget"])) > 1e-5: errs.append("kkt_budget_violation")
            if np.min(ll) < -1e-6 or np.min(lu) < -1e-6: errs.append("kkt_neg_multiplier")
            if dl < -1e-7 or po < dl - 1e-6 or (po - dl) > 2e-4: errs.append(f"kkt_duality_gap_or_sign:{po:.6f}_vs_{dl:.6f}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 7 — `execute_in_remote_docker` (`21934 ms` LLM)
- **Rationale**: Implement the periodic stencil selected independently by both pulses and retain dynamic calibration and the archived box-simplex solver. The isolated schema check will catch import, output-shape, or basic certificate failures before independent physical and numerical stress tests.
- **Alternatives Weighed**: `Carry fitted wall coefficients into the solver` (rejected: Both tracer folds independently select the same parameter-free stencil, with lower worst-shift error than the fitted coefficients.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py').read_text()
start, end = p.index('def _axis('), p.index('def _forward(')
p = p[:start] + '''def _axis(kappa, velocity, h):
    """Periodic row-local upwind operator, including the endpoint face."""
    face_right = 0.5 * (kappa + np.roll(kappa, -1)) / h**2
    left = np.roll(face_right, 1) + np.maximum(velocity, 0) / h
    right = face_right + np.maximum(-velocity, 0) / h
    n = len(kappa)
    axis = diags((-left[1:], left + right, -right[:-1]),
                 (-1, 0, 1), format='csc')
    seam = coo_matrix(([-left[0], -right[-1]], ([0, n - 1], [n - 1, 0])),
                      shape=(n, n))
    return axis + seam.tocsc()


''' + p[end:]
p = p.replace('Steady transport tomography with calibrated', 'Periodic steady transport tomography with calibrated').replace('from scipy.sparse import diags', 'from scipy.sparse import coo_matrix, diags').replace(", study['robin_x'])", ')').replace(", study['robin_y'])", ')')
Path('/holdout_task/batch_solver.py').write_text(p)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`446 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=35f03ee8c869a313 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35f03ee8 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`39229 ms` LLM)
- **Rationale**: The public schema and independent host probe passed. This last test targets failures those inputs may miss: incorrect seam indexing on rectangular grids, calibration tied to public gains, and noncomplementary multipliers at saturated budgets. A failure would point to the specific solver component to correct.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    base = pickle.load(f)[0]['study']

def axis(k, v, h):
    n = len(k)
    a = np.zeros((n, n))
    for i in range(n):
        for j, speed in (((i - 1) % n, v[i]), ((i + 1) % n, -v[i])):
            c = (k[i] + k[j]) / (2 * h * h) + max(speed, 0) / h
            a[i, j] -= c
            a[i, i] += c
    return a

rng = np.random.default_rng(983)
worst = dict(field=0., budget=0., stationarity=0., complementarity=0., gap=0.)
for nx, ny in ((5, 8), (9, 4)):
    st = dict(base, nx=nx, ny=ny, Lx=1.1, Ly=2.3)
    for d, n in (('x', nx), ('y', ny)):
        st['kappa_'+d] = rng.uniform(.15, .8, n)
        st['vel_'+d] = rng.uniform(-1, 1, n)
    st['V_reaction_2d'] = rng.uniform(.2, 1, (nx, ny))
    st['source_modes'] = rng.uniform(0, 1, (8, nx, ny))
    st['sensor_ij'] = np.column_stack((rng.integers(nx, size=32), rng.integers(ny, size=32)))
    ax = axis(st['kappa_x'], st['vel_x'], st['Lx']/(nx-1))
    ay = axis(st['kappa_y'], st['vel_y'], st['Ly']/(ny-1))
    L = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx)) + np.diag(st['V_reaction_2d'].ravel(order='F'))
    rhs = np.column_stack([a.ravel(order='F') for a in st['source_modes']])
    fields = [np.linalg.solve(L + mu*np.eye(nx*ny), rhs) for mu in st['mu_shifts']]
    ij = st['sensor_ij']
    S = np.vstack([a[ij[:,0]+nx*ij[:,1]] for a in fields])
    ref = rng.normal(size=(80, 4))
    mix = np.eye(4) + rng.normal(0, .1, (4, 4))
    bias = rng.normal(0, .1, 4)
    st['sensor_calibration_sweeps'] = dict(base['sensor_calibration_sweeps'], ref_rosette=ref, raw_rosette=ref@mix+bias)
    Z = st['Z_nuisance']
    y = S @ np.array([1., 0., .4, 0., 0., 0., 0., 0.]) + Z @ rng.normal(size=Z.shape[1])
    st['y_obs_raw'] = (y.reshape(-1, 4) @ mix + bias).ravel()
    P = np.eye(len(y)) - Z @ np.linalg.solve(Z.T@Z + st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    cap = st['u_box']
    for budget in (0., cap, 1.4, 8*cap):
        out = m.solve_batch(dict(st, total_budget=budget))
        w, cert = out['weights'], out['kkt_certificate']
        lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
        assert np.min(w)>=-1e-12 and np.max(w)<=cap+1e-12 and np.min(lo)>=0 and np.min(hi)>=0
        np.testing.assert_array_equal(w, cert['weights'])
        r = P @ (S @ w-y)
        g = (P@S).T @ r + st['source_ridge']*w
        objective = .5*(r@r+st['source_ridge']*(w@w))
        dual = objective - g@w + cert['nu_sum']*budget + cap*np.minimum(g-cert['nu_sum'], 0).sum()
        np.testing.assert_allclose([cert['primal_objective'], cert['dual_lower_bound']], [objective, dual], atol=1e-11, rtol=0)
        expected = (fields[-1] @ w).reshape(nx, ny, order='F')
        trial = dict(field=np.max(np.abs(out['terminal_field']-expected)), budget=abs(w.sum()-budget), stationarity=np.max(np.abs(g-cert['nu_sum']-lo+hi)), complementarity=max(np.max(abs(lo*w)), np.max(abs(hi*(cap-w)))), gap=abs(objective-dual))
        for k, value in trial.items():
            worst[k] = max(worst[k], value)
assert max(worst.values()) < 1e-9, worst
print('Independent dense periodic reference; eight grid/budget slices:', worst)
PY
```
- **Container Exit Code**: `0` (`498 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Independent dense periodic reference; eight grid/budget slices: {'field': 1.3322676295501878e-15, 'budget': 5.551115123125783e-17, 'stationarity': 4.263256414560601e-14, 'complementarity': 7.46069872548105e-15, 'gap': 1.5987211554602254e-14}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=35f03ee8c869a313 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35f03ee8 (directory).
```

### Turn 9 — `submit_completed_task` (`15093 ms` LLM)
- **Rationale**: Both tracer folds selected the same periodic stencil. The final artifact passed isolated schema validation, the independent public probe, and eight grid-and-budget stress cases against a dense reference, with worst KKT stationarity error of 4.27e-14.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5397'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, and the worst held-out tracer-by-shift RMSE was 1.23e-6.`

### Turn 10 — `execute_in_remote_docker` (`33179 ms` LLM)
- **Rationale**: The new out-of-fold sweep test can detect nonlinear calibration errors hidden by an in-sample affine fit. Tracer residuals before and after nuisance projection, broken down by sensor bank, can reveal whether pooled agreement hides a systematic bias. A clear signal would justify testing a targeted correction against both tracer pulses; otherwise retain the verified solver.
- **Alternatives Weighed**: `Run classification, relay-mode, or drift-operator checks` (rejected: This task has no class labels, relay state, or stochastic drift operator. The relevant tests are held-out calibration error and tracer residuals by shift and sensor bank.)

```bash
cat > /tmp/audit98.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
sw = st['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
scale = np.std(x, axis=0)

def features(x, model, scale):
    cols = [np.ones(len(x)), *x.T]
    if model == 'quadratic':
        cols += [x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]
    if model == 'rational':
        cols += [x[:, i]/(1+(x[:, i]/scale[i])**2) for i in range(4)]
        cols += [(x[:, i]-x[:, j])/(1+((x[:, i]-x[:, j])/np.hypot(scale[i],scale[j]))**2) for i in range(4) for j in range(i+1,4)]
    return np.column_stack(cols)

folds = np.array_split(np.random.default_rng(98).permutation(len(x)), 5)
for name in ('affine', 'quadratic', 'rational'):
    out = np.empty_like(y)
    worst = 0
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)),test)
        scale = np.std(x[train],axis=0)
        coef = np.linalg.lstsq(features(x[train],name,scale),y[train],rcond=None)[0]
        out[test] = features(x[test],name,scale)@coef-y[test]
        worst = max(worst,np.max(np.sqrt(np.mean(out[test]**2,axis=0))))
    print(name,'OOF RMS by channel',np.sqrt(np.mean(out**2,axis=0)),'worst fold/channel RMS',worst)
S, _ = m._forward(st)
Z = st['Z_nuisance']
B = np.asarray(st['row_bank_id'])
for i,p in enumerate(item['campaign_diagnostics']):
    r = m._calibrate(dict(st,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse']
    q = r-Z@np.linalg.solve(Z.T@Z+st['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@r)
    print('pulse',i,'raw/profiled RMS',np.sqrt(np.mean(r*r)),np.sqrt(np.mean(q*q)))
    print('pulse',i,'shift RMS',np.sqrt(np.mean(r.reshape(len(st['mu_shifts']),-1)**2,axis=1)))
    print('pulse',i,'banks: count, signed mean, RMS',[(int(b),int((B==b).sum()),float(r[B==b].mean()),float(np.sqrt(np.mean(r[B==b]**2)))) for b in np.unique(B)])
print('single-bank sweep shapes', {k: np.shape(v) for k,v in sw['single_bank_sweeps'].items()})
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit98.py
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine OOF RMS by channel [1.03180719e-05 1.17383571e-05 1.00131370e-05 8.29772171e-06] worst fold/channel RMS 1.5166580434512854e-05
quadratic OOF RMS by channel [1.16133908e-05 1.40771968e-05 1.11955076e-05 9.85623939e-06] worst fold/channel RMS 1.7898999636817845e-05
rational OOF RMS by channel [1.21144154e-05 1.31330168e-05 1.09522100e-05 9.80596529e-06] worst fold/channel RMS 1.7516834163947924e-05
pulse 0 raw/profiled RMS 1.0546348982844154e-06 1.0388715305038363e-06
pulse 0 shift RMS [8.39987991e-07 1.07587831e-06 1.21394825e-06]
pulse 0 banks: count, signed mean, RMS [(0, 24, 8.762894493739616e-08, 9.457293413144409e-08), (1, 24, 1.930793093028275e-06, 1.9506393881593076e-06), (2, 24, -5.818306496718079e-07, 6.121781801987032e-07), (3, 24, 2.3837840376281227e-07, 5.102145508483396e-07)]
pulse 1 raw/profiled RMS 1.0745916376726895e-06 1.0567879401594934e-06
pulse 1 shift RMS [8.69533035e-07 1.09643722e-06 1.22718348e-06]
pulse 1 banks: count, signed mean, RMS [(0, 24, 7.254050502092611e-08, 9.416525378937575e-08), (1, 24, 1.95949126715559e-06, 1.9792244078881464e-06), (2, 24, -6.063512190040398e-07, 6.323138231701593e-07), (3, 24, 2.3975528214061764e-07, 5.412685361729676e-07)]
single-bank sweep shapes {'bank_id': (24,), 'ref_signal': (24,), 'raw_readout': (24,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=35f03ee8c869a313 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35f03ee8 (directory).
```

### Turn 11 — `submit_completed_task` (`15146 ms` LLM)
- **Rationale**: Both tracer folds selected the same periodic stencil. Affine calibration had the lowest worst-fold RMSE, and the unchanged artifact passed isolated schema and dense-reference tests. The independent public probe measured terminal-field relative L2 error of 9.1e-7.
- **Alternatives Weighed**: `Replace affine calibration with a quadratic or rational model` (rejected: Both alternatives increased worst-fold calibration RMSE, and neither is supported by the tracer residual audit.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5397'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and eight independent grid-and-budget stress cases passed. The worst held-out tracer-by-shift RMSE was 1.23e-6.`

### Turn 12 — `execute_in_remote_docker` (`30851 ms` LLM)
- **Rationale**: Bank 1 shows a same-sign residual on both tracer pulses. Compare the two calibration sweep types, then test whether pooling them improves agreement on both independent pulses. Keep the current calibration if the single-bank sweeps do not transfer.
- **Alternatives Weighed**: `Fit a bank correction directly from the tracer residuals` (rejected: That would mix calibration and transport effects and depend on tracer data the solver does not receive.)

```bash
cat > /tmp/bank98.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
sw = st['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
beta = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
single = sw['single_bank_sweeps']
bank = single['bank_id'].astype(int)
signal, readout = single['ref_signal'], single['raw_readout']
print('row bank IDs match rosette positions:', np.array_equal(st['row_bank_id'], np.tile(np.arange(4), len(st['y_obs_raw'])//4)))
S, _ = m._forward(st)
for j in np.unique(bank):
    mask = bank == j
    A = np.c_[np.ones(mask.sum()), signal[mask]]
    prior = beta[[0,j+1],j]
    fitted = np.linalg.lstsq(A, readout[mask], rcond=None)[0]
    loo = np.array([A[i]@np.linalg.lstsq(np.delete(A,i,axis=0), np.delete(readout[mask],i), rcond=None)[0]-readout[mask][i] for i in range(len(A))])
    print('bank', j, 'full-sweep bias/gain', prior, 'single-bank bias/gain', fitted, 'single-bank transfer/LOO RMS', np.sqrt(np.mean((A@prior-readout[mask])**2)), np.sqrt(np.mean(loo**2)))

def tracer_metric(b):
    errors = []
    for p in item['campaign_diagnostics']:
        ycal = np.linalg.solve(b[1:].T, (p['y_pulse_raw'].reshape(-1,4)-b[0]).T).T.ravel()
        errors.append((ycal-S@p['w_pulse']).reshape(len(st['mu_shifts']),-1,4))
    return np.sqrt(np.mean(np.asarray(errors)**2,axis=2))

B = np.zeros((len(bank),5,4))
B[np.arange(len(bank)),0,bank] = 1
B[np.arange(len(bank)),bank+1,bank] = signal
pooled = beta.copy()
for j in range(4):
    mask = bank == j
    pooled[:,j] = np.linalg.lstsq(np.vstack([np.c_[np.ones(len(x)),x],B[mask,:,j]]),np.r_[y[:,j],readout[mask]],rcond=None)[0]
for name, b in [('rosette-only', beta), ('pooled sweeps', pooled)]:
    rms = tracer_metric(b)
    print(name, 'worst shift-by-bank RMS per pulse', rms.max(axis=(1,2)), 'bank 1 by pulse and shift', rms[:,:,1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/bank98.py
```
- **Container Exit Code**: `0` (`496 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
row bank IDs match rosette positions: True
bank 0 full-sweep bias/gain [0.07999983 1.2900006 ] single-bank bias/gain [0.08000179 1.28999387] single-bank transfer/LOO RMS 9.989334640009175e-06 1.1222340402637835e-05
bank 1 full-sweep bias/gain [-0.07000199  0.80000277] single-bank bias/gain [-0.0699893   0.80000244] single-bank transfer/LOO RMS 1.3944255507160443e-05 8.10250508753785e-06
bank 2 full-sweep bias/gain [0.06000119 1.2499978 ] single-bank bias/gain [0.06000149 1.24999525] single-bank transfer/LOO RMS 1.232239917197423e-05 1.6070903246656662e-05
bank 3 full-sweep bias/gain [-0.05000095  0.83000114] single-bank bias/gain [-0.05000444  0.83000639] single-bank transfer/LOO RMS 7.935680567145541e-06 1.0475556982952396e-05
rosette-only worst shift-by-bank RMS per pulse [2.19518891e-06 2.21512676e-06] bank 1 by pulse and shift [[1.59526256e-06 2.01277546e-06 2.19518891e-06]
 [1.63828102e-06 2.03991091e-06 2.21512676e-06]]
pooled sweeps worst shift-by-bank RMS per pulse [9.87293242e-07 1.00897599e-06] bank 1 by pulse and shift [[5.64526702e-07 8.58136074e-07 9.87293242e-07]
 [6.05872863e-07 8.86076884e-07 1.00897599e-06]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=35f03ee8c869a313 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_35f03ee8 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`41562 ms` LLM)
- **Rationale**: The pooled fit improves both tracer pulses without fitting to them or changing transport. Verify the implementation on the public pulses and in isolation with changed sensor gains, offsets, and cross-talk at interior and saturated budgets.
- **Alternatives Weighed**: `Correct only bank 1` (rejected: That would select a bank from this campaign’s tracer residuals. Pooling both sweep types applies the same calibration rule to every bank and improves both tracers.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
import re
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
start = s.index('def _calibrate(')
end = start + re.search(r'\n(?=def )', s[start:]).start()
s = s[:start] + '''def _calibrate(study):
    """Fit an affine rosette, adding each isolated sweep to its own bank."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    design = np.column_stack((np.ones(len(ref)), ref))
    single = sweeps['single_bank_sweeps']
    banks = np.asarray(single['bank_id'])
    signal = np.asarray(single['ref_signal'], dtype=float)
    readout = np.asarray(single['raw_readout'], dtype=float)
    coef = np.empty((5, 4))
    for j in range(4):
        rows = banks == j
        isolated = np.zeros((np.count_nonzero(rows), 5))
        isolated[:, 0] = 1
        isolated[:, j + 1] = signal[rows]
        coef[:, j] = np.linalg.lstsq(
            np.vstack((design, isolated)), np.r_[raw[:, j], readout[rows]], rcond=None
        )[0]
    obs = np.asarray(study['y_obs_raw'], dtype=float).reshape(-1, 4)
    return np.linalg.solve(coef[1:].T, (obs - coef[0]).T).T.ravel()

''' + s[end:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
st = item['study']
S, U = m._forward(st)
worst = []
for pulse in item['campaign_diagnostics']:
    residual = m._calibrate(dict(st,y_obs_raw=pulse['y_pulse_raw'])) - S @ pulse['w_pulse']
    rms = np.sqrt(np.mean(residual.reshape(len(st['mu_shifts']),-1,4)**2, axis=1))
    worst.append(float(rms.max()))
assert max(worst)<1.1e-6, worst
print('Worst shift-by-bank RMSE per tracer:', worst)
Z=st['Z_nuisance']
P=np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
A=P@S
rng=np.random.default_rng(98)
x=rng.normal(size=(80,4))
bank=np.repeat(np.arange(4),6)
signal=rng.uniform(-1,1,len(bank))
ycal=m._calibrate(st)
u=st['u_box']
worst_gap=0.
for _ in range(3):
    mix=np.eye(4)+rng.normal(0,.17,(4,4))
    b=rng.normal(0,.1,4)
    sw=dict(ref_rosette=x,raw_rosette=x@mix+b,single_bank_sweeps=dict(bank_id=bank,ref_signal=signal,raw_readout=b[bank]+mix[bank,bank]*signal))
    obs=(ycal.reshape(-1,4)@mix+b).ravel()
    synth=dict(st,sensor_calibration_sweeps=sw,y_obs_raw=obs)
    np.testing.assert_allclose(m._calibrate(synth), ycal, rtol=0, atol=1e-12)
    for budget in (0., u, st['total_budget'], S.shape[1]*u):
        out=m.solve_batch(dict(synth,total_budget=budget))
        w, c=out['weights'],out['kkt_certificate']
        r=A@w-P@ycal
        g=A.T@r+st['source_ridge']*w
        J=.5*(r@r+st['source_ridge']*(w@w))
        nu=c['nu_sum']
        dlo=J-g@w+nu*budget+u*np.minimum(g-nu,0).sum()
        lo,hi=np.asarray(c['lam_lower']),np.asarray(c['lam_upper'])
        assert np.min(w)>=-1e-12 and np.max(w)<=u+1e-12 and abs(w.sum()-budget)<1e-11
        np.testing.assert_allclose([c['primal_objective'],c['dual_lower_bound']],[J,dlo],rtol=0,atol=1e-11)
        assert max(np.max(abs(g-nu-lo+hi)),np.max(abs(lo*w)),np.max(abs(hi*(u-w))))<1e-9
        np.testing.assert_allclose(out['terminal_field'], np.tensordot(w,U,axes=(0,0)),rtol=0,atol=1e-12)
        worst_gap=max(worst_gap,abs(J-dlo))
assert worst_gap<1e-9
print('Three changed rosettes by four budgets: worst independent duality gap',worst_gap)
PY
```
- **Container Exit Code**: `0` (`731 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Worst shift-by-bank RMSE per tracer: [9.872932421424433e-07, 1.0089759906864615e-06]
Three changed rosettes by four budgets: worst independent duality gap 2.6645352591003757e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cb58963c3a479583 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_cb58963c (directory).
```

### Turn 14 — `submit_completed_task` (`18838 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation. Three changed rosettes across four budgets yielded a worst independently computed duality gap of 2.66e-15. The periodic stencil was unchanged from the dense-reference validation.
- **Alternatives Weighed**: `Keep rosette-only calibration` (rejected: Joint calibration improved the worst shift-by-bank error on both tracer pulses without fitting to either pulse.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. Joint calibration reduced the worst tracer RMSE across shifts and sensor banks from 2.22e-6 to 1.01e-6. The public schema and all 12 calibration-and-budget stress cases passed.`

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
