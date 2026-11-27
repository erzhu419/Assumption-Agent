# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m96_w02_batch_04__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M96-Autonomous-Structure-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_m96_w02_step_04` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `444,787` (`429,274` / `85,760` / `15,513`) |
| **Total Episode Wall Time (s)** | `515.94s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_11_split_preloaded_relaxation_allocat`**: [BIFURCATED SUB-REGIME `missing_feature_count > 0`]: Pooled raw features confound the specimen signal with the laboratory gauge, and cell-balanced imputation restores the feature coordinate but not the specimen-specific drift: a 0.9 onset-innovation correction helped all four incomplete-feature slices and worsened all four complete-feature slices. Calibrate loads with per-group affine check consensus, using a supplied historical gain when the checks identify only an offset; align coupon features against the reference panel before imputation, then fit the measured preload and exact two-branch recurrence jointly to strain and onset using separate 2.5-MAD Cauchy scales. Require the smallest singular value of the column-normalized robust Jacobian to exceed 1e-7 times the largest, and update only the linear drift on rows with at least one missing feature and finite onset, using missing-drift variance divided by missing-drift plus onset-noise variance. Apply the ridge residual map to both the modulated allocation design and observations before solving the box-simplex QP; certify feasibility, stationarity, and the gap to an independently computable tangent lower bound rather than trusting optimizer status alone. The episode probe recorded strain MAE 0.003892, relative time-constant error 0.003441, zero panel-gauge discrepancy, and zero allocation duality gap; it did not establish cross-epoch persistence or extrapolation to unseen regimes. (Counter-regime `missing_feature_count` warning: Complete-feature rows retain feature-based drift. Incomplete rows without finite measured onset also retain it; with finite onset, update only drift, and leave it unchanged if the estimated missing-drift variance is zero.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The cell-balanced feature mean is sufficient to recover the coupon-specific linear creep coefficient when a feature is missing.` (witness: `All four held-out labs benefited from a 0.9 initial-rate innovation correction on missing-feature rows, while all four complete-feature slices worsened.`) -> Pivot `Keep reference alignment and hereditary dynamics; infer a coupon-specific correction only where feature information is missing, with shrinkage for rate noise.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`13520 ms` LLM)
- **Rationale**: Inspect the public contract and prior solver before reusing any equations; differences in time conventions or history features would change the model design.

```bash
cd /holdout_task && cat validate_public_schema.py && sed -n '1,210p' public_asset_index.json && ls -lh /tmp/hegel_continual_archive && sed -n '1,260p' /tmp/hegel_continual_archive/alpha_m96_w02_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment imputation robust calibration'
```
- **Container Exit Code**: `0` (`364 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_calibrated_hereditary_creep@2ed1c9bd | align_reference_features(raw: np.ndarray, panel: np.ndarray, group: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align row-matched reference measurements under one affine gauge per group | REQ: {'field': 'yield_stress', 'op': '>', 'value': 0}; {'field': 'maximum_calibrated_absolute | --describe autogenous_calibrated_hereditary_creep.align_reference_features
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
```

### Turn 2 — `execute_in_remote_docker` (`15343 ms` LLM)
- **Rationale**: Recover the prior design equations and inspect the current arrays and calibration archive structure. This determines whether the prior time convention transfers and whether sparse lab sweeps can be reconciled with archived data.

```bash
cd /holdout_task; tail -35 validate_public_schema.py; cat /tmp/hegel_continual_archive/alpha_m96_w02_batch_03_solver.py; python3 - <<'PY'
import numpy as np,pickle,json
with open('public_alpha_m96_w02_batch_04.pkl','rb') as f:d=pickle.load(f)
def show(o,level=0):
 if isinstance(o,dict):
  for k,v in o.items():
   print(' '*level,k,end=' ')
   if isinstance(v,dict):print();show(v,level+2)
   else:
    a=np.asarray(v);print(a.shape,a.dtype,a.ravel()[:8])
show(d)
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'));print(str(h)[:3800])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_calibrated_hereditary_creep.align_reference_features
```
- **Container Exit Code**: `0` (`554 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ue": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_reference_features",
          "signature": "align_reference_features(raw: np.ndarray, panel: np.ndarray, group: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray]",
          "doc_summary": "Align row-matched reference measurements under one affine gauge per group."
        }
      ],
      "index_cards": [
        "autogenous_calibrated_hereditary_creep@2ed1c9bd | align_reference_features(raw: np.ndarray, panel: np.ndarray, group: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align row-matched reference measurements under one affine gauge per group | REQ: {'field': 'yield_stress', 'op': '>', 'value': 0}; {'field': 'maximum_calibrated_absolute | --describe autogenous_calibrated_hereditary_creep.align_reference_features",
        "autogenous_calibrated_hereditary_creep@2ed1c9bd | creep_hold(past_stress: np.ndarray, hold_stress: np.ndarray, durations: np.ndarray, times: np.ndarray, aligned: np.ndarray, taus: np.ndarray, *, weights: np.ndarray | None=None, strain0: np.ndarray | None=None, onset: np.ndarray | None=None, incomplete: np.ndarray | None=None) -> np.ndarray | tuple[np.ndarray, np.ndarray] | Evaluate an isothermal, piecewise-constant sub-yield history from zero memory | REQ: {'field': 'yield_stress', 'op': '>', 'value': 0}; {'field': 'maximum_calibrated_absolute | --describe autogenous_calibrated_hereditary_creep.creep_hold",
        "autogenous_calibrated_hereditary_creep@2ed1c9bd | fit_calibrated_creep(past_command: np.ndarray, hold_command: np.ndarray, durations: np.ndarray, times: np.ndarray, aligned: np.ndarray, group: np.ndarray, strain0: np.ndarray, strain: np.ndarray, rate: np.ndarray, check_group: np.ndarray, check_command: np.ndarray, check_stress: np.ndarray, *, tau_bounds: tuple[float, float], yield_stress: float, check_noise_fraction: float=0.0001, rank_tol: float=1e-07) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Fit two shared hereditary branches and positive group load-cell gains | REQ: {'field': 'yield_stress', 'op': '>', 'value': 0}; {'field': 'maximum_calibrated_absolute | --describe autogenous_calibrated_hereditary_creep.fit_calibrated_creep"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_calibrated_hereditary_creep",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`18307 ms` LLM)
- **Rationale**: The previous output truncated the inherited design and calibration rows. A compact inspection will determine the exact Prony basis and identify which lab sweeps need archival support rather than a standalone affine fit.
- **Alternatives Weighed**: `Reuse the archived creep skill directly` (rejected: Its applicability requires strictly sub-yield stresses, whereas the current contract also covers prior and current overstress.)

```bash
cd /holdout_task; sed -n '1,105p' /tmp/hegel_continual_archive/alpha_m96_w02_batch_03_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
D=pickle.load(open('public_alpha_m96_w02_batch_04.pkl','rb'))[0]['dataset']
for k,v in D.items():
 if isinstance(v,dict): print(k,[(a,np.shape(b)) for a,b in v.items()]);continue
 a=np.asarray(v);print(k,a.shape,a.ravel()[:4])
c=D['calibration_load_cells'];l=np.asarray(c['lab_id']);x=np.asarray(c['cmd_stress']);y=np.asarray(c['true_stress']);
for i in range(4):
 m=l==i; print('checks',i,np.round(np.c_[x[m],y[m]],6).tolist())
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'));print('archive',type(h).__name__,list(h) if isinstance(h,dict) else len(h));print(str(h)[:550])
PY
```
- **Container Exit Code**: `0` (`633 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.median(abs(a - np.median(a)))
        s = max(1e-7, mad(r[:, :-1]))
        return np.r_[np.full(y.shape[1] - 1, s), max(1e-7, mad(r[:, -1]))]

    def linear_start(a):
        a = a.reshape(-1, 7)
        yflat = y.ravel()
        b = np.linalg.lstsq(a, yflat, rcond=None)[0]
past_dt (18,) [0.7698 0.3214 0.7986 0.5057]
eval_times (10,) [0.25 0.8  1.35 1.9 ]
yield_stress () [1.52]
calibration_load_cells [('lab_id', (40,)), ('cmd_stress', (40,)), ('true_stress', (40,))]
train [('lab_id', (200,)), ('stratum', (200,)), ('coupon_features_raw', (200, 4)), ('ref_coupons', (200, 32)), ('past_stress_cmd', (200, 18)), ('strain_0', (200,)), ('stress_hold_cmd', (200,)), ('instant_rate_obs', (200,)), ('strain_curves_obs', (200, 10))]
test [('lab_id', (80,)), ('stratum', (80,)), ('coupon_features_raw', (80, 4)), ('ref_coupons', (80, 32)), ('past_stress_cmd', (80, 18)), ('strain_0', (80,)), ('stress_hold_cmd', (80,)), ('instant_rate_obs', (80,)), ('residual_plastic_state', (80,))]
checks 0 [[-2.137417, -2.456585], [-2.038153, -2.337875], [0.441746, 0.62932], [-0.923704, -1.004281], [-1.646941, -1.869737], [-1.909118, -2.183572], [1.721682, 2.160487], [-0.827312, -0.889093], [0.907314, 1.186205], [2.060784, 2.566157]]
checks 1 [[-2.146344, -1.745215], [0.539238, 0.253653], [-1.206333, -1.04564], [1.565345, 1.01746], [2.055058, 1.38169], [-1.496642, -1.261789], [1.806548, 1.19684], [1.720753, 1.133003], [2.22856, 1.511244], [-1.102923, -0.968706]]
checks 2 [[-1.689439, -2.042248], [0.268036, 0.457024], [1.257286, 1.719933], [1.787653, 2.396951], [-0.071829, 0.022965], [1.937655, 2.588445], [0.648325, 0.942368], [0.479652, 0.727122], [0.646366, 0.939859], [-1.632327, -1.969182]]
checks 3 [[-0.391862, -0.458899], [-2.051023, -1.749526], [-0.09675, -0.229386], [0.149779, -0.037718], [2.202939, 1.559303], [-2.112328, -1.797171], [-0.109734, -0.239583], [2.03645, 1.429609], [-1.353096, -1.206641], [-2.198558, -1.864133]]
archive list 4
[{'episode_short_name': 'alpha_m96_w02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.102436, 2.271537, -0.390137, 1.663003, 1.192465, 0.070258, -0.56265, -0.707913, 1.520913, 2.264016, -1.768567, 0.50801, -1.707781, 2.302804, 2.138134, 2.25892, -0.964218, -0.748286, -0.495635, -1.168253, 0.884886, -0.331964, -0.086177, 2.162354, -1.907811, 0.698151, 0.681071, 1.89468, 0.31516, 0.087615, -0.474789, -1.783433, 0.724747, 1.895774, -0.312701
```

### Turn 4 — `execute_in_remote_docker` (`23557 ms` LLM)
- **Rationale**: Recover the exact hereditary feature ordering and determine which lab and overstress slices the public data actually supports. Compare the test-only state with the specified history integral before deciding whether that field belongs in the constitutive fit.
- **Alternatives Weighed**: `Pool all historical and current calibration checks` (rejected: Pooling could transfer stale gains into a supported current sweep; compare episode-wise fits before selecting a fallback.)

```bash
cd /holdout_task; sed -n '54,78p;103,215p' /tmp/hegel_continual_archive/alpha_m96_w02_batch_03_solver.py; grep -inE 'plastic|alpha_m96_w02|strain' public_asset_index.json | cut -c1-240; python3 - <<'PY'
import pickle,json,numpy as np
P=pickle.load(open('public_alpha_m96_w02_batch_04.pkl','rb')); print('public entries',len(P),[list(z) for z in P]); D=P[0]['dataset']; C=D['calibration_load_cells']; H=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for s in [*H,C]:
 l,x,y=map(np.asarray,(s['lab_id'],s['cmd_stress'],s['true_stress'])); print(s.get('episode_short_name','current'),[(np.round(p:=np.polyfit(x[l==i],y[l==i],1),6).tolist(),round(np.max(abs(y[l==i]-np.polyval(p,x[l==i]))),6)) for i in range(4)])
p=np.array([np.polyfit(x[l==i],y[l==i],1) for i in range(4)])
for key in ['train','test']:
 a=D[key]; g,b=p[a['lab_id']].T; past=g[:,None]*a['past_stress_cmd']+b[:,None]; hold=g*a['stress_hold_cmd']+b; d=np.maximum(0,abs(past)-D['yield_stress'])**2@D['past_dt']; active=abs(hold)>D['yield_stress']; print(key,'missing feature rows',np.sum(np.isnan(a['coupon_features_raw']).any(axis=1))); print('by lab, history-active/hold-active counts')
 for lab in range(4):
  ix=a['lab_id']==lab; print(lab,np.bincount(2*(d[ix]>0)+active[ix],minlength=4).tolist())
 if 'residual_plastic_state' in a:
  r=a['residual_plastic_state']; print('state sample',np.round(np.c_[r,d,d*hold][:12],6)); print('state finite',np.sum(np.isfinite(r)))
PY
```
- **Container Exit Code**: `0` (`552 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ale shared by the two hereditary branches.
    def residual(p):
        a = design.make(p[:2])
        for q, (m, contrast) in zip(p[9:], contrasts):
            a[m, :, :2] += np.expm1(q) * contrast.make(p[:2])[:, :, :2]
        return ((a @ p[2:9] - target) / scale).ravel()

    free = np.full(7 + len(sparse), np.inf)
    p = least_squares(residual, np.r_[tau, theta, np.zeros(len(sparse))],
                      bounds=(np.r_[.2, .2, -free], np.r_[8., 8., free]),
                      loss='cauchy', x_scale='jac', max_nfev=150, ftol=1e-9).x
    for (c, anchor), q in zip(sparse, p[9:]):
        cal[c, 0] = np.exp(q)
        cal[c, 1] -= np.expm1(q) * anchor
    return _order_branches(p[:9])


def solve_alpha_m96_w02_s00(dataset: dict) -> dict:
3:    "/holdout_task/public_alpha_m96_w02_batch_04.pkl": {
84:              "residual_plastic_state": {
91:              "strain_0": {
152:              "strain_0": {
159:              "strain_curves_obs": {
public entries 1 [['dataset', 'gt']]
alpha_m96_w02_s00 [([0.999986, 4.9e-05], 0.000212), ([0.999965, 3.1e-05], 0.000323), ([1.000011, -9e-06], 0.000288), ([1.000022, -7e-06], 0.000265)]
alpha_m96_w02_batch_01 [([1.000001, -4e-06], 0.000143), ([1.0, 6.1e-05], 0.000175), ([0.999975, -2.6e-05], 0.000157), ([0.999993, -2.4e-05], 0.000243)]
alpha_m96_w02_batch_02 [([0.999987, 2.1e-05], 0.000141), ([1.000017, -9e-06], 0.000156), ([1.000052, 6e-05], 0.000289), ([0.999976, -2.4e-05], 9.6e-05)]
alpha_m96_w02_batch_03 [([1.196397, 0.100718], 0.000129), ([0.744335, -0.147725], 0.00017), ([1.276731, 0.114669], 0.000172), ([0.777786, -0.154173], 0.000136)]
current [([1.196421, 0.100679], 0.000179), ([0.744319, -0.147739], 0.000223), ([1.276681, 0.114712], 0.000115), ([0.777786, -0.154197], 0.000117)]
train missing feature rows 19
by lab, history-active/hold-active counts
0 [18, 0, 17, 16]
1 [12, 0, 13, 30]
2 [9, 0, 10, 26]
3 [11, 0, 10, 28]
test missing feature rows 6
by lab, history-active/hold-active counts
0 [4, 0, 4, 10]
1 [5, 0, 2, 11]
2 [7, 0, 8, 9]
3 [4, 0, 6, 10]
state sample [[  0.         2.228168  -5.084019]
 [  0.         2.468317  -6.397009]
 [  0.         2.171708   5.893565]
 [  0.         2.573233   7.215579]
 [  0.         9.90264  -25.385961]
 [  0.         5.278544  14.178478]
 [  0.         2.303986   4.855252]
 [  0.         5.376895  13.958061]
 [  0.         1.20001   -2.875229]
 [  0.         3.889139  10.221026]
 [  0.         1.848993   4.634386]
 [  0.         6.313073  14.804703]]
state finite 80
```

### Turn 5 — `execute_in_remote_docker` (`84597 ms` LLM)
- **Rationale**: Build the explicit hereditary and plastic basis, with the same coefficients governing onset and later strain. Check output schema and fitted parameters before testing held-out labs, prior-overstress specimens, and whether test onset measurements improve imputation.
- **Alternatives Weighed**: `Use the test-only residual-plastic field as a constitutive predictor` (rejected: Observed zero states coexist with substantial calculated history work; the contract defines the mechanism using the calibrated loading history.)

```bash
cat > /holdout_task/alpha_m96_w02_batch_04_solver.py <<'PY'
"""Calibrate the four load cells and fit a two-branch creep law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(checks, lab):
    x = np.asarray(checks['cmd_stress'], float)
    y = np.asarray(checks['true_stress'], float)
    m = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _line(x, y, prior=None):
    # Find consensus at the measured sweep precision, not at the EMI scale.
    i, j = np.triu_indices(len(x), 1)
    distinct = np.abs(x[j] - x[i]) >= .18
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    candidates = list(zip(slopes, y[i] - slopes * x[i]))
    if prior is not None:
        candidates.append(prior)
        candidates.extend((prior[0], v - prior[0] * u) for u, v in zip(x, y))
    best = None
    for g, b in candidates:
        if g <= 0:
            continue
        error = np.abs(y - (g * x + b))
        use = error <= 8e-4
        key = (int(use.sum()), -float(np.minimum(error, 8e-4).sum()))
        if best is None or key > best[0]:
            best = key, use
    if best is None:
        return (np.array([1., np.median(y - x)]) if len(x) else np.array([1., 0.])) if prior is None else prior
    use = best[1]
    u, v = x[use], y[use]
    if len(u) >= 3 and np.ptp(u) >= .35:
        return np.linalg.lstsq(np.column_stack((u, np.ones(len(u)))), v, rcond=None)[0]
    if prior is not None:
        if len(u) >= 2 and np.ptp(u) >= .35 and np.max(np.abs(v - (prior[0] * u + prior[1]))) > .003:
            # Two points support an affine update only if they are mutually consistent with the prior slope.
            g = (v[np.argmax(u)] - v[np.argmin(u)]) / np.ptp(u)
            if abs(g - prior[0]) < .08:
                return np.array([g, np.median(v - g * u)])
        if len(u) and np.max(np.abs(v - (prior[0] * u + prior[1]))) <= .003:
            return np.array([prior[0], np.median(v - prior[0] * u)])
        return prior
    if len(u) >= 2 and np.ptp(u) >= .18:
        return np.linalg.lstsq(np.column_stack((u, np.ones(len(u)))), v, rcond=None)[0]
    return np.array([1., np.median(y - x) if len(x) else 0.])


def _calibrate(checks):
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.is_file() else []
    out = []
    for lab in range(4):
        prior = None
        for episode in history:
            x, y = _sweep(episode, lab)
            if len(x):
                prior = _line(x, y, prior)
        out.append(_line(*_sweep(checks, lab), prior))
    return np.array(out)


def _align(train, test):
    lab, stratum, raw, refs = (np.concatenate((train[k], test[k])) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = raw.astype(float)
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        spread = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (x[use] - offset) / spread
    _, cells, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    w = (1. / counts[cells])[:, None]
    finite = np.isfinite(x)
    mass = (w * finite).sum(axis=0)
    fill = np.divide((w * np.where(finite, x, 0.)).sum(axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    return np.where(finite, x, fill)


class _Design:
    def __init__(self, dataset, split, x, cal):
        gain, bias = cal[np.asarray(split['lab_id'], int)].T
        self.past = split['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = split['stress_hold_cmd'] * gain + bias
        self.dt = np.asarray(dataset['past_dt'])
        self.t = np.asarray(dataset['eval_times'])
        sy = dataset['yield_stress']
        ov = np.maximum(0., np.abs(self.hold) - sy)**2 * np.sign(self.hold)
        self.hist = (np.maximum(0., np.abs(self.past) - sy)**2 * self.dt).sum(axis=1) * self.hold
        self.active = np.array([np.any(ov), np.any(self.hist)])
        drift = np.column_stack((np.ones(len(x)), x, ov))
        self.base = np.zeros((len(x), len(self.t)+1, 9))
        self.base[:, :-1, 2:8] = drift[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:8] = drift

    def make(self, taus, derivative=False):
        a = self.base.copy()
        deriv = np.zeros(a.shape[:2] + (2, 2)) if derivative else None
        for branch, tau in enumerate(taus):
            e = np.exp(-self.dt / tau)
            f = -np.expm1(-self.dt / tau)
            z = np.zeros(len(self.hold))
            dz = np.zeros_like(z)
            for j in range(len(e)):
                if derivative:
                    dz = e[j] * dz + e[j] * self.dt[j] / tau**2 * (z - self.past[:, j])
                z = e[j] * z + f[j] * self.past[:, j]
            gap = self.hold - z
            h = -np.expm1(-self.t / tau)
            a[:, :-1, branch] = gap[:, None] * h
            a[:, -1, branch] = gap / tau
            if derivative:
                dh = -np.exp(-self.t / tau) * self.t / tau**2
                deriv[:, :-1, branch, 0] = gap[:, None] * dh - dz[:, None] * h
                deriv[:, -1, branch, 0] = -dz / tau - gap / tau**2
            if branch == 0:
                a[:, :-1, 8] = self.hist[:, None] * h
                a[:, -1, 8] = self.hist / tau
                if derivative:
                    deriv[:, :-1, 0, 1] = self.hist[:, None] * dh
                    deriv[:, -1, 0, 1] = -self.hist / tau**2
        return (a, deriv) if derivative else a


def _mad(x):
    return max(1e-7, 1.4826 * np.median(np.abs(x - np.median(x))))


def _fit(design, split):
    y = np.column_stack((split['strain_curves_obs'] - split['strain_0'][:, None],
                         split['instant_rate_obs']))
    present = np.isfinite(y)
    curves = present.copy()
    curves[:, -1] = False
    rate = present.copy()
    rate[:, :-1] = False
    is_rate = rate[present]
    y = y[present]
    cols = np.r_[np.ones(7, dtype=bool), design.active]
    n = int(cols.sum())
    has_history = cols[8]
    seeds = []
    grid = (.19, .35, .65, 1.15, 2.1, 3.8, 7., 12.)
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            a = design.make((t1, t2))[present][:, cols]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for _ in range(2):
                r = np.einsum('ij,j->i', a, b) - y
                s = 2.5 * np.where(is_rate, _mad(r[is_rate]) if np.any(is_rate) else 1., _mad(r[~is_rate]))
                w = 1. / np.hypot(s, r)
                b = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
            r = np.einsum('ij,j->i', a, b) - y
            seeds.append((np.r_[t1, t2, b], r))
    _, rr = min(seeds, key=lambda z: np.median(np.abs(z[1][~is_rate])))
    scale = 2.5 * np.where(is_rate, _mad(rr[is_rate]) if np.any(is_rate) else 1., _mad(rr[~is_rate]))
    seeds.sort(key=lambda z: np.log1p((z[1] / scale)**2).sum())

    # Optimize the slow/fast separation directly; the plastic history always uses the fast branch.
    def pack(p):
        t1, t2 = p[:2]
        return np.r_[t1, (t2 - t1) / (24.9 - t1), p[2:]]

    def taus(p):
        return np.array([p[0], p[0] + (24.9 - p[0]) * p[1]])

    def residual(p):
        a = design.make(taus(p))[present][:, cols]
        return (np.einsum('ij,j->i', a, p[2:]) - y) / scale

    def jacobian(p):
        a, dp = design.make(taus(p), derivative=True)
        d = dp[present]
        d1 = d[:, 0, 0] * p[2]
        if has_history:
            d1 += d[:, 0, 1] * p[-1]
        d2 = d[:, 1, 0] * p[3]
        return np.column_stack((d1 + (1 - p[1]) * d2, (24.9 - p[0]) * d2,
                                a[present][:, cols])) / scale[:, None]

    bounds = (np.r_[.051, 1e-5, np.full(n, -np.inf)], np.r_[19., .99999, np.full(n, np.inf)])
    polished = [least_squares(residual, pack(p), jac=jacobian, bounds=bounds, loss='cauchy',
                              x_scale='jac', max_nfev=100, ftol=1e-8, xtol=1e-8)
                for p, _ in seeds[:3]]
    p = min(polished, key=lambda fit: fit.cost).x
    weights = np.zeros(9)
    weights[cols] = p[2:]
    return taus(p), weights


def solve_alpha_m96_w02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    cal = _calibrate(dataset['calibration_load_cells'])
    tau, weights = _fit(_Design(dataset, train, x[:n], cal), train)
    pred = _Design(dataset, test, x[n:], cal).make(tau) @ weights
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
path='/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec=importlib.util.spec_from_file_location('coupon',path); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb'))[0]['dataset']; o=s.solve_alpha_m96_w02_batch_04(d)
for k in ('estimated_taus','hereditary_weights','viscoplastic_weights'): print(k,o[k])
print('calibration',s._calibrate(d['calibration_load_cells'])); r=d['test']['residual_plastic_state']; print('state nonzero',np.count_nonzero(r),'range',r.min(),r.max())
PY
```
- **Container Exit Code**: `0` (`932 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus [0.55140842 3.71751563]
hereditary_weights [ 0.70580893  0.52968652  0.08007653  0.14696779 -0.12758608  0.10792954
 -0.08810511]
viscoplastic_weights [0.2504824  0.21994363]
calibration [[ 1.19642074  0.10067919]
 [ 0.74431944 -0.14773912]
 [ 1.27668055  0.11471248]
 [ 0.77778639 -0.15419686]]
state nonzero 0 range 0.0 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 6 — `execute_in_remote_docker` (`62716 ms` LLM)
- **Rationale**: Leave each lab’s outcomes out of the fit and measure strain and onset errors separately, especially after prior overstress with a subyield hold. Sparse and EMI-contaminated sweep checks will show whether the archive fallback preserves the physical stress scale or needs revision.

```bash
cat > /tmp/coupon04_validate.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('coupon', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = s._calibrate(d['calibration_load_cells'])
x = s._align(tr, te)[:len(tr['lab_id'])]
lab = tr['lab_id']
full = s._Design(d, tr, x, cal)
y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']))
res = np.empty_like(y)
for c in range(4):
    m = lab != c
    sub = {k: v[m] for k, v in tr.items()}
    tau, weights = s._fit(s._Design(d, sub, x[m], cal), sub)
    res[~m] = full.make(tau)[~m] @ weights - y[~m]
    print('held lab', c, 'taus', np.round(tau, 5), 'plastic', np.round(weights[7:], 6))

def metrics(mask):
    err = res[mask]
    if len(err):
        return tuple(np.round([np.sqrt(np.nanmean(err[:, :-1]**2)), np.sqrt(np.nanmean(err[:, -1]**2)), np.nanmedian(np.abs(err[:, :-1])), np.nanmedian(np.abs(err[:, -1]))], 6))

h, o = full.hist != 0, full.base[:, -1, 7] != 0
print('metric order: strain RMSE, onset RMSE, strain median absolute error, onset median absolute error')
for c in range(4):
    for name, use in [('subyield', ~h & ~o), ('history-only', h & ~o), ('hold-active', o)]:
        m = (lab == c) & use
        print(c, name, int(m.sum()), metrics(m))
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
print('missing features', int(missing.sum()), metrics(missing), 'complete', metrics(~missing))
print('horizon RMSE', np.round(np.sqrt(np.nanmean(res[:, :-1]**2, axis=0)), 6))

checks = d['calibration_load_cells']
for c in range(4):
    own = np.flatnonzero(checks['lab_id'] == c)
    for case in ('single spot', 'three EMI spikes'):
        keep = (checks['lab_id'] != c) | (np.arange(len(checks['lab_id'])) == own[len(own)//2]) if case == 'single spot' else np.ones(len(checks['lab_id']), bool)
        corrupt = {k: v[keep].copy() for k, v in checks.items()}
        if case == 'three EMI spikes':
            corrupt['true_stress'][own[:3]] += np.array([.27, -.35, .42])
        g,b = s._calibrate(corrupt)[c] - cal[c]
        worst = max(abs(b-3*g), abs(b+3*g))
        print('calibration', c, case, 'max stress delta', round(worst, 6))
        assert worst < .002
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/coupon04_validate.py
```
- **Container Exit Code**: `0` (`819 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 taus [0.55192 3.72191] plastic [0.250427 0.220031]
held lab 1 taus [0.55237 3.68489] plastic [0.250602 0.220205]
held lab 2 taus [0.55001 3.71018] plastic [0.250622 0.21977 ]
held lab 3 taus [0.5514  3.74948] plastic [0.250312 0.219807]
metric order: strain RMSE, onset RMSE, strain median absolute error, onset median absolute error
0 subyield 18 (0.334247, 0.069281, 0.010014, 0.008607)
0 history-only 17 (0.272017, 0.025615, 0.008882, 0.012679)
0 hold-active 16 (0.278917, 0.019255, 0.008047, 0.004848)
1 subyield 12 (0.42072, 0.115749, 0.008513, 0.015576)
1 history-only 13 (0.305464, 0.015598, 0.008907, 0.011294)
1 hold-active 30 (0.314027, 0.040095, 0.007941, 0.00727)
2 subyield 9 (0.365699, 0.023519, 0.01181, 0.009761)
2 history-only 10 (0.28534, 0.008381, 0.008043, 0.003736)
2 hold-active 26 (0.334201, 0.055641, 0.009365, 0.012655)
3 subyield 11 (0.251982, 0.014327, 0.010256, 0.007086)
3 history-only 10 (0.267636, 0.008677, 0.012748, 0.005992)
3 hold-active 28 (0.310363, 0.03506, 0.010903, 0.008226)
missing features 19 (0.529474, 0.147871, 0.181384, 0.069824) complete (0.283195, 0.011467, 0.008397, 0.007928)
horizon RMSE [0.268593 0.32287  0.271932 0.324159 0.284201 0.275782 0.299124 0.317379
 0.400578 0.35948 ]
calibration 0 single spot max stress delta 0.00026
calibration 0 three EMI spikes max stress delta 5e-05
calibration 1 single spot max stress delta 9.2e-05
calibration 1 three EMI spikes max stress delta 9.2e-05
calibration 2 single spot max stress delta 0.000283
calibration 2 three EMI spikes max stress delta 7.3e-05
calibration 3 single spot max stress delta 3.6e-05
calibration 3 three EMI spikes max stress delta 3e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 7 — `execute_in_remote_docker` (`30917 ms` LLM)
- **Rationale**: Compare against the supplied public reference where available to distinguish observation outliers from prediction errors. Independently check exact subyield weight gating and the hold-only and history-only terms; a failure would identify which part of the constitutive basis to fix.
- **Alternatives Weighed**: `Replace prescribed imputation with specimen-level onset correction` (rejected: The noisy held-lab residuals do not establish a constitutive error, and the contract prescribes cell-balanced feature imputation.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m96_w02_batch_04_solver.py'
sp=importlib.util.spec_from_file_location('coupon',p); s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; tr,te=d['train'],d['test']; out=s.solve_alpha_m96_w02_batch_04(d)
gt=item['gt']; print('public reference fields',[(k,np.shape(v)) for k,v in gt.items()] if isinstance(gt,dict) else type(gt).__name__)
cal=s._calibrate(d['calibration_load_cells']); x=s._align(tr,te); n=len(tr['lab_id']); des=s._Design(d,te,x[n:],cal); h=des.hist!=0; o=des.base[:,-1,7]!=0
if isinstance(gt,dict):
 for k,v in out.items():
  if k not in gt or np.shape(gt[k])!=v.shape: continue
  r=v-np.asarray(gt[k]); print('public reference',k,'rmse',np.sqrt(np.mean(r*r)),'max',np.max(abs(r)))
  if k in ('predicted_strain','initial_rate_pred'):
   for label,m in [('subyield',~h&~o),('history-only',h&~o),('hold-active',o),('missing features',np.isnan(te['coupon_features_raw']).any(axis=1))]:
    print(' ',label,int(m.sum()),np.sqrt(np.mean(r[m]**2)),np.max(abs(r[m])))
# Turn off plasticity everywhere to check exact parameter gating.
off=dict(d,yield_stress=1e6); z=s.solve_alpha_m96_w02_batch_04(off)
assert np.array_equal(z['viscoplastic_weights'],np.zeros(2))
# Exercise a hold-only boundary and an unloaded prior-overstress boundary independently.
tau=out['estimated_taus']; sy=d['yield_stress']; past=np.zeros((2,len(d['past_dt']))); past[1,0]=-2*sy; hold=np.array([-2*sy,sy/3]); a=s._Design(d,dict(lab_id=np.zeros(2,dtype=int),past_stress_cmd=past,stress_hold_cmd=hold),np.zeros((2,4)),np.tile([1.,0.],(4,1))).make(tau)
hist=sy**2*d['past_dt'][0]*hold[1]
assert np.allclose(a[0,:-1,7],-sy**2*d['eval_times']) and np.isclose(a[0,-1,7],-sy**2) and np.all(a[0,:,8]==0)
assert np.all(a[1,:,7]==0) and np.allclose(a[1,:-1,8],hist*(-np.expm1(-d['eval_times']/tau[0]))) and np.isclose(a[1,-1,8],hist/tau[0])
print('subyield weight and both isolated plastic mechanisms passed')
PY
```
- **Container Exit Code**: `0` (`662 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public reference fields [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
subyield weight and both isolated plastic mechanisms passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 8 — `execute_in_remote_docker` (`22792 ms` LLM)
- **Rationale**: The public reference uses different keys from the solver, so the previous check verified the mechanism boundaries but did not compare prediction accuracy. Map all five outputs explicitly and separate missing-feature from complete-feature errors to determine whether calibration, alignment, or fitting needs adjustment.

```bash
sed -n '27,53p' /tmp/hegel_continual_archive/alpha_m96_w02_batch_03_solver.py; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m96_w02_batch_04_solver.py'
sp=importlib.util.spec_from_file_location('coupon',p); s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb') as f: item=pickle.load(f)[0]
d,gt=item['dataset'],item['gt']; tr,te=d['train'],d['test']; out=s.solve_alpha_m96_w02_batch_04(d)
cal=s._calibrate(d['calibration_load_cells']); x=s._align(tr,te); des=s._Design(d,te,x[len(tr['lab_id']):],cal); h=des.hist!=0; o=des.base[:,-1,7]!=0
missing=np.isnan(te['coupon_features_raw']).any(axis=1)
for key,ref in zip(('estimated_taus','hereditary_weights','viscoplastic_weights','predicted_strain','initial_rate_pred'),('taus','theta','vp_theta','te_clean','te_true_rate0')):
 r=out[key]-gt[ref]; print(key,'RMSE',np.sqrt(np.mean(r*r)),'max',np.max(abs(r)))
 if r.ndim==1 and r.size<10: print('  parameters',gt[ref], 'error',r)
 if r.shape[0]==len(h):
  for name,m in [('subyield',~h&~o),('history-only',h&~o),('hold-active',o),('missing',missing),('complete',~missing)]:
   if m.any(): print(' ',name,int(m.sum()),'RMSE',np.sqrt(np.mean(r[m]**2)),'max',np.max(abs(r[m])))
for key,a in [('te_hold_true',des.hold),('te_hist_true',des.past)]: print(key,'max calibrated stress error',np.max(abs(a-gt[key])))
PY
```
- **Container Exit Code**: `0` (`595 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
features_raw'], test['coupon_features_raw'])).astype(float)
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0))
        x[use] = (x[use] - offset) / gain
    # Give each observed lab/stratum cell equal row mass before excluding missing values.
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[cell, None]
    valid = np.isfinite(x)
    mass = np.sum(w * valid, axis=0)
    fill = np.divide(np.sum(w * np.where(valid, x, 0.), axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0.)
    return np.where(valid, x, fill)


estimated_taus RMSE 0.05196297940728185 max 0.07348436900139221
  parameters [0.552 3.791] error [-0.00059158 -0.07348437]
hereditary_weights RMSE 0.003025565446509047 max 0.0061910748849609565
  parameters [ 0.712  0.528  0.08   0.15  -0.13   0.11  -0.09 ] error [-6.19107488e-03  1.68651731e-03  7.65338628e-05 -3.03220916e-03
  2.41392133e-03 -2.07046219e-03  1.89488664e-03]
viscoplastic_weights RMSE 0.0003434297923396569 max 0.00048240094760515495
  parameters [0.25 0.22] error [ 4.82400948e-04 -5.63681673e-05]
predicted_strain RMSE 0.1525718872658057 max 1.45242042966969
  subyield 20 RMSE 0.10994815098754335 max 0.8125274194155697
  history-only 20 RMSE 0.13837207559890627 max 1.02139492100356
  hold-active 40 RMSE 0.17589385424776763 max 1.45242042966969
  missing 6 RMSE 0.5569462880490875 max 1.45242042966969
  complete 74 RMSE 0.0038894453124196076 max 0.014523390184856155
initial_rate_pred RMSE 0.04854343904376356 max 0.28013319993740815
  subyield 20 RMSE 0.03461209734797053 max 0.15253225994447694
  history-only 20 RMSE 0.04440047764642861 max 0.19666488753080436
  hold-active 40 RMSE 0.0559305918023668 max 0.28013319993740815
  missing 6 RMSE 0.17635632316009947 max 0.28013319993740815
  complete 74 RMSE 0.005077669214648127 max 0.010916620792346343
te_hold_true max calibrated stress error 0.00010294572025237869
te_hist_true max calibrated stress error 0.00014286495509896469

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 9 — `execute_in_remote_docker` (`24176 ms` LLM)
- **Rationale**: Check whether the test inputs contain any independent onset measurement and whether imputed training rows bias the shared coefficients. If excluding those rows improves the public diagnostic, test the change across held-lab folds before adopting it; keep the prescribed inference-time imputation.
- **Alternatives Weighed**: `Change the constitutive basis` (rejected: Both plastic boundary checks pass, and clean-reference strain RMSE is only 0.00389 on complete specimens.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('coupon','/holdout_task/alpha_m96_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb'))[0]; d,gt=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=s._align(tr,te); cal=s._calibrate(d['calibration_load_cells']); print('test fields:',[(k,np.shape(v)) for k,v in te.items()]); miss=np.isnan(tr['coupon_features_raw']).any(axis=1); design=s._Design(d,te,x[n:],cal)
for name,m in [('all',np.ones(n,bool)),('complete',~miss)]:
 sub={k:v[m] for k,v in tr.items()}; tau,w=s._fit(s._Design(d,sub,x[:n][m],cal),sub); q=design.make(tau)@w; err=q[:,:-1]+te['strain_0'][:,None]-gt['te_clean']; dr=q[:,-1]-gt['te_true_rate0']; complete=np.isfinite(te['coupon_features_raw']).all(axis=1); print(name,'taus',tau,'max hereditary error',max(abs(w[:7]-gt['theta'])),'max plastic error',max(abs(w[7:]-gt['vp_theta'])),'complete strain/rate RMSE',np.sqrt(np.mean(err[complete]**2)),np.sqrt(np.mean(dr[complete]**2)))
PY
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
test fields: [('lab_id', (80,)), ('stratum', (80,)), ('coupon_features_raw', (80, 4)), ('ref_coupons', (80, 32)), ('past_stress_cmd', (80, 18)), ('strain_0', (80,)), ('stress_hold_cmd', (80,)), ('instant_rate_obs', (80,)), ('residual_plastic_state', (80,))]
all taus [0.55140842 3.71751563] max hereditary error 0.0061910748849609565 max plastic error 0.00048240094760515495 complete strain/rate RMSE 0.0038894453124196076 0.005077669214648127
complete taus [0.55201615 3.80241768] max hereditary error 0.003268364036461857 max plastic error 0.0005356291297680116 complete strain/rate RMSE 0.003712430255981522 0.0012746940688866649

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 10 — `execute_in_remote_docker` (`28124 ms` LLM)
- **Rationale**: Use held-lab strain curves, not the onset values used for the correction, to test whether onset assimilation recovers the missing drift. Compare complete-case fitting in the same folds; retain either change only if it improves the intended slice.
- **Alternatives Weighed**: `Replace balanced feature imputation` (rejected: The contract specifies that alignment. The measured onset instead provides an independent observation of the remaining rate offset, which contributes linearly to strain.)

```bash
cat > /tmp/coupon04_onset.py <<'PY'
exec(open('/tmp/coupon04_validate.py').read().split('res = np.empty_like(y)')[0])
miss = np.isnan(tr['coupon_features_raw']).any(axis=1)
print('Test onset finite',np.isfinite(te['instant_rate_obs']).sum(), 'test missing with onset',np.sum(np.isnan(te['coupon_features_raw']).any(axis=1)&np.isfinite(te['instant_rate_obs'])))
for c in range(4):
    learn = lab != c
    a = {k:v[learn] for k,v in tr.items()}
    tau0,w0 = s._fit(s._Design(d,a,x[learn],cal),a)
    fit = learn & ~miss
    b = {k:v[fit] for k,v in tr.items()}
    tau,w = s._fit(s._Design(d,b,x[fit],cal),b)
    prior = full.make(tau0) @ w0
    pred = full.make(tau) @ w
    delta = y[:,-1] - pred[:,-1]
    delta = np.where(miss & np.isfinite(delta),delta,0.)
    corrections = [np.column_stack((q[:,None]*full.t, q)) for q in (delta, np.clip(delta,-.35,.35), .8*np.clip(delta,-.35,.35))]
    print('lab',c,'complete-only fit',tau,'missing onset deltas',np.round(delta[(lab==c)&miss],4))
    for name, v in [('original',prior),('complete-fit',pred),*[(f'correction {i}',pred+z) for i,z in enumerate(corrections)]]:
        for m in ((lab==c)&miss, (lab==c)&~miss):
            err = (v-y)[m,:-1]
            print(name,len(err),np.round([np.sqrt(np.nanmean(err**2)), np.nanmedian(abs(err)), np.nanmedian(abs(err[:,-1]))],6))
item = pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb'))[0]
ref = item['gt']; mt = np.isnan(te['coupon_features_raw']).any(axis=1)
print('test onset reference error, missing rows',np.round((te['instant_rate_obs']-ref['te_true_rate0'])[mt],6),'complete RMSE',np.sqrt(np.mean((te['instant_rate_obs'][~mt]-ref['te_true_rate0'][~mt])**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/coupon04_onset.py
```
- **Container Exit Code**: `0` (`901 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.0407 -0.085   0.2486 -0.1544]
original 5 [0.512925 0.259326 0.371023]
original 46 [0.263675 0.007932 0.011205]
complete-fit 5 [0.51311  0.262414 0.370997]
complete-fit 46 [0.263546 0.007218 0.011435]
correction 0 5 [0.322187 0.031301 0.043436]
correction 0 46 [0.263546 0.007218 0.011435]
correction 1 5 [0.322187 0.031301 0.043436]
correction 1 46 [0.263546 0.007218 0.011435]
correction 2 5 [0.322507 0.033954 0.036433]
correction 2 46 [0.263546 0.007218 0.011435]
lab 1 complete-only fit [0.55261264 3.77343095] missing onset deltas [ 0.2155 -0.0366  0.0101 -0.3956]
original 4 [0.753738 0.263044 0.97975 ]
original 51 [0.280859 0.007728 0.010075]
complete-fit 4 [0.754596 0.264292 0.980202]
complete-fit 51 [0.280732 0.007401 0.010369]
correction 0 4 [0.299137 0.017599 0.04666 ]
correction 0 51 [0.280732 0.007401 0.010369]
correction 1 4 [0.304909 0.032439 0.116877]
correction 1 51 [0.280732 0.007401 0.010369]
correction 2 4 [0.347324 0.088448 0.392216]
correction 2 51 [0.280732 0.007401 0.010369]
lab 2 complete-only fit [0.55119162 3.80401293] missing onset deltas [ 0.048   0.1152 -0.1822 -0.1625  0.0039 -0.064 ]
original 6 [0.434388 0.215617 0.504636]
original 39 [0.31172  0.008083 0.008798]
complete-fit 6 [0.434324 0.217959 0.504226]
complete-fit 39 [0.311661 0.007855 0.009317]
correction 0 6 [0.192458 0.012687 0.035625]
correction 0 39 [0.311661 0.007855 0.009317]
correction 1 6 [0.192458 0.012687 0.035625]
correction 1 39 [0.311661 0.007855 0.009317]
correction 2 6 [0.215378 0.056651 0.141248]
correction 2 39 [0.311661 0.007855 0.009317]
lab 3 complete-only fit [0.55203043 3.82920435] missing onset deltas [-0.0108  0.1814  0.0015 -0.0401]
original 4 [0.389362 0.046555 0.087476]
original 45 [0.279103 0.010271 0.014266]
complete-fit 4 [0.390808 0.044081 0.091367]
complete-fit 45 [0.27915  0.009914 0.012425]
correction 0 4 [0.151885 0.019313 0.052334]
correction 0 45 [0.27915  0.009914 0.012425]
correction 1 4 [0.151885 0.019313 0.052334]
correction 1 45 [0.27915  0.009914 0.012425]
correction 2 4 [0.187761 0.017709 0.051188]
correction 2 45 [0.27915  0.009914 0.012425]
test onset reference error, missing rows [-0.005423  0.014617  0.004413 -0.00651  -0.006979  0.000502] complete RMSE 0.01049258039493343

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e6ce1bab5581a5fb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e6ce1bab (directory).
```

### Turn 11 — `execute_in_remote_docker` (`46042 ms` LLM)
- **Rationale**: The held-lab strain curves independently validate using measured onset to recover the missing linear drift. Apply the correction only to incomplete specimens, then verify that it reduces public-reference errors without degrading complete specimens or changing either plastic mechanism.
- **Alternatives Weighed**: `Clip or shrink the onset correction` (rejected: The full correction lowered held-lab strain RMSE in all four labs and matched or beat the clipped and shrunk variants.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m96_w02_batch_04_solver.py')
s = p.read_text()
s = s.replace('    curves = present.copy()\n    curves[:, -1] = False\n', '')
old = '''    tau, weights = _fit(_Design(dataset, train, x[:n], cal), train)
    pred = _Design(dataset, test, x[n:], cal).make(tau) @ weights
'''
new = '''    # Missing covariates have an unknown coupon-specific drift. Fit the shared
    # coefficients on observed covariates when they span the full design.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    design = _Design(dataset, train, x[:n], cal)
    if not complete.all() and complete.any():
        subset = {k: v[complete] for k, v in train.items()}
        clean = _Design(dataset, subset, x[:n][complete], cal)
        cols = np.r_[np.ones(7, dtype=bool), design.active]
        observed = np.isfinite(np.column_stack((subset['strain_curves_obs'],
                                               subset['instant_rate_obs'])))
        if np.linalg.matrix_rank(clean.make((.55, 3.8))[observed][:, cols]) == cols.sum():
            train, design = subset, clean
    tau, weights = _fit(design, train)
    test_design = _Design(dataset, test, x[n:], cal)
    pred = test_design.make(tau) @ weights

    # The imputed features supply the prior drift. If covariates are missing,
    # an observed onset identifies its remaining linear-in-time contribution.
    onset = np.asarray(test.get('instant_rate_obs', np.full(len(pred), np.nan)))
    update = (~np.isfinite(test['coupon_features_raw']).all(axis=1)
              & np.isfinite(onset))
    delta = onset[update] - pred[update, -1]
    pred[update, :-1] += delta[:, None] * test_design.t
    pred[update, -1] += delta
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
path = '/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('coupon', path)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
out = s.solve_alpha_m96_w02_batch_04(d)
x = s._align(tr, te)[len(tr['lab_id']):]
model = s._Design(d, te, x, s._calibrate(d['calibration_load_cells']))
h, o = model.hist != 0, model.base[:, -1, 7] != 0
m = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
e = out['predicted_strain'] - gt['te_clean']; r = out['initial_rate_pred'] - gt['te_true_rate0']
rmse = lambda q: np.sqrt(np.mean(q*q))
for name, mask in [('all', np.ones(len(h),bool)), ('missing',m), ('complete',~m), ('subyield',~h&~o), ('history-only',h&~o), ('hold-active',o), *[(f'lab {j}',te['lab_id']==j) for j in range(4)]]:
    print(name, int(mask.sum()), 'strain/onset RMSE', round(rmse(e[mask]),6),round(rmse(r[mask]),6))
for k, v in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
    print(k, 'max abs parameter error',np.max(abs(out[k]-gt[v])))
assert rmse(e[m]) < .06 and rmse(r[m]) < .03
assert rmse(e[~m]) < .005 and rmse(r[~m]) < .005
no_onset = dict(te); no_onset.pop('instant_rate_obs'); prior = s.solve_alpha_m96_w02_batch_04(dict(d,test=no_onset))
correction = out['initial_rate_pred'] - prior['initial_rate_pred']
np.testing.assert_allclose(out['predicted_strain'] - prior['predicted_strain'], correction[:, None]*model.t, atol=1e-12)
assert np.all(correction[~m] == 0)
assert np.array_equal(s.solve_alpha_m96_w02_batch_04(dict(d,yield_stress=1e6))['viscoplastic_weights'],np.zeros(2))
# Independently exercise signed hold-only and unloaded history-only terms.
sy = d['yield_stress']; past = np.zeros((2,len(model.dt))); past[1,0] = -2*sy
hold = np.array([-2*sy,sy/3]); tau = out['estimated_taus']
a = s._Design(d,dict(lab_id=np.zeros(2,int),past_stress_cmd=past,stress_hold_cmd=hold),np.zeros((2,4)),np.tile([1.,0.],(4,1))).make(tau)
np.testing.assert_allclose(a[0,:,7],-sy**2*np.r_[model.t,1]); assert not np.any(a[0,:,8])
np.testing.assert_allclose(a[1,:,8],sy**2*model.dt[0]*hold[1]*np.r_[-np.expm1(-model.t/tau[0]),1/tau[0]]); assert not np.any(a[1,:,7])
print('Onset correction, subyield gating, and isolated plastic terms verified')
PY
```
- **Container Exit Code**: `0` (`1068 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
all 80 strain/onset RMSE 0.007753 0.002434
missing 6 strain/onset RMSE 0.025131 0.00768
complete 74 strain/onset RMSE 0.003712 0.001275
subyield 20 strain/onset RMSE 0.003326 0.001173
history-only 20 strain/onset RMSE 0.00618 0.002037
hold-active 40 strain/onset RMSE 0.009778 0.003015
lab 0 18 strain/onset RMSE 0.007791 0.002277
lab 1 18 strain/onset RMSE 0.00664 0.002136
lab 2 24 strain/onset RMSE 0.003171 0.001109
lab 3 20 strain/onset RMSE 0.01158 0.003669
estimated_taus max abs parameter error 0.011417675872498645
hereditary_weights max abs parameter error 0.003268364036461857
viscoplastic_weights max abs parameter error 0.0005356291297680116
Onset correction, subyield gating, and isolated plastic terms verified

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b98dc9b48e16e04d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b98dc9b4 (directory).
```

### Turn 12 — `submit_completed_task` (`12544 ms` LLM)
- **Rationale**: The final artifact passed clean-process validation. The onset-informed correction reduced missing-feature strain error in all four held-lab folds, and the final verification confirmed exact subyield gating and both isolated plastic mechanisms.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m96_w02_batch_04_solver.py': 'FILE:10137'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m96_w02_batch_04_solver.py`. The public schema passed. Public-reference RMSE was 0.00775 for strain and 0.00243 for initial rate. The subyield, hold-only, and prior-overstress checks passed.`

### Turn 13 — `execute_in_remote_docker` (`37374 ms` LLM)
- **Rationale**: Hold out each entire wing to test whether shared parameters transfer and whether onset correction improves strain observations independently of the onset used for correction. Separately perturb one of two real calibration spots: if the resulting stress error exceeds 0.002, revise the sparse-sweep fallback rather than the constitutive law.
- **Alternatives Weighed**: `Search unrelated rational, stochastic-drift, or relay bases` (rejected: The coupon contract specifies a deterministic Prony and plastic-history basis; testing parameter transfer across wings directly probes whether that basis leaves a systematic residual.)

```bash
cat > /tmp/coupon04_margin.py <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('coupon', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
cal = s._calibrate(d['calibration_load_cells'])
worst = 0.
for wing in (0, 1):
    held = tr['lab_id'] // 2 == wing
    train = {k: v[~held] for k, v in tr.items()}
    test = {k: v[held] for k, v in tr.items()}
    without = {k: v for k, v in test.items() if k != 'instant_rate_obs'}
    obs = test['strain_curves_obs']
    out = s.solve_alpha_m96_w02_batch_04(dict(d, train=train, test=test))
    prior = s.solve_alpha_m96_w02_batch_04(dict(d, train=train, test=without))
    a = s._Design(d, test, s._align(train, test)[len(train['lab_id']):], cal)
    h, o = a.hist != 0, a.base[:, -1, 7] != 0
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    print('held wing', wing, 'taus', np.round(out['estimated_taus'], 5), 'plastic', np.round(out['viscoplastic_weights'], 5))
    for name, m in [('subyield',~h&~o),('history-only',h&~o),('hold-active',o),('missing',missing),('complete',~missing)]:
        if not m.any(): continue
        err = (out['predicted_strain'] - obs)[m]
        base = (prior['predicted_strain'] - obs)[m]
        rmse = np.sqrt(np.nanmean(err**2))
        worst = max(worst, rmse)
        print(name, m.sum(), 'prior/new RMSE', np.round([np.sqrt(np.nanmean(base**2)),rmse],6), 'median abs', round(np.nanmedian(abs(err)),6), 'late signed median', round(np.nanmedian(err[:,-1]),6))
print('worst held-wing strain RMSE', round(worst,6))
checks = d['calibration_load_cells']
for lab in range(4):
    own = np.flatnonzero(checks['lab_id'] == lab)
    order = own[np.argsort(checks['cmd_stress'][own])]
    pair = order[[0,-1]]
    keep = (checks['lab_id'] != lab) | np.isin(np.arange(len(checks['lab_id'])), pair)
    for idx in pair:
        for jump in (-.27,.27):
            c = {k:v[keep].copy() for k,v in checks.items()}
            c['true_stress'][np.count_nonzero(keep[:idx])] += jump
            g,b = s._calibrate(c)[lab] - cal[lab]
            delta = max(abs(b-3*g),abs(b+3*g))
            if delta >= .002:
                print('sparse + EMI', lab, 'spot',round(checks['cmd_stress'][idx],4),'spike',jump,'max stress delta',round(delta,6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/coupon04_margin.py
```
- **Container Exit Code**: `0` (`854 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held wing 0 taus [0.55444 3.75963] plastic [0.24926 0.22058]
subyield 30 prior/new RMSE [0.368553 0.254946] median abs 0.008734 late signed median -0.008316
history-only 30 prior/new RMSE [0.287418 0.280753] median abs 0.009268 late signed median -0.004512
hold-active 46 prior/new RMSE [0.303471 0.287786] median abs 0.0085 late signed median 0.001034
missing 9 prior/new RMSE [0.628958 0.315358] median abs 0.023603 late signed median -0.016283
complete 97 prior/new RMSE [0.272999 0.272999] median abs 0.008281 late signed median -0.001278
held wing 1 taus [0.55111 3.84763] plastic [0.24883 0.21979]
subyield 20 prior/new RMSE [0.309351 0.304648] median abs 0.010842 late signed median 0.002679
history-only 20 prior/new RMSE [0.276875 0.276844] median abs 0.010223 late signed median 0.005379
hold-active 54 prior/new RMSE [0.324625 0.279771] median abs 0.010248 late signed median -0.00044
missing 10 prior/new RMSE [0.429328 0.179231] median abs 0.016075 late signed median -0.03013
complete 84 prior/new RMSE [0.294685 0.294685] median abs 0.00989 late signed median 0.001716
worst held-wing strain RMSE 0.315358
sparse + EMI 0 spot -2.1374 spike -0.27 max stress delta 0.325475
sparse + EMI 0 spot -2.1374 spike 0.27 max stress delta 0.325476
sparse + EMI 0 spot 2.0608 spike -0.27 max stress delta 0.330507
sparse + EMI 0 spot 2.0608 spike 0.27 max stress delta 0.330301
sparse + EMI 1 spot -2.1463 spike -0.27 max stress delta 0.32262
sparse + EMI 1 spot -2.1463 spike 0.27 max stress delta 0.322748
sparse + EMI 1 spot 2.2286 spike -0.27 max stress delta 0.317364
sparse + EMI 1 spot 2.2286 spike 0.27 max stress delta 0.317856
sparse + EMI 2 spot -1.6894 spike -0.27 max stress delta 0.367662
sparse + EMI 2 spot -1.6894 spike 0.27 max stress delta 0.367454
sparse + EMI 2 spot 1.9377 spike -0.27 max stress delta 0.3491
sparse + EMI 2 spot 1.9377 spike 0.27 max stress delta 0.349061
sparse + EMI 3 spot -2.1986 spike -0.27 max stress delta 0.319092
sparse + EMI 3 spot -2.1986 spike 0.27 max stress delta 0.319233
sparse + EMI 3 spot 2.2029 spike -0.27 max stress delta 0.318808
sparse + EMI 3 spot 2.2029 spike 0.27 max stress delta 0.318979

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b98dc9b48e16e04d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_b98dc9b4 (directory).
```

### Turn 14 — `execute_in_remote_docker` (`36300 ms` LLM)
- **Rationale**: Keep the archived gain when the current sweep cannot independently support an affine fit, and use only archive-consistent checks to update the intercept. Rerun the failed sparse-plus-EMI cases and verify that the change preserves public accuracy and transfer across wings.
- **Alternatives Weighed**: `Refit gain from two widely spaced checks` (rejected: Two points give an exact line even when one is contaminated; wider spacing did not prevent the observed errors.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m96_w02_batch_04_solver.py')
s = p.read_text()
a = s.index('    if prior is not None:\n        if len(u) >= 2')
b = s.index('    if len(u) >= 2 and np.ptp(u) >= .18:', a)
s = s[:a] + '''    if prior is not None:
        # With fewer than three well-spaced inliers, a pair can fit exactly
        # even if one check is an EMI spike. Keep the archived gain and
        # update the intercept only from checks consistent with that archive.
        offset = y - prior[0] * x
        supported = np.abs(offset - prior[1]) <= .003
        if np.any(supported):
            return np.array([prior[0], np.median(offset[supported])])
        return prior
''' + s[b:]
p.write_text(s)
q = Path('/tmp/coupon04_margin.py')
t = q.read_text(); t = t[:t.index('checks = d[')] + t[t.index('checks = d['):].replace('checks = d[', 'worst_stress = 0.\nchecks = d[', 1).replace("            if delta >= .002:\n                print('sparse + EMI', lab, 'spot',round(checks['cmd_stress'][idx],4),'spike',jump,'max stress delta',round(delta,6))", "            worst_stress = max(worst_stress, delta)\nprint('worst of 16 sparse + EMI cases', round(worst_stress, 6))\nassert worst_stress < .002")
assert 'assert worst_stress < .002' in t
q.write_text(t)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/coupon04_validate.py
python3 -I /tmp/coupon04_margin.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
path='/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec=importlib.util.spec_from_file_location('coupon',path)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb') as f: p=pickle.load(f)[0]
d,g=p['dataset'],p['gt']; o=s.solve_alpha_m96_w02_batch_04(d)
for name,ref,limit in [('estimated_taus','taus',.02),('hereditary_weights','theta',.005),('viscoplastic_weights','vp_theta',.001)]:
 err=np.max(np.abs(o[name]-g[ref])); print(name,'max error',err); assert err<limit
te=d['test']; missing=~np.isfinite(te['coupon_features_raw']).all(axis=1)
for label,m,e_lim,r_lim in [('overall',np.ones(len(missing),bool),.008,.003),('missing',missing,.03,.01),('complete',~missing,.005,.005)]:
 e=np.sqrt(np.mean((o['predicted_strain'][m]-g['te_clean'][m])**2)); r=np.sqrt(np.mean((o['initial_rate_pred'][m]-g['te_true_rate0'][m])**2)); print(label,'strain/onset RMSE',e,r); assert e<e_lim and r<r_lim
assert np.array_equal(s.solve_alpha_m96_w02_batch_04(dict(d,yield_stress=1e6))['viscoplastic_weights'],np.zeros(2))
PY
```
- **Container Exit Code**: `0` (`2011 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m96_w02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
.008677, 0.012748, 0.005992)
3 hold-active 28 (0.310363, 0.03506, 0.010903, 0.008226)
missing features 19 (0.529474, 0.147871, 0.181384, 0.069824) complete (0.283195, 0.011467, 0.008397, 0.007928)
horizon RMSE [0.268593 0.32287  0.271932 0.324159 0.284201 0.275782 0.299124 0.317379
 0.400578 0.35948 ]
calibration 0 single spot max stress delta 0.00026
calibration 0 three EMI spikes max stress delta 5e-05
calibration 1 single spot max stress delta 9.2e-05
calibration 1 three EMI spikes max stress delta 9.2e-05
calibration 2 single spot max stress delta 0.000283
calibration 2 three EMI spikes max stress delta 7.3e-05
calibration 3 single spot max stress delta 3.6e-05
calibration 3 three EMI spikes max stress delta 3e-05
held wing 0 taus [0.55444 3.75963] plastic [0.24926 0.22058]
subyield 30 prior/new RMSE [0.368553 0.254946] median abs 0.008734 late signed median -0.008316
history-only 30 prior/new RMSE [0.287418 0.280753] median abs 0.009268 late signed median -0.004512
hold-active 46 prior/new RMSE [0.303471 0.287786] median abs 0.0085 late signed median 0.001034
missing 9 prior/new RMSE [0.628958 0.315358] median abs 0.023603 late signed median -0.016283
complete 97 prior/new RMSE [0.272999 0.272999] median abs 0.008281 late signed median -0.001278
held wing 1 taus [0.55111 3.84763] plastic [0.24883 0.21979]
subyield 20 prior/new RMSE [0.309351 0.304648] median abs 0.010842 late signed median 0.002679
history-only 20 prior/new RMSE [0.276875 0.276844] median abs 0.010223 late signed median 0.005379
hold-active 54 prior/new RMSE [0.324625 0.279771] median abs 0.010248 late signed median -0.00044
missing 10 prior/new RMSE [0.429328 0.179231] median abs 0.016075 late signed median -0.03013
complete 84 prior/new RMSE [0.294685 0.294685] median abs 0.00989 late signed median 0.001716
worst held-wing strain RMSE 0.315358
worst of 16 sparse + EMI cases 0.000283
estimated_taus max error 0.011417675872498645
hereditary_weights max error 0.003268364036461857
viscoplastic_weights max error 0.0005356291297680116
overall strain/onset RMSE 0.007753431858989233 0.002434424639867901
missing strain/onset RMSE 0.025130917894597947 0.007679793166169021
complete strain/onset RMSE 0.003712430255981522 0.0012746940688866649

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f7959e0f5ef3759 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_7f7959e0 (directory).
```

### Turn 15 — `submit_completed_task` (`13491 ms` LLM)
- **Rationale**: The final artifact passed isolated Python schema and accuracy checks. The sparse-sweep fix closed the observed calibration failure without degrading public-reference accuracy or held-wing transfer, and exact subyield gating still passed.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m96_w02_batch_04_solver.py`. The public schema passed. Public-reference RMSE was 0.00775 for strain and 0.00243 for initial rate. The calibration fix reduced the worst stress deviation across 16 sparse, EMI-contaminated checks from 0.368 to 0.000283 without changing public prediction accuracy.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M96-Autonomous-Structure-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
