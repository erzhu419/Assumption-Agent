# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `324,561` (`312,741` / `69,760` / `11,820`) |
| **Total Episode Wall Time (s)** | `393.66s` |

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

### Turn 1 — `execute_in_remote_docker` (`14660 ms` LLM)
- **Rationale**: Identify the dataset layout and validator contract before implementing, so incorrect field mappings or unavailable calibration inputs change the implementation rather than silently contaminating the fit.

```bash
cd /holdout_task && ls -l && cat public_asset_index.json validate_public_schema.py && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle
with open('public_alpha_m97_c02_batch_01.pkl','rb') as f: d=pickle.load(f)
def describe(x, depth=0):
 if isinstance(x,dict) and depth<2:
  for k,v in x.items():
   print(' '*depth,k,end=': '); describe(v,depth+1)
 elif hasattr(x,'shape'): print(type(x).__name__,x.shape,x.dtype)
 elif isinstance(x,(list,tuple)): print(type(x).__name__,len(x),str(x[0])[:250] if len(x) else '')
 else: print(str(x)[:250])
describe(d)
PY
```
- **Container Exit Code**: `0` (`616 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"dtype": "float64",
              "shape": [
                72
              ],
              "type": "ndarray"
            },
            "te_true_rate0": {
              "dtype": "float64",
              "shape": [
                72
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
      "bytes": 1111
    }
  },
  "schema": "hegel_public_asset_index_v1"
}import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "alpha_m97_c02_batch_01_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_alpha_m97_c02_batch_01.pkl"), "rb"))[0]
ds = item["dataset"]
out = mod.solve_alpha_m97_c02_batch_01(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("regime_mode_weights", (2,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
total 20
-rw-r--r-- 1 root root 7423 Nov 27 06:46 alpha_m97_c02_s00_solver.py
-r--r--r-- 1 root root 1793 Nov 27 06:46 archive_calibration_history.json
-rw-r--r-- 1 root root  462 Nov 27 06:46 archive_manifest.json
-rw-r--r-- 1 root root  429 Nov 27 06:46 pre_llm_gate_diagnostic.json
list 1 {'dataset': {'past_dt': array([0.6862, 0.3536, 0.4115, 0.5101, 0.4903, 0.7741, 0.4359, 0.5228,
       0.5538, 0.2493, 0.5622, 0.3497, 0.6793, 0.3183, 0.7778, 0.7773,
       0.754 , 0.2048]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778
```

### Turn 2 — `execute_in_remote_docker` (`16138 ms` LLM)
- **Rationale**: Check the archived panel layout and available training observations before fitting; a wrong panel reshape or a missing rate label would change the preprocessing or objective.

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation' || true; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
for k,v in D.items():
 if isinstance(v,dict):
  print(k, [(a,np.shape(b)) for a,b in v.items()]);
  if 'ref_coupons' in v: print('ref examples', v['ref_coupons'][:2], 'missing raw', np.isnan(v['coupon_features_raw']).sum(axis=0))
 else: print(k,v)
PY
```
- **Container Exit Code**: `0` (`640 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.7741 0.4359 0.5228 0.5538 0.2493
 0.5622 0.3497 0.6793 0.3183 0.7778 0.7773 0.754  0.2048]
eval_times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
crit_stress 2.65
yield_stress 2.65
shear_banding_index 0.12
calibration_load_cells [('lab_id', (40,)), ('cmd_stress', (40,)), ('true_stress', (40,))]
train [('lab_id', (180,)), ('stratum', (180,)), ('coupon_features_raw', (180, 4)), ('ref_coupons', (180, 32)), ('past_stress_cmd', (180, 18)), ('stress_hold_cmd', (180,)), ('strain_0', (180,)), ('aux_chamber_telemetry', (180, 3)), ('strain_curves_obs', (180, 10)), ('instant_rate_obs', (180,))]
ref examples [[ 0.25229101 -0.61031303 -1.75209797 -2.01246643  0.31465362  0.18192417
  -0.08853356 -0.46327493 -0.76053971 -0.97595278 -0.81320085 -0.16085888
  -0.7270434          nan -0.49396558 -0.81637355 -1.69819377  0.53757301
   0.5876544          nan -1.26658519 -1.60204875 -0.07785045 -1.60127165
  -0.89755477 -0.47875201 -0.76680671 -0.34010083  0.24423529  0.44790027
  -0.02502198  1.03527839]
 [ 1.62609958  2.66549873  2.0099448  -0.9507348   0.66042867 -0.48756258
   1.20824424 -0.44985318  2.07351417 -1.71934612  0.14478814  2.03377857
   0.16094029  3.42905047  0.97719508 -0.99161485  1.94306308 -0.02254746
   1.38170666  2.04541755  1.20514714  1.91361873 -0.00420489  0.09969053
   1.25008817  1.60107641 -0.53626641  3.04807427 -1.29786865  0.43507265
   0.19248326 -0.82106938]] missing raw [3 9 3 5]
test [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 32)), ('past_stress_cmd', (72, 18)), ('stress_hold_cmd', (72,)), ('strain_0', (72,)), ('aux_chamber_telemetry', (72, 3))]
ref examples [[-0.34049248 -1.30066948 -0.34558436  0.72083732 -1.53314778 -1.38968563
  -0.51756789 -0.91320917 -0.96263308 -1.01544027 -1.25135383 -1.71937243
   0.19927979 -0.48038076 -1.22008985 -0.0922118  -1.00932121 -1.72233421
  -1.12751347 -0.10133167  0.05663992 -0.94901491  1.37125767 -1.44360729
  -0.49865214 -0.56909193 -0.84612505 -0.08730568 -1.4616715  -1.71610666
   1.05148582 -0.0533517 ]
 [ 1.93815054 -0.16202596  2.03233903  1.13816982         nan  1.05320602
  -0.55832545  0.80373599  1.09453892  3.13325474  1.76493621  0.721382
  -0.01965817  0.88621554 -0.59731544 -1.76068639  1.70767123 -1.41552962
   2.09060948  0.12681331  3.43313548  0.32844675  0.93231946  1.07775296
   1.91199424  1.12290811  1.93355929  0.61605825 -0.51938721  0.7821473
   2.5701184   2.13664448]] missing raw [1 0 2 1]
```

### Turn 3 — `execute_in_remote_docker` (`23111 ms` LLM)
- **Rationale**: Recover the panel reshape and imputation convention from the truncated archive listing. Calibration residuals determine whether the archive fallback is needed; label coverage and cell counts inform the fit and lab-held-out validation.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: Its linear terminal flow omits the specified logarithmic plateau and crystallization terms.)

```bash
sed -n '45,165p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; cat /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Prony variable projection' 2>/dev/null | cut -c1-210 | head -6; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel 2>/dev/null | head -18; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
d=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']
for lab in range(4):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('calibration',lab,len(x),p,'max residual',np.max(np.abs(y-np.polyval(p,x))))
for name in ('train','test'):
 s=d[name]; print(name,'lab/stratum counts',np.unique(np.c_[s['lab_id'],s['stratum']],axis=0,return_counts=True));
 if name=='train':
  for k in ('strain_curves_obs','instant_rate_obs'):
   y=s[k]; print(k,'nonfinite',np.sum(~np.isfinite(y)),'quantiles',np.nanquantile(y,[0,.01,.5,.99,1]))
PY
```
- **Container Exit Code**: `0` (`706 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins:
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite pre
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite de
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
calibration 0 10 [ 0.79004725 -0.13998828] max residual 0.00019010847293876587
calibration 1 10 [1.30996232 0.16005601] max residual 0.00019176057234515653
calibration 2 10 [0.84996132 0.10003695] max residual 0.00012464239328244253
calibration 3 10 [ 1.24000154 -0.12996203] max residual 0.00014402498836063593
train lab/stratum counts (array([[0, 0],
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
       [3, 2]]), array([19, 16,  6,  7, 20, 10, 17, 15, 20, 10, 15, 25]))
strain_curves_obs nonfinite 0 quantiles [-17.7810202   -2.81361827   0.21296175   3.48446996  25.32372712]
instant_rate_obs nonfinite 0 quantiles [-5.1110138  -4.54625931  0.30886069  4.45692555  4.849737  ]
test lab/stratum counts (array([[0, 0],
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
       [3, 2]]), array([6, 5, 6, 4, 8, 4, 4, 5, 7, 8, 6, 9]))
```

### Turn 4 — `execute_in_remote_docker` (`50529 ms` LLM)
- **Rationale**: Implement the specified basis and onset derivative with shared weights, preserving the archived panel convention. The public validator will catch import, shape, or nonfinite-output failures; the archive key inspection determines whether the sparse calibration path can read the supplied history.
- **Alternatives Weighed**: `Ordinary least squares for all curve points` (rejected: Observed curve tails are several times larger than the central range; robust profiling limits their influence while retaining the prescribed physical basis.)

```bash
set -e
cd /holdout_task
p=alpha_m97_c02_batch_01_solver.py
awk '/^def _basis\(/ {exit} {print}' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py > "$p"
sed -i '1c\"""Calibrated two-branch Prony model with a crystallization plateau."""' "$p"
cat >> "$p" <<'PY'
def _basis(taus, past, hold, features, dt, times, activity, plateau=True):
    # A stress increment is attenuated by all steps after its own step.
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gap = hold[:, None] - z
    rise = -np.expm1(-times[:, None] / taus)
    flow = np.where(activity[:, None] > 0, np.log1p(times), times) if plateau else np.broadcast_to(times, (len(hold), len(times)))
    cols = 9 if plateau else 7
    curves = np.empty((len(hold), len(times), cols))
    rates = np.empty((len(hold), cols))
    curves[:, :, :2] = gap[:, None, :] * rise
    rates[:, :2] = gap / taus
    curves[:, :, 2:7] = flow[:, :, None] * features[:, None, :]
    rates[:, 2:7] = features
    if plateau:
        extra = activity * hold
        network = activity * z[:, 0]
        curves[:, :, 7] = extra[:, None] * rise[:, 0]
        curves[:, :, 8] = network[:, None] * flow
        rates[:, 7] = extra / taus[0]
        rates[:, 8] = network
    return curves, rates


def _activity(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
    return np.tanh(excess**2 @ dt)


def _fit(past, hold, features, dt, times, activity, observed, rates, plateau=True):
    plateau = plateau and np.any(activity > 0)
    ncols = 9 if plateau else 7
    nc = observed.size
    y = np.r_[observed.ravel(), rates]
    mask = np.isfinite(y)
    is_rate = (np.arange(y.size) >= nc)[mask]
    y = y[mask]

    def project(log_tau, weight):
        curves, derivatives = _basis(np.exp(log_tau), past, hold, features, dt, times, activity, plateau)
        A = np.concatenate((curves.reshape(-1, ncols), derivatives))[mask]
        coef = np.linalg.lstsq(A * weight[:, None], y * weight, rcond=None)[0]
        return A @ coef - y, coef

    lo, hi = np.log([0.025, 150.0])
    weight = np.where(is_rate, 1.0, 1.0 / np.sqrt(len(times)))

    def optimize(start):
        return least_squares(lambda p: project(p, weight)[0] * weight, start,
                             bounds=(lo, hi), max_nfev=70)

    starts = [(0.35, 2.0), (0.7, 4.0), (1.4, 8.0)]
    init = [optimize(np.log(s)) for s in starts]
    fit = min(init, key=lambda r: r.cost)
    for _ in range(12):
        residual, _ = project(fit.x, weight)
        wnew = np.ones_like(weight)
        for m in (is_rate, ~is_rate):
            if not np.any(m):
                continue
            r = residual[m]
            scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-5)
            wnew[m] = 1.0 / np.hypot(2.385 * scale, r)
        wnew /= np.max(wnew)
        weight = wnew
        fit = optimize(fit.x)
    _, coef = project(fit.x, weight)
    tau = np.exp(fit.x)
    if not plateau:
        order = np.argsort(tau)
        tau = tau[order]
        coef[:2] = coef[order]
        coef = np.r_[coef, 0.0, 0.0]
    return tau, coef


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    activity = _activity(dataset, past, dt)
    tau, coef = _fit(past[:n], hold[:n], features[:n], dt, times, activity[:n],
                     train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs'])
    curves, rates = _basis(tau, past[n:], hold[n:], features[n:], dt, times, activity[n:])
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:], 'predicted_strain': strain0[n:, None] + curves @ coef,
            'initial_rate_pred': rates @ coef}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import json
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive structure:', {k:list(v) if isinstance(v,dict) else len(v) if isinstance(v,list) else v for k,v in h.items()})
PY
```
- **Container Exit Code**: `1` (`683 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b641f16cca61ce93 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b641f16c (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 5 — `execute_in_remote_docker` (`28021 ms` LLM)
- **Rationale**: Measure all five outputs against the public references and compare the plateau and linear-flow models on held-out labs. Concentrated errors would point to alignment, onset-rate fitting, or regime gating. The archive probe also checks whether the solver can use the documented sparse-calibration fallback; the prior inspection failed because it treated the history list as a dictionary.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
p='alpha_m97_c02_batch_01_solver.py'; s=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c02_batch_01(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k, 'estimated',o[k], 'reference',g[j], 'max abs error',np.max(np.abs(o[k]-g[j])))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive rows:',len(h),[list(r) for r in h]); print('parsed archive:',[m._archived_calibration(c) for c in range(4)])
past,hold,x,b=m._prepare(d); n=len(d['train']['lab_id']); dt=d['past_dt']; t=d['eval_times']; a=m._activity(d,past,dt)
def score(e): return tuple(np.round([np.sqrt(np.mean(e**2)),np.median(np.abs(e)),np.max(np.abs(e))],6))
e=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; lab=d['test']['lab_id']; stratum=d['test']['stratum']; raw=d['test']['coupon_features_raw']; print('clean test curve/rate RMSE, MAD, max:',score(e),score(er)); print('per horizon RMSE',np.sqrt(np.mean(e**2,axis=0)))
for name,mask in [(f'lab {c}',lab==c) for c in range(4)]+[(f'stratum {c}',stratum==c) for c in np.unique(stratum)]+[('inactive',a[n:]==0),('active',a[n:]>0),('imputed',np.isnan(raw).any(axis=1))]:
 if mask.any(): print(name,mask.sum(),score(e[mask]),score(er[mask]))
y=d['train']['strain_curves_obs']-b[:n,None]; yr=d['train']['instant_rate_obs']; labs=d['train']['lab_id']
for c in range(4):
 tr=labs!=c; te=~tr
 for regime in (True,False):
  tau,w=m._fit(past[:n][tr],hold[:n][tr],x[:n][tr],dt,t,a[:n][tr],y[tr],yr[tr],regime)
  A,R=m._basis(tau,past[:n][te],hold[:n][te],x[:n][te],dt,t,a[:n][te],regime); w=w[:A.shape[-1]]
  print('held lab',c,'plateau',regime,'tau',tau,'curve/rate residual',score(A@w-y[te]),score(R@w-yr[te]))
PY
```
- **Container Exit Code**: `0` (`1123 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ights estimated [0.28487287 0.23458083] reference [0.28 0.24] max abs error 0.005419173159076585
archive rows: 1 [['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']]
parsed archive: [None, None, None, None]
clean test curve/rate RMSE, MAD, max: (0.217974, 0.029852, 1.328705) (0.040387, 0.017877, 0.257776)
per horizon RMSE [0.01036603 0.03197877 0.0626944  0.10024371 0.14248431 0.188066
 0.23616117 0.28618929 0.3377753  0.39061468]
lab 0 17 (0.24494, 0.029509, 1.29222) (0.031335, 0.021549, 0.075158)
lab 1 16 (0.184256, 0.025799, 1.124802) (0.068075, 0.016316, 0.257776)
lab 2 16 (0.100231, 0.028594, 0.510725) (0.028513, 0.022612, 0.063943)
lab 3 23 (0.271634, 0.036493, 1.328705) (0.02431, 0.013536, 0.04946)
stratum 0 22 (0.214082, 0.030537, 0.986368) (0.028091, 0.014561, 0.067449)
stratum 1 24 (0.261285, 0.033595, 1.328705) (0.057953, 0.022935, 0.257776)
stratum 2 26 (0.172556, 0.026136, 1.124802) (0.027369, 0.015285, 0.075158)
inactive 24 (0.367472, 0.1276, 1.328705) (0.02085, 0.016316, 0.047482)
active 48 (0.06125, 0.021525, 0.475948) (0.047216, 0.017994, 0.257776)
imputed 3 (0.204554, 0.075541, 0.475948) (0.156614, 0.067449, 0.257776)
held lab 0 plateau True tau [0.50281467 3.62586907] curve/rate residual (1.020278, 0.057633, 17.655527) (0.083262, 0.048294, 0.351226)
held lab 0 plateau False tau [ 0.4109268 22.0387627] curve/rate residual (1.072243, 0.164398, 17.97608) (0.48662, 0.305239, 1.552565)
held lab 1 plateau True tau [0.49939459 3.81677273] curve/rate residual (1.416603, 0.028391, 25.518773) (0.046946, 0.018931, 0.18895)
held lab 1 plateau False tau [ 0.43780099 22.80237031] curve/rate residual (1.382573, 0.15395, 25.045651) (0.556157, 0.35675, 1.349108)
held lab 2 plateau True tau [0.50045683 3.72511079] curve/rate residual (0.959035, 0.024377, 18.427439) (0.026429, 0.013568, 0.072404)
held lab 2 plateau False tau [ 0.42645505 24.83380557] curve/rate residual (0.943657, 0.168505, 18.4662) (0.516967, 0.274114, 1.35492)
held lab 3 plateau True tau [0.48660003 2.81504205] curve/rate residual (0.383871, 0.094316, 5.547825) (0.116082, 0.09001, 0.262402)
held lab 3 plateau False tau [ 0.4284998  21.89097835] curve/rate residual (0.434068, 0.182037, 5.320272) (0.549254, 0.336117, 1.351939)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b641f16cca61ce93 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b641f16c (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32530 ms` LLM)
- **Rationale**: Compare the two drift bases on held-out labs before editing; retain the broader logarithmic basis only if the measured inactive rows support it. Inspect the archive row structure to fix the confirmed fallback parsing failure.
- **Alternatives Weighed**: `Change the optimizer first` (rejected: The error grows with time on inactive rows even though their onset rates are accurate, which points first to the drift time basis.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,json,pickle,numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; p,h,x,b=m._prepare(d); n=len(d['train']['lab_id']); dt=d['past_dt']; t=d['eval_times']; a=m._activity(d,p,dt); lab=d['train']['lab_id']; y=d['train']['strain_curves_obs']-b[:n,None]; yr=d['train']['instant_rate_obs']
for entry in json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')):
 print('archive fields', {k:(np.shape(v),str(v)[:55]) for k,v in entry.items()}); print('archive counts',np.unique(entry['lab_id'],return_counts=True))
orig=m._basis
for uniform in (False,True):
 def basis(taus,past,hold,features,dt,times,activity,plateau=True):
  A,R=orig(taus,past,hold,features,dt,times,activity,plateau)
  if uniform and plateau: A[:,:,2:7]=np.log1p(times)[None,:,None]*features[:,None,:]
  return A,R
 m._basis=basis
 print('uniform log flow',uniform)
 for c in (-1,0,1,2,3):
  tr=(lab!=c); tau,w=m._fit(p[:n][tr],h[:n][tr],x[:n][tr],dt,t,a[:n][tr],y[tr],yr[tr]); print('fold',c,'taus',tau,'weights',w if c==-1 else '')
  if c==-1:
   A,R=basis(tau,p[n:],h[n:],x[n:],dt,t,a[n:]); e=b[n:,None]+A@w-g['te_clean']; er=R@w-g['te_true_rate0']; print('clean errors overall/inactive/active/nonmissing/imputed:',[(round(np.sqrt(np.mean(e[k]**2)),6),round(np.sqrt(np.mean(er[k]**2)),6)) for k in [np.ones(len(e),bool),a[n:]==0,a[n:]>0,~np.isnan(d['test']['coupon_features_raw']).any(axis=1),np.isnan(d['test']['coupon_features_raw']).any(axis=1)]])
   A,R=basis(g['taus'],p[n:],h[n:],x[n:],dt,t,a[n:]); ew=b[n:,None]+A@np.r_[g['theta'],g['vp_theta']]-g['te_clean']; print('public coefficient nonmissing diagnostic RMSE',np.sqrt(np.mean(ew[~np.isnan(d['test']['coupon_features_raw']).any(axis=1)]**2)))
  else:
   te=~tr; A,R=basis(tau,p[:n][te],h[:n][te],x[:n][te],dt,t,a[:n][te]); e=A@w-y[te]; er=R@w-yr[te]; print('OOF curve abs median overall/inactive/active and rate abs median',*[round(np.median(np.abs(e[k])),6) for k in [np.ones(len(e),bool),a[:n][te]==0,a[:n][te]>0]],round(np.median(np.abs(er)),6))
PY
```
- **Container Exit Code**: `0` (`1331 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0, 1, 1, 1, 1, 1, 1, 1, 1, '), 'cmd_stress': ((40,), '[1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.'), 'true_stress': ((40,), '[1.619593, -1.308033, -0.646936, 0.95752, 2.133164, -0.')}
archive counts (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
uniform log flow False
fold -1 taus [0.49927005 3.65346062] weights [ 0.67203732  0.54148651  0.12722214  0.14021881 -0.11727796  0.10958469
 -0.07944296  0.28487287  0.23458083]
clean errors overall/inactive/active/nonmissing/imputed: [(0.217974, 0.040387), (0.367472, 0.02085), (0.06125, 0.047216), (0.218539, 0.025211), (0.204554, 0.156614)]
public coefficient nonmissing diagnostic RMSE 0.26017262487445036
fold 0 taus [0.50281467 3.62586907] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.057633 0.115255 0.053927 0.048294
fold 1 taus [0.49939459 3.81677273] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.028391 0.23739 0.017463 0.018931
fold 2 taus [0.50045683 3.72511079] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.024377 0.162989 0.014299 0.013568
fold 3 taus [0.48660003 2.81504205] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.094316 0.076555 0.098275 0.09001
uniform log flow True
fold -1 taus [0.49971729 3.73097831] weights [ 0.67607939  0.54808314  0.13901646  0.15636619 -0.12764482  0.11746708
 -0.08817207  0.27990035  0.24057203]
clean errors overall/inactive/active/nonmissing/imputed: [(0.040479, 0.030804), (0.004117, 0.003575), (0.049491, 0.037642), (0.004522, 0.003985), (0.197117, 0.149692)]
public coefficient nonmissing diagnostic RMSE 0.009621749513319027
fold 0 taus [0.49940442 3.71122809] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.013471 0.01372 0.013407 0.012276
fold 1 taus [0.4996954  3.72557965] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.012993 0.01617 0.011633 0.016999
fold 2 taus [0.49997472 3.74113103] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.011876 0.01203 0.011422 0.013184
fold 3 taus [0.49992838 3.77306524] weights 
OOF curve abs median overall/inactive/active and rate abs median 0.01431 0.014872 0.014193 0.013111

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b641f16cca61ce93 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b641f16c (directory).
```

### Turn 7 — `execute_in_remote_docker` (`49114 ms` LLM)
- **Rationale**: Promote the held-lab-supported flow law and fix the archive reader for the supplied list of sweeps. Verify the exact edited artifact on the public outputs and force the documented sparse and EMI cases to catch fallback and channel-selection errors.
- **Alternatives Weighed**: `Keep linear drift for inactive coupons and retune the fit` (rejected: The inactive error grows with time despite accurate onset rates, and changing the flow law improves all four held-out labs.)
- **ATMS NOGOOD DECLARED**: Refuted `Low-banding coupons with zero accumulated damage return to linear terminal flow.` | Witness: `In all four held-out labs the inactive curve median absolute error fell when drift remained logarithmic; the clean inactive RMSE fell from 0.367472 to 0.004117.` | Blanket Diagnosis: `The original gate incorrectly used damage to switch the drift law as well as to activate the network contributions.` | Sublated Pivot: `Keep logarithmic drift throughout the low-banding campaign and retain damage gating on the two network contributions.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
a = s.index('def _archived_calibration(')
b = s.index('def _calibrate(', a)
s = s[:a] + '''def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == channel
        cmd = np.asarray(sweep['cmd_stress'])[mask]
        true = np.asarray(sweep['true_stress'])[mask]
        if np.unique(cmd).size >= 2:
            return np.polyfit(cmd, true, 1)
    return None


''' + s[b:]
s = s.replace('        archive = _archived_calibration(channel) if x.size < 2 or spike else None\n','')
s = s.replace('''        elif archive is not None:
            result[channel] = archive
''','''        elif (x.size < 2 or spike) and (archive := _archived_calibration(channel)) is not None:
            result[channel] = archive
''')
s = s.replace('''    flow = np.where(activity[:, None] > 0, np.log1p(times), times) if plateau else np.broadcast_to(times, (len(hold), len(times)))''','''    # Shear banding selects the flow law; damage scales the two network terms.
    flow = np.log1p(times) if plateau else times''')
s = s.replace('flow[:, :, None] * features[:, None, :]', 'flow[None, :, None] * features[:, None, :]')
s = s.replace('network[:, None] * flow', 'network[:, None] * flow[None, :]')
s = s.replace('    plateau = plateau and np.any(activity > 0)\n','')
s = s.replace('    if not plateau:\n        order = np.argsort(tau)', '    if not plateau or not np.any(activity > 0):\n        order = np.argsort(tau)')
s = s.replace('        coef = np.r_[coef, 0.0, 0.0]', '    if not plateau:\n        coef = np.r_[coef, 0.0, 0.0]')
s = s.replace('    activity = _activity(dataset, past, dt)\n', "    plateau = dataset['shear_banding_index'] <= 0.50\n    activity = _activity(dataset, past, dt)\n")
s = s.replace("train['instant_rate_obs'])", "train['instant_rate_obs'], plateau)")
s = s.replace('dt, times, activity[n:])', 'dt, times, activity[n:], plateau)\n    prediction_coef = coef[:curves.shape[-1]]')
s = s.replace('curves @ coef', 'curves @ prediction_coef').replace('rates @ coef', 'rates @ prediction_coef')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c02_batch_01(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'max abs error',np.max(np.abs(o[k]-g[j])))
p,h,x,b=m._prepare(d); n=len(d['train']['lab_id']); a=m._activity(d,p,d['past_dt'])[n:]; miss=np.isnan(d['test']['coupon_features_raw']).any(axis=1)
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; rmse=lambda z:float(np.sqrt(np.mean(z*z)))
for name,mask in [('overall',np.ones(len(a),bool)),('inactive',a==0),('active',a>0),('complete',~miss),('imputed',miss)]:
 if mask.any(): print(name,int(mask.sum()),'curve/rate RMSE',rmse(e[mask]),rmse(r[mask]))
for channel in range(4):
 c=d['calibration_load_cells']; select=c['lab_id']!=channel; sparse={k:v[select] for k,v in c.items()}; expected=m._archived_calibration(channel); assert expected is not None; np.testing.assert_allclose(m._calibrate(sparse)[channel],expected)
 c={k:v.copy() for k,v in c.items()}; dirty=np.flatnonzero(c['lab_id']==channel)[-1]; c['true_stress'][dirty]+=4.5; mask=(c['lab_id']==channel)&(np.arange(len(select))!=dirty)
 np.testing.assert_allclose(m._calibrate(c)[channel],np.polyfit(c['cmd_stress'][mask],c['true_stress'][mask],1))
print('Sparse archive fallback and EMI rejection passed on all four channels.')
PY
```
- **Container Exit Code**: `0` (`967 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus max abs error 0.09402168978669279
hereditary_weights max abs error 0.0039206141600862665
regime_mode_weights max abs error 0.0005720293457299463
overall 72 curve/rate RMSE 0.04047915399193919 0.030803810514765897
inactive 24 curve/rate RMSE 0.004117342616860508 0.0035747778645318163
active 48 curve/rate RMSE 0.0494910760313653 0.037642032290433385
complete 69 curve/rate RMSE 0.004521696690500113 0.003985405990789991
imputed 3 curve/rate RMSE 0.1971173096092142 0.14969193102017117
Sparse archive fallback and EMI rejection passed on all four channels.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c93eb62068b05b3c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c93eb620 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`24374 ms` LLM)
- **Rationale**: Check whether imputation follows the requested cell-balanced convention and whether its residual has the shared log-curve and onset-rate signature of a drift error. Correct a demonstrated alignment mismatch; otherwise retain the prescribed imputation rather than tuning to the three public test rows.
- **Alternatives Weighed**: `Retune the Prony fit immediately` (rejected: Complete-panel curve RMSE is already 0.004522; the remaining errors are concentrated on three coupons with missing features.)

```bash
cd /holdout_task && sed -n '49,108p' alpha_m97_c02_batch_01_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('candidate','alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; p,h,x,b=m._prepare(d); n=len(d['train']['lab_id']); t=d['eval_times']; dt=d['past_dt']; a=m._activity(d,p,dt)
raw=np.concatenate([d[k]['coupon_features_raw'] for k in ('train','test')]); lab=np.concatenate([d[k]['lab_id'] for k in ('train','test')]); st=np.concatenate([d[k]['stratum'] for k in ('train','test')]); mask=np.isnan(raw); print('gt keys/shapes:',[(k,np.shape(v)) for k,v in g.items()]); print('reference shapes:', [np.shape(d[k]['ref_coupons']) for k in ('train','test')]);
for j in np.flatnonzero(mask[n:].any(1)):
 k=n+j; print('missing case',j,'cell',lab[k],st[k],'columns',np.flatnonzero(mask[k]),'aligned values',x[k,1:]);
A,R=m._basis(g['taus'],p[n:],h[n:],x[n:],dt,t,a[n:]); coef=np.r_[g['theta'],g['vp_theta']]; e=b[n:,None]+A@coef-g['te_clean']; r=R@coef-g['te_true_rate0']; good=~mask[n:].any(1); flow=np.log1p(t); drift=e@flow/(flow@flow); print('reference coefficients complete-panel RMS drift, curve remainder, rate remainder:',*[np.sqrt(np.mean(v**2)) for v in (drift[good],(e-drift[:,None]*flow)[good],(r-drift)[good])]); print('missing cases drift signatures:',np.c_[drift[~good],r[~good]])
PY
```
- **Container Exit Code**: `0` (`434 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(taus, past, hold, features, dt, times, activity, plateau=True):
    # A stress increment is attenuated by all steps after its own step.
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gap = hold[:, None] - z
    rise = -np.expm1(-times[:, None] / taus)
    # Shear banding selects the flow law; damage scales the two network terms.
    flow = np.log1p(times) if plateau else times
    cols = 9 if plateau else 7
    curves = np.empty((len(hold), len(times), cols))
    rates = np.empty((len(hold), cols))
    curves[:, :, :2] = gap[:, None, :] * rise
    rates[:, :2] = gap / taus
    curves[:, :, 2:7] = flow[None, :, None] * features[:, None, :]
    rates[:, 2:7] = features
    if plateau:
        extra = activity * hold
        network = activity * z[:, 0]
        curves[:, :, 7] = extra[:, None] * rise[:, 0]
        curves[:, :, 8] = network[:, None] * flow[None, :]
        rates[:, 7] = extra / taus[0]
        rates[:, 8] = network
    return curves, rates
gt keys/shapes: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
reference shapes: [(180, 32), (72, 32)]
missing case 32 cell 0 0 columns [2] aligned values [ 0.12699299 -0.92968471 -0.15754602  1.07752927]
missing case 40 cell 1 0 columns [0] aligned values [ 0.03404788  1.82399888 -1.30247945 -1.2230961 ]
missing case 43 cell 1 1 columns [2 3] aligned values [ 0.0211652  -0.38397089 -0.15754602  0.0801342 ]
reference coefficients complete-panel RMS drift, curve remainder, rate remainder: 0.00734720587961929 1.2683439896959808e-05 7.172649778523135e-05
missing cases drift signatures: [[ 0.08329622  0.08313299]
 [-0.05180833 -0.05173948]
 [-0.24084501 -0.24078351]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c93eb62068b05b3c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c93eb620 (directory).
```

### Turn 9 — `submit_completed_task` (`23834 ms` LLM)
- **Rationale**: The verified candidate reduced worst held-lab inactive median strain error from 0.237390 to 0.016170. Isolated schema validation, sparse and EMI calibration checks, and the host scientific probe passed on the submitted artifact.
- **Alternatives Weighed**: `Retune imputation using public test outcomes` (rejected: The residuals are consistent with missing-feature drift, not a Prony basis error. Tuning to three test coupons would abandon the prescribed cell-balanced estimator without evidence of generalization.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_01_solver.py': 'FILE:7263'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema and four-channel calibration checks passed. Public test RMSE was 0.04048 for strain and 0.03080 for initial rate. The three coupons with missing features were the least accurate slice.`

### Turn 10 — `execute_in_remote_docker` (`43423 ms` LLM)
- **Rationale**: An independent stepwise recurrence can expose a sign, branch, drift, or onset-rate error shared by the fit and forecast; a mismatch would require a basis correction. Paired held-lab fits test whether imputed training rows bias the slow relaxation estimate without changing the prescribed prediction-time imputation. Retain the verified candidate unless the challenger improves the held-lab margin.
- **Alternatives Weighed**: `Add rational, pairwise, HJB, or relay terms; score batch classification` (rejected: This task specifies continuous strain and onset-rate outputs from two memory branches and a drift law. None of those terms or classification metrics is part of its model or data. Held-lab error and signed terminal bias directly test whether calibration leaves a lab shortcut.)

```bash
set -e
cd /holdout_task
cat >/tmp/rheology_margin_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
p, h, x, b = m._prepare(d)
tr = d['train']; labs = tr['lab_id']; n = len(labs)
dt, t = d['past_dt'], d['eval_times']
a = m._activity(d, p, dt)
y = tr['strain_curves_obs'] - b[:n, None]; r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('Incomplete training rows per lab:', [(c, int(sum((labs == c) & ~complete))) for c in range(4)])

out = m.solve_alpha_m97_c02_batch_01(d)
tau = out['estimated_taus']; w = out['hereditary_weights']; v = out['regime_mode_weights']
z = np.zeros((len(p), 2))
work = np.zeros(len(p))
for j, step in enumerate(dt):
    decay = np.exp(-step / tau)
    z = decay * z + (1 - decay) * p[:, j, None]
    work += np.maximum(0, np.abs(p[:, j]) - d['crit_stress'])**2 * step
np.testing.assert_allclose(a, np.tanh(work), atol=1e-13)
gap = h[:, None] - z
rise = 1 - np.exp(-t[:, None] / tau)
for plateau in (False, True):
    activity = a if plateau else np.zeros_like(a)
    drift = x @ w[2:] + v[1] * activity * z[:, 0]
    amp = gap * w[:2]; amp[:, 0] += v[0] * activity * h
    expected_curve = amp @ rise.T + drift[:, None] * (np.log1p(t) if plateau else t)
    expected_rate = amp @ (1 / tau) + drift
    A, R = m._basis(tau, p, h, x, dt, t, activity, plateau)
    coef = np.r_[w, v] if plateau else w
    np.testing.assert_allclose(A @ coef, expected_curve, atol=1e-12)
    np.testing.assert_allclose(R @ coef, expected_rate, atol=1e-12)
    if plateau:
        np.testing.assert_allclose(out['predicted_strain'], b[n:, None] + expected_curve[n:], atol=1e-12)
        np.testing.assert_allclose(out['initial_rate_pred'], expected_rate[n:], atol=1e-12)
print('Independent recurrence, both flow laws, onset derivative and output handoff passed.')

for label, eligible in [('all', np.ones(n, dtype=bool)), ('complete-only', complete)]:
    worst = np.zeros(3)
    for c in range(4):
        fit = (labs != c) & eligible
        test = (labs == c) & complete
        ta, co = m._fit(p[:n][fit], h[:n][fit], x[:n][fit], dt, t, a[:n][fit], y[fit], r[fit])
        A, R = m._basis(ta, p[:n][test], h[:n][test], x[:n][test], dt, t, a[:n][test])
        ec, er = A @ co - y[test], R @ co - r[test]
        metric = np.array([np.median(np.abs(ec)), np.median(np.abs(er)), abs(np.median(ec[:, -1]))])
        worst = np.maximum(worst, metric)
        print(label, 'held lab', c, 'tau', np.round(ta, 5), 'curve/rate MedAE and terminal bias', np.round(metric, 6))
    print(label, 'worst held-lab metrics:', np.round(worst, 6))
    ta, co = m._fit(p[:n][eligible], h[:n][eligible], x[:n][eligible], dt, t, a[:n][eligible], y[eligible], r[eligible])
    A, R = m._basis(ta, p[n:], h[n:], x[n:], dt, t, a[n:])
    ec, er = b[n:, None] + A @ co - gt['te_clean'], R @ co - gt['te_true_rate0']
    rmse = lambda v: np.sqrt(np.mean(v*v))
    print(label, 'public relative tau max, curve RMSE, rate RMSE:', max(abs(ta / gt['taus'] - 1)), rmse(ec), rmse(er))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1791 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Incomplete training rows per lab: [(0, 5), (1, 4), (2, 5), (3, 5)]
Independent recurrence, both flow laws, onset derivative and output handoff passed.
all held lab 0 tau [0.4994  3.71123] curve/rate MedAE and terminal bias [0.011788 0.008143 0.000918]
all held lab 1 tau [0.4997  3.72558] curve/rate MedAE and terminal bias [0.011485 0.014551 0.001957]
all held lab 2 tau [0.49997 3.74113] curve/rate MedAE and terminal bias [1.0717e-02 1.2130e-02 9.0000e-05]
all held lab 3 tau [0.49993 3.77307] curve/rate MedAE and terminal bias [0.012691 0.011805 0.005032]
all worst held-lab metrics: [0.012691 0.014551 0.005032]
all public relative tau max, curve RMSE, rate RMSE: 0.024580833931161505 0.04047915399193919 0.030803810514765897
complete-only held lab 0 tau [0.50001 3.74828] curve/rate MedAE and terminal bias [0.01174  0.008573 0.000805]
complete-only held lab 1 tau [0.50055 3.77517] curve/rate MedAE and terminal bias [0.011703 0.012922 0.003869]
complete-only held lab 2 tau [0.50109 3.77907] curve/rate MedAE and terminal bias [0.010672 0.011068 0.000278]
complete-only held lab 3 tau [0.50059 3.80872] curve/rate MedAE and terminal bias [0.012514 0.01205  0.004015]
complete-only worst held-lab metrics: [0.012514 0.012922 0.004015]
complete-only public relative tau max, curve RMSE, rate RMSE: 0.014546395243486554 0.04049410838816158 0.030886634083673523
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c93eb62068b05b3c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c93eb620 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`29745 ms` LLM)
- **Rationale**: Promote the fit that better identifies the relaxation spectrum without materially changing prediction error. Verify that the edited solver selects the tested fit and still passes the independent recurrence, onset-rate handoff, and public schema checks; a failure would block promotion.
- **Alternatives Weighed**: `Keep all training coupons in the fit` (rejected: Complete-coupon fitting improved all three worst held-lab metrics and reduced public maximum relative tau error from 2.46% to 1.45%; public strain and rate RMSE changed by less than 0.0001.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
old = '''    tau, coef = _fit(past[:n], hold[:n], features[:n], dt, times, activity[:n],
                     train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs'], plateau)'''
new = '''    # Imputation supplies prediction features, not measured training covariates.
    # Prefer complete coupons for identifying the shared relaxation spectrum.
    fit_rows = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if fit_rows.sum() <= n / 2 or (np.any(activity[:n] > 0) and not np.any(activity[:n][fit_rows] > 0)):
        fit_rows = slice(None)
    tau, coef = _fit(past[:n][fit_rows], hold[:n][fit_rows], features[:n][fit_rows],
                     dt, times, activity[:n][fit_rows],
                     (train['strain_curves_obs'] - strain0[:n, None])[fit_rows],
                     train['instant_rate_obs'][fit_rows], plateau)'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_margin_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c02_batch_01(d)
rel=np.max(np.abs(o['estimated_taus']/g['taus']-1)); print('Solver tau maximum relative error:',rel); assert rel < 0.02
for k,j in [('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'maximum absolute error',np.max(np.abs(o[k]-g[j])))
miss=~np.isfinite(d['test']['coupon_features_raw']).all(axis=1); rmse=lambda e:np.sqrt(np.mean(e**2))
for key,ref,limit in [('predicted_strain','te_clean',0.041),('initial_rate_pred','te_true_rate0',0.0315)]:
 e=o[key]-g[ref]; print(key,'RMSE all/complete/imputed',rmse(e),rmse(e[~miss]),rmse(e[miss])); assert rmse(e)<limit
PY
```
- **Container Exit Code**: `0` (`2145 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Incomplete training rows per lab: [(0, 5), (1, 4), (2, 5), (3, 5)]
Independent recurrence, both flow laws, onset derivative and output handoff passed.
all held lab 0 tau [0.4994  3.71123] curve/rate MedAE and terminal bias [0.011788 0.008143 0.000918]
all held lab 1 tau [0.4997  3.72558] curve/rate MedAE and terminal bias [0.011485 0.014551 0.001957]
all held lab 2 tau [0.49997 3.74113] curve/rate MedAE and terminal bias [1.0717e-02 1.2130e-02 9.0000e-05]
all held lab 3 tau [0.49993 3.77307] curve/rate MedAE and terminal bias [0.012691 0.011805 0.005032]
all worst held-lab metrics: [0.012691 0.014551 0.005032]
all public relative tau max, curve RMSE, rate RMSE: 0.024580833931161505 0.04047915399193919 0.030803810514765897
complete-only held lab 0 tau [0.50001 3.74828] curve/rate MedAE and terminal bias [0.01174  0.008573 0.000805]
complete-only held lab 1 tau [0.50055 3.77517] curve/rate MedAE and terminal bias [0.011703 0.012922 0.003869]
complete-only held lab 2 tau [0.50109 3.77907] curve/rate MedAE and terminal bias [0.010672 0.011068 0.000278]
complete-only held lab 3 tau [0.50059 3.80872] curve/rate MedAE and terminal bias [0.012514 0.01205  0.004015]
complete-only worst held-lab metrics: [0.012514 0.012922 0.004015]
complete-only public relative tau max, curve RMSE, rate RMSE: 0.014546395243486554 0.04049410838816158 0.030886634083673523
Solver tau maximum relative error: 0.014546395243486554
hereditary_weights maximum absolute error 0.0036627060060459293
regime_mode_weights maximum absolute error 0.0007657028637663432
predicted_strain RMSE all/complete/imputed 0.04049410838816158 0.004483074204260213 0.1972112942766501
initial_rate_pred RMSE all/complete/imputed 0.030886634083673523 0.003798715029123283 0.15021226492994047

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0725cc728ddcce69 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_0725cc72 (directory).
```

### Turn 12 — `submit_completed_task` (`14884 ms` LLM)
- **Rationale**: The final artifact passed isolated Python schema validation and independent checks of the memory recurrence, both flow laws, and the onset-rate derivative. Held-lab and public-reference results support the complete-coupon fit without changing the prescribed prediction-time imputation.
- **Alternatives Weighed**: `Retain all training coupons in the fit` (rejected: Fitting complete coupons reduced the maximum relative relaxation-time error from 2.46% to 1.45% and improved worst held-lab errors, while both public prediction RMSEs changed by less than 0.0001.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. Public test RMSE was 0.04049 for strain and 0.03089 for initial rate; the largest relative error across the two relaxation times was 1.45%. The three coupons requiring feature imputation remained the least accurate slice.`

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
