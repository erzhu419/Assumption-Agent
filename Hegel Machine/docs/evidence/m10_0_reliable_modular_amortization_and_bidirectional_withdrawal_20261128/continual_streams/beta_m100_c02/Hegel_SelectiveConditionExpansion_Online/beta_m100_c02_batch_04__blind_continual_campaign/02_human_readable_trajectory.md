# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c02_batch_04__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `13 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `331,427` (`320,952` / `98,944` / `10,475`) |
| **Total Episode Wall Time (s)** | `350.33s` |

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

### Turn 1 — `execute_in_remote_docker` (`14019 ms` LLM)
- **Rationale**: Inspect the calibrated input contract and determine whether pulse data or archive telemetry can distinguish the boundary conditions before implementing the solver.

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py validate_public_schema.py _upstream_param_payload.json public_asset_index.json; ls -lh /tmp/hegel_continual_archive; python3 - <<'PY'
import pickle,numpy as np
for i,x in enumerate(pickle.load(open('/holdout_task/public_batch.pkl','rb'))):
 print(i,x.keys(), 'diagnostics',type(x.get('campaign_diagnostics')), 'count',len(x.get('campaign_diagnostics') or []))
 for k,v in x['study'].items(): print(k, (v.shape,str(v.dtype),v.ravel()[:4]) if isinstance(v,np.ndarray) else (list(v) if isinstance(v,dict) else repr(v)[:200]))
 if x.get('campaign_diagnostics'): print(x['campaign_diagnostics'][0].keys())
PY
```
- **Container Exit Code**: `0` (`373 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
}total 44K
-r--r--r-- 1 root root  19K Nov 28 03:41 archive_apparatus_log.json
-rw-r--r-- 1 root root  675 Nov 28 03:41 archive_manifest.json
-rw-r--r-- 1 root root 4.7K Nov 28 03:41 beta_m100_c02_batch_01_solver.py
-rw-r--r-- 1 root root 4.7K Nov 28 03:41 beta_m100_c02_s00_solver.py
-rw-r--r-- 1 root root  127 Nov 28 03:41 pre_llm_gate_diagnostic.json
0 dict_keys(['study', 'campaign_diagnostics']) diagnostics <class 'list'> count 3
Lx 2.4
Ly 2.0
nx 14
ny 12
kappa_x ((14,), 'float64', array([0.47110686, 0.50461106, 0.53616811, 0.56394404]))
kappa_y ((12,), 'float64', array([0.57938047, 0.57451963, 0.56033089, 0.53796376]))
vel_x ((14,), 'float64', array([0.35      , 0.34500413, 0.33015913, 0.30588881]))
vel_y ((12,), 'float64', array([-0.        , -0.05560533, -0.10899563, -0.1580441 ]))
robin_x 1.0141673808964704
robin_y 1.1585219318384303
V_reaction_2d ((14, 12), 'float64', array([0.25, 0.25, 0.25, 0.25]))
mu_shifts ((3,), 'float64', array([0.15, 0.55, 1.1 ]))
source_modes ((8, 14, 12), 'float64', array([2.70117782e-11, 1.24329054e-09, 3.62029408e-08, 6.66909400e-07]))
sensor_ij ((32, 2), 'int64', array([10,  4,  8,  8]))
row_bank_id ((96,), 'int64', array([0, 1, 2, 3]))
sensor_calibration_sweeps ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ((96, 3), 'float64', array([-0.78126842,  0.41263868, -0.05312056, -3.7779729 ]))
y_obs_raw ((96,), 'float64', array([0.01682945, 0.96230258, 0.27997598, 0.48115801]))
aux_manifold_telemetry ((24, 4), 'float64', array([-1.97709076, -0.64910059,  2.33022354,  0.11634972]))
nuisance_ridge 0.08
source_ridge 0.04
u_box 0.42
total_budget 1.4
dict_keys(['w_pulse', 'y_pulse_raw'])
```

### Turn 2 — `execute_in_remote_docker` (`17332 ms` LLM)
- **Rationale**: Read the upstream return contract and the archive's boundary evidence. These determine which wall stencils to test against the pulses and prevent fitting boundary effects to the production observations.
- **Alternatives Weighed**: `Reuse a prior campaign's boundary operator directly` (rejected: The wall law can change between campaigns; compare archive evidence with the current calibrated pulses first.)

```bash
cd /holdout_task && tail -30 _verified_upstream.py && cat validate_public_schema.py; cd /tmp/hegel_continual_archive && cat archive_manifest.json pre_llm_gate_diagnostic.json && for f in *_solver.py; do echo "--- $f"; grep -nE 'def |robin|boundary|kap\[|periodic|wall' "$f"; done; python3 - <<'PY'
import json
p=json.load(open('archive_apparatus_log.json'))
print(type(p), list(p) if isinstance(p,dict) else len(p))
for k,v in (p.items() if isinstance(p,dict) else [('records',p)]):
 print(k, type(v).__name__, len(v) if isinstance(v,(dict,list,str)) else v)
 if isinstance(v,list) and v:
  for r in v[-4:]:
   print(str(r)[:4500])
 elif isinstance(v,dict): print(str(v)[:4500])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
```
- **Container Exit Code**: `0` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.26305, 2.449], [-0.63874, -0.32698, 0.04527, -0.67542, 0.84364, -0.81366, 0.40489, 2.4774]]}
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c02_batch_03', 'facility_telemetry_trace': [[-0.24626, 2.50359, -0.37308, -0.87533, 0.83152, -0.19964, -0.80421, -0.40314], [-0.05086, 2.59738, -0.77707, -0.7417, 0.63401, 0.47049, -0.86958, -0.40649], [0.16724, 2.51288, -0.76352, -0.57696, 0.37906, 0.81554, -0.86913, -0.45468], [0.40011, 2.49938, -0.31905, -0.43889, -0.14408, 0.70361, -0.77167, -0.39508], [0.54113, 2.58549, 0.31388, -0.23083, -0.52056, 0.2464, -0.67125, -0.43909], [0.74491, 2.63069, 0.76039, 0.03644, -0.7906, -0.46588, -0.52081, -0.35051], [0.82574, 2.48278, 0.78945, 0.25922, -0.84635, -0.7951, -0.32498, -0.38147], [0.82412, 2.52464, 0.33193, 0.47739, -0.63432, -0.71961, -0.13416, -0.42365], [0.84601, 2.46833, -0.36982, 0.56707, -0.30879, -0.19306, 0.12852, -0.39273], [0.76731, 2.69742, -0.82237, 0.77764, 0.14586, 0.42425, 0.3797, -0.30955], [0.64567, 2.60283, -0.78326, 0.77724, 0.55261, 0.84634, 0.55194, -0.36698], [0.47381, 2.59417, -0.31923, 0.86902, 0.71321, 0.73517, 0.72105, -0.33789], [0.28045, 2.58187, 0.3794, 0.83279, 0.81861, 0.17427, 0.78983, -0.41776], [0.00808, 2.53651, 0.78911, 0.77358, 0.643, -0.40979, 0.86542, -0.35124], [-0.17224, 2.40762, 0.7852, 0.62486, 0.29158, -0.80859, 0.89491, -0.25366], [-0.33689, 2.43824, 0.29637, 0.43045, -0.14068, -0.69853, 0.75133, -0.3311], [-0.57671, 2.6982, -0.33911, 0.21269, -0.49744, -0.17799, 0.65984, -0.43983], [-0.68123, 2.60841, -0.76891, -0.05439, -0.81927, 0.42523, 0.50594, -0.42878], [-0.83338, 2.48649, -0.75338, -0.17363, -0.86922, 0.82754, 0.33525, -0.36908], [-0.84296, 2.55261, -0.28745, -0.48383, -0.60176, 0.68294, 0.10286, -0.31619], [-0.81697, 2.63478, 0.34098, -0.63749, -0.29957, 0.23249, -0.16984, -0.5169], [-0.75287, 2.60074, 0.80517, -0.72083, 0.08803, -0.44581, -0.37555, -0.33179], [-0.6197, 2.56754, 0.79711, -0.82136, 0.48554, -0.86443, -0.52417, -0.39047], [-0.46214, 2.50722, 0.35196, -0.86747, 0.77876, -0.74906, -0.69201, -0.34306]]}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`16634 ms` LLM)
- **Rationale**: Extract prior boundary stencils and compact telemetry summaries. A prior wall stencil or a documented campaign change will determine the discrete candidates tested on the current pulses.
- **Alternatives Weighed**: `Use the sparse ridge primitive` (rejected: The objective has a budget equality and box bounds, not an L0 penalty; inspect the primitive contract but use the specified ridge projection.)

```bash
cd /tmp/hegel_continual_archive && cat archive_manifest.json pre_llm_gate_diagnostic.json && diff -u *_solver.py | head -100; sed -n '1,115p' /tmp/hegel_continual_archive/*batch_01_solver.py; python3 - <<'PY'
import json, numpy as np
p=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for k,v in p.items():
 print(k, type(v).__name__)
 if isinstance(v,list):
  for r in v:
   if not isinstance(r,dict): print(str(r)[:150]);continue
   print({k:v for k,v in r.items() if k!='facility_telemetry_trace'})
   if 'facility_telemetry_trace' in r:
    a=np.array(r['facility_telemetry_trace']); print('mean',a.mean(0).round(3),'std',a.std(0).round(3))
 else: print(str(v)[:800])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -30
```
- **Container Exit Code**: `0` (`402 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.column_stack((g[free], np.ones(len(free))))).T
        nu = z.sum() / q.sum()
        d = nu*q-z
        target = w[free] + d
        hits = ((target < -2e-14) | (target > cap+2e-14))
        if hits.any():
            hits = np.flatnonzero(hits)
            dest = np.where(d[hits] < 0, 0., cap)
            alpha = (dest-w[free[hits]])/d[hits]
            j = alpha.argmin()
            w[free] += np.clip(alpha[j], 0., 1.)*d
            k = free[hits[j]]
            w[k] = dest[j]
            side[k] = -1 if dest[j] == 0 else 1
            continue
        w[free] = np.clip(target, 0., cap)
        g = h @ w - c
        nu = g[free].mean()
        violation = side*(g-nu)
        k = violation.argmax()
        if violation[k] <= 2e-13:
            return w, float(nu)
        side[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s, terminal = chamber_response(study)
    y = calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    zz = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    sy = np.column_stack((s, y))
    sy -= z @ np.linalg.solve(zz, z.T @ sy)
    a, r = sy[:, :-1], sy[:, -1]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    h = a.T @ a + ridge * np.eye(a.shape[1])
    w, nu = box_simplex(h, a.T @ r, cap, budget)
    residual = a @ w - r
    g = a.T @ residual + ridge*w
    lower, upper = np.maximum(0, g-nu), np.maximum(0, nu-g)
    obj = 0.5 * (residual @ residual + ridge*(w @ w))
    dual = obj - g @ w + nu*budget - cap*upper.sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
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
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 4 — `execute_in_remote_docker` (`17996 ms` LLM)
- **Rationale**: Check pulse scale and nuisance content before fitting any wall stencil. If the pulses carry nuisance contamination, compare projected as well as raw residuals; use the archived sensor ordering to avoid mistaking a layout error for a boundary change.

```bash
sed -n '1,53p' /tmp/hegel_continual_archive/*batch_01_solver.py; cd /holdout_task; python3 - <<'PY'
import json,pickle,numpy as np,_verified_upstream as u
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:v for k,v in r.items() if k!='facility_telemetry_trace'}); a=np.array(r['facility_telemetry_trace']); print('mean',a.mean(0).round(3),'std',a.std(0).round(3))
i=pickle.load(open('public_batch.pkl','rb'))[0]; s=u.calibrate_upstream(dict(i['study'], reference_tracer_pulses=i['campaign_diagnostics'])); Z=s['Z_nuisance'];
for p in s['reference_tracer_pulses']:
 y=p['y_pulse_raw'];print('pulse',p['w_pulse'].round(5),'y range',y.min(),y.max(),'norm',np.linalg.norm(y),'nuisance residual',np.linalg.norm(y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]))
print('sensor bounds',s['sensor_ij'].min(0),s['sensor_ij'].max(0))
PY
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(study['source_modes']).reshape(-1, nx*ny, order='F').T
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx*ny, format='csc')
    readings = []
    for mu in study['mu_shifts']:
        response = splu(operator + mu * eye).solve(rhs)
        readings.append(response[sensors])
    return np.concatenate(readings), response.T.reshape(-1, nx, ny, order='F')


def calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    affine = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)),
                             sweeps['raw_rosette'], rcond=None)[0]
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(affine[1:].T, (raw-affine[0]).T).T.ravel()


def box_simplex(h, c, cap, budget):
    """Active-set solve of 1/2 w'Hw - c'w with a fixed sum."""
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'}
mean [-0.013 -0.42   0.002 -0.002 -0.012 -0.005  0.003 -0.354] std [0.609 0.06  0.597 0.597 0.609 0.601 0.606 0.071]
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m100_c02_s00'}
mean [-2.000e-03 -3.830e-01  1.000e-03  1.300e-02  2.000e-03 -3.000e-03
 -9.000e-03  2.433e+00] std [0.604 0.07  0.611 0.597 0.597 0.612 0.595 0.085]
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c02_batch_01'}
mean [ 0.003 -0.414  0.003  0.011  0.004  0.002 -0.003 -0.343] std [0.597 0.09  0.599 0.603 0.594 0.609 0.605 0.066]
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c02_batch_02'}
mean [ 3.000e-03 -3.640e-01 -1.200e-02  6.000e-03 -4.000e-03 -5.000e-03
 -1.000e-03  2.459e+00] std [0.6   0.054 0.602 0.591 0.609 0.6   0.605 0.086]
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c02_batch_03'}
mean [ 6.000e-03  2.555e+00  3.000e-03  1.000e-03 -5.000e-03  2.000e-03
 -2.000e-03 -3.810e-01] std [0.601 0.073 0.603 0.613 0.59  0.598 0.614 0.055]
pulse [1.  0.  0.  0.5 0.  0.  0.  0. ] y range 0.040275785733081486 0.4446500765601271 norm 1.998164801828218 nuisance residual 1.9478995918504014
pulse [0.  1.  0.  0.  0.5 0.  0.  0. ] y range 0.020534830142624336 0.38135662892798194 norm 1.7979903496696086 nuisance residual 1.769884946610709
pulse [0.  0.  1.  0.  0.  0.5 0.  0. ] y range 0.053518832562562985 0.39162427015654505 norm 1.9305610572793372 nuisance residual 1.8886880661728032
sensor bounds [0 0] [12 10]
```

### Turn 5 — `execute_in_remote_docker` (`35516 ms` LLM)
- **Rationale**: Compare discrete Robin, zero-flux, reflected, and wraparound stencils first. If none match, fit boundary coefficients using two pulses at a time and score the third to distinguish a reproducible boundary law from an in-sample fit.
- **Alternatives Weighed**: `Infer the wall law from facility telemetry alone` (rejected: The available trace ends at the preceding campaign; current calibrated pulses can test the wall stencil directly.)

```bash
cat > /tmp/probe_wall.py <<'PY'
import sys,pickle,itertools,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: i=pickle.load(f)[0]
s=calibrate_upstream(dict(i['study'],reference_tracer_pulses=i['campaign_diagnostics']))
p=s['reference_tracer_pulses']; W=np.stack([a['w_pulse'] for a in p],axis=1); Y=np.stack([a['y_pulse_raw'] for a in p],axis=1)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
rhs=s['source_modes'].reshape(-1,nx*ny,order='F').T@W
base=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); eye=sparse.eye(nx*ny,format='csc')
def axis(k,v,h,r,p):
 n=len(k); a=np.zeros((n,n)); faces=(k[:-1]+k[1:])/(2*h*h); j=np.arange(1,n-1)
 a[j,j-1]=-faces[:-1]-np.maximum(v[1:-1],0)/h; a[j,j+1]=-faces[1:]-np.maximum(-v[1:-1],0)/h; a[j,j]=-a.sum(1)[j]
 for end,near,far,(leak,inside,wrap) in [(0,1,n-1,p[0]),(n-1,n-2,0,p[1])]:
  a[end,near]=-inside*k[end]/h**2; a[end,far]=-wrap*(k[0]+k[-1])/(2*h*h); a[end,end]=-a[end].sum()+leak*r/h
 return sparse.csc_matrix(a)
def pred(p):
 ax,ay=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d],q) for d,q in zip('xy',np.asarray(p).reshape(2,2,3))]; op=base+sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')
 return np.concatenate([splu(op+mu*eye).solve(rhs)[idx] for mu in s['mu_shifts']])
def err(p): return pred(p)-Y
modes={'R':(1,1,0),'N':(0,1,0),'P':(0,1,1),'M':(0,2,0),'D':(0,0,0)}
r=[]
for x,y in itertools.product(modes,repeat=2):
 q=np.linalg.norm(err([modes[x]]*2+[modes[y]]*2),axis=0)/np.linalg.norm(Y,axis=0); r.append((q.max(),x+y,q.round(7)))
for v in sorted(r,key=lambda t:t[0])[:10]+[a for a in r if a[1]=='RR']: print(v)
for free in ('leak','local','full'):
 q0=np.tile([0.05,1,0.05],(4,1)); js={'leak':[0],'local':[0,1],'full':[0,1,2]}[free]; fixed=list(set(range(3))-set(js)); q0[:,fixed]=np.array([0,1,0])[fixed]
 def pars(z):
  q=q0.copy(); q[:,js]=z.reshape(4,len(js)); return q
 for held in [-1,0,1,2]:
  keep=[j for j in range(3) if j!=held]; fit=least_squares(lambda z:err(pars(z))[:,keep].ravel(),q0[:,js].ravel(),bounds=(0,np.inf),max_nfev=100,gtol=1e-13)
  print(free,held,pars(fit.x).round(6).tolist(),(np.linalg.norm(err(pars(fit.x)),axis=0)/np.linalg.norm(Y,axis=0)).round(8))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_wall.py
```
- **Container Exit Code**: `0` (`6800 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0.013522386854547395, 'PN', array([0.0063648, 0.0071292, 0.0135224]))
(0.06961398563456477, 'PM', array([0.0550633, 0.0525508, 0.069614 ]))
(0.21622648341685843, 'MN', array([0.0759904, 0.0767388, 0.2162265]))
(0.23375794133037933, 'MM', array([0.1292516, 0.1189807, 0.2337579]))
(0.23527274163277662, 'NN', array([0.0540587, 0.0758633, 0.2352727]))
(0.2401449348749572, 'NM', array([0.0697893, 0.0820315, 0.2401449]))
(0.25522947017251435, 'PP', array([0.1509301, 0.2552295, 0.0408051]))
(0.25569760161285515, 'NP', array([0.1646466, 0.2556976, 0.2281714]))
(0.276621880227529, 'MP', array([0.1392843, 0.2766219, 0.2074865]))
(0.532138920731401, 'RM', array([0.4642833, 0.4346174, 0.5321389]))
(0.6985851425471022, 'RR', array([0.6822836, 0.6458569, 0.6985851]))
leak -1 [[0.0, 1.0, 0.0], [0.007219, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]] [0.05444999 0.07561295 0.23496712]
leak 0 [[0.0, 1.0, 0.0], [0.01061, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]] [0.05485279 0.07563326 0.23492006]
leak 1 [[0.0, 1.0, 0.0], [0.006995, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]] [0.0544281  0.07561463 0.23497235]
leak 2 [[0.0, 1.0, 0.0], [0.002892, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]] [0.054136   0.0757137  0.23511572]
local -1 [[0.0, 567173.263801, 0.0], [0.019247, 0.504995, 0.0], [0.0, 1.330816, 0.0], [0.0, 1.087949, 0.0]] [0.08066311 0.02461971 0.1734286 ]
local 0 [[0.0, 931788.041433, 0.0], [0.050028, 0.472549, 0.0], [0.0, 2.134248, 0.0], [0.0, 2.209805, 0.0]] [0.10029953 0.04642538 0.16512221]
local 1 [[0.0, 1072080.051387, 0.0], [0.022858, 0.44838, 0.0], [0.0, 1.664781, 0.0], [0.0, 1.177113, 0.0]] [0.08500998 0.03362744 0.17066342]
local 2 [[0.0, 2.59975, 0.0], [0.0, 0.605483, 0.0], [0.0, 1.232895, 0.0], [0.015428, 1.271314, 0.0]] [0.04620077 0.02617091 0.20415606]
full -1 [[0.0, 1.035874, 1.137616], [2e-06, 1.035347, 1.080618], [0.0, 1.000072, 0.0], [2e-06, 1.000027, 0.0]] [1.211e-05 1.304e-05 1.248e-05]
full 0 [[0.0, 1.036489, 1.137935], [1e-06, 1.034964, 1.080124], [0.0, 1.000062, 0.0], [0.0, 1.000014, 0.0]] [2.166e-05 1.285e-05 1.185e-05]
full 1 [[0.0, 1.035931, 1.137621], [2e-06, 1.035232, 1.080481], [0.0, 1.000078, 2e-06], [0.0, 1.000037, 1e-06]] [1.211e-05 1.489e-05 1.228e-05]
full 2 [[0.0, 1.03575, 1.138051], [6e-06, 1.035396, 1.081075], [0.0, 1.000059, 0.0], [0.0, 1.000032, 0.0]] [1.145e-05 1.253e-05 2.622e-05]
```

### Turn 6 — `execute_in_remote_docker` (`22944 ms` LLM)
- **Rationale**: Compare averaged face diffusivity and signed upwind flux independently. If the exact periodic rule reaches the observed pulse error floor, it transfers to unseen grids without fitting boundary coefficients at runtime.
- **Alternatives Weighed**: `Hardcode the fitted endpoint multipliers` (rejected: Multipliers can vary with diffusivity, velocity, and grid spacing. Test the finite-volume rule directly before adopting fitted constants.)
- **ATMS NOGOOD DECLARED**: Refuted `A local endpoint stencil without cross-seam coupling describes the current chamber.` | Witness: `Calibrated pulses give 69.86% worst relative error for two Robin axes and 23.53% for two zero-flux axes, versus 1.35% for periodic x with zero-flux y.` | Blanket Diagnosis: `The wall rows omit cross-seam transport along x; adding the seam leaves a smaller discrepancy consistent with the diffusion and upwind conventions at its endpoints.` | Sublated Pivot: `Extend the stated row-local finite-volume rule to both endpoints of the periodic x axis, retaining the archived zero-flux y stencil.`

```bash
sed '/^modes=/,$d' /tmp/probe_wall.py > /tmp/probe_periodic.py
cat >> /tmp/probe_periodic.py <<'PY'
k,v=s['kappa_x'],s['vel_x']; h=s['Lx']/(nx-1); seam=(k[0]+k[-1])/2
local=[(k[0]+k[1])/2,(k[-1]+k[-2])/2]
for avg,adv in itertools.product((False,True),repeat=2):
 p=[]
 for j,e in enumerate((0,-1)):
  inward=((-1)**(j+1))*v[e]
  p.append([0,((local[j] if avg else k[e])+adv*h*max(inward,0))/k[e],(seam+adv*h*max(-inward,0))/seam])
 p += [[0,1,0]]*2
 print('averaged faces',avg,'upwind at seam',adv,'ratios',np.array(p)[:2,1:].round(7),'relative pulse error',np.linalg.norm(err(p),axis=0)/np.linalg.norm(Y,axis=0))
print('k and v endpoints',k[[0,1,-2,-1]],v[[0,-1]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_periodic.py
```
- **Container Exit Code**: `0` (`536 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
averaged faces False upwind at seam False ratios [[1. 1.]
 [1. 1.]] relative pulse error [0.00636476 0.00712924 0.01352239]
averaged faces False upwind at seam True ratios [[1.        1.1371565]
 [1.        1.0806186]] relative pulse error [0.00462011 0.00402347 0.00265432]
averaged faces True upwind at seam False ratios [[1.035559 1.      ]
 [1.035559 1.      ]] relative pulse error [0.00222905 0.00324688 0.01116186]
averaged faces True upwind at seam True ratios [[1.035559  1.1371565]
 [1.035559  1.0806186]] relative pulse error [1.33095072e-05 1.49936514e-05 1.38246745e-05]
k and v endpoints [0.47110686 0.50461106 0.50461106 0.47110686] [ 0.35       -0.20572484]
```

### Turn 7 — `execute_in_remote_docker` (`48732 ms` LLM)
- **Rationale**: Implement the pulse-validated boundary law, compose it with the verified upstream calibration, and run the public schema gate before independently checking the forward field and KKT certificate.
- **Alternatives Weighed**: `Fit four endpoint factors to the pulse data` (rejected: The physical face and upwind formulas already match at the calibration residual scale and transfer across grids without fitted factors.)

```bash
set -e
cd /holdout_task
sed -n '/^def calibrate_upstream/,$p' _verified_upstream.py
cat validate_public_schema.py
cat > downstream_solver.py <<'PY'
"""Steady chamber inversion with cyclic x transport and zero-flux y walls."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, periodic):
    n = len(kappa)
    face = 0.5 * (kappa[:-1] + kappa[1:]) / h**2
    west = np.empty(n - 1)
    east = np.empty(n - 1)
    center = np.empty(n)
    west[:-1] = -face[:-1] - np.maximum(velocity[1:-1], 0) / h
    east[1:] = -face[1:] - np.maximum(-velocity[1:-1], 0) / h
    center[1:-1] = -west[:-1] - east[1:]
    if periodic:
        seam = 0.5 * (kappa[-1] + kappa[0]) / h**2
        east[0] = -face[0] - max(-velocity[0], 0) / h
        west[-1] = -face[-1] - max(velocity[-1], 0) / h
        wrap_left = -seam - max(velocity[0], 0) / h
        wrap_right = -seam - max(-velocity[-1], 0) / h
        center[0] = -east[0] - wrap_left
        center[-1] = -west[-1] - wrap_right
    else:
        east[0] = -kappa[0] / h**2
        west[-1] = -kappa[-1] / h**2
        center[0] = -east[0]
        center[-1] = -west[-1]
    operator = sparse.diags((west, center, east), (-1, 0, 1), format='csc')
    if periodic:
        operator += sparse.csc_matrix(([wrap_left, wrap_right], ([0, n-1], [n-1, 0])), shape=(n, n))
    return operator


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), True)
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), False)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    readings = []
    for mu in study['mu_shifts']:
        response = splu(operator + mu * eye).solve(rhs)
        readings.append(response[sensors])
    return np.concatenate(readings), response.T.reshape(-1, nx, ny, order='F')


PY
sed -n '/^def box_simplex/,/^def solve_batch/{ /^def solve_batch/!p; }' /tmp/hegel_continual_archive/beta_m100_c02_batch_01_solver.py >> downstream_solver.py
cat >> downstream_solver.py <<'PY'
def solve_downstream(cal_item: dict) -> dict:
    s, terminal = chamber_response(cal_item)
    y = np.asarray(cal_item['CALIBRATED_OBSERVATIONS']).ravel()
    z = np.asarray(cal_item['Z_nuisance'])
    sy = np.column_stack((s, y))
    gram_z = z.T @ z + cal_item['nuisance_ridge'] * np.eye(z.shape[1])
    sy -= z @ np.linalg.solve(gram_z, z.T @ sy)
    a, r = sy[:, :-1], sy[:, -1]
    ridge, cap, budget = (cal_item[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    h = a.T @ a + ridge * np.eye(a.shape[1])
    w, nu = box_simplex(h, a.T @ r, cap, budget)
    residual = a @ w - r
    g = a.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    obj = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = obj - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': float(obj), 'dual_lower_bound': float(dual)}}
PY
python3 - <<'PY'
import pickle
from pathlib import Path
import numpy as np
from _verified_upstream import calibrate_upstream
with open('public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
out = calibrate_upstream(study)
keys = [k for k in out if k not in study and k.startswith('y_obs')]
if not keys and not np.array_equal(out['y_obs_raw'], study['y_obs_raw']):
    keys = ['y_obs_raw']
assert len(keys) == 1, list(out)
p = Path('downstream_solver.py')
p.write_text(p.read_text().replace('CALIBRATED_OBSERVATIONS', keys[0]))
PY
cat > batch_solver.py <<'PY'
"""Compose the verified sensor calibration and chamber inversion."""
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream


def solve_batch(study: dict) -> dict:
    return solve_downstream(calibrate_upstream(study))
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`659 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py', '/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without modifying the input."""
    out = study.copy()
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coeff = _fit_forward(ref, raw)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coeff)
    if study.get('reference_tracer_pulses') is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coeff))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in study['reference_tracer_pulses']
        ]
    calibration = sweeps.copy()
    calibration['raw_rosette'] = sweeps['ref_rosette'].copy()
    banks = sweeps['single_bank_sweeps']
    calibration['single_bank_sweeps'] = dict(banks, raw_readout=banks['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = calibration
    return out
import copy, importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(copy.deepcopy(st))
m_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (m_m,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not isinstance(out.get("kkt_certificate"), dict): errs.append("missing_or_bad:kkt_certificate")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=47fade5c3d2a6f23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_47fade5c (directory).
```

### Turn 8 — `execute_in_remote_docker` (`42899 ms` LLM)
- **Rationale**: Run an isolated check against an independent dense stencil to catch seam and field-ordering errors, and stress the optimizer near the budget limits to catch invalid KKT certificates that the schema check cannot detect.

```bash
set -e
cat > /tmp/check_batch04.py <<'PY'
import sys, pickle
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, axis_operator, box_simplex
from _verified_upstream import calibrate_upstream

def dense_axis(k,v,h,periodic):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        if not periodic and i in (0,n-1):
            a[i,1 if i==0 else n-2] = -k[i]/h**2
        else:
            l,r=(i-1)%n,(i+1)%n
            a[i,l] += -(k[l]+k[i])/(2*h*h)-max(v[i],0)/h
            a[i,r] += -(k[r]+k[i])/(2*h*h)-max(-v[i],0)/h
        a[i,i] = -a[i].sum()
    return a

def audit(h,c,w,nu,cap,budget):
    g=h@w-c
    lo=np.maximum(g-nu,0); hi=np.maximum(nu-g,0)
    errors=np.array([abs(w.sum()-budget), max(0.,-w.min(),w.max()-cap), max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))), lo@w+hi@(cap-w)])
    assert errors.max()<1e-8, errors
    return errors

with open('/holdout_task/public_batch.pkl','rb') as f:
    items=pickle.load(f)
maxerrors=np.zeros(4)
for j,item in enumerate(items):
    st=item['study']; cal=calibrate_upstream(dict(st,reference_tracer_pulses=item.get('campaign_diagnostics')))
    nx,ny=st['nx'],st['ny']
    ax,ay=[dense_axis(st['kappa_'+d],st['vel_'+d],st['L'+d]/(st['n'+d]-1),d=='x') for d in 'xy']
    l=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(st['V_reaction_2d'].ravel(order='F'))
    rhs=np.stack([m.ravel(order='F') for m in st['source_modes']],axis=1)
    ij=st['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
    replies=[np.linalg.solve(l+mu*np.eye(nx*ny),rhs) for mu in st['mu_shifts']]
    s=np.concatenate([x[idx] for x in replies]); y=cal['y_obs_raw'].ravel(); z=st['Z_nuisance']
    p=np.eye(len(y))-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    a,b=p@s,p@y; rho=st['source_ridge']; cap,budget=st['u_box'],st['total_budget']
    h=a.T@a+rho*np.eye(a.shape[1]); c=a.T@b
    out=solve_batch(st); w=out['weights']; cert=out['kkt_certificate']; nu=cert['nu_sum']; g=h@w-c
    dn=solve_downstream(cal)
    assert np.array_equal(w,dn['weights'])
    np.testing.assert_array_equal(cert['weights'],w)
    np.testing.assert_allclose(cert['lam_lower'],np.maximum(g-nu,0),rtol=0,atol=1e-10)
    np.testing.assert_allclose(cert['lam_upper'],np.maximum(nu-g,0),rtol=0,atol=1e-10)
    expected_field=(replies[-1]@w).reshape(nx,ny,order='F')
    field_err=np.max(abs(out['terminal_field']-expected_field))
    assert field_err<1e-10, field_err
    obj=(np.linalg.norm(a@w-b)**2+rho*np.linalg.norm(w)**2)/2
    dual=obj-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    np.testing.assert_allclose([cert['primal_objective'],cert['dual_lower_bound']],[obj,dual],rtol=0,atol=1e-10)
    maxerrors=np.maximum(maxerrors,audit(h,c,w,nu,cap,budget))
    print('public study',j,'field L_inf',field_err,'primal-dual gap',obj-dual)
    for i,pulse in enumerate(cal.get('reference_tracer_pulses') or []):
        yp=pulse['y_pulse_raw'].ravel()
        err=np.linalg.norm(s@pulse['w_pulse']-yp)/np.linalg.norm(yp)
        assert err<3e-5, err
        print('boundary pulse',i,'relative L2',err)

rng=np.random.default_rng(1)
for n in (3,5,15):
    k=rng.uniform(.15,2,n); v=rng.uniform(-2,2,n)
    for periodic in (False,True):
        np.testing.assert_allclose(axis_operator(k,v,.21,periodic).toarray(),dense_axis(k,v,.21,periodic),atol=1e-13)
for j in range(350):
    n=int(rng.integers(3,19)); d=rng.normal(size=(n,n)); h=d.T@d+.04*np.eye(n); c=rng.normal(size=n)*rng.choice([.1,1,10])
    cap=.42; budget=(rng.choice([.00001,.0001,.4,.9999,.99999])*n*cap if j%7 else int(rng.integers(1,n))*cap)
    w,nu=box_simplex(h,c,cap,budget)
    maxerrors=np.maximum(maxerrors,audit(h,c,w,nu,cap,budget))
print('350 budget stress cases plus public: max sum error, box violation, complementarity, gap',maxerrors)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch04.py
```
- **Container Exit Code**: `0` (`820 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public study 0 field L_inf 1.3877787807814457e-16 primal-dual gap 9.055256544598933e-16
boundary pulse 0 relative L2 1.3309507174530808e-05
boundary pulse 1 relative L2 1.4993651423695204e-05
boundary pulse 2 relative L2 1.382467447122633e-05
350 budget stress cases plus public: max sum error, box violation, complementarity, gap [1.99840144e-15 0.00000000e+00 1.15300295e-15 3.34394972e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=47fade5c3d2a6f23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_47fade5c (directory).
```

### Turn 9 — `execute_in_remote_docker` (`13312 ms` LLM)
- **Rationale**: The host rejected the previous probe receipt despite numerical agreement. Run the untouched public validator and existing isolated boundary and KKT audit in a single read-only step. A schema failure would point to composition; a pulse failure would reopen the wall stencil.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_batch04.py
```
- **Container Exit Code**: `0` (`947 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public study 0 field L_inf 1.3877787807814457e-16 primal-dual gap 9.055256544598933e-16
boundary pulse 0 relative L2 1.3309507174530808e-05
boundary pulse 1 relative L2 1.4993651423695204e-05
boundary pulse 2 relative L2 1.382467447122633e-05
350 budget stress cases plus public: max sum error, box violation, complementarity, gap [1.99840144e-15 0.00000000e+00 1.15300295e-15 3.34394972e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=47fade5c3d2a6f23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_47fade5c (directory).
```

### Turn 10 — `submit_completed_task` (`14783 ms` LLM)
- **Rationale**: The unchanged candidate passed the public schema and isolated numerical checks, with a worst primal-dual gap of 3.34e-15 across the public and stress cases. The host has verified the artifact bundle but has not accepted the separate structural resolution receipt.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:261', '/holdout_task/downstream_solver.py': 'FILE:4947'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema and all 350 budget stress cases passed. Across three calibrated tracer pulses, the worst relative L2 error was 1.50e-5. The terminal field agreed with an independent dense solve to 1.39e-16 in maximum absolute error.`

### Turn 11 — `execute_in_remote_docker` (`37331 ms` LLM)
- **Rationale**: The full-pulse norm could hide a bad sensor rosette or a boundary residual that lies in the nuisance span. Test seam terms separately against real pulses, score held-out rosettes with nuisance coefficients fit on the remaining rows, and check that the transport stencil annihilates a constant field. If an ablation matches the exact stencil after nuisance removal, reopen the boundary identification.
- **Alternatives Weighed**: `Apply classification, relay, or stochastic drift checks` (rejected: The task is a steady linear forward solve with continuous observations from one campaign; it has no class labels, relay modes, or stochastic drift operator.)

```bash
set -e
cat > /tmp/audit_boundary_resolution.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import axis_operator, chamber_response

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
cal = calibrate_upstream(dict(st, reference_tracer_pulses=item['campaign_diagnostics']))
pulses = cal['reference_tracer_pulses']
w = np.column_stack([p['w_pulse'] for p in pulses])
y = np.column_stack([p['y_pulse_raw'] for p in pulses])
z = st['Z_nuisance']
nx, ny = st['nx'], st['ny']
hx, hy = (st['L'+a] / (st['n'+a] - 1) for a in 'xy')
k, v = st['kappa_x'], st['vel_x']
ax = axis_operator(k, v, hx, True).toarray()
ay = axis_operator(st['kappa_y'], st['vel_y'], hy, False).toarray()
row_error = max(np.max(np.abs(ax.sum(axis=1))), np.max(np.abs(ay.sum(axis=1))))
assert row_error < 1e-12
s, _ = chamber_response(cal)
full = s @ w
rhs = np.stack([m.ravel(order='F') for m in st['source_modes']], axis=1) @ w
ij = st['sensor_ij']
idx = ij[:, 0] + nx * ij[:, 1]
fixed = np.kron(ay, np.eye(nx)) + np.diag(st['V_reaction_2d'].ravel(order='F'))
shifts = st['mu_shifts']
eye = np.eye(nx * ny)
def predict(a):
    l = fixed + np.kron(np.eye(ny), a)
    return np.concatenate([np.linalg.solve(l + mu * eye, rhs)[idx] for mu in shifts])
np.testing.assert_allclose(predict(ax), full, rtol=0, atol=1e-12)
seam = (k[0] + k[-1]) / (2 * hx**2)
variants = {'exact': full}
for name in ('no_cross_seam', 'no_seam_upwind', 'no_endpoint_face_average'):
    a = ax.copy()
    for i, far, near in ((0, nx-1, 1), (nx-1, 0, nx-2)):
        if name == 'no_cross_seam':
            a[i, far] = 0
        elif name == 'no_seam_upwind':
            a[i, far] = -seam
        else:
            a[i, near] += (k[near] - k[i]) / (2 * hx**2)
        a[i, i] = 0
        a[i, i] = -a[i].sum()
    variants[name] = predict(a)
print('constant-field transport residual Linf:', row_error)
# Hold out each sensor rosette. Fit only the nuisance coefficients on the
# remaining rows; the boundary and calibrated forward response stay fixed.
for name, pred in variants.items():
    raw = pred - y
    corrected = np.empty_like(raw)
    for start in range(0, len(y), 4):
        test = np.arange(start, start+4)
        train = np.ones(len(y), dtype=bool)
        train[test] = False
        coef = np.linalg.lstsq(z[train], raw[train], rcond=None)[0]
        corrected[test] = raw[test] - z[test] @ coef
    def worst(error):
        return (np.linalg.norm(error.reshape(-1, 4, w.shape[1]), axis=1) / np.linalg.norm(y.reshape(-1, 4, w.shape[1]), axis=1)).max()
    print(name, 'worst rosette relative L2 raw:', worst(raw), 'after held-out nuisance removal:', worst(corrected))
    if name == 'exact':
        assert worst(raw) < 3e-4 and worst(corrected) < 3e-4
    else:
        assert worst(corrected) > 1e-3
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_boundary_resolution.py
```
- **Container Exit Code**: `0` (`522 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
constant-field transport residual Linf: 3.552713678800501e-15
exact worst rosette relative L2 raw: 5.452141682447867e-05 after held-out nuisance removal: 5.4309927138296195e-05
no_cross_seam worst rosette relative L2 raw: 0.46847951856836534 after held-out nuisance removal: 0.5633878716116334
no_seam_upwind worst rosette relative L2 raw: 0.01975914258955831 after held-out nuisance removal: 0.02665223576279248
no_endpoint_face_average worst rosette relative L2 raw: 0.0069993196874361665 after held-out nuisance removal: 0.007184993833360891

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=47fade5c3d2a6f23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_47fade5c (directory).
```

### Turn 12 — `execute_in_remote_docker` (`17487 ms` LLM)
- **Rationale**: The host rejected the structural receipt for the invocation that created the temporary audit script. Run the existing audit in isolation with bytecode writes disabled to obtain a read-only receipt on the unchanged candidate. If the audit fails, reopen boundary identification; if it passes, no solver change is warranted.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/audit_boundary_resolution.py
```
- **Container Exit Code**: `0` (`562 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
constant-field transport residual Linf: 3.552713678800501e-15
exact worst rosette relative L2 raw: 5.452141682447867e-05 after held-out nuisance removal: 5.4309927138296195e-05
no_cross_seam worst rosette relative L2 raw: 0.46847951856836534 after held-out nuisance removal: 0.5633878716116334
no_seam_upwind worst rosette relative L2 raw: 0.01975914258955831 after held-out nuisance removal: 0.02665223576279248
no_endpoint_face_average worst rosette relative L2 raw: 0.0069993196874361665 after held-out nuisance removal: 0.007184993833360891

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=47fade5c3d2a6f23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_47fade5c (directory).
```

### Turn 13 — `submit_completed_task` (`15368 ms` LLM)
- **Rationale**: The unchanged bundle passed the schema gate, independent dense field comparison, and isolated KKT stress tests. On real tracer pulses, removing the seam, seam upwinding, or endpoint face averaging each increased held-out error after nuisance adjustment, supporting the implemented stencil without further edits.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py` and `/holdout_task/downstream_solver.py`. The public schema and all 350 budget stress cases passed. The worst held-out sensor-rosette relative error was 5.43e-5 after nuisance adjustment, and the largest primal–dual gap was 3.34e-15.`

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
