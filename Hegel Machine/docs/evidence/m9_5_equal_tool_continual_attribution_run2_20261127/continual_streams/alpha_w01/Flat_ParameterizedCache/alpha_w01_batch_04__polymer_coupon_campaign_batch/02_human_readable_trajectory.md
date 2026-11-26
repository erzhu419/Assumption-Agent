# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w01_batch_04__polymer_coupon_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w01_step_04` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `366,553` (`354,092` / `96,896` / `12,461`) |
| **Total Episode Wall Time (s)** | `368.25s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_10_split_preloaded_hereditary_transfe`**: [BIFURCATED SUB-REGIME `observed_feature_fraction_before_imputat < 1`]: Two exponential increments alone left maximum strain and onset-rate residuals of 0.30575 and 0.50642; the working model retained feature-dependent linear drift and propagated the supplied preload through both memory branches. Instead of pooling lab offsets or freely fitting gains to sparse checks, estimate each gauge from at least three positive-gain consensus checks spanning 0.10 of the stress scale and forming a strict majority; otherwise transfer an explicit prior, with a two-point override only when both well-separated checks disagree with it. Align features against each lab’s reference panel, impute with equal weight per observed lab–stratum cell, and fit ordered relaxation times on complete training rows with separately MAD-scaled strain and onset-rate residuals; reject a dimensionless robust Jacobian singular-value ratio at or below 1e-8. For an incomplete prediction specimen, use a finite onset rate to correct only its linear drift, then report the first interpolated crossing of the sampled curve from the onset anchor: analytic roots had crossing RMSE 0.004088 even on the exact public trajectory, while sampled interpolation reproduced those labels to numerical precision. The public probe measured strain MAE 0.00316, relative relaxation-time error 0.00056, and zero panel-affine gauge difference; it did not establish cross-epoch parameter persistence or extrapolation to other constitutive regimes. (Counter-regime `observed_feature_fraction_before_imputat` warning: Below full feature coverage, a finite trusted onset rate identifies a specimen-specific drift correction; without that anchor the prediction retains the imputed drift. At full coverage the model retains feature-predicted drift and does not fit to the specimen’s prediction-time onset rate. This is a split in drift identifiability, not evidence for two material laws.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Two exponential increments alone span the observed hold trajectory.` (witness: `At the public reference relaxation times, independently fitted two-mode amplitudes leave maximum strain residual 0.30575 and maximum onset-rate residual 0.50642.`) -> Pivot `Retain the archived feature-dependent drift and propagate the supplied initial memory states through the two relaxation modes.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The public hazard labels are roots of the continuous constitutive trajectory.` (witness: `With the exact public trajectory, the analytic root has crossing RMSE 0.004088; first-segment linear interpolation anchored at t=0 reproduces the labels to 7.08e-17 RMSE.`) -> Pivot `Interpolate the first threshold crossing on the sampled curve, including the onset anchor.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15898 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver to recover the seven hereditary basis terms and identify which calibration checks are available before fitting or choosing validation slices.

```bash
cd /holdout_task && ls -lh && cat validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation'; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_alpha_w01_batch_04.pkl','rb') as f: d=pickle.load(f)
def scan(obj, depth=0):
 if isinstance(obj,dict):
  for k,v in obj.items():
   print('  '*depth,k, type(v).__name__, getattr(v,'shape',''), repr(v)[:200] if np.isscalar(v) else '')
   if isinstance(v,dict) and depth<2: scan(v,depth+1)
   if isinstance(v,(list,tuple)) and v: print('  '*(depth+1),len(v),repr(v[0])[:320])
scan(d)
PY
p=/tmp/hegel_continual_archive/alpha_w01_batch_03_solver.py; if test -f "$p"; then sed -n '1,260p' "$p"; fi
```
- **Container Exit Code**: `0` (`497 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0FJTlMgPSBucC5hcnJheShbMC45OTk5ODQ3OSwgMS4wMDAwNDAxNCwgMS4wMDAwMDQ0NSwgMC45OTk5ODU2NV0sIGR0eXBlPWZsb2F0KQpfQklBU0VTID0gbnAuYXJyYXkoWzMuMDUyZS0wNSwgLTYuNzllLTA2LCAtNS44NGUtMDYsIC00LjVlLTA3XSwgZHR5cGU9ZmxvYXQpCl9tb2QgPSB0eXBlcy5Nb2R1bGVUeXBlKCdfaGVnZWxfbTk1X2FscGhhX2JvdW5kJykKZXhlYyhjb21waWxlKGJhc2U2NC5iNjRkZWNvZGUoX1NSQ19CNjQpLmRlY29kZSgndXRmLTgnKSwgJzxhbHBoYV9ib3VuZD4nLCAnZXhlYycpLCBfbW9kLl9fZGljdF9fKQpfZm4gPSBnZXRhdHRyKF9tb2QsIF9TUkNfRVApCgpkZWYgc29sdmVfYWxwaGFfdzAxX2JhdGNoXzAyKGRhdGFzZXQ6IGRpY3QpIC0+IGRpY3Q6CiAgICBkczIgPSBjb3B5LmRlZXBjb3B5KGRhdGFzZXQpCiAgICBmb3Igc3BfayBpbiAoJ3RyYWluJywgJ3Rlc3QnKToKICAgICAgICBzcCA9IGRpY3QoZHMyW3NwX2tdKQogICAgICAgIGxiID0gbnAuYXNhcnJheShzcFsnbGFiX2lkJ10sIGR0eXBlPWludCkucmF2ZWwoKQogICAgICAgIHNwWydwYXN0X3N0cmVzc19jbWQnXSA9IG5wLmFzYXJyYXkoc3BbJ3Bhc3Rfc3RyZXNzX2NtZCddLCBkdHlwZT1mbG9hdCkgKiBfR0FJTlNbbGIsIE5vbmVdICsgX0JJQVNFU1tsYiwgTm9uZV0KICAgICAgICBzcFsnc3RyZXNzX2hvbGRfY21kJ10gPSBucC5hc2FycmF5KHNwWydzdHJlc3NfaG9sZF9jbWQnXSwgZHR5cGU9ZmxvYXQpICogX0dBSU5TW2xiXSArIF9CSUFTRVNbbGJdCiAgICAgICAgZHMyW3NwX2tdID0gc3AKICAgIHJlZl9jbWQgPSBucC50aWxlKG5wLmFycmF5KFstMS41LCAtMC41LCAwLjUsIDEuNV0sIGR0eXBlPWZsb2F0KSwgNCkKICAgIGRzMlsnY2FsaWJyYXRpb25fbG9hZF9jZWxscyddID0gewogICAgICAgICdsYWJfaWQnOiBucC5yZXBlYXQobnAuYXJhbmdlKDQsIGR0eXBlPWludCksIDQpLAogICAgICAgICdjbWRfc3RyZXNzJzogcmVmX2NtZCwKICAgICAgICAndHJ1ZV9zdHJlc3MnOiByZWZfY21kLmNvcHkoKSwKICAgIH0KICAgIHJldHVybiBfZm4oZHMyKQo='
_SRC_EP = 'solve_alpha_w01_batch_02'
_GAINS = np.array([0.75358184, 1.2377899, 0.70754455, 1.28023218], dtype=float)
_BIASES = np.array([-0.1397032, 0.09426679, -0.11031877, 0.13347379], dtype=float)
_mod = types.ModuleType('_hegel_m95_alpha_bound')
exec(compile(base64.b64decode(_SRC_B64).decode('utf-8'), '<alpha_bound>', 'exec'), _mod.__dict__)
_fn = getattr(_mod, _SRC_EP)

def solve_alpha_w01_batch_03(dataset: dict) -> dict:
    ds2 = copy.deepcopy(dataset)
    for sp_k in ('train', 'test'):
        sp = dict(ds2[sp_k])
        lb = np.asarray(sp['lab_id'], dtype=int).ravel()
        sp['past_stress_cmd'] = np.asarray(sp['past_stress_cmd'], dtype=float) * _GAINS[lb, None] + _BIASES[lb, None]
        sp['stress_hold_cmd'] = np.asarray(sp['stress_hold_cmd'], dtype=float) * _GAINS[lb] + _BIASES[lb]
        ds2[sp_k] = sp
    ref_cmd = np.tile(np.array([-1.5, -0.5, 0.5, 1.5], dtype=float), 4)
    ds2['calibration_load_cells'] = {
        'lab_id': np.repeat(np.arange(4, dtype=int), 4),
        'cmd_stress': ref_cmd,
        'true_stress': ref_cmd.copy(),
    }
    return _fn(ds2)
```

### Turn 2 — `execute_in_remote_docker` (`15300 ms` LLM)
- **Rationale**: The first log obscured the constitutive basis and load-cell rows. Read those selectively to distinguish sparse checks from EMI and recover the alignment and fitting equations without copying opaque archived wrappers.

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np,json
p=pickle.load(open('public_alpha_w01_batch_04.pkl','rb')); print('cases',len(p))
for z in p:
 d=z['dataset'];print('case',list(z),[(k,v) for k,v in z.items() if k!='dataset' and np.isscalar(v)])
 for k,v in d.items():
  if isinstance(v,dict): print(k,{a:np.shape(b) for a,b in v.items()});
  else: print(k,np.shape(v),str(v)[:120])
 c=d['calibration_load_cells']; print('checks',np.column_stack(list(c.values())).round(5).tolist())
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive',str(h)[:3000])
PY
sed -n '1,260p' /tmp/hegel_continual_archive/alpha_w01_batch_01_solver.py | awk 'length<400'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel | head -45
```
- **Container Exit Code**: `0` (`531 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
        "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
        "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
      },
      "functions": [
        {
          "function": "align_reference_panel",
          "signature": "align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray",
          "doc_summary": "Return aligned real values (n, p) using real controls (n, q) and finite real groups (n,), with nonempty columns and a supported positive control gain per group."
        }
      ],
      "index_cards": [
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_reference_panel_balancing",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`19811 ms` LLM)
- **Rationale**: Decode the archived equations as text without executing wrappers, and compare robust affine fits by lab and batch. This will determine which historical sweeps share the current calibration and whether ordinary least squares would absorb EMI outliers.

```bash
python3 - <<'PY'
import ast,base64,pathlib,json,pickle,numpy as np
root=pathlib.Path('/tmp/hegel_continual_archive')
for fn in ['alpha_w01_batch_01_solver.py','alpha_w01_s00_solver.py']:
 s=(root/fn).read_text(); depth=0
 while True:
  vals={n.targets[0].id: n.value.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
  if '_SRC_B64' not in vals: break
  depth+=1;s=base64.b64decode(vals['_SRC_B64']).decode()
 print(fn,'wrapper depth',depth); print(s[:14000] if 's00' in fn else '\n'.join(x for x in s.splitlines() if 'def ' in x or 'calib' in x.lower()))
p=pickle.load(open('/holdout_task/public_alpha_w01_batch_04.pkl','rb'))[0];d=p['dataset'];print('public truth fields (reserved for verification)',list(p['gt']));
h=json.load(open(root/'archive_calibration_history.json')); c=d['calibration_load_cells'];
for z in h+[dict(c,episode_short_name='current')]:
 a,g,b=[np.asarray(z[k]) for k in ('lab_id','cmd_stress','true_stress')];print(z['episode_short_name'])
 for i in range(4):
  x,y=g[a==i],b[a==i]; ix,jx=np.triu_indices(len(x),1); diff=x[ix]-x[jx]; ok=abs(diff)>1e-6;sl=np.median((y[ix]-y[jx])[ok]/diff[ok]);m=np.median(y-sl*x);print(i,len(x),np.round([sl,m],7),np.sort(abs(y-sl*x-m)).round(6))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Prony hereditary relaxation variable projection' | cut -c1-245 | head -12
```
- **Container Exit Code**: `0` (`589 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.52e-04
 1.93e-04 2.16e-04]
alpha_w01_batch_03
0 10 [ 0.7535654 -0.1396748] [2.60e-05 2.60e-05 2.90e-05 4.00e-05 6.50e-05 1.09e-04 1.12e-04 1.12e-04
 1.22e-04 1.83e-04]
1 10 [1.2377843 0.0942787] [1.40e-05 1.40e-05 1.90e-05 3.40e-05 3.40e-05 3.80e-05 4.20e-05 7.40e-05
 1.86e-04 2.32e-04]
2 10 [ 0.7075702 -0.1103846] [2.10e-05 2.10e-05 3.00e-05 3.00e-05 5.20e-05 6.20e-05 6.60e-05 1.52e-04
 1.90e-04 2.38e-04]
3 10 [1.2802308 0.1334663] [2.00e-06 2.00e-06 2.00e-06 1.20e-05 3.90e-05 6.10e-05 9.00e-05 1.10e-04
 1.11e-04 1.49e-04]
current
0 10 [ 0.7536218 -0.1396894] [1.00e-05 1.00e-05 1.00e-05 2.00e-05 3.10e-05 4.60e-05 1.07e-04 1.23e-04
 1.35e-04 2.35e-04]
1 10 [1.2378279 0.0942879] [0.00e+00 0.00e+00 4.00e-06 4.80e-05 5.50e-05 6.90e-05 1.26e-04 1.31e-04
 1.34e-04 1.71e-04]
2 10 [ 0.707505  -0.1102435] [3.60e-05 3.60e-05 3.60e-05 4.00e-05 4.30e-05 4.80e-05 5.80e-05 6.50e-05
 9.50e-05 1.62e-04]
3 10 [1.2802314 0.1334896] [1.00e-06 1.00e-06 1.00e-06 2.00e-06 1.30e-05 3.80e-05 8.80e-05 9.10e-05
 9.80e-05 1.76e-04]
autogenous_hereditary_cohort_transfer@f2c729ba | fit_predict_hereditary(history: np.ndarray, hold: np.ndarray, durations: np.ndarray, times: np.ndarray, initial_strain: np.ndarray, features: np.ndarray, complete: np.ndarray, training: np.ndarray
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, floa
autogenous_preloaded_hereditary_transfer@14a7603e | fit_predict_preloaded(history: np.ndarray, hold: np.ndarray, durations: np.ndarray, times: np.ndarray, preload: np.ndarray, initial_strain: np.ndarray, features: np.ndarray, complete: np.ndarra
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarra
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ:
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to |
```

### Turn 4 — `execute_in_remote_docker` (`19638 ms` LLM)
- **Rationale**: Recover the exact hereditary predictors and their instantaneous derivatives from the unobscured source so the new solver can extend them without changing the baseline law. Compare the test-only plastic state with the specified history integral before deciding whether it is redundant.
- **Alternatives Weighed**: `Pool all archived sweeps with the current checks` (rejected: Batches through 02 used near-identity maps, whereas batch 03 agrees with the current nonidentity maps.)

```bash
p=/tmp/hegel_continual_archive/alpha_w01_s00_solver.py; grep -n '^def ' "$p"; sed -n '/^def _alignment/,$p' "$p" | head -170; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_hereditary_cohort_transfer.fit_predict_hereditary | grep -E 'yield|preload|rate' | cut -c1-220 | head -12; python3 - <<'PY'
import pickle,numpy as np
z=pickle.load(open('/holdout_task/public_alpha_w01_batch_04.pkl','rb'))[0];d=z['dataset'];print('truth keys, not values:',list(z['gt']))
for name in ('train','test'):
 s=d[name];print(name,'missing counts',np.isnan(s['coupon_features_raw']).sum(axis=0),'cells',np.unique(np.column_stack((s['lab_id'],s['stratum'])),axis=0,return_counts=True)[1]);print('rate sd',np.std(s['instant_rate_obs']));
 if 'residual_plastic_state' in s:
  p=np.maximum(abs(s['past_stress_cmd']*np.array([.75362,1.23783,.70751,1.28023])[s['lab_id'],None]+np.array([-.13969,.09429,-.11024,.13349])[s['lab_id'],None])-d['yield_stress'],0)**2 @ d['past_dt'];r=s['residual_plastic_state'];print('plastic field vs computed work: rms',np.sqrt(np.mean((p-r)**2)),'work range',p.min(),p.max())
PY
```
- **Container Exit Code**: `0` (`639 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
y in (
            'lab_id', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0',
            'strain_curves_obs', 'instant_rate_obs')}
        taus, w = _fit(fit_train, x[complete], maps, dt, time)
    else:
        taus, w = _fit(train, x, maps, dt, time)
    hist, hold = _stress(test, maps)
    cov = np.column_stack((np.ones(len(xt)), xt))
    a, b = _design(taus, hist, hold, cov, dt, time)
    rate = b @ w
    # An onset-rate residual identifies the unobserved linear drift, not memory.
    observed = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    informed = missing & np.isfinite(observed)
    drift_delta = np.where(informed, observed - rate, 0.)
    return {'estimated_taus': taus, 'hereditary_weights': w,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + a @ w + drift_delta[:, None] * time,
            'initial_rate_pred': rate + drift_delta}
          "field": "maximum_absolute_calibrated_stress_over_yield_stress",
          "signature": "fit_predict_hereditary(history: np.ndarray, hold: np.ndarray, durations: np.ndarray, times: np.ndarray, initial_strain: np.ndarray, features: np.ndarray, complete: np.ndarray, training: np.ndarray
          "doc_summary": "Return physical relaxation times, weights, strain curves and onset rates."
        "autogenous_hereditary_cohort_transfer@f2c729ba | calibrate_affine_gauges(command: np.ndarray, measured: np.ndarray, check_group: np.ndarray, groups: np.ndarray, *, stress_scale: float, prior: np.ndarray | None=N
        "autogenous_hereditary_cohort_transfer@f2c729ba | align_reference_features(raw: np.ndarray, reference: np.ndarray, group: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Return reference-align
        "autogenous_hereditary_cohort_transfer@f2c729ba | fit_predict_hereditary(history: np.ndarray, hold: np.ndarray, durations: np.ndarray, times: np.ndarray, initial_strain: np.ndarray, features: np.ndarray, complete
truth keys, not values: ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_cross', 'te_secant', 'te_hold_true', 'te_hist_true']
train missing counts [5 6 1 3] cells [29 16 13  9 21 16 12 15 23 13 12 21]
rate sd 2.586546762835197
test missing counts [2 3 3 0] cells [ 7  6  7 10  6  5  3  1 13  9  8  5]
rate sd 2.5240571549149498
plastic field vs computed work: rms 2.340548487029827 work range 0.0 11.612487343044418
```

### Turn 5 — `execute_in_remote_docker` (`69015 ms` LLM)
- **Rationale**: Implement the recovered hereditary basis and both prescribed extensions, pool only compatible calibration checks, and first verify the executable output contract before measuring held-out prediction and parameter errors.
- **Alternatives Weighed**: `Substitute the test-only residual plastic state for the history integral` (rejected: It differs from the specified integral by 2.34 RMS on the public data.)

```bash
cat > /holdout_task/alpha_w01_batch_04_solver.py <<'PY'
"""Calibrated two-branch creep model with hold and history overstress."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
# Batch 03 load-cell fits, retained for standalone use if the sweep is not mounted.
_PRIOR = np.array([[.75358184, -.13970320], [1.23778990, .09426679],
                   [.70754455, -.11031877], [1.28023218, .13347379]])


def _checks(panel, lab):
    x = np.asarray(panel['cmd_stress'], float)
    y = np.asarray(panel['true_stress'], float)
    ok = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(panel):
    history = json.loads(_ARCHIVE.read_text())[-1] if _ARCHIVE.is_file() else None
    maps = _PRIOR.copy()
    for lab, prior in enumerate(_PRIOR):
        x, y = _checks(panel, lab)
        a, b = np.triu_indices(len(x), 1)
        dx = x[b] - x[a]
        good = np.abs(dx) >= .10
        a, b, dx = a[good], b[good], dx[good]
        gains = (y[b] - y[a]) / dx
        fits = np.vstack((prior, np.column_stack((gains, y[a] - gains*x[a]))))
        er = np.abs(y - fits[:, :1]*x - fits[:, 1:])
        clean = er < .001
        n = clean.sum(1)
        score = np.sum(np.where(clean, er**2, 0), axis=1)
        best = np.lexsort((score, -n))[0]
        keep = clean[best]
        agrees = np.abs(y - prior[0]*x - prior[1]) < .001
        if n[best] >= 3 and np.ptp(x[keep]) >= .10:
            local = _line(x[keep], y[keep])
            # Pool only checks that support the same calibration regime.
            if np.max(np.abs(local - prior)) > .001:
                maps[lab] = local
                continue
            agrees = keep
        if history is not None:
            hx, hy = _checks(history, lab)
            hist_ok = np.abs(hy - prior[0]*hx - prior[1]) < .001
            xx, yy = np.r_[hx[hist_ok], x[agrees]], np.r_[hy[hist_ok], y[agrees]]
            if len(xx) >= 2 and np.ptp(xx) >= .10:
                maps[lab] = _line(xx, yy)
        elif n[best] >= 3 and np.ptp(x[keep]) >= .10:
            maps[lab] = local
        elif np.any(agrees):
            maps[lab, 1] = np.median(y[agrees] - prior[0]*x[agrees])
    return maps


def _align(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    complete = np.isfinite(raw).all(axis=1)
    for lab in np.unique(labs):
        m = labs == lab
        p = refs[m]
        p = p[:, np.isfinite(p).sum(axis=0) >= 2]
        raw[m] = (raw[m] - np.median(np.nanmean(p, axis=0))) / np.median(np.nanstd(p, axis=0))
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    valid = np.isfinite(raw)
    counts = np.zeros((cell.max()+1, raw.shape[1]))
    sums = np.zeros_like(counts)
    np.add.at(counts, cell, valid)
    np.add.at(sums, cell, np.where(valid, raw, 0.))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / np.maximum(1, (counts > 0).sum(axis=0))
    x = np.column_stack((np.ones(len(raw)), np.where(valid, raw, fill)))
    n = len(train['lab_id'])
    return x[:n], x[n:], complete[:n], complete[n:]


def _stress(block, maps):
    gain, bias = maps[np.asarray(block['lab_id'], int)].T
    return block['past_stress_cmd'] * gain[:, None] + bias[:, None], block['stress_hold_cmd'] * gain + bias


def _plastic(hist, hold, dt, yield_stress):
    ov = np.maximum(np.abs(hold) - yield_stress, 0.)**2 * np.sign(hold)
    work_hold = (np.maximum(np.abs(hist) - yield_stress, 0.)**2 @ dt) * hold
    return ov, work_hold


def _design(taus, hist, hold, cov, dt, time, ov, work_hold):
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-tail[:, None]/taus) * -np.expm1(-dt[:, None]/taus)
    jump = hold[:, None] - hist @ memory
    response = -np.expm1(-time[:, None]/taus)
    curve = np.concatenate((jump[:, None]*response, cov[:, None]*time[None, :, None],
                            (ov[:, None]*time)[..., None],
                            (work_hold[:, None]*response[:, 0])[..., None]), axis=2)
    rate = np.column_stack((jump/taus, cov, ov, work_hold/taus[0]))
    return curve, rate


def _mad(e):
    return max(1e-8, 1.4826*np.median(np.abs(e-np.median(e))))


def _fit(train, x, complete, maps, dt, time, sy):
    hist, hold = _stress(train, maps)
    ov, work = _plastic(hist, hold, dt, sy)
    # The onset-rate observation identifies missing linear drift at prediction;
    # estimate shared coefficients from measured covariates when possible.
    if complete.sum() < 12:
        complete = np.ones_like(complete)
    data = (hist[complete], hold[complete], x[complete], dt, time, ov[complete], work[complete])
    dy = (train['strain_curves_obs'] - train['strain_0'][:, None])[complete].ravel()
    rates = train['instant_rate_obs'][complete]
    yfull = np.r_[dy, rates]
    valid = np.isfinite(yfull)
    is_curve = (np.arange(len(yfull)) < len(dy))[valid]
    y = yfull[valid]
    active = np.r_[np.ones(7, bool), np.any(data[-2]), np.any(data[-1])]
    scale = np.ones_like(y)
    robust = np.ones_like(y)

    def unpack(p):
        fast = np.exp(p[0]); slow = fast + (24.999-fast)*p[1]
        return np.array([fast, slow])

    def profile(p, result=False):
        tau = unpack(p)
        curve, rate = _design(tau, *data)
        A = np.concatenate((curve.reshape(-1, 9), rate), axis=0)[valid][:, active]
        w = robust/scale
        coeff = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
        err = np.einsum('ij,j->i', A, coeff) - y
        if result:
            full = np.zeros(9)
            full[active] = coeff
            return tau, full, err
        return err*w

    bounds = ([np.log(.05001), 1e-5], [np.log(24.), .99999])
    starts = [(a, (b-a)/(24.999-a)) for a, b in ((.2, 1.5), (.6, 4.), (1.5, 9.), (3., 16.))]
    opts = [least_squares(profile, [np.log(a), b], bounds=bounds, max_nfev=60) for a, b in starts]
    p = min(opts, key=lambda fit: fit.cost).x
    e = profile(p, True)[2]
    for m in (is_curve, ~is_curve):
        if np.any(m):
            scale[m] = _mad(e[m])
    for _ in range(6):
        robust[:] = 1/np.sqrt(1 + (e/(2.5*scale))**2)
        p = least_squares(profile, p, bounds=bounds, max_nfev=60).x
        taus, theta, e = profile(p, True)
    return taus, theta


def solve_alpha_w01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], float)
    time = np.asarray(dataset['eval_times'], float)
    sy = float(dataset['yield_stress'])
    maps = _calibrate(dataset['calibration_load_cells'])
    x, xt, complete, complete_test = _align(train, test)
    taus, theta = _fit(train, x, complete, maps, dt, time, sy)
    hist, hold = _stress(test, maps)
    ov, work = _plastic(hist, hold, dt, sy)
    A, B = _design(taus, hist, hold, xt, dt, time, ov, work)
    rate = B @ theta
    observed = test['instant_rate_obs']
    drift_delta = np.where(~complete_test & np.isfinite(observed), observed-rate, 0.)
    return {'estimated_taus': taus, 'hereditary_weights': theta[:7],
            'viscoplastic_weights': theta[7:], 'initial_rate_pred': rate + drift_delta,
            'predicted_strain': test['strain_0'][:, None] + A @ theta + drift_delta[:, None]*time}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed2ce7267ea2a74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed2ce72 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`28714 ms` LLM)
- **Rationale**: Compare predictions with public reference values by lab, yield regime, and feature completeness, and refit with each lab’s outcomes held out. Excess error in a specific slice would distinguish alignment or missing-drift problems from a shared constitutive fit error.

```bash
cat > /tmp/b04_eval.py <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w01_batch_04_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
z=pickle.load(open(p+'public_alpha_w01_batch_04.pkl','rb'))[0]; d=z['dataset']; g=z['gt']
o=s.solve_alpha_w01_batch_04(d)
for ok,gk in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
 a,b=o[ok],np.asarray(g[gk]); print(ok,'fit',a.round(6),'truth',b.round(6),'max abs error',np.max(abs(a-b)))
rms=lambda x: np.sqrt(np.mean(np.asarray(x)**2))
st=d['test']; dt=d['past_dt']; t=d['eval_times']; sy=d['yield_stress']; h,hold=s._stress(st,s._calibrate(d['calibration_load_cells'])); ov,w=s._plastic(h,hold,dt,sy)
missing=~np.isfinite(st['coupon_features_raw']).all(1)
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; print('public total curve/rate RMS',rms(e),rms(r),'first/last curve RMS',rms(e[:,0]),rms(e[:,-1]))
for key,m in [(f'lab {i}',st['lab_id']==i) for i in range(4)]+[('complete',~missing),('missing',missing),('hold only',(ov!=0)&(w==0)),('history only',(ov==0)&(w!=0)),('hold and history',(ov!=0)&(w!=0)),('neither',(ov==0)&(w==0))]:
 if m.any(): print(key,m.sum(),'curve/rate RMS',rms(e[m]),rms(r[m]))
print('test onset observation noise RMS',rms(st['instant_rate_obs']-g['te_true_rate0']))
tr=d['train']; lab=tr['lab_id']; fullx,_,complete,_=s._align(tr,st)
for j in range(4):
 a=lab!=j; b=~a
 ds=dict(d,train={k:v[a] for k,v in tr.items()},test={k:v[b] for k,v in tr.items() if k!='strain_curves_obs'})
 # Preserve the original reference-panel population by passing the original test
 # coupons as unlabeled controls; only measured curves/rates from other labs fit the law.
 maps=s._calibrate(d['calibration_load_cells']); tau,theta=s._fit(ds['train'],fullx[a],complete[a],maps,dt,t,sy)
 hist,hold=s._stress(ds['test'],maps); A,B=s._design(tau,hist,hold,fullx[b],dt,t,*s._plastic(hist,hold,dt,sy))
 rate=B@theta; delta=np.where(~complete[b],tr['instant_rate_obs'][b]-rate,0.)
 pred=tr['strain_0'][b,None]+A@theta+delta[:,None]*t
 print('OOF lab',j,'curve RMS',rms(pred-tr['strain_curves_obs'][b]),'complete rate RMS',rms((rate-tr['instant_rate_obs'][b])[complete[b]]),'taus',tau.round(5),'plastic',theta[7:].round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b04_eval.py
```
- **Container Exit Code**: `0` (`773 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus fit [0.487437 4.252277] truth [0.488 4.211] max abs error 0.04127724786386633
hereditary_weights fit [ 0.700985  0.526622  0.079309  0.151943 -0.131362  0.110802 -0.090977] truth [ 0.7   0.52  0.08  0.15 -0.13  0.11 -0.09] max abs error 0.00662196197839815
viscoplastic_weights fit [0.249213 0.169606] truth [0.25 0.17] max abs error 0.0007870988416182889
public total curve/rate RMS 0.01700643841433301 0.006793061735397527 first/last curve RMS 0.0015755119181596366 0.027920922474416743
lab 0 20 curve/rate RMS 0.017476905767577475 0.006234507692933137
lab 1 21 curve/rate RMS 0.007945312622368158 0.0048605465081677564
lab 2 17 curve/rate RMS 0.01264184400423038 0.005443362207572367
lab 3 22 curve/rate RMS 0.024295629532143898 0.009328443478760776
complete 73 curve/rate RMS 0.016454841226260078 0.006727892369346316
missing 7 curve/rate RMS 0.02194767940611703 0.007438739435527018
hold and history 40 curve/rate RMS 0.01970736672256019 0.007090040363923133
neither 40 curve/rate RMS 0.013786137673167572 0.006482492045485661
test onset observation noise RMS 0.009928770728810685
OOF lab 0 curve RMS 0.2580699717057997 complete rate RMS 0.011762159340286965 taus [0.48787 4.15019] plastic [0.25086 0.16985]
OOF lab 1 curve RMS 0.2969083937803172 complete rate RMS 0.009760603984654127 taus [0.4873  4.33033] plastic [0.24789 0.16953]
OOF lab 2 curve RMS 0.2907622988551382 complete rate RMS 0.011120590447573893 taus [0.48696 4.18384] plastic [0.24983 0.16939]
OOF lab 3 curve RMS 0.2675046387516075 complete rate RMS 0.01478623369569087 taus [0.48802 4.31744] plastic [0.24863 0.16986]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed2ce7267ea2a74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed2ce72 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`35409 ms` LLM)
- **Rationale**: Check whether the held-out observation error reflects sparse large residuals or a systematic horizon bias; only the latter would justify revisiting the fit. Separately withhold or corrupt calibration checks on each wing and isolate the threshold terms to detect failures the clean public case cannot expose.

```bash
cat > /tmp/b04_stress.py <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w01_batch_04_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(p+'public_alpha_w01_batch_04.pkl','rb'))[0]['dataset']
tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; sy=d['yield_stress']; c=d['calibration_load_cells']
x,_,complete,_=s._align(tr,te); maps=s._calibrate(c)
h,hold=s._stress(tr,maps); ov,w=s._plastic(h,hold,dt,sy)
for j in range(4):
    b=tr['lab_id']==j; a=~b
    tau,theta=s._fit({k:v[a] for k,v in tr.items()},x[a],complete[a],maps,dt,t,sy)
    A,B=s._design(tau,h[b],hold[b],x[b],dt,t,ov[b],w[b]); r=B@theta
    e=tr['strain_0'][b,None]+A@theta-tr['strain_curves_obs'][b]; re=r-tr['instant_rate_obs'][b]
    masks=[('all',np.ones(b.sum(),bool)),('complete',complete[b]),('missing',~complete[b]),('below',(ov[b]==0)&(w[b]==0)),('above',(ov[b]!=0)|(w[b]!=0))]
    print('lab',j)
    for name,m in masks:
        if m.any():
            err=e[m]-np.where(complete[b][m],0,re[m])[:,None]*t
            print(name,int(m.sum()),'curve abs p50/90/95/99',np.round(np.quantile(abs(err),[.5,.9,.95,.99]),5),'rate abs p50/90/95',np.round(np.quantile(abs(re[m]),[.5,.9,.95]),5))
    print('signed median by horizon',np.round(np.median(e[complete[b]],axis=0),5))
for name, labs, retain, glitch in [('sparse A',(0,1),1,0),('sparse B',(2,3),1,0),('EMI A',(0,1),10,3),('EMI B',(2,3),10,3),('sparse+EMI B',(2,3),2,1)]:
    keep=np.ones(len(c['lab_id']),bool); y=c['true_stress'].copy()
    for j in labs:
        i=np.flatnonzero(c['lab_id']==j); keep[i[retain:]]=False; y[i[:glitch]]+=np.linspace(.3,.9,glitch)
    cand=s._calibrate(dict(lab_id=c['lab_id'][keep],cmd_stress=c['cmd_stress'][keep],true_stress=y[keep]))
    out=s.solve_alpha_w01_batch_04(dict(d,calibration_load_cells=dict(lab_id=c['lab_id'][keep],cmd_stress=c['cmd_stress'][keep],true_stress=y[keep])))
    print(name,'max calibration shift',np.max(abs(cand-maps)),'taus',out['estimated_taus'].round(6),'vp',out['viscoplastic_weights'].round(6))
    assert np.max(abs(cand-maps)) < .001
    assert np.isfinite(out['predicted_strain']).all()
sub=dict(d,yield_stress=1e6)
out=s.solve_alpha_w01_batch_04(sub)
assert np.array_equal(out['viscoplastic_weights'],np.zeros(2))
# Isolate the two threshold mechanisms; check their independent onset derivatives.
h2=np.array([[0.]*len(dt),[2*sy]*len(dt),[0.]*len(dt)])
hold2=np.array([2*sy,.5*sy,-2*sy]); o2,w2=s._plastic(h2,hold2,dt,sy)
assert np.array_equal(o2!=0,[True,False,True]) and np.array_equal(w2!=0,[False,True,False]) and o2[-1]<0
T=np.array([.5,4.2]); eps=1e-7
A,B=s._design(T,h2,hold2,np.ones((3,5)),dt,np.array([eps]),o2,w2)
np.testing.assert_allclose(A[:,0,7:]/eps,B[:,7:],rtol=1e-6,atol=1e-10)
print('plastic zero-activation, sign, and onset-derivative checks passed')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b04_stress.py
```
- **Container Exit Code**: `0` (`1201 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[0.01138 0.77238 0.92769 1.13707] rate abs p50/90/95 [0.00972 0.02057 0.02294]
above 25 curve abs p50/90/95/99 [0.01174 0.0772  0.87473 1.16348] rate abs p50/90/95 [0.00491 0.01319 0.0324 ]
signed median by horizon [-0.00255 -0.00129  0.00196  0.00433  0.00441  0.00172  0.00113  0.00775
  0.0078   0.00922]
lab 2
all 50 curve abs p50/90/95/99 [0.00831 0.03    0.94659 1.12844] rate abs p50/90/95 [0.00916 0.02528 0.06847]
complete 46 curve abs p50/90/95/99 [0.00831 0.03122 0.94659 1.12536] rate abs p50/90/95 [0.00818 0.01647 0.02436]
missing 4 curve abs p50/90/95/99 [0.00788 0.01904 0.06688 1.07499] rate abs p50/90/95 [0.1112  0.14748 0.14827]
below 25 curve abs p50/90/95/99 [0.00912 0.75749 0.98254 1.14964] rate abs p50/90/95 [0.00918 0.0157  0.02327]
above 25 curve abs p50/90/95/99 [0.0076  0.0265  0.89382 1.08166] rate abs p50/90/95 [0.00664 0.04386 0.07412]
signed median by horizon [ 0.00396  0.0012   0.00177 -0.00387 -0.00257 -0.00283 -0.00325 -0.00479
 -0.00446 -0.00484]
lab 3
all 46 curve abs p50/90/95/99 [0.01672 0.09443 0.85624 1.15243] rate abs p50/90/95 [0.01081 0.02616 0.02789]
complete 44 curve abs p50/90/95/99 [0.01613 0.09897 0.85624 1.15292] rate abs p50/90/95 [0.01081 0.02572 0.02794]
missing 2 curve abs p50/90/95/99 [0.03114 0.0575  0.10687 0.8456 ] rate abs p50/90/95 [0.01653 0.02408 0.02503]
below 27 curve abs p50/90/95/99 [0.01512 0.07345 0.83789 1.14535] rate abs p50/90/95 [0.00864 0.02155 0.02381]
above 19 curve abs p50/90/95/99 [0.02025 0.10932 0.85502 1.18853] rate abs p50/90/95 [0.01106 0.02805 0.0287 ]
signed median by horizon [-0.00103  0.00272  0.00248 -0.00091 -0.00138  0.00077 -0.00266 -0.00163
  0.0042   0.00205]
sparse A max calibration shift 2.6853108518576363e-05 taus [0.487448 4.251581] vp [0.249249 0.169627]
sparse B max calibration shift 3.8091004855786426e-05 taus [0.487442 4.250914] vp [0.249214 0.169598]
EMI A max calibration shift 1.1540710952620548e-05 taus [0.48744  4.252277] vp [0.249216 0.169609]
EMI B max calibration shift 1.2953449516017734e-05 taus [0.487436 4.253029] vp [0.249211 0.16961 ]
sparse+EMI B max calibration shift 3.820591357010794e-05 taus [0.487441 4.251021] vp [0.249213 0.169599]
plastic zero-activation, sign, and onset-derivative checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed2ce7267ea2a74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed2ce72 (directory).
```

### Turn 8 — `submit_completed_task` (`12730 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation and targeted stress tests. Sparse and corrupted checks shifted calibration coefficients by at most 0.000039. Large held-out curve residuals were concentrated in the tail rather than showing a consistent bias across horizons.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_w01_batch_04_solver.py': 'FILE:7663'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_batch_04_solver.py`. The public schema passed. The solver remained stable under sparse and EMI-corrupted calibration checks on both wings, and both plastic terms passed activation and onset-derivative checks.`

### Turn 9 — `execute_in_remote_docker` (`32711 ms` LLM)
- **Rationale**: Test all nine basis columns against the onset-rate identity on measured histories, then check signed late-horizon residuals within each held-out lab and yield regime. A consistent bias would justify revisiting the law; a consistent missing-feature improvement would justify changing how strongly the solver uses measured onset rates.
- **Alternatives Weighed**: `Test relay states, classification balance, or unrelated drift operators` (rejected: The dataset contains continuous creep measurements and no relay state or classification target; lab-by-yield residuals are the relevant test for confounding.)

```bash
cat > /tmp/b04_margin.py <<'PY'
import runpy
import numpy as np
m = runpy.run_path('/tmp/b04_eval.py')
s,d,g,o = (m[k] for k in ('s','d','g','o'))
tr,te = d['train'],d['test']
dt,t,sy = (d[k] for k in ('past_dt','eval_times','yield_stress'))
maps = s._calibrate(d['calibration_load_cells'])
x,xt,complete,ct = s._align(tr,te)

def rmse(a):
    return np.sqrt(np.mean(a*a))

def audit(block, x, complete, tau, theta, target, tag):
    h,hold = s._stress(block,maps)
    ov,work = s._plastic(h,hold,dt,sy)
    A,B = s._design(tau,h,hold,x,dt,t,ov,work)
    E = block['strain_0'][:,None]+A@theta-target
    rate_error = B@theta-block['instant_rate_obs']
    E -= np.where(complete,0,rate_error)[:,None]*t
    for name,b in [('below',(ov==0)&(work==0)),('above',(ov!=0)|(work!=0))]:
        if b.any():
            tail = E[b,-1]
            print(tag,name,int(b.sum()),'tail signed median',round(float(np.median(tail)),6),'tail median abs',round(float(np.median(abs(tail))),6),'curve rmse',round(float(rmse(E[b])),6),'first/last median abs',np.round(np.median(abs(E[b][:,[0,-1]]),axis=0),6))
    if (~complete).any():
        for w in (0,.5,1):
            e = E[~complete]+(1-w)*rate_error[~complete,None]*t
            print(tag,'missing drift fraction',w,'curve median abs',round(float(np.median(abs(e))),6),'last median abs',round(float(np.median(abs(e[:,-1]))),6))
    return A,B

for j in range(4):
    b=tr['lab_id']==j
    tau,theta=s._fit({k:v[~b] for k,v in tr.items()},x[~b],complete[~b],maps,dt,t,sy)
    audit({k:v[b] for k,v in tr.items()},x[b],complete[b],tau,theta,tr['strain_curves_obs'][b],f'OOF lab {j}')
tau=o['estimated_taus']; theta=np.r_[o['hereditary_weights'],o['viscoplastic_weights']]
A,B=audit(te,xt,ct,tau,theta,g['te_clean'],'public')
# At t=0 both memories have zero displacement. Check every rate column on the measured stress histories.
h,hold=s._stress(te,maps)
args=(h,hold,xt,dt)
ov,work=s._plastic(h,hold,dt,sy)
eps=1e-6
Z,_=s._design(tau,*args,np.array([0.,eps,eps/2]),ov,work)
np.testing.assert_array_equal(Z[:,0],np.zeros_like(B))
np.testing.assert_allclose((4*Z[:,2]-Z[:,1])/eps,B,rtol=1e-8,atol=1e-9)
print('all nine basis columns: zero displacement and independent onset derivative passed')
print('calibration max gain/bias abs error',np.max(abs(maps-np.column_stack((g['stress_gains'],g['stress_biases']))),axis=0))
print('public output max curve/rate abs error',np.max(abs(o['predicted_strain']-g['te_clean'])),np.max(abs(o['initial_rate_pred']-g['te_true_rate0'])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b04_margin.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1336 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.007344 0.01464 ]
OOF lab 1 above 25 tail signed median 0.016067 tail median abs 0.022763 curve rmse 0.286229 first/last median abs [0.007568 0.022763]
OOF lab 1 missing drift fraction 0 curve median abs 0.169892 last median abs 0.456355
OOF lab 1 missing drift fraction 0.5 curve median abs 0.091718 last median abs 0.224822
OOF lab 1 missing drift fraction 1 curve median abs 0.042436 last median abs 0.061711
OOF lab 2 below 25 tail signed median -7e-06 tail median abs 0.0146 curve rmse 0.313063 first/last median abs [0.004875 0.0146  ]
OOF lab 2 above 25 tail signed median -0.004853 tail median abs 0.009988 curve rmse 0.266602 first/last median abs [0.006143 0.009988]
OOF lab 2 missing drift fraction 0 curve median abs 0.265281 last median abs 0.578498
OOF lab 2 missing drift fraction 0.5 curve median abs 0.129314 last median abs 0.282477
OOF lab 2 missing drift fraction 1 curve median abs 0.007881 last median abs 0.013544
OOF lab 3 below 27 tail signed median 0.009257 tail median abs 0.026597 curve rmse 0.268522 first/last median abs [0.007863 0.026597]
OOF lab 3 above 19 tail signed median -0.0072 tail median abs 0.047967 curve rmse 0.266053 first/last median abs [0.00546  0.047967]
OOF lab 3 missing drift fraction 0 curve median abs 0.068159 last median abs 0.140947
OOF lab 3 missing drift fraction 0.5 curve median abs 0.05224 last median abs 0.097965
OOF lab 3 missing drift fraction 1 curve median abs 0.031141 last median abs 0.054983
public below 40 tail signed median 0.002004 tail median abs 0.010347 curve rmse 0.013786 first/last median abs [0.000914 0.010347]
public above 40 tail signed median -0.005245 tail median abs 0.011979 curve rmse 0.019707 first/last median abs [0.000843 0.011979]
public missing drift fraction 0 curve median abs 0.221716 last median abs 0.423865
public missing drift fraction 0.5 curve median abs 0.115079 last median abs 0.250926
public missing drift fraction 1 curve median abs 0.009598 last median abs 0.023506
all nine basis columns: zero displacement and independent onset derivative passed
calibration max gain/bias abs error [3.24868745e-05 3.09874506e-05]
public output max curve/rate abs error 0.08551336258254438 0.016708592403118394
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed2ce7267ea2a74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed2ce72 (directory).
```

### Turn 10 — `submit_completed_task` (`14775 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation and the onset-derivative check for all nine basis terms. Sparse and EMI-corrupted checks shifted calibration coefficients by less than 0.000039, and held-out lab-by-yield residuals showed no consistent late-horizon bias.
- **Alternatives Weighed**: `Reduce the onset-rate correction for coupons with missing features` (rejected: The full correction gave lower median curve error than partial or no correction in all four held-out labs.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_w01_batch_04_solver.py': 'FILE:7663'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_batch_04_solver.py`. The public schema passed, and the solver remained stable with sparse or EMI-corrupted calibration checks on either wing. Worst-lab public strain RMSE was 0.02430.`

### Turn 11 — `execute_in_remote_docker` (`32857 ms` LLM)
- **Rationale**: The previous audit found larger errors late in the hold without a consistent signed drift. Test whether fitting a noise scale at each horizon improves predictions in held-out labs, and report lab-by-stratum errors so pooled improvement cannot hide a worse subgroup. Leave the verified solver untouched unless the challenger improves the bottleneck.
- **Alternatives Weighed**: `Add rational terms or relay dynamics to the strain law` (rejected: The nine specified terms already pass the onset identity; no measured residual pattern points to these additions, and the dataset has no relay state.)

```bash
cat > /tmp/b04_horizon_audit.py <<'PY'
import importlib.util, pathlib, pickle, types
import numpy as np
p=pathlib.Path('/holdout_task/alpha_w01_batch_04_solver.py')
spec=importlib.util.spec_from_file_location('baseline',p)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
source=p.read_text()
old='    for m in (is_curve, ~is_curve):\n        if np.any(m):\n            scale[m] = _mad(e[m])'
new='    channel = np.r_[np.tile(np.arange(len(time)), len(rates)), np.full(len(rates),len(time))][valid]\n    for j in range(len(time)+1):\n        m = channel == j\n        if np.any(m):\n            scale[m] = _mad(e[m])'
assert source.count(old)==1
q=types.ModuleType('challenger'); exec(compile(source.replace(old,new),'/tmp/b04_horizon_challenger.py','exec'),q.__dict__)
z=pickle.load(open(p.parent/'public_alpha_w01_batch_04.pkl','rb'))[0]; d=z['dataset']; tr,te=d['train'],d['test']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress'))
x,_,complete,_=s._align(tr,te); maps=s._calibrate(d['calibration_load_cells']); h,hold=s._stress(tr,maps); ov,w=s._plastic(h,hold,dt,sy)
for mod in (s,q):
    err=np.empty_like(tr['strain_curves_obs']); re=np.empty(len(err))
    for lab in range(4):
        b=tr['lab_id']==lab; a=~b
        tau,theta=mod._fit({k:v[a] for k,v in tr.items()},x[a],complete[a],maps,dt,t,sy)
        A,B=s._design(tau,h[b],hold[b],x[b],dt,t,ov[b],w[b]); rate=B@theta
        re[b]=rate-tr['instant_rate_obs'][b]
        err[b]=tr['strain_0'][b,None]+A@theta-np.where(complete[b],0,re[b])[:,None]*t-tr['strain_curves_obs'][b]
    labels=np.column_stack((tr['lab_id'],tr['stratum']))
    cells=[]
    for cell in np.unique(labels,axis=0):
        b=np.all(labels==cell,axis=1)
        cells.append((tuple(cell),round(float(np.median(abs(err[b]))),6),round(float(np.median(err[b,-1])),6)))
    print(mod.__name__,'lab-stratum (curve MedAE, final signed median)',cells)
    print(mod.__name__,'worst lab curve MedAE',max(np.median(abs(err[tr['lab_id']==i])) for i in range(4)),'worst lab complete-rate MedAE',max(np.median(abs(re[(tr['lab_id']==i)&complete])) for i in range(4)))
    o=mod.solve_alpha_w01_batch_04(d); g=z['gt']; e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']
    print(mod.__name__,'public worst lab curve/rate RMSE',*[max(np.sqrt(np.mean(v[te['lab_id']==i]**2)) for i in range(4)) for v in (e,r)],'taus',o['estimated_taus'],'plastic',o['viscoplastic_weights'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/b04_horizon_audit.py
```
- **Container Exit Code**: `0` (`1048 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline lab-stratum (curve MedAE, final signed median) [((0, 0), 0.017127, -0.017286), ((0, 1), 0.02464, 0.0138), ((0, 2), 0.012407, -0.003466), ((1, 0), 0.012487, 0.015178), ((1, 1), 0.014535, 0.01464), ((1, 2), 0.008638, 0.004541), ((2, 0), 0.007989, -0.009858), ((2, 1), 0.008005, -0.002647), ((2, 2), 0.008908, -0.001957), ((3, 0), 0.020497, -0.0072), ((3, 1), 0.018393, 0.019961), ((3, 2), 0.014654, -0.008597)]
baseline worst lab curve MedAE 0.01749261367235244 worst lab complete-rate MedAE 0.010805645748756754
baseline public worst lab curve/rate RMSE 0.024295629532143898 0.009328443478760776 taus [0.48743712 4.25227725] plastic [0.2492129 0.1696062]
challenger lab-stratum (curve MedAE, final signed median) [((0, 0), 0.017329, -0.017401), ((0, 1), 0.024693, 0.013261), ((0, 2), 0.012729, -0.004522), ((1, 0), 0.012165, 0.0153), ((1, 1), 0.014749, 0.014835), ((1, 2), 0.008386, 0.003987), ((2, 0), 0.007858, -0.010076), ((2, 1), 0.007981, -0.004171), ((2, 2), 0.008846, -0.001979), ((3, 0), 0.020512, -0.006992), ((3, 1), 0.018471, 0.020221), ((3, 2), 0.01472, -0.008757)]
challenger worst lab curve MedAE 0.017597607261764292 worst lab complete-rate MedAE 0.010400130611729064
challenger public worst lab curve/rate RMSE 0.023992084858700465 0.009222972421318297 taus [0.4876933 4.2275082] plastic [0.24932567 0.16966456]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3ed2ce7267ea2a74 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3ed2ce72 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`23460 ms` LLM)
- **Rationale**: Promote the tested horizon-specific weighting. Re-run the isolated schema, onset-derivative, and sparse and EMI calibration checks on the saved artifact to catch any implementation discrepancy or loss of stability under the new weighting.
- **Alternatives Weighed**: `Retain one pooled strain noise scale` (rejected: It gives a slightly lower worst held-out curve median error, but the horizon-specific fit improves held-out onset-rate error and materially reduces the public slow relaxation-time error without changing the constitutive law.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_w01_batch_04_solver.py')
s = p.read_text()
old = '    is_curve = (np.arange(len(yfull)) < len(dy))[valid]'
new = '    # Rate observations form a separate channel from each strain horizon.\n    channel = np.r_[np.tile(np.arange(len(time)), len(rates)),\n                    np.full(len(rates), len(time))][valid]'
assert s.count(old) == 1
s = s.replace(old, new)
old = '    for m in (is_curve, ~is_curve):\n        if np.any(m):\n            scale[m] = _mad(e[m])'
new = '    for j in range(len(time) + 1):\n        m = channel == j\n        if np.any(m):\n            scale[m] = _mad(e[m])'
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/b04_margin.py | grep -E 'weights fit|taus fit|public total|lab [0-3] [0-9]+ curve|basis columns|public output max|calibration max'
python3 -I /tmp/b04_stress.py | tail -6
```
- **Container Exit Code**: `0` (`2212 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus fit [0.487693 4.227508] truth [0.488 4.211] max abs error 0.01650819809133708
hereditary_weights fit [ 0.70099   0.525005  0.079268  0.151921 -0.131306  0.110723 -0.090922] truth [ 0.7   0.52  0.08  0.15 -0.13  0.11 -0.09] max abs error 0.005005148807472715
viscoplastic_weights fit [0.249326 0.169665] truth [0.25 0.17] max abs error 0.0006743260785605498
public total curve/rate RMS 0.016951109663000837 0.006602658169216168 first/last curve RMS 0.0015480567325707043 0.02783362183041394
lab 0 20 curve/rate RMS 0.017596760747112812 0.005905442034705574
lab 1 21 curve/rate RMS 0.007967522038219528 0.004511022103085898
lab 2 17 curve/rate RMS 0.01282813356661186 0.0053763977908575025
lab 3 22 curve/rate RMS 0.023992084858700465 0.009222972421318297
all nine basis columns: zero displacement and independent onset derivative passed
calibration max gain/bias abs error [3.24868745e-05 3.09874506e-05]
public output max curve/rate abs error 0.08590943404340745 0.017664668767325864
sparse A max calibration shift 2.6853108518576363e-05 taus [0.4877   4.227011] vp [0.24936  0.169684]
sparse B max calibration shift 3.8091004855786426e-05 taus [0.487695 4.226671] vp [0.249325 0.169655]
EMI A max calibration shift 1.1540710952620548e-05 taus [0.487696 4.227538] vp [0.249329 0.169667]
EMI B max calibration shift 1.2953449516017734e-05 taus [0.487693 4.227949] vp [0.249323 0.169669]
sparse+EMI B max calibration shift 3.820591357010794e-05 taus [0.487695 4.22673 ] vp [0.249324 0.169656]
plastic zero-activation, sign, and onset-derivative checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=26da978010e40ed0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_26da9780 (directory).
```

### Turn 13 — `submit_completed_task` (`13672 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, onset-derivative checks for all nine basis terms, and calibration stress tests. The horizon-specific fit improved slow relaxation-time recovery without changing the constitutive law.
- **Alternatives Weighed**: `Retain the pooled strain noise scale` (rejected: Its worst held-out lab median strain error was lower by 0.00011, but the horizon-specific fit improved worst held-out onset-rate error and reduced public slow relaxation-time error from 0.04128 to 0.01651.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_batch_04_solver.py`. The public schema passed. Worst-lab public strain RMSE was 0.02399, and worst leave-one-lab-out median absolute strain error was 0.01760. Sparse and EMI-corrupted calibration checks on either wing shifted the fitted calibration coefficients by less than 0.000039.`

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
