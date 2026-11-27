# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_c01_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `329,298` (`316,072` / `122,496` / `13,226`) |
| **Total Episode Wall Time (s)** | `390.44s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14152 ms` LLM)
- **Rationale**: Inspect field shapes, calibration and pulse availability, and archive structure to detect missing inputs or convention mismatches before implementing the forward model.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle
s=pickle.load(open('public_batch.pkl','rb'))
for k,v in s.items():
 if isinstance(v,dict): print(k, {a:(type(b).__name__,getattr(b,'shape',None),len(b) if isinstance(b,(list,dict)) else '') for a,b in v.items()})
 elif isinstance(v,list): print(k,len(v), list(v[0]) if v and isinstance(v[0],dict) else str(v[:2])[:200])
 elif hasattr(v,'shape'): print(k,v.shape,v.dtype,v.ravel()[:8])
 else: print(k,repr(v)[:400])
PY
if test -d /tmp/hegel_continual_archive; then du -ah /tmp/hegel_continual_archive | sort -h | tail -30; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex QP' || true
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`17101 ms` LLM)
- **Rationale**: The first inspection found a list wrapper rather than a bare study. Read the actual study and available apparatus records to determine which inputs can constrain the forward model on blind campaigns.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
batch=pickle.load(open('public_batch.pkl','rb')); print('batch',len(batch));
def show(v,depth=0):
 if isinstance(v,dict):
  for k,x in v.items(): print(' '*depth,k,end=': '); show(x,depth+2)
 elif isinstance(v,list) and v and isinstance(v[0],dict): print('list',len(v)); show(v[0],depth+2)
 elif hasattr(v,'shape'): print(v.shape,v.dtype,np.asarray(v).ravel()[:12])
 else: print(str(v)[:350])
show(batch[0]); print('batch entries:',[list(e) for e in batch]);
PY
if test -d /tmp/hegel_continual_archive; then ls -lR /tmp/hegel_continual_archive | head -70; if test -f /tmp/hegel_continual_archive/archive_apparatus_log.json; then head -c 5500 /tmp/hegel_continual_archive/archive_apparatus_log.json; printf '\n'; fi; else echo 'No continual archive'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -36
```
- **Container Exit Code**: `0` (`403 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
        "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
        "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
      },
      "functions": [
        {
          "function": "fit_ridge_nuisance",
          "signature": "fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
        }
      ],
      "index_cards": [
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching",
      "reference_source_txt": null
    }
```

### Turn 3 — `execute_in_remote_docker` (`19301 ms` LLM)
- **Rationale**: Calibration residuals will determine whether nonlinear terms are needed; pulse scales and sensor placement will determine how to test Robin discretizations.
- **Alternatives Weighed**: `Assume a fixed sensor inverse` (rejected: The supplied sweeps identify the active calibration directly and can reveal nonlinearity or channel mixing.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; np.set_printoptions(precision=7,suppress=True,linewidth=130)
print('all keys',list(s)); print('unique levels', [np.unique(x[:,j]).size for j in range(4)]); print('sweep activation counts',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('sweeps',np.c_[x,y][:22]);
for k,v in c['single_bank_sweeps'].items(): print('single',k,v.shape,v[:16]);
a=np.c_[np.ones(len(x)),x]; f=np.linalg.lstsq(a,y,rcond=None)[0]; print('affine',f,'RMSE',np.sqrt(np.mean((a@f-y)**2,axis=0))); print('sensors',s['sensor_ij']); print('pulse scales',[(p['w_pulse'],np.linalg.norm(p['y_pulse_raw'])) for p in s['reference_tracer_pulses']]); print('source norm',np.linalg.norm(s['source_modes'],axis=(1,2))); t=s['aux_manifold_telemetry']; print('telemetry mean std svd',t.mean(0),t.std(0),np.linalg.svd(t,compute_uv=False)); print('telemetry edge rows',t[:4],t[-4:]);
PY
```
- **Container Exit Code**: `0` (`340 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
00058 -0.4000003]
 [ 0.         0.         0.         0.5       -0.0000141 -0.0000113  0.0000114  0.499992 ]
 [ 0.         0.         0.         1.3        0.0000081  0.0000055 -0.0000079  1.2999872]
 [-0.0387259 -0.704883   1.3794483 -0.1474809 -0.0387263 -0.7048879  1.3794418 -0.1474707]
 [ 0.7898369  1.5779146 -1.0573722 -1.0827809  0.7898156  1.5779121 -1.0573853 -1.0827903]
 [ 1.090665  -1.3606879  0.0356641 -0.9737087  1.0906707 -1.3606901  0.0356608 -0.9737115]
 [ 1.2931502  0.391234   0.5335765 -1.2297051  1.2931356  0.3912227  0.5335825 -1.2297028]
 [-0.295442  -1.1525896  1.7636719  1.4668463 -0.2954375 -1.1525813  1.7636727  1.4668415]
 [-1.3057168 -1.1038675  1.4794974  0.8929949 -1.3057107 -1.1038673  1.4794855  0.8929901]]
single bank_id (24,) [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2]
single ref_signal (24,) [-1.1 -0.6 -0.1  0.4  0.9  1.4 -1.1 -0.6 -0.1  0.4  0.9  1.4 -1.1 -0.6 -0.1  0.4]
single raw_readout (24,) [-1.1000036 -0.6000014 -0.1000116  0.3999999  0.9000046  1.4000039 -1.1000059 -0.5999994 -0.1000148  0.4000021  0.9000093
  1.3999791 -1.1000087 -0.59999   -0.0999996  0.4000104]
affine [[ 0.0000013  0.0000004 -0.0000007 -0.0000018]
 [ 1.0000001  0.0000014 -0.0000004  0.0000004]
 [-0.0000014  0.9999975 -0.0000013  0.0000021]
 [ 0.0000015 -0.0000007  1.0000014  0.0000007]
 [-0.0000003  0.0000007 -0.0000003  1.000001 ]] RMSE [0.0000095 0.0000105 0.0000102 0.0000089]
sensors [[ 0  6]
 [ 6  4]
 [10  4]
 [12  6]
 [10  0]
 [10  6]
 [ 2  4]
 [ 6  0]
 [12  0]
 [ 4  4]
 [10  8]
 [ 6 10]
 [ 4  6]
 [10  2]
 [ 0  8]
 [ 0 10]
 [ 0  0]
 [ 2  2]
 [10 10]
 [12 10]
 [ 6  6]
 [ 8  0]
 [ 8  4]
 [ 6  2]
 [ 2  6]
 [12  4]
 [ 8  8]
 [ 0  2]
 [ 2  0]
 [ 4  2]
 [ 2 10]
 [ 6  8]]
pulse scales [(array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]), 0.9203533795500654), (array([0. , 1. , 0. , 0. , 0.5, 0. , 0. , 0. ]), 0.9431361390949579)]
source norm [2.7328905 2.7328998 2.7286019 2.7329002 2.7328998 2.7319424 2.7324841 2.7327776]
telemetry mean std svd [-0.3032927  0.1313976  0.2736689  0.0260855] [1.0646252 1.0244723 1.0925783 1.1028606] [6.3439517 5.6048184 5.231071  3.955251 ]
telemetry edge rows [[-0.6633395  1.0719685  0.2499651  1.4838147]
 [-2.7078674  0.7721202 -1.5507076  0.7836598]
 [ 0.564505   1.5412976 -0.1816943  0.5947575]
 [-0.9031015  0.0532385  1.5622575 -0.1867578]] [[ 0.3450347 -0.7886119  2.5428394  0.9847891]
 [-0.6475638 -1.2414542  0.7867462 -2.2755968]
 [ 1.2141765 -0.8680061 -0.1270906 -0.2881768]
 [-0.2440376  1.4449562 -0.3524212 -0.4750624]]
```

### Turn 4 — `execute_in_remote_docker` (`32058 ms` LLM)
- **Rationale**: Compare nominal and fitted endpoint losses across finite-volume conventions. Residuals near the measurement noise would support a stencil; larger residuals would show that boundary adjustment alone does not explain that convention.
- **Alternatives Weighed**: `Fit only two Robin coefficients with a fixed stencil` (rejected: A discretization or sign mismatch could masquerade as a change in the boundary coefficients.)

```bash
cat >/tmp/probe98.py <<'PY'
import itertools,pickle,numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; p=s['reference_tracer_pulses']; B=np.array([np.tensordot(a['w_pulse'],s['source_modes'],axes=1).ravel(order='F') for a in p]).T; Y=np.array([a['y_pulse_raw'] for a in p]).T

def axis(k,v,h,mean,adv,sign):
    k=(k[:-1]+k[1:])/2 if mean==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); d=k/h**2; v=v*sign/h; a=(v[:-1]+v[1:])/2; pos=np.maximum(a,0); neg=np.minimum(a,0)
    if adv==0: lo=-d-pos; hi=-d+neg; mid=np.r_[d+pos,0]+np.r_[0,d-neg]
    if adv==1: lo=-d-pos; hi=-d+neg; mid=-np.r_[hi,0]-np.r_[0,lo]
    if adv==2:
        lo=-d-np.maximum(v[1:],0); hi=-d+np.minimum(v[:-1],0); mid=-np.r_[hi,0]-np.r_[0,lo]
    if adv==3: lo=-d-np.maximum(v[:-1],0); hi=-d+np.minimum(v[1:],0); mid=np.r_[d+np.maximum(v[:-1],0),0]+np.r_[0,d-np.minimum(v[1:],0)]
    return diags([lo,mid,hi],[-1,0,1],format='csc')

ends=np.array([np.tile(np.arange(nx)==0,ny),np.tile(np.arange(nx)==nx-1,ny),np.repeat(np.arange(ny)==0,nx),np.repeat(np.arange(ny)==ny-1,nx)],dtype=float).T
E=eye(nx*ny,format='csc')
def predict(A,r):
    AA=A+diags(ends@r,format='csc'); return np.concatenate([splu(AA+mu*E).solve(B)[idx] for mu in s['mu_shifts']])

def candidates():
 for grid,mean,adv,sign in itertools.product(range(2),range(2),range(4),(1,-1)):
    xx=s['Lx']/(nx-grid); yy=s['Ly']/(ny-grid)
    A=kron(eye(ny),axis(s['kappa_x'],s['vel_x'],xx,mean,adv,sign),format='csc')+kron(axis(s['kappa_y'],s['vel_y'],yy,mean,adv,sign),eye(nx),format='csc')+V
    yield (grid,mean,adv,sign), A, np.array([xx,xx,yy,yy])
if __name__=='__main__':
 results=[]; nom=np.repeat([s['robin_x'],s['robin_y']],2); Z=s['Z_nuisance']; print('pulse nuisance coefficients against zero',np.linalg.lstsq(Z,Y,rcond=None)[0]);
 for key,A,h in candidates():
    known=[]
    for label,r in [('direct',nom),('flux',nom/h),('ghost',nom/h**2)]: known.append((np.std(predict(A,r)-Y),label))
    fit=least_squares(lambda r:(predict(A,r)-Y).ravel(),nom/h,bounds=(0,250)); results.append((np.std(fit.fun),key,fit.x,np.array(fit.x*h),min(known)))
 for v in sorted(results,key=lambda v:v[0])[:12]: print(v)
 print('nominal',nom)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe98.py
ls -la /pretrain_task /holdout_task
```
- **Container Exit Code**: `0` (`2007 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4783952834e-05, (1, 1, 2, 1), array([4.9901488 , 4.99621766, 7.07848584, 7.16632831]), array([0.92125824, 0.92237865, 1.28699743, 1.30296878]), (0.0002415741493183668, 'flux'))
(0.00020938005553827843, (1, 1, 1, 1), array([4.90276286, 4.8151135 , 7.21933486, 7.04543195]), array([0.90512545, 0.88894403, 1.31260634, 1.28098763]), (0.000279159855799448, 'flux'))
(0.00022868197321332297, (1, 0, 1, 1), array([4.90169796, 4.80886208, 7.21631203, 7.04703051]), array([0.90492885, 0.88778992, 1.31205673, 1.28127828]), (0.00029261348682984613, 'flux'))
(0.00370018005876272, (0, 0, 3, 1), array([ 5.70921142,  9.47981378, 12.49008721,  9.48486282]), array([0.97872196, 1.62511093, 2.0816812 , 1.58081047]), (0.006798575660393921, 'flux'))
(0.003709931021462701, (0, 1, 3, 1), array([ 5.71353361,  9.49967707, 12.4976604 ,  9.48698391]), array([0.97946291, 1.62851607, 2.0829434 , 1.58116399]), (0.00681293801174965, 'flux'))
(0.0037962704827363687, (0, 0, 0, 1), array([ 6.5525306 ,  9.23964831, 12.19653646,  9.94239403]), array([1.12329096, 1.58393971, 2.03275608, 1.65706567]), (0.006374539707553262, 'flux'))
(0.0038083278113224383, (0, 1, 0, 1), array([ 6.55781786,  9.25868201, 12.20417733,  9.94520635]), array([1.12419735, 1.58720263, 2.03402956, 1.65753439]), (0.006390576999265296, 'flux'))
(0.006073395135636851, (0, 1, 2, 1), array([4.70468629, 3.69874477, 6.69890375, 5.29739236]), array([0.80651765, 0.63407053, 1.11648396, 0.88289873]), (0.006293466410167792, 'flux'))
(0.006093360591390885, (0, 0, 2, 1), array([4.70417796, 3.69461306, 6.69655543, 5.29897841]), array([0.80643051, 0.63336224, 1.11609257, 0.88316307]), (0.006313372049675265, 'flux'))
(0.006211993780346243, (0, 1, 1, 1), array([4.61171208, 3.57804291, 6.83840897, 5.23896679]), array([0.79057921, 0.61337879, 1.13973483, 0.87316113]), (0.006450926629377305, 'flux'))
(0.006231978180245928, (0, 0, 1, 1), array([4.61118726, 3.57418344, 6.83604276, 5.24068339]), array([0.79048924, 0.61271716, 1.13934046, 0.87344723]), (0.00647072225715028, 'flux'))
nominal [0.88093934 0.88093934 1.29318727 1.29318727]
/holdout_task:
total 44
drwxr-xr-x 2 root root  4096 Nov 27 12:26 .
drwxr-xr-x 1 root root  4096 Nov 27 12:27 ..
-rw-r--r-- 1 root root  3666 Nov 27 12:26 public_asset_index.json
-rw-r--r-- 1 root root 26378 Nov 27 12:26 public_batch.pkl
-rw-r--r-- 1 root root  1064 Nov 27 12:26 validate_public_schema.py

/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 12:27 ..
```

### Turn 5 — `execute_in_remote_docker` (`38068 ms` LLM)
- **Rationale**: Hold out each tracer pulse to test whether wall corrections transfer. Separately hold out whole shifts when inferring walls from routine observations; if that fails to improve prediction, do not substitute blind residual fitting for the nominal or archive-informed prior.
- **Alternatives Weighed**: `Use the supplied Robin values without testing them` (rejected: The public pulse discrepancy is larger than calibration noise; leave-one-pulse-out testing will determine whether the wall fit transfers beyond the pulse used to estimate it.)
- **ATMS NOGOOD DECLARED**: Refuted `The L/n cell-width convention describes the public chamber after refitting the wall losses.` | Witness: `The best tested L/n stencil retained a pulse residual standard deviation of 3.70e-3 with four independent wall losses; the L/(n-1) arithmetic-face, nodal-upwind stencil reached 2.00e-5 on the same observations.` | Blanket Diagnosis: `Grid spacing changes the interior operator; wall-only adjustments cannot compensate for it across sensors and shifts.` | Sublated Pivot: `Use L/(n-1) and nodal upwinding, then validate wall-loss estimation on held-out pulses and shifts.`

```bash
cat >/tmp/validate98.py <<'PY'
from probe98 import *
from scipy.linalg import null_space
key,A,h=next(t for t in candidates() if t[0]==(1,0,2,1)); nom=np.repeat([s['robin_x'],s['robin_y']],2)/h
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; des=np.c_[np.ones(len(x)),x]; affine=np.linalg.lstsq(des,raw,rcond=None)[0]; cal=lambda y:np.linalg.solve(affine[1:].T,(y.reshape(-1,4)-affine[0]).T).T.ravel(); Y=np.array([cal(p['y_pulse_raw']) for p in p]).T; y=cal(s['y_obs_raw']); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; F=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); n=len(y); ss=len(idx); q=y.reshape(-1,4); fp=lambda p:p.reshape(-1,4); rms=lambda z:np.sqrt(np.mean(z*z));
def design(r):
    AA=A+diags(ends@r,format='csc'); return np.concatenate([splu(AA+mu*E).solve(F)[idx] for mu in s['mu_shifts']]);
W=np.column_stack([a['w_pulse'] for a in p]);
for model, g, x0 in [('nom',lambda u:nom,[]),('axis',lambda u:np.repeat(u,2),nom[::2]),('wall',lambda u:u,nom)]:
    for i in [0,1,None]:
        train=[i] if i is not None else [0,1]; test=[1-i] if i is not None else [0,1]; r=g(least_squares(lambda u:(design(g(u))@W[:,train]-Y[:,train]).ravel(),x0,ftol=1e-13,gtol=1e-13).x) if len(x0) else g(None); R=design(r)@W-Y; print(model,i,'Robin',r*h,'train/test',rms(R[:,train]),rms(R[:,test]),'max-wall-shift',max(rms(a) for a in np.split(R[:,test],3)))
    if model=='wall': r_pulse=r
N=null_space(np.ones((1,W.shape[0]))); bm=np.ones(W.shape[0])*s['total_budget']/W.shape[0]
def obs_fit(r, mask):
    S=design(r); D=np.c_[S@N,Z]; b=y-S@bm; a=np.linalg.lstsq(D[mask],b[mask],rcond=None)[0]; return D@a-b,bm+N@a[:N.shape[1]]
for label,r in [('nom',nom),('pulse',r_pulse)]:
    err,w=obs_fit(r,np.ones(n,dtype=bool)); print(label,'blind residual',rms(err),'weights',w,'pulse/nom sensitivity',np.max(abs(r-nom)))
for k in [0,1,2,None]:
    mask=np.ones(n,dtype=bool); 
    if k is not None: mask[k*ss:(k+1)*ss]=False
    fit=least_squares(lambda r:obs_fit(r,mask)[0][mask],nom,bounds=(.2*nom,3*nom),ftol=1e-12,gtol=1e-12); err,w=obs_fit(fit.x,mask); print('blindfit',k,fit.x*h,'within / out',rms(err[mask]),rms(err[~mask]) if k is not None else 0,'nom/pulse out',[rms(obs_fit(r,mask)[0][~mask]) for r in [nom,r_pulse]] if k is not None else [],'weights',w,'jacobian sv',np.linalg.svd(fit.jac,compute_uv=False))
# Split by rosette so neither channel nor a physical sensor pair leaks between folds.
for fold in range(4):
    tr=np.arange(len(x))%4!=fold; f=np.linalg.lstsq(des[tr],raw[tr],rcond=None)[0]; print('cal',fold,'affine',np.max(np.sqrt(np.mean((des[~tr]@f-raw[~tr])**2,axis=0))),'identity',rms((x-raw)[~tr]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/validate98.py
```
- **Container Exit Code**: `0` (`708 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2354595 1.28852264 1.28852264] train/test 5.475439516102749e-05 5.475439516102749e-05 max-wall-shift 7.101720484484058e-05
wall 0 Robin [0.92128254 0.9217886  1.28672922 1.30133869] train/test 1.856968688975987e-05 2.4262006212798328e-05 max-wall-shift 2.7473475590736182e-05
wall 1 Robin [0.92099468 0.92105133 1.2855643  1.30372719] train/test 2.0928268289777957e-05 2.3203089292321243e-05 max-wall-shift 2.7202412701269062e-05
wall None Robin [0.92107951 0.9211281  1.28646561 1.30310303] train/test 2.062369707801902e-05 2.062369707801902e-05 max-wall-shift 2.5291103363084892e-05
nom blind residual 0.008253905994551248 weights [-0.02281886  0.26451846  0.23635374  0.06811953  0.70680844  0.0776918
  0.07719022 -0.00786332] pulse/nom sensitivity 0.0
pulse blind residual 0.008166722404627496 weights [-0.0093888   0.25965406  0.23635692  0.06415022  0.71497473  0.05476127
  0.08620149 -0.0067099 ] pulse/nom sensitivity 0.2176891203709319
blindfit 0 [0.8735056  1.34339149 1.15220456 1.50987831] within / out 0.0072968096206196984 0.010078712578883969 nom/pulse out [0.010941902844480156, 0.010816636027941171] weights [ 0.24647819  0.30426334  0.23287665  0.00223049  0.69130939 -0.2034806
  0.20252684 -0.07620431] jacobian sv [0.00772842 0.002778   0.00256273 0.00187102]
blindfit 1 [0.42005226 0.98015741 1.55034792 2.44128106] within / out 0.007141378268916655 0.009593132043837107 nom/pulse out [0.009858101438820628, 0.009792115103133926] weights [ 0.18534406  0.55754473  0.26021789 -0.01403214  0.37387965 -0.11155171
  0.16944514 -0.0208476 ] jacobian sv [0.0108819  0.00605636 0.00339613 0.00157433]
blindfit 2 [0.34955251 1.4955986  1.11737734 2.12984315] within / out 0.00703065814207077 0.009715408635995557 nom/pulse out [0.009437572087262338, 0.009436396334505718] weights [ 0.04654908 -0.07707968  0.33167907  0.22675213  0.96060295 -0.10670707
  0.12894729 -0.11074377] jacobian sv [0.01190797 0.00623489 0.0032263  0.00200982]
blindfit None [0.38461462 1.25113077 1.35884418 1.97866937] within / out 0.007582315045181666 0 nom/pulse out [] weights [ 0.17935046  0.27128464  0.27647465  0.06808776  0.65221217 -0.12976066
  0.16169605 -0.07934507] jacobian sv [0.01269524 0.00695596 0.00391378 0.00241647]
cal 0 affine 1.2546823512058231e-05 identity 1.0743860058767931e-05
cal 1 affine 1.1626967202112046e-05 identity 9.584567163840771e-06
cal 2 affine 1.4061491709850203e-05 identity 1.104188293958827e-05
cal 3 affine 1.120946476507479e-05 identity 8.616045041578106e-06
```

### Turn 6 — `execute_in_remote_docker` (`72269 ms` LLM)
- **Rationale**: Implement the validated operator and calibration, then verify the output schema. Use an active-set solve and a Lagrangian lower bound so stationarity, complementarity, and the dual gap can be tested independently.
- **Alternatives Weighed**: `Fit walls from routine observations immediately` (rejected: Unconstrained blind fits produce inconsistent wall estimates across shifts and worsen one held-out shift. Keep nominal walls as the initial blind baseline.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrate the rosettes, reconstruct the chamber, and certify the source QP."""
import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _monomials(degree):
    return np.array([[idx.count(j) for j in range(4)]
                     for d in range(degree + 1)
                     for idx in itertools.combinations_with_replacement(range(4), d)])


def _features(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=-1)


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    # Score entire rosettes rather than splitting their four coupled channels.
    fold = np.arange(len(x)) % 4
    models = []
    for degree in (1, 2, 3):
        powers = _monomials(degree)
        a = _features(x, powers)
        if len(a) * .75 <= a.shape[1] + 4:
            break
        scores = []
        for k in range(4):
            tr = fold != k
            b = np.linalg.lstsq(a[tr], (y-x)[tr], rcond=None)[0]
            scores.append(np.mean((a[~tr] @ b - (y-x)[~tr]) ** 2))
        models.append((np.mean(scores), np.std(scores) / 2, powers, a))
    best = min(models, key=lambda m: m[0])
    _, _, powers, a = next(m for m in models if m[0] <= 1.10 * best[0])
    b = np.linalg.lstsq(a, y-x, rcond=None)[0]
    linear = (np.eye(4) + b[1:5]).T
    derivatives = np.maximum(powers[None, :, :] - np.eye(4, dtype=int)[:, None, :], 0)

    def invert(raw):
        raw = np.asarray(raw, dtype=float).reshape(-1, 4)
        z = np.linalg.solve(linear, (raw - b[0]).T).T
        if len(powers) == 5:
            return z.ravel()
        for i, target in enumerate(raw):
            def fun(t):
                return t + _features(t[None, :], powers)[0] @ b - target

            def jac(t):
                d = np.prod(t[None, None, :] ** derivatives, axis=-1) * powers.T
                return np.eye(4) + (d @ b).T

            z[i] = least_squares(fun, z[i], jac=jac, gtol=1e-12, ftol=1e-12,
                                 xtol=1e-12).x
        return z.ravel()

    return invert


def _axis(kappa, velocity, step):
    d = (np.asarray(kappa[:-1]) + kappa[1:]) / (2 * step**2)
    a = np.asarray(velocity) / step
    lower = -d - np.maximum(a[1:], 0)
    upper = -d + np.minimum(a[:-1], 0)
    center = -np.r_[upper, 0.] - np.r_[0., lower]
    return diags((lower, center, upper), (-1, 0, 1), format='csc')


def _chamber(s):
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx']/(nx-1), s['Ly']/(ny-1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx)
    ay = _axis(s['kappa_y'], s['vel_y'], hy)
    ident = eye(nx*ny, format='csc')
    base = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
            + diags(np.ravel(s['V_reaction_2d'], order='F'), format='csc'))
    xx, yy = np.tile(np.arange(nx), ny), np.repeat(np.arange(ny), nx)
    walls = np.column_stack((xx == 0, xx == nx-1, yy == 0, yy == ny-1)).astype(float)
    walls /= [hx, hx, hy, hy]
    si = np.asarray(s['sensor_ij'])
    indices = si[:, 0] + nx*si[:, 1]
    sources = np.column_stack([m.ravel(order='F') for m in s['source_modes']])

    def forward(robin, rhs):
        operator = base + diags(walls @ robin, format='csc')
        readings = []
        for shift in s['mu_shifts']:
            u = splu(operator + shift*ident).solve(rhs)
            readings.append(u[indices])
        return np.concatenate(readings), u

    nominal = np.repeat([s['robin_x'], s['robin_y']], 2).astype(float)
    return forward, nominal, sources


def _pulse_fit(pulses, invert, forward, nominal, sources):
    rhs = sources @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([invert(p['y_pulse_raw']) for p in pulses])
    fit = least_squares(lambda r: (forward(r, rhs)[0] - observed).ravel(), nominal,
                        bounds=(0., np.inf), ftol=1e-12, gtol=1e-12, xtol=1e-12)
    return fit.x


def _box_qp(h, c, upper, budget):
    """Feasible active set for a convex quadratic with one equality."""
    n = len(c)
    w = budget * upper / upper.sum() if upper.sum() else np.zeros(n)
    g = lambda: h @ w + c
    if budget == 0:
        return w, float(-min(g()))
    if budget == upper.sum():
        return upper.copy(), float(-max(h @ upper + c))
    status = np.where(upper > 0, 0, 2)
    tol = 5e-13
    for _ in range(20*n*n):
        free = np.flatnonzero(status == 0)
        grad = g()
        if not len(free):
            lower, high = np.where(status == -1)[0], np.where(status == 1)[0]
            i, j = lower[np.argmin(grad[lower])], high[np.argmax(grad[high])]
            if grad[j] <= grad[i] + tol:
                return w, float(-(grad[i] + grad[j])/2)
            status[[i, j]] = 0
            continue
        f = h[np.ix_(free, free)]
        v, d = np.linalg.solve(f, np.column_stack((-grad[free], np.ones(len(free))))).T
        nu = v.sum()/d.sum()
        p = v - nu*d
        alpha = np.ones(len(free))
        positive, negative = p > tol, p < -tol
        alpha[positive] = (upper[free[positive]] - w[free[positive]]) / p[positive]
        alpha[negative] = -w[free[negative]] / p[negative]
        j = np.argmin(alpha)
        if alpha[j] < 1:
            w[free] += max(0., alpha[j])*p
            i = free[j]
            status[i] = 1 if p[j] > 0 else -1
            w[i] = upper[i] if p[j] > 0 else 0.
            continue
        w[free] += p
        reduced = g() + nu
        violations = np.where(status == -1, -reduced, np.where(status == 1, reduced, 0.))
        i = np.argmax(violations)
        if violations[i] <= tol:
            return w, float(nu)
        status[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    invert = _calibration(s['sensor_calibration_sweeps'])
    y = invert(s['y_obs_raw'])
    forward, robin, sources = _chamber(s)
    pulses = s.get('reference_tracer_pulses')
    if pulses:
        robin = _pulse_fit(pulses, invert, forward, robin, sources)
    design, last = forward(robin, sources)
    z = np.asarray(s['Z_nuisance'])
    a = np.column_stack((design, y))
    a -= z @ np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T @ a)
    design, y = a[:, :-1], a[:, -1]
    ridge = s['source_ridge']
    h = design.T @ design + ridge*np.eye(design.shape[1])
    c = -design.T @ y
    budget = float(s['total_budget'])
    upper = np.broadcast_to(s['u_box'], c.shape).astype(float)
    w, nu = _box_qp(h, c, upper, budget)
    reduced = h @ w + c + nu
    lo = np.where(w <= 1e-11, np.maximum(reduced, 0.), 0.)
    hi = np.where(w >= upper-1e-11, np.maximum(-reduced, 0.), 0.)
    residual = design @ w - y
    obj = .5*(residual @ residual + ridge*(w @ w))
    # Minimize the Lagrangian over unconstrained weights to get a dual bound.
    lag_grad = reduced - lo + hi
    dual = (obj + nu*(w.sum()-budget) - lo @ w + hi @ (w-upper)
            - .5*lag_grad @ np.linalg.solve(h, lag_grad))
    return {'weights': w, 'terminal_field': (last @ w).reshape((s['nx'], s['ny']), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': lo,
                                'lam_upper': hi, 'primal_objective': float(obj),
                                'dual_lower_bound': float(min(obj, dual))}}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`443 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c45a2d5fc284a74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c45a2d5f (directory).
```

### Turn 7 — `execute_in_remote_docker` (`22983 ms` LLM)
- **Rationale**: Check whether the archive supplies an external wall prior and whether routine residuals show structured mismatch. Those results determine whether to test a constrained blind correction or retain the nominal baseline.
- **Alternatives Weighed**: `Promote the unconstrained blind wall fit` (rejected: Its wall estimates vary materially by held-out shift and one shift predicts worse than nominal.)

```bash
ls -la /tmp/hegel_continual_archive 2>/dev/null || echo 'No continual archive mounted'; cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import batch_solver as m
b=pickle.load(open('public_batch.pkl','rb')); print('batch length:', len(b)); s=b[0]['study']; print('scalars:', {k:v for k,v in s.items() if np.isscalar(v)}); print('item metadata:',{k:str(v)[:300] for k,v in b[0].items() if k!='study'}); print('row bank:',s['row_bank_id'].shape,s['row_bank_id'].ravel()[:24]); print('upper:',s['u_box']); print('shifts:',s['mu_shifts']); print('nuisance head:',s['Z_nuisance'][:6]);
for pulse in [True,False]:
 a=dict(s); a['reference_tracer_pulses']=s['reference_tracer_pulses'] if pulse else None; o=m.solve_batch(a); print('pulse mode',pulse,'weights',o['weights'],'objective',o['kkt_certificate']['primal_objective'],'gap',o['kkt_certificate']['primal_objective']-o['kkt_certificate']['dual_lower_bound'])
inv=m._calibration(s['sensor_calibration_sweeps']); forward,r,F=m._chamber(s); r=m._pulse_fit(s['reference_tracer_pulses'],inv,forward,r,F); D,_=forward(r,F); w=m.solve_batch(s)['weights']; Z=s['Z_nuisance']; e=inv(s['y_obs_raw'])-D@w; g=np.linalg.lstsq(Z,e,rcond=None)[0]; e-=Z@g; print('residual statistics:', 'gamma',g,'shift std',e.reshape(3,-1).std(1),'bank std',e.reshape(-1,4).std(0),'range',e.min(),e.max()); print('paired sensors across shifts:', np.corrcoef(e.reshape(3,-1))); print('repeated sensors:',len(np.unique(s['sensor_ij'],axis=0)))
PY
```
- **Container Exit Code**: `0` (`527 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
No continual archive mounted
batch length: 1
scalars: {'Lx': 2.4, 'Ly': 2.0, 'nx': 14, 'ny': 12, 'robin_x': 0.880939340203039, 'robin_y': 1.2931872722124154, 'nuisance_ridge': 0.08, 'source_ridge': 0.04, 'u_box': 0.42, 'total_budget': 1.4}
item metadata: {}
row bank: (96,) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
upper: 0.42
shifts: [0.15 0.55 1.1 ]
nuisance head: [[ 0.37211897  0.4724303   1.56662757]
 [ 0.709251    0.034015    1.46999931]
 [ 1.3950572  -1.61046071 -0.3936241 ]
 [-0.19418066  1.50555262 -0.14397216]
 [ 0.31851321 -1.81541884 -0.42117595]
 [ 0.03971741 -1.05029228  0.13981814]]
pulse mode True weights [0.08854811 0.33161789 0.29302007 0.14214732 0.33659508 0.04563901
 0.16243251 0.        ] objective 0.011635030560985766 gap 0.0
pulse mode False weights [0.0890606  0.33058922 0.2935778  0.14255966 0.3339381  0.04882831
 0.16144632 0.        ] objective 0.011626358218643496 gap 0.0
residual statistics: gamma [-0.09210022  0.10884763  0.07261254] shift std [0.00857152 0.01036485 0.00928835] bank std [0.00995405 0.00974393 0.00945787 0.00706863] range -0.021139095435878053 0.0251322703366354
paired sensors across shifts: [[1.         0.37583547 0.13052197]
 [0.37583547 1.         0.22596689]
 [0.13052197 0.22596689 1.        ]]
repeated sensors: 32

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c45a2d5fc284a74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c45a2d5f (directory).
```

### Turn 8 — `execute_in_remote_docker` (`27661 ms` LLM)
- **Rationale**: Mask the public pulses and test wall inference with the actual constrained source solve. Promote it only if it improves held-out shift and sensor prediction without degrading agreement with the independently fitted wall losses.
- **Alternatives Weighed**: `Select blind wall corrections by observation residual alone` (rejected: A correction can fit nuisance and source residuals while moving wall losses away from the independently measured values.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
import batch_solver as m
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; inv=m._calibration(s['sensor_calibration_sweeps']); f,r0,F=m._chamber(s); rp=m._pulse_fit(s['reference_tracer_pulses'],inv,f,r0,F); y=inv(s['y_obs_raw']); Z=s['Z_nuisance']; M=F.shape[1]; b=s['total_budget']; u=np.broadcast_to(s['u_box'],(M,)); n=len(y); g=len(s['sensor_ij']); ix=np.arange(n); ridge=s['source_ridge']; rms=lambda x:np.sqrt(np.mean(x*x)); print('nominal/pulse loss:',r0,rp)
for key in ('kappa_x','kappa_y','vel_x','vel_y'):
 a=s[key]; print(key,a[[0,1,-2,-1]])
def source(r,tr):
 S,U=f(r,F); A=np.column_stack((S[tr],y[tr])); z=Z[tr]; p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); D,T=(A-z@(p@A))[:,:-1],(A-z@(p@A))[:,-1]; w,_=m._box_qp(D.T@D+ridge*np.eye(M),-D.T@T,u,b); gamma=p@(y[tr]-S[tr]@w); return S@w+Z@gamma-y,w,U@w
ref=source(rp,np.ones(n,dtype=bool));
for alpha in (np.inf,.1,.03,.01,.003):
 for kind in ('shift','sensor'):
  results=[]
  for k in range(3 if kind=='shift' else 4):
   tr=(ix//g!=k) if kind=='shift' else (((ix%g)//4)%4!=k)
   def residual(q):
    e=source(r0*np.exp(q),tr)[0][tr]; return np.r_[e,alpha*q]
   r=r0 if not np.isfinite(alpha) else r0*np.exp(least_squares(residual,np.zeros(4),bounds=(-.7,.7)).x)
   e,w,U=source(r,tr); results.append((rms(e[~tr]),np.max(np.abs(r-rp)),rms(U-ref[2])/rms(ref[2])))
  print(alpha,kind,'worst RMSE',max(v[0] for v in results),'worst wall dev',max(v[1] for v in results),'worst rel field dev',max(v[2] for v in results))
PY
```
- **Container Exit Code**: `0` (`1506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nominal/pulse loss: [0.88093934 0.88093934 1.29318727 1.29318727] [0.92107951 0.9211281  1.28646561 1.30310303]
kappa_x [0.46954204 0.50304623 0.50304623 0.46954204]
kappa_y [0.56785556 0.56299472 0.3327164  0.32785556]
vel_x [ 0.35        0.34500413 -0.15511704 -0.20572484]
vel_y [-0.         -0.05560533 -0.25469696 -0.22652476]
inf shift worst RMSE 0.012097861835270946 worst wall dev 0.04018876075720268 worst rel field dev 0.02598392326770887
inf sensor worst RMSE 0.011556073265402407 worst wall dev 0.04018876075720268 worst rel field dev 0.027960056857038576
0.1 shift worst RMSE 0.01204016838806082 worst wall dev 0.12018262760017806 worst rel field dev 0.0263637232165723
0.1 sensor worst RMSE 0.011391607555559513 worst wall dev 0.14780559017727546 worst rel field dev 0.029399986457507247
0.03 shift worst RMSE 0.012010120528328212 worst wall dev 0.3639498427322465 worst rel field dev 0.031069504126571595
0.03 sensor worst RMSE 0.011539343285173856 worst wall dev 0.5070281721889107 worst rel field dev 0.039756754986934435
0.01 shift worst RMSE 0.012045079580917625 worst wall dev 0.5785139952581577 worst rel field dev 0.04022410987788135
0.01 sensor worst RMSE 0.011851590767468398 worst wall dev 0.7067959360770286 worst rel field dev 0.04466716705724825
0.003 shift worst RMSE 0.012088952766599072 worst wall dev 0.7988804482035966 worst rel field dev 0.04357114239387559
0.003 sensor worst RMSE 0.011904978256065731 worst wall dev 0.7426353476554184 worst rel field dev 0.04557782698730849

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c45a2d5fc284a74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c45a2d5f (directory).
```

### Turn 9 — `execute_in_remote_docker` (`37691 ms` LLM)
- **Rationale**: Verify the exact saved candidate in isolation. Held-out tracer error tests whether the implemented stencil transfers; an independent QP solve, KKT residuals, field reconstruction, and active-bound stress cases distinguish certificate or ordering defects from a schema-only pass.
- **Alternatives Weighed**: `Promote ridge-regularized blind wall inference` (rejected: It improved observation prediction by at most 0.00017 on the worst sensor fold, while worsening agreement with tracer-identified wall losses and the reference terminal field.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import minimize, least_squares
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rms=lambda a:np.sqrt(np.mean(a*a)); inv=m._calibration(s['sensor_calibration_sweeps']); f,r0,F=m._chamber(s); P=s['reference_tracer_pulses']; worst=0.
for train,test in ((0,1),(1,0)):
 r=m._pulse_fit([P[train]],inv,f,r0,F); err=f(r,F@P[test]['w_pulse'])[0]-inv(P[test]['y_pulse_raw']); worst=max(worst,max(rms(a) for a in np.split(err,len(s['mu_shifts']))))
print('Candidate worst held-out pulse/shift RMSE:',worst); assert worst<3e-5
for pulses in (P,None):
 a=dict(s,reference_tracer_pulses=pulses); o=m.solve_batch(a); w=o['weights']; k=o['kkt_certificate']; assert np.array_equal(w,k['weights']); z=s['Z_nuisance']; y=inv(s['y_obs_raw']); r=m._pulse_fit(P,inv,f,r0,F) if pulses else r0; S,U=f(r,F); R=np.eye(len(y))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); D=R@S; T=R@y; u=s['u_box']; b=s['total_budget']; rid=s['source_ridge']; obj=lambda v:.5*(np.linalg.norm(D@v-T)**2+rid*(v@v)); grad=lambda v:D.T@(D@v-T)+rid*v
 opt=minimize(obj,np.full(len(w),b/len(w)),jac=grad,bounds=[(0,u)]*len(w),constraints={'type':'eq','fun':lambda v:v.sum()-b,'jac':lambda v:np.ones_like(v)},method='SLSQP',options={'ftol':1e-14}); assert opt.success and abs(obj(w)-opt.fun)<1e-11
 lo,hi=k['lam_lower'],k['lam_upper']; stat=np.max(np.abs(grad(w)+k['nu_sum']-lo+hi)); comp=max(np.max(np.abs(w*lo)),np.max(np.abs((u-w)*hi))); gap=k['primal_objective']-k['dual_lower_bound']; assert min(w.min(),(u-w).min(),lo.min(),hi.min())>=-1e-11 and abs(w.sum()-b)<1e-11; assert max(stat,comp,gap,abs(obj(w)-k['primal_objective']))<1e-10 and gap>=0
 assert np.allclose(o['terminal_field'],(U@w).reshape(s['nx'],s['ny'],order='F'),rtol=0,atol=1e-13)
 print('pulses:',bool(pulses),'stationarity:',stat,'complementarity:',comp,'dual gap:',gap,'independent QP objective delta:',abs(obj(w)-opt.fun))
rng=np.random.default_rng(981); worst=0.
for j in range(500):
 n=2+j%11; A=rng.normal(size=(n+3,n)); H=A.T@A+.04*np.eye(n); c=rng.normal(size=n); u=rng.uniform(.05,1,size=n); u[::5]=0.; b=[0.,u.sum(),u[1],u.sum()*.002,u.sum()*.998,rng.uniform()*u.sum()][j%6]; w,nu=m._box_qp(H,c,u,b); g=H@w+c+nu; d=max(abs(w.sum()-b),max(0.,-min(w)),max(0.,max(w-u)),np.max(np.maximum(g,0)*w),np.max(np.maximum(-g,0)*(u-w))); worst=max(d,worst)
assert worst<1e-9; print('Worst feasibility/complementarity, 500 QPs:',worst)
x=rng.uniform(-1,1,(80,4)); t=rng.uniform(-.3,.3,(64,4)); C=np.eye(4)+rng.normal(scale=.04,size=(4,4)); raw=lambda a:a@C+.015*np.column_stack((a[:,0]*a[:,2],a[:,1]**2,a[:,2]*a[:,3],a[:,0]**3)); err=m._calibration({'ref_rosette':x,'raw_rosette':raw(x)})(raw(t)).reshape(-1,4)-t; assert np.max(abs(err))<1e-9; print('Mixed nonlinear calibration round-trip max:',np.max(abs(err)))
PY
v=$?; OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && exit "$v"
```
- **Container Exit Code**: `0` (`841 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Candidate worst held-out pulse/shift RMSE: 2.7473485360589437e-05
pulses: True stationarity: 1.3877787807814457e-16 complementarity: 0.0 dual gap: 0.0 independent QP objective delta: 4.621303340002214e-15
pulses: False stationarity: 2.246466901389965e-16 complementarity: 0.0 dual gap: 0.0 independent QP objective delta: 4.355890648177763e-15
Worst feasibility/complementarity, 500 QPs: 3.279663743899368e-15
Mixed nonlinear calibration round-trip max: 1.1102230246251565e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c45a2d5fc284a74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c45a2d5f (directory).
```

### Turn 10 — `submit_completed_task` (`14045 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, held-out tracer tests, terminal-field reconstruction, and independent objective and KKT checks. The host receipt verifies the schema only; the scientific metrics come from the separate public-data tests.
- **Alternatives Weighed**: `Infer wall losses from routine observations` (rejected: The small gains in held-out prediction did not offset worse agreement with tracer-estimated wall losses and terminal fields.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:7411'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` and verified it in isolated Python. The public schema passed, and the solver matched an independent optimizer on the public study with and without tracer pulses. No apparatus archive was available; pulse-free batches use the supplied Robin coefficients.`

### Turn 11 — `execute_in_remote_docker` (`35870 ms` LLM)
- **Rationale**: Nested rosette folds can show whether pairwise rational terms improve calibration enough to revisit the inverse. Held-out tracer errors stratified by shift and bank can show whether the pooled wall-fit improvement hides a failing sensor slice.
- **Alternatives Weighed**: `Apply classification or relay diagnostics` (rejected: This study has continuous rosette measurements, one public campaign, and a steady PDE. Within-bank prediction errors test the relevant pooling failure; balanced accuracy and relay modes are not defined.)

```bash
cat >/tmp/audit_m98_margin.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
y = raw - x
rms = lambda a: np.sqrt(np.mean(a*a))
p2, p3 = m._monomials(2), m._monomials(3)
phi2, phi3 = m._features(x, p2), m._features(x, p3)
models = {'affine': phi2[:, :5], 'quadratic': phi2, 'cubic': phi3}
for j in range(4):
    for k in range(j+1, 4):
        for sign in (-1, 1):
            q = (x[:, j] + sign*x[:, k])/np.sqrt(2)
            for a in (.25, 1.):
                models[f'rational_{j}_{k}_{sign}_{a}'] = phi2/(1+a*q[:, None]**2)

def error(a, tr, te):
    b = np.linalg.lstsq(a[tr], y[tr], rcond=None)[0]
    return a[te] @ b - y[te]

ix = np.arange(len(x))
for family in ('affine', 'polynomial', 'rational'):
    options = [name for name in models if (family == 'rational' and name.startswith('rational')) or (family == 'polynomial' and not name.startswith('rational')) or name == family]
    chosen, scores, banks = [], [], []
    for outer in range(4):
        test = ix%4 == outer
        train = ~test
        inner = np.cumsum(train)%3
        def score(name):
            a = models[name]
            folds = [(train & (inner!=k), train & (inner==k)) for k in range(3)]
            if any(tr.sum() <= a.shape[1]+4 for tr, _ in folds):
                return np.inf
            return np.mean([rms(error(a, tr, te))**2 for tr, te in folds])
        name = min(options, key=score)
        chosen.append(name)
        e = error(models[name], train, test)
        scores.append(rms(e)); banks.append(max(rms(z) for z in e.T))
    print('Nested calibration', family, 'worst rosette/bank RMSE', max(scores), max(banks), 'chosen', chosen)

forward, nominal, sources = m._chamber(s)
pulses = s['reference_tracer_pulses']
invert = m._calibration(c)
for ax in ('x','y'):
    a = m._axis(s['kappa_'+ax], s['vel_'+ax], s['L'+ax]/(s['n'+ax]-1))
    defect = np.max(np.abs(np.asarray(a.sum(axis=1))))
    print('Interior no-flux constant-mode defect', ax, defect)
    assert defect < 1e-12
banks = np.asarray(s['row_bank_id'])
worst = {'nominal': 0., 'fitted': 0.}
for i, train in enumerate(pulses):
    test = pulses[1-i]
    robin = m._pulse_fit([train], invert, forward, nominal, sources)
    for name, r in [('nominal', nominal), ('fitted', robin)]:
        pred = forward(r, sources @ test['w_pulse'])[0]
        e = pred - invert(test['y_pulse_raw'])
        e, ids = e.reshape(len(s['mu_shifts']), -1), banks.reshape(len(s['mu_shifts']), -1)
        v = max(rms(row[bid == k]) for row, bid in zip(e, ids) for k in np.unique(bid))
        worst[name] = max(worst[name], v)
print('Worst held-out pulse x shift x channel bank RMSE', worst)
assert worst['fitted'] < 5e-5 and worst['fitted'] < worst['nominal']
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_m98_margin.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`748 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Nested calibration affine worst rosette/bank RMSE 1.1462029741919508e-05 1.4061491709757004e-05 chosen ['affine', 'affine', 'affine', 'affine']
Nested calibration polynomial worst rosette/bank RMSE 1.1462029741919508e-05 1.4061491709757004e-05 chosen ['affine', 'affine', 'affine', 'affine']
Nested calibration rational worst rosette/bank RMSE 1.4669799549695674e-05 1.859485867030367e-05 chosen ['rational_1_3_-1_1.0', 'rational_0_3_1_1.0', 'rational_0_2_-1_1.0', 'rational_2_3_1_1.0']
Interior no-flux constant-mode defect x 3.552713678800501e-15
Interior no-flux constant-mode defect y 3.552713678800501e-15
Worst held-out pulse x shift x channel bank RMSE {'nominal': 0.0006435196600458936, 'fitted': 3.478330889054288e-05}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c45a2d5fc284a74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c45a2d5f (directory).
```

### Turn 12 — `submit_completed_task` (`16302 ms` LLM)
- **Rationale**: The exact saved candidate passed isolated schema validation, independent QP and KKT checks, and terminal-field reconstruction. The separate held-out tracer audit confirmed that the wall-fit improvement persisted across shifts and sensor banks.
- **Alternatives Weighed**: `Add pairwise rational calibration terms` (rejected: Nested validation increased worst sensor-bank RMSE from 1.41e-5 to 1.86e-5.); `Infer wall losses from routine observations alone` (rejected: Small improvements in held-out observation error came with worse agreement with the tracer-estimated wall losses and terminal field.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed in isolated Python, and the certified source solution matched an independent optimizer with and without tracer pulses. No apparatus archive was available, so pulse-free batches use the supplied Robin coefficients.`

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
