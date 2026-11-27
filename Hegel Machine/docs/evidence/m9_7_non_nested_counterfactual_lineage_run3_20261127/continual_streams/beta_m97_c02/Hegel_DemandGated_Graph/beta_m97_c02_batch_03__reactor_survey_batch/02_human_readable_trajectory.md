# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m97_c02_batch_03__reactor_survey_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream` |
| **Wave / Wormhole Family** | `beta_m97_c02_step_03` / `fam_kronecker_boundary_regimes_and_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `227,753` (`220,415` / `78,464` / `7,338`) |
| **Total Episode Wall Time (s)** | `269.28s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`root_A03_ill_posed_regularization_and_certificates`**: AXIOM A03 (HEAVY-TAILED REGULARIZATION & DUAL CERTIFICATES): Under outliers, stage drift, or ill-posed combinatorial landscapes, replace naive L2 least squares with heavy-tailed robust losses (Huber/Cauchy/MAD) or exact variable projection, and verify optimality/safety margins via dual bounds or worst-corner simulation.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14390 ms` LLM)
- **Rationale**: Inspect the validator and predecessor to identify interface or boundary differences before reusing code, and look for a verified nuisance-profiling or KKT primitive.

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && sed -n '1,240p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex KKT'
```
- **Container Exit Code**: `0` (`1063 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator
```

### Turn 2 — `execute_in_remote_docker` (`16054 ms` LLM)
- **Rationale**: The first output was truncated. Read the missing operator and validator sections to resolve array ordering and multiplier sign, and inspect the sweep coverage to determine when archive fallback is permitted.

```bash
cd /holdout_task; sed -n '60,180p' validate_public_schema.py; sed -n '1,155p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -45; python3 - <<'PY'
import json,pickle,numpy as np
s=pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))
for k,v in s.items():
 a=np.asarray(v); print(k,type(v).__name__,a.shape,str(v)[:350] if a.size<18 else '')
print('sweeps',str(s['sensor_calibration_sweeps'])[:2500]); h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive structure',str(h)[:1600])
PY
```
- **Container Exit Code**: `1` (`405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ecution_available": true,
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
  ]
}
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 3 — `execute_in_remote_docker` (`19468 ms` LLM)
- **Rationale**: Read the studies using the validator's list format. The focused operator excerpt resolves indexing and coefficient conventions; sweep coverage determines which calibration paths need testing.

```bash
cd /holdout_task && python3 - <<'PY'
import json,pickle,numpy as np
items=pickle.load(open('public_beta_m97_c02_batch_03.pkl','rb'))
print('Study count:',len(items))
for key,value in items[0]['study'].items():
 a=np.asarray(value); print(key,a.shape,str(value) if a.ndim==0 and not isinstance(value,dict) else '')
for idx,item in enumerate(items):
 s=item['study']; c=s['sensor_calibration_sweeps']; b=np.asarray(c['bank_id']); print(idx,'mode',s['recirculation_loop_mode'],'bank counts',np.bincount(b,minlength=4).tolist(),'GT keys',list(item['gt']));
 if idx<3: print('sweep',c)
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history records',len(h),'; last record preview',str(h[-1])[:450])
PY
sed -n '16,104p' /tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bration(bank)
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offsets[rows]) / gains[rows]


def _axis(kappa, velocity, robin, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    mat = np.zeros((n, n))
    face = (kappa[1:] + kappa[:-1]) / (2.0 * h * h)
    i = np.arange(1, n - 1)
    v = velocity[i] / h
    mat[i, i - 1] = -face[:-1] - np.maximum(v, 0)
    mat[i, i + 1] = -face[1:] - np.maximum(-v, 0)
    mat[i, i] = face[:-1] + face[1:] + abs(v)
    mat[0, 0] = kappa[0] / (h * h) + robin / h
    mat[0, 1] = -kappa[0] / (h * h)
    mat[-1, -1] = kappa[-1] / (h * h) + robin / h
    mat[-1, -2] = -kappa[-1] / (h * h)
    return mat


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    lx, ly = study['Lx'], study['Ly']
    excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace']) - study['critical_vortex_circulation'])
    v_mem = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0.0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], study['Ly'] / (ny - 1))
    operator = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx))
    diag = np.diag_indices(nx * ny)
    reaction = np.asarray(study['V_reaction_2d'])
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        walls = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * walls * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator[diag] += reaction.ravel(order='F')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij'], dtype=int).T
    sensor_rows = ix + nx * iy
    blocks = []
    for mu in study['mu_shifts']:
        shifted = operator.copy()
        shifted[diag] += mu
        fields = np.linalg.solve(shifted, rhs)
        blocks.append(fields[sensor_rows])
    return np.vstack(blocks), fields


def _box_simplex(design, target, ridge, upper, budget):
    n = design.shape[1]
    hess = design.T @ design + ridge * np.eye(n)
    linear = design.T @ target
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    if budget == 0:
        state[:] = -1
```

### Turn 4 — `execute_in_remote_docker` (`42298 ms` LLM)
- **Rationale**: Test the forward matrix, calibrated target, inverse weights, and terminal field separately so a mismatch can be traced to its source. Compute the tangential gap against the public reference rather than relying on schema validation alone.
- **Alternatives Weighed**: `Change the periodic grid to endpoint-exclusive coordinates immediately` (rejected: The task specifies a boundary and reaction change but no new grid convention; compare the inherited grid against the public forward matrix first.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
prior = Path('/tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py').read_text()
prior = prior.replace('Calibrated history-dependent reactor tomography', 'Calibrated periodic and history-dependent reactor tomography').replace('solve_beta_m97_c02_batch_01', 'solve_beta_m97_c02_batch_03').replace('solve_beta_m97_c02_batch_02', 'solve_beta_m97_c02_batch_03')
prior = prior.replace('def _axis(kappa, velocity, robin, h):', 'def _axis(kappa, velocity, robin, h, periodic=False):')
old = '    face = (kappa[1:] + kappa[:-1]) / (2.0 * h * h)'
new = '''    if periodic:
        i = np.arange(n)
        im, ip = (i - 1) % n, (i + 1) % n
        km = (kappa[i] + kappa[im]) / (2.0 * h * h)
        kp = (kappa[i] + kappa[ip]) / (2.0 * h * h)
        v = velocity / h
        np.add.at(mat, (i, im), -km - np.maximum(v, 0))
        np.add.at(mat, (i, ip), -kp - np.maximum(-v, 0))
        mat[i, i] += km + kp + np.abs(v)
        return mat
''' + old
assert old in prior
prior = prior.replace(old, new)
start, end = prior.index('    sealed ='), prior.index('    operator =')
prior = prior[:start] + '''    mode = study['recirculation_loop_mode']
    periodic = mode > 0.50
    sealed = mode == 0 and v_mem > 0.0
    hx, hy = lx / (nx - 1), ly / (ny - 1)
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], hx, periodic)
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], hy, periodic)
''' + prior[end:]
old = "    reaction = np.asarray(study['V_reaction_2d'])\n    if sealed:"
new = """    reaction = np.asarray(study['V_reaction_2d'])
    if periodic:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        reaction = reaction + 0.75 * mode * np.cos(2 * np.pi * xs / lx) * np.cos(2 * np.pi * ys / ly)
    elif sealed:"""
assert old in prior
Path('beta_m97_c02_batch_03_solver.py').write_text(prior.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/beta_m97_c02_batch_03_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_beta_m97_c02_batch_03.pkl','rb'))[0]
s,g=item['study'],item['gt']; f,fields=m._forward(s); z=s['Z_nuisance']; vals=np.column_stack((m._calibrate(s),f)); vals-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@vals)
o=m.solve_beta_m97_c02_batch_03(s)
for k,a in [('S_tilde',vals[:,1:]),('r_y',vals[:,0]),('w_opt',o['weights']),('terminal_field_A',o['terminal_field'])]:
 ref=np.asarray(g[k]); d=a-ref; print(k,'maxabs',np.max(abs(d)), 'rel_l2',np.linalg.norm(d)/np.linalg.norm(ref))
c=o['kkt_certificate']; w=o['weights']; S=g['S_tilde']; y=g['r_y']; grad=S.T@(S@w-y)+s['source_ridge']*w; nu=c['nu_sum']; print('reference tangential gap',grad@w-nu*s['total_budget']-s['u_box']*np.minimum(0,grad-nu).sum())
PY
```
- **Container Exit Code**: `0` (`2119 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
S_tilde maxabs 1.1102230246251565e-16 rel_l2 8.741892313206741e-16
r_y maxabs 7.420397807322843e-08 rel_l2 5.104468814810004e-07
w_opt maxabs 2.6505391939601886e-07 rel_l2 5.370919684767995e-07
terminal_field_A maxabs 1.3292430045552805e-08 rel_l2 1.754351769321318e-07
reference tangential gap 3.7617936199732527e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0952c45bef40509d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0952c45b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`42244 ms` LLM)
- **Rationale**: Run the full reference certificate verifier, not just its gap formula. Stress the boundary switch, sparse and spiky calibration banks, and exact budget corners so any failure points to a specific branch to fix.
- **Alternatives Weighed**: `Retune the periodic discretization` (rejected: The forward matrix already agrees at roundoff; the remaining public discrepancy is in the calibrated target.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import ast
import importlib.util
import json
import math
import pickle
from pathlib import Path
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('candidate', 'beta_m97_c02_batch_03_solver.py')
prior = load('prior', '/tmp/hegel_continual_archive/beta_m97_c02_batch_02_solver.py')
tree = ast.parse(Path('validate_public_schema.py').read_text())
checker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate')
ns = dict(np=np, math=math)
exec(compile(ast.Module(body=[checker], type_ignores=[]), 'validate_public_schema.py', 'exec'), ns)
verify = ns[checker.name]
with open('public_beta_m97_c02_batch_03.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, g = item['study'], item['gt']
o = m.solve_beta_m97_c02_batch_03(s)
public = verify(g['S_tilde'], g['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], o['kkt_certificate'])
print('Public reference certificate:', public)
assert public['valid'], public
for mode, trace in [(0.50, s['past_vortex_circulation_trace']), (0.0, s['past_vortex_circulation_trace']), (0.0, np.zeros_like(s['past_vortex_circulation_trace']))]:
    case = dict(s, recirculation_loop_mode=mode, past_vortex_circulation_trace=trace)
    for actual, expected in zip(m._forward(case), prior._forward(case)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
print('Robin threshold and sealed/unsealed lineage branches agree with predecessor')
# Check that wraparound restores constant-field conservation and uses both endpoint velocities.
k = np.array([1., 2., 4., 3.]); v = np.array([0.2, -0.3, 0.1, -0.4]); h = 0.25
A = m._axis(k, v, 12., h, periodic=True)
np.testing.assert_allclose(A @ np.ones(k.size), 0, atol=1e-12)
np.testing.assert_allclose([A[0,-1], A[-1,0]], [-2/h**2-0.2/h, -2/h**2-0.4/h], atol=1e-12)
# Verify bank-local history use and rejection of a single EMI spike when clean coverage remains.
c = s['sensor_calibration_sweeps']; b = c['bank_id']; rows = s['row_bank_id']
keep = (b != 1); keep[np.flatnonzero(b == 1)[0]] = True
trimmed = {key: np.asarray(value)[keep].copy() for key, value in c.items()}
trimmed['raw_readout'][np.flatnonzero(trimmed['bank_id'] == 2)[0]] += 9
actual = m._calibrate(dict(s, sensor_calibration_sweeps=trimmed))
assert np.all(np.isfinite(actual))
expected = m._calibrate(s)
gain, offset = prior._historical_calibration(1)
expected[rows == 1] = (s['y_obs_raw'][rows == 1] - offset) / gain
for bank in [0, 3]: np.testing.assert_allclose(actual[rows == bank], expected[rows == bank], atol=1e-13, rtol=0)
emi_delta = np.max(np.abs(actual[rows == 2] - expected[rows == 2]))
assert emi_delta < 2e-6, emi_delta
print('Sparse-bank and isolated EMI checks passed; EMI calibration delta:', emi_delta)
# Stress lower, upper and mixed active sets on the measured design, including exact budget corners.
worst = 0.
for reverse in (False, True):
    S = g['S_tilde']; y = g['r_y'] * (-4 if reverse else 1)
    for budget in (0., 0.001, 1., 2.6, 6., 11., 11.999, 12.):
        upper, ridge = 1., s['source_ridge']
        w, nu = m._box_simplex(S, y, ridge, upper, budget)
        r = S @ w - y; grad = S.T @ r + ridge*w
        j = float(0.5*(r@r + ridge*(w@w)))
        cert = dict(weights=w, nu_sum=nu, lam_lower=np.maximum(grad-nu,0), lam_upper=np.maximum(nu-grad,0), primal_objective=j, dual_lower_bound=j-grad@w+nu*budget+upper*np.minimum(0,grad-nu).sum())
        result = verify(S, y, ridge, upper, budget, cert)
        assert result['valid'], (reverse, budget, result)
        worst = max(worst, result['gap'])
print('Worst active-set stress gap:', worst)
PY
```
- **Container Exit Code**: `0` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public reference certificate: {'valid': True, 'reason': 'ok', 'true_primal': 0.017882437181084634, 'verified_dual_lb': 0.017882061001722637, 'gap': 3.7617936199732527e-07}
Robin threshold and sealed/unsealed lineage branches agree with predecessor
Sparse-bank and isolated EMI checks passed; EMI calibration delta: 9.251299681878322e-08
Worst active-set stress gap: 3.552713678800501e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0952c45bef40509d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0952c45b (directory).
```

### Turn 6 — `submit_completed_task` (`14644 ms` LLM)
- **Rationale**: The final artifact passed clean-process verification. The public forward matrix agrees to roundoff, and the certificate passed the independent verifier on both the public reference and the budget stress cases.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_03_solver.py': 'FILE:8270'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public schema and independent KKT checks passed, as did the boundary, calibration, and budget stress tests.`

### Turn 7 — `execute_in_remote_docker` (`36792 ms` LLM)
- **Rationale**: Compare the specified physics and nuisance projection with ablations on the public measurements. Holding out real sweep points tests calibration sensitivity, while thinning each bank in turn tests whether archive fallback changes other banks or breaks the KKT certificate.
- **Alternatives Weighed**: `Test classifier balance, relay modes, or HJB drift` (rejected: These are not part of this reactor inversion. The relevant checks are bank-local calibration, nuisance projection, and the periodic transport stencil.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
from pathlib import Path
import numpy as np

p = Path('beta_m97_c02_batch_03_solver.py').resolve()
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_beta_m97_c02_batch_03.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, g = item['study'], item['gt']
sw = s['sensor_calibration_sweeps']; banks = sw['bank_id']; rows = s['row_bank_id']
z = s['Z_nuisance']; ridge = s['nuisance_ridge']
project = lambda a: a - z @ np.linalg.solve(z.T @ z + ridge*np.eye(z.shape[1]), z.T @ a)

def audit(st):
    y = m._calibrate(st)
    S, fields = m._forward(st)
    mat, target = project(S), project(y)
    out = m.solve_beta_m97_c02_batch_03(st)
    w = out['weights']; nu = out['kkt_certificate']['nu_sum']
    grad = mat.T @ (mat @ w - target) + st['source_ridge']*w
    budget, upper = st['total_budget'], st['u_box']
    gap = grad@w - nu*budget - upper*np.minimum(0, grad-nu).sum()
    assert abs(w.sum()-budget) < 1e-8 and w.min() >= -1e-8 and w.max() <= upper+1e-8 and abs(gap) < 1e-8
    np.testing.assert_allclose(out['terminal_field'].ravel(order='F'), fields @ w, atol=1e-12, rtol=1e-12)
    return out, gap

out, gap = audit(s)
raw = m._calibrate(s)
reference_y = g['r_y']
ignored, _ = audit(dict(s, aux_manifold_telemetry=-1000*np.asarray(s['aux_manifold_telemetry'])))
for k in ('weights','terminal_field'):
    np.testing.assert_array_equal(ignored[k], out[k])
print('Profiled target max error:', np.max(np.abs(project(raw)-reference_y)), '; omitting nuisance projection:', np.max(np.abs(raw-reference_y)))
old = dict(s, recirculation_loop_mode=0.50)
print('Forward max error with periodic physics:', np.max(np.abs(project(m._forward(s)[0])-g['S_tilde'])), '; substituting Robin physics:', np.max(np.abs(project(m._forward(old)[0])-g['S_tilde'])))
worst = (0., None)
for bank in range(4):
    ii = np.flatnonzero(banks == bank)
    for i in ii:
        keep = np.arange(banks.size) != i
        c = {k: v[keep] for k,v in sw.items()}
        err = np.max(np.abs(project(m._calibrate(dict(s, sensor_calibration_sweeps=c)))-reference_y))
        worst = max(worst, (float(err), int(bank)), key=lambda v:v[0])
    mask = (banks != bank) | (np.arange(banks.size) == ii[0])
    c = {k:v[mask] for k,v in sw.items()}
    sparse = dict(s, sensor_calibration_sweeps=c)
    _, sparse_gap = audit(sparse)
    cy = m._calibrate(sparse)
    np.testing.assert_array_equal(cy[rows != bank], raw[rows != bank])
    print('Sparse bank', bank, 'max unprofiled transfer shift', np.max(np.abs(cy-raw)), 'self-consistent gap', sparse_gap)
print('Worst leave-one-sweep-point-out profiled target max error, bank:', worst)
assert worst[0] < 1e-6
PY
```
- **Container Exit Code**: `0` (`573 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Profiled target max error: 7.420397807322843e-08 ; omitting nuisance projection: 2.6789289643459564
Forward max error with periodic physics: 1.1102230246251565e-16 ; substituting Robin physics: 0.041889056150938256
Sparse bank 0 max unprofiled transfer shift 4.268617446179235e-07 self-consistent gap 3.434752482434078e-16
Sparse bank 1 max unprofiled transfer shift 2.1128806926284938e-07 self-consistent gap 2.5500435096859064e-16
Sparse bank 2 max unprofiled transfer shift 4.953942043250947e-07 self-consistent gap 1.4051260155412137e-16
Sparse bank 3 max unprofiled transfer shift 1.9818775420787915e-07 self-consistent gap 3.937822290467352e-16
Worst leave-one-sweep-point-out profiled target max error, bank: (1.2980145180918612e-07, 3)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0952c45bef40509d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0952c45b (directory).
```

### Turn 8 — `submit_completed_task` (`13992 ms` LLM)
- **Rationale**: Clean-process verification passed on the unchanged artifact. Across all four banks, the worst leave-one-sweep-point-out profiled target error was 1.30e-7, and sparse-bank fallback left the other banks unchanged.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/beta_m97_c02_batch_03_solver.py': 'FILE:8270'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public schema and KKT certificate passed, along with boundary, calibration, and budget stress tests.`

### Turn 9 — `execute_in_remote_docker` (`26837 ms` LLM)
- **Rationale**: The prior jackknife measured error in the profiled target, not its effect on the reference certificate. This check measures that effect directly: a failure identifies the bank and sweep row to inspect; a pass supports submitting the unchanged artifact.
- **Alternatives Weighed**: `Add nonlinear calibration terms or classifier diagnostics` (rejected: The study specifies affine calibration and a continuous inverse problem. The relevant remaining question is whether observed calibration sensitivity can consume the certificate margin.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import ast, importlib.util, math, pickle
from pathlib import Path
import numpy as np
p = Path('beta_m97_c02_batch_03_solver.py').resolve()
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
f = next(n for n in ast.parse(Path('validate_public_schema.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate')
ns = dict(np=np, math=math)
exec(compile(ast.Module(body=[f], type_ignores=[]), 'validate_public_schema.py', 'exec'), ns)
verify = ns[f.name]
with open('public_beta_m97_c02_batch_03.pkl', 'rb') as src:
    item = pickle.load(src)[0]
s, g = item['study'], item['gt']
sweep = s['sensor_calibration_sweeps']
banks = np.asarray(sweep['bank_id'])
results = []
for idx in range(banks.size):
    keep = np.arange(banks.size) != idx
    st = dict(s, sensor_calibration_sweeps={k: np.asarray(v)[keep] for k, v in sweep.items()})
    out = m.solve_beta_m97_c02_batch_03(st)
    cert = verify(g['S_tilde'], g['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    results.append((idx, int(banks[idx]), cert, float(np.max(np.abs(out['terminal_field'] - g['terminal_field_A'])))))
print('Reference certificate failures:', [(i,b,r) for i,b,r,e in results if not r['valid']])
valid = [(r['gap'], i, b) for i,b,r,e in results if r['valid']]
print('Worst passing reference gap (gap, sweep row, bank):', max(valid, default=None))
print('Worst terminal max absolute error:', max(e for i,b,r,e in results))
assert all(r['valid'] for i,b,r,e in results)
PY
```
- **Container Exit Code**: `0` (`1003 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Reference certificate failures: []
Worst passing reference gap (gap, sweep row, bank): (5.493453561142281e-07, 24, 2)
Worst terminal max absolute error: 1.8180403071887952e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0952c45bef40509d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0952c45b (directory).
```

### Turn 10 — `submit_completed_task` (`13468 ms` LLM)
- **Rationale**: The saved artifact passed clean-process verification. The worst held-out terminal-field error was 1.82e-8, and the boundary, sparse-bank, EMI, and budget stress tests passed. No further edit is warranted.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_m97_c02_batch_03_solver.py`. The public schema and KKT certificate passed, including all 48 leave-one-sweep-point-out tests.`

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
