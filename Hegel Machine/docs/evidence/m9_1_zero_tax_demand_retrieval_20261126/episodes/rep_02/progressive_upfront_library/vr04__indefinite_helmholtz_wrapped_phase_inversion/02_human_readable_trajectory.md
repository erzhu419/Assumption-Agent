# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (8T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Indefinite Acoustic Helmholtz Scattering & Wrapped-Phase Receiver Calibration` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `4` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `204,975` (`197,726` / `47,488` / `7,249`) |
| **Total Episode Wall Time (s)** | `247.88s` |

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

### Turn 1 — `execute_in_remote_docker` (`17445 ms` LLM)
- **Rationale**: Inspect the public case schema before implementing the solver. Field names and array shapes determine the input mapping, and the available reference values determine which output errors can be measured directly.
- **Alternatives Weighed**: `Reuse the Darcy multigrid solver` (rejected: Its SPD precondition is incompatible with the indefinite complex Helmholtz operator.)

```bash
ls -lh /holdout_task; python3 - <<'PY'
import pickle
import numpy as np
from scipy import sparse
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
    data = pickle.load(f)
def show(x, depth=0):
    if isinstance(x, dict):
        return {k:show(v,depth+1) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return {'type':type(x).__name__,'count':len(x),'items':[show(v,depth+1) for v in x[:8]]}
    if isinstance(x,np.ndarray):
        return (x.shape,str(x.dtype))
    return x
print(show(data))
PY
```
- **Container Exit Code**: `0` (`783 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 01:38 public_helmholtz_cases.pkl
{'type': 'list', 'count': 2, 'items': [{'problem': {'Lx': 1.0, 'Ly': 1.0, 'k0': 11.5, 'beta_imp': 0.85, 'n_sq': ((32, 32), 'float64'), 'basis': ((6, 32, 32), 'float64'), 'pilot_calibration': {'known_field_re': ((12,), 'float64'), 'known_field_im': ((12,), 'float64'), 'amplitude_obs': ((12,), 'float64'), 'wrapped_phase_obs': ((12,), 'float64')}, 'survey': {'sensor_ij': ((140, 2), 'int64'), 'amplitude_obs': ((140,), 'float64'), 'wrapped_phase_obs': ((140,), 'float64'), 'tikhonov_alpha': 1e-06}}, 'gt': {'source_amplitudes': ((6,), 'float64'), 'global_phase_rad': -0.8141122105046681, 'predicted_field_abs': ((32, 32), 'float64')}}, {'problem': {'Lx': 1.0, 'Ly': 1.0, 'k0': 11.5, 'beta_imp': 0.85, 'n_sq': ((32, 32), 'float64'), 'basis': ((6, 32, 32), 'float64'), 'pilot_calibration': {'known_field_re': ((12,), 'float64'), 'known_field_im': ((12,), 'float64'), 'amplitude_obs': ((12,), 'float64'), 'wrapped_phase_obs': ((12,), 'float64')}, 'survey': {'sensor_ij': ((140, 2), 'int64'), 'amplitude_obs': ((140,), 'float64'), 'wrapped_phase_obs': ((140,), 'float64'), 'tikhonov_alpha': 1e-06}}, 'gt': {'source_amplitudes': ((6,), 'float64'), 'global_phase_rad': -1.3120678293854167, 'predicted_field_abs': ((32, 32), 'float64')}}]}
```

### Turn 2 — `execute_in_remote_docker` (`46083 ms` LLM)
- **Rationale**: Compare the forward map at the supplied truth to detect stencil errors, compare ordinary and robust regression to measure the effect of impulses, and jackknife receivers to detect undue dependence on a single channel.
- **Alternatives Weighed**: `Unwrap survey phases and fit them directly` (rejected: Polar-to-complex conversion removes the branch cut exactly and retains the linear source dependence.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from pilot and survey phasors."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    impedance = -1j * k * problem['beta_imp']
    cells = np.arange(nx * ny).reshape(nx, ny)
    diag = np.asarray(-hx * hy * k**2 * n_sq, dtype=np.complex128)
    diag[[0, -1], :] += hy * impedance
    diag[:, [0, -1]] += hx * impedance
    ends0, ends1, vals = [], [], []
    for a, b, w in ((cells[:-1, :], cells[1:, :], hy / hx),
                    (cells[:, :-1], cells[:, 1:], hx / hy)):
        a, b = a.ravel(), b.ravel()
        diag.ravel()[a] += w
        diag.ravel()[b] += w
        ends0.extend((a, b))
        ends1.extend((b, a))
        vals.extend((np.full(a.size, -w), np.full(b.size, -w)))
    matrix = coo_matrix((np.concatenate([diag.ravel(), *vals]),
                         (np.concatenate([cells.ravel(), *ends0]),
                          np.concatenate([cells.ravel(), *ends1]))),
                        shape=(cells.size, cells.size)).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = (hx * hy * basis.reshape(basis.shape[0], -1).T).astype(np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(
        1j * np.asarray(telemetry['wrapped_phase_obs']))


def _fit(a, y, alpha):
    # Pair the real and imaginary equations so each receiver gets one weight.
    x = np.stack((a.real, a.imag), axis=1)
    z = np.stack((y.real, y.imag), axis=1)
    ridge = np.sqrt(alpha) * np.eye(a.shape[1])
    def solve(w):
        sw = np.sqrt(w)
        lhs = np.vstack(((x * sw[:, None, None]).reshape(-1, a.shape[1]), ridge))
        rhs = np.r_[(z * sw[:, None]).ravel(), np.zeros(a.shape[1])]
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    coeff = solve(np.ones(len(y)))
    floor = np.finfo(float).eps * max(np.max(np.abs(y)), np.finfo(float).tiny)
    for _ in range(25):
        residual = np.abs(y - a @ coeff)
        scale = max(np.median(residual), floor)
        new = solve(np.minimum(1.0, 1.5 * scale / np.maximum(residual, floor)))
        if np.linalg.norm(new - coeff) <= 1e-8 * max(np.linalg.norm(new), floor):
            coeff = new
            break
        coeff = new

    # Refit without shrinking the weight of clean receivers. For circular
    # complex Gaussian noise, the 3.5-median gate rejects negligible clean mass.
    residual = np.abs(y - a @ coeff)
    clean = residual <= 3.5 * max(np.median(residual), floor)
    return solve(clean)


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    responses = _responses(problem)
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    received = _phasors(survey) * np.exp(-1j * phase)
    coeff = _fit(responses[i, j], received, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coeff)}
PY
cat > /tmp/helm_public.py <<'PY'
import runpy, pickle
import numpy as np
m=runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
    cases=pickle.load(f)
rel=lambda x,y: np.linalg.norm(x-y)/np.linalg.norm(y)
for case in cases:
    p,gt=case['problem'],case['gt']; s=p['survey']; u=m['_responses'](p); i,j=s['sensor_ij'].T; a=u[i,j]; y=m['_phasors'](s)*np.exp(-1j*gt['global_phase_rad']); x=gt['source_amplitudes']; obs=np.abs(y-a@x)
    est=m['invert_helmholtz_sources_and_phase'](p); ls=np.linalg.lstsq(np.r_[a.real,a.imag],np.r_[y.real,y.imag],rcond=None)[0]
    jack=np.array([m['_fit'](a[np.arange(len(y))!=l],y[np.arange(len(y))!=l],s['tikhonov_alpha']) for l in range(len(y))]); se=max(rel(t,x) for t in jack); fe=max(rel(np.abs(u@t),gt['predicted_field_abs']) for t in jack)
    print('truth forward',rel(np.abs(u@x),gt['predicted_field_abs']),'condition',np.linalg.cond(np.r_[a.real,a.imag]),'sv',np.linalg.svd(np.r_[a.real,a.imag],compute_uv=False),'true residual quantiles',np.quantile(obs,[0,.5,.8,.9,.93,.95,1])); print('OLS source',rel(ls,x),'robust source',rel(est['source_amplitudes'],x),'robust field',rel(est['predicted_field_abs'],gt['predicted_field_abs']),'phase',abs(np.angle(np.exp(1j*(est['global_phase_rad']-gt['global_phase_rad'])))),'jackknife source,field',se,fe)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helm_public.py
```
- **Container Exit Code**: `0` (`753 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
truth forward 1.1272979772244929e-15 condition 2.9618123405365124 sv [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722] true residual quantiles [6.47881901e-06 2.04397750e-03 2.75476983e-03 3.74096331e-03
 1.00151822e-01 3.72538342e-01 7.40687115e-01]
OLS source 0.17180313491718766 robust source 0.0035790065572627386 robust field 0.002289175829731044 phase 0.0006227225657142733 jackknife source,field 0.004379487462965869 0.0026087538267926453
truth forward 9.93191916978576e-16 condition 3.1136094055719403 sv [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252] true residual quantiles [2.09377251e-04 1.80389227e-03 2.88952944e-03 4.26435661e-03
 9.99000576e-02 4.09010034e-01 6.86029873e-01]
OLS source 0.2727211285456088 robust source 0.004715349089309417 robust field 0.003201396421629485 phase 0.00026689595102702407 jackknife source,field 0.0053896201386166446 0.0035909147350485474

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ee95427f17a139b9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ee95427f (directory).
```

### Turn 3 — `execute_in_remote_docker` (`46357 ms` LLM)
- **Rationale**: Make the diagonal C-contiguous so C-order indexed updates cannot silently land in a copy. Verify the exact artifact in isolated Python: a flux-form residual detects grid or boundary assembly errors, pilot jackknifing tests phase calibration dependence, and wrap-boundary cases with concentrated receiver impulses test the inversion where the public cases provide no direct coverage.
- **Alternatives Weighed**: `Retune the clipping threshold on the two public cases` (rejected: The existing margin is substantial, and threshold tuning on two cases would not provide independent evidence of improved generalization.)
- **ATMS NOGOOD DECLARED**: Refuted `Equal receiver weights provide sufficient source accuracy under the observed survey telemetry.` | Witness: `Unweighted source errors are 17.18% and 27.27% on the public cases, versus 0.358% and 0.472% for the robust fit; the forward stencil matches the supplied fields to about 1e-15.` | Blanket Diagnosis: `A small number of receiver impulses dominate the squared complex residual even when the forward model and pilot phase are accurate.` | Sublated Pivot: `Keep the exact forward responses and pilot calibration, and downweight entire receiver phasors before the clean-channel refit.`

```bash
sed -i 's/dtype=np.complex128)$/dtype=np.complex128, order="C")/' /holdout_task/helmholtz_phase_inversion.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helm_public.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np

m = runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
invert = m['invert_helmholtz_sources_and_phase']
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)

def phase_error(a,b):
    return abs(np.angle(np.exp(1j*(a-b))))

def polar(z):
    return dict(amplitude_obs=abs(z), wrapped_phase_obs=np.angle(z))

worst = np.zeros(3)
for case in cases:
    p, gt = case['problem'], case['gt']
    baseline = invert(p)
    assert baseline['source_amplitudes'].shape == (6,)
    assert baseline['predicted_field_abs'].shape == (32,32)
    assert baseline['source_amplitudes'].dtype == baseline['predicted_field_abs'].dtype == np.float64
    for l in range(12):
        pilot = {k: np.delete(v,l) for k,v in p['pilot_calibration'].items()}
        result = invert(dict(p, pilot_calibration=pilot))
        metrics = [rel(result['source_amplitudes'],gt['source_amplitudes']), rel(result['predicted_field_abs'],gt['predicted_field_abs']), phase_error(result['global_phase_rad'],gt['global_phase_rad'])]
        worst = np.maximum(worst,metrics)
print('Worst public leave-one-pilot-out source, field, phase:', worst, flush=True)
assert np.all(worst <= [.03,.03,.02])

rng = np.random.default_rng(2026)
worst = np.zeros(3)
for case in cases:
    p, gt = case['problem'], case['gt']
    for layout in ('C', 'F'):
        q = dict(p, Lx=1.08, Ly=.93, n_sq=np.array(p['n_sq'], order=layout))
        u = m['_responses'](q)
        nx,ny = q['n_sq'].shape
        hx,hy = q['Lx']/nx,q['Ly']/ny
        k, beta = q['k0'], q['beta_imp']
        hu = -hx*hy*k*k*q['n_sq'][...,None]*u
        fx = (u[:-1]-u[1:])*(hy/hx)
        fy = (u[:,:-1]-u[:,1:])*(hx/hy)
        hu[:-1] += fx; hu[1:] -= fx
        hu[:,:-1] += fy; hu[:,1:] -= fy
        hu[[0,-1]] -= 1j*k*beta*hy*u[[0,-1]]
        hu[:,[0,-1]] -= 1j*k*beta*hx*u[:,[0,-1]]
        rhs = hx*hy*np.moveaxis(q['basis'],0,-1)
        assert rel(hu,rhs) < 1e-11, (layout, rel(hu,rhs))
        x = gt['source_amplitudes']
        f = u@x
        s = q['survey']; i,j = s['sensor_ij'].T
        a = u[i,j]; see = f[i,j]
        clean_pilot = q['pilot_calibration']
        known = clean_pilot['known_field_re'] + 1j*clean_pilot['known_field_im']
        for phase in (np.pi-1e-8, -np.pi+1e-8, 0.):
            for attack in ('random', 'largest-response'):
                outliers = rng.choice(len(see), 10, replace=False) if attack=='random' else np.argsort(np.linalg.norm(a,axis=1))[-10:]
                y = see + .0015*(rng.normal(size=len(see))+1j*rng.normal(size=len(see)))
                y[outliers] += rng.uniform(.10,.75,size=10)*np.exp(1j*rng.uniform(-np.pi,np.pi,size=10))
                result = invert(dict(q, pilot_calibration=dict(clean_pilot, **polar(known*np.exp(1j*phase))), survey=dict(s, **polar(y*np.exp(1j*phase)))))
                assert -np.pi < result['global_phase_rad'] <= np.pi
                worst = np.maximum(worst,[rel(result['source_amplitudes'],x),rel(result['predicted_field_abs'],abs(f)),phase_error(result['global_phase_rad'],phase)])
print('Worst rectangular, C/F layout, wrap-boundary and impulse stress source, field, phase:', worst, flush=True)
assert np.all(worst <= [.03,.03,.02])
PY
```
- **Container Exit Code**: `0` (`1604 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
truth forward 1.1272979772244929e-15 condition 2.9618123405365124 sv [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722] true residual quantiles [6.47881901e-06 2.04397750e-03 2.75476983e-03 3.74096331e-03
 1.00151822e-01 3.72538342e-01 7.40687115e-01]
OLS source 0.17180313491718766 robust source 0.0035790065572627386 robust field 0.002289175829731044 phase 0.0006227225657142733 jackknife source,field 0.004379487462965869 0.0026087538267926453
truth forward 9.93191916978576e-16 condition 3.1136094055719403 sv [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252] true residual quantiles [2.09377251e-04 1.80389227e-03 2.88952944e-03 4.26435661e-03
 9.99000576e-02 4.09010034e-01 6.86029873e-01]
OLS source 0.2727211285456088 robust source 0.004715349089309417 robust field 0.003201396421629485 phase 0.00026689595102702407 jackknife source,field 0.0053896201386166446 0.0035909147350485474
Worst public leave-one-pilot-out source, field, phase: [0.00473207 0.00321022 0.00074135]
Worst rectangular, C/F layout, wrap-boundary and impulse stress source, field, phase: [5.78680770e-03 4.45260908e-03 4.44089210e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60d9e3b32d6119ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_60d9e3b3 (directory).
```

### Turn 4 — `submit_completed_task` (`13200 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification. Public receiver and pilot jackknifes, independent stencil residual checks, and rectangular-grid stresses near the phase wrap with concentrated impulses all passed.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3439'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across public and stress tests, worst errors were 0.579% for source amplitudes, 0.446% for field magnitude, and 0.000742 rad for global phase—all below the specified tolerances.`

### Turn 5 — `execute_in_remote_docker` (`29829 ms` LLM)
- **Rationale**: Single-receiver jackknifing can miss dependence on an entire region or phase sector. Remove each block from the real surveys and score all three outputs against supplied truth. If a block breaches tolerance, use design conditioning and the unweighted comparison to distinguish loss of source identifiability from residual contamination.
- **Alternatives Weighed**: `Add nonlinear basis terms or center survey groups` (rejected: The specified forward map is exactly linear in the real source amplitudes after pilot rotation. The data define no acquisition batches, and centering spatial groups would remove physical field information.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(y)

worst = np.zeros(3)
worst_ols = 0.
for cid, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    s = p['survey']
    i, j = s['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    quadrant = 2*(i >= nx/2).astype(int)+(j >= ny/2).astype(int)
    z = m['_phasors'](s)
    sector = np.floor((np.angle(z)+np.pi)/(np.pi/2)).astype(int) % 4
    responses = m['_responses'](p)
    for name, group in [('spatial quadrant', quadrant), ('reported phase sector', sector)]:
        for g in np.unique(group):
            keep = group != g
            reduced = {**s, **{k: s[k][keep] for k in ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs')}}
            r = m['invert_helmholtz_sources_and_phase']({**p, 'survey': reduced})
            phase = abs(np.angle(np.exp(1j*(r['global_phase_rad']-gt['global_phase_rad']))))
            err = np.array([rel(r['source_amplitudes'],gt['source_amplitudes']), rel(r['predicted_field_abs'],gt['predicted_field_abs']), phase])
            worst = np.maximum(worst, err)
            a, y = responses[i[keep],j[keep]], z[keep]*np.exp(-1j*r['global_phase_rad'])
            ar = np.r_[a.real,a.imag]
            ols = np.linalg.lstsq(ar,np.r_[y.real,y.imag],rcond=None)[0]
            ols_err = rel(ols,gt['source_amplitudes'])
            worst_ols = max(worst_ols,ols_err)
            print(cid, name, int(g), 'removed', int((~keep).sum()), 'design condition', round(np.linalg.cond(ar),2), 'robust errors', err, 'OLS source error', round(ols_err,5))
print('Worst blocked public errors (sources, field, phase):',worst,'; worst OLS source:',worst_ols)
assert np.all(worst <= [0.03,0.03,0.02])
PY
```
- **Container Exit Code**: `0` (`583 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 spatial quadrant 0 removed 36 design condition 3.14 robust errors [0.00444805 0.00344887 0.00062272] OLS source error 0.23057
0 spatial quadrant 1 removed 36 design condition 3.51 robust errors [0.00275664 0.00182275 0.00062272] OLS source error 0.18947
0 spatial quadrant 2 removed 31 design condition 3.23 robust errors [0.00555353 0.00302461 0.00062272] OLS source error 0.2312
0 spatial quadrant 3 removed 37 design condition 3.3 robust errors [0.00349652 0.00287067 0.00062272] OLS source error 0.11985
0 reported phase sector 0 removed 38 design condition 2.78 robust errors [0.00429471 0.00319893 0.00062272] OLS source error 0.16166
0 reported phase sector 1 removed 46 design condition 3.43 robust errors [0.00308926 0.00182481 0.00062272] OLS source error 0.22004
0 reported phase sector 2 removed 25 design condition 3.09 robust errors [0.0046567  0.00282297 0.00062272] OLS source error 0.21924
0 reported phase sector 3 removed 31 design condition 2.89 robust errors [0.00361606 0.00265698 0.00062272] OLS source error 0.17344
1 spatial quadrant 0 removed 42 design condition 3.5 robust errors [0.0059749  0.00399188 0.0002669 ] OLS source error 0.4501
1 spatial quadrant 1 removed 36 design condition 3.42 robust errors [0.00602678 0.0039954  0.0002669 ] OLS source error 0.29147
1 spatial quadrant 2 removed 31 design condition 3.31 robust errors [0.00484482 0.00333085 0.0002669 ] OLS source error 0.25562
1 spatial quadrant 3 removed 31 design condition 3.68 robust errors [0.00628773 0.00384921 0.0002669 ] OLS source error 0.55322
1 reported phase sector 0 removed 37 design condition 3.34 robust errors [0.004867   0.00311621 0.0002669 ] OLS source error 0.46338
1 reported phase sector 1 removed 31 design condition 3.32 robust errors [0.00627976 0.00448755 0.0002669 ] OLS source error 0.21991
1 reported phase sector 2 removed 43 design condition 3.1 robust errors [0.00518866 0.00366993 0.0002669 ] OLS source error 0.36635
1 reported phase sector 3 removed 29 design condition 3.3 robust errors [0.00400304 0.00257333 0.0002669 ] OLS source error 0.35035
Worst blocked public errors (sources, field, phase): [0.00628773 0.00448755 0.00062272] ; worst OLS source: 0.5532223525226884

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60d9e3b32d6119ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_60d9e3b3 (directory).
```

### Turn 6 — `submit_completed_task` (`13799 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, independent stencil residual checks, phase-wrap and concentrated-impulse stresses, and spatial and phase-sector holdouts on the public measurements. The block holdouts stayed below tolerance without retuning.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3439'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and stress tests, worst errors were 0.629% for source amplitudes, 0.449% for field magnitude, and 0.000742 rad for global phase. All were below the specified tolerances.`

### Turn 7 — `execute_in_remote_docker` (`33003 ms` LLM)
- **Rationale**: Jointly withhold the strongest pilot and an entire survey quadrant to test a failure that separate holdouts can miss. Use the fitted complex responses to measure coherent phase discrepancy on unseen receivers, with the impulse cutoff set only on training receivers. A material discrepancy would warrant inspecting the shared-phase assumption; otherwise retain the verified linear model.
- **Alternatives Weighed**: `Fit separate phase offsets for survey quadrants` (rejected: The specification has one shared oscillator, and the survey has no batch labels. Test for coherent held-out phase discrepancies before considering any change.)

```bash
cat > /tmp/helm_phase_audit.py <<'PY'
import pickle
import runpy
import numpy as np

solver = runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)

def circular(a):
    return abs(np.angle(np.exp(1j*a)))

worst = np.zeros(3)
worst_drift = 0.
for case in cases:
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    known = pilot['known_field_re'] + 1j*pilot['known_field_im']
    drop = np.argmax(abs(known))
    reduced_pilot = {k: np.delete(v, drop) for k, v in pilot.items()}
    i, j = survey['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    quadrant = 2*(i >= nx/2).astype(int) + (j >= ny/2).astype(int)
    a = solver['_responses'](p)[i,j]
    raw = solver['_phasors'](survey)
    for g in np.unique(quadrant):
        held = quadrant == g
        subset = {**survey, **{k: survey[k][~held] for k in ('sensor_ij','amplitude_obs','wrapped_phase_obs')}}
        result = solver['invert_helmholtz_sources_and_phase']({**p, 'pilot_calibration': reduced_pilot, 'survey': subset})
        x, phase = result['source_amplitudes'], result['global_phase_rad']
        err = [rel(x,gt['source_amplitudes']), rel(result['predicted_field_abs'],gt['predicted_field_abs']), circular(phase-gt['global_phase_rad'])]
        worst = np.maximum(worst, err)
        pred = a @ x
        aligned = raw * np.exp(-1j*phase)
        residual = abs(aligned-pred)
        # Fit and choose the impulse cutoff on training receivers only.
        clean_held = held & (residual <= 3.5*np.median(residual[~held]))
        drift = abs(np.angle(np.vdot(pred[clean_held], aligned[clean_held])))
        worst_drift = max(worst_drift, drift)
        print(f'quadrant {g}: {clean_held.sum()}/{held.sum()} retained, held-out phase discrepancy {drift:.6f} rad, errors {err}')
print('Worst joint holdouts (source, field, phase):', worst)
print('Worst held-out survey-to-pilot phase discrepancy:', worst_drift)
assert np.all(worst <= [.030,.030,.020])
assert worst_drift <= .020, 'Inspect coherent spatial phase residual before accepting a shared oscillator phase'
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helm_phase_audit.py
```
- **Container Exit Code**: `0` (`522 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
quadrant 0: 33/36 retained, held-out phase discrepancy 0.002657 rad, errors [0.004499210678349117, 0.003464615174508801, 0.00043660711014181786]
quadrant 1: 34/36 retained, held-out phase discrepancy 0.001044 rad, errors [0.002786756752715461, 0.001841914762933808, 0.00043660711014181786]
quadrant 2: 29/31 retained, held-out phase discrepancy 0.001469 rad, errors [0.005593340692830217, 0.0030410658635175816, 0.00043660711014181786]
quadrant 3: 34/37 retained, held-out phase discrepancy 0.005858 rad, errors [0.0034939763962888635, 0.002867686880561217, 0.00043660711014181786]
quadrant 0: 38/42 retained, held-out phase discrepancy 0.003994 rad, errors [0.005966248660662151, 0.003988468915335879, 0.00031465875954528316]
quadrant 1: 34/36 retained, held-out phase discrepancy 0.001695 rad, errors [0.006033244807908133, 0.003996570775519799, 0.00031465875954528316]
quadrant 2: 28/31 retained, held-out phase discrepancy 0.002928 rad, errors [0.0048481685472903015, 0.0033386360170887946, 0.00031465875954528316]
quadrant 3: 30/31 retained, held-out phase discrepancy 0.000136 rad, errors [0.006298882042978582, 0.003853045330664969, 0.00031465875954528316]
Worst joint holdouts (source, field, phase): [0.00629888 0.00399657 0.00043661]
Worst held-out survey-to-pilot phase discrepancy: 0.005858216346033883

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60d9e3b32d6119ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_60d9e3b3 (directory).
```

### Turn 8 — `submit_completed_task` (`16354 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification, an independent finite-volume residual check, rectangular-grid and phase-wrap stresses with concentrated impulses, and joint pilot and survey-quadrant holdouts on public measurements. No tested slice breached tolerance or warranted a model change.
- **Alternatives Weighed**: `Fit separate phase offsets by survey quadrant` (rejected: The specification defines one shared oscillator. Even after withholding the strongest pilot, the largest held-out survey-to-pilot phase discrepancy was only 0.00586 rad.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across public and stress tests, the worst errors were 0.630% for source amplitudes, 0.449% for field magnitude, and 0.000742 rad for global phase. All three met the specified tolerances.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/helmholtz_phase_inversion.py": "FILE:3439"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_source_rel_l2": 0.004235,
    "max_phase_abs_err_rad": 0.000852,
    "max_field_abs_rel_l2": 0.003,
    "elapsed_sec": 0.1084
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1084}\n__HEGEL_VERIFIER_AUTH_438265a41a99c9c7605d1bf6c9b9b43d__:{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1084}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5939.69,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
