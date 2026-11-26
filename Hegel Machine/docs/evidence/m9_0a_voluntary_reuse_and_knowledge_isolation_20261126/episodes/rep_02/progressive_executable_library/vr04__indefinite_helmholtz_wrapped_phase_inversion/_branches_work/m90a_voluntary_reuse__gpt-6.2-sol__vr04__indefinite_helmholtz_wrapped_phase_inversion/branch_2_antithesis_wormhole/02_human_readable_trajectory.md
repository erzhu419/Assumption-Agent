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
| **Turns Used / Max Turns** | `8 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `216,324` (`207,862` / `36,352` / `8,462`) |
| **Total Episode Wall Time (s)** | `268.81s` |

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

### Turn 1 — `execute_in_remote_docker` (`15552 ms` LLM)
- **Rationale**: Inspect the public input layout, receiver counts, and available reference outputs before choosing the robust fit or writing the validation harness.
- **Alternatives Weighed**: `Reuse the SPD Darcy solver` (rejected: The absorbing Helmholtz operator is complex symmetric, not Hermitian positive definite.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle, numpy as np
from collections.abc import Mapping
with open('public_helmholtz_cases.pkl','rb') as f: data=pickle.load(f)
def show(x,d=0):
 if isinstance(x,Mapping):
  for k,v in x.items():
   print('  '*d, k, end=': ')
   if isinstance(v,(Mapping,list,tuple)) and d<4: print(); show(v,d+1)
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype, 'range', (np.min(np.abs(v)),np.max(np.abs(v))) if v.size and np.issubdtype(v.dtype,np.number) else '')
   else: print(repr(v)[:180])
 elif isinstance(x,(list,tuple)):
  print('  '*d,type(x).__name__,len(x))
  if d<4:
   for v in x[:8]: show(v,d+1)
 else: print('  '*d,repr(x)[:180])
show(data)
PY
```
- **Container Exit Code**: `0` (`640 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
list 2
   problem: 
     Lx: 1.0
     Ly: 1.0
     k0: 11.5
     beta_imp: 0.85
     n_sq: (32, 32) float64 range (0.8013327008377176, 2.1484758890462508)
     basis: (6, 32, 32) float64 range (7.777719264893782e-58, 24.813476666399524)
     pilot_calibration: 
       known_field_re: (12,) float64 range (0.06040417987375535, 1.9271942759646705)
       known_field_im: (12,) float64 range (0.0029404880356104714, 0.7959843180131354)
       amplitude_obs: (12,) float64 range (0.33798887730235655, 2.0410925014957595)
       wrapped_phase_obs: (12,) float64 range (0.1557422688262286, 2.332925420945619)
     survey: 
       sensor_ij: (140, 2) int64 range (0, 31)
       amplitude_obs: (140,) float64 range (0.003940612776623732, 0.6957272476555351)
       wrapped_phase_obs: (140,) float64 range (0.005426360299870929, 3.1403266841958404)
       tikhonov_alpha: 1e-06
   gt: 
     source_amplitudes: (6,) float64 range (0.1518719487599487, 1.414714791178747)
     global_phase_rad: -0.8141122105046681
     predicted_field_abs: (32, 32) float64 range (0.0029623351218316853, 0.22726746347359125)
   problem: 
     Lx: 1.0
     Ly: 1.0
     k0: 11.5
     beta_imp: 0.85
     n_sq: (32, 32) float64 range (0.8013327008377176, 2.1484758890462508)
     basis: (6, 32, 32) float64 range (7.777719264893782e-58, 24.813476666399524)
     pilot_calibration: 
       known_field_re: (12,) float64 range (0.027533252859279322, 1.2552135205309114)
       known_field_im: (12,) float64 range (0.031297347691756655, 0.6534837285226502)
       amplitude_obs: (12,) float64 range (0.21619322732772722, 1.3575513988296413)
       wrapped_phase_obs: (12,) float64 range (0.6076904610419879, 2.9695938176749612)
     survey: 
       sensor_ij: (140, 2) int64 range (0, 31)
       amplitude_obs: (140,) float64 range (0.003413999204516145, 0.7813177048211306)
       wrapped_phase_obs: (140,) float64 range (0.027672989783847125, 3.1039441222422894)
       tikhonov_alpha: 1e-06
   gt: 
     source_amplitudes: (6,) float64 range (0.23023357171847825, 1.35)
     global_phase_rad: -1.3120678293854167
     predicted_field_abs: (32, 32) float64 range (0.003642904351227908, 0.22973187462246958)
```

### Turn 2 — `execute_in_remote_docker` (`47917 ms` LLM)
- **Rationale**: Compare all three outputs against public truth and compare the robust fit with ordinary ridge. A field mismatch at true coefficients would point to assembly; a source-only miss or large ridge gap would point to the receiver fit.
- **Alternatives Weighed**: `Fit real and imaginary channels with separate robust weights` (rejected: The corruption is receiver-level; one complex residual norm makes the fit invariant to the choice of real and imaginary axes.)

```bash
cd /holdout_task && cat > helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _face_laplacian(n):
    # Each interior face contributes opposite fluxes to its two cells.
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((off, degree, off), (-1, 0, 1), shape=(n, n))


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    x_wall = np.zeros(nx)
    y_wall = np.zeros(ny)
    x_wall[0] += hy
    x_wall[-1] += hy
    y_wall[0] += hx
    y_wall[-1] += hx
    wall_length = x_wall[:, None] + y_wall[None, :]
    diagonal = -area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall_length
    operator = ((hy / hx) * kron(_face_laplacian(nx), eye(ny))
                + (hx / hy) * kron(eye(nx), _face_laplacian(ny))
                + diags(diagonal.ravel()))
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator.tocsc()).solve(rhs.astype(np.complex128))


def _complex_obs(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(1j * np.asarray(telemetry['wrapped_phase_obs']))


def _pilot_phase(pilot):
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_obs(pilot))))
    return np.pi if phase <= -np.pi else phase


def _fit_sources(response, observed, alpha):
    design = np.vstack((response.real, response.imag))
    target = np.concatenate((observed.real, observed.imag))
    penalty = np.sqrt(alpha) * np.eye(response.shape[1])
    rhs = np.r_[target, np.zeros(response.shape[1])]

    def fit(weights):
        root = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((design * root[:, None], penalty)),
                               np.r_[rhs[:target.size] * root, rhs[target.size:]],
                               rcond=None)[0]

    coef = fit(np.ones(observed.size))
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(observed)))
    # A receiver's real and imaginary residuals have one shared weight.
    # Huber first establishes a stable fit; bisquare then rejects impulses.
    for cutoff, redescend in ((1.8, False), (4.685, True)):
        for _ in range(35):
            magnitude = np.abs(response @ coef - observed)
            scale = max(np.median(magnitude) / np.sqrt(2 * np.log(2)), floor)
            radius = magnitude / (cutoff * scale)
            if redescend:
                weights = np.maximum(1 - radius**2, 0)**2
            else:
                weights = 1 / np.maximum(radius, 1)
            updated = fit(weights)
            if np.linalg.norm(updated - coef) <= 1e-9 * (1 + np.linalg.norm(updated)):
                coef = updated
                break
            coef = updated
    return coef


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source coefficients, oscillator phase, and pressure magnitude."""
    shape = np.shape(problem['n_sq'])
    survey = problem['survey']
    phase = _pilot_phase(problem['pilot_calibration'])
    responses = _responses(problem)
    ij = np.asarray(survey['sensor_ij'])
    indices = np.ravel_multi_index((ij[:, 0], ij[:, 1]), shape)
    observed = _complex_obs(survey) * np.exp(-1j * phase)
    coefficients = _fit_sources(responses[indices], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coefficients,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coefficients).reshape(shape)}
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion','/holdout_task/helmholtz_phase_inversion.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_helmholtz_cases.pkl','rb') as f: cases=pickle.load(f)
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
for n, case in enumerate(cases):
 p, g = case['problem'],case['gt']; out=m.invert_helmholtz_sources_and_phase(p)
 R=m._responses(p); s=p['survey']; A=R[np.ravel_multi_index(np.asarray(s['sensor_ij']).T,p['n_sq'].shape)]; z=m._complex_obs(s)*np.exp(-1j*out['global_phase_rad']); alpha=s['tikhonov_alpha'];
 b=np.linalg.lstsq(np.vstack([A.real,A.imag,np.sqrt(alpha)*np.eye(A.shape[1])]),np.r_[z.real,z.imag,np.zeros(A.shape[1])],rcond=None)[0]
 truth_res=np.sort(np.abs(A@g['source_amplitudes']-z));
 print(n,'errors source/phase/field',rel(out['source_amplitudes'],g['source_amplitudes']),abs(np.angle(np.exp(1j*(out['global_phase_rad']-g['global_phase_rad'])))),rel(out['predicted_field_abs'],g['predicted_field_abs']),'ridge source/field',rel(b,g['source_amplitudes']),rel(abs(R@b).reshape(p['n_sq'].shape),g['predicted_field_abs']),'truth residual quantiles',np.quantile(truth_res,[.5,.9,.92,.94,.98,1]),'design cond',np.linalg.cond(np.vstack([A.real,A.imag])))
PY
```
- **Container Exit Code**: `0` (`718 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 errors source/phase/field 0.0030435559166623843 0.0006227225657142733 0.002167665975161924 ridge source/field 0.17182283124053396 0.18377960814573083 truth residual quantiles [0.0020254  0.00374545 0.00407118 0.36168832 0.50697455 0.74069889] design cond 2.9618123405365107
1 errors source/phase/field 0.005838141079609052 0.00026689595102702407 0.0037066542820577493 ridge source/field 0.2726702340356176 0.160987316325593 truth residual quantiles [0.00181359 0.00427394 0.00484774 0.37427004 0.64324622 0.6860372 ] design cond 3.11360940557194

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9dccdf48fe686847 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9dccdf48 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`45002 ms` LLM)
- **Rationale**: Use a clean-process jackknife to detect dependence on any single public receiver or pilot. Independently assemble the rectangular stencil to distinguish assembly errors from inverse-fit failures, and concentrate impulses at high-leverage receivers to test the robust fit before submission.
- **Alternatives Weighed**: `Use ordinary ridge regression` (rejected: Its public source error reaches 27.27%, above the 3% limit.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def errors(out, gt):
    assert set(out) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    a, phi, field = (out[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert a.shape == gt['source_amplitudes'].shape and field.shape == gt['predicted_field_abs'].shape
    assert np.issubdtype(a.dtype, np.floating) and np.issubdtype(field.dtype, np.floating)
    assert isinstance(phi, float) and -np.pi < phi <= np.pi and np.all(field >= 0)
    rel = lambda x, y: np.linalg.norm(x-y)/np.linalg.norm(y)
    return [rel(a, gt['source_amplitudes']), abs(np.angle(np.exp(1j*(phi-gt['global_phase_rad'])))), rel(field, gt['predicted_field_abs'])]

public, jackknife = [], []
for case in cases:
    p, gt = case['problem'], case['gt']
    public.append(errors(m.invert_helmholtz_sources_and_phase(p), gt))
    for group, cols in [('survey', ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs')), ('pilot_calibration', ('known_field_re', 'known_field_im', 'amplitude_obs', 'wrapped_phase_obs'))]:
        for i in range(len(p[group]['amplitude_obs'])):
            reduced = dict(p[group])
            for col in cols:
                reduced[col] = np.delete(reduced[col], i, axis=0)
            jackknife.append(errors(m.invert_helmholtz_sources_and_phase(dict(p, **{group: reduced})), gt))

# An independently assembled face stencil on a rectangular physical domain
# detects axis, wall-length, area, corner, and radiation-sign mistakes.
p = dict(cases[0]['problem'], Lx=1.23, Ly=0.82, k0=13.2, beta_imp=0.53)
nx, ny = p['n_sq'].shape
hx, hy = p['Lx']/nx, p['Ly']/ny
k = p['k0']
rows, cols, vals = [], [], []
for i in range(nx):
    for j in range(ny):
        s = i*ny+j
        d = -hx*hy*k*k*p['n_sq'][i,j] + 0j
        for di, dj, conductance, length in [(1,0,hy/hx,hy),(-1,0,hy/hx,hy),(0,1,hx/hy,hx),(0,-1,hx/hy,hx)]:
            ii, jj = i+di, j+dj
            if 0 <= ii < nx and 0 <= jj < ny:
                rows.append(s); cols.append(ii*ny+jj); vals.append(-conductance)
                d += conductance
            else:
                d -= 1j*k*p['beta_imp']*length
        rows.append(s); cols.append(s); vals.append(d)
H = coo_matrix((vals,(rows,cols)), shape=(nx*ny,nx*ny)).tocsc()
rhs = hx*hy*p['basis'].reshape(6,-1).T
forward_error = np.linalg.norm(H @ m._responses(p) - rhs)/np.linalg.norm(rhs)
assert forward_error < 1e-12, forward_error
R = spsolve(H, rhs)

# Add receiver-level impulses to both random and high-leverage receivers;
# place the oscillator across either side of the wrap seam.
stress = []
cond = []
for seed in range(40):
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=6)
    field = R @ coef
    phase = (-1 if seed % 2 else 1)*(np.pi-1e-5)
    indices = rng.choice(nx*ny, 140, replace=False)
    leverage = np.linalg.norm(R[indices], axis=1)
    bad = np.argsort(leverage)[-10:] if seed % 2 else rng.choice(len(indices), 10, replace=False)
    z = field[indices]*np.exp(1j*phase)
    z += 0.0015*(rng.normal(size=z.size)+1j*rng.normal(size=z.size))
    z[bad] += 0.8*np.exp(1j*rng.uniform(-np.pi,np.pi,len(bad)))
    known = rng.normal(size=12)+1j*rng.normal(size=12)
    pilot = known*np.exp(1j*phase)
    problem = dict(p, survey=dict(sensor_ij=np.column_stack(np.unravel_index(indices,(nx,ny))), amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z), tikhonov_alpha=1e-6), pilot_calibration=dict(known_field_re=known.real, known_field_im=known.imag, amplitude_obs=np.abs(pilot), wrapped_phase_obs=np.angle(pilot)))
    gt = dict(source_amplitudes=coef, global_phase_rad=phase, predicted_field_abs=np.abs(field).reshape(nx,ny))
    stress.append(errors(m.invert_helmholtz_sources_and_phase(problem), gt))
    usable = np.delete(R[indices], bad, axis=0)
    cond.append(np.linalg.cond(np.vstack((usable.real, usable.imag))))

worst = np.max(public+jackknife+stress,axis=0)
print('public max [source, phase rad, field]', np.max(public,axis=0))
print('public leave-one-receiver-or-pilot-out max', np.max(jackknife,axis=0), 'folds', len(jackknife))
print('rectangular stencil relative residual', forward_error)
print('synthetic random impulses max', np.max(stress[::2],axis=0))
print('synthetic high-leverage impulses max', np.max(stress[1::2],axis=0))
print('worst clean receiver design condition number', max(cond))
assert np.all(worst <= [0.03,0.02,0.03]), worst
PY
```
- **Container Exit Code**: `0` (`3088 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public max [source, phase rad, field] [0.00583814 0.00062272 0.00370665]
public leave-one-receiver-or-pilot-out max [0.00641197 0.00074135 0.00416389] folds 304
rectangular stencil relative residual 1.1885712284260123e-14
synthetic random impulses max [7.03483422e-03 4.44089210e-16 6.31626001e-03]
synthetic high-leverage impulses max [0.0102988 0.        0.0069505]
worst clean receiver design condition number 2.7225244149106587

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9dccdf48fe686847 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9dccdf48 (directory).
```

### Turn 4 — `submit_completed_task` (`13970 ms` LLM)
- **Rationale**: The final artifact passed isolated Python verification. The independent rectangular stencil gave a 1.19e-14 relative forward residual, and all three outputs met their tolerances across the public, leave-one-out, and impulse stress tests.
- **Alternatives Weighed**: `Ordinary ridge regression` (rejected: Public source error reached 27.27%, exceeding the 3% tolerance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3763'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across both public cases, 304 leave-one-out folds, and 40 synthetic stress cases, worst errors were 1.030% for source amplitudes, 0.000742 rad for phase, and 0.696% for field magnitude. All were within tolerance.`

### Turn 5 — `execute_in_remote_docker` (`45218 ms` LLM)
- **Rationale**: Check whether the public reference fields agree with the exact operator and whether pilot differences support the same phase rather than a sign-flipped solution. Then withhold entire receiver quadrants and the highest-leverage quartile to detect geometry dependence that single-receiver folds could miss; compare rational weights on the same folds without modifying the verified artifact.
- **Alternatives Weighed**: `Replace bisquare weights with rational Cauchy weights` (rejected: Do not switch without evidence that the rational weights improve a limiting error on real receiver holdouts.); `Apply batch centering or relay/drift tests` (rejected: The inputs contain no batch labels, classification targets, relay modes, or time-evolution operator. Spatial and leverage slices are relevant to this inverse problem.)

```bash
cd /holdout_task && cat > /tmp/helmholtz_margin_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)


def rel(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(y)


def cauchy(A, z, alpha, coef):
    X = np.vstack((A.real, A.imag))
    y = np.r_[z.real, z.imag]
    penalty = np.sqrt(alpha) * np.eye(A.shape[1])
    for _ in range(35):
        r = np.abs(A @ coef-z)
        scale = max(np.median(r)/np.sqrt(2*np.log(2)), np.finfo(float).eps)
        root = np.sqrt(np.tile(1/(1+(r/(2.385*scale))**2), 2))
        new = np.linalg.lstsq(np.vstack((X*root[:, None], penalty)), np.r_[y*root, np.zeros(A.shape[1])], rcond=None)[0]
        if np.linalg.norm(new-coef) <= 1e-9*(1+np.linalg.norm(new)):
            return new
        coef = new
    return coef


worst = []
for i, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    shape = p['n_sq'].shape
    R = m._responses(p)
    baseline = m.invert_helmholtz_sources_and_phase(p)
    phi = baseline['global_phase_rad']
    phase_error = abs(np.angle(np.exp(1j*(phi-gt['global_phase_rad']))))
    assert phase_error <= .02 and -np.pi < phi <= np.pi
    forward_truth = rel(np.abs(R @ gt['source_amplitudes']).reshape(shape), gt['predicted_field_abs'])
    assert forward_truth < 1e-12, forward_truth
    k = pilot['known_field_re'] + 1j*pilot['known_field_im']
    w = m._complex_obs(pilot)
    j, l = np.triu_indices(len(k), 1)
    raw_resid = np.abs(w-np.exp(1j*phi)*k)
    pair_resid = np.abs((w[j]-w[l])-np.exp(1j*phi)*(k[j]-k[l]))
    print(f'case {i}: phase error={phase_error:.7g} rad, truth forward error={forward_truth:.3g}, pilot direct RMS={np.sqrt(np.mean(raw_resid**2)):.4g}, pair-difference RMS={np.sqrt(np.mean(pair_resid**2)):.4g}, opposite-sign pilot RMS={np.sqrt(np.mean(np.abs(w+np.exp(1j*phi)*k)**2)):.4g}')
    ij = survey['sensor_ij']
    A = R[np.ravel_multi_index(ij.T, shape)]
    z = m._complex_obs(survey)*np.exp(-1j*phi)
    quadrant = 2*(ij[:,0] >= shape[0]/2) + (ij[:,1] >= shape[1]/2)
    high = np.argsort(np.linalg.norm(A, axis=1))[-len(A)//4:]
    high_mask = np.isin(np.arange(len(A)), high)
    splits = [(f'quadrant {q}', quadrant==q) for q in range(4)] + [('highest leverage quartile', high_mask)]
    public = [rel(baseline['source_amplitudes'], gt['source_amplitudes']), rel(baseline['predicted_field_abs'], gt['predicted_field_abs'])]
    worst.append(public)
    all_scores = []
    for name, test in splits:
        train = ~test
        reduced = dict(survey, **{c: survey[c][train] for c in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        out = m.invert_helmholtz_sources_and_phase(dict(p, survey=reduced))
        a = out['source_amplitudes']
        b = cauchy(A[train], z[train], survey['tikhonov_alpha'], a)
        scores = []
        for coef in (a, b):
            scores.append([rel(coef, gt['source_amplitudes']), rel(np.abs(R@coef).reshape(shape), gt['predicted_field_abs']), np.median(np.abs(A[test]@coef-z[test]))])
        all_scores.append(scores)
        worst.append(scores[0][:2])
        print(f'  hold out {name}: n={test.sum()}, current vs rational [source rel, field rel, held-out median complex residual] = {np.array(scores).round(6).tolist()}')
    print('  worst slab [current, rational]:', np.max(all_scores, axis=0))
print('worst current public and blocked holdouts [source, field]:', np.max(worst, axis=0))
assert np.all(np.max(worst, axis=0) <= .03)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_margin_audit.py
```
- **Container Exit Code**: `0` (`616 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
opposite-sign pilot RMS=2.13
  hold out quadrant 0: n=36, current vs rational [source rel, field rel, held-out median complex residual] = [[0.004075, 0.003484, 0.002338], [0.004071, 0.00352, 0.002335]]
  hold out quadrant 1: n=36, current vs rational [source rel, field rel, held-out median complex residual] = [[0.002264, 0.001794, 0.001886], [0.002222, 0.001809, 0.001884]]
  hold out quadrant 2: n=31, current vs rational [source rel, field rel, held-out median complex residual] = [[0.005003, 0.002846, 0.001861], [0.004917, 0.002844, 0.001862]]
  hold out quadrant 3: n=37, current vs rational [source rel, field rel, held-out median complex residual] = [[0.003239, 0.002681, 0.001997], [0.003226, 0.002696, 0.001995]]
  hold out highest leverage quartile: n=35, current vs rational [source rel, field rel, held-out median complex residual] = [[0.003672, 0.002629, 0.002107], [0.003644, 0.002647, 0.002109]]
  worst slab [current, rational]: [[0.00500309 0.00348353 0.00233773]
 [0.00491725 0.0035203  0.00233506]]
case 1: phase error=0.000266896 rad, truth forward error=8.89e-16, pilot direct RMS=0.002511, pair-difference RMS=0.003655, opposite-sign pilot RMS=1.755
  hold out quadrant 0: n=42, current vs rational [source rel, field rel, held-out median complex residual] = [[0.005921, 0.003907, 0.00193], [0.005904, 0.003946, 0.001936]]
  hold out quadrant 1: n=36, current vs rational [source rel, field rel, held-out median complex residual] = [[0.007276, 0.004505, 0.001815], [0.006919, 0.004368, 0.001796]]
  hold out quadrant 2: n=31, current vs rational [source rel, field rel, held-out median complex residual] = [[0.005531, 0.003938, 0.001574], [0.005404, 0.003894, 0.001571]]
  hold out quadrant 3: n=31, current vs rational [source rel, field rel, held-out median complex residual] = [[0.007439, 0.004357, 0.001792], [0.00724, 0.004273, 0.001806]]
  hold out highest leverage quartile: n=35, current vs rational [source rel, field rel, held-out median complex residual] = [[0.005695, 0.003513, 0.001495], [0.00521, 0.003315, 0.0015]]
  worst slab [current, rational]: [[0.00743882 0.00450508 0.00193009]
 [0.00724015 0.00436838 0.00193649]]
worst current public and blocked holdouts [source, field]: [0.00743882 0.00450508]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9dccdf48fe686847 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9dccdf48 (directory).
```

### Turn 6 — `submit_completed_task` (`14669 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification. The independent rectangular stencil had a relative forward residual of 1.19e-14. Withholding whole receiver quadrants or the highest-leverage quartile left source and field errors below 0.744% and 0.451%, respectively.
- **Alternatives Weighed**: `Replace bisquare weights with rational Cauchy weights` (rejected: Cauchy weights modestly improved coefficient and field errors on blocked public holdouts but did not improve the worst held-out median receiver residual or establish an advantage on the limiting synthetic slice.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3763'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver and pilot holdouts, and synthetic impulse tests, the worst errors were 1.030% for source amplitudes, 0.000742 rad for phase, and 0.696% for field magnitude. All met the specified tolerances.`

### Turn 7 — `execute_in_remote_docker` (`37328 ms` LLM)
- **Rationale**: The earlier high-leverage stress used synthetic impulses on a different geometry. This check tests whether actual measured impulse sizes and phases can push the current fit over tolerance when they hit receivers that most strongly influence individual coefficients. A failure would warrant comparing a trimmed initialization with the current Huber initialization; a pass needs no solver change.
- **Alternatives Weighed**: `Repeat the pilot, algebraic, or batch audit` (rejected: Pilot sign and pair differences, stencil correctness, and rational weights were already checked. There are no batch labels, classification outcomes, or relay states in these data.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

# Keep the measured complex errors and actual receiver geometries. Move only
# the clearly separated impulses onto the most informative receivers.
worst = np.zeros(3)
for j, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    s = p['survey']
    shape = p['n_sq'].shape
    R = m._responses(p)
    A = R[np.ravel_multi_index(s['sensor_ij'].T, shape)]
    phi = g['global_phase_rad']
    signal = (A @ g['source_amplitudes']) * np.exp(1j*phi)
    err = m._complex_obs(s) - signal
    size = np.abs(err)
    bad = np.flatnonzero(size > 10*np.median(size))
    assert 0 < len(bad) < len(err)/5
    clean = np.setdiff1d(np.arange(len(err)), bad)
    X = np.r_[A[clean].real, A[clean].imag]
    covariance = np.linalg.inv(X.T @ X)
    directions = list(np.eye(A.shape[1])) + list(np.linalg.eigh(covariance)[1].T)
    rankings = [np.sum((A @ covariance)*A.conj(), axis=1).real]
    rankings.extend(np.abs(A @ (covariance @ d)) for d in directions)
    local = np.zeros(3)
    rel = lambda a, b: np.linalg.norm(a-b)/np.linalg.norm(b)
    for rank in rankings:
        dest = np.argsort(rank)[-len(bad):]
        for order in (bad[np.argsort(size[bad])], bad[np.argsort(size[bad])[::-1]]):
            perm = np.empty(len(err), dtype=int)
            perm[dest] = order
            perm[np.setdiff1d(np.arange(len(err)), dest)] = clean
            z = signal + err[perm]
            out = m.invert_helmholtz_sources_and_phase(dict(p, survey=dict(s, amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))))
            local = np.maximum(local, [rel(out['source_amplitudes'], g['source_amplitudes']), abs(np.angle(np.exp(1j*(out['global_phase_rad']-phi)))), rel(out['predicted_field_abs'], g['predicted_field_abs'])])
    print(f'public geometry {j}: {len(bad)} measured impulses, {2*len(rankings)} placements, max [source, phase rad, field] {local}')
    worst = np.maximum(worst, local)
assert np.all(worst <= [0.03, 0.02, 0.03]), worst
PY
```
- **Container Exit Code**: `0` (`874 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public geometry 0: 10 measured impulses, 26 placements, max [source, phase rad, field] [0.00410136 0.00062272 0.00289965]
public geometry 1: 10 measured impulses, 26 placements, max [source, phase rad, field] [0.00656887 0.0002669  0.00311281]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9dccdf48fe686847 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9dccdf48 (directory).
```

### Turn 8 — `submit_completed_task` (`16542 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, an independent rectangular-stencil check, public receiver and pilot holdouts, and both synthetic and measured-impulse stress tests. No tested slice approached a specification limit.
- **Alternatives Weighed**: `Add a trimmed initialization` (rejected: Across 52 targeted placements of measured impulses, the current fit stayed below 0.657% source error; the test did not reveal an initialization failure.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver and pilot holdouts, and impulse stress tests, worst errors were 1.030% for source amplitudes, 0.000742 rad for phase, and 0.696% for field magnitude. All met the specified tolerances.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
