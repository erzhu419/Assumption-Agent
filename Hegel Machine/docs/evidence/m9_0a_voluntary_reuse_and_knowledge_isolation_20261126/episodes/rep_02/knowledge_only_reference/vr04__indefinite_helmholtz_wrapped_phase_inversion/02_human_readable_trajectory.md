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
| **Turns Used / Max Turns** | `8 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `211,986` (`204,669` / `40,704` / `7,317`) |
| **Total Episode Wall Time (s)** | `238.05s` |

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

### Turn 1 — `execute_in_remote_docker` (`15866 ms` LLM)
- **Rationale**: Inspect public data keys, shapes, and available truth before assembling the operator; schema or label differences determine how to implement and validate the solver. Consult the robust weighting contract before applying it to grouped complex residuals.
- **Alternatives Weighed**: `Reuse the SPD Darcy solver` (rejected: Its positive-definite operator contract does not apply to the complex indefinite Helmholtz stencil.)

```bash
cd /holdout_task && ls -lh && python3 - <<'PY'
import pickle, numpy as np, scipy
with open('public_helmholtz_cases.pkl','rb') as f: x=pickle.load(f)
def describe(x, depth=0):
 if isinstance(x,dict) and depth<5:
  for k,v in list(x.items())[:12]:
   print('  '*depth,str(k),end=': '); describe(v,depth+1)
 elif isinstance(x,(list,tuple)) and depth<5:
  print(type(x).__name__,len(x));
  if x: describe(x[0],depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,'sample',x.ravel()[:4])
 else: print(repr(x)[:160])
describe(x)
print('numpy/scipy',np.__version__,scipy.__version__)
PY
python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.robust_residual_weights
```
- **Container Exit Code**: `0` (`648 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 00:46 public_helmholtz_cases.pkl
list 2
   problem:      Lx: 1.0
     Ly: 1.0
     k0: 11.5
     beta_imp: 0.85
     n_sq: (32, 32) float64 sample [0.80163524 0.80204006 0.80263744 0.80348888]
     basis: (6, 32, 32) float64 sample [0.000461   0.00215903 0.00802474 0.02367123]
     pilot_calibration:        known_field_re: (12,) float64 sample [0.70152797 0.73476006 1.90297056 1.08085754]
       known_field_im: (12,) float64 sample [-0.36152527 -0.65790536 -0.37345604 -0.08335344]
       amplitude_obs: (12,) float64 sample [0.78821059 0.98674527 1.93760065 1.08525412]
       wrapped_phase_obs: (12,) float64 sample [-1.28850398 -1.54397385 -1.0075838  -0.89001087]
     survey:        sensor_ij: (140, 2) int64 sample [ 3 29 29 20]
       amplitude_obs: (140,) float64 sample [0.04516173 0.06261364 0.04173044 0.04717036]
       wrapped_phase_obs: (140,) float64 sample [-2.05372953 -0.20593495 -1.60097692  0.96523401]
       tikhonov_alpha: 1e-06
   gt:      source_amplitudes: (6,) float64 sample [ 1.35       -0.57020875 -0.72756369 -0.18032955]
     global_phase_rad: -0.8141122105046681
     predicted_field_abs: (32, 32) float64 sample [0.01746595 0.01663818 0.01827211 0.02375379]
numpy/scipy 1.26.4 1.13.0
{
  "ok": true,
  "module_stem": "autogenous_prim_robust_affine_projection",
  "compiled_code_sha256": "abc35d3defaecd9adc5e1091c1b53ab480677bd3aac9cbe7e691e6bd41e49062",
  "mathematical_preconditions": [
    "scale > 0 finite scalar or broadcastable array; loss in {'huber', 'cauchy'}.",
    "weights >= 0 when provided; sensitivity must have nonzero weighted norm above gain_floor for identifiable slope."
  ],
  "function_level_applicability": {},
  "functions": [
    {
      "function": "robust_residual_weights",
      "signature": "robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray",
      "doc_summary": "For residuals of shape S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_robust_affine_projection.reference.txt"
}
```

### Turn 2 — `execute_in_remote_docker` (`40275 ms` LLM)
- **Rationale**: Check source, field and phase errors independently. Pilot offsets and survey residuals by signal strength will show whether a missed tolerance calls for a calibration change or a different receiver-noise model.
- **Alternatives Weighed**: `Unwrap survey angles before fitting` (rejected: Complex phasors preserve the telemetry without branch choices and allow the real source coefficients to enter linearly.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion with a shared pilot phase reference."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    impedance = -1j * k * problem['beta_imp']

    def axis_operator(size, coupling, wall):
        graph_diag = np.full(size, 2 * coupling, dtype=complex)
        graph_diag[[0, -1]] += wall - coupling
        return sparse.diags((-coupling, graph_diag, -coupling), (-1, 0, 1),
                            shape=(size, size), format='csc')

    # Each interior face enters both cell balances, and each corner has two
    # separate impedance faces. The real stiffness is not positive definite.
    operator = (sparse.kron(axis_operator(nx, hy / hx, hy * impedance), sparse.eye(ny), format='csc')
                + sparse.kron(sparse.eye(nx), axis_operator(ny, hx / hy, hx * impedance), format='csc')
                - sparse.diags(hx * hy * k**2 * n_sq.ravel(), format='csc'))
    basis = np.asarray(problem['basis'])
    rhs = hx * hy * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator).solve(rhs.astype(np.complex128)).reshape(nx, ny, -1)


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(1j * np.asarray(telemetry['wrapped_phase_obs']))


def _robust_sources(design, observed, alpha):
    # Treat each complex observation as one sensor: its two quadratures
    # receive the same robust weight and the source coefficients stay real.
    gram = np.einsum('ni,nj->nij', design.conj(), design).real
    rhs = (design.conj() * observed[:, None]).real
    penalty = alpha * np.eye(design.shape[1])
    a = np.linalg.solve(gram.sum(axis=0) + penalty, rhs.sum(axis=0))
    floor = np.finfo(float).eps * np.max(np.abs(observed))
    for _ in range(40):
        residual = np.abs(design @ a - observed)
        cutoff = max(2.5 * np.median(residual), floor)
        if cutoff == 0:
            break
        weights = 1 / (1 + (residual / cutoff)**2)
        new = np.linalg.solve(np.einsum('n,nij->ij', weights, gram) + penalty, weights @ rhs)
        if np.linalg.norm(new - a) <= 1e-10 * (1 + np.linalg.norm(new)):
            return new
        a = new
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source amplitudes, oscillator phase and pressure modulus."""
    responses = _responses(problem)
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    a = _robust_sources(responses[i, j], _phasors(survey) * np.exp(-1j * phase), survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ a)}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import numpy as np, pickle
import helmholtz_phase_inversion as m
for case in pickle.load(open('public_helmholtz_cases.pkl','rb')):
 p,g=case['problem'],case['gt']; out=m.invert_helmholtz_sources_and_phase(p)
 print('errors', *(np.linalg.norm(out[k]-g[k])/np.linalg.norm(g[k]) for k in ('source_amplitudes','predicted_field_abs')), abs(np.angle(np.exp(1j*(out['global_phase_rad']-g['global_phase_rad'])))))
 responses=m._responses(p); i,j=p['survey']['sensor_ij'].T; X=responses[i,j]; pred=X@g['source_amplitudes']; z=m._phasors(p['survey'])*np.exp(-1j*g['global_phase_rad']); err=z-pred
 pilot=p['pilot_calibration']; ref=pilot['known_field_re']+1j*pilot['known_field_im']; ratio=m._phasors(pilot)/ref*np.exp(-1j*g['global_phase_rad']); print('pilot phase offsets',np.round(np.angle(ratio),6))
 print('survey noisefloor/tails',np.quantile(abs(err),[0,.25,.5,.75,.9,.93,.95,.99,1]),'design singular values',np.linalg.svd(np.r_[X.real,X.imag],compute_uv=False)); print('signal vs error median bins',[(float(np.median(abs(pred)[b])),float(np.median(abs(err)[b]))) for b in np.array_split(np.argsort(abs(pred)),4)])
PY
```
- **Container Exit Code**: `0` (`490 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
errors 0.003127922921622004 0.00218582033266458 0.0006227225657142733
pilot phase offsets [ 0.001452  0.000407  0.000315  0.001067  0.0004    0.001182  0.001045
  0.000203  0.021354 -0.005787 -0.007246 -0.003115]
survey noisefloor/tails [6.47881901e-06 1.38830670e-03 2.04397750e-03 2.60904210e-03
 3.74096331e-03 1.00151822e-01 3.72538342e-01 6.29259867e-01
 7.40687115e-01] design singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722]
signal vs error median bins [(0.02436413531049968, 0.002188406087948547), (0.051457197886480455, 0.001856595664229905), (0.0736225078024716, 0.002080000644928901), (0.12143967182519738, 0.0018473520545497291)]
errors 0.0053080477437224445 0.003496833206680231 0.00026689595102702407
pilot phase offsets [ 2.336e-03 -2.798e-03 -4.071e-03 -7.500e-05  1.839e-03 -1.433e-03
 -1.974e-03 -1.661e-03  1.050e-04 -1.184e-03  1.147e-03  2.717e-03]
survey noisefloor/tails [2.09377251e-04 1.16546417e-03 1.80389227e-03 2.52441899e-03
 4.26435661e-03 9.99000576e-02 4.09010034e-01 6.54635093e-01
 6.86029873e-01] design singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252]
signal vs error median bins [(0.05185372192499902, 0.0017665177565040252), (0.07467136438679042, 0.0020113018104835883), (0.09930466318131946, 0.001997802075792147), (0.15257531928875945, 0.0018423434915958085)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76a4deb689c899a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76a4deb6 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`41820 ms` LLM)
- **Rationale**: Recompute face balances to detect stencil errors that a square grid can hide. Hold out each public receiver to detect undue dependence on one channel, and target high-leverage receivers with spikes to test whether the robust fit survives concentrated contamination. Exercise both sides of the phase seam and check each output against its own tolerance.
- **Alternatives Weighed**: `Tune Cauchy weights against the two public cases` (rejected: All public errors are comfortably within tolerance, and residuals show little dependence on signal strength. Stress testing is more informative than tuning to two cases.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
limits = np.array([0.030, 0.020, 0.030])

def errors(out, a, phase, field):
    assert set(out) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    assert out['source_amplitudes'].shape == a.shape and out['source_amplitudes'].dtype.kind == 'f'
    assert isinstance(out['global_phase_rad'], float) and -np.pi < out['global_phase_rad'] <= np.pi
    assert out['predicted_field_abs'].shape == field.shape and out['predicted_field_abs'].dtype.kind == 'f'
    return np.array([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(out['global_phase_rad']-phase)))), np.linalg.norm(out['predicted_field_abs']-field)/np.linalg.norm(field)])

def check_balance(p, u):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    q = hx*hy*np.moveaxis(p['basis'], 0, -1)
    balance = -(hx*hy*k*k)*p['n_sq'][..., None]*u
    fx = hy/hx*(u[1:] - u[:-1]); fy = hx/hy*(u[:, 1:] - u[:, :-1])
    balance[1:] += fx; balance[:-1] -= fx
    balance[:, 1:] += fy; balance[:, :-1] -= fy
    balance[[0, -1]] -= 1j*k*p['beta_imp']*hy*u[[0, -1]]
    balance[:, [0, -1]] -= 1j*k*p['beta_imp']*hx*u[:, [0, -1]]
    return np.linalg.norm(balance-q)/np.linalg.norm(q)

def polar(z):
    return dict(amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))

public, jack, stress, balance_errors = [], [], [], []
rng = np.random.default_rng(4132)
for case in cases:
    p, g = case['problem'], case['gt']
    a, phase, field = (g[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    public.append(errors(m.invert_helmholtz_sources_and_phase(p), a, phase, field))
    survey = p['survey']
    n = len(survey['sensor_ij'])
    for held in range(n):
        keep = np.arange(n) != held
        fold = {**survey, **{k: survey[k][keep] for k in ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs')}}
        jack.append(errors(m.invert_helmholtz_sources_and_phase({**p, 'survey': fold}), a, phase, field))
    for lx, ly, k in [(0.83, 1.17, 10.9), (1.19, 0.87, 12.3)]:
        perturbed = {**p, 'Lx': lx, 'Ly': ly, 'k0': k}
        u = m._responses(perturbed)
        balance_errors.append(check_balance(perturbed, u))
        ij = survey['sensor_ij']
        x = u[ij[:,0], ij[:,1]]
        pilot = p['pilot_calibration']
        ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
        count = round(0.07*n)
        leverage = np.argsort(np.linalg.norm(x, axis=1))[-count:]
        for branch_phase in (np.pi, np.pi - 0.0005, -np.pi + 0.0005):
            for bad in (rng.choice(n, count, replace=False), leverage):
                truth = a if branch_phase > 0 else -a
                z = np.exp(1j*branch_phase)*(x @ truth) + 0.0016*(rng.normal(size=n)+1j*rng.normal(size=n))
                z[bad] += 0.4*np.exp(1j*rng.uniform(-np.pi, np.pi, count))
                cal = {**pilot, **polar(ref*np.exp(1j*branch_phase))}
                data = {**survey, **polar(z)}
                out = m.invert_helmholtz_sources_and_phase({**perturbed, 'pilot_calibration': cal, 'survey': data})
                stress.append(errors(out, truth, branch_phase, np.abs(u @ truth)))
for name, vals in [('public', public), ('receiver jackknife', jack), ('rectangular/phase seam/spike stress', stress)]:
    worst = np.max(vals, axis=0)
    print(name, len(vals), 'worst source / phase rad / field:', worst, flush=True)
    assert np.all(worst <= limits)
print('Worst independently recomputed finite-volume balance:', max(balance_errors))
assert max(balance_errors) < 1e-10
PY
```
- **Container Exit Code**: `0` (`2044 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 2 worst source / phase rad / field: [0.00530805 0.00062272 0.00349683]
receiver jackknife 280 worst source / phase rad / field: [0.00588427 0.00062272 0.00393776]
rectangular/phase seam/spike stress 24 worst source / phase rad / field: [5.41666623e-03 4.44089210e-16 5.05055553e-03]
Worst independently recomputed finite-volume balance: 1.2789020932951477e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76a4deb689c899a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76a4deb6 (directory).
```

### Turn 4 — `submit_completed_task` (`13423 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification, including independent face-balance checks and stress tests for rectangular grids, phase wrapping, and spikes on high-leverage receivers. No measured bottleneck warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3073'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across both public cases, 280 leave-one-receiver-out runs, and 24 stress tests, worst errors were 0.588% for source amplitudes, 0.000623 rad for phase, and 0.505% for field magnitude. All met the requested tolerances.`

### Turn 5 — `execute_in_remote_docker` (`35773 ms` LLM)
- **Rationale**: The stencil already uses exact face differences, and the Cauchy weight is the relevant rational quotient. Compare it with plain ridge on measured data, holding out spatial blocks to detect dependence on neighboring receivers and holding out individual pilots to detect undue calibration leverage. A missed tolerance would identify which stage needs revision.
- **Alternatives Weighed**: `Batch-center receiver data or add drift and relay terms` (rejected: The schema has no batches, drift dynamics, or relay states. The shared oscillator phase is identified from independent pilots; centering survey measurements would change the pressure model.)

```bash
cat > /tmp/helmholtz_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for case_id, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    s = p['survey']; pilot = p['pilot_calibration']
    a0 = truth['source_amplitudes']; phase0 = truth['global_phase_rad']; f0 = truth['predicted_field_abs']
    field = m._responses(p)
    i,j = s['sensor_ij'].T
    x = field[i,j]
    out = m.invert_helmholtz_sources_and_phase(p)
    y = m._phasors(s) * np.exp(-1j*out['global_phase_rad'])
    alpha = s['tikhonov_alpha']
    pred = x @ a0
    def score(a):
        return np.array([np.linalg.norm(a-a0)/np.linalg.norm(a0),np.linalg.norm(np.abs(field@a)-f0)/np.linalg.norm(f0)])
    stacked = np.r_[x.real,x.imag]
    ridge = np.linalg.solve(stacked.T@stacked+alpha*np.eye(x.shape[1]), stacked.T@np.r_[y.real,y.imag])
    print('case',case_id, 'source / field plain ridge:',score(ridge),'grouped Cauchy:',score(out['source_amplitudes']))
    # Spatial block holdouts test whether neighboring channels hide a local failure.
    block = 2*(i >= field.shape[0]//2).astype(int) + (j >= field.shape[1]//2)
    folds=[]; predicted=[]; ridge_predicted=[]
    for held in np.unique(block):
        keep = block != held
        fit = m._robust_sources(x[keep], y[keep], alpha)
        fit_l2 = m._robust_sources(x[keep], y[keep], alpha)
        a_train = np.r_[x[keep].real,x[keep].imag]; y_train = np.r_[y[keep].real,y[keep].imag]
        fit_l2 = np.linalg.solve(a_train.T@a_train+alpha*np.eye(x.shape[1]),a_train.T@y_train)
        folds.append(score(fit))
        predicted.append(np.linalg.norm(x[~keep]@fit-pred[~keep])/np.linalg.norm(pred[~keep]))
        ridge_predicted.append(np.linalg.norm(x[~keep]@fit_l2-pred[~keep])/np.linalg.norm(pred[~keep]))
    known = pilot['known_field_re']+1j*pilot['known_field_im']
    pilot_z = m._phasors(pilot)
    pilot_errors=[]; pilot_source_field=[]
    for held in range(len(known)):
        keep=np.arange(len(known))!=held
        ph=np.angle(np.vdot(known[keep],pilot_z[keep]))
        pilot_errors.append(abs(np.angle(np.exp(1j*(ph-phase0)))))
        pilot_source_field.append(score(m._robust_sources(x,m._phasors(s)*np.exp(-1j*ph),alpha)))
    print('spatial blocks counts:',np.bincount(block),'worst source / field:',np.max(folds,axis=0),'worst held-out complex prediction Cauchy / ridge:',max(predicted),max(ridge_predicted))
    print('pilot jackknife worst phase / source / field:',max(pilot_errors),*np.max(pilot_source_field,axis=0))
    assert np.all(np.max(folds,axis=0)<=.03) and np.all(np.max(pilot_source_field,axis=0)<=.03) and max(pilot_errors)<=.02
PY
sed -i '/fit_l2 = m\._robust_sources/d' /tmp/helmholtz_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_audit.py
```
- **Container Exit Code**: `0` (`520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 source / field plain ridge: [0.17182283 0.18377961] grouped Cauchy: [0.00312792 0.00218582]
spatial blocks counts: [36 36 31 37] worst source / field: [0.00509218 0.00347621] worst held-out complex prediction Cauchy / ridge: 0.008597861688079356 0.6346023258276098
pilot jackknife worst phase / source / field: 0.0007413489539971341 0.003170407331204782 0.0022020479074387
case 1 source / field plain ridge: [0.27267023 0.16098732] grouped Cauchy: [0.00530805 0.00349683]
spatial blocks counts: [42 36 31 31] worst source / field: [0.00700268 0.00430111] worst held-out complex prediction Cauchy / ridge: 0.00923514278341055 0.5113440005877524
pilot jackknife worst phase / source / field: 0.0005897335318925111 0.005341814836158671 0.0035059238918722082

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76a4deb689c899a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76a4deb6 (directory).
```

### Turn 6 — `submit_completed_task` (`16774 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process verification, independent face-balance checks, phase-wrap and concentrated-spike stresses, and public receiver, spatial-block, and pilot holdouts. The expanded audit found no remaining tolerance bottleneck.
- **Alternatives Weighed**: `Replace grouped Cauchy regression with plain ridge` (rejected: Plain ridge missed the source and field tolerances on both public cases.)
- **ATMS NOGOOD DECLARED**: Refuted `Uniform quadratic receiver weights suffice for the specified survey telemetry.` | Witness: `Plain ridge had 17.18% and 27.27% source error on the two public cases, versus 0.313% and 0.531% with grouped Cauchy weights.` | Blanket Diagnosis: `Uniform quadratic weights give large complex receiver residuals undue influence even after independent pilot alignment.` | Sublated Pivot: `Retain the verified finite-volume stencil and pilot calibration, and assign both quadratures of each receiver one shared Cauchy weight.`
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3073'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across public cases, receiver and pilot holdouts, and stress tests, the worst errors were 0.700% for source amplitudes, 0.000742 rad for phase, and 0.506% for field magnitude. All met the requested tolerances.`

### Turn 7 — `execute_in_remote_docker` (`32971 ms` LLM)
- **Rationale**: The previous spike stress used generated impulses. This check moves the largest measured residuals onto receivers with the most source influence, preserves their magnitudes, and carries the measured pilot errors across both sides of the phase seam. A source or field miss would motivate a stronger robust initializer; a phase miss would point to pilot calibration.
- **Alternatives Weighed**: `Add batch centering or alter the forward operator` (rejected: There are no batches in the input. The independent balance check already verifies the face differences, and clean pilot references identify the shared phase.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/helmholtz_phase_inversion.py'
spec = importlib.util.spec_from_file_location('inversion', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def polar(z):
    return {'amplitude_obs': np.abs(z), 'wrapped_phase_obs': np.angle(z)}

worst = np.zeros(3)
for idx, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    phase = gt['global_phase_rad']
    a = gt['source_amplitudes']
    i, j = survey['sensor_ij'].T
    x = m._responses(p)[i, j]
    clean = x @ a
    residual = m._phasors(survey) * np.exp(-1j * phase) - clean
    count = round(0.07 * len(residual))
    # Rank influence before choosing where to place the measured impulses.
    stacked = np.r_[x.real, x.imag]
    normal = stacked.T @ stacked + survey['tikhonov_alpha'] * np.eye(x.shape[1])
    sensitivity = np.linalg.solve(normal, x.T)
    influence = np.linalg.norm(sensitivity, axis=0)
    targets = np.argsort(influence)[-count:]
    tails = np.argsort(np.abs(residual))[-count:]
    displaced = np.setdiff1d(targets, tails)
    vacated = np.setdiff1d(tails, targets)
    ranked = residual.copy()
    ranked[vacated] = residual[displaced]
    ranked[targets] = residual[tails]
    case_worst = np.zeros(3)
    for sign in (1, -1):
        trial_phase = sign * (np.pi - 0.0003)
        rotate = np.exp(1j * trial_phase)
        cal = {**pilot, **polar(m._phasors(pilot) * rotate * np.exp(-1j * phase))}
        for noise in (residual, ranked, ranked.conj()):
            data = {**survey, **polar((clean + noise) * rotate)}
            out = m.invert_helmholtz_sources_and_phase({**p, 'survey': data, 'pilot_calibration': cal})
            err = np.array([np.linalg.norm(out['source_amplitudes'] - a) / np.linalg.norm(a), abs(np.angle(np.exp(1j * (out['global_phase_rad'] - trial_phase)))), np.linalg.norm(out['predicted_field_abs'] - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs'])])
            case_worst = np.maximum(case_worst, err)
    print('case', idx, 'worst source / phase rad / field:', case_worst)
    worst = np.maximum(worst, case_worst)
print('worst across 12 measured-residual stress runs:', worst)
assert np.all(worst <= [0.03, 0.02, 0.03])
PY
```
- **Container Exit Code**: `0` (`610 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 worst source / phase rad / field: [0.00444942 0.00062272 0.00344773]
case 1 worst source / phase rad / field: [0.00530805 0.0002669  0.00349683]
worst across 12 measured-residual stress runs: [0.00530805 0.00062272 0.00349683]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76a4deb689c899a6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76a4deb6 (directory).
```

### Turn 8 — `submit_completed_task` (`13878 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification and independent finite-volume balance checks. Grouped Cauchy weighting addressed the observed plain-ridge failure and stayed within tolerance when the largest measured residuals were moved to high-influence receivers and the pilot phase crossed the wrap boundary.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, holdouts, and stress tests, worst errors were 0.701% for source amplitudes, 0.000742 rad for global phase, and 0.506% for field magnitude. All met the requested tolerances.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/helmholtz_phase_inversion.py": "FILE:3073"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_source_rel_l2": 0.003931,
    "max_phase_abs_err_rad": 0.000852,
    "max_field_abs_rel_l2": 0.002855,
    "elapsed_sec": 0.0938
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.003931, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.002855, \"elapsed_sec\": 0.0938}\n__HEGEL_VERIFIER_AUTH_4a1e98bf842b58081c643f1a095b534a__:{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.003931, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.002855, \"elapsed_sec\": 0.0938}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5472.48,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
