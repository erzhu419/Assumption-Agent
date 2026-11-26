# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `dialectical_trajectory_synthesis_high`
- **Synthesis Rationale**: [blinded_slot:candidate_b->antithesis] Both candidates report passing both public cases against the three limits: 3.0% relative source error, 0.020 rad circular phase error, and 3.0% relative field-magnitude error. Candidate A's worst full-fixture errors were 0.472%, 0.000623 rad, and 0.321%, with the forward field matching the reference to roundoff. Candidate B's worst errors were 1.069%, 0.000623 rad, and 0.814% when withholding entire receiver regions, and 0.415%, 0.000623 rad, and 0.307% when relocating measured spikes to high-leverage receivers or receivers most sensitive to the weakest source direction. Candidate B also reports stencil agreement to roundoff against an independent implementation. All mechanical public metrics are null, and the task sets no runtime cap. | Select B for its stronger validation on the public measurements. Withholding wall bands, interiors, and quadrants tests dependence on receiver placement; relocating actual measured spikes tests robustness where those spikes can have the most influence. A's full-fixture results are sound, but they do not test those failure modes. The selection rests on the measured-data checks, not B's synthetic trial count.
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `True` (4T) | Antithesis (`Wormhole`) = `True` (9T)

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
| **Turns Used / Max Turns** | `9 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `253,354` (`244,779` / `71,680` / `8,575`) |
| **Total Episode Wall Time (s)** | `270.82s` |

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

### Turn 1 — `execute_in_remote_docker` (`16032 ms` LLM)
- **Rationale**: Inspect the public schema and available numerical libraries before implementing: the field layout determines stencil and receiver indexing, and the presence of reference outputs determines which accuracy checks can be run directly.
- **Alternatives Weighed**: `Reuse the SPD Darcy solver` (rejected: Its definiteness and Hermitian symmetry requirements do not hold for the prescribed outgoing Helmholtz operator.)

```bash
ls -la /holdout_task; ls -la /tmp/hegel_knowledge_ref; python3 - <<'PY'
import pickle, numpy as np, scipy
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f: data=pickle.load(f)
print('SciPy',scipy.__version__)
def show(x,p='',d=0):
 if isinstance(x,dict):
  for k,v in list(x.items())[:20]: show(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if d<5:
   for i,v in enumerate(x[:2]): show(v,p+'/'+str(i),d+1)
 elif isinstance(x,np.ndarray): print(p,x.shape,x.dtype, 'min/max', (x.min(),x.max()) if x.size and x.dtype.kind in 'fi' else '-')
 else: print(p,repr(x)[:160])
show(data)
PY
```
- **Container Exit Code**: `0` (`595 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
n.reference.txt
-rw-r--r-- 1 ubuntu ubuntu  1676 Nov 26 00:46 describe_ref.py
SciPy 1.13.0
 list 2
/0/problem/Lx 1.0
/0/problem/Ly 1.0
/0/problem/k0 11.5
/0/problem/beta_imp 0.85
/0/problem/n_sq (32, 32) float64 min/max (0.8013327008377176, 2.1484758890462508)
/0/problem/basis (6, 32, 32) float64 min/max (7.777719264893782e-58, 24.813476666399524)
/0/problem/pilot_calibration/known_field_re (12,) float64 min/max (-0.5609946845948346, 1.9271942759646705)
/0/problem/pilot_calibration/known_field_im (12,) float64 min/max (-0.6647931232127696, 0.7959843180131354)
/0/problem/pilot_calibration/amplitude_obs (12,) float64 min/max (0.33798887730235655, 2.0410925014957595)
/0/problem/pilot_calibration/wrapped_phase_obs (12,) float64 min/max (-1.5439738505952283, 2.332925420945619)
/0/problem/survey/sensor_ij (140, 2) int64 min/max (0, 31)
/0/problem/survey/amplitude_obs (140,) float64 min/max (0.003940612776623732, 0.6957272476555351)
/0/problem/survey/wrapped_phase_obs (140,) float64 min/max (-3.1167049813598284, 3.1403266841958404)
/0/problem/survey/tikhonov_alpha 1e-06
/0/gt/source_amplitudes (6,) float64 min/max (-0.7275636911125073, 1.414714791178747)
/0/gt/global_phase_rad -0.8141122105046681
/0/gt/predicted_field_abs (32, 32) float64 min/max (0.0029623351218316853, 0.22726746347359125)
/1/problem/Lx 1.0
/1/problem/Ly 1.0
/1/problem/k0 11.5
/1/problem/beta_imp 0.85
/1/problem/n_sq (32, 32) float64 min/max (0.8013327008377176, 2.1484758890462508)
/1/problem/basis (6, 32, 32) float64 min/max (7.777719264893782e-58, 24.813476666399524)
/1/problem/pilot_calibration/known_field_re (12,) float64 min/max (-0.027533252859279322, 1.2552135205309114)
/1/problem/pilot_calibration/known_field_im (12,) float64 min/max (-0.49403393542292023, 0.6534837285226502)
/1/problem/pilot_calibration/amplitude_obs (12,) float64 min/max (0.21619322732772722, 1.3575513988296413)
/1/problem/pilot_calibration/wrapped_phase_obs (12,) float64 min/max (-2.9695938176749612, -0.6076904610419879)
/1/problem/survey/sensor_ij (140, 2) int64 min/max (0, 31)
/1/problem/survey/amplitude_obs (140,) float64 min/max (0.003413999204516145, 0.7813177048211306)
/1/problem/survey/wrapped_phase_obs (140,) float64 min/max (-2.988561738334312, 3.1039441222422894)
/1/problem/survey/tikhonov_alpha 1e-06
/1/gt/source_amplitudes (6,) float64 min/max (-1.200183432565852, 1.35)
/1/gt/global_phase_rad -1.3120678293854167
/1/gt/predicted_field_abs (32, 32) float64 min/max (0.003642904351227908, 0.22973187462246958)
```

### Turn 2 — `execute_in_remote_docker` (`29244 ms` LLM)
- **Rationale**: An independent face-by-face solve will reveal any stencil or indexing mismatch against the supplied reference field. Pilot and survey residuals will distinguish calibration error, ordinary measurement noise, and impulsive contamination before choosing a robust fit.
- **Alternatives Weighed**: `Unwrap receiver phases before fitting` (rejected: The polar observations reconstruct complex pressures directly, without phase branches between different receivers.)

```bash
python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.robust_residual_weights; cat > /tmp/helmholtz_public_probe.py <<'PY'
import pickle
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

CASES = pickle.load(open('/holdout_task/public_helmholtz_cases.pkl','rb'))

def reference_transfer(p):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    A = lil_matrix((nx*ny,nx*ny), dtype=complex)
    for i, j in np.ndindex(nx,ny):
        r=i*ny+j
        v=-hx*hy*k*k*p['n_sq'][i,j]
        for di,dj,g,ell in ((-1,0,hy/hx,hy),(1,0,hy/hx,hy),(0,-1,hx/hy,hx),(0,1,hx/hy,hx)):
            a,b=i+di,j+dj
            if 0<=a<nx and 0<=b<ny:
                A[r,a*ny+b]=-g
                v+=g
            else:
                v-=1j*k*p['beta_imp']*ell
        A[r,r]=v
    return spsolve(A.tocsc(),hx*hy*p['basis'].reshape(len(p['basis']),-1).T)

def polar(d):
    return d['amplitude_obs']*np.exp(1j*d['wrapped_phase_obs'])

if __name__ == '__main__':
    for i,c in enumerate(CASES):
        p,g=c['problem'],c['gt']; s=p['survey']; q=p['pilot_calibration']
        T=reference_transfer(p)
        known=q['known_field_re']+1j*q['known_field_im']
        ph=np.angle(np.vdot(known,polar(q)))
        print('case',i,'pilot fit error',abs(np.angle(np.exp(1j*(ph-g['global_phase_rad'])))), 'pilot pointwise abs errors',np.round(np.abs(polar(q)-np.exp(1j*g['global_phase_rad'])*known),6))
        u=T@g['source_amplitudes']; ij=s['sensor_ij']; idx=np.ravel_multi_index(ij.T,p['n_sq'].shape)
        y=polar(s)*np.exp(-1j*g['global_phase_rad']); X=T[idx]
        e=y-u[idx]
        a=np.linalg.lstsq(np.concatenate([X.real,X.imag]),np.r_[y.real,y.imag],rcond=None)[0]
        print('forward abs error',np.linalg.norm(np.abs(u).reshape(p['n_sq'].shape)-g['predicted_field_abs'])/np.linalg.norm(u),'survey residual quantiles',np.quantile(abs(e),[0,.25,.5,.75,.85,.9,.92,.93,.95,.99,1]),'OLS amplitude rel error',np.linalg.norm(a-g['source_amplitudes'])/np.linalg.norm(g['source_amplitudes']),'unique receivers',len(np.unique(idx)))
        print('real design singular values', np.linalg.svd(np.r_[X.real,X.imag],compute_uv=False))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/helmholtz_public_probe.py
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
case 0 pilot fit error 0.0006227225657142733 pilot pointwise abs errors [0.001516 0.000629 0.001777 0.001658 0.001207 0.001115 0.003254 0.001962
 0.00727  0.002777 0.003417 0.001539]
forward abs error 1.1272979772244927e-15 survey residual quantiles [6.47881901e-06 1.38830670e-03 2.04397750e-03 2.60904210e-03
 2.99061952e-03 3.74096331e-03 4.01262154e-03 1.00151822e-01
 3.72538342e-01 6.29259867e-01 7.40687115e-01] OLS amplitude rel error 0.17180313491718766 unique receivers 140
real design singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722]
case 1 pilot fit error 0.00026689595102702407 pilot pointwise abs errors [0.005139 0.003659 0.001628 0.001572 0.000922 0.001817 0.002248 0.00301
 0.000128 0.001652 0.001466 0.002876]
forward abs error 9.931919169785765e-16 survey residual quantiles [2.09377251e-04 1.16546417e-03 1.80389227e-03 2.52441899e-03
 3.47499659e-03 4.26435661e-03 4.83633554e-03 9.99000576e-02
 4.09010034e-01 6.54635093e-01 6.86029873e-01] OLS amplitude rel error 0.2727211285456088 unique receivers 140
real design singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252]
```

### Turn 3 — `execute_in_remote_docker` (`58856 ms` LLM)
- **Rationale**: The independent stencil reproduces the supplied field to roundoff, so retain the physical model and change the survey estimator. Verify all three outputs against public truth and omit each survey receiver and pilot in turn to detect undue channel dependence.
- **Alternatives Weighed**: `Unweighted Tikhonov fit` (rejected: The public residuals have a distinct spike tail, and unweighted fits exceed the 3% source tolerance on both cases.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume acoustic source inversion from calibrated polar receivers."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_flux(n, conductance):
    links = np.full(n - 1, conductance)
    diagonal = np.r_[links, 0.0] + np.r_[0.0, links]
    return diags((-links, diagonal, -links), (-1, 0, 1), shape=(n, n))


def _unit_source_fields(problem):
    index_squared = np.asarray(problem['n_sq'])
    nx, ny = index_squared.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    area = hx * hy

    # Each interior face contributes one symmetric flux pair. Corners
    # receive both exterior-face contributions to the radiation term.
    wall_length = np.zeros((nx, ny))
    wall_length[0, :] += hy
    wall_length[-1, :] += hy
    wall_length[:, 0] += hx
    wall_length[:, -1] += hx
    potential = -area * k**2 * index_squared - 1j * k * problem['beta_imp'] * wall_length
    operator = (kron(_axis_flux(nx, hy / hx), eye(ny), format='csc')
                + kron(eye(nx), _axis_flux(ny, hx / hy), format='csc')
                + diags(potential.ravel(), format='csc'))
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(len(basis), nx * ny).T
    fields = splu(operator).solve(rhs.astype(np.complex128))
    return fields.reshape(nx, ny, len(basis))


def _complex_readings(panel):
    return np.asarray(panel['amplitude_obs']) * np.exp(1j * np.asarray(panel['wrapped_phase_obs']))


def _fit_real_sources(design, observations, alpha):
    """Cauchy IRLS with one weight for both components of each receiver."""
    n, m = design.shape
    full_design = np.vstack((design.real, design.imag, np.sqrt(alpha) * np.eye(m)))
    target = np.r_[observations.real, observations.imag, np.zeros(m)]

    def fit(weights):
        root = np.sqrt(weights)
        rows = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(full_design * rows[:, None], target * rows, rcond=None)[0]

    coefficients = fit(np.ones(n))
    floor = np.finfo(float).eps * max(np.max(np.abs(observations)), 1.0)
    # Re-estimate scale during the warm start so the initial, contaminated
    # least-squares residual does not set the final rejection threshold.
    for _ in range(8):
        residual = np.abs(design @ coefficients - observations)
        scale = max(3.0 * np.median(residual), floor)
        coefficients = fit(1.0 / (1.0 + (residual / scale)**2))

    # Fix the robust scale for the final objective and iterate to convergence.
    scale = max(3.0 * np.median(np.abs(design @ coefficients - observations)), floor)
    for _ in range(25):
        residual = np.abs(design @ coefficients - observations)
        updated = fit(1.0 / (1.0 + (residual / scale)**2))
        change = np.linalg.norm(updated - coefficients)
        coefficients = updated
        if change <= 1e-10 * max(1.0, np.linalg.norm(coefficients)):
            break
    return coefficients


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover the real source coefficients, pilot phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_readings(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _unit_source_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observations = _complex_readings(survey) * np.exp(-1j * phase)
    amplitudes = _fit_real_sources(fields[i, j], observations, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': amplitudes,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(fields @ amplitudes),
    }
PY
cat > /tmp/verify_helmholtz.py <<'PY'
import importlib.util
import time
import numpy as np
from pathlib import Path

# Import the exact deliverable without relying on the working directory.
path = Path('/holdout_task/helmholtz_phase_inversion.py')
spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
exec(Path('/tmp/helmholtz_public_probe.py').read_text().split("if __name__ == '__main__':")[0])


def metrics(out, gt):
    a, t, f = (out[key] for key in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert set(out) == set(gt)
    assert a.shape == gt['source_amplitudes'].shape and a.dtype.kind == 'f' and np.isfinite(a).all()
    assert isinstance(t, float) and -np.pi < t <= np.pi
    assert f.shape == gt['predicted_field_abs'].shape and f.dtype.kind == 'f' and np.isfinite(f).all() and (f >= 0).all()
    return np.array([np.linalg.norm(a - gt['source_amplitudes']) / np.linalg.norm(gt['source_amplitudes']),
                     abs(np.angle(np.exp(1j * (t - gt['global_phase_rad'])))),
                     np.linalg.norm(f - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs'])])


limits = np.array([.030, .020, .030])
for number, case in enumerate(CASES):
    p, gt = case['problem'], case['gt']
    start = time.perf_counter()
    out = solver.invert_helmholtz_sources_and_phase(p)
    elapsed = time.perf_counter() - start
    error = metrics(out, gt)
    transfer = solver._unit_source_fields(p)
    ref = reference_transfer(p).reshape(transfer.shape)
    transfer_error = np.linalg.norm(transfer - ref) / np.linalg.norm(ref)
    survey, pilot = p['survey'], p['pilot_calibration']
    phase = out['global_phase_rad']
    fields = transfer[tuple(survey['sensor_ij'].T)]
    y = polar(survey) * np.exp(-1j * phase)
    all_fields = []
    held_out = []
    for j in range(len(y)):
        take = np.arange(len(y)) != j
        a = solver._fit_real_sources(fields[take], y[take], survey['tikhonov_alpha'])
        all_fields.append(metrics(dict(out, source_amplitudes=a, predicted_field_abs=np.abs(transfer @ a)), gt))
        held_out.append(abs(fields[j] @ a - y[j]))
    pilot_errors = []
    for j in range(len(pilot['known_field_re'])):
        take = np.arange(len(pilot['known_field_re'])) != j
        reduced = {k: val[take] for k, val in pilot.items()}
        pilot_errors.append(metrics(solver.invert_helmholtz_sources_and_phase(dict(p, pilot_calibration=reduced)), gt))
    survey_worst = np.max(all_fields, axis=0)
    pilot_worst = np.max(pilot_errors, axis=0)
    print(f'public {number}: source/phase/field {error}; seconds {elapsed:.4f}; transfer relative error {transfer_error:.3g}')
    print(f'  survey leave-one-out worst {survey_worst}; held-out residual median/p90/max {np.quantile(held_out,[.5,.9,1])}')
    print(f'  pilot leave-one-out worst {pilot_worst}')
    assert transfer_error < 1e-12
    assert np.all(np.maximum.reduce([error, survey_worst, pilot_worst]) <= limits)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_helmholtz.py
```
- **Container Exit Code**: `0` (`975 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
public 0: source/phase/field [0.00323753 0.00062272 0.00220617]; seconds 0.0065; transfer relative error 1.31e-15
  survey leave-one-out worst [0.0039096  0.00062272 0.00246613]; held-out residual median/p90/max [0.00209494 0.00385698 0.74073162]
  pilot leave-one-out worst [0.00328098 0.00074135 0.00222293]
public 1: source/phase/field [0.00517179 0.0002669  0.00342805]; seconds 0.0057; transfer relative error 1.31e-15
  survey leave-one-out worst [0.00574849 0.0002669  0.0038653 ]; held-out residual median/p90/max [0.00176102 0.00452383 0.6861525 ]
  pilot leave-one-out worst [0.00520344 0.00058973 0.00343738]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=89d11065fdf39f6b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_89d11065 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`34143 ms` LLM)
- **Rationale**: Both public cases pass, including all receiver and pilot omissions. An independent rectangular-grid solve can expose swapped face weights that square cases cannot; phase-cut and clustered-spike trials can expose calibration or regression failures. Change only the component implicated by a failed check.
- **Alternatives Weighed**: `Retune the robust fit on the two public cases` (rejected: All three public metrics already have substantial margin, and both cases share one geometry.)

```bash
cat > /tmp/stress_helmholtz.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('inversion', root / 'helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
exec(Path('/tmp/helmholtz_public_probe.py').read_text().split("if __name__ == '__main__':")[0])

rng = np.random.default_rng(7423)
worst = {}
for Lx, Ly, k, beta in [(1.0, 1.0, 11.5, .85), (1.2, .8, 12.2, .65), (.82, 1.16, 10.8, 1.05)]:
    p = dict(CASES[0]['problem'], Lx=Lx, Ly=Ly, k0=k, beta_imp=beta)
    shape = p['n_sq'].shape
    ref = reference_transfer(p)
    transfer_error = np.linalg.norm(solver._unit_source_fields(p).reshape(ref.shape) - ref) / np.linalg.norm(ref)
    assert transfer_error < 1e-12, (Lx, Ly, transfer_error)
    for mode in ('random', 'clustered'):
        errors = []
        for trial in range(16):
            a = rng.normal(size=ref.shape[1]); a *= 2.1 / np.linalg.norm(a)
            u = ref @ a
            phase = [np.pi, np.pi-0.0001, -np.pi+0.0001, 0.47][trial % 4]
            rotate = np.exp(1j * phase)
            known = rng.normal(size=12) + 1j*rng.normal(size=12)
            zp = rotate * known + .002 * (rng.normal(size=12) + 1j*rng.normal(size=12))
            idx = rng.choice(u.size, 140, replace=False)
            ij = np.array(np.unravel_index(idx, shape)).T
            z = rotate * u[idx] + .003 * (rng.normal(size=140) + 1j*rng.normal(size=140))
            if mode == 'random':
                bad = rng.choice(len(idx), 10, replace=False)
            else:
                center = ij[np.argmax(np.linalg.norm(ref[idx], axis=1))]
                bad = np.argsort(np.sum((ij-center)**2, axis=1))[:10]
            z[bad] += rng.uniform(.10,.75,size=10) * np.exp(1j*rng.uniform(-np.pi,np.pi,size=10))
            p['pilot_calibration'] = dict(known_field_re=known.real, known_field_im=known.imag, amplitude_obs=abs(zp), wrapped_phase_obs=np.angle(zp))
            p['survey'] = dict(sensor_ij=ij, amplitude_obs=abs(z), wrapped_phase_obs=np.angle(z), tikhonov_alpha=1e-6)
            out = solver.invert_helmholtz_sources_and_phase(p)
            t = out['global_phase_rad']
            assert isinstance(t,float) and -np.pi < t <= np.pi
            errors.append([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(t-phase)))), np.linalg.norm(out['predicted_field_abs'].ravel()-abs(u))/np.linalg.norm(u)])
        worst[(Lx, Ly, mode)] = np.max(errors, axis=0)
        print(f'{Lx:g} x {Ly:g}, {mode}: source/phase/field worst {worst[(Lx,Ly,mode)]}; transfer error {transfer_error:.3g}')
assert np.all(np.max(list(worst.values()), axis=0) <= [.030,.020,.030])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_helmholtz.py
```
- **Container Exit Code**: `0` (`1210 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1 x 1, random: source/phase/field worst [0.0132707 0.0009623 0.0085954]; transfer error 1.31e-15
1 x 1, clustered: source/phase/field worst [0.01294212 0.00087037 0.00873855]; transfer error 1.31e-15
1.2 x 0.8, random: source/phase/field worst [0.01393651 0.00076027 0.01059634]; transfer error 2.78e-15
1.2 x 0.8, clustered: source/phase/field worst [0.01486309 0.00101655 0.00926638]; transfer error 2.78e-15
0.82 x 1.16, random: source/phase/field worst [0.01509058 0.00142058 0.00747411]; transfer error 1.85e-15
0.82 x 1.16, clustered: source/phase/field worst [0.01199888 0.00093924 0.00707125]; transfer error 1.85e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=89d11065fdf39f6b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_89d11065 (directory).
```

### Turn 5 — `submit_completed_task` (`13037 ms` LLM)
- **Rationale**: The exact deliverable passed isolated-process verification. The transfer matched an independent stencil to roundoff on square and rectangular domains, and phase-cut and clustered-spike trials stayed below all three tolerances. No further changes are warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`23` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3806'}): Do NOT lock prematurely with `23` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, all leave-one-out checks, and all 96 stress trials passed. Worst stress errors were 1.51% for source amplitudes, 0.00142 rad for global phase, and 1.06% for field magnitude.`

### Turn 6 — `execute_in_remote_docker` (`29698 ms` LLM)
- **Rationale**: Check the shared-rotation identity on real pilot differences and withhold whole receiver regions to expose spatial bias or dependence that individual omissions can miss. A failed slice would identify whether to revisit calibration, receiver bias, or the conditioning of the retained design.
- **Alternatives Weighed**: `Fit separate receiver-region offsets` (rejected: The input specifies one oscillator and no batch labels; first check whether held-out spatial groups show coherent residual offsets or a tolerance breach.); `Add relay, drift, or classification checks` (rejected: This is a steady-state complex wave inverse problem with no relay states, time drift, or class labels.)

```bash
cat > /tmp/audit_helmholtz_slices.py <<'PY'
import runpy
import numpy as np
v = runpy.run_path('/tmp/verify_helmholtz.py')
solver, metrics, polar = (v[key] for key in ('solver', 'metrics', 'polar'))
worst = np.zeros(3)
for case_id, case in enumerate(v['CASES']):
    p, gt = case['problem'], case['gt']
    s, q = p['survey'], p['pilot_calibration']
    out = solver.invert_helmholtz_sources_and_phase(p)
    phase = out['global_phase_rad']
    known = q['known_field_re'] + 1j*q['known_field_im']
    # Each calibrated pilot pair must obey the same rotation, including differences.
    ii, jj = np.triu_indices(len(known), 1)
    d_known = known[ii] - known[jj]
    measured = polar(q)
    d_measured = measured[ii] - measured[jj]
    pair_phase = np.angle(np.vdot(d_known, d_measured))
    pair_error = abs(np.angle(np.exp(1j*(pair_phase-gt['global_phase_rad']))))
    assert pair_error <= .020
    fields = solver._unit_source_fields(p)
    ij = s['sensor_ij']
    x = fields[tuple(ij.T)]
    y = polar(s)*np.exp(-1j*phase)
    nx, ny = fields.shape[:2]
    wall_dist = np.min(np.c_[ij, np.array([nx-1, ny-1])-ij], axis=1)
    masks = {'wall band': wall_dist <= 3, 'interior': wall_dist > 3}
    for left, bottom in np.ndindex(2, 2):
        masks[f'quadrant {left}{bottom}'] = ((ij[:,0] < nx/2) == left) & ((ij[:,1] < ny/2) == bottom)
    for name, held in masks.items():
        a = solver._fit_real_sources(x[~held], y[~held], s['tikhonov_alpha'])
        error = metrics(dict(out, source_amplitudes=a, predicted_field_abs=np.abs(fields @ a)), gt)
        r = y[held] - x[held] @ a
        center = np.median(r.real) + 1j*np.median(r.imag)
        worst = np.maximum(worst, error)
        print(f'public {case_id}, withheld {name} ({held.sum()}): errors {error}; held-out median |residual| {np.median(abs(r)):.4g}, |complex componentwise median| {abs(center):.4g}')
    print(f'public {case_id}: pilot pair-difference phase error {pair_error:.6g} rad')
print('worst grouped omission (source, phase, field):', worst)
assert np.all(worst <= v['limits'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_helmholtz_slices.py
```
- **Container Exit Code**: `0` (`853 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4849 0.0002669  0.0038653 ]; held-out residual median/p90/max [0.00176102 0.00452383 0.6861525 ]
  pilot leave-one-out worst [0.00520344 0.00058973 0.00343738]
public 0, withheld wall band (68): errors [0.004169   0.00062272 0.00265638]; held-out median |residual| 0.002077, |complex componentwise median| 0.0003501
public 0, withheld interior (72): errors [0.01068755 0.00062272 0.00814227]; held-out median |residual| 0.002002, |complex componentwise median| 0.0005007
public 0, withheld quadrant 00 (37): errors [0.00331838 0.00062272 0.00276846]; held-out median |residual| 0.002001, |complex componentwise median| 0.0003395
public 0, withheld quadrant 01 (31): errors [0.00521825 0.00062272 0.00292737]; held-out median |residual| 0.001863, |complex componentwise median| 0.0001403
public 0, withheld quadrant 10 (36): errors [0.00240705 0.00062272 0.0017758 ]; held-out median |residual| 0.001883, |complex componentwise median| 0.0002128
public 0, withheld quadrant 11 (36): errors [0.0042031  0.00062272 0.00345202]; held-out median |residual| 0.002339, |complex componentwise median| 0.001216
public 0: pilot pair-difference phase error 0.000243878 rad
public 1, withheld wall band (66): errors [0.00575415 0.0002669  0.00393232]; held-out median |residual| 0.001718, |complex componentwise median| 0.0004259
public 1, withheld interior (74): errors [0.00553704 0.0002669  0.00362226]; held-out median |residual| 0.0019, |complex componentwise median| 0.0003033
public 1, withheld quadrant 00 (31): errors [0.00683346 0.0002669  0.00410075]; held-out median |residual| 0.001825, |complex componentwise median| 0.0001499
public 1, withheld quadrant 01 (31): errors [0.00503707 0.0002669  0.00361301]; held-out median |residual| 0.001588, |complex componentwise median| 0.0003064
public 1, withheld quadrant 10 (36): errors [0.00663734 0.0002669  0.00425084]; held-out median |residual| 0.001804, |complex componentwise median| 0.0003136
public 1, withheld quadrant 11 (42): errors [0.00590991 0.0002669  0.00395   ]; held-out median |residual| 0.001924, |complex componentwise median| 0.0003446
public 1: pilot pair-difference phase error 0.00107975 rad
worst grouped omission (source, phase, field): [0.01068755 0.00062272 0.00814227]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=89d11065fdf39f6b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_89d11065 (directory).
```

### Turn 7 — `submit_completed_task` (`14100 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process verification. Its transfer agrees with an independent stencil to roundoff, pilot differences support the shared rotation, and both grouped public omissions and synthetic stress trials stayed within all three tolerances.
- **Alternatives Weighed**: `Add receiver-region offsets` (rejected: Every held-out region met all tolerances, and the spatial residual audit did not establish a need for separate offsets.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`21` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3806'}): Do NOT lock prematurely with `21` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, all receiver and pilot omission checks, and all 96 stress trials passed. Worst observed errors were 1.51% for source amplitudes, 0.00142 rad for global phase, and 1.06% for field magnitude.`

### Turn 8 — `execute_in_remote_docker` (`32261 ms` LLM)
- **Rationale**: Keep the measured residuals intact but relocate the largest ones to high-leverage channels and channels most informative about the weakest source direction. A tolerance breach would warrant testing a more robust starting fit; otherwise retain the verified artifact.
- **Alternatives Weighed**: `Repeat the spatial omission and pilot-pair audits` (rejected: Those checks already passed; they do not test whether impulses on the most informative receivers can pull the robust fit off target.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import runpy
import numpy as np

ref = runpy.run_path('/tmp/helmholtz_public_probe.py')
spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
worst = np.zeros(3)
for case_id, case in enumerate(ref['CASES']):
    p, gt = case['problem'], case['gt']
    s = p['survey']
    a = gt['source_amplitudes']
    t = gt['global_phase_rad']
    idx = np.ravel_multi_index(s['sensor_ij'].T, p['n_sq'].shape)
    X = ref['reference_transfer'](p)[idx]
    clean = np.exp(1j*t) * (X @ a)
    residual = ref['polar'](s) - clean
    n = len(idx)
    U, _, Vh = np.linalg.svd(np.r_[X.real, X.imag], full_matrices=False)
    leverage = np.sum(U[:n]**2 + U[n:]**2, axis=1)
    bad = np.argsort(abs(residual))[-round(.07*n):][::-1]
    for label, score in [('joint leverage', leverage), ('weakest source direction', abs(X @ Vh[-1]))]:
        # Exchange measured residuals without changing their amplitudes, phases, or count.
        selected = np.argsort(score)[-len(bad):][::-1]
        old = bad[~np.isin(bad, selected)]
        new = selected[~np.isin(selected, bad)]
        moved = residual.copy()
        moved[new], moved[old] = residual[old], residual[new]
        z = clean + moved
        out = solver.invert_helmholtz_sources_and_phase(dict(p, survey=dict(s, amplitude_obs=abs(z), wrapped_phase_obs=np.angle(z))))
        f = gt['predicted_field_abs']
        err = np.array([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(out['global_phase_rad']-t)))), np.linalg.norm(out['predicted_field_abs']-f)/np.linalg.norm(f)])
        worst = np.maximum(worst, err)
        print(f'public {case_id}, impulse relocation to {label}: source/phase/field errors {err}')
assert np.all(worst <= [.030, .020, .030]), worst
PY
```
- **Container Exit Code**: `0` (`583 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0, impulse relocation to joint leverage: source/phase/field errors [0.00288003 0.00062272 0.00247806]
public 0, impulse relocation to weakest source direction: source/phase/field errors [0.00249818 0.00062272 0.00202716]
public 1, impulse relocation to joint leverage: source/phase/field errors [0.00351129 0.0002669  0.0025299 ]
public 1, impulse relocation to weakest source direction: source/phase/field errors [0.00415121 0.0002669  0.00307004]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=89d11065fdf39f6b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_89d11065 (directory).
```

### Turn 9 — `submit_completed_task` (`14425 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated-process verification. The transfer matched an independent stencil to roundoff, and spatial omissions, phase-cut trials, and measured-spike relocation all passed with margin. No additional change is warranted.
- **Alternatives Weighed**: `Add a more robust starting fit` (rejected: Relocating measured spikes to high-leverage receivers and to receivers most informative about the weakest source direction did not threaten any tolerance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, receiver and pilot omission checks, and all 96 synthetic stress trials passed. Worst observed errors were 1.51% for source amplitudes, 0.00142 rad for global phase, and 1.06% for field magnitude.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/helmholtz_phase_inversion.py": "FILE:3806"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_source_rel_l2": 0.003994,
    "max_phase_abs_err_rad": 0.000852,
    "max_field_abs_rel_l2": 0.002879,
    "elapsed_sec": 0.0981
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.003994, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.002879, \"elapsed_sec\": 0.0981}\n__HEGEL_VERIFIER_AUTH_425fdd9f9aac1ecc15a3ddc2fe393925__:{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.003994, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.002879, \"elapsed_sec\": 0.0981}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5422.59,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
