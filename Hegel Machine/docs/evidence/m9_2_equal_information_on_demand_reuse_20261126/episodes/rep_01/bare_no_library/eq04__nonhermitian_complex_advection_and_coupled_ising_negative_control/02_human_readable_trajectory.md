# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq04__nonhermitian_complex_advection_and_coupled_ising_negative_control`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Non-Hermitian Complex Advection-Reaction PDE & Non-Separable Pairwise Co-Activation Subset Selection` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `186,843` (`178,405` / `59,776` / `8,438`) |
| **Total Episode Wall Time (s)** | `288.95s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`18627 ms` LLM)
- **Rationale**: Inspect the validator, public input metadata, and SciPy availability to detect schema or dependency mismatches that would change the implementation or test harness. No case arrays are transferred.
- **Alternatives Weighed**: `Greedy support selection` (rejected: Pair penalties couple activation decisions, and enumerating all 2,048 supports is tractable.)

```bash
ls -lah /holdout_task && sed -n '1,220p' /holdout_task/validate_public_schema.py && python3 - <<'PY'
import pickle
from pathlib import Path
import scipy
p = Path('/holdout_task/public_eq04_cases.pkl')
with p.open('rb') as f:
    cases = pickle.load(f)
print('scipy', scipy.__version__, 'public container', type(cases).__name__, 'count', len(cases))
if isinstance(cases, dict):
    print('top-level keys', list(cases)[:20]); seq = cases.values()
else:
    seq = cases
for case in list(seq)[:3]:
    if isinstance(case, dict):
        print({k: (getattr(v, 'shape', None), getattr(v, 'dtype', None)) if hasattr(v, 'shape') else (v if isinstance(v, (int, float, str)) else type(v).__name__) for k, v in case.items()})
    else:
        print(type(case).__name__)
PY
```
- **Container Exit Code**: `2` (`382 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`17677 ms` LLM)
- **Rationale**: Locate the fixtures and confirm that isolated Python can import SciPy. The results determine whether validation can use public cases and whether the planned sparse solver is available.
- **Alternatives Weighed**: `Assume public fixtures are unavailable and use synthetic cases immediately` (rejected: The failed lookup establishes only that the declared directory is absent; the fixtures may be under the reported /pretrain_task mount.)

```bash
printf 'Working directory: '; pwd
for d in /holdout_task /pretrain_task /app /workspace /results; do
  if [ -d "$d" ]; then
    printf '\n%s\n' "$d"
    find "$d" -maxdepth 3 -type f \( -name '*eq04*' -o -name 'validate_public_schema.py' -o -name 'AGENTS.md' -o -name '*nonhermitian*' -o -iname 'readme*' \) -printf '%p (%s bytes)\n' | head -60
  fi
done
printf '\nRoot entries:\n'; ls -1 /
python3 -I -c 'import numpy, scipy; print("NumPy", numpy.__version__, "SciPy", scipy.__version__)'
```
- **Container Exit Code**: `0` (`744 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Working directory: /pretrain_task

/pretrain_task

Root entries:
bin
boot
dev
etc
home
lib
lib64
media
mnt
opt
pretrain_task
proc
root
run
sbin
srv
sys
tmp
usr
var
NumPy 1.26.4 SciPy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`54335 ms` LLM)
- **Rationale**: SciPy is available, but the public fixtures and validator are not mounted. Write the solver at the requested path, then use an independent stencil and exhaustive reference to detect sign, indexing, or support-ranking errors that would require changes.
- **Alternatives Weighed**: `Solve each support through normal equations` (rejected: A shared augmented QR retains the ridge objective without squaring the condition number on nearly dependent basis responses.)

```bash
set -e
mkdir -p /holdout_task
cat > /holdout_task/nonhermitian_advection_coupled_select.py <<'PY'
"""Complex upwind PDE inversion with exhaustive coupled support selection."""

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _basis_responses(problem, basis):
    m, nx, ny = basis.shape
    hx = float(problem.get('Lx', 1.0)) / (nx + 1)
    hy = float(problem.get('Ly', 1.0)) / (ny + 1)
    dx = float(problem['nu']) / hx**2
    dy = float(problem['nu']) / hy**2
    vx = np.asarray(problem['vx'], dtype=float) / hx
    vy = np.asarray(problem['vy'], dtype=float) / hy
    ids = np.arange(nx * ny).reshape(nx, ny)
    diagonal = (2 * (dx + dy) + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma'], dtype=float)
                + 1j * float(problem['omega']))

    # Each velocity belongs to the row (receiving stencil), not the neighbor.
    rows = (ids, ids[1:, :], ids[:-1, :], ids[:, 1:], ids[:, :-1])
    cols = (ids, ids[:-1, :], ids[1:, :], ids[:, :-1], ids[:, 1:])
    values = (diagonal, -dx - np.maximum(vx[1:, :], 0),
              -dx + np.minimum(vx[:-1, :], 0),
              -dy - np.maximum(vy[:, 1:], 0),
              -dy + np.minimum(vy[:, :-1], 0))
    matrix = coo_matrix(
        (np.concatenate([v.ravel() for v in values]),
         (np.concatenate([r.ravel() for r in rows]),
          np.concatenate([c.ravel() for c in cols]))),
        shape=(nx * ny, nx * ny), dtype=np.complex128).tocsc()
    return splu(matrix).solve(basis.reshape(m, -1).T)


def _select_support(g, y, lam2, lam0, pairs):
    m = g.shape[1]
    # One QR compresses every augmented ridge problem to at most M rows,
    # without squaring the condition number as normal equations would.
    augmented = np.vstack((g, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:len(y)].conj().T @ y
    orthogonal = np.concatenate((y, np.zeros(m, dtype=np.complex128))) - q @ target
    constant = 0.5 * float(np.vdot(orthogonal, orthogonal).real)

    pair_masks = [((1 << int(u)) | (1 << int(v)), float(gamma))
                  for u, v, gamma in pairs]
    best_score = 0.5 * float(np.vdot(target, target).real)
    best_weights = np.zeros(m, dtype=np.complex128)
    for mask in range(1, 1 << m):
        penalty = lam0 * mask.bit_count() + sum(
            gamma for pair_mask, gamma in pair_masks if (mask & pair_mask) == pair_mask)
        if penalty >= best_score:
            continue
        support = [i for i in range(m) if mask & (1 << i)]
        columns = r[:, support]
        coeffs = lstsq(columns, target, lapack_driver='gelsy', check_finite=False)[0]
        if np.any(np.abs(coeffs) <= 1e-8):
            continue
        residual = columns @ coeffs - target
        score = 0.5 * float(np.vdot(residual, residual).real) + penalty
        if score < best_score:
            best_score = score
            best_weights.fill(0)
            best_weights[support] = coeffs
    return best_weights, float(best_score + constant)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return complex coefficients, the penalized objective, and field magnitude."""
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    _, nx, ny = basis.shape
    responses = _basis_responses(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=int)
    g = responses[sensors[:, 0] * ny + sensors[:, 1], :]
    y = (np.asarray(problem['y_obs_re'], dtype=float)
         + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    weights, objective = _select_support(
        g, y, float(problem['lam2']), float(problem['lam0']), problem['pair_penalties'])
    return {
        'weights_real': weights.real.copy(),
        'weights_imag': weights.imag.copy(),
        'objective': objective,
        'predicted_field_abs': np.abs((responses @ weights).reshape(nx, ny)),
    }
PY
```
- **Container Exit Code**: `0` (`332 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac1350248f551b98 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ac135024 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`62101 ms` LLM)
- **Rationale**: Use isolated Python to compare the exact artifact against an independently assembled stencil and all 2,048 reference supports. An operator residual failure would point to assembly; an objective or support discrepancy would point to selection; the paired negative control tests whether the penalty actually changes the result.
- **Alternatives Weighed**: `Validate only output schema` (rejected: Correct shapes would not catch a stencil sign error or a solver that ignores pair coupling.)

```bash
set -e
cat > /tmp/verify_eq04.py <<'PY'
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
spec = importlib.util.spec_from_file_location('candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference_operator(p):
    nx, ny = p['vx'].shape
    hx, hy = p['Lx'] / (nx + 1), p['Ly'] / (ny + 1)
    a = lil_matrix((nx * ny, nx * ny), dtype=np.complex128)
    for i, j in np.ndindex(nx, ny):
        k = i * ny + j
        a[k, k] = p['sigma'][i, j] + 1j * p['omega']
        for di, dj, h, v in [(1, 0, hx, p['vx'][i, j]), (0, 1, hy, p['vy'][i, j])]:
            a[k, k] += 2 * p['nu'] / h**2 + abs(v) / h
            for sign in (-1, 1):
                ii, jj = i + sign * di, j + sign * dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    a[k, ii * ny + jj] = -p['nu'] / h**2 - max(-sign * v, 0) / h
    return a.tocsr()


def objective(p, g, y, w):
    residual = g @ w - y
    active = abs(w) > 1e-8
    return (0.5 * np.vdot(residual, residual).real + 0.5 * p['lam2'] * np.vdot(w, w).real
            + p['lam0'] * active.sum() + sum(c for u, v, c in p['pair_penalties'] if active[u] and active[v]))


def reference(p, g, y):
    b = g.conj().T @ y
    h = g.conj().T @ g + p['lam2'] * np.eye(11)
    best = (np.inf, None)
    second = np.inf
    for mask in range(2048):
        s = [i for i in range(11) if (mask >> i) & 1]
        w = np.zeros(11, dtype=complex)
        if s:
            w[s] = solve(h[np.ix_(s, s)], b[s], assume_a='pos')
            if np.any(abs(w[s]) <= 1e-8):
                continue
        value = objective(p, g, y, w)
        if value < best[0]:
            second = best[0]
            best = (value, w)
        elif value < second:
            second = value
    return *best, second - best[0]


rng = np.random.default_rng(20504)
max_errors = np.zeros(5)
for case in ('mixed-sign', 'anisotropic', 'low-diffusion', 'duplicate-columns', 'empty', 'pair-coupled', 'pair-free'):
    p = dict(Lx=1.7, Ly=0.8, nu=0.009, omega=7.1, lam2=0.003, lam0=0.001)
    if case == 'anisotropic':
        p.update(Lx=3.1, Ly=0.21, omega=-70, nu=0.03)
    if case == 'low-diffusion':
        p.update(nu=1e-6, omega=-0.3)
    p.update(vx=rng.uniform(-2, 2, (32, 32)), vy=rng.uniform(-1, 1, (32, 32)),
             sigma=rng.uniform(-0.15, 0.4, (32, 32)), basis=rng.normal(size=(11, 32, 32)),
             pair_penalties=[(0, 1, .09), (1, 6, .02), (6, 4, .008), (4, 9, .01)])
    if case == 'duplicate-columns':
        p['lam2'] = 1e-7
        p['basis'][1] = p['basis'][0] + 1e-8 * p['basis'][1]
    if case in ('pair-coupled', 'pair-free'):
        p.update(vx=np.zeros((32, 32)), vy=np.zeros((32, 32)), sigma=np.zeros((32, 32)), lam0=.02, lam2=.01)
        if case == 'pair-free':
            p['pair_penalties'] = []
        else:
            p['pair_penalties'] = [(0, 1, .8)]
        idx = np.arange(11) * 76 + 34
    else:
        idx = np.unique(np.r_[0, 31, 992, 1023, rng.choice(1024, size=65, replace=False)])
    p['sensor_ij'] = np.column_stack(np.unravel_index(idx, (32, 32)))
    a = reference_operator(p)
    if case in ('pair-coupled', 'pair-free'):
        # Delta forcing at widely separated sensors makes the singleton gains comparable.
        p['basis'].fill(0)
        p['basis'].reshape(11, -1)[np.arange(11), idx] = abs(a[0, 0])
    rhs = p['basis'].reshape(11, -1).T
    fields = spsolve(a, rhs)
    g = fields[idx]
    w0 = np.zeros(11, dtype=complex)
    w0[[0, 1, 6]] = [1+0.2j, -0.5+0.9j, 0.7-0.1j]
    if case in ('pair-coupled', 'pair-free'):
        w0[[0, 1, 6]] = [1.05+0.2j, -0.5+0.9j, 0]
    y = g @ w0
    if case == 'empty':
        y *= 0
    p.update(y_obs_re=y.real, y_obs_im=y.imag)
    expected, w_ref, gap = reference(p, g, y)
    start = time.perf_counter()
    out = module.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter() - start
    assert set(out) == {'weights_real', 'weights_imag', 'objective', 'predicted_field_abs'}
    for name, shape in [('weights_real', (11,)), ('weights_imag', (11,)), ('predicted_field_abs', (32, 32))]:
        assert out[name].shape == shape and out[name].dtype.kind == 'f' and np.isfinite(out[name]).all()
    assert isinstance(out['objective'], float)
    w = out['weights_real'] + 1j * out['weights_imag']
    f_ref = abs((fields @ w_ref).reshape(32, 32))
    fields_candidate = module._basis_responses(p, p['basis'].astype(complex))
    errors = [abs(out['objective'] - expected), abs(out['objective'] - objective(p, g, y, w)),
              np.linalg.norm(out['predicted_field_abs'] - f_ref) / max(np.linalg.norm(f_ref), 1e-15),
              np.linalg.norm(a @ fields_candidate - rhs) / np.linalg.norm(rhs), np.linalg.norm(w - w_ref)]
    max_errors = np.maximum(max_errors, errors)
    support = np.flatnonzero(abs(w) > 1e-8).tolist()
    print(case, 'errors', ['%.2e' % e for e in errors], 'support', support, 'reference gap', '%.2e' % gap, 'seconds', '%.3f' % elapsed)
    assert max(errors[:2]) <= 1e-6 and errors[2] <= 1e-5 and errors[3] < 1e-10 and errors[4] < 1e-5
    if case == 'pair-coupled':
        assert support == [0]
    elif case == 'pair-free':
        assert support == [0, 1]
    elif case == 'empty':
        assert not support
print('worst errors:', max_errors)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; else printf 'Public fixtures and validator unavailable in this container.\n'; fi
```
- **Container Exit Code**: `1` (`1163 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mixed-sign errors ['3.47e-17', '2.08e-17', '8.10e-16', '6.90e-16', '1.22e-15'] support [0, 3, 6, 7, 9] reference gap 5.51e-05 seconds 0.035
anisotropic errors ['1.08e-19', '1.08e-19', '0.00e+00', '1.22e-15', '0.00e+00'] support [] reference gap 9.20e-04 seconds 0.004
low-diffusion errors ['2.78e-17', '2.78e-17', '3.35e-15', '1.02e-14', '9.37e-15'] support [0, 1, 6] reference gap 1.00e-03 seconds 0.041
duplicate-columns errors ['0.00e+00', '0.00e+00', '6.76e-16', '7.04e-16', '1.12e-15'] support [0, 6] reference gap 1.00e-03 seconds 0.005
empty errors ['0.00e+00', '0.00e+00', '0.00e+00', '6.92e-16', '0.00e+00'] support [] reference gap inf seconds 0.003
pair-coupled errors ['0.00e+00', '0.00e+00', '1.94e-16', '3.91e-16', '2.99e-16'] support [0, 1] reference gap 2.00e-02 seconds 0.031
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_eq04.py", line 118, in <module>
    assert support == [0]
           ^^^^^^^^^^^^^^
AssertionError
```

### Turn 5 — `execute_in_remote_docker` (`21746 ms` LLM)
- **Rationale**: The candidate and reference both correctly selected [0, 1] at the original penalty. Increase only the synthetic penalty and rerun in isolated Python. If the reference switches to [0] but the candidate does not, revise support selection; if neither switches, recalibrate the fixture.
- **Alternatives Weighed**: `Remove the support assertion` (rejected: That would lose the explicit test that changing only the pair penalty changes the selected support.)

```bash
set -e
sed -i "s/\[(0, 1, \.8)\]/[(0, 1, 2.0)]/" /tmp/verify_eq04.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; else printf 'Public fixtures and validator unavailable in this container.\n'; fi
```
- **Container Exit Code**: `0` (`1744 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mixed-sign errors ['3.47e-17', '2.08e-17', '8.10e-16', '6.90e-16', '1.22e-15'] support [0, 3, 6, 7, 9] reference gap 5.51e-05 seconds 0.042
anisotropic errors ['1.08e-19', '1.08e-19', '0.00e+00', '1.22e-15', '0.00e+00'] support [] reference gap 9.20e-04 seconds 0.007
low-diffusion errors ['2.78e-17', '2.78e-17', '3.35e-15', '1.02e-14', '9.37e-15'] support [0, 1, 6] reference gap 1.00e-03 seconds 0.049
duplicate-columns errors ['0.00e+00', '0.00e+00', '6.76e-16', '7.04e-16', '1.12e-15'] support [0, 6] reference gap 1.00e-03 seconds 0.012
empty errors ['0.00e+00', '0.00e+00', '0.00e+00', '6.92e-16', '0.00e+00'] support [] reference gap inf seconds 0.004
pair-coupled errors ['2.22e-16', '2.22e-16', '2.41e-16', '3.91e-16', '2.22e-16'] support [0] reference gap 2.00e-02 seconds 0.043
pair-free errors ['6.94e-18', '6.94e-18', '1.94e-16', '3.91e-16', '2.99e-16'] support [0, 1] reference gap 2.00e-02 seconds 0.008
worst errors: [2.22044605e-16 2.22044605e-16 3.35324451e-15 1.01606865e-14
 9.37353910e-15]
Public fixtures and validator unavailable in this container.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac1350248f551b98 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ac135024 (directory).
```

### Turn 6 — `submit_completed_task` (`14808 ms` LLM)
- **Rationale**: Isolated Python verified the saved artifact against an independent stencil and exhaustive support reference. Both error metrics passed their tolerances, and the paired control confirmed that the co-activation penalty affects support selection.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3849'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Seven synthetic stress cases passed against an independent exhaustive reference, including a control where the pair penalty changed the selected support. Worst absolute objective error was 2.22e-16; worst relative field error was 3.35e-15. The public cases and validator were not available in the container.`

### Turn 7 — `execute_in_remote_docker` (`37096 ms` LLM)
- **Rationale**: Check for available public cases, then test the unchanged artifact in isolated Python against independent per-support ridge solves. The new slices target full-support pruning, fewer sensors than coefficients, and uneven column scaling. A mismatch would warrant revising pruning or the reduced least-squares solve; agreement would support retaining the verified artifact.
- **Alternatives Weighed**: `Audit batch confounding or relay modes` (rejected: The specified inverse problem has neither batch labels nor relay state.)

```bash
set -e
if [ -f /holdout_task/public_eq04_cases.pkl ] && [ -f /holdout_task/validate_public_schema.py ]; then
  python3 -I /holdout_task/validate_public_schema.py
else
  printf 'Public fixtures and validator unavailable; running synthetic contract slices.\n'
fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('eq04', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(493)
m = 11
cases = [
    ('all-active', np.eye(m, dtype=complex) * (1 + 2j), np.arange(1, m+1) + 0.4j, 2e-2, 1e-3, [(0, 10, 2e-3)]),
    ('four-sensors', rng.normal(size=(4,m)) + 1j*rng.normal(size=(4,m)), rng.normal(size=4) + 1j*rng.normal(size=4), 1e-3, 1e-4, [(0, 10, .02), (2, 9, .005), (9, 10, .03)]),
    ('scaled-columns', (rng.normal(size=(18,m)) + 1j*rng.normal(size=(18,m))) * np.geomspace(1e-4, 1e4, m), rng.normal(size=18) + 1j*rng.normal(size=18), 1e-5, 1e-3, [(0, 10, .01), (1, 8, .02), (4, 10, .03)]),
]
worst_obj = worst_pred = 0.
for name, g, y, lam2, lam0, pairs in cases:
    def cost(w):
        active = abs(w) > 1e-8
        return float(.5 * (np.linalg.norm(g @ w - y)**2 + lam2 * np.linalg.norm(w)**2) + lam0*active.sum() + sum(c for i,j,c in pairs if active[i] and active[j]))
    ref = np.zeros(m, complex)
    ref_cost = cost(ref)
    for mask in range(1, 1 << m):
        idx = [i for i in range(m) if mask >> i & 1]
        n = len(idx)
        x = np.linalg.lstsq(np.vstack((g[:,idx], np.sqrt(lam2)*np.eye(n))), np.r_[y, np.zeros(n)], rcond=None)[0]
        if np.any(abs(x) <= 1e-8):
            continue
        trial = np.zeros(m, complex)
        trial[idx] = x
        score = cost(trial)
        if score < ref_cost:
            ref_cost, ref = score, trial
    w, reported = mod._select_support(g, y, lam2, lam0, pairs)
    e_obj = max(abs(reported - ref_cost), abs(reported - cost(w)))
    e_pred = np.linalg.norm(g @ (w - ref)) / max(np.linalg.norm(g @ ref), 1e-15)
    print(f'{name}: objective error {e_obj:.3e}; relative sensor prediction error {e_pred:.3e}; active count {np.count_nonzero(w)}')
    assert e_obj <= 1e-6 and e_pred <= 1e-5
    if name == 'all-active':
        assert np.count_nonzero(w) == 11
    worst_obj, worst_pred = max(worst_obj, e_obj), max(worst_pred, e_pred)
print(f'Audit worst: objective {worst_obj:.3e}, relative sensor prediction {worst_pred:.3e}')
PY
```
- **Container Exit Code**: `0` (`1259 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixtures and validator unavailable; running synthetic contract slices.
all-active: objective error 0.000e+00; relative sensor prediction error 9.188e-17; active count 11
four-sensors: objective error 6.505e-19; relative sensor prediction error 8.990e-16; active count 5
scaled-columns: objective error 2.665e-15; relative sensor prediction error 1.210e-10; active count 10
Audit worst: objective 2.665e-15, relative sensor prediction 1.210e-10

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ac1350248f551b98 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ac135024 (directory).
```

### Turn 8 — `submit_completed_task` (`15338 ms` LLM)
- **Rationale**: The saved artifact passed isolated-Python verification against independent references. The pair-penalty control changed the selected support as expected, and the additional full-support, few-sensor, and uneven-scale slices passed without requiring an edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3849'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/nonhermitian_advection_coupled_select.py`. Seven end-to-end synthetic cases and three additional support-selection stress cases passed. The worst absolute objective error was 2.67e-15, and the worst end-to-end relative field error was 3.35e-15, both below tolerance. The public fixtures and validator were unavailable.`

### Turn 9 — `submit_completed_task` (`15881 ms` LLM)
- **Rationale**: The verification receipt is bound to the saved artifact. Independent references confirmed both error tolerances, and the negative control showed that changing the pair penalty changes the selected support.
- **Alternatives Weighed**: `Repeat the generic audit` (rejected: The task has no batch labels or relay state, and the unchanged artifact already passed isolated-Python validation of the stencil, pair coupling, and numerical tolerances.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/nonhermitian_advection_coupled_select.py`. Seven end-to-end synthetic cases and three additional support-selection stress cases passed. Worst absolute objective error was 2.67e-15; worst end-to-end relative field error was 3.35e-15. The public fixtures and validator were unavailable.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/nonhermitian_advection_coupled_select.py": "FILE:3849"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_weights_rel_l2": 0.0,
    "max_objective_abs_err": 0.0,
    "max_field_rel_l2": 0.0,
    "student_elapsed_sec": 0.29,
    "instance_diagnostics": [
      {
        "seed": 99431,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      },
      {
        "seed": 99432,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      },
      {
        "seed": 99433,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.29, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_c07993bc2e3773018189e855ea53d634__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.29, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5946.39,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
