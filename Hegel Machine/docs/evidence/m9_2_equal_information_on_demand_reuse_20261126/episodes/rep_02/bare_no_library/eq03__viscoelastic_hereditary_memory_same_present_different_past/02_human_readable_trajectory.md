# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq03__viscoelastic_hereditary_memory_same_present_different_past`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Multi-Scale Viscoelastic Hereditary Memory Rebinding & Coupon-Calibrated Creep Hazard Forecasting` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_multiscale_memory_and_outlier_deconfounding` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `199,535` (`189,409` / `55,680` / `10,126`) |
| **Total Episode Wall Time (s)** | `314.45s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17932 ms` LLM)
- **Rationale**: Inspect only the public schema and array metadata to catch an incorrect assumption about pickle nesting or validator invocation before implementing and measuring the specified estimator.
- **Alternatives Weighed**: `Fit a memoryless predictor` (rejected: Identical present-state inputs cannot distinguish the specified history-dependent twins.)

```bash
cd /holdout_task && ls -lh && sed -n '1,220p' validate_public_schema.py && python3 -I - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_eq03_datasets.pkl','rb') as f:
    data = pickle.load(f)
def describe(obj, depth=0):
    pad='  '*depth
    if isinstance(obj, dict) and depth < 4:
        for k,v in obj.items():
            print(f'{pad}{k}:', end=' ')
            if isinstance(v, dict):
                print(); describe(v,depth+1)
            else:
                print(type(v).__name__, getattr(v,'shape',''),getattr(v,'dtype',''),repr(v)[:100] if np.isscalar(v) else '')
    elif isinstance(obj,(tuple,list)):
        print(type(obj).__name__, len(obj));
        if obj: describe(obj[0], depth+1)
    else:
        print(type(obj).__name__)
describe(data)
PY
```
- **Container Exit Code**: `1` (`360 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: cd: /holdout_task: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`15211 ms` LLM)
- **Rationale**: Distinguish a path mismatch from missing public assets without transferring data. If the files are present, use them for validation; if not, create the required destination and use synthetic contract tests while reporting the missing public validation.
- **Alternatives Weighed**: `Assume the public data are unavailable and validate only on synthetic data` (rejected: The initial snapshot names /pretrain_task; a bounded metadata search can establish whether the public assets are staged elsewhere.)

```bash
printf 'Working directory: '; pwd; ls -la / /pretrain_task; find /pretrain_task /app /workspace /results /mnt /tmp -maxdepth 4 -type f \( -iname '*eq03*' -o -name 'validate_public_schema.py' -o -name 'AGENTS.md' -o -iname '*viscoelastic*' \) -printf '%p (%s bytes)\n' 2>/dev/null; python3 -I -c 'import numpy; print("NumPy", numpy.__version__)'
```
- **Container Exit Code**: `0` (`727 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Working directory: /pretrain_task
/:
total 60
drwxr-xr-x   1 root root 4096 Nov 26 04:50 .
drwxr-xr-x   1 root root 4096 Nov 26 04:50 ..
-rwxr-xr-x   1 root root    0 Nov 26 04:50 .dockerenv
lrwxrwxrwx   1 root root    7 Apr 22  2024 bin -> usr/bin
drwxr-xr-x   2 root root 4096 Apr 22  2024 boot
drwxr-xr-x   5 root root  340 Nov 26 04:50 dev
drwxr-xr-x   1 root root 4096 Nov 26 04:50 etc
drwxr-xr-x   3 root root 4096 Nov 10 14:56 home
lrwxrwxrwx   1 root root    7 Apr 22  2024 lib -> usr/lib
lrwxrwxrwx   1 root root    9 Apr 22  2024 lib64 -> usr/lib64
drwxr-xr-x   2 root root 4096 Nov 10 14:48 media
drwxr-xr-x   2 root root 4096 Nov 10 14:48 mnt
drwxr-xr-x   2 root root 4096 Nov 10 14:48 opt
drwxr-xr-x   2 root root 4096 Nov 23 09:31 pretrain_task
dr-xr-xr-x 654 root root    0 Nov 26 04:50 proc
drwx------   2 root root 4096 Nov 10 14:55 root
drwxr-xr-x   4 root root 4096 Nov 10 14:56 run
lrwxrwxrwx   1 root root    8 Apr 22  2024 sbin -> usr/sbin
drwxr-xr-x   2 root root 4096 Nov 10 14:48 srv
dr-xr-xr-x  13 root root    0 Nov 26 04:50 sys
drwxrwxrwt   2 root root 4096 Nov 10 14:56 tmp
drwxr-xr-x  12 root root 4096 Nov 10 14:48 usr
drwxr-xr-x  11 root root 4096 Nov 10 14:55 var

/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 04:50 ..
NumPy 1.26.4
```

### Turn 3 — `execute_in_remote_docker` (`49041 ms` LLM)
- **Rationale**: The specified model is linear once the chronological Prony states are reconstructed. Implement it and check the isolated entry point before testing coefficient recovery, twin divergence, and first crossings. The public assets were not found at the supplied location.
- **Alternatives Weighed**: `Use ordinary least squares for the final calibration` (rejected: It would give transducer spikes full weight and would not implement the required Cauchy objective.)

```bash
set -eu
mkdir -p /holdout_task
cat > /holdout_task/viscoelastic_hereditary_forecast.py <<'PY'
"""Coupon-calibrated hereditary creep and first-passage forecasts."""

import numpy as np


def _align(train, test):
    """Estimate lab calibration and balanced imputation on the pooled coupons."""
    raw, refs, lab, stratum = (
        np.concatenate((train[key], test[key]), axis=0)
        for key in ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')
    )
    aligned = np.empty_like(raw, dtype=float)
    for label in np.unique(lab):
        mask = lab == label
        controls = refs[mask]
        controls = controls[:, np.isfinite(controls).any(axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - offset) / gain

    cells = np.column_stack((lab, stratum))
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = len(lab) / (len(counts) * counts[inverse])
    observed = np.isfinite(aligned)
    means = (weights @ np.where(observed, aligned, 0.0)) / (weights @ observed)
    aligned = np.where(observed, aligned, means)
    return np.split(aligned, [len(train['lab_id'])], axis=0)


def _design(split, features, times, dt, taus):
    """Build the linear basis for strain increments above the initial strain."""
    history = np.asarray(split['past_stress_history'], dtype=float)
    hold = np.asarray(split['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    memory = np.zeros((len(hold), len(taus)))
    for stress in history.T:
        memory = decay * memory + stress[:, None] * injection

    relaxation = -np.expm1(-times[:, None] / taus)
    design = np.empty((len(hold), len(times), len(taus) + 1 + features.shape[1]))
    design[:, :, :len(taus)] = (hold[:, None] - memory)[:, None, :] * relaxation
    design[:, :, len(taus)] = times
    design[:, :, len(taus) + 1:] = features[:, None, :] * times[None, :, None]
    return design


def _cauchy_fit(design, target):
    """Robust linear calibration by Cauchy IRLS with a MAD residual scale."""
    x = design.reshape(-1, design.shape[-1])
    y = target.ravel()
    # Equilibrate columns without changing the physical parameterization.
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1.0
    x = x / norm
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = 1e-12 * max(1.0, np.median(np.abs(y)))
    for _ in range(60):
        residual = y - x @ beta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(2.0 * 1.4826 * mad, floor)
        root_weight = scale / np.hypot(scale, residual)
        updated = np.linalg.lstsq(x * root_weight[:, None], y * root_weight, rcond=None)[0]
        if np.linalg.norm(updated - beta) <= 1e-10 * (1.0 + np.linalg.norm(updated)):
            beta = updated
            break
        beta = updated
    return beta / norm


def _first_crossing(initial, forecast, critical, times):
    """Interpolate the first upward crossing, including the initial knot."""
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = reached.argmax(axis=1)
    result = np.full(len(initial), times[-1], dtype=float)
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    right = first[rows]
    left = right - 1
    ya = values[rows, left]
    yb = values[rows, right]
    result[rows] = knots[left] + (critical[rows] - ya) / (yb - ya) * (knots[right] - knots[left])
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the two memory modes and predict test curves and crossing times."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    aligned_train, aligned_test = _align(train, test)
    design = _design(train, aligned_train, times, dt, taus)
    target = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    theta = _cauchy_fit(design, target)
    initial = np.asarray(test['strain_0'], dtype=float)
    forecast = initial[:, None] + _design(test, aligned_test, times, dt, taus) @ theta
    return {
        'aligned_test_features': aligned_test,
        'hereditary_weights': theta,
        'predicted_strain': forecast,
        'crossing_times': _first_crossing(initial, forecast, np.asarray(test['critical_strain']), times),
    }
PY
python3 -I - <<'PY'
import runpy
from pathlib import Path
path = Path('/holdout_task/viscoelastic_hereditary_forecast.py')
assert callable(runpy.run_path(str(path))['solve_viscoelastic_hereditary_forecast'])
print('Isolated module load passed')
PY
```
- **Container Exit Code**: `0` (`631 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_hereditary_forecast.py']`
- **Stdout (tail)**:
```text
Isolated module load passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e82bf5c0e4c5fd37 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e82bf5c0 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`56718 ms` LLM)
- **Rationale**: Independent convolution, cell aggregation, and scalar crossing references can localize a contract mismatch. Seeded held-out twins, sparse cells, and noisy finite reference coupons test whether any error warrants changing the candidate.
- **Alternatives Weighed**: `Submit after the isolated import check` (rejected: Import success cannot detect chronology, pooled imputation, spike sensitivity, twin collapse, or first-crossing errors.)

```bash
set -eu
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy
import numpy as np
from pathlib import Path

root = Path('/holdout_task')
module = runpy.run_path(str(root / 'viscoelastic_hereditary_forecast.py'))
solve = module['solve_viscoelastic_hereditary_forecast']
taus = np.array([1., 4.5])
dt = .25
theta = np.array([.43, .69, .12, .028, -.034, .019, .023])
offset = np.array([-2.7, .2, 2.2, 5.1])
gain = np.array([.65, 1.8, .95, 2.65])

def fixture(seed, latent):
    rng = np.random.default_rng(seed)
    ntr, nte, pairs = 480, 160, 48
    n = ntr + nte
    times = np.r_[.05, np.sort(rng.uniform(.1, 7.95, 26)), 8.]
    lab = rng.choice(4, n, p=[.45, .26, .19, .10])
    stratum = (rng.random(n) < np.array([.18, .8, .32, .7])[lab]).astype(int)
    x = rng.normal(size=(n, 4)) + .5 * stratum[:, None]
    x[rng.random(x.shape) < .05] = np.nan
    raw = x * gain[lab, None] + offset[lab, None]
    ref = rng.normal(size=(n, 4)) * gain[lab, None] + offset[lab, None]
    ref[rng.random(ref.shape) < .08] = np.nan
    hold = rng.uniform(.8, 2.1, n)
    initial = rng.uniform(.1, .8, n)
    rate = rng.normal(.2, .05, n)
    history = rng.uniform(-.1, 2., (n, 1)) + rng.normal(0., .18, (n, 40))
    a, b = ntr + np.arange(pairs), ntr + pairs + np.arange(pairs)
    for v in (lab, stratum, raw, x, hold, initial, rate):
        v[b] = v[a]
    history[a] = rng.uniform(-.1, .2, (pairs, 40))
    history[b] = history[a] + rng.uniform(1., 1.6, (pairs, 1))
    calibrated = np.empty_like(raw)
    for j in range(4):
        mask = lab == j
        cols = [col[np.isfinite(col)] for col in ref[mask].T]
        mu = np.median([col.mean() for col in cols if col.size])
        sd = np.median([col.std() for col in cols if col.size])
        calibrated[mask] = (raw[mask] - mu) / sd

    def fill(values):
        filled = values.copy()
        for c, column in enumerate(values.T):
            numerator = denominator = 0.
            for j in range(4):
                for s in range(2):
                    cell = column[(lab == j) & (stratum == s)]
                    valid = cell[np.isfinite(cell)]
                    numerator += valid.sum() / cell.size
                    denominator += valid.size / cell.size
            filled[np.isnan(column), c] = numerator / denominator
        return filled

    aligned = fill(calibrated)
    physical = fill(x) if latent else aligned
    d = np.exp(-dt / taus)
    # Closed-form convolution, independent of the candidate recurrence.
    z = history @ ((1. - d) * d ** np.arange(39, -1, -1)[:, None])
    clean = (initial[:, None] + ((hold[:, None] - z) * theta[:2]) @ (1. - np.exp(-times / taus[:, None])) + (theta[2] + physical @ theta[3:])[:, None] * times)
    obs = clean[:ntr].copy()
    spikes = rng.random(obs.shape) < .09
    obs[spikes] += np.clip(.45 * rng.standard_cauchy(spikes.sum()), -.95, .95)
    if latent:
        obs += rng.normal(0., .008, obs.shape)
    cols = dict(lab_id=lab, stratum=stratum, coupon_features_raw=raw, ref_coupons=ref, stress_hold_0=hold, strain_0=initial, instant_rate_0=rate, past_stress_history=history)
    train = {k: v[:ntr] for k, v in cols.items()}
    test = {k: v[ntr:] for k, v in cols.items()}
    train['strain_curves_obs'] = obs
    truth = clean[ntr:]
    y0 = initial[ntr:]
    critical = y0 + rng.uniform(.02, 1.4, nte) * (np.maximum(truth.max(axis=1) - y0, .03))
    critical[::17] = y0[::17]
    test['critical_strain'] = critical
    out = solve(dict(train=train, test=test, history_dt=dt, memory_taus=taus, eval_times=times))
    assert {k: v.shape for k, v in out.items()} == dict(aligned_test_features=(nte,4), hereditary_weights=(7,), predicted_strain=(nte,len(times)), crossing_times=(nte,))
    assert all(np.isfinite(v).all() for v in out.values())
    np.testing.assert_allclose(out['aligned_test_features'], aligned[ntr:], rtol=1e-12, atol=1e-12)
    crossing = []
    for ini, ys, crit in zip(y0, truth, critical):
        at, ay, result = 0., ini, times[-1]
        if ini >= crit:
            result = 0.
        else:
            for bt, by in zip(times, ys):
                if by >= crit:
                    result = at + (bt-at)*(crit-ay)/(by-ay)
                    break
                at, ay = bt, by
        crossing.append(result)
    pe = abs(out['predicted_strain'] - truth)
    ce = abs(out['crossing_times'] - crossing)
    fe = abs(out['aligned_test_features'] - physical[ntr:])
    metrics = [fe.mean(), np.linalg.norm(out['hereditary_weights']-theta)/np.linalg.norm(theta), pe.mean(), pe[:2*pairs].mean(), ce.mean()]
    assert np.all(np.array(metrics) <= [.15,.10,.08,.08,.18]), (seed, latent, metrics)
    assert np.all(out['predicted_strain'][a-ntr,-1] > out['predicted_strain'][b-ntr,-1])
    slices = [(lab[ntr:] == j) & (stratum[ntr:] == s) for j in range(4) for s in range(2)]
    return metrics + [max(pe[m].mean() for m in slices if m.any()), max(ce[m].mean() for m in slices if m.any()), pe.mean(axis=0).max()]

for latent in (False, True):
    scores = np.array([fixture(seed, latent) for seed in range(24)])
    print('24-seed worst', 'latent controls' if latent else 'exact contract', dict(zip(('features', 'relative_weights', 'trajectory', 'twins', 'crossing', 'cell_trajectory', 'cell_crossing', 'horizon_trajectory'), scores.max(axis=0))))
# Re-crossing after a fall must not replace the first crossing. Cover equality,
# nonuniform interpolation, initial exceedance and censoring separately.
p = module['_first_crossing'](np.array([0., 0., 1., 0., 0.]), np.array([[2.,0.,2.],[0.,1.,2.],[0.,0.,0.],[.1,.2,.3],[.2,1.,.1]]), np.ones(5), np.array([.5,2.,8.]))
np.testing.assert_allclose(p, [.25,2.,0.,8.,2.])
print('Isolated entrypoint, output schema, and first-passage branches passed.')
print('Public assets staged:', (root / 'public_eq03_datasets.pkl').exists(), (root / 'validate_public_schema.py').exists())
PY
```
- **Container Exit Code**: `1` (`898 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
24-seed worst exact contract {'features': 2.990149123585032e-16, 'relative_weights': 3.2660464542541562e-15, 'trajectory': 2.165981927545627e-15, 'twins': 2.335226919244112e-15, 'crossing': 5.429250798938412e-15, 'cell_trajectory': 2.864771911755984e-15, 'cell_crossing': 9.769962616701378e-15, 'horizon_trajectory': 3.066991105526995e-15}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e82bf5c0e4c5fd37 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e82bf5c0 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 102, in <module>
  File "<stdin>", line 96, in fixture
AssertionError: (0, True, [0.2530441581848766, 0.0190076896836713, 0.0408377196526695, 0.04118085220141448, 0.14868091061846603])
```

### Turn 5 — `execute_in_remote_docker` (`55111 ms` LLM)
- **Rationale**: The prior test stopped before checking first-crossing branches. Complete those checks and stress the unchanged artifact with small imbalanced cohorts, background noise, and coupon-clustered spikes. A coefficient or forecast failure against the contract-defined reference would warrant revisiting the robust fit; an interpolation mismatch would instead localize to crossing logic.
- **Alternatives Weighed**: `Change the prescribed coupon calibration to reduce the latent-feature error` (rejected: That test compared against unobservable generator features rather than the contract-defined pooled alignment; the candidate matched the independent alignment calculation.)

```bash
set -eu
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
module = runpy.run_path(str(root / 'viscoelastic_hereditary_forecast.py'))
solve = module['solve_viscoelastic_hereditary_forecast']
np.testing.assert_allclose(module['_first_crossing'](np.array([0.,0.,1.,0.,0.]), np.array([[2.,0.,2.],[0.,1.,2.],[0.,0.,0.],[.1,.2,.3],[.2,1.,.1]]), np.ones(5), np.array([.5,2.,8.])), [.25,2.,0.,8.,2.])

def crossing(initial, forecast, threshold, times):
    out = []
    for y0, curve, crit in zip(initial, forecast, threshold):
        t, y, hit = 0., y0, times[-1]
        if y >= crit:
            hit = 0.
        else:
            for nt, ny in zip(times, curve):
                if ny >= crit:
                    hit = t + (nt-t)*(crit-y)/(ny-y)
                    break
                t, y = nt, ny
        out.append(hit)
    return np.array(out)

def trial(seed, ntr, noise, burst):
    rng = np.random.default_rng(seed)
    nte, pairs = 96, 32
    n = ntr + nte
    taus = np.array([1.,4.5])
    dt = .25
    times = np.r_[.04, np.sort(rng.uniform(.15,7.9,18)),8.]
    theta = np.array([.38,.64,.115,.024,-.032,.019,.028])
    lab = rng.choice(4, n, p=[.49,.24,.17,.10])
    s = rng.binomial(1,np.array([.14,.68,.42,.75])[lab])
    offset = np.array([-4.,2.,.8,6.])[lab,None]
    gain = np.array([.54,2.8,.93,1.7])[lab,None]
    ref = offset + gain*rng.normal(size=(n,4))
    ref[rng.random(ref.shape)<.08] = np.nan
    raw = offset + gain*(rng.normal(size=(n,4)) + .7*s[:,None])
    raw[rng.random(raw.shape)<.05] = np.nan
    y0 = rng.uniform(-.2,.4,n)
    hold = rng.uniform(.7,1.9,n)
    rate = rng.normal(size=n)
    base = rng.uniform(.05,1.6,(n,1))
    hist = base + rng.normal(0,.21,(n,40)) + rng.uniform(-.45,.45,(n,1))*np.linspace(-1,1,40)
    a = np.arange(ntr,ntr+pairs)
    b = a + pairs
    for arr in (lab,s,raw,y0,hold,rate):
        arr[b] = arr[a]
    hist[b] = hist[a] + rng.uniform(.65,1.25,(pairs,1))
    x = raw.copy()
    for j in np.unique(lab):
        mask = lab == j
        columns = [v[np.isfinite(v)] for v in ref[mask].T]
        x[mask] = (raw[mask]-np.median([v.mean() for v in columns if v.size]))/np.median([v.std(ddof=0) for v in columns if v.size])
    for col in x.T:
        num = den = 0.
        for j,k in set(zip(lab,s)):
            v = col[(lab==j)&(s==k)]
            keep = v[np.isfinite(v)]
            num += keep.sum()/v.size
            den += keep.size/v.size
        col[np.isnan(col)] = num/den
    decay = np.exp(-dt/taus)
    memory = hist @ ((1.-decay)*decay**np.arange(39,-1,-1)[:,None])
    clean = y0[:,None] + ((hold[:,None]-memory)*theta[:2]) @ (1.-np.exp(-times/taus[:,None])) + (theta[2]+x@theta[3:])[:,None]*times
    obs = clean[:ntr] + rng.normal(0.,noise,(ntr,len(times)))
    spikes = rng.random(obs.shape)<.09 if not burst else np.roll(np.broadcast_to(rng.random((ntr,1))<.09,obs.shape),1,axis=0)
    obs[spikes] += np.clip(.5*rng.standard_cauchy(spikes.sum()),-.95,.95)
    cols = dict(lab_id=lab,stratum=s,ref_coupons=ref,coupon_features_raw=raw,past_stress_history=hist,stress_hold_0=hold,strain_0=y0,instant_rate_0=rate)
    tr = {k:v[:ntr] for k,v in cols.items()}
    te = {k:v[ntr:] for k,v in cols.items()}
    tr['strain_curves_obs'] = obs
    init = y0[ntr:]
    truth = clean[ntr:]
    crit = init + rng.uniform(.03,1.35,nte)*np.maximum(.03,truth.max(axis=1)-init)
    crit[::19] = init[::19]
    te['critical_strain'] = crit
    out = solve(dict(train=tr,test=te,history_dt=dt,memory_taus=taus,eval_times=times))
    assert {k:v.shape for k,v in out.items()} == dict(aligned_test_features=(nte,4),hereditary_weights=(7,),predicted_strain=(nte,len(times)),crossing_times=(nte,))
    assert all(np.isfinite(v).all() for v in out.values())
    np.testing.assert_allclose(out['aligned_test_features'],x[ntr:],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(out['crossing_times'],crossing(init,out['predicted_strain'],crit,times),atol=1e-12,rtol=1e-12)
    assert np.all(out['predicted_strain'][:pairs] > out['predicted_strain'][pairs:2*pairs])
    pe = np.abs(out['predicted_strain']-truth)
    ce = np.abs(out['crossing_times']-crossing(init,truth,crit,times))
    cells = [(te['lab_id']==j)&(te['stratum']==k) for j,k in set(zip(te['lab_id'],te['stratum']))]
    scores = [np.abs(out['aligned_test_features']-x[ntr:]).mean(),np.linalg.norm(out['hereditary_weights']-theta)/np.linalg.norm(theta),pe.mean(),pe[:2*pairs].mean(),ce.mean(),max(pe[m].mean() for m in cells),max(ce[m].mean() for m in cells),pe.mean(axis=0).max()]
    assert np.all(np.array(scores[:5]) <= [.15,.10,.08,.08,.18]),(seed,ntr,noise,burst,scores)
    return scores

names = 'features weights trajectory twins crossing cell_trajectory cell_crossing horizon_trajectory'.split()
for ntr,noise,burst in [(240,0.,False),(128,.012,False),(128,.012,True)]:
    scores = np.array([trial(seed,ntr,noise,burst) for seed in range(24)])
    print((ntr,noise,burst),dict(zip(names,scores.max(axis=0))))
print('Isolated output schema, twin separation, and first-crossing branches passed.')
public = root/'public_eq03_datasets.pkl'
validator = root/'validate_public_schema.py'
print('Public dataset and validator available:',public.is_file(),validator.is_file())
PY
```
- **Container Exit Code**: `0` (`1375 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(240, 0.0, False) {'features': 4.3245888279562955e-16, 'weights': 4.993086478028572e-15, 'trajectory': 8.724213481266693e-16, 'twins': 8.196541318936102e-16, 'crossing': 3.3746154019335486e-15, 'cell_trajectory': 1.429412144204889e-15, 'cell_crossing': 1.2434497875801753e-14, 'horizon_trajectory': 1.5151363959760762e-15}
(128, 0.012, False) {'features': 3.492847648843093e-16, 'weights': 0.009011169413879333, 'trajectory': 0.0007923789231574581, 'twins': 0.0008327718573459448, 'crossing': 0.003980479933479512, 'cell_trajectory': 0.001425740281859549, 'cell_crossing': 0.01932619593709095, 'horizon_trajectory': 0.0011468148587194516}
(128, 0.012, True) {'features': 3.492847648843093e-16, 'weights': 0.007634706997537628, 'trajectory': 0.0007747879603948768, 'twins': 0.0008123128121361141, 'crossing': 0.004545209576299755, 'cell_trajectory': 0.001354127070266415, 'cell_crossing': 0.01471687645454356, 'horizon_trajectory': 0.0013793912518245601}
Isolated output schema, twin separation, and first-crossing branches passed.
Public dataset and validator available: False False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e82bf5c0e4c5fd37 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e82bf5c0 (directory).
```

### Turn 6 — `submit_completed_task` (`13557 ms` LLM)
- **Rationale**: Isolated verification of the saved artifact passed the output schema, independent alignment and crossing checks, and twin separation. Worst synthetic relative coefficient error was 0.00902, twin strain MAE was 0.000833, and overall crossing MAE was 0.00455, all below their contract limits.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4547'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_hereditary_forecast.py`. All 72 synthetic stress trials met the specified tolerances. The public dataset and schema validator were not mounted, so public-data validation was not possible.`

### Turn 7 — `execute_in_remote_docker` (`42527 ms` LLM)
- **Rationale**: The public assets were absent on the previous turn. This focused isolated audit tests whether reversing the loading order while preserving its histogram, endpoints, and present observables separates twins. It also checks whether calibration transfers to a lab with no labeled curves in the fit. A twin-ordering failure would point to the memory basis; a lab-specific error would point to calibration or its transfer assumptions.
- **Alternatives Weighed**: `Add rational drift terms, relay modes, or classification diagnostics` (rejected: The specified equation is linear in seven coefficients after the exact stress-minus-memory transformation. There is no relay, classification label, or drift operator in the contract.)

```bash
set -eu
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
from pathlib import Path
import runpy
import numpy as np

root = Path('/holdout_task')
print('Public dataset and validator available:', (root / 'public_eq03_datasets.pkl').is_file(), (root / 'validate_public_schema.py').is_file())
solve = runpy.run_path(str(root / 'viscoelastic_hereditary_forecast.py'))['solve_viscoelastic_hereditary_forecast']
rng = np.random.default_rng(313)
lab = np.repeat(np.arange(4), 160)
n = lab.size
stratum = (rng.random(n) < np.array([.12,.25,.76,.89])[lab]).astype(int)
x = rng.normal(size=(n,4)) + 0.65*stratum[:,None]
ref = np.vstack([np.tile(np.array([[-1.,1.,-1.,1.],[1.,-1.,1.,-1.]]), (80,1)) for _ in range(4)])
offset = np.array([-7.,-2.,3.,8.])[lab,None]
gain = np.array([.55,2.6,.8,1.9])[lab,None]
raw = offset + gain*x
ref = offset + gain*ref
hold = rng.uniform(.8,2.1,n)
initial = rng.uniform(-.1,.4,n)
hist = rng.uniform(0.,1.7,(n,1)) + rng.normal(0.,.24,(n,40))
times = np.array([.05,.27,.8,1.4,2.3,3.7,5.1,7.2,8.])
taus = np.array([1.,4.5])
theta = np.array([.4,.66,.11,.028,-.033,.019,.023])
errors = []
for held_lab in range(4):
    tr = lab != held_lab
    te = ~tr
    ids = np.flatnonzero(te)
    a,b = ids[:40],ids[40:80]
    xx,rr,ss,hh,yy,h = (v.copy() for v in (x,raw,stratum,hold,initial,hist))
    for arr in (xx,rr,ss,hh,yy):
        arr[b] = arr[a]
    # Identical stress histograms and endpoints, but high stress is older in a.
    h[a,1:-1] = np.sort(h[a,1:-1],axis=1)[:,::-1]
    h[b] = h[a,::-1]
    h[a,0] = h[a,-1] = h[b,0] = h[b,-1] = .5
    d = np.exp(-.25/taus)
    z = h @ ((1-d)*d**np.arange(39,-1,-1)[:,None])
    truth = yy[:,None] + ((hh[:,None]-z)*theta[:2]) @ (1-np.exp(-times/taus[:,None])) + (theta[2]+xx@theta[3:])[:,None]*times
    obs = truth[tr] + rng.normal(0,.012,truth[tr].shape)
    spikes = rng.random(obs.shape) < .09
    obs[spikes] += np.clip(.5*rng.standard_cauchy(spikes.sum()),-.95,.95)
    cols = dict(lab_id=lab,stratum=ss,coupon_features_raw=rr,ref_coupons=ref,stress_hold_0=hh,strain_0=yy,instant_rate_0=np.zeros(n),past_stress_history=h)
    train,test = ({k:v[mask] for k,v in cols.items()} for mask in (tr,te))
    train['strain_curves_obs'] = obs
    test['critical_strain'] = yy[te] + .4*np.maximum(.05,truth[te].max(axis=1)-yy[te])
    out = solve(dict(train=train,test=test,eval_times=times,history_dt=.25,memory_taus=taus))
    np.testing.assert_allclose(out['aligned_test_features'], xx[te], atol=1e-12, rtol=1e-12)
    pred = out['predicted_strain']
    assert (pred[:40] > pred[40:80]).all(), 'Chronology reversal did not separate twins'
    err = np.abs(pred-truth[te])
    coeff = np.linalg.norm(out['hereditary_weights']-theta)/np.linalg.norm(theta)
    errors.append((coeff,err[:80].mean(),*(err[ss[te]==s].mean() for s in (0,1))))
worst = np.max(errors,axis=0)
print('Synthetic leave-one-lab-out worst: relative coefficients, reversed-history twins MAE, stratum 0 MAE, stratum 1 MAE:', worst)
assert np.all(worst <= [.10,.08,.08,.08])
PY
```
- **Container Exit Code**: `0` (`658 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public dataset and validator available: False False
Synthetic leave-one-lab-out worst: relative coefficients, reversed-history twins MAE, stratum 0 MAE, stratum 1 MAE: [0.00263416 0.00036321 0.00035268 0.00036819]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e82bf5c0e4c5fd37 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e82bf5c0 (directory).
```

### Turn 8 — `submit_completed_task` (`15051 ms` LLM)
- **Rationale**: The saved artifact passed isolated execution, output shape checks, independent calibration and crossing checks, and held-out-lab tests that changed history order while preserving the present and stress histogram. No tested slice exposed a contract failure that warrants a model change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4547'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All 72 synthetic stress trials met the specified tolerances, and all four synthetic leave-one-lab-out folds distinguished twins with reversed loading histories. The public dataset and schema validator were not mounted, so public-data validation remains pending.`

### Turn 9 — `submit_completed_task` (`18571 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution, schema checks, independent alignment and crossing checks, and chronology-sensitive held-out-lab tests. No tested slice warrants a model change, and the public inputs required for a real-measurement audit are unavailable.
- **Alternatives Weighed**: `Add rational, drift, or relay terms or run a within-batch classification audit` (rejected: The contract specifies a seven-coefficient regression after the exact Prony recurrence. It has no relay states or classification labels, and the applicable lab calibration was tested on held-out synthetic labs.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All 72 synthetic stress trials met the specified tolerances, and all four synthetic leave-one-lab-out folds distinguished twins with reversed loading histories. The public dataset and schema validator were not mounted, so validation on real measurements remains pending.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_hereditary_forecast.py": "FILE:4547"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_aligned_feat_mae": 0.11361,
    "max_hereditary_w_rel_l2": 0.05883,
    "max_strain_mae": 0.04337,
    "max_twin_same_present_mae": 0.03994,
    "max_crossing_time_mae": 0.09606,
    "student_elapsed_sec": 0.0765,
    "instance_diagnostics": [
      {
        "seed": 99421,
        "feat_mae": 0.10318,
        "w_rel_l2": 0.05883,
        "strain_mae": 0.04337,
        "twin_mae": 0.03994,
        "cross_mae": 0.09606
      },
      {
        "seed": 99422,
        "feat_mae": 0.1073,
        "w_rel_l2": 0.02218,
        "strain_mae": 0.03437,
        "twin_mae": 0.03595,
        "cross_mae": 0.03364
      },
      {
        "seed": 99423,
        "feat_mae": 0.11361,
        "w_rel_l2": 0.00892,
        "strain_mae": 0.04309,
        "twin_mae": 0.03834,
        "cross_mae": 0.08658
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05883, \"max_strain_mae\": 0.04337, \"max_twin_same_present_mae\": 0.03994, \"max_crossing_time_mae\": 0.09606, \"student_elapsed_sec\": 0.0765, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05883, \"strain_mae\": 0.04337, \"twin_mae\": 0.03994, \"cross_mae\": 0.09606}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02218, \"strain_mae\": 0.03437, \"twin_mae\": 0.03595, \"cross_mae\": 0.03364}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00892, \"strain_mae\": 0.04309, \"twin_mae\": 0.03834, \"cross_mae\": 0.08658}]}\n__HEGEL_VERIFIER_AUTH_10688e03036399fe1c3ac5cf29e79dfa__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05883, \"max_strain_mae\": 0.04337, \"max_twin_same_present_mae\": 0.03994, \"max_crossing_time_mae\": 0.09606, \"student_elapsed_sec\": 0.0765, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05883, \"strain_mae\": 0.04337, \"twin_mae\": 0.03994, \"cross_mae\": 0.09606}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02218, \"strain_mae\": 0.03437, \"twin_mae\": 0.03595, \"cross_mae\": 0.03364}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00892, \"strain_mae\": 0.04309, \"twin_mae\": 0.03834, \"cross_mae\": 0.08658}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5321.92,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
