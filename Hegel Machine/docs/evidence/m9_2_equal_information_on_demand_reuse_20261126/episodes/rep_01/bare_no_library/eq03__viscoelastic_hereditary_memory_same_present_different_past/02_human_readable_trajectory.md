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
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `160,665` (`152,578` / `46,720` / `8,087`) |
| **Total Episode Wall Time (s)** | `260.43s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16756 ms` LLM)
- **Rationale**: Inspect the public schema and summarize missingness in place. Unexpected dataset nesting or nonfinite values outside the coupon fields would change how the validation or fitting code handles the public data.

```bash
cd /holdout_task && ls -lh && sed -n '1,220p' validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_eq03_datasets.pkl','rb') as f:
    data=pickle.load(f)
def inspect(x, prefix='', depth=0):
    if isinstance(x,dict) and depth < 4:
        for k,v in list(x.items())[:18]: inspect(v,prefix+'/'+str(k),depth+1)
    elif isinstance(x,(list,tuple)) and len(x) and isinstance(x[0],dict) and depth < 3:
        print(prefix,type(x).__name__,len(x)); inspect(x[0],prefix+'/0',depth+1)
    elif hasattr(x,'shape'):
        print(prefix,x.shape,x.dtype, 'nonfinite='+str(np.count_nonzero(~np.isfinite(x))) if x.dtype.kind in 'fc' else '')
    else:
        print(prefix,repr(x)[:100])
inspect(data)
PY
```
- **Container Exit Code**: `1` (`368 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: cd: /holdout_task: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`14498 ms` LLM)
- **Rationale**: The requested directory was absent. Check the known task roots for public data and the validator without transferring either; if unavailable, create the specified output directory and use synthetic contract tests.

```bash
pwd; ls -ld /pretrain_task /holdout_task /app /workspace /results 2>/dev/null; find /pretrain_task /holdout_task /app /workspace /results -maxdepth 4 -type f \( -name '*eq03*' -o -name 'validate_public_schema.py' -o -name 'AGENTS.md' -o -name '*viscoelastic*' \) -printf '%p %s bytes\n' 2>/dev/null; if [ -d /pretrain_task ]; then ls -la /pretrain_task; fi; if [ -d /holdout_task ]; then ls -la /holdout_task; fi
```
- **Container Exit Code**: `0` (`357 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/pretrain_task
drwxr-xr-x 2 root root 4096 Nov 23 09:31 /pretrain_task
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 04:50 ..
```

### Turn 3 — `execute_in_remote_docker` (`47319 ms` LLM)
- **Rationale**: Implement the exact causal basis and cell-balanced calibration before testing. Independent synthetic labels will distinguish errors in calibration, hereditary integration, robust parameter recovery, and threshold interpolation.
- **Alternatives Weighed**: `Fit a model using only instantaneous observables` (rejected: The specified twins have identical instantaneous inputs but different loading histories, so a present-only model cannot distinguish them.)

```bash
set -e
mkdir -p /holdout_task
cat > /holdout_task/viscoelastic_hereditary_forecast.py <<'PY'
"""Coupon calibration, hereditary creep forecasting, and first passage times."""

import numpy as np


def _align(train, test):
    def pooled(name, dtype):
        return np.concatenate((np.asarray(train[name], dtype=dtype),
                               np.asarray(test[name], dtype=dtype)), axis=0)

    lab = pooled('lab_id', int)
    stratum = pooled('stratum', int)
    ref = pooled('ref_coupons', float)
    aligned = pooled('coupon_features_raw', float)
    for lab_id in np.unique(lab):
        rows = lab == lab_id
        controls = ref[rows]
        controls = controls[:, np.any(np.isfinite(controls), axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - offset) / gain

    _, cell, count = np.unique(2 * lab + stratum, return_inverse=True,
                               return_counts=True)
    weights = (lab.size / count.size) / count[cell]
    valid = np.isfinite(aligned)
    weighted_mean = ((weights @ np.where(valid, aligned, 0.0)) /
                     (weights @ valid))
    return np.where(valid, aligned, weighted_mean)


def _design(split, aligned, dt, taus, times):
    history = np.asarray(split['past_stress_history'], dtype=float)
    decay = np.exp(-dt / taus)
    uptake = -np.expm1(-dt / taus)
    memory = np.zeros((history.shape[0], taus.size))
    for stress in history.T:
        memory = decay * memory + uptake * stress[:, None]

    hold = np.asarray(split['stress_hold_0'], dtype=float)
    response = -np.expm1(-times[:, None] / taus)
    transient = (hold[:, None] - memory)[:, None, :] * response[None, :, :]
    flow = np.column_stack((np.ones(len(aligned)), aligned))
    return np.concatenate((transient, times[None, :, None] * flow[:, None, :]),
                          axis=2)


def _fit(design, observations):
    theta = np.linalg.lstsq(design, observations, rcond=None)[0]
    # Re-estimate the inlier scale from the residual MAD; the floor also
    # allows IRLS to finish on data with exactly noiseless inliers.
    floor = 1e-8 * max(1.0, np.median(np.abs(observations)))
    for _ in range(80):
        residual = observations - design @ theta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(floor, 2.385 * 1.4826 * mad)
        sqrt_weight = scale / np.hypot(scale, residual)
        update = np.linalg.lstsq(design * sqrt_weight[:, None],
                                 observations * sqrt_weight, rcond=None)[0]
        if np.linalg.norm(update - theta) <= 1e-10 * max(1.0, np.linalg.norm(update)):
            return update
        theta = update
    return theta


def _first_passage(initial, strain, threshold, times):
    crossing = np.full(initial.shape, times[-1], dtype=float)
    at_start = initial >= threshold
    crossing[at_start] = 0.0
    reached = strain >= threshold[:, None]
    rows = np.flatnonzero(~at_start & reached.any(axis=1))
    knots = np.column_stack((initial, strain))
    time_knots = np.r_[0.0, times]
    left = reached[rows].argmax(axis=1)
    y_a, y_b = knots[rows, left], knots[rows, left + 1]
    t_a, t_b = time_knots[left], time_knots[left + 1]
    crossing[rows] = t_a + (threshold[rows] - y_a) / (y_b - y_a) * (t_b - t_a)
    return crossing


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the two-mode hereditary model and forecast test creep and crossings."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    initial_train = np.asarray(train['strain_0'], dtype=float)
    initial_test = np.asarray(test['strain_0'], dtype=float)
    n_train = initial_train.size
    aligned = _align(train, test)
    train_design = _design(train, aligned[:n_train], dt, taus, times)
    observations = np.asarray(train['strain_curves_obs'], dtype=float)
    theta = _fit(train_design.reshape(-1, train_design.shape[-1]),
                 (observations - initial_train[:, None]).ravel())
    aligned_test = aligned[n_train:]
    test_design = _design(test, aligned_test, dt, taus, times)
    strain = initial_test[:, None] + test_design @ theta
    threshold = np.asarray(test['critical_strain'], dtype=float)
    return {'aligned_test_features': aligned_test,
            'hereditary_weights': theta,
            'predicted_strain': strain,
            'crossing_times': _first_passage(initial_test, strain, threshold, times)}
PY
ls -l /holdout_task/viscoelastic_hereditary_forecast.py
```
- **Container Exit Code**: `0` (`373 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_hereditary_forecast.py']`
- **Stdout (tail)**:
```text
-rw-r--r-- 1 root root 4565 Nov 26 04:51 /holdout_task/viscoelastic_hereditary_forecast.py

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=18398cc3a36a39a1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_18398cc3 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`56296 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated process. Independent closed-form memory labels and scalar crossing labels distinguish recurrence, calibration, robustness, and interpolation failures; each failure would identify a different component to fix.

```bash
set -e
cd /holdout_task
if [ -f validate_public_schema.py ]; then python3 -I /holdout_task/validate_public_schema.py; fi
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import importlib.util
import numpy as np
from pathlib import Path
p = Path('/holdout_task/viscoelastic_hereditary_forecast.py')
s = importlib.util.spec_from_file_location('forecast', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
solve = m.solve_viscoelastic_hereditary_forecast
limits = np.array([.15, .10, .08, .08, .18])
worst = np.zeros(5)
worst_lab = np.zeros(2)
worst_status = np.zeros(3)
min_twin_gap = np.inf
for seed in range(24):
    rng = np.random.default_rng(seed)
    ntr, nte = (96, 64) if seed < 12 else (224, 112)
    n = ntr + nte
    lab = rng.choice(4, n, p=[.49, .26, .16, .09])
    stratum = rng.binomial(1, .12 + .22*lab)
    lab[:8], stratum[:8] = np.repeat(np.arange(4), 2), np.tile([0, 1], 4)
    offsets, gains = np.array([-1.8, .4, 1.9, 4.1]), np.array([.55, 1.2, 1.8, 2.5])
    X = offsets[lab, None] + gains[lab, None] * (rng.normal(size=(n, 4)) + stratum[:, None]*(lab[:, None]-1))
    ref = offsets[lab, None] + gains[lab, None]*rng.normal(size=(n, 4))
    X[rng.random(X.shape) < .05] = np.nan
    ref[rng.random(ref.shape) < .08] = np.nan
    hold = rng.uniform(.55, 1.55, n)
    initial = rng.uniform(.01, .2, n)
    rate = rng.normal(size=n)
    hist = rng.uniform(0, 1.8, (n, 40)) + rng.uniform(0, 1, (n, 1))
    a = np.arange(ntr, ntr+16, 2)
    b = a+1
    for f in (lab, stratum, X, hold, initial, rate): f[b] = f[a]
    hist[a, :] = rng.uniform(0, .18, (len(a), 40))
    hist[b, :] = rng.uniform(1.65, 2.0, (len(a), 40))
    cells = [(lab == i) & (stratum == j) for i in range(4) for j in range(2)]
    aligned = X.copy()
    for i in range(4):
        mask = lab == i
        valid = [col[np.isfinite(col)] for col in ref[mask].T]
        center = sorted(v.mean() for v in valid if v.size)
        gains_est = sorted(v.std() for v in valid if v.size)
        aligned[mask] = (X[mask]-np.median(center))/np.median(gains_est)
    for col in aligned.T:
        sums = [np.nansum(col[c])/c.sum() for c in cells]
        mass = [np.isfinite(col[c]).mean() for c in cells]
        col[np.isnan(col)] = sum(sums)/sum(mass)
    dt = .25
    tau = np.array([1., 4.5])
    t = np.r_[np.sort(rng.uniform(.05, 7.8, 22 if seed < 12 else 42)), 8.]
    weights = np.array([.19, .36, .055, -.017, .024, .012, -.015]) * rng.uniform(.75, 1.25, 7)
    z = hist @ (np.exp(-np.arange(39, -1, -1)[:, None]*dt/tau) * (1-np.exp(-dt/tau)))
    clean = (initial[:, None] + ((hold[:, None]-z)*weights[:2]) @ (1-np.exp(-t[None,:]/tau[:,None]))
             + (weights[2] + aligned @ weights[3:])[:, None]*t)
    peak = np.maximum(initial, clean.max(1))
    critical = initial + rng.uniform(.05, 1.15, n)*np.maximum(.01, peak-initial)
    critical[ntr+16::7] = initial[ntr+16::7]
    critical[ntr+17::7] = peak[ntr+17::7]+.03
    y = clean[:ntr] + rng.normal(0, [0., .002, .008][seed % 3], (ntr,len(t)))
    mask = rng.random(y.shape) < .09
    y[mask] += np.clip(.30*rng.standard_cauchy(mask.sum()), -.95, .95)
    fields = dict(lab_id=lab, stratum=stratum, coupon_features_raw=X, ref_coupons=ref,
                  past_stress_history=hist, stress_hold_0=hold, strain_0=initial, instant_rate_0=rate)
    train = {k:v[:ntr] for k,v in fields.items()}
    test = {k:v[ntr:] for k,v in fields.items()}
    train['strain_curves_obs'] = y
    test['critical_strain'] = critical[ntr:]
    d = dict(train=train,test=test,eval_times=t,memory_taus=tau,history_dt=dt)
    out = solve(d)
    expected = dict(aligned_test_features=(nte,4), hereditary_weights=(7,), predicted_strain=(nte,len(t)), crossing_times=(nte,))
    assert out.keys() == expected.keys()
    for k, shape in expected.items(): assert out[k].shape == shape and np.isfinite(out[k]).all(), k
    cross, status = [], []
    for y0, curve, crit in zip(initial[ntr:], clean[ntr:], critical[ntr:]):
        if y0 >= crit: cross.append(0.); status.append(0); continue
        prev_t, prev_y = 0., y0
        for ti, yi in zip(t,curve):
            if yi >= crit:
                cross.append(prev_t+(ti-prev_t)*(crit-prev_y)/(yi-prev_y)); status.append(1); break
            prev_t, prev_y = ti, yi
        else: cross.append(t[-1]); status.append(2)
    err = np.abs(out['predicted_strain']-clean[ntr:])
    ce = np.abs(out['crossing_times']-cross)
    metrics = [np.mean(np.abs(out['aligned_test_features']-aligned[ntr:])),
        np.linalg.norm(out['hereditary_weights']-weights)/np.linalg.norm(weights),
        err.mean(), err[:16].mean(), ce.mean()]
    worst = np.maximum(worst, metrics)
    for i in range(4):
        sub = test['lab_id'] == i
        if sub.any(): worst_lab = np.maximum(worst_lab, [err[sub].mean(), ce[sub].mean()])
    for i in range(3):
        sub = np.asarray(status) == i
        if sub.any(): worst_status[i] = max(worst_status[i], ce[sub].mean())
    min_twin_gap = min(min_twin_gap, np.min(out['predicted_strain'][a-ntr,-1]-out['predicted_strain'][b-ntr,-1]))
np.testing.assert_allclose(m._first_passage(np.array([0., 0., .5, 0.]),
    np.array([[.25, .75, .1], [.2, .4, .49], [.1, .2, .3], [.5, .2, .75]]),
    np.full(4, .5), np.array([1., 3., 8.])), [2., 8., 0., 1.])
print('Public bundle mounted:', Path('public_eq03_datasets.pkl').exists())
print('Worst of 24 independent synthetic seeds [alignment MAE, coefficient rel L2, strain MAE, twin MAE, crossing MAE]:', worst)
print('Worst lab slice [strain MAE, crossing MAE]:',worst_lab)
print('Worst crossing status MAE [at start, crossed, censored]:',worst_status)
print('Minimum predicted twin gap at final knot:',min_twin_gap)
assert np.all(worst <= limits), (worst,limits)
assert min_twin_gap > .2
PY
```
- **Container Exit Code**: `0` (`827 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public bundle mounted: False
Worst of 24 independent synthetic seeds [alignment MAE, coefficient rel L2, strain MAE, twin MAE, crossing MAE]: [3.13578373e-16 8.65947208e-03 4.26310867e-04 7.00435500e-04
 5.30329308e-03]
Worst lab slice [strain MAE, crossing MAE]: [0.00076185 0.01017915]
Worst crossing status MAE [at start, crossed, censored]: [0.         0.00904057 0.        ]
Minimum predicted twin gap at final knot: 0.5957092435485727

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=18398cc3a36a39a1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_18398cc3 (directory).
```

### Turn 5 — `submit_completed_task` (`12782 ms` LLM)
- **Rationale**: The isolated verification of the unchanged artifact passed all five contract metrics, including crossing edge cases. No measured bottleneck warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4565'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All acceptance thresholds passed across 24 synthetic datasets, including identical-present twins. The public dataset and schema validator were not mounted, so public-data validation was not run.`

### Turn 6 — `execute_in_remote_docker` (`55580 ms` LLM)
- **Rationale**: Check the specific risk that the earlier stress tests did not isolate chronology from average stress or lab composition. Equal-histogram twins test the order-sensitive Prony basis, while shifted lab proportions and within-cell errors can expose calibration failures masked by pooled scores.
- **Alternatives Weighed**: `Audit rational drift, relay modes, or balanced classification accuracy` (rejected: This task specifies continuous creep and crossing times, with no relay state or class labels. Pulse-order twins and lab-by-stratum errors test the relevant structural and confounding risks.)

```bash
set -e
cd /holdout_task
if [ -f validate_public_schema.py ] && [ -f public_eq03_datasets.pkl ]; then python3 -I validate_public_schema.py; else printf 'Public dataset/validator unavailable; running a synthetic chronology and lab-confounding audit.\n'; fi
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('forecast', '/holdout_task/viscoelastic_hereditary_forecast.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

budgets = np.array([.15, .10, .08, .08, .18])
worst = np.zeros(5)
worst_cell = np.zeros(2)
min_memoryless_twin_mae = np.inf
for seed in range(12):
    rng = np.random.default_rng(390 + seed)
    lab = np.repeat(np.arange(4), 160)
    n = lab.size
    stratum = np.concatenate([rng.permutation(np.repeat([0, 1], [160-k, k])) for k in (12, 32, 128, 148)])
    # Shift the train/test lab proportions without leaving out any lab or stratum.
    test_idx = np.concatenate([rng.choice(np.flatnonzero(lab == l), k, replace=False) for l, k in enumerate((18, 34, 62, 84))])
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    pairs = test_idx[:16].reshape(-1, 2)
    a, b = pairs.T
    stratum[b] = stratum[a]
    gain = np.array([.45, 2.8, .85, 1.7])[lab, None]
    offset = np.array([5., 1.5, -2., -5.])[lab, None]
    controls = offset + gain * rng.normal(size=(n, 4))
    raw = offset + gain * (rng.normal(size=(n, 4)) + .8 * stratum[:, None])
    controls[rng.random(controls.shape) < .08] = np.nan
    raw[rng.random(raw.shape) < .05] = np.nan
    hold = rng.uniform(.8, 1.5, n)
    y0 = rng.uniform(.02, .18, n)
    rate = rng.normal(size=n)
    for field in (raw, hold, y0, rate):
        field[b] = field[a]
    history = rng.uniform(.1, 1.9, (n, 40)) + rng.uniform(-.25, .25, (n, 1))
    # Same histogram and final sample, but the order of the internal load pulses differs.
    pulse = np.r_[.9, np.full(18, 1.9), np.full(20, .1), .9]
    history[a], history[b] = pulse, pulse[::-1]
    x = raw.copy()
    for l in range(4):
        rows = lab == l
        c = controls[rows]
        x[rows] = (raw[rows] - np.median(np.nanmean(c, axis=0))) / np.median(np.nanstd(c, axis=0))
    cells = [((lab == l) & (stratum == s)) for l in range(4) for s in range(2)]
    for col in x.T:
        num = sum(np.nansum(col[cell]) / cell.sum() for cell in cells)
        den = sum(np.isfinite(col[cell]).mean() for cell in cells)
        col[np.isnan(col)] = num / den
    taus = np.array([1., 4.5])
    dt = .25
    times = np.r_[.03, .18, .36, np.linspace(.65, 8., 28)]
    theta = np.array([.23, .30, .05, -.026, .014, .009, -.012])
    lag = .25 * np.arange(39, -1, -1)[:, None]
    kernel = np.exp(-lag / taus) - np.exp(-(lag + dt) / taus)
    z = history @ kernel
    clean = y0[:, None] + ((hold[:, None] - z) * theta[:2]) @ (1 - np.exp(-times / taus[:, None])) + (theta[2] + x @ theta[3:])[:, None] * times
    observed = clean[train_idx] + rng.normal(0, .006, (len(train_idx), len(times)))
    spikes = rng.random(observed.shape) < .09
    observed[spikes] += np.clip(.35 * rng.standard_cauchy(spikes.sum()), -.95, .95)
    peak = np.maximum(y0, clean.max(axis=1))
    crit = y0 + rng.uniform(.03, .98, n) * np.maximum(.005, peak-y0)
    crit[test_idx[16::9]] = y0[test_idx[16::9]]
    crit[test_idx[17::9]] = peak[test_idx[17::9]] + .03
    fields = dict(lab_id=lab, stratum=stratum, coupon_features_raw=raw, ref_coupons=controls, past_stress_history=history, stress_hold_0=hold, strain_0=y0, instant_rate_0=rate)
    train = {key: value[train_idx] for key, value in fields.items()}
    test = {key: value[test_idx] for key, value in fields.items()}
    train['strain_curves_obs'] = observed
    test['critical_strain'] = crit[test_idx]
    out = module.solve_viscoelastic_hereditary_forecast(dict(train=train, test=test, history_dt=dt, memory_taus=taus, eval_times=times))
    truth = []
    for i in test_idx:
        if y0[i] >= crit[i]:
            truth.append(0.)
            continue
        ta, ya = 0., y0[i]
        for tb, yb in zip(times, clean[i]):
            if yb >= crit[i]:
                truth.append(ta + (tb-ta)*(crit[i]-ya)/(yb-ya))
                break
            ta, ya = tb, yb
        else:
            truth.append(times[-1])
    err = np.abs(out['predicted_strain'] - clean[test_idx])
    hazard_err = np.abs(out['crossing_times'] - truth)
    worst = np.maximum(worst, [np.abs(out['aligned_test_features'] - x[test_idx]).mean(), np.linalg.norm(out['hereditary_weights'] - theta) / np.linalg.norm(theta), err.mean(), err[:16].mean(), hazard_err.mean()])
    for cell in cells:
        sub = cell[test_idx]
        if sub.any():
            worst_cell = np.maximum(worst_cell, [err[sub].mean(), hazard_err[sub].mean()])
    min_memoryless_twin_mae = min(min_memoryless_twin_mae, .5 * np.abs(clean[a]-clean[b]).mean())
print('Worst overall, 12 seeds [feature MAE, relative coefficient L2, strain MAE, twin MAE, crossing MAE]:', worst)
print('Worst occupied lab/stratum [strain MAE, crossing MAE]:', worst_cell)
print('Best-case shared-prediction lower bound on twin MAE:', min_memoryless_twin_mae)
assert np.isfinite(worst).all() and np.all(worst < budgets), worst
assert np.all(worst_cell < budgets[[2, 4]]), worst_cell
assert min_memoryless_twin_mae > budgets[3]
PY
```
- **Container Exit Code**: `0` (`1146 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public dataset/validator unavailable; running a synthetic chronology and lab-confounding audit.
Worst overall, 12 seeds [feature MAE, relative coefficient L2, strain MAE, twin MAE, crossing MAE]: [1.22713862e-15 5.34203228e-03 1.59827655e-04 2.62851578e-04
 2.76112421e-03]
Worst occupied lab/stratum [strain MAE, crossing MAE]: [0.00026368 0.01877506]
Best-case shared-prediction lower bound on twin MAE: 0.1905824929943773

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=18398cc3a36a39a1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_18398cc3 (directory).
```

### Turn 7 — `submit_completed_task` (`14925 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated verification and the follow-up audit with reordered stress histories and shifted lab proportions. Neither audit identified a tolerance bottleneck that warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4565'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All five acceptance thresholds passed across 36 synthetic datasets. Worst twin strain MAE was 0.000700, and worst lab-by-stratum crossing-time MAE was 0.0188. Public-data validation was not run because the dataset and validator were not mounted.`

### Turn 8 — `submit_completed_task` (`15669 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution and both synthetic stress audits. Another synthetic rerun would not resolve the missing public-data check.
- **Alternatives Weighed**: `Add rational terms or relay-state dynamics` (rejected: The contract specifies an exact two-mode Prony basis with linear creep and no relay state. Chronology and lab-by-stratum audits found no tolerance failure.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_hereditary_forecast.py`. All five acceptance thresholds passed across 36 synthetic datasets. The public dataset and schema validator were not mounted, so validation on public measurements remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_hereditary_forecast.py": "FILE:4565"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_aligned_feat_mae": 0.11361,
    "max_hereditary_w_rel_l2": 0.06128,
    "max_strain_mae": 0.04367,
    "max_twin_same_present_mae": 0.04024,
    "max_crossing_time_mae": 0.09645,
    "student_elapsed_sec": 0.0864,
    "instance_diagnostics": [
      {
        "seed": 99421,
        "feat_mae": 0.10318,
        "w_rel_l2": 0.06128,
        "strain_mae": 0.04367,
        "twin_mae": 0.04024,
        "cross_mae": 0.09645
      },
      {
        "seed": 99422,
        "feat_mae": 0.1073,
        "w_rel_l2": 0.02068,
        "strain_mae": 0.03363,
        "twin_mae": 0.03489,
        "cross_mae": 0.03317
      },
      {
        "seed": 99423,
        "feat_mae": 0.11361,
        "w_rel_l2": 0.00822,
        "strain_mae": 0.04318,
        "twin_mae": 0.03826,
        "cross_mae": 0.08609
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.06128, \"max_strain_mae\": 0.04367, \"max_twin_same_present_mae\": 0.04024, \"max_crossing_time_mae\": 0.09645, \"student_elapsed_sec\": 0.0864, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.06128, \"strain_mae\": 0.04367, \"twin_mae\": 0.04024, \"cross_mae\": 0.09645}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02068, \"strain_mae\": 0.03363, \"twin_mae\": 0.03489, \"cross_mae\": 0.03317}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00822, \"strain_mae\": 0.04318, \"twin_mae\": 0.03826, \"cross_mae\": 0.08609}]}\n__HEGEL_VERIFIER_AUTH_4055ef971e71afef2eb81c49657c36ac__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.06128, \"max_strain_mae\": 0.04367, \"max_twin_same_present_mae\": 0.04024, \"max_crossing_time_mae\": 0.09645, \"student_elapsed_sec\": 0.0864, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.06128, \"strain_mae\": 0.04367, \"twin_mae\": 0.04024, \"cross_mae\": 0.09645}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02068, \"strain_mae\": 0.03363, \"twin_mae\": 0.03489, \"cross_mae\": 0.03317}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00822, \"strain_mae\": 0.04318, \"twin_mae\": 0.03826, \"cross_mae\": 0.08609}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5754.15,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
