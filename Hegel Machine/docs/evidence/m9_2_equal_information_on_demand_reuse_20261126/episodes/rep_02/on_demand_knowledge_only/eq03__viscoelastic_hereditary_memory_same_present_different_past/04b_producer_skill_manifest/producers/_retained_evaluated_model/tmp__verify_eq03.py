import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('forecast', '/holdout_task/viscoelastic_hereditary_forecast.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Check first crossings independently of fit quality: immediate, repeated,
# exact knot, censored, and interpolation on a later segment.
y0 = np.array([1., 0., 0., 0., 0.])
y = np.array([[0., 0., 0.], [2., 0., 2.], [0., 1., 0.], [.3, .5, .9], [.2, .6, 1.4]])
np.testing.assert_allclose(mod._crossing_times(np.array([1., 2., 4.]), y0, y, np.ones(5)), [0., .5, 2., 4., 3.])

def crossing(t, initial, curves, crit):
    result = []
    for start, curve, cap in zip(initial, curves, crit):
        hit = t[-1]
        if start >= cap:
            result.append(0.)
            continue
        prev_t, prev_y = 0., start
        for next_t, next_y in zip(t, curve):
            if next_y >= cap:
                hit = prev_t + (next_t - prev_t) * (cap - prev_y) / (next_y - prev_y)
                break
            prev_t, prev_y = next_t, next_y
        result.append(hit)
    return np.array(result)

worst = np.zeros(7)
min_twin_gap = np.inf
for seed in range(12):
    rng = np.random.default_rng(seed)
    n_train, n_test = 240, 98
    n = n_train + n_test
    cell = rng.choice(8, n, p=[.34, .04, .08, .07, .07, .15, .20, .05])
    cell[:8] = np.arange(8)
    lab, stratum = cell // 2, cell % 2
    gain = np.array([.55, 1.7, 2.4, .8])[lab]
    offset = np.array([-3., 1.3, 2., -1.7])[lab]
    raw = offset[:, None] + gain[:, None] * (rng.normal(size=(n, 4)) + 1.8 * stratum[:, None])
    refs = offset[:, None] + gain[:, None] * rng.normal(size=(n, 4))
    raw[rng.random(raw.shape) < .05] = np.nan
    refs[rng.random(refs.shape) < .08] = np.nan
    hold = rng.uniform(.8, 2.8, n)
    initial = rng.normal(.1, .04, n)
    rate = rng.normal(size=n)
    history = rng.uniform(.1, 3.3, (n, 40)) + rng.uniform(-.2, 1.2, (n, 1))
    twins = np.arange(n_train, n_train + 20, 2)
    for arr in (cell, lab, stratum, raw, hold, initial, rate):
        arr[twins + 1] = arr[twins]
    history[twins, -12:] += .9
    history[twins + 1] = history[twins]
    history[twins + 1, -12:] -= 1.7

    aligned = raw.copy()
    for group in range(4):
        ix = np.flatnonzero(lab == group)
        col = [[v for v in refs[ix, j] if np.isfinite(v)] for j in range(4)]
        mu = np.median([sum(a) / len(a) for a in col if a])
        sigma = np.median([np.std(a) for a in col if len(a) > 1])
        aligned[ix] = (raw[ix] - mu) / sigma
    for j in range(4):
        keep = np.isfinite(aligned[:, j])
        values = []
        weights = []
        for c in range(8):
            rows = cell == c
            values.extend(aligned[rows & keep, j])
            weights.extend([1 / rows.sum()] * np.count_nonzero(rows & keep))
        aligned[~keep, j] = np.average(values, weights=weights)

    dt = .25
    taus = np.array([1., 4.5])
    times = np.r_[0.07, np.sort(rng.uniform(.13, 7.9, size=37)), 8.]
    theta = np.array([.17, .31, .055, .025, -.014, .008, .018])
    # Closed-form convolution and sum of separately evaluated processes.
    z = np.column_stack([history @ (np.exp(-np.arange(39, -1, -1) * dt / tau) * (1 - np.exp(-dt / tau))) for tau in taus])
    clean = initial[:, None] + (theta[2] + aligned @ theta[3:])[:, None] * times
    for j, tau in enumerate(taus):
        clean += theta[j] * (hold - z[:, j])[:, None] * (1 - np.exp(-times / tau))
    obs = clean[:n_train].copy()
    if seed % 4:
        obs += rng.normal(0, [.0, .002, .008, .014][seed % 4], obs.shape)
    spike = rng.random(obs.shape) < .09
    obs[spike] += np.clip(rng.standard_cauchy(size=spike.sum()) * .65, -.95, .95)
    crit = initial[n_train:] + rng.uniform(.04, .53, n_test)
    crit[:2] = initial[n_train:n_train+2] - .01
    crit[-2:] = clean[-2:].max(axis=1) + np.abs(initial[-2:]) + 1.
    test = slice(n_train, None)
    def split(s):
        return dict(lab_id=lab[s], stratum=stratum[s], coupon_features_raw=raw[s], ref_coupons=refs[s], past_stress_history=history[s], stress_hold_0=hold[s], strain_0=initial[s], instant_rate_0=rate[s])
    d = dict(train=dict(split(slice(None, n_train)), strain_curves_obs=obs), test=dict(split(test), critical_strain=crit), eval_times=times, memory_taus=taus, history_dt=dt)
    o = mod.solve_viscoelastic_hereditary_forecast(d)
    shapes = dict(aligned_test_features=(n_test, 4), hereditary_weights=(7,), predicted_strain=(n_test, len(times)), crossing_times=(n_test,))
    assert set(o) == set(shapes)
    for k, shape in shapes.items():
        assert o[k].shape == shape and np.isfinite(o[k]).all(), (k, o[k].shape)
    np.testing.assert_allclose(o['aligned_test_features'], aligned[test], atol=1e-12, rtol=0)
    np.testing.assert_allclose(o['crossing_times'], crossing(times, initial[test], o['predicted_strain'], crit), atol=1e-12, rtol=0)
    e = np.abs(o['predicted_strain'] - clean[test])
    h = np.abs(o['crossing_times'] - crossing(times, initial[test], clean[test], crit))
    gap = clean[twins + 1] - clean[twins]
    gap_pred = o['predicted_strain'][twins + 1 - n_train] - o['predicted_strain'][twins - n_train]
    min_twin_gap = min(min_twin_gap, float(gap[:, -1].min()))
    metrics = [np.linalg.norm(o['hereditary_weights'] - theta) / np.linalg.norm(theta), e.mean(), e[:20].mean(), h.mean(), max(e[cell[test] == c].mean() for c in np.unique(cell[test])), max(e[:, j].mean() for j in range(len(times))), np.abs(gap - gap_pred).mean()]
    worst = np.maximum(worst, metrics)
    assert np.all(np.array(metrics) < [.10, .08, .08, .18, .08, .08, .08]), (seed, metrics)
print('12 synthetic held-out test sets; alignment and crossing algorithm agree with independent oracles to 1e-12')
print('Worst errors [rel theta, strain MAE, twin MAE, crossing MAE, cell strain MAE, horizon strain MAE, twin gap MAE]:', np.array2string(worst, precision=8))
print('Minimum terminal twin strain gap:', min_twin_gap)
