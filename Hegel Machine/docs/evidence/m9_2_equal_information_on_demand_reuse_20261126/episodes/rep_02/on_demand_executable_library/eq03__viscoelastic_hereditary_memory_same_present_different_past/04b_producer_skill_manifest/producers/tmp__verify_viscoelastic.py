import importlib.util
import numpy as np

path = '/holdout_task/viscoelastic_hereditary_forecast.py'
spec = importlib.util.spec_from_file_location('forecast', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Independent scalar check for initial contact, nonmonotone first passage,
# exact contact at a knot, and the no-contact convention.
np.testing.assert_allclose(m._crossings(np.array([1., 3., 8.]), np.array([1., 0., 0., 0., 0.]), np.array([[0, 0, 0], [2, -2, 4], [0, 1, 0], [-1, 0, .9], [-2, -1, 2]]), np.ones(5)), [0, .5, 3, 8, 3 + 10/3])

T = np.array([.12, .35, .7, 1.15, 1.8, 2.6, 3.8, 5.1, 6.5, 8.])
taus = np.array([1., 4.5])
dt = .25
truth = np.array([.53, .39, .029, .021, -.016, .013, .012])

def simulate(seed, severe_imbalance):
    rng = np.random.default_rng(seed)
    counts = [115, 27, 36, 13, 24, 8, 18, 5] if severe_imbalance else [35]*8
    train_cells = np.repeat(np.arange(8), counts)
    rng.shuffle(train_cells)
    ntr = len(train_cells)
    cells = np.r_[train_cells, np.repeat(np.arange(8), 14)]
    labs, strata = cells//2, cells%2
    n = len(cells)
    gains, offsets = np.array([.52, 1.8, .9, 2.2])[labs], np.array([-.9, .8, .2, -1.4])[labs]
    refs = offsets[:, None] + gains[:, None]*rng.normal(size=(n, 4))
    raw = offsets[:, None] + gains[:, None]*(rng.normal(size=(n, 4)) + .8*strata[:, None])
    refs[rng.random(refs.shape) < .08] = np.nan
    raw[rng.random(raw.shape) < .05] = np.nan
    if severe_imbalance:
        refs[labs == 3, 0] = np.nan
    history = rng.normal(.45, .65, size=(n, 40)) + rng.normal(0, .45, (n, 1))
    hold = rng.uniform(.3, 1.3, n)
    initial = rng.uniform(-.1, .25, n)
    rate = rng.uniform(0, 1, n)
    twin_left = ntr + np.arange(0, n-ntr, 2)
    twin_right = twin_left + 1
    for shared in (raw, hold, initial, rate):
        shared[twin_right] = shared[twin_left]
    history[twin_left, -15:] -= 1.3
    history[twin_right, -15:] += 1.3

    align = raw.copy()
    for lab in range(4):
        rows = labs == lab
        controls = [c[np.isfinite(c)] for c in refs[rows].T]
        means = [c.mean() for c in controls if len(c)]
        stds = [np.sqrt(np.mean((c-c.mean())**2)) for c in controls if len(c)]
        align[rows] = (raw[rows] - np.median(means))/np.median(stds)
    weights = np.array([1/np.count_nonzero(cells == c) for c in cells])
    for column in align.T:
        available = np.isfinite(column)
        column[~available] = np.average(column[available], weights=weights[available])

    # Closed-form convolution integral, independent of the solver recurrence.
    boundaries = dt*np.arange(40, -1, -1)
    mode = []
    for tau in taus:
        integrated = np.exp(-boundaries[1:]/tau) - np.exp(-boundaries[:-1]/tau)
        mode.append((hold - history @ integrated)[:, None]*(1-np.exp(-T/tau)))
    clean = initial[:, None] + truth[0]*mode[0] + truth[1]*mode[1] + (truth[2] + align@truth[3:])[:, None]*T
    noisy = clean[:ntr] + rng.normal(0, .008 if severe_imbalance else .003, (ntr, len(T)))
    spikes = rng.random(noisy.shape) < .09
    noisy[spikes] += np.clip(.95*rng.standard_cauchy(spikes.sum()), -.95, .95)
    fields = dict(lab_id=labs, stratum=strata, coupon_features_raw=raw, ref_coupons=refs, past_stress_history=history, stress_hold_0=hold, strain_0=initial, instant_rate_0=rate)
    train = {k:v[:ntr] for k,v in fields.items()}
    train['strain_curves_obs'] = noisy
    test = {k:v[ntr:] for k,v in fields.items()}
    crit = rng.uniform(.12, .85, n-ntr) + test['strain_0']
    crit[::13] = test['strain_0'][::13]
    test['critical_strain'] = crit
    result = m.solve_viscoelastic_hereditary_forecast(dict(train=train, test=test, history_dt=dt, memory_taus=taus, eval_times=T))
    exact_times = []
    for start, values, level in zip(test['strain_0'], clean[ntr:], crit):
        t_a, y_a = 0., start
        crossing = T[-1]
        if start >= level:
            crossing = 0.
        else:
            for t_b, y_b in zip(T, values):
                if y_b >= level:
                    crossing = t_a + (t_b-t_a)*(level-y_a)/(y_b-y_a)
                    break
                t_a, y_a = t_b, y_b
        exact_times.append(crossing)
    assert set(result) == {'aligned_test_features', 'hereditary_weights', 'predicted_strain', 'crossing_times'}
    for key, shape in zip(('aligned_test_features', 'hereditary_weights', 'predicted_strain', 'crossing_times'), ((n-ntr, 4), (7,), (n-ntr, len(T)), (n-ntr,))):
        assert result[key].shape == shape and np.isfinite(result[key]).all(), key
    pred = result['predicted_strain']
    e = np.abs(pred - clean[ntr:])
    h = np.abs(result['crossing_times'] - exact_times)
    worst_cell = max(e[cells[ntr:] == j].mean() for j in range(8))
    worst_cross_cell = max(h[cells[ntr:] == j].mean() for j in range(8))
    twin_delta_error = np.abs((pred[::2] - pred[1::2]) - (clean[twin_left] - clean[twin_right])).mean()
    np.testing.assert_allclose(result['aligned_test_features'], align[ntr:], atol=1e-12)
    return [np.abs(result['aligned_test_features'] - align[ntr:]).mean(), np.linalg.norm(result['hereditary_weights'] - truth)/np.linalg.norm(truth), e.mean(), e.mean(axis=0).max(), worst_cell, twin_delta_error, h.mean(), worst_cross_cell]

worst = np.max([simulate(i, bool(i % 2)) for i in range(24)], axis=0)
for name, value in zip(('alignment_mae', 'theta_rel_l2', 'trajectory_twin_mae', 'worst_horizon_trajectory_mae', 'worst_cell_trajectory_mae', 'twin_contrast_mae', 'crossing_mae', 'worst_cell_crossing_mae'), worst):
    print(f'{name}={value:.9g}')
assert np.all(worst <= [.15, .10, .08, .08, .08, .08, .18, .18]), worst
print('Isolated import and stress checks passed (24 synthetic datasets).')
