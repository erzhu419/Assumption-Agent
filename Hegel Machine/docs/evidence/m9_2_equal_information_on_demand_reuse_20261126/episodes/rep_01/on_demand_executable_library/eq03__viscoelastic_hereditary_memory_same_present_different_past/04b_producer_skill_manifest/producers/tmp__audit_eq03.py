import importlib.util
import numpy as np

path = '/holdout_task/viscoelastic_hereditary_forecast.py'
spec = importlib.util.spec_from_file_location('forecast', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def crossing(t, y, z):
    if y[0] >= z:
        return 0.
    for j in range(1, len(t)):
        if y[j] >= z:
            return t[j-1] + (t[j]-t[j-1]) * (z-y[j-1])/(y[j]-y[j-1])
    return t[-1]

np.testing.assert_allclose(m._first_crossing(np.array([1., 3., 8.]), np.array([1., 0., 0., 0., 0.]), np.array([[0., 0., 0.], [0., 0.3, 0.4], [1., 0., 2.], [0.2, 1., 0.], [0., 0., 2.]]), np.array([1., 0.6, 0.5, 1., 1.])), [0., 8., 0.5, 3., 5.5])
worst = np.zeros(7)
audit = []
limits = np.array([.15, .10, .08, .08, .08, .08, .18])
labels = ['alignment', 'relative coefficient L2', 'trajectory', 'twin', 'lab/stratum/horizon', 'paired contrast', 'crossing']
for seed in range(24):
    rng = np.random.default_rng(seed)
    nt, nv = 192, 96
    n = nt + nv
    t = np.r_[0.07, np.sort(rng.uniform(.12, 7.8, 31)), 8.]
    taus = np.array([1., 4.5])
    lab = rng.choice(4, n, p=[.52, .26, .15, .07])
    stratum = (rng.random(n) < np.array([.16, .8, .4, .7])[lab]).astype(int)
    gains = np.array([.43, 1.1, 1.9, 2.7])[lab, None]
    offsets = np.array([-1.4, .55, 2.6, -.38])[lab, None]
    ref = rng.normal(size=(n, 4))*gains + offsets
    raw = (rng.normal(size=(n, 4)) + .65*stratum[:, None])*gains + offsets
    ref[rng.random(ref.shape) < .08] = np.nan
    raw[rng.random(raw.shape) < .05] = np.nan
    h = rng.uniform(-.4, 1.2, (n, 1)) + rng.normal(0, .45, (n, 40))
    hold, y0, rate = rng.uniform(.55, 1.2, n), rng.uniform(-.12, .12, n), rng.normal(size=n)
    left = np.arange(nt, nt+40, 2)
    right = left + 1
    for arr in (lab, stratum, raw, hold, y0, rate):
        arr[right] = arr[left]
    h[right] = h[left]
    h[left[:10], 0] += 1.3
    h[right[:10], -1] += 1.3
    h[left[10:], -18:] -= 1.15
    h[right[10:], -9:] += .9

    # Independent scalar calibration and equal-cell weighted mean.
    x = np.empty_like(raw)
    for g in range(4):
        mask = lab == g
        cols = [col[np.isfinite(col)] for col in ref[mask].T]
        cols = [col for col in cols if len(col) >= 2]
        x[mask] = (raw[mask] - np.median([col.mean() for col in cols])) / np.median([col.std() for col in cols])
    cell_sizes = {(g,s): np.sum((lab==g)&(stratum==s)) for g,s in zip(lab,stratum)}
    inv = np.array([1/cell_sizes[(g,s)] for g,s in zip(lab,stratum)])
    for j in range(4):
        good = np.isfinite(x[:,j])
        x[~good,j] = np.average(x[good,j], weights=inv[good])

    # Evaluate the causal integral via exponential edge differences, not recurrence.
    age = .25*np.arange(40, -1, -1)
    kernel = np.diff(np.exp(-age[:, None]/taus), axis=0)
    theta = np.array([.56, .83, .045, .039, -.031, .019, .041])
    clean = y0[:, None] + ((hold[:, None] - h @ kernel)*theta[:2]) @ (1 - np.exp(-t[:, None]/taus)).T
    clean += (theta[2] + x @ theta[3:])[:, None] * t
    noise = rng.normal(0, [.005, .012, .025][seed % 3], (nt, len(t)))
    spike = rng.random(noise.shape) < .09
    noise[spike] += np.clip(.38*rng.standard_cauchy(spike.sum()), -.95, .95)
    crit = np.maximum(y0 + .04, clean[:, -1] + rng.uniform(-.2, .2, n))
    crit[nt:nt+8] = y0[nt:nt+8]
    crit[nt+40:nt+48] = clean[nt+40:nt+48].max(axis=1) + 1.
    fields = dict(lab_id=lab, stratum=stratum, ref_coupons=ref, coupon_features_raw=raw, past_stress_history=h, stress_hold_0=hold, strain_0=y0, instant_rate_0=rate, critical_strain=crit)
    tr = {k:v[:nt].copy() for k,v in fields.items()}
    te = {k:v[nt:].copy() for k,v in fields.items()}
    tr['strain_curves_obs'] = clean[:nt] + noise
    d = dict(train=tr, test=te, history_dt=.25, memory_taus=taus, eval_times=t)
    out = m.solve_viscoelastic_hereditary_forecast(d)
    shapes = dict(aligned_test_features=(nv,4), hereditary_weights=(7,), predicted_strain=(nv,len(t)), crossing_times=(nv,))
    assert set(out) == set(shapes)
    for k, shape in shapes.items():
        assert out[k].shape == shape and np.isfinite(out[k]).all(), k
    np.testing.assert_allclose(out['aligned_test_features'], x[nt:], atol=2e-14)
    pred = out['predicted_strain']
    kt = np.r_[0., t]
    ct = np.array([crossing(kt, np.r_[y, yy], c) for y,yy,c in zip(y0[nt:], clean[nt:], crit[nt:])])
    np.testing.assert_allclose(out['crossing_times'], [crossing(kt, np.r_[y, yy], c) for y,yy,c in zip(y0[nt:], pred, crit[nt:])], atol=2e-14)
    short = (clean[right] - clean[left])[:10]
    strong = (clean[right] - clean[left])[10:]
    # Any present-only model must return the same curve for each twin.
    present_only_bound = np.abs(strong).mean() / 2
    paired_err = np.abs((pred[right-nt] - pred[left-nt])[10:] - strong).mean()
    # Reversing the past preserves every history sum but reverses its recency.
    reversed_kernel_gap = h[left[:10]] @ (kernel - kernel[::-1])
    assert np.max(np.abs(reversed_kernel_gap)) > .01
    np.testing.assert_allclose(h[left[:10]].sum(axis=1), h[right[:10]].sum(axis=1))
    audit.append((present_only_bound, paired_err, np.abs(short).mean()))
    err = np.abs(pred-clean[nt:])
    slice_err = max(err[rows][:, cols].mean() for g,s in set(zip(lab[nt:],stratum[nt:])) for rows in [(lab[nt:]==g)&(stratum[nt:]==s)] for cols in np.array_split(np.arange(t.size), 3))
    contrast = np.abs((pred[right-nt]-pred[left-nt]) - (clean[right]-clean[left]))
    values = np.array([np.abs(out['aligned_test_features']-x[nt:]).mean(), np.linalg.norm(out['hereditary_weights']-theta)/np.linalg.norm(theta), err.mean(), err[:40].mean(), slice_err, max(contrast[:10].mean(),contrast[10:].mean()), np.abs(out['crossing_times']-ct).mean()])
    assert np.all(values <= limits), (seed, dict(zip(labels, values)))
    worst = np.maximum(worst, values)
print('Worst over 24 synthetic split scenarios:')
for label, value, limit in zip(labels, worst, limits):
    print(f'{label}: {value:.6g} (limit {limit})')

audit = np.array(audit)
assert audit[:, 0].min() > .08, audit
assert audit[:, 1].max() < .08, audit
print(f'Strong twins: minimum present-only MAE bound = {audit[:,0].min():.6g}; maximum hereditary contrast MAE = {audit[:,1].max():.6g}')
print(f'Equal-history-sum twins: minimum mean trajectory contrast = {audit[:,2].min():.6g}')
