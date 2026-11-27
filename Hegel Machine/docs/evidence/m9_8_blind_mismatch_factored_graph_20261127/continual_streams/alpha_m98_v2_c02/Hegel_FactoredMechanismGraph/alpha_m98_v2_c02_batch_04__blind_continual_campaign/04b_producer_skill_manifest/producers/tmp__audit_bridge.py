import sys, pickle, itertools
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _invert, _design, _POWERS, calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    d = item['dataset']
    c = d['calibration_load_cells']
    x, y = c['true_channel_loads'], c['bridge_raw']
    b, clean = _fit(x, y)
    norm = np.linalg.norm(x, axis=1)
    mixed = np.count_nonzero(x, axis=1) > 1
    high = norm >= np.quantile(norm[clean], .75)
    # Pairwise sums and differences squared/cubed are already in the cubic span.
    # Test a bridge-wide rational denominator as a challenger to that span.
    directions = list(np.eye(4))
    for i, j in itertools.combinations(range(4), 2):
        for sign in (-1, 1):
            directions.append(np.eye(4)[i] + sign * np.eye(4)[j])
    for name, held in [('upper-load quartile', high), ('upper-load mixed', high & mixed)]:
        bt, good = _fit(x[~held], y[~held])
        xt, yt = x[~held][good], y[~held][good]
        xv, yv = x[held & clean], y[held & clean]
        at, av = _design(xt, _POWERS), _design(xv, _POWERS)
        er = np.abs(_invert(yv, bt) - xv)
        base = np.sqrt(np.mean((av @ bt - yv)**2))
        best_train = np.inf
        for q in directions:
            u, v = (xt @ q)**2, (xv @ q)**2
            for alpha in (1e-4, 1e-3, 1e-2, .1, .5):
                pred = np.empty_like(yv)
                rss = 0.
                for k in range(4):
                    support = bt[:, k] != 0
                    z = at[:, support] / (1 + alpha * u[:, None])
                    coef = np.linalg.lstsq(z, yt[:, k], rcond=None)[0]
                    rss += np.sum((z @ coef - yt[:, k])**2)
                    pred[:, k] = av[:, support] @ coef / (1 + alpha * v)
                if rss < best_train:
                    best_train = rss
                    challenger = np.sqrt(np.mean((pred - yv)**2))
                    chosen = (q.tolist(), alpha)
        print(name, 'n', len(xv), 'max held-out inverse error', er.max(), 'forward RMSE cubic/rational', base, challenger, 'rational denominator', chosen)
    out = calibrate_upstream(d)
    trial = np.concatenate([out[s][k].reshape(-1, 4) for s in ('train', 'test', 'chamber_verification_trials') if out.get(s) is not None for k in ('past_channel_raw', 'hold_channel_raw') if k in out[s]])
    j = np.stack([_design(trial, np.maximum(_POWERS - e, 0)) @ (_POWERS[:, i, None] * b) for i, e in enumerate(np.eye(4, dtype=int))], axis=-1)
    print('Minimum singular value at public trial inverses:', np.linalg.svd(j, compute_uv=False)[:, -1].min())
    for k in ('lab_id', 'stratum'):
        if k in d['test']:
            print('Test', k, 'counts:', np.unique(d['test'][k], return_counts=True))
