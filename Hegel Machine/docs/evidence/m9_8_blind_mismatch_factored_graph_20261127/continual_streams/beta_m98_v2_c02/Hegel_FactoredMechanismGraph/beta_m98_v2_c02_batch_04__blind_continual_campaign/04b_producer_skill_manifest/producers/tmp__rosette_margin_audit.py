import pickle
import runpy
import sys
import numpy as np

root = '/holdout_task'
sys.path.insert(0, root)
from upstream_transducer import _fit, _forward, _inverse, calibrate_upstream

items = pickle.load(open(root + '/public_batch.pkl', 'rb'))
print('Public studies:', len(items))
for item in items:
    st = item['study']
    c = st['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']
    # Flag gross spikes independently of every candidate; keep them in training.
    clean = np.linalg.norm(y - x, axis=1) < 8
    print('Sweep rows:', len(x), 'gross spike rows:', np.flatnonzero(~clean).tolist())
    print('Observed cohort/relay metadata:', [k for k in st if any(t in k.lower() for t in ('batch_id', 'cohort', 'relay', 'meter'))])
    n = len(x)
    cols = np.arange(4)
    neighbor = (cols + 1) % 4
    adjacent = x * x[:, neighbor]
    cubes = x**3
    base = [np.column_stack((np.ones(n), x, adjacent[:, j], cubes[:, j])) for j in cols]
    # Pairwise sums/differences cubed add mixed cubic terms absent from the compact grammar.
    mixed = [np.column_stack((base[j], (x[:,j] + x[:,neighbor[j]])**3, (x[:,j] - x[:,neighbor[j]])**3)) for j in cols]
    # Rational corrections test remaining smooth saturation along local and differential coordinates.
    rational = [np.column_stack((base[j], x[:,j]/(1 + .5*x[:,j]**2), x[:,j]/(1 + .5*(x[:,j] - x[:,neighbor[j]])**2))) for j in cols]
    partitions = [('radius', np.linalg.norm(x, axis=1)), ('row-order', np.arange(n))] + [(f'channel-{j}', x[:,j]) for j in cols]
    worst = np.zeros(4)
    witness = ['']*4
    for name, coordinate in partitions:
        for k, h in enumerate(np.array_split(np.argsort(coordinate), 5)):
            t = np.ones(n, bool); t[h] = False
            h = h[clean[h]]
            if not len(h):
                continue
            coef = _fit(x[t], y[t])
            compact_pred = _forward(x[h], coef)
            # Select inliers from training residuals, never from the held-out fit.
            train_res = np.linalg.norm(_forward(x[t], coef) - y[t], axis=1)
            keep = np.flatnonzero(t)[train_res <= max(1e-6, 8*np.median(train_res))]
            preds = [compact_pred]
            for basis in (mixed, rational):
                preds.append(np.column_stack([a[h] @ np.linalg.lstsq(a[keep], y[keep,j], rcond=None)[0] for j,a in enumerate(basis)]))
            errs = [np.sqrt(np.mean((p-y[h])**2, axis=0)).max() for p in preds]
            errs.append(np.sqrt(np.mean((_inverse(y[h], coef)-x[h])**2, axis=0)).max())
            for j, err in enumerate(errs):
                if err > worst[j]:
                    worst[j], witness[j] = err, f'{name}/{k}'
    print('Blocked worst channel RMSE (compact, mixed cubic, rational, compact inverse):', dict(zip(('compact','mixed','rational','inverse'),zip(worst.tolist(),witness))))
    assert worst[3] < 1e-4
    warm = calibrate_upstream(st)
    coeff = _fit(x,y)
    raw = st['y_obs_raw']; obs = warm['y_obs_raw'].reshape(-1,4)
    forward_error = np.max(np.abs(_forward(obs,coeff).reshape(raw.shape)-raw))
    print('Live observation closure:', forward_error)
    assert forward_error < 1e-9
sys.argv = [root + '/validate_public_schema.py', root + '/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
