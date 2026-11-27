import itertools
import pickle
import sys
import numpy as np

root = '/holdout_task'
sys.path.insert(0, root)
from upstream_transducer import _features, _fit_forward, _invert, _robust

with open(root + '/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']

def gross_clean(raw, reference):
    center = np.median(reference, axis=0)
    scale = np.median(np.linalg.norm(reference - center, axis=1))
    return np.linalg.norm(raw - center, axis=1) <= 6 * scale

axis = np.count_nonzero(x, axis=1) == 1
extreme = ~axis & (np.max(abs(x), axis=1) >= np.quantile(np.max(abs(x[~axis]), axis=1), .75))
linear = [()] + [(j,) for j in range(4)]
quadratic = linear + list(itertools.combinations_with_replacement(range(4), 2))
for name, train, target in [('axial series withheld', ~axis, axis), ('outer quartile mixed withheld', ~extreme, extreme)]:
    test = target & gross_clean(y, y[train])
    results = {}
    for model, terms in [('affine', linear), ('quadratic', quadratic), ('delivered cubic', None)]:
        if terms is None:
            terms, b = _fit_forward(x[train], y[train])
        else:
            a = _features(x[train], terms)
            b = _robust(a, y[train], np.ones((a.shape[1], 4), dtype=bool), gross_clean(y[train], y[train]))
        bridge = _features(x[test], terms) @ b - y[test]
        results[model] = float(np.sqrt(np.mean(bridge**2, axis=0)).max())
    inverse_error = _invert(y[test], terms, b) - x[test]
    inverse_rmse = float(np.sqrt(np.mean(inverse_error**2, axis=0)).max())
    print(name, 'n=', int(test.sum()), 'held-out worst-channel bridge RMSE:', results, 'delivered inverse RMSE:', inverse_rmse)
    assert results['delivered cubic'] * 10 < min(results['affine'], results['quadratic'])
    assert np.max(abs(inverse_error)) < 0.001
print('ng_t05: matched, spike-screened outer splits directly validate the delivered structural replacement.')
