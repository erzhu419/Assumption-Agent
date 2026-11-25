import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('assay_ic50', '/holdout_task/assay_ic50.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_screen.pkl', 'rb') as f:
    screen = pickle.load(f)
plates, truth = screen['plates'], screen['gt']
keys = ('log10_ic50_uM', 'hill_slope')
limit = np.array([0.09, 0.18])

def score(plates, expected=truth):
    pred = m.estimate_compound_hill_parameters(plates)
    assert pred.keys() == expected.keys()
    assert all(set(v) == set(keys) and all(type(x) is float and np.isfinite(x) for x in v.values()) for v in pred.values())
    errors = np.array([[abs(pred[c][k] - expected[c][k]) for k in keys] for c in expected])
    return errors.max(axis=0), [list(expected)[i] for i in errors.argmax(axis=0)]

full, ids = score(plates)
print('Public full', full, ids)
assert np.all(full <= limit)
for size in (3, 2, 1):
    slices = [(tuple(p['plate_id'] for p in subset), *score(subset)) for subset in itertools.combinations(plates, size)]
    print('Worst subsets of', size, [max(slices, key=lambda s: s[1][j]) for j in range(2)])
for p in plates:
    a = np.asarray([(w['row'], w['col'], w['fluorescence']) for w in p['wells'] if w['well_type'] == 'DMSO'])
    b = np.median([w['fluorescence'] for w in p['wells'] if w['well_type'] == 'POS_KILL'])
    length, coeff = m._fit_gain(a[:,0], a[:,1], a[:,2]-b)
    res = []
    for j in range(len(a)):
        sel = np.arange(len(a)) != j
        t, co = m._fit_gain(a[sel,0], a[sel,1], a[sel,2]-b)
        res.append(((m._spatial_basis(a[j:j+1,0], a[j:j+1,1], t) @ co)[0] - (a[j,2]-b))/np.median(a[:,2]-b))
    print(p['plate_id'], 'decay length', round(length,3), 'held-control RMSE / max', round(np.sqrt(np.mean(np.square(res))),4), round(max(np.abs(res)),4))
