import pickle
import sys
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from assay_ic50 import _coordinates, _calibrate_plate, _fit_hill, estimate_compound_hill_parameters

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    public = pickle.load(f)
plates, truth = public['plates'], public['gt']
fields = ('log10_ic50_uM', 'hill_slope')

def errors(est):
    assert est.keys() == truth.keys()
    assert all(set(v) == set(fields) and all(type(x) is float and np.isfinite(x) for x in v.values()) for v in est.values())
    return np.max([[abs(est[name][f] - gt[f]) for f in fields] for name, gt in truth.items()], axis=0)

records = {}
rms = {'constant': [], 'spatial': []}
for p in plates:
    wells = p['wells']
    pos = np.array([w['well_type'] == 'POS_KILL' for w in wells])
    d, r, c = _coordinates(wells)
    X = np.column_stack([np.ones(len(d)), np.exp(-d/2), r, c])
    y = np.array([w['fluorescence'] for w in wells])
    A, z = X[pos], y[pos]
    b = np.median(z)
    sd = max(1., 1.4826*np.median(abs(z-b)))
    def fit(M, v, scale):
        return least_squares(lambda q: M@q-v, [np.median(v), 0., 0., 0.], loss='soft_l1', f_scale=scale).x
    miss = []
    for j in range(len(z)):
        keep = np.arange(len(z)) != j
        m = np.median(z[keep])
        scale = max(1., 1.4826*np.median(abs(z[keep]-m)))
        miss.append([z[j]-m, z[j]-A[j]@fit(A[keep], z[keep], scale)])
    miss = np.array(miss)
    for s, model in enumerate(rms):
        rms[model].append([np.sqrt(np.mean(miss[region,s]**2)) for region in (d[pos] == 0, d[pos] > 0)])
    spatial_b = X @ fit(A, z, sd)
    corrected = [dict(w, fluorescence=val) for w, val in zip(wells, y-spatial_b+b)]
    for w, v in zip(wells, _calibrate_plate(corrected)):
        if w['well_type'] == 'COMPOUND':
            records.setdefault(w['compound_id'], []).append((w['log10_conc_uM'], v))

for model, slices in rms.items():
    print(model, 'worst-plate held-out background RMS, border/interior:', np.round(np.max(slices, axis=0), 3))
print('Spatial-background challenger Hill errors:', np.round(errors({n: _fit_hill(*np.array(pairs).T) for n, pairs in records.items()}), 5))
full = errors(estimate_compound_hill_parameters(plates))
pairwise = np.max([errors(estimate_compound_hill_parameters(list(ps))) for ps in combinations(plates,2)], axis=0)
print('Exact artifact full-screen and worst two-plate errors:', np.round(full, 5), np.round(pairwise, 5))
assert np.all(full <= [.09, .18]) and np.all(pairwise <= [.09, .18])
