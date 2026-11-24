import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from assay_ic50 import _coordinates, _gain, _fit_hill, estimate_compound_hill_parameters

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    screen = pickle.load(f)
fields = ('log10_ic50_uM', 'hill_slope')
truth = screen['gt']
records = {m: {} for m in ('scalar', 'exponential', 'rational', 'interaction')}
control_errors = {m: {} for m in records}
edge_fractions = {}


def gain(p, coords, model):
    if model == 'exponential':
        return _gain(p, coords)
    d, r, c = coords
    if model == 'scalar':
        return np.ones_like(d) * p[0]
    s = d / np.exp(p[4])
    if model == 'rational':
        return p[0] + p[1] / (1 + s*s) + p[2]*r + p[3]*c
    return (p[0] + p[1]*np.exp(-s)) * (1 + p[2]*r + p[3]*c)


def calibrate(x, y, model):
    if model == 'scalar':
        return [np.median(y)]
    residual = lambda p: gain(p, x, model) - y
    bounds = ([.1, 0., -1., -1., np.log(.25)], [3., 2., 1., 1., np.log(12.)])
    p = least_squares(residual, [.9, .4, 0., 0., np.log(2.)], bounds=bounds, loss='soft_l1', f_scale=.008).x
    return least_squares(residual, p, bounds=bounds, loss='cauchy', f_scale=.015).x

for plate in screen['plates']:
    pid = plate['plate_id']
    wells = plate['wells']
    coords = _coordinates(wells)
    y = np.array([w['fluorescence'] for w in wells])
    kind = np.array([w['well_type'] for w in wells])
    b = np.median(y[kind == 'POS_KILL'])
    dm = np.flatnonzero(kind == 'DMSO')
    tr = np.flatnonzero(kind == 'COMPOUND')
    for j in tr:
        edge_fractions.setdefault(wells[j]['compound_id'], []).append(coords[0][j] <= 1)
    for m in records:
        for j in dm:
            fit_ids = dm[dm != j]
            scale = np.median(y[fit_ids]) - b
            p = calibrate(tuple(a[fit_ids] for a in coords), (y[fit_ids]-b)/scale, m)
            pred = scale*gain(p, tuple(a[[j]] for a in coords), m)[0]
            control_errors[m].setdefault((pid, 'border' if coords[0][j] == 0 else 'interior'), []).append((pred-(y[j]-b))/(y[j]-b))
        scale = np.median(y[dm])-b
        p = calibrate(tuple(a[dm] for a in coords), (y[dm]-b)/scale, m)
        v = (y[tr]-b)/scale/gain(p, tuple(a[tr] for a in coords), m)
        for j, val in zip(tr, v):
            w = wells[j]
            records[m].setdefault(w['compound_id'], []).append((w['log10_conc_uM'], val))

def maxima(out, names):
    return np.round(np.max([[abs(out[n][k]-truth[n][k]) for k in fields] for n in names], axis=0), 5).tolist()

rank = sorted(truth, key=lambda n: np.mean(edge_fractions[n]))
print('Mean near-edge occupancy by compound half:', [round(float(np.mean([np.mean(edge_fractions[n]) for n in names])), 3) for names in (rank[:6], rank[6:])])
exact = estimate_compound_hill_parameters(screen['plates'])
for m, by_name in records.items():
    out = {n: _fit_hill(*np.asarray(v).T) for n,v in by_name.items()}
    if m == 'exponential':
        np.testing.assert_allclose([[out[n][k] for k in fields] for n in rank], [[exact[n][k] for k in fields] for n in rank], rtol=0, atol=2e-7)
    slices = {s: 100*np.sqrt(np.mean(np.square(v))) for s,v in control_errors[m].items()}
    print(m, 'worst control RMS [%] by region:', [max((round(v, 3), s[0]) for s,v in slices.items() if s[1] == region) for region in ('border','interior')], 'worst Hill errors by lower/higher near-edge occupancy:', maxima(out, rank[:6]), maxima(out, rank[6:]))
