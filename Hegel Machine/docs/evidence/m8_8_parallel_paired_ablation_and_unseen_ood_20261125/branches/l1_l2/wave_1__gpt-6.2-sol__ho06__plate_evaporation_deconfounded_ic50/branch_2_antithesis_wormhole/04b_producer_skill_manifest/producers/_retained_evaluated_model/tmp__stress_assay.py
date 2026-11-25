import pickle
from collections import defaultdict
import numpy as np
from scipy.special import expit
from assay_ic50 import estimate_compound_hill_parameters

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    public = pickle.load(f)
layout = public['plates']
keys = ('log10_ic50_uM', 'hill_slope')

def simulate(seed, regime):
    rng = np.random.default_rng(seed)
    ids = list(public['gt'])
    truth = {k: (v, rng.uniform(.75, 1.5)) for k, v in zip(ids, np.sort(rng.uniform(-2.3, 1.8, len(ids))))}
    plates = []
    common_decay = rng.uniform(1.3, 2.8)
    for plate in layout:
        background, unit = rng.uniform(100, 300), rng.uniform(1650, 3100)
        a, tx, ty = rng.uniform(.20, .42), rng.uniform(-.075, .075), rng.uniform(-.075, .075)
        decay = rng.uniform(1.25, 3.0) if regime == 'variable_decay' else common_decay
        wells = []
        for w in plate['wells']:
            r, c = w['row'], w['col']
            x, y = (r - 7.5)/7.5, (c - 11.5)/11.5
            edge = a*np.exp(-min(r, 15-r, c, 23-c)/decay)
            tilt = tx*x + ty*y
            gain = (1 + edge)*(1 + tilt) if regime == 'product' else 1 + edge + tilt
            kind = w['well_type']
            if kind == 'COMPOUND':
                ic50, slope = truth[w['compound_id']]
                v = expit(np.log(10)*slope*(ic50-w['log10_conc_uM']))
            else:
                v = float(kind == 'DMSO')
            signal = unit*(gain*v + rng.normal(0, .014 if kind != 'POS_KILL' else .008)) + background
            if rng.random() < .03:
                signal += unit*rng.choice([-1, 1])*rng.uniform(.25, .65)
            wells.append(dict(w, fluorescence=signal))
        plates.append(dict(plate, wells=wells))
    return plates, truth

def score(plates, truth):
    result = estimate_compound_hill_parameters(plates)
    assert set(result) == set(truth)
    assert all(set(v) == set(keys) and all(type(x) is float and np.isfinite(x) for x in v.values()) and v['hill_slope'] > 0 for v in result.values())
    return np.array([max(abs(result[k][p]-truth[k][j]) for k in truth) for j,p in enumerate(keys)])

if __name__ == '__main__':
    for regime in ('additive', 'variable_decay', 'product'):
        errors = np.array([score(*simulate(seed, regime)) for seed in range(40)])
        print(regime, 'max', np.max(errors,axis=0).round(4), 'p95', np.quantile(errors,.95,axis=0).round(4), 'worst seeds', np.argmax(errors,axis=0), 'failing screens', np.sum(np.any(errors > [.09,.18],axis=1)))
