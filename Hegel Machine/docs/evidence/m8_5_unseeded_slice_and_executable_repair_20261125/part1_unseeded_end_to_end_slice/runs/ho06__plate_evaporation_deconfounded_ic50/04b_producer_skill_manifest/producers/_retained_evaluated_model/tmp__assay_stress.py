import copy
import pickle
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, '/holdout_task')
import assay_ic50 as assay

with open('/holdout_task/public_screen.pkl', 'rb') as stream:
    public = pickle.load(stream)
KEYS = ('log10_ic50_uM', 'hill_slope')
LIMITS = np.array([0.09, 0.18])


def synthetic(shape, seed):
    rng = np.random.default_rng(seed)
    plates = copy.deepcopy(public['plates'])
    truth = {c: dict(zip(KEYS, (v[KEYS[0]] + rng.uniform(-0.19, 0.19),
                                rng.uniform(0.7, 1.65)))) for c, v in public['gt'].items()}
    for plate in plates:
        background = rng.uniform(90, 310)
        scale = rng.uniform(1700, 2800)
        strength = rng.uniform(0.26, 0.42)
        decay = rng.uniform(0.85, 3.4)
        tilt_r, tilt_c = rng.uniform(-0.045, 0.045, 2)
        for w in plate['wells']:
            r, c = w['row'], w['col']
            dr, dc = min(r, 15-r), min(c, 23-c)
            if shape == 'separable':
                perimeter = 0.5 * (np.exp(-dr/decay) + np.exp(-dc/(1.25*decay)))
            elif shape == 'gaussian':
                perimeter = np.exp(-(min(dr, dc)/decay)**2)
            else:
                perimeter = np.exp(-min(dr, dc)/decay)
            gain = 1 + strength * perimeter + tilt_r * (r-7.5)/7.5 + tilt_c * (c-11.5)/11.5
            kind = w['well_type']
            if kind == 'COMPOUND':
                h = truth[w['compound_id']]
                v = expit(np.log(10) * h[KEYS[1]] * (h[KEYS[0]] - w['log10_conc_uM']))
            else:
                v = float(kind == 'DMSO')
            if rng.random() < 0.03:
                v *= rng.uniform(0, 0.35)
            w['fluorescence'] = background + scale * (gain * v + rng.normal(0, 0.009))
    return plates, truth


def errors(plates, truth):
    pred = assay.estimate_compound_hill_parameters(plates)
    return np.array([[abs(pred[c][k] - row[k]) for k in KEYS] for c, row in truth.items()])


if __name__ == '__main__':
    for shape in ['exponential', 'separable', 'gaussian']:
        results = np.array([errors(*synthetic(shape, seed)) for seed in range(16)])
        max_by_seed = results.max(axis=1)
        print(shape, 'max', max_by_seed.max(axis=0).round(5), 'seeds over limits', (max_by_seed > LIMITS).sum(axis=0), 'worst seeds', max_by_seed.argmax(axis=0), flush=True)
