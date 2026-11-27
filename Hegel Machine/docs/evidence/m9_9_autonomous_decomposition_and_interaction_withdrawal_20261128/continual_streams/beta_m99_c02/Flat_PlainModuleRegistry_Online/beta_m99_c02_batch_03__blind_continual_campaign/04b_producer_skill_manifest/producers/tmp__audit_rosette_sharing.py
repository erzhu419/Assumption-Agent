import pickle
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _forward, _inverse

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
b = c['single_bank_sweeps']
axis = np.count_nonzero(x, axis=1) == 1
high = np.max(abs(x), axis=1) >= 1

def expand(p):
    return np.column_stack((p[:4], p[4:8], p[4:8, None] * p[None, 8:]))

def pooled(train, base):
    xr, yr = x[train], y[train]
    solo = np.zeros((len(b['bank_id']), 4))
    bank = np.asarray(b['bank_id'])
    idx = np.arange(len(bank))
    solo[idx, bank] = b['ref_signal']
    seed = np.r_[base[:, 0], base[:, 1], np.mean(base[:, 2:] / base[:, 1, None], axis=0)]
    # Estimate noise scale on training data only. Mask individual readings
    # so an EMI spike need not discard the other three channel readings.
    rr = abs(_forward(xr, base) - yr)
    sr = abs(_forward(solo, base)[idx, bank] - b['raw_readout'])
    gate = max(8 * np.median(np.r_[rr.ravel(), sr]), 1e-9)
    rm, sm = rr < gate, sr < gate
    def residual(p):
        co = expand(p)
        return np.r_[(_forward(xr, co) - yr)[rm], (_forward(solo, co)[idx, bank] - b['raw_readout'])[sm]]
    return expand(least_squares(residual, seed, gtol=1e-12).x)

folds = [('axis', axis), ('mixed-low', ~axis & ~high), ('mixed-high', ~axis & high)]
for i, ids in enumerate(np.array_split(np.random.default_rng(120).permutation(len(x)), 5)):
    folds.append((f'random-{i}', np.isin(np.arange(len(x)), ids)))
worst = np.zeros(2)
for name, test in folds:
    train = ~test
    base = _fit(dict(c, ref_rosette=x[train], raw_rosette=y[train]))
    co = pooled(train, base)
    clean = test & (np.max(abs(_forward(x, base) - y), axis=1) < 1e-3)
    errors = np.array([np.max(abs(_inverse(y[clean], p) - x[clean])) for p in (base, co)])
    worst = np.maximum(worst, errors)
    print(name, 'n', clean.sum(), 'inverse max independent/shared', *errors)
print('worst independent/shared', *worst)
print('full-fit shared normalized coefficients', (co[:, 2:]/co[:, 1, None]).mean(axis=0))
