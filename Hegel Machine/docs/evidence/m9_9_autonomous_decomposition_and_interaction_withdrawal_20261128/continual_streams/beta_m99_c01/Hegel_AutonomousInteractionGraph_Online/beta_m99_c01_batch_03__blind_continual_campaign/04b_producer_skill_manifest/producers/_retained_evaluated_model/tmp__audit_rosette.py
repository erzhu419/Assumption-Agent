import pickle, sys
from itertools import combinations_with_replacement
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_rosette, _invert_rosette
from batch_solver import solve_batch

study = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
sweep = study['sensor_calibration_sweeps']
x, y = sweep['ref_rosette'], sweep['raw_rosette']
j = np.arange(4)
k = (j + 1) % 4
amplitude = np.max(abs(y), axis=1)
# The three EMI rows are separated from the clean rows by a >8-unit gap.
clean = amplitude < 8
terms = [p for degree in (2, 3) for p in combinations_with_replacement(range(4), degree)]
full = np.column_stack([np.prod(x[:, p], axis=1) for p in terms])

def design(ch, family, a):
    base = np.column_stack((np.ones(len(x)), x, x[:, ch]**3, x[:, ch]*x[:, k[ch]]))
    if family == 'mixed':
        extra = [i for i, p in enumerate(terms) if p not in {(ch,)*3, tuple(sorted((ch, k[ch])))}]
        return np.column_stack((base, full[:, extra]))
    if family in ('self', 'sum', 'difference'):
        q = x[:, ch] if family == 'self' else x[:, ch] + (1 if family == 'sum' else -1)*x[:, k[ch]]
        return base/(1 + a*q[:, None]**2)
    return base

def alternative(train, test, family, a=0):
    tr = train & clean
    return np.column_stack([d[test] @ np.linalg.lstsq(d[tr], y[tr, ch], rcond=None)[0]
                            for ch in j for d in [design(ch, family, a)]])

folds = [('shuffled', np.random.default_rng(1203).permutation(len(x)) % 5, range(5)),
         ('high amplitude', np.max(abs(x), axis=1) > 1.45, [1]),
         ('axis', np.arange(len(x)) < 16, [1])]
worst = 0
for name, fold, keys in folds:
    for key in keys:
        train = fold != key
        test = ~train & clean
        coef = _fit_rosette(x[train], y[train])
        forward = coef[:, 0] + x[test] @ coef[:, 1:5].T + coef[:, 5]*x[test]**3 + coef[:, 6]*x[test]*x[test][:, k]
        inverse = np.max(abs(_invert_rosette(y[test], coef) - x[test]))
        worst = max(worst, inverse)
        print(name, key, 'inverse max', f'{inverse:.3g}', 'forward max', f'{np.max(abs(forward-y[test])):.3g}', end='; ')
        print('full mixed max', f'{np.max(abs(alternative(train,test,"mixed")-y[test])):.3g}', end='; ')
        rows = np.flatnonzero(train & clean)
        inner = np.full(len(x), -1)
        inner[rows] = np.random.default_rng(42).permutation(len(rows)) % 3
        options = [(family, a) for family in ('self', 'sum', 'difference') for a in (0, 1e-5, 1e-4, 1e-3, 1e-2, .1)]
        def score(option):
            return sum(np.sum((alternative(train & (inner != i), inner == i, *option)-y[inner == i])**2) for i in range(3))
        opt = min(options, key=score)
        print('nested rational', opt, 'max', f'{np.max(abs(alternative(train,test,*opt)-y[test])):.3g}')
assert worst < 1e-4, worst
out = solve_batch(study)
print('isolated composed solve:', len(out['weights']), 'weights; budget error', abs(out['weights'].sum()-study['total_budget']))
