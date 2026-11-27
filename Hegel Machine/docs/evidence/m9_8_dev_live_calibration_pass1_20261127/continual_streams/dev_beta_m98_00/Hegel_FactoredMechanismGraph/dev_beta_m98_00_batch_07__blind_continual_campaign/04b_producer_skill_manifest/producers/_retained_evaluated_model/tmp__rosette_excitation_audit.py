import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _Rosette, _cyclic_design, _robust_fit

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    public = pickle.load(f)

def rms(r):
    return float(np.sqrt(np.mean(r*r, axis=0)).max())

for item in public:
    s = item['study']['sensor_calibration_sweeps']
    x, y = s['ref_rosette'], s['raw_rosette']
    clean = np.max(np.abs(y), axis=1) < 7
    amplitude = np.max(np.abs(x), axis=1)
    lo, hi = np.quantile(amplitude, [.25, .75])
    max_inverse = 0.
    for name, held in [('low excitation', amplitude <= lo), ('high excitation', amplitude >= hi)]:
        train, test = ~held, held & clean
        m = _Rosette(x[train], y[train])
        base = [_cyclic_design(x, j) for j in range(4)]
        spec = {'quadratic ablation': [a[:, :-1] for a in base]}
        for sign in (-1, 1):
            spec[f'cubic plus rational contrast {sign:+d}'] = [
                np.column_stack((a, a[:, 4] / (1 + .2*(x[:, j] + sign*x[:, (j+1)%4])**2)))
                for j, a in enumerate(base)]
        errors = {}
        for label, designs in spec.items():
            coeff, _ = _robust_fit([a[train] for a in designs], y[train])
            pred = np.column_stack([a[test] @ coeff[:, j] for j, a in enumerate(designs)])
            errors[label] = rms(pred - y[test])
        cubic = rms(m.predict(x[test]) - y[test]); inv = rms(m.invert(y[test]) - x[test])
        max_inverse = max(max_inverse, inv)
        print(name, 'clean test rows', test.sum(), 'cubic forward RMSE', cubic, 'cubic inverse RMSE', inv, 'challenger forward RMSE', errors)
    print('Worst excitation-block inverse channel RMSE:', max_inverse)
    assert max_inverse < 1e-4, 'Reopen calibration: inverse error exceeds the existing audit threshold on a held-out excitation band'
