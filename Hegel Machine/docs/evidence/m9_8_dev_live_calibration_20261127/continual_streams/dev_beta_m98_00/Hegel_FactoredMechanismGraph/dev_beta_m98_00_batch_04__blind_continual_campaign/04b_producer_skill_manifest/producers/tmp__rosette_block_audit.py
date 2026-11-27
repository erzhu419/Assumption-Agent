import pickle
import runpy
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
a = up._features(x)
clean = np.max(np.abs(y), axis=1) < 8  # Gross spikes are separated in the public measurements.
print('EMI row indices:', np.flatnonzero(~clean).tolist())
errors = []
for block in np.array_split(np.arange(len(x)), 5):
    train = np.ones(len(x), dtype=bool)
    train[block] = False
    val = block[clean[block]]
    coef = up._fit(x[train], y[train])
    pred = a @ coef
    base = np.linalg.norm(pred[val]-y[val], axis=1)
    inv = np.linalg.norm(up._invert(y[val], coef)-x[val], axis=1)
    errors.extend(inv)
    fit_error = np.linalg.norm(pred-y, axis=1)
    use = train & (fit_error <= 6*np.median(fit_error[train]))
    variants = []
    for sign in (1, -1):
        for alpha in (.1, .5, 2.):
            p = np.empty_like(y[val])
            for j in range(4):
                v = x[:, j] + sign*x[:, (j+1)%4]
                d = np.column_stack((a[:, coef[:, j] != 0], v**3/(1+alpha*v**2)))
                p[:, j] = d[val] @ np.linalg.lstsq(d[use], y[use, j], rcond=None)[0]
            variants.append(np.linalg.norm(p-y[val], axis=1).max())
    print('held block', int(block[0]), int(block[-1]), 'clean/EMI', len(val), len(block)-len(val), 'inverse max', inv.max(), 'forward max', base.max(), 'best augmented forward max', min(variants))
print('blocked worst inverse L2:', max(errors))
assert max(errors) < 2e-4
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
