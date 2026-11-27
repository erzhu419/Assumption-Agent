import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
sweeps = study['sensor_calibration_sweeps']
x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
D = np.stack([m._rosette_basis(x, j) for j in range(4)], axis=1)

def factor(p):
    return np.column_stack((p[:4], p[4:8, None] * np.r_[1., p[8:]][None, :]))

def fit(train):
    limited = dict(sweeps, ref_rosette=x[train], raw_rosette=y[train], single_bank_sweeps={k: v[:0] for k, v in sweeps['single_bank_sweeps'].items()})
    base = m._fit_rosette(limited)
    errors = np.max(np.abs(m._rosette_raw(x[train], base)-y[train]), axis=1)
    good = np.flatnonzero(train)[errors <= max(5*np.median(errors), 1e-9)]
    seed = np.r_[base[:,0], base[:,1], np.mean(base[:,2:]/base[:,1,None], axis=0)]
    result = least_squares(lambda p: (np.einsum('ijk,jk->ij', D[good], factor(p))-y[good]).ravel(), seed, gtol=1e-13)
    return base, factor(result.x)

# Hold out observed sweeps by amplitude, including the block farthest from the observation range.
# The raw-unit separation is used only to exclude grossly corrupted outcomes from scoring.
clean = np.flatnonzero(np.max(np.abs(y), axis=1) < 9)
order = clean[np.argsort(np.linalg.norm(x[clean], axis=1))]
errors = []
for fold in np.array_split(order, 5):
    train = np.ones(len(x), dtype=bool)
    train[fold] = False
    errors.append([np.sqrt(np.mean((m._rosette_raw(x[fold], coef)-y[fold])**2, axis=0)).max() for coef in fit(train)])
print('Worst sweep-channel RMSE, separate versus shared gain-normalized terms:', np.max(errors, axis=0))
_, shared = fit(np.ones(len(x), dtype=bool))
reference = m.solve_batch(study)
original = m._fit_rosette
try:
    m._fit_rosette = lambda _: shared
    challenger = m.solve_batch(study)
finally:
    m._fit_rosette = original
for key in ('weights', 'terminal_field'):
    a, b = reference[key], challenger[key]
    print(key, 'relative change from pooling gain-normalized terms:', np.linalg.norm(a-b)/np.linalg.norm(a))
print('Candidate primary outputs and certificate were not modified.')
