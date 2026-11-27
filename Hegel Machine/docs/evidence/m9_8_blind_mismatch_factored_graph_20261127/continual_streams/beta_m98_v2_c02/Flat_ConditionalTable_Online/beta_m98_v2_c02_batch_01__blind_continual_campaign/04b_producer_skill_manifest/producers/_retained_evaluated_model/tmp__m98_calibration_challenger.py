import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sw = s['sensor_calibration_sweeps']
x, y = (np.asarray(sw[k]) for k in ('ref_rosette', 'raw_rosette'))
F = np.c_[np.ones(len(x)), x]
folds = np.array_split(np.argsort(np.max(np.abs(x), axis=1)), 5)

def design(i, j, a):
    q = x[:, i] if i == j else (x[:, i] - x[:, j]) / np.sqrt(2)
    return F / (1 + a*q[:, None]**2)

def fit(D, rows):
    return np.linalg.lstsq(D[rows], y[rows], rcond=None)[0]

scores = [[], []]
choices = [(i, j) for i in range(4) for j in range(i, 4)]
for test in folds:
    train = np.setdiff1d(np.arange(len(x)), test)
    def train_loss(i, j, a):
        D = design(i, j, a)
        return np.mean((D[train] @ fit(D, train) - y[train])**2)
    candidates = [(train_loss(0, 0, 0.), 0, 0, 0.)]
    for i, j in choices:
        opt = minimize_scalar(lambda a: train_loss(i, j, a), bounds=(0., .1), method='bounded', options={'xatol': 1e-10})
        candidates.append((opt.fun, i, j, opt.x))
    _, i, j, a = min(candidates)
    for D, out in zip((F, design(i, j, a)), scores):
        out.append(np.sqrt(np.mean((D[test] @ fit(D, train) - y[test])**2, axis=0)))
    print(f'held-out rows={len(test)} pair=({i},{j}) a={a:.3g} affine={max(scores[0][-1]):.3g} rational={max(scores[1][-1]):.3g}')
print('worst amplitude-fold/channel RMSE, affine / rational:', np.max(scores[0]), np.max(scores[1]))
S, _ = m._forward(s)
D = np.vstack([(S @ p['w_pulse']).reshape(-1, 4) for p in item['campaign_diagnostics']])
print('reference ranges, pulse:', D.min(), D.max(), 'lowest amplitude fold:', np.max(np.abs(x[folds[0]])))
