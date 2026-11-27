import importlib.util as iu
import itertools
import pickle
from pathlib import Path

import numpy as np


def load(path):
    spec = iu.spec_from_file_location(path.stem, path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


root = Path('/holdout_task')
m = load(root / 'batch_solver.py')
with (root / 'public_batch.pkl').open('rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
coef, keep = m._fit_rosette(s)
terms = list(itertools.combinations_with_replacement(range(4), 2)) + list(itertools.combinations_with_replacement(range(4), 3))
poly = np.column_stack([np.prod(x[:, t], axis=1) for t in terms])
errs = {name: [] for name in ('compact', 'full_cubic', 'compact_plus_rational')}
for i in np.flatnonzero(keep):
    train = np.arange(len(x)) != i
    sub = dict(s, sensor_calibration_sweeps={'ref_rosette': x[train], 'raw_rosette': raw[train]})
    _, fold_keep = m._fit_rosette(sub)
    rows = np.flatnonzero(train)[fold_keep]
    ee = {name: [] for name in errs}
    for j in range(4):
        A = m._rosette_design(x, j)
        own, prev, nxt = x[:, j], x[:, (j-1)%4], x[:, (j+1)%4]
        R = [p / (1 + a * q*q) for p in (own, own*nxt) for q in (own-nxt, own+nxt, own-prev, own+prev) for a in (.25, 1.)]
        for name, design in [('compact', A), ('full_cubic', np.c_[np.ones(len(x)), x, poly]), ('compact_plus_rational', np.column_stack([A, *R]))]:
            b = np.linalg.lstsq(design[rows], raw[rows, j], rcond=None)[0]
            ee[name].append(design[i] @ b - raw[i,j])
    for name in errs:
        errs[name].append(ee[name])
for name, e in errs.items():
    print('calibration', name, 'worst-bank LOO RMS', np.max(np.sqrt(np.mean(np.square(e), axis=0))), 'worst absolute', np.max(np.abs(e)))

archive = Path('/tmp/hegel_continual_archive')
sealed = load(next(archive.glob('*01_solver.py')))._axis
periodic = load(next(archive.glob('*03_solver.py')))._axis
robin = m._axis
axes = {'R': robin, 'S': lambda k,v,l,r: sealed(k,v,l), 'P': lambda k,v,l,r: periodic(k,v,l)}
y = m._calibrate(s)
Z = s['Z_nuisance']
n = len(s['sensor_ij'])
idx = np.arange(len(y))
rr = s['nuisance_ridge']
sr = s['source_ridge']
prior = np.eye(Z.shape[1])*rr
metrics = []
for xx, yy in itertools.product(axes, repeat=2):
    calls = iter((axes[xx], axes[yy]))
    m._axis = lambda *args: next(calls)(*args)
    S, _ = m._forward(s)
    errors = np.empty_like(y)
    for i in range(n):
        test = idx % n == i
        tr = ~test
        z, A, b = Z[tr], S[tr], y[tr]
        P = np.linalg.solve(z.T @ z + prior, z.T)
        At, bt = A-z@(P@A), b-z@(P@b)
        w, _ = m._box_budget(At.T@At+sr*np.eye(S.shape[1]), -At.T@bt, s['u_box'], s['total_budget'])
        errors[test] = y[test]-S[test]@w-Z[test]@(P@(b-A@w))
    e = errors.reshape(-1,n)
    rms = np.sqrt(np.mean(e**2))
    worst = np.sqrt(np.mean(e**2,axis=0)).max()
    banks = np.asarray(s['row_bank_id']).ravel()
    bank_rms = [np.sqrt(np.mean(errors[banks==i]**2)) for i in np.unique(banks)]
    metrics.append((rms, xx+yy))
    print('walls', xx+yy, 'pooled', rms, 'worst sensor', worst, 'bank RMS', np.round(bank_rms, 5))
m._axis = robin
print('wall ranking', sorted(metrics))
assert np.isfinite(m.solve_batch(s)['terminal_field']).all()
