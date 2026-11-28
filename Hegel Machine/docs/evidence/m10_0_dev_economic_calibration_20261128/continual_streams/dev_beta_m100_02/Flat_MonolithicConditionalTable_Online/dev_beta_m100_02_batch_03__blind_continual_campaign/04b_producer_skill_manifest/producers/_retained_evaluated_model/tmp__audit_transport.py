import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
b = c['single_bank_sweeps']

def design(x, k, extra=False):
    j = (k + 1) % 4
    cols = [np.ones(len(x)), *x.T, x[:, k]**3, x[:, k] * x[:, j]]
    if extra:
        cols += [x[:, a] * x[:, d] for a in range(4) for d in range(a, 4) if (a, d) != tuple(sorted((k, j)))]
    return np.column_stack(cols)

errors = {key: np.empty_like(y) for key in ('cubic_cross', 'all_quadratics', 'rational_self', 'rational_difference')}
for test in np.array_split(np.arange(len(x)), 6):
    train = np.setdiff1d(np.arange(len(x)), test)
    fold = dict(s, sensor_calibration_sweeps=dict(c, ref_rosette=x[train], raw_rosette=y[train]))
    ts = m._sweep_fit(fold)
    errors['cubic_cross'][test] = m._raw(x[test], ts) - y[test]
    for k, t in enumerate(ts):
        j = (k + 1) % 4
        sel = b['bank_id'].ravel() == k
        iso = np.zeros((np.count_nonzero(sel), 4))
        iso[:, k] = b['ref_signal'].ravel()[sel]
        xx = np.vstack((x[train], iso))
        yy = np.r_[y[train, k], b['raw_readout'].ravel()[sel]]
        # Isolate gross contamination using only this fold's training data.
        good = np.abs(design(xx, k) @ t - yy) < .01
        xx, yy = xx[good], yy[good]
        a, at = design(xx, k, True), design(x[test], k, True)
        errors['all_quadratics'][test, k] = at @ np.linalg.lstsq(a, yy, rcond=None)[0] - y[test, k]
        a, at = design(xx, k), design(x[test], k)
        for label, q, qt in [('rational_self', xx[:, k], x[test, k]), ('rational_difference', xx[:, k]-xx[:, j], x[test, k]-x[test, j])]:
            p = least_squares(lambda p: (a @ p[:-1]) / (1 + p[-1]*q*q) - yy, np.r_[t, 0.], bounds=(np.r_[np.full(len(t), -np.inf), 0.], np.inf), gtol=1e-13).x
            errors[label][test, k] = (at @ p[:-1]) / (1 + p[-1]*qt*qt) - y[test, k]

base = errors['cubic_cross']
gross = np.abs(base) > .01
print('calibration held-out gross residuals:', np.count_nonzero(gross), 'across rows', np.flatnonzero(gross.any(axis=1)).tolist(), 'smallest gross', np.min(abs(base[gross]), initial=np.inf))
for name, e in errors.items():
    print(name, 'clean worst channel CV RMSE', max(np.sqrt(np.mean(e[~gross[:, k], k]**2)) for k in range(4)), 'clean max abs', max(abs(e[~gross])))

S, _ = m._responses(s, m._operator(s))
Y, Z = m._calibrate(s), s['Z_nuisance']
for nuisance in (False, True):
    rmse = []
    for test in np.array_split(np.arange(Y.size), len(s['mu_shifts'])):
        train = np.setdiff1d(np.arange(Y.size), test)
        a = np.c_[S[train], Y[train]]
        z = Z[train]
        p = np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T @ a) if nuisance else np.zeros((z.shape[1], a.shape[1]))
        a -= z @ p
        X, y_train = a[:, :-1], a[:, -1]
        w, _ = m._box_simplex(X.T @ X + s['source_ridge']*np.eye(X.shape[1]), -X.T @ y_train, s['u_box'], s['total_budget'])
        err = S[test] @ w + Z[test] @ (p[:, -1] - p[:, :-1] @ w) - Y[test]
        rmse.append(float(np.sqrt(np.mean(err**2))))
    print('profile nuisance', nuisance, 'held-shift RMSE', np.round(rmse, 8))
print('study fields:', ', '.join(sorted(s)))
