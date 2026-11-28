"""Calibrated panel-B coordinates with separate scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _fit(study, x, y, traj):
    a, b = study['panel_a'], study['panel_b']
    u, v = a['u'], a['v']
    keep = np.max(np.abs(v-u), axis=1) < 1.0
    u, v = u[keep], v[keep]
    du = u-u.mean(axis=0)
    gain = np.sum(du*(v-v.mean(axis=0)), axis=0)/np.sum(du**2, axis=0)
    offset = v.mean(axis=0)-gain*u.mean(axis=0)
    w = (x-offset)/gain

    def features(q, p):
        k, power, linear = p
        return _affine(q**3/(1+k*np.abs(q)**power)+linear*q)

    def scalar(z):
        return np.column_stack((_affine(z), z[:,0]*z[:,2], z[:,1]*z[:,3]))

    def profile(p):
        panel = features(b['u'], p)
        operator = _lstsq(panel, b['v'])
        z = features(w, p)@operator
        sy, st = scalar(z), _affine(z)
        cy = _lstsq(sy, y)
        ct = _lstsq(st, traj)
        left, value, right = np.linalg.svd(ct, full_matrices=False)
        ct = (left[:,:2]*value[:2])@right[:2]
        residual = np.concatenate(((panel@operator-b['v']).ravel(), 0.7*(sy@cy-y), 0.7*(st@ct-traj).ravel()))
        return residual, operator, cy, ct

    bounds = ([0.015, 0.6, -4.0], [5.0, 5.0, 4.0])
    fits = [least_squares(lambda p: profile(p)[0], p, bounds=bounds, max_nfev=90)
            for p in ([0.5, 2.5, -0.09], [1.3, 2.0, -0.08])]
    params = min(fits, key=lambda r: r.cost).x
    _, operator, cy, ct = profile(params)
    def coordinates(query):
        return features((query-offset)/gain, params)@operator
    def predict(query):
        z = coordinates(query)
        return scalar(z)@cy, _affine(z)@ct
    return predict, coordinates


def solve_batch(study: dict) -> dict:
    train = study['train']
    predict, _ = _fit(study, train['x'], train['y'], train['traj'])
    y, state = predict(study['test']['x'])
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
