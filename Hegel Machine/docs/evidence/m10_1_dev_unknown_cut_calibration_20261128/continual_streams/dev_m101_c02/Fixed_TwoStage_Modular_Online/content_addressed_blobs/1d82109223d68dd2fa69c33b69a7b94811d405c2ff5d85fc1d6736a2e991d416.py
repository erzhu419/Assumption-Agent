"""Predict responses using inverse calibration and a panel-B representation."""
import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.distance import cdist


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _inverse_calibration(panel, x):
    u, v = panel['u'], panel['v']
    good = np.max(np.abs(v - u), axis=1) < 2
    u, v = u[good], v[good]
    coeff = np.array([
        _ls(_bias(np.column_stack((a, b, a*b, a**3))), y)
        for a, b, y in zip(u.T, np.roll(u, -1, axis=1).T, v.T)
    ]).T
    offset, gain, cross, bilinear, cubic = coeff
    j = np.arange(x.shape[1])
    z = x.copy()
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(12):
        q = np.roll(z, -1, axis=1)
        err = offset + gain*z + cross*q + bilinear*z*q + cubic*z**3 - x
        jac[:, j, j] = gain + bilinear*q + 3*cubic*z*z
        jac[:, j, (j+1) % len(j)] = cross + bilinear*z
        z -= np.linalg.solve(jac, err[..., None])[..., 0]
    return z


def _panel_features(panel, z):
    u, v = panel['u'], panel['v']

    def odd(x, p):
        return _bias(np.column_stack((np.arctan(p[0]*np.sign(x)*np.abs(x)**p[1]), x, x**3)))

    def residual(p):
        a = odd(u, p)
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [.8, 3], bounds=([.005, 1.02], [25, 7]), max_nfev=120).x
    atan = odd(z, p) @ _ls(odd(u, p), v)

    def rational(x, k):
        return _bias(np.column_stack((x, x**3/(1+k*x*x))))

    def objective(k):
        a = rational(u, k)
        return np.mean((a @ _ls(a, v) - v)**2)

    k = minimize_scalar(objective, bounds=(.03, 3), method='bounded').x
    rat = rational(z, k) @ _ls(rational(u, k), v)
    return atan, rat


def _features(study, x):
    z = _inverse_calibration(study['panel_a'], x)
    return (z, *_panel_features(study['panel_b'], z))


def _kernel(w, z):
    return 1 + w @ w.T + .03*np.exp(-3*cdist(z, z, 'sqeuclidean')/(4*.65**2))


def _predict(features, target, train, test):
    z, atan, rat = features
    kernel = _kernel(atan, z)
    k = kernel[np.ix_(train, train)] + .0003*np.eye(len(train))
    pred = kernel[np.ix_(test, train)] @ solve(k, target, assume_a='pos')

    j, l = np.triu_indices(rat.shape[1])
    q = np.column_stack((rat, rat[:, j]*rat[:, l]))
    mu = q[train].mean(axis=0)
    q = (q - mu) / q[train].std(axis=0)
    a, b = q[train], q[test]
    y = target[:, 0]
    quad = y.mean() + b @ solve(a.T @ a + .3*np.eye(a.shape[1]), a.T @ (y-y.mean()), assume_a='pos')
    pred[:, 0] = .25*pred[:, 0] + .75*quad
    return pred


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x = np.concatenate((tr['x'], study['test']['x']))
    split = len(tr['x'])
    target = np.column_stack((tr['y'], tr['traj']))
    p = _predict(_features(study, x), target, np.arange(split), np.arange(split, len(x)))
    y, state = p[:, 0], p[:, 1:]
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
