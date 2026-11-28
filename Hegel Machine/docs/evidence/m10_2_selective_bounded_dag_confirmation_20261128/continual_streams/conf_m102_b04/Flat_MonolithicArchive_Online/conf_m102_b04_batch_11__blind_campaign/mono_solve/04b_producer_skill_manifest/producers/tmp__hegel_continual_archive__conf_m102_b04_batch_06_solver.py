"""Calibrate inputs and fit a shared nonlinear operator with separate readouts."""
import numpy as np
from scipy.optimize import least_squares


def _least(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibration_design(u, j):
    k = (j + 1) % u.shape[1]
    a, b = u[:, j], u[:, k]
    return np.column_stack((np.ones(len(u)), a, b, a*b, a**3))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    designs = [_calibration_design(u, j) for j in range(u.shape[1])]
    distance = np.linalg.norm(v-u, axis=1)
    keep = distance < max(5*np.median(distance), 1e-8)
    for _ in range(4):
        w = np.column_stack([_least(g[keep], v[keep, j]) for j, g in enumerate(designs)])
        err = np.linalg.norm(np.column_stack([g @ w[:, j] for j, g in enumerate(designs)])-v, axis=1)
        keep = err < max(6*np.median(err), 1e-8)
    return w


def _inverse(x, w):
    j = np.arange(x.shape[1]); k = np.roll(j, -1)
    linear = np.diag(w[1]); linear[j, k] = w[2]
    u = np.linalg.solve(linear, (x-w[0]).T).T
    jac = np.zeros((len(u), len(j), len(j)))
    for _ in range(40):
        nxt = u[:, k]
        r = w[0] + w[1]*u + w[2]*nxt + w[3]*u*nxt + w[4]*u**3 - x
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac[:, j, j] = w[1] + w[3]*nxt + 3*w[4]*u**2
        jac[:, j, k] = w[2] + w[3]*u
        du = np.linalg.solve(jac, r[..., None])[..., 0]
        u -= du/np.maximum(1., np.max(np.abs(du), axis=1, keepdims=True))
    return u


def _activation(x, params):
    knee, linear, shoulder = params
    excess = np.maximum(np.abs(x) - knee, 0.0)
    return x**3 + linear*x + shoulder*np.sign(x)*excess**2


def _fit_operator(panel, u, traj):
    p, v = panel['u'], panel['v']

    def profile(params, coefficients=False):
        a = _affine(_activation(u, params))
        b = _affine(_activation(p, params))
        state_weights = _least(a, traj)
        operator_weights = _least(b, v)
        if coefficients:
            return operator_weights, state_weights
        return np.concatenate(((a @ state_weights - traj).ravel(),
                               (b @ operator_weights - v).ravel()))

    params = least_squares(profile, [.6, -.07, -1.2],
                           bounds=([.1, -.5, -5], [1.2, .5, 1])).x
    return params, *profile(params, coefficients=True)


def _scalar_features(g):
    a, b, c, d = g.T
    return np.column_stack((g, a*c, b*d))


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibrate(study['panel_a'])
    u = _inverse(train['x'], calibration)
    test = _inverse(study['test']['x'], calibration)
    params, operator, state = _fit_operator(study['panel_b'], u, train['traj'])
    a = _affine(_activation(u, params))
    b = _affine(_activation(test, params))
    scalar = _least(_scalar_features(a @ operator), train['y'])
    y = _scalar_features(b @ operator) @ scalar
    t = b @ state
    return {'y_hat': y, 'state_pred': t, 'diagnostics_vec':
            np.array([y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()])}
