"""Invert the reference calibration and fit readouts of the panel B operator."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize, root

def _affine(z):
    return np.column_stack((np.ones(len(z)), z))

def _calibrate(panel, observed):
    u, v = panel['u'], panel['v']
    next_u = np.roll(u, -1, axis=1)
    designs = [np.column_stack((np.ones(len(u)), a, b, a * b, a**3))
               for a, b in zip(u.T, next_u.T)]
    # The sparse corruptions are much larger than the calibration nonlinearity.
    affine = _affine(u)
    keep = np.arange(len(u))
    for _ in range(3):
        residual = np.linalg.norm(affine @ _ls(affine[keep], v[keep]) - v, axis=1)
        keep = np.flatnonzero(residual < 5 * np.median(residual))
    coef = np.array([_ls(p[keep], v[keep, j]) for j, p in enumerate(designs)]).T

    def forward(z):
        w = np.roll(z, -1)
        return coef[0] + coef[1]*z + coef[2]*w + coef[3]*z*w + coef[4]*z**3

    return np.array([root(lambda z: forward(z) - row, row, tol=1e-10).x
                     for row in observed])

def _panel_b(panel, x):
    u, v = panel['u'], panel['v']

    def features(q, p):
        return _affine(q + p[1] * np.sin(p[0] * q))

    def residual(p):
        a = features(u, p)
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [2.85, -.4], bounds=([1.5, -1.], [4.5, -.05])).x
    return features(x, p) @ _ls(features(u, p), v)


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _scalar_predict(w, wt, a, at, y, terminal):
    # Favor directions that change either observed readout, while retaining
    # Euclidean distance in the rest of the calibrated operator space.
    directions = _ls(a, np.column_stack((y, terminal)))[1:]
    directions /= np.where(np.linalg.norm(directions, axis=0) > 0,
                           np.linalg.norm(directions, axis=0), 1.0)
    metric = np.column_stack((np.eye(w.shape[1]), directions))

    def squared_distance(left, right):
        return np.sum(((left[:, None, :] - right[None, :, :]) @ metric)**2, axis=-1)

    d = squared_distance(w, w)
    eye = np.eye(len(w))
    rhs = np.column_stack((a, y))
    dof = len(w) - a.shape[1]

    def profile(p, predict=False):
        scale = np.exp(p[0])
        factor = cho_factor(np.exp(-0.5 * scale * d) + np.exp(p[1]) * eye)
        q = cho_solve(factor, rhs)
        gram = a.T @ q[:, :-1]
        beta = np.linalg.solve(gram, a.T @ q[:, -1])
        alpha = q[:, -1] - q[:, :-1] @ beta
        if predict:
            return at @ beta + np.exp(-0.5 * scale * squared_distance(wt, w)) @ alpha
        variance = max(1e-15, (y - a @ beta) @ alpha / dof)
        return (dof * np.log(variance) + 2 * np.log(np.diag(factor[0])).sum()
                + np.linalg.slogdet(gram)[1])

    opt = minimize(profile, [0., np.log(.005)], method='L-BFGS-B',
                   bounds=[(-2.5, 2.5), (-10., -1.)])
    return profile(opt.x, predict=True)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    w = _panel_b(study['panel_b'], _calibrate(study['panel_a'], x))
    a, at = _affine(w[:n]), _affine(w[n:])
    state = at @ _ls(a, train['traj'])
    y = _scalar_predict(w[:n], w[n:], a, at, train['y'], train['traj'][:, -1])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
