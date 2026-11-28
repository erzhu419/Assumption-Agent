import itertools
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(u, v, x, warp, starts, bounds):
    def residual(p):
        z = _aff(warp(u, p))
        return (z @ _ls(z, v) - v).ravel()

    fit = min((least_squares(residual, p, bounds=bounds, max_nfev=120)
               for p in starts), key=lambda fit: fit.cost)
    z = _aff(warp(u, fit.x))
    return _aff(warp(x, fit.x)) @ _ls(z, v), np.mean(np.abs(residual(fit.x)))


def _cross(x):
    return _aff(np.column_stack([x[:, i] * x[:, j] for i, j in
                                 ((0, 1), (0, 3), (1, 2), (2, 3))]))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    w = np.asarray(inputs['node_00']['w'], dtype=float)
    a = study['panel_b']
    affine = _ls(_aff(a['u']), a['v'])
    x = (w - affine[0]) @ np.linalg.pinv(affine[1:])

    def warp(z, p):
        return (z**3 + p[0]*z)/(1 + p[1]*z*z)

    lower = -0.85 / max(1., float(np.max(np.abs(a['u'])))**2)
    b, b_error = _calibrate(a['u'], a['v'], x, warp,
                            ((.22, .22), (1.25, 1.25), (4., .2),
                             (.1, max(-.2, .9*lower))),
                            ([-20., lower], [40., 30.]))
    c = study['panel_c']
    cz = _cross(c['u'])
    c_coef = _ls(cz, c['v1'])
    bilinear = _cross(x) @ c_coef
    c_error = np.mean(np.abs(cz @ c_coef - c['v1']))
    quadratic = [b[:, i]*b[:, j] for i, j in
                 itertools.combinations_with_replacement(range(b.shape[1]), 2)]
    f = np.column_stack((b, bilinear, *quadratic))

    train = study['train']
    y = np.asarray(train['y'], dtype=float).reshape(-1)
    target = np.column_stack((y, train['traj']))
    n = len(y)
    z = _aff((f-f[:n].mean(axis=0))/np.maximum(f[:n].std(axis=0), 1e-8))
    t = z[:n]
    pen = .02*np.eye(t.shape[1]); pen[0, 0] = 0
    coef = np.linalg.solve(t.T @ t + pen, t.T @ target)
    pred = z[n:] @ coef
    y_hat, state_pred = pred[:, 0], pred[:, 1:]
    tail = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         tail.mean(), tail.std()])}
