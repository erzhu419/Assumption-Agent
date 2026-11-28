import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _coordinates(study, x):
    panel = study['panel_a']
    design = _affine(panel['u'])
    v = panel['v']
    err = np.linalg.norm(design @ _fit(design, v) - v, axis=1)
    keep = np.argsort(err)[:int(.86 * len(err))]
    calibration = _fit(design[keep], v[keep])
    z = (x - calibration[0]) / np.diag(calibration[1:])
    panel = study['panel_b']
    return _affine(z) @ _fit(_affine(panel['u']), panel['v'])


def _readout(w, a, y):
    """Fit the affine saturation index, profiling out linear coefficients."""
    n = len(y)
    pair = np.column_stack((w[:, 0] * w[:, 2], w[:, 1] * w[:, 3]))

    def design(affine, products, index_coefs):
        index = affine @ index_coefs
        inv_den = 1 / (1 + index**2)
        return np.column_stack((affine, inv_den, index * inv_den,
                                products, products * inv_den[:, None]))

    a_train, p_train = a[:n], pair[:n]

    def residual(index_coefs):
        f = design(a_train, p_train, index_coefs)
        return f @ _fit(f, y) - y

    initial = .2 * _fit(a_train, y)
    index_coefs = least_squares(residual, initial, max_nfev=100).x
    f = design(a, pair, index_coefs)
    return f[n:] @ _fit(f[:n], y)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    w = _coordinates(study, np.vstack((train['x'], study['test']['x'])))
    a = _affine(w)
    state_pred = a[n:] @ _fit(a[:n], train['traj'])
    y_hat = _readout(w, a, train['y'])
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
