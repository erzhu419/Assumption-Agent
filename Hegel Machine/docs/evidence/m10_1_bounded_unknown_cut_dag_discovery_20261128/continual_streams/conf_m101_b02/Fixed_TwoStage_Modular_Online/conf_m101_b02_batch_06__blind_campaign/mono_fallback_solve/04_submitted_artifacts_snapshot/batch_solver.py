"""Recover latent coordinates and fit scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, root
from scipy.special import erf


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _calibration(x):
    i, j = np.triu_indices(4)
    return _design(x, x[:, i] * x[:, j], x**3)


def _latent(panel, x):
    z, v = _calibration(panel['u']), panel['v']
    keep = np.arange(len(z))
    for _ in range(8):
        w = _ls(z[keep], v[keep])
        keep = np.argsort(np.linalg.norm(z @ w - v, axis=1))[:45]
    w = _ls(z[keep], v[keep])
    return np.array([root(lambda p: _calibration(p[None])[0] @ w - row, row).x
                     for row in x])


def _bridge(panel, x, kind):
    u, v = panel['u'], panel['v']
    def design(z, p):
        if kind == 'gaussian':
            zz = z + p[2:]
            return _design(zz * (1 - p[1] * np.exp(-(p[0] * zz)**2)))
        return _design(z - p[1] * erf(p[0] * z), z*z)

    def residual(p):
        z = design(u, p)
        return (z @ _ls(z, v) - v).ravel()

    init = [1., 1.08, 0., 0., 0., 0.] if kind == 'gaussian' else [1.45, .65]
    p = least_squares(residual, init).x
    return design(x, p) @ _ls(design(u, p), v)


def _pairs(x):
    return _design(x[:, 0]*x[:, 1], x[:, 0]*x[:, 3],
                   x[:, 1]*x[:, 2], x[:, 2]*x[:, 3])


def _features(study, x):
    a = _latent(study['panel_a'], x)
    pc = study['panel_c']
    w = _ls(_pairs(pc['u']), pc['v1'])
    for kind in ('gaussian', 'erf'):
        c = _pairs(_bridge(study['panel_b'], a, kind)) @ w
        yield np.column_stack((a, c, (a[:, :, None] * c[:, None, :]).reshape(len(a), 16)))


def _readout(x, y, n, ridge):
    mean = x[:n].mean(axis=0)
    scale = np.maximum(x[:n].std(axis=0), 1e-5)
    z = (x - mean) / scale
    u, s, vt = np.linalg.svd(z[:n], full_matrices=False)
    center = y.mean(axis=0)
    return center + ((z[n:] @ vt.T) * (s / (s*s + ridge))) @ (u.T @ (y - center))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    targets = np.column_stack((train['y'], train['traj']))
    x = np.vstack((train['x'], study['test']['x']))
    # The scalar readout is more sensitive to high-leverage interactions.
    features = list(_features(study, x))
    scalar = np.mean([_readout(f, targets[:, :1], n, .03) for f in features], axis=0)
    trajectory = np.mean([_readout(f, targets[:, 1:], n, .003) for f in features], axis=0)
    pred = np.column_stack((scalar, trajectory))
    y_hat, state_pred = pred[:, 0], pred[:, 1:]
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
