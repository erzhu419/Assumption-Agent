"""Panel-calibrated spatial operators and separate scalar and temporal readouts."""
import numpy as np


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    params = []
    for u, v in zip(panel['u'].T, panel['v'].T):
        design = _aff(u)
        coeff = _fit(design, v)
        n_keep = max(2, int(.875 * len(u)))
        for _ in range(8):
            keep = np.argsort(np.abs(design @ coeff - v))[:n_keep]
            coeff = _fit(design[keep], v[keep])
        error = np.abs(design @ coeff - v)
        inliers = error <= max(.002, 10 * np.median(error))
        coeff = _fit(design[inliers], v[inliers])
        params.append(coeff)
    offset, gain = np.asarray(params).T
    return (x - offset) / gain


def _products(x):
    return np.column_stack((x[:, 0] * x[:, 1], x[:, 0] * x[:, 3],
                            x[:, 1] * x[:, 2], x[:, 2] * x[:, 3]))


def _latent(study, x):
    a, b, c = (study[k] for k in ('panel_a', 'panel_b', 'panel_c'))
    z = _calibrate(a, x)
    w = _aff(z) @ _fit(_aff(b['u']), b['v'])
    return _aff(_products(w)) @ _fit(_aff(_products(c['u'])), c['v1'])


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    h = _latent(study, np.vstack((train['x'], test['x'])))
    space = _aff(h)
    scalar = np.column_stack((space, h[:, 0] * h[:, 2], h[:, 1] * h[:, 3]))
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    coeff = _fit(space[:n], train['traj'])
    # Estimate the shared two-dimensional temporal subspace from the signal,
    # not from raw trajectories that include measurement noise.
    modes = np.linalg.svd(space[:n] @ coeff, full_matrices=False)[2][:2]
    state = ((space[n:] @ coeff) @ modes.T) @ modes
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
