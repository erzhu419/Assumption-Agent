from itertools import combinations_with_replacement

import numpy as np


def _features(x):
    return np.column_stack([np.ones(len(x)), x, *[
        x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)
    ]])


def _predict(x, y, traj, test, panel_b):
    u = np.asarray(panel_b['u'])
    operator = np.linalg.lstsq(np.column_stack([np.ones(len(u)), u]), panel_b['v'], rcond=None)[0][1:]
    dx, dq = _features(x), _features(test)
    gram, cross = dx @ dx.T, dq @ dx.T

    def fit(target, z, q, nugget):
        scale = z.std(axis=0)
        z, q = z / scale, q / scale

        def matern(a, b):
            r = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1) / 1.6
            return (1 + r + r*r/3) * np.exp(-r)

        k = gram + matern(z, z)
        k.flat[::len(x)+1] += nugget
        return (cross + matern(q, z)) @ np.linalg.solve(k, target)

    inv = np.linalg.inv(operator)
    scalar = fit(y, x @ inv, test @ inv, 0.0003)
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    latent = traj @ modes.T
    state = (0.75 * fit(latent, x, test, 0.003) +
             0.25 * fit(latent, x @ operator, test @ operator, 0.03)) @ modes
    return scalar, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    y, state = _predict(train['x'], train['y'], train['traj'], study['test']['x'], study['panel_b'])
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
