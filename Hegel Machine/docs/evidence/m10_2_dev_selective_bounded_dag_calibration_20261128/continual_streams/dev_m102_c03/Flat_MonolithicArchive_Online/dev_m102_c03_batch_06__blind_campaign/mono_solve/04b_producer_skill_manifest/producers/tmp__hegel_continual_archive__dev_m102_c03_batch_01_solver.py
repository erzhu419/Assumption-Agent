"""Calibrated regression with a cycle-interaction covariance."""
import numpy as np

_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_CYCLE = [0, 2, 3, 5]


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        us, vs = u[keep], v[keep]
        um, vm = us.mean(axis=0), vs.mean(axis=0)
        centered = us - um
        gain = np.sum(centered * (vs - vm), axis=0) / np.sum(centered**2, axis=0)
        offset = vm - um * gain
        error = np.linalg.norm(v - offset - u * gain, axis=1)
        keep = error <= max(.001, 3 * np.median(error))
    return offset, gain


def _products(z):
    return z[:, _PAIRS[:, 0]] * z[:, _PAIRS[:, 1]]


def _kernels(left, right):
    a, b = _products(left), _products(right)
    linear = 1 + left @ right.T
    quadratic = linear + a @ b.T + (left**2) @ (right**2).T
    cycle = a[:, _CYCLE] @ b[:, _CYCLE].T
    state = quadratic + .04 * cycle * linear**2
    d = (np.sqrt(5) / 2) * np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    matern = (1 + d + d**2 / 3) * np.exp(-d)
    return state + .04 * cycle * matern, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, gain = _calibrate(study['panel_a'])
    z = (np.asarray(train['x']) - offset) / gain
    test = (np.asarray(study['test']['x']) - offset) / gain
    ky, ks = _kernels(z, z)
    cy, cs = _kernels(test, z)
    ridge = .002 * np.eye(len(z))
    y = cy @ np.linalg.solve(ky + ridge, np.asarray(train['y']))
    state = cs @ np.linalg.solve(ks + ridge, np.asarray(train['traj']))
    endpoint = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), endpoint.mean(), endpoint.std()])}
