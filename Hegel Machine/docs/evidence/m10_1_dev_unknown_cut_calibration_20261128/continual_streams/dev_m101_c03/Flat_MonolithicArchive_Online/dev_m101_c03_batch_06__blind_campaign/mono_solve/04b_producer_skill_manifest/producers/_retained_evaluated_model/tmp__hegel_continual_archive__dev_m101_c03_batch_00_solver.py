"""Panel-calibrated, two-mode prediction for the public study."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    coeff = []
    for uj, vj in zip(u.T, v.T):
        dx = uj[i] - uj[j]
        valid = dx != 0
        slope = np.median((vj[i] - vj[j])[valid] / dx[valid])
        offset = np.median(vj - slope * uj)
        r = abs(vj - offset - slope * uj)
        keep = r <= 6 * np.median(r) + 1e-8
        coeff.append(np.linalg.lstsq(np.column_stack((np.ones(keep.sum()), uj[keep])), vj[keep], rcond=None)[0])
    return np.asarray(coeff).T


def _branch(u, v, first):
    def basis(z, p):
        return 0.45*z + np.tanh(p*z) if first else z + p*z*z

    def fit(p):
        f = basis(u, p)
        mean = f.mean(axis=0)
        g = f - mean
        left, singular, right = np.linalg.svd(g.T @ (v - v.mean(axis=0)))
        mat = (left @ right) * (singular.sum() / np.sum(g*g))
        bias = v.mean(axis=0) - mean @ mat
        return mat, bias

    def error(p):
        mat, bias = fit(p)
        return np.sum((basis(u, p) @ mat + bias - v)**2)

    p = minimize_scalar(error, bounds=(0.5, 4) if first else (0.05, 1), method='bounded').x
    mat, bias = fit(p)
    return lambda z: basis(z, p) @ mat + bias


def _features(study, inputs):
    offset, gain = _calibration(study['panel_a'])
    z = (inputs - offset) / gain
    panel = study['panel_c']
    a = _branch(panel['u'], panel['v1'], True)(z)
    b = _branch(panel['u'], panel['v2'], False)(z)
    mixed = (a[:, :, None] * b[:, None, :]).reshape(len(z), 16)
    linear = np.column_stack((np.ones(len(z)), a, b))
    return z, linear, mixed


def _ridge(x, y, penalty):
    gram = x.T @ x
    gram[np.diag_indices_from(gram)] += penalty
    gram[0, 0] -= penalty
    return np.linalg.solve(gram, x.T @ y)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    _, linear, mixed = _features(study, np.vstack((train['x'], study['test']['x'])))
    dot = mixed[:, [0, 5, 10, 15]].sum(axis=1)
    common = np.column_stack((linear, dot))
    # The slow mode is driven primarily by the first-by-second cross-branch product.
    slow = np.column_stack((linear, mixed[:, 1]))
    times = np.asarray(study['eval_times'])[:, None]

    def time_basis(rates):
        return -np.expm1(-times * rates)

    def residual(rates):
        basis = time_basis(rates)
        observed = train['traj'].T
        return (basis @ np.linalg.lstsq(basis, observed, rcond=None)[0] - observed).ravel()

    rates = least_squares(residual, [1.8, 0.4], bounds=([1.0, 0.01], [10.0, 1.0])).x
    temporal = time_basis(rates)
    amplitudes = np.linalg.lstsq(temporal, train['traj'].T, rcond=None)[0].T
    main = common[n:] @ _ridge(common[:n], np.column_stack((train['y'], amplitudes[:, 0])), 0.00015)
    tail = slow[n:] @ _ridge(slow[:n], amplitudes[:, 1], 0.0003)
    y = main[:, 0]
    trajectory = np.column_stack((main[:, 1], tail)) @ temporal.T
    end = trajectory[:, -1]
    return {'y_hat': y, 'state_pred': trajectory,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
