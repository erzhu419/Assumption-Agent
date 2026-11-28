"""Panel-calibrated branch fusion and shared relaxation prediction."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    i, j = np.triu_indices(len(u), 1)
    columns = []
    for a, b, c in zip(u.T, v.T, x.T):
        dx = a[i] - a[j]
        good = np.abs(dx) > 1e-9
        slope = np.median((b[i[good]] - b[j[good]]) / dx[good])
        residual = b - slope * a
        deviation = np.abs(residual - np.median(residual))
        good = deviation <= max(.001, 6 * np.median(deviation))
        bias, gain = _ls(_design(a[good]), b[good])
        columns.append((c - bias) / gain)
    return np.column_stack(columns)


def _branches(panel, z):
    u = np.asarray(panel['u'])

    def left(q, k):
        return _design(np.c_[q, q / (1 + k * np.abs(q))])

    def right(q, k):
        return _design(np.c_[q, q * np.roll(q, -1, axis=1), np.arctan(k * q)])

    outputs = []
    for key, features in [('v1', left), ('v2', right)]:
        v = np.asarray(panel[key])

        def loss(k):
            a = features(u, k)
            return np.mean((a @ _ls(a, v) - v)**2)

        k = minimize_scalar(loss, bounds=(.2, 2), method='bounded').x
        outputs.append(features(z, k) @ _ls(features(u, k), v))
    return outputs


def _relaxation(times, trajectory):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        h = basis(rates)
        return (h @ _ls(h, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [1.8, .4], bounds=([.8, .01], [5, 1.2]), ftol=1e-10).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    p, q = _branches(study['panel_c'], z)
    common = np.c_[p, q, np.sum(p * q, axis=1)]
    a = _design(common)
    y = a[n:] @ _ls(a[:n], train['y'])
    shifted = p * np.roll(q, -1, axis=1)
    a = _design(np.c_[common, shifted])
    h = _relaxation(np.asarray(study['eval_times']), np.asarray(train['traj']))
    amplitudes = _ls(h, np.asarray(train['traj']).T).T
    state = (a[n:] @ _ls(a[:n], amplitudes)) @ h.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
