"""Calibrated sinusoidal operator with separate scalar and state readouts."""
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_affine(x), x[:, i] * x[:, j]))


def _calibration(panel):
    offsets, gains = [], []
    for u, v in zip(panel['u'].T, panel['v'].T):
        i, j = np.triu_indices(len(u), 1)
        du = u[j] - u[i]
        i, j = i[du != 0], j[du != 0]
        gain = np.median((v[j] - v[i]) / (u[j] - u[i]))
        offset = np.median(v - gain * u)
        err = abs(v - offset - gain * u)
        keep = err <= max(8 * np.median(err), 1e-8)
        offset, gain = np.linalg.lstsq(_affine(u[keep, None]), v[keep], rcond=None)[0]
        offsets.append(offset)
        gains.append(gain)
    return np.array(offsets), np.array(gains)


def _operator(panel, q):
    u, v = panel['u'], panel['v']

    def design(x, frequency):
        return _affine(np.sin(frequency * x)**2)

    def fit(frequency):
        f = design(u, frequency)
        w = np.linalg.lstsq(f, v, rcond=None)[0]
        return np.sum((f @ w - v)**2), w

    frequency = minimize_scalar(lambda k: fit(k)[0], bounds=(1.05, 1.55),
                                method='bounded', options={'xatol': 1e-10}).x
    return design(q, frequency) @ fit(frequency)[1]


def _scalar_readout(f, y, query):
    """Keep the affine terms and admit interactions only when BIC improves."""
    n = len(y)
    cols = list(range(5))
    remaining = list(range(5, f.shape[1]))

    def fit(columns):
        a = f[:, columns]
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        mse = np.mean((a @ w - y)**2)
        return n * np.log(max(mse, 1e-30)) + len(columns) * np.log(n), w

    score, w = fit(cols)
    while remaining:
        trial = [(j, fit(cols + [j])) for j in remaining]
        j, (next_score, next_w) = min(trial, key=lambda pair: pair[1][0])
        if next_score >= score:
            break
        cols.append(j)
        remaining.remove(j)
        score, w = next_score, next_w
    return query[:, cols] @ w


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    offset, gain = _calibration(study['panel_a'])
    b = _operator(study['panel_b'], (x - offset) / gain)

    f = _quadratic(b)
    y_hat = _scalar_readout(f[:n], train['y'], f[n:])

    z = train['traj']
    modes = np.linalg.svd(z, full_matrices=False)[2][:2]
    f = _affine(b)
    state_pred = (f[n:] @ np.linalg.lstsq(f[:n], z @ modes.T, rcond=None)[0]) @ modes
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
