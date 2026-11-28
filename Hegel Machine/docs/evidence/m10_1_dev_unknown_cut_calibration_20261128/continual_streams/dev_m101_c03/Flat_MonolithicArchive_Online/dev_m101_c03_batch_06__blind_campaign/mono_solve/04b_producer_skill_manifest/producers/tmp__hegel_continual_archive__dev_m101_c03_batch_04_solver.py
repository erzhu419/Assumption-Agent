"""Panel-calibrated prediction with two shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    coeff = []
    for raw, observed in zip(u.T, v.T):
        dx = raw[i] - raw[j]
        keep = dx != 0
        gain = np.median((observed[i] - observed[j])[keep] / dx[keep])
        residual = observed - gain * raw
        residual = np.abs(residual - np.median(residual))
        good = residual <= 6 * np.median(residual) + 1e-8
        coeff.append(_fit(np.column_stack((np.ones(good.sum()), raw[good])), observed[good]))
    offset, gain = np.array(coeff).T
    return (x - offset) / gain


def _branches(study, x):
    z = _calibrate(study['panel_a'], x)
    panel = study['panel_c']
    u = panel['u']

    def cross(w):
        a, b, c, d = w.T
        return np.column_stack((np.ones(len(w)), a*b, a*d, b*c, c*d))

    def cosine(w, frequency):
        return np.column_stack((np.ones(len(w)), np.cos(frequency*w)))

    a = cross(z) @ _fit(cross(u), panel['v1'])

    def loss(frequency):
        m = cosine(u, frequency)
        return np.sum((m @ _fit(m, panel['v2']) - panel['v2'])**2)

    frequency = minimize_scalar(loss, bounds=(1.5, 3.1), method='bounded', options={'xatol': 1e-10}).x
    b = cosine(z, frequency) @ _fit(cosine(u, frequency), panel['v2'])
    return a, b


def _predict_state(times, target, slow_train, fast_train, slow_test, fast_test):
    def design(rates, slow, fast):
        f, g = -np.expm1(-rates[:, None] * times)
        m = np.concatenate((slow[:, None, :]*f[None, :, None], fast[:, None, :]*g[None, :, None]), axis=2)
        return m.reshape(len(slow)*len(times), m.shape[-1])

    observed = target.ravel()

    def residual(rates):
        m = design(rates, slow_train, fast_train)
        return m @ _fit(m, observed) - observed

    rates = least_squares(residual, [0.41, 1.81], bounds=([0.05, 0.9], [0.85, 4.0])).x
    coeff = _fit(design(rates, slow_train, fast_train), observed)
    return (design(rates, slow_test, fast_test) @ coeff).reshape(len(slow_test), len(times))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    a, b = _branches(study, np.vstack((train['x'], study['test']['x'])))
    dot = np.sum(a*b, axis=1)
    mix = a[:, 0]*b[:, 1]
    scalar = np.column_stack((a, b, dot))
    slow = np.column_stack((b, mix))
    fast = np.column_stack((a, dot, b, mix))
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    state = _predict_state(study['eval_times'], train['traj'], slow[:n], fast[:n], slow[n:], fast[n:])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
