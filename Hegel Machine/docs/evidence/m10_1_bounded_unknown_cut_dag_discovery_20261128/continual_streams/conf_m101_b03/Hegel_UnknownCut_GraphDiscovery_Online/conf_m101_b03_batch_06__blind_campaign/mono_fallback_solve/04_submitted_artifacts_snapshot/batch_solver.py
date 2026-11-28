"""Predict the scalar response and two-mode trajectory from calibrated operators."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-9)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _pairs(z):
    return z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    # Coordinatewise affine calibration. Joint row trimming rejects panel outliers.
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        uc, vc = u[keep], v[keep]
        centered = uc - uc.mean(axis=0)
        gain = np.sum(centered * (vc - vc.mean(axis=0)), axis=0) / np.sum(centered**2, axis=0)
        shift = vc.mean(axis=0) - uc.mean(axis=0) * gain
        error = np.linalg.norm(u * gain + shift - v, axis=1)
        updated = error <= 4 * np.median(error)
        if np.array_equal(keep, updated):
            break
        keep = updated
    return (x - shift) / gain


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    pb, pc = study['panel_b'], study['panel_c']
    b = _affine(z) @ _fit(_affine(pb['u']), pb['v'])
    q = _pairs(z)
    c = _affine(q) @ _fit(_affine(_pairs(pc['u'])), pc['v1'])

    def cosine_design(rate):
        return _affine(np.cos(rate * pc['u']))

    def residual(rate):
        f = cosine_design(rate)
        return np.linalg.norm(f @ _fit(f, pc['v2']) - pc['v2'])

    rate = minimize_scalar(residual, bounds=(2.2, 2.6), method='bounded').x
    h = np.cos(rate * z)
    d = _affine(h) @ _fit(cosine_design(rate), pc['v2'])
    return z, b, q, c, h, d


def _dot(a, b, shift=0):
    return np.sum(a * np.roll(b, shift, axis=1), axis=1)


def _designs(study, x):
    z, b, q, c, h, d = _features(study, x)
    scalar = np.column_stack((_affine(q), h, _dot(c, d)))
    mixed = (q[:, :, None] * h[:, None, :]).reshape(len(x), 16)
    state = np.column_stack((_affine(q), h, _dot(b, c, 2), _dot(z, c), mixed))
    return scalar, state


def _time_modes(times, trajectory):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        h = basis(rates)
        return (h @ _fit(h, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x
    return basis(np.sort(rates))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    scalar, state = _designs(study, x)
    h = _time_modes(study['eval_times'], train['traj'])
    amps = _fit(h, train['traj'].T).T
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    state_pred = (state[n:] @ _fit(state[:n], amps)) @ h.T
    endpoint = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
