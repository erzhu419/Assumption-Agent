"""Inverse calibration, panel operators, and a two-mode response model."""
import numpy as np
from scipy.optimize import least_squares


def _affine(*columns):
    return np.column_stack((np.ones(len(columns[0])), *columns))


def _ols(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        bias, scale = np.array([_ols(_affine(u[keep, j]), v[keep, j])
                                for j in range(u.shape[1])]).T
        e = np.linalg.norm(u * scale + bias - v, axis=1)
        next_keep = e <= max(1e-9, 4 * np.median(e))
        if next_keep.sum() < len(u) // 2 or np.array_equal(keep, next_keep):
            break
        keep = next_keep
    return bias, scale


def _pairs(z):
    return np.column_stack([z[:, i] * z[:, j]
                            for i, j in ((0, 1), (0, 3), (1, 2), (2, 3))])


def _c2_features(z, curvature, amplitude):
    return _affine(z + (amplitude / curvature) * np.log1p(curvature * z**2))


def _spatial(study, x):
    bias, scale = _calibration(study['panel_a'])
    z = (x - bias) / scale
    panel = study['panel_c']
    u, v1, v2 = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))
    p = _affine(_pairs(z)) @ _ols(_affine(_pairs(u)), v1)

    def residual(params):
        f = _c2_features(u, *params)
        return (f @ _ols(f, v2) - v2).ravel()

    params = least_squares(residual, [.36, .55],
                           bounds=([.001, .01], [3., 2.])).x
    r = _c2_features(z, *params) @ _ols(_c2_features(u, *params), v2)
    dot = np.sum(p * r, axis=1)
    cross = p[:, 0] * r[:, 1]
    return _affine(p, r, dot), _affine(p, r, dot, cross), _affine(r, cross)


def _time_basis(t, rates):
    return -np.expm1(-t[:, None] * rates[None, :])


def _trajectory(fast, slow, times, n, measured):
    def design(rates, stop):
        h = _time_basis(times, rates)
        a = fast[:stop, None, :] * h[None, :, 0, None]
        b = slow[:stop, None, :] * h[None, :, 1, None]
        return np.concatenate((a, b), axis=2).reshape(-1, fast.shape[1] + slow.shape[1])

    target = measured.ravel()

    def residual(rates):
        d = design(rates, n)
        return d @ _ols(d, target) - target

    rates = least_squares(residual, [1.8, .42],
                          bounds=([1.3, .1], [2.3, .8])).x
    coef = _ols(design(rates, n), target)
    k = fast.shape[1]
    amplitudes = np.column_stack((fast[n:] @ coef[:k], slow[n:] @ coef[k:]))
    return amplitudes @ _time_basis(times, rates).T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    y_features, fast, slow = _spatial(study, x)
    y_hat = y_features[n:] @ _ols(y_features[:n], np.asarray(train['y']))
    state = _trajectory(fast, slow, np.asarray(study['eval_times']), n,
                        np.asarray(train['traj']))
    final_state = state[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final_state.mean(), final_state.std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
