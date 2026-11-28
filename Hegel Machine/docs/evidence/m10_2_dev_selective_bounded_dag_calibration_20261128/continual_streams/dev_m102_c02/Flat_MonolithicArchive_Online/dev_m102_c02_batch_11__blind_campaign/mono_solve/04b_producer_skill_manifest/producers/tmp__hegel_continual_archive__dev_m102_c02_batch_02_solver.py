"""Calibrated panel-B coordinates with a bilinear scalar index and shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    result = []
    for a, b in zip(u.T, v.T):
        dx = a[j] - a[i]
        valid = np.abs(dx) > 0.1 * np.ptp(a)
        slope = np.median((b[j[valid]] - b[i[valid]]) / dx[valid])
        residual = b - slope * a
        residual -= np.median(residual)
        keep = np.abs(residual) <= max(1e-6, 8 * np.median(np.abs(residual)))
        result.append(_fit(_affine(a[keep, None]), b[keep]))
    return np.asarray(result).T


def _scalar_design(w):
    paired = np.sum(w[:, :2] * w[:, 2:], axis=1)
    return np.column_stack((_affine(w), paired))


def _scalar_prediction(w, y, w_test):
    a = _scalar_design(w)
    b = _scalar_design(w_test)
    seed = _fit(a, y)
    preds = []
    # Both saturating links fit the observed range near the measurement noise floor.
    # Averaging them reduces sensitivity to the unidentified tail of the link.
    for link in (np.tanh, lambda h: h / (1 + h * h)):
        def design(x, q):
            h = x @ q
            return _affine(np.column_stack((h, link(h))))

        def residual(q):
            d = design(a, q)
            return d @ _fit(d, y) - y

        fits = [least_squares(residual, seed * k, max_nfev=100)
                for k in (0.35, 0.7, 1.25)]
        q = min(fits, key=lambda r: r.cost).x
        preds.append(design(b, q) @ _fit(design(a, q), y))
    return np.mean(preds, axis=0)


def _state_prediction(w, traj, t, w_test):
    # Profile the mode amplitudes in closed form; optimize only the two rates.
    time_coeff = _fit(w, traj)

    def projected(q):
        modes = np.vstack((t, -np.expm1(-np.exp(q[:, None]) * t)))
        amplitudes = _fit(modes.T, time_coeff.T).T
        return amplitudes @ modes

    def residual(q):
        return (w @ (projected(q) - time_coeff)).ravel()

    fit = least_squares(residual, np.log([0.5, 2.0]), bounds=(-6, 3), max_nfev=100)
    return w_test @ projected(fit.x)


def solve_batch(study: dict) -> dict:
    bias, gain = _calibrate(study['panel_a'])
    panel = study['panel_b']
    operator = _fit(_affine(np.asarray(panel['u'], dtype=float)), panel['v'])

    def coordinates(x):
        return _affine((np.asarray(x, dtype=float) - bias) / gain) @ operator

    train = study['train']
    w, wt = coordinates(train['x']), coordinates(study['test']['x'])
    y_hat = _scalar_prediction(w, np.asarray(train['y'], dtype=float), wt)
    state_pred = _state_prediction(w, train['traj'], np.asarray(study['eval_times'], dtype=float), wt)
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
