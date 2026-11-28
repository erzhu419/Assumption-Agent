"""Calibrated paired-index scalar response and two-mode trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    a = study['panel_a']
    keep = np.max(np.abs(a['v'] - a['u']), axis=1) < 1
    u, v = a['u'][keep], a['v'][keep]
    du = u - u.mean(axis=0)
    gain = np.sum(du * (v - v.mean(axis=0)), axis=0) / np.sum(du**2, axis=0)
    offset = v.mean(axis=0) - u.mean(axis=0) * gain
    b = study['panel_b']
    operator = _fit(_affine(b['u']), b['v'])
    return _affine((x - offset) / gain) @ operator


def _scalar_features(z):
    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((_affine(z), interaction))


def _scalar(z, target, query):
    d, dq = _scalar_features(z), _scalar_features(query)

    def design(d, p):
        h = d @ p
        return np.column_stack((np.ones(len(h)), h, np.arctan(h)))

    def residual(p):
        a = design(d, p)
        return a @ _fit(a, target) - target

    init = _fit(d, target)
    opt = min((least_squares(residual, scale * init, max_nfev=150)
               for scale in (0.35, 0.7, 1.15, 1.6)), key=lambda r: r.cost)
    return design(dq, opt.x) @ _fit(design(d, opt.x), target)


def _trajectory(z, target, query, t):
    coefficients = _fit(z, target)

    def basis(rates):
        return np.stack((np.log1p(t * rates[0]), -np.expm1(-t * rates[1])))

    def residual(rates):
        b = basis(rates)
        return (z @ (_fit(b.T, coefficients.T).T @ b - coefficients)).ravel()

    rates = least_squares(residual, [1.4, 0.7], bounds=(0.001, 20)).x
    b = basis(rates)
    return query @ _fit(b.T, coefficients.T).T @ b


def solve_batch(study: dict) -> dict:
    train = study['train']
    z = _coordinates(study, train['x'])
    q = _coordinates(study, study['test']['x'])
    y = _scalar(z, train['y'], q)
    state = _trajectory(z, train['traj'], q, study['eval_times'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
