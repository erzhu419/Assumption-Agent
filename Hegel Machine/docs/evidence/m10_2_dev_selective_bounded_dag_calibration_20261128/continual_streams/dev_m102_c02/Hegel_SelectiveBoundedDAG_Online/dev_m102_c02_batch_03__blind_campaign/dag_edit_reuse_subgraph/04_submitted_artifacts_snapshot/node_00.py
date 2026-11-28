"""Invert the coupled sensor calibration, then apply the reference-panel map."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _terms(u, i):
    a, b = u[:, i], u[:, (i + 1) % u.shape[1]]
    return np.column_stack((np.ones(len(u)), a, b, a*b, a**3))


def _calibrate(u, v):
    # Reject gross contamination jointly rather than fitting each channel to
    # a different set of panel rows.
    score = np.max(np.abs(v - np.median(v, axis=0)), axis=1)
    center = np.median(score)
    spread = np.median(np.abs(score-center))
    keep = score <= center + 6*max(spread, 1e-12)
    coef = []
    for i in range(u.shape[1]):
        design = _terms(u, i)
        coef.append(np.linalg.lstsq(design[keep], v[keep, i], rcond=None)[0])
    return np.asarray(coef)


def _invert(x, coef):
    bias, gain, cross, mixed, cubic = coef.T
    n = len(gain)
    own = np.arange(n)
    nxt = (own + 1) % n

    def residual(z, target):
        b = z[nxt]
        return bias + gain*z + cross*b + mixed*z*b + cubic*z**3 - target

    def jacobian(z, target):
        j = np.zeros((n, n))
        j[own, own] = gain + mixed*z[nxt] + 3*cubic*z*z
        j[own, nxt] += cross + mixed*z
        return j

    return np.array([least_squares(residual, np.clip((row-bias)/gain, -2, 2),
                                    jac=jacobian, args=(row,), ftol=1e-12,
                                    xtol=1e-12, gtol=1e-12).x for row in x]).reshape(-1, n)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    first, second = study['panel_a'], study['panel_b']
    coef = _calibrate(np.asarray(first['u'], float), np.asarray(first['v'], float))
    operator = np.linalg.lstsq(_affine(np.asarray(second['u'], float)),
                               np.asarray(second['v'], float), rcond=None)[0]
    train, test = (np.asarray(study[k]['x'], float) for k in ('train', 'test'))
    coordinates = _affine(_invert(np.vstack((train, test)), coef)) @ operator
    return {'c': coordinates[:len(train)], 'query': coordinates[len(train):]}
