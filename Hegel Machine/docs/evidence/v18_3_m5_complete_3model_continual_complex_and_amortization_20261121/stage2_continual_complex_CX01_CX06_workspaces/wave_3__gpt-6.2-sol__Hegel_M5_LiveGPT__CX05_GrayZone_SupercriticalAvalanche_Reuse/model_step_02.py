import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s, u / (1.2 + u), s / (1.0 + s), math.sqrt(u)


def _law(p, u, s, sat, sink, root):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * sink) * sat + c_loc * root - k_aval * np.maximum(0.0, s - s_crit) * u


def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float).T
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        return _law(p, *X) - y

    bounds = ([1.0, 0.05, -2.0, 0.0, 1.25], [15.0, 0.95, 2.0, np.inf, 1.62])
    fits = [least_squares(residual, [6.5, 0.7, 0.15, 1.0, t], bounds=bounds, ftol=1e-12, xtol=1e-12, gtol=1e-12) for t in (1.34, 1.45)]
    _params = min(fits, key=lambda fit_result: fit_result.cost).x
    return _params


def predict(row) -> float:
    return float(_law(_params, *_features(row)))
