import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.05


def _features(row):
    u_n = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    base = (1.0 - s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base, s / (1.0 + s) * (u_n / (1.2 + u_n)), math.sqrt(max(u_n, 0.0))


def fit(train_rows):
    X = np.asarray([_features(r) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)

    # y = vmax*base - (vmax*rho)*gradient_term + c_loc*sqrt(u_n)
    # Fit numerically, then convert the coupled coefficients to vmax and rho.
    def residual(q):
        return X @ q - y

    result = least_squares(residual, x0=np.array([6.0, 4.2, 0.1]), max_nfev=10000)
    a, b, c_loc = result.x
    vmax = float(a)
    rho = float(b / a) if abs(a) > 1e-12 else 0.70
    return {'vmax': vmax, 'rho': rho, 'c_loc': float(c_loc)}


def predict(row):
    params = fit([row]) if False else None
    # Parameters are fitted by fit(train_rows); predict uses the fitted values
    # attached by the standard harness after fitting.
    if params is None:
        params = _FITTED_PARAMS
    base, grad_term, root_term = _features(row)
    return float(params['vmax'] * base - params['vmax'] * params['rho'] * grad_term + params['c_loc'] * root_term)


def _fit_and_bind(train_rows):
    global _FITTED_PARAMS
    _FITTED_PARAMS = fit(train_rows)
    return _FITTED_PARAMS


# Ensure the standard fit(train_rows) call also binds parameters for predict(row).
_fit_impl = fit
def fit(train_rows):
    params = _fit_impl(train_rows)
    global _FITTED_PARAMS
    _FITTED_PARAMS = params
    return params
