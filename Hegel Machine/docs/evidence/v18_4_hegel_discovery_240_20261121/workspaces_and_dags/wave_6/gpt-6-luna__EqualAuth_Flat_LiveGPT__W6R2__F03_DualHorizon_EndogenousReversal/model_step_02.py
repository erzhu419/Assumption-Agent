import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    def unpack(z):
        amp_fast, amp_slow, tau_fast, gap = np.exp(z)
        return amp_fast, amp_slow, tau_fast, tau_fast + gap
    def predict_params(row, params):
        amp_fast, amp_slow, tau_fast, tau_slow = params
        h = float(row['H'])
        u = float(row['u_now'])
        return u * (amp_fast * (-math.expm1(-h / tau_fast))
                    - amp_slow * (-math.expm1(-h / tau_slow)))
    def residual(z):
        params = unpack(z)
        return np.array([predict_params(r, params) - float(r['y']) for r in rows])
    starts = [
        np.log([1.0, 1.1, 0.5, 10.0]),
        np.log([1.2, 1.3, 1.0, 20.0]),
        np.log([0.8, 1.0, 0.2, 5.0]),
    ]
    best = None
    for z0 in starts:
        result = least_squares(residual, z0, max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, unpack(result.x))
    return {'params': best[1]}


def predict(row):
    return predict_params_for_model(row)


def predict_params_for_model(row):
    # Parameters are attached by fit for the standard fit-then-predict interface.
    params = _FITTED_MODEL['params']
    amp_fast, amp_slow, tau_fast, tau_slow = params
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast))
                - amp_slow * (-math.expm1(-h / tau_slow)))