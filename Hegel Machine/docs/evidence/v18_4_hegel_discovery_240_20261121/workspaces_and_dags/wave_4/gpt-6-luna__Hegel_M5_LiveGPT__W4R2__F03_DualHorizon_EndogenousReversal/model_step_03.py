import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    a_fast, tau_fast, a_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (a_fast * (-math.expm1(-h / tau_fast)) - a_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [1.3, 0.5, 0.6, 10.0], bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 1000.0, 20.0, 1000.0]), max_nfev=20000)
    return result.x

def predict(row):
    # The confirmation harness supplies fitted parameters in the row when predicting.
    p = row.get('_fit_params')
    if p is None:
        raise ValueError('predict requires _fit_params from fit(train_rows)')
    return float(_value(row, p))
