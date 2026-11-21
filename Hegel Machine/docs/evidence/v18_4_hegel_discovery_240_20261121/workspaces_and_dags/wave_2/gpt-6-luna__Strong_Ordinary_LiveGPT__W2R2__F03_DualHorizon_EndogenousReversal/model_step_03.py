import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    # y = u * (A_fast*(1-exp(-H/tau_fast)) - A_slow*(1-exp(-H/tau_slow)))
    def residual(p):
        af, tf, ass, ts = p
        pred = u * (af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts)))
        return pred - y

    fit_result = least_squares(
        residual,
        x0=np.array([1.0, 0.2, 1.0, 10.0]),
        bounds=([0.0, 1e-5, 0.0, 1e-4], [100.0, 100.0, 100.0, 1000.0]),
        max_nfev=50000
    )
    return fit_result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    af, tf, ass, ts = p
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts))))

_TRAIN_ROWS = []
# fit() stores parameters for predict() through its input rows, so predict expects
# the evaluator to call fit(train_rows) first.
_PARAMS = None

def fit(train_rows):
    global _TRAIN_ROWS, _PARAMS
    _TRAIN_ROWS = list(train_rows)
    rows = _TRAIN_ROWS
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        af, tf, ass, ts = p
        return u * (af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts))) - y
    res = least_squares(residual, [1.0, 0.2, 1.0, 10.0], bounds=([0.0, 1e-5, 0.0, 1e-4], [100.0, 100.0, 100.0, 1000.0]), max_nfev=50000)
    _PARAMS = res.x
    return _PARAMS

def predict(row):
    af, tf, ass, ts = _PARAMS
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts))))