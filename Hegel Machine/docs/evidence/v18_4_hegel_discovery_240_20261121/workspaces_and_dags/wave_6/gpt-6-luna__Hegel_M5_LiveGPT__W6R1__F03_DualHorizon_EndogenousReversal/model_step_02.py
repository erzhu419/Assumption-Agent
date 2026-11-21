import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(logp):
        af, ass, tf, ts = np.exp(logp)
        pred = u * (af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts)))
        return pred - y

    result = least_squares(residual, np.log([1.0, 1.0, 1.0, 10.0]), max_nfev=20000)
    _params = np.exp(result.x)
    return _params

def predict(row) -> float:
    af, ass, tf, ts = _params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts))))