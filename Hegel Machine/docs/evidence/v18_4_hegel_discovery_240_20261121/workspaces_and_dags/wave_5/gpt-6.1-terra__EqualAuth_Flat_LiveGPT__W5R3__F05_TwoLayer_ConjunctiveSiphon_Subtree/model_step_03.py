import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    vmax, rho, km, c = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [3.0, 0.5, 1.2, 0.4], bounds=([0.0, 0.0, 0.001, 0.0], [50.0, 2.0, 50.0, 10.0]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    return float(_value(_params, row))