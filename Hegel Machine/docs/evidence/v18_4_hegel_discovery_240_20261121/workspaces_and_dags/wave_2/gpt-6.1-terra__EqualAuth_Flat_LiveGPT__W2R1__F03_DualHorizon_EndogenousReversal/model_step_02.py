import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array [float(r['u_now']) for r in train_rows]
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        af, tf, a_s, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y
    result = least_squares(residual, [1.8, 0.5, 2.5, 12.0], bounds=([0, 0.001, 0, 0.01], [100, 100, 100, 1000]), max_nfev=10000)
    _params = result.x

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts)))
