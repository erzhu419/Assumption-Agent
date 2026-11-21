import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        af, tf, a_s, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y
    best = None
    for guess in ([2, 0.8, 3, 10], [3, 1, 4, 20], [1.5, 0.5, 2, 5]):
        result = least_squares(residual, np.log(guess), max_nfev=5000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))
