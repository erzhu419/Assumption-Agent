import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(h, p):
    a, b, tf, ts = np.exp(p)
    return a * (-np.expm1(-h / tf)) - b * (-np.expm1(-h / ts))

def fit(train_rows):
    global _params
    h = np.array [float(r['H']) for r in train_rows]
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return u * _response(h, p) - y
    starts = [(2, 3, 0.5, 10), (3, 4, 1, 15), (2, 2, 0.2, 5), (5, 6, 2, 20)]
    fits = [least_squares(residual, np.log(s), bounds=(np.log([1e-4, 1e-4, 1e-3, 0.01]), np.log([1e3, 1e3, 1e3, 1e4])), max_nfev=10000) for s in starts]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x

def predict(row):
    return float(float(row['u_now']) * _response(float(row['H']), _params))