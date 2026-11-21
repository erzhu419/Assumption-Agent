import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a, b, c, d = p
        return u / (a + b*u + c*s + d*u*s) - y
    result = least_squares(residual, [0.03, 0.15, 0.20, 0.08], bounds=(0, np.inf), max_nfev=10000)
    _params = result.x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    a, b, c, d = _params
    return float(u / (a + b*u + c*s + d*u*s))
