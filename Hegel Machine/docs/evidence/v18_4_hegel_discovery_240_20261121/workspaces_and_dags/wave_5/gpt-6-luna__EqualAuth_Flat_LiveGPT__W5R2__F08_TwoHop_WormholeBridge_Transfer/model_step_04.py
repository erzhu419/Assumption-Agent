import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, u, s):
        a, b, c, d, e, f = p
        c = abs(c) + 1e-8
        z = s - 1.0
        return a + b * (1.0 - np.exp(-c * u)) + d*z + e*z*z + f*u*z
    p0 = np.array([0.0, 4.0, 0.6, 0.0, 0.0, 0.0])
    result = least_squares(lambda p: pred(p, u, s) - y, p0, max_nfev=20000)
    return result.x

def predict(row):
    u, s = _features(row)
    a, b, c, d, e, f = _FIT_PARAMS
    return float(a + b * (1.0 - np.exp(-(abs(c) + 1e-8) * u)) + d*(s-1.0) + e*(s-1.0)**2 + f*u*(s-1.0))