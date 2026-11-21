import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    numerator = p[0] + p[1]*u + p[2]*u*u
    denominator = 1.0 + p[3]*u + p[4]*u*u + p[5]*s + p[6]*u*s + p[7]*s*s
    return numerator / denominator

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        predictions = np.array([_value(p, r) for r in rows])
        return np.concatenate((predictions-y, 1e-5*p))
    starts = [np.array([0., 3., 0., 0.5, 0., 0.4, 0., 0.]), np.array([0.1, 4., 0.1, 1., 0.1, 0.5, 0.1, 0.1])]
    results = [least_squares(residual, p, max_nfev=10000) for p in starts]
    _params = min(results, key=lambda r: np.linalg.norm(residual(r.x))).x
    return _params

def predict(row):
    return float(_value(_params, row))