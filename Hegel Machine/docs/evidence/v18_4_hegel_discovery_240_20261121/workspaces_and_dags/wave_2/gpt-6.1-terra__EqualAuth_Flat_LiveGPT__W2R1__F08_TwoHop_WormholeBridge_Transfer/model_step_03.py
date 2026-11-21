import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return p[0] * u / (1.0 + p[1] * u + p[2] * s) + p[3] * u / (1.0 + p[4] * u + p[5] * s)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_evaluate(p, r) for r in rows]) - y
    starts = [[2, 0.4, 0.1, 2, 0.4, 1], [4, 0.7, 0.5, 1, 0.1, 0.1], [1, 0.1, 1, 4, 0.8, 0.1], [3, 0.5, 0.3, 3, 1.5, 2]]
    fits = [least_squares(residual, x, bounds=(np.zeros(6), np.full(6, 100.0)), max_nfev=10000) for x in starts]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x

def predict(row):
    return float(_evaluate(_params, row))
