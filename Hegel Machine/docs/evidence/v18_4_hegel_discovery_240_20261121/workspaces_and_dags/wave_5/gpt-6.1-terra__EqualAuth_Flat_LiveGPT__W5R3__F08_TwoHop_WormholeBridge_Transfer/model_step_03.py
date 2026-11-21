import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    a, b, power, c, q = np.exp(p)
    u = max(float(row['u_now']), 1e-9)
    s = max(float(row['s_grad']), 0.0)
    z = u ** power
    return a * z / (b + z) * (1.0 + c * s) ** (-q)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    bounds = (np.log([0.1, 0.001, 0.2, 0.001, 0.05]),
              np.log([100.0, 100.0, 3.0, 100.0, 5.0]))
    starts = [[8, 1, 1, 0.5, 1], [10, 2, 1, 1, 0.5],
              [6, 0.5, 1, 0.2, 2]]
    fits = [least_squares(residual, np.log(start), bounds=bounds, max_nfev=5000)
            for start in starts]
    _params = min(fits, key=lambda result: np.sum(result.fun ** 2)).x
    return _params

def predict(row):
    return float(_value(row, _params))
