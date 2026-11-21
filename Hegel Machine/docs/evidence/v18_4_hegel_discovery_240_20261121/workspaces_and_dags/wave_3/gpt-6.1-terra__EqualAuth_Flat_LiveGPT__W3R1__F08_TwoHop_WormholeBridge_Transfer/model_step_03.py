import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    a, b, power, c, q = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    up = u ** power
    return a * up / (b + up) * ((1.0 + c) / (1.0 + c * s)) ** q

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_evaluate(p, r) for r in train_rows]) - y
    bounds = ([0.01, 0.001, 0.1, 0.00001, 0.05], [100.0, 100.0, 4.0, 20.0, 5.0])
    best = None
    for start in ([6.0, 1.0, 1.0, 0.4, 1.0], [8.0, 2.0, 0.8, 1.0, 0.5], [5.5, 0.7, 1.2, 0.2, 2.0]):
        result = least_squares(residual, start, bounds=bounds, max_nfev=3000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, row))
