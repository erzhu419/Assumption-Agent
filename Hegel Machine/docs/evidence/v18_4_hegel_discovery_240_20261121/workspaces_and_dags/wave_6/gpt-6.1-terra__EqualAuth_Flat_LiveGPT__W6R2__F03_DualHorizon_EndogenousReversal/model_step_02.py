import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, h):
    af, tf, a_s, ts = p
    return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array [float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for start in ([1.5, 0.5, 2.5, 10.0], [2.0, 1.0, 3.0, 20.0], [1.0, 0.3, 2.0, 5.0]):
        result = least_squares(lambda p: _evaluate(p, u, h) - y, start, bounds=([0.0, 1e-5, 0.0, 1e-5], [100.0, 1000.0, 100.0, 1000.0]), max_nfev=5000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['H'])))