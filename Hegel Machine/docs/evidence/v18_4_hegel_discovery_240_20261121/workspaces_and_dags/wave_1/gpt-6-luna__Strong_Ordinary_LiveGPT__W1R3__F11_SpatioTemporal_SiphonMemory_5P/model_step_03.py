import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']), 1e-9)
    s = max(float(row['s_grad']), 0.0)
    base = (u / (1.2 + u))
    sink = s / (1.0 + s)
    return u, base, sink

def _pred(row, p):
    u, base, sink = _features(row)
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * sink) * base + c_loc * np.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(r, p) for r in rows]) - y
    best = None
    for start in ([2.0, 0.7, 0.0], [2.0, 0.7, 0.3], [2.0, 0.7, -0.3]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    global _params
    return float(_pred(row, _params))

_params = np.array([1.0, 0.5, 0.0])