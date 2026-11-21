import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    un = float(row['u_now']) / 0.05
    sg = float(row['s_grad'])
    base_factor = (1.0 - sg / (1.0 + sg)) * (un / (1.2 + un))
    return un, base_factor

def fit(train_rows):
    global _params
    rows = list(train_rows)
    yy = np.array([float(r['y']) for r in rows])
    X = [_features(r) for r in rows]
    def residual(p):
        vmax, rho, cloc = p
        return np.array([vmax * (1.0 - rho * float(r['s_grad']) / (1.0 + float(r['s_grad']))) * (un / (1.2 + un)) + cloc * math.sqrt(max(un, 0.0)) - float(r['y']) for r, (un, _) in zip(rows, X)])
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, cloc = _params
    un = float(row['u_now']) / 0.05
    sg = float(row['s_grad'])
    return float(vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un)) + cloc * math.sqrt(max(un, 0.0)))