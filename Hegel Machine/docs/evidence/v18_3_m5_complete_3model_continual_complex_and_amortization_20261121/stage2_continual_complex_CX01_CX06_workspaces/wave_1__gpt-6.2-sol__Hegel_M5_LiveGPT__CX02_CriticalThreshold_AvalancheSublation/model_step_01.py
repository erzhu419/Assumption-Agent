import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u_n = float(row['u_now']) / 1.0
    s = float(row['s_grad'])
    return u_n / (1.2 + u_n), s / (1.0 + s), math.sqrt(u_n)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    drive, sink, branch = np.array([_features(row) for row in rows], dtype=float).T
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * branch - y

    def jacobian(p):
        vmax, rho, _ = p
        return np.column_stack(((1.0 - rho * sink) * drive, -vmax * sink * drive, branch))

    _params = least_squares(residual, [6.5, 0.70, 0.15], jac=jacobian, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    drive, sink, branch = _features(row)
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * sink) * drive + c_loc * branch)
