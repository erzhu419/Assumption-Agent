import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([row['u_now'] for row in train_rows], dtype=float)
    s = np.asarray([row['s_grad'] for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    branch = np.sqrt(u)

    def residual(p):
        vmax, rho, c_sqrt, leak, threshold = p
        return vmax * (1.0 - rho * sink) * drive + c_sqrt * branch - leak * u * np.maximum(0.0, s - threshold) - y

    result = least_squares(residual, [6.0, 0.7, 0.4, 1.5, 2.55], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf]))
    _params = tuple(map(float, result.x))

def predict(row) -> float:
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c_sqrt, leak, threshold = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u) - leak * u * max(0.0, s - threshold))