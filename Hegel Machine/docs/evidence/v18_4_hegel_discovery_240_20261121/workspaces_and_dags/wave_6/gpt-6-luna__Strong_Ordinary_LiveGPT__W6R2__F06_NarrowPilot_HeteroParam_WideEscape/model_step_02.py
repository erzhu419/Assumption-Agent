import math
import numpy as np
from scipy.optimize import least_squares


def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    u_n = u  # characteristic_tau_scale = 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(u_n)


def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        np.array([10.0, 0.3, 0.5]),
        np.array([8.0, 0.7, 1.0]),
        np.array([12.0, 0.1, -0.5]),
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, -10.0], [100.0, 5.0, 10.0])) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    return best.x


def predict(row):
    return float(_value(row, fit(train_rows)))