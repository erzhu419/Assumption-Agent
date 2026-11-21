import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def _law(p, u, s):
    vmax, rho, strength, u_mid, u_slope, s_mid, log_s_slope = p
    z = u / 4.0
    g = s / (1.0 + s)
    basin = vmax * (1.0 - rho * g) * z / (1.2 + z)
    opening = _sigmoid(u_slope * (z - u_mid)) * _sigmoid(math.exp(log_s_slope) * (s - s_mid))
    return basin - strength * g * opening

def fit(train_rows):
    global _params
    u, s, y = np.asarray([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float).T
    center = math.log(15.0)
    def residual(p):
        return np.append(_law(p, u, s) - y, 0.006 * (p[-1] - center))
    start = [5.7, 0.72, 0.45, 1.6, 0.95, 0.57, center]
    lower = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, math.log(2.0)]
    upper = [np.inf, 1.0, np.inf, 6.0, 10.0, 2.5, math.log(60.0)]
    _params = least_squares(residual, start, bounds=(lower, upper), x_scale='jac').x

def predict(row):
    return float(_law(_params, float(row['u_now']), float(row['s_grad'])))