import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s / (1.0 + s), float(row['H'])

def fit(train_rows):
    global _params
    u, sink, h = np.asarray([_inputs(row) for row in train_rows], dtype=float).T
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    base = u / (1.2 + u)

    def residual(p):
        vmax, rho, fast, amp, gap = p
        gain = vmax * (1.0 - rho * sink) * base * (-np.expm1(-h / fast))
        adaptation = amp * u * (-np.expm1(-h / (fast + gap)))
        return gain - adaptation - y

    result = least_squares(residual, [5.0, 0.7, 0.22, 1.8, 13.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [np.inf, 1.0, np.inf, np.inf, np.inf]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row):
    u, sink, h = _inputs(row)
    vmax, rho, fast, amp, gap = _params
    gain = vmax * (1.0 - rho * sink) * (u / (1.2 + u)) * (-math.expm1(-h / fast))
    adaptation = amp * u * (-math.expm1(-h / (fast + gap)))
    return float(gain - adaptation)
