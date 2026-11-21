import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a_fast, t_fast, a_slow, gap = p
        t_slow = t_fast + gap
        gain = -a_fast * np.expm1(-h / t_fast) + a_slow * np.expm1(-h / t_slow)
        return u * gain - y

    result = least_squares(residual, [1.7, 0.45, 2.5, 12.0], bounds=([0.0, 1e-8, 0.0, 1e-8], np.inf), x_scale='jac')
    _params = result.x

def predict(row):
    a_fast, t_fast, a_slow, gap = _params
    h = row['H']
    gain = -a_fast * math.expm1(-h / t_fast) + a_slow * math.expm1(-h / (t_fast + gap))
    return float(row['u_now'] * gain)
