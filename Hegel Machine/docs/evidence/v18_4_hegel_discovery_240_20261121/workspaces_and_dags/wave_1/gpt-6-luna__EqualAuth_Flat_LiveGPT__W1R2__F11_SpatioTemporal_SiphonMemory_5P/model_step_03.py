import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    H = max(0.0, float(row['H']))
    x = float(row['x_now'])
    m = float(row['mem_conv'])
    return u, s, H, x, m

def _pred(row, p):
    u, s, H, x, m = _features(row)
    j0, beta, rho, c_sqrt, gamma = p
    base = j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)
    # Lag modulation from the observed memory/current-state mismatch, accumulated over the forecast horizon.
    return base * math.exp(max(-20.0, min(20.0, gamma * H * (m - x))))

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.2, 0.0]),
                           bounds=([0.0, 0.0, 0.0, 0.0, -10.0], [20.0, 20.0, 0.999, 20.0, 10.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    raise ValueError('predict requires parameters fitted through fit')