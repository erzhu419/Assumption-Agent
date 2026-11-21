import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[max(1e-8, float(r['u_now'])), max(0.0, float(r['s_grad']))] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # Saturating base-field response with a smooth, multiplicative sink-gradient effect.
    def pred(p, x):
        vmax, km, rho, c = p
        u, s = x
        return vmax * (u / (km + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, x) for x in X]) - y
    result = least_squares(residual, x0=np.array([100.0, 1.0, 0.7, 10.0]), bounds=([0.0, 1e-6, 0.0, 0.0], [1e5, 1e3, 0.999, 1e4]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, km, rho, c = p
    return float(vmax * (u / (km + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))
