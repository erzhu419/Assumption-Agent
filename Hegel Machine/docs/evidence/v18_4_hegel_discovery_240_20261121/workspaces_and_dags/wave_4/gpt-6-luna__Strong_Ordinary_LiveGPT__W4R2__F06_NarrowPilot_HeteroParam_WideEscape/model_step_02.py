import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def predict_params(p, r):
        vmax, rho, c_loc = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        un = u  # characteristic tau scale is 1.0
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
        return base + c_loc * math.sqrt(max(un, 0.0))
    def residual(p):
        return np.array([predict_params(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
    return result.x

def predict(row):
    p = fit([row])
    return float(p[0] * (1.0 - p[1] * float(row['s_grad']) / (1.0 + float(row['s_grad']))) * (float(row['u_now']) / (1.2 + float(row['u_now']))) + p[2] * math.sqrt(max(float(row['u_now']), 0.0)))