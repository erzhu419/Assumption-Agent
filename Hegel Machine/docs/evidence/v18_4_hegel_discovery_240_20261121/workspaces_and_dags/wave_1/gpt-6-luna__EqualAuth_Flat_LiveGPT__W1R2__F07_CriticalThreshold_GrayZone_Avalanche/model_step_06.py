import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-6, r['u_now']) for r in train_rows], dtype=float)
    s = np.array([max(0.0, r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, ss):
        vmax, km, rho, threshold, width, drop = p
        base = vmax * (1.0 - rho * ss / (1.0 + ss)) * uu / (km + uu)
        gate = 1.0 / (1.0 + np.exp(np.clip(-(ss - threshold) / width, -60, 60)))
        return base - drop * gate * np.maximum(uu, 0.0)
    p0 = [5.0, 1.2, 0.7, 3.0, 0.25, 2.0]
    lo = [0.0, 0.01, 0.0, 0.0, 0.03, 0.0]
    hi = [20.0, 20.0, 1.5, 8.0, 3.0, 20.0]
    res = least_squares(lambda p: pred(p, u, s) - y, p0, bounds=(lo, hi), max_nfev=20000)
    return res.x.tolist()

def predict(row):
    p = predict._params
    u = max(1e-6, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, km, rho, threshold, width, drop = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
    gate = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -(s - threshold) / width))))
    return float(base - drop * gate * u)
