import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    # Model cold-process response as a saturating surface term plus sqrt(raw input.
    # For processed rows, estimate an additive adjustment from the clamped mediators.
    def pred_params(p, r):
        vmax, km, c_raw, b0, b1, b2, b3 = p
        u = max(0.0, float(r.get('u_raw', 0.0)))
        s = max(0.0, float(r.get('s_surf', 0.0)))
        base = vmax * s / (km + s + 1e-12) + c_raw * math.sqrt(u)
        t = float(r.get('t_proc', 0.0))
        if t <= 0.0:
            return base
        return base + t * (b0 + b1*float(r.get('m_1', 0.0)) + b2*float(r.get('m_2', 0.0)) + b3*float(r.get('m_3', 0.0)))
    def residual(p):
        return np.array([pred_params(p, r) - float(r['y']) for r in rows])
    p0 = np.array([8.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    lower = [-100.0, 1e-6, -100.0, -100.0, -100.0, -100.0, -100.0]
    upper = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # Mild penalty regularizes the poorly identified thermal coefficients.
    def objective(p):
        data = residual(p)
        return np.concatenate([data, 0.03*np.asarray(p[3:])])
    result = least_squares(objective, p0, bounds=(lower, upper), max_nfev=10000)
    return result.x

def predict(row):
    p = fit(_train_rows)
    vmax, km, c_raw, b0, b1, b2, b3 = p
    u = max(0.0, float(row.get('u_raw', 0.0)))
    s = max(0.0, float(row.get('s_surf', 0.0)))
    y = vmax*s/(km+s+1e-12) + c_raw*math.sqrt(u)
    t = float(row.get('t_proc', 0.0))
    if t > 0.0:
        y += t*(b0 + b1*float(row.get('m_1', 0.0)) + b2*float(row.get('m_2', 0.0)) + b3*float(row.get('m_3', 0.0)))
    return float(y)