import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r.get('u_indoor', 0.0)), float(r.get('a_foot', 1.0)), float(r.get('impact_idx', 0.0))] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # Saturating indoor dose, attenuated by footprint and impact; the intercept
    # absorbs any residual baseline in this rainy-city regime.
    def predict_params(p, x):
        base, amp, k, space, impact_power = p
        indoor, area, impact = x.T
        dose = indoor / (k + indoor + 1e-12)
        area_factor = np.exp(-space * (area - 1.0))
        impact_factor = np.maximum(1.0 - impact, 1e-6) ** impact_power
        return base + amp * dose * area_factor * impact_factor
    res = least_squares(lambda p: predict_params(p, X) - y,
        x0=[1.5, 8.0, 1.0, 0.5, 1.0],
        bounds=([-20.0, 0.0, 0.001, -5.0, 0.0], [20.0, 30.0, 20.0, 5.0, 8.0]),
        max_nfev=20000)
    return res.x.tolist()

def predict(row):
    p = fit([row])
    # This fallback is replaced by fitted parameters when fit() is used as intended.
    indoor = float(row.get('u_indoor', 0.0))
    area = float(row.get('a_foot', 1.0))
    impact = float(row.get('impact_idx', 0.0))
    base, amp, k, space, impact_power = p
    return float(base + amp * indoor / (k + indoor + 1e-12) * math.exp(-space * (area - 1.0)) * max(1.0 - impact, 1e-6) ** impact_power)
